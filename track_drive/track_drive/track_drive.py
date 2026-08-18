#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# 본 프로그램은 자이트론에서 제작한 것입니다.
# 상업라이센스에 의해 제공되므로 무단배포 및 상업적 이용을 금합니다.
# 교육과 실습 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
#
#  ┌───────────────────────────────────────────────────────────────┐
#  │  자율주행 제어 노드 (track_drive.py) — 기능별(Feature) 섹션 구조 │
#  │                                                                 │
#  │  [데이터 흐름]  센서 → 인지 → 판단 → 제어 → 모터                 │
#  │                                                                 │
#  │  [코스 시나리오] (실차 전환 후 재정의)                          │
#  │   1. 신호등 인식 후 출발                                        │
#  │   2. 차선주행                                                   │
#  │   3. 4구 신호등 교차로 — 직진/지름길 경로 선택                  │
#  │      ├ 직진 선택 → 차선주행(S1) 복귀 후 순서대로 진행:          │
#  │      │    4. 라바콘 주행         (B1_LAVACON)                  │
#  │      │    5. 고정장애물 회피     (B2_OBSTACLE, ★재설계 예정)    │
#  │      │    6. 방해차량 추월       (B3_VEHICLE,  ★재설계 예정)    │
#  │      └ 지름길 선택 → 좌회전 → 지름길(S3) → 좌회전 → 차선주행 복귀│
#  │                                                                 │
#  │  [섹션 목차]                                                    │
#  │   [0] 설정  [1] 통신I/O  [2] 인지  [3] 판단  [4] 제어            │
#  │   [5] 메인루프  [6] 유틸/디버그                                  │
#  │   ※ 각 인지 섹션의 [담당]/[협업] 표기 참고 (한 기능=한 담당자)   │
#  └───────────────────────────────────────────────────────────────┘
#=============================================
import rclpy, cv2, math, time
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from std_msgs.msg import Float32MultiArray, Float32
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from .perception.perc_lavacon import process_lavacon
from .perception.hough_lane import HoughLaneDetector
from .perception.perc_floor import check_stopline, LaneDetector as ClassicLaneDetector
from .perception.lane_util import CameraProcessor, SlideWindow
from .perception.dl_lane import DLLaneDetector
from .perception.traffic_signal import SignalDetector
from .perception.yolo_cone import YoloConeDetector
from .controller.obstacle_avoidance import ObstacleAvoidance, AvoidPhase
# vehicle_overtake.py 의 구 VehicleOvertake 는 더 이상 쓰지 않는다.
#   추월/회피가 규정상 같은 기동("타겟이 없는 차선으로 지나간다")이라
#   obstacle_avoidance.TargetPassing 한 클래스로 통합했다(moving 플래그로 구분).
from .planner.hybrid_astar import HybridAStar
from .planner.occupancy import OccupancyGrid
from .controller.stanley import StanleyController
from .controller.pure_pursuit import PurePursuitController
from .planner.node import Node as PlannerNode
from .localization.pose_estimator import EncoderPoseEstimator
from .kr_text import put_text_kr_multi
# #############################################################
# [0] 설정 (Config)
# #############################################################
#   튜닝 파라미터/디버그 플래그/state 관련 상수는 전부 config.py로 모아뒀다.
#   실차 테스트 중 값을 바꿔야 하면 이 파일이 아니라 config.py를 고칠 것 —
#   MissionState/BehaviorState/Phase Enum도 config.py에 있다(START_STATE가
#   Enum 값을 쓰기 때문에 같이 옮겨야 순환 import가 안 생긴다).
from .config import *  # noqa: F401,F403 — 아래 전체가 이 파일 곳곳에서 이름 그대로 쓰인다


# #############################################################
# ROS2 노드
# #############################################################
class TrackDriverNode(Node):

    def __init__(self):
        super().__init__('driver')
        self.bridge = CvBridge()

        # ── 원본 센서 버퍼 ──
        self.img_front = self.img_left = self.img_right = self.img_behind = None
        self.lidar_ranges = None
        self.imu_yaw = 0.0
        # [2026-08-06] IMU 각속도(yaw_rate, rad/s) — cb_imu()가 msg.angular_velocity.z를
        # 그대로 저장. _imu_curvature_px()가 VESC 실측속도와 묶어 pure_pursuit의 코너
        # 감쇠(lookahead_curvature_gain)를 보강하는 데 쓴다(controller/pure_pursuit.py
        # imu_curvature_px 주석 참고). _imu_t는 VESC의 _vesc_t와 동일한 생존 체크 용도 —
        # IMU가 죽으면(메시지가 안 옴) imu_yaw_rate가 마지막 값에 얼어붙는데, 그 상태로
        # 계속 curvature 계산에 쓰면 "코너가 아닌데 코너로 착각해 lookahead가 계속
        # 눌려있는" 문제가 생기므로 반드시 이 타임스탬프로 살아있는지 먼저 확인할 것.
        self.imu_yaw_rate = 0.0
        self._imu_t = None
        # [2026-08-06] imu_yaw_rate 저역통과 상태 — _imu_curvature_px()가 IMU_YAW_RATE_EMA_ALPHA로
        # 갱신한다(config.py 주석 참고). IMU/VESC가 죽어 _imu_curvature_px()가 None을 반환하는
        # 동안엔 갱신을 건너뛰어(아래 함수 참고) 그대로 얼어있는다 — held 프레임에 last_curvature를
        # 안 건드리는 것과 같은 원칙, 다시 살아나면 몇 프레임 안에 EMA로 자연스럽게 수렴한다.
        self._imu_yaw_rate_ema = 0.0
        self._img_front_t = 0.0   # 전방 카메라 최근 수신 시각(디버그: 카메라 살아있는지 나이로 판단)
        self._scan_t       = 0.0  # 라이다 최근 수신 시각(디버그용)

        # ── 인터페이스 변수 (인지 → 판단/제어) ──
        # [2-1 차선]
        self.lane_offset = 0.0      # 근거리 중앙편차(px, 우측+) — [4] 속도계획(turn_preview)에 계속 사용
        self.lane_valid  = False    # 차선 검출 여부
        self.lane_lookahead = 0.0   # 원거리(앞쪽) 편차 → 코너 진입 전 예측감속용
        self.lane_path = []         # 명시적 경로(ROI 픽셀좌표 웨이포인트, 가까운점→먼점)
                                     #   perc_lane()이 갱신, _lane_steer()가 조향각 계산에 사용
        # [2026-08-11] DL 추론 워커의 result_seq(perception/dl_lane.py) 기반 "새 결과 없음"
        # 판정 — perc_lane() 참고. _lane_seq_seen=None은 "아직 한 번도 안 비교함"(백엔드가
        # result_seq를 안 가진 hough/classic_cv에서는 계속 None으로 남아 아래 로직 자체가
        # 스킵된다 — 그 백엔드들은 매 틱 동기 계산이라 애초에 이 문제가 없음).
        self._lane_seq_seen = None
        self._lane_fresh_t = time.time()  # 마지막으로 result_seq가 바뀐 걸 확인한 시각
        self.lane_stale = False     # LANE_STALE_SEC 이상 새 추론 결과가 안 나온 상태 — _lane_drive()가 SPEED_LANE_STALE로 강제 감속
        # [2026-08-17m] lane_stale은 "추론 워커가 완전히 죽었을 때"(result_seq가 안 바뀜)만
        # 잡는다 — 워커는 매 틱 살아서 새 결과를 내는데 그 결과 자체가 계속 무효(lane_valid=
        # False)인 경우(예: 커브 진입부에서 da 밴드 핏이 몇 틱 연속 실패)는 여기 안 걸려서
        # 아무 감속도 안 걸렸다. 실차 재현: 좌회전 진입에서 da 검출이 약 2초간 반복
        # 실패하는 동안 SPEED_CORNER_MIN이 SPEED_NORMAL에 거의 붙어있어(§speed15 프리셋)
        # 감속이 사실상 없었고, 그 블라인드 구간에서 트랙 밖으로 튀어나감(카카오톡 영상
        # 2026-08-17 15:44). lane_valid가 연속으로 False인 프레임 수를 따로 세서
        # _lane_drive()가 LANE_UNSTABLE_FRAMES 이상이면 SPEED_LANE_STALE과 동일한 캡을
        # 걸도록 보강한다 — "워커가 죽었나"가 아니라 "지금 이 프레임을 믿을 수 있나"
        # 자체를 보는 게 원래 목적에 더 맞다.
        self._lane_invalid_streak = 0
        self.lane_unstable = False
        self._lane_prev_width = 448.0  # 도로폭 직전값(px, EMA)
        self.lane_side   = LANE_SIDE   # 현재 주행 차선: +1=우측차선(노란선이 왼쪽) / -1=좌측차선
                                        #   노란 중앙선 위치로 매 프레임 갱신(_update_lane_side)
        # [2-2 신호등] S0(출발)/S2(교차로) 공통 — 둘 다 같은 4구 신호등을 본다(대회 규정 변경)
        self.signal_red_on      = False  # 빨강 (단일 프레임 순간값, 디바운스 안 됨)
        self.signal_straight_on = False  # 직진(=초록만 점등) 순간값
        self.signal_left_on     = False  # 좌회전(=초록+빨강 동시 점등) 순간값
        # FSM은 반드시 아래 confirmed 값만 봐야 한다(디바운스 통과분) — 순간값은 빛반사·블러로
        # 한 프레임만 튈 수 있어 그대로 쓰면 오출발/오좌회전 위험이 있다.
        self.signal_straight_confirmed = False
        self.signal_left_confirmed     = False
        self._sig_straight_cnt = 0   # signal_straight_on 연속 유지 프레임 수
        self._sig_left_cnt     = 0   # signal_left_on 연속 유지 프레임 수
        self.stopline = False            # 굵은 가로 흰선(정지선/지름길 끝 단서)
        self._stopline_cooldown_t = 0.0  # 이 시각까지 정지선 재감지 무시
        self._last_stopline_t = 0.0      # [10] 정지선을 마지막으로 본 시각(교차로 근처 기동 금지용)
        # [2-3 장애물(전방/측면)]
        self.obstacle_front = False   # 전방 장애물
        self.obstacle_dist  = 999.0   # 전방 거리(m)
        self.obstacle_side  = 'none'  # 'left'/'right'/'center'/'none'
        self.obstacle_type  = 'none'  # 'fixed'/'vehicle'/'none' (라이다 점수로 판별)
        self.left_clear     = True    # 좌측 차선 비었는지(순간값, 디버그 표시용)
        self.right_clear    = True    # 우측 차선 비었는지(순간값, 디버그 표시용)
        # [2026-08-13] choose_side()가 실제로 받는 디바운스된 값 — SIDE_CLEAR_CONFIRM_FRAMES
        # 연속 프레임 "비었음"이 유지돼야 True로 확정되고, 한 프레임이라도 "막힘"이 나오면
        # 즉시 False로 리셋된다(비대칭 디바운스 — config.py SIDE_CLEAR_CONFIRM_FRAMES 주석 참고).
        self.left_clear_confirmed  = True
        self.right_clear_confirmed = True
        self._left_clear_cnt  = 0
        self._right_clear_cnt = 0
        # [2026-08-13] 판정(임계값 비교) 이전에 원본 점개수 자체를 먼저 EMA로 스무딩한다 —
        # 바로 아래 _ema_y(장애물 좌우 위치)와 같은 패턴을 left_cnt/right_cnt에도 적용한 것.
        # 히스테리시스/디바운스가 "판정 이후" 안정화라면, 이건 "판정 이전" 안정화다.
        self._left_cnt_ema  = 0.0
        self._right_cnt_ema = 0.0
        self._ema_y         = 0.0     # 전방 장애물 횡위치 EMA(obstacle_side 안정화)
        self.obstacle_width = 0.0     # [6] 전방 장애물 실측 횡폭(m) — 거리무관 분류 기준
        self._obstacle_last_seen = 0.0  # [9] 마지막으로 실제 검출된 시각(인식 끊김 보상)
        # ── 추월 판단용 타겟 정보 (가장 가까운 전방 클러스터 기준) ──
        self.obstacle_y     = 0.0     # 타겟 횡중심(m, +=좌). 어느 쪽으로 비킬지의 1차 근거
        self.obstacle_y_min = 0.0     # 타겟 우측 끝(m)
        self.obstacle_y_max = 0.0     # 타겟 좌측 끝(m)
        self.obstacle_rate  = 0.0     # 접근율(m/s, 음수=접근). 추돌 방지·교차확인용
        self._obstacle_prev_dist = None
        self._obstacle_prev_t    = 0.0
        # [2026-08-14] 회피 "복귀 유예"(avoid-hold) — _update_avoid_hold()가 perc_obstacle()
        # 직후 갱신하고, perc_lane()이 DL 차선인식 백엔드로 그대로 넘긴다(config.py
        # AVOID_HOLD_TRIGGER_DIST_M/AVOID_HOLD_SEC_* 주석, README §2.32/§2.33 참고).
        self._avoid_hold_until_t = 0.0  # 이 시각까지는 avoid_hold_active=True
        self.avoid_hold_active   = False
        # [2026-08-15] avoid-hold 개선 — avoid_hold_improvement_proposal.md "1차 적용 결정"
        # 적용1(가변 유예시간+거리기반 조기해제)/적용3(방향 힌트) 상태. 전부
        # _update_avoid_hold()가 매 틱 갱신하고, _debug_viz_avoid_hold()가 그대로 보여준다.
        self.avoid_hold_hold_sec = AVOID_HOLD_SEC_BASE  # 이번 유예 구간에 쓰이는 hold_sec(트리거 시점 스냅샷)
        self.avoid_hold_side = 0             # choose_side() 결과(-1/0/+1) — 0이면 "양쪽 다 막힘"(적용4 안전판 트리거)
        self.avoid_hold_release_reason = ''  # 직전 유예가 끝난 사유: 'timeout'/'early_dist'/''(아직 없음)
        self._avoid_hold_release_cnt = 0            # obstacle_front=False 연속 프레임(조기해제 디바운스)
        self._avoid_hold_last_valid_dist = 999.0    # 마지막으로 obstacle_front=True였던 순간의 obstacle_dist
        self._avoid_hold_target_speed_est = 0.0     # 트리거 시점 target_speed_est 스냅샷(디버그 표시용)
        # [2-4 라바콘]
        self.lavacon_offset = 0.0    # 디버그/로깅용(중심선 y평균) — 조향엔 더 이상 안 씀
        self.lavacon_done   = False
        self.lavacon_path   = []     # _lane_steer()에 그대로 태우는 px 스케일 경로(perc_lavacon() 참고)
        self._lavacon_path_m = []    # 위와 같은 경로의 원본(라이다 미터 좌표) — DEBUG_VIZ_LAVACON 시각화용
        self._lavacon_empty_cnt = 0   # 우측콘 연속 미검출 프레임 수(Phase 전환 디바운스)
        self.lavacon_left_detected  = False  # 좌측 라이다 클러스터 검출 여부(B1 진입 트리거용)
        self.lavacon_right_detected = False  # 우측 라이다 클러스터 검출 여부(B1 진입 트리거용)
        self.cone_detected_yolo     = False  # YOLO 카메라 콘 검출 여부(B1 진입 트리거 이중확인용)
        self.lavacon_trigger        = False  # (YOLO 검출 AND 좌우 라이다 동시검출)이 디바운스 프레임수만큼 유지되면 True
        self._lavacon_trigger_cnt   = 0      # 동시검출 연속 프레임 수(디바운스 카운터)
        self._lavacon_dbg = (0, 0, 0, 0)     # 디버그용 (좌ROI점수, 좌최대연속묶음, 우ROI점수, 우최대연속묶음)
        self._lavacon_mask_dbg = (0, -1.0)   # 디버그용 (BODY_LO~HI 마스킹 구간 원본 점수, 최소거리)
        # [2-6 방해차량 트리거]
        self.vehicle_trigger       = False   # 라이다 디바운스 통과 → B3 진입 트리거
        self._vehicle_trigger_cnt  = 0       # 동시검출 연속 프레임 수(디바운스 카운터)
        # [2-7 장애물 위치 판단]
        self.lane_center   = 320.0           # 차선 중앙 x좌표(px) — 첫 카메라 프레임 전까지 화면 중앙 기본값

        # ── 외부 차선 인식 모듈 초기화 (LANE_DETECTOR_BACKEND로 선택, 인터페이스는 셋 다 동일) ──
        self.lane_detector = self._build_lane_detector(LANE_DETECTOR_BACKEND)
        self.signal_detector = SignalDetector()          # 신호등(3구/4구) Hough Circle 인식기

        # 라바콘 카메라 이중확인용 YOLO 콘 검출기. onnxruntime 미설치/모델 파일 부재 등으로
        # 초기화가 실패하면 _build_lane_detector()의 dl→hough 폴백과 달리 대체 백엔드가
        # 없으므로(카메라 이중확인 자체가 선택사항), None으로 두고 perc_lavacon_trigger()가
        # "카메라 확인 불가 시 라이다 단독 판정으로 폴백"하도록 한다 — 원인은 에러 로그로 남긴다.
        try:
            self.yolo_cone_detector = YoloConeDetector(logger=self.get_logger())
        except Exception as e:
            self.get_logger().error(
                f'YOLO 콘 검출기 초기화 실패, 라바콘 트리거는 라이다 단독 판정으로 폴백합니다: {e}'
            )
            self.yolo_cone_detector = None

        # ── 판단/제어 상태 ──
        self.mission_state  = START_STATE
        self.behavior_state = BehaviorState.B0_NORMAL
        self.phase          = Phase.LAVACON     # S1 내부 진행 순서(라바콘부터 시작)
        # [2026-08-15] Phase.OBSTACLE_ZONE 통합(da_based_b2b3_proposal.md B안) —
        # B2/B3 각각 최소 한 번 완료됐는지 추적. 둘 다 True가 돼야 Phase.DONE으로
        # 넘어간다(_mark_behavior_passed() 참고) — 순서를 안 따지므로 어느 쪽이 먼저
        # 끝나도 상관없다.
        self._b2_passed = False
        self._b3_passed = False
        self._behavior_enabled = TEST_FORCE_BEHAVIOR  # 원래 S2 교차로 "직진"으로 S1 재진입 시에만 True
                                                       #   (TEST_FORCE_BEHAVIOR=True면 라바콘 단독 테스트용으로 시작부터 강제 ON)
        self._lavacon_engaged  = False          # B1_LAVACON 진입 확정 latch (트리거 이후 잠깐 한쪽 클러스터가
                                                 #   끊겨도 중간에 일반주행으로 안 튀도록 유지, lavacon_done으로 해제)
        self.ctrl_angle = 0.0
        self.ctrl_speed = SPEED_STOP
        self._prev_angle_out = 0.0    # [5] 직전 발행 조향각(변화율 제한용)
        self._pid_prev_error = 0.0
        self._pid_integral   = 0.0
        self._turn_yaw_start = None   # 좌회전 진행 중 플래그 (None=미회전)
        self._turn_frame_cnt = 0      # 좌회전 경과 프레임 수
        self._s2_commit_dist = None   # S2 신호 확정 후 물리적 분기 커밋 구간 누적 이동거리(m, None=미진입)
        self._s2_commit_dir  = None   # 커밋 구간에서 진행 중인 방향 ('straight'/'left')
        self._approach_t0    = None   # [진입] 정지선 감지 후 감속 시작 시각
        self._exit_approach_t0 = None # [진출] S3 탈출 정지선 감지 후 감속 시작 시각
        self._shortcut_t0    = None   # 지름길 진입 시각(끝감지 타이밍용)
        self._shortcut_ref_yaw = None # S3 진입 1초 후 기록한 기준 yaw (탈출 좌회전 전 보정용)
        self._prev_speed     = 0.0    # 가속 속도제한용(직전 출력 속도)
        self._corner_hold    = 0.0    # 코너 활성도(감쇠 peak-hold)
        self._corner_signal  = 0.0    # 코너 감속 판단용 조향각 signed EMA(부호 유지, _lane_drive() 참고)
        self._last_debug_t   = 0.0

        # ── 바퀴 카운트 상태 ──
        self.lap = 1                    # 현재 몇 바퀴째 (출발 직후 = 1바퀴째 주행중)
        self._yaw_accum = 0.0           # 이번 바퀴 시작 이후 누적 yaw(rad, 부호 유지)
        self._prev_yaw_accum_ref = None # 누적 계산용 직전 yaw (첫 호출 때 초기화)
        self._lap_t0 = time.time()      # 이번 바퀴 시작 시각
        self._lap_stopline_used = False # 이번 바퀴에서 정지선으로 이미 확정했는지
        self._lap_yaw_over_cnt = 0      # 누적 임계 초과 연속 프레임 수(디바운스)

        # 전방 차량 통과 컨트롤러 — 두 미션이 같은 기동이라 같은 클래스를 쓰고
        # moving 플래그로 재평가 강도만 다르게 한다.
        #   B2 고정장애물 = '고장난 차량'(정지)  → moving=False
        #   B3 방해차량   = 느리게 주행하며 차선을 오감 → moving=True
        self.obstacle_controller = ObstacleAvoidance(moving=False)
        self.vehicle_controller  = ObstacleAvoidance(moving=True)

        # OccupancyGrid의 angle_offset_deg/body_lo/body_hi는 perc_obstacle()의
        # LIDAR_ANGLE_OFFSET_DEG(88행 부근), BODY_LO/BODY_HI(395행 부근)와 반드시
        # 같은 값이어야 한다 — 라이다 장착 보정이 여기만 빠지면 장애물이 실제와
        # 다른 각도에 찍힌다.
        self.occupancy = OccupancyGrid(
            resolution=0.1, width=10, height=10,
            angle_offset_deg=LIDAR_ANGLE_OFFSET_DEG,
            body_lo=215, body_hi=305
        )
        self.planner = HybridAStar()
        self.stanley = StanleyController()

        # 차선 세그멘테이션 경로(self.lane_path) 추종용 — _lane_pid()(PID)를 대체.
        # _lane_pid()는 B2/B3 장애물회피 behavior(apply_behavior_override())가 여전히
        # 쓰므로 그대로 남겨둔다 — 없앤 게 아니라 "일반 차선주행" 용도에서만 교체한 것.
        # 튜닝값은 전부 config.py의 PP_* 에서 가져온다 — 클래스 자체의 기본값은 config.py를
        # 안 거치고 pure_pursuit.py를 직접 쓸 때(단독 테스트 등)를 위한 fallback이라,
        # 여기서 명시적으로 넘기지 않으면 config.py를 고쳐도 반영이 안 된다.
        # [2026-08-14] LQR 컨트롤러(self.lqr)와 그 사이를 고르던 STEERING_CONTROLLER는
        # 실차 미검증 상태로 한 번도 켜본 적 없어 코드베이스에서 제거했다 — config.py
        # section 4 주석 참고.
        self.pure_pursuit = PurePursuitController(
            lookahead_base_px=PP_LOOKAHEAD_BASE_PX,
            lookahead_speed_gain=PP_LOOKAHEAD_SPEED_GAIN,
            lookahead_max_px=PP_LOOKAHEAD_MAX_PX,
            wheelbase_px=PP_WHEELBASE_PX,
            angle_max_deg=ANGLE_MAX,
            alpha=PP_ALPHA,
            min_lookahead_px=PP_MIN_LOOKAHEAD_PX,
            dx_deadzone_px=PP_DX_DEADZONE_PX,
            lookahead_curvature_gain=PP_LOOKAHEAD_CURVATURE_GAIN,
            lookahead_min_px=PP_LOOKAHEAD_MIN_PX,
            straight_curvature_eps=PP_STRAIGHT_CURVATURE_EPS,
            straight_confirm_frames=PP_STRAIGHT_CONFIRM_FRAMES,
            straight_deadzone_px=PP_STRAIGHT_DEADZONE_PX,
            straight_alpha=PP_STRAIGHT_ALPHA,
            straight_bias_ema_alpha=PP_STRAIGHT_BIAS_EMA_ALPHA,
        )

        self.path = None
        self.grid = None
        self.goal = None

        # ── B3(방해차량) hybrid A* 대안 전용 상태 (USE_HYBRID_ASTAR_FOR_B3=True일 때만 의미있음) ──
        #   self.path/self.goal은 B2와 공유(두 Phase가 동시에 활성화되지 않으므로 안전).
        #   아래는 B3만의 재계획 트리거/폴백 상태라 별도로 둔다 — _handle_overtake_astar() 참고.
        self._b3_tick = 0          # 마지막 전체 재탐색 후 경과 틱 (ASTAR_B3_REPLAN_TICKS 주기용)
        self._b3_switch_cnt = 0    # 타겟이 통과방향쪽으로 넘어온 연속 프레임 (SWITCH_FRAMES 트리거용)
        self._b3_fail_cnt = 0      # 연속 탐색실패 프레임 (ASTAR_B3_FAIL_GRACE_TICKS 초과시 폴백)
        self._b3_side = 0          # 현재 채택된 통과방향(-1/0/+1), TargetPassing.side와 동일 부호규약
        self._b3_using_fallback = False  # True면 이번 통과가 끝날 때까지 TargetPassing으로 고정 위임


        # pose 변수 - 추후 수정(정식 오도메트리 연동 전까지는 _handle_fixed_obstacle에서
        # replan 시점을 원점으로 삼아 IMU yaw 실측값 + 명령속도 적분으로 근사 추적)
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_yaw = 0.0
        self.vehicle_speed = 0.0
        self._plan_ref_yaw = 0.0
        self._plan_last_t = 0.0

        # ── VESC 기반 실측 속도 (엔코더 대체, 2026-08-06 LQR 브랜치의 ROS1 연동 작업에서 이식) ──
        #   cb_vesc()가 '/vesc_speed_erpm'(ROS1 launch/vesc_speed_bridge.py가 중계)을 받을 때마다
        #   갱신한다. 그 브리지 노드가 안 떠 있거나 아직 메시지를 한 번도 못 받았으면 0.0으로
        #   유지된다 — _speed_for_lookahead()가 VESC_MIN_SPEED_MPS 가드로 이 상태를 걸러내고
        #   self._prev_speed(명령속도)로 폴백한다(cb_vesc()/VESC_MIN_SPEED_MPS 주석 참고).
        self.v_mps = 0.0
        self._vesc_t = None

        # ── 엔코더(VESC) 기반 pose 추정기 (localization/pose_estimator.py) ──
        #   위 self.vehicle_x/y/yaw(플래너용, 명령속도 적분 근사)와는 별개 컴포넌트. wheelbase_m은
        #   2026-08-06 실측값(config.py WHEELBASE_M, 옛 이름 LQR_WHEELBASE_M). v_mps는 cb_vesc()가
        #   갱신하는 self.v_mps를 control_loop()에서 매 주기 넣어준다. IMU를 yaw 소스로 쓰려면
        #   set_yaw_source('imu') 후 update(..., imu_yaw=self.imu_yaw).
        self.pose_estimator = EncoderPoseEstimator(wheelbase_m=WHEELBASE_M)


        # ── ROS 통신 ──
        self.motor_msg = Float32MultiArray()
        self.motor_pub = self.create_publisher(Float32MultiArray, 'xycar_motor', 10)
        image_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image,     '/usb_cam/image_raw/front',  self.cb_img_front,  image_qos)
        self.create_subscription(Image,     '/usb_cam/image_raw/left',   self.cb_img_left,   image_qos)
        self.create_subscription(Image,     '/usb_cam/image_raw/right',  self.cb_img_right,  image_qos)
        self.create_subscription(Image,     '/usb_cam/image_raw/behind', self.cb_img_behind, image_qos)
        self.create_subscription(LaserScan, '/scan',                     self.cb_scan,       qos_profile_sensor_data)
        self.create_subscription(Imu,       '/imu',                      self.cb_imu,        qos_profile_sensor_data)
        # VESC 실측 속도(ERPM, 2026-08-06 LQR 브랜치에서 이식) — launch/vesc_speed_bridge.py
        # (ROS1, noetic_ws에 별도 배치)가 /sensors/core의 state.speed만 뽑아 std_msgs/Float32로
        # 다시 뿌린 '/vesc_speed_erpm'을 구독한다(커스텀 메시지 vesc_msgs를 이 ROS2 워크스페이스에
        # 안 깔아도 되게 하려는 우회 — config.py "VESC 실측 속도 연동" 절 참고). 표준 메시지라
        # import 실패 걱정 없이 항상 구독 가능.
        self.create_subscription(Float32,   '/vesc_speed_erpm',          self.cb_vesc,        qos_profile_sensor_data)
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info(f'초기화 완료 | 시작={START_STATE.name}')


    # #########################################################
    # [1] 통신 I/O    담당: 공통(수정 X)
    # #########################################################
    def cb_img_front(self, msg):
        try:
            self.img_front = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._img_front_t = time.time()
            self.get_logger().info(f'[front] 첫 수신 OK enc={msg.encoding} shape={self.img_front.shape}', once=True)
        except Exception as e:
            self.get_logger().error(f'[front] 이미지 변환 실패 enc={msg.encoding}: {e}', throttle_duration_sec=2.0)
    def cb_img_left(self, msg):   self.img_left   = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    def cb_img_right(self, msg):  self.img_right  = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    def cb_img_behind(self, msg): self.img_behind = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    def cb_scan(self, msg):       self.lidar_ranges = msg.ranges; self._scan_t = time.time()
    def cb_imu(self, msg):
        q = msg.orientation
        self.imu_yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.imu_yaw_rate = msg.angular_velocity.z
        self._imu_t = time.time()

    def cb_vesc(self, msg):
        """'/vesc_speed_erpm'(std_msgs/Float32) — launch/vesc_speed_bridge.py(ROS1)가 VESC
        드라이버의 state.speed(ERPM, 전기적 회전수/분)를 그대로 실어 보낸 것. 이 로봇의 사실상
        엔코더. VESC_SPEED_TO_ERPM_GAIN(config.py, vesc.yaml의 speed_to_erpm_gain 실측값)으로
        나눠야 실제 선속도(m/s)가 된다. offset 보정은 하지 않음(vesc_to_odom.cpp 참고 — 기본
        offset=0.0이라 보정해도 차이가 없다고 판단, 실차에서 정지 상태 값이 0에 가깝지 않으면
        여기에 offset 보정을 추가할 것)."""
        try:
            self.v_mps = float(msg.data) / VESC_SPEED_TO_ERPM_GAIN
            self._vesc_t = time.time()
        except Exception as e:
            self.get_logger().error(f'[vesc] /vesc_speed_erpm 파싱 실패: {e}', throttle_duration_sec=2.0)

    def drive(self, angle, speed):
        # ROS1 xycar_motor.py 노드가 XycarMotor 대신 Float32MultiArray(data=[angle, speed])를
        # 구독하도록 이미 전환되어 있음(ros1_bridge가 커스텀 XycarMotor 타입을 못 넘기는 문제 우회).
        clipped_angle = float(np.clip(angle, -ANGLE_MAX, ANGLE_MAX))
        # [5] 변화율 제한: 크기(clip)만으로는 한 주기에 0도→-60도 같은 계단 명령이 그대로 나간다
        #     (_do_left_turn 의 즉시 대입, B0→B1 전환, Stanley 출력 교체 등에서 실제로 발생).
        #     서보 기계부하와 제어 오실레이션을 동시에 줄인다.
        clipped_angle = float(np.clip(clipped_angle,
                                      self._prev_angle_out - ANGLE_RATE_MAX,
                                      self._prev_angle_out + ANGLE_RATE_MAX))
        self._prev_angle_out = clipped_angle
        clipped_speed = float(np.clip(speed, -100.0, 100.0))
        self.motor_msg.data = [clipped_angle, clipped_speed]
        for _ in range(7):
            self.motor_pub.publish(self.motor_msg)


    def _build_lane_detector(self, backend):
        """LANE_DETECTOR_BACKEND 값에 따라 차선인식 백엔드 객체를 만든다. 셋 다
        detect(frame) -> (valid, offset, lookahead, lane_center, path, debug_img) 인터페이스가
        동일하므로 perc_lane()/_update_lane_side()는 어떤 백엔드가 골라졌는지 몰라도 된다."""
        if backend == 'classic_cv':
            detector = ClassicLaneDetector()
            detector.set_processor(CameraProcessor(), SlideWindow())
            return detector

        if backend == 'dl':
            try:
                return DLLaneDetector(logger=self.get_logger())
            except Exception as e:
                # onnxruntime 미설치, models/twinlitenetplus_kmu_v1.2.0.onnx(.data) 부재 등으로 초기화가 실패하면
                # 원인을 명확히 남기고 검증된 백엔드(hough)로 폴백한다 — 조용히 무시하지 않는다.
                self.get_logger().error(
                    f'DL 차선인식 백엔드 초기화 실패, hough로 폴백합니다: {e}'
                )
                return HoughLaneDetector()

        return HoughLaneDetector()

    # #########################################################
    # [2] 인지 (Perception)
    # #########################################################
    def perceive_all(self):
        self.perc_lane()        # 비전
        self.perc_signal()      # 비전
        self.perc_obstacle()    # 라이다
        self._update_avoid_hold()  # 라이다(위 obstacle_front/dist 기반) — perc_obstacle() 직후여야 함
        self.perc_lavacon()     # 라이다
        self.perc_yolo_cone()   # 비전 (YOLO, perc_lavacon_trigger()가 라이다와 AND 결합해서 씀)
        self.perc_lavacon_trigger()  # 라이다+비전 (YOLO 콘 검출 AND 좌우 클러스터 동시검출 → B1_LAVACON 진입 트리거)
        self.perc_vehicle_trigger()  # 라이다 (전방 장애물 근접 → B3_VEHICLE 진입 트리거)
        self.perc_stopline()    # 비전

    # [2-4a] 라바콘 카메라 이중확인 (YOLO)
    #   입력 self.img_front → 출력 self.cone_detected_yolo
    #   yolo_cone.py가 별도 스레드에서 자기 페이스로 추론하므로 여기선 논블로킹으로
    #   최신 결과만 받아온다(dl_lane.py의 perc_lane()과 동일한 패턴).
    def perc_yolo_cone(self):
        if self.yolo_cone_detector is None:
            # 초기화 실패 상태 — perc_lavacon_trigger()가 이 경우 라이다 단독 판정으로
            # 폴백하므로 여기선 그냥 False로 둔다(카메라 확인 "안 됨"이 아니라 "못 함").
            self.cone_detected_yolo = False
            return
        if self.img_front is None:
            return
        self.cone_detected_yolo = self.yolo_cone_detector.detect(self.img_front)
        self.yolo_cone_detector.show_debug_windows()  # 메인 스레드에서만 호출(yolo_cone.py 주석 참고)

    # [2-1] 차선
    #   입력 self.img_front → 출력 self.lane_offset(우측+), self.lane_valid
    def perc_lane(self):
        if self.img_front is None:
            self.lane_valid = False
            self.lane_stale = True   # 카메라 프레임 자체가 아직/더 이상 없음 — 당연히 신선하지 않음
            self._lane_invalid_streak += 1
            self.lane_unstable = self._lane_invalid_streak >= LANE_UNSTABLE_FRAMES
            return

        # [2026-08-14] avoid-hold(§2.32) 상태를 이번 detect() 호출 전에 DL 백엔드로 넘긴다 —
        # hough/classic_cv처럼 이 메서드가 없는 백엔드는 getattr가 조용히 no-op을 반환해
        # 건너뛴다(show_debug_windows()와 동일 관례). perceive_all()에서 perc_lane()이
        # perc_obstacle()/_update_avoid_hold()보다 먼저 도는 순서라 여기 값은 엄밀히는
        # 직전 틱 기준(0.05s 이내 오차)이다 — DL 추론 자체도 이미 논블로킹 백그라운드
        # 워커라 결과가 한두 프레임 지연되는 걸 감안하고 설계됐으므로(모듈 상단 주석)
        # 무시 가능한 오차로 판단.
        # [2026-08-15] 적용3 — avoid_hold_side(방향 힌트, -1/0/+1)도 같이 넘긴다
        # (config.py AVOID_HOLD_DIR_BIAS_PX 주석, perception/dl_lane.py _clip_da_by_ll() 참고).
        getattr(self.lane_detector, 'set_avoid_hold', lambda *_a, **_k: None)(
            self.avoid_hold_active, self.avoid_hold_side)
        # [2026-08-17g] 현재 속도(m/s)도 같이 넘긴다 — DL 백엔드의 da 안전마진이 방해차량
        # "뒤" 방향 추가 마진을 속도에 비례해 늘리는 데 쓴다(perception/dl_lane.py
        # set_speed()/config.py DL_DA_REAR_MARGIN_* 주석 참고). hough/classic_cv처럼 이
        # 메서드가 없는 백엔드는 set_avoid_hold()와 동일하게 getattr로 조용히 건너뛴다.
        getattr(self.lane_detector, 'set_speed', lambda *_a, **_k: None)(self.v_mps)

        # hough_lane.py의 HoughLaneDetector를 사용하여 차선 인식 수행
        valid, offset, lookahead, lane_center, path, debug_img = self.lane_detector.detect(self.img_front)
        # DL 백엔드는 추론이 별도 스레드에서 도는데, cv2.imshow()/waitKey()는 스레드
        # 세이프하지 않아 반드시 메인 스레드(여기, control_loop 타이머 콜백)에서만 호출해야
        # 한다(dl_lane.DLLaneDetector.show_debug_windows() 주석 참고). hough/classic_cv
        # 백엔드는 이 메서드가 없으므로 getattr로 조용히 건너뛴다.
        #   속도 적응형 look-ahead 목표점(pure_pursuit.py last_target_xy)도 같이 넘겨서
        #   result 패널에 찍는다 — self._lane_steer()가 이번 틱에 아직 안 돌았으므로
        #   엄밀히는 직전 틱 값(0.05s 이내 오차, 디버깅 목적엔 무시 가능).
        lookahead_xy = self.pure_pursuit.last_target_xy
        lookahead_px = self.pure_pursuit.last_lookahead_px
        # [2026-08-17d] 직전 틱의 직진/커브대응 상태도 같이 넘겨서 result 패널에 표시한다
        # (dl_lane.DLLaneDetector.show_debug_windows() 주석 참고) — lookahead_xy/px와 동일하게
        # 한 틱(0.05s) 지연 가능.
        is_straight = self.pure_pursuit.is_straight
        # [2026-08-17g] dl_lane 창 맨 아래 yellow 패널이 속도+커브대응 상태 패널로
        # 바뀌면서(perception/dl_lane.py show_debug_windows() 참고) v_mps도 같이 넘긴다.
        getattr(self.lane_detector, 'show_debug_windows', lambda *a, **k: None)(
            lookahead_xy, lookahead_px, is_straight, self.v_mps)

        # [2026-08-11] "재사용된 최신값"과 "완전히 안 갱신됨"을 구분 — DLLaneDetector가
        # 추론 1회 끝날 때마다 올리는 result_seq(dl_lane.py 참고)가 직전 틱에서 본 값과
        # 같으면 이번 틱도 같은 결과를 다시 받았다는 뜻이다. 그게 LANE_STALE_SEC 넘게
        # 계속되면(=추론 워커가 죽었거나 카메라가 끊겼거나) lane_stale=True로 표시한다.
        # hough/classic_cv처럼 result_seq가 없는(=매 틱 동기 계산이라 항상 새 값인) 백엔드는
        # getattr가 None을 반환해 이 판정 자체를 건너뛰고 항상 fresh로 취급한다.
        seq = getattr(self.lane_detector, 'result_seq', None)
        if seq is not None:
            if seq != self._lane_seq_seen:
                self._lane_seq_seen = seq
                self._lane_fresh_t = time.time()
            self.lane_stale = (time.time() - self._lane_fresh_t) >= LANE_STALE_SEC
        else:
            self.lane_stale = False

        self.lane_center = lane_center
        self.lane_valid = valid
        self._lane_invalid_streak = 0 if valid else self._lane_invalid_streak + 1
        self.lane_unstable = self._lane_invalid_streak >= LANE_UNSTABLE_FRAMES
        if valid:
            # 기존 제어 코드와 호환되도록 필터링 적용
            self.lane_offset = 0.7 * self.lane_offset + 0.3 * offset
            self.lane_lookahead = 0.5 * self.lane_lookahead + 0.5 * lookahead
        # [2026-08-10] `if path:`만 보던 예전 조건은 디바운스가 전혀 없어서, 밴드 판정이
        # 프레임마다 흔들릴 때 그 흔들림을 거의 그대로 조향에 전달했다 — 그래서 `valid`도
        # 같이 요구하도록 바꿨었다(offset과 동일한 안정성 검증).
        # [2026-08-17n, §2.36 재발 수정] 그런데 `valid`(=lane_valid, 근접 밴드 필수)를
        # 그대로 쓰면 급커브 진입처럼 근접만 일시적으로 안 보이는 구간에서 원거리 정보가
        # 있어도 경로 자체가 못 갱신되는 문제가 실차로 확인됐다(perception/dl_lane.py
        # DLSlideWindow._debounce_path_ok() 주석 참고). `path_ok`는 근접 OR 원거리 중
        # 하나만 있어도 통과하는 별도 신호를, `valid`와 동일한 디바운스 강도로 걸러낸
        # 것 — dl_lane.py 내부 self.path 갱신도 이제 이 값으로 가드하므로(같은 값을
        # 안팎이 같이 봄) §2.36의 "내부/외부 조건 불일치로 인한 점프" 크래시가 재발할
        # 여지는 없다. hough/classic_cv처럼 이 속성이 없는 백엔드는 getattr가 `valid`로
        # 폴백해 기존 동작 그대로다.
        path_ok = getattr(self.lane_detector, 'path_ok', valid)
        if path_ok and path:
            self.lane_path = path

        self._update_lane_side()

    def _update_lane_side(self):
        """노란 중앙선이 화면 어느 쪽에 있는지로 '지금 어느 차선에 있는지'를 판정한다.

        코스는 차선 2개 + 가운데 점선 노란선 + 바깥 흰 실선 구조다. 회피는 항상
        노란선을 넘어 반대편 차선으로 가야 하므로(흰 실선은 넘으면 안 됨),
        회피 방향은 '장애물이 어디 있나'가 아니라 '내가 어느 차선에 있나'로 정해진다.
          노란선이 화면 왼쪽  → 나는 우측 차선 (+1) → 회피는 왼쪽으로
          노란선이 화면 오른쪽 → 나는 좌측 차선 (-1) → 회피는 오른쪽으로
        노란선이 안 보이는 프레임은 직전 판정을 유지한다(점선이라 끊기는 구간이 있음).
        """
        ld = self.lane_detector
        centers = getattr(ld, 'yellow_centers', None)
        if not centers or not ld.roi_w:
            return
        xs = [c[1] for c in centers if c is not None]
        if not xs:
            return
        # [2026-08-17] _lane_steer()와 동일하게 vehicle_center_x가 있으면 그걸 기준으로
        # 좌/우를 가른다(roi_w/2.0은 그 속성이 없는 백엔드용 폴백).
        vehicle_center_x = getattr(ld, 'vehicle_center_x', None)
        if vehicle_center_x is None:
            vehicle_center_x = ld.roi_w / 2.0
        self.lane_side = 1 if (sum(xs) / len(xs)) < vehicle_center_x else -1

    # [2-2] 신호등
    #   입력 self.img_front
    #   출력 signal_red/straight/left_on (S0/S2 공통)
    #   주의 4구는 직진·좌회전 모두 초록 → 점등 '위치'로 구분
    def perc_signal(self):
        """신호등 판별 — traffic_signal.py의 SignalDetector.detect_s2()(4구 Hough Circle)에 위임.
          대회 규정 변경으로 S0(출발)도 S2(교차로)와 동일한 4구 신호등을 재사용한다:
            S0 → signal_straight_confirmed(초록만 점등) = 출발
            S2 → signal_straight_confirmed = 직진, signal_left_confirmed(초록+빨강 동시) = 좌회전
        detect_s2()는 원 4개가 정확히 안 잡히면(초과분은 pick_best_4()로 어느 정도 흡수하지만,
        미달은 흡수 불가) 그 프레임은 인식 실패로 순간값이 False가 될 수 있다. 여기서
        SIG_CONFIRM_FRAMES 연속 유지를 확인해 confirmed로 승격시켜, 단발성 오검출/오검출실패가
        바로 FSM 전환(출발/좌회전)으로 새는 걸 막는다(라바콘/차량 트리거와 동일한 패턴).
        DEBUG_LOG_SIGNAL=True면(기본값) 매 프레임 _log_signal_debug()로 실패 원인+힌트를 찍는다
        — DEBUG_VIZ_SIGNAL(창)과 별개 스위치라 터미널 로그만 원하면 이것만 켜도 됨."""
        if self.img_front is None:
            return

        if self.mission_state in (MissionState.S0_WAIT_GREEN, MissionState.S2_INTERSECTION):
            self.signal_red_on, self.signal_straight_on, self.signal_left_on = \
                self.signal_detector.detect_s2(self.img_front)

            self._sig_straight_cnt = self._sig_straight_cnt + 1 if self.signal_straight_on else 0
            self._sig_left_cnt     = self._sig_left_cnt + 1 if self.signal_left_on else 0
            self.signal_straight_confirmed = self._sig_straight_cnt >= SIG_CONFIRM_FRAMES
            self.signal_left_confirmed     = self._sig_left_cnt >= SIG_CONFIRM_FRAMES

    # [2026-08-13] mask(불리언 배열)에서 True인 인덱스들 중 가장 긴 "연속(인접 인덱스)"
    # 구간의 길이를 구한다. 전방 장애물 그룹핑(perc_obstacle()의 fidx/groups, np.split을
    # np.diff(fidx)>1 지점마다 나누는 방식)과 완전히 동일한 로직 — 흩어진 노이즈 점들과
    # 진짜 붙어있는 물체(연속된 각도에서 찍힘)를 구분하려고 좌/우 판정에도 같은 방식을
    # 적용한다. 단순 개수와 달리, 노이즈가 여기저기 떨어져서 총 개수가 우연히 많아져도
    # 연속 구간 자체는 짧게 나와 "막힘"으로 잘못 판정되지 않는다.
    def _largest_run(self, mask):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return 0
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        return max(len(g) for g in groups)

    # [2-3] 장애물(전방+측면)
    #   입력 self.lidar_ranges
    #   출력 obstacle_front/dist/side, left_clear, right_clear
    def perc_obstacle(self):
        # ── 튜닝 파라미터 ──
        FRONT_X_MIN, FRONT_X_MAX = 0.0, 5.0   # 전방 ROI 종방향(m)
        FRONT_Y_HALF             = 1.5         # 전방 ROI 횡방향 반폭(m)
        FRONT_MIN_PTS            = 2           # 전방 장애물 확정 최소 포인트
        FRONT_VEHICLE_PTS        = 12          # 이 이상이면 차량, 미만이면 고정장애물
        SIDE_X_MIN, SIDE_X_MAX   = 0.8, 5.5   # 측면 ROI 종방향(m)
        LEFT_Y_MIN,  LEFT_Y_MAX  = 0.7, 1.5   # 좌측 ROI 횡방향(m)
        RIGHT_Y_MIN, RIGHT_Y_MAX = 0.7, 1.5   # 우측 ROI 횡방향(m)
        # [2026-08-13] 히스테리시스(이중 임계값, Schmitt trigger) — 점 개수가 임계값 바로
        # 근처에서 오락가락하면 단일 임계값으로는 self.left_clear가 매 프레임 뒤집힌다.
        # '비었음→막힘'과 '막힘→비었음' 전환에 서로 다른 임계값을 써서, 일단 어느 상태로
        # 들어가면 그 경계를 확실히 넘어야만 반대로 넘어가게 한다(아래 계산부 참고).
        # _LOW는 실차 미검증 첫 추정치 — 상단(_BLOCK_TH)의 절반으로 잡았다.
        LEFT_BLOCK_TH            = 8           # 좌측: '비었음→막힘' 전환 임계 (기존과 동일)
        LEFT_CLEAR_TH            = 4           # 좌측: '막힘→비었음' 전환 임계 (신규, 더 낮음)
        RIGHT_BLOCK_TH           = 5           # 우측: '비었음→막힘' 전환 임계 (기존과 동일)
        RIGHT_CLEAR_TH           = 2           # 우측: '막힘→비었음' 전환 임계 (신규, 더 낮음)
        SIDE_DEADZONE            = 0.25        # |EMA(mean_y)| 이하이면 'center'
        SIDE_EMA_ALPHA           = 0.3         # EMA 계수
        BODY_LO, BODY_HI         = 215, 305    # 차체 자기가림 구간 (최종 확정 2026-07-22)

        if self.lidar_ranges is None:
            self.obstacle_front = False
            self.obstacle_dist  = 999.0
            self.obstacle_side  = 'none'
            self.obstacle_type  = 'none'
            self.left_clear     = True
            self.right_clear    = True
            self.left_clear_confirmed  = True
            self.right_clear_confirmed = True
            return

        # LUT 지연 초기화 (최초 1회)
        if not hasattr(self, '_obs_cos'):
            _deg = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False) - math.radians(LIDAR_ANGLE_OFFSET_DEG)
            self._obs_cos = np.cos(_deg).astype(np.float32)
            self._obs_sin = np.sin(_deg).astype(np.float32)

        ranges = np.array(self.lidar_ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = 0.0
        ranges[ranges <= 0.0]        = 0.0
        if BODY_MASK_ENABLED:
            ranges[BODY_LO:BODY_HI]  = 0.0   # 차체 자기가림 마스킹

        n = len(ranges)
        m = min(n, 360)
        cos_d, sin_d = self._obs_cos[:m], self._obs_sin[:m]
        r = ranges[:m]
        x = r * cos_d        # 전방(+앞)
        y = r * sin_d        # 횡방향(+좌/-우)
        valid = r > 0.0

        # ── 전방 장애물 (고정장애물/차량 공통) ──
        front_mask = valid & (x > FRONT_X_MIN) & (x < FRONT_X_MAX) & (np.abs(y) < FRONT_Y_HALF)
        front_cnt  = int(np.count_nonzero(front_mask))
        detected_now = front_cnt > FRONT_MIN_PTS
        now = time.time()

        if detected_now:
            self._obstacle_last_seen = now
            self.obstacle_front = True

            # ── 타겟 분리: 전방 점들을 인접 인덱스(=인접 각도)끼리 묶고, 가장 가까운
            #    묶음 하나만 '지금 문제되는 타겟'으로 삼는다.
            #    구 구현은 ROI 안 점을 전부 뭉뚱그려 평균냈다. 벽과 차량이 같이 잡히면
            #    횡위치가 둘의 중간으로 나와서 '어느 쪽으로 비킬까' 판단이 통째로 틀어진다.
            fidx = np.where(front_mask)[0]
            groups = np.split(fidx, np.where(np.diff(fidx) > 1)[0] + 1)
            tgt = min(groups, key=lambda g: float(np.min(r[g])))

            ty = y[tgt]
            self.obstacle_dist  = float(np.min(r[tgt]))
            # [6] 분류 기준을 '점 개수' → '실제 횡폭(m)' 으로.
            #   점 개수는 거리에 반비례해서 같은 물체도 거리에 따라 분류가 뒤집힌다
            #   (실측: 폭 0.35m 차량이 2.0m 에선 fixed(11점), 1.5m 에선 vehicle(13점)).
            #   y 범위는 거리와 무관한 물리적 폭이라 안정적이다.
            self.obstacle_y_min = float(np.min(ty))
            self.obstacle_y_max = float(np.max(ty))
            self.obstacle_width = self.obstacle_y_max - self.obstacle_y_min
            self.obstacle_type = ('vehicle' if self.obstacle_width >= OBSTACLE_VEHICLE_WIDTH_M
                                  else 'fixed')

            # 접근율(m/s). obstacle_dist 는 라이다 실측이라 이 값은 '진짜 m/s' 다
            # (자차 속도만 모터단위라 미보정). 음수 = 가까워지는 중.
            #   정지 타겟이면 ≈ -자차속도, 같은 방향으로 달리는 차면 그보다 훨씬 작다.
            #   분류의 주 근거는 Phase 이고, 이 값은 교차확인·추돌방지용이다.
            if self._obstacle_prev_dist is not None:
                dt_o = now - self._obstacle_prev_t
                if dt_o > 1e-3:
                    raw = (self.obstacle_dist - self._obstacle_prev_dist) / dt_o
                    self.obstacle_rate = 0.7 * self.obstacle_rate + 0.3 * raw
            self._obstacle_prev_dist = self.obstacle_dist
            self._obstacle_prev_t    = now

            mean_y = float(np.mean(ty))
            self._ema_y = SIDE_EMA_ALPHA * mean_y + (1.0 - SIDE_EMA_ALPHA) * self._ema_y
            self.obstacle_y = self._ema_y
            if   self._ema_y >  SIDE_DEADZONE: self.obstacle_side = 'left'
            elif self._ema_y < -SIDE_DEADZONE: self.obstacle_side = 'right'
            else:                              self.obstacle_side = 'center'
        elif (now - self._obstacle_last_seen) < OBSTACLE_HOLD_T:
            # [9] 인식 끊김 보상: 마지막 관측 후 OBSTACLE_HOLD_T 안이면 계속 있다고 본다.
            #   직전 dist/type/side/width 를 그대로 유지한다(새로 계산할 근거가 없으므로).
            self.obstacle_front = True
        else:
            self.obstacle_front = False
            self.obstacle_dist = 999.0
            self.obstacle_side = 'none'
            self.obstacle_type = 'none'
            self.obstacle_width = 0.0
            self.obstacle_rate = 0.0
            self._obstacle_prev_dist = None
            self._ema_y *= (1.0 - SIDE_EMA_ALPHA)
            self.obstacle_y = self._ema_y

        # ── 좌/우 차선 공간 (추월 이동·복귀 판단) ──
        left_mask  = valid & (x > SIDE_X_MIN) & (x < SIDE_X_MAX) & (y >  LEFT_Y_MIN)  & (y <  LEFT_Y_MAX)
        right_mask = valid & (x > SIDE_X_MIN) & (x < SIDE_X_MAX) & (y < -RIGHT_Y_MIN) & (y > -RIGHT_Y_MAX)
        # [2026-08-13] 총 점개수 대신 "가장 긴 연속(인접 인덱스) 묶음의 길이"를 쓴다
        # (_largest_run() 참고) — 흩어진 노이즈 점들이 우연히 합쳐서 개수가 많아져도
        # 연속 구간 자체는 짧아 '막힘'으로 잘못 잡히지 않는다. ★주의★ 값의 의미가
        # "총 개수"에서 "최대 연속 길이"로 바뀌었으므로 LEFT/RIGHT_BLOCK_TH(위,
        # 원래 총 개수 8/5 기준으로 잡힌 값)도 재해석이 필요할 수 있다 — 연속 길이는
        # 보통 총 개수보다 작거나 같으므로, 같은 물체를 여전히 '막힘'으로 잡으려면
        # 임계값을 낮춰야 할 가능성이 높다. 실차에서 DEBUG_VIZ_LIDAR의 run 표시로
        # 실제 값 범위를 보고 재조정할 것(현재는 기존 값 그대로 유지 — 실차 미검증).
        left_cnt_raw  = self._largest_run(left_mask)
        right_cnt_raw = self._largest_run(right_mask)
        # [2026-08-13] 판정(임계값 비교) 전에 원본 점개수를 먼저 EMA로 스무딩한다 — 위
        # obstacle_y용 _ema_y와 같은 SIDE_EMA_ALPHA를 재사용(새 튜닝값 추가 없이 기존
        # 패턴만 확장). 히스테리시스/디바운스가 "판정 이후" 안정화라면 이건 "판정 이전"
        # 안정화라 서로 다른 축 — 셋을 순서대로(스무딩→히스테리시스→디바운스) 겹쳐 쓴다.
        self._left_cnt_ema  = SIDE_EMA_ALPHA * left_cnt_raw  + (1.0 - SIDE_EMA_ALPHA) * self._left_cnt_ema
        self._right_cnt_ema = SIDE_EMA_ALPHA * right_cnt_raw + (1.0 - SIDE_EMA_ALPHA) * self._right_cnt_ema
        left_cnt  = self._left_cnt_ema
        right_cnt = self._right_cnt_ema
        # 히스테리시스: 현재 '비었음' 상태면 높은 임계값(_BLOCK_TH)을 넘어야 '막힘'으로
        # 전환하고, 현재 '막힘' 상태면 낮은 임계값(_CLEAR_TH) 밑으로 내려가야 '비었음'으로
        # 전환한다 — 두 임계값 사이 구간에서는 직전 상태를 그대로 유지해 경계 근처 잔떨림을
        # 없앤다(우변은 갱신 전 self.left_clear/right_clear, 즉 직전 프레임 상태를 읽는다).
        self.left_clear  = (left_cnt  < LEFT_BLOCK_TH)  if self.left_clear  else (left_cnt  < LEFT_CLEAR_TH)
        self.right_clear = (right_cnt < RIGHT_BLOCK_TH) if self.right_clear else (right_cnt < RIGHT_CLEAR_TH)

        # [2026-08-13] 비대칭 디바운스 — "비었음"은 SIDE_CLEAR_CONFIRM_FRAMES 연속 유지돼야
        # 확정되고, "막힘"은 한 프레임만 나와도 즉시 카운터가 리셋된다(config.py
        # SIDE_CLEAR_CONFIRM_FRAMES 주석 참고). choose_side()에는 이 확정값을 넘긴다.
        self._left_clear_cnt  = self._left_clear_cnt + 1 if self.left_clear  else 0
        self._right_clear_cnt = self._right_clear_cnt + 1 if self.right_clear else 0
        self.left_clear_confirmed  = self._left_clear_cnt  >= SIDE_CLEAR_CONFIRM_FRAMES
        self.right_clear_confirmed = self._right_clear_cnt >= SIDE_CLEAR_CONFIRM_FRAMES

        if DEBUG_VIZ_LIDAR:
            # [2026-08-11] PPM=125(표시 범위 2m)였을 때는 실제 장애물 감지 ROI(FRONT_X_MAX=5.0m,
            #   SIDE_X_MAX=5.5m)의 대부분이 캔버스 밖으로 잘려서 안 보였다(예: to_px(5.0, 1.5)의
            #   y픽셀이 음수) — "감지 범위가 안 보인다"는 요청으로 ROI 전체가 들어오도록 축척을
            #   낮춤. PPM=40 → 반경 250px/40=6.25m까지 표시되어 5.0/5.5m ROI 박스가 온전히 보인다.
            # [2026-08-11] 각도 컴퍼스(8방향 i-라벨)/자기가림 경계선(MASK_LO/HI)/포인트별 인덱스
            #   숫자를 지웠다 — 셋 다 LIDAR_ANGLE_OFFSET_DEG(§6.2)·BODY_LO/HI(§0 공통주의) 값을
            #   맞추던 캘리브레이션용이었는데 두 값 다 2026-07-22 최종 확정됐다. 지금 실차
            #   테스트에서 매 프레임 봐야 하는 건 "지금 뭘 장애물로 잡았고 좌우가 비었는지"뿐이라
            #   ROI 박스/포인트/상태텍스트만 남기고 나머지는 지웠다 — 각도 재검증이 다시 필요해지면
            #   git 이력에서 이 줄 이전 버전을 복원할 것.
            PPM = 40          # 1m = 40px (표시 범위 약 6.25m — 장애물 감지 ROI 전체 포함)
            W, H = 500, 500
            EX, EY = 250, 250  # 자차 위치(캔버스 정중앙 — 전/후/좌/우 전체 360도가 다 보이도록)
            bev = np.zeros((H, W, 3), dtype=np.uint8)

            for d in range(1, 7):
                cv2.circle(bev, (EX, EY), d * PPM, (50, 50, 50), 1)
                cv2.putText(bev, f'{d}m', (EX + 4, EY - d*PPM + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

            def to_px(wx, wy): return (int(EX - wy*PPM), int(EY - wx*PPM))

            # 세 ROI 박스 = 지금 실제로 장애물을 감지하는 범위(perc_obstacle() 상단 튜닝 파라미터와 동일).
            #   청록 = 전방(FRONT_X_MIN~MAX × ±FRONT_Y_HALF, B2/B3 공용 obstacle_dist 산출 범위)
            #   초록 = 좌측 차선공간(SIDE_X_MIN~MAX × LEFT_Y_MIN~MAX, 추월 이동/복귀 판단용)
            #   주황 = 우측 차선공간(SIDE_X_MIN~MAX × RIGHT_Y_MIN~MAX)
            cv2.rectangle(bev, to_px(FRONT_X_MIN, FRONT_Y_HALF), to_px(FRONT_X_MAX, -FRONT_Y_HALF), (0, 220, 220), 1)
            cv2.putText(bev, f'FRONT ROI ({FRONT_X_MIN:.1f}~{FRONT_X_MAX:.1f}m)',
                        to_px(FRONT_X_MAX, FRONT_Y_HALF - 0.15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 220), 1, cv2.LINE_AA)
            cv2.rectangle(bev, to_px(SIDE_X_MIN, LEFT_Y_MAX),  to_px(SIDE_X_MAX,  LEFT_Y_MIN), (0, 220, 0),   1)
            cv2.rectangle(bev, to_px(SIDE_X_MIN, -RIGHT_Y_MIN), to_px(SIDE_X_MAX, -RIGHT_Y_MAX), (0, 140, 255), 1)

            for i in range(len(r)):
                if not valid[i]: continue
                sx = int(EX - y[i] * PPM)
                sy = int(EY - x[i] * PPM)
                if not (0 <= sx < W and 0 <= sy < H): continue
                if front_mask[i]:   col = (0, 0, 255)
                elif left_mask[i]:  col = (0, 255, 0)
                elif right_mask[i]: col = (0, 140, 255)
                else:               col = (60, 60, 60)
                cv2.circle(bev, (sx, sy), 2, col, -1)

            cv2.circle(bev, (EX, EY), 7, (255, 220, 0), -1)
            cv2.line(bev, (EX, EY), (EX, EY - 18), (255, 220, 0), 2)

            type_col = (0, 0, 255) if self.obstacle_front else (0, 255, 0)
            cv2.putText(bev, f'{self.obstacle_type.upper()} {self.obstacle_dist:.1f}m  {self.obstacle_side}  pts={front_cnt}',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, type_col, 1, cv2.LINE_AA)
            # [2026-08-13] 최대연속길이(run, 총 점개수 아님) → EMA 스무딩값 → 히스테리시스
            # 판정(순간값) → 디바운스 확정값, 4단계를 한눈에 비교할 수 있게 전부 표시한다 —
            # 실차 디버깅 시 어느 단계에서 값이 흔들리는지/뒤집히는지 바로 구분하기 위함.
            cv2.putText(bev, f'run L:{left_cnt_raw}->{self._left_cnt_ema:.1f}  R:{right_cnt_raw}->{self._right_cnt_ema:.1f}',
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(bev, f'L:{"CLR" if self.left_clear else "BLK"}({self._left_clear_cnt}/{SIDE_CLEAR_CONFIRM_FRAMES})'
                             f' R:{"CLR" if self.right_clear else "BLK"}({self._right_clear_cnt}/{SIDE_CLEAR_CONFIRM_FRAMES})',
                        (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(bev, f'confirmed L:{"CLR" if self.left_clear_confirmed else "BLK"}'
                             f'  R:{"CLR" if self.right_clear_confirmed else "BLK"}',
                        (8, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if (self.left_clear_confirmed or self.right_clear_confirmed) else (0, 0, 255),
                        1, cv2.LINE_AA)
            cv2.imshow('lidar_bev', bev)
            cv2.waitKey(1)

    # [2-3b] 회피 "복귀 유예"(avoid-hold) — perc_obstacle() 직후에만 호출할 것
    #   (obstacle_front/obstacle_dist가 이번 틱 기준으로 이미 갱신돼 있어야 함).
    #   입력 self.obstacle_front/obstacle_dist → 출력 self.avoid_hold_active
    def _update_avoid_hold(self):
        """da 안전마진 회피(§2.30) 중 너무 이른 복귀를 막기 위한 타이머 — config.py
        AVOID_HOLD_* 주석, README §2.32, avoid_hold_improvement_proposal.md 참고.
        obstacle_front/obstacle_dist는 TEST_DISABLE_B2_B3와 무관하게 매 틱 갱신되므로
        (perc_obstacle() 참고), B2/B3 미션 자체가 꺼져있어도 이 신호는 그대로 쓸 수 있다.

        [2026-08-15 개선] 기존엔 트리거~해제 둘 다 "장애물이 AVOID_HOLD_TRIGGER_DIST_M
        안에 있는가" 하나와 고정 AVOID_HOLD_SEC뿐이었다. 아래 네 갈래를 추가했다
        (avoid_hold_improvement_proposal.md "1차 적용 결정" 적용1/2/3):
          ① 트리거가 걸리는 순간 target_speed_est(=v_mps+obstacle_rate)를 1회 스냅샷해
             hold_sec을 가변으로 정한다(정지 장애물엔 짧게, 방해차량처럼 빠르게 붙는
             경우엔 길게, AVOID_HOLD_SEC_MAX로 캡).
          ② da 연속성(perception/dl_lane.py DLSlideWindow.da_area_jump_detected)을 라이다
             obstacle_front와 OR로 결합 — 라이다 사각지대를 카메라 쪽이 보완한다.
          ③ obstacle_front=False가 AVOID_HOLD_RELEASE_CONFIRM_FRAMES 연속 + 마지막
             유효 obstacle_dist가 AVOID_HOLD_RELEASE_DIST_M 이상이면, hold_sec을 다
             채우기 전에도 즉시 해제한다("안 보임=멀어짐"이 아니라 "마지막으로 봤을 때
             이미 멀었음"으로 조건화).
          ④ TargetPassing.choose_side()로 "지금 어느 쪽이 안전한가"를 매 틱 갱신해
             self.avoid_hold_side에 저장한다(perc_lane()이 DL 백엔드로 전달 → 방향
             힌트로 씀, side==0이면 apply_behavior_override() 이전 _lane_drive()가
             안전판 감속을 건다 — 적용4).
        """
        now = time.time()

        # ② da 연속성 보조 트리거 — 백엔드에 없으면(hough/classic_cv) getattr가 False를
        # 반환해 조용히 건너뛴다(show_debug_windows()와 동일 관례).
        da_area_jump = bool(getattr(self.lane_detector, 'da_area_jump', False))
        triggered_now = ((self.obstacle_front and self.obstacle_dist < AVOID_HOLD_TRIGGER_DIST_M)
                          or da_area_jump)

        if triggered_now:
            if self._avoid_hold_until_t <= now:
                # 직전엔 유예가 꺼져있었다 = 이번이 새 트리거 — 여기서만 target_speed_est를
                # 스냅샷해서 hold_sec을 정한다(① — 노이즈가 유예시간 자체의 흔들림으로
                # 새는 것을 막는 게 핵심, 매 틱 재계산하지 않는다).
                target_speed_est = self.v_mps + self.obstacle_rate
                self._avoid_hold_target_speed_est = target_speed_est
                gain_term = (AVOID_HOLD_RATE_GAIN * abs(target_speed_est)
                             if abs(target_speed_est) > OBSTACLE_STATIC_SPEED_TH_MPS else 0.0)
                self.avoid_hold_hold_sec = min(
                    AVOID_HOLD_SEC_MAX, max(AVOID_HOLD_SEC_MIN, AVOID_HOLD_SEC_BASE + gain_term))
                self._avoid_hold_release_cnt = 0
                self.avoid_hold_release_reason = ''
            self._avoid_hold_until_t = now + self.avoid_hold_hold_sec

        # ③ 조기 해제 판정용 상태 갱신 — obstacle_front가 True인 동안(=아직 가까움)은
        # "마지막 유효 거리"를 계속 최신으로 두고 디바운스 카운터를 리셋한다.
        if self.obstacle_front:
            self._avoid_hold_last_valid_dist = self.obstacle_dist
            self._avoid_hold_release_cnt = 0
        else:
            self._avoid_hold_release_cnt += 1

        was_active = self.avoid_hold_active
        if (was_active
                and self._avoid_hold_release_cnt >= AVOID_HOLD_RELEASE_CONFIRM_FRAMES
                and self._avoid_hold_last_valid_dist >= AVOID_HOLD_RELEASE_DIST_M):
            self._avoid_hold_until_t = now  # 조기 해제 — hold_sec을 다 채우기 전에 즉시 종료
            self.avoid_hold_release_reason = 'early_dist'

        self.avoid_hold_active = now < self._avoid_hold_until_t
        if was_active and not self.avoid_hold_active and self.avoid_hold_release_reason != 'early_dist':
            self.avoid_hold_release_reason = 'timeout'

        # ④ 방향 힌트 — choose_side()는 입력값(obstacle_y/left_clear_confirmed/
        # right_clear_confirmed/lane_side)만 보는 순수 판정 함수라 self.vehicle_controller의
        # 다른 상태(phase/side 등, B2/B3 FSM 전용)를 건드리지 않는다. left_clear_confirmed/
        # right_clear_confirmed는 perc_obstacle()이 TEST_DISABLE_B2_B3와 무관하게 매 틱
        # 갱신하므로(코드 확인 완료) 별도 stale 가드 없이 그대로 호출해도 안전하다.
        self.avoid_hold_side = self.vehicle_controller.choose_side(
            self.obstacle_y, self.left_clear_confirmed, self.right_clear_confirmed, self.lane_side)

    # [2-4] 라바콘
    #   출력 lavacon_offset(디버그용)/lavacon_done, lavacon_path(조향용 — _handle_lavacon() 참고)
    def perc_lavacon(self):
        self.lavacon_offset, self.lavacon_done, path_m = process_lavacon(self.lidar_ranges)
        # [2026-08-11] 라바콘 조향 파라미터를 라인주행(_lane_steer())과 완전히 일치시키기로
        # 한 결정 — LAVACON_KP 같은 라바콘 전용 P게인 대신, self.lane_path와 동일하게
        # self.pure_pursuit(같은 PP_* 게인)에 태운다. "1m=DL_PIXELS_PER_METER px,
        # x=오른쪽+, 전방=이미지 위쪽(y 감소)" 스케일로 실측 축거(PP_WHEELBASE_PX)를
        # 캘리브레이션해뒀으므로(controller/pure_pursuit.py 상단 주석 참고), 라이다 미터
        # 좌표(x=전방+, y=좌측+)를 그 스케일로 그대로 변환하면 물리적으로 일관된 입력이
        # 된다 — 차량 기준점은 원점(0,0)으로 두고(_handle_lavacon()이 vehicle_x=0.0으로
        # 호출), 좌측(+y_m)은 이미지 왼쪽(-col_px)에 대응한다.
        self.lavacon_path = [(-y * DL_PIXELS_PER_METER, -x * DL_PIXELS_PER_METER) for x, y in path_m]
        # 위 px 변환 전 원본(라이다 미터 좌표) — _draw_lavacon_bev()가 DEBUG_VIZ_LAVACON일 때
        # 그대로 그려서 "실제로 조향에 쓰이는 경로"를 시각적으로 보여준다.
        self._lavacon_path_m = path_m

    # [2-4b] 라바콘 좌우 클러스터 검출 + YOLO 카메라 이중확인 → B1_LAVACON 진입 트리거
    #   입력 self.lidar_ranges, self.cone_detected_yolo(perc_yolo_cone()이 먼저 채워둠)
    #   출력 lavacon_left_detected/right_detected, lavacon_trigger
    #   설계 의도: 라이다 포인트가 "존재"하는 것만으로는 벽·바닥 잡음과 구분이 안 되므로,
    #     인접 인덱스(=인접 각도)로 붙어있는 포인트 묶음(클러스터)이 좌/우 각각 최소 1개씩
    #     동시에 있어야 "라바콘 구간 진입"으로 인정한다. perc_obstacle()과 동일한 차체 마스킹/
    #     극좌표 변환 방식을 사용하되, ROI와 목적은 별개(장애물 회피용이 아니라 콘 게이트 진입 판단용)이므로
    #     여기서 독립적으로 계산한다.
    #   [2026-08-07] 라이다 클러스터 판정 단독으로는 벽 모서리 등에서 오검출 여지가 있어,
    #     YOLO(perception/yolo_cone.py)로 카메라에도 실제 cone 클래스가 보이는지 AND 조건을
    #     추가했다. YOLO 콘 검출 AND 좌우 라이다 동시검출이 LAVACON_TRIGGER_FRAMES 연속
    #     유지되면 진입 확정(디바운스). YOLO 검출기 초기화 실패(self.yolo_cone_detector is
    #     None)로 카메라 확인 자체가 불가능하면, 카메라 이중확인 없이 라이다 단독 판정으로
    #     폴백한다(전체가 영영 안 켜지는 것보다 낫다는 판단).
    def perc_lavacon_trigger(self):
        # ── 튜닝 파라미터 (실측 라바콘 간격 기준 추정치, 실차 튜닝 필요) ──
        LON_MIN, LON_MAX = 0.3, 0.5   # 트리거 ROI 전방 종방향(m) — 너무 가깝거나(차체 반사) 먼 점 배제
        LAT_MAX           = 2.0        # 트리거 ROI 횡방향 한계(m)
        CLUSTER_MIN_PTS   = 2          # 클러스터로 인정할 최소 연속 포인트 수(단일 반사점 노이즈 배제)
        CLUSTER_MAX_GAP   = 0.35       # 같은 클러스터로 볼 최대 거리편차(m) — 콘 지름 근사
        BODY_LO, BODY_HI  = 215, 305   # 차체 자기가림 구간 (perc_obstacle과 동일, 최종 확정 2026-07-22)

        if self.lidar_ranges is None:
            self.lavacon_left_detected  = False
            self.lavacon_right_detected = False
            self._lavacon_trigger_cnt   = 0
            self.lavacon_trigger        = False
            self._lavacon_dbg = (0, 0, 0, 0)
            self._lavacon_mask_dbg = (0, -1.0)
            return

        ranges_raw = np.array(self.lidar_ranges, dtype=np.float32)
        ranges_raw[~np.isfinite(ranges_raw)] = 0.0
        ranges_raw[ranges_raw <= 0.0] = 0.0

        ranges = ranges_raw.copy()
        n = len(ranges)
        if BODY_MASK_ENABLED and n > BODY_LO:
            ranges[BODY_LO:min(BODY_HI, n)] = 0.0   # 차체 자기가림 마스킹

        m = min(n, 360)
        deg = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False) - math.radians(LIDAR_ANGLE_OFFSET_DEG)
        r = ranges[:m]
        r_raw = ranges_raw[:m]
        x = r * np.cos(deg)          # 전방(+앞)
        y = r * np.sin(deg)          # 횡방향(+좌/-우)
        roi = (r > 0.0) & (x > LON_MIN) & (x < LON_MAX) & (np.abs(y) < LAT_MAX)

        # 진단용: BODY_LO~BODY_HI로 "차체 자기가림"이라 보고 지워버리는 구간에
        # 마스킹 전 원본 raw range가 실제로 얼마나/얼마나 가깝게 찍히는지 확인.
        # 여기 값이 크고 거리도 콘 간격과 비슷하면 이 마스크가 진짜 우측 콘 반사까지
        # 같이 지우고 있다는 뜻(마스크 구간 자체를 재보정해야 함).
        body_hi_eff = min(BODY_HI, n)
        masked_raw = ranges_raw[BODY_LO:body_hi_eff]
        masked_valid = masked_raw[masked_raw > 0.0]
        masked_pts = int(masked_valid.size)
        masked_min = float(masked_valid.min()) if masked_pts else -1.0
        self._lavacon_mask_dbg = (masked_pts, masked_min)

        def _has_cluster(side_mask):
            idx = np.where(roi & side_mask)[0]
            pts = len(idx)
            if pts < CLUSTER_MIN_PTS:
                return False, pts, 0
            # 인덱스(=각도) 순 배열이므로, 인덱스가 서로 붙어있으면 공간적으로도 인접한 점으로 보고 묶는다.
            splits = np.where(np.diff(idx) > 1)[0] + 1
            found, best_run = False, 0
            for g in np.split(idx, splits):
                best_run = max(best_run, len(g))
                if len(g) >= CLUSTER_MIN_PTS and (np.max(r[g]) - np.min(r[g])) <= CLUSTER_MAX_GAP:
                    found = True   # 콘 하나 크기로 뭉친 클러스터 발견
            return found, pts, best_run

        self.lavacon_left_detected,  left_pts,  left_run  = _has_cluster(y > 0.0)   # 좌측(y>0)
        self.lavacon_right_detected, right_pts, right_run = _has_cluster(y < 0.0)   # 우측(y<0)
        # 디버그용: ROI 안에 몇 점이 잡혔는지 / 그중 최대 연속 묶음 길이(클러스터 기준 CLUSTER_MIN_PTS=2 통과 여부 진단)
        self._lavacon_dbg = (left_pts, left_run, right_pts, right_run)

        if DEBUG_VIZ_LAVACON:
            self._draw_lavacon_bev(r, x, y, roi, LON_MIN, LON_MAX, LAT_MAX,
                                    left_pts, left_run, right_pts, right_run,
                                    r_raw, deg, BODY_LO, body_hi_eff)

        # YOLO 콘 검출 AND 좌우 라이다 클러스터 동시검출이 연속 프레임 유지되면 진입 확정(디바운스).
        # yolo_cone_detector 초기화 실패 시엔 카메라 조건을 무조건 통과(True)시켜 라이다
        # 단독 판정으로 자연스럽게 폴백한다.
        cone_confirmed_cam = self.cone_detected_yolo if self.yolo_cone_detector is not None else True
        if cone_confirmed_cam and self.lavacon_left_detected and self.lavacon_right_detected:
            self._lavacon_trigger_cnt += 1
        else:
            self._lavacon_trigger_cnt = 0
        self.lavacon_trigger = self._lavacon_trigger_cnt >= LAVACON_TRIGGER_FRAMES

    # [2-4c] [DEBUG_VIZ_LAVACON] 라바콘 트리거 ROI/좌우 클러스터 BEV 시각화
    #   perc_obstacle()의 DEBUG_VIZ_LIDAR 창과 같은 스타일, ROI/축척만 라바콘 트리거에 맞게 확대.
    #   초록=좌측(y>0) ROI점, 주황=우측(y<0) ROI점, 회색=ROI 밖. 자홍(magenta)=BODY_LO~BODY_HI
    #   "차체 자기가림"이라고 보고 지워버리는 구간의 마스킹 전 원본(raw) 점 — 이 구간에 실제
    #   물체(콘)가 있는데도 마스크가 지우고 있는 건 아닌지 진단용. 텍스트로 pts(ROI 내 점수)/
    #   run(최대 연속묶음, CLUSTER_MIN_PTS=2 이상이어야 클러스터로 인정) 표시.
    def _draw_lavacon_bev(self, r, x, y, roi, lon_min, lon_max, lat_max,
                           left_pts, left_run, right_pts, right_run,
                           r_raw, deg, body_lo, body_hi_eff):
        PPM = 80           # 1m = 80px (좁은 트리거 ROI라 perc_obstacle보다 확대)
        W, H = 500, 500
        EX, EY = 250, 460  # 자차 위치(하단 중앙)
        bev = np.zeros((H, W, 3), dtype=np.uint8)

        for d in (1, 2, 3):
            cv2.circle(bev, (EX, EY), d * PPM, (50, 50, 50), 1)
            cv2.putText(bev, f'{d}m', (EX + 4, EY - d * PPM + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

        def to_px(wx, wy): return (int(EX - wy * PPM), int(EY - wx * PPM))
        cv2.rectangle(bev, to_px(lon_min, lat_max), to_px(lon_max, -lat_max), (0, 220, 220), 1)

        # 마스킹 전 원본(raw) 점 중 BODY_LO~BODY_HI(자기가림 구간)에 해당하는 것만 자홍색으로 표시.
        # ROI 필터 없이 원거리까지 전부 그려서, 이 "자기가림 구간"에 실제로 뭔가 찍히는지 그대로 보여준다.
        masked_x = r_raw[body_lo:body_hi_eff] * np.cos(deg[body_lo:body_hi_eff])
        masked_y = r_raw[body_lo:body_hi_eff] * np.sin(deg[body_lo:body_hi_eff])
        for xi, yi, ri in zip(masked_x, masked_y, r_raw[body_lo:body_hi_eff]):
            if ri <= 0.0:
                continue
            sx, sy = int(EX - yi * PPM), int(EY - xi * PPM)
            if 0 <= sx < W and 0 <= sy < H:
                cv2.circle(bev, (sx, sy), 2, (255, 0, 255), -1)

        left_mask, right_mask = roi & (y > 0.0), roi & (y < 0.0)
        for i in range(len(r)):
            if r[i] <= 0.0:
                continue
            sx, sy = int(EX - y[i] * PPM), int(EY - x[i] * PPM)
            if not (0 <= sx < W and 0 <= sy < H):
                continue
            if left_mask[i]:    col = (0, 255, 0)
            elif right_mask[i]: col = (0, 140, 255)
            else:                col = (60, 60, 60)
            cv2.circle(bev, (sx, sy), 3, col, -1)

        # [2026-08-11] 실제 조향에 쓰이는 경로(self._lavacon_path_m, perc_lavacon()이 채운
        # 보로노이 정점 → x오름차순 정렬 결과) 시각화. 노란 선분이 Pure Pursuit/LQR이
        # _target_point()에서 그대로 걷는 꺾은선이다 — 차량(원점)에서 시작해 정점을 순서대로
        # 잇는다. 원은 정점 하나하나(§질문 답변: "영역"이 아니라 점 하나씩).
        path_m = self._lavacon_path_m
        if path_m:
            prev_px = (EX, EY)
            for wx, wy in path_m:
                cur_px = to_px(wx, wy)
                cv2.line(bev, prev_px, cur_px, (0, 255, 255), 2)
                prev_px = cur_px
            for wx, wy in path_m:
                cv2.circle(bev, to_px(wx, wy), 5, (0, 255, 255), -1)
                cv2.circle(bev, to_px(wx, wy), 5, (0, 0, 0), 1)

        cv2.circle(bev, (EX, EY), 6, (255, 220, 0), -1)
        cv2.line(bev, (EX, EY), (EX, EY - 18), (255, 220, 0), 2)

        l_col = (0, 255, 0)   if self.lavacon_left_detected  else (0, 0, 255)
        r_col = (0, 140, 255) if self.lavacon_right_detected else (0, 0, 255)
        cv2.putText(bev, f'L pts={left_pts} run={left_run} (need run>=2)',  (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, l_col, 1, cv2.LINE_AA)
        cv2.putText(bev, f'R pts={right_pts} run={right_run} (need run>=2)', (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, r_col, 1, cv2.LINE_AA)
        cone_col = (0, 255, 0) if self.cone_detected_yolo else (0, 0, 255)
        cv2.putText(bev, f'YOLO cone={self.cone_detected_yolo}', (8, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cone_col, 1, cv2.LINE_AA)
        cv2.putText(bev, f'trig={self._lavacon_trigger_cnt}/{LAVACON_TRIGGER_FRAMES}',
                    (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        masked_pts, masked_min = self._lavacon_mask_dbg
        masked_min_s = f'{masked_min:.2f}m' if masked_min >= 0 else 'N/A'
        cv2.putText(bev, f'masked(magenta) pts={masked_pts} min={masked_min_s}',
                    (8, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(bev, f'yellow=lavacon_path(voronoi vertex, n={len(path_m)})',
                    (8, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('lavacon_bev', bev)
        cv2.waitKey(1)

    # [2-6] 방해차량 진입 트리거 (라이다)
    #   입력 obstacle_front/dist (라이다)
    #   출력 vehicle_trigger
    #   설계 의도: 라이다 거리 단독·즉시 판정은 순간 오검출에 취약하므로,
    #     근접 상태가 VEHICLE_TRIGGER_FRAMES 연속 유지될 때만 진입을 확정한다(디바운스).
    def perc_vehicle_trigger(self):
        lidar_hit = self.obstacle_front and self.obstacle_dist < OVERTAKE_TRIGGER
        if lidar_hit:
            self._vehicle_trigger_cnt += 1
        else:
            self._vehicle_trigger_cnt = 0
        self.vehicle_trigger = self._vehicle_trigger_cnt >= VEHICLE_TRIGGER_FRAMES

    # [2-5] 정지선(굵은 가로 흰선)
    #   입력 self.img_front → 출력 self.stopline
    #   용도 : S1→S2 진입 / 지름길 끝(탈출 좌회전 지점) 단서
    def perc_stopline(self):
        if self.img_front is None:
            self.stopline = False
            return
        self.stopline = check_stopline(self.img_front)
        if self.stopline:
            self._last_stopline_t = time.time()   # [10] 교차로 근처 기동 금지 타이머

    # #########################################################
    # [3] 판단 (Decision)
    # #########################################################
    # ── 기동 진행중 여부 ──
    #   B2/B3는 한번 진입하면 트리거가 사라져도 기동이 끝날 때까지 유지돼야 한다.
    #   진행중 플래그를 별도 변수로 따로 들고 있으면 실제 컨트롤러 상태와 어긋날 수 있으므로,
    #   각 기동이 이미 들고 있는 실제 진행상태를 그대로 읽어서 판단한다.
    @property
    def _obstacle_active(self):
        """B2(고정장애물): 회피 기동이 진행 중인가."""
        return self.obstacle_controller.phase != AvoidPhase.IDLE

    @property
    def _maneuver_allowed(self):
        """[10] 교차로(정지선) 근처에서는 회피/추월 기동을 시작하지 않는다.
        Apollo 는 신호/정지선 20m·접속부 15m 클리어런스를 쓰지만, 우리는 거리 환산이
        아직 없으므로 '정지선을 최근에 봤는가'로 근사한다(B-2 실측 후 거리 기준 전환 가능).
        교차로 한복판에서 회피 기동이 걸리는 것을 막는 가드다."""
        return (time.time() - self._last_stopline_t) > MANEUVER_BLOCK_AFTER_STOPLINE_T

    @property
    def _overtake_active(self):
        """B3(방해차량): 추월 기동이 진행 중인가."""
        return self.vehicle_controller.phase != AvoidPhase.IDLE

    # ── 바퀴 카운트 ──
    def _update_lap(self):
        """누적 yaw(주) + 정지선(보조)으로 바퀴 수를 센다. 매 제어주기 1회 호출.

        누적은 ±pi 경계를 넘어가도 끊기지 않도록 '직전 yaw 대비 증분'을 더해서
        만든다(20Hz면 한 주기 변화가 pi를 넘을 일이 없으므로 언랩이 항상 안전하다).
        imu_yaw 자체는 -pi~pi 로 감기기 때문에 그 값을 직접 비교하면 안 된다.
        """
        if self._prev_yaw_accum_ref is None:
            self._prev_yaw_accum_ref = self.imu_yaw
            return

        d = self.imu_yaw - self._prev_yaw_accum_ref
        self._yaw_accum += math.atan2(math.sin(d), math.cos(d))   # 언랩된 증분
        self._prev_yaw_accum_ref = self.imu_yaw

        # 최소 소요시간 가드 — 제자리 회전이나 순간 오검출로 연속 카운트되는 것 방지
        if (time.time() - self._lap_t0) < LAP_MIN_T:
            return

        turned = abs(self._yaw_accum)
        # 임계 초과가 연속 유지돼야 확정 — 슬라럼 순간 피크 배제
        if turned >= LAP_YAW_FULL:
            self._lap_yaw_over_cnt += 1
        else:
            self._lap_yaw_over_cnt = 0
        by_yaw = self._lap_yaw_over_cnt >= LAP_YAW_CONFIRM_FRAMES
        # 정지선은 '충분히 돈 상태'에서만 인정한다. 흰 노면표시를 정지선으로
        # 오검출하는 경우가 있어(재현됨) 단독 신뢰는 위험하다.
        by_stopline = (self.stopline and turned >= LAP_YAW_MIN
                       and not self._lap_stopline_used)

        if by_yaw or by_stopline:
            self._begin_new_lap('yaw' if by_yaw else 'stopline', turned)

    def _begin_new_lap(self, cause, turned):
        self.lap += 1
        self.get_logger().info(
            f'[LAP] {self.lap - 1} → {self.lap} 바퀴 (근거={cause}, '
            f'누적={math.degrees(turned):.0f}도, {time.time() - self._lap_t0:.1f}s)')

        self._yaw_accum = 0.0
        self._lap_t0 = time.time()
        self._lap_stopline_used = (cause == 'stopline')
        self._lap_yaw_over_cnt = 0

        if self.lap > TOTAL_LAPS:
            self._change_state(MissionState.S4_FINISH)
            return

        if RESET_PHASE_EACH_LAP:
            # 장애물은 트랙에 그대로 있으므로 매 바퀴 처음부터 다시 만난다.
            self.phase = Phase.LAVACON
            self._lavacon_engaged = False
            self._lavacon_empty_cnt = 0
            self._lavacon_trigger_cnt = 0
            self._vehicle_trigger_cnt = 0
            self.obstacle_controller.reset()
            self.vehicle_controller.reset()
            self.behavior_state = BehaviorState.B0_NORMAL
            self._b2_passed = False   # [2026-08-15] Phase.OBSTACLE_ZONE 통합 — 완료 추적도 매 바퀴 리셋
            self._b3_passed = False

    def run_mission_fsm(self):
        {
            MissionState.S0_WAIT_GREEN  : self._s0_wait_green,
            MissionState.S1_LANE_FOLLOW : self._s1_lane_follow,
            MissionState.S2_INTERSECTION: self._s2_intersection,
            MissionState.S3_SHORTCUT    : self._s3_shortcut,
            MissionState.S4_FINISH      : self._s4_finish,
        }[self.mission_state]()

    def _change_state(self, new_state):
        """
        Mission 상태 전환 공통 처리.
          - 전환 로그 출력(디버깅 추적용)
          - PID 누적값 초기화: 이전 상태에서 쌓인 적분/미분 잔여가 새 상태로 넘어와 튀는 것을 방지한다.
        모든 상태 전환은 반드시 이 함수를 통해서만 한다(직접 대입 금지).
        """
        self.get_logger().info(f'[전환] {self.mission_state.name} → {new_state.name}')
        prev_state = self.mission_state
        self.mission_state = new_state
        self._pid_prev_error = 0.0
        self._pid_integral   = 0.0
        self.ctrl_angle = 0.0
        self.ctrl_speed = SPEED_STOP
        # S2 진입 시 신호값 초기화 (안정화는 S1 감속구간에서 이미 완료)
        if new_state == MissionState.S2_INTERSECTION:
            self.signal_red_on      = False
            self.signal_straight_on = False
            self.signal_left_on     = False
            self.signal_straight_confirmed = False
            self.signal_left_confirmed     = False
            self._sig_straight_cnt = 0
            self._sig_left_cnt     = 0
            self._s2_commit_dist = None
            self._s2_commit_dir  = None
        # S1 진입 시 감속 플래그 초기화
        if new_state == MissionState.S1_LANE_FOLLOW:
            self._approach_t0 = None
            # 출발(S0) 직후 첫 S1 진입 시 잠깐 정지선 오검출 억제
            if prev_state == MissionState.S0_WAIT_GREEN:
                self._stopline_cooldown_t = time.time() + 3.0
                # 실제 주행은 지금부터 — 신호 대기하며 서 있던 시간이 1바퀴 시간에
                # 섞이지 않도록 바퀴 기준점을 여기서 다시 잡는다.
                self._lap_t0 = time.time()
                self._yaw_accum = 0.0
                self._prev_yaw_accum_ref = self.imu_yaw
        # S3 진입 시 탈출 감속 플래그 + 기준 yaw 초기화
        if new_state == MissionState.S3_SHORTCUT:
            self._exit_approach_t0 = None
            self._shortcut_ref_yaw = None

    # ── S0: 출발 (신호등 인식) ──
    def _s0_wait_green(self):
        """
        출발선에서 정지한 채 4구 신호등을 본다(대회 규정 변경: S2와 동일한 신호등 재사용).
          - 초록불 켜지기 전: 완전 정지 (신호위반 감점 방지)
          - 초록불(직진 위치만 점등) 감지: S1(차선주행)로 전환하여 출발
        """
        self.ctrl_angle, self.ctrl_speed = 0.0, SPEED_STOP
        if self.signal_straight_confirmed:
            self._change_state(MissionState.S1_LANE_FOLLOW)

    # ── S1: 차선인식 주행 (라바콘·고정장애물·추월 Behavior를 이 상태 안에서 처리) ──
    def _s1_lane_follow(self):
        """
        차선을 따라 안정 주행.
          - S1에는 두 번 진입한다: ①S0 직후(교차로 가기 전, 순수 주행만)
                                  ②S2 교차로 "직진" 선택 후 복귀(Behavior B1→B2→B3 순서 진행)
          - ①에서는 정지선 감지 시 S2(교차로)로 전환.
          - ②에서는 Behavior가 조향/속도를 전담하므로 여기선 PID를 돌리지 않는다(적분 오염 방지).
        """
        # Behavior가 조향을 전담하는 구간에서는 Mission의 차선 PID를 건너뛴다.
        # phase==LAVACON이어도 좌우 클러스터 동시검출 트리거(_lavacon_engaged)가 확정되기 전까지는
        # 여기서 안 걸리고 아래 else 분기의 일반 차선주행(_lane_drive)이 계속 돈다.
        if self._behavior_enabled and self.phase == Phase.LAVACON and self._lavacon_engaged:
            return
        if self._obstacle_active or self._overtake_active:
            return

        if self._approach_t0 is not None:
            # 감속 구간: 차선 조향 유지 + 극저속 → 거의 정지 상태로 S2 진입
            elapsed = time.time() - self._approach_t0
            self.ctrl_angle = self._lane_steer()
            self.ctrl_speed = APPROACH_SPEED
            self._prev_speed = APPROACH_SPEED
            if elapsed >= APPROACH_TIME:
                self._change_state(MissionState.S2_INTERSECTION)
        else:
            self._lane_drive()
            # TEST_DISABLE_INTERSECTION=True면 정지선을 감지해도 아래 조건이 항상 False가 되어
            # _approach_t0가 절대 세팅되지 않음 → S2_INTERSECTION 전환 자체가 원천 차단되고
            # 계속 이 else 분기(_lane_drive)만 반복하며 차선주행을 이어간다.
            if (not TEST_DISABLE_INTERSECTION and self.stopline
                    and time.time() >= self._stopline_cooldown_t):  # 정지선 감지(쿨다운 지난 뒤만)
                self._approach_t0 = time.time()                             # 감속 구간 시작

    # ── S2: 교차로 — 정지 후 신호로 경로 판단 ──
    def _s2_intersection(self):
        """
        4구 신호등 교차로 진입 후 흐름 (순수 신호 인식만으로 경로 선택):
          1. 진입 즉시 정지 (기본값 STOP, 명시적 신호만 출발)
          2. 직진 초록(signal_straight_confirmed) → 커밋 구간(S2_COMMIT_DIST_M) 거쳐 S1 복귀
             + Behavior 활성화(라바콘부터 진행)
             좌회전 신호(signal_left_confirmed) → 커밋 구간 거쳐 좌회전 후 S3(지름길)
          3. 좌회전 진행 중이면 신호와 무관하게 완료 우선
          4. 커밋 구간(_s2_commit_dist)에서는 신호와 무관하게 직진만 유지 — 신호가 보이는
             지점과 실제 도로가 갈라지는 물리적 분기 지점이 떨어져 있고(config.py
             S2_COMMIT_DIST_M 주석 참고), 그 사이에 _lane_drive()(비전)를 켜면 분기가
             보이기 시작하는 순간 da가 반대쪽 갈래로 끌려간다(실측 재현됨). 신호로
             이미 확정된 방향이므로 이 구간은 비전을 아예 참조하지 않는다. 커밋 구간
             종료 판정은 시간이 아니라 VESC 실측(v_mps) 적분 거리로 한다 — 대회 주행 때
             APPROACH_SPEED 근방 실속도가 튜닝 시점과 달라져도 물리적 분기 지점과
             안 어긋나게(config.py S2_COMMIT_DIST_M 주석 참고).
        """
        if self._turn_yaw_start is not None:
            self._do_left_turn(next_state=MissionState.S3_SHORTCUT)
            return

        if self._s2_commit_dist is not None:
            self.ctrl_angle = 0.0
            self.ctrl_speed = APPROACH_SPEED
            self._s2_commit_dist += self._commit_speed_mps() * 0.05  # 20Hz 제어주기(control_loop) 가정
            if self._s2_commit_dist >= S2_COMMIT_DIST_M:
                commit_dir = self._s2_commit_dir
                self._s2_commit_dist = None
                self._s2_commit_dir  = None
                if commit_dir == 'straight':
                    self._behavior_enabled = True
                    self._stopline_cooldown_t = time.time() + STOPLINE_COOLDOWN
                    self._change_state(MissionState.S1_LANE_FOLLOW)
                else:
                    self._begin_left_turn()
            return

        self.ctrl_angle, self.ctrl_speed = 0.0, SPEED_STOP

        if self.signal_straight_confirmed:
            self._s2_commit_dist = 0.0
            self._s2_commit_dir  = 'straight'
        elif self.signal_left_confirmed:
            self._s2_commit_dist = 0.0
            self._s2_commit_dir  = 'left'

    # ── S3: 지름길 — 직진(+차선소실 대비), 끝에서 좌회전 ──
    def _s3_shortcut(self):
        """
        지름길 직진. 중간 차선소실 구간은 라이다로 딸 것이 없으므로 그냥 직진.
        끝에 도달하면 신호없이 좌회전으로 S1(차선주행) 복귀 (Behavior는 켜지 않음).

        지름길 출구(본선 합류부)는 신호등이 없어 정지선 검출로만 끝(_shortcut_end())을
        판단하는데, 합류부는 도로가 서서히 넓어지는 형태라 정지선이 실제로 잡히기 전에
        da 세그멘테이션이 합류 쪽으로 먼저 끌려가는 문제가 있다(ㅓ교차로와 동일한
        실패모드, config.py SHORTCUT_VISION_CUTOFF_T 주석 참고). 그래서 정지선 검출을
        기다리지 않고 SHORTCUT_VISION_CUTOFF_T가 지나면 미리 비전(_lane_drive())을 끄고
        _shortcut_ref_yaw 기준 헤딩홀드로 직진을 유지한다 — 좌회전 스크립트는 아직
        시작하지 않고, 정지선이 실제로 잡히거나 SHORTCUT_MAX_T에 도달해 _shortcut_end()가
        확정된 뒤에야 진출 시퀀스(감속+좌회전)로 넘어간다.
        """
        if self._turn_yaw_start is not None:
            self._do_left_turn(next_state=MissionState.S1_LANE_FOLLOW)
            return

        if self._shortcut_t0 is None:
            self._shortcut_t0 = time.time()

        if self._shortcut_ref_yaw is None and (time.time() - self._shortcut_t0) >= 1.0:
            self._shortcut_ref_yaw = self.imu_yaw
            self.get_logger().info(f'[S3] 기준 yaw 기록: {math.degrees(self._shortcut_ref_yaw):.1f}°')

        if self._shortcut_end():
            if self._exit_approach_t0 is None:
                self._exit_approach_t0 = time.time()
            elapsed = time.time() - self._exit_approach_t0
            if elapsed < APPROACH_EXIT_TIME:
                if self._shortcut_ref_yaw is not None:
                    yaw_err = self._yaw_delta(self._shortcut_ref_yaw)
                    self.ctrl_angle = float(np.clip(-yaw_err * 100.0, -30.0, 30.0))
                else:
                    self.ctrl_angle = 0.0
                self.ctrl_speed = APPROACH_EXIT_SPEED
            else:
                self._shortcut_t0 = None
                self._exit_approach_t0 = None
                self._begin_left_turn()
            return

        shortcut_elapsed = time.time() - self._shortcut_t0
        if shortcut_elapsed >= SHORTCUT_VISION_CUTOFF_T and self._shortcut_ref_yaw is not None:
            # 합류부 근접 구간 — 정지선이 아직 안 잡혔어도 비전을 더는 믿지 않고
            # 헤딩홀드로 직진 유지(진출 시퀀스는 _shortcut_end() 확정 후에만 시작).
            yaw_err = self._yaw_delta(self._shortcut_ref_yaw)
            self.ctrl_angle = float(np.clip(-yaw_err * 100.0, -30.0, 30.0))
            self.ctrl_speed = SPEED_NORMAL
            return

        if self.lane_valid:
            self._lane_drive()
        else:
            self.ctrl_angle = 0.0
            self.ctrl_speed = SPEED_NORMAL

    def _shortcut_end(self):
        """지름길 끝(탈출 좌회전 지점) 감지."""
        if self._shortcut_t0 is None:
            return False
        elapsed = time.time() - self._shortcut_t0
        if elapsed < SHORTCUT_MIN_T:
            return False
        return self.stopline or elapsed > SHORTCUT_MAX_T

    # ── S4: 종료 ──
    def _s4_finish(self):
        self.ctrl_angle, self.ctrl_speed = 0.0, SPEED_STOP



    # ── 좌회전 공통 (실차 전환: 후진 없이 무난한 좌회전) ──
    def _begin_left_turn(self):
        self._turn_yaw_start = self.imu_yaw   # closed-loop 기준점(IMU 죽었을 때만 미사용)
        self._turn_frame_cnt = 0
        self.get_logger().info('좌회전 시작')

    def _do_left_turn(self, next_state):
        """무난한(후진 없는) 좌회전 후 next_state로 전환.

        [2026-08-18] 종료 판정을 프레임 카운트(open-loop)에서 IMU yaw 실측
        기반(closed-loop)으로 변경. IMU가 살아있으면(_imu_live()) 실제 회전각
        (_yaw_delta(_turn_yaw_start))이 목표(trn_yaw_target, config.py TURN_YAW_TARGET_DEG류)에
        도달했을 때 끝낸다 — 같은 (조향각, 속도) 명령이어도 배터리 전압 강하·노면·속도
        변동에 따라 실제 요레이트가 매번 달라질 수 있어, "N프레임 지남"보다 정확하다.
        trn_f(TURN_FRAMES류)는 이제 트리거가 아니라 IMU가 죽어있을 때만 걸리는 안전
        타임아웃 상한이다 — 무한 회전 방지 + IMU 장애 시 예전과 동일한 동작으로 열화."""
        if next_state == MissionState.S3_SHORTCUT:
            trn_ang, trn_spd, trn_f, trn_yaw_target = (
                TURN_ANGLE, TURN_SPEED, TURN_FRAMES, TURN_YAW_TARGET_DEG)
        else:
            trn_ang, trn_spd, trn_f, trn_yaw_target = (
                TURN_EXIT_ANGLE, TURN_EXIT_SPEED, TURN_EXIT_FRAMES, TURN_EXIT_YAW_TARGET_DEG)

        turn_done = self._turn_frame_cnt >= trn_f   # IMU 죽었을 때를 위한 안전 타임아웃
        if self._imu_live():
            yaw_turned_deg = abs(math.degrees(self._yaw_delta(self._turn_yaw_start)))
            turn_done = turn_done or yaw_turned_deg >= trn_yaw_target

        if not turn_done:
            self.ctrl_angle = trn_ang
            self.ctrl_speed = trn_spd
        else:
            self.get_logger().info('좌회전 완료')
            self._turn_yaw_start = None
            self._turn_frame_cnt = 0
            if next_state == MissionState.S1_LANE_FOLLOW:
                self._stopline_cooldown_t = time.time() + STOPLINE_COOLDOWN
            self._change_state(next_state)
            return
        self._turn_frame_cnt += 1

    def _yaw_delta(self, start):
        """현재 yaw - start (−π~π wrap)"""
        d = self.imu_yaw - start
        return math.atan2(math.sin(d), math.cos(d))

    # ── Behavior FSM (Phase에 따라 순차 전용으로 배타 실행, 우선순위 판단 불필요) ──
    def run_behavior_fsm(self):
        """
        S1(차선주행) 재진입 후 Phase 순서(LAVACON→OBSTACLE_ZONE→DONE)에 따라 딱 하나의
        Behavior만 활성화한다. Phase 전환은 각 핸들러가 완료 시점에 직접 수행
        (`_mark_behavior_passed()` 참고).

        [2026-08-15] OBSTACLE_ZONE 통합(da_based_b2b3_proposal.md B안) — 예전엔
        FIXED_OBSTACLE/VEHICLE 두 Phase로 미리 순서를 나눠 "지금이 고정장애물 구간인지
        방해차량 구간인지"를 Phase가 알려줬는데, da 안전마진(§2.30)+avoid-hold
        (§2.32/§2.33)로 회피 기동 자체가 정적/동적 구분 없이 da 경로를 그대로 신뢰하는
        쪽으로 가면서 그 구분을 미리 나눠둘 이유가 약해졌다. 이제 정적/동적 구분은
        Phase가 아니라 매 프레임 `obstacle_type`(라이다 실측 폭 기반, `perc_obstacle()`
        참고)으로 그때그때 판단한다 — 예전에 "Phase가 순서를 강제하니 트리거에서
        `obstacle_type` 조건을 뺐다"던 이유(아래 참고)가 Phase 통합으로 없어졌으니
        다시 넣을 수 있게 된 것.
        """
        if self.phase == Phase.LAVACON:
            # 좌우 라이다 클러스터가 동시에(디바운스 프레임수만큼) 검출되면 B1_LAVACON 진입을 확정(latch)한다.
            # 한번 확정된 뒤에는 중간에 한쪽 클러스터가 잠깐 끊겨도(occlusion 등) B0로 되돌아가지 않고
            # lavacon_done 디바운스(_lavacon_empty_cnt)로 정상 종료될 때까지 유지한다.
            if self.lavacon_trigger:
                self._lavacon_engaged = True
            self.behavior_state = (BehaviorState.B1_LAVACON
                                    if self._lavacon_engaged
                                    else BehaviorState.B0_NORMAL)
        elif self.phase == Phase.OBSTACLE_ZONE:
            # TEST_DISABLE_B2_B3=True면 트리거 검사를 아예 안 하고 바로 리턴 — 장애물/방해차량이
            # 실제로 잡혀도 B2_OBSTACLE/B3_VEHICLE로 안 넘어가고 B0로 고정되어 _s1_lane_follow의
            # 일반 차선 PID가 계속 돎(placeholder 회피 기동이 실행 안 됨). 이 게이팅은 통합 전과
            # 동일하게 유지된다 — False로 바꾸면 트리거/디스패치가 예전과 같은 조건으로 그대로
            # 동작한다(달라진 건 "두 Phase로 미리 나눔" → "한 Phase 안에서 타입으로 나눔"뿐).
            if TEST_DISABLE_B2_B3:
                self.behavior_state = BehaviorState.B0_NORMAL
                return

            # 이미 진행 중인 기동이 있으면 이번 프레임 obstacle_type이 흔들려도 핸들러를
            # 중간에 바꾸지 않는다(진행 중 스왑은 위험 — 기존 "or self._obstacle_active"/
            # "or self._overtake_active" 유지 패턴과 동일 원칙, 우선순위만 명시적으로 첫
            # 분기로 올렸다).
            if self._obstacle_active:
                self.behavior_state = BehaviorState.B2_OBSTACLE
                return
            if self._overtake_active:
                self.behavior_state = BehaviorState.B3_VEHICLE
                return

            # 새로 트리거를 판단할 때만 obstacle_type으로 B2/B3를 가른다. 두 트리거
            # (SAFETY_DIST 기반 triggered_fixed, OVERTAKE_TRIGGER+디바운스 기반
            # perc_vehicle_trigger()의 self.vehicle_trigger)는 원래 서로 다른 Phase에서만
            # 봐서 겹칠 일이 없었는데, 이제 같은 프레임에 동시에 참일 수 있다 —
            # obstacle_type=='vehicle'이면 방해차량 쪽을 우선한다(vehicle_trigger가 아직
            # 디바운스 중이라도 폭이 이미 vehicle로 잡혔으면 B3로 본다). ★알려진 한계★:
            # 트리거 발동 직후 몇 프레임은 obstacle_type/vehicle_trigger 디바운스 타이밍이
            # 안 맞아 잠깐 B2로 걸렸다가 B3로 바뀌는 등 짧은 오분류가 있을 수 있다 —
            # `_obstacle_active`/`_overtake_active` latch가 실제로 걸리기 전(TargetPassing이
            # 아직 IDLE)에만 해당하는 구간이라 영향은 제한적이라고 판단.
            triggered_fixed = (self.obstacle_front
                                and self.obstacle_dist < SAFETY_DIST
                                and self._maneuver_allowed)
            triggered_vehicle = self.vehicle_trigger and self._maneuver_allowed

            if triggered_vehicle and self.obstacle_type == 'vehicle':
                self.behavior_state = BehaviorState.B3_VEHICLE
            elif triggered_fixed:
                self.behavior_state = BehaviorState.B2_OBSTACLE
            else:
                self.behavior_state = BehaviorState.B0_NORMAL
        else:  # Phase.DONE
            self.behavior_state = BehaviorState.B0_NORMAL

    # #########################################################
    # [4] 제어 (Control)
    # #########################################################
    def _corner_radius_speed_scale(self):
        """회전반경 기반 코너 감속 배율(0~1) — ROS2 Nav2 Regulated Pure Pursuit의
        curvatureConstraint()와 동일한 공식(CORNER_MIN_RADIUS_PX 주석 참고): 회전반경이
        CORNER_MIN_RADIUS_PX보다 작아지면 그 비율만큼 속도를 깎는다. 반경이 0에 가까워져도
        속도가 0으로 죽지 않게 CORNER_MIN_SPEED_SCALE로 하한을 둔다.

        [2026-08-06, 2026-08-10 복원] curvature는 self.pure_pursuit.last_curvature(이번 틱의
        순간값)가 아니라 _lane_drive()가 매 틱 갱신하는 self._corner_signal(조향각의 signed
        EMA)에서 역산한다 — 이유는 _lane_drive() 상단 주석 참고(진동을 매번 급코너로 오인해
        감속하는 문제). [2026-08-10] 이 신호 전환이 커밋 80aefe3("디버그창 적용", 조향과 무관한
        디버그 캔버스 레이아웃 변경)에서 실수로 되돌려져 있던 걸 발견해 복원함 — README §0.5.3
        참고."""
        curvature = math.tan(math.radians(self._corner_signal)) / self.pure_pursuit.wheelbase_px
        if curvature == 0.0:
            return 1.0
        radius = abs(1.0 / curvature)
        if radius >= CORNER_MIN_RADIUS_PX:
            return 1.0
        return max(CORNER_MIN_SPEED_SCALE, radius / CORNER_MIN_RADIUS_PX)

    def _lane_drive(self):
        """S1/S3 공통 차선 조향+감속 로직. ctrl_angle·ctrl_speed·_prev_speed·_corner_hold 갱신."""
        self.ctrl_angle = self._lane_steer()

        # [2026-08-06, 2026-08-10 복원] 코너 감속 판단은 "지금 순간 조향각이 얼마나 큰가"가
        # 아니라 "최근 한동안 같은 방향으로 얼마나 꺾여 있었는가"를 봐야 한다. pure_pursuit은
        # 구조상 좌우로 조금씩 진동("와리가리")하는데, turn_now를 매 틱 abs(ctrl_angle)로 그대로
        # 계산하면 진동의 절반(방향이 바뀌는 쪽)마다 급코너로 오인해 아래 3제곱 감속식과
        # _corner_radius_speed_scale()이 실제 코너가 아닌데도 속도를 깎는다(실차에서 재현:
        # 진동할 때마다 속도가 팍팍 줄었다 늘었다 함). self._corner_signal은 ctrl_angle을
        # signed(부호 유지) EMA로 누적한 값이다 — 진동처럼 부호가 계속 바뀌면 서로 상쇄돼 0
        # 근처로 수렴하고, 실제 코너처럼 한 방향으로 계속 꺾이면 EMA가 실제 각도로 수렴한다.
        # abs()는 반드시 이 EMA를 취한 "이후"에 적용해야 한다 — abs(ctrl_angle)을 먼저 평균내면
        # 부호 정보가 지워져서 진동도 그대로 다 더해져 상쇄 효과가 없어진다.
        # [2026-08-10] 이 로직이 커밋 80aefe3("디버그창 적용")에서 실수로 되돌려져 순간값
        # abs(ctrl_angle) 방식으로 퇴행해 있던 걸 발견해 복원함(README §0.5.3 참고) — 급조향
        # (30도 이상) 후 직진 복귀 구간에서 차선인식/조향이 흔들린다는 실차 보고의 유력한
        # 원인 중 하나로 지목됨.
        self._corner_signal = (CORNER_SIGN_EMA_ALPHA * self.ctrl_angle
                                + (1.0 - CORNER_SIGN_EMA_ALPHA) * self._corner_signal)
        turn_now     = min(1.0, abs(self._corner_signal) / ANGLE_MAX)
        turn_preview = min(1.0, abs(self.lane_lookahead) / LANE_LOOKAHEAD_REF)
        # [2026-08-17] 명시적 직진 모드(README §0.5.9)를 코너 감속에도 병합 — pure_pursuit이
        # (조향 출력과는 독립된 경로 curvature+IMU 신호로) "직진 확정"을 판단해준 프레임에는
        # turn_now/turn_preview에 낀 잔여 노이즈로 인한 유령 감속을 무시하고 그대로 전속력을
        # 낸다. is_straight가 아닌 프레임(코너 포함 전부)은 기존 연속값 로직을 그대로 쓴다 —
        # 코너 감속 감도 자체는 전혀 안 바뀜.
        is_straight = getattr(self.pure_pursuit, 'is_straight', False)
        turn_for_speed = 0.0 if is_straight else max(turn_now, turn_preview * 0.3)
        target_speed = max(SPEED_CORNER_MIN,
                           SPEED_NORMAL * (1.0 - 0.90 * turn_for_speed ** 3))
        # 코너 진입(회전반경 감소) 시 추가 감속 — 기존 turn_for_speed 기반 감속과는 독립적으로
        # 계산해서 더 낮은 쪽을 쓴다(대체가 아니라 추가 안전판). 직진 확정 중엔 이 반경 계산도
        # 같은 이유로 건너뛴다(1.0 = 감속 없음).
        corner_radius_scale = 1.0 if is_straight else self._corner_radius_speed_scale()
        target_speed = max(SPEED_CORNER_MIN, target_speed * corner_radius_scale)
        # [2026-08-18] SPEED_LL_DEGRADED 캡 제거 — DL_CENTER_MODE='ll'일 때만 의미있던
        # 로직인데 현재 DL_CENTER_MODE='da'로 완전히 전환되어 차선(ll) 기반 주행을
        # 더 이상 쓰지 않는다(요청 반영). config.py SPEED_LL_DEGRADED 상수도 함께 삭제.
        # [2026-08-11] LANE_STALE_SEC 이상 새 차선인식 결과가 안 나온 상태(perc_lane()의
        # lane_stale, config.py LANE_STALE_SEC 주석 참고) — 조향 자체는 이미 self.lane_path가
        # 고정돼 있어 안전하게(발산 없이) 마지막 판단을 유지하지만, 그것만으론 "지금 인지가
        # 죽어있다"는 걸 아무도 모른다. 코너가 아닌데도 이 속도로 깎이면 그 부자연스러움
        # 자체가 사람이 알아챌 수 있는 신호가 되도록 일부러 감속으로 표시한다(요청 반영).
        # 정상적인 프레임 재사용(추론이 20Hz보다 느려 몇 틱 같은 값을 재사용하는 것,
        # dl_lane.py 모듈 상단 주석의 의도된 동작)은 result_seq가 계속 바뀌므로 여기 안 걸림 —
        # 진짜로 갱신이 멈춘 경우에만 발동한다.
        if self.lane_stale:
            target_speed = min(target_speed, SPEED_LANE_STALE)
        # [2026-08-17m] lane_stale은 "워커가 죽었을 때"만 잡고, "워커는 살아서 매 틱 새
        # 결과를 내지만 그 결과가 연속으로 무효"인 경우는 못 잡는다(위 self.lane_unstable
        # 주석, config.py LANE_UNSTABLE_FRAMES 주석 참고) — 커브 진입에서 da 밴드 핏이
        # 몇 틱 연속 실패하는 실패모드가 실차에서 이 갭으로 빠져나가 무감속으로 트랙을
        # 이탈했다. lane_stale과 같은 SPEED_LANE_STALE 캡을 여기도 건다.
        if self.lane_unstable:
            target_speed = min(target_speed, SPEED_LANE_STALE)
        # [2026-08-18] avoid-hold 적용4(SPEED_AVOID_HOLD_BLOCKED 안전판) 삭제 — 실차 테스트에서
        # "속도 5 고정" 증상의 실제 원인으로 확인됨(README §2.43). TEST_DISABLE_B2_B3=True라
        # 실제 회피 기동(옆차선 이동)은 꺼져있는데 이 캡만 무관하게 계속 걸려서, 트리거를
        # 풀어줄 수단이 없어 무한정 5.0에 고정되는 구조였다. avoid_hold_active/avoid_hold_side
        # 자체(타이머, _update_avoid_hold())와 DA 클리핑 방향 편향(적용3,
        # perception/dl_lane.py set_avoid_hold()) 소비부는 그대로 유지 — 요청 반영(속도캡만 제거).
        speed_ratio = min(1.0, self._prev_speed / SPEED_NORMAL)
        corner_decay = CORNER_HOLD_DECAY_LO + (CORNER_HOLD_DECAY_HI - CORNER_HOLD_DECAY_LO) * speed_ratio
        self._corner_hold = max(turn_now, self._corner_hold * corner_decay)
        accel_step = SPEED_ACCEL_STEP * max(0.25, 1.0 - self._corner_hold)
        # [2026-08-17n] 정지(또는 데드존 이하)에서 새로 출발하는 틱 — SPEED_ACCEL_STEP
        # 계단 램프로 데드존(≈1.4)을 여러 틱에 걸쳐 오르면 그동안 바퀴는 안 움직이는데
        # (RPM=0, 역기전력 없음) duty만 계속 올라가 락터 전류가 가장 큰 구간에 가장 오래
        # 머문다 — "틱틱거림/초반 힘딸림"의 유력 원인(config.py SPEED_KICK_START 주석,
        # README §7.2). 이 조건이 참인 틱 1회만 accel_step 램프를 건너뛰고 SPEED_KICK_START로
        # 점프해 정지마찰 구간 체류를 줄인다 — target_speed가 이미 그보다 낮으면(코너 진입 등
        # 저속 유지 상황) 점프하지 않고 그대로 둔다.
        if self._prev_speed < SPEED_KICK_START < target_speed:
            target_speed = SPEED_KICK_START
        elif target_speed > self._prev_speed + accel_step:
            target_speed = self._prev_speed + accel_step
        self.ctrl_speed = target_speed
        self._prev_speed = target_speed

    def _vesc_live(self):
        """VESC 실측속도(self.v_mps)를 지금 믿을 수 있는가 — _speed_for_lookahead()와
        _imu_curvature_px()가 동일 기준으로 공유(2026-08-06 분리, 기존엔 _speed_for_lookahead()
        안에 인라인돼 있던 걸 IMU curvature 쪽도 똑같이 필요해져서 뺐다)."""
        return (self._vesc_t is not None
                and (time.time() - self._vesc_t) < VESC_STALE_SEC
                and abs(self.v_mps) >= VESC_MIN_SPEED_MPS)

    def _imu_live(self):
        """IMU 실측(self.imu_yaw)을 지금 믿을 수 있는가 — _vesc_live()와 동일 철학의
        스테일 가드. 기존엔 _imu_curvature_px() 안에 인라인돼 있었는데(imu_live 지역변수),
        [2026-08-18] _do_left_turn()의 yaw 목표각 판정도 똑같은 가드가 필요해져서 뺐다."""
        return self._imu_t is not None and (time.time() - self._imu_t) < IMU_STALE_SEC

    def _imu_curvature_px(self):
        """IMU 각속도(yaw_rate) + VESC 실측속도(v_mps)로 "차량이 지금 실제로 얼마나
        도는지" curvature(1/px)를 구해 pure_pursuit의 코너 감쇠(lookahead_curvature_gain)를
        보강한다(controller/pure_pursuit.py control()의 imu_curvature_px 주석 참고) —
        비전 경로만으로 뽑던 probe_curvature와 달리 픽셀 노이즈에 영향을 안 받는다.

        [2026-08-06] imu_yaw_rate를 그대로 쓰지 않고 IMU_YAW_RATE_EMA_ALPHA로 저역통과한
        self._imu_yaw_rate_ema를 쓴다 — probe_curvature는 경로 위 여러 점을 누적한 값이라
        어느 정도 스무딩이 걸려있는데, 자이로 순간값은 그런 스무딩이 없다. curvature
        damping이 두 값 중 절댓값이 큰 쪽을 그대로 채택하는 구조(controller/pure_pursuit.py
        control() 참고)라, 스무딩 없는 쪽이 노이즈 스파이크 한 프레임만으로 감쇠를 확
        눌러버릴 위험이 있었다.

        kappa_m = yaw_rate(rad/s) / v_mps(m/s) 는 물리적으로 옳은 실제 curvature(1/m)다.
        이걸 픽셀 curvature로 바꾸려면 DL_PIXELS_PER_METER(BEV 목적캔버스 스케일, =200px/m)로
        나누면 되는데, 이 환산은 dl+BEV 조합(self.lane_path가 그 스케일일 때)에서만 유효하다
        — PP_WHEELBASE_PX를 물리 기반 값으로 바꿀 때와 동일 전제(config.py 참고).
        부호는 안 맞춘다 — pure_pursuit이 abs()로만 소비하므로 crash/오조향 위험 없이
        magnitude만 넘기는 게 더 안전하다(IMU 부호규약 실차 미검증).

        IMU 또는 VESC 둘 중 하나라도 죽어있으면(stale/미수신) None을 반환해 pure_pursuit이
        기존처럼 probe_curvature 단독 판단으로 자동 폴백하게 한다."""
        if not (LANE_DETECTOR_BACKEND == 'dl' and DL_USE_BEV):
            return None
        if not (self._imu_live() and self._vesc_live()):
            return None
        self._imu_yaw_rate_ema = (IMU_YAW_RATE_EMA_ALPHA * self.imu_yaw_rate
                                   + (1.0 - IMU_YAW_RATE_EMA_ALPHA) * self._imu_yaw_rate_ema)
        kappa_m = self._imu_yaw_rate_ema / self.v_mps
        return kappa_m / DL_PIXELS_PER_METER

    def _speed_for_lookahead(self):
        """pure_pursuit의 속도 적응형 lookahead(PP_LOOKAHEAD_SPEED_GAIN)에 넣을 속도값을
        고른다(2026-08-06, VESC 실측속도 연동). 예전엔 항상 self._prev_speed(직전 "명령"
        속도, 모터단위)를 근사치로 썼는데, lookahead는 원래 "실제로 얼마나 빨리 달리고
        있는가"에 대한 개념이라 명령값은 근사일 뿐이다 — 모터 데드존/가속 지연/슬립처럼
        명령≠실제인 구간(§2.3에서 다룬 코너 급감속 직후 등)에서는 이 근사가 특히 틀어진다.

        VESC 실측값(self.v_mps, m/s)이 살아있으면(최근에 받았고 VESC_MIN_SPEED_MPS 이상)
        그걸 METERS_PER_SPEED_UNIT(README §6.5 실측 회귀)으로 "명령속도와 같은 단위"로
        역환산해서 쓴다 — 그러면 PP_LOOKAHEAD_SPEED_GAIN/PP_LOOKAHEAD_BASE_PX 등 기존에
        명령속도 스케일로 튜닝된 게인을 그대로 재사용할 수 있다(단위를 바꾸면 게인도 전부
        다시 튜닝해야 하므로). VESC 브리지가 안 떠 있거나 메시지가 끊겼거나(_vesc_t가
        VESC_STALE_SEC 이상 지남) 값이 너무 작으면(정지 근방, 노이즈 대비 신뢰 불가)
        예전처럼 self._prev_speed로 폴백한다 — cb_vesc()/VESC_MIN_SPEED_MPS 주석과 동일한
        가드 원칙."""
        if self._vesc_live():
            return abs(self.v_mps) / METERS_PER_SPEED_UNIT
        return self._prev_speed

    def _commit_speed_mps(self):
        """_s2_intersection()의 커밋 구간 이동거리 적분에 쓸 현재 속도(m/s) 추정.
        VESC 실측이 살아있으면(_vesc_live()) 그 값을 그대로 쓴다 — S2_COMMIT_DIST_M을
        이 값으로 적분하면 대회 당일 실속도가 얼마든 실제 이동거리 기준으로 맞는다.
        VESC가 죽어있을 때만 APPROACH_SPEED(명령속도)를 METERS_PER_SPEED_UNIT으로
        환산해 폴백한다 — 예전 시간 기반(S2_COMMIT_T)이 암묵적으로 가정하던 것과
        동일한 근사치라 VESC 장애 시에도 이전과 같은 동작으로 안전하게 열화된다."""
        if self._vesc_live():
            return abs(self.v_mps)
        return APPROACH_SPEED * METERS_PER_SPEED_UNIT

    def _lane_steer(self, path=None, vehicle_x=None):
        """path(ROI 픽셀좌표 경로, 가까운점→먼점)를 pure_pursuit(controller/pure_pursuit.py)로
        추종해 조향각(도)을 계산한다. 차량 기준점은 (vehicle_x, path[0]의 y좌표)로 둔다.

        인자를 생략하면(기본 호출부인 _lane_drive() 등) 기존과 동일하게 self.lane_path와
        ROI 하단 중앙(roi_w/2)을 쓴다 — path[0].y는 lane_util._fit_and_sample_path()가
        self.roi_h로 샘플링해둔 값이라 별도로 백엔드별 roi_h를 조회할 필요가 없다.
        [2026-08-11] _handle_lavacon()이 self.lavacon_path/vehicle_x=0.0을 명시적으로
        넘겨 호출한다 — 라바콘 조향 파라미터를 라인주행과 완전히 일치시키기 위해, 별도
        게인을 두지 않고 이 함수를 그대로 재사용하기로 한 결정(perc_lavacon() 주석 참고).

        경로가 비어있으면(첫 프레임, 혹은 roi_w를 아직 모르는 백엔드) 직전 조향각을
        그대로 유지한다 — pure_pursuit.control()이 내부적으로 이렇게 처리한다.
        [2026-08-14] STEERING_CONTROLLER로 pure_pursuit/lqr 중 고르던 분기를 LQR 컨트롤러
        제거와 함께 없앴다 — 이제 pure_pursuit 고정."""
        if path is None:
            path = self.lane_path
            # [2026-08-17] roi_w/2.0(캔버스 단순 절반) 대신, 백엔드가 노출하면
            # vehicle_center_x(BEV 사다리꼴 실측 기하로 구한 실제 차량 중심,
            # dl_lane.py DL_BEV_VEHICLE_CENTER_X 참고)를 우선 쓴다. 그 속성이 없는
            # 백엔드(hough/classic_cv)는 기존과 동일하게 roi_w/2.0로 폴백한다.
            roi_w = getattr(self.lane_detector, 'roi_w', 0) or 0
            vehicle_x = getattr(self.lane_detector, 'vehicle_center_x', None)
            if vehicle_x is None:
                vehicle_x = roi_w / 2.0
        if not path or vehicle_x is None:
            return self.pure_pursuit.prev_steer_deg
        vehicle_xy = (vehicle_x, path[0][1])
        return self.pure_pursuit.control(path, vehicle_xy, speed=self._speed_for_lookahead(),
                                          imu_curvature_px=self._imu_curvature_px())

    # [DEBUG_VIZ_STEER] 조향 컨트롤러가 이번 주기에 "새로 계산"했는지(초록/현재값 반영)
    # "직전 조향각을 그대로 유지"했는지(주황/직전값 유지)를 별도 창으로 바로 확인.
    # cv2 기본폰트는 한글을 못 그려서 kr_text.put_text_kr_multi(PIL 기반)로 그린다
    # (kr_text.py 상단 주석 참고). control_loop()에서 매 주기 호출 — 이 함수를 부르는
    # 시점(run_mission_fsm 직후, behavior override 이전)이 중요: 표시값은 self.ctrl_angle이
    # 아니라 controller.prev_steer_deg를 쓴다 — B1/B2/B3 behavior가 나중에 self.ctrl_angle을
    # 덮어써도(apply_behavior_override) 이 창은 항상 "차선 조향 컨트롤러 자체"의 상태를
    # 보여주기 위함이다.
    def _debug_viz_steer(self):
        controller = self.pure_pursuit
        held = getattr(controller, 'held', False)

        # 주황 = 직전값 유지(경로 부족/노이즈로 신뢰 못 함), 초록 = 현재값 반영(정상 계산됨)
        status_color = (0, 140, 255) if held else (0, 200, 0)
        status_text = '직전값 유지 (경로 부족/노이즈)' if held else '현재값 반영'

        lines = [
            ('컨트롤러: Pure Pursuit', (10, 8), (255, 255, 255), 20, 'Controller: Pure Pursuit'),
            (f'상태: {status_text}', (10, 40), status_color, 20,
             f'Status: {"HOLD (prev)" if held else "LIVE (fresh)"}'),
            (f'조향각: {controller.prev_steer_deg:+.1f}도', (10, 72), (255, 255, 255), 20,
             f'Steer: {controller.prev_steer_deg:+.1f}deg'),
        ]
        # [2026-08-06] IMU curvature damping 보강이 실제로 반영되는지(그냥 코드만
        # 들어가고 IMU/VESC 둘 중 하나가 죽어서 조용히 폴백 중인 건 아닌지) 여기서
        # 바로 확인할 수 있게 노출한다 — _imu_curvature_px()가 None을 주면
        # pure_pursuit.last_imu_curvature_px도 None으로 유지되므로(control() 참고)
        # None이면 "지금은 probe_curvature 단독 판단 중"이라는 뜻.
        lookahead_px = getattr(controller, 'last_lookahead_px', None)
        curvature = getattr(controller, 'last_curvature', None)
        imu_kappa = getattr(controller, 'last_imu_curvature_px', None)
        if lookahead_px is not None and curvature is not None:
            lines.append((f'lookahead: {lookahead_px:.0f}px  curvature: {curvature:+.4f}',
                           (10, 8 + 32 * len(lines)), (255, 255, 255), 18,
                           f'lookahead: {lookahead_px:.0f}px  curvature: {curvature:+.4f}'))
        imu_color = (0, 200, 0) if imu_kappa is not None else (140, 140, 140)
        imu_text = f'{imu_kappa:+.4f}' if imu_kappa is not None else '미반영(IMU/VESC 확인)'
        lines.append((f'IMU curvature: {imu_text}', (10, 8 + 32 * len(lines)), imu_color, 18,
                       f'IMU curvature: {imu_text if imu_kappa is not None else "N/A"}'))

        # [2026-08-17] 명시적 직진 모드(README §0.5.9, 조향 데드존 + 코너감속 둘 다 이 상태를
        # 참고) 표시 — 확정까지 남은 프레임 수를 같이 보여줘서, 직전 몇 프레임이 이미
        # 저곡률이었는지(곧 확정될지)를 실차에서 바로 확인할 수 있게 한다.
        is_straight = getattr(controller, 'is_straight', False)
        straight_frames = getattr(controller, '_straight_frames', 0)
        straight_confirm = getattr(controller, 'straight_confirm_frames', 0)
        straight_color = (0, 200, 0) if is_straight else (140, 140, 140)
        straight_text = f'확정({straight_frames}프레임, 조향+속도 적용)' if is_straight else f'대기({straight_frames}/{straight_confirm})'
        lines.append((f'직진모드: {straight_text}', (10, 8 + 32 * len(lines)), straight_color, 18,
                       f'Straight mode: {"CONFIRMED (steer+speed)" if is_straight else f"waiting({straight_frames}/{straight_confirm})"}'))

        # DA(주행가능영역) 면적 — DL_DA_MAX_AREA_PX 실측 튜닝용. 원래 da_debug라는 별도
        # 창이었는데 조향 상태랑 같이 한눈에 보고 싶다는 요청으로 이 창에 합쳤다(2026-08-06).
        # 'dl' 백엔드 전용(_slide 속성이 없는 hough/classic_cv 백엔드에서는 0px로만 표시됨).
        #   [2026-08-06] 마스크 전체 대비 비율이 아니라 절대 픽셀값으로 바꿨다(config.py의
        #   DL_DA_MAX_AREA_PX 주석 참고) — 직선 구간에서 이 창의 "largest" 값을 그대로 읽어서
        #   DL_DA_MAX_AREA_PX 실측값으로 쓰면 된다.
        #   초록 : 임계값 대비 80% 미만 — 여유 있음
        #   주황 : 80~100% — 임계값에 근접
        #   빨강 : 100% 초과 — 이번 프레임 실제로 outlier 처리됨(_largest_da_component() 참고)
        slide = getattr(self.lane_detector, '_slide', None)
        da_largest = getattr(slide, 'da_largest_area_px', 0) if slide is not None else 0
        da_chosen = getattr(slide, 'da_chosen_area_px', 0) if slide is not None else 0
        da_fallback = getattr(slide, 'da_fallback_used', False) if slide is not None else False
        da_ratio_of_max = (da_largest / DL_DA_MAX_AREA_PX) if DL_DA_MAX_AREA_PX > 0 else 0.0
        if da_largest > DL_DA_MAX_AREA_PX:
            da_color = (0, 0, 220)
            da_kr = f'임계값 초과(outlier) {da_ratio_of_max*100:.0f}%'
            da_en = f'OVER THRESHOLD {da_ratio_of_max*100:.0f}%'
        elif da_ratio_of_max >= 0.8:
            da_color = (0, 140, 255)
            da_kr, da_en = f'임계값 근접 {da_ratio_of_max*100:.0f}%', f'NEAR THRESHOLD {da_ratio_of_max*100:.0f}%'
        else:
            da_color = (0, 200, 0)
            da_kr, da_en = f'정상 {da_ratio_of_max*100:.0f}%', f'OK {da_ratio_of_max*100:.0f}%'
        lines.append((f'DA 면적: {da_kr}', (10, 8 + 32 * len(lines)), da_color, 20, f'DA area: {da_en}'))
        lines.append((
            f'DA largest:{da_largest}px chosen:{da_chosen}px'
            f'{" [FALLBACK]" if da_fallback else ""} max:{DL_DA_MAX_AREA_PX}px',
            (10, 8 + 32 * len(lines)), (255, 255, 255), 18,
            f'DA largest:{da_largest}px chosen:{da_chosen}px max:{DL_DA_MAX_AREA_PX}px'))

        # [2026-08-10] 시드 위치(차량 바로 앞) 덩어리의 bounding box 가로폭 실측용 —
        # 면적 대신 너비로 da 판단 로직을 바꿀지 결정하기 전에 실제 값 분포부터 관찰하려는
        # 목적. 판단 로직에는 아직 전혀 안 쓰인다(dl_lane.py DLSlideWindow.da_seed_width_px
        # 주석 참고) — 아직 임계값이 없어 색 구분 없이 그냥 값만 표시한다. 시드 위치에
        # 아무것도 없었던 프레임(가려짐 등)은 0px으로 뜬다.
        da_seed_width = getattr(slide, 'da_seed_width_px', 0) if slide is not None else 0
        lines.append((
            f'DA seed width:{da_seed_width}px',
            (10, 8 + 32 * len(lines)), (255, 255, 255), 18,
            f'DA seed width:{da_seed_width}px'))

        # [2026-08-18] SPEED_LL_DEGRADED 표시 블록 제거 — DL_CENTER_MODE='ll' 전용이었는데
        # 현재 항상 'da'라 이 조건 자체가 죽어있었다(요청 반영, 위 target_speed 계산부의
        # 동일 제거 사유 참고).

        canvas = np.full((8 + 32 * len(lines) + 16, 380, 3), 30, dtype=np.uint8)
        put_text_kr_multi(canvas, lines)
        # 한글 폰트가 없어 fallback(영문)만 그려진 경우에도 색 테두리만으로 상태 구분 가능하게.
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), status_color, 3)

        cv2.imshow('steer_debug', canvas)
        cv2.waitKey(1)

    # [DEBUG_VIZ_VESC] VESC 실측속도(/vesc_speed_erpm) 연동이 실제로 살아있는지 한눈에 보여주는
    # 창(2026-08-06 LQR 브랜치에서 이식). cb_vesc()가 self.v_mps/self._vesc_t를 갱신하는 동안은
    # 로그를 계속 지켜보지 않는 한 "지금 진짜 실측 속도를 받고 있는지" 알기 어려워서 별도 창으로
    # 뺐다. 상태 3가지:
    #   빨강/NEVER_RECEIVED : 지금까지 메시지를 한 번도 못 받음 — ROS1쪽 launch/vesc_speed_bridge.py
    #                         노드가 안 떠 있거나, 토픽 이름이 다르거나, ros1_bridge가 이 토픽을
    #                         안 넘기고 있거나.
    #   주황/STALE          : 예전엔 받았는데 VESC_STALE_SEC 이상 새 메시지가 없음 — 브리지 노드나
    #                         VESC 드라이버가 죽었을 가능성.
    #   초록/LIVE           : 정상 수신 중.
    # control_loop()에서 매 주기 호출.
    def _debug_viz_vesc(self):
        now = time.time()
        if self._vesc_t is None:
            color = (0, 0, 220)
            text_kr, text_en = '/vesc_speed_erpm 메시지 수신 안 됨', 'NO MESSAGE RECEIVED YET'
        else:
            age = now - self._vesc_t
            if age > VESC_STALE_SEC:
                color = (0, 140, 255)
                text_kr, text_en = f'수신 끊김 (마지막 {age:.1f}초 전)', f'STALE (last {age:.1f}s ago)'
            else:
                color = (0, 200, 0)
                text_kr, text_en = f'정상 수신 중 ({age*1000:.0f}ms 전)', f'LIVE ({age*1000:.0f}ms ago)'

        gain_fed = abs(self.v_mps) >= VESC_MIN_SPEED_MPS
        canvas = np.full((160, 380, 3), 30, dtype=np.uint8)
        lines = [
            (f'VESC 연동: {text_kr}', (10, 8), color, 16, f'VESC link: {text_en}'),
            (f'v_mps: {self.v_mps:+.3f} m/s', (10, 44), (255, 255, 255), 20,
             f'v_mps: {self.v_mps:+.3f} m/s'),
            (f'LQR 게인 반영: {"O" if gain_fed else "X (VESC_MIN_SPEED_MPS 미만)"}',
             (10, 76), (255, 255, 255) if gain_fed else (150, 150, 150), 16,
             f'LQR gain fed: {"YES" if gain_fed else "NO"}'),
            ('토픽: /vesc_speed_erpm (std_msgs/Float32, vesc_speed_bridge.py 경유)',
             (10, 108), (180, 180, 180), 12, 'topic: /vesc_speed_erpm'),
        ]
        put_text_kr_multi(canvas, lines)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), color, 3)

        cv2.imshow('vesc_debug', canvas)
        cv2.waitKey(1)

    # [DEBUG_VIZ_AVOID_HOLD] avoid-hold(§2.32, avoid_hold_improvement_proposal.md) 전용
    # 상태창 — 2026-08-15에 추가. 지금 유예가 걸려있는지/왜 걸렸는지/직전엔 왜 풀렸는지/
    # 방향 힌트가 뭔지를 실차에서 한눈에 보기 위한 것과 동시에, 이 기능이 새로 들여온
    # 파라미터 중 실측이 안 된 값들을 매 프레임 같이 띄워서 "이 숫자는 아직 지어낸
    # 값"이라는 걸 계속 상기시키는 용도(실측 절차는 avoid_hold_measurement_todo.md 참고).
    # 아래 5개(RATE_GAIN/SEC_MAX/RELEASE_DIST_M/DA_AREA_JUMP_RATIO/DIR_BIAS_PX)는
    # 그 문서에 적힌 측정 절차를 실차에서 그대로 따라가며
    # 이 창의 실시간 값을 관찰하는 용도로도 쓰인다(예: da_area_jump가 실제 통과 순간에만
    # True로 뜨는지, 노이즈 프레임에서도 뜨는지 여기서 직접 눈으로 확인). control_loop()
    # 에서 매 주기 호출.
    def _debug_viz_avoid_hold(self):
        now = time.time()
        remaining = max(0.0, self._avoid_hold_until_t - now)
        active_color = (0, 200, 0) if self.avoid_hold_active else (110, 110, 110)
        side_label = {1: '오른쪽(+1)', -1: '왼쪽(-1)', 0: '방향없음/양쪽막힘(0)'}
        side_color = (0, 140, 255) if self.avoid_hold_side == 0 else (255, 255, 255)

        slide = getattr(self.lane_detector, '_slide', None)
        da_area = getattr(slide, 'da_chosen_area_px', 0)
        da_jump = bool(getattr(self.lane_detector, 'da_area_jump', False))

        UNMEASURED = (60, 160, 255)  # 실측 미검증 파라미터 강조색(주황) — 다른 창의 STALE 색과 통일
        lines = [
            (f'AVOID-HOLD: {"활성" if self.avoid_hold_active else "대기"}  '
             f'(남은 {remaining:.2f}s / hold_sec={self.avoid_hold_hold_sec:.2f}s)',
             (10, 8), active_color, 17,
             f'AVOID-HOLD: {"ACTIVE" if self.avoid_hold_active else "idle"} '
             f'({remaining:.2f}s left / hold_sec={self.avoid_hold_hold_sec:.2f}s)'),
            (f'직전 해제 사유: {self.avoid_hold_release_reason or "(아직 없음)"}   '
             f'방향 힌트: {side_label[self.avoid_hold_side]}',
             (10, 34), side_color, 14,
             f'last release: {self.avoid_hold_release_reason or "(none)"}  '
             f'dir hint: {self.avoid_hold_side:+d}'),
            (f'트리거 입력 — front={self.obstacle_front} dist={self.obstacle_dist:.2f}m '
             f'rate={self.obstacle_rate:+.2f}m/s  v_mps={self.v_mps:+.2f}',
             (10, 62), (255, 255, 255), 13,
             f'trigger — front={self.obstacle_front} dist={self.obstacle_dist:.2f}m '
             f'rate={self.obstacle_rate:+.2f} v={self.v_mps:+.2f}'),
            (f'target_speed_est(트리거 스냅샷) = {self._avoid_hold_target_speed_est:+.2f} m/s',
             (10, 84), (255, 255, 255), 13,
             f'target_speed_est(snapshot) = {self._avoid_hold_target_speed_est:+.2f} m/s'),
            (f'da 연속성 보조트리거(적용2) — chosen_area={da_area}px  jump={"O" if da_jump else "X"}',
             (10, 106), (0, 200, 255) if da_jump else (150, 150, 150), 13,
             f'da continuity — area={da_area}px jump={"YES" if da_jump else "no"}'),
            (f'조기해제 진행 — front=False {self._avoid_hold_release_cnt}/{AVOID_HOLD_RELEASE_CONFIRM_FRAMES}'
             f'  마지막유효dist={self._avoid_hold_last_valid_dist:.2f}m (>= {AVOID_HOLD_RELEASE_DIST_M}m 필요)',
             (10, 128), (200, 200, 200), 12,
             f'early-release {self._avoid_hold_release_cnt}/{AVOID_HOLD_RELEASE_CONFIRM_FRAMES}, '
             f'last_valid_dist={self._avoid_hold_last_valid_dist:.2f} (need>={AVOID_HOLD_RELEASE_DIST_M})'),
            ('── ★ 실차 미검증(실측 필요) — avoid_hold_measurement_todo.md 참고 ★',
             (10, 156), UNMEASURED, 13, '-- UNMEASURED, see avoid_hold_measurement_todo.md --'),
            (f'RATE_GAIN={AVOID_HOLD_RATE_GAIN}   SEC_MAX={AVOID_HOLD_SEC_MAX}s   '
             f'RELEASE_DIST_M={AVOID_HOLD_RELEASE_DIST_M}m',
             (10, 178), UNMEASURED, 12,
             f'RATE_GAIN={AVOID_HOLD_RATE_GAIN} SEC_MAX={AVOID_HOLD_SEC_MAX} '
             f'RELEASE_DIST_M={AVOID_HOLD_RELEASE_DIST_M}'),
            (f'DA_AREA_JUMP_RATIO={AVOID_HOLD_DA_AREA_JUMP_RATIO}   DIR_BIAS_PX={AVOID_HOLD_DIR_BIAS_PX}px',
             (10, 198), UNMEASURED, 12,
             f'DA_AREA_JUMP_RATIO={AVOID_HOLD_DA_AREA_JUMP_RATIO} '
             f'DIR_BIAS_PX={AVOID_HOLD_DIR_BIAS_PX}'),
        ]
        canvas = np.full((222, 620, 3), 30, dtype=np.uint8)
        put_text_kr_multi(canvas, lines)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), active_color, 3)

        cv2.imshow('avoid_hold_debug', canvas)
        cv2.waitKey(1)

    def _lane_pid(self, offset, deadzone=LANE_DEADZONE):
        """차선 중앙편차(offset)를 PID 제어로 조향각(angle)으로 변환한다."""
        if abs(offset) < deadzone:
            offset = 0.0
        self._pid_integral += offset
        # [anti-windup] 적분항 기여를 ±LANE_INTEGRAL_TERM_MAX(도)로 제한한다.
        #   클램프가 없으면 카메라 정렬오차·좌우 검출 신뢰도 차이 등 아주 작은 편향도
        #   시간이 지날수록 계속 누적돼서 결국 한쪽으로 조향이 쏠리는 문제가 생긴다
        #   (실측으로 재현됨 — "주행하다가 점점 오른쪽으로 도는" 증상의 주 원인).
        integral_limit = (LANE_INTEGRAL_TERM_MAX / LANE_KI) if LANE_KI else 0.0
        self._pid_integral = float(np.clip(self._pid_integral, -integral_limit, integral_limit))
        deriv = offset - self._pid_prev_error
        self._pid_prev_error = offset
        boost_ratio = min(1.0, max(0.0, abs(offset) - LANE_CORNER_MIN) / (LANE_CORNER_REF - LANE_CORNER_MIN))
        kp_eff = LANE_KP * (1.0 + LANE_CORNER_BOOST * boost_ratio)
        angle = kp_eff*offset + LANE_KI*self._pid_integral + LANE_KD*deriv
        return float(np.clip(angle, -ANGLE_MAX, ANGLE_MAX))

    def apply_behavior_override(self):
        """Behavior 상태에 따라 Mission이 계산한 ctrl_angle/ctrl_speed를 덮어쓴다."""
        if self.behavior_state == BehaviorState.B1_LAVACON:
            self._handle_lavacon()
        elif self.behavior_state == BehaviorState.B2_OBSTACLE:
            self._cross_check_obstacle_motion('B2')
            self._handle_fixed_obstacle()
        elif self.behavior_state == BehaviorState.B3_VEHICLE:
            self._cross_check_obstacle_motion('B3')
            self._handle_overtake()
        # B0_NORMAL: 아무것도 안 함(Mission 출력 그대로)

    # [2026-08-11] 정적/동적 분류는 여전히 Phase(B2=고정장애물, B3=방해차량, 순차 미션
    #   설계)가 기준이고 이 함수가 그걸 바꿔타지는 않는다 — 실시간 속도추정을 판단
    #   기준으로 승격하면 라이다 노이즈에 더 취약해질 위험이 있고, 대회 규정상 미션이
    #   고정 순서/구간으로 보장되므로 Phase 기준으로 충분하다는 판단(사용자 확인,
    #   controller/obstacle_avoidance.py 상단 주석과 동일 전제). 여기서는 실측
    #   (self.obstacle_rate, self.v_mps)이 그 전제와 어긋날 때 로그만 남긴다 —
    #   README §5 "알려진 한계"의 "콘이 남아있는데 B3로 오인 진입" 같은 케이스를
    #   실차 로그로 잡아내기 위함.
    def _cross_check_obstacle_motion(self, tag):
        """target_speed_est ≈ v_mps + obstacle_rate. 타겟이 정지해 있으면 자차가
        다가가는 속도만큼 obstacle_rate가 음수라 합이 0에 가깝고, 자차와 같은 속도로
        달리면 obstacle_rate≈0이라 합이 v_mps에 가깝다(perc_obstacle() 주석 참고)."""
        if not (self.obstacle_front and self._vesc_live()):
            return

        target_speed_est = self.v_mps + self.obstacle_rate
        looks_static = abs(target_speed_est) < OBSTACLE_STATIC_SPEED_TH_MPS

        if tag == 'B2' and not looks_static:
            self.get_logger().warn(
                f'[B2] Phase는 고정장애물인데 obstacle_rate 기준 타겟이 움직이는 것처럼 '
                f'보임(추정 속도={target_speed_est:+.2f}m/s) — 오검출/오판 의심',
                throttle_duration_sec=1.0)
        elif tag == 'B3' and looks_static:
            self.get_logger().warn(
                f'[B3] Phase는 방해차량인데 obstacle_rate 기준 타겟이 정지해있는 것처럼 '
                f'보임(추정 속도={target_speed_est:+.2f}m/s) — 오검출/오판 의심',
                throttle_duration_sec=1.0)

    # ── B1-라바콘: 보로노이 편차 기반 P제어 ──
    def _handle_lavacon(self):
        """
        Phase.LAVACON 동안 항상 활성(트리거 조건 없음).
        우측 콘이 연속 LAVACON_DONE_FRAMES 프레임 미검출되면 고정장애물 구간으로 전환.

        [2026-08-11] 조향은 라바콘 전용 P게인(LAVACON_KP, 폐기) 대신 _lane_steer()를
        그대로 재사용한다 — 라인주행(_lane_drive())과 조향 파라미터(self.pure_pursuit
        인스턴스, PP_* 게인, ANGLE_MAX/ANGLE_RATE_MAX)를 완전히 일치시키기로 한 결정.
        self.lavacon_path는 perc_lavacon()이 라이다 미터 좌표를
        self.lane_path와 같은 px 스케일로 변환해둔 것이고, 차량 기준점은 그 변환의
        원점인 (0.0, path[0].y)다.
        """
        self.ctrl_angle = self._lane_steer(path=self.lavacon_path, vehicle_x=0.0)
        self.ctrl_speed = SPEED_LAVACON

        if self.lavacon_done:
            self._lavacon_empty_cnt += 1
            if self._lavacon_empty_cnt >= LAVACON_DONE_FRAMES:
                self._lavacon_empty_cnt = 0
                self._pid_prev_error = 0.0
                self._pid_integral   = 0.0
                self._lavacon_engaged = False   # B1 진입 latch 해제 (구간 재진입 대비)
                self.phase = Phase.OBSTACLE_ZONE
                self.get_logger().info('[LAVACON] 구간 통과 완료 → 장애물 구간')
        else:
            self._lavacon_empty_cnt = 0

    # ── B2-고정장애물 회피 ──
    #   차선 2개 + 넘어도 되는 노란 중앙선 구조라, 방향은 '반대편 차선' 하나로 정해진다.
    #   좌우 선택 로직은 ObstacleAvoidance.decide_lane() 이 lane_side 로 처리한다.
    def _handle_fixed_obstacle(self):
        if not self._da_avoidance_failed():
            # da 기반 경로(Mission의 lane-follow 출력)가 알아서 피하고 있다고 신뢰 —
            # TargetPassing으로 덮어쓰지 않고 이번 틱은 그냥 둔다.
            return

        self._run_passing(self.obstacle_controller, 'B2')

    # [2026-08-11] da 기반 회피가 이 정지 장애물을 알아서 피하고 있다고 믿을 수 있는지
    #   판단한다. False면 TargetPassing 개입 없이 da 경로 그대로, True면 위
    #   _handle_fixed_obstacle()이 TargetPassing(실측 기반 하드코딩 SHIFT/ALONGSIDE/
    #   RETURN)으로 override한다. Hybrid A*(검색 기반) 대신 이 하드코딩 방식을 폴백으로
    #   쓰기로 함 — 구조화된 2차선 환경에서 검색은 과한 방식이라는 기존 결론(§4)과 같은
    #   이유.
    #
    #   실패로 보는 조건 두 가지(OR):
    #     ① 경로 끊김/불안정 — self.lane_valid/self.lane_stale. 이건 지금도 실제로
    #        동작하는 신호다(perc_lane(), §2.24 LANE_STALE_SEC).
    #     ② da가 장애물을 반영해서 회피했다는 근거가 없음 — da 세그멘테이션
    #        (perception/dl_lane.py)은 아직 차선표시만 학습돼 있고 장애물 인지가
    #        전혀 없다(2026-08-11 기준). 그래서 지금은 이 조건이 항상 참이고, 결과적으로
    #        B2 트리거만 걸리면 매번 TargetPassing이 켜진다 — 오늘 기준 동작은 이 함수를
    #        넣기 전과 동일하다. da가 장애물 인지형으로 바뀌는 날 이 한 줄(da_unaware_
    #        of_obstacle)만 실제 판단 로직으로 교체하면 되고, TargetPassing 쪽은 손댈
    #        필요가 없다 — 그게 이 함수를 분리해둔 이유다.
    def _da_avoidance_failed(self):
        path_broken = (not self.lane_valid) or self.lane_stale
        da_unaware_of_obstacle = True  # ★da가 장애물 인지형이 되면 실제 조건으로 교체

        return path_broken or da_unaware_of_obstacle

    # ── B3-방해차량 추월 ──★재설계 예정(임시 placeholder) ──
    # target_lane 반영 수정
    # 회피 후 복귀하는 로직 추가 : idle -> avoid -> idle+B2/B3 완료 => idel -> avoid -> return -> idle
    # (2026-08-15: Phase.VEHICLE는 Phase.OBSTACLE_ZONE 통합으로 없어짐 — _mark_behavior_passed() 참고)
    def _handle_overtake(self):
        """방해차량 추월. 규정상 '방해차량이 주행하지 않는 차선'으로 지나가야 하고,
        그 차량이 1·2차선을 오가므로 매 프레임 타겟 횡위치를 다시 보고 필요하면
        진행 방향을 바꾼다(moving=True 로 생성된 컨트롤러가 이걸 처리).

        USE_HYBRID_ASTAR_FOR_B3=True면 _handle_overtake_astar()를 대신 쓰되, 그쪽이
        탐색실패로 TargetPassing에 폴백한 뒤(_b3_using_fallback)엔 이번 통과가 끝날
        때까지 계속 TargetPassing에 맡긴다 — 매틱 astar/TargetPassing을 오가면 진행
        중이던 SHIFT/ALONGSIDE 기동이 중간에 끊기기 때문(_run_passing()의 done 처리에서
        다음 통과를 위해 다시 풀어준다)."""
        if USE_HYBRID_ASTAR_FOR_B3 and not self._b3_using_fallback:
            self._handle_overtake_astar()
            return

        self._run_passing(self.vehicle_controller, 'B3')

    # ── B3 대안: Hybrid A* 경로계획 방식 (USE_HYBRID_ASTAR_FOR_B3=True 일 때만) ──
    #   [2026-08-11] B2 쪽 Hybrid A* 대안(_handle_fixed_obstacle_astar)은 삭제됐다 —
    #   그 방식("1회 계획 후 최대 1초 재사용")은 애초에 정지 장애물 전제라 여기(타겟이
    #   움직이는 B3)엔 그대로 못 쓴다. 그래서 "그리드/충돌검사는 매틱, 전체 재탐색은
    #   트리거 기반"으로 별도 설계했다.
    #   재탐색 트리거 3가지(하나라도 걸리면 발동):
    #     ① 경로 무효화 — 남은 waypoint가 최신 그리드와 충돌 (즉시, 주기 무시)
    #     ② 타겟 진입 — 통과중인 방향으로 타겟이 SWITCH_FRAMES 연속 넘어옴 (즉시)
    #     ③ 주기적 — ASTAR_B3_REPLAN_TICKS 틱마다 최소 한 번
    #   ①②로 강제된 재탐색이 실패하면(빈 경로) 그 프레임은 이전 경로를 버리고 감속하며
    #   유예(ASTAR_B3_FAIL_GRACE_TICKS)를 준다 — 넘기면 TargetPassing으로 폴백한다.
    #   ③(주기적)만으로 트리거됐는데 새 탐색이 실패한 경우는 기존 경로가 아직 안전하다는
    #   뜻이므로(무효화 검사를 통과했으니) 그냥 기존 경로를 계속 따라간다.
    def _handle_overtake_astar(self):
        self.grid = self.occupancy.build(self.lidar_ranges)

        replan = self.path is None
        force = False  # True면 기존 경로를 더 못 믿는다는 뜻 — 재탐색 실패시 바로 폴백 카운트

        if not replan:
            if self._path_blocked(self.path, self.grid):
                replan = True
                force = True

            if self._b3_side < 0:
                cuts_in = self.obstacle_y > CENTER_DEADZONE_M
            elif self._b3_side > 0:
                cuts_in = self.obstacle_y < -CENTER_DEADZONE_M
            else:
                cuts_in = False

            if cuts_in:
                self._b3_switch_cnt += 1
                if self._b3_switch_cnt >= SWITCH_FRAMES:
                    replan = True
                    force = True
                    self._b3_switch_cnt = 0
            else:
                self._b3_switch_cnt = 0

            self._b3_tick += 1
            if self._b3_tick >= ASTAR_B3_REPLAN_TICKS:
                replan = True
                self._b3_tick = 0

        if replan:
            side = self.vehicle_controller.choose_side(
                self.obstacle_y, self.left_clear_confirmed, self.right_clear_confirmed, self.lane_side)

            if side == 0:
                # 양쪽 다 막혔다 — TargetPassing의 'blocked' 서행재시도 동작을 그대로 재사용.
                self._b3_astar_reset()
                self._b3_using_fallback = True
                self._run_passing(self.vehicle_controller, 'B3')
                return

            goal = self.planner.make_goal_by_side(self.obstacle_dist, side)
            start = PlannerNode(0.0, 0.0, 0.0)
            new_path = self.planner.plan(start, goal, self.grid)

            if new_path:
                self.path = new_path
                self.goal = goal
                self._b3_side = side
                self._b3_fail_cnt = 0
                self._b3_tick = 0
                # 로컬 pose 원점 리셋 (재계획 시점을 (0,0,0)으로 잡는 패턴) — target_idx도 같이 리셋해야
                # 한다. 안 하면 새 경로(짧을 수 있음)에 옛 인덱스가 그대로 남아
                # nearest_index()가 범위를 벗어나 다음 control()에서 인덱스 에러가 난다.
                self.vehicle_x = 0.0
                self.vehicle_y = 0.0
                self.vehicle_yaw = 0.0
                self._plan_ref_yaw = self.imu_yaw
                self._plan_last_t = time.time()
                self.stanley.reset()
            elif force or self.path is None:
                # 기존 경로를 더 못 믿는데(무효화/진입) 새 탐색도 실패했거나, 애초에
                # 경로가 아예 없었다 — 유예 프레임 소진 전까지는 감속만 하고 재시도.
                self._b3_fail_cnt += 1
                if self._b3_fail_cnt >= ASTAR_B3_FAIL_GRACE_TICKS:
                    self._b3_astar_reset()
                    self._b3_using_fallback = True
                    self._run_passing(self.vehicle_controller, 'B3')
                    return
                self.path = None
                self.ctrl_speed = max(SPEED_NORMAL * 0.15, 0.5)
                return
            # else: 주기적 트리거만으로 재탐색했는데 실패 — 기존 경로가 여전히 안전하다는
            #   뜻이므로(위에서 무효화 검사를 이미 통과) self.path를 그대로 유지하고 계속 추종.

        now = time.time()
        dt = now - self._plan_last_t
        self._plan_last_t = now

        # yaw는 IMU 실측 기준(B2와 동일). 속도는 B2와 달리 VESC 실측(self.v_mps)이 살아있으면
        # 그걸 우선 쓴다 — B3는 재탐색이 잦아 적분 구간(dt 누적)이 훨씬 짧아지므로 드리프트
        # 영향은 작지만, 이미 있는 실측 인프라(_vesc_live())를 쓰지 않을 이유가 없다.
        self.vehicle_yaw = self._yaw_delta(self._plan_ref_yaw)
        self.vehicle_speed = self.v_mps if self._vesc_live() else self.ctrl_speed
        self.vehicle_x += self.vehicle_speed * math.cos(self.vehicle_yaw) * dt
        self.vehicle_y += self.vehicle_speed * math.sin(self.vehicle_yaw) * dt

        steer, speed = self.stanley.control(
            self.vehicle_x, self.vehicle_y, self.vehicle_yaw, self.vehicle_speed, self.path
        )

        self.ctrl_angle = math.degrees(steer)
        self.ctrl_speed = speed

        if self.stanley.goal_reached(self.vehicle_x, self.vehicle_y, self.path):
            self._b3_astar_reset()
            self.stanley.reset()

            self._mark_behavior_passed('B3')  # [2026-08-15] Phase.OBSTACLE_ZONE 통합 — B2도 끝나야 DONE
            self.behavior_state = BehaviorState.B0_NORMAL

            self._pid_prev_error = 0.0
            self._pid_integral = 0.0

        if DEBUG_PLANNER:
            cv2.imshow(
                "Occupancy_B3",
                cv2.resize(self.grid, None, fx=4, fy=4)
            )
            cv2.waitKey(1)

    def _path_blocked(self, path, grid):
        """남은 경로 waypoint가 최신 그리드와 충돌하는지 저렴하게 재검사.
        collision()은 5점 투영뿐이라 그리드 생성보다도 훨씬 싸다 — 매틱 돌려도 된다."""
        for x, y, yaw in path[self.stanley.target_idx:]:
            if self.planner.collision(PlannerNode(x, y, yaw), grid):
                return True
        return False

    def _b3_astar_reset(self):
        self.path = None
        self.goal = None
        self._b3_tick = 0
        self._b3_switch_cnt = 0
        self._b3_fail_cnt = 0
        self._b3_side = 0

    # [2026-08-15] Phase.OBSTACLE_ZONE 통합(da_based_b2b3_proposal.md B안) — B2/B3가
    # 완료를 알리는 공통 지점. 예전엔 각자 다음 Phase를 직접 지정했는데(B2 완료 →
    # Phase.VEHICLE, B3 완료 → Phase.DONE, 트랙 순서가 고정이라는 가정), 이제 Phase가
    # 하나로 합쳐져서 "둘 다 최소 한 번 끝났는가"로 판단해야 한다 — 어느 쪽이 먼저
    # 끝나도 상관없다. 트랙에 둘 중 하나가 아예 없어서 영영 안 끝나면 Phase는
    # OBSTACLE_ZONE에 계속 남지만, 그래도 무해하다(run_behavior_fsm()이 매 프레임
    # 트리거를 다시 보므로 일반 차선주행과 다를 바 없다).
    def _mark_behavior_passed(self, tag):
        if tag == 'B2':
            self._b2_passed = True
        elif tag == 'B3':
            self._b3_passed = True
        if self._b2_passed and self._b3_passed:
            self.phase = Phase.DONE
            self.get_logger().info('[OBSTACLE_ZONE] B2/B3 모두 통과 완료 → DONE')

    # ── B2/B3 공통 실행부 ──
    def _run_passing(self, controller, tag):
        steer_offset, speed, done, status = controller.update(
            obstacle_front=self.obstacle_front,
            obstacle_dist=self.obstacle_dist,
            obstacle_y=self.obstacle_y,
            lane_offset=self.lane_offset,
            lane_lookahead=self.lane_lookahead,
            lane_side=self.lane_side,
            left_clear=self.left_clear_confirmed,
            right_clear=self.right_clear_confirmed,
            allow_maneuver=self._maneuver_allowed,
        )

        self.ctrl_angle = self._lane_pid(steer_offset)

        if status == 'blocked':
            # 양쪽 다 막혔다 → 흰 실선 밖으로 나가지 않고 서행하며 기다린다.
            #   규정상 '주행 중 정지'는 위험하다(멈춘 뒤 1분 내 재개 못하면 실격).
            #   그래서 완전정지 대신 최저속으로 유지하며 다음 프레임에 다시 판단한다.
            self.ctrl_speed = max(SPEED_NORMAL * 0.15, 0.5)
            self.get_logger().warn(f'[{tag}] 양쪽 통과 불가 — 서행 후 재시도',
                                   throttle_duration_sec=1.0)
            return

        self.ctrl_speed = speed

        if done:
            self._mark_behavior_passed(tag)
            self._pid_prev_error = 0.0
            self._pid_integral = 0.0
            if tag == 'B3':
                # astar가 실패해서 여기로 폴백해 있던 상태였다면, 이번 통과가 끝났으니
                # 다음 방해차량 조우 때는 다시 astar부터 시도할 수 있게 풀어준다.
                self._b3_using_fallback = False
            self.get_logger().info(f'[{tag}] 통과 완료 (phase={self.phase.name})')


    # #########################################################
    # [5] 메인 루프
    # #########################################################
    def control_loop(self):
        """
        20Hz(0.05초)마다 호출되는 제어의 심장.
        매 주기 '인지 → 판단 → 제어 → 발행' 한 사이클을 순서대로 실행한다.
        ※ Behavior 게이팅: S1(차선주행) 상태이면서 _behavior_enabled=True일 때만 B1/B2/B3가 작동.
          (S0/S2/S3 및 S1 최초 진입 구간에서는 꺼져서 오검출로 인한 오작동을 막는다)
        """
        self.perceive_all()                 # 1. 인지
        self._update_lap()                  #    바퀴 카운트(누적 yaw + 정지선)
        self.run_mission_fsm()              # 2. 판단(Mission)

        if DEBUG_VIZ_STEER:
            # behavior override 이전 시점 — 차선 조향 컨트롤러 자체의 상태를 보여준다
            # (_debug_viz_steer() 상단 주석 참고).
            self._debug_viz_steer()
        if DEBUG_VIZ_VESC:
            self._debug_viz_vesc()
        if DEBUG_VIZ_AVOID_HOLD:
            self._debug_viz_avoid_hold()

        if ENABLE_BEHAVIOR and self.mission_state == MissionState.S1_LANE_FOLLOW and self._behavior_enabled:
            self.run_behavior_fsm()         #    Behavior 상태 결정
            self.apply_behavior_override()  #    필요 시 조향/속도 덮어쓰기
        else:
            self.behavior_state = BehaviorState.B0_NORMAL   # OFF 구간은 항상 정상

        # pose_estimator는 behavior override까지 반영된 "최종" ctrl_angle로 갱신한다(차량이 실제로
        # 명령받는 조향각이 이거라서, 2026-08-06 LQR 브랜치에서 이식). v_mps=0.0 고정 상태
        # (vesc_speed_bridge 노드 미실행 등)에서도 그냥 안 움직이는 것으로 적분될 뿐 안전하게 동작한다.
        self.pose_estimator.update(self.v_mps, math.radians(self.ctrl_angle), 0.05)

        self.drive(self.ctrl_angle, self.ctrl_speed)   # 4. 발행
        if DEBUG_LOG:                                    # 5. 디버그
            self._print_debug()


    # #########################################################
    # [6] 유틸/디버그
    # #########################################################
    def _print_debug(self):
        """0.5초마다 여러 줄로 상태 덤프. 별도 터미널(rqt/topic echo) 없이 이 로그만으로
        센서 원시상태(카메라 살아있는지·신호색·라이다 포인트수)부터 트리거 카운터까지 확인 가능하게 함.
        80컬럼 좁은 터미널에서도 안 잘리게 줄당 길이를 짧게 나눴다.
          [SENS] cam = 카메라 나이(s, 값이 계속 커지면 미수신)
                 sig = 신호등 상태. 앞 R/L/S(0/1)는 이번 프레임 순간값, confirmS/L는 디바운스
                       통과 후 FSM이 실제로 보는 확정값(카운터/기준 SIG_CONFIRM_FRAMES 같이 표시)
                 lidar = 유효포인트수(min거리m, 나이s)
          [LANE] 차선편차px(검출여부) stale=LANE_STALE_SEC 이상 새 추론결과 없음(1) 여부
                 (SPEED_LANE_STALE로 강제감속 중이라는 뜻 — 코너 아닌데 감속되면 이거 확인)
                 / obs = 라이다 전방장애물(거리m,좌우,타입)
          lava   = 라바콘 보로노이 편차(구간종료 판정)
          trigL  = 라바콘 진입: 본선카운트/기준 (좌클러스터,우클러스터 검출여부)
          trigV  = 차량 진입:   본선카운트/기준
          [LAVA-ROI] 라바콘 트리거 ROI(전방0.3~3.0m,좌우2.0m) 안에 잡힌 점 개수(pts)와
                     그중 최대 연속(붙어있는 인덱스) 묶음 길이(run). run>=2여야 클러스터로
                     인정되어 L/R detected=True가 됨. pts=0이면 ROI 안에 아예 점이 없는 것
                     (LON_MIN/MAX·LAT_MAX 범위나 콘 배치 확인), pts>0인데 run<2면 점은
                     있지만 서로 인덱스가 안 붙어있어 클러스터로 안 뭉치는 것(노이즈/각도해상도 문제).
                     DEBUG_VIZ_LAVACON=True로 켜면 'lavacon_bev' 창에서 같은 정보를 시각으로 확인 가능.
                     masked= BODY_LO~BODY_HI(차체 자기가림 구간)를 "무조건 자기가림"으로 보고
                     지워버리기 전, 원본(raw) 라이다 값 기준 그 구간 안의 점 개수/최소거리.
                     이 값이 크고 거리도 콘 간격과 비슷하면, 그 마스크가 진짜 콘 반사까지
                     같이 지우고 있다는 뜻 — BODY_LO/BODY_HI 구간 재보정이 필요할 수 있음.
          [SIG](S0/S2 상태에서만 출력) 4구 신호등 원 검출이 어느 단계에서 막혔는지 진단:
            roi     = ROI 픽셀 좌표(t,b,l,r) — 신호등이 이 영역 안에 실제로 들어오는지 확인용
            circles = HoughCircles가 찾은 원 개수(0=원 자체를 못 찾음, 4가 아니면 배치/블러 의심)
            reason  = 실패 사유(OK=성공) — circle_count=N / vert_spread.../horiz_spread.../
                      gap[i]=... (배치 불량)
            bright  = 성공적으로 4개+배치 통과 시 좌→우(빨강,노랑,좌회전,직진) 밝기값
        """
        now = time.time()
        if now - self._last_debug_t < DEBUG_PERIOD: return
        self._last_debug_t = now

        cam_age = now - self._img_front_t if self._img_front_t else -1.0
        scan_age = now - self._scan_t if self._scan_t else -1.0
        if self.lidar_ranges is not None:
            r = np.asarray(self.lidar_ranges, dtype=np.float32)
            valid = r[np.isfinite(r) & (r > 0.0)]
            lidar_desc = f'{valid.size}pt(min={float(valid.min()):.2f}m,age={scan_age:.1f}s)' if valid.size else f'0pt(age={scan_age:.1f}s)'
        else:
            lidar_desc = 'NONE'

        lava_lp, lava_lrun, lava_rp, lava_rrun = self._lavacon_dbg
        masked_pts, masked_min = self._lavacon_mask_dbg
        masked_min_s = f'{masked_min:.2f}m' if masked_min >= 0 else 'N/A'

        sig_line = ''
        if self.mission_state in (MissionState.S0_WAIT_GREEN, MissionState.S2_INTERSECTION):
            sd = self.signal_detector
            reason = sd.s2_reject_reason or 'OK'
            sig_line = (
                f'\n  [SIG] roi={sd.s2_roi_px} circles={sd.s2_circle_count} '
                f'reason={reason} bright={sd.s2_brightness}'
            )

        sig_flags = (f'R{int(self.signal_red_on)}L{int(self.signal_left_on)}S{int(self.signal_straight_on)} '
                     f'confirmS{int(self.signal_straight_confirmed)}({self._sig_straight_cnt}/{SIG_CONFIRM_FRAMES})'
                     f'L{int(self.signal_left_confirmed)}({self._sig_left_cnt}/{SIG_CONFIRM_FRAMES})')
        self.get_logger().info(
            f'[{self.mission_state.name}|{self.behavior_state.name}|{self.phase.name}] '
            f'ang={self.ctrl_angle:+.1f} spd={self.ctrl_speed:.1f}\n'
            f'  [LAP] {self.lap}/{TOTAL_LAPS} 바퀴 '
            f'누적={math.degrees(self._yaw_accum):+.0f}도/{math.degrees(LAP_YAW_FULL):.0f} '
            f'경과={time.time() - self._lap_t0:.0f}s\n'
            f'  [SENS] cam={cam_age:.1f}s sig={sig_flags} lidar={lidar_desc}\n'
            f'  [LANE] lane={self.lane_offset:+.1f}({int(self.lane_valid)}) stale={int(self.lane_stale)} '
            f'side={"R" if self.lane_side >= 0 else "L"}차선 '
            f'obs={self.obstacle_front}({self.obstacle_dist:.1f}m,w={self.obstacle_width:.2f}m,{self.obstacle_type}) '
            f'lava={self.lavacon_offset:+.2f}(done={int(self.lavacon_done)})\n'
            f'  [TRIG] trigL={self._lavacon_trigger_cnt}/{LAVACON_TRIGGER_FRAMES}'
            f'(L{int(self.lavacon_left_detected)}R{int(self.lavacon_right_detected)}'
            f'Y{int(self.cone_detected_yolo)}) '
            f'trigV={self._vehicle_trigger_cnt}/{VEHICLE_TRIGGER_FRAMES}\n'
            f'  [LAVA-ROI] L pts={lava_lp} run={lava_lrun}(need>=2) '
            f'R pts={lava_rp} run={lava_rrun}(need>=2) '
            f'masked_raw_pts={masked_pts} masked_min={masked_min_s}'
            f'{sig_line}')

    def _log_signal_debug(self):
        """DEBUG_LOG_SIGNAL 전용 상세 로그. 전역 DEBUG_LOG의 [SIG] 요약 줄(0.5초 주기, roi/circles/
        reason/bright만 나열)과 달리, 신호등이 "왜" 안 잡히는지 단계별 원인 + 대응 힌트를 붙여서
        찍는다 — DEBUG_LOG를 꺼둔 채로 신호등만 디버깅하고 싶을 때 이것만 켜면 됨.
        perc_signal()에서 S0/S2 상태일 때만 호출된다(그 외 상태는 detect_s2() 자체를 안 돌림).
        0.2초 스로틀(대략 4~5프레임당 1번, 20Hz 기준) — 매 프레임 찍으면 터미널이 너무 빨리
        흘러가서 오히려 못 읽는다."""
        sd = self.signal_detector
        reason = sd.s2_reject_reason or 'OK'

        if reason == 'no_circles':
            hint = 'ROI 안에 원이 아예 안 잡힘 → 신호등이 ROI 밖이거나 노출/대비 문제 (SIG4_ROI_*)'
        elif 'too noisy' in reason:
            hint = '원이 너무 많이 잡힘(MAX_CANDIDATES 초과) → 반사광 등 잡음, ROI를 좁히는 것 고려'
        elif '(<4)' in reason:
            hint = '원이 4개 미만 → 가림/블러 또는 반지름 범위 밖 (SIG4_MIN/MAX_RADIUS)'
        elif 'vert_spread' in reason:
            hint = '찾은 4개가 세로로 너무 퍼짐 → 오검출이 섞였거나 카메라 각도 틀어짐 의심'
        elif 'horiz_spread' in reason:
            hint = '찾은 4개가 가로로 너무 퍼짐 → 오검출이 섞임 의심'
        elif 'gap[' in reason:
            hint = '인접한 두 원 사이 간격이 너무 좁음 → 반사광 등이 신호등 원 사이에 끼어듦'
        elif 'no_valid_4subset' in reason:
            hint = '원 5개 이상 중 배치조건을 만족하는 4개 조합이 없음 → 오검출 비율이 높음'
        else:
            hint = '배치검사 통과 — 아래 bright/lit이 실제 밝기 대비 판정 결과(정상 동작)'

        state_kr = ('좌회전' if self.signal_left_on else
                    '직진'   if self.signal_straight_on else
                    '정지(빨강)' if self.signal_red_on else '미검출')

        self.get_logger().info(
            f'[SIG-DEBUG] {self.mission_state.name} state={state_kr}\n'
            f'  roi(px)=(t={sd.s2_roi_px[0]},b={sd.s2_roi_px[1]},l={sd.s2_roi_px[2]},r={sd.s2_roi_px[3]}) '
            f'radius={SIG4_MIN_RADIUS}~{SIG4_MAX_RADIUS}px circles_found={sd.s2_circle_count}\n'
            f'  reason={reason}\n'
            f'  → {hint}\n'
            f'  bright(빨강,노랑,좌회전,직진)={sd.s2_brightness} margin={SIG4_BRIGHT_MARGIN} '
            f'lit={[int(v) for v in sd.s2_lit]}\n'
            f'  confirm: 직진={self._sig_straight_cnt}/{SIG_CONFIRM_FRAMES} '
            f'좌회전={self._sig_left_cnt}/{SIG_CONFIRM_FRAMES}',
            throttle_duration_sec=0.2)


# #############################################################
# 메인
# #############################################################
def main(args=None):
    rclpy.init(args=args)
    node = TrackDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try: node.drive(0.0, 0.0)
        except Exception: pass
        # DL 백엔드는 백그라운드 추론 스레드를 띄우므로(dl_lane.DLLaneDetector) 정상 종료시킨다.
        # hough/classic_cv 백엔드는 stop()이 없으므로 getattr 기본값으로 조용히 건너뛴다.
        try: getattr(node.lane_detector, 'stop', lambda: None)()
        except Exception: pass
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
