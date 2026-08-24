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
#  │  [코스 시나리오] (2026-08-20, README §대회 규정 요약 기준으로 정정)│
#  │   1. 신호등 인식 후 출발(S0_SIGNAL) → 곧바로 Behavior 활성화     │
#  │   2. 차선주행(S1) 중 순서대로 진행:                              │
#  │      라바콘 주행(B1_LAVACON) → 고정장애물 회피(B2_OBSTACLE) →   │
#  │      방해차량 추월(B3_VEHICLE)                                  │
#  │   3. 트랙 중앙 분기점 4구 신호등(S0_SIGNAL 재진입, 3바퀴 중      │
#  │      2·3바퀴째 한 번만 좌회전=지름길 옵션 등장, 그 외엔 직진)    │
#  │      ├ 직진 → 차선주행(S1) 복귀, 다음 바퀴도 Behavior 순서대로   │
#  │      └ 좌회전(지름길, 최대 1회) → 좌회전 → 지름길(S3) → 좌회전   │
#  │         → 차선주행 복귀                                         │
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
# [2026-08-22k] CONE_LON_MAX/CONE_LAT_LIMIT/BOX_LON_START(→ LAVACON_PATH_LON_MAX/
#   LAVACON_PATH_LAT_LIMIT/LAVACON_BOX_LON_START) 별칭 삭제 — 이 값들을 쓰던 lavacon_bev의
#   박스 스택 시각화(흰 cone ROI 선/파란 박스 경계)를 지우면서 더 이상 안 쓰임
#   (_draw_lavacon_bev() 상단 주석 참고). BOX_LON_WIDTH(→ LAVACON_BOX_LON_WIDTH)는 자차
#   마커 위치 계산에 여전히 쓰여 유지.
from .perception.perc_lavacon import process_lavacon, nearest_cone_lateral, BOX_LON_WIDTH as LAVACON_BOX_LON_WIDTH
# [2026-08-22] §5.10 유령 점 임시 마스크 좌표 — lavacon_bev에 빨간 원으로 시각화해 마스킹
#   구역과 실제 유령 점 위치가 맞는지 눈으로 대조하기 위해 import(마스킹 자체는
#   perc_lavacon.py `_lidar_to_xy()`가 이미 적용).
from .perception.perc_lavacon import GHOST_POINT_X_M, GHOST_POINT_Y_M, GHOST_POINT_RADIUS_M
from .perception.hough_lane import HoughLaneDetector
from .perception.perc_floor import check_stopline, LaneDetector as ClassicLaneDetector
from .perception.lane_util import CameraProcessor, SlideWindow
from .perception.dl_lane import DLLaneDetector
from .perception.yolo_cone import YoloConeDetector
from .perception.yolo_vehicle import YoloVehicleDetector
from .perception.yolo_signal_state import YoloSignalStateDetector
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
        # [2026-08-22] 디버그 창을 처음 띄울 때만 cv2.moveWindow로 위치를 잡기 위한
        #   1회성 가드(DEBUG_WIN_POS_* 참고) — 매 프레임 moveWindow를 부르면 사용자가
        #   직접 옮겨놔도 다음 프레임에 도로 스냅백돼서, 창 이름을 여기 기록해두고 한 번만 옮긴다.
        self._dbg_windows_positioned = set()

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
        # [2026-08-18] 기준을 lane_valid(근접 전용)에서 path_ok(근접 OR 원거리)로 통일
        # (perc_lane() 참고, 요청 반영) — 실제 조향 경로(self.lane_path) 갱신 조건과
        # 정확히 같은 신호를 보게 되어, "경로는 갱신되는데 속도만 이유 없이 깎이는"
        # 불일치가 사라진다. 근접+원거리 둘 다 없어야(=path_ok=False) 스트릭이 쌓인다.
        self._lane_invalid_streak = 0
        self.lane_unstable = False
        self._lane_prev_width = 448.0  # 도로폭 직전값(px, EMA)
        self.lane_side   = LANE_SIDE   # 현재 주행 차선: +1=우측차선(노란선이 왼쪽) / -1=좌측차선
                                        #   노란 중앙선 위치로 매 프레임 갱신(_update_lane_side)
        # [2-2 신호등] MissionState.S0_SIGNAL 공용 — 출발/교차로 둘 다 같은 4구 신호등을 재사용하고
        # [2026-08-20]부터 같은 state로 통합됐다(perc_signal()/README §1 참고)
        self.signal_red_on      = False  # 빨강 (단일 프레임 순간값, 디바운스 안 됨)
        self.signal_straight_on = False  # 직진(=초록만 점등) 순간값
        self.signal_left_on     = False  # 좌회전(=초록+빨강 동시 점등) 순간값
        # FSM은 반드시 아래 confirmed 값만 봐야 한다(디바운스 통과분) — 순간값은 빛반사·블러로
        # 한 프레임만 튈 수 있어 그대로 쓰면 오출발/오좌회전 위험이 있다.
        self.signal_straight_confirmed = False
        self.signal_left_confirmed     = False
        self._sig_straight_cnt = 0   # signal_straight_on 연속 유지 프레임 수
        self._sig_left_cnt     = 0   # signal_left_on 연속 유지 프레임 수
        # [2026-08-23, 요청 반영] 직진/좌회전 어느 쪽이든 확정되는 즉시 신호등 YOLO 추론을
        # 끈다(perc_signal() 참고) — 원래는 직진만 Phase.DONE 동안 계속 켜져 있다가 다음
        # 바퀴(_update_lap()의 Phase.LAVACON 리셋)에야 꺼졌고, 좌회전 쪽은 S0_SIGNAL 커밋
        # 구간/게이트 램프 내내 아예 안 꺼졌었다(_active_yolo_stage() S0_SIGNAL 분기가 이
        # 플래그를 안 보고 무조건 'signal' 반환) — 이미 색이 확정된 뒤라 더 볼 필요가 없는데도
        # 불필요한 추론이 계속 돌던 비대칭을 없앴다. _active_yolo_stage() S0_SIGNAL/Phase.DONE
        # 분기 참고, RESET_PHASE_EACH_LAP에서 다음 바퀴 시작 시 해제.
        self._signal_yolo_off = False
        # [2026-08-23, 요청 반영] "확정되자마자 바로 꺼서 디버그창에서 검출된 걸 확인할
        # 새도 없다"는 보고로, 확정 순간 바로 끄지 않고 SIGNAL_YOLO_OFF_HOLD_FRAMES만큼
        # 더 돈 뒤에 끄도록 유예를 둔다. None=아직 확정 안 됨(카운트 시작 전), 확정되는
        # 순간 0부터 세기 시작 — perc_signal() 참고. mission_state가 S0_SIGNAL로 바뀌면서
        # _change_state()가 signal_left_confirmed 등을 곧바로 리셋해버려도(좌회전 확정
        # 시 같은 틱에 전환됨) 이 카운터 자체는 안 건드리므로 유예가 끊기지 않는다.
        self._signal_off_hold_cnt = None
        # [2026-08-22] S1→S0_SIGNAL 진입 트리거를 "보드 인식(색상 무관)" → "색상 확정
        # (signal_straight/left_confirmed)"으로 교체(요청 반영) — 예전엔 보드만 보여도
        # 곧장 S0_SIGNAL로 넘어가 그 안에서 다시 멈춰 서서 색상을 판독했다(그 사이 속도가
        # 0으로 굳음). 이제 색상이 실제로 확정될 때까지 S1 차선주행을 그대로 유지하고,
        # 확정되는 순간 정지 없이 곧장 커밋 구간으로 들어간다 — perc_signal()/
        # _s1_lane_follow() 참고. 이에 따라 보드 인식 전용 트리거(signal_board_confirmed)는
        # 더 이상 쓰지 않는다.
        self.stopline = False            # 굵은 가로 흰선(정지선 단서, 바퀴카운트용)
        self._signal_reentry_cooldown_t = 0.0  # 이 시각까지 신호등 보드 재감지(S0_SIGNAL 재진입) 무시
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
        # [2026-08-19] avoid_hold 디버그창(_debug_viz_avoid_hold)이 "어떤 라이다 클러스터를
        # 보고 판단했는지"를 그릴 수 있도록, perc_obstacle()이 매 틱 갱신하는 타겟 클러스터
        # 원본 좌표(전방(+x)/횡(+y), m 단위) — self._obstacle_cluster_x/y가 실제 선택된
        # 클러스터(tgt), self._obstacle_front_all_x/y가 같은 전방 ROI의 나머지 점(비교용 배경).
        self._obstacle_cluster_x = np.empty(0, dtype=np.float32)
        self._obstacle_cluster_y = np.empty(0, dtype=np.float32)
        self._obstacle_front_all_x = np.empty(0, dtype=np.float32)
        self._obstacle_front_all_y = np.empty(0, dtype=np.float32)
        self._obstacle_cluster_group_count = 0  # 이번 틱 전방 ROI 안에서 발견된 별개 클러스터 개수
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
        # [2026-08-19] avoid_hold "왜 트리거됐는지" 라이다 클러스터 스냅샷 — 위
        # self._obstacle_cluster_x/y(perc_obstacle()이 매 틱 덮어씀)를 트리거되는 그
        # 순간(_update_avoid_hold() "새 트리거" 분기)에 한 번 복사해 avoid_hold_hold_sec
        # 동안 그대로 유지한다. 매틱 최신값을 그대로 보여주면 장애물이 이미 멀어진 뒤
        # (avoid_hold_active만 유예로 남아있는 동안)엔 화면이 비어 "왜 아직 유예 중인지"를
        # 설명 못 한다 — _debug_viz_avoid_hold()가 이 스냅샷을 그린다.
        self._avoid_hold_trigger_cluster_x = np.empty(0, dtype=np.float32)
        self._avoid_hold_trigger_cluster_y = np.empty(0, dtype=np.float32)
        self._avoid_hold_trigger_front_all_x = np.empty(0, dtype=np.float32)
        self._avoid_hold_trigger_front_all_y = np.empty(0, dtype=np.float32)
        self._avoid_hold_trigger_obstacle_dist  = 999.0
        self._avoid_hold_trigger_obstacle_width = 0.0
        self._avoid_hold_trigger_obstacle_type  = 'none'
        self._avoid_hold_trigger_obstacle_side  = 'none'
        self._avoid_hold_trigger_cluster_pts    = 0
        self._avoid_hold_trigger_group_count    = 0
        self._avoid_hold_trigger_cause          = ''  # 'lidar' / 'da_jump'
        # [2-4 라바콘]
        self.lavacon_offset = 0.0    # 디버그/로깅용(중심선 y평균) — 조향엔 더 이상 안 씀
        self.lavacon_done   = False
        self.lavacon_path   = []     # _lane_steer()에 그대로 태우는 px 스케일 경로(perc_lavacon() 참고)
        self._lavacon_path_m = []    # 위와 같은 경로의 원본(라이다 미터 좌표) — DEBUG_VIZ_LAVACON 시각화용
        # [2026-08-19] process_lavacon()의 프레임간 좌/우 EMA(LAVACON_TEMPORAL_EMA_ENABLED)용
        # 상태 — process_lavacon() 자체는 무상태라 이 노드가 직전 boxes를 들고 있다가
        # 매 틱 다시 넘겨준다(perc_lavacon.py 상단 주석 6) 참고). 첫 틱엔 None.
        self._lavacon_boxes_prev = None
        self._lavacon_empty_cnt = 0   # 우측콘 연속 미검출 프레임 수(Phase 전환 디바운스)
        self.lavacon_left_detected  = False  # 좌측 라이다 클러스터 검출 여부(B1 진입 트리거용)
        self.lavacon_right_detected = False  # 우측 라이다 클러스터 검출 여부(B1 진입 트리거용)
        self.cone_detected_yolo     = False  # YOLO 카메라 콘 검출 여부(원시값, 박스 1개 이상) —
                                              #   B1/B2 진입 트리거 이중확인용(perception/yolo_cone.py).
                                              #   면적 게이트(YOLO_CONE_MIN_BOX_AREA_PX_B1/_B2)는
                                              #   perc_lavacon_trigger()/perc_obstacle_cut_trigger()가 각자 건다.
        self.cone_max_box_area      = 0.0    # [2026-08-23] 이번 프레임 콘 검출 박스 중 최대 면적(px², 위 게이트/디버그 표시용)
        self._cone_area_b1_cnt      = 0      # [2026-08-24] 면적게이트(_B1) 통과 연속 프레임 수 — YOLO_AREA_CONFIRM_FRAMES 참고
        self._cone_area_b2_cnt      = 0      # [2026-08-24] 면적게이트(_B2) 통과 연속 프레임 수
        self.lavacon_trigger        = False  # (YOLO 검출 AND 좌우 라이다 동시검출)이 디바운스 프레임수만큼 유지되면 True
        self._lavacon_trigger_cnt   = 0      # 동시검출 연속 프레임 수(디바운스 카운터)
        self._lavacon_dbg = (0, 0, 0, 0)     # 디버그용 (좌ROI점수, 좌최대연속묶음, 우ROI점수, 우최대연속묶음)
        self._lavacon_mask_dbg = (0, -1.0)   # 디버그용 (BODY_LO~HI 마스킹 구간 원본 점수, 최소거리)
        # [2026-08-21] 좌회전 진입 랜드마크 후보 — 체크무늬 게이트 라이다 기둥쌍 검출
        # (perc_checker_pillar(), config.py "체크무늬 게이트 라이다 기둥쌍 검출" 절 참고).
        # _s0_signal()의 'left' 커밋 종료 트리거로 연결됨(요청 반영) — 로직 자체는 실제
        # 라이다 캡처로 검증 못한 상태이니 실차에서 반드시 확인할 것.
        self.checker_pillar_left_detected  = False  # 좌측 기둥 클러스터 검출 여부
        self.checker_pillar_right_detected = False  # 우측 기둥 클러스터 검출 여부
        self.checker_pillar_lat_dist_m     = 0.0    # 좌우 클러스터 사이 실측 횡방향 거리(m, 둘 다 검출됐을 때만 유효)
        self.checker_pillar_trigger        = False  # (좌우 검출 AND 간격이 실측값과 일치)가 디바운스 프레임수만큼 유지되면 True
        self._checker_pillar_trigger_cnt   = 0      # 디바운스 카운터
        self._checker_pillar_dbg = (0, 0, 0, 0, 0.0)  # 디버그용 (좌pts, 좌run, 우pts, 우run, 횡방향거리)
        self._checker_pillar_bev_img = None  # 최근 라이다 BEV 프레임(_draw_checker_pillar_bev()가 채움) — left_turn_debug 통합창이 그대로 재사용
        self._checker_ramp_dist = None  # None=램프 비활성, 아니면 트리거 이후 누적 이동거리(m) — _s2_commit_dist와 동일한 VESC 거리적분 패턴
        self._left_turn_last_done_t = None  # 좌회전 램프가 가장 최근에 "완료"된 time.time() — left_turn_debug 창의 "실행끝" 표시용(None=아직 한 번도 완료 안 됨)
        # [2026-08-24] 지름길 출구 T자 강제 좌회전(config.py SHORTCUT_EXIT_DIST_M 주석 참고) —
        #   입구 램프 완료 시점부터만 채워지는 별도 거리 추적. None=추적 안 함(입구 램프를 아직
        #   안 거쳤거나 이미 출구 램프까지 끝난 상태), 숫자면 그 시점부터 누적 이동거리(m).
        #   _checker_ramp_dist와 별개 변수인 이유: 저건 "지금 램프를 돌고 있는가", 이건 "램프를
        #   시작하기 전 T자까지 얼마나 남았는가"라 동시에(램프 도는 동안엔 이 값 갱신 안 함)
        #   서로 다른 걸 추적한다.
        self._shortcut_exit_dist = None
        # 출구 램프 실행 중 플래그 — True인 동안 _s1_lane_follow()가 일반 차선주행 대신
        # _do_shortcut_exit_ramp_turn()으로 조향을 넘긴다(_s0_signal()이 self._checker_ramp_dist
        # is not None으로 입구 램프를 감지하는 것과 동일한 역할, 다만 이쪽은 S0_SIGNAL이 아니라
        # S1_LANE_FOLLOW 안에서 발동하므로 별도 bool 플래그로 뺐다).
        self._shortcut_exit_ramp_active = False
        # [2-6 방해차량 트리거]
        self.vehicle_trigger       = False   # 라이다 디바운스 통과 → B3 진입 트리거
        self._vehicle_trigger_cnt  = 0       # 동시검출 연속 프레임 수(디바운스 카운터)

        # [2026-08-20] da 근접 컷(ENABLE_OBSTACLE_CUT) — TargetPassing(B2/B3 FSM)과
        # avoid_hold_active(§2.32)와는 완전히 독립된 상태. avoid_hold_active는 이미
        # DL 워커의 _clip_da_by_ll() 재활성화용으로 쓰이고 있어(perception/dl_lane.py),
        # 재사용하면 두 메커니즘이 뒤섞인다.
        self.vehicle_detected_yolo_cut = False   # 이번 프레임 YOLO 'car' 검출 원시값(박스 1개 이상,
                                                  #   근접컷 전용 인스턴스, perception/yolo_vehicle.py).
                                                  #   면적 게이트(YOLO_VEHICLE_MIN_BOX_AREA_PX_B3)는
                                                  #   perc_obstacle_cut_trigger()(B3)가 건다.
        self.vehicle_max_box_area_cut  = 0.0     # [2026-08-23] 이번 프레임 차량 검출 박스 중 최대 면적(px², 위 게이트/디버그 표시용)
        self._vehicle_area_b3_cnt      = 0       # [2026-08-24] 면적게이트(_B3) 통과 연속 프레임 수 — YOLO_AREA_CONFIRM_FRAMES 참고
        self._obstacle_cut_trigger_cnt = 0       # (라이다 근접 AND YOLO(콘 또는 차량)) 연속확인 프레임 수(디바운스)
        self.obstacle_cut_trigger      = False   # 위 디바운스가 OBSTACLE_CUT_TRIGGER_FRAMES 넘겨 확정됐는지
        # [2026-08-20] 요청 반영 — 이 트리거를 B2(고정장애물=라바콘 1개)/B3(방해차량) 공용으로
        # 확장하면서 추가. 'fixed'(YOLO 콘 검출로 확정)/'vehicle'(YOLO 차량 검출로 확정) —
        # perc_obstacle_cut_trigger() 참고, run_behavior_fsm()의 Phase.OBSTACLE_ZONE 분기가
        # 진입 순간 이 값을 스냅샷해 B2/B3 중 어느 쪽을 통과 중인지 결정한다.
        self.obstacle_cut_type         = 'none'
        self._obstacle_cut_y           = None    # 매틱 갱신되는 라이다 최근접점 횡위치(m, +좌측) — 아래 _obstacle_cut_y_locked의 원본 소스
        # [2026-08-24, 요청 반영] 활성 유지 중 실제로 set_obstacle()에 넘기는 값 — 진입
        # 확정 순간 self._obstacle_cut_y를 스냅샷해 고정한다(_update_obstacle_cut_hold()
        # 참고). 방해차량(B3)처럼 차체가 라이다에 넓게 걸치면 매틱 최근접점이 좌우로
        # 넘나들어 self._obstacle_cut_y 부호가 바뀔 수 있는데, 그걸 그대로 매틱
        # _clip_da_by_obstacle()에 넘기면 컷 방향이 한 회피 안에서 좌→우로 뒤집혀
        # "cut이 좌우로 2번 뜨는" 증상으로 보였다(실차 확인) — 그래서 기본은 고정.
        # [2026-08-24b, 요청 반영] 완전 고정 대신, B3(vehicle)에 한해 이번 틱 라이다·YOLO
        # 좌우가 실제로 일치(교차검증 통과)하면 갱신을 허용한다 — 라이다 단독 노이즈로는
        # 안 바뀌고, 카메라도 같은 방향을 확인해줄 때만 넘어간다(_update_obstacle_cut_hold()
        # 의 새 진입 아닌 분기 참고). B2(cone)는 좌우 교차검증 자체가 없어 계속 완전 고정.
        self._obstacle_cut_y_locked    = None
        # [2026-08-23r] 이번 틱 트리거 ROI에 실제로 쓰인 횡방향 반폭(m) — B2/B3 공용값
        # (OBSTACLE_CUT_TRIGGER_Y_HALF_M) 또는 B3 전용값(_VEHICLE) 중 perc_obstacle_cut_trigger()가
        # 그때그때 고른 값. _debug_viz_obstacle_cut()이 박스를 그릴 때 그대로 읽는다.
        self._obstacle_cut_y_half      = OBSTACLE_CUT_TRIGGER_Y_HALF_M
        # [2026-08-23r2] 위 y_half와 동일한 이유로 전방 트리거 거리(m)도 이번 틱 실사용값을
        # 별도 저장 — B3가 OBSTACLE_CUT_TRIGGER_X_MAX_M_VEHICLE(2.5m)로 확장된 뒤 디버그 뷰가
        # 여전히 B2 값(1.0m)만 표시/그리던 불일치를 없앤다.
        self._obstacle_cut_x_max       = OBSTACLE_CUT_TRIGGER_X_MAX_M
        self._obstacle_cut_until_t     = 0.0     # 이 시각까지는 obstacle_cut_active=True 유지
        self.obstacle_cut_active       = False   # perc_lane()이 set_obstacle()로 DL 백엔드에 전달하는 최종 상태
        self._obstacle_cut_hold_start_t = 0.0    # 최소유지시간(OBSTACLE_CUT_HOLD_SEC_MIN) 판정 기준 시각
        # [2026-08-21] 이번 진입에 적용 중인 최소유지시간 — B2(fixed)면 OBSTACLE_CUT_HOLD_SEC_MIN_FIXED,
        # 그 외(B3/vehicle 등)는 OBSTACLE_CUT_HOLD_SEC_MIN. _update_obstacle_cut_hold() 진입 순간 결정.
        self._obstacle_cut_hold_sec_min = OBSTACLE_CUT_HOLD_SEC_MIN
        self._obstacle_cut_release_cnt  = 0      # 해제 판단용 라이다 클리어 연속 프레임 수(디바운스 카운터)
        self.obstacle_cut_release_reason = ''    # 디버그용 — 'floor_and_lidar_clear' 등
        self._obstacle_cut_lidar_near = False    # 디버그용 — 이번 프레임 트리거 ROI에 라이다 점이 잡혔는지(원시값)
        # [2026-08-21] 방해차량 좌우 교차검증 디버그용(perc_obstacle_cut_trigger() 참고) —
        # 매틱 갱신, _debug_viz_obstacle_cut()에서 그대로 표시.
        self._obstacle_cut_lidar_side = None   # 'L'/'R'/None(lidar_near 아니었으면)
        self._obstacle_cut_yolo_side  = None   # 'L'/'R'/None(YOLO 미검출)
        self._obstacle_cut_side_veto  = False  # True면 이번 틱 좌우 불일치로 vehicle_seen 취소됨
        # [2026-08-21] 진입을 확정지은 그 순간의 장애물 좌/우('L'/'R') — _update_obstacle_cut_hold()의
        # 새 진입 분기에서 self._obstacle_cut_y 부호로 한 번 캡처해두고, 활성 유지 중엔 안 바뀐다.
        # _obstacle_cut_roi_clear()가 해제 판정 시 이 쪽만 본다(요청 반영 — 아래 해당 함수 주석 참고).
        self._obstacle_cut_trigger_side = None  # 'L'/'R'/None(아직 한 번도 진입 안 함)
        # [2026-08-21] obstacle_cut_active 진입 순간 PP_LOOKAHEAD_CURVATURE_GAIN을
        # PP_CURVATURE_BOOST_GAIN으로 잠깐 올려두는 타이머 — 이 시각까지는 부스트 유지
        # (_update_obstacle_cut_hold()가 진입 엣지에서 세팅, _lane_steer()가 매틱 소비).
        self._pp_curvature_boost_until_t = 0.0
        # avoid_hold 새 트리거(=da 마진 확장) 시점에 obstacle_type=='vehicle'면 세팅 —
        # 이 시각까지 lookahead 고정(_update_avoid_hold()가 세팅, _lane_steer()가 소비).
        self._pp_vehicle_lookahead_fix_until_t = 0.0
        # [2026-08-20] _debug_viz_obstacle_cut() BEV 패널용 — avoid_hold_debug의
        # _obstacle_front_all_x/y·_obstacle_cluster_x/y와 동일 패턴(매틱 갱신되는 라이브 값).
        # bg_*는 표시범위 안의 배경점 전부(회색), roi_*는 실제 트리거 ROI 안에 잡힌 점만(빨강).
        self._obstacle_cut_bg_x  = np.array([])
        self._obstacle_cut_bg_y  = np.array([])
        self._obstacle_cut_roi_x = np.array([])
        self._obstacle_cut_roi_y = np.array([])

        # [2-7 장애물 위치 판단]
        self.lane_center   = 320.0           # 차선 중앙 x좌표(px) — 첫 카메라 프레임 전까지 화면 중앙 기본값

        # ── 외부 차선 인식 모듈 초기화 (LANE_DETECTOR_BACKEND로 선택, 인터페이스는 셋 다 동일) ──
        self.lane_detector = self._build_lane_detector(LANE_DETECTOR_BACKEND)

        # 라바콘 카메라 이중확인용 YOLO 콘 검출기. onnxruntime 미설치/모델 파일 부재 등으로
        # 초기화가 실패하면 _build_lane_detector()의 dl→hough 폴백과 달리 대체 백엔드가
        # 없으므로(카메라 이중확인 자체가 선택사항), None으로 두고 perc_lavacon_trigger()가
        # "카메라 확인 불가 시 라이다 단독 판정으로 폴백"하도록 한다 — 원인은 에러 로그로 남긴다.
        # [2026-08-20] ENABLE_OBSTACLE_CUT과 동일 패턴으로 YOLO_CONE_ENABLE 게이트 추가(요청
        # 반영) — ENABLE_BEHAVIOR=False로 라바콘 자체를 안 쓰는 지금, 이 검출기가 매 프레임
        # 백그라운드에서 계속 도는 게 순전한 오버헤드라 꺼둔다(config.py YOLO_CONE_ENABLE
        # 주석 참고). perc_yolo_cone()은 self.yolo_cone_detector=None일 때 조용히 스킵한다.
        self.yolo_cone_detector = None
        if YOLO_CONE_ENABLE:
            try:
                self.yolo_cone_detector = YoloConeDetector(logger=self.get_logger())
            except Exception as e:
                self.get_logger().error(
                    f'YOLO 콘 검출기 초기화 실패, 라바콘 트리거는 라이다 단독 판정으로 폴백합니다: {e}'
                )
                self.yolo_cone_detector = None

        # [2026-08-20] da 근접 컷(ENABLE_OBSTACLE_CUT) 전용 YOLO 차량 검출기 — 이 저장소의
        # 다른 YOLO 검출기와 동일 패턴(초기화 실패 시 None, 라이다 단독 판정으로 폴백).
        # ENABLE_OBSTACLE_CUT=False(기본값)면 초기화 자체를 건너뛴다 — 이미 dl_lane/
        # yolo_cone/yolo_signal_state 세 개가 돌고 있는 상황에서 안 쓰는 4번째 추론 스레드를
        # 굳이 띄워 자원경합을 더할 이유가 없다(꺼져있을 땐 정말로 아무 영향 없게).
        self.yolo_vehicle_cut_detector = None
        if ENABLE_OBSTACLE_CUT:
            try:
                self.yolo_vehicle_cut_detector = YoloVehicleDetector(logger=self.get_logger())
            except Exception as e:
                self.get_logger().error(
                    f'YOLO 차량(근접컷) 검출기 초기화 실패, 근접 컷 트리거는 라이다 단독 판정으로 폴백합니다: {e}'
                )

        # [2026-08-21] 신호등 인식은 이 YOLO 단독 모델(위치+색상상태를 한 스테이지로 동시
        # 예측) 하나뿐이다 — 예전에 있던 HSV/Hough Circle 기반 검출(traffic_signal.py/
        # frst.py) 및 배경판 위치 전용 YOLO(yolo_signal.py)는 삭제했다(README §1.18). 초기화
        # 실패 시(모델 파일 없음 등) 폴백 없이 None으로 두면 perc_signal()이 신호등을 계속
        # "미검출"로만 보고한다 — 다른 YOLO 검출기와 달리 이 모델이 없으면 대체 경로가 없다.
        try:
            self.yolo_signal_state_detector = YoloSignalStateDetector(logger=self.get_logger())
        except Exception as e:
            self.get_logger().error(f'YOLO 신호등 검출기 초기화 실패, 신호등을 인식할 수 없습니다: {e}')
            self.yolo_signal_state_detector = None

        # ── 판단/제어 상태 ──
        self.mission_state  = START_STATE
        self.behavior_state = BehaviorState.B0_NORMAL
        # [2026-08-24, 요청 반영] B2 대기 단독 테스트용 오버라이드(Phase.OBSTACLE_ZONE 시작)
        # 원복 — 전체 바퀴/신호 흐름(S0_SIGNAL→B1→B2→B3) 검증으로 다시 전환.
        self.phase          = Phase.LAVACON
        # [2026-08-15] Phase.OBSTACLE_ZONE 통합(da_based_b2b3_proposal.md B안) —
        # B2/B3 각각 최소 한 번 완료됐는지 추적. 둘 다 True가 돼야 Phase.DONE으로
        # 넘어간다(_mark_behavior_passed() 참고).
        # [2026-08-20] 대회 트랙은 고정장애물(B2)이 항상 이동장애물(B3)보다 먼저 나오는 게
        # 확정된 순서다(§5.2) — §5.4에서 잠깐 반대로 뒤집었다가(요청 반영), [2026-08-22]
        # 요청으로 다시 "B2 먼저"로 되돌렸다(README §5.5). run_behavior_fsm()에서 B3 트리거를
        # self._b2_passed가 True일 때만 받아들이도록 순서를 강제한다(아래 참고).
        # [2026-08-23] 위 Phase.LAVACON 원복과 짝 — B2/B3 둘 다 아직 안 지났으므로 False로
        # 시작(원래 정상 레이스 시작값).
        self._b2_passed = False
        self._b3_passed = False
        # [2026-08-20] 요청 반영 — B2/B3 실제 처리를 da 근접 컷(obstacle_cut_active) 기반으로
        # 바꾸면서 추가. obstacle_cut_active 진입 순간 'B2'/'B3' 중 하나를 latch해뒀다가,
        # obstacle_cut_active가 다시 꺼지는 순간(탈출) 그 태그로 _mark_behavior_passed()를
        # 부른다(run_behavior_fsm()의 Phase.OBSTACLE_ZONE 분기 참고) — B1의 _lavacon_engaged와
        # 동일한 진입~탈출 latch 패턴.
        self._obscut_zone_tag = None
        self._behavior_enabled = TEST_FORCE_BEHAVIOR  # 원래 S0_SIGNAL "직진" 확정 시에만 True
                                                       #   (TEST_FORCE_BEHAVIOR=False — S0_SIGNAL부터
                                                       #    정상 시작이라 강제 ON 불필요, 직진 신호
                                                       #    확정 분기가 그때 가서 직접 켠다)
        # [2026-08-20] S0_SIGNAL 통합(S0_WAIT_GREEN+S2_INTERSECTION → 하나)으로 같은 state를
        # 출발 때와 매 바퀴 교차로에서 반복 재진입하게 됐다 — "이번이 진짜 첫 출발인지"는
        # prev_state 비교로 더 이상 구분이 안 되므로(둘 다 같은 state) 이 플래그로 직접
        # 추적한다. _change_state()가 S1_LANE_FOLLOW 진입 시 이 값을 보고 바퀴 타이머/yaw
        # 누적 기준점(_lap_t0 등)을 최초 1회만 리셋한다.
        self._departed = False
        self._lavacon_engaged  = False          # B1_LAVACON 진입 확정 latch (트리거 이후 잠깐 한쪽 클러스터가
                                                 #   끊겨도 중간에 일반주행으로 안 튀도록 유지, lavacon_done으로 해제)
        # [2026-08-22] _lavacon_steer_da_push()가 이번 틱에 실제로 경로를 옆으로 밀었는지
        # (push_m != 0.0) — perc_lane()이 set_lavacon_push()로 DL 백엔드에 넘겨 DA
        # 디버그창의 경로 색(자홍/주황)에 반영한다(lane_util.py draw_path() 참고).
        # run_behavior_fsm()이 _lavacon_engaged가 꺼지는 즉시 False로 되돌린다.
        self._lavacon_push_active = False
        # [2026-08-22] 위 플래그와 함께 넘기는 실제 push량(px, +면 우측으로 밀림) —
        # DA 디버그창에서 밀리기 전(보라) 원본 경로와 밀린 뒤(주황) 경로를 나란히 그려
        # 게인 튜닝 시 "어느 정도 밀었는지"를 한눈에 보려는 용도(lane_util.py draw_path()).
        self._lavacon_push_px = 0.0
        # [2026-08-23, 요청 반영] LAVACON_KICK_ENABLED 실험용 — B1 진입 확정 순간부터 남은
        # "강제 조향각 유지" 프레임 수. run_behavior_fsm()이 진입 상승엣지에서 채우고,
        # _handle_lavacon()이 매 틱 1씩 깎으며 0보다 큰 동안 push 계산 대신
        # LAVACON_KICK_ANGLE_DEG를 그대로 ctrl_angle에 꽂는다(config.py 주석 참고).
        self._lavacon_kick_cnt = 0
        self.ctrl_angle = 0.0
        self.ctrl_speed = SPEED_STOP
        self._prev_angle_out = 0.0    # [5] 직전 발행 조향각(변화율 제한용)
        self._pid_prev_error = 0.0
        self._pid_integral   = 0.0
        self._s2_commit_dist = None   # 좌회전 커밋 구간(체크무늬 게이트까지) 누적 이동거리(m, None=미진입)
        self._s2_commit_dir  = None   # [2026-08-22h] 직진은 커밋 구간 자체가 없어져 이제 'left'만 들어온다
        self._s2_commit_start_t = None  # [2026-08-24] 커밋 구간 진입 time.time() — CHECKER_PILLAR_LIDAR_TIMEOUT_SEC 초 경과 판정용
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
            ld_floor_px=PP_LD_FLOOR_PX,
            dx_deadzone_px=PP_DX_DEADZONE_PX,
            lookahead_curvature_gain=PP_LOOKAHEAD_CURVATURE_GAIN,
            lookahead_min_px=PP_LOOKAHEAD_MIN_PX,
            wheelbase_boost_enable=PP_WHEELBASE_BOOST_ENABLE,
            wheelbase_boost_gain_per_deg=PP_WHEELBASE_BOOST_GAIN_PER_DEG,
            wheelbase_boost_max_scale=PP_WHEELBASE_BOOST_MAX_SCALE,
            lookahead_alpha=PP_LOOKAHEAD_ALPHA,
            lookahead_speed_anchor=PP_LOOKAHEAD_SPEED_ANCHOR,
        )

        # [2026-08-24] B1(라바콘) 전용 Pure Pursuit 인스턴스 — 위 self.pure_pursuit(일반
        # 차선주행용)와 완전히 분리된 별도 상태(prev_steer_deg/last_lookahead_px 등)를
        # 갖는다. 게인도 전부 config.py의 _LAVACON 전용 상수(PP_TUNE_PRESETS 프리셋과
        # 무관하게 고정된 스냅샷, config.py 해당 블록 주석 참고)에서만 가져오므로, 이후
        # speed15 프리셋을 재튜닝하거나 다른 프리셋으로 바꿔도 라바콘 조향은 지금 거동
        # 그대로 유지된다. _lavacon_pure_pursuit_steer()가 이 인스턴스를 사용한다.
        self.pure_pursuit_lavacon = PurePursuitController(
            lookahead_base_px=PP_LOOKAHEAD_BASE_PX_LAVACON,
            lookahead_speed_gain=PP_LOOKAHEAD_SPEED_GAIN_LAVACON,
            lookahead_max_px=PP_LOOKAHEAD_MAX_PX_LAVACON,
            wheelbase_px=PP_WHEELBASE_PX_LAVACON,
            angle_max_deg=ANGLE_MAX_LAVACON,
            alpha=PP_ALPHA_LAVACON,
            ld_floor_px=PP_LD_FLOOR_PX_LAVACON,
            dx_deadzone_px=PP_DX_DEADZONE_PX_LAVACON,
            lookahead_curvature_gain=PP_LOOKAHEAD_CURVATURE_GAIN_LAVACON,
            lookahead_min_px=PP_LOOKAHEAD_MIN_PX_LAVACON,
            wheelbase_boost_enable=PP_WHEELBASE_BOOST_ENABLE_LAVACON,
            wheelbase_boost_gain_per_deg=PP_WHEELBASE_BOOST_GAIN_PER_DEG_LAVACON,
            wheelbase_boost_max_scale=PP_WHEELBASE_BOOST_MAX_SCALE_LAVACON,
            lookahead_alpha=PP_LOOKAHEAD_ALPHA_LAVACON,
            lookahead_speed_anchor=PP_LOOKAHEAD_SPEED_ANCHOR_LAVACON,
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
        #     (_begin_checker_ramp_turn() 진입 시 즉시 대입, B0→B1 전환, Stanley 출력 교체 등에서 실제로 발생).
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
    # [2026-08-20] 요청 반영 — YOLO 카메라 검출기 3종(콘/차량/신호등색상)을 항상 다 같이
    #   돌리지 않고, 지금 mission_state/phase가 실제로 필요로 하는 것 하나만 이번 틱에
    #   추론시킨다. YoloConeDetector 등은 detect(frame)에 새 프레임을 안 넘기면 백그라운드
    #   추론 스레드가 할 일이 없어 그냥 논다(perception/yolo_cone.py _worker() 참고) —
    #   그래서 perceive_all()에서 "호출 자체를 건너뛴다"가 곧 "그 모델의 추론을 끈다"와
    #   같다(스레드/세션은 살아있지만 GPU/CPU 추론 자체는 안 돎).
    #
    #   매핑: S0_SIGNAL(신호등 판단 대기) → 신호등. S1_LANE_FOLLOW 중 Phase.LAVACON(B1
    #   진입 대기) → 콘. Phase.OBSTACLE_ZONE 중 B2(고정장애물) 아직 안 지났으면 → 콘,
    #   지났으면(B3 방해차량 대기) → 차량. Phase.DONE(다음 교차로 신호등 대기) → 신호등.
    #   S4_FINISH는 카메라 YOLO 자체가 불필요해 전부 끈다.
    #   [2026-08-22, 요청 반영] §5.4에서 "B3(방해차량) → B2(고정장애물)"로 뒤집었던 트랙
    #   순서를 다시 원래 순서 "B1(라바콘) → B2(고정장애물) → B3(방해차량)"로 되돌렸다
    #   (README §5.5) — 아래 Phase.OBSTACLE_ZONE 분기, run_behavior_fsm() 참고.
    #
    #   [2026-08-20] §1.16은 신호등 YOLO를 "S3/S4 포함 항상 켜서 오탐률을 전체 구간에서
    #   로그로 본다"는 의도로 상시 가동시켰는데, 이번 요청으로 그 부분을 뒤집는다 —
    #   상시 오탐 로깅보다 동시 추론 개수를 줄이는 쪽(연산 자원 절약)을 우선했다. 오탐률을
    #   다시 전체 구간에서 보고 싶으면 이 함수가 'signal' 아닌 다른 stage일 때도
    #   perc_yolo_signal_state()를 호출하도록 되돌리면 된다.
    def _active_yolo_stage(self):
        if TEST_FORCE_SIGNAL_YOLO:
            # [2026-08-23, 요청 반영] "욜로 안 끊기게, 검출만 테스트" 전용 — mission_state/
            # phase/_signal_yolo_off 등 FSM 상태와 완전히 무관하게 신호등 YOLO를 항상 켠다.
            # TEST_SIGNAL_LOOP의 phase 조기 이탈이나 확정 후 hold-off 로직(perc_signal())에
            # 전혀 영향받지 않으므로, 순수하게 YOLO_신호등 창의 검출 정확도만 보고 싶을 때 켤 것.
            return 'signal'
        if self.mission_state == MissionState.S0_SIGNAL:
            # [2026-08-23] 좌회전 확정 뒤 커밋구간/게이트 램프(_s2_commit_dist/
            # _checker_ramp_dist)를 도는 동안은 색상이 이미 확정된 뒤라 더 볼 필요가 없다
            # (_signal_yolo_off, perc_signal() 참고). 아직 신호를 못 읽어 대기 중인
            # 출발선(START_STATE로 S0_SIGNAL 시작, 확정 전이라 _signal_yolo_off=False)
            # 에서는 그대로 계속 켜둔다.
            return None if self._signal_yolo_off else 'signal'
        if self.mission_state == MissionState.S1_LANE_FOLLOW:
            if self.phase == Phase.LAVACON:
                # [2026-08-23] 진입 확정(_lavacon_engaged) 전까지는 진입 트리거 판정
                # (perc_lavacon_trigger()의 cone_confirmed_cam)에 cone YOLO가 필요하지만,
                # 일단 진입이 확정된 뒤(라바콘 사이를 실제로 통과 주행하는 동안)엔 콘이
                # 카메라 시야를 가려 프레임이 제대로 안 나오는 구간이라 추론을 계속 돌릴
                # 실익이 없다 — 여기서 끈다. 탈출(lavacon_done 확정 → Phase.OBSTACLE_ZONE
                # 전환) 시점부터는 아래 OBSTACLE_ZONE 분기가 고정장애물(B2) 판정을 위해
                # 자동으로 다시 'cone'을 켠다.
                return 'cone' if not self._lavacon_engaged else None
            if self.phase == Phase.OBSTACLE_ZONE:
                return 'cone' if not self._b2_passed else 'vehicle'
            # Phase.DONE — 다음 교차로 신호등 보드 대기. 단, 이번 바퀴 신호등을 이미
            # 직진/좌회전 어느 쪽으로든 확정했다면(_signal_yolo_off) 다음 바퀴 리셋
            # 전까지 추론 자체를 끈다.
            return None if self._signal_yolo_off else 'signal'
        return None  # S4_FINISH — 카메라 YOLO 불필요

    def perceive_all(self):
        self.perc_lane()        # 비전 — set_obstacle()가 여기서 "직전 틱"의 obstacle_cut_active를
                                 #   DL 백엔드로 넘긴다(set_avoid_hold()와 동일한 1틱 지연 허용, 아래 참고)
        yolo_stage = self._active_yolo_stage()

        # [2026-08-21] perc_signal()이 이 결과(self.signal_red/straight/left_on)를 그대로
        #   디바운스(확정) 처리하므로, 같은 틱의 최신값을 넘겨주려면 perc_signal()보다 먼저
        #   돌아야 한다(1틱 지연을 추가로 만들지 않기 위함).
        if yolo_stage == 'signal':
            self.perc_yolo_signal_state() # 비전 (YOLO, 신호등 위치+색상상태 동시 예측)
        else:
            self.signal_red_on = self.signal_straight_on = self.signal_left_on = False
        self.perc_signal()      # 비전
        self.perc_obstacle()    # 라이다
        self._update_avoid_hold()  # 라이다(위 obstacle_front/dist 기반) — perc_obstacle() 직후여야 함
        # [2026-08-20] da 근접 컷(ENABLE_OBSTACLE_CUT) — 라이다+YOLO AND 트리거 → 유지/해제
        #   타이머 순서. run_behavior_fsm()의 B2/B3 진입~탈출 신호로 쓰인다(§4/§5 README).
        #   [2026-08-20] perc_obstacle_cut_trigger()가 B2(고정장애물=라바콘) 판정에
        #   self.cone_detected_yolo를 쓰게 되면서, 그 값을 채우는 perc_yolo_cone()을
        #   perc_lavacon_trigger()용으로 두던 원래 위치보다 앞으로 당겼다 — 안 당기면
        #   1틱 지연된(직전 프레임) 콘 검출값을 obstacle_cut 판정에 쓰게 된다.
        if yolo_stage == 'vehicle':
            self.perc_yolo_vehicle_cut()      # 비전 (YOLO 차량, perc_obstacle_cut_trigger()가 라이다와 AND 결합)
        else:
            self.vehicle_detected_yolo_cut = False
        if yolo_stage == 'cone':
            self.perc_yolo_cone()             # 비전 (YOLO 콘, perc_obstacle_cut_trigger()/perc_lavacon_trigger() 공용)
        else:
            self.cone_detected_yolo = False
        self.perc_obstacle_cut_trigger()  # 라이다+비전 → 근접 컷 진입 트리거(디바운스), B2/B3 타입 판정
        self._update_obstacle_cut_hold()  # 최소유지시간 + 라이다 클리어 디바운스로 유지/해제 판단
        self.perc_lavacon()     # 라이다
        self.perc_lavacon_trigger()  # 라이다+비전 (YOLO 콘 검출 AND 좌우 클러스터 동시검출 → B1_LAVACON 진입 트리거)
        self.perc_vehicle_trigger()  # 라이다 (전방 장애물 근접 → B3_VEHICLE 진입 트리거)
        self.perc_stopline()    # 비전
        self.perc_checker_pillar()  # 라이다 (체크무늬 게이트 좌우 기둥쌍 → 좌회전 램프 진입 트리거, S0_SIGNAL 'left' 커밋 중에만 실사용)

    # [2-4a] 라바콘 카메라 이중확인 (YOLO)
    #   입력 self.img_front → 출력 self.cone_detected_yolo
    #   yolo_cone.py가 별도 스레드에서 자기 페이스로 추론하므로 여기선 논블로킹으로
    #   최신 결과만 받아온다(dl_lane.py의 perc_lane()과 동일한 패턴).
    def perc_yolo_cone(self):
        if self.yolo_cone_detector is None:
            # 초기화 실패 상태 — perc_lavacon_trigger()가 이 경우 라이다 단독 판정으로
            # 폴백하므로 여기선 그냥 False로 둔다(카메라 확인 "안 됨"이 아니라 "못 함").
            self.cone_detected_yolo = False
            self.cone_max_box_area = 0.0
            return
        if self.img_front is None:
            return
        self.cone_detected_yolo = self.yolo_cone_detector.detect(self.img_front)
        self.cone_max_box_area = self.yolo_cone_detector.get_latest_max_area()
        self.yolo_cone_detector.show_debug_windows()  # 메인 스레드에서만 호출(yolo_cone.py 주석 참고)

    # [2026-08-20] da 근접 컷 전용 YOLO 차량 이중확인 (ENABLE_OBSTACLE_CUT)
    #   입력 self.img_front → 출력 self.vehicle_detected_yolo_cut
    #   perc_yolo_cone()과 동일 패턴 — yolo_vehicle.py가 별도 스레드에서 자기 페이스로
    #   추론하므로 여기선 논블로킹으로 최신 결과만 받아온다. perc_yolo_vehicle()(다른
    #   브랜치, near_obstacle 트리거용)과는 별개 인스턴스/용도다.
    def perc_yolo_vehicle_cut(self):
        if not ENABLE_OBSTACLE_CUT or self.yolo_vehicle_cut_detector is None:
            self.vehicle_detected_yolo_cut = False
            self.vehicle_max_box_area_cut = 0.0
            return
        if self.img_front is None:
            return
        self.vehicle_detected_yolo_cut = self.yolo_vehicle_cut_detector.detect(self.img_front)
        self.vehicle_max_box_area_cut = self.yolo_vehicle_cut_detector.get_latest_max_area()
        # [2026-08-20] 여기서 별도 창으로 안 띄운다 — _debug_viz_obstacle_cut()이
        # get_latest_debug_frame()으로 이 프레임을 가져다 라이다 ROI 패널과 한 창에 합쳐 그린다.

    # [2-4b] 신호등 위치+색상상태 YOLO
    #   입력 self.img_front → 출력 self.signal_red/straight/left_on
    #   yolo_signal_state.py가 별도 스레드에서 자기 페이스로 추론하므로 여기선 논블로킹으로
    #   최신 결과만 받아온다(perc_yolo_cone()과 동일한 패턴). perceive_all()이 _active_yolo_stage()
    #   로 'signal' 단계일 때만 이 함수를 호출한다 — perc_signal()보다 먼저 돌도록 순서를 맞춰뒀다.
    def perc_yolo_signal_state(self):
        if self.yolo_signal_state_detector is None:
            self.signal_red_on = self.signal_straight_on = self.signal_left_on = False
            return
        if self.img_front is None:
            return
        self.signal_red_on, self.signal_straight_on, self.signal_left_on = \
            self.yolo_signal_state_detector.detect(self.img_front)
        self.yolo_signal_state_detector.show_debug_windows()  # 메인 스레드에서만 호출(yolo_signal_state.py 주석 참고)

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
        # [2026-08-20] da 근접 컷(ENABLE_OBSTACLE_CUT) 상태도 같이 넘긴다 — set_avoid_hold()와
        # 동일한 1틱 지연 허용(위 주석 참고). obstacle_cut_active=False면(기본, 또는 아직
        # 트리거 전) _clip_da_by_obstacle()이 그대로 통과시키므로 영향 없다.
        # [2026-08-23] obstacle_cut_type('fixed'/'vehicle')도 같이 넘긴다 — B2/B3 컷
        # 좌우폭을 다르게 쓰기 위함(config.py OBSTACLE_CUT_LANE_HALF_WIDTH_PX_* 참고).
        # [2026-08-24, 요청 반영] 활성 유지 중엔 self._obstacle_cut_y_locked(진입 확정
        # 순간 고정값)을 넘긴다 — 매틱 갱신되는 self._obstacle_cut_y를 그대로 넘기면
        # 컷 방향이 회피 도중 좌우로 뒤집힐 수 있었다(위 self._obstacle_cut_y_locked
        # 주석 참고).
        getattr(self.lane_detector, 'set_obstacle', lambda *_a, **_k: None)(
            self._obstacle_cut_y_locked if self.obstacle_cut_active else self._obstacle_cut_y,
            self.obstacle_cut_active, self.obstacle_cut_type)
        # [2026-08-22] B1 콘 침범 push(_lavacon_steer_da_push()) 상태도 같이 넘긴다 —
        # set_obstacle()과 동일한 1틱 지연 허용(위 주석 참고). DA 디버그창 경로 색
        # (자홍/보라/주황, lane_util.py draw_path())과, 밀리기 전/후 두 경로를 나란히
        # 그리기 위한 실제 push량(px)에 쓰인다.
        getattr(self.lane_detector, 'set_lavacon_push', lambda *_a, **_k: None)(
            self._lavacon_push_active, self._lavacon_push_px)

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
        # [2026-08-19] wheelbase 부스트 전/후 조향각도 같이 넘겨서 result 패널 하단에
        # 표시한다(perception/dl_lane.py show_debug_windows() steer_deg_raw/steer_deg_final
        # 주석, 요청 반영) — lookahead_xy와 동일하게 이번 틱 _lane_steer() 실행 전 시점이라
        # 직전 틱 값(0.05s 이내 오차, 무시 가능).
        # [2026-08-20] 디버그창 간소화(요청 반영) — v_mps(실측) 대신 self.ctrl_speed(지금
        # drive()로 실제 발행 중인 speed 명령값)를 넘긴다. show_debug_windows() docstring
        # ctrl_speed 주석 참고.
        getattr(self.lane_detector, 'show_debug_windows', lambda *a, **k: None)(
            lookahead_xy, lookahead_px, self.ctrl_speed,
            steer_deg_raw=self.pure_pursuit.last_pre_boost_steer_deg,
            steer_deg_final=self.pure_pursuit.prev_steer_deg)

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
        # [2026-08-18] lane_unstable(SPEED_LANE_STALE 트리거)의 기준을 `valid`(근접 전용)
        # 에서 `path_ok`(근접 OR 원거리, 위와 동일)로 통일 — 요청 반영: "경로만 있으면
        # 충분하다, 경로가 안 찍힐 때 속도를 낮추는 것처럼 둘의 검출조건을 일치시키자".
        # 실제 조향 경로(self.lane_path, 바로 아래)도 path_ok로 갱신되므로, 이제 "경로가
        # 갱신 안 되는 시점"과 "속도가 깎이는 시점"이 정확히 같은 조건을 본다 — 근접만
        # 비고 원거리로 경로가 계속 갱신되는 동안에는 더 이상 속도가 깎이지 않는다.
        self._lane_invalid_streak = 0 if path_ok else self._lane_invalid_streak + 1
        self.lane_unstable = self._lane_invalid_streak >= LANE_UNSTABLE_FRAMES
        if valid:
            # 기존 제어 코드와 호환되도록 필터링 적용
            self.lane_offset = 0.7 * self.lane_offset + 0.3 * offset
            self.lane_lookahead = 0.5 * self.lane_lookahead + 0.5 * lookahead
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
    #   출력 signal_red/straight/left_on (S0_SIGNAL 공용 — 출발/교차로 겸용)
    #   주의 4구는 직진·좌회전 모두 초록 → 점등 '위치'로 구분
    def perc_signal(self):
        """신호등 판별 — perc_yolo_signal_state()가 같은 틱에 먼저 갱신해둔
          self.signal_red/straight/left_on(YOLO 단독, 위치+색상 동시 예측)을 디바운스만
          적용해 확정값으로 승격시킨다. 대회 규정 변경으로 출발과 교차로가 동일한 4구
          신호등을 재사용하고, [2026-08-20]부터는 그 둘을 아예 하나의
          state(MissionState.S0_SIGNAL)로 합쳤다 — 이 함수는 매번 같은 의미로 판독한다:
          signal_straight_confirmed(초록만 점등) = 직진(출발 시점엔 "출발"과 동의어),
          signal_left_confirmed(초록+빨강 동시) = 좌회전(지름길, 출발 지점에선 사실상 안 뜸).
        YOLO 검출이 그 프레임에 실패하면(신뢰도 미달/각도 등) signal_*_on이 False가 될 수
        있다. 여기서 SIG_CONFIRM_FRAMES 연속 유지를 확인해 confirmed로 승격시켜, 단발성
        오검출/미검출이 바로 FSM 전환(출발/좌회전)으로 새는 걸 막는다(라바콘/차량 트리거와
        동일한 패턴).

        [2026-08-20] S1_LANE_FOLLOW 중에도 신호등 검출을 돌리도록 확장(요청 반영) — 예전엔
        S0_SIGNAL 진입 후에야 신호등을 "보기" 시작해서, 그 진입 자체는 정지선(바닥 흰선)
        검출이 트리거였다. 그런데 "정지선을 밟아야 신호를 읽기 시작한다"는 신호등 자체를
        인식하는 것과는 다른 신호원이라, 정지선 인식이 어긋나면(치우침/블러 등) 신호등이
        이미 빨간불인데도 계속 차선주행 속도로 접근하는 상황이 가능했다. 정지선은 더 이상
        이 전환에 관여하지 않는다(다른 용도— _update_lap()/교차로 근처 기동 금지—는
        그대로 유지).

        [2026-08-22] "보드 인식(색상 무관) → S0_SIGNAL 진입 → 그 안에서 색상 확정"이던
        2단계 트리거를 없앴다(요청 반영, S0_SIGNAL 진입 시 속도가 0으로 굳는 문제) —
        S1_LANE_FOLLOW/S0_SIGNAL 구분 없이 매 프레임 직진/좌회전 색상 확정
        (signal_straight/left_confirmed)을 갱신하고, _s1_lane_follow()가 이 확정값을
        직접 보고 커밋 구간 진입을 트리거한다. 색상 확정 전까지는 S1 차선주행이 멈추지
        않는다."""
        if self.img_front is None:
            return
        if self.mission_state not in (MissionState.S1_LANE_FOLLOW, MissionState.S0_SIGNAL):
            return

        self._sig_straight_cnt = self._sig_straight_cnt + 1 if self.signal_straight_on else 0
        self._sig_left_cnt     = self._sig_left_cnt + 1 if self.signal_left_on else 0
        self.signal_straight_confirmed = self._sig_straight_cnt >= SIG_CONFIRM_FRAMES
        self.signal_left_confirmed     = self._sig_left_cnt >= SIG_CONFIRM_FRAMES
        # ★★★ [2026-08-23j, 요청 반영, 중대 임시 스위치 — 반드시 원복!] ★★★
        # TEST_FORCE_LEFT_TURN_SIGNAL=True면 실제 YOLO 판독 결과를 완전히 무시하고
        # "좌회전 신호를 이미 받은 것"으로 강제한다 — 좌회전 로직(커밋 구간 →
        # perc_checker_pillar() 좌우 라이다 기둥쌍 검출 → 조향 램프)만 신호등 인식과
        # 분리해서 단독 검증하려는 목적(config.py TEST_FORCE_LEFT_TURN_SIGNAL 주석 참고).
        # ⚠️ 검증 끝나면 config.py에서 False로 되돌릴 것 — 실차 레이스 중 켜진 채로
        # 있으면 신호와 무관하게 항상 좌회전으로 우겨서 코스를 이탈한다.
        if TEST_FORCE_LEFT_TURN_SIGNAL:
            self.signal_left_confirmed = True
        # [2026-08-23] 직진/좌회전 둘 중 하나라도 확정되면 신호등 YOLO 추론을 끈다 —
        # 색상이 이미 확정된 이상 더 볼 필요가 없다(다음 바퀴 리셋까지, _update_lap()
        # RESET_PHASE_EACH_LAP 분기 참고). 예전엔 직진 확정 분기(_s1_lane_follow())에서만
        # 개별적으로 껐는데, 좌회전 확정 뒤 S0_SIGNAL 커밋구간에서는 _active_yolo_stage()가
        # S0_SIGNAL이면 이 플래그와 무관하게 무조건 'signal'을 켜고 있어 좌회전 쪽만
        # 추론이 안 꺼지는 비대칭이 있었다 — 여기 한 곳에서 양쪽 다 같은 순간(확정되는 틱)에
        # 끄도록 합쳤다(_active_yolo_stage() S0_SIGNAL 분기도 같이 수정).
        # [2026-08-23b, 요청 반영] "YOLO_신호등 창에 좌회전이 분명히 찍혔는데도 확정
        # 표시가 바로 사라져서 확인이 안 된다"는 보고 — 좌회전 확정은 같은 틱에 곧장
        # S0_SIGNAL로 전환되고 _change_state()가 signal_left_confirmed/signal_left_on/
        # _sig_left_cnt를 그 자리에서 즉시 리셋해버리는데(S0_SIGNAL 진입 시 새로 판독
        # 시작하려는 의도, 그 자체는 정상), 곧이어 이 조건이 True→False로 바로 꺼져
        # YOLO도 다음 틱부터 바로 멈춰버려 육안/디버그창으로 확인할 틈이 없었다. 그래서
        # 확정되는 즉시 끄지 않고, 확정된 틱부터 SIGNAL_YOLO_OFF_HOLD_FRAMES만큼 더 돈
        # 뒤에 끄도록 유예를 둔다 — _signal_off_hold_cnt는 _change_state()가 안 건드리는
        # 별도 필드라 위 리셋과 무관하게 계속 유지된다. FSM의 실제 상태전환(좌회전이면
        # S0_SIGNAL 진입, 직진이면 Behavior 재활성화)은 이 유예와 무관하게 확정되는 그
        # 틱에 이미 끝나 있으므로, 이 유예는 순수하게 "YOLO를 몇 프레임 더 돌려서 눈으로
        # 확인 가능하게" 하는 것뿐 — FSM 반응 속도에는 영향 없다.
        if (self.signal_straight_confirmed or self.signal_left_confirmed) \
                and self._signal_off_hold_cnt is None:
            self._signal_off_hold_cnt = 0
        if self._signal_off_hold_cnt is not None:
            self._signal_off_hold_cnt += 1
            if self._signal_off_hold_cnt >= SIGNAL_YOLO_OFF_HOLD_FRAMES:
                self._signal_yolo_off = True

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
        # [2026-08-22k] 1.5 → 0.5로 축소했다가(라바콘 트리거 ROI 축소와 같이 묶어서 요청)
        #   다시 1.5로 원복(요청 반영) — 이 값은 lidar_bev 표시 전용이 아니라 B2/B3 회피
        #   판정(front_mask → obstacle_front/dist/type/width)에 직접 쓰이는 ROI라, 라바콘과는
        #   별개로 다뤄야 한다는 판단. 라바콘 쪽 축소(perc_lavacon_trigger() LAT_MAX,
        #   perc_lavacon.py CONE_LAT_LIMIT, 둘 다 0.5)는 그대로 유지.
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

            # [avoid_hold 디버그용] 선택된 타겟 클러스터(tgt)와 전방 ROI 전체 점(fidx, 배경
            # 비교용) 원본 좌표를 저장 — _update_avoid_hold()가 트리거 순간에 이 값을
            # 스냅샷해 _debug_viz_avoid_hold()가 그려준다(위 클래스 초기화부 주석 참고).
            self._obstacle_cluster_x = x[tgt]
            self._obstacle_cluster_y = y[tgt]
            self._obstacle_front_all_x = x[fidx]
            self._obstacle_front_all_y = y[fidx]
            self._obstacle_cluster_group_count = len(groups)

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
            # [2026-08-19] 위 obstacle_dist/type처럼 완전히 리셋 — 안 하면 장애물이 실제로
            # 시야에서 사라진 뒤에도 _debug_viz_avoid_hold()의 BEV 패널에 몇 틱 전 클러스터
            # 점이 계속 남아있어 "창이 멈췄다"로 오해하기 쉽다.
            self._obstacle_cluster_x = np.empty(0, dtype=np.float32)
            self._obstacle_cluster_y = np.empty(0, dtype=np.float32)
            self._obstacle_front_all_x = np.empty(0, dtype=np.float32)
            self._obstacle_front_all_y = np.empty(0, dtype=np.float32)
            self._obstacle_cluster_group_count = 0

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
            # [2026-08-23] 'lidar_bev' 창만 다른 디버그 창들과 달리 moveWindow가 아예 없어서
            #   OS 기본 위치(대개 화면 좌상단)에 떠 다른 창(YOLO_신호등 등)과 겹쳤다(요청 반영
            #   — "디버깅창 뜰때 안겹치게") — 나머지 창과 동일한 1회성 가드 패턴 적용.
            if 'lidar_bev' not in self._dbg_windows_positioned:
                cv2.namedWindow('lidar_bev', cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow('lidar_bev', *DEBUG_WIN_POS_LIDAR)
                self._dbg_windows_positioned.add('lidar_bev')
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
                # [2026-08-19] "어떤 라이다 클러스터를 보고 트리거됐는지" 디버그용 스냅샷 —
                # target_speed_est와 같은 이유로 트리거 순간에만 찍는다(매 틱 최신값으로
                # 덮어쓰면 장애물이 멀어진 뒤엔 화면이 비어버림, 위 클래스 초기화부 주석 참고).
                self._avoid_hold_trigger_cluster_x = self._obstacle_cluster_x.copy()
                self._avoid_hold_trigger_cluster_y = self._obstacle_cluster_y.copy()
                self._avoid_hold_trigger_front_all_x = self._obstacle_front_all_x.copy()
                self._avoid_hold_trigger_front_all_y = self._obstacle_front_all_y.copy()
                self._avoid_hold_trigger_obstacle_dist  = self.obstacle_dist
                self._avoid_hold_trigger_obstacle_width = self.obstacle_width
                self._avoid_hold_trigger_obstacle_type  = self.obstacle_type
                self._avoid_hold_trigger_obstacle_side  = self.obstacle_side
                self._avoid_hold_trigger_cluster_pts    = int(self._obstacle_cluster_x.size)
                self._avoid_hold_trigger_group_count    = self._obstacle_cluster_group_count
                self._avoid_hold_trigger_cause = (
                    'lidar' if (self.obstacle_front and self.obstacle_dist < AVOID_HOLD_TRIGGER_DIST_M)
                    else 'da_jump')
                # da 마진 확장(avoid_hold)이 걸리는 이 시점이 실제 "차량 감지" 순간이다 —
                # obstacle_cut_active(ENABLE_OBSTACLE_CUT+라이다근접+YOLO+디바운스)는 훨씬
                # 늦게/드물게 걸려서 여기 걸어야 lookahead 고정이 실제로 동작한다.
                if self.obstacle_type == 'vehicle':
                    self._pp_vehicle_lookahead_fix_until_t = now + PP_VEHICLE_LOOKAHEAD_FIX_SEC
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
        self.lavacon_offset, self.lavacon_done, path_m, self._lavacon_boxes_prev = \
            process_lavacon(self.lidar_ranges, self._lavacon_boxes_prev)
        # [2026-08-19] process_lavacon()이 프레임간 좌/우 EMA(LAVACON_TEMPORAL_EMA_ENABLED,
        # perc_lavacon.py 상단 주석 6) 참고)용으로 반환한 boxes를 self._lavacon_boxes_prev에
        # 그대로 들고 있다가 다음 틱에 다시 넘긴다 — process_lavacon() 자체는 무상태.
        # [2026-08-11] 라바콘 조향 파라미터를 라인주행(_lane_steer())과 완전히 일치시키기로
        # 한 결정 — LAVACON_KP 같은 라바콘 전용 P게인 대신, self.lane_path와 동일하게
        # self.pure_pursuit(같은 PP_* 게인)에 태운다. "1m=DL_PIXELS_PER_METER px,
        # x=오른쪽+, 전방=이미지 위쪽(y 감소)" 스케일로 실측 축거(PP_WHEELBASE_PX)를
        # 캘리브레이션해뒀으므로(controller/pure_pursuit.py 상단 주석 참고), 라이다 미터
        # 좌표(x=전방+, y=좌측+)를 그 스케일로 그대로 변환하면 물리적으로 일관된 입력이
        # 된다 — 차량 기준점은 원점(0,0)으로 두고(_handle_lavacon()이 vehicle_x=0.0으로
        # 호출), 좌측(+y_m)은 이미지 왼쪽(-col_px)에 대응한다.
        # [2026-08-19] path_m 생성 로직 자체는 보로노이 → 콘 클러스터 페어링 → 박스 스택
        # 페어링 순으로 여러 차례 교체됐지만(perc_lavacon.py 상단 주석 참고), 여기 변환/
        # 호출부는 출력 형식이 그대로라 변경 없음.
        self.lavacon_path = [(-y * DL_PIXELS_PER_METER, -x * DL_PIXELS_PER_METER) for x, y in path_m]
        # 위 px 변환 전 원본(라이다 미터 좌표) — _draw_lavacon_bev()가 DEBUG_VIZ_LAVACON일 때
        # 그대로 그려서 "실제로 조향에 쓰이는 경로"를 시각적으로 보여준다.
        self._lavacon_path_m = path_m
        # [2026-08-19] self._lavacon_boxes_prev는 process_lavacon()이 이번 프레임에 반환한
        # boxes(LAVACON_TEMPORAL_EMA_ENABLED면 프레임간 EMA 이미 반영됨)로 위에서 막 갱신됨 —
        # [2026-08-22] 예전엔 여기서 곧장 별도 창(lavacon_ema_bev)에 그렸으나, DEBUG_VIZ_LAVACON
        # 통합 창으로 합치면서 실제 그리기는 이 프레임의 뒤 순서인 perc_lavacon_trigger() →
        # _draw_lavacon_bev()로 넘겼다(같은 틱 안에서 self._lavacon_boxes_prev를 그대로
        # 읽으므로 값은 최신 그대로).

    # [2026-08-20] da 근접 컷(ENABLE_OBSTACLE_CUT) 진입 트리거 — perc_lavacon_trigger()와
    #   동일한 "라이다 AND YOLO 카메라" 이중확인 패턴이지만, perc_obstacle()의 공유 ROI
    #   (FRONT_X_MAX/FRONT_Y_HALF, 다른 B2/B3/avoid_hold 소비처와 공유)를 재사용하지
    #   않고 이 목적 전용의 독립 라이다 ROI(OBSTACLE_CUT_TRIGGER_X_MAX_M/Y_HALF_M)를
    #   새로 계산한다 — 그 소비처들의 튜닝이 나중에 이 트리거와 갈라져도 서로 간섭하지
    #   않게(config.py 해당 상수 주석 참고).
    #   [2026-08-20] 요청 반영 — B3(방해차량) 전용이던 걸 B2(고정장애물=라바콘 1개)까지
    #   포함하도록 확장했다. run_behavior_fsm()의 Phase.OBSTACLE_ZONE 분기가 이 트리거를
    #   B2/B3 공용 진입~탈출 신호로 그대로 재사용한다(§4/§5 README 참고).
    #   입력 self.lidar_ranges, self.cone_detected_yolo(perc_yolo_cone()이 먼저 채워둠),
    #        self.vehicle_detected_yolo_cut(perc_yolo_vehicle_cut()이 먼저 채워둠)
    #   출력 self.obstacle_cut_trigger, self.obstacle_cut_type('fixed'/'vehicle'),
    #        self._obstacle_cut_y(트리거 확정 순간 장애물 횡위치, m, +좌측)
    def perc_obstacle_cut_trigger(self):
        # [2026-08-22] 요청 반영 — B1(Phase.LAVACON) 중엔 이 트리거 자체를 죽인다.
        # perc_yolo_cone()이 라바콘 구간에서도 계속 돌아서(위 perceive_all()) 콘이
        # 잡히는 순간 여기서도 라이다+YOLO AND가 성립해버려, 아직 B2 대기 상태(Phase.
        # OBSTACLE_ZONE)로 넘어가기도 전에 obstacle_cut_active가 켜지는 문제가 있었다
        # (DA창에 CUT이 뜨는 원인). ENABLE_OBSTACLE_CUT=False와 동일한 완전 비활성
        # 경로를 그대로 태워 처리 — Phase.OBSTACLE_ZONE 진입 후에만 실제로 발동한다.
        if not ENABLE_OBSTACLE_CUT or self.phase == Phase.LAVACON:
            self._obstacle_cut_trigger_cnt = 0
            self.obstacle_cut_trigger = False
            return

        BODY_LO, BODY_HI = 215, 305   # perc_obstacle()/perc_lavacon_trigger()와 동일(차체 자기가림)

        if self.lidar_ranges is None:
            self._obstacle_cut_trigger_cnt = 0
            self.obstacle_cut_trigger = False
            self._obstacle_cut_lidar_near = False
            self._obstacle_cut_bg_x = self._obstacle_cut_bg_y = np.array([])
            self._obstacle_cut_roi_x = self._obstacle_cut_roi_y = np.array([])
            return

        ranges = np.array(self.lidar_ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = 0.0
        ranges[ranges <= 0.0] = 0.0
        n = len(ranges)
        if BODY_MASK_ENABLED and n > BODY_LO:
            ranges[BODY_LO:min(BODY_HI, n)] = 0.0

        m = min(n, 360)
        deg = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False) - math.radians(LIDAR_ANGLE_OFFSET_DEG)
        r = ranges[:m]
        x = r * np.cos(deg)          # 전방(+앞)
        y = r * np.sin(deg)          # 횡방향(+좌/-우, perc_lavacon_trigger()와 동일 규약)
        # [2026-08-23r, 요청 반영] B3(방해차량, self._b2_passed==True → _active_yolo_stage()가
        # 'vehicle') 단계에서는 트리거 ROI 횡방향 반폭을 OBSTACLE_CUT_TRIGGER_Y_HALF_M_VEHICLE로
        # 좁힌다 — _debug_viz_obstacle_cut()도 이 값을 그대로 읽어 박스를 그리도록
        # self._obstacle_cut_y_half에 저장.
        # [2026-08-23r2, 요청 반영] B3는 전방 트리거 거리도 OBSTACLE_CUT_TRIGGER_X_MAX_M_VEHICLE로
        # 별도 확장(y_half와 동일하게 self._b2_passed로 분기).
        y_half = (OBSTACLE_CUT_TRIGGER_Y_HALF_M_VEHICLE if self._b2_passed
                  else OBSTACLE_CUT_TRIGGER_Y_HALF_M)
        x_max = (OBSTACLE_CUT_TRIGGER_X_MAX_M_VEHICLE if self._b2_passed
                 else OBSTACLE_CUT_TRIGGER_X_MAX_M)
        self._obstacle_cut_y_half = y_half
        self._obstacle_cut_x_max = x_max
        roi_mask = ((r > 0.0) & (x > 0.0) & (x < x_max)
                    & (np.abs(y) < y_half))

        lidar_near = bool(np.any(roi_mask))
        self._obstacle_cut_lidar_near = lidar_near
        if lidar_near:
            idx = np.where(roi_mask)[0]
            nearest = idx[int(np.argmin(r[idx]))]  # 가장 가까운 점의 횡위치를 대표값으로
            self._obstacle_cut_y = float(y[nearest])

        # [2026-08-20] _debug_viz_obstacle_cut() BEV 패널용 라이브 스냅샷 — ROI보다 조금
        # 넓게 잡아서(여백 없으면 박스 바로 밖 점이 왜 트리거 안 됐는지 안 보임) 배경으로
        # 같이 그린다. roi_mask 안쪽만 별도로 빼서 "실제로 트리거에 쓰인 점"을 구분.
        disp_mask = ((r > 0.0) & (x > 0.0) & (x < x_max + 1.0)
                     & (np.abs(y) < y_half + 0.45))
        self._obstacle_cut_bg_x, self._obstacle_cut_bg_y = x[disp_mask], y[disp_mask]
        self._obstacle_cut_roi_x, self._obstacle_cut_roi_y = x[roi_mask], y[roi_mask]

        # [2026-08-20] 요청 반영 — B2(고정장애물=라바콘 1개)/B3(방해차량) 공용 트리거로 확장.
        # 카메라 확인을 "차량 YOLO 단독"에서 "콘 YOLO OR 차량 YOLO"로 넓힌다 — B1 진입
        # 트리거(perc_lavacon_trigger())가 이미 매 틱 갱신해두는 self.cone_detected_yolo를
        # 그대로 재사용(B1이 지금 placeholder라 해당 검출기가 유휴 상태인 것과 무관하게
        # perc_yolo_cone() 자체는 계속 돈다, perceive_all() 참고).
        # [2026-08-23] 요청 반영 — B2(콘)/B3(차량) 각자 다른 임계값(YOLO_CONE_MIN_BOX_AREA_PX_B2/
        # YOLO_VEHICLE_MIN_BOX_AREA_PX_B3)으로 "가장 큰 검출 박스 면적" 게이트를 건다. B1은
        # perc_lavacon_trigger()가 별도 임계값(_B1)으로 독립적으로 건다(config.py 참고).
        # [2026-08-24] 요청 반영 — 면적게이트 통과가 1프레임 순간값이면 안 믿고, YOLO_AREA_CONFIRM_FRAMES
        # 연속 유지돼야 cone_seen/vehicle_seen을 True로 친다(하나라도 빠지면 즉시 0 리셋).
        cone_area_ok_now = (self.cone_detected_yolo
                        and self.cone_max_box_area >= YOLO_CONE_MIN_BOX_AREA_PX_B2) if self.yolo_cone_detector is not None else False
        self._cone_area_b2_cnt = self._cone_area_b2_cnt + 1 if cone_area_ok_now else 0
        cone_seen    = self._cone_area_b2_cnt >= YOLO_AREA_CONFIRM_FRAMES

        vehicle_area_ok_now = (self.vehicle_detected_yolo_cut
                        and self.vehicle_max_box_area_cut >= YOLO_VEHICLE_MIN_BOX_AREA_PX_B3) if self.yolo_vehicle_cut_detector is not None else False
        self._vehicle_area_b3_cnt = self._vehicle_area_b3_cnt + 1 if vehicle_area_ok_now else 0
        vehicle_seen = self._vehicle_area_b3_cnt >= YOLO_AREA_CONFIRM_FRAMES
        # [2026-08-21] 방해차량(vehicle) 오검출 방지 — 라이다/YOLO가 각각 판단한 좌우가
        # 일치할 때만 vehicle_seen을 신뢰한다. 라이다·카메라가 "이번 틱에 뭔가 있다"까지는
        # 둘 다 맞아도 서로 다른 위치(예: 라이다는 우측 근접 장애물, YOLO는 좌측 배경
        # 오검출)를 보고 있으면 우연히 같은 프레임에 겹친 것뿐이라 방해차량 확정 근거로
        # 부족하다 — 좌우 부호가 어긋나면 이번 틱은 vehicle_seen을 취소해 cone_seen/라이다
        # 단독 판정 경로로 폴백시킨다(아래 cam_confirmed 계산 참고). self._obstacle_cut_y는
        # 위에서 lidar_near일 때만 갱신되므로 그 경우에만 비교한다. y>0=좌측 규약(위 x,y
        # 계산부 주석)과 yolo_vehicle.py get_latest_side()의 'L'/'R'은 둘 다 전방 카메라/
        # 라이다 기준 같은 실세계 좌우라 부호만 맞춰주면 바로 비교 가능하다.
        # [2026-08-21 수정, 요청 반영] "가장 가까운 점 하나"만으로 정한 lidar_side가 YOLO와
        # 어긋나도, ROI 안에 YOLO가 가리키는 쪽 점이 실제로 있으면(=좌우 양쪽 다 찍힌 경우,
        # 가장 가까운 점이 우연히 반대쪽이었을 뿐) 더 이상 veto하지 않는다 — 대신 YOLO가
        # 가리키는 쪽 점들 중 가장 가까운 점으로 self._obstacle_cut_y를 다시 골라, 실제
        # 컷 방향이 YOLO 검출 방향을 따라가게 한다(set_obstacle()이 이 값을 그대로 씀,
        # perc_lane() 참고). veto는 이제 "YOLO가 가리키는 쪽엔 라이다 점이 아예 없다"는
        # 진짜 불일치일 때만 발동한다.
        self._obstacle_cut_lidar_side = self._obstacle_cut_yolo_side = None
        self._obstacle_cut_side_veto = False
        if vehicle_seen and lidar_near and self.yolo_vehicle_cut_detector is not None:
            yolo_side = self.yolo_vehicle_cut_detector.get_latest_side()
            lidar_side = 'L' if self._obstacle_cut_y > 0.0 else 'R'
            self._obstacle_cut_lidar_side, self._obstacle_cut_yolo_side = lidar_side, yolo_side
            if yolo_side is not None and yolo_side != lidar_side:
                yolo_side_mask = roi_mask & ((y > 0.0) if yolo_side == 'L' else (y < 0.0))
                if np.any(yolo_side_mask):
                    yolo_side_idx = np.where(yolo_side_mask)[0]
                    nearest_yolo_side = yolo_side_idx[int(np.argmin(r[yolo_side_idx]))]
                    self._obstacle_cut_y = float(y[nearest_yolo_side])
                    self._obstacle_cut_lidar_side = yolo_side
                else:
                    vehicle_seen = False
                    self._obstacle_cut_side_veto = True
        # 두 검출기 다 초기화 실패면(드묾) perc_lavacon_trigger()와 동일 원칙으로 라이다
        # 단독 판정으로 폴백(카메라 확인 자체를 생략, cam_confirmed=True).
        both_cams_unavailable = self.yolo_cone_detector is None and self.yolo_vehicle_cut_detector is None
        cam_confirmed = cone_seen or vehicle_seen or both_cams_unavailable
        confirmed_now = lidar_near and cam_confirmed
        self._obstacle_cut_trigger_cnt = self._obstacle_cut_trigger_cnt + 1 if confirmed_now else 0
        self.obstacle_cut_trigger = self._obstacle_cut_trigger_cnt >= OBSTACLE_CUT_TRIGGER_FRAMES

        if confirmed_now:
            if vehicle_seen and not cone_seen:
                self.obstacle_cut_type = 'vehicle'
            elif cone_seen and not vehicle_seen:
                self.obstacle_cut_type = 'fixed'
            else:
                # 둘 다 잡히거나(드묾) 둘 다 카메라 폴백인 경우 — perc_obstacle()의 라이다
                # 폭 기반 obstacle_type으로 타이브레이크, 그것도 미확정('none')이면 트랙
                # 순서상 먼저 나오는 'fixed'를 기본값으로 둔다.
                self.obstacle_cut_type = self.obstacle_type if self.obstacle_type != 'none' else 'fixed'

    # [2026-08-20] da 근접 컷 유지/해제 타이머 — "카메라에서 장애물이 잠깐 안 보인다고
    #   회피가 바로 꺼지면 안 된다"(요청 원문)에 대한 대응. avoid_hold(§2.32)와 완전히
    #   독립된 상태이며, avoid_hold와 정확히 같은 이유(README §2.32: "카메라가 차량
    #   앞코에 있어 장애물을 지나치는 순간 즉시 원래 폭으로 돌아와... 너무 이른 복귀가
    #   충돌로 이어질 위험")를 이 컷 메커니즘에도 그대로 적용한다.
    #   설계:
    #     진입: perc_obstacle_cut_trigger()가 디바운스 통과시키는 순간 hold-start 시각을 찍는다.
    #     최소유지(floor): OBSTACLE_CUT_HOLD_SEC_MIN 동안은 라이다/YOLO가 뭐라 하든 무조건 유지.
    #     해제: floor를 넘긴 뒤 "진입과 동일한 전용 트리거 ROI"가 더 이상 아무것도 안 잡음이
    #           OBSTACLE_CUT_RELEASE_CONFIRM_FRAMES 연속 유지돼야 해제한다 — 일부러 YOLO는
    #           여기서 다시 안 본다(카메라가 옆/뒤로 빠진 차를 놓치는 건 정상 현상이지 "회피가
    #           끝났다"는 근거가 아니므로, 진입 확신에만 쓰고 퇴장 판단에는 안 쓴다).
    #   perc_obstacle()의 공유 obstacle_front/obstacle_dist(범위가 다름, 5.0m/1.5m)가 아니라
    #   위 트리거와 같은 독립 ROI로 재계산한 clear 상태를 쓴다 — 범위가 다르면 해제 타이밍이
    #   트리거 설계 의도와 어긋난다.
    def _update_obstacle_cut_hold(self):
        # [2026-08-22] 요청 반영 — perc_obstacle_cut_trigger()와 동일하게 B1(Phase.
        # LAVACON) 중엔 유지 타이머도 강제로 끈다. 안 끄면 라바콘 구간에서 직전 틱까지
        # 남아있던 obstacle_cut_active/hold 타이머가 그대로 유지될 수 있다.
        if not ENABLE_OBSTACLE_CUT or self.phase == Phase.LAVACON:
            self.obstacle_cut_active = False
            return

        now = time.time()
        if self.obstacle_cut_trigger:
            if self._obstacle_cut_until_t <= now:   # 새 진입(직전엔 비활성이었음)
                self._obstacle_cut_hold_start_t = now
                # [2026-08-21] B2(고정장애물)는 정지해 있어 회피가 끝나면 바로 지나쳐가므로
                # B3(방해차량, OBSTACLE_CUT_HOLD_SEC_MIN)보다 짧은 최소유지시간을 쓴다 —
                # perc_obstacle_cut_trigger()가 바로 이번 틱에 obstacle_cut_type을 이미
                # 확정해뒀으므로(같은 틱 내 호출 순서, perceive_all() 참고) 여기서 바로 읽어도 된다.
                self._obstacle_cut_hold_sec_min = (
                    OBSTACLE_CUT_HOLD_SEC_MIN_FIXED if self.obstacle_cut_type == 'fixed'
                    else OBSTACLE_CUT_HOLD_SEC_MIN)
                # [2026-08-21, 요청 반영] 진입을 확정지은 이번 틱의 self._obstacle_cut_y
                # 부호로 트리거 쪽을 캡처 — 해제 판정(_obstacle_cut_roi_clear())이 이 쪽만
                # 보고 반대쪽 라이다 상황과 무관하게 판단하게 한다.
                self._obstacle_cut_trigger_side = (
                    'L' if self._obstacle_cut_y is not None and self._obstacle_cut_y > 0.0 else 'R')
                self._obstacle_cut_y_locked = self._obstacle_cut_y
            elif (self.obstacle_cut_type == 'vehicle' and self._obstacle_cut_yolo_side is not None
                  and not self._obstacle_cut_side_veto and self._obstacle_cut_y is not None):
                # [2026-08-24, 요청 반영] 유지 중(진입 아님)에도 방향 갱신을 허용하되, 이번 틱
                # 라이다·YOLO 좌우가 실제로 일치(perc_obstacle_cut_trigger()의 교차검증 통과 —
                # self._obstacle_cut_yolo_side가 None이 아니고 veto도 안 걸렸다는 뜻)할 때만.
                # 라이다 최근접점 단독 흔들림(노이즈)만으론 안 바뀌고, 카메라도 같은 쪽을
                # 확인해줄 때만 컷 방향이 실제로 넘어간다 — 그래야 B3처럼 차체가 라이다에
                # 넓게 걸쳐 최근접점이 순간적으로 반대쪽으로 튀는 경우와, 차량이 실제로
                # 이동해 컷 방향이 바뀌어야 하는 경우를 구분할 수 있다. B2(cone)는 이
                # 좌우 교차검증 자체가 없어(perc_obstacle_cut_trigger() 참고) 적용 안 됨 —
                # 진입 시 고정값 그대로 유지.
                self._obstacle_cut_trigger_side = self._obstacle_cut_yolo_side
                self._obstacle_cut_y_locked = self._obstacle_cut_y
            self._obstacle_cut_until_t = now + self._obstacle_cut_hold_sec_min
            self._obstacle_cut_release_cnt = 0
        else:
            # 트리거 ROI가 클리어됐는지(라이다만, YOLO는 안 봄 — 위 클래스 주석 참고)
            lidar_clear = self.lidar_ranges is not None and self._obstacle_cut_roi_clear()
            self._obstacle_cut_release_cnt = self._obstacle_cut_release_cnt + 1 if lidar_clear else 0

        floor_elapsed = (now - self._obstacle_cut_hold_start_t) >= self._obstacle_cut_hold_sec_min
        was_active = self.obstacle_cut_active
        if (was_active and floor_elapsed
                and self._obstacle_cut_release_cnt >= OBSTACLE_CUT_RELEASE_CONFIRM_FRAMES):
            self._obstacle_cut_until_t = now
            self.obstacle_cut_release_reason = 'floor_and_lidar_clear'

        self.obstacle_cut_active = now < self._obstacle_cut_until_t
        if was_active and not self.obstacle_cut_active and self.obstacle_cut_release_reason != 'floor_and_lidar_clear':
            self.obstacle_cut_release_reason = 'timeout'
        # [2026-08-21] obstacle_cut 진입 엣지(직전엔 비활성 → 이번 틱 활성)에서만 부스트
        # 타이머를 새로 찍는다 — 활성 유지 중에는 안 건드려서(재진입 아닌 한) 1초가 매틱
        # 늘어나 계속 부스트 상태로 안 남게 한다. config.py PP_CURVATURE_BOOST_GAIN 주석 참고.
        if not was_active and self.obstacle_cut_active:
            self._pp_curvature_boost_until_t = now + PP_CURVATURE_BOOST_SEC

    def _obstacle_cut_roi_clear(self):
        """perc_obstacle_cut_trigger()와 동일한 전용 ROI(OBSTACLE_CUT_RELEASE_DIST_M로
        약간 넓힌 히스테리시스 거리)에 아무 점도 안 잡히는지 — 해제 판단 전용이라
        obstacle_cut_trigger 계산과 별개로 다시 라이다를 훑는다(가벼운 연산).

        [2026-08-21, 요청 반영] 좌우 대칭(|y|<Y_HALF_M) 전체가 아니라, 진입을 확정지었던
        그 순간의 쪽(self._obstacle_cut_trigger_side, 'L'/'R')만 본다 — 반대쪽에 뭐가
        잡히든 안 잡히든 해제 판정과 무관하게, "그때 트리거됐던 라이다"가 사라졌는지로만
        판단한다. side가 아직 None(진입 이력 없음)이면 이전처럼 양쪽 다 본다(안전 폴백)."""
        BODY_LO, BODY_HI = 215, 305
        ranges = np.array(self.lidar_ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = 0.0
        ranges[ranges <= 0.0] = 0.0
        n = len(ranges)
        if BODY_MASK_ENABLED and n > BODY_LO:
            ranges[BODY_LO:min(BODY_HI, n)] = 0.0
        m = min(n, 360)
        deg = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False) - math.radians(LIDAR_ANGLE_OFFSET_DEG)
        r = ranges[:m]
        x = r * np.cos(deg)
        y = r * np.sin(deg)
        side = self._obstacle_cut_trigger_side
        if side == 'L':
            side_mask = (y > 0.0) & (y < OBSTACLE_CUT_TRIGGER_Y_HALF_M)
        elif side == 'R':
            side_mask = (y < 0.0) & (y > -OBSTACLE_CUT_TRIGGER_Y_HALF_M)
        else:
            side_mask = np.abs(y) < OBSTACLE_CUT_TRIGGER_Y_HALF_M
        roi_mask = (r > 0.0) & (x > 0.0) & (x < OBSTACLE_CUT_RELEASE_DIST_M) & side_mask
        return not bool(np.any(roi_mask))

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
        # [2026-08-19] LON_MAX 0.5→0.7(폭 0.2→0.4m) — perc_lavacon.py BOX_LON_WIDTH를
        #   실측 콘 간격(옛 박스 폭 기준 2칸≈0.4m)에 맞춰 키운 것과 동일 폭으로 유지
        #   (perc_lavacon.py 상단 "박스 스택 페어링 파라미터" 주석 — 두 값은 항상 같이 바꿀 것).
        # [2026-08-22] 0.3~0.7 → -0.1~0.3(요청 반영, 폭 0.4m 유지한 채 통째로 앞으로 당김) —
        #   push ROI(config.py LAVACON_PUSH_LON_MIN/MAX)를 같은 이유로 먼저 옮긴 것과 맞춤
        #   (자차 마커 기준 전방 0.3m부터 시작하도록 — 마커는 라이다 원점보다
        #   LAVACON_BOX_LON_WIDTH(0.4m)만큼 뒤에 그려지므로 라이다 원점 기준으로는
        #   0.3-0.4=-0.1부터). 두 ROI는 항상 같이 맞출 것(위 push ROI 주석 참고) — 하나만
        #   바꾸면 lavacon_bev에서 노란/보라 박스 시작점이 다시 어긋나 보인다.
        LON_MIN, LON_MAX = -0.1, 0.3   # 트리거 ROI 전방 종방향(m) — 너무 가깝거나(차체 반사) 먼 점 배제
        # [2026-08-22k] 2.0 → 0.5 → 0.75(요청 반영) — 좌우 각 0.75m(총 1.5m). 처음 0.5로
        #   좁힐 땐 perc_lavacon.py CONE_LAT_LIMIT(좌우 검출 박스 폭)도 같은 값으로 맞춰서
        #   lavacon_bev의 트리거 박스와 검출 박스 사이 빈 공간을 없앴었는데, 그 박스 스택
        #   시각화 자체를 §3.8(2026-08-22k)에서 지우면서 "두 값을 항상 같이 맞출" 이유가
        #   없어졌다 — 이번엔 트리거 박스(LAT_MAX)만 단독으로 조정. CONE_LAT_LIMIT은 여전히
        #   0.5로 남아있고(perc_lavacon.py, cone 후보 필터링용), 필요하면 별도로 바꿀 것.
        LAT_MAX           = 0.75       # 트리거 ROI 횡방향 한계(m)
        # [2026-08-23p, 요청 반영] 2 → 1(최소치) — B1 라바콘 진입 트리거가 너무 늦게/안
        #   걸린다는 판단으로 라이다 클러스터 인정 기준을 완화. CHECKER_PILLAR_CLUSTER_MIN_PTS
        #   =1(config.py, 체크무늬 게이트 기둥쌍 검출)에서 이미 쓰던 것과 동일한 완화 —
        #   사이드당 1포인트만 잡혀도 클러스터로 인정한다. 노이즈(단일 반사점)에 더 민감해질
        #   여지가 있으니, 실차에서 유령 트리거(콘이 없는데 B1 진입)가 보이면 되돌릴 것.
        CLUSTER_MIN_PTS   = 1          # 클러스터로 인정할 최소 연속 포인트 수(단일 반사점 노이즈 배제)
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
            # [2026-08-22k, 요청 반영] 박스 스택 페어링(boxes/path_m) 시각화는 지워서 더 이상
            # 안 넘긴다 — 그 조향 방식 자체가 이미 폐기됐다(_draw_lavacon_bev() 상단 주석 참고).
            self._draw_lavacon_bev(r, x, y, roi, LON_MIN, LON_MAX, LAT_MAX,
                                    left_pts, left_run, right_pts, right_run,
                                    r_raw, deg, BODY_LO, body_hi_eff)

        # YOLO 콘 검출 AND 좌우 라이다 클러스터 동시검출이 연속 프레임 유지되면 진입 확정(디바운스).
        # yolo_cone_detector 초기화 실패 시엔 카메라 조건을 무조건 통과(True)시켜 라이다
        # 단독 판정으로 자연스럽게 폴백한다.
        # [2026-08-23] 요청 반영 — B1 진입 트리거는 B2(perc_obstacle_cut_trigger())와
        # 다른 임계값(YOLO_CONE_MIN_BOX_AREA_PX_B1)으로 "가장 큰 검출 박스 면적"을 따로
        # 건다(config.py 해당 상수 주석 참고) — 같은 콘 검출기를 공유하지만 진입 전(B1)과
        # 이미 진입해 지나치는 중(B2)은 콘까지의 거리 감이 달라질 수 있어서 분리했다.
        # [2026-08-24] 요청 반영 — 면적게이트(_B1)도 YOLO_AREA_CONFIRM_FRAMES 연속 통과해야
        # cone_confirmed_cam=True (B2/B3와 동일 패턴, 카운터는 _cone_area_b1_cnt로 별도).
        if self.yolo_cone_detector is not None:
            cone_area_ok_now = (
                self.cone_detected_yolo and self.cone_max_box_area >= YOLO_CONE_MIN_BOX_AREA_PX_B1
            )
            self._cone_area_b1_cnt = self._cone_area_b1_cnt + 1 if cone_area_ok_now else 0
            cone_confirmed_cam = self._cone_area_b1_cnt >= YOLO_AREA_CONFIRM_FRAMES
        else:
            cone_confirmed_cam = True
        if cone_confirmed_cam and self.lavacon_left_detected and self.lavacon_right_detected:
            self._lavacon_trigger_cnt += 1
        else:
            self._lavacon_trigger_cnt = 0
        self.lavacon_trigger = self._lavacon_trigger_cnt >= LAVACON_TRIGGER_FRAMES

    # [2-4c] [DEBUG_VIZ_LAVACON] 라바콘 통합 디버그 BEV — 'lavacon_bev' 창 하나.
    #   [2026-08-22k, 요청 반영] 예전엔 박스 스택 페어링(perc_lavacon.py `_pick_boxed_sides`/
    #   `_build_path`, 초록/주황 채움 박스 + EMA 좌우 차선 + 노란 경로선)까지 같이 그렸는데,
    #   그 조향 방식 자체가 이미 두 단계로 폐기됐다 — 2026-08-19에 `LAVACON_STEER_MODE_DA_PUSH
    #   =True`로 전환되며 박스 스택 경로는 `_handle_lavacon()`의 안 쓰는 폴백 분기로 밀려났고,
    #   2026-08-20엔 `_handle_lavacon()` 자체가 아예 안 불리게 됐다(`run_behavior_fsm()`이
    #   B1_LAVACON 진입/탈출 트리거만 쓰고 그 사이는 그냥 S1 차선주행). 화면엔 계속 그려지고
    #   있었지만 조향엔 이미 관여하지 않던 잔존 시각화라 지웠다. 지금 실제로 쓰이는 정보만
    #   그린다:
    #   ① 트리거 ROI(노란 박스, BGR로는 청록 `(0,220,220)`) — `perc_lavacon_trigger()`의
    #      `LAT_MAX`/`LON_MIN~MAX`. 좌우 라이다 클러스터 동시검출로 B1_LAVACON 진입을
    #      확정하는 판정 범위(YOLO 콘 AND 조건은 별도, 아래 텍스트 참고).
    #   ② push ROI(자홍 박스) — `_lavacon_steer_da_push()`(`_lane_drive()`가
    #      `self._lavacon_engaged`일 때 매 틱 호출, 지금 실제 조향에 쓰이는 유일한 라바콘
    #      전용 로직)가 보는 `LAVACON_PUSH_LON_MIN~MAX × ±LAVACON_PUSH_LAT_LIMIT` 범위 —
    #      이 안에서 좌/우 각각 가장 가까운 콘 1점의 횡위치(y)만 본다. 자홍 가로 눈금선이
    #      실제 검출된 y위치(정확한 종방향 위치가 아니라 ROI 안에 있다는 표시일 뿐 — 함수
    #      자체가 y만 반환하고 x는 안 준다), 어두운 자홍 실선이 안전마진
    #      (`LAVACON_PUSH_SAFETY_MARGIN_M`) 경계 — 콘이 이 선보다 안쪽으로 들어오면 반대쪽
    #      으로 밀린다(파랑=마진 침범 중, 초록=안전).
    #   · perc_obstacle()의 DEBUG_VIZ_LIDAR 창과 같은 스타일, ROI/축척만 라바콘 트리거에
    #     맞게 확대. 초록=좌측(y>0) ROI점, 주황=우측(y<0) ROI점, 회색=ROI 밖.
    #   · 자홍(magenta) 점=BODY_LO~BODY_HI "차체 자기가림"이라고 보고 지워버리는 구간의
    #     마스킹 전 원본(raw) 점 — 이 구간에 실제 물체(콘)가 있는데도 마스크가 지우고
    #     있는 건 아닌지 진단용(push ROI 박스 테두리와 색이 비슷하니 혼동 주의 — 점은 항상
    #     원본 라이다 반사점, 박스/눈금선은 항상 ROI 경계·검출값).
    #   · 반투명 회색 부채꼴 = 그 구간의 각도 범위 자체(데드존, 135~224도/정후방 중심
    #     90도 폭) — 실제 반사점 유무와 무관하게 항상 표시.
    #   · 텍스트: pts(트리거 ROI 내 점수)/run(최대 연속묶음, CLUSTER_MIN_PTS=2 이상이어야
    #     클러스터로 인정), YOLO 콘 검출, 트리거 디바운스 카운터, push ROI 좌우 검출값/현재
    #     push량/방향/engaged 여부.
    def _draw_lavacon_bev(self, r, x, y, roi, lon_min, lon_max, lat_max,
                           left_pts, left_run, right_pts, right_run,
                           r_raw, deg, body_lo, body_hi_eff):
        # [2026-08-22k] 80→100(요청 반영) — 트리거/검출 ROI 폭이 좌우 ±2.0m/±1.0m에서
        #   ±0.5m로 좁아진 만큼 배율도 같이 키웠다. 100으로 잡은 이유: 전방 시야 한도가
        #   ORIGIN_EY(아래 정의, 460px)/PPM이라 CONE_LON_MAX(perc_lavacon.py, 4.0m, 박스
        #   스택 전체 깊이)를 잘라먹지 않을 최댓값 근방을 골라야 한다 — 100이면
        #   460/100=4.6m로 4.0m에 여유(0.6m)가 남는다. 그보다 더 키우면(예: 120 →
        #   460/120≈3.83m) 원거리 박스/경로가 창 위로 잘려나갈 위험.
        PPM = 100          # 1m = 100px (좁은 트리거 ROI라 perc_obstacle보다 확대)
        W, H = 500, 600    # [2026-08-19] 상단이 너무 길다는 요청으로 900→600(2/3)로 축소
        # [2026-08-19] 진짜 원인 발견: EX,EY를 "라이다 원점(모든 센서점의 좌표변환 기준)"과
        #   "자차 마커(파란 점) 그리는 위치"에 동시에 같은 값으로 써왔다 — to_px()도, 거리원도,
        #   ROI 점(라인 1208/1216 부근)도 전부 EX,EY를 원점으로 삼는다. 그래서 EX,EY를 옮기면
        #   마커와 센서점(실제로 찍히는 라바콘 반사점)이 "같이" 움직여서, 화면상 마커와
        #   반사점의 상대 위치는 절대 안 바뀌었던 것 — 사용자가 실측 물체를 두고 그 반사점과
        #   마커를 맞춰보려 했는데 계속 그대로였던 진짜 원인. 라이다 원점(ORIGIN_EX/EY, 센서
        #   좌표변환 전용 — 실측 캘리브레이션 기준이라 고정)과 자차 마커 위치(MARKER_EX/EY)를
        #   분리해서, 마커만 따로 움직이게 고쳤다.
        ORIGIN_EX, ORIGIN_EY = 250, 460   # 라이다 원점(센서점 좌표변환 전용, 고정)
        BEAK_LEN = 18      # 자차 마커 헤딩 표시선("주둥이") 길이(px, 표시용) — 위치 오프셋과는 무관
        # 자차 마커(파란 원)를 실제 물리거리 기준으로 라이다 원점보다 뒤로 당겨서 그린다 —
        #   지금은 "적절한 위치가 어디인지" 찾는 튜닝 단계라 이 상수를 눈으로 보면서 조절 중.
        #   이후 값이 확정되면 이게 곧 "라이다 원점 vs 차량 실제 기준점 차이"라는 뜻이 되므로,
        #   그때 process_lavacon()/perc_lavacon_trigger()가 쓰는 x=0 원점 자체도 이 값만큼
        #   실제로 옮길 예정(요청에 따라 순서상 지금은 시각화만, 원점 이동은 다음 단계).
        # [2026-08-19] 임의 픽셀 대신 파란 박스 스택 경계선 한 칸의 세로(종방향) 길이
        #   (LAVACON_BOX_LON_WIDTH, perc_lavacon.py에서 import된 값 — 하드코딩 안 하고
        #   상수로 참조해 값이 바뀌어도 항상 최신값을 따라감)만큼 뒤로 당김 — 요청 반영.
        #   값을 또 바꾸려면 여기 말고 perc_lavacon.py의 BOX_LON_WIDTH를 바꿀 것
        #   (그래야 실제 박스 스택 페어링 로직과 이 마커 위치가 계속 같은 값을 공유한다).
        EGO_MARKER_PULLBACK_PX = int(LAVACON_BOX_LON_WIDTH * PPM)
        MARKER_EX, MARKER_EY = ORIGIN_EX, ORIGIN_EY + EGO_MARKER_PULLBACK_PX
        bev = np.zeros((H, W, 3), dtype=np.uint8)

        for d in (1, 2, 3):
            cv2.circle(bev, (ORIGIN_EX, ORIGIN_EY), d * PPM, (50, 50, 50), 1)
            cv2.putText(bev, f'{d}m', (ORIGIN_EX + 4, ORIGIN_EY - d * PPM + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

        def to_px(wx, wy): return (int(ORIGIN_EX - wy * PPM), int(ORIGIN_EY - wx * PPM))

        # [2026-08-19] 라이다 자기가림 데드존(BODY_LO~BODY_HI, LIDAR_ANGLE_OFFSET_DEG 보정 후
        # 135~224도, 폭 90도, 정후방 중심) 시각화 — 반투명 회색 부채꼴로 "이 각도 구간은
        # 애초에 아예 안 본다"는 걸 항상 보여준다. 기존 자홍 점(아래)은 그 구간에 실제로
        # 뭔가 찍혔는지만 보여줄 뿐 각도 범위 자체는 안 보여줬어서 추가. deg 배열(라디안,
        # 이미 LIDAR_ANGLE_OFFSET_DEG 보정됨)을 그대로 재사용해 각도 계산을 이중화하지 않는다.
        DEADZONE_R = 3.0  # 부채꼴 표시 반경(m) — 3m 거리원까지 채워서 눈에 띄게
        dz_pts = ([(ORIGIN_EX, ORIGIN_EY)]
                  + [to_px(DEADZONE_R * math.cos(a), DEADZONE_R * math.sin(a)) for a in deg[body_lo:body_hi_eff]]
                  + [(ORIGIN_EX, ORIGIN_EY)])
        overlay = bev.copy()
        cv2.fillPoly(overlay, [np.array(dz_pts, dtype=np.int32)], (70, 70, 70))
        cv2.addWeighted(overlay, 0.4, bev, 0.6, 0, bev)

        cv2.rectangle(bev, to_px(lon_min, lat_max), to_px(lon_max, -lat_max), (0, 220, 220), 1)

        # [2026-08-22] §5.10 유령 점 임시 마스크(GHOST_POINT_*, perc_lavacon.py) 시각화용
        #   빨간 원 — 실제 마스킹된 라이다 점은 ranges=0으로 지워지므로 이 화면에 안 찍힌다,
        #   이 원은 "마스킹 중인 구역"을 보여줄 뿐이다. 실제 반경(6cm→PPM=100이면 6px)이
        #   화면에서 거의 안 보여 픽셀로 눈대중 대조하다 오차가 컸던 문제(2026-08-22 사용자
        #   지적) — 실제 마스킹 반경은 원(반투명 채움)으로 정확히 그리되, 중심은 십자선으로
        #   또렷이 표시하고 옆에 미터 좌표를 텍스트로 같이 찍어 픽셀이 아니라 숫자로
        #   대조할 수 있게 한다(위 "range,x ref only" 텍스트 줄의 실제 검출 좌표와 비교).
        _gpx, _gpy = to_px(GHOST_POINT_X_M, GHOST_POINT_Y_M)
        _gp_overlay = bev.copy()
        cv2.circle(_gp_overlay, (_gpx, _gpy), max(int(GHOST_POINT_RADIUS_M * PPM), 1), (0, 0, 255), -1)
        cv2.addWeighted(_gp_overlay, 0.35, bev, 0.65, 0, bev)
        cv2.circle(bev, (_gpx, _gpy), max(int(GHOST_POINT_RADIUS_M * PPM), 1), (0, 0, 255), 1)
        cv2.drawMarker(bev, (_gpx, _gpy), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
        cv2.putText(bev, f'GHOST MASK ({GHOST_POINT_X_M:.3f},{GHOST_POINT_Y_M:.3f}) r={GHOST_POINT_RADIUS_M:.2f}m',
                    (_gpx + 8, _gpy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        # ── push ROI(자홍) — 지금 실제 조향(_lavacon_steer_da_push())이 보는 좌우 최근접 콘 ──
        # [2026-08-22] return_range=True — "y가 작은 쪽"과 "실제로 더 가까운 쪽"이 다를 수
        # 있다는 문제 제기로 range도 같이 받아 아래 PUSH 텍스트에 표시(순수 표시용, push
        # 판정 자체는 여전히 y-마진 비교 그대로 — nearest_cone_lateral() docstring 참고).
        (push_left_y, push_right_y, push_left_range, push_right_range,
         push_left_x, push_right_x) = nearest_cone_lateral(
            self.lidar_ranges, LAVACON_PUSH_LON_MIN, LAVACON_PUSH_LON_MAX, LAVACON_PUSH_LAT_LIMIT,
            return_range=True, lon_max_l=LAVACON_PUSH_LON_MAX_L)
        cv2.rectangle(bev, to_px(LAVACON_PUSH_LON_MIN, 0.0),
                      to_px(LAVACON_PUSH_LON_MAX, -LAVACON_PUSH_LAT_LIMIT), (200, 0, 200), 1)
        # [2026-08-24, 테스트] 좌측(y>0)만 LAVACON_PUSH_LON_MAX_L까지 별도 박스로 표시
        cv2.rectangle(bev, to_px(LAVACON_PUSH_LON_MIN, LAVACON_PUSH_LAT_LIMIT),
                      to_px(LAVACON_PUSH_LON_MAX_L, 0.0), (200, 0, 200), 1)
        cv2.putText(bev, 'PUSH ROI', to_px(LAVACON_PUSH_LON_MAX, LAVACON_PUSH_LAT_LIMIT),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 200), 1, cv2.LINE_AA)
        # 안전마진 경계선(좌 LAVACON_PUSH_SAFETY_MARGIN_L_M / 우 _R_M) — 이 선보다 안쪽으로
        # 콘이 들어오면 반대쪽으로 밀린다.
        cv2.line(bev, to_px(LAVACON_PUSH_LON_MIN, LAVACON_PUSH_SAFETY_MARGIN_L_M),
                 to_px(LAVACON_PUSH_LON_MAX, LAVACON_PUSH_SAFETY_MARGIN_L_M), (140, 0, 140), 1, cv2.LINE_AA)
        cv2.line(bev, to_px(LAVACON_PUSH_LON_MIN, -LAVACON_PUSH_SAFETY_MARGIN_R_M),
                 to_px(LAVACON_PUSH_LON_MAX, -LAVACON_PUSH_SAFETY_MARGIN_R_M), (140, 0, 140), 1, cv2.LINE_AA)
        # 검출된 좌/우 최근접 콘의 y위치를 ROI 폭 전체에 걸친 가로 눈금선으로 표시. 마진
        # 침범 중이면 파랑(=지금 미는 중), 안전하면 초록/주황.
        for py, base_col, margin in ((push_left_y, (0, 255, 0), LAVACON_PUSH_SAFETY_MARGIN_L_M),
                                      (push_right_y, (0, 140, 255), LAVACON_PUSH_SAFETY_MARGIN_R_M)):
            if py is None:
                continue
            col = (255, 80, 0) if abs(py) < margin else base_col
            cv2.line(bev, to_px(LAVACON_PUSH_LON_MIN, py), to_px(LAVACON_PUSH_LON_MAX, py), col, 2)
        # [2026-08-22] 위 눈금선은 "이 y값이다"라고 ROI 전체 폭에 걸쳐 그은 안내선이라, 그
        # y값을 만든 원본 점이 실제로 어디(전후 x까지) 있는지는 안 보여준다는 문제 제기 —
        # 그 점의 (x,y) 정확한 위치에 노란 링 마커를 따로 찍어 "안내선"과 "진짜 그 점"을
        # 구분한다(PPM=100이라 margin=0.13m이 화면상 ±13px밖에 안 돼서 눈금선끼리 거의
        # 겹쳐 보일 수 있음 — 이 마커가 없으면 그 좁은 구간에서 실제 점을 못 짚어낸다).
        for px_, py_ in ((push_left_x, push_left_y), (push_right_x, push_right_y)):
            if px_ is None or py_ is None:
                continue
            cv2.circle(bev, to_px(px_, py_), 6, (0, 255, 255), 2)

        # 마스킹 전 원본(raw) 점 중 BODY_LO~BODY_HI(자기가림 구간)에 해당하는 것만 자홍색으로 표시.
        # ROI 필터 없이 원거리까지 전부 그려서, 이 "자기가림 구간"에 실제로 뭔가 찍히는지 그대로 보여준다.
        masked_x = r_raw[body_lo:body_hi_eff] * np.cos(deg[body_lo:body_hi_eff])
        masked_y = r_raw[body_lo:body_hi_eff] * np.sin(deg[body_lo:body_hi_eff])
        for xi, yi, ri in zip(masked_x, masked_y, r_raw[body_lo:body_hi_eff]):
            if ri <= 0.0:
                continue
            sx, sy = int(ORIGIN_EX - yi * PPM), int(ORIGIN_EY - xi * PPM)
            if 0 <= sx < W and 0 <= sy < H:
                cv2.circle(bev, (sx, sy), 2, (255, 0, 255), -1)

        left_mask, right_mask = roi & (y > 0.0), roi & (y < 0.0)
        for i in range(len(r)):
            if r[i] <= 0.0:
                continue
            sx, sy = int(ORIGIN_EX - y[i] * PPM), int(ORIGIN_EY - x[i] * PPM)
            if not (0 <= sx < W and 0 <= sy < H):
                continue
            if left_mask[i]:    col = (0, 255, 0)
            elif right_mask[i]: col = (0, 140, 255)
            else:                col = (60, 60, 60)
            cv2.circle(bev, (sx, sy), 3, col, -1)

        cv2.circle(bev, (MARKER_EX, MARKER_EY), 6, (255, 220, 0), -1)
        cv2.line(bev, (MARKER_EX, MARKER_EY), (MARKER_EX, MARKER_EY - BEAK_LEN), (255, 220, 0), 2)

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

        # push ROI(자홍) 검출값 + 지금 실제로 나가는 push량/방향 — _lavacon_steer_da_push()와
        # 완전히 같은 부호 규약(좌측 콘 침범→+, 우측 콘 침범→-).
        push_m = 0.0
        if push_left_y is not None and push_left_y < LAVACON_PUSH_SAFETY_MARGIN_L_M:
            push_m += LAVACON_PUSH_SAFETY_MARGIN_L_M - push_left_y
        if push_right_y is not None and -push_right_y < LAVACON_PUSH_SAFETY_MARGIN_R_M:
            push_m -= LAVACON_PUSH_SAFETY_MARGIN_R_M - (-push_right_y)
        push_m *= LAVACON_PUSH_GAIN  # [2026-08-22b] _lavacon_steer_da_push()와 동일 배율 — 표시값 동기화
        push_left_s = 'N/A' if push_left_y is None else f'{push_left_y:.3f}m'
        push_right_s = 'N/A' if push_right_y is None else f'{push_right_y:.3f}m'
        push_col = (255, 80, 0) if push_m != 0.0 else (200, 0, 200)
        cv2.putText(bev, f'PUSH L_y={push_left_s} R_y={push_right_s} '
                          f'push={push_m:+.2f}m engaged={self._lavacon_engaged}',
                    (8, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.45, push_col, 1, cv2.LINE_AA)
        # [2026-08-22] L_y/R_y(횡방향 침범량, push 판정 기준)만으로는 "어느 쪽 콘이
        # 실제로 더 가까운지"(직선거리) 판단이 안 돼서 헷갈린다는 문제 제기 — range를
        # 따로 한 줄 더 찍는다. 어느 한쪽 range가 더 작아도 push 방향은 여전히 L_y/R_y
        # 마진 침범 여부로만 결정된다(설계 그대로, 아래 텍스트는 순수 참고용).
        rl_s = 'N/A' if push_left_range is None else f'{push_left_range:.3f}m'
        rr_s = 'N/A' if push_right_range is None else f'{push_right_range:.3f}m'
        # [2026-08-22] §5.10 유령 점 좌표를 화면(픽셀)으로 눈대중 대조하다 오차가 컸던 문제
        #   — x,y를 소수점 3자리까지 텍스트로 바로 찍어 GHOST_POINT_X_M/Y_M을 픽셀 눈대중 대신
        #   숫자로 정확히 맞출 수 있게 함(아래 GHOST MASK 원 옆 좌표 텍스트와 대조).
        xl_s = 'N/A' if push_left_x is None else f'{push_left_x:.3f}'
        xr_s = 'N/A' if push_right_x is None else f'{push_right_x:.3f}'
        cv2.putText(bev, f'(range,x ref only) L=({xl_s},{rl_s}) R=({xr_s},{rr_s}) - push uses L_y/R_y only',
                    (8, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(bev, f'margin L={LAVACON_PUSH_SAFETY_MARGIN_L_M:.2f}m R={LAVACON_PUSH_SAFETY_MARGIN_R_M:.2f}m gain={LAVACON_PUSH_GAIN:.1f}x '
                          f'lon={LAVACON_PUSH_LON_MIN:.1f}~{LAVACON_PUSH_LON_MAX:.1f}m '
                          f'lat=+-{LAVACON_PUSH_LAT_LIMIT:.1f}m',
                    (8, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        if 'lavacon_bev' not in self._dbg_windows_positioned:
            cv2.namedWindow('lavacon_bev', cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow('lavacon_bev', *DEBUG_WIN_POS_LAVACON)
            self._dbg_windows_positioned.add('lavacon_bev')
        cv2.imshow('lavacon_bev', bev)
        cv2.waitKey(1)

    # [2-4d'] 좌회전 진입 랜드마크 후보 — 체크무늬 게이트 라이다 기둥쌍 검출
    #   입력 self.lidar_ranges → 출력 checker_pillar_left/right_detected, checker_pillar_trigger
    #   perc_lavacon_trigger()와 완전히 동일한 패턴(극좌표→직교좌표, 종방향 ROI, 좌/우
    #   클러스터 탐지)이지만, 좌우 클러스터 간 횡방향 거리가 실측 기둥 간격
    #   (CHECKER_PILLAR_LAT_TARGET_M) 이상이어야만 확정한다는 조건이 추가됐다 — config.py
    #   "체크무늬 게이트 라이다 기둥쌍 검출" 절 참고. [2026-08-22b] 검출조건 완화 이력은
    #   README §1.19 참고.
    def perc_checker_pillar(self):
        if self.lidar_ranges is None:
            self.checker_pillar_left_detected  = False
            self.checker_pillar_right_detected = False
            self.checker_pillar_lat_dist_m     = 0.0
            self._checker_pillar_trigger_cnt   = 0
            self.checker_pillar_trigger        = False
            self._checker_pillar_dbg = (0, 0, 0, 0, 0.0)
            return

        ranges_raw = np.array(self.lidar_ranges, dtype=np.float32)
        ranges_raw[~np.isfinite(ranges_raw)] = 0.0
        ranges_raw[ranges_raw <= 0.0] = 0.0

        ranges = ranges_raw.copy()
        n = len(ranges)
        if BODY_MASK_ENABLED and n > 215:
            ranges[215:min(305, n)] = 0.0   # 차체 자기가림 마스킹 (perc_obstacle/perc_lavacon_trigger와 동일 구간)

        m = min(n, 360)
        deg = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False) - math.radians(LIDAR_ANGLE_OFFSET_DEG)
        r = ranges[:m]
        x = r * np.cos(deg)          # 전방(+앞)
        y = r * np.sin(deg)          # 횡방향(+좌/-우)
        roi = (r > CHECKER_PILLAR_MIN_RANGE_M) & (x > CHECKER_PILLAR_LON_MIN) & (x < CHECKER_PILLAR_LON_MAX) \
              & (np.abs(y) < CHECKER_PILLAR_LAT_MAX)

        def _cluster_center(side_mask):
            """side_mask 안에서 클러스터 조건(연속 포인트 수/거리편차)을 만족하는 가장 큰
            묶음을 찾아 (found, pts, best_run, lat_center) 반환 — lat_center는 그 묶음의
            y좌표 평균(횡방향 위치, m)."""
            idx = np.where(roi & side_mask)[0]
            pts = len(idx)
            if pts < CHECKER_PILLAR_CLUSTER_MIN_PTS:
                return False, pts, 0, 0.0
            splits = np.where(np.diff(idx) > 1)[0] + 1
            found, best_run, lat_center = False, 0, 0.0
            for g in np.split(idx, splits):
                if len(g) > best_run:
                    best_run = len(g)
                if len(g) >= CHECKER_PILLAR_CLUSTER_MIN_PTS and (np.max(r[g]) - np.min(r[g])) <= CHECKER_PILLAR_CLUSTER_MAX_GAP:
                    found = True
                    lat_center = float(np.mean(y[g]))
            return found, pts, best_run, lat_center

        self.checker_pillar_left_detected,  left_pts,  left_run,  left_y  = _cluster_center(y > 0.0)
        self.checker_pillar_right_detected, right_pts, right_run, right_y = _cluster_center(y < 0.0)

        if self.checker_pillar_left_detected and self.checker_pillar_right_detected:
            self.checker_pillar_lat_dist_m = abs(left_y - right_y)
        else:
            self.checker_pillar_lat_dist_m = 0.0
        self._checker_pillar_dbg = (left_pts, left_run, right_pts, right_run, self.checker_pillar_lat_dist_m)

        lat_ok = self.checker_pillar_lat_dist_m >= CHECKER_PILLAR_LAT_TARGET_M
        if self.checker_pillar_left_detected and self.checker_pillar_right_detected and lat_ok:
            self._checker_pillar_trigger_cnt += 1
        else:
            self._checker_pillar_trigger_cnt = 0
        self.checker_pillar_trigger = self._checker_pillar_trigger_cnt >= CHECKER_PILLAR_CONFIRM_FRAMES

        # [2026-08-22j] DEBUG_VIZ_LEFT_TURN 통합창도 이 BEV 프레임을 그대로 넣어 보여주므로
        # (_debug_viz_left_turn() 참고, 요청 반영: "좌회전 디버깅창에 라이다영상도 추가"),
        # 독립 창 플래그(DEBUG_VIZ_CHECKER_PILLAR)가 꺼져 있어도 통합창이 켜져 있으면 계속
        # BEV를 만들어 self._checker_pillar_bev_img에 채워둔다.
        if DEBUG_VIZ_CHECKER_PILLAR or DEBUG_VIZ_LEFT_TURN:
            self._draw_checker_pillar_bev(r, x, y, roi, lat_ok,
                                           left_pts, left_run, right_pts, right_run)

    # [2-4d''] [DEBUG_VIZ_CHECKER_PILLAR] 체크무늬 게이트 라이다 기둥쌍 검출 BEV 시각화
    #   _draw_lavacon_bev()와 동일 스타일(1/2/3m 거리원, 청록 ROI 박스, 좌=초록/우=주황 점).
    #   좌우 기둥쌍 사이 실측 횡방향 거리(checker_pillar_lat_dist_m)가 실측 기준값
    #   (CHECKER_PILLAR_LAT_TARGET_M±TOLERANCE)과 맞는지가 이 게이트만의 추가 확정 조건이라
    #   화면에 그 비교값도 같이 표시한다.
    def _draw_checker_pillar_bev(self, r, x, y, roi, lat_ok,
                                  left_pts, left_run, right_pts, right_run):
        PPM = 80
        W, H = 500, 500
        ORIGIN_EX, ORIGIN_EY = 250, 460
        bev = np.zeros((H, W, 3), dtype=np.uint8)

        for d in (1, 2, 3):
            cv2.circle(bev, (ORIGIN_EX, ORIGIN_EY), d * PPM, (50, 50, 50), 1)
            cv2.putText(bev, f'{d}m', (ORIGIN_EX + 4, ORIGIN_EY - d * PPM + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

        def to_px(wx, wy): return (int(ORIGIN_EX - wy * PPM), int(ORIGIN_EY - wx * PPM))

        cv2.rectangle(bev, to_px(CHECKER_PILLAR_LON_MIN, CHECKER_PILLAR_LAT_MAX),
                      to_px(CHECKER_PILLAR_LON_MAX, -CHECKER_PILLAR_LAT_MAX), (0, 220, 220), 1)

        # [2026-08-22e] 극근접 무시 데드존(CHECKER_PILLAR_MIN_RANGE_M) 시각화 — 이 원
        # 안쪽 점은 roi 계산에서 이미 제외되지만(위 perc_checker_pillar() 참고), 화면에는
        # 여전히 회색 점으로 그려지므로(무시=미검출이지 미표시가 아님) 필터가 실제로
        # 걸리는 범위를 빨간 원으로 눈에 보이게 표시해 "왜 안 사라지지" 오해를 줄인다.
        cv2.circle(bev, (ORIGIN_EX, ORIGIN_EY), int(CHECKER_PILLAR_MIN_RANGE_M * PPM), (0, 0, 255), 1)
        cv2.putText(bev, f'ignore<{CHECKER_PILLAR_MIN_RANGE_M:.2f}m',
                    (ORIGIN_EX + 4, ORIGIN_EY - int(CHECKER_PILLAR_MIN_RANGE_M * PPM) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        left_mask, right_mask = roi & (y > 0.0), roi & (y < 0.0)
        for i in range(len(r)):
            if r[i] <= 0.0:
                continue
            sx, sy = int(ORIGIN_EX - y[i] * PPM), int(ORIGIN_EY - x[i] * PPM)
            if not (0 <= sx < W and 0 <= sy < H):
                continue
            if left_mask[i]:    col = (0, 255, 0)
            elif right_mask[i]: col = (0, 140, 255)
            else:                col = (60, 60, 60)
            cv2.circle(bev, (sx, sy), 3, col, -1)

        cv2.circle(bev, (ORIGIN_EX, ORIGIN_EY), 6, (255, 220, 0), -1)
        cv2.line(bev, (ORIGIN_EX, ORIGIN_EY), (ORIGIN_EX, ORIGIN_EY - 18), (255, 220, 0), 2)

        l_col = (0, 255, 0)   if self.checker_pillar_left_detected  else (0, 0, 255)
        r_col = (0, 140, 255) if self.checker_pillar_right_detected else (0, 0, 255)
        cv2.putText(bev, f'L pts={left_pts} run={left_run}', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, l_col, 1, cv2.LINE_AA)
        cv2.putText(bev, f'R pts={right_pts} run={right_run}', (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, r_col, 1, cv2.LINE_AA)
        dist_col = (0, 255, 0) if lat_ok else (0, 0, 255)
        cv2.putText(bev, f'lat_dist={self.checker_pillar_lat_dist_m:.2f}m '
                          f'(target>={CHECKER_PILLAR_LAT_TARGET_M:.2f})',
                    (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dist_col, 1, cv2.LINE_AA)
        trig_col = (0, 255, 0) if self.checker_pillar_trigger else (200, 200, 200)
        cv2.putText(bev, f'trigger={self.checker_pillar_trigger} '
                          f'cnt={self._checker_pillar_trigger_cnt}/{CHECKER_PILLAR_CONFIRM_FRAMES}',
                    (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, trig_col, 1, cv2.LINE_AA)

        self._checker_pillar_bev_img = bev  # left_turn_debug 통합창(_debug_viz_left_turn())이 재사용

        if DEBUG_VIZ_CHECKER_PILLAR:
            if 'checker_pillar_bev' not in self._dbg_windows_positioned:
                cv2.namedWindow('checker_pillar_bev', cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow('checker_pillar_bev', *DEBUG_WIN_POS_CHECKER_PILLAR)
                self._dbg_windows_positioned.add('checker_pillar_bev')
            # [2026-08-23, 요청 반영: "카메라욜로랑 검출라이다는 크기 완전 작게"] 원본
            # 500x500(self._checker_pillar_bev_img, left_turn_debug 통합창도 재사용)은
            # 그대로 두고, 이 독립 창에 띄울 때만 표시용으로 축소한다(정사각형이라 종횡비
            # 유지 위해 160x160 — 카메라 창 YOLO_신호등의 160x120과 같은 크기 관례).
            small_bev = cv2.resize(bev, (160, 160), interpolation=cv2.INTER_AREA)
            cv2.imshow('checker_pillar_bev', small_bev)
            cv2.waitKey(1)

    # [2-4e] 체크무늬 게이트 통과 후 완만한 조향 램프 (_s0_signal() 'left' 커밋 종료 후 사용)
    #   checker_pillar_trigger가 뜬 시점에 호출부(_begin_checker_ramp_turn())가
    #   self._checker_ramp_dist=0.0으로 시작하면, 이후 매 제어주기 이 함수가 누적거리를
    #   갱신하며 CHECKER_TURN_RAMP_START_ANGLE→
    #   END_ANGLE 사이 조향각을 반환한다. TURN_DIST_M류와 동일하게 _speed_mps_fallback()
    #   기반 거리적분이라 VESC 죽음에도 안전(무한 램프 없음). 램프가 끝나면(None, angle=
    #   END_ANGLE) 튜플을 반환 — 호출부가 ramp_dist를 None으로 리셋하고 다음 로직(예:
    #   차선인식 복귀)으로 넘어가면 된다.
    #   [2026-08-22] CHECKER_TURN_RAMP_CURVE='smoothstep'(3t²-2t³)이 기본값 — t=0(램프
    #   시작)과 t=1(END_ANGLE 고정값으로 넘어가는 지점) 양쪽 다 기울기 0이라 조향이 저크
    #   없이 부드러운 S자 곡선으로 붙는다. 'linear'는 단순 비례(요청의 "선형" 옵션, 튜닝
    #   비교용으로 남겨둠).
    def _checker_turn_ramp_angle(self, cmd_speed):
        if self._checker_ramp_dist is None:
            return CHECKER_TURN_RAMP_START_ANGLE, False

        t = min(self._checker_ramp_dist / CHECKER_TURN_RAMP_DIST_M, 1.0)
        if CHECKER_TURN_RAMP_CURVE == 'smoothstep':
            t = t * t * (3.0 - 2.0 * t)   # 양끝 기울기 0인 S자 곡선 — 시작/종료 모두 저크 없음
        angle = CHECKER_TURN_RAMP_START_ANGLE + (CHECKER_TURN_RAMP_END_ANGLE - CHECKER_TURN_RAMP_START_ANGLE) * t

        done = self._checker_ramp_dist >= CHECKER_TURN_RAMP_DIST_M
        if not done:
            self._checker_ramp_dist += self._speed_mps_fallback(cmd_speed) * 0.05  # 20Hz 제어주기(control_loop) 가정
        return angle, done

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

        # [2026-08-24, 요청 반영] TOTAL_LAPS 기반 레이스 종료 판정(S4_FINISH 전환) 삭제 —
        # yaw/정지선 바퀴 판정 자체를 신뢰하지 않기로 함(요청). 이제 이 함수는 바퀴 수
        # 표시(디버그용 self.lap)만 하고 정지를 유발하지 않는다.

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
            self._signal_yolo_off = False  # [2026-08-23] 새 바퀴 시작 — 다음 교차로 신호등 YOLO 재개
            self._signal_off_hold_cnt = None  # [2026-08-23b] 유예 카운터도 같이 리셋(perc_signal() 참고)
            self._b2_passed = False   # [2026-08-15] Phase.OBSTACLE_ZONE 통합 — 완료 추적도 매 바퀴 리셋
            self._b3_passed = False
            self._obscut_zone_tag = None  # [2026-08-20] da 근접 컷 기반 B2/B3 latch도 매 바퀴 리셋

    def run_mission_fsm(self):
        {
            MissionState.S0_SIGNAL      : self._s0_signal,
            MissionState.S1_LANE_FOLLOW : self._s1_lane_follow,
            MissionState.S4_FINISH      : self._s4_finish,
        }[self.mission_state]()

    def _change_state(self, new_state):
        """
        Mission 상태 전환 공통 처리.
          - 전환 로그 출력(디버깅 추적용)
          - PID 누적값 초기화: 이전 상태에서 쌓인 적분/미분 잔여가 새 상태로 넘어와 튀는 것을 방지한다.
        모든 상태 전환은 반드시 이 함수를 통해서만 한다(직접 대입 금지).

        [2026-08-23] 예전엔 여기서 ctrl_angle/ctrl_speed를 무조건 0.0/SPEED_STOP으로
        찍었는데, 그게 그대로 이번 틱 drive()에 발행되면서 "주행 중 전환"(예:
        _s1_lane_follow()의 좌회전 확정 → S0_SIGNAL 커밋 구간, _do_checker_ramp_turn()
        완료 → S0_SIGNAL)에서 순항/램프 속도로 달리다가 한 틱만 조향0/급정지 명령이
        나가는 "뚝 끊김" 현상을 만들었다(실차 확인). 실제로 정지가 필요한 전환
        (S4_FINISH, S0_SIGNAL의 "정지 대기" 분기)은 각 state 핸들러(_s4_finish(),
        _s0_signal())가 그 상태로 실행될 때 스스로 0/STOP을 세팅하므로 여기서 미리
        찍어둘 필요가 없다 — 그래서 제거. 이제 전환 직후 첫 틱은 직전 틱의 ctrl_angle/
        ctrl_speed가 그대로 유지된 채 발행되고, 새 state 핸들러가 실행되는 시점부터
        그 값을 이어받거나 필요하면 스스로 정지시킨다.
        """
        self.get_logger().info(f'[전환] {self.mission_state.name} → {new_state.name}')
        self.mission_state = new_state
        self._pid_prev_error = 0.0
        self._pid_integral   = 0.0
        # S0_SIGNAL 진입 시 신호값 초기화 (색상 확정은 이 state 진입 후부터 새로 시작)
        if new_state == MissionState.S0_SIGNAL:
            self.signal_red_on      = False
            self.signal_straight_on = False
            self.signal_left_on     = False
            self.signal_straight_confirmed = False
            self.signal_left_confirmed     = False
            self._sig_straight_cnt = 0
            self._sig_left_cnt     = 0
            self._s2_commit_dist = None
            self._s2_commit_dir  = None
            self._s2_commit_start_t = None
            # [2026-08-24] 방어적 정리 — 지름길 출구 거리추적 중(_shortcut_exit_dist is not
            # None) 뭔가 다른 이유로 S0_SIGNAL로 전환되면(정상 트랙에선 이 구간에 더 볼
            # 신호등이 없어 일어날 일이 없다고 보지만) 그 값을 여기서 지워 다음 바퀴로
            # 새는 걸 막는다 — 출구 램프 자체(_shortcut_exit_ramp_active)는 이 함수 진입
            # 훨씬 전인 _s1_lane_follow() 맨 위에서 이미 최우선으로 가로채므로 여기까지
            # 안 온다.
            if self._shortcut_exit_dist is not None:
                self.get_logger().warn(
                    '지름길 출구 거리추적 중 예상 밖 S0_SIGNAL 전환 — 추적 취소',
                    throttle_duration_sec=1.0)
                self._shortcut_exit_dist = None
        # S1 진입 시 처리
        if new_state == MissionState.S1_LANE_FOLLOW:
            # [2026-08-20] S0_SIGNAL 통합 이후 이 state는 출발 때 1번 + 매 바퀴 교차로에서
            # 반복 진입하므로, prev_state만으로는 "진짜 첫 출발"을 구분할 수 없다(항상
            # S0_SIGNAL이라 같음) — self._departed로 직접 추적한다.
            if not self._departed:
                self._departed = True
                # 신호등 오검출 억제 + 바퀴 기준점 리셋은 진짜 출발 시점에만 필요 —
                # 신호 대기하며 서 있던 시간이 1바퀴 시간에 섞이지 않도록 한다.
                self._signal_reentry_cooldown_t = time.time() + SIGNAL_REENTRY_COOLDOWN
                self._lap_t0 = time.time()
                self._yaw_accum = 0.0
                self._prev_yaw_accum_ref = self.imu_yaw

    # ── S1: 차선인식 주행 (라바콘·고정장애물·추월 Behavior를 이 상태 안에서 처리) ──
    def _s1_lane_follow(self):
        """
        차선을 따라 안정 주행.
          - [2026-08-20] S0_SIGNAL 통합 이후 S1은 매번 "직진 확정" 직후에만 진입하고,
            그때마다 _behavior_enabled=True로 Behavior(B1→B2→B3)가 바로 활성화된다
            (README §대회 규정 요약: 라바콘 등은 출발 직후부터 시작). Behavior가 조향/속도를
            전담하는 구간에서는 여기서 PID를 돌리지 않는다(적분 오염 방지) — 아래 조기 return.
          - [2026-08-22] "신호등 보드가 인식되는 시점"(색상 무관)에 곧장 S0_SIGNAL로
            전환해 그 안에서 멈춰 서서 색상을 판독하던 방식을 없앴다(요청 반영 — 그 정지
            구간 동안 속도가 0으로 굳는 문제). 이제 색상이 실제로 확정
            (signal_straight/left_confirmed, perc_signal() 참고)될 때까지 이 함수(S1
            차선주행)를 그대로 유지한다.
            좌회전(지름길) 확정이면 정지 없이 곧장 S0_SIGNAL로 넘어가 커밋 구간
            (_s2_commit_dist)을 같은 틱에 바로 시작한다 — 아래 _change_state() 직후
            _s2_commit_dist를 세팅하는 순서 참고(먼저 세팅하면 _change_state()가
            S0_SIGNAL 진입 처리에서 그 값을 다시 None으로 지운다). 이 커밋 구간은 라이다
            체크무늬 게이트 검출(checker_pillar_trigger)로 끝나고 그 자리에서 좌회전을
            시작한다(_s0_signal() 참고) — 정지 이벤트 자체가 없다.
            [2026-08-22h] 직진 확정이면(요청 반영) 커밋 구간/상태전환 자체가 없다 — 신호가
            이미 "직진"이라고 확정했으니 물리적 분기까지 저속으로 기다릴 이유가 없고, 그냥
            S1을 유지한 채 Behavior만 재활성화해 다음 바퀴(라바콘부터)를 바로 준비한다.
        """
        # [2026-08-24] 지름길 출구 T자 강제 좌회전 램프 실행 중이면 다른 모든 판단(신호
        # 재확인, B1/B2/B3 가드 등)보다 최우선으로 이걸 돈다 — 위 signal_left_confirmed
        # 처리를 함수 맨 앞으로 옮긴 것과 동일한 이유("최상위 조향, 다른 로직에 안 씹히게").
        if self._shortcut_exit_ramp_active:
            self._do_shortcut_exit_ramp_turn()
            return

        # [2026-08-23q, 요청 반영] 신호 확정(특히 좌회전) 처리를 아래 B1_LAVACON/장애물회피
        # 조기 return보다 먼저 본다 — "최상위 조향, 다른 Behavior에 씹히지 않게". 원래는
        # 이 검사가 함수 맨 끝(_lane_drive() 이후)에 있어서, B1_LAVACON이 조향을 잡고 있거나
        # (_behavior_enabled and behavior_state==B1_LAVACON) 장애물회피/추월 중이면 함수가
        # 그 위 가드에서 곧장 return해버려 signal_left_confirmed가 그 틱에 확정돼도 아예
        # 확인조차 안 되고 넘어갈 수 있었다(다음 틱에 재확인되긴 하지만, 최악의 경우
        # confirmed 래치가 그 사이 풀리면 좌회전 자체를 놓친다). 좌회전이 확정되는 즉시
        # mission_state를 S0_SIGNAL로 바꿔버리면, control_loop()의 Behavior 게이트
        # (mission_state==S1_LANE_FOLLOW 조건)가 그 즉시 꺼져서 이후 B1/B2/B3가 이 조향을
        # 덮어쓸 수 없다.
        # TEST_DISABLE_INTERSECTION=True면 아래 조건이 항상 False가 되어 S0_SIGNAL 재진입
        # 자체가 원천 차단된다.
        if not (TEST_DISABLE_INTERSECTION or time.time() < self._signal_reentry_cooldown_t):
            if self.signal_left_confirmed:
                self._change_state(MissionState.S0_SIGNAL)
                self._s2_commit_dist = 0.0   # _change_state()가 진입 처리에서 None으로 리셋한 뒤 다시 세팅
                self._s2_commit_dir  = 'left'
                self._s2_commit_start_t = time.time()
                return
            elif self.signal_straight_confirmed:
                # [2026-08-22h] 요청 반영 — 커밋 구간 없이 곧장 다음 바퀴 준비만 하고 S1 유지.
                # (_signal_yolo_off는 이제 perc_signal()이 확정되는 순간 바로 세팅한다.)
                self._behavior_enabled = True
                self._signal_reentry_cooldown_t = time.time() + SIGNAL_REENTRY_COOLDOWN
                # [2026-08-24, 요청 반영] TEST_SIGNAL_LOOP 조건 제거 — B1/B2/B3 재무장을
                # _update_lap()(IMU yaw 누적/정지선 기반 바퀴 완주)이 아니라 신호등 직진
                # 확정 하나로 통일한다. 신호등을 다시 받는 것 자체가 "다음 구간 시작"이라는
                # 판단(요청) — 더 이상 테스트 전용이 아니라 정상 레이스 경로에서도 항상 동작.
                # _update_lap()/_begin_new_lap()의 바퀴 카운트(self.lap, 디버그 표시용)는
                # 그대로 유지하되, 이제 그쪽 phase 리셋과 TOTAL_LAPS 기반 종료 판정
                # (S4_FINISH 전환)은 삭제됐다(요청 반영, 아래 _begin_new_lap() 참고) — 재무장은
                # 여기(신호등) 하나로만 결정된다.
                self.phase = Phase.LAVACON
                self._b2_passed = False
                self._b3_passed = False
                self._lavacon_engaged = False
                self._lavacon_empty_cnt = 0
                self._lavacon_trigger_cnt = 0

        # Behavior가 조향을 전담하는 구간에서는 Mission의 차선 PID를 건너뛴다.
        # [2026-08-22] 요청 반영 — run_behavior_fsm()이 다시 behavior_state=B1_LAVACON을
        # 세팅하게 되면서(_lavacon_engaged인 동안) 이 가드가 다시 걸린다. B1 구간 조향은
        # apply_behavior_override()의 _handle_lavacon()(→ _lavacon_steer_da_push())이
        # 전담하고, 여기 아래 else 분기의 일반 차선주행(_lane_drive)은 그 구간엔 안 돈다.
        if self._behavior_enabled and self.behavior_state == BehaviorState.B1_LAVACON:
            return
        if self._obstacle_active or self._overtake_active:
            return

        self._lane_drive()

        # [2026-08-24] 지름길 출구 T자 강제 좌회전 — 입구 램프 완료 시점부터 시작된 거리
        # 추적(self._shortcut_exit_dist, _do_checker_ramp_turn() 참고)을 여기서 누적한다.
        # _s2_commit_dist(_s0_signal())와 동일하게 이번 틱 _lane_drive()가 갱신한
        # self.ctrl_speed를 쓰기 위해 그 호출 "다음"에 둔다 — 먼저 하면 아직 갱신 안 된
        # 직전 틱 속도로 적분된다.
        if self._shortcut_exit_dist is not None:
            if not self._vesc_live():
                # VESC 실측이 죽으면 명령속도 기반 근사(_speed_mps_fallback() 내부 폴백)로
                # 열화된다 — 거리기반 트리거라 조용히 부정확해질 수 있어 실차 로그로
                # 바로 보이게 경고만 남긴다(디바운스: 1초에 한 번).
                self.get_logger().warn(
                    '지름길 출구 거리추적 중 VESC 실측 끊김 — 명령속도 근사로 폴백',
                    throttle_duration_sec=1.0)
            self._shortcut_exit_dist += self._speed_mps_fallback(self.ctrl_speed) * 0.05  # 20Hz 가정
            if self._shortcut_exit_dist >= SHORTCUT_EXIT_DIST_M:
                self._begin_shortcut_exit_ramp()

    # ── S0_SIGNAL: 4구 신호등 판단 — 출발선/교차로 공용 (정지 후 신호로 경로 판단) ──
    def _s0_signal(self):
        """
        4구 신호등 앞에서 정지한 채 판독한다. [2026-08-20] 원래 출발(S0_WAIT_GREEN)과 교차로
        (S2_INTERSECTION)로 나뉘어 있던 걸 하나로 합쳤다 — 둘 다 로직이 완전히 같았기 때문
        (정지 → 4구 신호 판독 → 직진/좌회전 확정).

        [2026-08-22h] 요청 반영 — 직진 확정(signal_straight_confirmed)은 이제 이 state
        자체에 들어오지 않는다(_s1_lane_follow() 참고: 확정되는 순간 S1을 유지한 채
        Behavior만 재활성화하고 끝). 그래서 이 state에 진입하는 경로는 좌회전(지름길)
        확정 하나뿐이다 — 아래 "1. 진입 즉시 정지" 분기도 좌회전 신호만 본다. 그 분기가
        실제로 쓰이는 건 (a) START_STATE=S0_SIGNAL로 노드가 맨 처음 시작하는 출발선
        경우(대회 규정상 심판이 신호를 초록으로 바꾸기 전까지 출발 자체가 금지 — 출발선
        신호가 직진이면 곧장 S1로 넘어가고 이 state에 남지 않는다) 하나뿐이다 — 아직
        신호를 못 읽어 정지 대기 중인 경우. [2026-08-23, 08-23q 되돌림] 좌회전 램프
        완료(_do_checker_ramp_turn())는 이제 여기로 돌아오지 않고 곧장 S1_LANE_FOLLOW로
        복귀한다 — 이 트랙은 좌회전 직후 바로 차선주행이라 읽을 다음 신호등이 없고,
        예전처럼 여기로 돌아오면 확정될 리 없는 신호를 영원히 기다리며 속도 0으로
        굳는 문제가 있었다.
          1. 진입 즉시 정지 (기본값 STOP, 좌회전 신호만 커밋 구간 시작) — 출발선 전용, 위 참고
          2. 좌회전 신호(signal_left_confirmed) → 커밋 구간(_s2_commit_dist) 거쳐 좌회전
             후 S1 복귀(3바퀴 중 2·3바퀴째에 한 번만 등장). 커밋 구간 동안은 S1과 동일한
             차선주행(_lane_drive())을 유지하고, checker_pillar_trigger(체크무늬 게이트
             라이다 기둥쌍 검출)로 커밋 종료를 판정한다 — 거리 기반이 아니라 물리적
             트리거라 신호 확정 지점과 실제 분기 사이에 커브가 있어도 안전하다.
          3. 좌회전 진행 중이면 신호와 무관하게 완료 우선
        """
        if self._checker_ramp_dist is not None:
            self._do_checker_ramp_turn()
            return

        if self._s2_commit_dist is not None:
            self._lane_drive()
            self._s2_commit_dist += self._speed_mps_fallback(self.ctrl_speed) * 0.05  # 20Hz 제어주기(control_loop) 가정

            # [2026-08-24] 커밋 종료 트리거는 체크무늬 게이트 라이다 기둥쌍 검출
            # (config.py "체크무늬 게이트 라이다 기둥쌍 검출" 절 참고). 좌우 기둥쌍이
            # CHECKER_PILLAR_LIDAR_TIMEOUT_SEC초 넘도록 안 잡히면(라이다 죽음/오검출)
            # 더 이상 좌회전을 강제하지 않는다 — 직진 신호를 받았을 때와 완전히 동일한
            # 경로(S1_LANE_FOLLOW 유지 + Behavior 재무장, _s1_lane_follow()의
            # signal_straight_confirmed 분기 참고)로 넘어가 정상 차선주행을 이어간다.
            if self.checker_pillar_trigger:
                self._s2_commit_dist = None
                self._s2_commit_dir  = None
                self._s2_commit_start_t = None
                self._begin_checker_ramp_turn()
                return

            timed_out = (not TEST_DISABLE_CHECKER_PILLAR_TIMEOUT
                         and self._s2_commit_start_t is not None
                         and time.time() - self._s2_commit_start_t >= CHECKER_PILLAR_LIDAR_TIMEOUT_SEC)
            if timed_out:
                self.get_logger().warn(
                    f'좌회전 커밋 {CHECKER_PILLAR_LIDAR_TIMEOUT_SEC:.0f}초 경과했는데 '
                    '체크무늬 게이트 기둥쌍 미검출 — 좌회전 포기, 직진 신호와 동일하게 처리')
                self._s2_commit_dist = None
                self._s2_commit_dir  = None
                self._s2_commit_start_t = None
                self._behavior_enabled = True
                self._signal_reentry_cooldown_t = time.time() + SIGNAL_REENTRY_COOLDOWN
                self._change_state(MissionState.S1_LANE_FOLLOW)
                self.phase = Phase.LAVACON
                self._b2_passed = False
                self._b3_passed = False
                self._lavacon_engaged = False
                self._lavacon_empty_cnt = 0
                self._lavacon_trigger_cnt = 0
            return

        self.ctrl_angle, self.ctrl_speed = 0.0, SPEED_STOP

        if self.signal_straight_confirmed:
            self._behavior_enabled = True
            self._signal_reentry_cooldown_t = time.time() + SIGNAL_REENTRY_COOLDOWN
            self._change_state(MissionState.S1_LANE_FOLLOW)
        elif self.signal_left_confirmed:
            self._s2_commit_dist = 0.0
            self._s2_commit_dir  = 'left'
            self._s2_commit_start_t = time.time()

    # ── S4: 종료 ──
    def _s4_finish(self):
        self.ctrl_angle, self.ctrl_speed = 0.0, SPEED_STOP

    # ── 진입 좌회전 — 체크무늬 게이트 라이다 기둥쌍 트리거 + 완만한 조향 램프 ──
    #   [2026-08-21] _begin_left_turn()/_do_left_turn(next_state=S3_SHORTCUT)을 대체(요청
    #   반영) — S2_COMMIT_DIST_M 거리기반 대신 perc_checker_pillar()의 라이다 기둥쌍
    #   검출로 커밋 종료를 판단하고(_s0_signal() 'left' 분기 참고), 고정각 즉시 진입 대신
    #   CHECKER_TURN_RAMP_START_ANGLE→END_ANGLE로 서서히 조향을 올린다.
    #   [2026-08-22g] 램프 완료 후 전환할 곳을 MissionState.S3_SHORTCUT(지름길 전용 직진+
    #   정지선 감지+탈출 좌회전 스크립트)에서 MissionState.S1_LANE_FOLLOW로 바꿨다(요청
    #   반영) — 지름길 구간도 물리적으로는 그냥 트랙의 일부라 일반 차선주행과 다를 게
    #   없고(끝에 신호등 없는 갈림길도 아니고, B3 방해차량 회피를 마친 뒤 다음 교차로
    #   신호등을 기다리며 차선주행하는 상태와 동일), 별도 탈출 스크립트가 필요하다는 전제
    #   자체가 더 이상 유효하지 않다는 판단(실차 트랙 재확인). S3_SHORTCUT
    #   전용이던 _s3_shortcut()/_shortcut_end()/_begin_left_turn()/_do_left_turn()과
    #   MissionState.S3_SHORTCUT enum 값 자체를 삭제했다(README §1.19b 참고) — 되돌릴
    #   경우 git 이력에서 복원할 것.
    #   [2026-08-23q] 좌회전 직후 Phase가 아직 LAVACON(이번 바퀴 B1을 아직 안 거친 상태)로
    #   남아있으면 Behavior가 곧장 켜져 B1이 검증 없이 바로 발동하는 문제가 실차에서
    #   드러나, 램프 완료 후 곧장 S0_SIGNAL(신호 대기)로 돌아가 "다음 직진 신호"가 확정될
    #   때까지 정지 대기하도록 바꿨었다. 그런데 이 트랙은 좌회전(지름길) 직후 곧바로
    #   차선주행 구간으로 이어지고 그 사이엔 읽을 신호등 자체가 없어서, 확정될 리 없는
    #   신호를 영원히 기다리며 속도 0으로 완전히 굳어버리는 문제로 이어졌다(실차 확인,
    #   2026-08-23). S1_LANE_FOLLOW+Behavior 즉시 재개로 되돌린다 — 위 08-23q가 우려한
    #   Phase.LAVACON 상태의 B1 오발동은, 그 사이 유령 라이다 점 마스킹(perc_lavacon.py
    #   GHOST_POINT_*, 2026-08-22 작업기록 참고)으로 완화됐을 가능성이 있어 실차에서
    #   재확인할 것 — 다시 문제가 보이면 이 되돌림 대신 좌회전 직후에만 짧게 B1 감지를
    #   유예하는 방향으로 갈 것.
    #   [2026-08-24] 요청 반영 — 위에서 우려했던 "짧게 B1 감지를 유예"를 실제로 적용했다.
    #   단, S0_SIGNAL(정지 대기)로 돌아가진 않는다(그건 08-23q가 겪은 속도0 고정 재현) —
    #   S1_LANE_FOLLOW+B0로 계속 달리면서 다음 신호등 직진 확정을 기다리고, 확정되는
    #   순간에만 phase=LAVACON(B1)이 열린다(_s1_lane_follow() 상단 분기). 좌회전 직후
    #   구간에 실제로 읽을 신호등이 없으면 이 바퀴 B1이 영원히 안 열린다 — 트랙에 신호등이
    #   있는지 실차로 재확인할 것.
    def _begin_checker_ramp_turn(self):
        self._checker_ramp_dist = 0.0
        self.get_logger().info('체크무늬 게이트 통과 — 좌회전 램프 시작')

    def _do_checker_ramp_turn(self):
        angle, done = self._checker_turn_ramp_angle(TURN_SPEED)
        self.ctrl_angle = angle
        self.ctrl_speed = TURN_SPEED
        if done:
            self._checker_ramp_dist = None
            self._left_turn_last_done_t = time.time()  # left_turn_debug 창 "실행끝" 표시용
            self._signal_yolo_off = False
            self._signal_off_hold_cnt = None
            self._signal_reentry_cooldown_t = time.time() + SIGNAL_REENTRY_COOLDOWN
            self.get_logger().info('체크무늬 게이트 좌회전 램프 완료 — 곧장 B1 검출로 진입')
            self._behavior_enabled = True
            self.phase = Phase.LAVACON
            self._b2_passed = False
            self._b3_passed = False
            self._lavacon_engaged = False
            self._lavacon_empty_cnt = 0
            self._lavacon_trigger_cnt = 0
            self._change_state(MissionState.S1_LANE_FOLLOW)
            # [2026-08-24] 지름길 출구 T자 강제 좌회전(config.py SHORTCUT_EXIT_DIST_M 참고) —
            #   이 램프(입구)가 끝나는 시점부터 출구까지 거리를 재기 시작한다. 위에서 이미
            #   Phase/B1~B3를 리셋해뒀지만, 출구 램프 자체는 그걸 다시 건드리지 않는다
            #   (_do_shortcut_exit_ramp_turn() 참고) — 지름길을 빠져나오면 B1/B2/B3 없이
            #   바로 결승선으로 이어지는 트랙이라(사용자 확인, 2026-08-24) 출구 쪽에서
            #   Phase를 또 리셋할 이유가 없다.
            self._shortcut_exit_dist = 0.0

    # ── 지름길 출구 T자 강제 좌회전 (2026-08-24, config.py SHORTCUT_EXIT_DIST_M 주석 참고) ──
    #   T자 교차로엔 라이다 랜드마크도 정지선도 없어(사용자 실측 확인) 물리 트리거를 못 쓰고,
    #   da도 반시계방향 규칙을 모르니 T자에서 방향을 못 정한다 — 그래서 입구 램프 완료
    #   시점부터 잰 거리(self._shortcut_exit_dist, _s1_lane_follow() 참고)로 강제 트리거한다.
    #   조향 계산 자체는 입구와 완전히 대칭이라 _checker_turn_ramp_angle()을 그대로 재사용
    #   (각도/거리/곡선 재튜닝 불필요) — 다만 완료 처리는 입구(_do_checker_ramp_turn())와
    #   분리된 별도 함수다: 입구는 Phase를 LAVACON으로 리셋해 다음 구간의 B1을 다시 열지만,
    #   출구 뒤엔 더 볼 Behavior가 없으므로(위 참고) Phase/B1~B3엔 전혀 손대지 않는다 —
    #   여기서 리셋해버리면 이미 지나온 B1~B3 완료 기록이 지워지거나, 결승선 직전 구간에서
    #   불필요하게 B1 콘 검출(YOLO 추론)이 다시 켜지는 부작용이 생긴다.
    def _begin_shortcut_exit_ramp(self):
        self._shortcut_exit_dist = None
        self._shortcut_exit_ramp_active = True
        self._checker_ramp_dist = 0.0  # 입구와 같은 누적변수 재사용 — 두 램프가 동시에 도는 경우가 없어 안전
        self.get_logger().info(
            f'지름길 출구 거리 도달({SHORTCUT_EXIT_DIST_M:.1f}m) — 강제 좌회전 램프 시작')

    def _do_shortcut_exit_ramp_turn(self):
        angle, done = self._checker_turn_ramp_angle(TURN_SPEED)
        self.ctrl_angle = angle
        self.ctrl_speed = TURN_SPEED
        if done:
            self._checker_ramp_dist = None
            self._shortcut_exit_ramp_active = False
            self.get_logger().info('지름길 출구 좌회전 램프 완료 — 일반 주행 복귀(Phase 변경 없음)')

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
            # [2026-08-20] 요청 반영 — B1은 진입/탈출 두 트리거로 latch를 관리한다.
            # 진입: 좌우 라이다 클러스터 동시검출(lavacon_trigger)로 self._lavacon_engaged를
            # latch. 탈출: process_lavacon()이 매 틱 계산해두는 lavacon_done(우측 콘
            # 연속 미검출)이 LAVACON_DONE_FRAMES만큼 유지되면 확정 — 이 판정은 exit 시점을
            # 여기 한 곳에서만 확정하고 _handle_lavacon()에서는 중복으로 하지 않는다
            # (그 함수 docstring 참고).
            # [2026-08-22] 요청 반영 — behavior_state를 B0_NORMAL로 묶어두던 걸 되돌렸다.
            # _lavacon_engaged인 동안은 B1_LAVACON으로 세팅해 apply_behavior_override()가
            # 실제로 _handle_lavacon()(→ _lavacon_steer_da_push(), 콘 침범 시 옆으로 밀기)을
            # 호출하게 한다 — 그 전까진 검출/진입판정만 돌고 조향은 그냥 일반 S1 차선 PID가
            # 대신하고 있었다(콘 전용 안전마진 없이 DA가 우연히 피해가는 수준). 그 PID 스킵
            # 가드(behavior_state==B1_LAVACON 기준, _s1_lane_follow() 참고)가 이제 다시 걸린다.
            was_engaged = self._lavacon_engaged
            if self.lavacon_trigger:
                self._lavacon_engaged = True
            if LAVACON_KICK_ENABLED and (not was_engaged) and self._lavacon_engaged:
                # [2026-08-23, 요청 반영] 진입 확정 상승엣지 — 여기서 딱 한 번만 걸린다
                # (_lavacon_engaged가 True로 유지되는 동안엔 다시 안 걸림). 프레임수는
                # 20Hz 고정주기(control_loop() 타이머, 0.05초) 기준 환산 — 이 값을 바꾸려면
                # LAVACON_KICK_DURATION_S(config.py)만 조정하면 된다.
                self._lavacon_kick_cnt = int(round(LAVACON_KICK_DURATION_S / 0.05))
            if self._lavacon_engaged:
                if self.lavacon_done:
                    self._lavacon_empty_cnt += 1
                    if self._lavacon_empty_cnt >= LAVACON_DONE_FRAMES:
                        self._lavacon_empty_cnt = 0
                        self._lavacon_engaged = False
                        self.phase = Phase.OBSTACLE_ZONE
                        self.get_logger().info(
                            '[LAVACON] 탈출 트리거 확정 → 장애물 구간')
                else:
                    self._lavacon_empty_cnt = 0
            if self._lavacon_engaged:
                self.behavior_state = BehaviorState.B1_LAVACON
            else:
                self.behavior_state = BehaviorState.B0_NORMAL
                self._lavacon_push_active = False  # 진입 전/탈출 직후엔 항상 꺼둔다
                self._lavacon_push_px = 0.0
        elif self.phase == Phase.OBSTACLE_ZONE:
            # TEST_DISABLE_B2_B3=True면 트리거 검사를 아예 안 하고 바로 리턴 — 장애물/방해차량이
            # 실제로 잡혀도 통과 처리가 안 되어 Phase가 영원히 OBSTACLE_ZONE에 머문다(placeholder
            # 회피 자체는 ENABLE_OBSTACLE_CUT과 별개 스위치라 계속 돌 수 있음에 주의).
            if TEST_DISABLE_B2_B3:
                self.behavior_state = BehaviorState.B0_NORMAL
                return

            # [2026-08-20] 요청 반영 — B2/B3도 B1(라바콘, 위 Phase.LAVACON 분기)과 같은
            # 패턴으로 단순화했다. 실제 회피 조향/감속은 behavior_state와 무관하게 상시로
            # 도는 da 근접 컷(ENABLE_OBSTACLE_CUT — perc_obstacle_cut_trigger()의 라이다+
            # YOLO(콘 또는 차량) 이중확인 → obstacle_cut_active → _clip_da_by_obstacle()의
            # da 클리핑 + SPEED_PRE_OBSTACLE_CAP 속도캡(감지 전 구간, _update_speed() 참고),
            # perception/dl_lane.py)이 이미 처리하므로,
            # 여기서는 obstacle_cut_active의 진입~탈출을 그대로 B2/B3 신호로 재사용해서
            # Phase 완료 처리(_mark_behavior_passed)만 담당한다 — B1의 lavacon_trigger/
            # lavacon_done latch와 완전히 같은 구조. TargetPassing 기반
            # _handle_fixed_obstacle()/_handle_overtake()는 더 이상 호출되지 않는다(구현은
            # 나중을 위해 보존, 아래 함수 docstring 참고).
            if self.obstacle_cut_active and self._obscut_zone_tag is None:
                # obstacle_cut_type: perc_obstacle_cut_trigger()가 YOLO 콘/차량 이중확인으로
                # 매 틱 갱신해두는 값을 트리거 활성화 순간 스냅샷 — 'fixed'(고정장애물=B2) 아니면
                # 'vehicle'(방해차량=B3). 트랙 순서는 "B2 먼저"(§5.2, §5.4에서 잠깐 반대로
                # 뒤집혔다가 §5.5로 원복) — _b2_passed가 아직 False면 타입이 vehicle로 잡혀도
                # B2로 취급한다 — 초반 오분류로 B3를 먼저 통과 처리해버리는 사고를 원천 차단.
                if self.obstacle_cut_type == 'vehicle' and self._b2_passed:
                    self._obscut_zone_tag = 'B3'
                else:
                    self._obscut_zone_tag = 'B2'

            if self._obscut_zone_tag is not None and not self.obstacle_cut_active:
                self._mark_behavior_passed(self._obscut_zone_tag)
                self._obscut_zone_tag = None

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
        참고.

        [2026-08-19] 반경 역산에 쓰는 축거리를 self.pure_pursuit.wheelbase_px(PP_WHEELBASE_PX,
        조향 게인 튜닝값)에서 config.CORNER_RADIUS_WHEELBASE_PX(물리 기반 고정값, 67.0)로
        분리했다(요청 반영) — PP_WHEELBASE_PX를 조향 반응성 목적으로 낮출 때마다 이 반경
        계산도 같이 작아져서, 살짝만 꺾여도 코너 감속이 상시로 걸리는 부작용이 있었다
        (config.py CORNER_RADIUS_WHEELBASE_PX 주석 참고)."""
        curvature = math.tan(math.radians(self._corner_signal)) / CORNER_RADIUS_WHEELBASE_PX
        if curvature == 0.0:
            return 1.0
        radius = abs(1.0 / curvature)
        if radius >= CORNER_MIN_RADIUS_PX:
            return 1.0
        return max(CORNER_MIN_SPEED_SCALE, radius / CORNER_MIN_RADIUS_PX)

    def _imu_corner_confirm_scale(self):
        """turn_now/turn_preview(비전+조향출력 신호)가 코너라고 판단해도, IMU 실측
        회전량(self.pure_pursuit.last_imu_curvature_px)이 이를 뒷받침하지 않으면 코너감속을
        절반 이하로 깎는다(config.py CORNER_IMU_CONFIRM_KAPPA_PX/CORNER_IMU_MIN_SCALE 주석 —
        2023 KMU AuTURBO rookie 팀의 ModeController가 IMU yaw 변화량으로 "진짜 커브"를
        확인하던 패턴 참고). last_imu_curvature_px는 _lane_steer()가 이번 틱에 이미
        갱신해뒀다(_imu_curvature_px() 호출 순서 참고) — 여기서 다시 계산하지 않는다.
        None이면(IMU/VESC 죽었거나 dl+BEV 조합이 아니면) 기존처럼 비전 신호만 믿도록
        1.0(무감쇠)을 반환한다. 실차 미검증 첫 추정치."""
        imu_kappa = getattr(self.pure_pursuit, 'last_imu_curvature_px', None)
        if imu_kappa is None:
            return 1.0
        return max(CORNER_IMU_MIN_SCALE, min(1.0, abs(imu_kappa) / CORNER_IMU_CONFIRM_KAPPA_PX))

    def _lane_drive(self):
        """S1/S3 공통 차선 조향+감속 로직. ctrl_angle·ctrl_speed·_prev_speed·_corner_hold 갱신.

        [2026-08-21] §3.4가 남긴 "알려진 한계"(라바콘 구간을 회피조향 없이 일반 차선 PID로만
        지나가서, 콘이 차선 폭 안쪽까지 침범하면 충돌 위험) 대응 — `self._lavacon_engaged`
        (perc_lavacon_trigger()의 라이다 AND YOLO 이중확인으로 이미 확정된 "지금 라바콘
        구간 안" latch, run_behavior_fsm()의 Phase.LAVACON 분기가 계속 관리)가 True인 동안만
        `_lavacon_steer_da_push()`(da 경로 + 콘 침범 시 옆으로 밀기)로 바꿔 쓴다.
        `behavior_state`는 여전히 B0_NORMAL로 유지되므로(§3.4 결정 유지, B2/B3 단독 검증에
        영향 없음) 이건 `_handle_lavacon()`을 되살리는 게 아니라, 상시로 도는 안전 보정
        하나를 여기 얹는 것뿐이다(§4.3 da 근접 컷이 behavior_state와 무관하게 상시로 도는
        것과 같은 패턴).

        [2026-08-22] 속도 계산(코너 감속/가속 램프)은 `_update_speed()`로 분리했다 — B1
        (`_handle_lavacon()`)도 조향만 라바콘 전용으로 바꿔 쓰고 속도는 이 S1 로직을
        "똑같이" 그대로 타게 하려는 목적(요청 반영, speed15 프리셋 기준 SPEED_NORMAL=12/
        SPEED_CORNER_MIN=10로 자연히 그 사이에서 움직임). behavior_state==B1_LAVACON인
        동안은 `_s1_lane_follow()`가 이 함수 자체를 안 부르므로(가드 참고), 이 함수는
        여전히 S1(+커밋 구간의 S0_SIGNAL)에서만 실행된다."""
        if self._lavacon_engaged:
            self.ctrl_angle = self._lavacon_steer_da_push()
        else:
            self.ctrl_angle = self._lane_steer()
        self._update_speed()

    def _update_speed(self):
        """`_lane_drive()`(S1)와 `_handle_lavacon()`(B1)이 공유하는 속도 계산 — 코너 감속/
        가속 램프 로직 자체는 조향 소스(일반 차선 PID든 라바콘 da-push든)와 무관하게
        self.ctrl_angle 하나만 보고 돌아간다(아래 self._corner_signal 갱신 참고). ctrl_speed·
        _prev_speed·_corner_hold 갱신. [2026-08-22] _lane_drive()에서 분리 — B1 진입 시
        속도가 SPEED_LAVACON(2.5 고정)으로 "굳던" 문제(이전 수정으로 일단 "진입 시점 속도
        유지"로 바꿨었음) 대신, S1과 동일한 동적 속도 로직을 B1에도 그대로 적용하기 위함
        (요청 반영)."""
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
        # [2026-08-18] IMU 실측 회전량으로 "비전이 본 코너가 진짜인가" 교차검증
        # (_imu_corner_confirm_scale() 주석 참고) — turn_now/turn_preview가 비전 잡음만으로
        # 감속을 거는 걸 막는다.
        imu_corner_scale = self._imu_corner_confirm_scale()
        turn_for_speed = max(turn_now, turn_preview * 0.3) * imu_corner_scale
        # [2026-08-23] 리터럴 0.80 → SPEED_CORNER_STEER_GAIN(config.py, 요청 반영) — 프리셋별로
        #   다르게 가져갈 수 있게 이름 있는 상수로 분리. 이력은 config.py 해당 상수 주석 참고
        #   (0.90 → 0.80, 조향 기반 감속이 너무 강하다는 피드백으로 완화, 실차 재검증 필요).
        target_speed = max(SPEED_CORNER_MIN,
                           SPEED_NORMAL * (1.0 - SPEED_CORNER_STEER_GAIN * turn_for_speed ** 3))
        # 코너 진입(회전반경 감소) 시 추가 감속 — 기존 turn_for_speed 기반 감속과는 독립적으로
        # 계산해서 더 낮은 쪽을 쓴다(대체가 아니라 추가 안전판).
        corner_radius_scale = self._corner_radius_speed_scale()
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
        # [2026-08-19] 근접 밴드 hold 타임아웃(config.py DL_NEAR_HOLD_MAX_FRAMES,
        # perception/dl_lane.py detect() 참고) — 근접 밴드가 그 프레임 수 넘게 안 잡혀
        # hold도 포기한 상태면, "지금 차량 바로 앞 정보를 오래 못 믿고 있다"는 뜻이라
        # lane_stale/lane_unstable과 동일하게 감속 신호로 드러낸다. hough/classic_cv
        # 백엔드엔 이 속성이 없으니 getattr로 조회(다른 DL 전용 속성과 동일 관례).
        if getattr(self.lane_detector, 'near_band_stale', False):
            target_speed = min(target_speed, SPEED_LANE_STALE)
        # [2026-08-23, 요청 반영 후 취소] 라바콘 탈출 후(Phase.OBSTACLE_ZONE) 장애물 미감지
        # 구간을 SPEED_PRE_OBSTACLE_CAP(8.0)으로 캡하던 로직을 없앴다 — 요청 반영, 속도 8
        # 유지는 B1(라바콘 주행) 중에만 적용하고 그 이후(장애물 구간)는 감지 여부와 무관하게
        # 곧장 일반 코너감속 로직(SPEED_NORMAL 기반)을 탄다. SPEED_PRE_OBSTACLE_CAP(config.py)
        # 상수 자체는 남겨뒀지만 이제 아무 데서도 참조하지 않는다.
        # [2026-08-23, 요청 반영] B1(Phase.LAVACON) 중엔 목표속도 상한을 SPEED_LAVACON_CAP으로
        # 추가 제한 — 과거 SPEED_LAVACON(2.5 고정, 삭제됨)처럼 매 틱 정확한 값으로 강제
        # 고정하는 게 아니라, 위 코너감속 등과 동일하게 target_speed에 min()으로만 얹는다.
        # 아래 accel_step 램프가 그대로 적용되므로 급감속/굳는 증상 없이 부드럽게 이 상한까지
        # 내려간다(위 SPEED_PRE_OBSTACLE_CAP과 동일 패턴, config.py SPEED_LAVACON_CAP 주석 참고).
        # [2026-08-24, 요청 반영] `and self.cone_detected_yolo` 제거 — 그 값은 YOLO 원시
        # 프레임 검출(perc_yolo_cone(), 박스 1개 이상)이라 B1 구간 안에서도 그 틱에 카메라가
        # 콘을 놓치면 캡이 안 걸려 SPEED_NORMAL(=12) 쪽으로 튀는 게 실차에서 확인됨(요청).
        # [2026-08-24b, 요청 반영] `self.phase == Phase.LAVACON` 단독 조건 버그 수정 —
        # Phase.LAVACON은 B1 진입 전(아직 콘 게이트를 못 만나 그냥 S1 차선주행 중인)
        # 대기 구간도 포함한다(_active_yolo_stage()의 Phase.LAVACON 분기 참고, 'cone' 스테이지
        # if not self._lavacon_engaged). 그래서 위 수정 직후엔 신호 확정→S1 진입 직후부터
        # 실제 콘 게이트를 만나기 전까지도 계속 8로 캡이 걸려 "라바콘 진입 전인데 속도가
        # 8로 굳어있다"는 증상으로 나타났다(실차 확인). B1 진입 확정 latch
        # self._lavacon_engaged(실제로 콘 사이를 통과 주행 중인지)로 조건을 좁혀 진짜 B1
        # 구간에서만 캡이 걸리게 한다.
        if self.phase == Phase.LAVACON and self._lavacon_engaged:
            target_speed = min(target_speed, SPEED_LAVACON_CAP)
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

    def _speed_mps_fallback(self, cmd_speed):
        """거리 적분(_s2_commit_dist/_checker_ramp_dist)에 쓸 현재 속도(m/s) 추정.
        VESC 실측이 살아있으면(_vesc_live()) 그 값을 그대로 쓴다 — 대회 당일 실속도가
        얼마든 실제 이동거리 기준으로 맞는다. VESC가 죽어있을 때만 그 구간에서 명령
        중인 속도(cmd_speed, 모터단위)를 METERS_PER_SPEED_UNIT으로 환산해 폴백한다 —
        예전 시간 기반(S2_COMMIT_T)이 암묵적으로 가정하던 것과 동일한 근사치라 VESC
        장애 시에도 이전과 같은 동작으로 안전하게 열화되고, 거리 적분 자체가 멈춰
        무한정 그 구간에 머무는(예: 좌회전 조향을 계속 유지하는) 상황이 없다.
        [2026-08-20] S2_COMMIT_DIST_M 전용이던 _commit_speed_mps()를 일반화 — 체크무늬
        게이트 램프(_checker_turn_ramp_angle())의 거리 기반 종료 판정에도 동일 철학으로
        재사용한다."""
        if self._vesc_live():
            return abs(self.v_mps)
        return cmd_speed * METERS_PER_SPEED_UNIT

    def _lane_steer(self, path=None, vehicle_x=None, vehicle_y_px=None):
        """path(ROI 픽셀좌표 경로, 가까운점→먼점)를 pure_pursuit(controller/pure_pursuit.py)로
        추종해 조향각(도)을 계산한다. 차량 기준점은 (vehicle_x, vehicle_y_px)로 둔다
        (vehicle_y_px를 안 주면 path[0]의 y좌표를 그대로 쓴다 — 기존 동작).

        인자를 생략하면(기본 호출부인 _lane_drive() 등) 기존과 동일하게 self.lane_path와
        ROI 하단 중앙(roi_w/2)을 쓴다 — path[0].y는 lane_util._fit_and_sample_path()가
        self.roi_h로 샘플링해둔 값이라 별도로 백엔드별 roi_h를 조회할 필요가 없다.
        [2026-08-11] _handle_lavacon()이 self.lavacon_path/vehicle_x=0.0을 명시적으로
        넘겨 호출한다 — 라바콘 조향 파라미터를 라인주행과 완전히 일치시키기 위해, 별도
        게인을 두지 않고 이 함수를 그대로 재사용하기로 한 결정(perc_lavacon() 주석 참고).

        [2026-08-19] vehicle_y_px — path가 "센서 원점 기준" 좌표계일 때(라바콘의 경우
        라이다 원점, row 0 = 라이다 자체 위치), path[0]의 행(row)을 차량 위치로 대신
        쓰는 건 "차량이 대략 첫 웨이포인트 부근에 있다"는 근사일 뿐 실제 차량 위치가
        아니다 — 실측으로 알아낸 진짜 차량 기준점 행이 있으면(예: 라이다가 차량 맨
        앞부분보다 LIDAR_TO_VEHICLE_FRONT_M만큼 앞에 있다는 게 확인됨, config.py 참고)
        path[0]와 무관하게 그 절대값을 직접 넘긴다 — path[0]에 "더하는" 식으로 쓰면 안 됨
        (path[0]의 행 자체가 BOX_LON_START 등 다른 값에 좌우되는 임의 기준이라, 더하면
        엉뚱한 위치가 나온다). None(기본값)이면 기존과 완전히 동일하게 path[0][1]을 쓴다
        (라인주행 등 다른 호출부는 전부 이 값을 안 넘겨서 영향 없음). _handle_lavacon()이
        LIDAR_TO_VEHICLE_FRONT_M*DL_PIXELS_PER_METER(라이다 원점 기준 차량의 절대 행)를
        직접 계산해 넘긴다.

        경로가 비어있으면(첫 프레임, 혹은 roi_w를 아직 모르는 백엔드) 직전 조향각을
        그대로 유지한다 — pure_pursuit.control()이 내부적으로 이렇게 처리한다.
        [2026-08-14] STEERING_CONTROLLER로 pure_pursuit/lqr 중 고르던 분기를 LQR 컨트롤러
        제거와 함께 없앴다 — 이제 pure_pursuit 고정."""
        # [2026-08-19] 근접 장애물 급회피 대응 — path 인자를 명시로 넘기는 호출부
        # (_handle_lavacon() 등)는 아래 if path is None 분기를 안 타므로 near_obstacle이
        # 항상 False로 남는다 — 라바콘 등 다른 주행모드는 이 변경과 구조적으로 무관하다.
        near_obstacle = False
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
            # obstacle_front/obstacle_dist는 TEST_DISABLE_B2_B3와 무관하게 perc_obstacle()가
            # 매 틱 갱신하므로 별도 stale 가드 없이 바로 써도 안전하다(_update_avoid_hold()와
            # 동일 근거). AVOID_HOLD_TRIGGER_DIST_M(기존 상수, 1.5m) 재사용 — 별도 상수
            # 안 늘림(요청 반영). pure_pursuit.py _target_point_max_deviation() 참고.
            # [2026-08-22] 요청 반영 — B1(Phase.LAVACON) 중엔 이 근접회피도 죽인다.
            # perc_obstacle()의 전방 ROI(5m×1.5m 반폭)가 라바콘 자체를 "고정장애물"로
            # 잡아 obstacle_front/dist가 갱신되면, B1 진입 트리거(좌우 라이다 클러스터
            # 동시검출)가 아직 안 걸린 대기 구간에서도 목표점 선택이 _target_point_max_deviation()
            # 으로 바뀌어 콘 구간 진입 전부터 급조향(=B2 회피처럼 보이는 거동)이 나가는
            # 문제가 실차에서 확인됨 — perc_obstacle_cut_trigger()/_update_obstacle_cut_hold()에
            # 이미 걸려있는 것과 동일한 Phase.LAVACON 가드를 여기도 추가한다. Phase.OBSTACLE_ZONE
            # 진입(=B1 완료) 후에만 실제로 발동한다.
            near_obstacle = (self.phase != Phase.LAVACON
                              and self.obstacle_front and self.obstacle_dist < AVOID_HOLD_TRIGGER_DIST_M)
        if not path or vehicle_x is None:
            return self.pure_pursuit.prev_steer_deg
        vehicle_xy = (vehicle_x, path[0][1] if vehicle_y_px is None else vehicle_y_px)
        # [2026-08-21] obstacle_cut 진입 후 PP_CURVATURE_BOOST_SEC 동안만
        # lookahead_curvature_gain을 PP_CURVATURE_BOOST_GAIN으로 올린다 — pure_pursuit
        # 인스턴스가 하나뿐이라(__init__ 참고) 매틱 여기서 스위칭해도 다른 호출부
        # (_handle_lavacon() 등, near_obstacle=False 경로)에도 그대로 적용된다.
        # _update_obstacle_cut_hold()가 타이머만 세팅, 실제 반영은 여기서.
        now_t = time.time()
        vehicle_lookahead_fix = now_t < self._pp_vehicle_lookahead_fix_until_t
        # lookahead를 고정하는 동안은 curvature 부스트를 겹쳐 걸지 않고 평상시 게인으로 둔다(요청 반영).
        self.pure_pursuit.lookahead_curvature_gain = (
            PP_LOOKAHEAD_CURVATURE_GAIN if vehicle_lookahead_fix
            else PP_CURVATURE_BOOST_GAIN if now_t < self._pp_curvature_boost_until_t
            else PP_LOOKAHEAD_CURVATURE_GAIN)
        # [2026-08-24] B1(라바콘) 중엔 lookahead 상한을 PP_LOOKAHEAD_MAX_PX_LAVACON으로
        # 낮춘다 — SPEED_LAVACON_CAP 조건(2026-08-24b)과 동일하게 self._lavacon_engaged로
        # 좁혀 진짜 B1 구간에서만 적용(대기 구간은 일반 상한 유지).
        self.pure_pursuit.lookahead_max_px = (
            PP_LOOKAHEAD_MAX_PX_LAVACON
            if self.phase == Phase.LAVACON and self._lavacon_engaged
            else PP_LOOKAHEAD_MAX_PX)
        return self.pure_pursuit.control(
            path, vehicle_xy, speed=self._speed_for_lookahead(),
            imu_curvature_px=self._imu_curvature_px(), near_obstacle=near_obstacle,
            lookahead_override_px=PP_VEHICLE_LOOKAHEAD_FIX_PX if vehicle_lookahead_fix else None)

    # [2026-08-24] B1(라바콘) 전용 조향 — self.pure_pursuit_lavacon(§ __init__, config.py
    #   _LAVACON 상수)만 쓴다. 위 _lane_steer()와 달리 근접 장애물 회피(near_obstacle),
    #   차량감지 lookahead 고정(vehicle_lookahead_fix), obstacle_cut curvature 부스트처럼
    #   B2/B3 상태에 좌우되는 "일반상태" 로직을 전혀 안 거친다 — 라바콘 조향이 그런 다른
    #   구간의 타이머/상태에 영향받지 않고, 오직 이 함수에 넘어온 path/vehicle_xy와
    #   config.py의 고정된 _LAVACON 게인만으로 결정되게 하려는 목적(요청 반영). speed/IMU
    #   curvature는 실측 라이브 상태라 그대로 공유한다(_update_speed()가 S1과 동일 로직을
    #   쓰는 것과 같은 이유 — 튜닝 상수만 분리, 실측값까지 얼릴 이유는 없음).
    def _lavacon_pure_pursuit_steer(self, path, vehicle_x, vehicle_y_px=None):
        if not path or vehicle_x is None:
            return self.pure_pursuit_lavacon.prev_steer_deg
        vehicle_xy = (vehicle_x, path[0][1] if vehicle_y_px is None else vehicle_y_px)
        return self.pure_pursuit_lavacon.control(
            path, vehicle_xy, speed=self._speed_for_lookahead(),
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

        # [2026-08-23] 조향각(아래) = Pure Pursuit 컨트롤러 자체의 출력, behavior override
        # (B1 라바콘 등)나 drive()의 ANGLE_MAX/ANGLE_RATE_MAX 클립을 아직 안 거친 값이라
        # 실제 발행값과 다를 수 있다. 이 창의 호출 시점이 drive() 이후로 옮겨졌으니(호출부
        # control_loop() 참고) self._prev_angle_out(drive()가 이번 틱에 마지막으로 클립해
        # 발행한 각도)을 "발행조향"으로 같이 보여줘서 둘이 다를 때 바로 보이게 한다.
        override_active = self.behavior_state != BehaviorState.B0_NORMAL
        publish_color = (0, 140, 255) if override_active else (255, 255, 255)
        publish_suffix = f' [override:{self.behavior_state.name}]' if override_active else ''
        lines = [
            ('컨트롤러: Pure Pursuit', (10, 8), (255, 255, 255), 20, 'Controller: Pure Pursuit'),
            (f'상태: {status_text}', (10, 40), status_color, 20,
             f'Status: {"HOLD (prev)" if held else "LIVE (fresh)"}'),
            (f'PP조향각: {controller.prev_steer_deg:+.1f}도', (10, 72), (255, 255, 255), 20,
             f'PP steer: {controller.prev_steer_deg:+.1f}deg'),
            (f'발행조향: {self._prev_angle_out:+.1f}도{publish_suffix}', (10, 104), publish_color, 20,
             f'Published: {self._prev_angle_out:+.1f}deg{publish_suffix}'),
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
        canvas = np.full((190, 380, 3), 30, dtype=np.uint8)
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
        # [2026-08-20] 좌회전(_do_checker_ramp_turn())의 진행상황 표시를 IMU 창
        # (_debug_viz_imu(), 예전 yaw 기반)에서 여기로 옮겼다 — 좌회전 진행 판정이 VESC
        # 적분(_checker_ramp_dist)이라 이 창이 더 맞는 위치. [2026-08-22g] S3_SHORTCUT
        # 진출 좌회전(_do_left_turn(), 옛 _turn_dist 기반)은 S3 자체를 삭제하며 같이
        # 없어져 이제 진입 램프 하나만 표시한다.
        if self._checker_ramp_dist is not None:
            lines.append((
                f'체크무늬 램프 중: {self._checker_ramp_dist:.2f}m / {CHECKER_TURN_RAMP_DIST_M:.2f}m '
                f'(angle={self.ctrl_angle:.1f}°)',
                (10, 140), (0, 255, 255), 14,
                f'CHECKER RAMP: {self._checker_ramp_dist:.2f} / {CHECKER_TURN_RAMP_DIST_M:.2f} m '
                f'(angle={self.ctrl_angle:.1f} deg)'))
        else:
            lines.append(('좌회전 진행 중 아님', (10, 140), (150, 150, 150), 14, 'not turning'))
        put_text_kr_multi(canvas, lines)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), color, 3)

        cv2.imshow('vesc_debug', canvas)
        cv2.waitKey(1)

    # [DEBUG_VIZ_IMU] IMU(/imu) 연동이 실제로 살아있는지 + 지금 imu_yaw 값이 얼마인지를
    # 한눈에 보여주는 창(2026-08-18, _debug_viz_vesc()와 동일 패턴). [2026-08-20] 좌회전
    # 진행상황 표시는 _debug_viz_vesc()로 옮겼다 — 좌회전(체크무늬 게이트 램프) 자체는
    # IMU를 참조하지 않는다. 이 창은 여전히 바퀴카운트(_update_lap()의 _yaw_accum) 등
    # imu_yaw를 쓰는 다른 로직 확인용으로 남긴다. control_loop()에서 매 주기 호출.
    def _debug_viz_imu(self):
        now = time.time()
        if self._imu_t is None:
            color = (0, 0, 220)
            text_kr, text_en = '/imu 메시지 수신 안 됨', 'NO MESSAGE RECEIVED YET'
        else:
            age = now - self._imu_t
            if age > IMU_STALE_SEC:
                color = (0, 140, 255)
                text_kr, text_en = f'수신 끊김 (마지막 {age:.1f}초 전)', f'STALE (last {age:.1f}s ago)'
            else:
                color = (0, 200, 0)
                text_kr, text_en = f'정상 수신 중 ({age*1000:.0f}ms 전)', f'LIVE ({age*1000:.0f}ms ago)'

        yaw_deg = math.degrees(self.imu_yaw)
        canvas = np.full((360, 380, 3), 30, dtype=np.uint8)  # 방위 다이얼(cy=250, r=90)까지 포함하는 높이
        lines = [
            (f'IMU 연동: {text_kr}', (10, 8), color, 16, f'IMU link: {text_en}'),
            (f'imu_yaw: {yaw_deg:+7.2f}° ({self.imu_yaw:+.3f} rad)', (10, 44),
             (255, 255, 255), 18, f'imu_yaw: {yaw_deg:+7.2f} deg'),
        ]

        put_text_kr_multi(canvas, lines)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), color, 3)

        # 방위 다이얼 — atan2(y,x) 규약 그대로(0도=+x축, 반시계 증가) 그린 것뿐이라 실제
        # 좌/우회전과 화살표 회전방향이 맞는지는 이 창으로 실차에서 직접 봐야 한다(부호
        # 규약 미검증, 위 주석 참고). "값이 지금 이거다"를 시각적으로 보여주는 용도.
        cx, cy, r = canvas.shape[1] // 2, 250, 90
        cv2.circle(canvas, (cx, cy), r, (90, 90, 90), 2)
        cv2.circle(canvas, (cx, cy), 3, (255, 255, 255), -1)
        nx = int(cx + r * math.cos(self.imu_yaw))
        ny = int(cy - r * math.sin(self.imu_yaw))
        cv2.arrowedLine(canvas, (cx, cy), (nx, ny), (0, 255, 0), 2, tipLength=0.15)

        cv2.imshow('imu_debug', canvas)
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
        # [2026-08-19] "어떤 라이다 클러스터를 보고 판단했는지" 미니 BEV 패널 — 위 텍스트
        # 줄의 'front=.. dist=..'는 숫자로만 알려주는데, 벽/다른 차량 등 여러 클러스터가 같은
        # 전방 ROI에 동시에 잡혀도 실제로 '이 결정에 쓰인' 게 어느 점 묶음인지는 안 보였다.
        # [2026-08-19 수정] 처음엔 트리거 "순간"에만 스냅샷해 유예 내내 고정 표시했는데,
        # 그러면 avoid_hold가 다시 트리거되기 전까진 패널이 그대로 멈춰있어("이 창 멈췄나?"
        # 오해 재현됨) — perc_obstacle()이 매 틱 갱신하는 self._obstacle_cluster_x/y(현재
        # 타겟 클러스터)/self._obstacle_front_all_x/y(같은 ROI 배경점)를 매 프레임 그대로
        # 그리는 라이브 표시로 바꿨다. 장애물이 실제로 안 보이면(전방 ROI에 아무 점도 없으면)
        # 패널도 정직하게 빈 채로 나온다 — 그게 "멈춤"이 아니라 "지금 아무것도 안 잡힘"이라는
        # 뜻임을 밑에 텍스트로 같이 알려준다. "왜 아직 유예 중인지"는 _avoid_hold_trigger_*
        # (트리거 시점 스냅샷, _update_avoid_hold() 참고)를 오른쪽 텍스트로 별도 표시한다.
        PANEL_W, PANEL_H = 200, 200
        panel_x0, panel_y0 = 15, 226
        canvas = np.full((panel_y0 + PANEL_H + 12, 620, 3), 30, dtype=np.uint8)
        put_text_kr_multi(canvas, lines)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), active_color, 3)

        cv2.putText(canvas, 'LIVE FRONT LIDAR CLUSTER (매틱 갱신)',
                    (panel_x0, panel_y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        bev = canvas[panel_y0:panel_y0 + PANEL_H, panel_x0:panel_x0 + PANEL_W]
        FRONT_X_MAX_V, FRONT_Y_HALF_V = 5.0, 1.5   # perc_obstacle() FRONT ROI와 동일(표시축척용 재선언)
        PPM = min(PANEL_H / (FRONT_X_MAX_V + 0.3), PANEL_W / (2 * FRONT_Y_HALF_V + 0.6))
        EX, EY = PANEL_W // 2, PANEL_H - 8

        def to_px(wx, wy):
            return (int(EX - wy * PPM), int(EY - wx * PPM))

        for d in (1, 2, 3, 4, 5):
            if d > FRONT_X_MAX_V + 0.3:
                break
            cv2.circle(bev, (EX, EY), int(d * PPM), (55, 55, 55), 1)
        cv2.rectangle(bev, to_px(0.0, FRONT_Y_HALF_V), to_px(FRONT_X_MAX_V, -FRONT_Y_HALF_V), (0, 150, 150), 1)

        all_x = self._obstacle_front_all_x   # 매틱 갱신되는 라이브 값(perc_obstacle())
        all_y = self._obstacle_front_all_y
        for i in range(all_x.size):
            px, py = to_px(float(all_x[i]), float(all_y[i]))
            if 0 <= px < PANEL_W and 0 <= py < PANEL_H:
                cv2.circle(bev, (px, py), 2, (90, 90, 90), -1)   # 같은 ROI의 다른 점(미사용, 회색)

        cl_x = self._obstacle_cluster_x       # 매틱 갱신되는 라이브 값 — 지금 avoid_hold 트리거
        cl_y = self._obstacle_cluster_y       # 판정(obstacle_front/obstacle_dist)에 쓰이는 바로 그 클러스터
        for i in range(cl_x.size):
            px, py = to_px(float(cl_x[i]), float(cl_y[i]))
            if 0 <= px < PANEL_W and 0 <= py < PANEL_H:
                cv2.circle(bev, (px, py), 4, (0, 0, 255), -1)    # 실제 판정에 쓰인 타겟 클러스터(빨강)

        if all_x.size == 0:
            cv2.putText(bev, '(전방 ROI에 점 없음)', (8, PANEL_H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

        cv2.circle(bev, (EX, EY), 5, (255, 220, 0), -1)          # 자차 위치
        cv2.rectangle(bev, (0, 0), (PANEL_W - 1, PANEL_H - 1), (80, 80, 80), 1)

        info_x = panel_x0 + PANEL_W + 24
        info_lines = [
            ('빨강=현재 avoid_hold 판정에 쓰이는 클러스터', (info_x, panel_y0 - 4), (255, 255, 255), 13,
             'red = cluster currently used'),
            ('회색=같은 ROI의 다른 점(미사용)', (info_x, panel_y0 + 18), (150, 150, 150), 12,
             'gray = other ROI points (unused)'),
            (f'── 마지막 트리거 시점(hold_sec 재시작 순간) 스냅샷 ──',
             (info_x, panel_y0 + 44), (110, 160, 255), 12,
             '-- last trigger-moment snapshot --'),
            (f'원인={self._avoid_hold_trigger_cause or "(아직 없음)"}  '
             f'클러스터 {self._avoid_hold_trigger_cluster_pts}점'
             f'/전체클러스터 {self._avoid_hold_trigger_group_count}개',
             (info_x, panel_y0 + 66), (255, 255, 255), 12,
             f'cause={self._avoid_hold_trigger_cause or "(none)"} '
             f'pts={self._avoid_hold_trigger_cluster_pts}/{self._avoid_hold_trigger_group_count}'),
            (f'거리={self._avoid_hold_trigger_obstacle_dist:.2f}m  '
             f'폭={self._avoid_hold_trigger_obstacle_width:.2f}m  '
             f'{self._avoid_hold_trigger_obstacle_type}/{self._avoid_hold_trigger_obstacle_side}',
             (info_x, panel_y0 + 88), (255, 255, 255), 12,
             f'dist={self._avoid_hold_trigger_obstacle_dist:.2f}m '
             f'width={self._avoid_hold_trigger_obstacle_width:.2f}m '
             f'{self._avoid_hold_trigger_obstacle_type}/{self._avoid_hold_trigger_obstacle_side}'),
        ]
        put_text_kr_multi(canvas, info_lines)

        cv2.imshow('avoid_hold_debug', canvas)
        cv2.waitKey(1)

    # [DEBUG_VIZ_OBSTACLE_CUT] da 근접 컷(ENABLE_OBSTACLE_CUT) 전용 상태창 —
    #   avoid_hold_debug와 같은 구조(텍스트 줄로 라이다 raw/YOLO raw/AND확정/유지타이머
    #   잔여시간/해제카운터를 한곳에 모아 보여줌). 실측 안 된 파라미터도 항상 같이
    #   띄운다(config.py OBSTACLE_CUT_* 주석 참고).
    # [2026-08-20] YOLO 차량검출(카메라) + 그 판정에 실제로 쓰인 라이다 ROI를 텍스트
    # 상태 한 창에 합쳐서 그린다 — 원래는 YOLO 원시 박스가 'yolo_vehicle_result'라는
    # 별도 창, 라이다 근접 여부는 이 창의 텍스트 한 줄(True/False)로만 나뉘어 있어서
    # "욜로는 car라는데 라이다는 왜 안 잡히나/그 반대" 같은 AND 불일치 원인을 창 두 개를
    # 오가며 봐야 했다 — 카메라 박스와 라이다 ROI 안 점을 나란히 놓고 바로 대조할 수 있게
    # 한 창으로 합침(요청 반영). yolo_vehicle.py는 이제 전용 창을 안 띄우고
    # get_latest_debug_frame()으로 프레임만 넘긴다(해당 함수 주석 참고).
    # [2026-08-22, 요청 반영] "지금 B1/B2/B3 중 어느 단계고 뭘 기다리는지"를 터미널 로그
    # (_print_debug()의 [{mission_state}|{behavior_state}|{phase}] 요약 줄, 상황 대응 중
    # 알아보기 힘들다는 요청)가 아니라 _debug_viz_obstacle_cut() 창 상단에 한글 한 줄로
    # 보여주기 위한 헬퍼 — (한글, BGR색, 영문) 튜플을 반환한다. run_behavior_fsm()의
    # Phase.LAVACON/OBSTACLE_ZONE 분기와 정확히 같은 판단 순서(B1→B2→B3)를 그대로 따라간다.
    # [2026-08-22h, 요청 반영] 좌회전 진입 램프 중(checker_pillar_trigger를 넘어
    # _do_checker_ramp_turn()이 도는 구간)도 여기서 최우선으로 걸러 보여준다.
    def _current_stage_label(self):
        # [2026-08-22h, 요청 반영] 체크무늬 게이트 라이다 임계값(checker_pillar_trigger)을
        # 넘어 좌회전 진입 램프(_do_checker_ramp_turn(), -10°→-30° 조향)로 넘어간 순간도
        # 이 창 헤드라인에서 바로 보이게 — mission_state는 그동안 계속 S0_SIGNAL이라(램프가
        # 끝나야 S1_LANE_FOLLOW로 전환됨) 아래 S0_SIGNAL 분기보다 먼저 걸러야 한다.
        if self._checker_ramp_dist is not None:
            return (f'좌회전 진입 중 (게이트 통과, 조향 {self.ctrl_angle:.1f}°, '
                     f'{self._checker_ramp_dist:.2f}/{CHECKER_TURN_RAMP_DIST_M:.2f}m)',
                    (0, 200, 0),
                    f'LEFT TURN ENTRY (angle={self.ctrl_angle:.1f} deg, '
                    f'{self._checker_ramp_dist:.2f}/{CHECKER_TURN_RAMP_DIST_M:.2f} m)')

        ms = self.mission_state
        if ms == MissionState.S0_SIGNAL:
            return '신호등 판독 대기', (0, 200, 255), 'S0_SIGNAL: waiting for light'
        if ms == MissionState.S4_FINISH:
            return '주행 종료', (150, 150, 150), 'S4_FINISH'

        if self.phase == Phase.LAVACON:
            engaged = self._lavacon_engaged
            state_kr = '진행 중(탈출 대기)' if engaged else '대기 중(좌우 동시검출 트리거 대기)'
            state_en = 'engaged' if engaged else 'waiting for entry trigger'
            color = (0, 200, 0) if engaged else (0, 140, 255)
            return f'B1 라바콘 — {state_kr}', color, f'B1 LAVACON — {state_en}'

        if self.phase == Phase.OBSTACLE_ZONE:
            if not self._b2_passed:
                tag, label_kr, label_en = 'B2', 'B2 고정장애물', 'B2 FIXED OBSTACLE'
            elif not self._b3_passed:
                tag, label_kr, label_en = 'B3', 'B3 방해차량', 'B3 VEHICLE'
            else:
                return 'OBSTACLE_ZONE 종료 처리 중', (150, 150, 150), 'OBSTACLE_ZONE finishing'
            active = self._obscut_zone_tag == tag
            state_kr = '감지됨(회피/통과 중)' if active else '대기 중(감지 안 됨)'
            state_en = 'detected' if active else 'waiting'
            color = (0, 200, 0) if active else (0, 140, 255)
            return f'{label_kr} — {state_kr}', color, f'{label_en} — {state_en}'

        return 'B1/B2/B3 모두 통과 — 다음 교차로 대기', (0, 200, 0), 'ALL PASSED — waiting for next intersection'

    def _debug_viz_obstacle_cut(self):
        now = time.time()
        remaining = max(0.0, self._obstacle_cut_until_t - now)
        active_color = (0, 200, 0) if self.obstacle_cut_active else (110, 110, 110)
        slide = getattr(self.lane_detector, '_slide', None)
        col_range = getattr(slide, 'obstacle_cut_col_range', None)
        UNMEASURED = (60, 160, 255)

        # [2026-08-21] B2(고정장애물=콘)/B3(방해차량) 공용 창 — obstacle_cut 메커니즘 자체가
        # 라이다 ROI/트리거/유지타이머까지 완전히 공유라(perc_obstacle_cut_trigger() 참고)
        # 창 대부분은 손 안 대고, 카메라 패널/검출값 텍스트만 지금 활성 스테이지
        # (_active_yolo_stage(), Phase.OBSTACLE_ZONE에서 _b2_passed 기준 'cone'/'vehicle')에
        # 맞춰 콘 검출기 ↔ 차량 검출기를 동적으로 바꿔 보여준다.
        # [2026-08-22, 요청 반영] 상단 헤드라인에 "지금 B1/B2/B3 중 어느 단계고 뭘 기다리는지"
        # 를 한 줄로 보여준다 — 예전엔 이 정보가 _print_debug()의 터미널 [SIG] 요약 줄로만
        # 나와서 실시간으로 알아보기 어렵다는 요청 반영(_current_stage_label() 참고).
        cam_stage = self._active_yolo_stage()
        # [2026-08-23] 요청 반영 — 면적 임계값이 B1/B2/B3마다 다른 변수(YOLO_CONE_MIN_BOX_AREA_PX_B1/
        # _B2, YOLO_VEHICLE_MIN_BOX_AREA_PX_B3)로 분리됐다 — 콘 카메라는 cam_stage만으로는
        # B1(라바콘 진입 트리거 대기)인지 B2(고정장애물)인지 못 가리므로 self.phase로 한 번 더
        # 나눈다(_active_yolo_stage()의 Phase.LAVACON/OBSTACLE_ZONE 분기와 동일 판단 순서).
        # yolo_detected도 "박스 존재" 원시값이 아니라 실제 트리거에 쓰이는 면적 게이트까지
        # 통과한 값으로 보여준다(perc_lavacon_trigger()/perc_obstacle_cut_trigger()와 동일 식).
        if cam_stage == 'vehicle':
            cam_detector, cam_label = self.yolo_vehicle_cut_detector, 'YOLO VEHICLE CAM'
            viz_flag_name = 'DEBUG_VIZ_YOLO_VEHICLE'
            cam_max_area, cam_min_area, area_stage_tag = (
                self.vehicle_max_box_area_cut, YOLO_VEHICLE_MIN_BOX_AREA_PX_B3, 'B3')
            yolo_detected, yolo_flag_label = (
                self.vehicle_detected_yolo_cut and cam_max_area >= cam_min_area), 'car'
        else:
            cam_detector, cam_label = self.yolo_cone_detector, 'YOLO CONE CAM'
            viz_flag_name = 'DEBUG_VIZ_YOLO_CONE'
            if self.phase == Phase.LAVACON:
                cam_min_area, area_stage_tag = YOLO_CONE_MIN_BOX_AREA_PX_B1, 'B1'
            else:
                cam_min_area, area_stage_tag = YOLO_CONE_MIN_BOX_AREA_PX_B2, 'B2'
            cam_max_area = self.cone_max_box_area
            yolo_detected, yolo_flag_label = (
                self.cone_detected_yolo and cam_max_area >= cam_min_area), 'cone'

        # --- 상단 카메라(YOLO)/라이다 BEV 패널 레이아웃 -------------------------------
        CAM_W, CAM_H = 300, 220
        cam_x0, cam_y0 = 10, 58   # [2026-08-22] 아래 stage 줄 추가로 34→58(+24px)
        PANEL_W, PANEL_H = 240, CAM_H
        panel_x0, panel_y0 = cam_x0 + CAM_W + 20, cam_y0
        text_y0 = cam_y0 + CAM_H + 16

        canvas = np.full((text_y0 + 208, panel_x0 + PANEL_W + 10, 3), 30, dtype=np.uint8)

        stage_kr, stage_color, stage_en = self._current_stage_label()
        headline = [
            (f'OBSTACLE-CUT: {"활성" if self.obstacle_cut_active else "대기"}  '
             f'(남은 {remaining:.2f}s / floor={self._obstacle_cut_hold_sec_min:.2f}s)',
             (10, 8), active_color, 17,
             f'OBSTACLE-CUT: {"ACTIVE" if self.obstacle_cut_active else "idle"} '
             f'({remaining:.2f}s left / floor={self._obstacle_cut_hold_sec_min:.2f}s)'),
            (f'단계: {stage_kr}', (10, 30), stage_color, 17, f'STAGE: {stage_en}'),
        ]
        lines = [
            (f'직전 해제 사유: {self.obstacle_cut_release_reason or "(아직 없음)"}   '
             f'장애물 y={self._obstacle_cut_y if self._obstacle_cut_y is not None else "N/A"}',
             (10, text_y0), (255, 255, 255), 14,
             f'last release: {self.obstacle_cut_release_reason or "(none)"}'),
            (f'[{cam_stage or "?"}단계] 트리거 — 라이다 근접={self._obstacle_cut_lidar_near}  '
             f'YOLO {yolo_flag_label}={yolo_detected}  '
             f'AND확정={self._obstacle_cut_trigger_cnt}/{OBSTACLE_CUT_TRIGGER_FRAMES}',
             (10, text_y0 + 28), (255, 255, 255), 13,
             f'[{cam_stage}] trigger — lidar={self._obstacle_cut_lidar_near} yolo={yolo_detected} '
             f'confirm={self._obstacle_cut_trigger_cnt}/{OBSTACLE_CUT_TRIGGER_FRAMES}'),
            (f'해제 진행 — 라이다클리어 {self._obstacle_cut_release_cnt}/{OBSTACLE_CUT_RELEASE_CONFIRM_FRAMES}',
             (10, text_y0 + 50), (200, 200, 200), 13,
             f'release progress {self._obstacle_cut_release_cnt}/{OBSTACLE_CUT_RELEASE_CONFIRM_FRAMES}'),
            (f'좌우 교차검증 — 라이다={self._obstacle_cut_lidar_side} YOLO={self._obstacle_cut_yolo_side} '
             + ('★불일치→vehicle_seen 취소★' if self._obstacle_cut_side_veto else '(일치 또는 미해당)'),
             (10, text_y0 + 72), (0, 0, 255) if self._obstacle_cut_side_veto else (180, 180, 180), 13,
             f'side-check lidar={self._obstacle_cut_lidar_side} yolo={self._obstacle_cut_yolo_side} '
             f'veto={self._obstacle_cut_side_veto}'),
            (f'컷 열(px) 범위 = {col_range}  (da BEV 좌표계)',
             (10, text_y0 + 94), (255, 0, 255), 13, f'cut col range = {col_range}'),
            (f'YOLO 검출기({cam_stage or "?"}): {"정상" if cam_detector is not None else "초기화실패→라이다단독폴백"}',
             (10, text_y0 + 116), (255, 255, 255), 12,
             f'yolo detector({cam_stage}): {"OK" if cam_detector is not None else "FAILED->lidar-only"}'),
            ('── ★ 전 파라미터 실차 미검증 ★', (10, text_y0 + 140), UNMEASURED, 13,
             '-- ALL PARAMS UNMEASURED --'),
            (f'TRIGGER X_MAX={self._obstacle_cut_x_max}m Y_HALF={self._obstacle_cut_y_half}m  '
             f'NEAR_M={OBSTACLE_CUT_NEAR_M}m  PRE_CUT_SPEED_CAP={SPEED_PRE_OBSTACLE_CAP}(감지 전만)',
             (10, text_y0 + 162), UNMEASURED, 12,
             f'trigger_x={self._obstacle_cut_x_max} y_half={self._obstacle_cut_y_half} '
             f'near={OBSTACLE_CUT_NEAR_M} pre_cut_speed_cap={SPEED_PRE_OBSTACLE_CAP}(pre-detect only)'),
            # [2026-08-23] 요청 반영 — "박스 하나라도 찍히면 검출"이 아니라 "가장 큰 박스
            # 면적이 임계값 이상"으로 바뀐 검출조건을 실차에서 바로 보고 조정할 수 있게
            # 값/임계값을 나란히 표시. 임계값은 B1/B2/B3마다 다른 변수(config.py
            # YOLO_CONE_MIN_BOX_AREA_PX_B1/_B2, YOLO_VEHICLE_MIN_BOX_AREA_PX_B3)라
            # area_stage_tag로 지금 어느 임계값이 적용 중인지도 같이 보여준다.
            (f'[{area_stage_tag}] YOLO {yolo_flag_label} 최대 박스면적={cam_max_area:.0f}px²  '
             f'(임계값 {cam_min_area:.0f}px² 이상이어야 검출 인정)',
             (10, text_y0 + 184), (0, 200, 0) if cam_max_area >= cam_min_area else (180, 180, 180), 13,
             f'[{area_stage_tag}] yolo {yolo_flag_label} max_box_area={cam_max_area:.0f}px² '
             f'(min={cam_min_area:.0f}px²)'),
        ]
        put_text_kr_multi(canvas, headline + lines)

        # --- 카메라(YOLO 원시 박스) 패널 — cam_stage에 따라 콘/차량 검출기 중 하나 --------
        cam_vis = cam_detector.get_latest_debug_frame() if cam_detector is not None else None
        cam_region = canvas[cam_y0:cam_y0 + CAM_H, cam_x0:cam_x0 + CAM_W]
        if cam_vis is not None:
            # 종횡비 유지 없이 단순 리사이즈(표시 전용 — yolo_vehicle.py/yolo_cone.py
            # preprocess()와 동일 관례, "검출 여부"만 보면 되고 좌표 왜곡은 이 창의 목적에
            # 영향 없음).
            cam_region[:] = cv2.resize(cam_vis, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
        else:
            reason = (f'{viz_flag_name} 확인' if cam_detector is not None else '검출기 초기화 실패')
            cv2.putText(cam_region, f'카메라 프레임 없음 ({reason})', (8, CAM_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (cam_x0, cam_y0), (cam_x0 + CAM_W - 1, cam_y0 + CAM_H - 1), (80, 80, 80), 1)
        cv2.putText(canvas, cam_label, (cam_x0, cam_y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # --- 라이다 ROI(트리거에 실제 쓰인 박스) BEV 패널 ---------------------------
        # [2026-08-23] cv2 기본폰트(Hershey)는 한글을 못 그려 '매틱 갱신' 부분이 '????'로
        # 깨져 보이던 버그 수정 — 다른 텍스트들과 동일하게 put_text_kr_multi(PIL 기반)로 교체.
        put_text_kr_multi(canvas, [
            ('LIDAR TRIGGER ROI (매틱 갱신)', (panel_x0, panel_y0 - 20), (180, 180, 180), 15,
             'LIDAR TRIGGER ROI (updated every tick)'),
        ])
        bev = canvas[panel_y0:panel_y0 + PANEL_H, panel_x0:panel_x0 + PANEL_W]
        DISP_X_MAX = self._obstacle_cut_x_max + 1.0
        DISP_Y_HALF = self._obstacle_cut_y_half + 0.45
        PPM = min(PANEL_H / (DISP_X_MAX + 0.2), PANEL_W / (2 * DISP_Y_HALF + 0.2))
        EX, EY = PANEL_W // 2, PANEL_H - 8

        def to_px(wx, wy):
            return (int(EX - wy * PPM), int(EY - wx * PPM))

        for d in (0.5, 1.0, 1.5, 2.0):
            if d > DISP_X_MAX:
                break
            cv2.circle(bev, (EX, EY), int(d * PPM), (55, 55, 55), 1)
        # 실제 트리거 박스(이번 틱 x_max x y_half) — "검증범위" 그 자체.
        # [2026-08-23r2] B3 단계면 self._obstacle_cut_x_max/_y_half가 이미 VEHICLE 전용값
        # (2.5m/0.75m)이라 박스도 자동으로 그만큼 넓게 그려진다(perc_obstacle_cut_trigger() 참고).
        cv2.rectangle(bev, to_px(0.0, self._obstacle_cut_y_half),
                      to_px(self._obstacle_cut_x_max, -self._obstacle_cut_y_half),
                      (0, 255, 255), 1)

        bg_x, bg_y = self._obstacle_cut_bg_x, self._obstacle_cut_bg_y
        for i in range(bg_x.size):
            px, py = to_px(float(bg_x[i]), float(bg_y[i]))
            if 0 <= px < PANEL_W and 0 <= py < PANEL_H:
                cv2.circle(bev, (px, py), 2, (90, 90, 90), -1)   # 표시범위 안 배경점(미트리거, 회색)

        roi_x, roi_y = self._obstacle_cut_roi_x, self._obstacle_cut_roi_y
        for i in range(roi_x.size):
            px, py = to_px(float(roi_x[i]), float(roi_y[i]))
            if 0 <= px < PANEL_W and 0 <= py < PANEL_H:
                cv2.circle(bev, (px, py), 4, (0, 0, 255), -1)   # 트리거 박스 안(실제 판정에 쓰인 점, 빨강)

        if bg_x.size == 0:
            # [2026-08-23] 위 패널 제목과 동일한 이유로 '????' 깨짐 수정 — put_text_kr_multi로 교체.
            put_text_kr_multi(bev, [
                ('(표시범위 안 점 없음)', (8, PANEL_H - 34), (120, 120, 120), 13,
                 '(no points in range)'),
            ])

        cv2.circle(bev, (EX, EY), 5, (255, 220, 0), -1)          # 자차 위치
        cv2.rectangle(bev, (0, 0), (PANEL_W - 1, PANEL_H - 1), (80, 80, 80), 1)

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), active_color, 3)
        if 'obstacle_cut_debug' not in self._dbg_windows_positioned:
            cv2.namedWindow('obstacle_cut_debug', cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow('obstacle_cut_debug', *DEBUG_WIN_POS_OBSTACLE_CUT)
            self._dbg_windows_positioned.add('obstacle_cut_debug')
        cv2.imshow('obstacle_cut_debug', canvas)
        cv2.waitKey(1)

    # [DEBUG_VIZ_LEFT_TURN] 좌회전(체크무늬 게이트 진입) 전용 통합 디버그 창.
    #   [2026-08-22i, 요청 반영] "좌회전 관련 실행중/실행끝/발행각도/라이다감지 합친
    #   디버그창 하나"로 신설 — 예전엔 이 네 가지가 obstacle_cut_debug(_current_stage_label()
    #   헤드라인)/checker_pillar_bev(라이다 원시 BEV)에 흩어져 있었다. 같은 요청으로 이
    #   창과 DEBUG_VIZ_DL_LANE(차선인식)만 남기고 나머지 DEBUG_VIZ_*는 다 껐다(config.py 참고).
    #   [2026-08-22j/k/l, 요청 반영] 상태 텍스트 패널 아래에 전방 카메라(self.img_front)와
    #   라이다 BEV(self._checker_pillar_bev_img)를 좌우로 나란히 붙여 한 창에서 텍스트+영상을
    #   같이 본다.
    def _debug_viz_left_turn(self):
        running = self._checker_ramp_dist is not None
        now = time.time()
        since_done = (now - self._left_turn_last_done_t) if self._left_turn_last_done_t is not None else None

        canvas = np.full((216, 480, 3), 30, dtype=np.uint8)

        run_col = (0, 200, 0) if running else (110, 110, 110)
        run_txt = f'실행중: {"예 (게이트 통과, 조향 램프 진행 중)" if running else "아니오"}'
        run_en = f'RUNNING: {running}'

        if since_done is None:
            done_col, done_txt, done_en = (110, 110, 110), '실행끝: 아직 없음(한 번도 완료 안 됨)', 'FINISHED: never'
        elif since_done < 3.0:
            done_col, done_txt, done_en = (0, 255, 255), f'실행끝: 방금 완료 ({since_done:.1f}초 전)', f'FINISHED: {since_done:.1f}s ago'
        else:
            done_col, done_txt, done_en = (200, 200, 200), f'실행끝: 마지막 완료 {since_done:.1f}초 전', f'FINISHED: {since_done:.1f}s ago'

        if running:
            angle_txt = (f'발행각도: {self.ctrl_angle:.1f}°  '
                         f'({CHECKER_TURN_RAMP_START_ANGLE:.0f}°→{CHECKER_TURN_RAMP_END_ANGLE:.0f}°, '
                         f'{self._checker_ramp_dist:.2f}/{CHECKER_TURN_RAMP_DIST_M:.2f}m, '
                         f'{CHECKER_TURN_RAMP_CURVE})')
            angle_en = (f'ANGLE: {self.ctrl_angle:.1f} deg ({self._checker_ramp_dist:.2f}/'
                        f'{CHECKER_TURN_RAMP_DIST_M:.2f}m, {CHECKER_TURN_RAMP_CURVE})')
        else:
            angle_txt = f'발행각도: {self.ctrl_angle:.1f}° (좌회전 램프 비활성 — 현재 전체 발행값)'
            angle_en = f'ANGLE: {self.ctrl_angle:.1f} deg (ramp idle — current published value)'

        lidar_col = (0, 200, 0) if self.checker_pillar_trigger else (110, 110, 110)
        lidar_txt = (f'라이다감지: {self.checker_pillar_trigger}  '
                     f'(디바운스 {self._checker_pillar_trigger_cnt}/{CHECKER_PILLAR_CONFIRM_FRAMES}, '
                     f'좌{self.checker_pillar_left_detected} 우{self.checker_pillar_right_detected}, '
                     f'간격 {self.checker_pillar_lat_dist_m:.2f}m)')
        lidar_en = (f'LIDAR: trigger={self.checker_pillar_trigger} '
                    f'cnt={self._checker_pillar_trigger_cnt}/{CHECKER_PILLAR_CONFIRM_FRAMES}')

        # [2026-08-22m, 요청 반영: "욜로 신호등이 어떤 상태로 검출됐는지도 한줄 추가"]
        # perc_yolo_signal_state()가 매 프레임 갱신하는 원시 점등 상태(signal_red/left/
        # straight_on)와 perc_signal()이 SIG_CONFIRM_FRAMES 연속 유지로 승격시킨 확정값
        # (signal_left/straight_confirmed)을 한 줄로 같이 보여준다 — _print_debug()의
        # 터미널 [SIG] 로그(sig_flags)와 동일한 포맷.
        sig_col = (0, 200, 0) if (self.signal_left_confirmed or self.signal_straight_confirmed) else (200, 200, 200)
        sig_txt = (f'신호등(YOLO): R{int(self.signal_red_on)} L{int(self.signal_left_on)} '
                   f'S{int(self.signal_straight_on)}  확정 L={int(self.signal_left_confirmed)}'
                   f'({self._sig_left_cnt}/{SIG_CONFIRM_FRAMES}) '
                   f'S={int(self.signal_straight_confirmed)}({self._sig_straight_cnt}/{SIG_CONFIRM_FRAMES})')
        sig_en = (f'SIGNAL(YOLO): R={int(self.signal_red_on)} L={int(self.signal_left_on)} '
                  f'S={int(self.signal_straight_on)} confirmL={int(self.signal_left_confirmed)} '
                  f'confirmS={int(self.signal_straight_confirmed)}')

        # [2026-08-23c, 요청 반영] "YOLO_신호등 창엔 분명히 찍히는데 여기(확정 L/S)엔 안
        # 뜬다"는 혼란의 실제 원인은 대부분 _active_yolo_stage()가 이 틱에 'signal'을
        # 아예 리턴 안 해서(phase가 Phase.LAVACON/OBSTACLE_ZONE으로 넘어가 있으면
        # perc_yolo_signal_state() 자체가 안 불림) — 그럴 땐 YOLO_신호등 창은 마지막으로
        # 성공했던 추론 결과가 그대로 얼어붙어 있는 것뿐이다(백그라운드 워커가 새 프레임을
        # 못 받아 유휴 상태, yolo_signal_state.py _worker() 참고). 한눈에 구분되게
        # phase/현재 활성 YOLO 스테이지를 별도 줄로 보여준다.
        yolo_stage = self._active_yolo_stage()
        stage_col = (0, 200, 0) if yolo_stage == 'signal' else (0, 120, 220)
        stage_txt = (f'phase={self.phase.name}  활성 YOLO={yolo_stage!r}'
                     + ('' if yolo_stage == 'signal' else '  ← 신호등 YOLO 꺼짐, 위 확정 안 뜸'))
        stage_en = f'phase={self.phase.name} active_yolo_stage={yolo_stage!r}'

        put_text_kr_multi(canvas, [
            ('좌회전 진입 램프 통합 상태', (10, 8), (255, 255, 255), 18, 'LEFT TURN ENTRY STATUS'),
            (run_txt, (10, 40), run_col, 16, run_en),
            (done_txt, (10, 66), done_col, 16, done_en),
            (angle_txt, (10, 96), (0, 255, 255) if running else (200, 200, 200), 15, angle_en),
            (lidar_txt, (10, 126), lidar_col, 15, lidar_en),
            (sig_txt, (10, 156), sig_col, 14, sig_en),
            (stage_txt, (10, 182), stage_col, 14, stage_en),
        ])

        border_col = (0, 200, 0) if running else (80, 80, 80)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), border_col, 3)
        panel_w = canvas.shape[1]

        # [2026-08-22l, 요청 반영: "카메라 영상부분을 옆으로 붙여줘"] 전방 카메라
        # (self.img_front, YOLO 오버레이 없는 원본 그대로)와 라이다 BEV
        # (self._checker_pillar_bev_img)를 상태 패널 아래에 좌우로 나란히 붙인다 — 둘의
        # 종횡비가 서로 달라 자연 크기로는 나란히 맞추기 어려우므로, obstacle_cut_debug
        # 카메라 패널과 동일 관례(종횡비 무시 단순 리사이즈 — "표시 전용, 좌표 왜곡은 이
        # 창의 목적에 영향 없음")로 각각을 고정 박스에 맞춘다.
        # [2026-08-23] 이 안의 두 패널을 잠깐 축소했다가(요청 오해) 원복 — "카메라욜로랑
        # 검출라이다 크기 줄여줘"는 이 통합창 안의 패널이 아니라 독립 창인 'YOLO_신호등'
        # (perception/yolo_signal_state.py)과 'checker_pillar_bev'(DEBUG_VIZ_CHECKER_PILLAR
        # 단독 창) 얘기였다 — 그쪽에서 축소 처리.
        half_w = panel_w // 2
        HALF_H = 220

        if self.img_front is not None:
            cam_panel = cv2.resize(self.img_front, (half_w, HALF_H), interpolation=cv2.INTER_AREA)
        else:
            cam_panel = np.full((HALF_H, half_w, 3), 30, dtype=np.uint8)
            cv2.putText(cam_panel, 'no frame yet', (10, HALF_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(cam_panel, 'FRONT CAM', (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if self._checker_pillar_bev_img is not None:
            bev_panel = cv2.resize(self._checker_pillar_bev_img, (panel_w - half_w, HALF_H),
                                    interpolation=cv2.INTER_AREA)
        else:
            bev_panel = np.full((HALF_H, panel_w - half_w, 3), 30, dtype=np.uint8)
            cv2.putText(bev_panel, 'no frame yet', (10, HALF_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(bev_panel, 'LIDAR BEV', (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cam_row = np.hstack([cam_panel, bev_panel])
        cv2.line(cam_row, (half_w, 0), (half_w, HALF_H - 1), (80, 80, 80), 1)

        combined = np.vstack([canvas, cam_row])
        if 'left_turn_debug' not in self._dbg_windows_positioned:
            cv2.namedWindow('left_turn_debug', cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow('left_turn_debug', *DEBUG_WIN_POS_LEFT_TURN)
            self._dbg_windows_positioned.add('left_turn_debug')
        cv2.imshow('left_turn_debug', combined)
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

    # ── B1-라바콘: 박스 스택 페어링 경로 추종([2026-08-19] 보로노이 → 클러스터 매칭 → 박스 스택으로 교체) ──
    def _handle_lavacon(self):
        """
        [2026-08-22] 요청 반영 — run_behavior_fsm()이 다시 behavior_state=B1_LAVACON을
        세팅하게 되면서(_lavacon_engaged인 동안) 이 함수가 다시 호출된다. exit(구간 종료)
        판정은 여전히 run_behavior_fsm()의 Phase.LAVACON 분기가 전담하므로 여기서는
        중복으로 하지 않는다(아래 함수 끝 주석 참고).

        (아래는 실제 회피 조향 로직 자체 설명)
        우측 콘이 연속 LAVACON_DONE_FRAMES 프레임 미검출되면 고정장애물 구간으로 전환.

        [2026-08-11] 조향은 라바콘 전용 P게인(LAVACON_KP, 폐기) 대신 _lane_steer()를
        그대로 재사용한다 — 라인주행(_lane_drive())과 조향 파라미터(self.pure_pursuit
        인스턴스, PP_* 게인, ANGLE_MAX/ANGLE_RATE_MAX)를 완전히 일치시키기로 한 결정.
        self.lavacon_path는 perc_lavacon()이 라이다 미터 좌표를
        self.lane_path와 같은 px 스케일로 변환해둔 것이고(row 0 = 라이다 원점), 차량
        기준점은 그 변환의 원점에서 LIDAR_TO_VEHICLE_FRONT_M만큼 뒤로 민 지점이다.

        [2026-08-19] vehicle_x=0.0은 여전히 라이다의 좌우 중심(라이다가 차량에 좌우로는
        정렬돼 있다고 가정)과 같지만, 종방향(전후)은 라이다가 차량 맨 앞부분보다 실측
        LIDAR_TO_VEHICLE_FRONT_M(config.py, lavacon_bev 디버그창에서 자차 마커를 실측
        물체와 맞춰 확인)만큼 앞에 있다는 게 확인돼서, _lane_steer()의 vehicle_y_px에
        "라이다 원점(row 0) 기준 차량의 절대 행" = +LIDAR_TO_VEHICLE_FRONT_M*
        DL_PIXELS_PER_METER(전방=y감소 관례라 양수=뒤쪽)를 직접 넘긴다 — path[0]의 행에
        더하는 게 아니라 완전히 대체하는 것에 주의(path[0]은 BOX_LON_START 등 다른
        값에 좌우되는 별개 기준이라 더하면 틀린 값이 나온다, _lane_steer() 주석 참고).
        안 넘기면(구 동작) 차량이 실제보다 LIDAR_TO_VEHICLE_FRONT_M만큼 앞서 있다고
        착각해 Pure Pursuit lookahead 거리가 그만큼씩 짧게 계산된다. 실차 재검증
        필요(2026-08-19, 코드 리뷰 수준 반영).

        [2026-08-19] LAVACON_STEER_MODE_DA_PUSH=True면 위 박스 스택 경로(self.lavacon_path)
        대신 _lavacon_steer_da_push()(da 경로 + 콘 침범 시 옆으로 밀기)를 쓴다. 기본
        False라 안 켜면 이 함수의 나머지 동작은 전과 100% 동일 — self.lavacon_done 기반
        구간 종료 판정도 두 모드에서 공유한다(perc_lavacon()이 매 틱 그대로 계산해두므로).
        """
        if LAVACON_KICK_ENABLED and self._lavacon_kick_cnt > 0:
            # [2026-08-23, 요청 반영] 실험용 킥 구간 — push/차선 조향 계산을 건너뛰고 고정
            # 조향각을 강제. 디버그창(push 표시)이 이번 틱엔 "안 밀림"으로 보이도록 push
            # 플래그도 같이 꺼둔다. 속도는 손대지 않음 — 아래 _update_speed()가 그대로 돎.
            self._lavacon_kick_cnt -= 1
            self.ctrl_angle = LAVACON_KICK_ANGLE_DEG
            self._lavacon_push_active = False
            self._lavacon_push_px = 0.0
        elif LAVACON_STEER_MODE_DA_PUSH:
            self.ctrl_angle = self._lavacon_steer_da_push()
        else:
            self.ctrl_angle = self._lane_steer(
                path=self.lavacon_path, vehicle_x=0.0,
                vehicle_y_px=LIDAR_TO_VEHICLE_FRONT_M * DL_PIXELS_PER_METER)
        # [2026-08-22] SPEED_LAVACON(2.5 고정) 삭제(요청 반영) — 여기서 매 틱 고정속도로
        # 덮어써서 B1 진입 즉시 그 값에 "굳는" 증상이 실차에서 확인됐다. 이후(같은 날
        # 재요청) "S1 주행하던 대로 똑같이"로 다시 바뀌어, 진입 시점 속도를 얼려두는 대신
        # _update_speed()(_lane_drive()와 공유하는 코너 감속/가속 램프 로직, speed15
        # 프리셋 기준 SPEED_NORMAL=12/SPEED_CORNER_MIN=10 사이에서 동적으로 움직임)를
        # 그대로 재사용한다 — self.ctrl_angle만 위에서 이미 라바콘 조향으로 바뀐 상태라
        # _update_speed()는 조향 소스를 몰라도(코너 신호는 ctrl_angle 하나만 봄) 동일하게
        # 동작한다.
        self._update_speed()
        # exit(구간 종료) 판정은 run_behavior_fsm()의 Phase.LAVACON 분기로 옮겨졌다 —
        # 이 함수가 다시 불리게 되는 시점에도 여기서 중복으로 하지 않는다(위 docstring 참고).

    # ── B1-라바콘 대안 조향: da 경로 + 콘 침범 시 옆으로 밀기 (LAVACON_STEER_MODE_DA_PUSH) ──
    #   [2026-08-19] 라이다 박스 스택 페어링이 실차에서 계속 듬성한 검출/노이즈에 시달려서
    #   (요청 반영) 아예 다른 축으로 전환. da(주행가능영역) 경로(self.lane_path, S1/S3
    #   차선주행과 완전히 같은 신호)를 그대로 신뢰하고 조향하되, YOLO로 재확인된 콘이
    #   안전마진(LAVACON_PUSH_SAFETY_MARGIN_M) 안으로 들어왔을 때만 그만큼 반대쪽으로
    #   self.lane_path 전체를 옆으로 밀어서 _lane_steer()에 넘긴다. "정밀한 경로 재구성"
    #   대신 "위험할 때만 미는" 훨씬 단순하고 견고한 신호라, 라이다가 콘 1점만 듬성하게
    #   봐도(박스 스택이 실차에서 계속 시달린 문제) 바로 동작한다. da 단독으로도 라바콘
    #   구간을 그럭저럭 지나간다는 게 실차로 이미 확인됨(사용자, 2026-08-19).
    def _lavacon_steer_da_push(self):
        """[2026-08-19, 최초 도입] push는 원래 self.cone_detected_yolo(그 프레임에 카메라로도
        콘이 실제 보일 때)로만 켰다 — perc_lavacon_trigger()가 진입 판정에 라이다 단독 대신
        YOLO AND 라이다를 쓰는 것과 같은 이유(라이다 단독 클러스터는 벽 모서리 등에서 오검출
        여지가 있음).

        [2026-08-21, 요청 반영] 그 매 프레임 카메라 재확인 게이트를 뺐다 — 실차에서 라이다
        클러스터는 선명하게 잡히는데도 YOLO가 그 프레임에 콘을 놓쳐(카메라 각도/거리/조도 등)
        push 자체가 안 걸리는 문제가 있었다(사용자 실측). 이 함수는 이제 `_lane_drive()`에서
        `self._lavacon_engaged`가 True일 때만 불린다 — 그 latch 자체가 이미
        perc_lavacon_trigger()의 라이다 AND YOLO 이중확인을 거쳐 확정된 것이므로("지금
        라바콘 구간 안"이라는 전제가 이미 보장됨), 매 프레임 카메라 재확인 없이 라이다
        근접만으로 push를 켜도 진입 오검출 문제와는 별개다(진입 자체는 여전히 이중확인 그대로).

        push_m 부호: nearest_cone_lateral()의 y는 좌측+ — 좌측 콘이 안전마진을 침범하면
        그만큼 우측(+px)으로, 우측 콘이 침범하면 그만큼 좌측(-px)으로 민다. 두 콘이 동시에
        침범하면(통로 자체가 마진의 2배보다 좁음) 서로 상쇄돼 순 push가 줄어드는데, 이건
        "이미 중앙에 있으면 그대로 두는" 게 맞는 동작이라 의도된 결과다.

        vehicle_x는 밀지 않는다 — path만 밀어야 "차량이 경로 대비 반대쪽으로 치우쳤다"고
        Pure Pursuit이 해석해 실제로 옆으로 붙는 조향이 나온다. vehicle_x도 같이 밀면
        상대위치가 그대로라 아무 효과가 없다.
        """
        left_y, right_y = nearest_cone_lateral(
            self.lidar_ranges, LAVACON_PUSH_LON_MIN, LAVACON_PUSH_LON_MAX,
            LAVACON_PUSH_LAT_LIMIT, lon_max_l=LAVACON_PUSH_LON_MAX_L)
        push_m = 0.0
        if left_y is not None and left_y < LAVACON_PUSH_SAFETY_MARGIN_L_M:
            push_m += LAVACON_PUSH_SAFETY_MARGIN_L_M - left_y        # 좌측 콘 침범 → 우측(+)으로
        if right_y is not None and -right_y < LAVACON_PUSH_SAFETY_MARGIN_R_M:
            push_m -= LAVACON_PUSH_SAFETY_MARGIN_R_M - (-right_y)    # 우측 콘 침범 → 좌측(-)으로
        push_m *= LAVACON_PUSH_GAIN  # [2026-08-22b] 요청 반영 — 미는 세기 2배
        push_px = push_m * DL_PIXELS_PER_METER
        # [2026-08-22] 요청 반영 — 이번 틱에 실제로 밀렸는지(lavacon_bev의 push ROI/
        # 자홍 박스가 콘 침범을 잡아 push_m이 0이 아닌 경우)를 DA 디버그창에도 반영한다
        # (perc_lane()의 set_lavacon_push() 참고, lane_util.py draw_path()가 소비).
        # push_px(실제 밀린 양)도 같이 넘겨서, 디버그창이 밀리기 전(보라) 원본과 밀린 뒤
        # (주황) 경로를 나란히 그릴 수 있게 한다 — 게인 튜닝 시 밀림 정도를 눈으로 참고.
        self._lavacon_push_active = (push_m != 0.0)
        self._lavacon_push_px = push_px

        shifted_path = [(x + push_px, y) for x, y in self.lane_path]

        # [2026-08-17] _lane_steer()의 path=None 기본분기와 동일한 vehicle_x 산출(그대로
        # 복붙) — 여기선 path를 명시로 넘기므로 그 분기를 안 타 자동 계산이 안 된다.
        roi_w = getattr(self.lane_detector, 'roi_w', 0) or 0
        vehicle_x = getattr(self.lane_detector, 'vehicle_center_x', None)
        if vehicle_x is None:
            vehicle_x = roi_w / 2.0

        return self._lavacon_pure_pursuit_steer(path=shifted_path, vehicle_x=vehicle_x)

    # ── B2-고정장애물 회피 ──
    #   차선 2개 + 넘어도 되는 노란 중앙선 구조라, 방향은 '반대편 차선' 하나로 정해진다.
    #   좌우 선택 로직은 ObstacleAvoidance.decide_lane() 이 lane_side 로 처리한다.
    #   [2026-08-20] 요청 반영 — B2 실제 회피가 da 근접 컷(obstacle_cut_active)으로
    #   바뀌면서(run_behavior_fsm()의 Phase.OBSTACLE_ZONE 분기 참고) behavior_state가
    #   B2_OBSTACLE이 되는 경로 자체가 없어졌다 — 이 함수는 지금 호출되지 않는다.
    #   TargetPassing 기반 회피를 다시 쓰려면 그 분기에서 behavior_state를 B2_OBSTACLE로
    #   되돌리면 된다.
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
    # [2026-08-20] 요청 반영 — B3 실제 회피가 da 근접 컷(obstacle_cut_active, YOLO 차량+
    # 라이다 AND 트리거)으로 옮겨가면서(run_behavior_fsm()의 Phase.OBSTACLE_ZONE 분기 참고)
    # behavior_state가 B3_VEHICLE이 되는 경로 자체가 없어졌다 — 이 함수는 지금 호출되지
    # 않는다. TargetPassing 기반 추월을 다시 쓰려면 그 분기에서 behavior_state를
    # B3_VEHICLE으로 되돌리면 된다.
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
            # Phase.DONE 진입 즉시 신호등 재판독 시작 (예전엔 다음 바퀴 리셋 때까지 꺼져있었음)
            self.signal_red_on = self.signal_straight_on = self.signal_left_on = False
            self.signal_straight_confirmed = False
            self.signal_left_confirmed     = False
            self._sig_straight_cnt = 0
            self._sig_left_cnt     = 0
            self._signal_yolo_off = False
            self._signal_off_hold_cnt = None
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
          [2026-08-20] S0_SIGNAL "직진" 확정 시 항상 켜지므로(출발 포함), S0_SIGNAL/S3 구간에서만
          꺼져 있다.
        """
        self.perceive_all()                 # 1. 인지
        self._update_lap()                  #    바퀴 카운트(누적 yaw + 정지선)
        self.run_mission_fsm()              # 2. 판단(Mission)

        # [2026-08-23] DEBUG_VIZ_STEER는 여기 있었으나 behavior override/drive()의 각도
        # 클립(ANGLE_MAX, ANGLE_RATE_MAX) 이전 시점이라 "조향각" 줄이 실제 발행값과
        # 다를 수 있었다(디버그창엔 찍히는데 실제로는 다른 각도가 나가는 것처럼 보이는
        # 원인) — DEBUG_VIZ_LEFT_TURN과 동일하게 drive() 이후로 옮겼다(아래 참고).
        if DEBUG_VIZ_VESC:
            self._debug_viz_vesc()
        if DEBUG_VIZ_IMU:
            self._debug_viz_imu()
        if DEBUG_VIZ_AVOID_HOLD:
            self._debug_viz_avoid_hold()
        if DEBUG_VIZ_OBSTACLE_CUT:
            self._debug_viz_obstacle_cut()

        # [2026-08-24, 버그수정] `and not self._shortcut_exit_ramp_active` 추가 — 출구 램프는
        # mission_state가 S1_LANE_FOLLOW인 채로 도는 강제 조향 구간이라(입구 램프와 달리
        # S0_SIGNAL로 안 빠짐, _do_shortcut_exit_ramp_turn() 참고) 이 가드가 없으면 매틱
        # run_behavior_fsm()/apply_behavior_override()가 계속 돌면서, 만에 하나 B1이
        # (실제 콘 없이도 오검출로) 걸리면 apply_behavior_override()가 방금 _s1_lane_follow()
        # 가 찍어둔 강제 좌회전 각도/속도를 그대로 덮어써버린다 — 하필 "벽 피하려고 강제로
        # 꺾는" 그 순간에 조향이 다른 걸로 뺏기는 셈이라 위험도가 높다. 입구 램프는
        # mission_state!=S1_LANE_FOLLOW라 이 조건에 이미 자동으로 안 걸려서 문제가 없었다.
        if (ENABLE_BEHAVIOR and self.mission_state == MissionState.S1_LANE_FOLLOW
                and self._behavior_enabled and not self._shortcut_exit_ramp_active):
            self.run_behavior_fsm()         #    Behavior 상태 결정
            self.apply_behavior_override()  #    필요 시 조향/속도 덮어쓰기
        else:
            self.behavior_state = BehaviorState.B0_NORMAL   # OFF 구간은 항상 정상

        # pose_estimator는 behavior override까지 반영된 "최종" ctrl_angle로 갱신한다(차량이 실제로
        # 명령받는 조향각이 이거라서, 2026-08-06 LQR 브랜치에서 이식). v_mps=0.0 고정 상태
        # (vesc_speed_bridge 노드 미실행 등)에서도 그냥 안 움직이는 것으로 적분될 뿐 안전하게 동작한다.
        self.pose_estimator.update(self.v_mps, math.radians(self.ctrl_angle), 0.05)

        self.drive(self.ctrl_angle, self.ctrl_speed)   # 4. 발행
        if DEBUG_VIZ_STEER:
            # [2026-08-23] behavior override + drive()의 각도 클립(ANGLE_MAX,
            # ANGLE_RATE_MAX)까지 다 반영된 뒤(=이번 틱에 실제로 발행된 값)에 그린다.
            # _debug_viz_steer() 안의 "발행조향" 줄이 self._prev_angle_out(drive()가
            # 방금 갱신한 최종 클립값)을 쓰는 이유.
            self._debug_viz_steer()
        if DEBUG_VIZ_LEFT_TURN:
            # [2026-08-22i] "발행각도"가 이번 틱에 실제로 발행된 값이어야 하므로
            # drive() 이후에 그린다(위 VESC/IMU/OBSTACLE_CUT 창들은 behavior
            # override 이전 시점이라 이 창과 달리 최종 발행값과 어긋날 수 있음).
            self._debug_viz_left_turn()
        if DEBUG_LOG:                                    # 5. 디버그
            self._print_debug()


    # #########################################################
    # [6] 유틸/디버그
    # #########################################################
    # [2026-08-22, 요청 반영] _print_debug() 맨 앞 [mission|behavior|phase] 요약 줄의 가운데
    # 칸이 실제로는 거의 항상 "B0_NORMAL"로 고정돼 있어(§5.5 위 주석 — 실제 회피는
    # behavior_state가 아니라 obstacle_cut_active가 담당, run_behavior_fsm() 참고) 그 자리만
    # 보고는 지금 B1/B2/B3 중 어디에 있는지 알 수 없었다. self.behavior_state 자체를 바꾸면
    # apply_behavior_override()가 옛 TargetPassing 핸들러(_handle_lavacon()/
    # _handle_fixed_obstacle()/_handle_overtake())를 다시 불러버려 실제 주행이 깨지므로
    # (그래서 run_behavior_fsm()이 항상 B0_NORMAL로 고정해둔 것), 이 태그는 표시 전용이고
    # self.behavior_state는 건드리지 않는다 — run_behavior_fsm()과 동일한 판단 순서
    # (B1→B2→B3)를 그대로 따라간다. 뒤에 '+'가 붙으면 그 단계가 지금 실제로 감지/진행
    # 중이라는 뜻(대기 중이면 안 붙음) — obstacle_cut_debug 창의 _current_stage_label()과
    # 같은 근거(_lavacon_engaged/_b2_passed/_b3_passed/_obscut_zone_tag)를 쓴다.
    def _behavior_progress_tag(self):
        if self.mission_state != MissionState.S1_LANE_FOLLOW:
            return self.behavior_state.name
        if self.phase == Phase.LAVACON:
            return 'B1_LAVACON' + ('+' if self._lavacon_engaged else '')
        if self.phase == Phase.OBSTACLE_ZONE:
            pending = 'B2' if not self._b2_passed else ('B3' if not self._b3_passed else None)
            if pending is None:
                return 'B0_NORMAL'
            name = 'B2_OBSTACLE' if pending == 'B2' else 'B3_VEHICLE'
            return name + ('+' if self._obscut_zone_tag == pending else '')
        return 'B0_NORMAL'  # Phase.DONE — B1/B2/B3 모두 통과, 다음 교차로 대기

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
          lava   = 라바콘 박스 스택 페어링 중앙 편차(디버그/로깅용, 조향엔 미사용)
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
          [SIG](S0_SIGNAL 상태에서만 출력) 4구 신호등 원 검출이 어느 단계에서 막혔는지 진단:
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

        sig_flags = (f'R{int(self.signal_red_on)}L{int(self.signal_left_on)}S{int(self.signal_straight_on)} '
                     f'confirmS{int(self.signal_straight_confirmed)}({self._sig_straight_cnt}/{SIG_CONFIRM_FRAMES})'
                     f'L{int(self.signal_left_confirmed)}({self._sig_left_cnt}/{SIG_CONFIRM_FRAMES})')
        # [2026-08-21] B2/B3 실제 회피는 behavior_state(override 여부)가 아니라 da 근접 컷
        # (obstacle_cut_active)으로 이뤄져서 이 상태에선 behavior_state가 항상 B0_NORMAL로
        # 고정돼(run_behavior_fsm() Phase.OBSTACLE_ZONE 분기 참고) 로그만 보면 회피 중인지
        # 구분이 안 됐다 — obstacle_cut_debug(cv2 창)를 못 볼 때도 터미널에서 바로 보이게
        # 요약 줄에 추가.
        cut_desc = (f'{"ON" if self.obstacle_cut_active else "off"}'
                    f'({self.obstacle_cut_type})' if self.obstacle_cut_type != 'none'
                    else ("ON" if self.obstacle_cut_active else "off"))
        # [2026-08-22, 요청 반영] 가운데 칸을 self.behavior_state.name(거의 항상 B0_NORMAL)
        # 대신 _behavior_progress_tag()로 바꿔 B1→B2→B3 진행이 이 한 줄에서 바로 보이게 함.
        self.get_logger().info(
            f'[{self.mission_state.name}|{self._behavior_progress_tag()}|{self.phase.name}] '
            f'ang={self.ctrl_angle:+.1f} spd={self.ctrl_speed:.1f} cut={cut_desc}\n'
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
            f'masked_raw_pts={masked_pts} masked_min={masked_min_s}')


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
