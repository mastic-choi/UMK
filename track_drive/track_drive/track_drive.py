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
from .controller.obstacle_avoidance import ObstacleAvoidance, AvoidPhase
# vehicle_overtake.py 의 구 VehicleOvertake 는 더 이상 쓰지 않는다.
#   추월/회피가 규정상 같은 기동("타겟이 없는 차선으로 지나간다")이라
#   obstacle_avoidance.TargetPassing 한 클래스로 통합했다(moving 플래그로 구분).
from .planner.hybrid_astar import HybridAStar
from .planner.occupancy import OccupancyGrid
from .controller.stanley import StanleyController
from .controller.pure_pursuit import PurePursuitController
from .controller.lqr import LQRController
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
        self._img_front_t = 0.0   # 전방 카메라 최근 수신 시각(디버그: 카메라 살아있는지 나이로 판단)
        self._scan_t       = 0.0  # 라이다 최근 수신 시각(디버그용)

        # ── 인터페이스 변수 (인지 → 판단/제어) ──
        # [2-1 차선]
        self.lane_offset = 0.0      # 근거리 중앙편차(px, 우측+) — [4] 속도계획(turn_preview)에 계속 사용
        self.lane_valid  = False    # 차선 검출 여부
        self.lane_lookahead = 0.0   # 원거리(앞쪽) 편차 → 코너 진입 전 예측감속용
        self.lane_path = []         # 명시적 경로(ROI 픽셀좌표 웨이포인트, 가까운점→먼점)
                                     #   perc_lane()이 갱신, _lane_steer()가 조향각 계산에 사용
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
        self.left_clear     = True    # 좌측 차선 비었는지(추월 복귀 판단)
        self.right_clear    = True    # 우측 차선 비었는지(추월 이동 판단)
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
        # [2-4 라바콘]
        self.lavacon_offset = 0.0
        self.lavacon_done   = False
        self._lavacon_empty_cnt = 0   # 우측콘 연속 미검출 프레임 수(Phase 전환 디바운스)
        self.lavacon_left_detected  = False  # 좌측 라이다 클러스터 검출 여부(B1 진입 트리거용)
        self.lavacon_right_detected = False  # 우측 라이다 클러스터 검출 여부(B1 진입 트리거용)
        self.lavacon_trigger        = False  # 좌우 동시검출이 디바운스 프레임수만큼 유지되면 True
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

        # ── 판단/제어 상태 ──
        self.mission_state  = START_STATE
        self.behavior_state = BehaviorState.B0_NORMAL
        self.phase          = Phase.LAVACON     # S1 내부 진행 순서(라바콘부터 시작)
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
        self._s2_commit_t0  = None    # S2 신호 확정 후 물리적 분기 커밋 구간 시작 시각(None=미진입)
        self._s2_commit_dir = None    # 커밋 구간에서 진행 중인 방향 ('straight'/'left')
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
        # 튜닝값은 전부 config.py의 PP_*/LQR_* 에서 가져온다 — 클래스 자체의 기본값은
        # config.py를 안 거치고 pure_pursuit.py/lqr.py를 직접 쓸 때(단독 테스트 등)를
        # 위한 fallback이라, 여기서 명시적으로 넘기지 않으면 config.py를 고쳐도 반영이
        # 안 된다. STEERING_CONTROLLER로 아래 둘 중 _lane_steer()가 실제로 호출할 것을 고른다.
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
        )
        self.lqr = LQRController(
            # DL+BEV 조합일 때만 픽셀->미터 환산이 유효하다(DL_PIXELS_PER_METER는 BEV
            # 워프 목적캔버스 기준값이라 BEV가 꺼져있으면 의미가 없다) — 그 외(hough/
            # classic_cv, 혹은 DL_USE_BEV=False)엔 None을 넘겨 레거시 픽셀 게인 모드로
            # 자동 폴백한다(controller/lqr.py 상단 "좌표계" 주석 참고).
            pixels_per_meter=(DL_PIXELS_PER_METER
                               if (LANE_DETECTOR_BACKEND == 'dl' and DL_USE_BEV) else None),
            wheelbase_m=LQR_WHEELBASE_M,
            speed_mps=LQR_SPEED_MPS,
            heading_probe_m=LQR_HEADING_PROBE_M,
            min_path_m=LQR_MIN_PATH_M,
            wheelbase_gain=LQR_WHEELBASE_GAIN,
            speed_gain=LQR_SPEED_GAIN,
            q_lateral=LQR_Q_LATERAL,
            q_heading=LQR_Q_HEADING,
            r_steer=LQR_R_STEER,
            dt=LQR_DT,
            heading_probe_px=LQR_HEADING_PROBE_PX,
            angle_max_deg=ANGLE_MAX,
            alpha=LQR_ALPHA,
            min_path_px=LQR_MIN_PATH_PX,
        )

        self.path = None
        self.grid = None
        self.replan_count = 0
        self.goal = None


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
        #   유지된다 — 아래 control_loop()의 LQR_MIN_SPEED_MPS 가드가 이 상태에서 self.lqr 게인을
        #   건드리지 않게 막아준다(즉 이 값이 안 들어와도 LQRController 생성자 기본값
        #   speed_mps=LQR_SPEED_MPS로 조용히 폴백하지, v=0으로 게인이 퇴화하는 일은 없다).
        self.v_mps = 0.0
        self._vesc_t = None

        # ── 엔코더(VESC) 기반 pose 추정기 (localization/pose_estimator.py) ──
        #   위 self.vehicle_x/y/yaw(플래너용, 명령속도 적분 근사)와는 별개 컴포넌트. wheelbase_m은
        #   2026-08-06 실측값(LQR_WHEELBASE_M, config.py — LQR 컨트롤러와 같은 차량이므로 같은 값을
        #   공유). v_mps는 cb_vesc()가 갱신하는 self.v_mps를 control_loop()에서 매 주기 넣어준다.
        #   IMU를 yaw 소스로 쓰려면 set_yaw_source('imu') 후 update(..., imu_yaw=self.imu_yaw).
        self.pose_estimator = EncoderPoseEstimator(wheelbase_m=LQR_WHEELBASE_M)


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
                # onnxruntime 미설치, models/best.onnx 부재 등으로 초기화가 실패하면
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
        self.perc_lavacon()     # 라이다
        self.perc_lavacon_trigger()  # 라이다 (좌우 클러스터 동시검출 → B1_LAVACON 진입 트리거)
        self.perc_vehicle_trigger()  # 라이다 (전방 장애물 근접 → B3_VEHICLE 진입 트리거)
        self.perc_stopline()    # 비전

    # [2-1] 차선
    #   입력 self.img_front → 출력 self.lane_offset(우측+), self.lane_valid
    def perc_lane(self):
        if self.img_front is None:
            self.lane_valid = False
            return

        # hough_lane.py의 HoughLaneDetector를 사용하여 차선 인식 수행
        valid, offset, lookahead, lane_center, path, debug_img = self.lane_detector.detect(self.img_front)
        # DL 백엔드는 추론이 별도 스레드에서 도는데, cv2.imshow()/waitKey()는 스레드
        # 세이프하지 않아 반드시 메인 스레드(여기, control_loop 타이머 콜백)에서만 호출해야
        # 한다(dl_lane.DLLaneDetector.show_debug_windows() 주석 참고). hough/classic_cv
        # 백엔드는 이 메서드가 없으므로 getattr로 조용히 건너뛴다.
        #   속도 적응형 look-ahead 목표점(pure_pursuit.py last_target_xy)도 같이 넘겨서
        #   result 패널에 찍는다 — self._lane_steer()가 이번 틱에 아직 안 돌았으므로
        #   엄밀히는 직전 틱 값(0.05s 이내 오차, 디버깅 목적엔 무시 가능). LQR을 쓰면
        #   pure_pursuit이 안 갱신되므로 None이 넘어가 마커가 그려지지 않는다.
        lookahead_xy = lookahead_px = None
        if STEERING_CONTROLLER == 'pure_pursuit':
            lookahead_xy = self.pure_pursuit.last_target_xy
            lookahead_px = self.pure_pursuit.last_lookahead_px
        getattr(self.lane_detector, 'show_debug_windows', lambda *a, **k: None)(lookahead_xy, lookahead_px)

        self.lane_center = lane_center
        self.lane_valid = valid
        if valid:
            # 기존 제어 코드와 호환되도록 필터링 적용
            self.lane_offset = 0.7 * self.lane_offset + 0.3 * offset
            self.lane_lookahead = 0.5 * self.lane_lookahead + 0.5 * lookahead
        if path:
            # path가 빈 리스트면(이번 프레임 유효 슬라이스 2개 미만) 갱신하지 않고
            # 직전 경로를 유지한다 — lane_offset의 "무효 프레임엔 직전 값 유지" 폴백과 동일.
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
        self.lane_side = 1 if (sum(xs) / len(xs)) < (ld.roi_w / 2.0) else -1

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
        바로 FSM 전환(출발/좌회전)으로 새는 걸 막는다(라바콘/차량 트리거와 동일한 패턴)."""
        if self.img_front is None:
            return

        if self.mission_state in (MissionState.S0_WAIT_GREEN, MissionState.S2_INTERSECTION):
            self.signal_red_on, self.signal_straight_on, self.signal_left_on = \
                self.signal_detector.detect_s2(self.img_front)

            self._sig_straight_cnt = self._sig_straight_cnt + 1 if self.signal_straight_on else 0
            self._sig_left_cnt     = self._sig_left_cnt + 1 if self.signal_left_on else 0
            self.signal_straight_confirmed = self._sig_straight_cnt >= SIG_CONFIRM_FRAMES
            self.signal_left_confirmed     = self._sig_left_cnt >= SIG_CONFIRM_FRAMES

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
        LEFT_BLOCK_TH            = 8           # 좌측 차단 임계 (추월용)
        RIGHT_BLOCK_TH           = 5           # 우측 차단 임계
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
        self.left_clear  = int(np.count_nonzero(left_mask))  < LEFT_BLOCK_TH
        self.right_clear = int(np.count_nonzero(right_mask)) < RIGHT_BLOCK_TH

        if DEBUG_VIZ_LIDAR:
            PPM = 125         # 1m = 125px (표시 범위 2m)
            W, H = 500, 500
            EX, EY = 250, 250  # 자차 위치(캔버스 정중앙 — 전/후/좌/우 전체 360도가 다 보이도록)
            bev = np.zeros((H, W, 3), dtype=np.uint8)

            for d in range(1, 3):
                cv2.circle(bev, (EX, EY), d * PPM, (50, 50, 50), 1)
                cv2.putText(bev, f'{d}m', (EX + 4, EY - d*PPM + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

            def to_px(wx, wy): return (int(EX - wy*PPM), int(EY - wx*PPM))

            # 각도/인덱스 컴퍼스 (보정후 각도 기준, 0°=정면=위쪽, 반시계 방향)
            # 각 스포크의 'i라벨'은 현재 LIDAR_ANGLE_OFFSET_DEG 가정하에 그 방향이어야 할 원본 인덱스.
            for a_deg in range(0, 360, 45):
                raw_idx = int(round((a_deg + LIDAR_ANGLE_OFFSET_DEG) % 360))
                px_, py_ = to_px(1.9 * math.cos(math.radians(a_deg)), 1.9 * math.sin(math.radians(a_deg)))
                is_front = (a_deg == 0)
                spoke_col = (255, 220, 0) if is_front else (70, 70, 70)
                cv2.line(bev, (EX, EY), (px_, py_), spoke_col, 2 if is_front else 1)
                cv2.putText(bev, f'i{raw_idx}', (px_ - 14, py_),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, spoke_col, 1, cv2.LINE_AA)

            # 자기가림 마스킹 경계(BODY_LO~BODY_HI) — 이 두 스포크 사이 구간은 ranges가
            # 무조건 0으로 지워져서 안 찍힘(왼쪽이 안 보이는 원인).
            for body_idx, body_tag in ((BODY_LO, 'LO'), (BODY_HI, 'HI')):
                body_ang = body_idx - LIDAR_ANGLE_OFFSET_DEG
                bx_, by_ = to_px(1.9 * math.cos(math.radians(body_ang)), 1.9 * math.sin(math.radians(body_ang)))
                cv2.line(bev, (EX, EY), (bx_, by_), (200, 0, 200), 1)
                cv2.putText(bev, f'MASK_{body_tag}(i{body_idx})', (bx_ - 20, by_),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 200), 1, cv2.LINE_AA)

            cv2.rectangle(bev, to_px(FRONT_X_MIN, FRONT_Y_HALF), to_px(FRONT_X_MAX, -FRONT_Y_HALF), (0, 220, 220), 1)
            cv2.rectangle(bev, to_px(0.8, 1.5),  to_px(5.5,  0.7), (0, 220, 0),   1)
            cv2.rectangle(bev, to_px(0.8, -0.7), to_px(5.5, -1.5), (0, 140, 255), 1)

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
                if i % 10 == 0:   # 실제 원본 인덱스 번호(참값) — 컴퍼스 가정과 대조용
                    cv2.putText(bev, str(i), (sx + 3, sy - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.circle(bev, (EX, EY), 7, (255, 220, 0), -1)
            cv2.line(bev, (EX, EY), (EX, EY - 18), (255, 220, 0), 2)

            type_col = (0, 0, 255) if self.obstacle_front else (0, 255, 0)
            cv2.putText(bev, f'{self.obstacle_type.upper()} {self.obstacle_dist:.1f}m  {self.obstacle_side}  pts={front_cnt}',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, type_col, 1, cv2.LINE_AA)
            cv2.putText(bev, f'L:{"CLR" if self.left_clear else "BLK"}  R:{"CLR" if self.right_clear else "BLK"}',
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow('lidar_bev', bev)
            cv2.waitKey(1)

    # [2-4] 라바콘
    def perc_lavacon(self):
        self.lavacon_offset, self.lavacon_done = process_lavacon(self.lidar_ranges)

    # [2-4b] 라바콘 좌우 클러스터 검출 → B1_LAVACON 진입 트리거
    #   입력 self.lidar_ranges
    #   출력 lavacon_left_detected/right_detected, lavacon_trigger
    #   설계 의도: 라이다 포인트가 "존재"하는 것만으로는 벽·바닥 잡음과 구분이 안 되므로,
    #     인접 인덱스(=인접 각도)로 붙어있는 포인트 묶음(클러스터)이 좌/우 각각 최소 1개씩
    #     동시에 있어야 "라바콘 구간 진입"으로 인정한다. perc_obstacle()과 동일한 차체 마스킹/
    #     극좌표 변환 방식을 사용하되, ROI와 목적은 별개(장애물 회피용이 아니라 콘 게이트 진입 판단용)이므로
    #     여기서 독립적으로 계산한다. 좌우 동시검출이 LAVACON_TRIGGER_FRAMES 연속 유지되면 진입 확정(디바운스).
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

        # 좌우 클러스터 동시검출이 연속 프레임 유지되면 진입 확정(디바운스)
        if self.lavacon_left_detected and self.lavacon_right_detected:
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

        cv2.circle(bev, (EX, EY), 6, (255, 220, 0), -1)
        cv2.line(bev, (EX, EY), (EX, EY - 18), (255, 220, 0), 2)

        l_col = (0, 255, 0)   if self.lavacon_left_detected  else (0, 0, 255)
        r_col = (0, 140, 255) if self.lavacon_right_detected else (0, 0, 255)
        cv2.putText(bev, f'L pts={left_pts} run={left_run} (need run>=2)',  (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, l_col, 1, cv2.LINE_AA)
        cv2.putText(bev, f'R pts={right_pts} run={right_run} (need run>=2)', (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, r_col, 1, cv2.LINE_AA)
        cv2.putText(bev, f'trig={self._lavacon_trigger_cnt}/{LAVACON_TRIGGER_FRAMES}',
                    (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        masked_pts, masked_min = self._lavacon_mask_dbg
        masked_min_s = f'{masked_min:.2f}m' if masked_min >= 0 else 'N/A'
        cv2.putText(bev, f'masked(magenta) pts={masked_pts} min={masked_min_s}',
                    (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
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
        if USE_HYBRID_ASTAR_FOR_B2:
            return self.path is not None
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
            self._s2_commit_t0  = None
            self._s2_commit_dir = None
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
          2. 직진 초록(signal_straight_confirmed) → 커밋 구간(S2_COMMIT_T) 거쳐 S1 복귀
             + Behavior 활성화(라바콘부터 진행)
             좌회전 신호(signal_left_confirmed) → 커밋 구간 거쳐 좌회전 후 S3(지름길)
          3. 좌회전 진행 중이면 신호와 무관하게 완료 우선
          4. 커밋 구간(_s2_commit_t0)에서는 신호와 무관하게 직진만 유지 — 신호가 보이는
             지점과 실제 도로가 갈라지는 물리적 분기 지점이 떨어져 있고(config.py
             S2_COMMIT_T 주석 참고), 그 사이에 _lane_drive()(비전)를 켜면 분기가
             보이기 시작하는 순간 da가 반대쪽 갈래로 끌려간다(실측 재현됨). 신호로
             이미 확정된 방향이므로 이 구간은 비전을 아예 참조하지 않는다.
        """
        if self._turn_yaw_start is not None:
            self._do_left_turn(next_state=MissionState.S3_SHORTCUT)
            return

        if self._s2_commit_t0 is not None:
            self.ctrl_angle = 0.0
            self.ctrl_speed = APPROACH_SPEED
            if time.time() - self._s2_commit_t0 >= S2_COMMIT_T:
                commit_dir = self._s2_commit_dir
                self._s2_commit_t0  = None
                self._s2_commit_dir = None
                if commit_dir == 'straight':
                    self._behavior_enabled = True
                    self._stopline_cooldown_t = time.time() + STOPLINE_COOLDOWN
                    self._change_state(MissionState.S1_LANE_FOLLOW)
                else:
                    self._begin_left_turn()
            return

        self.ctrl_angle, self.ctrl_speed = 0.0, SPEED_STOP

        if self.signal_straight_confirmed:
            self._s2_commit_t0  = time.time()
            self._s2_commit_dir = 'straight'
        elif self.signal_left_confirmed:
            self._s2_commit_t0  = time.time()
            self._s2_commit_dir = 'left'

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
        self._turn_yaw_start = self.imu_yaw   # 플래그로만 사용 (None 여부 체크)
        self._turn_frame_cnt = 0
        self.get_logger().info(f'좌회전 시작 ({TURN_FRAMES}f)')

    def _do_left_turn(self, next_state):
        """무난한(후진 없는) 좌회전 후 next_state로 전환."""
        if next_state == MissionState.S3_SHORTCUT:
            trn_ang, trn_spd, trn_f = TURN_ANGLE, TURN_SPEED, TURN_FRAMES
        else:
            trn_ang, trn_spd, trn_f = TURN_EXIT_ANGLE, TURN_EXIT_SPEED, TURN_EXIT_FRAMES

        if self._turn_frame_cnt < trn_f:
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
        S1(차선주행) 재진입 후 Phase 순서(LAVACON→FIXED_OBSTACLE→VEHICLE→DONE)에 따라
        딱 하나의 Behavior만 활성화한다. Phase 전환은 각 핸들러가 완료 시점에 직접 수행.
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
        elif self.phase == Phase.FIXED_OBSTACLE:
            # TEST_DISABLE_B2_B3=True면 SAFETY_DIST 트리거 검사(아래 triggered 계산)를 아예 안 하고
            # 바로 리턴 — 장애물이 실제로 잡혀도 B2_OBSTACLE로 안 넘어가고 B0로 고정되어
            # _s1_lane_follow의 일반 차선 PID가 계속 돎(placeholder 회피 기동이 실행 안 됨).
            if TEST_DISABLE_B2_B3:
                self.behavior_state = BehaviorState.B0_NORMAL   # 테스트 범위 제한: B2 트리거 무시
                return
            # 트리거에서 obstacle_type=='fixed' 조건을 뺐다.
            #   [6]에서 분류 기준을 실폭으로 바꾸면서, 차선을 막고 선 '차량'(폭 넓음)은
            #   'vehicle' 로 분류된다. 옛 조건대로면 이 Phase 에서 영영 트리거되지 않아
            #   Phase.VEHICLE 로 넘어가지도 못하고 교착된다.
            #   회피 기동 자체는 고정물이든 차량이든 '반대편 차선으로 비킨다'로 동일하므로
            #   여기서는 '앞을 막고 있는가'만 본다(순서는 Phase 가 이미 강제하고 있음).
            triggered = (self.obstacle_front
                         and self.obstacle_dist < SAFETY_DIST
                         and self._maneuver_allowed)
            self.behavior_state = (BehaviorState.B2_OBSTACLE
                                    if (triggered or self._obstacle_active)
                                    else BehaviorState.B0_NORMAL)
        elif self.phase == Phase.VEHICLE:
            # 위와 동일한 이유로 트리거 검사를 건너뛰고 B0로 고정(placeholder 추월 기동 비활성화)
            if TEST_DISABLE_B2_B3:
                self.behavior_state = BehaviorState.B0_NORMAL   # 테스트 범위 제한: B3 트리거 무시
                return
            # 진입 판정은 perc_vehicle_trigger()의 라이다 디바운스 결과를 사용.
            # 한번 진입한 뒤(_overtake_active)에는 기존과 동일하게 라이다 단독으로 유지/종료 판단.
            self.behavior_state = (BehaviorState.B3_VEHICLE
                                    if ((self.vehicle_trigger and self._maneuver_allowed)
                                        or self._overtake_active)
                                    else BehaviorState.B0_NORMAL)
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
        STEERING_CONTROLLER가 'pure_pursuit'일 때만 적용한다 — lqr.py는 curvature가 아니라
        횡오차/헤딩오차 상태로 도는 별개 모델이라 이 반경 개념이 안 맞는다.

        [2026-08-06] curvature는 self.pure_pursuit.last_curvature(이번 틱의 순간값)가 아니라
        _lane_drive()가 매 틱 갱신하는 self._corner_signal(조향각의 signed EMA)에서 역산한다 —
        이유는 _lane_drive() 상단 주석 참고(진동을 매번 급코너로 오인해 감속하는 문제)."""
        if STEERING_CONTROLLER != 'pure_pursuit':
            return 1.0
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

        # [2026-08-06] 코너 감속 판단은 "지금 순간 조향각이 얼마나 큰가"가 아니라 "최근 한동안
        # 같은 방향으로 얼마나 꺾여 있었는가"를 봐야 한다. pure_pursuit은 구조상 좌우로 조금씩
        # 진동("와리가리")하는데, turn_now를 매 틱 abs(ctrl_angle)로 그대로 계산하면 진동의
        # 절반(방향이 바뀌는 쪽)마다 급코너로 오인해 아래 3제곱 감속식과
        # _corner_radius_speed_scale()이 실제 코너가 아닌데도 속도를 깎는다(실차에서 재현:
        # 진동할 때마다 속도가 팍팍 줄었다 늘었다 함). self._corner_signal은 ctrl_angle을
        # signed(부호 유지) EMA로 누적한 값이다 — 진동처럼 부호가 계속 바뀌면 서로 상쇄돼 0
        # 근처로 수렴하고, 실제 코너처럼 한 방향으로 계속 꺾이면 EMA가 실제 각도로 수렴한다.
        # abs()는 반드시 이 EMA를 취한 "이후"에 적용해야 한다 — abs(ctrl_angle)을 먼저 평균내면
        # 부호 정보가 지워져서 진동도 그대로 다 더해져 상쇄 효과가 없어진다.
        self._corner_signal = (CORNER_SIGN_EMA_ALPHA * self.ctrl_angle
                                + (1.0 - CORNER_SIGN_EMA_ALPHA) * self._corner_signal)
        turn_now     = min(1.0, abs(self._corner_signal) / ANGLE_MAX)
        turn_preview = min(1.0, abs(self.lane_lookahead) / LANE_LOOKAHEAD_REF)
        turn_for_speed = max(turn_now, turn_preview * 0.3)
        target_speed = max(SPEED_CORNER_MIN,
                           SPEED_NORMAL * (1.0 - 0.90 * turn_for_speed ** 3))
        # 코너 진입(회전반경 감소) 시 추가 감속 — 기존 turn_for_speed 기반 감속과는 독립적으로
        # 계산해서 더 낮은 쪽을 쓴다(대체가 아니라 추가 안전판).
        target_speed = max(SPEED_CORNER_MIN, target_speed * self._corner_radius_speed_scale())
        speed_ratio = min(1.0, self._prev_speed / SPEED_NORMAL)
        corner_decay = CORNER_HOLD_DECAY_LO + (CORNER_HOLD_DECAY_HI - CORNER_HOLD_DECAY_LO) * speed_ratio
        self._corner_hold = max(turn_now, self._corner_hold * corner_decay)
        accel_step = SPEED_ACCEL_STEP * max(0.25, 1.0 - self._corner_hold)
        if target_speed > self._prev_speed + accel_step:
            target_speed = self._prev_speed + accel_step
        self.ctrl_speed = target_speed
        self._prev_speed = target_speed

    def _lane_steer(self):
        """self.lane_path(DL/classic_cv/hough 백엔드가 만든 ROI 픽셀좌표 경로, 가까운점→
        먼점)를 STEERING_CONTROLLER로 고른 컨트롤러(pure_pursuit.py 또는 lqr.py)로
        추종해 조향각(도)을 계산한다. 차량 기준점은 항상 ROI 하단 중앙(roi_w/2, path의
        가장 가까운 점의 y좌표)으로 둔다 — path[0].y는 lane_util._fit_and_sample_path()가
        self.roi_h로 샘플링해둔 값이라 별도로 백엔드별 roi_h를 조회할 필요가 없다.
        경로가 비어있으면(첫 프레임, 혹은 roi_w를 아직 모르는 백엔드) 직전 조향각을
        그대로 유지한다 — 두 컨트롤러의 control()이 내부적으로 동일하게 처리.
        (구 이름 _pure_pursuit_steer — STEERING_CONTROLLER로 lqr도 고를 수 있게 되며
        컨트롤러 중립적인 이름으로 변경)
        pure_pursuit은 속도 적응형 lookahead 때문에 speed(_prev_speed 근사치)를 받지만,
        lqr은 자체 speed_gain 튜닝값을 쓰고 control()에 speed 인자가 없다(controller/lqr.py
        참고) — 그래서 여기서 컨트롤러별로 분기해서 호출한다(공통 kwarg로 합칠 수 없음)."""
        controller = self.lqr if STEERING_CONTROLLER == 'lqr' else self.pure_pursuit
        roi_w = getattr(self.lane_detector, 'roi_w', 0) or 0
        if not self.lane_path or not roi_w:
            return controller.prev_steer_deg
        vehicle_xy = (roi_w / 2.0, self.lane_path[0][1])
        if STEERING_CONTROLLER == 'lqr':
            return self.lqr.control(self.lane_path, vehicle_xy)
        return self.pure_pursuit.control(self.lane_path, vehicle_xy, speed=self._prev_speed)

    # [DEBUG_VIZ_STEER] 조향 컨트롤러가 이번 주기에 "새로 계산"했는지(초록/현재값 반영)
    # "직전 조향각을 그대로 유지"했는지(주황/직전값 유지)를 별도 창으로 바로 확인.
    # cv2 기본폰트는 한글을 못 그려서 kr_text.put_text_kr_multi(PIL 기반)로 그린다
    # (kr_text.py 상단 주석 참고). control_loop()에서 매 주기 호출 — 이 함수를 부르는
    # 시점(run_mission_fsm 직후, behavior override 이전)이 중요: 표시값은 self.ctrl_angle이
    # 아니라 controller.prev_steer_deg를 쓴다 — B1/B2/B3 behavior가 나중에 self.ctrl_angle을
    # 덮어써도(apply_behavior_override) 이 창은 항상 "차선 조향 컨트롤러 자체"의 상태를
    # 보여주기 위함이다.
    def _debug_viz_steer(self):
        controller = self.lqr if STEERING_CONTROLLER == 'lqr' else self.pure_pursuit
        held = getattr(controller, 'held', False)

        canvas = np.full((200, 380, 3), 30, dtype=np.uint8)
        # 주황 = 직전값 유지(경로 부족/노이즈로 신뢰 못 함), 초록 = 현재값 반영(정상 계산됨)
        status_color = (0, 140, 255) if held else (0, 200, 0)
        status_text = '직전값 유지 (경로 부족/노이즈)' if held else '현재값 반영'
        controller_name = 'LQR' if STEERING_CONTROLLER == 'lqr' else 'Pure Pursuit'

        lines = [
            (f'컨트롤러: {controller_name}', (10, 8), (255, 255, 255), 20, f'Controller: {controller_name}'),
            (f'상태: {status_text}', (10, 40), status_color, 20,
             f'Status: {"HOLD (prev)" if held else "LIVE (fresh)"}'),
            (f'조향각: {controller.prev_steer_deg:+.1f}도', (10, 72), (255, 255, 255), 20,
             f'Steer: {controller.prev_steer_deg:+.1f}deg'),
        ]
        if STEERING_CONTROLLER == 'lqr':
            e_y = getattr(controller, 'last_e_y', None)
            e_psi = getattr(controller, 'last_e_psi', None)
            if e_y is not None:
                lines.append((f'횡오차 e_y: {e_y:+.1f}px', (10, 104), (255, 255, 255), 20,
                               f'e_y: {e_y:+.1f}px'))
            if e_psi is not None:
                lines.append((f'헤딩오차 e_psi: {math.degrees(e_psi):+.1f}도', (10, 136), (255, 255, 255), 20,
                               f'e_psi: {math.degrees(e_psi):+.1f}deg'))

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

        gain_fed = abs(self.v_mps) >= LQR_MIN_SPEED_MPS
        canvas = np.full((160, 380, 3), 30, dtype=np.uint8)
        lines = [
            (f'VESC 연동: {text_kr}', (10, 8), color, 16, f'VESC link: {text_en}'),
            (f'v_mps: {self.v_mps:+.3f} m/s', (10, 44), (255, 255, 255), 20,
             f'v_mps: {self.v_mps:+.3f} m/s'),
            (f'LQR 게인 반영: {"O" if gain_fed else "X (LQR_MIN_SPEED_MPS 미만)"}',
             (10, 76), (255, 255, 255) if gain_fed else (150, 150, 150), 16,
             f'LQR gain fed: {"YES" if gain_fed else "NO"}'),
            ('토픽: /vesc_speed_erpm (std_msgs/Float32, vesc_speed_bridge.py 경유)',
             (10, 108), (180, 180, 180), 12, 'topic: /vesc_speed_erpm'),
        ]
        put_text_kr_multi(canvas, lines)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), color, 3)

        cv2.imshow('vesc_debug', canvas)
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
            self._handle_fixed_obstacle()
        elif self.behavior_state == BehaviorState.B3_VEHICLE:
            self._handle_overtake()
        # B0_NORMAL: 아무것도 안 함(Mission 출력 그대로)

    # ── B1-라바콘: 보로노이 편차 기반 P제어 ──
    def _handle_lavacon(self):
        """
        Phase.LAVACON 동안 항상 활성(트리거 조건 없음).
        우측 콘이 연속 LAVACON_DONE_FRAMES 프레임 미검출되면 고정장애물 구간으로 전환.
        """
        self.ctrl_angle = self.lavacon_offset * LAVACON_KP
        self.ctrl_speed = SPEED_LAVACON

        if self.lavacon_done:
            self._lavacon_empty_cnt += 1
            if self._lavacon_empty_cnt >= LAVACON_DONE_FRAMES:
                self._lavacon_empty_cnt = 0
                self._pid_prev_error = 0.0
                self._pid_integral   = 0.0
                self._lavacon_engaged = False   # B1 진입 latch 해제 (구간 재진입 대비)
                self.phase = Phase.FIXED_OBSTACLE
                self.get_logger().info('[LAVACON] 구간 통과 완료 → 고정장애물 구간')
        else:
            self._lavacon_empty_cnt = 0

    # ── B2-고정장애물 회피 ──
    #   차선 2개 + 넘어도 되는 노란 중앙선 구조라, 방향은 '반대편 차선' 하나로 정해진다.
    #   좌우 선택 로직은 ObstacleAvoidance.decide_lane() 이 lane_side 로 처리한다.
    def _handle_fixed_obstacle(self):
        if USE_HYBRID_ASTAR_FOR_B2:
            self._handle_fixed_obstacle_astar()
            return

        self._run_passing(self.obstacle_controller, 'B2',
                          done_next_phase=Phase.VEHICLE)

    # ── B2 대안: Hybrid A* 경로계획 방식 (USE_HYBRID_ASTAR_FOR_B2=True 일 때만) ──
    #   구조화된 2차선 환경에는 과한 방식이라 기본은 비활성. 비교/보존용으로 남겨둔다.
    #   ※ 이 경로는 아직 명령속도(모터단위)를 m/s 로 적분하는 단위 불일치가 남아있다
    #     — B-2(METERS_PER_SPEED_UNIT) 실측 후에 정리할 것.
    def _handle_fixed_obstacle_astar(self):

        if self.path is None:
            # Local Occupancy 생성
            self.grid = self.occupancy.build(
                self.lidar_ranges
            )
            # Local Frame 시작점 — 이번 replan 시점을 로컬 pose 원점으로 리셋
            start = PlannerNode(0.0,0.0,0.0)
            self.vehicle_x = 0.0
            self.vehicle_y = 0.0
            self.vehicle_yaw = 0.0
            self._plan_ref_yaw = self.imu_yaw
            self._plan_last_t = time.time()
            # Goal 생성
            if self.goal is None:
                self.goal = self.planner.make_goal(
                    self.obstacle_dist,
                    self.left_clear,
                    self.right_clear
                )
            # Hybrid A*
            self.path = self.planner.plan(start,self.goal,self.grid)

        if not self.path:
            self.path = None
            self.goal = None
            self.ctrl_speed = 0
            return

        # 로컬 pose 갱신 — yaw는 IMU 실측값 기준(_yaw_delta), x/y는 명령속도(ctrl_speed,
        # 미보정 단위) 적분 근사치. 정식 오도메트리가 생기기 전까지의 임시 추정값이다.
        now = time.time()
        dt = now - self._plan_last_t
        self._plan_last_t = now

        self.vehicle_yaw = self._yaw_delta(self._plan_ref_yaw)
        self.vehicle_speed = self.ctrl_speed
        self.vehicle_x += self.vehicle_speed * math.cos(self.vehicle_yaw) * dt
        self.vehicle_y += self.vehicle_speed * math.sin(self.vehicle_yaw) * dt

        # Stanley
        steer, speed = self.stanley.control(
            self.vehicle_x,
            self.vehicle_y,
            self.vehicle_yaw,
            self.vehicle_speed,
            self.path
        )

        self.ctrl_angle = math.degrees(steer)
        self.ctrl_speed = speed

        self.replan_count += 1
        if self.replan_count >= 20:
            self.path = None
            self.goal = None
            self.replan_count = 0
            return

        if self.stanley.goal_reached(self.vehicle_x, self.vehicle_y, self.path):
            self.path = None
            self.goal = None
            self.replan_count = 0

            self.stanley.reset()

            self.phase = Phase.VEHICLE
            self.behavior_state = BehaviorState.B0_NORMAL

            self._pid_prev_error = 0.0
            self._pid_integral = 0.0

        

        if DEBUG_PLANNER:
            cv2.imshow(
                "Occupancy",
                cv2.resize(self.grid,None,
                           fx=4,
                           fy=4)
                )

            cv2.waitKey(1)

    

    # ── B3-방해차량 추월 ──★재설계 예정(임시 placeholder) ──
    # target_lane 반영 수정
    # 회피 후 복귀하는 로직 추가 : idle -> avoid -> idle+Phase.VEHICLE => idel -> avoid -> return -> idle
    def _handle_overtake(self):
        """방해차량 추월. 규정상 '방해차량이 주행하지 않는 차선'으로 지나가야 하고,
        그 차량이 1·2차선을 오가므로 매 프레임 타겟 횡위치를 다시 보고 필요하면
        진행 방향을 바꾼다(moving=True 로 생성된 컨트롤러가 이걸 처리)."""
        self._run_passing(self.vehicle_controller, 'B3',
                          done_next_phase=Phase.DONE)

    # ── B2/B3 공통 실행부 ──
    def _run_passing(self, controller, tag, done_next_phase):
        steer_offset, speed, done, status = controller.update(
            obstacle_front=self.obstacle_front,
            obstacle_dist=self.obstacle_dist,
            obstacle_y=self.obstacle_y,
            lane_offset=self.lane_offset,
            lane_lookahead=self.lane_lookahead,
            lane_side=self.lane_side,
            left_clear=self.left_clear,
            right_clear=self.right_clear,
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
            self.phase = done_next_phase
            self._pid_prev_error = 0.0
            self._pid_integral = 0.0
            self.get_logger().info(f'[{tag}] 통과 완료 → {done_next_phase.name}')


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
        # VESC 실측 속도를 LQR 게인에 반영(2026-08-06 LQR 브랜치에서 이식) — run_mission_fsm()보다
        # 먼저 해야 이번 틱의 _lane_steer()가 최신 속도로 계산된 게인을 쓴다. LQR_MIN_SPEED_MPS
        # 미만(정지/거의정지, 혹은 vesc_speed_bridge 노드 미실행으로 self.v_mps가 계속 0.0)이면
        # 건너뛰고 직전 게인을 유지한다(cb_vesc()/LQR_MIN_SPEED_MPS 주석 참고).
        if abs(self.v_mps) >= LQR_MIN_SPEED_MPS:
            self.lqr.set_speed_mps(self.v_mps)
        self.run_mission_fsm()              # 2. 판단(Mission)

        if DEBUG_VIZ_STEER:
            # behavior override 이전 시점 — 차선 조향 컨트롤러 자체의 상태를 보여준다
            # (_debug_viz_steer() 상단 주석 참고).
            self._debug_viz_steer()
        if DEBUG_VIZ_VESC:
            self._debug_viz_vesc()

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
          [LANE] 차선편차px(검출여부) / obs = 라이다 전방장애물(거리m,좌우,타입)
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
            f'  [LANE] lane={self.lane_offset:+.1f}({int(self.lane_valid)}) '
            f'side={"R" if self.lane_side >= 0 else "L"}차선 '
            f'obs={self.obstacle_front}({self.obstacle_dist:.1f}m,w={self.obstacle_width:.2f}m,{self.obstacle_type}) '
            f'lava={self.lavacon_offset:+.2f}(done={int(self.lavacon_done)})\n'
            f'  [TRIG] trigL={self._lavacon_trigger_cnt}/{LAVACON_TRIGGER_FRAMES}'
            f'(L{int(self.lavacon_left_detected)}R{int(self.lavacon_right_detected)}) '
            f'trigV={self._vehicle_trigger_cnt}/{VEHICLE_TRIGGER_FRAMES}\n'
            f'  [LAVA-ROI] L pts={lava_lp} run={lava_lrun}(need>=2) '
            f'R pts={lava_rp} run={lava_rrun}(need>=2) '
            f'masked_raw_pts={masked_pts} masked_min={masked_min_s}'
            f'{sig_line}')


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
