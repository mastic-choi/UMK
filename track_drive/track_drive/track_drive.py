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
from enum import Enum
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from std_msgs.msg import Float32MultiArray
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from .perc_lavacon import process_lavacon, HALF_WIDTH_DEFAULT
from .lane_util import PolyLaneNetDetector, DEFAULT_WEIGHTS_PATH
from .perc_floor import check_stopline
from .traffic_signal import SignalDetector


# #############################################################
# [0] 설정 (Config)
# #############################################################

class MissionState(Enum):
    S0_WAIT_GREEN   = 0  # 3구 신호등 초록불 대기 후 출발
    S1_LANE_FOLLOW  = 1  # 차선인식 주행 (라바콘·고정장애물·추월 Behavior를 이 상태 안에서 처리)
    S2_INTERSECTION = 2  # 4구 신호등 교차로 (정지→라이다 경로판단→직진/좌회전)
    S3_SHORTCUT     = 3  # 지름길 (직진, 끝에서 좌회전)
    S4_FINISH       = 4  # 종료

class BehaviorState(Enum):
    B0_NORMAL   = 0  # Mission(차선주행) 출력 그대로
    B1_LAVACON  = 1  # 라바콘 구간 주행 (Phase.LAVACON일 때, 좌우 라이다 클러스터 동시검출 트리거로 활성)
    B2_OBSTACLE = 2  # 고정장애물 회피 (Phase.FIXED_OBSTACLE일 때, 감지 시 활성) — ★재설계 예정
    B3_VEHICLE  = 3  # 방해차량 추월   (Phase.VEHICLE일 때, 감지 시 활성)      — ★재설계 예정

# S1(차선주행) 내부 진행 순서 — 순서 고정(라바콘→고정장애물→방해차량→완료), 순차 전용(우선순위 판단 불필요)
class Phase(Enum):
    LAVACON        = 0
    FIXED_OBSTACLE = 1
    VEHICLE        = 2
    DONE           = 3  # 모든 Behavior 미션 완료 — 이후 계속 B0로 일반 차선주행

# ── 속도·각도 상수 ──
SPEED_NORMAL  = 5.0   # 차선주행(S1) 기본속도
                       # 출처: KUAC_2024-main lane_detection/src/lane_detection.py self.motor=30(고정)
                       #   기존 20.0 → 30.0. 모터/조향 스케일이 같은 xycar 플랫폼인지 미확인, 실차 저속 테스트 우선 권장
SPEED_LAVACON = 2.5    # KUAC_2024 라바콘 속도(12~30, fast/safe 라벨 앞뒤가 안 맞아 신뢰도 낮음) 참고만 하고 미반영
SPEED_STOP    = 0.0
ANGLE_MAX     = 100.0
SPEED_ACCEL_STEP = 0.85  # 가속 속도제한(주기당 최대 증가량)
CORNER_HOLD_DECAY_LO = 0.92  # 저속 시 코너 hold 감쇠 (빠른 회복)
CORNER_HOLD_DECAY_HI = 0.97  # 고속 시 코너 hold 감쇠 (느린 회복, 연속코너 대응)

# ── 라이다 장착 각도 보정 (재실측 2026-07-22) ──
# 차량 정면에 사람을 세우고 라이다 BEV 디버그 창(각도/인덱스 컴퍼스)으로 확인한 결과,
# 자기가림 마스크를 끄고 다시 보니 정면 클러스터가 인덱스 80에서 찍힘 — 즉 라이다 인덱스(도)
# 원점이 차량 정면 기준 80도 어긋나 있다(자기가림 마스킹과는 별개 문제).
# perc_obstacle()/perc_lavacon_trigger()에서 극좌표→직교좌표 변환 시
# "deg = 인덱스(도) - LIDAR_ANGLE_OFFSET_DEG" 로 이 오프셋을 빼서 보정한다.
#   보정 후 각도 약속: 0=정면, 90=좌측, 180=후방, 270(-90)=우측 (반시계 방향)
# perc_lavacon.py의 동일 상수와 반드시 값을 일치시킬 것.
LIDAR_ANGLE_OFFSET_DEG = 80.0  # 재실측(2026-07-22): i90이 아니라 i80이 정면

# 차체 자기가림 마스킹(BODY_LO~BODY_HI) 전체 스위치.
BODY_MASK_ENABLED = True  # 최종 확정(2026-07-22)

# ── 튜닝 파라미터 (한곳에 모음) ──
# 차선 PID — 출처: KUAC_2024-main lane_detection/src/lane_detection.py PID(safe모드)
#   safe: kp=0.70, ki=0.0008, kd=0.15 / fast: kp=0.78, ki=0.0005, kd=0.405 (참고, safe값 채택)
#   기존 0.14/0.0/1.40 대비 Kp 5배↑ Kd 9배↓ 로 큰 변화 — 실차 저속에서 오실레이션 여부 먼저 확인 후 정속 테스트 권장
LANE_KP, LANE_KI, LANE_KD = 0.70, 0.0008, 0.15  # 차선 PID
LANE_SIDE = 1               # 주행 차선: +1=노란선 오른쪽(우측차선), -1=왼쪽
LANE_CORNER_BOOST = 1.8    # 코너(큰 offset) 조향 가중
LANE_CORNER_REF   = 120.0  # 이 offset(px)에서 가중 최대
LANE_CORNER_MIN   = 40.0   # 코너 가중 시작 임계(px)
LANE_DEADZONE     = 40.0   # 중앙 데드존(px)
LANE_PREVIEW      = 0.38   # 코너 예측 조향 비중(0~1)
LANE_LOOKAHEAD_REF = 220.0  # 예측감속 최대가 되는 lookahead 편차(px)

LAVACON_KP   = 210.0
LAVACON_DONE_FRAMES = 80   # 우측콘 미검출이 연속 N프레임(20Hz→약 4초) 쌓이면 Phase 전환(순간누락 디바운스)
LAVACON_TRIGGER_FRAMES = 5   # 좌우 클러스터 동시검출이 연속 N프레임 쌓이면 B1_LAVACON 진입 확정(디바운스)
SAFETY_DIST      = 5.0    # B2(고정장애물) 발동 거리(m) — ★재설계 시 재검토
OVERTAKE_TRIGGER = 6.5    # B3(방해차량) 발동 거리(m)   — ★재설계 시 재검토
VEHICLE_TRIGGER_FRAMES = 5     # 라이다 단독검출 연속 N프레임이면 B3_VEHICLE 진입 확정(디바운스)

# ── 좌회전 공통 (실차 전환: 후진 없이 무난한 좌회전으로 단순화) ──
# 시뮬 전용이던 "후진 후 최대조향 좌회전" 방식 폐기 — 실차 튜닝 필요한 임시값
TURN_ANGLE       = -60.0   # [진입] S2 교차로 → S3 지름길 좌회전 조향각
TURN_SPEED       = 15.0    # [진입] 좌회전 속도
TURN_FRAMES      = 40      # [진입] 좌회전 유지 프레임 수 (20Hz 기준, 실차 튜닝 필요)
TURN_EXIT_ANGLE  = -60.0   # [진출] S3 지름길 → S1 차선주행 좌회전 조향각
TURN_EXIT_SPEED  = 15.0    # [진출] 좌회전 속도
TURN_EXIT_FRAMES = 40      # [진출] 좌회전 유지 프레임 수

SHORTCUT_MIN_T = 3.0   # 지름길 진입 후 끝감지 활성화까지 최소 주행시간(s, 오판 방지)
SHORTCUT_MAX_T = 15.0  # 지름길 최대 주행시간(s, 끝 못 찾을 때 강제 탈출 백업)
STOPLINE_TH    = 0.95  # 정지선 판정: 한 행 흰색비율 임계
STOPLINE_COOLDOWN = 3.0 # 상태 복귀 후 이 시간(s)간 정지선 재감지 무시(따다닥 전환 방지)
APPROACH_SPEED = 2.0    # [진입] 정지선 감지 후 S2 진입 전 감속 속도
APPROACH_TIME  = 1.0    # [진입] 감속 유지 시간(s)
APPROACH_EXIT_SPEED = 2.0  # [진출] S3 탈출 정지선 감지 후 감속 속도
APPROACH_EXIT_TIME  = 1.0  # [진출] 감속 유지 시간(s)

# 장애물회피 판단
AVOID_OFFSET = 100      # 차선 중앙에서 좌우로 이동할 거리(px)
RETURN_THRESHOLD = 10 
# ── 신호등 ROI/임계값은 traffic_signal.py(SignalDetector)로 이관 — 여기서 중복 정의하지 않음 ──

# ── Behavior 게이팅 ──
# 라바콘·고정장애물·방해차량 미션은 전부 S1(차선주행)에서만 나온다.
# 단, S1에는 두 번 진입한다: ①S0 직후(교차로 가기 전, 순수 주행만) ②S2 교차로 "직진" 선택 후 복귀(여기서만 Behavior 작동).
# → _behavior_enabled 로 ①/② 를 구분한다.


# ── 개발/테스트 플래그 ──
START_STATE     = MissionState.S1_LANE_FOLLOW
ENABLE_BEHAVIOR = True
DEBUG_LOG       = True
DEBUG_PERIOD    = 0.5
DEBUG_VIZ       = True  # 신호등/4구 디버그 창
DEBUG_VIZ_LANE  = True  # 차선 슬라이딩윈도우 디버그 창
DEBUG_VIZ_LIDAR = True  # 라이다 BEV 장애물 감지 디버그 창
DEBUG_VIZ_LAVACON = False  # 라바콘 트리거 좌우 클러스터 BEV 디버그 창

# ── 실차 테스트 범위 제한 ──
#   지금 단계에서 실차로 검증 가능한 건 딱 세 가지: ①신호등 인식 후 출발(S0) ②차선주행(S1)
#   ③라바콘 주행(B1). 나머지(S2 교차로/S3 지름길, B2 고정장애물/B3 방해차량)는 아직
#   실차 미검증(좌회전 각도·속도 placeholder, B2/B3는 감속-대기 placeholder라 실제 회피/추월 기동이 없음)이라
#   테스트 중 의도치 않게 발동하면 위험할 수 있어 아래 두 플래그로 강제로 꺼둔다.
#   → 전체 미션을 테스트할 준비가 되면(좌회전 튜닝 끝, B2/B3 실기동 구현 끝) 둘 다 False로 되돌릴 것.
TEST_DISABLE_INTERSECTION = True
#   True: _s1_lane_follow()에서 정지선(self.stopline)을 감지해도 감속→S2_INTERSECTION 전환을 아예 안 함.
#         즉 정지선을 계속 밟고 지나가도 무시하고 차선주행만 계속함(교차로 좌회전 로직 자체가 안 걸림).
#   False: 원래대로 정지선 감지 시 감속 후 S2로 정상 전환.
TEST_DISABLE_B2_B3 = True
#   True: run_behavior_fsm()에서 Phase가 FIXED_OBSTACLE/VEHICLE로 넘어가도 장애물/차량 감지 트리거
#         검사 자체를 건너뛰고 behavior_state를 무조건 B0_NORMAL로 고정 → 결과적으로 B1(라바콘) 끝난
#         뒤에도 계속 일반 차선주행만 함(장애물이 실제로 잡혀도 회피/추월 기동이 안 걸림).
#   False: 원래대로 SAFETY_DIST/OVERTAKE_TRIGGER 트리거 검사해서 B2/B3 정상 발동.
TEST_FORCE_BEHAVIOR = True
#   True: _behavior_enabled를 시작부터 강제로 True로 켠다.
#         원래 _behavior_enabled는 S2 교차로에서 "직진" 신호를 받아야만 켜지는데(딱 한 곳),
#         TEST_DISABLE_INTERSECTION=True로 S2 진입 자체가 막혀 있으면 그 경로가 사라져서
#         B1(라바콘)을 포함한 모든 Behavior가 영원히 비활성 상태가 된다(위 ③과 모순).
#         이 플래그는 S2/S3 로직을 건드리지 않고 게이트 변수만 우회하므로,
#         교차로를 끈 채로 라바콘(B1)만 독립적으로 실차 검증할 수 있다.
#   False: 원래대로 S2 교차로 직진 신호를 받아야만 Behavior가 켜짐.
#   → 전체 미션 테스트로 넘어갈 때는 TEST_DISABLE_INTERSECTION=False와 함께 이것도 False로 되돌릴 것
#     (둘 다 켜두면 S0 직후 첫 차선주행 구간에서도 Behavior가 켜져 시나리오 순서가 어긋난다).


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
        self.lane_offset = 0.0      # 근거리 중앙편차(px, 우측+)
        self.lane_valid  = False    # 차선 검출 여부
        self.lane_lookahead = 0.0   # 원거리(앞쪽) 편차 → 코너 진입 전 예측감속용
        self._lane_prev_width = 448.0  # 도로폭 직전값(px, EMA)
        # [2-2 신호등]
        self.signal_color   = 'unknown'  # [S0] 'red'/'yellow'/'blue'/'unknown'
        self.signal_red_on      = False  # [S2] 빨강
        self.signal_straight_on = False  # [S2] 직진(점등 위치)
        self.signal_left_on     = False  # [S2] 좌회전(점등 위치)
        self.stopline = False            # 굵은 가로 흰선(정지선/지름길 끝 단서)
        self._stopline_cooldown_t = 0.0  # 이 시각까지 정지선 재감지 무시
        # [2-3 장애물(전방/측면)]
        self.obstacle_front = False   # 전방 장애물
        self.obstacle_dist  = 999.0   # 전방 거리(m)
        self.obstacle_side  = 'none'  # 'left'/'right'/'center'/'none'
        self.obstacle_type  = 'none'  # 'fixed'/'vehicle'/'none' (라이다 점수로 판별)
        self.left_clear     = True    # 좌측 차선 비었는지(추월 복귀 판단)
        self.right_clear    = True    # 우측 차선 비었는지(추월 이동 판단)
        self._ema_y         = 0.0     # 전방 장애물 횡위치 EMA(obstacle_side 안정화)
        # [2-4 라바콘]
        self.lavacon_offset = 0.0
        self.lavacon_done   = False
        self.lavacon_half_width = HALF_WIDTH_DEFAULT
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

        # ── 외부 차선 인식 모듈 (lane_util.py) 초기화 ──
        #   PolyLaneNet 딥러닝 회귀 기반 차선 인식기. weights_path가 None이면 랜덤 초기화된
        #   backbone으로 동작하므로(=차선 인식 불가), 실차 투입 전 반드시 학습된 체크포인트를
        #   DEFAULT_WEIGHTS_PATH 경로에 배치하거나 아래 인자로 별도 경로를 지정할 것.
        self.lane_detector = PolyLaneNetDetector(weights_path=DEFAULT_WEIGHTS_PATH)
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
        self._pid_prev_error = 0.0
        self._pid_integral   = 0.0
        self._turn_yaw_start = None   # 좌회전 진행 중 플래그 (None=미회전)
        self._turn_frame_cnt = 0      # 좌회전 경과 프레임 수
        self._approach_t0    = None   # [진입] 정지선 감지 후 감속 시작 시각
        self._exit_approach_t0 = None # [진출] S3 탈출 정지선 감지 후 감속 시작 시각
        self._shortcut_t0    = None   # 지름길 진입 시각(끝감지 타이밍용)
        self._shortcut_ref_yaw = None # S3 진입 1초 후 기록한 기준 yaw (탈출 좌회전 전 보정용)
        self._overtake_phase_int = 0   # 방해차량(B3) 단계 정수 FSM (0=대기) — ★재설계 예정
        self._overtake_frame_cnt = 0   # 현재 단계 경과 프레임 수
        self._obstacle_phase     = 'idle'  # 고정장애물(B2) 단계 ('idle'/그 외)      — ★재설계 예정
        self._obstacle_frame_cnt = 0       # 현재 단계 경과 프레임 수
        self._prev_speed     = 0.0    # 가속 속도제한용(직전 출력 속도)
        self._corner_hold    = 0.0    # 코너 활성도(감쇠 peak-hold)
        self._last_debug_t   = 0.0

        self.target_lane = None       # LEFT / RIGHT
        self.target_offset = 0.0      # 목표 편차 (px)
        self._return_cnt = 0

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

    def drive(self, angle, speed):
        # ROS1 xycar_motor.py 노드가 XycarMotor 대신 Float32MultiArray(data=[angle, speed])를
        # 구독하도록 이미 전환되어 있음(ros1_bridge가 커스텀 XycarMotor 타입을 못 넘기는 문제 우회).
        clipped_angle = float(np.clip(angle, -ANGLE_MAX, ANGLE_MAX))
        clipped_speed = float(np.clip(speed, -100.0, 100.0))
        self.motor_msg.data = [clipped_angle, clipped_speed]
        for _ in range(7):
            self.motor_pub.publish(self.motor_msg)


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

        # lane_util.py의 PolyLaneNetDetector로 차선 인식 수행 (딥러닝 다항식 회귀, CV 후처리 없음)
        valid, offset, lookahead, lane_center, bev = self.lane_detector.detect(self.img_front)

        self.lane_center = lane_center
        self.lane_valid = valid
        if valid:
            # 기존 제어 코드와 호환되도록 필터링 적용
            self.lane_offset = 0.7 * self.lane_offset + 0.3 * offset
            self.lane_lookahead = 0.5 * self.lane_lookahead + 0.5 * lookahead

    # [2-2] 신호등
    #   입력 self.img_front
    #   출력 [S0] signal_color / [S2] signal_red/straight/left_on
    #   주의 4구는 직진·좌회전 모두 초록 → 점등 '위치'로 구분
    def perc_signal(self):
        """신호등 판별 (상태별) — traffic_signal.py의 SignalDetector(Hough Circle)에 위임:
          S0 → 3구 색 판별 → signal_color('blue'=초록=출발)
          S2 → 4구 직진/좌회전 → 빨강 동반 여부로 구분(좌회전=초록+빨강 동시, 직진=초록만)
        ★TODO(실차 테스트시 체크): 원 3개/4개 정확히 안 잡히면 그 프레임은 인식 실패 처리됨
          (디바운스/폴백 없음, 자세한 내용은 traffic_signal.py의 shape_ok 주석 참고)"""
        if self.img_front is None:
            return

        if self.mission_state == MissionState.S0_WAIT_GREEN:
            self.signal_color = self.signal_detector.detect_s0(self.img_front)

        elif self.mission_state == MissionState.S2_INTERSECTION:
            self.signal_red_on, self.signal_straight_on, self.signal_left_on = \
                self.signal_detector.detect_s2(self.img_front)

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
        self.obstacle_front = front_cnt > FRONT_MIN_PTS
        if self.obstacle_front:
            self.obstacle_dist = float(np.min(r[front_mask]))
            self.obstacle_type = 'vehicle' if front_cnt >= FRONT_VEHICLE_PTS else 'fixed'
            mean_y = float(np.mean(y[front_mask]))
            self._ema_y = SIDE_EMA_ALPHA * mean_y + (1.0 - SIDE_EMA_ALPHA) * self._ema_y
            if   self._ema_y >  SIDE_DEADZONE: self.obstacle_side = 'left'
            elif self._ema_y < -SIDE_DEADZONE: self.obstacle_side = 'right'
            else:                              self.obstacle_side = 'center'
        else:
            self.obstacle_dist = 999.0
            self.obstacle_side = 'none'
            self.obstacle_type = 'none'
            self._ema_y *= (1.0 - SIDE_EMA_ALPHA)

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
        self.lavacon_offset, self.lavacon_done, self.lavacon_half_width = process_lavacon(
            self.lidar_ranges, self.lavacon_half_width)

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

    # #########################################################
    # [3] 판단 (Decision)
    # #########################################################
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
        # S1 진입 시 감속 플래그 초기화
        if new_state == MissionState.S1_LANE_FOLLOW:
            self._approach_t0 = None
            # 출발(S0) 직후 첫 S1 진입 시 잠깐 정지선 오검출 억제
            if prev_state == MissionState.S0_WAIT_GREEN:
                self._stopline_cooldown_t = time.time() + 3.0
        # S3 진입 시 탈출 감속 플래그 + 기준 yaw 초기화
        if new_state == MissionState.S3_SHORTCUT:
            self._exit_approach_t0 = None
            self._shortcut_ref_yaw = None

    # ── S0: 출발 (신호등 인식) ──
    def _s0_wait_green(self):
        """
        출발선에서 정지한 채 3구 신호등을 본다.
          - 파란불 전: 완전 정지 (신호위반 감점 방지)
          - 파란불 감지: S1(차선주행)로 전환하여 출발
        """
        self.ctrl_angle, self.ctrl_speed = 0.0, SPEED_STOP
        if self.signal_color == 'blue':
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
        if self._obstacle_phase != 'idle' or self._overtake_phase_int != 0:
            return

        if self._approach_t0 is not None:
            # 감속 구간: 차선 조향 유지 + 극저속 → 거의 정지 상태로 S2 진입
            elapsed = time.time() - self._approach_t0
            self.ctrl_angle = self._lane_pid(
                (1.0 - LANE_PREVIEW) * self.lane_offset + LANE_PREVIEW * self.lane_lookahead)
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
          2. 직진 초록(signal_straight_on) → S1 복귀 + Behavior 활성화(라바콘부터 진행)
             좌회전 신호(초록+빨강 동시, signal_left_on) → 좌회전 후 S3(지름길)
          3. 좌회전 진행 중이면 신호와 무관하게 완료 우선
        """
        if self._turn_yaw_start is not None:
            self._do_left_turn(next_state=MissionState.S3_SHORTCUT)
            return

        self.ctrl_angle, self.ctrl_speed = 0.0, SPEED_STOP

        if self.signal_straight_on:
            # 직진 신호 → S1 복귀, 이때부터 Behavior 시작
            self._behavior_enabled = True
            self._stopline_cooldown_t = time.time() + STOPLINE_COOLDOWN
            self._change_state(MissionState.S1_LANE_FOLLOW)
        elif self.signal_left_on:
            # 좌회전 신호 → 지름길로 (Behavior 안 켬)
            self._begin_left_turn()

    # ── S3: 지름길 — 직진(+차선소실 대비), 끝에서 좌회전 ──
    def _s3_shortcut(self):
        """
        지름길 직진. 중간 차선소실 구간은 라이다로 딸 것이 없으므로 그냥 직진.
        끝에 도달하면 신호없이 좌회전으로 S1(차선주행) 복귀 (Behavior는 켜지 않음).
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

    # 목표 차선 결정
    # obstacle_side(라이다 기준 장애물 좌/우 위치)를 이용하여 회피할 목표 차선을 결정
    def decide_target_lane(self):

        if self.obstacle_side == 'left':

            self.target_lane = "RIGHT"

            #현재 차선 기준으로 오른쪽으로 100px 이동
            self.target_offset = self.lane_center + AVOID_OFFSET

        elif self.obstacle_side == 'right':
            self.target_lane = "LEFT"
            #현재 차선 기준으로 왼쪽으로 100px 이동
            self.target_offset = self.lane_center - AVOID_OFFSET

        else:
            self.target_lane = None
            self.target_offset = self.lane_offset

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
            triggered = self.obstacle_front and self.obstacle_type == 'fixed' and self.obstacle_dist < SAFETY_DIST
            self.behavior_state = (BehaviorState.B2_OBSTACLE
                                    if (triggered or self._obstacle_phase != 'idle')
                                    else BehaviorState.B0_NORMAL)
        elif self.phase == Phase.VEHICLE:
            # 위와 동일한 이유로 트리거 검사를 건너뛰고 B0로 고정(placeholder 추월 기동 비활성화)
            if TEST_DISABLE_B2_B3:
                self.behavior_state = BehaviorState.B0_NORMAL   # 테스트 범위 제한: B3 트리거 무시
                return
            # 진입 판정은 perc_vehicle_trigger()의 라이다 디바운스 결과를 사용.
            # 한번 진입한 뒤(_overtake_phase_int != 0)에는 기존과 동일하게 라이다 단독으로 유지/종료 판단.
            self.behavior_state = (BehaviorState.B3_VEHICLE
                                    if (self.vehicle_trigger or self._overtake_phase_int != 0)
                                    else BehaviorState.B0_NORMAL)
        else:  # Phase.DONE
            self.behavior_state = BehaviorState.B0_NORMAL

    # #########################################################
    # [4] 제어 (Control)
    # #########################################################
    def _lane_drive(self):
        """S1/S3 공통 차선 조향+감속 로직. ctrl_angle·ctrl_speed·_prev_speed·_corner_hold 갱신."""
        steer_offset = (1.0 - LANE_PREVIEW) * self.lane_offset + LANE_PREVIEW * self.lane_lookahead
        self.ctrl_angle = self._lane_pid(steer_offset)
        turn_now     = min(1.0, abs(self.ctrl_angle) / ANGLE_MAX)
        turn_preview = min(1.0, abs(self.lane_lookahead) / LANE_LOOKAHEAD_REF)
        turn_for_speed = max(turn_now, turn_preview * 0.3)
        target_speed = max(SPEED_NORMAL * 0.15,
                           SPEED_NORMAL * (1.0 - 0.90 * turn_for_speed ** 3))
        speed_ratio = min(1.0, self._prev_speed / SPEED_NORMAL)
        corner_decay = CORNER_HOLD_DECAY_LO + (CORNER_HOLD_DECAY_HI - CORNER_HOLD_DECAY_LO) * speed_ratio
        self._corner_hold = max(turn_now, self._corner_hold * corner_decay)
        accel_step = SPEED_ACCEL_STEP * max(0.25, 1.0 - self._corner_hold)
        if target_speed > self._prev_speed + accel_step:
            target_speed = self._prev_speed + accel_step
        self.ctrl_speed = target_speed
        self._prev_speed = target_speed

    def _lane_pid(self, offset, deadzone=LANE_DEADZONE):
        """차선 중앙편차(offset)를 PID 제어로 조향각(angle)으로 변환한다."""
        if abs(offset) < deadzone:
            offset = 0.0
        self._pid_integral += offset
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

    # ── B2-고정장애물 회피 ──★재설계 예정(임시 placeholder) ──
    # target_lane을 반영해 수정
    def _handle_fixed_obstacle(self):
        """
        ★ TODO: 실차 회피 기동 재설계 필요. 시뮬 전용이던 역C자 고정 프레임 시퀀스는 폐기.
        지금은 감지되면 감속만 하고 버티다가, 장애물이 사라지면 방해차량 구간으로 넘어가는
        임시(placeholder) 동작이다. 실제 회피 궤적/조향은 팀에서 별도로 설계해서 교체할 것.
        """
        is_obstacle = (
            self.obstacle_front and 
            self.obstacle_type == "fixed"
        )

        #idle
        if self._obstacle_phase == "idle":
            if is_obstacle:
                #회피 방향을 딱 한번 결정
                self.decide_target_lane()
                self._obstacle_phase = "avoid"
                self.get_logger().info(
                    f"[OBSTACLE] START lane={self.target_lane}"
                )
            return
        
        #avoid
        elif self._obstacle_phase == "avoid":
            steer_offset = (
                (1.0-LANE_PREVIEW)*self.target_offset + 
                LANE_PREVIEW*self.lane_lookahead
            )

            self.ctrl_angle = self._lane_pid(steer_offset)
            self.ctrl_speed = SPEED_LAVACON

            #장애물이 일전 프레임 동안 사라졌으면 복귀 시작
            if not is_obstacle:
                self._return_cnt += 1
            else:
                self._return_cnt = 0
            if self._return_cnt >= 5:
                self._return_cnt = 0
                self._obstacle_phase = "return"

                self.target_offset = 0.0

                self.get_logger().info("[OBSTACLE] RETURN")

        #return
        elif self._obstacle_phase == "return":
            steer_offset = (
                (1.0-LANE_PREVIEW)*self.lane_offset +
                LANE_PREVIEW*self.lane_lookahead
            )

            self.ctrl_angle = self._lane_pid(steer_offset)
            self.ctrl_speed = SPEED_NORMAL

            # 차선 중앙으로 거의 복귀
            if abs(self.lane_offset) < RETURN_THRESHOLD:
                self._obstacle_phase = "idle"
                self.phase = Phase.VEHICLE

                self._return_cnt = 0
                
                self._pid_prev_error = 0
                self._pid_integral = 0

                self.get_logger().info("[OBSTACLE] DONE")

    # ── B3-방해차량 추월 ──★재설계 예정(임시 placeholder) ──
    # target_lane 반영 수정
    # 회피 후 복귀하는 로직 추가 : idle -> avoid -> idle+Phase.VEHICLE => idel -> avoid -> return -> idle
    def _handle_overtake(self):
        """
        ★ TODO: 실차 추월 기동 재설계 필요. 시뮬 전용이던 6단계 고정 프레임 시퀀스는 폐기.
        지금은 감지되면 감속만 하고 버티다가, 차량이 사라지면 DONE으로 넘어가는
        임시(placeholder) 동작이다. 실제 추월 궤적/조향은 팀에서 별도로 설계해서 교체할 것.
        """
        is_vehicle = self.obstacle_front and self.obstacle_dist < OVERTAKE_TRIGGER

        if self._overtake_phase_int == 0:
            if is_vehicle:
                self._overtake_phase_int = 1
                self._overtake_frame_cnt = 0
                self.get_logger().info(f'[VEHICLE] 감지 dist={self.obstacle_dist:.2f}m')
            return

        # 활성 상태 — TODO: 실제 추월 기동으로 교체
        self.decide_target_lane()

        steer_offset = (
            (1.0 - LANE_PREVIEW) * self.target_offset
            + LANE_PREVIEW * self.lane_lookahead
        )
        self.ctrl_angle = self._lane_pid(steer_offset) 
        self.ctrl_speed = SPEED_LAVACON
        self._overtake_frame_cnt += 1

        if not is_vehicle and self._overtake_frame_cnt > 10:
            self._overtake_phase_int = 0
            self._overtake_frame_cnt = 0
            self._pid_prev_error = 0.0
            self._pid_integral   = 0.0
            self.phase = Phase.DONE
            self.get_logger().info('[VEHICLE] 추월 완료(placeholder) → DONE')


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
        self.run_mission_fsm()              # 2. 판단(Mission)

        if ENABLE_BEHAVIOR and self.mission_state == MissionState.S1_LANE_FOLLOW and self._behavior_enabled:
            self.run_behavior_fsm()         #    Behavior 상태 결정
            self.apply_behavior_override()  #    필요 시 조향/속도 덮어쓰기
        else:
            self.behavior_state = BehaviorState.B0_NORMAL   # OFF 구간은 항상 정상

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
          [SENS] cam = 카메라 나이(s, 값이 계속 커지면 미수신) / sig = 신호등 색(S0: unknown이면 원 검출 실패)
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
          [SIG-S0](S0 상태에서만 출력) 신호등 원 검출이 어느 단계에서 막혔는지 진단:
            roi     = ROI 픽셀 좌표(t,b,l,r) — 신호등이 이 영역 안에 실제로 들어오는지 확인용
            circles = HoughCircles가 찾은 원 개수(0=원 자체를 못 찾음, 3이 아니면 배치/블러 의심)
            reason  = 실패 사유(OK=성공) — circle_count=N / vert_spread.../horiz_spread.../
                      gap[i]=... (배치 불량) / bright_margin=... (밝기 대비 부족)
            bright  = 성공적으로 3개+배치 통과 시 좌→우(빨강,노랑,초록) 밝기값
            margin  = 최댓값-평균 (SIG_BRIGHT_MARGIN=15와 비교되는 실측값)
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

        sig_s0_line = ''
        if self.mission_state == MissionState.S0_WAIT_GREEN:
            sd = self.signal_detector
            reason = sd.s0_reject_reason or 'OK'
            sig_s0_line = (
                f'\n  [SIG-S0] roi={sd.s0_roi_px} circles={sd.s0_circle_count} '
                f'reason={reason} bright={sd.s0_brightness} margin={sd.s0_bright_margin:+.1f}'
            )

        self.get_logger().info(
            f'[{self.mission_state.name}|{self.behavior_state.name}|{self.phase.name}] '
            f'ang={self.ctrl_angle:+.1f} spd={self.ctrl_speed:.1f}\n'
            f'  [SENS] cam={cam_age:.1f}s sig={self.signal_color} lidar={lidar_desc}\n'
            f'  [LANE] lane={self.lane_offset:+.1f}({int(self.lane_valid)}) '
            f'obs={self.obstacle_front}({self.obstacle_dist:.1f}m,{self.obstacle_side},{self.obstacle_type}) '
            f'lava={self.lavacon_offset:+.2f}(done={int(self.lavacon_done)})\n'
            f'  [TRIG] trigL={self._lavacon_trigger_cnt}/{LAVACON_TRIGGER_FRAMES}'
            f'(L{int(self.lavacon_left_detected)}R{int(self.lavacon_right_detected)}) '
            f'trigV={self._vehicle_trigger_cnt}/{VEHICLE_TRIGGER_FRAMES}\n'
            f'  [LAVA-ROI] L pts={lava_lp} run={lava_lrun}(need>=2) '
            f'R pts={lava_rp} run={lava_rrun}(need>=2) '
            f'masked_raw_pts={masked_pts} masked_min={masked_min_s}'
            f'{sig_s0_line}')


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
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
