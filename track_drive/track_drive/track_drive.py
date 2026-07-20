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
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image, LaserScan, Imu
from std_srvs.srv import SetBool
from yolo_msgs.msg import DetectionArray
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from .perc_lavacon import process_lavacon
from .lane_util import CameraProcessor, SlideWindow
from .perc_floor import LaneDetector, check_stopline
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
SPEED_NORMAL  = 30.0   # 차선주행(S1) 기본속도
                       # 출처: KUAC_2024-main lane_detection/src/lane_detection.py self.motor=30(고정)
                       #   기존 20.0 → 30.0. 모터/조향 스케일이 같은 xycar 플랫폼인지 미확인, 실차 저속 테스트 우선 권장
SPEED_LAVACON = 2.5    # KUAC_2024 라바콘 속도(12~30, fast/safe 라벨 앞뒤가 안 맞아 신뢰도 낮음) 참고만 하고 미반영
SPEED_STOP    = 0.0
ANGLE_MAX     = 100.0
SPEED_ACCEL_STEP = 0.85  # 가속 속도제한(주기당 최대 증가량)
CORNER_HOLD_DECAY_LO = 0.92  # 저속 시 코너 hold 감쇠 (빠른 회복)
CORNER_HOLD_DECAY_HI = 0.97  # 고속 시 코너 hold 감쇠 (느린 회복, 연속코너 대응)

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

# ── YOLO 카메라 이중확인 (yolo_ros 연동) ──
# 라이다는 "무언가 있다"만 알고 그것이 콘인지 차량인지 모르므로, 카메라 YOLO의
# 클래스 판별을 라이다 트리거와 AND 결합해 이중확인한다.
# yolo_node 인스턴스 2개(yolo_cone/yolo_vehicle)를 launch에서 항상 띄워두고,
# Phase에 맞는 쪽만 enable 서비스(SetBool)로 켠다(모델 재로드 없이 즉시 전환).
YOLO_FRESH_SEC         = 0.5   # 이 시간(s) 이내 수신된 detection만 유효(카메라 추론 지연/끊김 대비)
YOLO_CONE_CLASSES      = ('cone',)          # 라바콘 파인튜닝 모델(cone_best.pt)의 클래스명
YOLO_VEHICLE_CLASSES   = ('car', 'truck')   # COCO 모델에서 xycar가 잡히는 클래스(실측: truck 위주)
VEHICLE_TRIGGER_FRAMES = 5     # (라이다+YOLO) 동시검출 연속 N프레임이면 B3_VEHICLE 진입 확정(디바운스)
YOLO_FALLBACK_FRAMES   = 60    # 라이다 단독검출이 연속 N프레임(20Hz→3초) 지속되면 YOLO 미확인이라도 진입 확정.
                               #   AND 결합이 만드는 단일 실패점(YOLO 노드 죽음/미검출 → 미션 전체 정지) 방어용 폴백.
                               #   폴백 발동은 warn 로그로 남으므로 실차 테스트에서 자주 보이면 YOLO 쪽 점검할 것.

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
START_STATE     = MissionState.S0_WAIT_GREEN
ENABLE_BEHAVIOR = True
DEBUG_LOG       = True
DEBUG_PERIOD    = 0.5
DEBUG_VIZ       = False  # 신호등/4구 디버그 창
DEBUG_VIZ_LANE  = False  # 차선 슬라이딩윈도우 디버그 창
DEBUG_VIZ_LIDAR = False  # 라이다 BEV 장애물 감지 디버그 창

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
        self._lavacon_empty_cnt = 0   # 우측콘 연속 미검출 프레임 수(Phase 전환 디바운스)
        self.lavacon_left_detected  = False  # 좌측 라이다 클러스터 검출 여부(B1 진입 트리거용)
        self.lavacon_right_detected = False  # 우측 라이다 클러스터 검출 여부(B1 진입 트리거용)
        self.lavacon_trigger        = False  # (좌우 동시검출 AND YOLO 콘)이 디바운스 프레임수만큼 유지되면 True
        self._lavacon_trigger_cnt   = 0      # 동시검출 연속 프레임 수(디바운스 카운터)
        self._lavacon_lidar_cnt     = 0      # 라이다 단독검출 연속 프레임 수(YOLO 미확인 폴백 카운터)
        # [2-6 YOLO 객체인식]
        self.yolo_cone_detected    = False   # YOLO 콘 검출 여부(신선도 통과분만, B1 진입 AND 조건)
        self.yolo_vehicle_detected = False   # YOLO 차량(car/truck) 검출 여부(B3 진입 AND 조건)
        self._yolo_cone_msg        = None    # 최신 DetectionArray 버퍼(yolo_cone)
        self._yolo_cone_t          = 0.0     # 최신 수신 시각(신선도 판정용)
        self._yolo_vehicle_msg     = None    # 최신 DetectionArray 버퍼(yolo_vehicle)
        self._yolo_vehicle_t       = 0.0     # 최신 수신 시각(신선도 판정용)
        # [2-7 방해차량 트리거]
        self.vehicle_trigger       = False   # (라이다 AND YOLO 차량) 디바운스 통과 → B3 진입 트리거
        self._vehicle_trigger_cnt  = 0       # 동시검출 연속 프레임 수(디바운스 카운터)
        self._vehicle_lidar_cnt    = 0       # 라이다 단독검출 연속 프레임 수(YOLO 미확인 폴백 카운터)
        # [2-8 장애물 위치 판단]
        self.obstacle_lane = None            # YOLO bbox 중심 vs 차선 중앙 비교 결과 ('LEFT'/'RIGHT'/None)
        self.lane_center   = 320.0           # 차선 중앙 x좌표(px) — 첫 카메라 프레임 전까지 화면 중앙 기본값

        # ── 외부 차선 인식 모듈 (lane_util.py / perc_floor.py) 초기화 ──
        self.camera_processor = CameraProcessor()       # BEV 변환 및 색상 마스크(흰/노랑) 처리기
        self.slide_window_processor = SlideWindow()     # 슬라이딩 윈도우 기반 차선 탐색 및 피팅기
        self.lane_detector = LaneDetector(self.camera_processor, self.slide_window_processor)
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
        self.motor_msg = XycarMotor()
        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)
        image_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image,     '/usb_cam/image_raw/front',  self.cb_img_front,  image_qos)
        self.create_subscription(Image,     '/usb_cam/image_raw/left',   self.cb_img_left,   image_qos)
        self.create_subscription(Image,     '/usb_cam/image_raw/right',  self.cb_img_right,  image_qos)
        self.create_subscription(Image,     '/usb_cam/image_raw/behind', self.cb_img_behind, image_qos)
        self.create_subscription(LaserScan, '/scan',                     self.cb_scan,       qos_profile_sensor_data)
        self.create_subscription(Imu,       '/imu',                      self.cb_imu,        qos_profile_sensor_data)
        # YOLO 검출 결과 (yolo_ros의 yolo_node 2개 — launch에서 namespace로 분리)
        self.create_subscription(DetectionArray, '/yolo_cone/detections',    self.cb_yolo_cone,    10)
        self.create_subscription(DetectionArray, '/yolo_vehicle/detections', self.cb_yolo_vehicle, 10)
        # Phase 전환 시 yolo_node on/off용 서비스 클라이언트 (_switch_yolo에서 비동기 호출)
        self._cli_yolo_cone_en    = self.create_client(SetBool, '/yolo_cone/enable')
        self._cli_yolo_vehicle_en = self.create_client(SetBool, '/yolo_vehicle/enable')
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info(f'초기화 완료 | 시작={START_STATE.name}')


    # #########################################################
    # [1] 통신 I/O    담당: 공통(수정 X)
    # #########################################################
    def cb_img_front(self, msg):
        try:
            self.img_front = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.get_logger().info(f'[front] 첫 수신 OK enc={msg.encoding} shape={self.img_front.shape}', once=True)
        except Exception as e:
            self.get_logger().error(f'[front] 이미지 변환 실패 enc={msg.encoding}: {e}', throttle_duration_sec=2.0)
    def cb_img_left(self, msg):   self.img_left   = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    def cb_img_right(self, msg):  self.img_right  = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    def cb_img_behind(self, msg): self.img_behind = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    def cb_scan(self, msg):       self.lidar_ranges = msg.ranges
    def cb_imu(self, msg):
        q = msg.orientation
        self.imu_yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
    # YOLO 검출은 카메라 추론 속도에 따라 비동기로 들어옴 → 버퍼+수신시각만 저장하고
    # 유효성(신선도) 판정은 제어루프의 perc_yolo()에서 일괄 수행한다.
    def cb_yolo_cone(self, msg):    self._yolo_cone_msg,    self._yolo_cone_t    = msg, time.time()
    def cb_yolo_vehicle(self, msg): self._yolo_vehicle_msg, self._yolo_vehicle_t = msg, time.time()

    def drive(self, angle, speed):
        self.motor_msg.angle = float(np.clip(angle, -ANGLE_MAX, ANGLE_MAX))
        self.motor_msg.speed = float(np.clip(speed, -100.0, 100.0))
        for _ in range(7):
            self.motor_pub.publish(self.motor_msg)


    # #########################################################
    # [2] 인지 (Perception)
    # #########################################################
    def perceive_all(self):
        self.perc_lane()        # 비전
        self.perc_signal()      # 비전
        self.perc_obstacle()    # 라이다
        self.perc_yolo()        # 비전 (YOLO 클래스 판별 — 아래 트리거들의 AND 조건 재료)
        self.perc_lavacon()     # 라이다
        self.perc_lavacon_trigger()  # 라이다+YOLO (좌우 클러스터 AND 콘 검출 → B1_LAVACON 진입 트리거)
        self.perc_vehicle_trigger()  # 라이다+YOLO (전방 장애물 AND 차량 검출 → B3_VEHICLE 진입 트리거)
        self.perc_obstacle_lane()    # 비전 (YOLO bbox 중심 vs 차선 중앙 → 장애물 좌/우 판단, B2/B3 회피방향 재료)
        self.perc_stopline()    # 비전

    # [2-1] 차선
    #   입력 self.img_front → 출력 self.lane_offset(우측+), self.lane_valid
    def perc_lane(self):
        if self.img_front is None:
            self.lane_valid = False
            return

        # lane_util.py의 LaneDetector를 사용하여 차선 인식 수행
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
        BODY_LO, BODY_HI         = 99, 263     # 차체 자기가림 구간

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
            _deg = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
            self._obs_cos = np.cos(_deg).astype(np.float32)
            self._obs_sin = np.sin(_deg).astype(np.float32)

        ranges = np.array(self.lidar_ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = 0.0
        ranges[ranges <= 0.0]        = 0.0
        ranges[BODY_LO:BODY_HI]      = 0.0   # 차체 자기가림 마스킹

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
            PPM = 50          # 1m = 50px
            W, H = 500, 500
            EX, EY = 250, 450  # 자차 위치(하단 중앙)
            bev = np.zeros((H, W, 3), dtype=np.uint8)

            for d in range(1, 6):
                cv2.circle(bev, (EX, EY), d * PPM, (50, 50, 50), 1)
                cv2.putText(bev, f'{d}m', (EX + 4, EY - d*PPM + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

            def to_px(wx, wy): return (int(EX - wy*PPM), int(EY - wx*PPM))
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
    #   입력 self.lidar_ranges, self.yolo_cone_detected([2-6] 카메라)
    #   출력 lavacon_left_detected/right_detected, lavacon_trigger
    #   설계 의도: 라이다 포인트가 "존재"하는 것만으로는 벽·바닥 잡음과 구분이 안 되므로,
    #     인접 인덱스(=인접 각도)로 붙어있는 포인트 묶음(클러스터)이 좌/우 각각 최소 1개씩
    #     동시에 있어야 "라바콘 구간 진입"으로 인정한다. perc_obstacle()과 동일한 차체 마스킹/
    #     극좌표 변환 방식을 사용하되, ROI와 목적은 별개(장애물 회피용이 아니라 콘 게이트 진입 판단용)이므로
    #     여기서 독립적으로 계산한다.
    #     + 카메라 이중확인: 좌우 클러스터가 있어도 그것이 콘이라는 보장은 없으므로(벽 모서리·
    #     사람 다리 등), YOLO 콘 검출(yolo_cone_detected)까지 AND로 만족해야 카운트를 올린다.
    #     + 폴백: 카메라/YOLO 실패가 미션 전체를 멈추는 단일 실패점이 되지 않도록,
    #     라이다 단독검출이 YOLO_FALLBACK_FRAMES 연속 지속되면 YOLO 미확인이라도 진입한다.
    def perc_lavacon_trigger(self):
        # ── 튜닝 파라미터 (실측 라바콘 간격 기준 추정치, 실차 튜닝 필요) ──
        LON_MIN, LON_MAX = 0.3, 3.0   # 트리거 ROI 전방 종방향(m) — 너무 가깝거나(차체 반사) 먼 점 배제
        LAT_MAX           = 2.0        # 트리거 ROI 횡방향 한계(m)
        CLUSTER_MIN_PTS   = 2          # 클러스터로 인정할 최소 연속 포인트 수(단일 반사점 노이즈 배제)
        CLUSTER_MAX_GAP   = 0.35       # 같은 클러스터로 볼 최대 거리편차(m) — 콘 지름 근사
        BODY_LO, BODY_HI  = 99, 263    # 차체 자기가림 구간 (perc_obstacle과 동일)

        if self.lidar_ranges is None:
            self.lavacon_left_detected  = False
            self.lavacon_right_detected = False
            self._lavacon_trigger_cnt   = 0
            self.lavacon_trigger        = False
            return

        ranges = np.array(self.lidar_ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = 0.0
        ranges[ranges <= 0.0] = 0.0
        n = len(ranges)
        if n > BODY_LO:
            ranges[BODY_LO:min(BODY_HI, n)] = 0.0   # 차체 자기가림 마스킹

        m = min(n, 360)
        deg = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False)
        r = ranges[:m]
        x = r * np.cos(deg)          # 전방(+앞)
        y = r * np.sin(deg)          # 횡방향(+좌/-우)
        roi = (r > 0.0) & (x > LON_MIN) & (x < LON_MAX) & (np.abs(y) < LAT_MAX)

        def _has_cluster(side_mask):
            idx = np.where(roi & side_mask)[0]
            if len(idx) < CLUSTER_MIN_PTS:
                return False
            # 인덱스(=각도) 순 배열이므로, 인덱스가 서로 붙어있으면 공간적으로도 인접한 점으로 보고 묶는다.
            splits = np.where(np.diff(idx) > 1)[0] + 1
            for g in np.split(idx, splits):
                if len(g) >= CLUSTER_MIN_PTS and (np.max(r[g]) - np.min(r[g])) <= CLUSTER_MAX_GAP:
                    return True   # 콘 하나 크기로 뭉친 클러스터 발견
            return False

        self.lavacon_left_detected  = _has_cluster(y > 0.0)   # 좌측(y>0)
        self.lavacon_right_detected = _has_cluster(y < 0.0)   # 우측(y<0)

        # 본선: (라이다 좌우 클러스터 AND YOLO 콘) 디바운스 / 폴백: 라이다 단독 장기지속(YOLO 실패 방어)
        if self.lavacon_left_detected and self.lavacon_right_detected:
            self._lavacon_lidar_cnt += 1
            self._lavacon_trigger_cnt = self._lavacon_trigger_cnt + 1 if self.yolo_cone_detected else 0
        else:
            self._lavacon_lidar_cnt   = 0
            self._lavacon_trigger_cnt = 0
        if self._lavacon_lidar_cnt == YOLO_FALLBACK_FRAMES and self._lavacon_trigger_cnt < LAVACON_TRIGGER_FRAMES:
            self.get_logger().warn('[LAVACON] YOLO 콘 미확인 — 라이다 단독 폴백으로 진입 (yolo_cone 노드 점검 요망)')
        self.lavacon_trigger = (self._lavacon_trigger_cnt >= LAVACON_TRIGGER_FRAMES
                                or self._lavacon_lidar_cnt >= YOLO_FALLBACK_FRAMES)

    # [2-6] YOLO 객체인식 (카메라)
    #   입력 self._yolo_cone_msg / _yolo_vehicle_msg (yolo_ros DetectionArray 버퍼)
    #   출력 yolo_cone_detected / yolo_vehicle_detected
    #   설계 의도: 라이다 클러스터/점개수 판단은 "무언가 있다"까지만 알 수 있고
    #     그것이 콘인지 차량인지 벽인지 구분하지 못한다. 카메라 YOLO의 클래스 판별을
    #     각 트리거([2-4b], [2-7])에 AND 조건으로 결합해 이중확인한다.
    #     카메라 추론은 제어루프(20Hz)보다 느릴 수 있으므로 YOLO_FRESH_SEC 이내
    #     수신분만 유효 처리(멈춘 노드의 낡은 검출로 인한 유령 트리거 방지).
    #     신뢰도(score) 필터는 yolo_node의 threshold 파라미터에서 이미 수행되므로 여기선 생략.
    def perc_yolo(self):
        now = time.time()

        def _has_class(msg, t, classes):
            if msg is None or (now - t) > YOLO_FRESH_SEC:
                return False
            return any(d.class_name in classes for d in msg.detections)

        self.yolo_cone_detected    = _has_class(self._yolo_cone_msg,    self._yolo_cone_t,    YOLO_CONE_CLASSES)
        self.yolo_vehicle_detected = _has_class(self._yolo_vehicle_msg, self._yolo_vehicle_t, YOLO_VEHICLE_CLASSES)

    # [2-7] 방해차량 진입 트리거 (라이다 + YOLO 이중확인)
    #   입력 obstacle_front/dist (라이다), yolo_vehicle_detected (카메라)
    #   출력 vehicle_trigger
    #   설계 의도: 기존 B3 진입은 라이다 거리 단독·즉시 판정이라 순간 오검출에 취약했다.
    #     [2-4b] 라바콘 트리거와 동일한 패턴으로 (라이다 AND YOLO) 동시검출이
    #     연속 N프레임 유지될 때만 진입을 확정한다(디바운스).
    #     폴백도 [2-4b]와 동일: 라이다 단독검출이 YOLO_FALLBACK_FRAMES 지속되면
    #     YOLO 미확인이라도 진입(카메라 실패로 미션이 멈추는 것 방지).
    def perc_vehicle_trigger(self):
        lidar_hit = self.obstacle_front and self.obstacle_dist < OVERTAKE_TRIGGER
        if lidar_hit:
            self._vehicle_lidar_cnt += 1
            self._vehicle_trigger_cnt = self._vehicle_trigger_cnt + 1 if self.yolo_vehicle_detected else 0
        else:
            self._vehicle_lidar_cnt   = 0
            self._vehicle_trigger_cnt = 0
        if self._vehicle_lidar_cnt == YOLO_FALLBACK_FRAMES and self._vehicle_trigger_cnt < VEHICLE_TRIGGER_FRAMES:
            self.get_logger().warn('[VEHICLE] YOLO 차량 미확인 — 라이다 단독 폴백으로 진입 (yolo_vehicle 노드 점검 요망)')
        self.vehicle_trigger = (self._vehicle_trigger_cnt >= VEHICLE_TRIGGER_FRAMES
                                or self._vehicle_lidar_cnt >= YOLO_FALLBACK_FRAMES)
        
    # [2-8] 장애물 위치 판단

    def perc_obstacle_lane(self):
        self.obstacle_lane = None
        if self._yolo_vehicle_msg is None:
            return
        # 오래된 메세지 무시
        if time.time() - self._yolo_vehicle_t > YOLO_FRESH_SEC:
            return
        # 가장 큰 차량 하나 선택
        best = None
        best_area = 0

        for det in self._yolo_vehicle_msg.detections:
            if det.class_name not in YOLO_VEHICLE_CLASSES:
                continue
            area = det.bbox.size.x * det.bbox.size.y   # yolo_msgs/BoundingBox2D: size는 Vector2(x,y) 필드

            if area > best_area:
                best = det
                best_area = area

        if best is None:
            return
        
        # Bounding Box 중심
        cx = best.bbox.center.position.x

        #화면 기준 좌/우 판단
        if cx < self.lane_center:
            self.obstacle_lane = "LEFT"
        else:
            self.obstacle_lane = "RIGHT"

        

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
    # obstacle_lane을 이용하여 회피할 목표 차선을 결정
    def decide_target_lane(self):

        if self.obstacle_lane == "LEFT":

            self.target_lane = "RIGHT"

            #현재 차선 기준으로 오른쪽으로 100px 이동
            self.target_offset = self.lane_center + AVOID_OFFSET
        
        elif self.obstacle_lane == "RIGHT":
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
            # 진입 판정은 perc_vehicle_trigger()의 (라이다 AND YOLO) 디바운스 결과를 사용.
            # 한번 진입한 뒤(_overtake_phase_int != 0)에는 기존과 동일하게 라이다 단독으로 유지/종료 판단.
            self.behavior_state = (BehaviorState.B3_VEHICLE
                                    if (self.vehicle_trigger or self._overtake_phase_int != 0)
                                    else BehaviorState.B0_NORMAL)
        else:  # Phase.DONE
            self.behavior_state = BehaviorState.B0_NORMAL

    # ── YOLO 노드 on/off 전환 (Phase 전환 시 호출) ──
    def _switch_yolo(self, cone, vehicle):
        """
        yolo_cone/yolo_vehicle 노드의 enable 서비스(SetBool)를 비동기로 호출한다.
          - 제어루프(20Hz)를 막지 않도록 응답을 기다리지 않음(call_async, fire-and-forget)
          - 서비스 서버가 아직 없으면(노드 미실행) 경고만 남기고 넘어감 — 재시도 없으므로
            launch에서 두 yolo_node를 반드시 함께 띄울 것
        """
        for cli, on, tag in ((self._cli_yolo_cone_en,    cone,    'yolo_cone'),
                             (self._cli_yolo_vehicle_en, vehicle, 'yolo_vehicle')):
            if not cli.service_is_ready():
                self.get_logger().warn(f'[YOLO] {tag}/enable 서비스 미발견 — 전환 실패(노드 실행 여부 확인)')
                continue
            req = SetBool.Request()
            req.data = bool(on)
            cli.call_async(req)
            self.get_logger().info(f'[YOLO] {tag} enable={on}')


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
                # 라바콘 구간 끝 → 콘 모델은 더 이상 불필요, 차량 모델을 미리 켜서
                # 다음 Phase.VEHICLE 진입 판정([2-7])에 대비한다(고정장애물 구간은 라이다 단독 판단).
                self._switch_yolo(cone=False, vehicle=True)
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
        now = time.time()
        if now - self._last_debug_t < DEBUG_PERIOD: return
        self._last_debug_t = now
        self.get_logger().info(
            f'[{self.mission_state.name}|{self.behavior_state.name}|{self.phase.name}] '
            f'ang={self.ctrl_angle:+.1f} spd={self.ctrl_speed:.1f} '
            f'lane={self.lane_offset:+.1f}({int(self.lane_valid)}) '
            f'obs={self.obstacle_front}({self.obstacle_dist:.1f}m,{self.obstacle_side},{self.obstacle_type}) '
            f'lava={self.lavacon_offset:+.2f}(done={int(self.lavacon_done)}) '
            f'yolo=cone:{int(self.yolo_cone_detected)}/veh:{int(self.yolo_vehicle_detected)}')


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
