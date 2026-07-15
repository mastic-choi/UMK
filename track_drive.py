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
#  │  [코스 시나리오]                                                │
#  │   출발(라바콘) → 차선주행 → 4구신호등 교차로                     │
#  │     ├ 경찰차 있음(라이다로 경로막힘) → 직진(차선인식)                │
#  │     │    · 보행자 미션(반사 회피)                               │
#  │     │    · 차량 미션(추월: 우측이동→추월→좌측복귀)              │
#  │     │    · 어린이 보호구역(감속)                                │
#  │     └ 경찰차 없음(경로열림)+좌회전신호 → 좌회전(yaw90) → 지름길  │
#  │          · 직진, 중간 차선소실 구간은 그냥 직진(heading hold)    │
#  │          · 끝에서 신호없이 좌회전(yaw90) → 차선인식 복귀             │
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
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from perc_lavacon import process_lavacon
from lane_util import CameraProcessor, SlideWindow
from perc_floor import LaneDetector, check_stopline

# #############################################################
# [0] 설정 (Config)
# #############################################################

class MissionState(Enum):
    S0_WAIT_GREEN  = 0  # 파란불 대기 후 출발
    S1_LAVACON     = 1  # 라바콘 주행
    S2_LANE_FOLLOW = 2  # 차선인식 주행 (보행자→차량 미션 포함)
    S3_INTERSECTION= 3  # 4구 신호등 교차로 (정지→라이다 경로판단→직진/좌회전)
    S4_SHORTCUT    = 4  # 지름길 (직진, 차선소실 구간 대비, 끝에서 좌회전)
    S5_FINISH      = 5  # 종료

class BehaviorState(Enum):
    B0_NORMAL = 0  # Mission 출력 그대로
    B1_SAFETY = 1  # 장애물(보행자/차량) 대응 (1순위)
    B2_POLICY = 2  # 어린이 보호구역 감속 (2순위)

# 차선인식 장애물 미션 구간 (순서 고정: 보행자 → 차량 → 완료)
class Segment(Enum):
    PEDESTRIAN = 0  # 보행자 회피 구간
    VEHICLE    = 1  # 차량 추월 구간
    DONE       = 2  # 모든 장애물 미션 완료 — 이후 B1 발동 시 아무것도 안 함

# 차량 추월 sub-FSM 단계
class Overtake(Enum):
    IDLE        = 0
    PASS_RIGHT  = 1  # 우측 차선으로 이동해 느린 좌측차 추월
    RETURN_LEFT = 2  # 우측차 후진 복귀 전에 좌측 차선으로 복귀

# ── 속도·각도 상수 ──
SPEED_NORMAL  = 20.0   # 차선주행(S2) 기본속도
SPEED_SLOW    = 5.5    # 어린이보호구역 상한 ≈9km/h (대회 패널티 3m/s=10.8km/h 미만 유지, 여유 둠)
SPEED_LAVACON = 2.5
# ── S1 라바콘 고정 시퀀스 (20Hz 프레임 기준) ──
S1_STRAIGHT_SPEED  = 100   # 1단계: 직진 속도
S1_STRAIGHT_FRAMES = 10    # 1단계: 직진 프레임 수
S1_TURN_ANGLE      = -100  # 2단계: 조향각
S1_TURN_SPEED      = 100   # 2단계: 속도
S1_TURN_FRAMES     = 10    # 2단계: 조향 프레임 수
SPEED_TURN    = 4.0    # 회전 (20→4, ×0.2)
SPEED_AVOID   = 3.6    # 보행자 회피 (18→3.6, ×0.2)
SPEED_OVERTAKE= 7.0    # 추월 가속 (35→7, ×0.2)
SPEED_STOP    = 0.0
ANGLE_MAX     = 100.0
SPEED_ACCEL_STEP = 0.85  # 가속 속도제한(주기당 최대 증가량): speed×1.33 비례 조정
CORNER_HOLD_DECAY_LO = 0.92  # 저속 시 코너 hold 감쇠 (빠른 회복)
CORNER_HOLD_DECAY_HI = 0.97  # 고속 시 코너 hold 감쇠 (느린 회복, 연속코너 대응)

# ── 튜닝 파라미터 (한곳에 모음) ──
LANE_KP, LANE_KI, LANE_KD = 0.14, 0.0, 1.40  # 차선 PID (speed×1.33: KP÷1.33, KD×1.33)
LANE_SIDE = 1               # 주행 차선: +1=노란선 오른쪽(우측차선), -1=왼쪽. 엉뚱한 차선이면 부호 반대로
LANE_CORNER_BOOST = 1.8    # 코너(큰 offset) 조향 가중: 최대 (1+BOOST)배까지 KP 증폭 (1.5→1.8)
LANE_CORNER_REF   = 120.0  # 이 offset(px)에서 가중 최대 — '코너' 판단 기준 픽셀
LANE_CORNER_MIN   = 40.0   # 코너 가중 시작 임계(px): 이 이하 offset(직선 흔들림)엔 가중 0 → 직선 안정
LANE_DEADZONE     = 40.0   # 중앙 데드존(px): 이내 offset은 직진(0)으로 간주 → 직선 휘청 억제(코너엔 영향 없음)
LANE_PREVIEW      = 0.38   # 코너 예측 조향 비중(0~1): 앞쪽 차선(lookahead) 섞는 비율.
# ── 슬라이딩윈도우 차선 인지 (BEV 시점) ──
LANE_ROI_TOP = 0.55         # ROI 상단(0=화면맨위,1=맨아래): 작을수록 멀리 봄
LANE_ROI_BOT = 0.95         # ROI 하단
LANE_LOOKAHEAD     = 0.22   # 예측감속용 원거리 행(ROI 위쪽 비율)
LANE_LOOKAHEAD_REF = 220.0  # 이 lookahead 편차(px)에서 예측감속 최대: 160→220 turn_preview 완화
SW_NWIN, SW_MARGIN, SW_MINPIX = 10, 110, 25  # 슬라이딩윈도우: 윈도우 수 / 폭 / 최소픽셀
LANE_YELLOW_WEIGHT = 0.4   # 노란 중앙선 블렌드 비중(0=비활성). 슬라이딩윈도우 추적 후 활성화
LANE_YELLOW_MAX_DEV = 80.0 # 노란선이 흰선 중앙에서 이 이상(px) 벗어나면 튄 것으로 보고 무시
# ── BEV 4점 캘리브레이션 (ROI 내 비율, 실측 후 조정) ──
# src: 투시 ROI에서 도로 4모서리 — (좌하, 우하, 우상, 좌상) 순서
# 차선 간격·소실점 위치에 맞게 x/y 비율을 시뮬에서 직접 재어 맞춤
BEV_SRC = np.float32([[0.009, 0.729],   # 좌하
                       [0.931, 0.729],   # 우하
                       [0.619, 0.021],   # 우상
                       [0.364, 0.021]])  # 좌상
# dst: BEV 출력에서 직사각형으로 펼 위치 (도로 폭을 동일 비율로 유지)
BEV_DST = np.float32([[0.15, 1.00],
                       [0.85, 1.00],
                       [0.85, 0.00],
                       [0.15, 0.00]])
LAVACON_KP   = 210.0
LAVACON_SMOOTH = 0.3   # 라바콘 편차 저역통과(EMA) 계수: 클수록 부드럽지만 느림(0~1)
LAVACON_OFFSET_CLAMP = 0.8  # 편차 물리한계(m): 콘 사이폭보다 크면 오검출(벽 등) → 잘라냄
LAVACON_AVOID = 2.8         # 한쪽 콘만 보일 때, 이 거리보다 '가까우면' 반대로 회피(멀면 직진, 다가가기 금지)
LAVACON_DONE_FRAMES = 80  # 우측콘 미검출이 연속 N프레임(20Hz→약 4초) 쌓이면 S2 전환(순간누락 디바운스)
SAFETY_DIST  = 5.0      # B1 발동 거리(m) — 보행자 5m, 차량은 run_behavior_fsm에서 4.5m 별도
AVOID_ANGLE  = 50.0     # 보행자 회피 조향
TURN_YAW     = math.radians(90)   # 좌회전 목표 회전각(rad) — 현재 미사용(시간기반 전환)
TURN_YAW_TOL = math.radians(8)    # 회전 완료 허용오차 — 현재 미사용
TURN_REVERSE_SPEED  = 100   # [진입] S3 교차로 좌회전 전 후진 속도
TURN_REVERSE_FRAMES = 4     # [진입] 후진 프레임 수 (0.18s→3.6f → 4f=0.20s ←애매, ±1 조정 권장)
TURN_FRAMES  = 16           # [진입] 좌회전 프레임 수 (0.8s→16.0f ✓)
TURN_SPEED   = 70           # [진입] S3 교차로 좌회전 속도
TURN_EXIT_REVERSE_SPEED  = 100  # [진출] S4 지름길 탈출 좌회전 전 후진 속도
TURN_EXIT_REVERSE_FRAMES = 5    # [진출] 후진 프레임 수 (0.22s→4.4f → 4f=0.20s ←애매, ±1 조정 권장)
TURN_EXIT_FRAMES = 12       # [진출] 좌회전 프레임 수 (0.6s→12.0f ✓)
TURN_EXIT_SPEED = 70        # [진출] S4 지름길 탈출 좌회전 속도
SHORTCUT_MIN_T = 3.0   # 지름길 진입 후 끝감지 활성화까지 최소 주행시간(s, 오판 방지)
SHORTCUT_MAX_T = 15.0  # 지름길 최대 주행시간(s, 끝 못 찾을 때 강제 탈출 백업)
STOPLINE_TH    = 0.95  # 정지선 판정: 한 행 흰색비율 임계(≈1.0=전폭 흰선만). 값은 0~1로 1.0이 최대
STOPLINE_COOLDOWN = 3.0 # S3→S2 복귀 후 이 시간(s)간 정지선 재감지 무시(같은 정지선 따다닥 전환 방지)
APPROACH_SPEED = 2.0    # [진입] 정지선 감지 후 S3 진입 전 감속 속도
APPROACH_TIME  = 1.0    # [진입] 감속 유지 시간(s)
APPROACH_EXIT_SPEED = 2.0  # [진출] S4 탈출 정지선 감지 후 감속 속도
APPROACH_EXIT_TIME  = 1.0  # [진출] 감속 유지 시간(s)
# 경찰차 감지 직사각형 ROI (차량 기준, 전방=x축, 좌측=y축 양수)
S3_ROI_FWD_MIN = 1.0   # 전방 최소(m)
S3_ROI_FWD_MAX = 8.0   # 전방 최대(m)
S3_ROI_LAT_MIN = 0.5   # 좌측 최소(m)
S3_ROI_LAT_MAX = 7.0   # 좌측 최대(m) — 박스 왼쪽 확장
S3_ROI_MIN_PTS = 3     # ROI 내 최소 포인트 수: 이 이상이어야 blocked 판정 (노이즈 오검출 방지)
FLOOR_ROI_TOP   = 0.6  # 어린이보호구역 감지 ROI 상단(0~1): 하단부만 봄
FLOOR_YELLOW_TH = 0.08 # 노란 비율 임계: 일반도로(<0.05)와 구역(0.08+) 사이
FLOOR_EXIT_FRAMES = 20 # 진입 후 연속 N프레임(20Hz→1초) 낮아야 해제 → 구역 내 깜빡임에도 감속 유지(sticky)
# ── 3구 신호등(S0 출발) ── ROI는 실측 위치로 튜닝(DEBUG_VIZ 'signal_roi' 보며 신호등만 들어오게)
SIG_ROI_T, SIG_ROI_B = 0.17, 0.32   # 신호등 ROI 상/하 (세로 비율) — test_lavacon 실측 튜닝값
SIG_ROI_L, SIG_ROI_R = 0.32, 0.63   # 신호등 ROI 좌/우 (가로 비율)
SIG_MIN_PIX_S0 = 2000                # [S0] 3구 신호등 점등 인정 최소 픽셀
SIG_MIN_PIX    = 3500                # [S3] 4구 신호등 점등 인정 최소 픽셀
# ── 4구 신호등(S3 교차로) ── ROI는 실측 위치로 튜닝(DEBUG_VIZ 'signal4_roi' 보며 4구 전체 들어오게)
SIG4_ROI_T, SIG4_ROI_B = 0.08, 0.28  # 4구 신호등 ROI 상/하 (세로 비율)
SIG4_ROI_L, SIG4_ROI_R = 0.04, 0.78  # 4구 신호등 ROI 좌/우 (가로 비율)
OVERTAKE_PASS_T   = 2.5  # 추월 최대 시간(백업 타이머, A)
OVERTAKE_RIGHT_A  = 45.0 # 우측 이동 조향
OVERTAKE_LEFT_A   = 45.0 # 좌측 복귀 조향
OVERTAKE_LANE_OFFSET = 112  # 추월 직진 중 흰선에서 유지할 거리(px): 클수록 차선 안쪽

# ── Behavior 게이팅 ──
# 보행자·차량·어린이보호구역 미션은 전부 차선인식(S2)에서만 나온다.
# → Behavior(B1 회피/B2 감속)는 S2에서만 켜고, 나머지 구간은 꺼서 오검출/오작동을 막는다.
#   (켤 상태가 S2 하나뿐이라 집합/in 검사 없이 control_loop에서 == 로 직접 비교)

# ── 개발/테스트 플래그 ──
# [정식 시작] S0부터: 3구 신호등 초록불 대기 → 출발
START_STATE     = MissionState.S2_LANE_FOLLOW #MissionState.S0_WAIT_GREEN #MissionState.S4_SHORTCUT #MissionState.S2_LANE_FOLLOW
ENABLE_BEHAVIOR = True
DEBUG_LOG       = True
DEBUG_PERIOD    = 0.5
DEBUG_VIZ       = False  # 차선/검출/4구 디버그 창
DEBUG_VIZ_LANE  = False  # 차선 슬라이딩윈도우 디버그 창 (lane_result)
DEBUG_VIZ_LIDAR = False  # 라이다 BEV 장애물 감지 디버그 창


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
        self.lane_valid  = False    # 차선 검출 여부 (지름길 갭/회전완료 판단에 사용)
        self.lane_lookahead = 0.0   # 원거리(앞쪽) 편차 → 코너 진입 전 예측감속용
        self._lane_prev_width = 448.0  # 도로폭(바깥 흰선↔흰선) 직전값(px, EMA). BEV 기준=(0.85-0.15)*roi_w≈448px
        # [2-2 신호등]
        self.signal_color   = 'unknown'  # [S0] 'red'/'yellow'/'blue'/'unknown'
        self.signal_red_on      = False  # [S3] 빨강
        self.signal_straight_on = False  # [S3] 직진(점등 위치)
        self.signal_left_on     = False  # [S3] 좌회전(점등 위치)
        self.intersection_ahead = False  # [S2→S3] 교차로 진입
        self.stopline = False            # [S4] 굵은 가로 흰선(정지선/지름길 끝 단서)
        self._stopline_cooldown_t = 0.0  # 이 시각까지 S2의 정지선 재감지 무시(S3 따다닥 방지)
        # [2-3 장애물(전방/측면)]
        self.obstacle_front = False   # 전방 장애물
        self.obstacle_dist  = 999.0   # 전방 거리(m)
        self.obstacle_side  = 'none'  # 'left'/'right'/'center'/'none'
        self.obstacle_type  = 'none'  # 'pedestrian'/'vehicle'/'none' (라이다 점수로 판별)
        self.left_blocked   = False   # [S3] 좌회전 경로(좌측) 막힘 = 경찰차(S3 전용 ROI)
        self.left_clear     = True    # 좌측 차선 비었는지(추월 복귀 판단)
        self.right_clear    = True    # 우측 차선 비었는지(추월 이동 판단)
        self._ema_y         = 0.0     # 전방 장애물 횡위치 EMA(obstacle_side 안정화)
        # [2-4 라바콘]
        self.lavacon_offset = 0.0
        self.lavacon_done   = False
        
        # [2-5 바닥글자]
        self.floor_zone = 'none'     # 'child_zone'/'release'/'none'
        self._floor_low = 0          # 어린이보호구역 해제 디바운스(연속 저비율 프레임 수)
        self._child_zone_enabled = False  # S4탈출/추월완료 후 ON, 구역 탈출 후 영구 OFF

        # ── 외부 차선 인식 모듈 (lane_util.py / perc_floor.py) 초기화 ──
        self.camera_processor = CameraProcessor()       # BEV 변환 및 색상 마스크(흰/노랑) 처리기
        self.slide_window_processor = SlideWindow()     # 슬라이딩 윈도우 기반 차선 탐색 및 피팅기
        self.lane_detector = LaneDetector(self.camera_processor, self.slide_window_processor) # 최종 통합 인식기

        # ── 판단/제어 상태 ──
        self.mission_state  = START_STATE
        self.behavior_state = BehaviorState.B0_NORMAL
        self.segment        = Segment.PEDESTRIAN     # 차선인식 미션 구간(보행자→차량)
        self.overtake_phase = Overtake.IDLE          # 차량 추월 sub-FSM
        self.ctrl_angle = 0.0
        self.ctrl_speed = SPEED_STOP
        self._pid_prev_error = 0.0
        self._pid_integral   = 0.0
        self._s1_phase      = 'straight'  # S1 라바콘 시퀀스 단계
        self._s1_frame_cnt  = 0           # S1 현재 단계 경과 프레임 수
        self._turn_yaw_start = None   # 좌회전 진행 중 플래그 (None=미회전)
        self._turn_frame_cnt = 0      # 좌회전 경과 프레임 수
        self._approach_t0    = None   # [진입] 정지선 감지 후 감속 시작 시각
        self._exit_approach_t0 = None # [진출] S4 탈출 정지선 감지 후 감속 시작 시각
        self._shortcut_t0    = None   # 지름길 진입 시각(끝감지 타이밍용)
        self._overtake_t0        = 0.0
        self._overtake_phase_int = 0   # 차량 추월 단계 정수 FSM (0=대기)
        self._overtake_frame_cnt = 0   # 현재 추월 단계 경과 프레임 수
        self._ped_phase          = 'idle'  # 보행자 C자 기동 단계 ('idle'/'p1'/'p2')
        self._ped_frame_cnt      = 0       # 현재 단계 경과 프레임 수
        self._s4_ref_yaw     = None   # S4 진입 1초 후 기록한 기준 yaw (탈출 좌회전 전 보정용)
        self._prev_speed     = 0.0    # 가속 속도제한용(직전 출력 속도)
        self._corner_hold    = 0.0    # 코너 활성도(감쇠 peak-hold): 연속 급코너 가속억제용
        self._lavacon_empty_cnt = 0   # 우측콘 연속 미검출 프레임 수(직진 후 S2 전환용)
        self._dbg_dL = self._dbg_dR = -1.0   # 디버그: 좌/우 콘 최근접 거리
        self._dbg_nL = self._dbg_nR = 0      # 디버그: 좌/우 콘 점 개수
        self._prev_behavior  = BehaviorState.B0_NORMAL
        self._last_debug_t   = 0.0

        # ── ROS 통신 ──
        self.motor_msg = XycarMotor()
        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)
        # 이미지(대용량)는 RELIABLE로 구독 — best-effort면 조각 유실 시 프레임 통째로 버려져
        # WSL2+CycloneDDS에서 카메라가 간헐적으로 안 들어오던 근본 원인(2026-06-22 확인)
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

    def drive(self, angle, speed):
        self.motor_msg.angle = float(np.clip(angle, -ANGLE_MAX, ANGLE_MAX))
        self.motor_msg.speed = float(np.clip(speed, -100.0, 100.0))
        for _ in range(7):
            self.motor_pub.publish(self.motor_msg)


    # #########################################################
    # [2] 인지 (Perception)
    #   ※ 각 기능별 [담당] = 코드를 채우는 팀 / [협업] = 그 결과를 쓰는 쪽
    #     - 카메라 기반(차선·신호등·바닥글자·정지선) → [담당] 비전팀
    #     - 라이다 기반(장애물·라바콘)               → [담당] 라이다팀
    #     - 모든 인지 결과의 소비자(판단/제어)        → [협업] 제어팀
    #   인지팀은 '인터페이스 변수'만 채우고, 그걸 제어팀이 읽어 주행에 사용한다.
    # #########################################################
    def perceive_all(self):
        self.perc_lane()        # 비전
        self.perc_signal()      # 비전
        self.perc_obstacle()    # 라이다
        self.perc_lavacon()     # 라이다
        self.perc_floor()       # 비전
        self.perc_stopline()    # 비전

    # [2-1] 차선
    #   [담당] 비전팀        [협업] 제어팀(S2 차선 PID, 추월 복귀, 좌회전 완료 판정에 사용)
    #   입력 self.img_front → 출력 self.lane_offset(우측+), self.lane_valid
    def perc_lane(self):
        if self.img_front is None:
            self.lane_valid = False
            return

        # perc_floor.py의 LaneDetector를 사용하여 차선 인식 수행
        valid, offset, lookahead, bev = self.lane_detector.detect(self.img_front)

        self.lane_valid = valid
        if valid:
            # 기존 제어 코드와 호환되도록 필터링 적용
            self.lane_offset = 0.7 * self.lane_offset + 0.3 * offset
            self.lane_lookahead = 0.5 * self.lane_lookahead + 0.5 * lookahead

    # [2-2] 신호등
    #   [담당] 비전팀        [협업] 제어팀(S0 출발, S3 직진/좌회전 분기)
    #   입력 self.img_front
    #   출력 [S0] signal_color / [S3] signal_red/straight/left_on / intersection_ahead
    #   주의 4구는 직진·좌회전 모두 초록 → 점등 '위치'로 구분
    def perc_signal(self):
        """신호등 판별 (상태별):
          S0 → 3구 색 판별 → signal_color('blue'=초록=출발)
          S3 → 4구 직진/좌회전 → 빨강 동반 여부로 구분(좌회전=초록+빨강 동시, 직진=초록만)"""
        if self.img_front is None:
            return
        h, w = self.img_front.shape[:2]

        if self.mission_state == MissionState.S0_WAIT_GREEN:
            roi = self.img_front[int(h*SIG_ROI_T):int(h*SIG_ROI_B), int(w*SIG_ROI_L):int(w*SIG_ROI_R)]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red = (cv2.inRange(hsv, np.array([0, 110, 110]),   np.array([10, 255, 255])) |
                   cv2.inRange(hsv, np.array([160, 110, 110]), np.array([180, 255, 255])))
            yellow = cv2.inRange(hsv, np.array([18, 110, 110]), np.array([35, 255, 255]))
            green  = cv2.inRange(hsv, np.array([45, 110, 110]), np.array([85, 255, 255]))
            rc, yc, gc = (int(np.count_nonzero(m)) for m in (red, yellow, green))
            if   rc > SIG_MIN_PIX_S0 and rc >= gc and rc >= yc: self.signal_color = 'red'
            elif gc > SIG_MIN_PIX_S0 and gc >= rc and gc >= yc: self.signal_color = 'blue'   # 초록=출발
            elif yc > SIG_MIN_PIX_S0:                           self.signal_color = 'yellow'
            else:                                            self.signal_color = 'unknown'

        elif self.mission_state == MissionState.S3_INTERSECTION:
            roi = self.img_front[int(h*SIG4_ROI_T):int(h*SIG4_ROI_B), int(w*SIG4_ROI_L):int(w*SIG4_ROI_R)]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red = (cv2.inRange(hsv, np.array([0, 110, 110]),   np.array([10, 255, 255])) |
                   cv2.inRange(hsv, np.array([160, 110, 110]), np.array([180, 255, 255])))
            green = cv2.inRange(hsv, np.array([45, 110, 110]), np.array([85, 255, 255]))
            rc, gc = int(np.count_nonzero(red)), int(np.count_nonzero(green))
            red_on, green_on = rc > SIG_MIN_PIX, gc > SIG_MIN_PIX
            self.signal_left_on     = green_on and red_on        # 좌회전: 초록(화살표)+빨강 동시
            self.signal_straight_on = green_on and not red_on    # 직진: 초록만
            self.signal_red_on      = red_on and not green_on    # 정지
            if DEBUG_VIZ:
                vis = roi.copy()
                cv2.putText(vis, f'R:{rc}  G:{gc}  th:{SIG_MIN_PIX}',
                            (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                state_txt = ('LEFT' if self.signal_left_on else
                             'STR'  if self.signal_straight_on else
                             'RED'  if self.signal_red_on else '---')
                cv2.putText(vis, state_txt, (4, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if 'LEFT' in state_txt or 'STR' in state_txt else (0, 0, 255) if 'RED' in state_txt else (180, 180, 180), 1, cv2.LINE_AA)
                cv2.imshow('signal4_roi', vis); cv2.waitKey(1)

    # [2-3] 장애물(전방+측면)
    #   [담당] 라이다팀      [협업] 제어팀(B1 보행자 회피·차량 추월, S3 경로 막힘 판단)
    #   입력 self.lidar_ranges
    #   출력 obstacle_front/dist/side, left_blocked(좌회전 경로=경찰차), left_clear, right_clear
    #   ※ 경찰차는 비전이 아니라 '교차로에서 좌측 경로가 라이다로 막혔는가'로 판단
    #      → 이 부분은 라이다팀이 채우되, S3 로직(제어팀)과 의미를 맞춰야 함(협업 포인트)
    def perc_obstacle(self):
        # ── 튜닝 파라미터 ──
        FRONT_X_MIN, FRONT_X_MAX = 0.0, 5.0   # 전방 ROI 종방향(m)
        FRONT_Y_HALF             = 1.5         # 전방 ROI 횡방향 반폭(m)
        FRONT_MIN_PTS            = 2           # 전방 장애물 확정 최소 포인트 (보행자 최소 2pts)
        FRONT_VEHICLE_PTS        = 12          # 이 이상이면 차량, 미만이면 보행자 (보행자 최대 10pts 실측)
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

        # ── 전방 장애물 (보행자/차량 공통) ──
        front_mask = valid & (x > FRONT_X_MIN) & (x < FRONT_X_MAX) & (np.abs(y) < FRONT_Y_HALF)
        front_cnt  = int(np.count_nonzero(front_mask))
        self.obstacle_front = front_cnt > FRONT_MIN_PTS
        if self.obstacle_front:
            self.obstacle_dist = float(np.min(r[front_mask]))
            self.obstacle_type = 'vehicle' if front_cnt >= FRONT_VEHICLE_PTS else 'pedestrian'
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

        # ── [S3 전용] 경찰차 ROI — left_blocked는 교차로 전용으로 별도 유지 ──
        if self.mission_state == MissionState.S3_INTERSECTION:
            x_fwd = ranges * np.cos(np.linspace(0.0, 2.0*math.pi, n, endpoint=False))
            y_lat = ranges * np.sin(np.linspace(0.0, 2.0*math.pi, n, endpoint=False))
            valid_s3 = np.isfinite(ranges) & (ranges > 0.1)
            in_roi = (valid_s3
                      & (x_fwd > S3_ROI_FWD_MIN) & (x_fwd < S3_ROI_FWD_MAX)
                      & (y_lat > S3_ROI_LAT_MIN)  & (y_lat < S3_ROI_LAT_MAX))
            self.left_blocked = bool(np.sum(in_roi) >= S3_ROI_MIN_PTS)
            s3_dist = float(np.min(ranges[in_roi])) if np.any(in_roi) else 999.0


        if DEBUG_VIZ_LIDAR:
            PPM = 50          # 1m = 50px
            W, H = 500, 500
            EX, EY = 250, 450  # 자차 위치(하단 중앙)
            bev = np.zeros((H, W, 3), dtype=np.uint8)

            # 거리 가이드 (1~5m 동심원, 회색)
            for d in range(1, 6):
                cv2.circle(bev, (EX, EY), d * PPM, (50, 50, 50), 1)
                cv2.putText(bev, f'{d}m', (EX + 4, EY - d*PPM + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

            # ROI 박스: 전방(노랑), 좌측(초록), 우측(주황)
            # 스크린 좌표: sx = EX - y*PPM, sy = EY - x*PPM
            def to_px(wx, wy): return (int(EX - wy*PPM), int(EY - wx*PPM))
            cv2.rectangle(bev, to_px(FRONT_X_MIN, FRONT_Y_HALF), to_px(FRONT_X_MAX, -FRONT_Y_HALF), (0, 220, 220), 1)   # 전방
            cv2.rectangle(bev, to_px(0.8, 1.5),  to_px(5.5,  0.7), (0, 220, 0),   1)   # 좌측
            cv2.rectangle(bev, to_px(0.8, -0.7), to_px(5.5, -1.5), (0, 140, 255), 1)   # 우측

            # 라이다 점 플로팅 (유효 포인트만)
            for i in range(len(r)):
                if not valid[i]: continue
                sx = int(EX - y[i] * PPM)
                sy = int(EY - x[i] * PPM)
                if not (0 <= sx < W and 0 <= sy < H): continue
                # front/left/right ROI 안이면 색상 강조
                if front_mask[i]:   col = (0, 0, 255)    # 전방 장애물 = 빨강
                elif left_mask[i]:  col = (0, 255, 0)    # 좌측 = 초록
                elif right_mask[i]: col = (0, 140, 255)  # 우측 = 주황
                else:               col = (60, 60, 60)   # 기타 = 어두운 회색
                cv2.circle(bev, (sx, sy), 2, col, -1)

            # 자차 마커
            cv2.circle(bev, (EX, EY), 7, (255, 220, 0), -1)
            cv2.line(bev, (EX, EY), (EX, EY - 18), (255, 220, 0), 2)

            # 상태 텍스트
            type_col = (0, 0, 255) if self.obstacle_front else (0, 255, 0)
            cv2.putText(bev, f'{self.obstacle_type.upper()} {self.obstacle_dist:.1f}m  {self.obstacle_side}  pts={front_cnt}',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, type_col, 1, cv2.LINE_AA)
            cv2.putText(bev, f'L:{"CLR" if self.left_clear else "BLK"}  R:{"CLR" if self.right_clear else "BLK"}',
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            if self.mission_state == MissionState.S3_INTERSECTION:
                s3_col = (0, 0, 255) if self.left_blocked else (0, 255, 0)
                cv2.putText(bev, f'[S3] {"BLOCKED" if self.left_blocked else "CLEAR"}',
                            (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, s3_col, 1, cv2.LINE_AA)
            cv2.imshow('lidar_bev', bev)
            cv2.waitKey(1)

    # [2-4] 라바콘
    #   [담당] 라이다팀      [협업] 제어팀(S1 라바콘 주행 조향)
    #   ※ offset 단위/부호를 제어팀과 합의해야 함(LAVACON_KP 튜닝이 여기 의존) — 협업 포인트
    def perc_lavacon(self):
        self.lavacon_offset, self.lavacon_done = process_lavacon(self.lidar_ranges)

    # [2-5] 바닥글자(어린이 보호구역)
    #   [담당] 비전팀        [협업] 제어팀(B2 감속 발동)
    def perc_floor(self):
        """어린이보호구역 감지: 하단 ROI의 노란색 비율이 임계 초과면 child_zone.
        근거: 보호구역은 측면 노란차선+바닥 노란글자로 노란색이 많고, 일반도로는 중앙 점선뿐이라 적음."""
        if self.img_front is None:
            self.floor_zone = 'none'
            return
        # 활성화 전에는 감지 스킵
        if not self._child_zone_enabled:
            self.floor_zone = 'none'
            return
        h, w = self.img_front.shape[:2]
        roi = self.img_front[int(h * FLOOR_ROI_TOP):h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([40, 255, 255]))
        ratio = np.count_nonzero(yellow) / yellow.size
        # 진입은 즉시, 해제는 연속 N프레임 낮을 때만(sticky) → 구역 내 깜빡임에도 감속 유지
        if ratio > FLOOR_YELLOW_TH:
            self.floor_zone = 'child_zone'
            self._floor_low = 0
        elif self.floor_zone == 'child_zone':
            self._floor_low += 1
            if self._floor_low >= FLOOR_EXIT_FRAMES:
                self.floor_zone = 'none'
                self._child_zone_enabled = False  # 구역 탈출 → 영구 비활성
                self.get_logger().info('[floor] 어린이보호구역 탈출 → 감지 OFF')
        else:
            self.floor_zone = 'none'

    # [2-6] 정지선(굵은 가로 흰선)
    #   [담당] 비전팀        [협업] 제어팀(S4 지름길 끝 = 탈출 좌회전 시점)
    #   입력 self.img_front → 출력 self.stopline
    #   용도 : 지름길 끝(탈출 좌회전 지점) 단서. 하단 ROI에서 가로로 긴 흰색 행 탐지.
    #   ※ 카메라 시점 이미지 확보 후 ROI 높이/임계(STOPLINE_TH) 튜닝 필요.
    def perc_stopline(self):
        if self.img_front is None:
            self.stopline = False
            return

        # perc_floor.py의 check_stopline 함수 사용
        self.stopline = check_stopline(self.img_front)

    # #########################################################
    # [3] 판단 (Decision)
    #   [담당] 제어팀
    #   [협업] 인지 결과(비전 lane/signal/floor, 라이다 obstacle/lavacon, IMU yaw)를
    #          읽어서 상태 전환과 기본 주행을 결정한다. 인지팀과는 '인터페이스 변수'로만 연결.
    # #########################################################
    def run_mission_fsm(self):
        {
            MissionState.S0_WAIT_GREEN  : self._s0_wait_green,
            MissionState.S1_LAVACON     : self._s1_lavacon,
            MissionState.S2_LANE_FOLLOW : self._s2_lane_follow,
            MissionState.S3_INTERSECTION: self._s3_intersection,
            MissionState.S4_SHORTCUT    : self._s4_shortcut,
            MissionState.S5_FINISH      : self._s5_finish,
        }[self.mission_state]()

    def _change_state(self, new_state):
        """
        Mission 상태 전환 공통 처리.
          - 전환 로그 출력(디버깅 추적용)
          - PID 누적값 초기화: 이전 상태에서 쌓인 적분/미분 잔여가
            새 상태로 넘어와 튀는 것을 방지한다.
        모든 상태 전환은 반드시 이 함수를 통해서만 한다(직접 대입 금지).
        """
        self.get_logger().info(f'[전환] {self.mission_state.name} → {new_state.name}')
        prev_state = self.mission_state
        self.mission_state = new_state
        self._pid_prev_error = 0.0
        self._pid_integral   = 0.0
        self.ctrl_angle = 0.0
        self.ctrl_speed = SPEED_STOP
        # S3 진입 시 신호값 초기화 (안정화는 S2 감속구간에서 이미 완료)
        if new_state == MissionState.S3_INTERSECTION:
            self.signal_red_on      = False
            self.signal_straight_on = False
            self.signal_left_on     = False
        # S1 진입 시 라바콘 시퀀스 초기화
        if new_state == MissionState.S1_LAVACON:
            self._s1_phase     = 'straight'
            self._s1_frame_cnt = 0
        # S2 진입 시 감속 플래그 초기화
        if new_state == MissionState.S2_LANE_FOLLOW:
            self._approach_t0 = None
            # S1(라바콘) 직후 S2 진입 시 3초간 정지선 감지 억제 (라바콘 출구 오감지 방지)
            if prev_state == MissionState.S1_LAVACON:
                self._stopline_cooldown_t = time.time() + 3.0
        # S4 진입 시 탈출 감속 플래그 + 기준 yaw 초기화
        if new_state == MissionState.S4_SHORTCUT:
            self._exit_approach_t0 = None
            self._s4_ref_yaw       = None

    # ── S0: 출발 (미션① 신호등) ──
    def _s0_wait_green(self):
        """
        출발선에서 정지한 채 3구 신호등을 본다.
          - 파란불 전: 완전 정지 (신호위반 감점 방지)
          - 파란불 감지: S1(라바콘)로 전환하여 출발
        입력: self.signal_color (perc_signal이 채움)
        """
        self.ctrl_angle, self.ctrl_speed = 0.0, SPEED_STOP
        if self.signal_color == 'blue':
            self._change_state(MissionState.S1_LAVACON)

    # ── S1: 라바콘 주행 (미션②) ──
    def _s1_lavacon(self):
        # 보로노이 편차를 비례 제어(P제어)로 조향에 반영
        self.ctrl_angle = self.lavacon_offset * LAVACON_KP
        self.ctrl_speed = SPEED_LAVACON

        # 우측 콘이 끝나면 S2(차선 인식)로 전환
        if self.lavacon_done:
            self.get_logger().info('라바콘 구간 통과 완료 → S2 전환')
            self.segment = Segment.PEDESTRIAN
            self._change_state(MissionState.S2_LANE_FOLLOW)

    # ── S2: 차선인식 주행 (미션③, 주력 상태) ──
    def _s2_lane_follow(self):
        """
        차선을 따라 안정 주행(전체 주행의 대부분).
          - 조향: 차선 중앙편차(lane_offset)를 PID로 변환
          - 속도: 일반 속도
          - 전환: 교차로(신호등+정지선) 감지 시 S3로
        ※ 보행자 회피·차량 추월은 이 함수가 아니라 Behavior(B1)가 덮어쓴다.
          (S2는 '기본 주행'만 책임지고, 돌발 대응은 상위 Behavior가 담당)
        입력: self.lane_offset, self.intersection_ahead
        """
        # 보행자/추월 기동 중 PID 완전 중단 — 차선 이탈 중 적분 오염 방지
        if self._overtake_phase_int != 0 or self._ped_phase != 'idle':
            return

        if self._approach_t0 is not None:
            # 감속 구간: 차선 조향 유지 + 극저속 → 거의 정지 상태로 S3 진입
            elapsed = time.time() - self._approach_t0
            self.ctrl_angle = self._lane_pid(
                (1.0 - LANE_PREVIEW) * self.lane_offset + LANE_PREVIEW * self.lane_lookahead)
            self.ctrl_speed = APPROACH_SPEED
            self._prev_speed = APPROACH_SPEED
            if elapsed >= APPROACH_TIME:
                self._change_state(MissionState.S3_INTERSECTION)
        else:
            self._lane_drive()
            if self.stopline and time.time() >= self._stopline_cooldown_t:  # 정지선 감지(쿨다운 지난 뒤만)
                self._approach_t0 = time.time()                             # 감속 구간 시작

    # ── S3: 교차로 — 정지 후 라이다로 경로 판단 ──
    def _s3_intersection(self):
        """
        4구 신호등 교차로 진입 후 흐름:
          1. 진입 즉시 정지 (기본값 STOP, 명시적 신호만 출발)
          2. 라이다(left_blocked)로 경찰차 유무 판단
             - 막힘: 직진 초록(signal_straight_on) → 직진 후 S2 복귀; 그 외 → 정지 대기
             - 열림: 좌회전 신호(초록+빨강, signal_left_on) → 좌회전 후 S4; 그 외 → 정지 대기
          3. 좌회전 진행 중이면 신호와 무관하게 완료 우선
        ※ 진입 시 _change_state에서 신호값 초기화 → stale 오작동 원천 차단
        """
        # 좌회전 진행 중이면 완료 우선 (신호 상태와 무관)
        if self._turn_yaw_start is not None:
            self._do_left_turn(next_state=MissionState.S4_SHORTCUT)
            return

        # 기본: 명시적 출발 신호가 올 때까지 정지 대기 (라이다/신호 안정화는 S2 감속구간에서 완료)
        self.ctrl_angle, self.ctrl_speed = 0.0, SPEED_STOP

        if self.left_blocked:
            # 경찰차 있음 → 직진 초록 신호만 기다림
            if self.signal_straight_on:
                self.ctrl_angle = 0.0
                self.ctrl_speed = SPEED_NORMAL
                self.segment = Segment.PEDESTRIAN
                self._stopline_cooldown_t = time.time() + STOPLINE_COOLDOWN
                self._change_state(MissionState.S2_LANE_FOLLOW)
        else:
            # 경찰차 없음 → 좌회전 신호(초록+빨강 동시, signal_left_on) 기다림
            if self.signal_left_on:
                self._begin_left_turn()

    # ── S4: 지름길 — 직진(+차선소실 대비), 끝에서 좌회전 ──
    def _s4_shortcut(self):
        """
        지름길 직진. 중간 차선소실 구간은 라이다로 딸 것이 없으므로 그냥 직진.
        끝에 도달하면 신호없이 좌회전(yaw90)으로 차선인식 복귀.
        """
        # 탈출 좌회전 진행 중이면 마무리 우선
        if self._turn_yaw_start is not None:
            self._do_left_turn(next_state=MissionState.S2_LANE_FOLLOW)
            return

        # 지름길 진입 시각 기록(최초 1회) — 좌회전 완료 후 재설정 방지를 위해 turn 체크 이후에 위치
        if self._shortcut_t0 is None:
            self._shortcut_t0 = time.time()

        # 진입 1초 후 기준 yaw 기록 (초기 흔들림 지난 뒤 안정된 방향 저장)
        if self._s4_ref_yaw is None and (time.time() - self._shortcut_t0) >= 1.0:
            self._s4_ref_yaw = self.imu_yaw
            self.get_logger().info(f'[S4] 기준 yaw 기록: {math.degrees(self._s4_ref_yaw):.1f}°')

        if self._shortcut_end():
            if self._exit_approach_t0 is None:
                self._exit_approach_t0 = time.time()    # yaw 보정 구간 시작
            elapsed = time.time() - self._exit_approach_t0
            if elapsed < APPROACH_EXIT_TIME:
                # yaw 보정: 기준 방향과 현재 yaw 차이로 P제어 조향
                if self._s4_ref_yaw is not None:
                    yaw_err = self._yaw_delta(self._s4_ref_yaw)
                    self.ctrl_angle = float(np.clip(-yaw_err * 100.0, -30.0, 30.0))
                else:
                    self.ctrl_angle = 0.0
                self.ctrl_speed = APPROACH_EXIT_SPEED
            else:
                self._shortcut_t0 = None                # 다음 바퀴 위해 리셋
                self._exit_approach_t0 = None
                self._begin_left_turn()                 # 탈출 좌회전 시작
            return

        if self.lane_valid:
            self._lane_drive()                                   # 차선 보이면 S2와 동일 조향+감속
        else:
            self.ctrl_angle = 0.0                                # 차선소실 → 직진 유지
            self.ctrl_speed = SPEED_NORMAL

    def _shortcut_end(self):
        """
        지름길 끝(탈출 좌회전 지점) 감지.
          - 진입 후 SHORTCUT_MIN_T 전에는 비활성(진입 직후 정지선/차선 오판 방지)
          - 1차: 정지선(굵은 가로 흰선) 감지
          - 백업: SHORTCUT_MAX_T 초과 시 강제 끝 처리(끝 못 찾는 상황 대비)
        ※ 카메라 시점 이미지 확보 후 정지선 임계/시간값 재튜닝 권장.
        """
        if self._shortcut_t0 is None:
            return False
        elapsed = time.time() - self._shortcut_t0
        if elapsed < SHORTCUT_MIN_T:
            return False
        return self.stopline or elapsed > SHORTCUT_MAX_T

    # ── S5: 종료 ──
    def _s5_finish(self):
        self.ctrl_angle, self.ctrl_speed = 0.0, SPEED_STOP

    # ── 좌회전 (IMU yaw 90도) 공통 ──
    def _begin_left_turn(self):
        self._turn_yaw_start = self.imu_yaw   # 플래그로만 사용 (None 여부 체크)
        self._turn_frame_cnt = 0
        self.get_logger().info(f'좌회전 시작 (후진{TURN_REVERSE_FRAMES}f → 좌회전{TURN_FRAMES}f)')

    def _do_left_turn(self, next_state):
        """후진(TURN_REVERSE_FRAMES) → 최대 조향 좌회전(TURN_FRAMES) 후 next_state로 전환.
        Behavior(B1/B2)는 S3/S4에서 OFF이므로 이 함수의 ctrl_angle/speed는 그대로 발행된다."""
        # 진입(S3→S4) / 진출(S4→S2) 파라미터 분기
        if next_state == MissionState.S4_SHORTCUT:
            rev_spd, rev_f, trn_spd, trn_f = TURN_REVERSE_SPEED, TURN_REVERSE_FRAMES, TURN_SPEED, TURN_FRAMES
        else:
            rev_spd, rev_f, trn_spd, trn_f = TURN_EXIT_REVERSE_SPEED, TURN_EXIT_REVERSE_FRAMES, TURN_EXIT_SPEED, TURN_EXIT_FRAMES

        if self._turn_frame_cnt < rev_f:
            self.ctrl_angle = 0.0
            self.ctrl_speed = -rev_spd              # 후진
        elif self._turn_frame_cnt < rev_f + trn_f:
            self.ctrl_angle = -ANGLE_MAX
            self.ctrl_speed = trn_spd               # 좌회전
        else:
            self.get_logger().info('좌회전 완료')
            self._turn_yaw_start = None
            self._turn_frame_cnt = 0
            if next_state == MissionState.S2_LANE_FOLLOW:
                self._child_zone_enabled = True
                self.get_logger().info('[floor] 어린이보호구역 감지 ON (S4 탈출)')
                self._stopline_cooldown_t = time.time() + STOPLINE_COOLDOWN
            self._change_state(next_state)
            return
        self._turn_frame_cnt += 1

    def _yaw_delta(self, start):
        """현재 yaw - start (−π~π wrap)"""
        d = self.imu_yaw - start
        return math.atan2(math.sin(d), math.cos(d))

    # ── Behavior FSM (override 우선순위 B1>B2>B0) ──
    def run_behavior_fsm(self):
        """
        매 주기 위험/규칙 상황을 평가해 behavior_state를 결정한다(우선순위 B1>B2>B0).
          B1(안전): 전방 장애물이 안전거리 이내 OR 추월 진행 중(끝까지 유지)
          B2(정책): 어린이 보호구역 진입
          B0(정상): 위 둘 다 아님
        ※ 추월이 시작되면(overtake_phase != IDLE) 장애물이 잠깐 안 잡혀도
          B1을 유지해 추월 시퀀스가 중간에 끊기지 않게 한다.
        """
        is_overtaking = self._overtake_phase_int != 0

        # obstacle_type이 vehicle이면 즉시 VEHICLE 세그먼트로 전환 (보행자→차량 미션 순서 보장)
        # DONE 상태에서는 전환 금지 — 추월 완료 후 재트리거 차단
        if (self.obstacle_type == 'vehicle' or is_overtaking) and self.segment != Segment.DONE:
            self.segment = Segment.VEHICLE

        # 차량은 4.5m 조기 발동(미리 준비), 보행자는 기존 안전거리
        trigger_dist = 4.5 if self.segment == Segment.VEHICLE else SAFETY_DIST
        b1 = (self.obstacle_front and self.obstacle_dist < trigger_dist) or is_overtaking

        if b1:
            self.behavior_state = BehaviorState.B1_SAFETY
        elif self.floor_zone == 'child_zone':
            self.behavior_state = BehaviorState.B2_POLICY
        else:
            self.behavior_state = BehaviorState.B0_NORMAL

        # 추월/보행자 기동 중에는 _update_segment 차단 (기동 완료 전 segment 자동전환 방지)
        if is_overtaking or self._ped_phase != 'idle':
            self._prev_behavior = self.behavior_state
        else:
            self._update_segment()

    def _update_segment(self):
        """보행자 회피(B1)가 끝나면 다음은 차량 구간으로 전환 (보행자→차량 순서)."""
        finished_b1 = (self._prev_behavior == BehaviorState.B1_SAFETY
                       and self.behavior_state != BehaviorState.B1_SAFETY)
        if finished_b1 and self.segment == Segment.PEDESTRIAN \
           and self.overtake_phase == Overtake.IDLE \
           and self._ped_phase == 'idle':   # 역C자 기동 완전히 끝난 후에만 전환
            self.segment = Segment.VEHICLE
            self.get_logger().info('보행자 구간 통과 → 차량(추월) 구간')
        self._prev_behavior = self.behavior_state


    # #########################################################
    # [4] 제어 (Control)
    #   [담당] 제어팀
    #   [협업] 인지가 준 편차/거리 값을 실제 angle/speed로 변환.
    #          게인·각도·시간 튜닝값은 인지 출력 단위(offset 부호/스케일 등)에 의존하므로
    #          비전·라이다팀과 단위 합의가 필요(특히 lane_offset, lavacon_offset).
    # #########################################################
    def _lane_drive(self):
        """S2/S4 공통 차선 조향+감속 로직. ctrl_angle·ctrl_speed·_prev_speed·_corner_hold 갱신."""
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
        """
        차선 중앙편차(offset)를 PID 제어로 조향각(angle)으로 변환한다.
          P(비례) : 현재 오차에 비례 — 많이 벗어날수록 많이 꺾음
          I(적분) : 오차 누적 — 한쪽으로 계속 치우치는 정상편차 보정 (현재 Ki=0)
          D(미분) : 오차 변화율 — 급변 억제(진동 감쇠)
        부호 약속: offset 우측 치우침(+) → 우회전(angle +).
                  시뮬에서 반대로 가면 마지막 return에 -1을 곱해 보정.
        deadzone=0 으로 호출하면 데드존 없이 동작 (추월 P3/P6 등에서 사용).
        """
        # 데드존: 중앙 근처 미세 offset은 0으로 → 직선에서 노이즈 추종(미세 휘청) 차단
        if abs(offset) < deadzone:
            offset = 0.0
        self._pid_integral += offset                 # 적분: 오차 누적
        deriv = offset - self._pid_prev_error        # 미분: 직전 대비 변화량
        self._pid_prev_error = offset                # 다음 주기용 저장
        # 코너 가중: LANE_CORNER_MIN 이상(=실제 코너)에서만 작동 → 직선의 작은 offset엔 base KP 유지
        boost_ratio = min(1.0, max(0.0, abs(offset) - LANE_CORNER_MIN) / (LANE_CORNER_REF - LANE_CORNER_MIN))
        kp_eff = LANE_KP * (1.0 + LANE_CORNER_BOOST * boost_ratio)
        angle = kp_eff*offset + LANE_KI*self._pid_integral + LANE_KD*deriv
        return float(np.clip(angle, -ANGLE_MAX, ANGLE_MAX))

    def apply_behavior_override(self):
        """
        Behavior 상태에 따라 Mission이 계산한 ctrl_angle/ctrl_speed를 덮어쓴다.
          B1(안전): 구간(segment)에 따라 보행자 회피 / 차량 추월로 분기.
                    조향+속도를 모두 덮어씀(완전 가로채기).
          B2(정책): 속도만 SPEED_SLOW로 제한. 조향은 Mission 결과 유지.
          B0(정상): 아무것도 안 함(Mission 출력 그대로).
        ※ B1 > B2 우선순위는 run_behavior_fsm에서 이미 결정되어 들어온다.
        """
        if self._ped_phase != 'idle':
            # 역C자 기동 진행 중 — B 상태와 무관하게 완주 (보행자가 사라져도 멈추지 않음)
            self._handle_pedestrian()
        elif self.behavior_state == BehaviorState.B1_SAFETY:
            if self.segment == Segment.VEHICLE:
                self._handle_overtake()
            elif self.segment == Segment.PEDESTRIAN:
                self._handle_pedestrian()    # 최초 트리거 (idle → p1 시작)
            # DONE이면 B1이어도 아무것도 안 함
        elif self.behavior_state == BehaviorState.B2_POLICY:
            self.ctrl_speed = min(self.ctrl_speed, SPEED_SLOW)

    # ── B1-보행자: 역C자 2단계 기동 (오른쪽 회피) ──
    def _handle_pedestrian(self):
        """
        보행자 방향 무관, 5m 이내 감지 시 역C자(⊃) 기동으로 오른쪽 회피.
        idle → (감지) → p1(오른쪽 꺾기) → p2(왼쪽 복귀) → p3(역조향 방향잡기) → idle
        """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # [1단계] 오른쪽 꺾기 — 보행자 옆으로 비켜남
        P1_ANGLE  =  65.0  # 우측 조향각(°)
        P1_SPEED  =  20.0  # 속도
        P1_FRAMES =   4    # 유지 프레임 수 (20Hz 기준 0.2s)

        # [2단계] 왼쪽 복귀 — 원래 차선으로 돌아옴
        P2_ANGLE  = -100.0  # 좌측 조향각(°)
        P2_SPEED  =  15.0   # 속도
        P2_FRAMES =  33     # 유지 프레임 수 33 (20Hz 기준 1.95s)

        # [3단계] 역조향 — 차체 방향 잡기 (카운터스티어)
        P3_ANGLE  =  100  # 우측 조향각(°)
        P3_SPEED  =  15.0  # 속도
        P3_FRAMES =  26    # 유지 프레임 수 22가 원본 (20Hz 기준 0.5s)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        is_ped = self.obstacle_front and self.obstacle_type == 'pedestrian'

        if self._ped_phase == 'idle':
            if is_ped:
                self._ped_phase     = 'p1'
                self._ped_frame_cnt = 0
                self.get_logger().info(f'[PED] 역C자 시작 dist={self.obstacle_dist:.1f}m side={self.obstacle_side}')
            # else: 아무것도 안 함

        if self._ped_phase == 'p1':  # elif → if: 트리거 프레임에서 즉시 P1 조향 적용
            self.ctrl_speed = P1_SPEED
            self.ctrl_angle = P1_ANGLE
            self._ped_frame_cnt += 1
            self.get_logger().info(f'[PED|P1] f={self._ped_frame_cnt}/{P1_FRAMES} ang={P1_ANGLE} spd={P1_SPEED}')
            if self._ped_frame_cnt >= P1_FRAMES:
                self._ped_phase     = 'p2'
                self._ped_frame_cnt = 0
                self.get_logger().info('[PED] P1→P2 왼쪽 복귀')

        elif self._ped_phase == 'p2':
            self.ctrl_speed = P2_SPEED
            self.ctrl_angle = P2_ANGLE
            self._ped_frame_cnt += 1
            self.get_logger().info(f'[PED|P2] f={self._ped_frame_cnt}/{P2_FRAMES} ang={P2_ANGLE} spd={P2_SPEED}')
            if self._ped_frame_cnt >= P2_FRAMES:
                self._ped_phase     = 'p3'
                self._ped_frame_cnt = 0
                self.get_logger().info('[PED] P2→P3 역조향 방향잡기')

        elif self._ped_phase == 'p3':
            self.ctrl_speed = P3_SPEED
            self.ctrl_angle = P3_ANGLE
            self._ped_frame_cnt += 1
            self.get_logger().info(f'[PED|P3] f={self._ped_frame_cnt}/{P3_FRAMES} ang={P3_ANGLE} spd={P3_SPEED}')
            if self._ped_frame_cnt >= P3_FRAMES:
                self._ped_phase      = 'idle'
                self._ped_frame_cnt  = 0
                self._pid_prev_error = 0.0
                self._pid_integral   = 0.0
                self.segment = Segment.VEHICLE   # 즉시 차량 구간으로 전환 → 재트리거 차단
                self.get_logger().info('[PED] 역C자 완료 → 차량(추월) 구간')

    # ── B1-차량: 6단계 추월 (우측이동→우측차선주행→좌측복귀→좌측차선주행) ──
    def _handle_overtake(self):
        """
        0.대기 → 1.우측꺾기 → 2.우측안정화 → 3.우측차선주행
               → 4.좌측꺾기 → 5.좌측안정화 → 6.좌측차선주행 → IDLE(S2복귀)
        각 단계 파라미터는 아래 블록에서 단계별로 수정.
        """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # [0단계] 대기 — 전방 차량 감지 시 추월 시작
        # ※ 감지 파라미터(ROI/점수 기준): perc_obstacle 내 541~544번 줄
        OVERTAKE_TRIGGER = 6.5     # 추월 시작 거리(m)

        # [1단계] 우측 꺾기 — 차선 변경 조향
        P1_ANGLE  = 70           # 우측 조향각(°)
        P1_FRAMES = 12             # 유지 프레임 수 (0.6s = 12f)
        P1_SPEED  = 12           # 속도

        # [2단계] 우측 안정화 — 카운터스티어로 차체 평행화
        P2_ANGLE  = -70          # 카운터 조향각(°)
        P2_FRAMES = 8           # 유지 프레임 수 (0.5s = 10f)
        P2_SPEED  = 12           # 속도

        # [3단계] 우측 차선 주행 — 차선 PID로 흰선 기준 직진
        P3_FRAMES = 50             # 주행 프레임 수 (1.5s = 30f)
        P3_SPEED  = 15.0           # 속도

        # [4단계] 좌측 꺾기 — 원래 차선으로 복귀 조향
        P4_ANGLE  = -35.0          # 좌측 조향각(°)
        P4_FRAMES = 25             # 유지 프레임 수 (1.0s = 20f)
        P4_SPEED  = 12.0           # 속도

        # [5단계] 좌측 안정화 — 카운터스티어로 차체 평행화
        P5_ANGLE  = 70           # 카운터 조향각(°)
        P5_FRAMES = 16             # 유지 프레임 수 (0.7s = 14f)
        P5_SPEED  = 12.0           # 속도

        # [6단계] 좌측 차선 주행 — 차선 PID로 흰선 기준 직진 후 복귀
        P6_FRAMES = 20             # 주행 프레임 수 (1.0s = 20f): 이후 IDLE → S2 인계
        P6_SPEED  = 15.0           # 속도
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        p = self._overtake_phase_int

        # 0. 대기
        if p == 0:
            if self.obstacle_front and self.obstacle_dist < OVERTAKE_TRIGGER:
                self._overtake_phase_int = 1
                self._overtake_frame_cnt = 0
                self.overtake_phase      = Overtake.PASS_RIGHT
                self.get_logger().info(f'[OVERTAKE] 시작 dist={self.obstacle_dist:.2f}m')
                p = 1   # fall-through: 트리거 프레임에서 즉시 P1 조향 적용
            else:
                return

        # 1. 우측 꺾기
        if p == 1:  # elif → if: fall-through 허용
            self.ctrl_speed = P1_SPEED
            self.ctrl_angle = P1_ANGLE
            self._overtake_frame_cnt += 1
            if self._overtake_frame_cnt >= P1_FRAMES:
                self._overtake_phase_int = 2
                self._overtake_frame_cnt = 0
                self.get_logger().info('[OVERTAKE] 1→2 우측 안정화')

        # 2. 우측 안정화 (카운터스티어)
        elif p == 2:
            self.ctrl_speed = P2_SPEED
            self.ctrl_angle = P2_ANGLE
            self._overtake_frame_cnt += 1
            if self._overtake_frame_cnt >= P2_FRAMES:
                self._overtake_phase_int = 3
                self._overtake_frame_cnt = 0
                self._pid_prev_error = 0.0
                self._pid_integral   = 0.0
                self.get_logger().info('[OVERTAKE] 2→3 우측 차선 주행')

        # 3. 우측 차선 주행 (차선 PID)
        elif p == 3:
            self.ctrl_speed = P3_SPEED
            self.ctrl_angle = self._lane_pid(self.lane_offset, deadzone=0.0) if self.lane_valid else 0.0
            self._overtake_frame_cnt += 1
            if self._overtake_frame_cnt >= P3_FRAMES:
                self._overtake_phase_int = 4
                self._overtake_frame_cnt = 0
                self.overtake_phase      = Overtake.RETURN_LEFT
                self.get_logger().info('[OVERTAKE] 3→4 좌측 꺾기')

        # 4. 좌측 꺾기
        elif p == 4:
            self.ctrl_speed = P4_SPEED
            self.ctrl_angle = P4_ANGLE
            self._overtake_frame_cnt += 1
            if self._overtake_frame_cnt >= P4_FRAMES:
                self._overtake_phase_int = 5
                self._overtake_frame_cnt = 0
                self.get_logger().info('[OVERTAKE] 4→5 좌측 안정화')

        # 5. 좌측 안정화 (카운터스티어)
        elif p == 5:
            self.ctrl_speed = P5_SPEED
            self.ctrl_angle = P5_ANGLE
            self._overtake_frame_cnt += 1
            if self._overtake_frame_cnt >= P5_FRAMES:
                self._overtake_phase_int = 6
                self._overtake_frame_cnt = 0
                self._pid_prev_error = 0.0
                self._pid_integral   = 0.0
                self.get_logger().info('[OVERTAKE] 5→6 좌측 차선 주행')

        # 6. 좌측 차선 주행 (차선 PID) → 완료
        elif p == 6:
            self.ctrl_speed = P6_SPEED
            self.ctrl_angle = self._lane_pid(self.lane_offset, deadzone=0.0) if self.lane_valid else 0.0
            self._overtake_frame_cnt += 1
            if self._overtake_frame_cnt >= P6_FRAMES:
                self._overtake_phase_int = 0
                self._overtake_frame_cnt = 0
                self.overtake_phase      = Overtake.IDLE
                self.segment             = Segment.DONE   # 추월 완료 → 재트리거 영구 차단
                self._child_zone_enabled = True
                self.get_logger().info('[OVERTAKE] 완료 → DONE / 어린이보호구역 감지 ON')


    # #########################################################
    # [5] 메인 루프
    # #########################################################
    def control_loop(self):
        """
        20Hz(0.05초)마다 호출되는 제어의 심장.
        매 주기 '인지 → 판단 → 제어 → 발행' 한 사이클을 순서대로 실행한다.

          1) perceive_all()      : 센서 원본 → 인터페이스 변수 갱신
          2) run_mission_fsm()   : 현재 코스 단계의 기본 조향/속도 계산
          3) (게이팅) Behavior    : 허용 상태에서만 위험/규칙 override
          4) drive()             : 최종 명령을 모터 토픽으로 발행
          5) (옵션) 디버그 로그

        ※ Behavior 게이팅: 차선인식(S2) 상태에서만 B1/B2가 작동.
          지름길(S4) 등 다른 구간에선 꺼져서 오검출로 인한 오작동을 막는다.
        """
        self.perceive_all()                 # 1. 인지
        self.run_mission_fsm()              # 2. 판단(Mission)

        # 3. Behavior는 차선인식(S2) 상태에서만 동작 (그 외 구간은 OFF)
        if ENABLE_BEHAVIOR and self.mission_state == MissionState.S2_LANE_FOLLOW:
            self.run_behavior_fsm()         #    위험/규칙 상태 결정
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
            f'[{self.mission_state.name}|{self.behavior_state.name}|{self.segment.name}|{self.overtake_phase.name}] '
            f'ang={self.ctrl_angle:+.1f} spd={self.ctrl_speed:.1f} '
            f'lane={self.lane_offset:+.1f}({int(self.lane_valid)}) '
            f'obs={self.obstacle_front}({self.obstacle_dist:.1f}m,{self.obstacle_side}) '
            f'Lblk={int(self.left_blocked)} '
            f'lava={self.lavacon_offset:+.2f}(done={int(self.lavacon_done)}) '
            f'L={self._dbg_dL:.2f}({self._dbg_nL}) R={self._dbg_dR:.2f}({self._dbg_nR})')


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
