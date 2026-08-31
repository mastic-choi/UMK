#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# config.py — 실차 테스트용 파라미터 통합 파일.
#
# 이 프로젝트의 거의 모든 모듈(track_drive.py, perception/*, controller/*)이
# 튜닝 파라미터를 여기서 가져다 씁니다(`from .config import *` 또는
# `from ..config import ...`). 즉 실차 테스트 중 값을 바꿔야 할 때 이 파일
# 하나만 고치면 됩니다 — 개별 모듈 파일을 헤집을 필요가 없습니다.
#
# ── 이 파일에 없는 값들 ──
#   전부 다 여기 모은 건 아닙니다. OpenCV 알고리즘 내부값(Canny/Hough 임계값,
#   블러 커널 크기 등)처럼 "행동을 튜닝한다"기보다 "그 함수 내부 구현"에 가까운
#   상수, 그리고 현재 기본값이 아닌 백엔드(hough/classic_cv 차선인식, Hybrid A*
#   B2 대안)의 세부 알고리즘 상수는 원래 파일에 그대로 남아있습니다. 각 절의
#   "관련 지식" 표를 보고 원래 위치를 찾으세요.
#
# ── 이번 정리에서 같이 치운 죽은 상수 ──
#   track_drive.py에 있던 DEBUG_VIZ, DEBUG_VIZ_LANE(자기 것), STOPLINE_TH,
#   RETURN_THRESHOLD는 정의만 있고 어디서도 참조되지 않는 죽은 상수라 이관하지
#   않고 삭제했습니다(README에 이미 죽은 플래그로 문서화돼 있던 것들 포함).
#=============================================
import math
from enum import Enum

import numpy as np


# #############################################################
# 미션 상태 Enum — START_STATE가 여기 값을 쓰므로 config.py에 둔다
# (track_drive.py가 여기서 import해서 그대로 쓴다. 순환 import 방지 목적).
# #############################################################

class MissionState(Enum):
    # S0_SIGNAL은 출발선/교차로 신호등 판독을 공용으로 처리한다(정지 → 4구 신호등 판독 →
    # 직진/좌회전 확정, track_drive.py _s0_signal() 참고). 출발 직후 1회, 이후 매 바퀴
    # 트랙 중앙 분기점에서 재진입한다 — "진짜 첫 출발인지"는 self._departed 플래그로 별도 추적.
    # 값 2, 3은 과거 상태가 쓰던 값이라 재사용하지 않는다.
    S0_SIGNAL       = 0  # 4구 신호등 판단 (출발선/교차로 공용, 정지→직진·좌회전 확정)
    S1_LANE_FOLLOW  = 1  # 차선인식 주행 (라바콘·고정장애물·추월 Behavior를 이 상태 안에서 처리)
    S4_FINISH       = 4  # 종료

class BehaviorState(Enum):
    B0_NORMAL   = 0  # Mission(차선주행) 출력 그대로
    B1_LAVACON  = 1  # 라바콘 구간 주행 (Phase.LAVACON일 때, 좌우 라이다 클러스터 동시검출 트리거로 활성)
    B2_OBSTACLE = 2  # 고정장애물 회피 (Phase.OBSTACLE_ZONE일 때 obstacle_type=='fixed'로 감지 시 활성)
    B3_VEHICLE  = 3  # 방해차량 추월   (Phase.OBSTACLE_ZONE일 때 obstacle_type=='vehicle'로 감지 시 활성)

# S1(차선주행) 내부 진행 순서 — 순서 고정(라바콘→장애물구간→완료), 순차 전용(우선순위 판단 불필요).
# 정적/동적 장애물 구분은 Phase가 아니라 매 프레임 obstacle_type(라이다 실측 폭 기반,
# perc_obstacle() 참고)으로 판단한다(track_drive.py run_behavior_fsm() 참고).
# OBSTACLE_ZONE→DONE 전환은 B2/B3 둘 다 최소 한 번씩 완료돼야 넘어간다(_mark_behavior_passed()).
class Phase(Enum):
    LAVACON        = 0
    OBSTACLE_ZONE  = 1  # 고정장애물 회피 + 방해차량 추월 통합 구간 (예전 FIXED_OBSTACLE/VEHICLE)
    DONE           = 2  # 모든 Behavior 미션 완료 — 이후 계속 B0로 일반 차선주행


# #############################################################
# 1. 차선인식 백엔드 선택
# #############################################################
#   track_drive.py의 perc_lane()은 self.lane_detector.detect(frame) ->
#   (valid, offset, lookahead, lane_center, path, debug_img) 인터페이스에만
#   의존하는 pluggable 구조라, 아래 셋 중 하나로 자유롭게 바꿔 끼울 수 있다.
#     'dl'         : perception/dl_lane.py의 DLLaneDetector (TwinLiteNet ONNX
#                    세그멘테이션, 별도 스레드 추론). **현재 기본값**.
#     'hough'      : perception/hough_lane.py의 HoughLaneDetector. 대안,
#                    실차 라바콘 테스트까지 검증됨. 'dl' 초기화 실패 시 자동 폴백.
#     'classic_cv' : perception/lane_util.py+perc_floor.py 조립. 보존용,
#                    현재 라이브 미검증.
LANE_DETECTOR_BACKEND = 'dl'  # 'hough' | 'classic_cv' | 'dl'


# #############################################################
# 2. 차량 속도 / 조향 기본값
# #############################################################
SPEED_NORMAL  = 15.0   # 차선주행(S1) 기본(직진) 속도. 이 값을 바꾸면 PP_LOOKAHEAD_MAX_PX
                        #   (§0.5.6/§0.5.10 공식: BASE+GAIN*SPEED_NORMAL)도 같이 재계산할 것.
                        #   0.0으로 두지 말 것 — _lane_drive()에서 나눗셈 분모로도 쓰여 ZeroDivisionError.
                        #   ★주의★ README §6.5의 METERS_PER_SPEED_UNIT 회귀는 speed=5/10 두 점만
                        #   실측한 것이라 15는 측정 범위 밖 외삽 — 실차 재검증 필요.
SPEED_STOP    = 0.0
# 코너 감속(_lane_drive())의 목표속도 하한. "6. 단위환산" 절 실측 근거상 speed≈1.4 미만은
#   모터 데드존으로 추정되며, 이 하한이 데드존 밑으로 내려가면 코너 중 속도가 죽어
#   lookahead도 안 바뀌는 계속-정지 증상이 재현된다 — 데드존보다 확실히 위로 유지할 것.
#   SPEED_CORNER_MIN < SPEED_NORMAL 관계가 깨지면(같거나 커지면) 코너 감속 자체가
#   no-op이 되므로 값을 조정할 때 반드시 함께 확인할 것.
SPEED_CORNER_MIN = 8.0
# DL 추론 워커(별도 스레드, dl_lane.py)가 LANE_STALE_SEC 이상 새 결과를 못 내놓았을 때
#   (카메라/추론 죽음 등, perc_lane()의 lane_stale 판정) 강제하는 속도 상한. 일부러
#   SPEED_CORNER_MIN보다 낮추지 않았다 — 급정지가 아니라 코너 감속과 비슷한 수준으로만
#   눈에 띄게 낮춘다는 설계. 실차 미검증.
SPEED_LANE_STALE = 5.0
ANGLE_MAX     = 80.0  # 조향각 클램프(도)
ANGLE_RATE_MAX = 12.0  # 조향 변화율 제한(도/주기, 20Hz 기준 12도/주기=240도/초) — drive()에서 모든 명령에 일괄 적용
# 가속 램프를 완만하게 잡아 정지→출발/가속 구간의 모터 전류 피크를 낮춘다 — 배터리 전압
#   강하로 인한 ESC/VESC 저전압·과전류 보호(LVC) 트립 완화 목적. 근본 해결은 배터리
#   점검/교체이며, 이 값은 소프트웨어 완화책. 재발하면 배터리 자체를 볼 것.
SPEED_ACCEL_STEP = 0.4  # 가속 속도제한(주기당 최대 증가량)
# 정지 출발 시 첫 틱에 한해 즉시 이 속도로 점프시켜(_lane_drive() 가속 램프 진입부)
#   정지마찰 구간 체류 시간을 줄이려는 시도였으나, 실차 재검증 결과 "틱틱거림" 개선
#   효과가 없었고 원인은 하드웨어(배터리/ESC/모터) 쪽으로 파악됨(track_drive/
#   실제속도측정.md §0.1 참고) — 이후 로직은 그대로 SPEED_ACCEL_STEP 램프를 이어간다.
SPEED_KICK_START = 3.0
CORNER_HOLD_DECAY_LO = 0.92  # 저속 시 코너 hold 감쇠 (빠른 회복)
CORNER_HOLD_DECAY_HI = 0.97  # 고속 시 코너 hold 감쇠 (느린 회복, 연속코너 대응)
# 코너 감속 판단용 조향각 signed EMA 계수(_lane_drive()의 self._corner_signal) —
#   pure_pursuit 특유의 좌우 진동("와리가리")이 매 스윙마다 급코너로 오인돼 속도가 팍팍
#   깎이는 문제 대응. 작을수록(더 스무딩) 진동 상쇄 효과는 커지지만 실제 코너 진입 반응은
#   느려진다. 너무 작으면 실제 코너 감속이 늦어져 위험할 수 있다.
CORNER_SIGN_EMA_ALPHA = 0.15
LANE_LOOKAHEAD_REF = 220.0   # 예측감속 최대가 되는 lookahead 편차(px) — _lane_drive() 속도계획용
# 조향각 기반 3제곱 감속식의 최대 감속 게인. target_speed = SPEED_NORMAL*(1 -
#   SPEED_CORNER_STEER_GAIN*turn_for_speed**3) 공식에서 turn_for_speed=1(최대 코너 신호)일
#   때 SPEED_NORMAL 대비 이 비율만큼 깎인다. 실차 재검증 필요 — 과하면 더 낮출 것.
SPEED_CORNER_STEER_GAIN = 0.80

# ── 코너 진입 시 회전반경 기반 감속 (ROS2 Nav2 Regulated Pure Pursuit 방식) ──
#   회전반경(1/curvature)이 CORNER_MIN_RADIUS_PX보다 작아지면 그 비율만큼 목표속도를
#   깎는다. PIXELS_PER_METER 미실측이라 반경은 픽셀 단위 — 실차 미검증 추정치.
CORNER_MIN_RADIUS_PX = 250.0
CORNER_MIN_SPEED_SCALE = 0.35  # 반경이 0에 가까워져도 속도가 0으로 죽지 않게 하는 하한 배율

# "직선인데 커브로 오검출돼 속도가 안 오른다" 대응 — turn_now(조향각 signed EMA)/
#   turn_preview(lane_lookahead)는 전부 비전+조향출력에서만 나오는 신호라, 세그멘테이션
#   잡음이나 조향 잔떨림만으로도 코너로 오인될 수 있다. 2023 KMU 대회 AuTURBO rookie 팀
#   저장소(ModeController.py)의 모드전환 로직 아이디어를 참고 — diff_degree(IMU yaw
#   변화량)로 실제 회전량을 교차검증한다. track_drive.py._imu_corner_confirm_scale()에서
#   "비전은 코너라는데 IMU 실측 회전율이 거의 0"이면 코너감속(turn_for_speed)을 절반
#   이하로 깎는다. IMU/VESC가 죽어있거나 dl+BEV 조합이 아니면 비전 신호만으로 판단
#   (무감쇠, 1.0)한다. 실차 미검증 첫 추정치.
CORNER_IMU_CONFIRM_KAPPA_PX = 1.0 / CORNER_MIN_RADIUS_PX  # = 0.004 — 이 이상 IMU curvature면 코너감속 100% 신뢰
CORNER_IMU_MIN_SCALE = 0.5  # IMU가 "회전 거의 없음"을 보고해도 비전신호 기반 감속을 최소 이만큼은 남겨두는 하한

# ── 좌회전 공통 (S0_SIGNAL 'left' 커밋 → 체크무늬 게이트 램프 진입) ──
#   좌회전은 진입(_do_checker_ramp_turn(), CHECKER_TURN_RAMP_* 절)만 있고, 끝나면 곧장
#   S1_LANE_FOLLOW로 돌아간다.
TURN_SPEED     = 12.0    # [진입] 체크무늬 게이트 램프 좌회전 속도(_do_checker_ramp_turn()이 ctrl_speed로 그대로 씀)

# ── 정지선 접근 감속 ──
#   정지선을 감지하면 곧장 S0_SIGNAL로 전환하고, 그 상태의 매 프레임 판정("신호
#   미확정=사실상 빨간불이면 속도 0", _s0_signal() 기본 동작)이 정지를 대신한다 —
#   "정지"라는 별도 이벤트/타이머는 없다. 직진 확정 시에도 커밋 구간 없이 곧장 S1로
#   돌아가 Behavior(라바콘부터)를 재활성화한다(track_drive.py _s1_lane_follow()/
#   _s0_signal() 참고). 좌회전 커밋 구간(_s2_commit_dist, checker_pillar_trigger로 종료)은
#   별도로 남아있다.


# #############################################################
# 3. 차선인식(dl_lane.py) 세부 튜닝 — BEV 적용여부 포함
# #############################################################
#   perception/dl_lane.py(현재 기본 백엔드)의 세그멘테이션 후처리 튜닝값 모음.
#   모델 자체의 입출력 스펙(DL_INPUT_W/H, DL_OUTPUT_NAMES 등)은 "이미 검증된
#   고정값"이라 dl_lane.py에 그대로 남아있다 — 여긴 실차 조건에 따라 실제로
#   바꿀 만한 값들만 모았다.

# ── BEV(원근변환) 적용 여부 ──
#   True면 da/ll 확률맵을 원근변환(BEV)한 뒤 중심선을 뽑는다 — 실차 검증 중.
#   DL_BEV_SRC_PX(원본 프레임 픽셀 4점)와 DL_PIXELS_PER_METER(설계값, 실측
#   아님)가 이 변환의 캘리브레이션 입력이다. 재측정 방법/실측값 근거는
#   track_drive/README.md "6.3 DL 백엔드 BEV 캘리브레이션" 절 참고. 카메라
#   마운트가 바뀌면(재장착, 진동 등) 아래 4점은 무효가 되므로 재측정 필요.
DL_USE_BEV = True

# 실측 픽셀좌표(원본 640×480 프레임 기준, ROI 자르기 전 절대좌표) — TL/TR/BR/BL 순.
# dl_lane.py가 여기서 DL_ROI_Y0만큼 뺀 ROI-상대좌표로 변환해서 쓴다.
DL_BEV_SRC_PX_RAW = np.float32([
    [246, 257],  # TL(좌상/먼왼쪽)
    [455, 257],  # TR(우상/먼오른쪽)
    [635, 333],  # BR(우하/가까운오른쪽)
    [60,  333],  # BL(좌하/가까운왼쪽)
])
DL_PIXELS_PER_METER = 200.0   # 설계값(실측 아님) — 목적 캔버스를 1m=200px 스케일로 만든다.

# 원거리 크롭 — da/ll 처리가 근거리 기준점(BL/BR)으로부터 몇 m까지만 보게 할지. 캔버스는
#   "ROI 전체가 여백 없이 들어가도록" 자동 확장되는데(perception/dl_lane.py의
#   DL_BEV_CANVAS_H 계산 참고), 그 결과 실측 캘리브레이션 지점(TL/TR, 1.0m)보다도 더 먼
#   외삽 영역까지 처리에 포함될 수 있다. 이 값을 낮추면 그 외삽 영역과 원거리 blur를
#   처리 대상에서 제외할 수 있다.
#   ★주의★ 이건 DL_BEV_SRC_PX_RAW/DL_PIXELS_PER_METER 같은 캘리브레이션 값이 아니다 — 그
#   값들은 그대로 두고 "이미 정확하게 아는 좌표계에서 먼 부분을 그냥 안 본다"는 크롭일
#   뿐이라 스케일 왜곡이 없다. 카메라가 실제로 보는 물리적 거리를 바꾸려면 DL_BEV_SRC_PX_RAW
#   4점의 실측 재측정이 필요하다(README §6.3 참고). 실차 미검증 — DEBUG_VIZ_DL_LANE에서
#   크롭 경계가 원하는 위치에 오는지 확인할 것. B1(라바콘) 이외 전 구간(B2/B3 포함)이
#   항상 이 값을 쓴다.
DL_BEV_FAR_LIMIT_M_NORMAL  = 1.0

# ── 세그멘테이션 결과에서 좌/우 차선 중심을 뽑을 관심영역 (원본 480행 기준 절대 픽셀, 실차 실측값) ──
DL_ROI_Y0 = 250
DL_ROI_Y1 = 390

DL_FG_THRESHOLD = 0.5   # da 확률맵 이진화 임계값(요구사항에 명시된 값) — da에만 쓴다

# ll(차선) 전용 이진화 임계값. BEV 워프는 카메라에서 먼 지점일수록 원근압축을 되돌리려고
#   더 크게 확대하는데(호모그래피 성질상 불가피), da/ll은 이진화 "전"(float 확률맵)
#   상태로 워프하기 때문에(위 DL_USE_BEV 주석 참고, 계단 현상 방지 목적) 모델 출력의
#   경계 blur(확률이 0.5 근방인 애매한 픽셀들)도 그 확대율만큼 같이 늘어난다. 원거리에서
#   ll이 실제 선 두께보다 두껍게 잡히면 _clip_da_by_ll()이 da를 필요 이상으로 깎아내
#   DL_DA_MIN_COMPONENT_AREA 밑으로 떨어뜨려 그 프레임이 무효 처리될 수 있다. 지금은
#   da와 같은 0.5 — 실차 재검증 전, DEBUG_VIZ_DL_LANE에서 ll_cov가 정상 범위로 올라오는지,
#   원거리 ll이 과하게 두꺼워지진 않는지 확인할 것.
DL_LL_FG_THRESHOLD = 0.5

# ── SlideWindow moments 재사용 DL 전용 튜닝값 (알고리즘은 lane_util.py의
#   MOMENT_*/LANE_SLICE_*/STABLE_*와 동일, DL은 원본 카메라 프레임 px 스케일이라
#   값만 따로 둔다) — 전부 실차 미검증 튜닝값 ──
DL_N_SLICES = 8               # da 중심선을 세로로 나눌 밴드 수
DL_MIN_PIXELS = 40            # 밴드 내 da 픽셀수가 이 미만이면 그 밴드는 "차선 없음" 처리
# B2/B3 회피주행 중(obstacle_cut_confirmed=True) 전용 픽셀수 문턱 — 회피 컷으로 da가
#   얇아진 상태에서 DL_MIN_PIXELS(40)를 그대로 쓰면 밴드가 통째로 None 처리될 수 있어
#   더 낮춘 값. 실차 미검증.
DL_MIN_PIXELS_OBSTACLE = 20
DL_NEAR_SLICES = 2            # 근거리(조향용) 편차 계산에 쓸 아래쪽 밴드 수
DL_FAR_SLICES = 2             # 원거리(코너 예측용) 편차 계산에 쓸 위쪽 밴드 수

# 근접 밴드(위 DL_NEAR_SLICES개) 전용 "임시 hold" 상한 — 원거리 밴드가 장애물 회피로
#   정당하게 휘면 leave-one-out 추세선(_reject_outliers())이 아직 정상인 근접 밴드를
#   이상치로 오판하던 문제의 2차 안전판(1차는 근접 밴드를 그 검사에서 아예 빼는 것,
#   dl_lane.py detect()/lane_util.py _reject_outliers() protect_indices 참고). 그 방어를
#   뚫고도 근접 밴드가 비면, 마지막으로 실제 찾았던 위치(_da_prev_band_x[i])를 이 프레임
#   수까지만 대신 채운다 — 찰나의 깜빡임은 넘기되, 오래 안 보이면(장애물이 실제로 근접까지
#   온 경우일 수 있음, README §2.30 "앞코가 뒷꽁지를 긁는다" 참고) 더는 안 믿고 포기해서
#   위험 신호를 숨기지 않는다. 넘기면 near_band_stale=True로 노출돼 track_drive.py
#   _lane_drive()가 SPEED_LANE_STALE급으로 감속시킨다. 20Hz 기준 3프레임≈0.15초. 실차 미검증.
DL_NEAR_HOLD_MAX_FRAMES = 3
# 아래 4개 값은 BEV 캔버스(585×298px, DL_PIXELS_PER_METER=200px/m) 스케일 기준으로 계산한
#   값 — 실차 미검증, DEBUG_VIZ_DL_LANE 오버레이로 교차로 진입 구간에서 확인 후 조정할 것.
DL_SLICE_OUTLIER_MAX = 40     # 반차로폭(0.4m=80px)의 1/2. 추세선에서 이 이상(px) 벗어난 밴드는 이상치로 제외
DL_SLICE_FIT_MIN = 3          # 유효 밴드가 이 미만이면 추세 판단 생략
# ref_x(_ll_yellow_white_centers()/_ll_slice_centers()의 탐색창 seed, dl_lane.py detect()
#   참고)가 이 프레임 수만큼 지연된 값이라, 값이 낮을수록 새 후보가 확정값으로 승격되는
#   지연이 줄어 급조향 복귀 구간 반응이 빨라진다(DL_STABLE_JUMP_MAX=20px 체크는 그대로라
#   노이즈 한 프레임이 바로 통과되진 않음). 실차 미검증 — 너무 낮추면(예: 1) 스파이크가
#   그대로 확정될 위험이 있다.
DL_STABLE_FRAME_MIN = 2       # "새 추론이 끝난 시점" 기준 연속 안정 프레임 수(디바운스)
DL_STABLE_JUMP_MAX = 20       # 반차로폭의 1/4. 이 이상(px) 차이나면 새 후보로 취급

# da 파편화 대응 — ConnectedComponents 최대 덩어리 면적이 이 미만이면 "da 안 보임" 처리.
DL_DA_MIN_COMPONENT_AREA = 1560
# da 과대검출 대응 — 최대 덩어리 면적이 이 절대 픽셀수를 넘으면 정상 자기차선 폭이
#   아니라고 보고 outlier로 버린다(_largest_da_component()가 그 다음으로 큰 덩어리를
#   대신 시도 — "차선책", 위 주석 참고). 교차로에서 da가 옆 갈림길까지 하나로 이어붙는
#   경우뿐 아니라, 차선(백선)이 아예 없는 맨바닥을 통째로 주행가능영역으로 오검출하는
#   경우도 실측으로 확인됨 — 두 실패모드 모두 정상 대비 면적이 비정상적으로 크다는
#   공통점이 있다. 원거리 크롭 DL_BEV_FAR_LIMIT_M=0.7 적용 시 직선 구간 실측 평균
#   13,219px(최대 13,361px) 기준으로 여유를 둔 값 — 캔버스 크기가 바뀌면(원거리 크롭 값
#   재조정 등) 이 값도 같이 재측정해야 한다.
DL_DA_MAX_AREA_PX = 16000

# _largest_da_component()의 시드(seed) 기반 최우선 후보 선택에 쓰는 탐색 범위 — ROI
#   최하단(차량과 가장 가까운 행)에서 이 행 수(세로) × 이 반경(가로, ROI 중앙 기준)만큼의
#   작은 영역을 보고, 그 영역과 물리적으로 맞닿은 덩어리가 있으면(면적이 유효 범위 안이면)
#   과거 판단(직전 프레임 연속성/면적순위)보다 우선 채택한다 — "차량이 실제로 서 있는
#   자리"라는 매 프레임 독립적인 물리 신호라, 직전 프레임의 오판이 계속 이어지는 걸 스스로
#   교정한다. DL_DA_SEED_HALF_WIDTH_PX는 실측 차로폭의 절반 근방, DL_DA_SEED_ROWS_PX는
#   "차량 바로 앞"만 보도록 작게 잡았다. 둘 다 실차 미검증 — 좁으면 시드 영역이 빈
#   프레임(=폴백)이 잦아지고, 너무 넓으면 옆 차선까지 시드에 걸려 잘못 채택될 수 있다.
DL_DA_SEED_ROWS_PX = 10
DL_DA_SEED_HALF_WIDTH_PX = 70.0

# DL_CENTER_MODE='da' 밴드 중심 계산 — 탐색창(prior) + 밴드 간 속도예측 + 프레임 간
#   앵커링. da가 옆 차선/여백까지 과검출(S자 커브에서 특히)되면 무보정 무게중심 계산은
#   그 넓어진 영역 전체에 쏠린다 — Mobileye/openpilot/drivable-area 연구에서 공통된
#   "탐색을 예측 위치 근방으로 제한" 아이디어를 _ll_slice_centers()(DL_LL_ALGO='lr')가
#   쓰던 패턴 그대로 da에도 적용한다(DLSlideWindow._da_slice_centers_windowed() 참고).
#   da는 좌/우 두 갈래가 아니라 중심선 "한 갈래"라 ll보다 단순한 단일-트랙 버전.
#   DL_DA_SEARCH_HALF_WIDTH_PX가 DL_LL_SEARCH_HALF_WIDTH_PX(60, 선 하나 전용)보다 넓은
#   이유 — da는 폭 있는 영역이라 반차로폭(LANE_WIDTH_M=0.4m=80px) 이상은 창 안에 들어와야
#   정상 시야까지 잘라내지 않는다. 전부 실차 미검증 초기값 — 창이 너무 좁아 정상 코너까지
#   놓치지 않는지(검출 밴드 수가 줄지 않는지) DEBUG_VIZ_DL_LANE으로 확인할 것.
DL_DA_SEARCH_HALF_WIDTH_PX = 100.0
DL_DA_SEARCH_WIDEN_STEP_PX = 20.0
DL_DA_SEARCH_WIDEN_MAX_PX = 200.0
DL_DA_VELOCITY_EMA_ALPHA = 0.3      # 밴드 간 이동 속도(px/밴드) EMA 계수 — DL_LL_VELOCITY_EMA_ALPHA와 동일 관례
DL_DA_VELOCITY_MAX_PX = 40.0        # 예측 이동량 클램프
DL_DA_BAND_ANCHOR_ALPHA = 0.35      # 밴드별 탐색창 중심 계산 시 "직전 프레임 그 밴드 위치"에 주는 가중치

# BEV 근접 밴드 모서리 사각지대 비대칭 편향 보정 — 근접 밴드(near_slices)에서 그 행의
#   visible da 폭이 기대 차선폭*비율보다 좁으면 cx를 vehicle_center_x 쪽으로 블렌드
#   (_da_slice_centers_windowed() 참고). 실차 미검증 — 비율/블렌드폭 둘 다 실측 필요.
DL_DA_NEAR_WIDTH_MIN_RATIO = 0.7
DL_DA_NEAR_WIDTH_BLEND_MAX = 0.5

# da가 옆 차선과 이어붙었을 때 ll 라인 바깥(옆 차선 쪽) 픽셀을 잘라내는 여유폭(px) —
#   실측 라인 두께 2.5cm(=5px @200px/m) + 세그멘테이션 경계 흔들림(1~2px) 여유.
DL_LL_CLIP_MARGIN_PX = 8

# _clip_da_by_ll() 전용 ll 잔상(decay) — da가 옆 차선과 완전히 한 덩어리로 붙어버리는
#   실패모드에서는(ll_cov 극히 낮음, ll_bands=0/8) da 자체가 두 차선을 구분하는 내부 경계
#   없이 뭉텅하게 나와 침식(erosion)으로 끊을 구조가 없다(실차 재현 확인). 대신 최근 몇
#   프레임 동안 확실했던 ll 픽셀을 감쇠 가중치로 유지해 이번 프레임 ll이 비어도 클리핑
#   근거로 계속 쓴다. DL_LL_DECAY_ALPHA는 매 프레임 곱해지는 감쇠율(1에 가까울수록 오래
#   남음), DL_LL_DECAY_MIN_VALUE는 "아직 보이는 것"으로 칠 최소 잔상값(0~255 스케일,
#   ll_mask와 동일) — 대략 3~4프레임(추론 프레임 기준) 뒤 자연 소멸한다. centerline
#   추출(_ll_slice_centers)에는 이 잔상을 안 쓴다 — waypoint를 과거 위치로 미는 건 더
#   위험하고, 클리핑은 "울타리" 역할이라 약간 stale해도 안전하다는 판단. 실차 미검증
#   초기값 — 짧은 끊김엔 도움 되는지, 너무 오래 남아 실제 경계 이동을 못 따라가진
#   않는지 DEBUG_VIZ_DL_LANE으로 확인할 것.
DL_LL_DECAY_ALPHA = 0.8
DL_LL_DECAY_MIN_VALUE = 128.0

# ── 밴드별 중심 계산 모드 스위치 — 세 모드가 서로 완전히 다른 알고리즘 ──
#   'da'    : 밴드별 중심을 da(주행가능영역) 무게중심(_slice_centers(), cv2.moments)으로
#             계산한다(main 기본값). 덩어리 선택은 DLSlideWindow._largest_da_component()
#             — ①시드(차량 위치와 맞닿은 덩어리) → ②연속성(직전 프레임과 가장 가까운
#             덩어리) → ③면적순위(최후 폴백) 순, 면적 상한(max) 체크는 없다(실차 검증
#             결과 면적만으로 da를 거르는 방식 자체가 불신뢰 — da가 옆 차선과 붙는 문제는
#             이제 전적으로 _clip_da_by_ll()이 담당). 하한(DL_DA_MIN_COMPONENT_AREA)은
#             "사실상 안 보임" 노이즈 필터로 유지.
#   'll_da' : "corridor" 알고리즘 — ll(차선)로 도로 폭 자체를 규정하고, da는 그 안에서
#             장애물 회피용 열린 공간을 찾는 데만 쓴다. 밴드마다 ll을 왼쪽부터 정렬해
#             (DLSlideWindow._ll_line_centers(), 흰/노랑 구분 없는 원본 ll_mask 사용 —
#             노란 중앙선도 "2번째 선"으로 그대로 센다) 1번째~3번째 선을 도로 경계(전체
#             트랙, 양쪽 차로 폭)로 삼는다. 그 x범위 안에서만 da를 보고 실제 열린(장애물
#             없는) 구간을 찾아(DLSlideWindow._pick_open_run(), 직전 프레임 위치에 가장
#             가까운 구간을 우선하는 히스테리시스 있음) 그 중심을 밴드 중심으로 쓴다
#             (DLSlideWindow._corridor_slice_centers()). "자기 차선 하나"를 전제로 한
#             _largest_da_component()/_clip_da_by_ll()은 건너뛰고 클리핑 전 원본
#             da(da_mask_all_roi)를 그대로 쓴다 — 장애물이 도로를 좌/우로 쪼갤 때 그 두
#             함수는 지나갈 수 있는 작은 쪽을 통째로 버리거나 잘라내 버려서 corridor
#             취지(양쪽 차로를 동시에 보고 그 안에서 고른다)와 반대다. 밴드에서 검출된
#             선이 3개 미만이거나 corridor 폭이 DL_CORRIDOR_WIDTH_MIN/MAX_PX 밖이면 그
#             밴드는 da 폴백 없이 그냥 드롭한다 — corridor 경계 자체가 ll에서 나오므로
#             ll이 불충분한 순간엔 "도로 폭이 얼마인지" 판단할 근거가 없기 때문.
#   'll'    : ll을 흰선/노란선으로 분리(DLSlideWindow._split_ll_by_yellow(), 커넥티드
#             컴포넌트 단위로 HSV 노란색 겹침 비율 투표 — 픽셀 단위로 빼는 것보다 dash
#             가장자리가 깔끔함)한 뒤, **노란 중앙선 + (내 차선 판정에 따른) 한쪽 흰색
#             경계선**을 추적한다(_ll_yellow_white_centers()). 실제 도로가 편도 1차로
#             기준 흰-노-흰 구조라 "좌/우 흰선 두 개를 독립 추적"하는 방식은 노란선이
#             있는 쪽에서 흰선 탐색이 구조적으로 실패한다(노란선은 흰선 마스크에서
#             제외됨) — 그래서 아래 순서로 재설계했다:
#               ① 차선 판정: 근거리 밴드의 노란선이 seed(차량 위치, x 중앙) 기준 왼쪽에
#                 있으면 "나는 우측차선 주행중"(흰 경계선은 오른쪽에서 탐색), 오른쪽에
#                 있으면 "좌측차선 주행중"(왼쪽에서 탐색) — self.lane_side에 기록.
#               ② 밴드마다 노란선/흰선을 각각 좁은 창(DL_LL_SEARCH_HALF_WIDTH_PX)으로
#                 독립 탐색. 둘 다 찾으면 중점 채택 + 간격(self._white_yellow_gap_px,
#                 DL_LL_YELLOW_GAP_EMA_ALPHA로 EMA) 갱신.
#               ③ 노란선만 찾으면(흰선 실패) → 간격만큼 흰선 위치를 추정해서 중점 계산.
#               ④ 노란선을 못 찾으면(이번 밴드) → 넓은 창(DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX)
#                 으로 흰선을 몇 개나 찾았는지로 3분기한다(좁은 창 하나로 흰선 하나만
#                 보고 gap을 역적용하던 옛 방식은 gap EMA가 노이즈로 부풀면 실제 흰선
#                 위치를 무시하고 차선 밖으로 waypoint를 밀어내는 문제가 있었다, README
#                 §2.18):
#                   - 2개(양쪽 다 보임): 두 실측 위치의 중점을 그대로 채택(간격
#                     추정치에 안 기댐 — 가장 신뢰도 높음).
#                   - 1개(한쪽만 보임): 그 흰선이 기준점(cur_yellow) 대비 좌/우
#                     어느 쪽인지 매 밴드 새로 판정한 뒤(직전 프레임 값에 안
#                     기댐), 그 방향으로 간격(클램프됨, 아래 참고)만큼 안쪽으로
#                     당겨 중앙선을 재구성.
#                   - 0개(둘 다 못 찾음): 직전까지 추적하던 위치를 그대로
#                     유지("잔상").
#             ③④는 전부 "저신뢰 추정"이라 self.ll_degraded=True로 표시되고, 이번
#             프레임에 하나라도 있으면 track_drive.py _lane_drive()가 속도를
#             SPEED_LL_DEGRADED로 강제 제한한다(_debug_viz_steer()에도 표시). da 파편화
#             대응/옆 차선 클리핑/ll sanity check는 이 모드에서도 그대로 적용된다
#             (da_mask 자체는 여전히 클리핑/디버그 패널용으로 계산됨) — 다만 그 da 결과가
#             중심점 계산에는 전혀 섞이지 않는다.
#   현재 기본값은 'da' — ll_da/ll 둘 다 노란선 인식 불안정으로 실차 검증이 막혀,
#   da 자체 검출 품질만 먼저 확인하는 방향으로 전환했다. 아래 DL_DA_SKIP_LL_CLIP과 짝.
DL_CENTER_MODE = 'da'  # 'da' | 'll_da' | 'll'

# DL_CENTER_MODE='da' 전용 — ll(차선) 기반으로 옆 차선을 잘라내는 _clip_da_by_ll() 단계를
#   통째로 건너뛴다. True면 largest-component만 남긴 da_mask를 클리핑 없이 그대로
#   센터라인 계산에 쓴다 — da_ll_clip_skipped/da_clip_band_virtual 등 기존 "클리핑
#   건너뜀" 디버그 표시를 그대로 재사용해 visualize()에도 반영된다(detect() 참고).
#   DL_CENTER_MODE='ll'에는 영향 없음 — 그쪽은 이 플래그와 무관하게 항상 클리핑 적용.
#   ★주의★ 클리핑을 끄면 옆 차선 침범 가드가 없어지므로, da 세그멘테이션 품질 자체만
#   보는 실험용이지 이 상태로 실주행에 쓰라는 뜻이 아니다. 실차 확인 결과 클리핑을 껐을
#   때(False)는 _clip_da_by_ll()이 의도한 대로 동작하지 않아 다시 True로 고정.
DL_DA_SKIP_LL_CLIP = True

# ── da 안전마진(차량 폭) 침식 — DL_CENTER_MODE='da' 전용 ──
#   장애물 회피 중 "앞코가 장애물 뒷꽁지를 긁는다"는 실차 보고 대응(README §2.30) —
#   _da_slice_centers_windowed()가 차량을 폭 0인 점으로 취급해 da 경계에 바짝 붙여
#   경로를 뽑던 문제. da 마스크 자체를 차폭(VEHICLE_WIDTH_M)+여유만큼 침식(erosion)해서
#   중심선이 da 경계에서 최소한 이만큼은 떨어지도록 강제한다(ROS2 Nav2의 costmap
#   inflation과 동일한 개념).
#   ★주의★ da 세그멘테이션 자체는 그대로 두고 후처리로 마진을 만드는 것뿐이라,
#   DL_PIXELS_PER_METER(설계값, 실측 아님 — 위 주석 참고)의 원거리 외삽/피치 변화 취약성을
#   그대로 물려받는다 — "몇 cm 여유"를 엄밀히 보장하진 못하고 근사치다. 커브가 심한
#   구간에서 침식이 과해 da가 DL_DA_MIN_COMPONENT_AREA 밑으로 꺼지면(그 프레임 무효
#   처리) DL_DA_VEHICLE_MARGIN_M을 낮출 것. 실차 미검증 초기값.
DL_DA_APPLY_VEHICLE_MARGIN = True
DL_DA_VEHICLE_MARGIN_M = 0.05   # ASTAR_VEHICLE_MARGIN_M(라이다/Hybrid A* 쪽)과 동일 관례 — 전/후(세로) 마진 기본값

# 좌/우(가로) 마진 전용 상수 — DL_DA_VEHICLE_MARGIN_M(등방)과 분리해 좌우만 따로 조정할
#   수 있게 한다(dl_lane.py _apply_vehicle_margin()이 좌우 경계 전체에 적용 — "장애물
#   옆이라서" 골라 적용되는 게 아니라 트랙 가장자리/커브 여백도 이 값을 따라 넓어진다).
#   실차 테스트로 0.1m을 확인. 이 값을 키우면 근접 장애물 앞에서 lookahead가 멀리(옆차선
#   중앙)찍혀 조향이 완만해지는 2차 증상이 날 수 있어 pure_pursuit.py
#   _target_point_max_deviation()(근접 장애물 한정)으로 같이 보완한다. 좁은 커브에서
#   da가 DL_DA_MIN_COMPONENT_AREA 밑으로 꺼지는지 계속 확인할 것.
DL_DA_SIDE_MARGIN_M = 0.1

# B2/B3 회피주행 중(obstacle_cut_confirmed=True) 전용 좌우 침식 커트라인 — 실차 미검증.
#   _clip_da_by_obstacle()의 차선 절반 컷과 위 DL_DA_SIDE_MARGIN_M 등방 침식이 같은
#   프레임에 겹쳐 쌓이면 da가 필요 이상으로 얇아져 밴드가 통째로 None(경로 단절)되기
#   쉽다는 문제 대응. 단위가 m이 아니라 px(반경 아님, 폭 기준) —
#   dl_lane.py _dl_da_margin_kernel()의 rx = 이 값 // 2.
DL_DA_SIDE_CUTLINE_PX_OBSTACLE = 20

# 위 DL_DA_VEHICLE_MARGIN_M(+VEHICLE_WIDTH_M/2)의 등방(isotropic) 원형 침식 커널과 별개로,
#   속도가 높을수록 접근 상대속도가 커져 "앞코가 장애물 뒷꽁지를 긁는" 여유가 더
#   필요하므로 da 마스크의 세로축(진행방향)만 v_mps에 비례해 추가로 침식폭을 키운다
#   (가로=좌우 폭은 DL_DA_SIDE_MARGIN_M이 독립적으로 결정).
#   extra_m = min(DL_DA_REAR_MARGIN_REACT_SEC * v_mps, DL_DA_REAR_MARGIN_MAX_M) —
#   REACT_SEC은 "제동/재계획까지 걸리는 시간"의 근사치로 삼은 설계값(실측 아님). MAX_M은
#   da가 통째로 침식돼 사라지는 걸(§2.30, _apply_vehicle_margin() 폴백 참고) 막는 상한.
#   실차 미검증 초기값 — 코너에서 da가 자주 무효 처리되면 REACT_SEC/MAX_M을 낮출 것.
DL_DA_REAR_MARGIN_REACT_SEC = 0.2   # 속도(v_mps)에 비례해 "뒤" 방향 마진을 추가로 늘리는 반응시간(s)
DL_DA_REAR_MARGIN_MAX_M = 0.5       # 위 추가 마진의 상한(m) — 대략 VEHICLE_LENGTH_M(0.64) 이내로 캡

# ── 회피 "복귀 유예"(avoid-hold) — DL_CENTER_MODE='da' 전용 (README §2.32) ──
#   위 안전마진 침식은 da에 뚫린 장애물 구멍 주변을 자연스럽게 우회하게 만들 뿐, "언제
#   원래 차선 중앙으로 돌아가도 되는지"는 전혀 모른다. 카메라가 차량 앞코에 달려있어서
#   장애물을 실제로 지나치는 순간 그 장애물이 화면(및 da 구멍)에서 사라지고, da가 그
#   프레임에 바로 원래 폭으로 돌아와 중심선도 즉시 원래 차선 중앙으로 복귀한다. 장애물이
#   정지해 있으면 문제없지만, 장애물이 방해차량처럼 계속 주행 중이면 "지나친 그 순간"엔
#   아직 옆이나 뒤에 바짝 붙어있을 수 있어 너무 이른 복귀가 그 차와의 충돌(추월 후
#   방해차량이 우리 뒤를 들이받는 상황)로 이어질 수 있다(실차 미검증).
#   대응: perc_obstacle()의 obstacle_front/obstacle_dist(라이다, 매 틱 갱신됨)를 근거로,
#   장애물이 AVOID_HOLD_TRIGGER_DIST_M 안으로 마지막으로 들어왔던 시점부터 AVOID_HOLD_SEC초
#   동안은 da를 raw centroid 그대로 쓰지 않고 ll(차선)로 "지금 차선 하나"만 남기도록
#   강제로 자른다 — 예전부터 있던 _clip_da_by_ll() 클리핑(DL_DA_SKIP_LL_CLIP=True로
#   평소엔 꺼둔 것)을 이 창에서만 되살리는 방식(track_drive.py _update_avoid_hold()/
#   perception/dl_lane.py detect() 참고). 장애물이 시야에서 사라진 직후에도 몇 초간은
#   옆 차선으로 안 새고 지금 차선 폭 안에서만 주행해, 급하게 원래 차선 중앙으로 꺾여
#   들어가는 걸 늦추는 목적. ★주의★ 둘 다 실차 미검증 초기값이다.
AVOID_HOLD_TRIGGER_DIST_M = 1.5   # 이 거리(m) 안으로 장애물이 들어오면 "회피 중"으로 본다

# avoid-hold 개선 — 가변 유예시간 + 거리기반 조기 해제 + da 연속성 보조트리거 + 방향 힌트 +
#   안전판(avoid_hold_improvement_proposal.md "1차 적용 결정" 적용1~4). 고정 유예시간
#   하나만으로는 "짧으면 방해차량에 위험, 길면 정지장애물엔 낭비"라는 트레이드오프를
#   피할 수 없었다 — 설계 배경/시나리오별 사이드이펙트/대비책은 위 proposal 문서 참고.
#   ★ 아래 값 중 실측이 필요한 건 track_drive/avoid_hold_measurement_todo.md에 측정
#   절차와 함께 정리해뒀다 — 실차 테스트 전에 그 문서부터 볼 것. DEBUG_VIZ_AVOID_HOLD
#   (아래 "5. 디버깅 ON/OFF" 참고) 창이 이 값들과 지금 상태를 실시간으로 같이 보여준다 ★

# 유예시간 계산(track_drive.py _update_avoid_hold()) —
#   hold_sec = clip(BASE + RATE_GAIN*|target_speed_est|, MIN, MAX)
#   target_speed_est = v_mps + obstacle_rate, 트리거가 걸리는 순간 1회만 스냅샷한다 —
#   매 틱 다시 계산하면 obstacle_rate의 프레임간 노이즈가 유예시간 자체의 흔들림으로
#   샌다(문제1 대비책). target_speed_est가 OBSTACLE_STATIC_SPEED_TH_MPS(아래 "7. 기타"
#   절) 이하인 애매한 회색지대는 정지로 보고 RATE_GAIN을 아예 적용하지 않는다
#   (_cross_check_obstacle_motion()이 이미 쓰는 데드존을 그대로 재사용).
AVOID_HOLD_SEC_BASE = 2.0    # 기존 AVOID_HOLD_SEC과 동일값 — 상대속도가 데드존 이하일 때(정지 장애물 등)
AVOID_HOLD_SEC_MIN  = 2.0    # clip 하한. BASE+RATE_GAIN*x는 항상 BASE 이상이라 지금은 사실상 BASE와
                              #   같지만, clip() 형태를 명시적으로 유지해 나중에 BASE를 낮출 때도
                              #   이 하한이 안전판으로 남게 한다.
AVOID_HOLD_SEC_MAX  = 3.5    # clip 상한 — 아무리 빠르게 접근해도 여기서 캡(무한정 길어지는 것 방지).
                              #   ★실측 필요★ — avoid_hold_measurement_todo.md 참고.
AVOID_HOLD_RATE_GAIN = 0.75  # target_speed_est(m/s) 1당 유예를 이만큼(초) 늘림. ★실측 필요★

# 조기 해제(같은 함수) — obstacle_front=False가 RELEASE_CONFIRM_FRAMES 연속 + 마지막으로
#   obstacle_front=True였을 때의 obstacle_dist가 RELEASE_DIST_M 이상이면 hold_sec을 다
#   채우기 전에도 즉시 해제한다. "안 보임=멀어짐"이 아니라 "마지막으로 봤을 때 이미
#   멀었음"으로 조건화해 라이다 사각지대와 실제 이탈을 최소한이나마 구분한다 — 그래도
#   둘을 완전히 못 가르는 건 여전한 한계(proposal 문서 참고).
AVOID_HOLD_RELEASE_DIST_M = 2.0        # TRIGGER_DIST_M(1.5)보다 크게 둬 히스테리시스 확보. ★실측 필요★
AVOID_HOLD_RELEASE_CONFIRM_FRAMES = 4  # SIG_CONFIRM_FRAMES(3)/VEHICLE_TRIGGER_FRAMES(5) 등과
                                        #   동일 관례 — 순간적 미검출(사각지대/flicker)로
                                        #   조기해제가 새는 것 방지.

# da 연속성 보조 트리거(perception/dl_lane.py DLSlideWindow) — 라이다 사각지대 보완.
#   이번 프레임 da_chosen_area_px가 직전 프레임 대비 이 배율 이상 급증하면(=방금까지
#   뚫려있던 구멍이 갑자기 메워짐) "뭔가 방금 시야에서 사라졌을 수 있다"는 신호로 보고,
#   라이다 obstacle_front 트리거와 OR로 결합한다(단독 트리거로는 안 씀 — 세그멘테이션
#   자체가 흔들리는 프레임에서 오발동할 수 있어서).
AVOID_HOLD_DA_AREA_JUMP_RATIO = 1.4    # ★실측 필요★

# 방향 힌트(track_drive.py _update_avoid_hold()가 매 틱 계산 → perc_lane()이 DL 백엔드로
#   전달 → perception/dl_lane.py _clip_da_by_ll()) — TargetPassing.choose_side()가 반환한
#   side(-1/0/+1, lane_offset과 동일한 "우측+" 부호규약)를, _clip_da_by_ll()의 "ll도
#   잔상도 없는" 최후수단 가상경계 폴백에서만 기준점을 이만큼(px) 안전한 쪽으로 미리
#   기울이는 데 쓴다 — 실측/잔상 등 실제 증거가 있는 밴드는 건드리지 않는다(방향 힌트가
#   실패해도 원래 로직으로 조용히 폴백되는 소프트 제약).
AVOID_HOLD_DIR_BIAS_PX = 20.0   # ≈ PASS_OFFSET(80.0, "7. 기타" 절, 실측 기반)의 1/4.
                                 #   ★비율 자체는 실측/재검증 필요★

# ── da 근접 컷(obstacle-cut) — avoid_hold(위)와 완전히 독립된 메커니즘 ──
#   배경: 장애물/방해차량이 잡혔을 때 da 안전마진(§2.30)의 국소 침식만으로는 반응이
#   너무 완만하다(장애물 바로 앞에서만 살짝 밈) — lookahead를 늘려도 Pure Pursuit
#   curvature=2·sin(α)/ld 공식상 ld가 커질수록 오히려 반응이 희석되는 역효과만 확인돼
#   ("lookahead 확장" 검토 후 폐기, README §2.5x 참고) 대신 차량↔장애물 사이
#   구간의 da를 장애물 쪽 절반만 통째로 잘라("근접 컷") 갈림길을 뚜렷하게 만드는
#   방식으로 전환 — perception/dl_lane.py _clip_da_by_obstacle() 참고.
#   ENABLE_OBSTACLE_CUT=False가 기본값이다 — 부호규약(장애물 쪽을 정확히 잘라야
#   하는지 반대로 잘라 오히려 장애물 쪽으로 조향하게 되는지)이 실차 미검증이라,
#   반드시 정지/저속에서 먼저 확인 후 켤 것.
ENABLE_OBSTACLE_CUT = True    # 현재 B2/B3의 실제 회피 메커니즘(위 배경 설명 참고) — 꺼진
                               # 채로는 트리거만 잡히고 실제 회피 조향/속도캡이 전혀 안 걸린다.
                               # ⚠ 위 "부호규약 실차 미검증" 경고 그대로 유효 — 반드시 정지/저속
                               # 구간에서 먼저 방향 확인 후 트랙 주행에 쓸 것.
                               # ENABLE_BEHAVIOR/TEST_DISABLE_B2_B3와 무관하게 독립적으로 켜고 끔

# ── da 근접 컷 진입 트리거 (perc_obstacle_cut_trigger(), track_drive.py) ──
#   perc_lavacon_trigger()와 동일한 "라이다 AND YOLO 카메라" 이중확인 패턴이지만,
#   perc_obstacle()의 공유 ROI(FRONT_X_MAX/FRONT_Y_HALF, 다른 B2/B3/avoid_hold
#   소비처와 공유)를 재사용하지 않고 이 트리거 전용의 독립 라이다 ROI를 새로 잡는다
#   — 나중에 그 소비처들의 튜닝이 이 트리거와 갈라져도 서로 간섭하지 않게.
#   트리거 거리 1.0m는 da BEV 캔버스의 표현 한계(DL_BEV_FAR_LIMIT_M=0.7m)보다 살짝
#   여유를 둔 값 — 디바운스(OBSTACLE_CUT_TRIGGER_FRAMES)가 끝나는 시점이 da가 실제로
#   컷을 보여줄 수 있는 경계(0.7m) 바로 앞에 오도록 확정했다(더 멀리 보는 것 자체는
#   의미 없음 — da가 0.7m보다 먼 거리를 표현 못 해서 어차피 컷의 "먼 경계"는 항상
#   캔버스 끝으로 클램프된다). B2(고정장애물)/일반 기본값.
OBSTACLE_CUT_TRIGGER_X_MAX_M  = 1.0   # 실차 미검증
# 트리거 ROI 전방 하한(m) — B2/B3 공용. 라이다 원점 근처 차체 반사/노이즈를 배제하려고
#   뒤로(멀리) 밀었다.
OBSTACLE_CUT_TRIGGER_X_MIN_M  = 0.25
# B3(방해차량)는 위 DL_BEV_FAR_LIMIT_M=0.7m 캡보다 훨씬 멀리 본다(트리거만 먼저 잡고,
#   실제 컷 지오메트리는 여전히 da가 보이는 범위로 클램프됨 — OBSTACLE_CUT_NEAR_M/da
#   캔버스 한계 참고). 즉 트리거가 da로 시각 확인되기 전에 먼저 울릴 수 있음 — 실차
#   미검증, 오검출 잦으면 다시 좁힐 것.
OBSTACLE_CUT_TRIGGER_X_MAX_M_VEHICLE = 2.5   # B3(방해차량) 전용 전방 트리거 거리(m)
OBSTACLE_CUT_TRIGGER_Y_HALF_M = 0.55  # 횡방향 반폭 — LANE_WIDTH_M(0.4m) 기준 한 차선+여유, 실차 미검증 추정치
                                       #   B2(고정장애물=콘)/일반 기본값. B3(방해차량)는 아래
                                       #   OBSTACLE_CUT_TRIGGER_Y_HALF_M_VEHICLE로 별도 사용
                                       #   (perc_obstacle_cut_trigger()가 self._b2_passed로 분기).
OBSTACLE_CUT_TRIGGER_Y_HALF_M_VEHICLE = 0.75  # m — B3(방해차량) 전용, 실차 미검증. 너무 넓어
                                               #   인접 차선 물체까지 잡히면 다시 좁힐 것.
OBSTACLE_CUT_TRIGGER_FRAMES   = 2     # 라이다 AND YOLO 연속확인 프레임 수(디바운스) — 실차 미검증
                                       #   ★값을 더 낮출수록(1까지) 반응은 빨라지지만 노이즈(순간 오검출) 하나로도 트리거가 걸릴 위험이 커진다 — 실차에서 오검출 잦으면 다시 3으로.

# ── da 근접 컷 지오메트리 + 유지/해제 타이머 ──
#   트리거 확정 시점엔 obstacle_dist가 항상 0.7~1.0m(=da 크롭 한계 이내)이므로 컷의 먼
#   경계를 obstacle_dist로 계산하지 않는다 — "지금 보이는 da 전체 깊이"를 그대로 먼
#   경계로 쓰고, 가까운 경계만 아래 OBSTACLE_CUT_NEAR_M로 고정한다.
OBSTACLE_CUT_NEAR_M = 0.1                 # 컷의 차량쪽 고정 경계(m) — 차량 뒤 더 넓은 범위 차단
# B2(고정장애물)/B3(방해차량)가 컷 좌우폭을 따로 가진다(둘 다 None이면 기존과 동일하게
#   LANE_WIDTH_M*DL_PIXELS_PER_METER로 계산, _clip_da_by_obstacle() 참고). B2는 "조향이
#   너무 크다"는 실차 체감 피드백으로 좌우폭을 10% 줄였다(OBSTACLE_CUT_HALF_WIDTH_SCALE_FIXED).
#   B3는 아직 실차 미검증 상태 그대로 유지.
OBSTACLE_CUT_LANE_HALF_WIDTH_PX_FIXED   = None   # B2(고정장애물) 전용
OBSTACLE_CUT_LANE_HALF_WIDTH_PX_VEHICLE = None   # B3(방해차량) 전용 — 기존 동작과 동일(배율 없음)
OBSTACLE_CUT_HALF_WIDTH_SCALE_FIXED = 0.9        # B2 전용 배율 — LANE_WIDTH_M*DL_PIXELS_PER_METER(80px) 대비 10%↓(→72px), 실차 미검증
OBSTACLE_CUT_MIN_REMAIN_PX = 25.0         # 클리핑 후 밴드에 이 폭(px) 미만만 남으면 그 밴드는 컷을 건너뛴다
                                           #   — da가 완전히 비면 pure_pursuit.control()의 "path 없으면 직전값 유지(held)"
                                           #   폴백이 걸려 회피가 가장 필요한 순간 조향이 얼어붙는 위험을 방지. 실차 미검증 추정치.
# 해제는 진입과 동일한 전용 트리거 ROI(위 OBSTACLE_CUT_TRIGGER_*)로 재계산한 "clear"
#   상태를 쓴다 — perc_obstacle()의 공유 ROI(범위가 다름)를 쓰면 해제 타이밍이 트리거
#   설계 의도와 어긋난다.
OBSTACLE_CUT_HOLD_SEC_MIN = 2.0           # 진입 확정 후 라이다/YOLO가 뭐라 하든 무조건 유지하는 최소시간(B3/방해차량 기준) — 실차 미검증
# B2(고정장애물=콘)는 정지해 있어 회피가 끝나면 바로 지나쳐가므로, B3와 같은 2.0초를
#   그대로 쓰면 이미 다 지나간 뒤에도 컷이 오래 남아있는다. B2로 진입할 때만 이 값을
#   최소유지시간으로 쓴다(_update_obstacle_cut_hold() 진입 순간 self.obstacle_cut_type으로
#   분기, README §4.3 참고). OBSTACLE_CUT_RELEASE_CONFIRM_FRAMES 해제 디바운스는 그대로
#   공유 — 이 값은 "floor"만 낮춘다.
OBSTACLE_CUT_HOLD_SEC_MIN_FIXED = 0.2     # B2(고정장애물) 전용 최소유지시간 — 실차 미검증
OBSTACLE_CUT_RELEASE_DIST_M = 1.0         # 트리거와 동일(히스테리시스 없음). 실차 미검증
OBSTACLE_CUT_RELEASE_CONFIRM_FRAMES = 4   # 해제 디바운스 — AVOID_HOLD_RELEASE_CONFIRM_FRAMES와 동일 관례, 실차 미검증

# ── B2 종료 → B3 무장 지연 ──
#   B2(고정장애물) 통과 확정(_mark_behavior_passed('B2')) 직후 곧장 B3(방해차량) 관련 판정
#   (_active_yolo_stage()의 vehicle YOLO 전환, perc_obstacle_cut_trigger()의 VEHICLE 전용
#   트리거 ROI 전환, obstacle_cut_type 'vehicle' 태깅)을 켜면 방금 지나친 B2 장애물이 아직
#   시야/라이다에 남아있는 채로 B3로 오인식될 위험이 있다 — 이 시간(초) 동안은 그 판정들을
#   전부 보류하고 B2와 동일하게(cone/fixed) 취급한다. track_drive.py _b3_armed() 참고.
B2_TO_B3_DELAY_SEC = 3.0  # 실차 미검증

# ── 컷 활성 "전"(장애물 미감지 구간) 속도 캡 ──
#   라바콘 탈출 직후(Phase.OBSTACLE_ZONE 진입)부터 실제로 장애물을 감지해 회피 조향이
#   들어가는 순간(obstacle_cut_active=True)까지 이 값으로 속도를 낮춰 유지하고, 회피가
#   시작되면 캡을 아예 풀어(_update_speed() 참고) 일반 주행속도(SPEED_NORMAL 기반
#   코너감속 로직)로 올린다. 실차에서 회피 기동 중 불안정하면 이 캡 대신 회피 조향
#   로직 자체(da 근접 컷) 쪽 튜닝을 먼저 볼 것. 실차 미검증.
SPEED_PRE_OBSTACLE_CAP = 8.0

# B1(Phase.LAVACON) 중 목표속도 상한 — track_drive.py `_update_speed()`가
#   SPEED_PRE_OBSTACLE_CAP과 동일한 방식(target_speed에 min()으로만 얹음)으로 적용한다.
#   accel_step 램프가 그대로 적용된 채로만 상한을 낮추는 방식이라, 매 틱 정확한 값으로
#   강제 고정해 "굳는" 증상을 일으키지 않는다 — 실차 미검증, 라바콘 구간에서 급감속/정지처럼
#   느껴지면 값을 올릴 것.
SPEED_LAVACON_CAP = 8.0

# DL_CENTER_MODE='ll' 내부에서 실제 밴드 중심 계산 알고리즘을 고르는 2차 스위치 — 두
#   구현을 둘 다 남기고 전환 가능하게 유지한다(README §2.19 참고).
#   'yw' (main 기본값, 팀원 작성) : 노란 중앙선 + (차선 판정에 따른) 한쪽 흰색
#        경계선을 짝지어 추적한다(DLSlideWindow._ll_yellow_white_centers()). 노란선이
#        안 보이면 3분기 폴백(양쪽 흰선 실측/한쪽만 실측/잔상). 관련 튜닝값:
#        DL_LL_YELLOW_GAP_INIT/EMA_ALPHA/MIN/MAX_PX, DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX
#        (아래).
#   'lr' (이지유 작성) : 좌/우 흰선을 각각 완전히 독립된 슬라이딩 윈도우로 추적한다
#        (DLSlideWindow._ll_slice_centers()) — 적응형 탐색창(속도 예측+확장) + 밴드별
#        프레임 간 앵커링 포함. 노란선이 있는 차로에서는 그쪽 흰선 탐색이 구조적으로
#        계속 실패하는 한계가 있다(팀원이 'yw'로 재설계한 이유이기도 함) — 그래서
#        main 기본값은 'yw'다. 관련 튜닝값: DL_LL_WIDTH_MIN/MAX_PX/EMA_ALPHA,
#        DL_LL_VELOCITY_*, DL_LL_SEARCH_WIDEN_*, DL_LL_BAND_ANCHOR_ALPHA(아래).
#   두 알고리즘 다 DL_LL_SEARCH_HALF_WIDTH_PX/DL_LL_SIDE_MIN_PIXELS는 공유한다(둘 다
#   "좁은 탐색창 + 최소 픽셀수" 기본 뼈대는 같아서). 실차에서 A/B 비교할 때 이 값만
#   바꾸면 된다 — DL_CENTER_MODE는 그대로 'll' 유지.
#   DL_LL_VELOCITY_EMA_ALPHA/DL_LL_VELOCITY_MAX_PX/DL_LL_BAND_ANCHOR_ALPHA(아래 'lr'
#   섹션에 있음)는 "밴드 간 이동 속도를 추적해 탐색창을 미리 옮기고, 직전 프레임 그
#   밴드 위치로 당긴다"는 동일한 물리적 개념이라 'yw'/'lr' 둘 다 공유해서 쓴다.
DL_LL_ALGO = 'yw'  # 'yw'(노란+흰선 짝짓기, main 기본) | 'lr'(좌우 흰선 독립 슬라이딩 윈도우)

# ── DL_CENTER_MODE='ll_da'(corridor 알고리즘) 전용 튜닝값 (전부 실차 미검증 초기값) ──
# 밴드 내 ll connected-component가 "진짜 선 하나"로 인정될 최소 픽셀수.
#   DL_LL_SIDE_MIN_PIXELS(=15, 'll' 모드용 "한쪽 절반" 기준)보다 살짝 낮게 잡았다 —
#   corridor는 밴드 안에서 선 하나하나를 개별로 세므로(좌/우 절반이 아니라 선 단위),
#   같은 밴드라도 선 1개가 차지하는 픽셀수는 그보다 적을 수 있다.
DL_CORRIDOR_LINE_MIN_PIXELS = 12
# 점선 틈으로 같은 물리적 선이 한 밴드 안에서 connected-component 2개로 쪼개졌을 때
# 병합할 x거리(px) 임계값 — 실측 아님, DEBUG_VIZ_DL_LANE에서 같은 선이 두 번 잡히는지
# 보고 조정할 것.
DL_CORRIDOR_LINE_MERGE_PX = 15
# corridor(1~3번째 선, 전체 트랙=양쪽 차로 폭) 정상 폭 범위(px) — 편도 1차로 정상범위
# (실측 차로폭 0.8m@200px/m=160px 근방)의 2배 근방으로 계산한 추정치(320px ±약 40%).
# 실측 아님 — 직선 구간에서 corridor_bounds 시각화로 재조정할 것.
DL_CORRIDOR_WIDTH_MIN_PX = 190
DL_CORRIDOR_WIDTH_MAX_PX = 450
# corridor 안에서 da가 "열려있다(지나갈 수 있다)"고 인정할 최소 폭(px) — 차량 실폭에
# 안전마진을 곱해 px로 환산한 추정치(약 0.31m*1.3≈0.40m=80px@200px/m). 실측 아님 —
# 좁은 틈으로 무리하게 끼어들지 않는지 실차에서 확인 후 조정할 것.
DL_CORRIDOR_MIN_PASSABLE_PX = 80

# _pick_open_run()의 프레임 간 히스테리시스(직전 프레임 채택 위치와 가장 가까운 open
#   run을 우선)는 정적이라, 빠른 S자에서 실제 열린 구간 위치가 그 사이 크게 이동하면
#   뒤처질 수 있다. da/ll 모드에 적용한 것과 동일한 원리로 밴드 간 이동 속도를 EMA
#   추적해(_corridor_slice_centers()) prefer_x를 "직전 위치 + 예측 이동량"으로 미리
#   옮긴다(README §2.27). corridor는 좌/우 두 갈래가 아니라 "열린 구간 하나"만 추적하므로
#   da처럼 스칼라 하나면 된다. 실차 미검증 초기값.
DL_CORRIDOR_VELOCITY_EMA_ALPHA = 0.3
DL_CORRIDOR_VELOCITY_MAX_PX = 40.0

# DL_CENTER_MODE='ll'일 때 쓰는 ll 중점 채택 임계값.
# 밴드 내 ll 픽셀수가 이 미만이면(노란선/흰선 각각 판정) "이 밴드는 그게 안 보임" 처리.
#   DL_MIN_PIXELS(=40, da용)보다 낮은 이유: ll은 da처럼 면을 채우는 마스크가 아니라 가는
#   선이라 같은 밴드 안에 있는 픽셀수 자체가 원래 훨씬 적다. 실차 미검증 초기값.
DL_LL_SIDE_MIN_PIXELS = 15

# _ll_yellow_white_centers()가 노란선/흰선을 찾을 때 보는 탐색창 반경(px). 좌/우 분리
#   기준점 하나로 밴드를 절반씩(왼쪽 전체/오른쪽 전체, 보통 수백 px) 나눠 그 안 전체
#   픽셀로 무게중심을 내면, 그 "반쪽"이 넓다 보니 옆 차선 선이나 반사광이 반쪽 어디에
#   있든 평균에 섞여 들어가는 문제가 있다(다중 후보 오탐). 참고:
#   github.com/junhyukch7/Advanced-Lane-Detection의 슬라이딩 윈도우가 폭 120px(반경
#   60px)짜리 좁은 창만 보는 것에서 착안 — 창 밖의 무관한 픽셀이 애초에 평균 계산에 안
#   들어오게 예상 위치 중심의 좁은 창만 보도록 한다. 실차 미검증 초기값 — 급커브에서
#   밴드 간 실제 선 이동량이 이 값보다 크면 창이 선을 놓치고 추적이 끊길 수 있으니,
#   그런 구간에서 검출 밴드 비율이 뚝 떨어지면 이 값을 키울 것.
DL_LL_SEARCH_HALF_WIDTH_PX = 60.0

# 노란선 대비 흰색 경계선까지의 간격(px) 러닝 추정치 self._white_yellow_gap_px의 초기값
#   /EMA 계수. 둘 다 찾은 밴드에서만 이 계수로 갱신한다(_ll_yellow_white_centers() 참고).
#   흰-노 간격 실측 0.4m을 DL_PIXELS_PER_METER(200px/m)로 환산해 80px로 잡았다 — 다만
#   DL_PIXELS_PER_METER 자체가 "설계값(실측 아님)"이라(위 주석 참고), 이 200px/m 환산이
#   실제로 맞는지는 별도 확인 필요. DEBUG_VIZ_DL_LANE에서 정상 구간의 gap 표시값이 80px
#   근방으로 수렴하는지 보고 재조정할 것.
DL_LL_YELLOW_GAP_INIT_PX = 80.0
DL_LL_YELLOW_GAP_EMA_ALPHA = 0.1

# gap EMA 상하한 클램프 — 노란선이 죽기 전 노이즈(글레어 등)로 큰 |흰선-노란선| 값이
#   잡히면 EMA가 크게 부풀 수 있고, 그 직후 노란선이 아예 안 잡히기 시작하면 "둘 다
#   찾았을 때만 갱신"되는 이 값이 부푼 채로 얼어붙는다(실측 40cm=80px 대비 과대). 그
#   상태로 한쪽 선 없는 밴드의 위치 추정에 부푼 gap을 그대로 쓰면 waypoint가 실제
#   흰선을 넘어 차선 밖까지 밀려나 급조향으로 이어질 수 있다(실차 재현 확인). 실측값
#   (80px) 근방으로 상하한을 걸어 어떤 노이즈가 껴도 이만큼은 안 부풀게 막는다 —
#   DEBUG_VIZ_DL_LANE의 gap 표시값이 이 범위 끝에 계속 붙어있으면 실제 트랙 폭이 이
#   범위 밖일 수 있다는 뜻이니 재조정할 것.
DL_LL_YELLOW_GAP_MIN_PX = 50.0
DL_LL_YELLOW_GAP_MAX_PX = 110.0

# 노란선이 이번 밴드에서 안 보일 때 흰선을 찾는 탐색창 반경(px) — DL_LL_SEARCH_HALF_WIDTH_PX
#   (60, 노란/흰 각각 하나씩 좁게 찾는 창)와 별개로, "노란선 없을 때 양쪽 흰선이 몇
#   개나 보이는지"를 세야 하므로 그보다 훨씬 넓게 잡는다 — 좌우 흰선이 각각 gap(최대
#   DL_LL_YELLOW_GAP_MAX_PX=110)만큼 떨어져 있을 수 있으므로 그보다 여유를 더 둔 값.
#   cur_yellow(기준점) 중심으로 이 반경 안의 흰선 connected component를 전부 찾아
#   (_ll_line_centers() 재사용) 개수로 3분기한다(_ll_yellow_white_centers() 참고): 2개=
#   양쪽 다 보임(중점 채택), 1개=한쪽만 보임(어느 쪽인지 실측 위치로 판정 후 gap만큼
#   안쪽으로 재구성), 0개=잔상. 실측 미검증 — 실제 트랙 폭 기준으로 재조정할 것.
DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX = 150.0

# ── DL_LL_ALGO='lr'(좌/우 흰선 독립 슬라이딩 윈도우, _ll_slice_centers()) 전용 튜닝값 ──
#   밴드 내 좌/우 ll 중점을 채택하기 위한 두 선 사이 거리(px) 허용범위 — 범위 밖이면
#   (반대쪽 밴드의 다른 차선을 잘못 짝지은 경우 등) 그 밴드는 버린다(da 폴백 없음).
#   실측 라인 간격이 75~80px로 나와 하한을 100에서 낮췄다 — 여전히 넓게 열려있으니
#   visualize()의 밴드별 실측 폭 표시로 좁힐 것. 실차 미검증 초기값.
DL_LL_WIDTH_MIN_PX = 50
DL_LL_WIDTH_MAX_PX = 200
# 좌/우 둘 다 찾아 실측 폭이 나온 밴드에서만 self._ll_half_width(차로 반폭 러닝
#   추정치)를 이 계수로 EMA 갱신한다. classic_cv 백엔드의 LANE_WIDTH_EMA_ALPHA(=0.1,
#   hough_lane.py)와 동일한 관례.
DL_LL_WIDTH_EMA_ALPHA = 0.1

# _ll_slice_centers()(DL_LL_ALGO='lr') 적응형 탐색창 — 밴드 간 실제 선 이동량이
#   DL_LL_SEARCH_HALF_WIDTH_PX보다 크면 창이 선을 놓치는 문제 대응. 두 갈래로 완화한다:
#   ①속도 예측 — 그 사이드에서 실제로 찾은 밴드들 사이의 x 이동량(밴드 간 간격으로
#     나눈 px/밴드)을 EMA로 추적해뒀다가, 다음 밴드 탐색창 중심을 "마지막으로 찾은
#     위치"가 아니라 "그 위치 + 예측 이동량"으로 미리 옮긴다(창이 곡선을 따라 먼저
#     움직임).
#   ②탐색창 확장 — 그 사이드가 연속으로 못 찾으면 매 실패마다 창 반경을 넓혀 다음
#     기회에 더 넓은 범위에서 재포착을 시도하고, 다시 찾으면 원래 반경으로 리셋한다.
#   실차 미검증 초기값.
DL_LL_VELOCITY_EMA_ALPHA = 0.3     # 밴드 간 이동 속도(px/밴드) EMA 계수
DL_LL_VELOCITY_MAX_PX = 40.0       # 예측 이동량 클램프 — 노이즈로 속도 추정이 튀어도 창이 한번에 너무 멀리 안 튀도록
DL_LL_SEARCH_WIDEN_STEP_PX = 15.0  # 연속 미검출 1회당 탐색창 반경 증가폭
DL_LL_SEARCH_WIDEN_MAX_PX = 120.0  # 탐색창 반경 상한(기본 반경 DL_LL_SEARCH_HALF_WIDTH_PX=60의 2배)

# [2026-08-10] _ll_slice_centers() 밴드별 프레임 간 앵커링 — 기존에는 band 0(근거리)
#   만 직전 프레임 전체 확정 lane_center(ref_x)에 앵커링되고, 그 위 밴드는 전부 "같은
#   프레임 안에서" 아래 밴드 결과를 이어받는 식으로만 전파됐다. band 0 검출이 노이즈로
#   살짝 틀어지면 그 오차가 위 밴드들까지 누적되어 번져나가는 문제가 있다. 밴드마다
#   "직전 프레임에 그 밴드(같은 y위치)에서 실제로 찾은 위치"를 따로 기억해뒀다가,
#   이번 프레임 그 밴드의 탐색창 중심을 (이번 프레임 내 전파값, 직전 프레임 그 밴드
#   값)의 가중평균으로 잡는다 — 도로 곡률이 프레임 간 급격히 안 변한다는 가정에 기대어,
#   band 0의 오차가 위로 그대로 번지지 않고 각 밴드 고유의 과거 위치 쪽으로 당겨지게
#   한다. 실차 미검증 초기값.
DL_LL_BAND_ANCHOR_ALPHA = 0.35     # 밴드별 탐색창 중심 계산 시 "직전 프레임 그 밴드 위치"에 주는 가중치(0=무시, 1=전적으로 그 값만 사용)

# ── 색상기반 노란 중앙선 보조 검출 (lane_side 판정용, hough_lane.py와 공유) ──
#   TwinLiteNet의 ll 출력은 흰/노랑을 구분하지 않아 HSV로 별도 검출한다.
YELLOW_LOWER = np.array([15, 80, 80])
YELLOW_UPPER = np.array([40, 255, 255])

# ── 좌회전 진입 랜드마크 후보 — 끊긴 노란 중앙선(dash) 카운팅 (hough_lane.YellowDashCounter) ──
#   [2026-08-21] "신호등→좌회전 입구" 트리거를 속도기반 거리적분(S2_COMMIT_DIST_M류) 대신
#   실제로 보이는 노란 파선 개수로 판단하는 대안 실험(요청 반영) — 정지위치 산포·제어주기
#   해상도 문제(TURN_DIST_M 넛지 논의 참고) 자체를 회피할 수 있다는 게 장점. 위
#   YELLOW_LOWER/UPPER와 동일 HSV 기준으로 근거리 ROI 안 노란 픽셀 유무를 매 프레임
#   판단하고, '있음↔없음' 전이를 디바운스해서 파선 하나가 지나갈 때마다 카운트를 올린다.
#   ★주의★ 카운팅을 언제부터 시작할지(예: 노란 하프 출발선 검출 시점)는 아직 미정 —
#   YellowDashCounter는 카운팅 로직만 담당하고, reset() 호출 시점은 호출부(상태머신) 책임.
#
#   [2026-08-21] ~/Downloads/lap_001 실제 캡처로 확인해보니, ROI 안 전체 픽셀수만 보는
#   방식은 트랙 밖 나무색 바닥재/크레이트를 노란 파선으로 오검출한다(YELLOW_LOWER/UPPER
#   범위가 나무색과도 겹침, frame_000678 재현됨) — 이걸 막으려고 요청받아 다음 두 조건을
#   추가했다: ROI 좌우를 살짝 좁히고(YELLOW_DASH_ROI_X0/X1), 픽셀수 대신 커넥티드컴포넌트
#   단위로 "덩어리 하나"가 ① 면적이 파선 하나 크기 범위 안이고(YELLOW_DASH_MIN/MAX_AREA_PX)
#   ② ROI 좌우 테두리에 안 닿아야(닿으면 ROI 밖으로 잘린 큰 덩어리의 일부라는 뜻) 파선으로
#   인정한다. 처음엔 "가로세로비(세로가 길어야 함)"도 필터로 넣으려 했는데, 실제 캡처에서
#   화면 가장자리 쪽 파선은 어안렌즈 왜곡+원근 때문에 대각선으로 찍혀 바운딩박스가 넓적해져
#   있었다(frame_001583 재현 — aspect비로 걸렀으면 진짜 파선을 놓쳤을 것) — 그래서 aspect비
#   필터는 빼고 면적상한+테두리비접촉 두 개만 쓴다. 5개 실제 프레임(0.1/0.3/0.5/0.7/0.9
#   지점)으로 검증: 진짜 파선 4장 모두 인정, 나무바닥 오검출 1장 모두 거부.
YELLOW_DASH_ROI_TOP = 0.60   # ROI 세로비율(0=위/원거리~1=아래/근거리) — 차량 바로 앞 근거리만 봄
YELLOW_DASH_ROI_BOT = 0.95
YELLOW_DASH_ROI_X0 = 0.05   # ROI 가로비율 좌측 시작 — 트랙 밖 바닥/사람이 자주 걸리는 맨 가장자리만 살짝 제외
YELLOW_DASH_ROI_X1 = 0.95   # ROI 가로비율 우측 끝
YELLOW_DASH_MIN_AREA_PX = 150     # 컴포넌트 면적이 이 이상이어야 "파선 후보"(잡음/반사 필터, 기존과 동일)
YELLOW_DASH_MAX_AREA_PX = 2000    # 컴포넌트 면적이 이 이하여야 함 — 실측 진짜 파선 최대 ≈1200px, 나무바닥 오검출은 ≈9000px라 여유있게 구분됨
YELLOW_DASH_PRESENT_FRAMES = 2    # 연속 이 프레임 이상 보여야 "파선 하나 확정"(카운트 +1)
YELLOW_DASH_ABSENT_FRAMES  = 2    # 연속 이 프레임 이상 안 보여야 "다음 파선 대기" 상태로 재무장
TURN_DASH_TRIGGER_COUNT = 4       # "3개 넘기고 4번째가 다가오는 순간" — count(완전히 지나친 개수)가 TURN_DASH_TRIGGER_COUNT-1(=3)이면서
                                   #   동시에 YellowDashCounter.present가 True(4번째가 지금 보이는 중)일 때 좌회전 트리거.
                                   #   count만으로는 판단 불가 — [2026-08-21] "보이자마자"가 아니라 "지나쳐야" 세는 걸로
                                   #   바꿨기 때문에(hough_lane.YellowDashCounter 주석 참고), 4번째가 다가오는 중엔 아직 count=3.
                                   #   아직 상태머신 미연결.

# ── 좌회전 진입 랜드마크 — 체크무늬(흑/노랑 교차) 게이트 밴드 (hough_lane.check_checker_band()) ──
#   [2026-08-21] 사용자가 실차 사진으로 확인해준 "노란 하프 출발선" = 흑/연노랑 체크무늬가
#   트랙을 가로지르는 밴드(정지선처럼 단색 굵은 선이 아니라 체커 패턴). check_stopline()
#   (perc_floor.py, 흰색 단일톤 비율 임계값)과 같은 ROI+비율 임계값 방식을 쓰되, 이건
#   "어두운 픽셀 비율"과 "노란 픽셀 비율"이 같은 밴드 안에 동시에 일정 이상 있어야
#   확정한다 — 그래야 맨 도로(둘 다 낮음)나 순수 노란 점선 중앙선(어두운 비율 낮음)과
#   헷갈리지 않는다. YELLOW_LOWER/UPPER는 위 것 그대로 재사용.
#   ★주의★ 색상값(특히 CHECKER_BLACK_MAX_V)은 사용자가 보여준 사진을 육안으로 보고 잡은
#   추정치일 뿐 실측 HSV 샘플링을 거치지 않았다 — 실차/실제 캡처로 반드시 재조정할 것.
CHECKER_ROI_TOP = 0.45   # ROI 세로비율 — 체크무늬는 하프 출발선이라 파선 중앙선(YELLOW_DASH_ROI)보다 더 먼 위치에서 잡힘
CHECKER_ROI_BOT = 0.75
CHECKER_BLACK_MAX_V = 60        # 그레이스케일 값이 이 이하면 "검정 사각형"으로 카운트
CHECKER_DARK_RATIO_TH   = 0.15  # ROI 내 검정 비율이 이 이상이어야 함
CHECKER_YELLOW_RATIO_TH = 0.10  # ROI 내 노란 비율이 이 이상이어야 함
CHECKER_CONFIRM_FRAMES = 2      # 연속 이 프레임 이상 위 조건을 만족해야 "게이트 확정"(오검출 디바운스, LAP_YAW_CONFIRM_FRAMES와 동일 패턴)

# ── 좌회전 진입 랜드마크 — 체크무늬 게이트 라이다 기둥쌍 검출 (perc_checker_pillar()) ──
#   [2026-08-21] 위 CHECKER_ROI_*(비전, HSV+비율)와 이번 세션에 시도한 3가지 변형(면적+
#   테두리접촉, 전이횟수, 폭균일성)까지 전부 실제 캡처(signal_masking_dataset + lap_001
#   오검출 프레임)로 검증했지만 배경(사람/의자/나무바닥)이 체커보드보다 더 어둡고 더
#   노란 경우가 많아 임계값으로 못 갈랐다 — 비전 휴리스틱은 이 마커엔 한계가 뚜렷하다고
#   판단(요청 반영). 대신 체크무늬 게이트가 걸린 신호등 게이트 구조물의 실측 좌우 기둥
#   간격(≈0.98m)을 라이다로 직접 재는 방식으로 실험 전환 — perc_lavacon_trigger()와
#   완전히 동일한 패턴(극좌표→직교좌표, 종방향 ROI, 좌/우(y>0/y<0) 클러스터 각각 탐지)을
#   재사용하되, 라바콘엔 없던 "좌우 클러스터 간 횡방향 거리가 실측값과 일치해야 함" 조건을
#   추가해 다른 트랙 구조물과의 오인식을 줄인다. 조명/색상에 전혀 안 흔들리는 순수 기하
#   측정이라 이번 세션에서 반복 실패한 비전 휴리스틱보다 신뢰도가 높을 것으로 기대되나,
#   이 저장소엔 실제 라이다 캡처 데이터가 없어 로직 자체를
#   실측 검증하지 못했다 — 실차에서 반드시 확인할 것. LON_MIN/MAX는 "기둥이 차량 바로
#   옆을 지나는 순간"을 잡으려는 추정치라 perc_lavacon_trigger()의 0.3~0.5m보다 더 넓게
#   잡아뒀다(게이트 구조물이 콘보다 커서 감지 가능 구간도 더 길 것으로 추정) — 전부 실차
#   미검증 초기값.
#   [2026-08-22] 실차 확인 후 ROI 재조정(요청 반영) — 횡방향 한계를 2.0→1.0m로 좁히고,
#   종방향 구간은 폭(0.7m)은 유지한 채 차량 쪽으로 0.5m 당김(0.1~0.8 → -0.4~0.3m).
#   [2026-08-22b] 실차 재확인 후 재조정(요청 반영) — 종방향 구간을 폭(0.7m) 유지한 채
#   전방으로 0.25m 밀어올림(-0.4~0.3 → -0.15~0.55m). 또한 라이다 자기반사/노이즈로
#   추정되는 극근접(0.1m 이내) 포인트를 아예 무시하도록 CHECKER_PILLAR_MIN_RANGE_M 신설.
#   [2026-08-22e] 0.1 → 0.3으로 상향(요청 반영) — 실차에서 0.1m로는 근접 노이즈가 여전히
#   안 걸러지는 게 확인됨(원인 미확정 — 실제 반사 거리가 0.1~0.3m 대라는 뜻일 수도, 다른
#   경로 문제일 수도 있음). 실차 미검증, 이번에도 안 걸러지면 코드 경로 자체를 재확인할 것.
CHECKER_PILLAR_LON_MIN = -0.15      # 트리거 ROI 전방 종방향 하한(m)
CHECKER_PILLAR_LON_MAX = 0.55       # 트리거 ROI 전방 종방향 상한(m)
CHECKER_PILLAR_LAT_MAX = 1.0        # 트리거 ROI 횡방향 한계(m) — perc_lavacon_trigger()와 동일
CHECKER_PILLAR_MIN_RANGE_M = 0.3    # 이 거리(m) 이내로 찍힌 포인트는 자기반사/노이즈로 보고 무시
# [2026-08-22b] 좌회전 신호 확정 커밋 중에만 쓰이는 트리거라(perc_checker_pillar() 호출부
# 주석 참고) 이 값을 낮추는 효과가 그 구간에만 국한된다 — 이전엔 한쪽이라도 연속
# 2포인트(+거리편차 이내)를 못 채우면 그 사이드 자체가 "미검출"이라 기둥쌍이 잘 안 잡힌다는
# 실차 보고(요청 반영) — 좌우 각 1포인트만 찍혀도 그 사이드는 검출된 것으로 완화.
CHECKER_PILLAR_CLUSTER_MIN_PTS = 1  # 클러스터 최소 연속 포인트 수(좌우 각각 이 이상이면 그 사이드 검출)
CHECKER_PILLAR_CLUSTER_MAX_GAP = 0.35  # 같은 클러스터로 볼 최대 거리편차(m) — 기둥 굵기 실측 후 재조정할 것(콘 지름 근사값 그대로 재사용 중)
CHECKER_PILLAR_LAT_TARGET_M = 0.98   # 좌우 기둥 사이 최소 횡방향 간격(m, lat_dist가 이 이상이면 통과)
                                     # [2026-08-22h] 0.5 → 0.98(실측 기둥 간격)로 복원(요청 반영) —
                                     # 실차 미검증, 무관한 라이다 클러스터쌍(예: 라바콘, 다른 장애물)을
                                     # 게이트로 오인할 위험을 줄이려면 실측값 그대로가 안전하다는 판단.
                                     # checker_pillar_bev 오탐/미탐 여부를 확인할 것.
# [2026-08-22b] 대칭 허용오차(± TOLERANCE) 대신 "간격이 이 값 이상이면 통과"로 완화(요청
# 반영) — 위 CLUSTER_MIN_PTS 완화와 짝: 사이드당 1포인트만으로 잡은 위치는 좌표가 덜
# 안정적이라, 상한까지 좁게 걸면 실제 기둥쌍인데도 근소하게 밀려나 놓칠 위험이 더 크다고
# 판단(실차 미검증, perc_checker_pillar()의 lat_ok 참고).
CHECKER_PILLAR_CONFIRM_FRAMES = 2   # 연속 이 프레임 이상 좌우+간격 조건을 만족해야 확정(디바운스)
CHECKER_PILLAR_LIDAR_TIMEOUT_SEC = 5.0  # [안전장치] 좌회전 커밋 시작 후 이 시간(초)까지 좌우
                                      #   기둥쌍이 안 잡히면(라이다 죽음/오검출 등) 좌회전 자체를
                                      #   포기한다 — [2026-08-24] 요청 반영, 예전처럼 거리기반으로
                                      #   좌회전을 "강제 시작"하지 않고, 직진 신호를 받았을 때와
                                      #   완전히 동일한 경로(S1_LANE_FOLLOW 유지 + Behavior
                                      #   재무장, track_drive.py _s0_signal() 참고)로 넘어가
                                      #   정상 차선주행을 그대로 이어간다. 실차 미검증 초기값.

# ── 좌회전 진입 — 체크무늬 게이트 통과 후 완만한 조향 램프 ──
#   [2026-08-21] 위 라이다 기둥쌍 트리거가 확정되는 순간부터, 고정 조향각을 즉시 걸지
#   않고 CHECKER_TURN_RAMP_START_ANGLE에서 CHECKER_TURN_RAMP_END_ANGLE까지
#   CHECKER_TURN_RAMP_DIST_M 동안 서서히 올린다(요청 반영: "-10~-30으로 완만히"). 거리
#   적분은 TURN_DIST_M류와 동일하게 _speed_mps_fallback() 기반(VESC 실측 + 명령속도
#   폴백)이라 이 램프도 그 안전장치를 그대로 물려받는다. 시간(time.time() 경과) 대신 거리
#   기준을 쓰는 이유: 조향각-거리 관계가 실제 주행 궤적의 곡률(회전반경)을 결정하므로,
#   속도가 바뀌어도(SPEED_NORMAL 3→10 등) 같은 거리 기준이면 게이트를 통과하는 물리적
#   곡선 모양이 그대로 유지된다 — 반대로 시간 기준이면 속도가 오를수록 곡선이 늘어져
#   버려 궤적이 달라진다. _s0_signal()의 'left' 커밋 구간 종료 트리거로 연결됨(요청 반영,
#   S2_COMMIT_DIST_M 거리기반 대신 기둥쌍 검출로 대체) — CHECKER_PILLAR_LIDAR_TIMEOUT_SEC
#   초과 시엔 좌회전 자체를 포기하고 직진 신호와 동일하게 처리(안전 폴백).
#   [2026-08-22] SPEED_NORMAL 3→10 상향 이후 0.5m가 실차에서 약 3프레임 만에 끝나버려
#   (README §"speed10 변경" 참고) 2.0으로 상향 — 20Hz·TURN_SPEED 기준 대략 십수 프레임
#   구간으로 늘어남. 실차 미검증 초기값이라 재검증 필요. CHECKER_TURN_RAMP_CURVE는
#   'linear'(거리에 선형 비례) 대신 기본을 'smoothstep'으로 바꿈 — 3t²-2t³ 형태라 양끝
#   (t=0 램프 시작, t=1 END_ANGLE로 고정 전환되는 지점) 모두 기울기 0이라 저크 없는 S자
#   곡선이 된다(예전 'ease_in'=t² 는 시작은 완만했지만 t=1에서 END_ANGLE 고정값으로
#   넘어가는 순간 기울기가 뚝 끊겨 저크가 있었음 — 이번에 대체).
CHECKER_TURN_RAMP_START_ANGLE = 0
CHECKER_TURN_RAMP_END_ANGLE   = -25.0   # [2026-08-25] 요청 반영: -20→-25
CHECKER_TURN_RAMP_DIST_M      = 2.5    # [2026-08-23] 요청 반영: 3.0→1.5 (짧게)
CHECKER_TURN_RAMP_CURVE       = 'smoothstep'  # 'linear' | 'smoothstep'

# ── 지름길 출구 T자 교차로 — 거리 기반 강제 좌회전 (2026-08-24) ──
#   T자 교차로라 직진(벽에 부딪힘)/좌회전/우회전 세 방향이 다 물리적으로 가능한데, 라이다
#   랜드마크도 없고 정지선도 없어서(사용자 실측 확인) 위 체크무늬 게이트 같은 물리 트리거를
#   쓸 수 없다 — da/차선인식도 T자 갈림에서는 어느 쪽이 "맞는" 방향인지 알 방법이 없다
#   (반시계방향 트랙이라 항상 좌회전이어야 하는데, da는 그런 규칙을 모름). 그래서 입구 램프
#   완료 시점(_do_checker_ramp_turn()의 done 분기)부터 누적거리를 재서 이 값에 도달하면
#   강제로 좌회전을 실행한다(track_drive.py _do_shortcut_exit_kick() 참고) — "언제 시작할지"
#   트리거만 여기서 정하고, 실제 조향 방식은 아래 SHORTCUT_EXIT_KICK_* 참고.
#   실측(사용자, 2026-08-24): 입구 램프 완료 지점 → T자 지점까지 직선거리 5.8m(단, 실제
#   주행경로는 입구 램프의 완만한 커브 때문에 직선보다 길다 — 곡선경로 길이 ≥ 두 점의
#   직선거리라 순수 직선값만 쓰면 항상 "조금 이르게" 트리거된다는 걸 의미). 여기에 VESC
#   적분 특유의 슬립/누적오차(코드베이스 관례상 늘 있는 오차원)까지 더해, 5.8m보다 여유를
#   얹어 시작한다 — 실차에서 좌/우로 튜닝할 것.
SHORTCUT_EXIT_DIST_M = 5.5 # [2026-08-25] 요청 반영: 6.3→4.5
# [2026-08-24] 근접 안전정지(obstacle_front/obstacle_dist 기반 SPEED_STOP 캡)는 도입했다가
#   삭제했다(요청 반영) — 대회 규정상 코스 이탈/충돌 시 사람이 차량을 들어 코스 안으로
#   복귀시키는 절차가 있어, 이 실패모드를 소프트웨어로 막을 필요가 없다는 판단. 거리
#   트리거(위) 단독으로만 판단한다.

# [2026-08-24b, 요청 반영] 출구 실제 조향을 입구(_checker_turn_ramp_angle(), 거리기반
#   smoothstep 램프)와 공유하던 걸 분리했다 — 입구/출구를 서로 독립된 별개 메커니즘으로
#   가져가는 게 낫다는 판단(사용자). LAVACON_KICK_*(config.py, B1 진입 킥)와 동일한
#   패턴 — 고정 각도를 정해진 시간(초) 동안 그대로 유지하는 오픈루프 킥. 거리 대신
#   시간 기준인 이유도 LAVACON_KICK과 동일: "짧고 확실하게 한 번 꺾어준다"가 목적이라
#   속도에 따라 궤적이 달라지는 걸 감수하고서라도 구현을 단순하게 가져감.
SHORTCUT_EXIT_KICK_ANGLE_DEG    = -20.0  # 부호규약은 ctrl_angle과 동일 — 체크무늬 램프의
                                          # END_ANGLE(-20°)과 같은 값으로 시작(실차 재조정 대상)
SHORTCUT_EXIT_KICK_DURATION_S   = 0.5    # 이 시간(초) 동안 고정 조향각 유지 — 20Hz로 환산해 프레임수로 씀

# [2026-08-07] ll을 흰선/노란선으로 분리하는 데 쓰는 커넥티드 컴포넌트 투표 기준
# (DL_CENTER_MODE='ll' 전용, DLSlideWindow._split_ll_by_yellow() 참고). ll 픽셀 자체는
#   흰/노랑 구분이 없으므로(위 YELLOW_LOWER/UPPER 주석 참고), ll_mask의 커넥티드
#   컴포넌트(=점선 한 조각/실선 한 덩어리) 단위로 그 안에 YELLOW_LOWER/UPPER 기준 노란
#   픽셀이 얼마나 겹치는지를 보고 "덩어리 전체"를 노란선/흰선 중 하나로 확정한다(픽셀
#   단위로 빼는 것보다 dash 가장자리가 깔끔하게 갈린다). DL_LL_YELLOW_MIN_AREA 미만인
#   자잘한 컴포넌트(반사/노이즈)는 투표 자체를 생략하고 흰선 쪽에 그대로 둔다(어차피
#   이후 DL_LL_SIDE_MIN_PIXELS/CCA에서 걸러짐). 실차 미검증 초기값 — 흰선 일부가
#   노랗게(역광/그림자) 물들어 오분류되면 DL_LL_YELLOW_VOTE_RATIO를 올릴 것, 노란
#   점선이 흰선으로 새면 내릴 것.
DL_LL_YELLOW_VOTE_RATIO = 0.35
DL_LL_YELLOW_MIN_AREA = 10

FPS_LOG_PERIOD_SEC = 5.0   # dl_lane.py 워커 스레드 FPS/provider 로그 주기(s)


# #############################################################
# 4. 조향 컨트롤러 (Pure Pursuit)
# #############################################################
#   track_drive.py의 _lane_steer()가 self.lane_path를 받아 controller/pure_pursuit.py의
#   PurePursuitController.control(path, vehicle_xy)로 조향각(도)을 계산한다.
# [2026-08-14] LQR 컨트롤러(controller/lqr.py)와 그 사이를 고르던 STEERING_CONTROLLER
#   스위치를 코드베이스에서 완전히 제거했다 — 실차 미검증 상태로 한 번도 켜본 적 없이
#   pure_pursuit만 계속 써온 죽은 분기라 유지보수 부담만 있었다. 과거 LQR 설계 배경/
#   튜닝값 기록은 README §0.5, §6.7, §7에 남아있다.

# ── Pure Pursuit 튜닝값 (controller/pure_pursuit.py PurePursuitController) ──
#   전부 실차 미검증 튜닝값. 각 값의 설계 배경은 pure_pursuit.py __init__ 상단
#   주석 참고 — 여기는 "현재 적용값"만 모아둔다.
# [2026-08-17h] 아래 14개 PP_* 값 전부 `pp_tune_gridsearch.py --speeds 15.0 10.0 --samples 400
#   --seed 0`(SPEED_NORMAL=15.0/SPEED_CORNER_MIN=10.0 재증속 후 재실행, §0.5.11)의 speed=15.0
#   best_params로 일괄 교체(요청 반영: "그리드서치 돌려서 파라미터가 적당한지 판단"). 직진은
#   baseline도 이미 최적(cte_rms 0.6cm, 직진태그 100%)이었지만, 90도커브/S자커브는
#   baseline cte_rms 7~9cm → best 1~1.6cm로 개선됨(score 24.37→3.53) — 특히 PP_WHEELBASE_PX가
#   25.0→49.64로 거의 2배가 되면서 조향 게인 부족(atan(curvature*wheelbase_px))이 커브
#   추종 오차의 가장 큰 원인이었던 것으로 나타남. 이제 이 14개 값은 더 이상 "PP_LOOKAHEAD_MAX_PX
#   = BASE+GAIN*SPEED_NORMAL" 같은 개별 공식 관계를 만족하지 않는다 — 그리드서치가 14개를
#   각각 독립적으로 샘플링한 조합이기 때문(§0.5.6~§0.5.11이 지켜온 수식 관례는 여기서부터 끊김,
#   SPEED_NORMAL을 또 바꿀 땐 그 수식이 아니라 이 그리드서치를 다시 돌릴 것).
#   ★★★ 실차 완전 미검증 ★★★ — pp_tune_gridsearch.py 자체가 화이트박스 합성 시뮬레이션(노이즈
#   1.5px 가정, 트랙 곡률 반경 1.2~1.3m 가정 등 전부 설계값)이고, PP_WHEELBASE_PX는 과거
#   저속(SPEED_NORMAL=3.0)에서 이보다 낮은 값(25.0)이 "진동 감소"로 실차 검증된 이력이 있다
#   (아래 개별 주석 참고) — 이번 상향이 새 속도(15.0)에서도 진동을 안 키우는지 실차에서
#   반드시 먼저 확인할 것. 문제 생기면 이 커밋 이전 값(BASE=90/GAIN=4/MAX=150/WHEELBASE=25/
#   ALPHA=0.8/MIN_LOOKAHEAD=90/DEADZONE=6/CURV_GAIN=100/LOOKAHEAD_MIN=40/EPS=0.0035/
#   CONFIRM=5/STRAIGHT_DEADZONE=20/STRAIGHT_ALPHA=0.4/BIAS_EMA=0.15)으로 되돌릴 것.
PP_LOOKAHEAD_BASE_PX = 65.26       # lookahead 하한(직진/저속 기준값) — 90.0→65.26(그리드서치)
PP_LOOKAHEAD_SPEED_GAIN = 3.305    # 속도가 오를수록 lookahead를 늘리는 게인 — 4.0→3.305(그리드서치)
# [2026-08-07] 150 → 190. speed_lookahead_px = BASE + GAIN*speed 공식이 SPEED_NORMAL=5
#   기준(90+4*5=110)으로 설계됐는데(pure_pursuit.py __init__ 주석), SPEED_NORMAL이 이후
#   25까지 오르면서(config.py 상단 SPEED_NORMAL 주석) 이론상 필요한 lookahead(90+4*25=190)가
#   구 상한(150)에 막혀 speed>=15부터는 lookahead가 더 안 늘어났다. 실차에서 "속도 5는
#   진동이 없는데 20으로 올리니 진동이 심해진다"는 증상으로 재현됨 — Pure Pursuit은 lookahead가
#   짧을수록 curvature=2*sin(alpha)/ld 공식에서 같은 픽셀오차도 더 크게 증폭되므로(§0.5.2
#   README), 속도만 오르고 lookahead가 그만큼 못 늘어나면 고속에서 과민 반응→진동이 커진다.
#   190은 SPEED_NORMAL=25를 그대로 대입한 값 — 실차 재검증 필요. 그래도 진동이 남으면
#   PP_ALPHA(현재 0.5)를 낮춰 조향각 저역통과를 더 강하게 거는 쪽을 다음으로 볼 것.
# [2026-08-17f] SPEED_NORMAL 3.0→10.0 증속에 맞춰 같은 공식으로 재계산: BASE(90) + GAIN(4)*10 = 130.
#   190(구 SPEED_NORMAL=25 기준)을 그대로 둬도 130<190이라 당장 클리핑되진 않지만, §0.5.6이 정한 관례
#   (상한 = BASE+GAIN*현재 SPEED_NORMAL)를 그대로 따름 — 실차 재검증 필요.
# [2026-08-17g] SPEED_NORMAL 10.0→15.0 증속에 맞춰 같은 공식으로 재계산: BASE(90) + GAIN(4)*15 = 150.
# [2026-08-17h] 위 PP_LOOKAHEAD_BASE_PX/SPEED_GAIN 주석 참고 — 이제부터는 공식이 아니라
#   그리드서치 독립 샘플값이라 90+4*15와 무관하게 180.7로 교체.
PP_LOOKAHEAD_MAX_PX = 180.7        # lookahead 상한 — 150.0→180.7(그리드서치)
# [2026-08-24] B1(라바콘) 전용 lookahead 상한 — track_drive.py _lane_steer()가
# self.phase==Phase.LAVACON일 때만 이 값으로 스위칭(그 외엔 PP_LOOKAHEAD_MAX_PX).
PP_LOOKAHEAD_MAX_PX_LAVACON = 140.0
PP_LOOKAHEAD_CURVATURE_GAIN = 224.8  # 직전 프레임 curvature가 클수록(코너) lookahead를 줄이는 게인 — 100.0→224.8(그리드서치)
PP_LOOKAHEAD_MIN_PX = 62.61        # 코너에서 lookahead가 줄어들 수 있는 하한 — 40.0→62.61(그리드서치)

# [2026-08-21, 요청 반영] da 근접 컷(obstacle_cut_active, B2/B3 회피) 진입 순간
#   PP_LOOKAHEAD_CURVATURE_GAIN을 잠깐 이 값으로 올렸다가 PP_CURVATURE_BOOST_SEC 뒤
#   원래 값(프리셋 적용 후의 PP_LOOKAHEAD_CURVATURE_GAIN, 예: speed15=120)으로 복귀시킨다
#   (_update_obstacle_cut_hold() 진입 엣지 감지 + _lane_steer() 적용, track_drive.py 참고).
#   회피 진입 순간 코너 감쇠를 세게 걸어 lookahead를 짧게 당겨서(=조향을 더 촘촘/민감하게)
#   장애물 근접 구간에서의 반응성을 순간적으로 높이려는 의도 — 실차 미검증, speed15 기준
#   요청값 180 그대로 사용(프리셋 무관하게 항상 이 값으로 튐).
PP_CURVATURE_BOOST_GAIN = 180.0
PP_CURVATURE_BOOST_SEC  = 1.0

# [2026-08-19] speed_lookahead_px = BASE + GAIN*(speed - ANCHOR)의 ANCHOR(요청 반영) —
#   기존엔 ANCHOR가 암묵적으로 0(=BASE가 "speed=0일 때" 값)이라, 실제로는 절대 안 나오는
#   speed=0 지점을 기준으로 삼다 보니 config.py만 봐서는 "지금 주행속도에서 lookahead가
#   대충 얼마인지"가 바로 안 보이는 문제가 있었다(예: speed15 프리셋 BASE=110.38인데 실제
#   SPEED_NORMAL=12에서 나오는 값은 146.5 — 숫자가 전혀 안 겹쳐서 헷갈림). ANCHOR를
#   SPEED_NORMAL로 두면 BASE 자체가 "주행속도에서의 lookahead값"이 되어 더 직관적이다.
#   기본값 0.0 = 기존과 완전히 동일한 동작(하위호환) — 이 전역 기본값과 speed15를 뺀
#   나머지 프리셋(None 포함)은 전부 이 0.0을 그대로 쓰므로 안 건드림. speed15 프리셋만
#   아래서 SPEED_NORMAL(12.0)로 덮어쓰고, 그에 맞춰 BASE_PX도 같이 재계산했다(아래
#   PP_TUNE_PRESETS['speed15'] 주석 참고) — 다른 프리셋으로 바꿔 쓰려면 그 프리셋도
#   같은 방식으로 ANCHOR+BASE_PX를 맞춰야 한다(아직 안 함).
PP_LOOKAHEAD_SPEED_ANCHOR = 0.0

# [2026-08-19] lookahead_px 자체에 거는 프레임간 저역통과(요청 반영, pure_pursuit.py
#   lookahead_alpha 주석 참고) — curvature_damp가 한 프레임 만에 lookahead를 base 근처에서
#   MIN_PX까지 확 끌어내리면, 좁아진 lookahead가 같은 dx도 더 큰 curvature로 증폭시켜
#   (curvature=2*sin(alpha)/ld) 급조향 스파이크를 키우거나 복귀 중 재흔들림을 만들 수
#   있다는 관찰에서 추가. 1.0(기본값)=필터 없음, 기존 동작과 완전히 동일 — 프리셋에서
#   명시적으로 낮춰야 켜진다. 실차 미검증.
PP_LOOKAHEAD_ALPHA = 1.0

# [2026-08-06] "곡률→조향각" 게인(pure_pursuit.py의 steer_deg = atan(curvature*wheelbase_px)).
#   원래 80.0은 "실제 축거리 대신 쓰는" 임의 튜닝값이었다(pure_pursuit.py 상단 주석: "카메라
#   픽셀→미터 변환이 아직 실측 전이라 wheelbase_px를 대신 쓴다, PIXELS_PER_METER가 실측되면
#   실제 축거리(m)*PIXELS_PER_METER로 대체 가능"). LANE_DETECTOR_BACKEND='dl'(기본값) +
#   DL_USE_BEV=True(기본값)에서는 self.lane_path가 정확히 DL_PIXELS_PER_METER(=200px/m,
#   BEV 캔버스의 정의상 스케일)로 만들어진 픽셀좌표이므로, 이제 실측 `WHEELBASE_M`
#   (0.335m, §6.7 — 옛 이름 LQR_WHEELBASE_M)을 그대로 곱해 물리 기반 값으로 대체할 수
#   있다: 0.335 * 200 = 67.0.
#   ★ 실차 재검증 필요 ★ — 80.0은 그 자체로 실차에서 "이 정도 조향 반응이 적당하더라"고
#   경험적으로 맞춰졌을 가능성이 있어(다른 근사 오차를 상쇄했을 수도 있음), 67.0로 바꾸면
#   같은 curvature에도 조향각이 더 작게(atan 인자가 작아짐) 나와 코너링이 더 완만해질 수
#   있다 — 너무 밋밋하게 느껴지면 이 값을 다시 올릴 것(단, 그때는 "튜닝값"임을 주석에 남길 것).
PP_WHEELBASE_PX = 49.64            # [2026-08-17h] 25.0→49.64(그리드서치) — 커브 추종 오차(cte_rms
                                    #   7~9cm→1~1.6cm)를 가장 크게 줄인 값. ★주의★ 아래 [2026-08-13]
                                    #   이력대로 25.0은 SPEED_NORMAL=3.0 저속에서 "진동 감소" 목적으로
                                    #   실차 검증된 값이었다 — 이번 상향이 새 속도에서 진동을 다시
                                    #   키우는지 최우선으로 실차 확인할 것(위 PP_* 블록 상단 주석 참고).
                                    # [2026-08-13] 67.0 → 40.0 → 25.0(요청 반영, 튜닝값 — 조향을 더
                                    #   줄이는 방향). atan(curvature*wheelbase_px) 공식상 값이 작을수록
                                    #   같은 curvature에도 조향각이 더 작게 나옴(반응 약화) — 실차 재검증 필요.
                                    # 원래 = WHEELBASE_M(0.335) * DL_PIXELS_PER_METER(200) 실측 기반 계산값(67.0)

# [2026-08-19] 조향각이 클수록 wheelbase_px를 비례해서 키우는 "부스트"(요청 반영).
#   atan(curvature*wheelbase_px) 공식상 wheelbase_px를 키우면 같은 curvature에도 조향각이
#   atan 포화 쪽으로 더 밀린다 — 1차 steer_deg의 절댓값에 그대로 비례해서(GAIN_PER_DEG)
#   wheelbase_px를 키운다(MAX_SCALE로 상한). [2026-08-19 재수정] 원래는 문턱각
#   (PP_WHEELBASE_BOOST_ANGLE_TH_DEG)을 넘어야만 부스트가 시작되는 계단형이었는데,
#   "조향각이 작을 땐 증폭도 작고 커지면 증폭도 커지게"(요청 반영) — 즉 문턱 없이 각도 0부터
#   연속적으로 커지는 형태로 바꾸면서 문턱각 파라미터 자체를 없앴다. GAIN_PER_DEG도 그에 맞춰
#   재조정했다 — 문턱이 있을 때는 "문턱을 넘은 초과분"에만 곱해졌지만 이제는 "전체 각도"에
#   곱해지므로, 예전과 같은 숫자를 쓰면 훨씬 작은 각도에서 MAX_SCALE에 도달해버린다(예:
#   GAIN=0.15면 MAX_SCALE=1.5 도달 각도가 (1.5-1)/0.15≈3.3°로, 사실상 상시 최대 부스트가
#   걸리는 것과 다름없어져 "미미할 땐 작게"라는 요청과 어긋난다). 0.03으로 낮춰
#   (1.5-1)/0.03≈16.7°에서 MAX_SCALE에 닿도록 재설정 — 순전히 추정치, 실차에서 체감보고
#   재조정할 것.
#   "speed15 프리셋일 때만 적용" 요청대로(2026-08-19 — 처음엔 런타임 speed==15로, 그 다음엔
#   별도 계산식으로 잘못/우회적으로 구현했었음) 셋 다 PP_WHEELBASE_PX처럼 "여기 top-level엔
#   기본값(비활성)만 두고, 아래 PP_TUNE_PRESETS['speed15']가 실제 값으로 덮어쓴다"는 이 파일의
#   기존 프리셋 관례를 그대로 따른다. PP_TUNE_ACTIVE_PRESET이 'speed15'가 아니거나 None이면
#   아래 ENABLE 기본값(False)이 그대로 남아 부스트가 꺼진다.
#   ★ 실차 완전 미검증 ★ — 부스트가 과하면 급코너에서 오버스티어처럼 느껴질 수 있으니
#   실차에서는 MAX_SCALE부터 낮춰가며 확인할 것.
PP_WHEELBASE_BOOST_ENABLE = False        # 기본값(비활성) — speed15 프리셋만 아래서 True로 덮어씀
PP_WHEELBASE_BOOST_GAIN_PER_DEG = 0.03   # 조향각 1도당 wheelbase_px를 이 비율만큼 확대(문턱 없음, 각도 0부터 연속 적용)
PP_WHEELBASE_BOOST_MAX_SCALE = 1.5       # wheelbase_px 배율 상한(최대 +50%) — 폭주 방지

# [2026-08-12] 직진 구간에서도 계속 진동("와리가리")한다는 보고 대응 — §0.5.3 "알려진
#   한계"에서 이미 "PP_ALPHA를 낮춰 조향각 저역통과를 더 강하게 거는 쪽을 다음으로 볼 것"
#   이라고 못박아뒀던 그 다음 단계. 0.5 → 0.35로 낮춰 프레임간 저역통과를 더 세게 건다
#   (README §0.5.7). 실차 미검증 — 너무 낮추면 실제 코너 진입 반응이 느려지니 같이 볼 것.
# [2026-08-13] 0.35 → 0.60(요청 반영, 저속(SPEED_NORMAL=3.0) 재튜닝과 함께 필터를 완화해
#   반응성↑) 했었으나, [2026-08-17] "천천히 가도 직진에서 계속 구불구불하다"는 재보고로
#   0.35로 원복 — 8/13 변경이 8/12에 검증 시도했던 진동억제 레버를 그대로 되돌린 것이었다는
#   게 뒤늦게 드러났다(원인 분석 대화 참고). 저속(SPEED_NORMAL=3.0 그대로)일수록
#   speed_lookahead_px가 하한(90px)에 가까워 픽셀노이즈 증폭이 가장 심한 구간이라, 필터를
#   더 강하게 걸어야 하는 상황에서 반대로 완화했던 것이 원인일 가능성이 높다.
# [2026-08-17c] 한때 아래 PP_STRAIGHT_ALPHA가 직진 확정 중 필터를 따로 담당하도록
#   분리했던 적이 있다("직진(A)/코너+S자(B) 2상태 분기") — 그 동안 이 값은 "코너/S자
#   등 직진이 아닐 때" 전용으로 좁혀져 반응성 우선으로 0.35→0.8까지 올라갔었다.
# [2026-08-17h] 0.8→0.5244(그리드서치, PP_WHEELBASE_PX 상향과 함께 재탐색된 조합).
# [2026-08-19] 요청 반영 — "직진(A)/코너+S자(B) 2상태 분기" 자체(PP_STRAIGHT_*, README
#   §0.5.9)를 완전히 제거했다. 이제 이 값이 모든 상황(직진 포함)에서 유일하게 쓰이는
#   저역통과 필터다 — 위 [2026-08-17c] 이전(2026-08-12~17 사이)의 "직진 진동 대응"
#   문제로 되돌아갈 수 있으니, 직진에서 다시 "와리가리"가 보이면 이 값을 낮추는 쪽으로
#   대응할 것(위 [2026-08-12]/[2026-08-13] 이력 참고).
PP_ALPHA = 0.5244                  # 프레임간 조향각 저역통과 필터(1=필터없음, 0=반응없음)
PP_LD_FLOOR_PX = 86.95        # curvature 분모(ld) 바닥값 — 노이즈 증폭 방지용. lookahead_px 자체의 하한인 PP_LOOKAHEAD_MIN_PX와는 다른 값(개명 전 이름: PP_MIN_LOOKAHEAD_PX)이니 헷갈리지 말 것. 90.0→86.95(그리드서치)
# [2026-08-12] 6.0 → 15.0. 직진 진동 대응 세 번째 레버 — 원래 값이 중앙 부근 잔떨림을
#   죽이기엔 너무 작아서(픽셀 몇 개짜리 노이즈도 그대로 통과) 직진에서도 매 프레임 미세한
#   조향이 나갔던 것으로 추정. LANE_DEADZONE(구 PID 전용, 40px)보다는 여전히 훨씬 작게
#   유지 — Pure Pursuit 목표점은 이미 lookahead 앞 실제 경로점이라 그만큼 크게 죽이면
#   완만한 커브 진입까지 무시하게 된다(pure_pursuit.py 상단 주석 참고). 실차 미검증.
# [2026-08-13] 15.0 → 5.0(요청 반영, PP_WHEELBASE_PX를 67→40으로 줄여 조향 반응 자체가
#   약해진 만큼 데드존도 같이 줄임 — 노란선 흔들림 대응) 했었으나, [2026-08-17] PP_ALPHA와
#   같은 이유로 12.0으로 절충 원복.
# [2026-08-17c] 한때 PP_ALPHA와 같은 이유로 "코너/S자(비직진) 상태 전용"으로 좁혀져
#   12.0→6.0으로 낮아졌던 적이 있다(직진 잡음 억제는 PP_STRAIGHT_DEADZONE_PX가 담당).
# [2026-08-17h] 6.0→4.445(재증속 후 재실행한 그리드서치, pp_tune_gridsearch.py).
# [2026-08-19] 요청 반영 — 위 PP_ALPHA와 동일하게 "직진(A)/코너+S자(B) 2상태 분기"를
#   완전히 제거했다. 이제 이 값이 모든 상황(직진 포함)에서 유일하게 쓰이는 데드존이다.
PP_DX_DEADZONE_PX = 4.445          # 이 이하 픽셀오차는 0으로 죽여 중앙 부근 잔떨림 제거

# [2026-08-19] 명시적 "직진 모드"(README §0.5.9, 2026-08-17 도입) 제거(요청 반영: "직진모드,
#   커브모드는 빼자 — 파라미터는 모두 커브대응상태만 남기고 날려줘"). pure_pursuit이
#   probe_curvature/dx 편향으로 "직진 확정" 상태를 판정해 그 동안만 더 넓은 데드존
#   (PP_STRAIGHT_DEADZONE_PX)과 다른 필터(PP_STRAIGHT_ALPHA)를 쓰던 2상태 분기였는데,
#   이제 위 PP_ALPHA/PP_DX_DEADZONE_PX 하나로 통일했다(코너/S자 대응 값이 상시 적용) —
#   PP_STRAIGHT_CURVATURE_EPS/CONFIRM_FRAMES/DEADZONE_PX/ALPHA/BIAS_EMA_ALPHA 5개
#   상수와 pure_pursuit.py의 판정 로직(is_straight, _straight_frames, _dx_bias_ema)을
#   전부 삭제했다. 되돌리려면 이 커밋 이전(2026-08-17h) 이력을 참고할 것.

# ── 차량 물리 상수 ──
# [2026-08-14] 옛 이름 LQR_WHEELBASE_M → WHEELBASE_M. LQR 컨트롤러 제거로 "LQR 전용"이
#   아니라 EncoderPoseEstimator(localization/pose_estimator.py)가 쓰는 일반 차량 상수임을
#   반영한 이름 변경 — 값 자체는 그대로(실측 유지).
WHEELBASE_M = 0.335         # 실측값(2026-08-06, 줄자로 앞바퀴-뒷바퀴 축간거리 실측 — LQR 브랜치에서
                             #   이식). planner/hybrid_astar.py의 wheelbase 기본값(같은 차량이므로
                             #   반드시 같은 값)과 일치시킬 것 — 재실측 시 둘 다 갱신.

# [2026-08-19] track_drive.py._corner_radius_speed_scale()가 코너 감속용 회전반경을 역산할 때
#   쓰는 "물리 기반" 축거리 — 여태 PP_WHEELBASE_PX(pure_pursuit.py의 "곡률→조향각" 게인,
#   실측 축거리가 아니라 자유롭게 재튜닝되는 값)를 그대로 재사용하고 있었는데, 최근 그 값을
#   조향 반응성 목적으로 49.64→16까지 낮추면서 이 반경 계산식(radius=wheelbase_px/tan(steer))
#   분자가 같이 줄어들어 살짝만 꺾여도(corner_signal≈10도) 반경이 확 작게 나와 코너 감속이
#   상시로 걸리는 부작용이 났다(요청 반영으로 분리 — "높은 조향에서 감속이 너무 세게
#   들어간다"). PP_WHEELBASE_PX 재튜닝과 코너 감속 민감도가 서로 안 엮이도록, 여기서는 그
#   대신 실측 WHEELBASE_M*DL_PIXELS_PER_METER(물리 기반, PP_WHEELBASE_PX 도입 전 원래
#   pure_pursuit.py가 쓰던 계산과 동일 — config.py PP_WHEELBASE_PX 주석 참고)를 쓴다.
#   실차 미검증 — 이 값 자체를 낮추면 코너 감속이 다시 더 민감해진다.
CORNER_RADIUS_WHEELBASE_PX = WHEELBASE_M * DL_PIXELS_PER_METER  # = 67.0

# ── VESC 실측 속도 연동 (2026-08-06, LQR 브랜치의 ROS1 연동 작업에서 이식) ──
#   이 로봇엔 별도 엔코더 토픽이 없고, VESC 드라이버(ROS1, vesc_driver)가 /sensors/core
#   (vesc_msgs/VescStateStamped)로 모터 홀센서 기반 회전속도를 발행한다. vesc_msgs가 이
#   ROS2 워크스페이스엔 안 빌드돼 있어(2026-08-06 실차 확인) ros1_bridge가 커스텀 메시지를
#   그대로 못 넘기므로, ROS1쪽에 작은 변환 노드(launch/vesc_speed_bridge.py — 이 워크스페이스
#   바깥 noetic_ws에 별도 배치해서 실행, 파일 상단 주석 참고)를 하나 더 띄워서 state.speed
#   (ERPM) 값 하나만 std_msgs/Float32로 '/vesc_speed_erpm'에 다시 뿌리게 하고,
#   track_drive.py의 cb_vesc()가 그 표준 메시지를 구독해 self.v_mps(m/s)로 변환한다.
VESC_SPEED_TO_ERPM_GAIN = 4614.0  # VESC 드라이버 vesc.yaml의 speed_to_erpm_gain 값 그대로(실차 확인,
                                   #   2026-08-06). 실속도(m/s) = state.speed(ERPM) / 이 값.
VESC_STALE_SEC = 0.5        # 마지막 /vesc_speed_erpm 수신 후 이 시간(s)이 지나면 vesc_debug 창에서
                             #   "끊김"으로 표시(20Hz 기준 약 10틱).
VESC_MIN_SPEED_MPS = 0.05    # v_mps가 이 미만(정지/거의정지, 혹은 vesc_speed_bridge 노드 미실행으로
                             #   0.0 고정)이면 "VESC 실측값을 못 믿는다"고 보고 _speed_for_lookahead()
                             #   (2026-08-06, pure_pursuit용)가 v_mps 대신 self._prev_speed(명령속도)로
                             #   폴백한다 — track_drive.py 참고. [2026-08-14] LQR 컨트롤러 제거 전엔
                             #   self.lqr.set_speed_mps() 갱신도 이 값으로 건너뛰었으나(v≈0에서 상태공간
                             #   게인이 퇴화하는 것을 피하기 위함) 그 용도는 LQR과 함께 사라졌다.

IMU_STALE_SEC = 0.5          # 마지막 /imu 수신 후 이 시간(s)이 지나면 "죽었다"고 본다(VESC_STALE_SEC과
                             #   동일 철학). imu_yaw 자체는 값이 없어도 초기값 0.0을 계속 들고 있어서
                             #   느낌만으로는 "살아있는지 그냥 직진 중인지" 구분이 안 되므로,
                             #   cb_imu()가 갱신하는 _imu_t 타임스탬프로 따로 생존을 체크한다 —
                             #   track_drive.py._imu_curvature_px() 전용(PP curvature damping 보강).
                             #   lap 카운트(_update_lap)는 이 가드 없이 imu_yaw를 그대로 쓰므로
                             #   IMU가 죽으면 [LAP] 누적이 0에 멈추는 것으로 바로 티가 난다(의도된 동작).

IMU_YAW_RATE_EMA_ALPHA = 0.3  # [2026-08-06] _imu_curvature_px() 전용 저역통과(1=필터없음, 0=반응없음,
                             #   PP_ALPHA/CORNER_SIGN_EMA_ALPHA와 동일한 관례). probe_curvature는
                             #   경로 위 여러 점을 누적한 값인데 imu_yaw_rate는 자이로 순간값을
                             #   그대로 썼었다 — curvature damping이 두 값 중 "더 큰 쪽"을 그대로
                             #   쓰는 구조라(controller/pure_pursuit.py control() 참고), 스무딩 없는
                             #   쪽(IMU)이 노이즈 스파이크 한 프레임만으로도 감쇠를 확 눌러버릴 수
                             #   있었다. 0.3은 CORNER_SIGN_EMA_ALPHA(0.15)보다 약간 반응성을 준
                             #   추정치 — 실차 미검증, VESC 복구 후 steer_debug의 IMU curvature
                             #   값이 진동하는지 보며 조정할 것.

# [2026-08-11] 차선인식 "새 결과 없음" 생존 체크 — VESC_STALE_SEC/IMU_STALE_SEC과 동일 철학.
#   DL 백엔드(perception/dl_lane.py)는 20Hz control_loop()과 분리된 별도 스레드에서 자기
#   페이스껏 추론하고, perc_lane()은 매 틱 그 시점의 최신 결과를 논블로킹으로 재사용한다
#   (dl_lane.py 모듈 상단 주석 — 추론 지연을 조향 발행 주기에 안 실으려는 의도된 설계).
#   문제는 "재사용"과 "완전히 죽어서 안 갱신됨"을 구분할 방법이 없었다는 것 — 추론
#   스레드가 예외로 죽거나 카메라가 끊겨도 self.lane_path/offset은 마지막 값에 조용히
#   얼어붙을 뿐 에러가 안 난다(IMU/VESC와 같은 "센서 죽음=조용한 고정값" 패턴이 비전
#   파이프라인에도 그대로 적용됨). DLLaneDetector.result_seq(추론 1회 끝날 때마다
#   증가하는 카운터)가 몇 틱 동안 그대로면 "새 데이터 없음" 상태 지속시간을
#   perc_lane()이 재고, 그게 이 값(초) 이상이면 lane_stale=True로 판정한다.
#   [2026-08-11 후속] 처음엔 VESC_STALE_SEC/IMU_STALE_SEC과 같은 0.5였는데, 그 둘은
#   원래 초당 수십~백 회 들어오는 고빈도 토픽 콜백이라 0.5초 무응답이면 거의 확실히
#   죽은 것이지만, DL 추론(TwinLiteNet 세그멘테이션)은 프레임 하나 처리가 정상적으로도
#   수십~수백ms 걸리는 무거운 연산이고, TensorRT provider는 최초 실행 시 엔진 빌드에
#   수십초~수분이 걸릴 수 있다는 경고까지 dl_lane.py에 있다(TwinLiteNetEngine.__init__
#   TensorRT 캐시 주석 참고) — 즉 "정상 상황에서도 순간적으로 훨씬 오래 걸리는 구간"이
#   VESC/IMU보다 훨씬 흔하다. 실측 FPS(FPS_LOG_PERIOD_SEC 로그) 없이 0.5를 그대로 갖다
#   쓴 건 근거가 약해서, 일반적인 세그멘테이션 프레임타임의 10배 이상 여유를 두는 2.0으로
#   올렸다 — 정상 지연을 고장으로 오판할 위험을 크게 줄이면서도, 실제 고장 시 2초 안에는
#   감지되도록 하는 절충점. 실차에서 FPS 로그를 실제로 보고 나면 그 주기의 몇 배 정도로
#   더 정확하게(너무 크면 진짜 고장 감지가 늦어짐) 재조정할 것.
LANE_STALE_SEC = 2.0

# [2026-08-17m] LANE_STALE_SEC(위)은 "추론 워커가 죽어서 결과 자체가 안 바뀌는" 경우만
#   잡는다 — 워커는 매 틱 새 결과를 내지만 그 결과가 계속 무효(lane_valid=False)인 경우엔
#   이 게이트를 안 거쳐서 감속이 전혀 안 걸렸다(실차 재현: 커브 진입에서 da 밴드 핏이
#   연속 실패하는 동안 무감속으로 트랙 이탈, track_drive.py self._lane_invalid_streak
#   주석 참고). lane_valid가 연속으로 이 프레임 수 이상 False면 SPEED_LANE_STALE과 동일한
#   캡을 건다 — 20Hz 기준 0.5초, LANE_STALE_SEC(2.0초)보다 훨씬 짧게 잡아 "워커 고장"보다
#   훨씬 흔할 "이번 프레임을 못 믿겠다" 상황에 더 빨리 반응하게 한다. 실차 미검증 첫 추정치.
LANE_UNSTABLE_FRAMES = 10


# #############################################################
# 5. 디버깅 ON/OFF
# #############################################################
DEBUG_LOG    = True   # 0.5초마다 CLI에 [LAP]/[SENS]/[LANE]/[TRIG]/[SIG]/[LAVA-ROI] 로그
DEBUG_PERIOD = 0.5     # 위 로그 주기(s)

# [2026-08-22] 디버그 창들이 전부 화면 기본 위치(보통 좌상단 근처)에 겹쳐서 뜬다는 요청으로
#   추가 — cv2.moveWindow로 창이 처음 뜰 때 한 번만 이 좌표로 옮긴다(각 창 imshow 호출부
#   참고: track_drive.py _draw_lavacon_bev/_debug_viz_obstacle_cut, perception/dl_lane.py
#   show_debug_windows). 화면 해상도에 안 맞으면 이 값만 바꾸면 됨 — (x, y)는 창의
#   좌상단 모서리가 놓일 스크린 픽셀 좌표.
# [2026-08-23a~e] 여러 차례 그리드를 다시 짰는데도 "또 가려져서 뜬다"는 요청이 반복됐다 —
#   원인은 좌표가 아니라 전제였다: 그동안 화면이 가로 ~1900px·세로 ~1150px는 될 거라 가정하고
#   배치를 짰는데, 실차 디버그 화면에서 `xrandr --current`로 실측하니 실제 해상도는
#   1280x800(HDMI-0)뿐이었다 — 그리드 전체가 항상 화면 밖으로 흘러넘쳐서 어떻게 좌표를
#   바꿔도 겹칠 수밖에 없었다.
# [2026-08-23f] 요청 반영("디버깅창 또 가려져서 뜬다 위치조절") — 실측 해상도(1280x800)
#   기준으로 그리드를 처음부터 다시 짬. (처음에 dl_lane/obstacle_cut_debug을 표시 직전에
#   축소하는 시도를 했다가 [2026-08-23g] "위치만 바꾸라고 시키지도 않은 짓 하지 말라"는
#   지적으로 되돌림 — 크기는 원본 그대로 두고 좌표만 조정한다.)
#   원본 크기(YOLO_신호등 600x450 / dl_lane ~585x650 / obstacle_cut_debug 580x480 /
#   yolo_cone_result 160x120 / lavacon_bev 160x160) 그대로 서로 안 겹치게 2열로 배치:
#     (  0,   0) YOLO_신호등        600x450
#     (600,   0) dl_lane            ~585x650
#     (  0, 450) obstacle_cut_debug 580x480  (세로가 800을 ~130px 넘어갈 수 있음 — 다른
#                                              창과는 안 겹치니 크기는 그대로 둠)
#     (600, 650) yolo_cone_result   160x120
#     (770, 650) lavacon_bev        160x160
#   화면 해상도가 다르면(`DISPLAY=:0 xrandr --current`로 확인) 이 값들만 바꾸면 됨
#   — (x, y)는 창의 좌상단 모서리가 놓일 스크린 픽셀 좌표.
DEBUG_WIN_POS_LIDAR         = (0, 450)        # 'lidar_bev' 창 (track_drive.py), 500x500 — 지금 꺼짐,
                                               #   켜면 obstacle_cut_debug와 같은 자리라 좌표 재검토 필요
DEBUG_WIN_POS_LAVACON       = (770, 650)      # 'lavacon_bev' 창 (track_drive.py), B1 콘 push 시각화, 160x160
DEBUG_WIN_POS_OBSTACLE_CUT  = (0, 450)        # 580x480 원본 그대로 — 화면(1280x800) 밖으로
                                               #   ~130px 넘칠 수 있으나(요청 반영: 크기는 안 건드림)
                                               #   다른 창과는 안 겹침
DEBUG_WIN_POS_DL_LANE       = (600, 0)        # 원본 크기 그대로(~585x650)
DEBUG_WIN_POS_YOLO_SIGNAL_STATE = (0, 0)      # 'YOLO_신호등' 창 (perception/yolo_signal_state.py)
                                               #   원본 640x480, [2026-08-23] 표시만 160x120로 축소했다가
                                               #   [2026-08-23d] 요청 반영으로 600x450으로 다시 키움 —
                                               #   최근 요청으로 키운 값이라 재배치에서도 안 건드림.
DEBUG_WIN_POS_CHECKER_PILLAR    = (770, 650)  # 'checker_pillar_bev' 창 (track_drive.py) — 지금 꺼짐,
                                               #   위 lavacon_bev와 같은 자리 재사용(둘 다 동시에
                                               #   켤 일 생기면 좌표 다시 확인할 것)
                                               #   원본 500x500, [2026-08-23] 표시만 160x160로 축소(요청 반영)
DEBUG_WIN_POS_LEFT_TURN         = (0, 450)    # 'left_turn_debug' 창 (track_drive.py), 480x436 — 지금 꺼짐,
                                               #   켜면 obstacle_cut_debug와 같은 자리라 좌표 재검토 필요
DEBUG_WIN_POS_YOLO_CONE         = (600, 650)  # 'yolo_cone_result' 창 (perception/yolo_cone.py), 160x120

# [2026-08-11] 라바콘 실차 테스트 중엔 라이다 창만 보고 싶다는 요청으로, 아래 DEBUG_VIZ_LIDAR만
#   켜고 나머지는 전부 잠시 끔. 다른 디버그창이 다시 필요하면(예: 차선 인식 디버깅) 개별적으로
#   다시 True로 되돌릴 것 — 서로 독립적인 스위치라 다른 항목엔 영향 없음.
# [2026-08-23] True → False(요청 반영) — 라이다 원시 BEV 대신 아래 DEBUG_VIZ_LAVACON
#   (B1 콘 회피 경로 push 시각화)을 보기로 함.
DEBUG_VIZ_LIDAR    = False  # 라이다 BEV 장애물 감지 디버그 창 (track_drive.py)
# [2026-08-19] ENABLE_BEHAVIOR=False(라바콘/장애물/추월 Behavior 전체 비활성, 순수 차선주행만
#   사용)로 전환하면서, 더 이상 발동되지 않는 라바콘 디버그창 관련 스위치도 함께 끔(요청 반영:
#   "디버깅창도 다른거 다 끄고 ll창만"). 나머지 창들도 아래에서 전부 꺼져 있고 DEBUG_VIZ_DL_LANE만
#   켜져 있다. 라바콘을 다시 켜면(ENABLE_BEHAVIOR=True) 아래도 다시 True로 되돌릴 것.
#   ([2026-08-22] 당시 "3개"였던 라바콘 창은 이후 통합·삭제로 아래 두 개(DEBUG_VIZ_LAVACON/
#   DEBUG_VIZ_LAVACON_SHOW_PATH)만 남음.)
# [2026-08-22] 원래 DEBUG_VIZ_LAVACON('lavacon_bev', 트리거 좌우 클러스터)과
#   DEBUG_VIZ_LAVACON_EMA('lavacon_ema_bev', 박스 클러스터링+좌우 EMA 차선)가 별개 창
#   두 개였는데, 요청 반영으로 'lavacon_bev' 창 하나로 통합했다(track_drive.py
#   _draw_lavacon_bev() 참고) — 스위치도 DEBUG_VIZ_LAVACON 하나면 충분해져 EMA 전용
#   스위치는 삭제.

# [2026-08-22k] False → True(요청 반영) — lavacon_bev 창 자체가 꺼져 있어서 라바콘 트리거
#   ROI(LAT_MAX)/검출 박스(CONE_LAT_LIMIT) 수정 결과가 실제로는 화면에 안 보이고 있었다
#   (2026-08-11 당시 "라바콘 실차 테스트 중엔 라이다 창만" 요청으로 꺼둔 게 그대로 남아있던
#   것 — 위 DEBUG_VIZ_LIDAR 주석 참고). 처음엔 DEBUG_VIZ_LIDAR(B2/B3 전용)와 같이 켰다가,
#   지금은 라바콘 검출 확인이 목적이라 위 DEBUG_VIZ_LIDAR를 다시 False로 끄고 이 창만 남김
#   (요청 반영).
# [2026-08-23, 요청 반영] True → False — B1 박스면적 테스트 창(DEBUG_VIZ_B1_CONE_AREA)만
#   보기 위한 임시 비활성화. 복원 시 True로.
DEBUG_VIZ_LAVACON  = False  # [2026-08-25, 요청 반영] 대회 실주행 준비 — 성능(imshow 오버헤드)
                             #   영향 없게 모든 디버그창 끔. 필요시 다시 True로.
                             # 라바콘 트리거 ROI + push ROI(지금 실제 조향에 쓰이는 좌우 최근접
                             #   콘 검출) 통합 BEV 디버그 창 (track_drive.py _draw_lavacon_bev())
                             #   — 콘 침범 시 경로를 옆으로 미는 push(margin/gain/lat) 표시.
                             # [2026-08-23] False → True(요청 반영) — 라이다 창 대신 B1 콘 push
                             #   시각화를 보기로 함. 이전엔 B3 통과 후 신호등 대기+좌회전
                             #   진입 검증 중이라 B1(라바콘)은 이번 검증 범위 밖.
# [2026-08-22k] DEBUG_VIZ_LAVACON_SHOW_PATH 삭제(요청 반영) — 박스 스택 페어링 조향
#   (_handle_lavacon()의 폴백 분기, 2026-08-20부터 이미 안 불림)이 쓰던 노란 경로선(path_m)
#   시각화를 _draw_lavacon_bev()에서 지우면서 이 스위치가 아무것도 안 하게 됐다.
DEBUG_PLANNER      = False  # Hybrid A* OccupancyGrid 디버그 창 (track_drive.py, USE_HYBRID_ASTAR_FOR_B3=True일 때만 의미있음)
DEBUG_VIZ_STEER    = False  # 조향 컨트롤러(직전값유지/현재값반영) 한글 디버그 창 (track_drive.py)
DEBUG_VIZ_VESC     = False  # VESC 실측속도(/vesc_speed_erpm) 연동 상태(수신중/끊김/미수신) +
                             #   좌회전 진행거리(TURN_DIST_M류) 디버그 창
                             #   (track_drive.py, 2026-08-06 LQR 브랜치에서 이식)
# [2026-08-18] IMU(/imu) 연동이 실제로 살아있는지 + 지금 imu_yaw 값이 얼마인지를 보여주는
#   창. [2026-08-20] 좌회전(_do_left_turn())을 실측거리 기반으로 되돌리며 좌회전 진행상황
#   표시는 DEBUG_VIZ_VESC 창으로 옮겼다 — 이제 좌회전이 IMU를 참조하지 않으므로.
DEBUG_VIZ_IMU      = False  # IMU(/imu) 연동 상태 + 현재 yaw값 디버그 창
                             # [2026-08-22] 요청 반영으로 끔 — 아래 "차선인식/좌회전 통합
                             #   창만 남기고 나머지 다 끄기" 일괄 정리, 필요하면 다시 True로.

# [2026-08-23, 요청 반영] True → False — B1 박스면적 테스트 창만 보기 위한 임시 비활성화.
#   복원 시 True로.
DEBUG_VIZ_DL_LANE    = False  # [2026-08-25, 요청 반영] 대회 실주행 준비 — 모든 디버그창 끔.
                               # 차선 — 기본 백엔드('dl') 디버그 창 (perception/dl_lane.py),
                               # da(주행가능영역) 오버레이+경로가 찍히는 'dl_lane' 창
                               # [2026-08-23] False → True(요청 반영) — B1/B2/B3(고정·이동장애물
                               #   회피) 검증 시작하며 da 창 다시 켬.
# [2026-08-10] offset 스파크라인이 몇 프레임을 보여줄지 — [2026-08-11] 원래 별도
#   'dl_lane_params' 창 하단에 붙었으나, 그 창의 파라미터 텍스트 목록(대부분 config
#   고정값)을 지우면서 스파크라인만 'dl_lane' 창 맨 아래로 옮겼다(DEBUG_VIZ_DL_LANE
#   하나로 통합 제어). 주행 성능에 영향 없는 순수 디버그 표시 설정이라 driving 튜닝값
#   묶음이 아니라 여기 DEBUG_VIZ_* 옆에 둔다. 너무 크면(예: 수백) 그래프가 납작해져
#   최근 흔들림이 잘 안 보이고, 너무 작으면 추세를 못 봄 — 90(대략 3~6초, FPS별로 다름)
#   으로 시작.
DL_DEBUG_HISTORY_LEN = 90
DEBUG_VIZ_HOUGH_LANE = False  # 차선 — 대안 백엔드('hough') 디버그 창 (perception/hough_lane.py)
DEBUG_VIZ_LANE       = False  # 차선 — 대안 백엔드('classic_cv') 디버그 창 (perception/lane_util.py)
DEBUG_VIZ_STOPLINE   = False  # 정지선 디버그 창, 백엔드 무관 항상 동작 (perception/perc_floor.py)
# [2026-08-21] 좌회전 진입 랜드마크(체커 게이트/노란 파선 카운터) 디버그 창 — 실차에서
#   ROI/비율/카운트가 실제로 의도대로 잡히는지 눈으로 확인하기 위해 켜둠(요청 반영).
#   신뢰도 검증 전이라 기본 True — 확정되면 다른 항목처럼 False로 내릴 것.
DEBUG_VIZ_CHECKER_GATE = False   # 체커무늬(하프 출발선) 게이트 디버그 창 (perception/hough_lane.py CheckerBandGate)
DEBUG_VIZ_DASH_COUNTER = False   # 노란 파선 카운터 디버그 창 (perception/hough_lane.py YellowDashCounter)
# [2026-08-22] 위 CHECKER_GATE(비전, hough_lane.py)와는 별개 — 이건 perc_checker_pillar()가
#   쓰는 라이다 좌우 기둥쌍 검출(체크무늬 게이트 라이다 기둥쌍, checker_pillar_trigger) 전용
#   BEV 디버그 창. 신호등 좌회전 확정 후 이 트리거가 실제 좌회전 시작 시점을 결정하는
#   구조로 바뀌면서(요청 반영, track_drive.py _s1_lane_follow()/_s0_signal() 참고) 실차에서
#   좌우 기둥쌍이 실제로 잡히는지 눈으로 바로 확인할 필요가 생겨 추가했다.
#   [2026-08-22i] 요청 반영으로 끔 — 이 창의 핵심 정보(라이다 감지 트리거 여부)는 아래
#   DEBUG_VIZ_LEFT_TURN 통합 창에도 요약으로 들어간다. 좌우 점 하나하나의 원시 BEV
#   좌표까지 봐야 할 때만 다시 True로.
# [2026-08-23] True → False(요청 반영: "신호등, 라이더, 좌회전 디버깅창 제외하고 다 꺼줘") —
#   이 창의 라이다 BEV는 DEBUG_VIZ_LIDAR/DEBUG_VIZ_LEFT_TURN 창에도 나오므로 중복.
DEBUG_VIZ_CHECKER_PILLAR = False   # 체크무늬 게이트 라이다 기둥쌍 검출 BEV 디버그 창 (track_drive.py)
# [2026-08-22i] 좌회전(체크무늬 게이트 진입) 전용 통합 디버그 창 — 실행중/실행끝/발행각도/
#   라이다감지 4가지를 한 창에 모아 보여준다(요청 반영: "좌회전 관련 ... 합친 디버그창 하나").
#   기존에 이 정보들이 obstacle_cut_debug(§_current_stage_label())/checker_pillar_bev/
#   VESC 창 등에 흩어져 있던 것을 좌회전만 따로 떼어 한 곳에 모음
#   (track_drive.py _debug_viz_left_turn() 참고). 같은 요청으로 이 창과
#   DEBUG_VIZ_DL_LANE(차선인식)만 남기고 나머지 DEBUG_VIZ_* 는 전부 껐다.
# [2026-08-22j] 요청 반영("좌회전 디버깅창에 라이다영상도 추가")으로 상태 패널 아래에
#   perc_checker_pillar()의 라이다 BEV 원시 프레임(_draw_checker_pillar_bev())도 이어붙였다
#   — DEBUG_VIZ_CHECKER_PILLAR가 꺼져 있어도(기본값) 이 창이 켜져 있으면 BEV가 계속 채워짐.
# [2026-08-22k] 요청 반영("카메라 영상도 같이 띄워줄래")으로 전방 카메라 원본
#   (self.img_front)도 이어붙였다. [2026-08-22l] 요청 반영("카메라 영상부분을 옆으로
#   붙여줘")으로 카메라/BEV를 세로가 아닌 좌우 나란히 배치로 변경 — 최종 레이아웃:
#   상태 텍스트(위) / FRONT CAM·LIDAR BEV 좌우 나란히(아래).
DEBUG_VIZ_LEFT_TURN = False  # 좌회전 실행중/실행끝/발행각도/라이다감지+카메라+BEV 통합 디버그 창 (track_drive.py)
                              # [2026-08-23] True → False(요청 반영) — B1(라바콘) 검증 중이라
                              #   당장 필요 없어 끔. 좌회전 진행 확인 필요해지면 다시 True로.
DEBUG_VIZ_YOLO_CONE  = False  # [2026-08-25, 요청 반영] 대회 실주행 준비 — 모든 디버그창 끔.
                               # 콘 원시검출 창('yolo_cone_result', yolo_cone.py
                               # show_debug_windows()) + obstacle_cut_debug 창의 콘 카메라
                               # 패널(cam_stage=='cone'일 때)용 vis 프레임(yolo_cone.py
                               # _worker() 참고, yolo_vehicle.py DEBUG_VIZ_YOLO_VEHICLE과
                               # 동일 관례). [2026-08-22i] 요청 반영으로 껐다가,
                               # [2026-08-23] B1(라바콘) 검증 위해 다시 켬.
# [2026-08-21] 신호등 위치+색상 판정을 YOLO 단독(yolo_signal_state.py) 하나로 정리하면서
#   (README §1.18) 이게 유일한 신호등 결과 창이 됐다 — 실차에서 지금 뭘 보고 판단 중인지
#   눈으로 확인하기 위함.
# [2026-08-23, 요청 반영] True → False — B1 박스면적 테스트 창만 보기 위한 임시 비활성화.
#   복원 시 True로.
DEBUG_VIZ_YOLO_SIGNAL_STATE = False   # [2026-08-25, 요청 반영] 대회 실주행 준비 — 모든 디버그창 끔.
                                      # 신호등 위치+색상상태 YOLO 검출 박스 디버그 창 (perception/yolo_signal_state.py, 창 이름 'YOLO_신호등')
                                      # [2026-08-22h] 요청 반영으로 껐다가, [2026-08-23] B3 통과 후
                                      #   S0_SIGNAL 대기 검증(아래 "6. 미션 State" override) 위해 다시 켰다가,
                                      #   [2026-08-23b] 요청 반영으로 다시 끔, [2026-08-23r] 요청 반영(주행용
                                      #   실차 검증 — S0_SIGNAL 직진/좌회전 판독을 눈으로 확인)으로 다시 켬.
                                      #   창 위치는 DEBUG_WIN_POS_YOLO_SIGNAL_STATE=(0,0), 600x450 —
                                      #   DL_LANE(600,0~)/LAVACON·YOLO_CONE(y=650~)과 안 겹치게 이미
                                      #   맞춰져 있다(config.py "디버그 창 위치" 절 참고).
# [2026-08-15] avoid-hold(§2.32) 전용 상태창 — 지금 유예가 걸려있는지/왜 걸렸는지/방향
#   힌트/조기해제 진행상황을 한곳에 모아 보여주고, 실측 안 된 파라미터 값도 항상 같이
#   띄워서 "이 숫자 아직 지어낸 값"이라는 걸 상기시킨다(track_drive.py
#   _debug_viz_avoid_hold(), avoid_hold_measurement_todo.md 참고). 다른 회피 관련 창
#   (lidar_bev 등)과 별개로 언제든 독립적으로 켜고 끌 수 있다.
DEBUG_VIZ_AVOID_HOLD = False
#   [2026-08-11] smooth-imu-yaw-rate 브랜치(0c0d88b)에서 수동 포팅 — 라바콘 실차 테스트 중
#   라이다 창과 함께 켜 두고 나머지는 꺼둔 상태(요청 반영).

# [2026-08-20] da 근접 컷(obstacle-cut, ENABLE_OBSTACLE_CUT 주석 참고) 전용 상태창 —
#   라이다 raw/YOLO raw/AND확정/유지타이머 잔여시간/해제카운터를 avoid_hold_debug와
#   같은 구조로 한곳에 모아 보여준다(track_drive.py _debug_viz_obstacle_cut()).
# [2026-08-23] False → True — B1/B2/B3 최대 박스면적(B1/B2/B3 태그 포함)을 이 창에서도
#   보고 싶다는 요청으로 다시 켬.
DEBUG_VIZ_OBSTACLE_CUT = False   # [2026-08-25, 요청 반영] 대회 실주행 준비 — 모든 디버그창 끔.
#   [2026-08-22i] 요청 반영으로 껐다가, [2026-08-22m] 회피(da 근접 컷)
#   검출범위 확인용으로 다시 켰다가, [2026-08-23a] 요청 반영으로 다시 끔, [2026-08-23b] B1/B2/B3
#   회피 전체 흐름 검증 시작하며 다시 켬 — dl_lane(da) 창과 같이 켜서 "라이다가 왜 지금
#   잡았는지"와 "그래서 경로가 실제로 밀렸는지"를 두 창에서 나란히 확인.
#   [2026-08-23r] 요청 반영으로 다시 끔 — 대신 DEBUG_VIZ_YOLO_SIGNAL_STATE를 켜서 주행 중
#   신호등 판독(직진/좌회전)을 확인하는 쪽으로 전환. 이 창의 자리(DEBUG_WIN_POS_OBSTACLE_CUT
#   =(0,450))는 비므로 겹칠 걱정 없음.


# #############################################################
# 6. 미션 State / 실차 테스트 범위 제한
# #############################################################
# [2026-08-25, 요청 반영] 최종주행코드 확정 — START_STATE를 실제 레이스 시작 상태인
#   S0_SIGNAL(출발선 신호 대기)로 원복. 그동안 B1/B2/B3 구간별 개별 검증을 위해 S1로
#   건너뛰던 디버그 스타트 세팅(아래 옛 이력 참고)을 전부 걷어내고, 정지→4구 신호
#   판독→직진/좌회전 확정→S1→B1→B2→B3 정상 경로로 시작한다. 아래 TEST_FORCE_BEHAVIOR도
#   짝으로 False 원복, track_drive.py __init__의 self.phase도 Phase.LAVACON으로 원복했다.
START_STATE     = MissionState.S0_SIGNAL
ENABLE_BEHAVIOR = True  # S1에서 라바콘/장애물/추월 Behavior를 켤지 여부(최상위 스위치)

# ── 실차 테스트 범위 제한 ──
TEST_DISABLE_INTERSECTION = False
#   True: 신호등 보드 인식(signal_board_confirmed, README §1.15 이전엔 정지선 self.stopline
#         기준이었음)이 확정돼도 감속→S0_SIGNAL 재진입을 아예 안 함(차선주행만 계속).
#   False: 원래대로 신호등 보드 인식 확정 시 감속 후 S0_SIGNAL로 정상 전환.
TEST_DISABLE_B2_B3 = False
#   True: Phase가 FIXED_OBSTACLE/VEHICLE로 넘어가도 트리거 검사를 건너뛰고
#         B0_NORMAL로 고정(B1 끝난 뒤 계속 일반 차선주행만 함). ENABLE_BEHAVIOR=False와
#         이중으로 걸어 B2/B3가 어떤 경로로도 안 켜지게 한다(안전판).
#   False: 원래대로 SAFETY_DIST/OVERTAKE_TRIGGER 트리거 검사해서 B2/B3 정상 발동 —
#         이번 B3 검증이 보고 싶은 게 바로 이 발동이라 False.
TEST_FORCE_BEHAVIOR = False
#   True: _behavior_enabled를 시작부터 강제 True로 켜서, START_STATE=S1_LANE_FOLLOW로
#         S0/S2를 건너뛴 채로도 B1→B2→B3가 정상 발동한다(그렇지 않으면 _behavior_enabled가
#         S0_SIGNAL 직진 확정 시에만 True가 되는데 그 경로 자체를 안 타므로 계속 False로
#         남아 Behavior가 영원히 안 켜짐).
#   False: 원래대로 S0_SIGNAL 직진 신호 확정 시에만 Behavior가 켜짐(정상 레이스 동작).
#   [2026-08-25, 요청 반영] 최종주행코드 확정 — True→False 원복. 위 START_STATE도
#   S0_SIGNAL로 같이 원복했으므로, S0_SIGNAL부터 정상 전환 경로(직진 신호 확정 시
#   _behavior_enabled=True)를 그대로 탄다.
TEST_SIGNAL_LOOP = False
#   [2026-08-24, 요청 반영] B1/B2/B3 phase 리셋(_s1_lane_follow() 직진 확정 분기)이
#   이 스위치와 무관하게 항상 동작하도록 바뀌면서(신호등 직진 확정을 재무장의 유일한
#   기준으로 통일, _update_lap() 바퀴완주는 더 이상 phase 리셋을 담당하지 않음) 이 플래그의
#   역할이 좁아졌다 — 이제 아래 True 항목 중 "phase=Phase.LAVACON/... 리셋" 부분은 이미
#   상시 동작이라 무관하고, `_do_checker_ramp_turn()`의 "phase==Phase.DONE일 때만
#   _signal_yolo_off를 다시 푸는" 좌회전 반복 테스트 전용 분기에만 영향을 준다.
#   True: 좌회전(지름길) 램프 완료 시점에 phase가 Phase.DONE이면(=신호판단 격리 테스트
#         상태에서 시작된 좌회전) 신호등 YOLO를 즉시 재개 — 좌회전 반복 테스트 편의용.
#   False: 원래대로 그 재개를 다음 바퀴 리셋에 맡긴다(정상 레이스 동작).
TEST_FORCE_SIGNAL_YOLO = False  # [2026-08-23p, 요청 반영] True→False로 원복 — 이 스위치가
                                  #   True면 _active_yolo_stage()가 mission_state/phase와 무관하게
                                  #   항상 'signal'만 반환해서, S0_SIGNAL 직진 확정 후 Phase.LAVACON
                                  #   진입 시 켜져야 할 cone YOLO 스테이지가 절대 안 켜진다 — B1
                                  #   라바콘 트리거(perc_lavacon_trigger()의 cone_confirmed_cam)가
                                  #   영원히 못 켜지므로, 신호→S1→B1 정상 전환 검증엔 반드시 꺼둘 것.
#   [2026-08-23, 요청 반영] "욜로 안 끊기게 띄워서 검출만 테스트" 전용 — True면
#   _active_yolo_stage()가 mission_state/phase/_signal_yolo_off 등 FSM 상태와 완전히
#   무관하게 항상 'signal'을 리턴해 신호등 YOLO가 절대 안 꺼진다. TEST_SIGNAL_LOOP의
#   phase 조기 이탈 문제(§1.19j)나 확정 후 hold-off(SIGNAL_YOLO_OFF_HOLD_FRAMES)에
#   전혀 영향받지 않는다 — 순수 검출 정확도(conf 수치 등)만 보고 싶을 때 켤 것.
#   True인 동안은 cone/vehicle YOLO 스테이지가 전혀 안 켜지므로 B1/B2/B3 검증과는
#   동시에 못 쓴다 — 신호등 검출만 볼 때만 켜고, 다른 검증으로 넘어가면 False로 끌 것.
#   대신 _da_avoidance_failed() 게이트 + TargetPassing(실측 기반 하드코딩)로 대체
#   — 구조화된 2차선 환경에서 검색 기반 계획은 과한 방식이라는 결론(README §4/§5.1)에
#   따름. B3(방해차량, 동적)는 여전히 USE_HYBRID_ASTAR_FOR_B3로 Hybrid A* 대안을 쓴다
#   (아래, 819행 부근).

# ★★★★★ [2026-08-23n, 요청 반영] True→False로 원복 — 실제 YOLO 신호등 판독(직진/좌회전
#   구분) 자체를 다시 확인하기 위해, 강제 좌회전 스위치를 끔. 좌회전 로직 단독 검증이
#   다시 필요하면 True로 되돌릴 것(아래 주석 참고). ★★★★★
TEST_FORCE_LEFT_TURN_SIGNAL = False
#   순전히 테스트 목적 — "지금 정상 주행(S1_LANE_FOLLOW) 중에 좌회전 신호를 이미 받은
#   것처럼" perc_signal()의 signal_left_confirmed를 무조건 True로 강제한다(YOLO
#   신호등 검출 결과와 완전히 무관). 이러면 실제 판단 소스는 신호등 YOLO가 아니라
#   perc_checker_pillar()의 좌우 라이다 기둥쌍 검출(checker_pillar_trigger) 하나만
#   남는다 — _s1_lane_follow()가 즉시 S0_SIGNAL 'left' 커밋 구간으로 전환해 그 안에서
#   차선주행을 유지하다, 라이다가 체크무늬 게이트 기둥쌍을 실제로 검출하는 순간
#   _begin_checker_ramp_turn()이 걸려 좌회전 램프가 시작된다 — 즉 "좌회전 로직 자체"
#   (커밋 구간 → 라이다 게이트 트리거 → 조향 램프)만 신호등 인식과 분리해서 단독
#   검증하려는 스위치.
#   True인 동안: TEST_SIGNAL_LOOP(위)와 맞물려 램프 완료 후 S1로 복귀하자마자 다시
#   signal_left_confirmed가 강제 True라 즉시 재커밋 → 게이트를 지날 때마다 계속
#   반복 진입한다(의도된 동작 — 반복 검증용). 신호등 색상 판단 자체는 이 스위치가
#   켜진 동안 사실상 무의미해진다(직진 신호를 봐도 좌회전 우선순위 규칙상 항상 좌회전
#   쪽이 이김, perc_signal() 참고).
#   ⚠️⚠️⚠️ 검증 끝나면 반드시 False로 되돌릴 것 — 실제 레이스에서 이게 켜진 채로 있으면
#   신호등이 빨간불/직진이어도 무조건 좌회전으로 우겨서 코스 완전 이탈한다. False로
#   되돌리면 track_drive.py의 override 코드가 자동으로 비활성화되고 원래 신호등 YOLO
#   기반 판단으로 돌아간다(다른 곳 되돌릴 필요 없음).

# [2026-08-23q, 요청 반영] True→False로 원복 — 좌회전 로직 단독 테스트 종료, 타임아웃
#   안전장치(CHECKER_PILLAR_LIDAR_TIMEOUT_SEC)를 다시 켠다.
TEST_DISABLE_CHECKER_PILLAR_TIMEOUT = False
#   순전히 테스트 목적 — CHECKER_PILLAR_LIDAR_TIMEOUT_SEC(위) 안전장치를 꺼서, S0_SIGNAL
#   'left' 커밋 구간이 기둥쌍 미검출을 이유로 좌회전을 포기하지 않고 무조건
#   checker_pillar_trigger(실제 라이다 기둥쌍 검출)만 기다리게 한다(track_drive.py의
#   _s0_signal() 참고) — 타임아웃 폴백이 실제 미검출 상황을 가려서 "왜 안 잡히는지"
#   디버깅이 안 되는 문제를 피하려는 스위치.
#   ⚠️⚠️⚠️ 검증 끝나면 반드시 False로 되돌릴 것 — 실제 레이스에서 이게 켜진 채로 있으면
#   라이다가 기둥쌍을 영원히 못 잡을 때(죽음/오검출) 좌회전 커밋 구간에서 영원히 못
#   벗어나고 차선을 놓친 채 계속 직진하는 사고로 이어진다.

# ── 바퀴(Lap) 카운트 — 트랙은 닫힌 곡선이라 한 바퀴 돌면 누적 yaw가 정확히 360도 ──
TOTAL_LAPS = 3
LAP_YAW_FULL   = math.radians(330.0)  # 이만큼 돌면 한 바퀴 완주로 확정(360도에 여유)
LAP_YAW_MIN    = math.radians(270.0)  # 정지선으로 조기 확정하려면 최소 이만큼은 돌아야 함
LAP_MIN_T      = 20.0   # 한 바퀴 최소 소요시간(s) — 연속 오검출 차단
LAP_YAW_CONFIRM_FRAMES = 10  # 누적 임계를 넘은 상태가 이 프레임(20Hz→0.5초) 연속 유지돼야 확정
RESET_PHASE_EACH_LAP = True
#   True : 새 바퀴가 시작되면 Phase를 LAVACON으로 되돌린다(라바콘 등은 매 바퀴 다시 만남).
#   False: Phase를 유지한다(1바퀴만 미션 수행하고 이후 순수 차선주행).


# #############################################################
# 7. 기타 튜닝 파라미터 (PID, 트리거, 회피, 신호등, 정지선, 단위환산 등)
# #############################################################

# ── 라이다 장착 각도 보정 (재실측 2026-07-22) ──
#   perc_obstacle()/perc_lavacon_trigger()에서 극좌표→직교좌표 변환 시
#   "deg = 인덱스(도) - LIDAR_ANGLE_OFFSET_DEG"로 이 오프셋을 뺀다.
#   보정 후 각도 약속: 0=정면, 90=좌측, 180=후방, 270(-90)=우측(반시계 방향).
LIDAR_ANGLE_OFFSET_DEG = 80.0
BODY_MASK_ENABLED = True  # 차체 자기가림 마스킹(BODY_LO~BODY_HI) 전체 스위치. 최종 확정(2026-07-22)

# ── 라이다 장착 위치(종방향) 보정 — 실측 2026-08-19 ──
#   라이다는 차량 맨 앞부분보다 이만큼(m) 더 앞으로 나가 있다(lavacon_bev 디버그창에서
#   자차 마커를 물체와 동일선상으로 맞춰가며 실측 — track_drive.py _draw_lavacon_bev()의
#   EGO_MARKER_PULLBACK_PX/LAVACON_BOX_LON_WIDTH 튜닝 이력 참고). process_lavacon()이
#   내놓는 라바콘 경로(perc_lavacon.py, path_m)는 전부 "라이다 원점 기준" 좌표라, 그걸
#   그대로 Pure Pursuit에 "차량이 여기 있다"고 넘기면 차량 위치를 실제보다 0.2m 앞으로
#   착각하게 된다 — _handle_lavacon()이 _lane_steer()를 호출할 때 이 값만큼 차량 기준점을
#   뒤로 밀어서 보정한다(track_drive.py _lane_steer() vehicle_y_px 참고). 라바콘
#   ROI/박스 스택 임계값(perc_lavacon.py LON_MIN/CONE_LON_MAX/BOX_LON_START 등)은 전부
#   "라이다가 실제로 뭘 보는지" 기준으로 튜닝된 값이라 이 보정과 무관 — 그대로 둔다.
LIDAR_TO_VEHICLE_FRONT_M = 0.20

# ── 차선 PID (_lane_pid(), B2/B3 behavior가 여전히 사용) ──
LANE_KP, LANE_KI, LANE_KD = 0.70, 0.0008, 0.15
LANE_INTEGRAL_TERM_MAX = 10.0  # [anti-windup] 적분항이 조향에 기여할 수 있는 최대 각도(도)
LANE_SIDE = 1               # 주행 차선: +1=노란선 오른쪽(우측차선), -1=왼쪽
LANE_CORNER_BOOST = 1.8     # 코너(큰 offset) 조향 가중
LANE_CORNER_REF   = 120.0   # 이 offset(px)에서 가중 최대
LANE_CORNER_MIN   = 40.0    # 코너 가중 시작 임계(px)
LANE_DEADZONE     = 40.0    # 중앙 데드존(px)

# ── 차선 인식 안정화 (perception/lane_util.py — dl/classic_cv 백엔드 공용) ──
# 프레임 간 스파이크 필터링(디바운스): 새로 들어온 (lane_valid, offset)이 직전
#   "후보"와 STABLE_JUMP_MAX(px) 이내로 비슷하면 후보 연속 프레임 수를 늘리고,
#   벗어나면 새 후보로 교체하며 카운트를 1로 리셋한다. 후보가 STABLE_FRAME_MIN
#   프레임 연속 유지돼야만 확정값으로 승격돼 실제 출력에 반영된다 — 1~2프레임짜리
#   튐이 조향에 바로 반영되지 않게 막는다. 실차 미검증 튜닝값.
STABLE_FRAME_MIN = 3   # 후보를 확정값으로 승격시키기 위해 필요한 연속 프레임 수
STABLE_JUMP_MAX = 15   # 이 이상(px) 차이나면 "같은 흐름"이 아닌 새 후보로 취급

# 경로(웨이포인트) 프레임 간 EMA 스무딩 — pure_pursuit이 실제로 추종하는 self.path는
#   매 프레임 그 프레임의 관측점만으로 처음부터 다시 피팅돼 대입되므로(dl_lane.py
#   detect(), lane_util.calc_center() 공통) 별도 필터가 없으면 프레임간 흔들림이
#   그대로 조향에 실린다. 웨이포인트 인덱스별로 새 값과 직전 값을 블렌딩(x,y 둘 다 —
#   y_far가 프레임마다 달라지므로 x만 블렌딩하면 인덱스가 가리키는 실제 지점이 어긋남).
#   값을 낮추면(더 스무딩) 저킹은 줄지만 코너 진입 반응이 늦어짐. 실차 미검증 튜닝값.
#   [2026-08-05] 실차 조향 오실레이션("와리가리") 대응으로 0.3→0.2 하향(pure_pursuit.py의
#   속도 적응형 lookahead와 같은 목적 — 입력 신호 자체의 프레임간 흔들림을 줄임).
#   [2026-08-06] 0.2→0.4로 재상향. pure_pursuit.py 쪽 lookahead 얼어붙음 버그 수정 +
#   curvature 기반 lookahead 축소로 "중심선이 waypoint에서 벗어나면 되돌아오는 반응이
#   한 박자 늦다"는 문제를 제어 단계에서 어느 정도 잡았으니, 인지 단계 경로 스무딩도
#   새 프레임 비중을 높여 지연 자체를 줄여본다. 오실레이션 재발 시 0.2~0.3으로 낮출 것.
#   [2026-08-12] 바로 그 "재발" 상황 — 직진 구간에서도 계속 진동한다는 보고로 0.4→0.25로
#   다시 낮췄다(위에서 예고한 0.2~0.3 범위 안). README §0.5.7 참고. A1/A2/A3(같은 날
#   추가된 da/ll/corridor 밴드 간 속도예측+앵커링)이 경로 자체의 프레임간 흔들림을
#   줄이는 근본 대응이고, 이 값은 그 위에 남는 잔여 흔들림을 한 번 더 죽이는 보강이다.
#   [2026-08-17] 실차 주행 중 커브 진입 구간에서 웨이포인트가 실제 커브를 따라가는 게
#   너무 늦어(0.25=새 프레임 비중 25%라 매 프레임 잔여오차의 75%가 남음) 코너를 못 돌고
#   박는 문제로 재현됨. lane_valid는 커브 중에도 계속 True라 §B2/B3 전용인 위쪽
#   "lane_valid일 때만 EMA 갱신" 프리즈 수정과는 무관한 별개 증상 — 순수하게 EMA 자체가
#   코너 추종 속도를 못 따라가는 경우다. 0.25→0.45로 재상향(과거 최댓값 0.4보다 살짝
#   위). 오실레이션 재발 시 0.3~0.4 사이로 낮출 것.
#   [2026-08-17j] 0.45→0.75(요청 반영, 추가 상향). 코너 추종 지연을 더 줄이는 방향 —
#   진동 대응은 이 값을 낮추는 대신 아래 PP_LOOKAHEAD_BASE_PX/MAX_PX 상향(같은 날
#   [2026-08-17k])으로 분리했다(lookahead가 길수록 curvature=2*sin(alpha)/ld 공식상
#   같은 픽셀 노이즈의 증폭이 줄어 진동에 더 강함 — §0.5.2). 실차 미검증, 특히 0.75는
#   과거 검증 구간(0.2~0.4)을 크게 벗어난 값이라 직진에서 잔떨림이 커지는지 최우선 확인할 것.
PATH_EMA_ALPHA = 0.25   # 새 프레임에 줄 가중치(작을수록 더 부드럽고, 더 느리게 반응)

# [2026-08-25] 룩어헤드 목표점이 "엄청 잠깐 짧게" 코앞으로 튀었다가(다음 프레임 바로 회복)
#   사라지는 현상 대응. dl_lane.py detect()의 path_ok_raw는 near_center/far_center 중
#   하나만 있어도 True라(_debounce_path_ok() 참고), 원거리 슬라이스들이 딱 한 프레임만
#   노이즈로 비어도(모델 순간 미검출, _largest_da_component() 덩어리 전환 등) 안 걸리고
#   그대로 _fit_and_sample_path()에 들어간다 — 그 함수의 y_far=min(ys)가 그 한 프레임만
#   차량 근처로 훅 당겨진 fitted_path를 만들고, PATH_EMA_ALPHA가 큰 프리셋(최대 0.93)에서는
#   그 한 프레임짜리 결과가 거의 그대로 self.path에 실려 pure_pursuit의 목표점이 코앞으로
#   튀는 것으로 보인다(_target_point()가 짧아진 경로 끝, path[-1]을 그대로 반환).
#   _debounce_path_reach_drop()(dl_lane.py)이 "reach(=path가 지금 얼마나 멀리까지 보는지,
#   px)가 한 프레임 만에 이 값 이상 줄면" 일단 노이즈로 의심하고 PATH_REACH_DROP_FRAMES
#   연속으로 재현돼야만 받아들인다 — offset 디바운스(_debounce())와 동일한 "N프레임 연속
#   확인" 철학을 reach에 적용한 것뿐이라, 늘어나는 쪽(위험하지 않음)은 즉시 통과시키고
#   진짜 상황 변화(장애물 컷, 급커브 진입 등 몇 프레임 지속되는 변화)는 그대로 반영된다.
#   실차 미검증 — 코너 진입 시 경로 갱신이 늦어 보이면 DROP_MAX_PX를 올리거나
#   DROP_FRAMES를 낮출 것, 반대로 튀는 현상이 여전히 보이면 DROP_MAX_PX를 낮출 것.
PATH_REACH_DROP_MAX_PX = 60.0
PATH_REACH_DROP_FRAMES = 2

# ── 라바콘/장애물/방해차량/신호등 트리거 ──
LAVACON_DONE_FRAMES = 40      # [2026-08-24, 요청 반영] 20→40(≈2초) — 종료가 너무 민감해서 복귀
# [2026-08-22, 요청 반영] 40(≈2초)→20 — 종료판정 ROI를
                               #   perc_lavacon_trigger() 트리거 박스와 동일 크기로 축소한
                               #   것과 함께, "그 박스에 1초 이상 아무것도 안 찍히면 종료"로
                               #   조건을 바꿨다(20Hz 고정주기 기준 20프레임=1초,
                               #   perc_lavacon.py EXIT_LON_MIN/MAX/LAT_LIMIT 주석 참고).
                               #   좌우 콘 미검출이 연속 N프레임 쌓이면 Phase 전환(디바운스)
LAVACON_TRIGGER_FRAMES = 2    # (YOLO 콘 검출 AND 좌우 라이다 클러스터 동시검출)이 연속 N프레임
                               #   쌓이면 B1_LAVACON 진입 확정. [2026-08-07] 카메라(YOLO)+라이다
                               #   이중확인으로 강화 — 값 자체는 기존 그대로 유지.
                               #   [2026-08-23] 5 → 2(요청 반영) — 실차에서 좌우 라이다가 분명히
                               #   찍히는데도 5프레임 연속을 못 채워 B1_LAVACON 진입이 안 되는
                               #   문제로 완화. 셋 중 하나라도 한 프레임 빠지면 즉시 0으로
                               #   리셋되는 디바운스 특성상(perc_lavacon_trigger() 참고) 5프레임
                               #   (0.25초)이 실차 노이즈엔 너무 엄격했던 것으로 추정 — 오검출
                               #   방지 효과가 필요하면 다시 올릴 것.

# [2026-08-19] 박스 스택 페어링(perc_lavacon.py `_build_path`)에서, 한쪽만 검출된 박스를
#   버리지 않고 반폭(half-width) 추정으로 살릴지 여부. 기본 False — 켜기 전엔 기존
#   `_pick_boxed_centers`와 결과가 100% 동일함이 보장돼야 한다(perc_lavacon.py 상단 주석
#   5) 참고). 실차에서 콘 간격이 벌어지는 구간 대응이 필요하면 켤 것.
# [2026-08-19] 자가 테스트로 기존 동작과 동일함이 확인된 상태에서(0d1b55) 실차 검증 단계로
#   전환 — sparse fallback + 프레임간 EMA(아래) 둘 다 True로 켬. [2026-08-22 이후]
#   DEBUG_VIZ_LAVACON 통합 창('lavacon_bev', track_drive.py `_draw_lavacon_bev()`)으로
#   좌우 EMA 차선을 직접 보면서 튜닝할 것(원래는 DEBUG_VIZ_LAVACON_EMA 전용 창이었으나
#   'lavacon_bev' 하나로 통합됨).
LAVACON_SPARSE_FALLBACK_ENABLED = True
LAVACON_HALFWIDTH_EMA_ALPHA     = 0.3   # 좌우 반폭 러닝 추정 EMA 계수(작을수록 더 느리게 반응)

# [2026-08-19] 라이다가 한 프레임 튀어도(반사 노이즈, 순간 미검출) waypoint가 그대로
#   같이 튀는 문제 대응 — 박스별 좌/우 바운더리 포인트("라바콘 차선")에 프레임간 EMA를
#   건다(perc_lavacon.py `_blend_boxes_temporal` 참고). PATH_EMA_ALPHA(da 차선 경로용)와
#   완전히 별개 — 저기는 값이 다르므로 여기 새 상수를 쓴다. 기본 False였다가 실차 검증
#   단계로 전환하며 잠깐 True였음 — 라바콘 간격이 넓어(박스 폭 0.2→0.4로 이미 대응) 콘이
#   가려져서 검출이 성긴 구간에서는 EMA로 스무딩할 재료(연속 프레임 검출) 자체가 부족해
#   급커브 대응에 도움이 안 된다고 판단, 다시 False로 뺌(요청 반영).
LAVACON_TEMPORAL_EMA_ENABLED = False
LAVACON_TEMPORAL_EMA_ALPHA   = 0.5      # 새 프레임에 줄 가중치(작을수록 더 부드럽고 느리게 반응)

# [2026-08-19] 라이다 박스 스택 페어링이 실차에서 계속 듬성한 검출/노이즈에 시달려서
#   (요청 반영) 아예 다른 축으로 전환 — da(주행가능영역) 단독으로도 라바콘 구간을
#   그럭저럭 지나간다는 게 실차로 확인됐으므로, da 경로를 기본으로 신뢰하고 콘이 안전
#   마진 안으로 들어왔을 때만 그만큼 옆으로 미는 방식(track_drive.py
#   `_lavacon_steer_da_push()`, perc_lavacon.py `nearest_cone_lateral()`)을 추가.
#   자가 테스트로 꺼진 상태에서 기존 박스 스택 조향과 동일함이 확인된 상태에서(ff6e80a),
#   기본 주행 방식으로 채택하며 True로 전환(요청 반영) — 박스 스택 조향은 이제
#   `_handle_lavacon()`의 폴백 분기로만 남는다.
LAVACON_STEER_MODE_DA_PUSH = True

# [2026-08-23, 요청 반영] 임시 실험용 — B1 진입 확정(_lavacon_engaged 상승 엣지) 직후
#   일정 시간 동안 push 계산을 무시하고 고정 조향각을 강제로 꽂아 넣는다(속도는 건드리지
#   않음, _update_speed()가 그대로 돎). "진입하자마자 확 꺾어서 초반 자세를 잡아준다"는
#   아이디어를 실차로 빠르게 테스트해보기 위한 스위치 — 효과 없거나 오히려 나쁘면
#   LAVACON_KICK_ENABLED만 False로 되돌리면 이 블록 전체가 비활성화된다(실차 미검증).
LAVACON_KICK_ENABLED    = True
LAVACON_KICK_DURATION_S = 0.4    # 이 시간(초) 동안 고정 조향각 유지 — CONTROL_HZ(20Hz)로 환산해 프레임수로 씀
                                  # [2026-08-23] 0.2 → 0.4 → 0.2 → 0.0 → 0.2(요청 반영)
                                  # [2026-08-25] 0.2 → 0.4(요청 반영)
LAVACON_KICK_ANGLE_DEG  = -30.0  # 강제 조향각(도) — 부호규약은 ctrl_angle과 동일(우측 콘 쪽으로 짐작, 실차에서 방향 확인 필요)
                                  # [2026-08-23] -20.0 → -30.0 → 0.0 → -20.0(요청 반영)
                                  # [2026-08-25] -10.0 유지(요청 반영)

# [2026-08-19] 안전마진(m) — 이 값보다 콘이 차량 중심에 가깝게 들어오면 그만큼 반대쪽으로
#   민다. VEHICLE_WIDTH_M(0.31, 실측)/2=0.155(차량 반폭) + DL_DA_SIDE_MARGIN_M(0.1, B2/B3
#   장애물 회피에서 "실차 테스트로 딱 적당하게 잘 주행함" 확인된 좌우 여유값, 성격이
#   동일해 재사용) = 0.255 → 0.26로 반올림. 콘 자체의 물리적 반지름은 따로 반영 안 함
#   (실측값 없음, perc_lavacon_trigger()의 CLUSTER_MAX_GAP=0.35는 "콘 지름 근사"라 참고는
#   되나 라이다 스프레드가 섞인 값이라 그대로 쓰지 않음) — 실차에서 콘을 스치면 이 값을
#   올릴 것.
# [2026-08-22] 0.26 → 0.13(요청 반영, 실차 미검증) — push가 너무 세게/일찍 걸린다는
#   판단으로 절반으로 낮춤. 콘을 스치면 다시 올릴 것.
# [2026-08-22b] 0.13 → 0.26(요청 반영, 실차 미검증) — lavacon_bev 디버그창에서 "인접하면
#   민다"고 판정하는 보라색 안전마진 라인(±margin) 구간이 너무 좁아 보인다는 지적으로
#   좌우 폭을 2배로 되돌림(공교롭게도 위 2026-08-22 절반값과 정확히 반대라 원래 0.26으로
#   복귀). push 세기 자체는 이 값과 별개로 LAVACON_PUSH_GAIN이 담당한다(아래).
# [2026-08-22c] 좌우 비대칭 요청 반영 — 좌측만 0.3으로 확대, 우측은 기존 0.2 유지.
LAVACON_PUSH_SAFETY_MARGIN_L_M = 0.35
LAVACON_PUSH_SAFETY_MARGIN_R_M = 0.23

# [2026-08-22b] push량(push_m) 배율 — 안전마진(위 LAVACON_PUSH_SAFETY_MARGIN_M) 침범량에
#   곱해 실제로 미는 세기를 정한다. margin을 넓힌 것과는 별개로 "밀리는 정도 자체"도
#   2배로 키워달라는 요청 반영(실차 미검증) — track_drive.py `_lavacon_steer_da_push()`와
#   lavacon_bev 디버그 표시(둘 다 같은 부호규약) 양쪽에 동일하게 곱한다.
LAVACON_PUSH_GAIN = 1.35 #1.13

# [2026-08-19] push 신호 전용 ROI — 박스 스택의 CONE_LON_MAX(4.0m, 구간 전체) 대신 훨씬
#   가까운 범위만 본다. "지금 당장 스칠 위험이 있는 콘"만 반응해야 하므로, 멀리 있는
#   콘까지 보면 아직 위협도 아닌데 미리 밀거나(오조향) 다음 콘으로 넘어가면서 push가
#   들쭉날쭉 튈 위험이 있다. 실측 아님 — 실차에서 튜닝 필요.
# [2026-08-22] 0.1~0.5 → 0.3~0.7(요청 반영, 폭 0.4m 유지한 채 통째로 뒤로 밈) — lavacon_bev
#   디버그창에서 push ROI(보라 박스)가 트리거 ROI(노란 박스, perc_lavacon_trigger()의
#   LON_MIN=0.3~LON_MAX=0.7)보다 앞쪽에서 시작해 둘의 검출 시작 지점이 어긋나 보인다는
#   지적으로, 두 박스의 종방향 시작점을 0.3m로 맞췄다. 폭은 그대로 유지했으므로 끝점도
#   0.5→0.7로 같이 밀렸다.
# [2026-08-22] 0.3~0.7 → -0.1~0.3(요청 반영, 폭 0.4m 유지한 채 통째로 앞으로 당김) — 위
#   0.3~0.7은 "라이다 원점" 기준값인데, lavacon_bev의 자차 마커(파란 점, _draw_lavacon_bev()
#   EGO_MARKER_PULLBACK_PX)는 라이다 원점보다 LAVACON_BOX_LON_WIDTH(0.4m)만큼 뒤에 그려진다
#   — 즉 자차 마커(=차량 실제 위치, 이 마커는 절대 건드리지 않기로 함)를 기준으로 보면
#   push ROI는 실제로 차량 앞 0.3+0.4=0.7m 지점부터 시작하고 있었다. "차량 전방 0.3m부터
#   감지"가 되려면 라이다 원점 기준으로는 0.3(요청값) - 0.4(마커 뒤당김량) = -0.1부터여야
#   한다(폭 0.4m는 그대로 유지 → 끝점도 0.7→0.3). 트리거 ROI(노란 박스,
#   perc_lavacon_trigger()의 LON_MIN/LON_MAX)도 같은 이유로 뒤이어 -0.1~0.3으로 맞췄다 —
#   두 ROI는 항상 같이 바꿀 것, 하나만 바꾸면 lavacon_bev에서 둘의 시작점이 다시 어긋나 보인다.
LAVACON_PUSH_LON_MIN     = -0.1  # 차량(자차 마커) 기준 전방 0.3m(=라이다 원점 기준 -0.1m)부터
LAVACON_PUSH_LON_MAX     = 0.25  # 이 거리보다 먼 콘은 아직 안 민다
LAVACON_PUSH_LAT_LIMIT   = 1.0   # 횡방향 탐색 한계 — CONE_LAT_LIMIT(perc_lavacon.py)와 동일값으로 시작
# [2026-08-24, 테스트] 좌측만 전방(LON_MAX) 0.1m 확장 — 우측(LAVACON_PUSH_LON_MAX)은 그대로.
LAVACON_PUSH_LON_MAX_L   = LAVACON_PUSH_LON_MAX + 0.1

# [2026-08-19] 박스 안 후보점의 좌/우 배정을 y부호(차량 헤딩 기준 고정 중앙선, y>0=좌/
#   y<0=우) 대신 직전 박스의 같은 라인과의 최근접 연속성으로 할지 여부
#   (perc_lavacon.py `_assign_by_continuity()` 참고). 급커브에서는 물리적으로 계속
#   "오른쪽 라인"이던 콘이 차량 헤딩 기준 y>0(수학적으로는 왼쪽)으로 넘어갈 수 있는데,
#   고정 y=0 경계로 가르면 그 점이 왼쪽 라인으로 잘못 편입된다 — 사용자가 lavacon_ema_bev
#   창에서 실제로 확인(오른쪽 콘 두 개가 색이 갈려서 찍힘). 자가 테스트로 기존 동작과
#   동일함이 확인된 상태에서 바로 True로 켬(요청 반영) — 끄면 기존 y부호 방식과 100%
#   동일하다.
LAVACON_LINE_CONTINUITY_ENABLED = True
LAVACON_LINE_TRACK_MAX_JUMP_M   = 0.6   # 직전 박스 같은 라인 점과의 최대 허용 거리(m) — 이보다 멀면 다른 콘/라인으로 보고 y부호 폴백

# ── 라바콘 카메라 이중확인 (perception/yolo_cone.py, YOLOv8n ONNX) ──
#   perc_lavacon_trigger()가 기존 라이다 좌우 클러스터 판정에 "카메라로도 콘이 보이는가"를
#   AND로 추가한다 — 라이다 단독 클러스터 판정은 벽 모서리 등에서 오검출 여지가 있어서,
#   실제로 콘(cone) 클래스가 화면에 잡힐 때만 진입을 인정하도록 이중화한다.
#   [2026-08-11] smooth-imu-yaw-rate 브랜치(0c0d88b)에서 수동 포팅.
# [2026-08-20] ENABLE_BEHAVIOR=False(라바콘/장애물/추월 Behavior 전체 비활성, 순수
#   차선주행만 사용) 상태인데도 perc_yolo_cone()이 매 프레임 백그라운드에서 계속 돌고
#   있어서(track_drive.py perceive_all()) 요청 반영으로 끈다 — ENABLE_OBSTACLE_CUT과
#   동일 패턴으로, False면 track_drive.py가 YoloConeDetector 자체를 생성하지 않는다
#   (self.yolo_cone_detector=None, perc_yolo_cone()이 None 체크로 조용히 스킵).
#   라바콘(B1) 실차 테스트를 다시 시작하면 True로 되돌릴 것.
# [2026-08-21, 임시] B3(방해차량) 회피 검증을 위해 다시 켬 — perc_obstacle_cut_trigger()가
#   콘/차량 카메라 이중확인으로 obstacle_cut_type을 가르는데, 이게 꺼져 있으면 B2(콘)가
#   카메라 폴백(라이다 단독)으로 흐려 "트랙 순서상 B2가 먼저 지나가야 B3 인정"이라는
#   _b2_passed 가드가 안정적으로 안 걸린다(위 START_STATE 근처 임시 테스트 블록 참고).
YOLO_CONE_ENABLE = True
YOLO_CONE_INPUT_SIZE = 640     # cone_best_n.onnx export 시 imgsz와 반드시 일치시킬 것
YOLO_CONE_CONF_THRESHOLD = 0.5 # 이 신뢰도 이상인 검출만 인정(모델이 nms=True로 export돼 좌표 디코딩은 불필요)
YOLO_CONE_MODEL_PATH = None    # None이면 yolo_ros/cone_best_n.onnx(형제 디렉터리)를 자동으로 찾음(perception/yolo_cone.py 참고)
# [2026-08-23] 요청 반영 — "화면에 콘이 찍힌다"만으로 검출 인정하지 않고, 이번 프레임
#   검출된 박스들 중 "가장 큰" 것의 면적(YOLO_CONE_INPUT_SIZE=640 스케일 기준 px²)이
#   이 값 이상일 때만 검출로 인정한다. perception/yolo_cone.py는 원시 검출 여부와
#   최대 박스면적(get_latest_max_area())만 돌려주고, 실제 면적 게이트는
#   track_drive.py가 B1(perc_lavacon_trigger())/B2(perc_obstacle_cut_trigger()) 각
#   호출부에서 건다 — 같은 콘 검출기를 공유하지만 B1/B2가 처한 상황(진입 트리거 vs
#   이미 진입해 지나치는 중)이 달라 멀리서 작게 찍힌 걸 걸러낼 기준도 다를 수 있다는
#   요청 반영(2026-08-23, "그 크기값을 b1/b2/b3마다 다른 변수로")으로 아래 두 값으로
#   분리했다.
#   [2026-08-23 밤, 요청 반영] b1_cone_area_debug/b2_cone_area_debug 창으로 실차 실측 완료 —
#   B1=5000.0, B2=2500.0으로 확정(placeholder 아님).
YOLO_CONE_MIN_BOX_AREA_PX_B1 = 5000.0   # B1(라바콘 진입 트리거) 전용 — 실차 실측 확정치
YOLO_CONE_MIN_BOX_AREA_PX_B2 = 2500.0   # B2(고정장애물 obstacle_cut) 전용 — 실차 실측 확정치
# [2026-08-24, 요청 반영] 면적게이트 자체가 1프레임짜리 순간 오검출로도 통과되는 문제 —
#   B1/B2/B3 모두 면적게이트 통과가 연속 N프레임 유지돼야 확정으로 친다(각자 독립 카운터,
#   track_drive.py의 _cone_area_b1_cnt/_cone_area_b2_cnt/_vehicle_area_b3_cnt).
#   20Hz 고정주기 기준 5프레임=0.25초. 라이다와 합쳐지는 하위 디바운스(LAVACON_TRIGGER_FRAMES/
#   OBSTACLE_CUT_TRIGGER_FRAMES)와는 별개로 YOLO 조건 자체에 거는 게이트.
YOLO_AREA_CONFIRM_FRAMES = 5

# ── [2026-08-20] 방해차량 카메라 이중확인 (perception/yolo_vehicle.py, YOLOv8n ONNX) ──
#   da 근접 컷(ENABLE_OBSTACLE_CUT, 위 참고) 전용. 최초 이식(fix/da-corridor-near-band-margin
#   브랜치 3be0fb6)은 파인튜닝 모델 없이 COCO 사전학습 yolov8n.pt를 그대로 ONNX(nms=True)
#   내보내기해서 'car'(class_id 2)만 필터링해 썼다(신뢰도 0.15~0.78, 평균 0.3대로 낮음 —
#   yolo_ros/yolov8n_car.onnx로 롤백/비교용 보존).
#   [2026-08-20 §2.57] 대회에서 실제로 회피해야 하는 그 차량 한 대(#46, TRAXXAS 검정/연두)
#   뒷모습만 잡도록 전용 파인튜닝한 target_vehicle_best.onnx(yolo-V8-KMU-xycar 저장소,
#   nc=1 'target_vehicle')로 교체 — YOLO_VEHICLE_CLASS_ID를 2(COCO car)에서 0으로 변경.
#   이때는 nms=False로 export됐었다(ultralytics nms=True 옵션이 안 먹혀서) — raw 출력을
#   perception/yolo_vehicle.py가 직접 디코딩+NMS하는 우회로 대응했었음.
#   [2026-08-20 §2.59] 원인(ultralytics 8.3.0의 DetectionModel ONNX export가 nms 인자를
#   참조 안 함, CoreML 전용 옵션이었음)을 찾아 export 단계에서 고쳤다 — v1.2.0부터는
#   torchvision.ops.batched_nms를 심은 커스텀 export로 output0가 다시 정상 NMS 내장
#   [1,N,6]이다(가중치는 v1.1.0과 동일). YOLO_VEHICLE_NMS_IOU_THRESHOLD(§2.57에서 쓰던
#   직접 NMS용 파라미터)는 더 이상 필요 없어 삭제 — perception/yolo_vehicle.py도 콘/구
#   vehicle 모델과 같은 단순 파싱으로 되돌림.
YOLO_VEHICLE_INPUT_SIZE = 640
YOLO_VEHICLE_CONF_THRESHOLD = 0.6  # [2026-08-20 §2.57] 처음엔 실측 분포 없이 ultralytics
                                    # 기본값(0.5)으로 시작.
                                    # [2026-08-21 §2.60] v1.2.0 실측 결과 신뢰도가 0.7 밑으로
                                    # 안 내려가는 것 확인 — 오탐 여유를 두면서도 정탐은 그대로
                                    # 다 통과시키도록 0.6으로 상향(0.7 그대로 쓰면 여유가 없어
                                    # 경계선 프레임을 놓칠 수 있음)
YOLO_VEHICLE_CLASS_ID = 0          # [2026-08-20 §2.57] target_vehicle_best.onnx 클래스 id
                                    # (nc=1, 'target_vehicle' 하나뿐이라 0부터 시작 — COCO
                                    # class_id=2였던 이전 모델과 다름, 반드시 같이 바꿀 것)
YOLO_VEHICLE_MODEL_PATH = None     # None이면 yolo_ros/target_vehicle_best.onnx(형제 디렉터리)를
                                    # 자동으로 찾음(perception/yolo_vehicle.py _default_model_path() 참고)
                                    # [2026-08-20] 가중치 파일을 yolo-V8-KMU-xycar 저장소
                                    # v1.0.0(seed_labeled 2,127장, mAP50-95=0.974) →
                                    # v1.1.0(seed+round2 6,041장, mAP50-95=0.985) →
                                    # [2026-08-20 §2.59] v1.2.0(가중치는 v1.1.0과 동일,
                                    # nms 내장 export로 교체)으로 갱신.
# [2026-08-23] 요청 반영 — YOLO_CONE_MIN_BOX_AREA_PX_B1/_B2와 동일한 목적/패턴. 이번
#   프레임 검출된 target_vehicle 박스들 중 가장 큰 것의 면적(YOLO_VEHICLE_INPUT_SIZE=640
#   스케일 기준 px²)이 이 값 이상일 때만 검출로 인정 — track_drive.py
#   perc_obstacle_cut_trigger()(B3)가 self.vehicle_detected_yolo_cut/vehicle_max_box_area_cut을
#   가지고 이 값으로 게이트를 건다. B3 전용이라 콘 쪽처럼 나눌 필요는 없어 하나만 둠.
#   [2026-08-23 밤, 요청 반영] b3_vehicle_area_debug 창으로 실차 실측 완료 — 4500.0으로
#   확정(placeholder 아님).
YOLO_VEHICLE_MIN_BOX_AREA_PX_B3 = 4500.0   # B3(방해차량 obstacle_cut) 전용 — 실차 실측 확정치
# [2026-08-23s, 요청 반영] False → True — B3 박스면적 실측 창(b3_vehicle_area_debug)에
#   카메라 화면이 안 보이는(면적 숫자는 찍히는데 프레임이 빈 칸) 문제 확인 후 복원.
DEBUG_VIZ_YOLO_VEHICLE = False      # [2026-08-25, 요청 반영] 대회 실주행 준비 — 모든 디버그창
                                     # 끔(성능 영향 없게). 끄면 _debug_viz_obstacle_cut()의 카메라
                                     # 패널이 B3(방해차량) 단계에서 get_latest_debug_frame()이 항상
                                     # None을 반환해 빈 칸으로 보이지만(yolo_vehicle.py _worker()
                                     # 참고), 위 DEBUG_VIZ_OBSTACLE_CUT도 같이 꺼서 그 창 자체가
                                     # 안 뜨므로 무관 — 실제 B3 YOLO 검출/트리거 로직에는 영향 없음.

# ── 신호등 위치+색상상태 YOLO (perception/yolo_signal_state.py, YOLOv8n ONNX) ──
#   [2026-08-19] datasets/signal_state/(라벨링 워크플로는 그쪽 README 참고)로 파인튜닝한
#   신호등 색상상태(빨강/직진초록/좌회전초록) 검출기 — "지금 어떤 색이 켜져 있는지"를
#   단일 스테이지로 직접 예측한다(배경판 위치 탐지와 색상 판정이 한 모델).
#   [2026-08-21] 이전까지 있던 HSV/Hough Circle 기반 검출(traffic_signal.py/frst.py/
#   yolo_signal.py — 흰 배경판을 먼저 찾고 원 4개의 밝기로 색을 판정하던 방식)과
#   "YOLO 단독 vs YOLO+HSV 하이브리드" 판단 소스 스위치(SIGNAL_USE_YOLO_STATE_FOR_DECISION)를
#   전부 삭제했다(README §1.18) — 이제 신호등 인식은 이 YOLO 모델 하나뿐이다.
YOLO_SIGNAL_STATE_INPUT_SIZE = 640     # signal_state_best_n.onnx export 시 imgsz와 반드시 일치시킬 것
YOLO_SIGNAL_STATE_CONF_THRESHOLD = 0.7 # 이 신뢰도 이상인 검출만 인정(모델이 nms=True로 export돼 좌표 디코딩은 불필요)
# [2026-08-25, 요청 반영] 0.7 → 0.65 → 0.7로 원복
# [2026-08-23] 0.5→0.8(요청 반영). [2026-08-23e, 요청 반영] 0.8→0.5로 다시 원복 —
#   green_left가 green_straight보다 평균 신뢰도가 낮게 나오는 것으로 보여, 0.8에서는
#   green_left만 유독 문턱을 못 넘어 못 잡히고 green_straight/red만 통과하는 쪽으로
#   편향됐다는 게 실차에서 의심됨(SIG_CONFIRM_FRAMES 재상향과 같이 묶어서 결정,
#   README §1.19k 참고). 단발 오검출 방어는 이제 여기(신뢰도)가 아니라 아래
#   SIG_CONFIRM_FRAMES(여러 프레임 연속 확인)가 전담한다.
# [2026-08-23s, 요청 반영] 0.5→0.8로 다시 올림 — ★위 §1.19k에서 green_left가 이 값에서
#   편향돼 안 잡혔던 이력이 있으니, 실차에서 좌회전 신호만 유독 안 잡히면(직진만 계속
#   확정되고 좌회전은 안 뜨면) 이게 원인일 가능성이 높다 — 그때는 다시 0.5로 낮추거나,
#   클래스별로 다른 문턱을 두는 방향을 고려할 것.
# [2026-08-20 §2.59] target_vehicle과 같은 문제(ultralytics nms=True가 CoreML 전용이라
# DetectionModel ONNX export엔 안 먹힘)가 여기서도 재현돼 v1.2.0으로 교체 — 가중치는
# v1.1.0과 완전히 동일, export만 NMS 내장 커스텀 스크립트로 바뀜(output0 [1,N,6] 유지,
# 이 파일의 파싱 코드는 원래부터 그 형식 전제라 변경 없음).
YOLO_SIGNAL_STATE_MODEL_PATH = None    # None이면 yolo_ros/signal_state_best_n.onnx(형제 디렉터리)를 자동으로 찾음(perception/yolo_signal_state.py 참고)
YOLO_SIGNAL_STATE_CLASS_NAMES = ('red', 'green_straight', 'green_left')  # class id(0/1/2) 순서 — datasets/signal_state/classes.txt와 반드시 일치시킬 것
# [2026-08-24] YOLO_CONE_MIN_BOX_AREA_PX_B1/_B2, YOLO_VEHICLE_MIN_BOX_AREA_PX_B3와 동일한
#   목적/패턴 — 클래스별(red/green_straight/green_left) 최고신뢰도 박스의 면적(px², 640
#   입력 스케일)이 이 값 초과일 때만 검출로 인정한다(yolo_signal_state.py detect() 참고).
#   실차 실측 확정치(요청 반영, 클래스 구분 없이 단일값).
YOLO_SIGNAL_MIN_BOX_AREA_PX = 1500.0

SAFETY_DIST      = 5.0        # B2(고정장애물) 발동 거리(m)
OVERTAKE_TRIGGER = 6.5        # B3(방해차량) 발동 거리(m)
VEHICLE_TRIGGER_FRAMES = 5    # 라이다 단독검출 연속 N프레임이면 B3_VEHICLE 진입 확정
SIG_CONFIRM_FRAMES = 5        # [2026-08-25, 요청 반영] 10→5 — 실차에서 10프레임 연속을
#   못 채우고 계속 0으로 리셋되는 게 관찰돼(confirm 자체가 거의 안 됨) 완화.
# [2026-08-23i] 100(5s)→10(20Hz 기준 0.5s)로 재조정 —
#   100은 오검출엔 강하지만 확정까지 체감이 너무 느려짐, 그렇다고 §1.19j/§1.19k에서
#   문제가 됐던 1(단발 오검출에 바로 걸림)로는 안 돌아가고 그 중간값으로 재시도.
#   신호등(직진/좌회전) 판정이 연속 N프레임 유지돼야 확정. 실차 관찰 결과 "연속된
#   프레임으로 보면 검출이 꽤 정확한데, 순간적으로 한 프레임만 끊어 보면 green_straight/
#   green_left가 동시에 뜨는 등 오검출이 섞이고, 시간이 좀 지나야 안정된다"는 게
#   확인돼(§1.19j/§1.19k의 단발 오검출과 같은 맥락) 1→3→10→200→100→10 순으로 조정됨.
#   이 값은 사용자 지시 없이 임의로 바꾸지 말 것(이전에 그렇게 했다가 문제가 됐음).
#   [주의] 이 카운터는 "연속" 프레임 기준이라(perc_signal() 참고, 한 프레임이라도
#   신호가 꺼지면 즉시 0으로 리셋) N=200처럼 크면 안정화 이후에도 아주 가끔 섞이는
#   단발 미검출 한 번에 처음부터 다시 세야 한다 — 확정까지 체감 시간이 꽤 늘어날 수
#   있으니, 실차에서 "확정이 너무 안 된다"고 느껴지면 이 카운터를 "최근 N프레임 중
#   비율"(연속이 아니라 sliding-window 다수결) 방식으로 바꾸는 것도 고려할 것.
SIGNAL_YOLO_OFF_HOLD_FRAMES = 10  # [2026-08-23b, 요청 반영] 신호 확정 후 신호등 YOLO를
#   즉시 끄지 않고 이 프레임 수(20Hz 기준 10=0.5s)만큼 더 돌리고 나서 끈다 — 확정과
#   동시에 꺼버리면(원래 동작) 특히 좌회전은 같은 틱에 S0_SIGNAL로 전환되면서
#   _change_state()가 confirmed/on/cnt를 그 자리에서 리셋해버려, YOLO_신호등 디버그창/
#   left_turn_debug에서 "확정"을 육안으로 확인할 틈도 없이 검출이 사라져 버렸다(perc_signal()
#   참고). FSM 전환 자체는 확정되는 틱에 이미 끝나므로 이 값은 순수 디버그 가시성 +
#   추론 지속시간 트레이드오프용 — 늘리면 확정 후에도 그만큼 더 오래 불필요한 추론이 돈다.

# [2026-08-13] 좌/우 차선 공간(left_clear/right_clear, perc_obstacle())이 회피 방향을
#   정하는 choose_side()에 직접 쓰이는데, 매 프레임 라이다 점 개수를 임계값과 그냥
#   비교만 한 순간값이라 디바운스가 없었다 — 점 개수가 임계값(LEFT/RIGHT_BLOCK_TH) 근처에서
#   흔들리면 회피 방향을 정하는 그 한 프레임에 우연히 어느 쪽이 찍히느냐로 결과가 갈릴 수
#   있었다(신호등의 signal_straight_on과 같은 종류의 문제인데 여긴 디바운스가 없었음).
#   SIG_CONFIRM_FRAMES와 동일한 패턴이되, 방향은 비대칭이다 — "비었다(clear)"는 N프레임
#   연속 유지돼야 확정(성급하게 빈 걸로 오판하면 위험), "막혔다(blocked)"는 한 프레임만
#   막혀도 즉시 카운터 리셋(막힘 쪽으로는 빨리 반응해도 손해가 적음). 실차 미검증 첫 추정치.
SIDE_CLEAR_CONFIRM_FRAMES = 3

# [2026-08-11] 정적/동적 분류는 여전히 Phase(순차 미션 설계)가 기준이다 — 실시간으로
#   바꿔타지 않는다(대회 규정상 라바콘→고정장애물→방해차량이 트랙 위 고정 순서/구간이라
#   원래 이걸로 충분하다는 전제). 대신 self.obstacle_rate(라이다 접근율, 이미 계산됨)로
#   Phase 가정과 실측이 어긋나는지 로그만 남긴다(_cross_check_obstacle_motion()) — 실차
#   미검증 임계값이라 처음엔 느슨하게 잡음.
OBSTACLE_STATIC_SPEED_TH_MPS = 0.3   # |v_mps + obstacle_rate|가 이 미만이면 '정지'로 봄

# ── 장애물회피(TargetPassing, controller/obstacle_avoidance.py) ──
# [2026-08-11] PASS_OFFSET 실측값 반영: 기존 100px는 "차선 폭 실측 후 교체 예정"이라 주석 붙어있던
#   placeholder였다. LANE_WIDTH_M=0.4m(실측, §6.1)를 DL_PIXELS_PER_METER=200px/m로 환산 —
#   PP_WHEELBASE_PX(config.py 위쪽)가 같은 방식으로 "계산은 오프라인에서 해두고 리터럴을 남기는"
#   패턴이라 그걸 따랐다(런타임에 다른 상수를 참조하면 이 파일 안에서 정의 순서에 묶이게 됨).
PASS_OFFSET = 80.0            # = LANE_WIDTH_M(0.4m) * DL_PIXELS_PER_METER(200px/m), 실측 기반
CENTER_DEADZONE_M = 0.12     # 타겟 횡중심이 이 값(m) 이내면 '정면'으로 보고 방향을 다른 근거로 정함
CLEAR_FRAMES_TO_RETURN = 6   # 타겟이 안 보이는 상태가 이만큼 연속되면 복귀 시작
SWITCH_FRAMES = 8            # 주행 타겟이 내 진행쪽으로 넘어온 상태가 이만큼 지속되면 방향 전환
LATERAL_ALPHA_OUT = 0.12     # 옆차선 이동 수렴 속도
LATERAL_ALPHA_BACK = 0.16    # 복귀 수렴 속도 — 90cm 규정 때문에 늑장 부리면 차선이탈, OUT보다 빠르게
LATERAL_DONE_PX = 8.0        # 이 이하로 좁혀지면 이동 완료로 판정
MIN_GAP_M = 0.6              # 추돌 방지 종방향 간격(m) — 이보다 가까우면 횡이동 끝날 때까지 속도를 죽임

# ── Hybrid A*(planner/) B2/B3 공용 — 차량 풋프린트 ──
#   [2026-08-11] 기존 planner/hybrid_astar.py는 vehicle_width=0.45/vehicle_length=0.70을
#   하드코딩하고 있었는데, 실측값(VEHICLE_WIDTH_M=0.31, 아래 VEHICLE_LENGTH_M=0.64)과
#   다른 추정치였다 — 충돌검사가 실제 차체보다 큰 여유를 이미 넣은 셈이라 위험하진
#   않았지만, "실측했다"는 착각을 막기 위해 실측값 + 명시적 마진으로 교체한다.
ASTAR_VEHICLE_MARGIN_M = 0.05  # 설계값(미검증) — 실측 풋프린트에 더할 편도 여유(각 변)

# ── Hybrid A* B3(방해차량, 동적) 대안 (USE_HYBRID_ASTAR_FOR_B3=True일 때만) ──
#   B2와 달리 타겟이 움직이므로 "1회 계획 후 재사용"이 아니라 "그리드는 매틱, 전체
#   재탐색은 트리거 기반"으로 다르게 설계했다 — track_drive.py _handle_overtake_astar() 참고.
USE_HYBRID_ASTAR_FOR_B3 = False
ASTAR_B3_REPLAN_TICKS = 4       # 20Hz 기준 0.2s — 이 주기마다 최소 한 번은 전체 재탐색
ASTAR_B3_FAIL_GRACE_TICKS = 3   # 탐색 실패가 이 틱 연속되면 TargetPassing으로 폴백

# ── 정지선(perception/perc_floor.py check_stopline(), 백엔드 무관 항상 사용) ──
STOPLINE_WHITE_LOW = 180        # 그레이스케일 흰색 임계
STOPLINE_WHITE_RATIO_TH = 0.06  # ROI 내 흰 픽셀 비율 임계 (실측: 1000/16500 ≈ 6%)

# ── 정지선 접근/이탈 판정 (track_drive.py) ──
# [2026-08-22g] SHORTCUT_MIN_T/MAX_T/VISION_CUTOFF_T 삭제 — S3_SHORTCUT state 자체가
# 없어지며(config.py "미션 상태 Enum"/"좌회전 공통" 절 참고) 이 셋의 유일한 소비자였던
# _s3_shortcut()/_shortcut_end()가 함께 삭제됐다.
# [2026-08-20] 이름을 STOPLINE_COOLDOWN → SIGNAL_REENTRY_COOLDOWN으로 변경 — S0_SIGNAL
#   재진입 트리거가 정지선에서 신호등 보드 인식(signal_board_confirmed)으로 바뀌면서
#   (track_drive.py perc_signal()/_s1_lane_follow() 참고), 이 쿨다운도 "정지선 재감지"가
#   아니라 "신호등 보드 재감지" 무시 시간이 됐다. 값(3.0s)은 그대로 유지.
SIGNAL_REENTRY_COOLDOWN = 3.0  # 상태 복귀 후 이 시간(s)간 신호등 보드 재감지 무시(따다닥 전환 방지)

# ── 인식 끊김 보상 / 교차로 근처 기동 금지 ──
OBSTACLE_HOLD_T = 0.6                   # 마지막 관측 후 이 시간(s)까지는 장애물이 있다고 본다
MANEUVER_BLOCK_AFTER_STOPLINE_T = 2.0   # 정지선을 최근에 봤으면 이 시간(s) 동안 회피/추월 기동 금지

# ── 단위 환산 상수 — 실측 후 값만 채울 것 ──
#   지금 코드에는 '모터 단위'(drive()가 ±100으로 클립하는 값)와 '미터'가 섞여
#   있다. 아래 값이 0.0이면 아직 미실측 상태라는 뜻이고, 거리 기반 로직은
#   보수적으로 동작한다.
# [2026-08-17~18 재실측] 줄자+시간차 실측 도구(measure_speed_calibration.py, track_drive/
#   실제속도측정.md)로 speed=5(3회 평균 0.427m/s)·10(4회 평균 0.860m/s)·15(3회 평균 1.264m/s,
#   전부 회귀 기반 steady-state 추정치)를 재측정한 결과, unit≈0.085 안팎으로 3개 속도가 거의
#   완벽히 선형이고 데드존도 사실상 0에 가까웠다 — 옛 2점(5,10) 회귀의 "0.1347/데드존≈1.4"
#   추정과 크게 다름(옛 값은 실제보다 약 58% 과대추정하고 있었다). 아래 값(0.0848)은 이
#   5/10/15 3점을 "명령속도 0→실속도 0"(위에서 확인된 데드존 없음 반영) 조건으로 원점고정
#   회귀한 것 — 지금 활성 프리셋이 SPEED_NORMAL=15라 실제 주행이 거의 항상 이 구간 안에서
#   일어나므로, 이 구간에서 가장 정확하도록 잡았다.
#   ★주의 — speed=20 이상은 이 상수를 쓰지 말 것★ speed=20 반복측정은 훨씬 불안정했고
#   (VESC=모터축 회전 기준값은 run마다 거의 고정인데 지상속도만 들쭉날쭉 — 견인력 순간손실/
#   슬립으로 진단, 실제속도측정.md §4.7), 이상치를 걸러낸 32구간 풀링 재분석 결과 대표값
#   1.449m/s(unit≈0.072)로 5~15의 평평한 구간보다 15~16% 낮다 — speed=20 부근부터 이미
#   비선형(견인력 포화) 구간에 들어섰을 가능성이 있어, 이 상수(5~15 전용)를 그대로 곱하면
#   20 이상에서는 실제보다 차가 더 빨리 간다고 과대추정하게 된다. pp_tune_gridsearch.py처럼
#   speed_norm 20 이상을 다루는 곳은 이 한계를 감안할 것 — 단일 선형식으로 표현 안 되는
#   구간이라 별도 처리(비선형 모델 또는 구간별 상수)가 필요할 수 있다. speed=25는 실차
#   하드웨어 신뢰성 문제(배터리/ESC/모터 의심, 실제속도측정.md §0.1)로 측정 보류 중 — 그
#   문제 해결 후 5점(5/10/15/20/25) 재회귀 예정.
#   VESC(v_mps) 실측 대조 결과 항상 실제보다 5~11% 높게 보고(방향은 일관, 크기는 변동,
#   VESC_SPEED_TO_ERPM_GAIN 재보정 여부 미결정). 원본 로그: track_drive/speed_calib_logs/,
#   상세 분석: track_drive/실제속도측정.md. 옛 도출 과정(2점 회귀)은 README.md "6.5 속도
#   단위 ↔ m/s 환산" 참고(갱신 예정).
METERS_PER_SPEED_UNIT = 0.0848   # 모터 속도단위 1당 m/s — 실측 3점(5~15) 원점고정 회귀. 20 이상엔 쓰지 말 것(위 주석 참고)
LANE_WIDTH_M          = 0.4   # 실측(2026-08-04): 흰선-흰선(도로 전체폭) 80cm, 노란선 정중앙 확인 → 차선 1개 폭 = 40cm
PIXELS_PER_METER      = 0.0   # BEV 픽셀 ↔ 미터 환산(전역) — 미실측. DL_USE_BEV가 검증돼 기본 전환되면 DL_PIXELS_PER_METER로 채울 것
VEHICLE_WIDTH_M       = 0.31  # 실측(2026-08-04): xycar 본체 가로 31cm (세로64cm×가로31cm×높이20cm)
VEHICLE_LENGTH_M      = 0.64  # 실측(2026-08-04): xycar 본체 세로(전후) 64cm — 위와 동일 실측, 그동안 config 상수로는 안 쓰였음
# 각폭 분류 임계 — 이 폭 이상이면 '차량', 미만이면 '고정장애물'.
#   실측(2026-08-04): 고정장애물(고장난 차량) 가로20cm×세로41cm×높이16cm,
#   방해차량 가로28cm×세로54cm×높이19cm → (0.20+0.28)/2 = 0.24
OBSTACLE_VEHICLE_WIDTH_M = 0.24

# #############################################################
# 8. Pure Pursuit + 코너감속 튜닝 프리셋 (pp_tune_gridsearch.py, 2026-08-17~18 전면 갱신)
# #############################################################
#   [2026-08-18 전면 교체] 아래 이전 버전(speed15/speed25, 22개 파라미터 조인트
#   랜덤서치 4000샘플 기반)을 폐기하고, 훨씬 확장된 새 시뮬레이터 결과로 교체한다 —
#   git 이력에 이전 버전이 그대로 남아있으니 되돌릴 일이 있으면 거기서 복원할 것.
#   달라진 점:
#     - 실제 트랙 도면 실측 곡률(코너 R1.95m, 지그재그 4원호) 반영한 '실전트랙'
#       시나리오 + 모드 전환구간 채점 추가.
#     - PATH_EMA_ALPHA(경로 스무딩)/DL_STABLE_FRAME_MIN·DL_STABLE_JUMP_MAX(코너
#       프리뷰 디바운스)까지 포함해 파라미터 22개 → 25개로 확장.
#     - METERS_PER_SPEED_UNIT을 실측(0.1347→0.0848, §7 주석 참고)으로 재보정한
#       뒤 재탐색 — 물리 자체가 바뀌어서 이전 버전과 점수가 직접 비교되지 않음.
#     - 랜덤서치 대신 Optuna(TPE) 사용, 이전 탐색 결과를 시드로 이어받으며 진행.
#     - 10/12.5/15/17.5/20/22.5/25 7개 속도 전부 개별 탐색(이전엔 15/25 2개뿐).
#   진행 기록/원시 로그는 track_drive/새로운_조향테스트.md 참고.
#
#   ★★★ 실차 완전 미검증 — 반드시 서행 가능한 곳에서 즉시 개입 준비하고 켤 것 ★★★
#   시뮬레이션이 가정한 트랙 곡률/세그멘테이션 노이즈(1.5px, 아직 실측 전)/노이즈-VESC
#   상관관계 등에 의존하므로 그대로 믿지 말 것. 특히 아래 두 가지는 **과거에 실차에서
#   문제를 일으켜 이미 한 번 되돌려진 값과 같은 구간으로 다시 수렴**했다는 걸 알고 켤 것:
#     - `PP_WHEELBASE_PX`(전 속도 48.8~50.0) — 이전에 47.36이었다가 "술 취한 듯한 좌우
#       진동"(카카오톡 영상 2026-08-17 16:31)으로 25.0으로 되돌려졌던 것과 거의 같은
#       크기. 이번엔 실전트랙(실측 곡률)+더 넓은 탐색으로 독립적으로 재확인된 값이라
#       방향성은 신뢰할 만하지만, 진동 재현 여부는 반드시 실차에서 최우선 확인할 것.
#     - `SPEED_ACCEL_STEP`(전 속도 1.08~1.58) — 과거 1.014(LVC 배터리 트립, 카카오톡
#       로그 2026-08-17 15:38)로 인해 0.4로 되돌려진 이력이 있는 구간과 겹친다. 이건
#       시뮬레이터가 아예 모델링하지 않는 영역(배터리 전류)이라 시뮬 점수가 좋다고 해서
#       재발 안 한다는 보장이 없다 — 첫 실차 테스트에서 최우선으로 지켜볼 것.
#     - `ANGLE_RATE_MAX`(조향 변화율 제한)는 이 시뮬레이터가 아직 반영하지 않은
#       상태에서 나온 값이다 — 급커브/지그재그 전환 구간에서 시뮬이 가정한 조향
#       반응이 실제 서보보다 빠를 수 있다(추가 검증 예정).
#   [2026-08-18 배포 직후 수정] `SPEED_CORNER_MIN`이 speed10/12.5/15 3개 프리셋에서
#   그리드서치 원값 그대로 `SPEED_NORMAL`보다 커서(예: speed15는 15.77 vs 15.0)
#   `max(SPEED_CORNER_MIN, ...)` 공식상 코너감속 경로가 완전히 죽어있던 버그를 실차
#   테스트로 발견 — §0.5.10과 동일 실패모드. 세 프리셋 다 `SPEED_NORMAL*0.7`로 완화
#   (아래 dict 안 해당 값 옆 주석 참고). 이 그리드서치(pp_tune_gridsearch.py)의 탐색
#   공간이 SPEED_CORNER_MIN을 SPEED_NORMAL과 무관하게 독립 샘플링해서 생긴 문제라,
#   재발 방지하려면 그 스크립트에 "speed_corner_min < speed_norm" 제약을 추가하는 게
#   근본 해결책 — 아직 미반영.
#
#   사용법: 아래 PP_TUNE_ACTIVE_PRESET을 None(기존 개별 값 그대로) 또는 7개 속도
#   프리셋 중 하나로 바꾸고 colcon build 후 실차에서 테스트. 프리셋이 활성화되면
#   이 파일 위쪽의 개별 PP_*/SPEED_NORMAL/SPEED_CORNER_MIN/CORNER_*/PATH_EMA_ALPHA/
#   DL_STABLE_FRAME_MIN/DL_STABLE_JUMP_MAX 값을 전부 덮어쓴다(아래 globals().update()
#   — 이 파일 맨 끝에 있어야 나중 정의가 안 덮어씀).
PP_TUNE_PRESETS = {
    'speed3': dict(
        # [2026-08-18] 커밋 0b365ee(2026-08-17, "[Pure Pursuit] 직진모드 curvature 문턱(eps)
        # 너무 타이트하던 문제 완화")의 값을 그대로 옮김 — 사용자가 이 커밋 시점 파라미터로
        # 주행이 "엄청 잘 됐다"고 확인. 그 커밋의 실제 SPEED_NORMAL은 3.0이었다(사용자가
        # "speed5"로 기억했으나 확인 결과 3.0 — 프리셋 이름/SPEED_NORMAL은 실측값 그대로
        # 'speed3'로 둠, 혼동 방지). 다른 프리셋과 달리 그리드서치 결과가 아니라 그 시점의
        # 수동 튜닝값을 그대로 옮긴 것 — PP_LD_FLOOR_PX/PP_LOOKAHEAD_MIN_PX는 그 커밋 당시
        # 이름(PP_MIN_LOOKAHEAD_PX/PP_LOOKAHEAD_MIN_PX)에서 이후 리네이밍(2026-08-18,
        # 9bfb67f)을 반영해 PP_LD_FLOOR_PX로만 이름을 맞췄고 값 자체는 그대로다.
        PP_LOOKAHEAD_BASE_PX=90.0, PP_LOOKAHEAD_SPEED_GAIN=4.0, PP_LOOKAHEAD_MAX_PX=190.0,
        PP_WHEELBASE_PX=25.0, PP_ALPHA=0.8, PP_LD_FLOOR_PX=90.0, PP_DX_DEADZONE_PX=6.0,
        PP_LOOKAHEAD_CURVATURE_GAIN=100.0, PP_LOOKAHEAD_MIN_PX=40.0,
        SPEED_CORNER_MIN=5.0, CORNER_SIGN_EMA_ALPHA=0.15, LANE_LOOKAHEAD_REF=220.0,
        SPEED_ACCEL_STEP=0.4, CORNER_HOLD_DECAY_LO=0.92, CORNER_HOLD_DECAY_HI=0.97,
        CORNER_MIN_RADIUS_PX=250.0, CORNER_MIN_SPEED_SCALE=0.35,
        PATH_EMA_ALPHA=0.25, DL_STABLE_FRAME_MIN=2, DL_STABLE_JUMP_MAX=20,
        SPEED_NORMAL=3.0,
    ),
    'speed10': dict(
        PP_LOOKAHEAD_BASE_PX=78.61, PP_LOOKAHEAD_SPEED_GAIN=1.476, PP_LOOKAHEAD_MAX_PX=263.7,
        PP_WHEELBASE_PX=49.39, PP_ALPHA=0.7678, PP_LD_FLOOR_PX=63.26, PP_DX_DEADZONE_PX=1.626,
        PP_LOOKAHEAD_CURVATURE_GAIN=446.4, PP_LOOKAHEAD_MIN_PX=41.66,
        # [2026-08-18 배포 직후 수정] 그리드서치 원값 14.16이 SPEED_NORMAL(10.0)보다 커서
        # max(SPEED_CORNER_MIN, ...) 공식상 코너감속 경로가 완전히 죽어있었다(§0.5.10과
        # 동일 실패모드, 실차 테스트로 재현됨 — "속도 5 고정" 증상은 이거 때문이 아니라
        # LL_DEGRADED/LANE_STALE/AVOID_HOLD_BLOCKED 캡이 원인이었지만, 이 버그 자체는
        # 별개로 진짜였음). SPEED_NORMAL*0.7로 완화.
        SPEED_CORNER_MIN=7.0, CORNER_SIGN_EMA_ALPHA=0.6406, LANE_LOOKAHEAD_REF=449.4,
        SPEED_ACCEL_STEP=1.562, CORNER_HOLD_DECAY_LO=0.9363, CORNER_HOLD_DECAY_HI=0.9278,
        CORNER_MIN_RADIUS_PX=582.4, CORNER_MIN_SPEED_SCALE=0.2266,
        PATH_EMA_ALPHA=0.4719, DL_STABLE_FRAME_MIN=10, DL_STABLE_JUMP_MAX=43.24,
        SPEED_NORMAL=10.0,
    ),
    'speed12_5': dict(
        PP_LOOKAHEAD_BASE_PX=78.44, PP_LOOKAHEAD_SPEED_GAIN=1.808, PP_LOOKAHEAD_MAX_PX=180.1,
        PP_WHEELBASE_PX=49.17, PP_ALPHA=0.2258, PP_LD_FLOOR_PX=52.63, PP_DX_DEADZONE_PX=10.56,
        PP_LOOKAHEAD_CURVATURE_GAIN=490.4, PP_LOOKAHEAD_MIN_PX=34.59,
        # [2026-08-18 배포 직후 수정] 위 speed10과 동일 버그(14.17 > SPEED_NORMAL 12.5) —
        # SPEED_NORMAL*0.7로 완화.
        SPEED_CORNER_MIN=8.75, CORNER_SIGN_EMA_ALPHA=1.0, LANE_LOOKAHEAD_REF=405.9,
        SPEED_ACCEL_STEP=1.084, CORNER_HOLD_DECAY_LO=0.8675, CORNER_HOLD_DECAY_HI=0.9038,
        CORNER_MIN_RADIUS_PX=460.3, CORNER_MIN_SPEED_SCALE=0.126,
        PATH_EMA_ALPHA=0.5028, DL_STABLE_FRAME_MIN=7, DL_STABLE_JUMP_MAX=10.55,
        SPEED_NORMAL=12.5,
    ),
    'speed15': dict(
        # [2026-08-18 재그리드서치] 이전 값(BASE=81.12/GAIN=1.225/MAX=278.0/WHEELBASE=49.97/
        # ALPHA=0.9349 등) 전체를 재실행 결과로 교체(요청 반영). 이번 원값은
        # SPEED_CORNER_MIN=13.34 < SPEED_NORMAL(15.0)을 그대로 만족해 위 speed10/12.5
        # 프리셋에서 겪은 "그리드서치 원값이 SPEED_NORMAL보다 커서 코너감속이 no-op"
        # 버그가 재현되지 않았다 — SPEED_NORMAL*0.7 클램프 불필요, 그리드서치 원값 그대로 적용.
        # 실차 재검증 전.
        # [2026-08-19] BASE_PX를 "speed=0 기준"에서 "SPEED_NORMAL(=12.0) 기준"으로 재해석
        #   (요청 반영, 위 전역 PP_LOOKAHEAD_SPEED_ANCHOR 주석 참고) — 실동작은 그대로 유지한 채
        #   숫자만 다시 계산: 구 BASE_PX(110.38) + GAIN(3.01)*ANCHOR(12.0) = 146.50.
        #   즉 speed=SPEED_NORMAL(12)일 때 speed_lookahead_px는 여전히 146.5로 그리드서치
        #   원값과 동일 — PP_LOOKAHEAD_SPEED_ANCHOR를 12.0으로 같이 지정해야 이 등가성이 성립한다.
        # [2026-08-25, 요청 반영] 상태별 구간 분리(_new_tuning_active/PP_*_FINAL) 되돌림 —
        #   BASE_PX/WHEELBASE_PX/CURVATURE_GAIN/MIN_PX/BOOST_GAIN_PER_DEG 5개를 다시 여기
        #   하나로 합쳐 B1(라바콘) 이외 전 구간(B2/B3 포함)에서 항상 쓴다.
        PP_LOOKAHEAD_BASE_PX=220, PP_LOOKAHEAD_SPEED_GAIN=3.01, PP_LOOKAHEAD_MAX_PX=265.4,
        PP_LOOKAHEAD_SPEED_ANCHOR=12.0,
        PP_WHEELBASE_PX=30, PP_ALPHA=0.9, PP_LD_FLOOR_PX=120.19, PP_DX_DEADZONE_PX=5,

        PP_LOOKAHEAD_CURVATURE_GAIN=70, PP_LOOKAHEAD_MIN_PX=140,
        SPEED_CORNER_MIN=10.0, CORNER_SIGN_EMA_ALPHA=0.15, LANE_LOOKAHEAD_REF=220.0,
        SPEED_CORNER_STEER_GAIN=0.50,
        SPEED_ACCEL_STEP=0.8, CORNER_HOLD_DECAY_LO=0.92, CORNER_HOLD_DECAY_HI=0.97,
        CORNER_MIN_RADIUS_PX=250.0, CORNER_MIN_SPEED_SCALE=0.35,
        PATH_EMA_ALPHA=0.7, DL_STABLE_FRAME_MIN=1, DL_STABLE_JUMP_MAX=37.44,
        SPEED_NORMAL=12.0, #직진 잘한 상태
        # [2026-08-19] 조향각 wheelbase 부스트(요청 반영) — "speed15 프리셋일 때만 적용"이라
        # 여기(=speed15 프리셋)에만 켜서(ENABLE=True) 넣는다. 다른 프리셋으로 바꾸면 이 키가
        # 아예 없어서 top-level 기본값(ENABLE=False)이 그대로 남아 자동으로 꺼진다.
        # [2026-08-19 재수정] 문턱각(구 PP_WHEELBASE_BOOST_ANGLE_TH_DEG=15.0) 없이 각도 0부터
        # 연속 비례하는 방식으로 바꾸면서(요청 반영, config.py 위 PP_WHEELBASE_BOOST_* 주석
        # 참고) 이 프리셋에 있던 GAIN_PER_DEG=0.15도 top-level 기본값(0.03)과 같은 값으로
        # 재조정했다 — 0.15를 문턱 없이 그대로 쓰면 (1.5-1)/0.15≈3.3°만 넘어도 MAX_SCALE에
        # 도달해 사실상 상시 최대 부스트가 걸린다(요청("미미할 땐 작게")과 어긋남). 순전히
        # 추정치, 실차에서 체감보고 재조정할 것.
        PP_WHEELBASE_BOOST_ENABLE=True, PP_WHEELBASE_BOOST_GAIN_PER_DEG=0.17,
        PP_WHEELBASE_BOOST_MAX_SCALE=2.75,
    ),
    'speed17_5': dict(
        PP_LOOKAHEAD_BASE_PX=82.53, PP_LOOKAHEAD_SPEED_GAIN=1.162, PP_LOOKAHEAD_MAX_PX=253.4,
        PP_WHEELBASE_PX=45.0, PP_ALPHA=0.5642, PP_LD_FLOOR_PX=56.15, PP_DX_DEADZONE_PX=1.694,
        PP_LOOKAHEAD_CURVATURE_GAIN=478.8, PP_LOOKAHEAD_MIN_PX=30.73,
        SPEED_CORNER_MIN=15.61, CORNER_SIGN_EMA_ALPHA=0.4682, LANE_LOOKAHEAD_REF=314.7,
        SPEED_ACCEL_STEP=1.518, CORNER_HOLD_DECAY_LO=0.8894, CORNER_HOLD_DECAY_HI=0.9731,
        CORNER_MIN_RADIUS_PX=551.4, CORNER_MIN_SPEED_SCALE=0.06394,
        PATH_EMA_ALPHA=0.6743, DL_STABLE_FRAME_MIN=4, DL_STABLE_JUMP_MAX=32.57,
        SPEED_NORMAL=17.5,
    ),
    'speed20': dict(
        # [2026-08-18 재그리드서치] 이전 값(BASE=80.62/GAIN=1.353/MAX=243.3/WHEELBASE=49.36/
        # ALPHA=0.7727 등, SPEED_CORNER_MIN=5.818) 전체를 재실행 결과로 교체(요청 반영).
        # 새 SPEED_CORNER_MIN=15.83 < SPEED_NORMAL(20.0) 만족 — 그리드서치 원값 그대로 적용.
        # DL_STABLE_JUMP_MAX가 21.91→104.2로 크게 뛰었다는 점은 실차에서 da/ll 튐 허용폭이
        # 훨씬 넓어졌다는 뜻이라 눈여겨볼 것. 실차 재검증 전.
        PP_LOOKAHEAD_BASE_PX=99.6, PP_LOOKAHEAD_SPEED_GAIN=0.8793, PP_LOOKAHEAD_MAX_PX=331.9,
        PP_WHEELBASE_PX=38.19, PP_ALPHA=0.9274, PP_LD_FLOOR_PX=59.4, PP_DX_DEADZONE_PX=3.577,
        PP_LOOKAHEAD_CURVATURE_GAIN=840.7, PP_LOOKAHEAD_MIN_PX=61.31,
        SPEED_CORNER_MIN=15.83, CORNER_SIGN_EMA_ALPHA=0.7536, LANE_LOOKAHEAD_REF=459.4,
        SPEED_ACCEL_STEP=4.678, CORNER_HOLD_DECAY_LO=0.9445, CORNER_HOLD_DECAY_HI=0.9143,
        CORNER_MIN_RADIUS_PX=678.6, CORNER_MIN_SPEED_SCALE=0.261,
        PATH_EMA_ALPHA=0.9288, DL_STABLE_FRAME_MIN=1, DL_STABLE_JUMP_MAX=104.2,
        SPEED_NORMAL=20.0,
    ),
    'speed22_5': dict(
        PP_LOOKAHEAD_BASE_PX=82.36, PP_LOOKAHEAD_SPEED_GAIN=1.23, PP_LOOKAHEAD_MAX_PX=283.5,
        PP_WHEELBASE_PX=49.29, PP_ALPHA=0.7039, PP_LD_FLOOR_PX=56.8, PP_DX_DEADZONE_PX=1.904,
        PP_LOOKAHEAD_CURVATURE_GAIN=442.0, PP_LOOKAHEAD_MIN_PX=46.28,
        SPEED_CORNER_MIN=15.71, CORNER_SIGN_EMA_ALPHA=0.1762, LANE_LOOKAHEAD_REF=246.7,
        SPEED_ACCEL_STEP=1.433, CORNER_HOLD_DECAY_LO=0.8567, CORNER_HOLD_DECAY_HI=0.9191,
        CORNER_MIN_RADIUS_PX=208.0, CORNER_MIN_SPEED_SCALE=0.3496,
        PATH_EMA_ALPHA=0.7615, DL_STABLE_FRAME_MIN=2, DL_STABLE_JUMP_MAX=52.65,
        SPEED_NORMAL=22.5,
    ),
    'speed25': dict(
        # [주의] speed=25는 METERS_PER_SPEED_UNIT 실측이 하드웨어 문제로 아직 안 끝난
        # 속도(§7 주석 참고) — 이 프리셋의 물리 전제 자체가 5개 프리셋 중 가장 약하다.
        PP_LOOKAHEAD_BASE_PX=80.26, PP_LOOKAHEAD_SPEED_GAIN=1.231, PP_LOOKAHEAD_MAX_PX=250.9,
        PP_WHEELBASE_PX=49.99, PP_ALPHA=0.7777, PP_LD_FLOOR_PX=73.1, PP_DX_DEADZONE_PX=1.989,
        PP_LOOKAHEAD_CURVATURE_GAIN=517.9, PP_LOOKAHEAD_MIN_PX=49.07,
        SPEED_CORNER_MIN=13.1, CORNER_SIGN_EMA_ALPHA=0.3186, LANE_LOOKAHEAD_REF=568.9,
        SPEED_ACCEL_STEP=1.57, CORNER_HOLD_DECAY_LO=0.9392, CORNER_HOLD_DECAY_HI=0.9318,
        CORNER_MIN_RADIUS_PX=202.0, CORNER_MIN_SPEED_SCALE=0.137,
        PATH_EMA_ALPHA=0.6929, DL_STABLE_FRAME_MIN=10, DL_STABLE_JUMP_MAX=32.76,
        SPEED_NORMAL=25.0,
    ),
}
# [2026-08-18] 이전 speed15/speed25(22개 파라미터, 구물리값) 프리셋 전면 교체 — 위
# 헤더 주석 참고. 'speed15'를 활성 속도로 선택(요청 반영) — **이전 프리셋도 동일하게
# 실차 미검증이었다는 전제로 선택된 것**이라, 위에서 경고한 PP_WHEELBASE_PX/
# SPEED_ACCEL_STEP/SPEED_CORNER_MIN 재발 위험을 서행 상태에서 최우선 확인할 것.
# 문제 생기면 None으로 되돌리거나 git 이력의 이전 speed15 프리셋으로 복원할 것.
PP_TUNE_ACTIVE_PRESET = 'speed15'   # None / 'speed3' / 'speed10' / 'speed12_5' / 'speed15' / 'speed17_5' / 'speed20' / 'speed22_5' / 'speed25'
if PP_TUNE_ACTIVE_PRESET is not None:
    globals().update(PP_TUNE_PRESETS[PP_TUNE_ACTIVE_PRESET])

# ── B1(라바콘) 전용 Pure Pursuit 상수 (2026-08-24) ──
#   지금까지 라바콘 조향(_lavacon_steer_da_push())은 track_drive.py의 self.pure_pursuit
#   하나를 일반 차선주행과 그대로 같이 썼다 — 즉 위 PP_TUNE_PRESETS['speed15']가 정한
#   PP_LOOKAHEAD_BASE_PX/SPEED_GAIN/SPEED_ANCHOR/CURVATURE_GAIN/MIN_PX, PP_WHEELBASE_PX,
#   PP_ALPHA, PP_LD_FLOOR_PX, PP_DX_DEADZONE_PX, PP_WHEELBASE_BOOST_* (그리고 이 프리셋
#   밖의 전역 기본값인 ANGLE_MAX/PP_LOOKAHEAD_ALPHA)까지 전부 "지금 이 순간의" 값을
#   그대로 물려받고 있었다 — 나중에 speed15 프리셋을 재튜닝하거나 다른 프리셋(speed20 등)
#   으로 바꾸면 그 즉시 라바콘 조향도 같이 바뀌는 구조(요청 반영해 분리하기로 함).
#   아래 _LAVACON 상수들은 오늘(2026-08-24) 기준 speed15 프리셋이 만들어낸 값을 그대로
#   숫자로 박아넣은 스냅샷이다 — 위 프리셋을 갈아끼우거나 재튜닝해도 이 값들은 안 바뀌므로,
#   track_drive.py가 이 상수들만 쓰는 전용 PurePursuitController를 라바콘 조향에 쓰면
#   지금 실차 거동이 그대로 고정된다. 값 자체를 라바콘만 따로 재튜닝하고 싶으면 이
#   블록만 수정할 것 — 위 프리셋과는 이제 완전히 무관.
#   [2026-08-24, 요청 반영] PP_WHEELBASE_PX_LAVACON/PP_WHEELBASE_BOOST_GAIN_PER_DEG_LAVACON
#   두 개만은 "지금" speed15 값(25 / 0.13) 대신 커밋 6840146(그 시점 speed15 프리셋,
#   WHEELBASE=20/GAIN_PER_DEG=0.15) 기준으로 스냅샷했다 — 그 커밋과 지금 speed15 사이에
#   실제로 값이 달랐던 두 항목이 이거라서, 라바콘 조향은 그 커밋 당시 거동을 기준으로 고정.
PP_LOOKAHEAD_BASE_PX_LAVACON          = 180.0
PP_LOOKAHEAD_SPEED_GAIN_LAVACON       = 3.01
PP_LOOKAHEAD_SPEED_ANCHOR_LAVACON     = 12.0
# PP_LOOKAHEAD_MAX_PX_LAVACON은 이미 위(§0.5.10 인근)에서 프리셋과 무관한 전용 상수로
# 정의돼 있어(140.0) 그대로 재사용 — 여기서 다시 선언하지 않는다.
PP_LOOKAHEAD_CURVATURE_GAIN_LAVACON   = 120.0
PP_LOOKAHEAD_MIN_PX_LAVACON           = 80.0
PP_LOOKAHEAD_ALPHA_LAVACON            = 1.0    # 전역 기본값(PP_LOOKAHEAD_ALPHA) 그대로 — speed15가 건드리지 않는 항목
PP_WHEELBASE_PX_LAVACON               = 20.0
PP_ALPHA_LAVACON                      = 0.9
PP_LD_FLOOR_PX_LAVACON                = 120.19
PP_DX_DEADZONE_PX_LAVACON             = 5.0
PP_WHEELBASE_BOOST_ENABLE_LAVACON     = True
PP_WHEELBASE_BOOST_GAIN_PER_DEG_LAVACON = 0.15
PP_WHEELBASE_BOOST_MAX_SCALE_LAVACON  = 2.75
ANGLE_MAX_LAVACON                     = 80.0   # 전역 기본값(ANGLE_MAX) 그대로 — 프리셋이 건드리지 않는 항목
