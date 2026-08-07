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
    S0_WAIT_GREEN   = 0  # 3구 신호등 초록불 대기 후 출발
    S1_LANE_FOLLOW  = 1  # 차선인식 주행 (라바콘·고정장애물·추월 Behavior를 이 상태 안에서 처리)
    S2_INTERSECTION = 2  # 4구 신호등 교차로 (정지→라이다 경로판단→직진/좌회전)
    S3_SHORTCUT     = 3  # 지름길 (직진, 끝에서 좌회전)
    S4_FINISH       = 4  # 종료

class BehaviorState(Enum):
    B0_NORMAL   = 0  # Mission(차선주행) 출력 그대로
    B1_LAVACON  = 1  # 라바콘 구간 주행 (Phase.LAVACON일 때, 좌우 라이다 클러스터 동시검출 트리거로 활성)
    B2_OBSTACLE = 2  # 고정장애물 회피 (Phase.FIXED_OBSTACLE일 때, 감지 시 활성)
    B3_VEHICLE  = 3  # 방해차량 추월   (Phase.VEHICLE일 때, 감지 시 활성)

# S1(차선주행) 내부 진행 순서 — 순서 고정(라바콘→고정장애물→방해차량→완료), 순차 전용(우선순위 판단 불필요)
class Phase(Enum):
    LAVACON        = 0
    FIXED_OBSTACLE = 1
    VEHICLE        = 2
    DONE           = 3  # 모든 Behavior 미션 완료 — 이후 계속 B0로 일반 차선주행


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
SPEED_NORMAL  = 25.0   # [2026-08-06] 차선주행(S1) 기본(직진) 속도. 8.0 → 25.0로 상향(요청 반영).
                        #   0.0으로 두지 말 것 — _lane_drive()에서 나눗셈 분모로도 쓰여 ZeroDivisionError.
                        #   ★주의★ README §6.5의 METERS_PER_SPEED_UNIT 회귀는 speed=5/10 두 점만 실측한
                        #   것이라 25는 측정 범위 밖(2.5배) 외삽 — 실제 m/s·제동거리·코너 반응이 그
                        #   선형식대로 나올지 실차 재검증 필요. PP_LOOKAHEAD_SPEED_GAIN(=4.0) 등 speed를
                        #   그대로 입력받는 게인들도 최고값이 8→25로 커진 만큼 lookahead가 더 크게 튈 수
                        #   있으니(PP_LOOKAHEAD_MAX_PX=150으로 클램프는 되지만) 같이 관찰할 것.
SPEED_LAVACON = 2.5    # 라바콘 구간 속도
SPEED_STOP    = 0.0
# [2026-08-06] 코너 감속(_lane_drive())의 목표속도 하한 — 원래 SPEED_NORMAL*0.15(=1.2, 두 곳에
#   하드코딩)로 잡혀 있었는데, config.py "6. 단위환산" 절의 실측 근거(정속 회귀선의 절편이
#   음수라 speed≈1.4 미만은 모터 데드존으로 추정)보다 낮다. 실차에서 코너가 계속돼 이 하한까지
#   깎이면 명령은 나가는데 실제로는 거의 안 움직이다 멈추고, 멈추면 조향 대상 lookahead도 안
#   바뀌어 감속이 안 풀리는(=계속 정지) 증상이 재현됨. 데드존(1.4)보다 확실히 위인 3.0으로
#   올렸다가, 이후 5.0으로 재상향(요청 반영) — 데드존 대비 여유를 더 두어 코너에서도 확실히
#   전진하도록 함. SPEED_NORMAL이 25.0으로 오른 만큼 최고/최저 속도 폭이 넓어졌으니, 코너
#   진입/탈출 시 속도 급변이 과하게 느껴지면 이 값을 올리는 쪽으로 완화할 것.
SPEED_CORNER_MIN = 5.0
ANGLE_MAX     = 80.0  # 조향각 클램프(도)
ANGLE_RATE_MAX = 12.0  # 조향 변화율 제한(도/주기, 20Hz 기준 12도/주기=240도/초) — drive()에서 모든 명령에 일괄 적용
#   [2026-08-07] 0.85 → 0.4. 실차에서 ros2 run 직후(정지→출발)와 가속 도중 차량이 갑자기
#   멈췄다 살아나는 증상이 재현됨 — ctrl_speed(spd)는 그동안 계속 25로 발행되고 있었고
#   v_mps(VESC 실측, vesc_debug 초록=LIVE 확인됨)만 순간 0으로 떨어졌다 회복되는 패턴이라
#   판단(FSM) 로직이 아니라 배터리 전압 강하로 인한 ESC/VESC 저전압·과전류 보호(LVC) 트립으로
#   추정된다(정지 직후 재출발·가속 구간 둘 다 모터 전류 요구가 가장 큰 지점과 일치).
#   SPEED_NORMAL이 5→25로 오르며 가속 시 전류 피크도 커졌을 것 — 근본 해결은 배터리 점검/교체지만,
#   가속 램프를 더 완만하게(0→25까지 약 1.5초→약 3.1초, 20Hz 기준) 늘려 전류 피크를 낮추는
#   소프트웨어 완화책으로 우선 적용. 실차 재검증 필요 — 그래도 반복되면 배터리 자체를 볼 것.
SPEED_ACCEL_STEP = 0.4  # 가속 속도제한(주기당 최대 증가량)
CORNER_HOLD_DECAY_LO = 0.92  # 저속 시 코너 hold 감쇠 (빠른 회복)
CORNER_HOLD_DECAY_HI = 0.97  # 고속 시 코너 hold 감쇠 (느린 회복, 연속코너 대응)
# [2026-08-06] 코너 감속 판단용 조향각 signed EMA 계수(_lane_drive()의 self._corner_signal) —
#   pure_pursuit 특유의 좌우 진동("와리가리")이 매 스윙마다 급코너로 오인돼 속도가 팍팍
#   깎이는 문제 대응. 작을수록(더 스무딩) 진동 상쇄 효과는 커지지만 실제 코너 진입 반응은
#   느려진다 — 0.15는 20Hz 기준 시정수 약 0.33초(진동 주기보다 확실히 길게, 실제 코너 진입
#   시간보다는 짧게 잡은 첫 추정치). 너무 작으면 실제 코너 감속이 늦어져 위험할 수 있으니
#   실차에서 진동 주기/코너 반응 둘 다 보며 조정할 것.
CORNER_SIGN_EMA_ALPHA = 0.15
LANE_LOOKAHEAD_REF = 220.0   # 예측감속 최대가 되는 lookahead 편차(px) — _lane_drive() 속도계획용
LAVACON_KP = 210.0           # 라바콘 조향 게인

# ── 코너 진입 시 회전반경 기반 감속 (ROS2 Nav2 Regulated Pure Pursuit 방식) ──
#   회전반경(1/curvature)이 CORNER_MIN_RADIUS_PX보다 작아지면 그 비율만큼 목표속도를
#   깎는다. PIXELS_PER_METER 미실측이라 반경은 픽셀 단위 — 실차 미검증 추정치.
CORNER_MIN_RADIUS_PX = 250.0
CORNER_MIN_SPEED_SCALE = 0.35  # 반경이 0에 가까워져도 속도가 0으로 죽지 않게 하는 하한 배율

# ── 좌회전 공통 (S2→S3 진입, S3→S1 진출) — 전부 실차 튜닝 필요한 임시값 ──
TURN_ANGLE       = -60.0   # [진입] S2 교차로 → S3 지름길 좌회전 조향각
TURN_SPEED       = 15.0    # [진입] 좌회전 속도
TURN_FRAMES      = 40      # [진입] 좌회전 유지 프레임 수 (20Hz 기준)
TURN_EXIT_ANGLE  = -60.0   # [진출] S3 지름길 → S1 차선주행 좌회전 조향각
TURN_EXIT_SPEED  = 15.0    # [진출] 좌회전 속도
TURN_EXIT_FRAMES = 40      # [진출] 좌회전 유지 프레임 수

# ── 정지선 접근 감속 (S1→S2 진입, S3→S1 진출) ──
APPROACH_SPEED      = 2.0  # [진입] 정지선 감지 후 S2 진입 전 감속 속도
APPROACH_TIME       = 1.0  # [진입] 감속 유지 시간(s)
APPROACH_EXIT_SPEED = 2.0  # [진출] S3 탈출 정지선 감지 후 감속 속도
APPROACH_EXIT_TIME  = 1.0  # [진출] 감속 유지 시간(s)

# ── S2 신호 확정 → 물리적 분기 커밋 구간 (실차 튜닝 필요) ──
#   4구 신호등의 ㅓ교차로는 신호가 보이는 지점과 실제 도로가 갈라지는 지점(물리적
#   분기)이 약 1m 떨어져 있고, 그 분기는 직각이 아니라 커브로 열린다. 신호 확정
#   직후 곧장 _lane_drive()(DL da 세그멘테이션 기반 비전 조향)를 켜면, 분기가
#   보이기 시작하는 순간 da 마스크가 반대쪽 갈래까지 넓게 이어붙어 중심선이
#   그쪽으로 끌려간다(실측 재현됨) — 즉 신호로 이미 확정된 진행방향을 비전이
#   뒤집어버리는 상황. 신호가 이미 정답을 알고 있으므로(직진 2/3바퀴 확정, 좌회전
#   1/3바퀴만 등장 — RACE_RULES.md 11절), 이 구간에서는 비전을 아예 참조하지 않고
#   APPROACH_SPEED로 직진(각도 0)만 유지하다가, 이 시간이 지난 뒤에야 실제 분기
#   방향(직진 복귀 or 좌회전 스크립트 시작)을 실행한다. 값은 물리적 1m를
#   APPROACH_SPEED로 주행하는 데 걸리는 시간으로 맞춰야 한다. METERS_PER_SPEED_UNIT은
#   이제 실측값이 있지만(README.md "6.5 속도 단위 ↔ m/s 환산" 참고) speed=5/10 두
#   점으로만 회귀한 값이라 APPROACH_SPEED=2.0 같은 저속 구간(추정 데드존 ≈1.4에
#   가까움)에 그대로 대입하면 못 미덥다 — 그래서 여기 대입해서 자동 계산하지 않고
#   여전히 우선 추정치로 둔다 — 실차에서 분기 진입 타이밍 보고 조정할 것.
S2_COMMIT_T = 1.0


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
DL_USE_BEV = True  # 실차 검증 중(2026-08-05). 검증 전까지는 기본 False로 되돌릴 것

# 실측 픽셀좌표(원본 640×480 프레임 기준, ROI 자르기 전 절대좌표) — TL/TR/BR/BL 순.
# dl_lane.py가 여기서 DL_ROI_Y0만큼 뺀 ROI-상대좌표로 변환해서 쓴다.
DL_BEV_SRC_PX_RAW = np.float32([
    [246, 257],  # TL(좌상/먼왼쪽)
    [455, 257],  # TR(우상/먼오른쪽)
    [635, 333],  # BR(우하/가까운오른쪽)
    [60,  333],  # BL(좌하/가까운왼쪽)
])
DL_PIXELS_PER_METER = 200.0   # 설계값(실측 아님) — 목적 캔버스를 1m=200px 스케일로 만든다.

# [2026-08-06] 원거리 크롭 — da/ll 처리가 근거리 기준점(BL/BR)으로부터 몇 m까지만 보게 할지.
#   캔버스는 "ROI 전체가 여백 없이 들어가도록" 자동 확장되는데(perception/dl_lane.py의
#   DL_BEV_CANVAS_H 계산 참고), 그 결과 실측 캘리브레이션 지점(TL/TR, 1.0m)보다도 더 먼
#   영역(외삽, 실측값 기준 약 1.30m까지)까지 처리에 포함되고 있었다. 이 값을 낮추면 그
#   외삽 영역과 원거리 blur(§2.2에서 다룬 S자 커브 ll 두께 과다검출의 원인)를 처리 대상에서
#   제외할 수 있다.
#   ★주의★ 이건 DL_BEV_SRC_PX_RAW/DL_PIXELS_PER_METER 같은 캘리브레이션 값이 아니다 — 그
#   값들은 그대로 두고 "이미 정확하게 아는 좌표계에서 먼 부분을 그냥 안 본다"는 크롭일
#   뿐이라 스케일 왜곡이 없다. 반대로 이 숫자를 바꾼다고 카메라가 실제로 보는 물리적 거리가
#   바뀌는 것도 아니다(그건 DL_BEV_SRC_PX_RAW 4점의 실측 재측정이 필요 — README §6.3 참고).
#   1.0 → 0.7로 낮춤(요청 반영) — 실차 미검증, DEBUG_VIZ_DL_LANE에서 크롭 경계가 원하는
#   위치에 오는지 확인할 것.
DL_BEV_FAR_LIMIT_M = 0.7

# ── 세그멘테이션 결과에서 좌/우 차선 중심을 뽑을 관심영역 (원본 480행 기준 절대 픽셀, 실차 실측값) ──
DL_ROI_Y0 = 250
DL_ROI_Y1 = 390

DL_FG_THRESHOLD = 0.5   # da 확률맵 이진화 임계값(요구사항에 명시된 값) — da에만 쓴다

# ll(차선) 전용 이진화 임계값 — DL_FG_THRESHOLD보다 높다. [2026-08-06, 실차 관찰] BEV 워프는
#   카메라에서 먼 지점일수록 원근압축을 되돌리려고 더 크게 확대하는데(호모그래피 성질상
#   불가피), da/ll은 이진화 "전"(float 확률맵) 상태로 워프하기 때문에(위 DL_USE_BEV 주석
#   참고, 계단 현상 방지 목적) 모델 출력의 경계 blur(확률이 0.5 근방인 애매한 픽셀들)도
#   그 확대율만큼 같이 늘어난다. 근거리는 원래도 확률이 뚜렷해 큰 영향이 없지만, 원거리는
#   실측 S자 커브 구간에서 ll이 실제 선 두께보다 눈에 띄게 두껍게 잡히는 게 확인됐다 — 두꺼운
#   ll은 _clip_da_by_ll()이 da를 필요 이상으로 깎아내 da가 DL_DA_MIN_COMPONENT_AREA
#   밑으로 떨어지고 그 프레임이 무효 처리되는 원인이 된다. ll은 "차선 있으면 그 위치만
#   보고 자르는" 용도라 da보다 확신이 필요하므로, 임계값을 높여 blur로 번진 저확률
#   가장자리를 미리 잘라낸다. 0.7은 첫 추정치 — DEBUG_VIZ_DL_LANE 오버레이(빨강)로 실제
#   선 두께가 근거리/원거리에서 비슷해지는지 보고 조정할 것.
DL_LL_FG_THRESHOLD = 0.7

# ── SlideWindow moments 재사용 DL 전용 튜닝값 (알고리즘은 lane_util.py의
#   MOMENT_*/LANE_SLICE_*/STABLE_*와 동일, DL은 원본 카메라 프레임 px 스케일이라
#   값만 따로 둔다) — 전부 실차 미검증 튜닝값 ──
DL_N_SLICES = 8               # da 중심선을 세로로 나눌 밴드 수
DL_MIN_PIXELS = 40            # 밴드 내 da 픽셀수가 이 미만이면 그 밴드는 "차선 없음" 처리
DL_NEAR_SLICES = 2            # 근거리(조향용) 편차 계산에 쓸 아래쪽 밴드 수
DL_FAR_SLICES = 2             # 원거리(코너 예측용) 편차 계산에 쓸 위쪽 밴드 수
# [LQR 브랜치 dl_lane_BEV_파라미터_변경사유.md에서 이식, 2026-08-05] 아래 4개 값은
#   원래 BEV 도입 전 원근(perspective) 640×140px ROI 스케일로 잡혔던 값을 그대로
#   들고 있었다. BEV(585×298px 캔버스, DL_PIXELS_PER_METER=200px/m)로 좌표계가
#   바뀌면서 "픽셀당 의미"가 달라졌는데도 재계산이 안 돼 있었던 것 — 반차로폭
#   (LANE_WIDTH_M=0.4m)의 상당 부분(예: 옛 DL_SLICE_OUTLIER_MAX=60px는 반차로폭의
#   75%)까지 "이상치 아님"으로 통과시켜서, 교차로 등에서 da/중심선이 옆 차선으로
#   번지는 걸 걸러내지 못하는 원인 중 하나였다(DL_DA_MAX_AREA_PX로 잡는
#   "면적이 통째로 큰 경우"와는 별개로, 이쪽은 "중심선이 서서히 옆으로 새는" 경우를
#   못 잡는 문제). 아래는 새 BEV 스케일 기준으로 다시 계산한 값 — 여전히 실차 미검증,
#   DEBUG_VIZ_DL_LANE 오버레이로 교차로 진입 구간에서 확인 후 조정할 것.
DL_SLICE_OUTLIER_MAX = 40     # 반차로폭(0.4m=80px)의 1/2. 추세선에서 이 이상(px) 벗어난 밴드는 이상치로 제외
DL_SLICE_FIT_MIN = 3          # 유효 밴드가 이 미만이면 추세 판단 생략
DL_STABLE_FRAME_MIN = 3       # "새 추론이 끝난 시점" 기준 연속 안정 프레임 수(디바운스)
DL_STABLE_JUMP_MAX = 20       # 반차로폭의 1/4. 이 이상(px) 차이나면 새 후보로 취급

# da 파편화 대응 — ConnectedComponents 최대 덩어리 면적이 이 미만이면 "da 안 보임" 처리
#   1560 = 옛 원근 ROI(89,600px²) 대비 800의 비율(0.893%)을 새 BEV 캔버스(174,330px²)에서
#   그대로 유지한 값(89,600→174,330, ×1.9456배) — 위 [LQR 브랜치 이식] 주석 참고.
DL_DA_MIN_COMPONENT_AREA = 1560
# da 과대검출 대응 — 최대 덩어리 면적이 이 절대 픽셀수를 넘으면 정상 자기차선 폭이
# 아니라고 보고 outlier로 버린다(_largest_da_component()가 그 다음으로 큰 덩어리를
# 대신 시도 — "차선책", 위 주석 참고). ㅓ교차로에서 da가 옆 갈림길까지 하나로 이어붙는
# 경우뿐 아니라, 차선(백선)이 아예 없는 맨바닥을 통째로 주행가능영역으로 오검출하는
# 경우도 실측으로 확인됨 — 두 실패모드 모두 정상 대비 면적이 비정상적으로 크다는
# 공통점이 있다.
#   [2026-08-06] 마스크 전체 대비 비율(DL_DA_MAX_AREA_RATIO=0.6, 캔버스 크기 안 타는 장점)
#   방식에서, 실차 직선 구간 da 면적 실측값 기반 절대 픽셀값으로 교체(요청 반영) — "정상
#   직진 구간엔 이 정도"라는 실측 근거가 비율 추정보다 정확하다.
#   실측(2026-08-06, 원거리 크롭 DL_BEV_FAR_LIMIT_M=0.7 적용 후, steer_debug 창 `DA largest:`
#   직선 구간 3프레임): 13349px, 13361px, 12946px(평균 13,219px, 최대 13,361px) — 여기에
#   여유를 두고 13,700으로 설정. 캔버스 크기가 또 바뀌면(원거리 크롭 값 재조정 등) 비율
#   방식과 달리 이 값도 같이 재측정해야 한다.
DL_DA_MAX_AREA_PX = 16000
# ll sanity check — ROI 내 ll(차선) foreground 비율이 이 미만이면 da 결과와 무관하게 무효 처리
DL_LL_SANITY_MIN_RATIO = 0.005
# da가 옆 차선과 이어붙었을 때 ll 라인 바깥(옆 차선 쪽) 픽셀을 잘라내는 여유폭(px)
#   8 = 실측 라인 두께 2.5cm(=5px @200px/m) + 세그멘테이션 경계 흔들림(1~2px) 여유
#   (위 [LQR 브랜치 이식] 주석 참고). 옛 값 15px은 필요 이상으로 넓게 잘라내 정상
#   자기차선 폭까지 깎아내는 부작용이 있었다.
DL_LL_CLIP_MARGIN_PX = 8

# ── [2026-08-06] 밴드별 중심 계산 모드 스위치 ──
#   'da'    : 밴드별 중심을 da(주행가능영역) 무게중심으로만 계산한다(main의 기존 방식).
#   'll_da' : 밴드마다 좌/우 ll(차선)이 둘 다 신뢰할 만하면(아래 DL_LL_SIDE_MIN_PIXELS/
#             DL_LL_WIDTH_MIN_PX~MAX_PX) 그 중점을 우선 채택하고, ll이 부족한(점선 틈/
#             마모/반사/편측 가려짐) 밴드만 da 무게중심으로 폴백한다. da는 "주행 가능한
#             영역"이지 "차로 중앙"이 아니라서, 갓길 등 여백이 넓은 구간에서 da 무게중심이
#             여백 쪽으로 쏠려 경로가 차로 중앙을 벗어나는 문제가 실측으로 확인됐는데,
#             ll(차선 자체)은 여백 크기와 무관하게 "선이 실제로 있는 위치"만 가리키므로 이
#             문제에서 자유롭다. da는 여전히 ll이 끊긴 구간을 메우는 안전망 역할로 남는다.
#   'll'    : 'll_da'에서 da 폴백을 아예 없앤 순수 ll 모드. 밴드마다 좌/우 ll이 둘 다
#             신뢰될 때만(DL_LL_SIDE_MIN_PIXELS/WIDTH_MIN~MAX_PX) 그 중점을 쓰고, 그
#             조건을 못 채우는 밴드는 da로 메우지 않고 그냥 None(무효 밴드)으로 둔다 —
#             da가 섞여 들어와 여백 쪽으로 경로가 쏠리는 걸 완전히 차단하고 싶을 때 쓴다.
#             대가로 ll이 끊기는 구간(반사/마모/점선 틈)에서는 그만큼 유효 밴드가 줄어
#             fit이 더 쉽게 실패한다(_fit_and_sample_path의 DL_SLICE_FIT_MIN 미만이면
#             경로가 갱신 안 되고 직전 값 유지). da 파편화 대응/옆 차선 클리핑/ll sanity
#             check는 이 모드에서도 그대로 적용된다(da_mask 자체는 여전히 클리핑용으로
#             계산됨) — 다만 그 da 결과가 중심점 계산에는 전혀 섞이지 않는다.
#   세 모드 다 da 파편화 대응(_largest_da_component)/옆 차선 클리핑(_clip_da_by_ll)/ll
#   sanity check는 동일하게 적용된다 — 차이는 "밴드별 중심점을 뭘로 뽑는가" 뿐이다.
#   main 기본값은 실차에서 이미 어느 정도 검증된 'da'로 둔다 — 'll_da'/'ll'은 아직 실차
#   미검증이라, 켤 때는 이 값을 직접 바꾸고 A/B 비교 후 재조정할 것.
DL_CENTER_MODE = 'da'  # 'da' | 'll_da' | 'll'

# DL_CENTER_MODE='ll_da'/'ll'일 때 쓰는 ll 중점 채택 임계값.
# 밴드 내 ll 픽셀수가 이 미만이면(좌/우 각각 판정) "이 밴드는 그쪽 선이 안 보임" 처리.
#   DL_MIN_PIXELS(=40, da용)보다 낮은 이유: ll은 da처럼 면을 채우는 마스크가 아니라 가는
#   선이라 같은 밴드 안에 있는 픽셀수 자체가 원래 훨씬 적다. 실차 미검증 초기값.
DL_LL_SIDE_MIN_PIXELS = 15
# 밴드 내 좌/우 ll 중점을 채택하기 위한 두 선 사이 거리(px) 허용범위 — 실측 차로폭
#   0.8m(=DL_LL_CLIP_MARGIN_PX 주석의 근거와 동일 실측, @200px/m=160px)에 ±40% 여유를
#   둔 값. 이 범위 밖이면(예: 반대쪽 밴드의 다른 차선을 잘못 짝지은 경우) 그 밴드는
#   버리고 da로 폴백한다. 실차 미검증 초기값 — DEBUG_VIZ_DL_LANE으로 실제 정상 주행
#   중 밴드별 폭이 이 범위 안에 드는지 보고 조정할 것.
DL_LL_WIDTH_MIN_PX = 100
DL_LL_WIDTH_MAX_PX = 220

# [2026-08-07] _ll_slice_centers()가 좌/우 ll을 찾을 때 보는 탐색창 반경(px). 원래는
#   좌/우 분리 기준점(cur_ref) 하나로 밴드를 절반씩(왼쪽 전체/오른쪽 전체, 보통 수백 px)
#   나눠 그 안 전체 픽셀로 무게중심을 냈는데, 그 "반쪽"이 넓다 보니 옆 차선 선이나
#   반사광이 반쪽 어디에 있든 평균에 섞여 들어가는 문제가 있었다(다중 후보 오탐).
#   참고: github.com/junhyukch7/Advanced-Lane-Detection의 슬라이딩 윈도우가 폭
#   120px(반경 60px)짜리 좁은 창만 보는 것에서 착안 — 창 밖의 무관한 픽셀이 애초에
#   평균 계산에 안 들어오게 좌/우 각각 예상 위치 중심의 좁은 창만 보도록 바꿨다.
#   실차 미검증 초기값(참고 프로젝트와 동일하게 60으로 시작) — 급커브에서 밴드 간
#   실제 선 이동량이 이 값보다 크면 창이 선을 놓치고 추적이 끊길 수 있으니, 그런
#   구간에서 ll_bands 비율이 뚝 떨어지면 이 값을 키울 것.
DL_LL_SEARCH_HALF_WIDTH_PX = 60.0

# ── 색상기반 노란 중앙선 보조 검출 (lane_side 판정용, hough_lane.py와 공유) ──
#   TwinLiteNet의 ll 출력은 흰/노랑을 구분하지 않아 HSV로 별도 검출한다.
YELLOW_LOWER = np.array([15, 80, 80])
YELLOW_UPPER = np.array([40, 255, 255])

FPS_LOG_PERIOD_SEC = 5.0   # dl_lane.py 워커 스레드 FPS/provider 로그 주기(s)


# #############################################################
# 4. 조향 컨트롤러 선택 (Pure Pursuit / LQR)
# #############################################################
#   track_drive.py의 _lane_steer()가 self.lane_path를 받아 조향각(도)을
#   계산하는데, 아래 값으로 어떤 컨트롤러를 쓸지 고른다. 둘 다
#   control(path, vehicle_xy)->조향각(도) 계약이 동일해 전환에 다른 코드
#   수정이 필요 없다.
#     'pure_pursuit' : controller/pure_pursuit.py (기하학적, 속도/커브 적응형
#                      lookahead). **현재 기본값**.
#     'lqr'          : controller/lqr.py (2-state 운동학 오차모델 LQR, 신규·
#                      실차 미검증). 처음 켤 때는 저속에서, 언제든 사람이
#                      개입할 수 있는 상태로 테스트할 것.
STEERING_CONTROLLER = 'pure_pursuit'  # 'pure_pursuit' | 'lqr'

# ── Pure Pursuit 튜닝값 (controller/pure_pursuit.py PurePursuitController) ──
#   전부 실차 미검증 튜닝값. 각 값의 설계 배경은 pure_pursuit.py __init__ 상단
#   주석 참고 — 여기는 "현재 적용값"만 모아둔다.
PP_LOOKAHEAD_BASE_PX = 90.0        # lookahead 하한(직진/저속 기준값)
PP_LOOKAHEAD_SPEED_GAIN = 4.0      # 속도가 오를수록 lookahead를 늘리는 게인
# [2026-08-07] 150 → 190. speed_lookahead_px = BASE + GAIN*speed 공식이 SPEED_NORMAL=5
#   기준(90+4*5=110)으로 설계됐는데(pure_pursuit.py __init__ 주석), SPEED_NORMAL이 이후
#   25까지 오르면서(config.py 상단 SPEED_NORMAL 주석) 이론상 필요한 lookahead(90+4*25=190)가
#   구 상한(150)에 막혀 speed>=15부터는 lookahead가 더 안 늘어났다. 실차에서 "속도 5는
#   진동이 없는데 20으로 올리니 진동이 심해진다"는 증상으로 재현됨 — Pure Pursuit은 lookahead가
#   짧을수록 curvature=2*sin(alpha)/ld 공식에서 같은 픽셀오차도 더 크게 증폭되므로(§0.5.2
#   README), 속도만 오르고 lookahead가 그만큼 못 늘어나면 고속에서 과민 반응→진동이 커진다.
#   190은 SPEED_NORMAL=25를 그대로 대입한 값 — 실차 재검증 필요. 그래도 진동이 남으면
#   PP_ALPHA(현재 0.5)를 낮춰 조향각 저역통과를 더 강하게 거는 쪽을 다음으로 볼 것.
PP_LOOKAHEAD_MAX_PX = 190.0        # lookahead 상한
PP_LOOKAHEAD_CURVATURE_GAIN = 100.0  # 직전 프레임 curvature가 클수록(코너) lookahead를 줄이는 게인
PP_LOOKAHEAD_MIN_PX = 40.0         # 코너에서 lookahead가 줄어들 수 있는 하한
# [2026-08-06] "곡률→조향각" 게인(pure_pursuit.py의 steer_deg = atan(curvature*wheelbase_px)).
#   원래 80.0은 "실제 축거리 대신 쓰는" 임의 튜닝값이었다(pure_pursuit.py 상단 주석: "카메라
#   픽셀→미터 변환이 아직 실측 전이라 wheelbase_px를 대신 쓴다, PIXELS_PER_METER가 실측되면
#   실제 축거리(m)*PIXELS_PER_METER로 대체 가능"). LANE_DETECTOR_BACKEND='dl'(기본값) +
#   DL_USE_BEV=True(기본값)에서는 self.lane_path가 정확히 DL_PIXELS_PER_METER(=200px/m,
#   BEV 캔버스의 정의상 스케일)로 만들어진 픽셀좌표이므로, 이제 실측 `LQR_WHEELBASE_M`
#   (0.335m, §6.7)을 그대로 곱해 물리 기반 값으로 대체할 수 있다: 0.335 * 200 = 67.0.
#   ★ 실차 재검증 필요 ★ — 80.0은 그 자체로 실차에서 "이 정도 조향 반응이 적당하더라"고
#   경험적으로 맞춰졌을 가능성이 있어(다른 근사 오차를 상쇄했을 수도 있음), 67.0로 바꾸면
#   같은 curvature에도 조향각이 더 작게(atan 인자가 작아짐) 나와 코너링이 더 완만해질 수
#   있다 — 너무 밋밋하게 느껴지면 이 값을 다시 올릴 것(단, 그때는 "튜닝값"임을 주석에 남길 것).
PP_WHEELBASE_PX = 67.0             # = LQR_WHEELBASE_M(0.335) * DL_PIXELS_PER_METER(200) 실측 기반 계산값
PP_ALPHA = 0.5                     # 프레임간 조향각 저역통과 필터(1=필터없음, 0=반응없음)
PP_MIN_LOOKAHEAD_PX = 90.0         # curvature 분모(ld) 바닥값 — 노이즈 증폭 방지용. PP_LOOKAHEAD_MIN_PX와 다른 값이니 헷갈리지 말 것
PP_DX_DEADZONE_PX = 6.0            # 이 이하 픽셀오차는 0으로 죽여 중앙 부근 잔떨림 제거

# ── LQR 튜닝값 (controller/lqr.py LQRController) — 전부 실차 미검증 ──
#   speed_gain: 클수록 반응 커짐. r_steer: 올릴수록 조향 억제(지그재그 완화,
#   q_lateral/q_heading 건드리기 전에 먼저 조정). q_lateral/q_heading: lateral
#   비중↑→중앙복귀 서두름(오버슈트 위험), heading 비중↑→각도부터 맞추고 천천히
#   복귀. wheelbase_gain: 조향 강도. alpha: 저역통과(반응 느리면 올리고 잔떨림
#   있으면 낮출 것). heading_probe_px/min_path_px: 노이즈 방지 안전장치(조향이
#   자꾸 직전값 유지로 빠지면 낮출 것).
#
#   [LQR 브랜치에서 이식, 2026-08-05] Q=diag(1,1)이 e_y(px, O(1~100))와 e_psi(rad,
#   O(0.01~0.5))를 같은 가중치로 취급하면 Riccati가 극단적으로 큰 K를 내놓아 미세한
#   오차에도 조향각이 클램프까지 튀는 버그가 실차에서 확인됐다(controller/lqr.py 상단
#   주석 참고). DL+BEV 조합(LANE_DETECTOR_BACKEND='dl' and DL_USE_BEV)일 때는
#   DL_PIXELS_PER_METER로 e_y를 미터로 환산하는 "미터 모드"를 쓰면 e_y·e_psi가 비슷한
#   크기가 되어 이 문제가 사라진다 — track_drive.py가 LQRController 생성 시 이 조건을
#   보고 pixels_per_meter를 넘길지(미터 모드) None을 넘길지(레거시 픽셀 모드, 아래
#   LQR_WHEELBASE_GAIN/LQR_SPEED_GAIN/LQR_HEADING_PROBE_PX/LQR_MIN_PATH_PX 사용) 자동
#   결정한다. 아래 LQR_WHEELBASE_M/LQR_SPEED_MPS/LQR_HEADING_PROBE_M/LQR_MIN_PATH_M은
#   미터 모드 전용값 — wheelbase_m은 줄자 실측, speed_mps는 엔코더 연동 전 임시값(실차
#   최우선 튜닝 대상).
LQR_WHEELBASE_GAIN = 50.0
LQR_SPEED_GAIN = 120.0
LQR_Q_LATERAL = 1.0
LQR_Q_HEADING = 1.0
LQR_R_STEER = 1.0
LQR_DT = 0.05               # control_loop 타이머 주기(20Hz)와 반드시 일치 — 튜닝값 아닌 시스템 상수
LQR_HEADING_PROBE_PX = 65.0
LQR_ALPHA = 0.5
LQR_MIN_PATH_PX = 65.0
LQR_WHEELBASE_M = 0.335     # [미터 모드] 실측값(2026-08-06, 줄자로 앞바퀴-뒷바퀴 축간거리 실측 —
                             #   LQR 브랜치에서 이식). planner/hybrid_astar.py의 wheelbase 기본값(같은
                             #   차량이므로 반드시 같은 값)과 일치시킬 것 — 재실측 시 둘 다 갱신.
LQR_SPEED_MPS = 1.0         # [미터 모드] 속도 추정치(m/s) — 아래 VESC 연동이 살아있으면 매 주기
                             #   set_speed_mps()로 실측값으로 덮어써진다(track_drive.py의 cb_vesc()
                             #   참고). 이 값은 그 전까지, 혹은 VESC 브리지가 안 떠 있을 때 쓰는 폴백.
LQR_HEADING_PROBE_M = 0.3   # [미터 모드] 헤딩오차 추정용 근거리 참조거리(m)
LQR_MIN_PATH_M = 0.3        # [미터 모드] 경로 전체 길이가 이보다 짧으면 직전값 유지

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
                             #   0.0 고정)이면 "VESC 실측값을 못 믿는다"고 보고 폴백한다. 두 곳에서 씀:
                             #   ① self.lqr.set_speed_mps() 갱신을 건너뛰고 직전 게인 유지 — v≈0에서
                             #     B≈0으로 게인이 퇴화(조향이 상태에 영향을 못 미치는 것으로 계산됨)하는
                             #     것을 피하기 위함.
                             #   ② _speed_for_lookahead()(2026-08-06, pure_pursuit용)가 v_mps 대신
                             #     self._prev_speed(명령속도)로 폴백 — track_drive.py 참고. 이름은
                             #     LQR 전용처럼 보이지만 "VESC 값을 신뢰할 최소 속도"라는 의미라
                             #     LQR_이 아니라 VESC_ 접두어를 씀.

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


# #############################################################
# 5. 디버깅 ON/OFF
# #############################################################
DEBUG_LOG    = True   # 0.5초마다 CLI에 [LAP]/[SENS]/[LANE]/[TRIG]/[SIG]/[LAVA-ROI] 로그
DEBUG_PERIOD = 0.5     # 위 로그 주기(s)

DEBUG_VIZ_LIDAR    = False  # 라이다 BEV 장애물 감지 디버그 창 (track_drive.py)
DEBUG_VIZ_LAVACON  = False  # 라바콘 트리거 좌우 클러스터 BEV 디버그 창 (track_drive.py)
DEBUG_PLANNER      = False  # Hybrid A* OccupancyGrid 디버그 창 (track_drive.py, USE_HYBRID_ASTAR_FOR_B2=True일 때만 의미있음)
DEBUG_VIZ_STEER    = True   # 조향 컨트롤러(직전값유지/현재값반영) 한글 디버그 창 (track_drive.py)
DEBUG_VIZ_VESC     = True   # VESC 실측속도(/vesc_speed_erpm) 연동 상태(수신중/끊김/미수신) 디버그 창
                             #   (track_drive.py, 2026-08-06 LQR 브랜치에서 이식)

DEBUG_VIZ_DL_LANE    = True   # 차선 — 기본 백엔드('dl') 디버그 창 (perception/dl_lane.py)
DEBUG_VIZ_HOUGH_LANE = True   # 차선 — 대안 백엔드('hough') 디버그 창 (perception/hough_lane.py)
DEBUG_VIZ_LANE       = True   # 차선 — 대안 백엔드('classic_cv') 디버그 창 (perception/lane_util.py)
DEBUG_VIZ_STOPLINE   = False  # 정지선 디버그 창, 백엔드 무관 항상 동작 (perception/perc_floor.py)
DEBUG_VIZ_SIGNAL     = False  # 신호등(S0/S2 공용) 디버그 창 (perception/traffic_signal.py)


# #############################################################
# 6. 미션 State / 실차 테스트 범위 제한
# #############################################################
START_STATE     = MissionState.S1_LANE_FOLLOW
ENABLE_BEHAVIOR = False   # S1에서 라바콘/장애물/추월 Behavior를 켤지 여부(최상위 스위치)

# ── 실차 테스트 범위 제한 ──
#   지금 단계에서 실차로 검증 가능한 건 딱 세 가지: ①신호등 인식 후 출발(S0)
#   ②차선주행(S1) ③라바콘 주행(B1). 나머지(S2 교차로/S3 지름길, B2 고정장애물/
#   B3 방해차량)는 아직 실차 미검증(좌회전 각도·속도 placeholder)이라 테스트 중
#   의도치 않게 발동하면 위험할 수 있어 아래 두 플래그로 강제로 꺼둔다.
#   → 전체 미션을 테스트할 준비가 되면(좌회전 튜닝 끝) 둘 다 False로 되돌릴 것.
TEST_DISABLE_INTERSECTION = True
#   True: 정지선을 감지해도 감속→S2_INTERSECTION 전환을 아예 안 함(차선주행만 계속).
#   False: 원래대로 정지선 감지 시 감속 후 S2로 정상 전환.
TEST_DISABLE_B2_B3 = True
#   True: Phase가 FIXED_OBSTACLE/VEHICLE로 넘어가도 트리거 검사를 건너뛰고
#         B0_NORMAL로 고정(B1 끝난 뒤 계속 일반 차선주행만 함).
#   False: 원래대로 SAFETY_DIST/OVERTAKE_TRIGGER 트리거 검사해서 B2/B3 정상 발동.
TEST_FORCE_BEHAVIOR = True
#   True: _behavior_enabled를 시작부터 강제 True로 켜서, 교차로를 끈 채로도
#         라바콘(B1)만 독립적으로 실차 검증할 수 있게 한다.
#   False: 원래대로 S2 교차로 직진 신호를 받아야만 Behavior가 켜짐.
#   → 전체 미션 테스트로 넘어갈 때는 TEST_DISABLE_INTERSECTION=False와 함께
#     이것도 False로 되돌릴 것(둘 다 켜두면 시나리오 순서가 어긋난다).

# ── B2 회피 방식 선택 ──
#   False = ObstacleAvoidance(차선 기반 횡이동, 기본값)
#   True  = Hybrid A* + OccupancyGrid + Stanley (비교/보존용) — planner/ 참고
USE_HYBRID_ASTAR_FOR_B2 = False

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
PATH_EMA_ALPHA = 0.4   # 새 프레임에 줄 가중치(작을수록 더 부드럽고, 더 느리게 반응)

# ── 라바콘/장애물/방해차량/신호등 트리거 ──
LAVACON_DONE_FRAMES = 80      # 우측콘 미검출이 연속 N프레임(20Hz→약 4초) 쌓이면 Phase 전환(디바운스)
LAVACON_TRIGGER_FRAMES = 5    # 좌우 클러스터 동시검출이 연속 N프레임 쌓이면 B1_LAVACON 진입 확정
SAFETY_DIST      = 5.0        # B2(고정장애물) 발동 거리(m)
OVERTAKE_TRIGGER = 6.5        # B3(방해차량) 발동 거리(m)
VEHICLE_TRIGGER_FRAMES = 5    # 라이다 단독검출 연속 N프레임이면 B3_VEHICLE 진입 확정
SIG_CONFIRM_FRAMES = 3        # 신호등(직진/좌회전) 판정이 연속 N프레임 유지돼야 확정(20Hz→0.15s)

# ── 장애물회피(TargetPassing, controller/obstacle_avoidance.py) ──
PASS_OFFSET = 100.0          # 반대 차선으로 이동할 목표 횡편차(px)
CENTER_DEADZONE_M = 0.12     # 타겟 횡중심이 이 값(m) 이내면 '정면'으로 보고 방향을 다른 근거로 정함
CLEAR_FRAMES_TO_RETURN = 6   # 타겟이 안 보이는 상태가 이만큼 연속되면 복귀 시작
SWITCH_FRAMES = 8            # 주행 타겟이 내 진행쪽으로 넘어온 상태가 이만큼 지속되면 방향 전환
LATERAL_ALPHA_OUT = 0.12     # 옆차선 이동 수렴 속도
LATERAL_ALPHA_BACK = 0.16    # 복귀 수렴 속도 — 90cm 규정 때문에 늑장 부리면 차선이탈, OUT보다 빠르게
LATERAL_DONE_PX = 8.0        # 이 이하로 좁혀지면 이동 완료로 판정
MIN_GAP_M = 0.6              # 추돌 방지 종방향 간격(m) — 이보다 가까우면 횡이동 끝날 때까지 속도를 죽임

# ── 신호등(S0/S2 공용 4구, perception/traffic_signal.py) ──
SIG4_ROI_T, SIG4_ROI_B = 0.08, 0.28
SIG4_ROI_L, SIG4_ROI_R = 0.04, 0.78
SIG4_MIN_RADIUS, SIG4_MAX_RADIUS = 15, 25
SIG4_BRIGHT_MARGIN  = 15
SIG4_MAX_CANDIDATES = 10  # 원이 이보다 많이 잡히면 조합 탐색 없이 바로 실패 처리(ROI 자체가 노이즈로 판단)

# ── 정지선(perception/perc_floor.py check_stopline(), 백엔드 무관 항상 사용) ──
STOPLINE_WHITE_LOW = 180        # 그레이스케일 흰색 임계
STOPLINE_WHITE_RATIO_TH = 0.06  # ROI 내 흰 픽셀 비율 임계 (실측: 1000/16500 ≈ 6%)

# ── 정지선 접근/이탈 판정 (track_drive.py) ──
SHORTCUT_MIN_T = 3.0     # 지름길 진입 후 끝감지 활성화까지 최소 주행시간(s, 오판 방지)
SHORTCUT_MAX_T = 15.0    # 지름길 최대 주행시간(s, 끝 못 찾을 때 강제 탈출 백업)
# 지름길 출구(본선 합류부)는 신호등이 없어 정지선 검출로만 끝을 판단하는데, 합류부는
# 도로가 서서히 넓어지는 형태라 정지선이 실제로 잡히기 "전"에 da 세그멘테이션이 이미
# 합류 쪽(실측: 우측)으로 넓어져 중심선이 끌려가는 문제가 있다(ㅓ교차로와 동일한
# 실패모드, 트리거만 신호 대신 시간 기반). 그래서 정지선 검출을 기다리지 않고, 이
# 시간이 지나면 미리 _lane_drive()(비전)를 끄고 _shortcut_ref_yaw 기준 헤딩홀드로
# 전환한다(좌회전 스크립트는 아직 시작 안 함 — 정지선이 실제로 잡히거나 SHORTCUT_MAX_T에
# 도달해야 _shortcut_end()가 확정되어 진출 시퀀스로 넘어간다). SHORTCUT_MIN_T보다 크고
# SHORTCUT_MAX_T보다 충분히 작아야 하며, 실차에서 지름길 실제 통과시간을 재서 합류부
# 도달 직전 시점으로 맞출 것 — 지금은 미실측 추정치.
SHORTCUT_VISION_CUTOFF_T = 10.0
STOPLINE_COOLDOWN = 3.0  # 상태 복귀 후 이 시간(s)간 정지선 재감지 무시(따다닥 전환 방지)

# ── 인식 끊김 보상 / 교차로 근처 기동 금지 ──
OBSTACLE_HOLD_T = 0.6                   # 마지막 관측 후 이 시간(s)까지는 장애물이 있다고 본다
MANEUVER_BLOCK_AFTER_STOPLINE_T = 2.0   # 정지선을 최근에 봤으면 이 시간(s) 동안 회피/추월 기동 금지

# ── 단위 환산 상수 — 실측 후 값만 채울 것 ──
#   지금 코드에는 '모터 단위'(drive()가 ±100으로 클립하는 값)와 '미터'가 섞여
#   있다. 아래 값이 0.0이면 아직 미실측 상태라는 뜻이고, 거리 기반 로직은
#   보수적으로 동작한다.
# 실측(2026-08-06): speed=5(3s/1.04m, 6s/2.50m), speed=10(3s/2.3m·2.06m 평균, 5s/4.5m) 각각
#   2개 시점 거리로 "정속구간 기울기(m/s)"와 "가속 오프셋"을 분리 추정(직선회귀, 아래 README
#   6.5절 근거). speed=5 → 정속 0.487m/s(가속구간≈1.73s), speed=10 → 정속 1.16m/s(가속구간≈2.24s).
#   두 점을 잇는 기울기 0.1347(m/s per unit)을 채택 — 단, 절편이 음수라 사실상 speed≈1.4
#   미만에서는 이 선형식이 안 맞는(모터 데드존 추정) 2점짜리 근사이니 저속(APPROACH_SPEED=2.0
#   등)에는 그대로 쓰지 말 것. 상세 도출 과정은 README.md "6.5 속도 단위 ↔ m/s 환산" 참고.
METERS_PER_SPEED_UNIT = 0.1347   # 모터 속도단위 1당 m/s(정속구간 기준, speed 5~10 구간 회귀)
LANE_WIDTH_M          = 0.4   # 실측(2026-08-04): 흰선-흰선(도로 전체폭) 80cm, 노란선 정중앙 확인 → 차선 1개 폭 = 40cm
PIXELS_PER_METER      = 0.0   # BEV 픽셀 ↔ 미터 환산(전역) — 미실측. DL_USE_BEV가 검증돼 기본 전환되면 DL_PIXELS_PER_METER로 채울 것
VEHICLE_WIDTH_M       = 0.31  # 실측(2026-08-04): xycar 본체 가로 31cm (세로64cm×가로31cm×높이20cm)
# 각폭 분류 임계 — 이 폭 이상이면 '차량', 미만이면 '고정장애물'.
#   실측(2026-08-04): 고정장애물(고장난 차량) 가로20cm×세로41cm×높이16cm,
#   방해차량 가로28cm×세로54cm×높이19cm → (0.20+0.28)/2 = 0.24
OBSTACLE_VEHICLE_WIDTH_M = 0.24
