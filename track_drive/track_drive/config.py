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
    B2_OBSTACLE = 2  # 고정장애물 회피 (Phase.OBSTACLE_ZONE일 때 obstacle_type=='fixed'로 감지 시 활성)
    B3_VEHICLE  = 3  # 방해차량 추월   (Phase.OBSTACLE_ZONE일 때 obstacle_type=='vehicle'로 감지 시 활성)

# S1(차선주행) 내부 진행 순서 — 순서 고정(라바콘→장애물구간→완료), 순차 전용(우선순위 판단 불필요)
# [2026-08-15] FIXED_OBSTACLE/VEHICLE을 OBSTACLE_ZONE 하나로 통합했다
# (da_based_b2b3_proposal.md "해결 방향 B안"). 예전엔 이 둘을 트랙 순서로 미리 나눠
# "지금이 고정장애물 구간인지 방해차량 구간인지"를 Phase가 알려줬는데, da 안전마진
# (§2.30)+avoid-hold(§2.32/§2.33)로 회피 기동 자체가 정적/동적 구분 없이 동일해지면서
# 그 구분이 굳이 필요 없어졌다 — 이제 정적/동적 구분은 Phase가 아니라 매 프레임
# obstacle_type(라이다 실측 폭 기반, perc_obstacle() 참고)으로 그때그때 판단한다
# (track_drive.py run_behavior_fsm() 참고). OBSTACLE_ZONE→DONE 전환은 B2/B3 둘 다
# 최소 한 번씩 완료돼야 넘어간다(_mark_behavior_passed(), 순서 고정 가정을 버렸으므로
# "마지막에 끝난 쪽"이 아니라 "둘 다 끝났는가"로 판단).
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
SPEED_NORMAL  = 15.0   # [2026-08-17g] 10.0 → 15.0(요청 반영, 증속). 이 값이 바뀌면
                        #   PP_LOOKAHEAD_MAX_PX(§0.5.6/§0.5.10 공식: BASE+GAIN*SPEED_NORMAL)도
                        #   반드시 같이 재계산할 것 — 아래에서 150.0으로 갱신함.
                        # [2026-08-17f] 3.0 → 10.0(요청 반영, 저속 재튜닝 검증 완료 후 증속). 이 값이
                        #   바뀌면 PP_LOOKAHEAD_MAX_PX(§0.5.6/§0.5.10 공식: BASE+GAIN*SPEED_NORMAL)도
                        #   반드시 같이 재계산할 것 — 아래에서 130.0으로 갱신함. SPEED_CORNER_MIN 등
                        #   "즉시 cap" 계열 하한값은 모터 데드존 등 물리적 근거로 고정된 값이라 이번엔
                        #   일부러 비례 조정하지 않았다(README §0.5.10 참고) — 실차에서 코너 진입/탈출
                        #   속도 급변이 과하게 느껴지면 그때 올릴 것.
                        # [2026-08-13] 15.0 → 3.0(요청 반영, 조향 파라미터 재튜닝 테스트용 저속화).
                        # [2026-08-10] 차선주행(S1) 기본(직진) 속도. 8.0 → 25.0 → 15.0(요청 반영,
                        #   DL_CENTER_MODE='ll' 기본 전환과 함께 하향 — ll 재설계가 아직 실차
                        #   미검증이라 우선 보수적으로 낮춤).
                        #   0.0으로 두지 말 것 — _lane_drive()에서 나눗셈 분모로도 쓰여 ZeroDivisionError.
                        #   ★주의★ README §6.5의 METERS_PER_SPEED_UNIT 회귀는 speed=5/10 두 점만 실측한
                        #   것이라 15도 측정 범위 밖 외삽 — 실제 m/s·제동거리·코너 반응이 그 선형식대로
                        #   나올지 실차 재검증 필요.
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
# [2026-08-17j] 15.0 → 7.0(요청 반영). SPEED_CORNER_MIN < SPEED_NORMAL(15.0) 관계가
#   다시 확보돼 코너 감속이 실제로 걸린다(직전 [2026-08-17i]에서 15.0으로 올려 SPEED_NORMAL과
#   같아지는 바람에 코너 감속이 no-op이었던 문제 해소). 단, SPEED_LANE_STALE(5.0, 아래
#   참고)은 요청에 따라 그대로 둠 — 인지 정지 상황에서는 여전히 5.0으로 더 깎인다.
# [2026-08-17i] 10.0 → 15.0(요청 반영). ★주의★ SPEED_NORMAL도 15.0이라 이제
#   SPEED_CORNER_MIN == SPEED_NORMAL — _lane_drive()의 코너 감속(목표속도 하한 clip)이
#   사실상 no-op이 된다(코너에서도 직진과 같은 속도로 주행, 감속 없음). 과거
#   SPEED_NORMAL=3.0 구간에서 겪은 것과 같은 종류의 문제([[project-track-drive-speed10-change]]
#   메모 참고)이니, 코너에서 실제로 느려지길 기대한다면 이 값을 SPEED_NORMAL보다 낮게
#   유지해야 한다 — 실차에서 코너 반응 확인 후 필요시 재조정할 것.
# [2026-08-17g] 5.0 → 10.0(요청 반영, SPEED_NORMAL 15.0 증속과 함께 코너 속도도 상향).
#   ★주의★ 이전엔 SPEED_NORMAL=3.0 구간에서 이 값이 SPEED_NORMAL보다 커서 코너 감속
#   ([[project-track-drive-speed10-change]] 메모 참고)이 사실상 no-op이었던 이력이 있다 —
#   SPEED_NORMAL(15.0) > SPEED_CORNER_MIN(10.0) 관계는 유지되므로 이번엔 그 문제가 재현되진
#   않지만, 코너 감속 폭(15→10, 33%)이 이전(10→5, 50%)보다 완만해졌다는 점은 실차에서
#   코너 진입 느낌으로 확인할 것 — 부족하면 이 값을 다시 낮출 것.
SPEED_CORNER_MIN = 7.0
# [2026-08-18] SPEED_LL_DEGRADED(DL_CENTER_MODE='ll' 전용 속도 상한) 삭제 — 이제
#   DL_CENTER_MODE='da'로 완전히 전환되어 차선(ll) 기반 주행 자체를 쓰지 않는다(요청
#   반영). track_drive.py _lane_drive()/​_debug_viz_steer()의 소비부도 함께 제거.
# [2026-08-11] DL 추론 워커(별도 스레드, dl_lane.py)가 LANE_STALE_SEC 이상 새 결과를 못
#   내놓았을 때(카메라/추론 죽음 등, perc_lane()의 lane_stale 판정) 강제하는 속도 상한.
#   일부러 SPEED_CORNER_MIN(5.0)보다 낮추지 않았다 — "코너가 아닌데도 이 속도로 깎였다"는
#   부자연스러움 자체가 사람이 알아챌 수 있는 신호가 되도록, 급정지가 아니라 코너 감속과
#   비슷한 수준으로만 눈에 띄게 낮춘다는 설계(요청 반영). 실차 미검증.
SPEED_LANE_STALE = 5.0
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
# [2026-08-17n] 정지 출발 시 "틱틱거림/초반 힘딸림" 완화용 신설(요청 반영, README §7.2).
#   제어루프가 20Hz(0.05초) 고정주기라(track_drive.py:357 create_timer), 위 SPEED_ACCEL_STEP
#   램프는 정지(_prev_speed=0)에서도 매 틱 0.4씩만 올라가는 "계단" 명령이 된다. 문제는 그
#   계단이 모터 데드존(≈1.4, 위 SPEED_CORNER_MIN 주석의 회귀 절편 근거)을 넘는 데만 4틱
#   (0.2초)이 걸리는데, 그 구간 내내 바퀴는 RPM=0이라 역기전력이 없어 같은 duty라도 이미
#   움직이는 상태보다 훨씬 큰(락터) 전류가 흐른다 — "가장 전류를 많이 먹는 구간에 가장 오래
#   머무는" 구조. 매 50ms 새 duty를 정지마찰에 막혀 못 넘기고 코깅음만 내는 게 반복되면
#   그게 "일정 주기로 툭툭거림"으로 체감된다는 게 실제 관찰과 일치. 정지에서 새로 출발하는
#   틱 1회에 한해 이 값으로 즉시 점프시켜(_lane_drive() 가속 램프 진입부) 정지마찰 구간
#   체류 시간을 4틱→1틱으로 줄인다 — 이후 틱은 그대로 SPEED_ACCEL_STEP 램프를 이어간다.
#   §7.1(2026-08-07)의 LVC 트립은 "고속까지 너무 빨리 가속"이 원인이었고 이건 그와 무관한
#   "정지 돌파" 구간만 건드리므로 재발 가능성은 낮다고 보지만, 이 값도 순간 전류 스파이크를
#   키우는 방향이라 실차에서 틱틱거림이 줄어드는지 + LVC 재발은 없는지 둘 다 확인 필요.
#   데드존(1.4)보다 확실히 위, SPEED_CORNER_MIN(7.0)보다는 아래로 잡은 실차 미검증 초기값 —
#   틱틱거림이 여전하면 올리고, 출발이 과하게 튀어나가면 낮출 것.
#   [2026-08-18 실차 재검증] 효과 없음으로 판명 — track_drive.py 정식 루프에 적용 후에도
#   틱틱거림 개선 체감 전혀 없었고, 같은 세션에서 출발 성공률이 오히려 30분에 1회 수준으로
#   급락함. 원인이 이 스틱션 체류시간 가설이 아니라 하드웨어(배터리/ESC/모터) 쪽으로
#   재이동함 — 자세한 내용은 track_drive/실제속도측정.md §0.1 참고. 이 값 자체를 더
#   올리거나 내리는 재튜닝은 하드웨어 원인 확인 전까지 보류.
SPEED_KICK_START = 3.0
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
# [2026-08-11] LAVACON_KP(라바콘 전용 P게인, 210.0) 삭제 — 라바콘 조향이 이제
# _lane_steer()(Pure Pursuit/LQR)를 그대로 재사용해서(track_drive.py _handle_lavacon()/
# perc_lavacon() 참고) 더 이상 쓰이지 않는다.

# ── 코너 진입 시 회전반경 기반 감속 (ROS2 Nav2 Regulated Pure Pursuit 방식) ──
#   회전반경(1/curvature)이 CORNER_MIN_RADIUS_PX보다 작아지면 그 비율만큼 목표속도를
#   깎는다. PIXELS_PER_METER 미실측이라 반경은 픽셀 단위 — 실차 미검증 추정치.
CORNER_MIN_RADIUS_PX = 250.0
CORNER_MIN_SPEED_SCALE = 0.35  # 반경이 0에 가까워져도 속도가 0으로 죽지 않게 하는 하한 배율

# [2026-08-18] "직선인데 커브로 오검출돼 속도가 안 오른다" 대응 — turn_now(조향각 signed
#   EMA)/turn_preview(lane_lookahead)는 전부 비전+조향출력에서만 나오는 신호라, 세그멘테이션
#   잡음이나 조향 잔떨림만으로도 코너로 오인될 수 있다. is_straight(§0.5.9, PP_STRAIGHT_*)가
#   이미 이 문제를 다루긴 하지만 여러 프레임 연속 확정이 필요한 이진 게이트라, 아직
#   확정 전인 애매한 프레임에서는 turn_now/turn_preview가 잡음만으로 감속을 걸어도 못 막는다.
#   2023 KMU 대회 AuTURBO rookie 팀 저장소(ModeController.py, github.com/AuTURBO/
#   2023_KMU_Autonomous_team_AuTURBO_rookie)의 모드전환 로직을 참고 — 거기서도 비전(픽셀
#   오차 평균)만으로는 "커브"를 확정하지 않고, diff_degree(IMU yaw 변화량 > 47도)로 실제
#   회전량을 교차검증한 뒤에야 realcurve로 카운트한다. 그 아이디어를 연속값 버전으로
#   가져와 track_drive.py._imu_corner_confirm_scale()에서 쓴다 — "비전은 코너라는데 IMU
#   실측 회전율이 거의 0"이면 코너감속(turn_for_speed)을 절반 이하로 깎는다.
#   CORNER_IMU_CONFIRM_KAPPA_PX = 1/CORNER_MIN_RADIUS_PX로 잡은 이유: 반경기반 추가감속
#   (_corner_radius_speed_scale())이 작동을 시작하는 커브(반경<=250px)에서는 IMU 신뢰도가
#   이미 1.0(무감쇠)에 도달해 있어야, 진짜 코너에서까지 이 게이트가 감속을 방해하지 않는다.
#   IMU/VESC가 죽어있거나(_imu_curvature_px()가 None) dl+BEV 조합이 아니면 기존처럼 비전
#   신호만으로 판단(무감쇠, 1.0)한다. 실차 미검증 첫 추정치.
CORNER_IMU_CONFIRM_KAPPA_PX = 1.0 / CORNER_MIN_RADIUS_PX  # = 0.004 — 이 이상 IMU curvature면 코너감속 100% 신뢰
CORNER_IMU_MIN_SCALE = 0.5  # IMU가 "회전 거의 없음"을 보고해도 비전신호 기반 감속을 최소 이만큼은 남겨두는 하한

# ── 좌회전 공통 (S2→S3 진입, S3→S1 진출) — 전부 실차 튜닝 필요한 임시값 ──
#   [2026-08-18] 종료 판정을 프레임 카운트(open-loop)에서 IMU yaw 실측 기반(closed-loop)으로
#   변경. 같은 (TURN_ANGLE, TURN_SPEED) 명령이어도 배터리 전압 강하·노면·속도 변동에 따라
#   실제 요레이트(초당 회전각)가 매번 조금씩 달라질 수 있어, "N프레임 지남"보다 "실제로
#   TURN_YAW_TARGET_DEG만큼 돌았음"이 더 정확하다(track_drive.py _do_left_turn() 참고).
#   TURN_FRAMES/TURN_EXIT_FRAMES는 이제 트리거가 아니라 IMU가 죽어있을 때만 쓰는 안전
#   타임아웃 상한(무한 회전 방지)이다 — _imu_live() 가드 패턴, _vesc_live()와 동일 철학.
TURN_ANGLE           = -60.0   # [진입] S2 교차로 → S3 지름길 좌회전 조향각
TURN_SPEED           = 15.0    # [진입] 좌회전 속도
TURN_YAW_TARGET_DEG  = 90.0    # [진입] 목표 회전각(도) — 분기가 직각이 아니라 커브로 열려서
                                #        정확히 90인지 미검증, 실차 재확인 필요
TURN_FRAMES          = 40      # [진입] 안전 타임아웃 상한 프레임 수 (20Hz 기준, IMU 죽었을 때만 씀)
TURN_EXIT_ANGLE      = -60.0   # [진출] S3 지름길 → S1 차선주행 좌회전 조향각
TURN_EXIT_SPEED      = 15.0    # [진출] 좌회전 속도
TURN_EXIT_YAW_TARGET_DEG = 90.0  # [진출] 목표 회전각(도) — 위와 동일 사유로 실차 재확인 필요
TURN_EXIT_FRAMES     = 40      # [진출] 안전 타임아웃 상한 프레임 수 (IMU 죽었을 때만 씀)

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
#   APPROACH_SPEED로 직진(각도 0)만 유지하다가, 이 거리를 채운 뒤에야 실제 분기
#   방향(직진 복귀 or 좌회전 스크립트 시작)을 실행한다.
#   [2026-08-18] 시간 기반(S2_COMMIT_T=1.0s)에서 거리 기반으로 변경 — 대회 주행 때
#   APPROACH_SPEED 근방 실제 속도가 튜닝 시점과 달라지면 "몇 초 동안 직진"은 물리적
#   분기 지점과 안 맞을 수 있다(속도가 빠르면 못 미치고, 느리면 지나침). 대신
#   track_drive.py _s2_intersection()이 매 제어주기 VESC 실측(self.v_mps, m/s)을
#   적분해 이 거리(m)를 채웠는지로 판단한다 — METERS_PER_SPEED_UNIT의 저속 회귀
#   신뢰도 문제(speed=5/10 두 점 회귀라 APPROACH_SPEED=2.0 같은 저속엔 못 미더움,
#   위 예전 주석 참고)와 무관하게 실제 이동거리 기준으로 맞는다. VESC가
#   죽어있을 때만(_vesc_live() 가드) METERS_PER_SPEED_UNIT 폴백을 쓴다 — 이전
#   S2_COMMIT_T가 암묵적으로 가정하던 것과 동일한 근사치다.
S2_COMMIT_DIST_M = 1.0   # 실측 물리적 분기 거리(≈1m) — 실차에서 재확인 후 조정할 것


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

# ll(차선) 전용 이진화 임계값. [2026-08-06, 실차 관찰] BEV 워프는
#   카메라에서 먼 지점일수록 원근압축을 되돌리려고 더 크게 확대하는데(호모그래피 성질상
#   불가피), da/ll은 이진화 "전"(float 확률맵) 상태로 워프하기 때문에(위 DL_USE_BEV 주석
#   참고, 계단 현상 방지 목적) 모델 출력의 경계 blur(확률이 0.5 근방인 애매한 픽셀들)도
#   그 확대율만큼 같이 늘어난다. 근거리는 원래도 확률이 뚜렷해 큰 영향이 없지만, 원거리는
#   실측 S자 커브 구간에서 ll이 실제 선 두께보다 눈에 띄게 두껍게 잡히는 게 확인됐다 — 두꺼운
#   ll은 _clip_da_by_ll()이 da를 필요 이상으로 깎아내 da가 DL_DA_MIN_COMPONENT_AREA
#   밑으로 떨어지고 그 프레임이 무효 처리되는 원인이 된다. ll은 "차선 있으면 그 위치만
#   보고 자르는" 용도라 da보다 확신이 필요하다는 논리로 원래 DL_FG_THRESHOLD(0.5)보다
#   높은 0.7로 잡았었다.
#   [2026-08-07] 그런데 실차 영상(변경 전 촬영분) 여러 개를 훑어보니 ll_cov가 거의 항상
#   0.03 미만으로 극히 낮았다 — "가끔 두껍게 잡힘"보다 "대부분의 프레임에서 ll이 거의
#   안 잡힘"이 더 큰 문제였다. 0.7이 blur 방지 목적을 넘어 정상 신뢰도(0.5~0.7)의 실제
#   차선 픽셀까지 통째로 걸러내고 있었을 가능성이 높다고 보고, 우선 da와 같은 0.5로
#   낮췄다(요청 반영). 원거리 blur로 두꺼워지는 문제가 다시 나타나면 §2.2(S자 커브 원거리
#   ll 두께 과다 검출) 대응이 이미 있으니 그쪽을 같이 볼 것. 실차 재검증 전 —
#   DEBUG_VIZ_DL_LANE에서 ll_cov가 정상 범위로 올라오는지, 원거리 ll이 다시 과하게
#   두꺼워지진 않는지 확인할 것.
DL_LL_FG_THRESHOLD = 0.5

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
# [2026-08-10] 3 → 2로 낮춤 — 급조향(30도 이상) 후 직진 복귀 구간에서 차선인식이
#   흔들린다는 실차 보고 조사 중, ref_x(_ll_yellow_white_centers()/_ll_slice_centers()의
#   탐색창 seed, dl_lane.py detect() 참고)가 이 프레임 수만큼 지연된 값이라는 게
#   드러났다 — 값이 빨리 바뀌는 급조향 복귀 구간일수록 지연이 가장 크게 문제된다.
#   3에서 2로 낮추면 새 후보가 확정값으로 승격되는 데 필요한 지연이 그만큼 줄어든다
#   (DL_STABLE_JUMP_MAX=20px 체크는 그대로라 노이즈 한 프레임이 바로 통과되진 않음).
#   실차 미검증 — 낮출수록 반응은 빨라지지만 노이즈에 대한 필터링은 약해지므로, 너무
#   낮추면(예: 1) 스파이크가 그대로 확정될 위험이 있다. 실차에서 급조향 복귀 흔들림
#   개선 여부와 노이즈 민감도를 같이 보며 재조정할 것.
DL_STABLE_FRAME_MIN = 2       # "새 추론이 끝난 시점" 기준 연속 안정 프레임 수(디바운스)
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

# [2026-08-07] _largest_da_component()의 시드(seed) 기반 최우선 후보 선택에 쓰는
#   탐색 범위 — ROI 최하단(차량과 가장 가까운 행)에서 이 행 수(세로) × 이 반경(가로,
#   ROI 중앙 기준)만큼의 작은 영역을 보고, 그 영역과 물리적으로 맞닿은 덩어리가 있으면
#   (면적이 유효 범위 안이면) 과거 판단(직전 프레임 연속성/면적순위)보다 우선 채택한다
#   — "차량이 실제로 서 있는 자리"라는 매 프레임 독립적인 물리 신호라, 직전 프레임의
#   오판(예: 교차로에서 잘못된 갈래를 물었던 경우)이 계속 이어지는 걸 스스로 교정한다.
#   DL_DA_SEED_HALF_WIDTH_PX는 DL_LL_WIDTH_MIN/MAX_PX(실측 차로폭 100~220px)의 절반
#   근방으로 잡은 초기 추정치, DL_DA_SEED_ROWS_PX는 "차량 바로 앞"만 보도록 작게 잡았다.
#   둘 다 실차 미검증 — 좁으면 시드 영역이 빈 프레임(=②/③ 폴백)이 잦아지고, 너무 넓으면
#   옆 차선까지 시드에 걸려 잘못 채택될 수 있으니 DEBUG_VIZ_DL_LANE으로 확인하며 조정할 것.
DL_DA_SEED_ROWS_PX = 10
DL_DA_SEED_HALF_WIDTH_PX = 70.0

# [2026-08-12] DL_CENTER_MODE='da' 밴드 중심 계산 — 탐색창(prior) + 밴드 간 속도예측 +
#   프레임 간 앵커링. 원래 _slice_centers()(무보정 cv2.moments, 밴드 전체 폭)를 그대로
#   썼는데, da가 옆 차선/여백까지 과검출(S자 커브에서 특히)되면 그 넓어진 영역 전체가
#   무게중심 계산에 그대로 들어가 중심이 쏠린다 — da 선택 단계(_largest_da_component,
#   시드/연속성)는 "어느 덩어리를 볼지"만 정하고, 그 덩어리 *안에서* 밴드 중심을 어떻게
#   뽑을지에는 아무 방어가 없었다. Mobileye(클로소이드+칼만 — 과거 상태로 예측/추적),
#   openpilot(프레임 간 hidden state), drivable-area 연구(공간 prior로 억제)에서 공통된
#   "탐색을 예측 위치 근방으로 제한" 아이디어를 _ll_slice_centers()(DL_LL_ALGO='lr',
#   §2.20 README)가 이미 쓰던 패턴 그대로 da에도 적용한다 —
#   DLSlideWindow._da_slice_centers_windowed() 참고. da는 좌/우 두 갈래가 아니라
#   중심선 "한 갈래"라 ll보다 단순한 단일-트랙 버전.
#   DL_DA_SEARCH_HALF_WIDTH_PX가 DL_LL_SEARCH_HALF_WIDTH_PX(60, 선 하나 전용)보다 넓은
#   이유 — da는 폭 있는 영역이라 반차로폭(LANE_WIDTH_M=0.4m=80px) 이상은 창 안에 들어와야
#   정상 시야까지 잘라내지 않는다. 전부 실차 미검증 초기값 — 처음 켤 때 창이 너무 좁아
#   정상 코너까지 놓치지 않는지(=검출 밴드 수가 줄지 않는지) DEBUG_VIZ_DL_LANE으로 확인할 것.
DL_DA_SEARCH_HALF_WIDTH_PX = 100.0
DL_DA_SEARCH_WIDEN_STEP_PX = 20.0
DL_DA_SEARCH_WIDEN_MAX_PX = 200.0
DL_DA_VELOCITY_EMA_ALPHA = 0.3      # 밴드 간 이동 속도(px/밴드) EMA 계수 — DL_LL_VELOCITY_EMA_ALPHA와 동일 관례
DL_DA_VELOCITY_MAX_PX = 40.0        # 예측 이동량 클램프
DL_DA_BAND_ANCHOR_ALPHA = 0.35      # 밴드별 탐색창 중심 계산 시 "직전 프레임 그 밴드 위치"에 주는 가중치

# [2026-08-18] DL_LL_SANITY_MIN_RATIO(ll sanity check) 삭제 — lane_valid/path_ok 모두
#   da 중심점 유무로만 판정하도록 바꿈(perception/dl_lane.py 참고, ll 미사용 확정에 따른
#   정리, README §2.42 연장선).
# da가 옆 차선과 이어붙었을 때 ll 라인 바깥(옆 차선 쪽) 픽셀을 잘라내는 여유폭(px)
#   8 = 실측 라인 두께 2.5cm(=5px @200px/m) + 세그멘테이션 경계 흔들림(1~2px) 여유
#   (위 [LQR 브랜치 이식] 주석 참고). 옛 값 15px은 필요 이상으로 넓게 잘라내 정상
#   자기차선 폭까지 깎아내는 부작용이 있었다.
DL_LL_CLIP_MARGIN_PX = 8

# [2026-08-07] _clip_da_by_ll() 전용 ll 잔상(decay) — da가 옆 차선과 완전히 한 덩어리로
#   붙어버리는 실패모드를 실차에서 재현해보니(캡처 프레임: ll_cov=0.022, ll_bands=0/8,
#   즉 ll이 프레임 전체에서 거의 안 보임), "얇은 다리로 이어붙는다"는 가정과 달리 da
#   자체가 애초에 두 차선을 구분하는 내부 경계 없이 뭉텅하게 나왔다 — 침식(erosion)으로
#   끊을 구조가 없어서 그 방향은 포기(실차 재현 결과 반영). 대신 최근 몇 프레임 동안
#   확실했던 ll 픽셀을 감쇠 가중치로 유지해 이번 프레임 ll이 비어도 클리핑 근거로 계속
#   쓴다. DL_LL_DECAY_ALPHA는 매 프레임 곱해지는 감쇠율(1에 가까울수록 오래 남음),
#   DL_LL_DECAY_MIN_VALUE는 "아직 보이는 것"으로 칠 최소 잔상값(0~255 스케일, ll_mask와
#   동일). 0.8^n < 128/255 ≈ 0.5 → n≈3.1이라 대략 3~4프레임(추론 프레임 기준, 제어루프
#   20Hz와 별개 — 모듈 상단 "실시간 전략" 주석 참고) 뒤 자연 소멸한다. centerline
#   추출(_ll_slice_centers)에는 이 잔상을 안 쓴다 — waypoint를 과거 위치로 미는 건 더
#   위험하고, 클리핑은 "울타리" 역할이라 약간 stale해도 안전하다는 판단. 실차 미검증
#   초기값 — 짧은 끊김엔 도움 되는지, 너무 오래 남아 실제 경계 이동을 못 따라가진
#   않는지 DEBUG_VIZ_DL_LANE으로 확인할 것.
DL_LL_DECAY_ALPHA = 0.8
DL_LL_DECAY_MIN_VALUE = 128.0

# ── [2026-08-10] 밴드별 중심 계산 모드 스위치 — 세 모드가 서로 완전히 다른 알고리즘 ──
#   'da'    : 밴드별 중심을 da(주행가능영역) 무게중심(_slice_centers(), cv2.moments)으로
#             계산한다(main 기본값). 덩어리 선택은 DLSlideWindow._largest_da_component()
#             — ①시드(차량 위치와 맞닿은 덩어리) → ②연속성(직전 프레임과 가장 가까운
#             덩어리) → ③면적순위(최후 폴백) 순, 면적 상한(max) 체크는 없다(2026-08-10
#             제거 — 실차 검증 결과 면적만으로 da를 거르는 방식 자체가 불신뢰. da가 옆
#             차선과 붙는 문제는 이제 전적으로 _clip_da_by_ll()이 담당). 하한
#             (DL_DA_MIN_COMPONENT_AREA)은 "사실상 안 보임" 노이즈 필터로 유지.
#   'll_da' : [2026-08-10] "corridor" 알고리즘으로 교체 — ll(차선)로 도로 폭 자체를
#             규정하고, da는 그 안에서 장애물 회피용 열린 공간을 찾는 데만 쓴다. 밴드마다
#             ll을 왼쪽부터 정렬해(DLSlideWindow._ll_line_centers(), 흰/노랑 구분 없는
#             원본 ll_mask 사용 — 노란 중앙선도 "2번째 선"으로 그대로 센다) 1번째~3번째
#             선을 도로 경계(전체 트랙, 양쪽 차로 폭)로 삼는다. 그 x범위 안에서만 da를
#             보고 실제 열린(장애물 없는) 구간을 찾아(DLSlideWindow._pick_open_run(),
#             직전 프레임 위치에 가장 가까운 구간을 우선하는 히스테리시스 있음) 그 중심을
#             밴드 중심으로 쓴다(DLSlideWindow._corridor_slice_centers()). "자기 차선
#             하나"를 전제로 한 _largest_da_component()/_clip_da_by_ll()은 건너뛰고
#             클리핑 전 원본 da(da_mask_all_roi)를 그대로 쓴다 — 장애물이 도로를 좌/우로
#             쪼갤 때 그 두 함수는 지나갈 수 있는 작은 쪽을 통째로 버리거나 잘라내
#             버려서 corridor 취지(양쪽 차로를 동시에 보고 그 안에서 고른다)와
#             반대다. 밴드에서 검출된 선이 3개 미만이거나 corridor 폭이
#             DL_CORRIDOR_WIDTH_MIN/MAX_PX 밖이면 그 밴드는 da 폴백 없이 그냥 드롭한다
#             — corridor 경계 자체가 ll에서 나오므로 ll이 불충분한 순간엔 "도로 폭이
#             얼마인지" 판단할 근거가 없기 때문.
#   'll'    : ll을 흰선/노란선으로 분리(DLSlideWindow._split_ll_by_yellow(), 커넥티드
#             컴포넌트 단위로 HSV 노란색 겹침 비율 투표 — 픽셀 단위로 빼는 것보다 dash
#             가장자리가 깔끔함)한 뒤, **노란 중앙선 + (내 차선 판정에 따른) 한쪽 흰색
#             경계선**을 추적한다(_ll_yellow_white_centers()). [2026-08-10] 원래는
#             "좌/우 흰선 두 개를 독립 추적"하는 모델이었는데, 실제 도로가 편도 1차로
#             기준 흰-노-흰 구조라 노란선이 있는 쪽엔 애초에 흰선 탐색이 실패할 수밖에
#             없어(노란선은 흰선 마스크에서 제외됨) 실차 영상에서 검출 밴드가 계속 0~1/8
#             이었던 게 확인돼 재설계했다:
#               ① 차선 판정: 근거리 밴드의 노란선이 seed(차량 위치, x 중앙) 기준 왼쪽에
#                 있으면 "나는 우측차선 주행중"(흰 경계선은 오른쪽에서 탐색), 오른쪽에
#                 있으면 "좌측차선 주행중"(왼쪽에서 탐색) — self.lane_side에 기록.
#               ② 밴드마다 노란선/흰선을 각각 좁은 창(DL_LL_SEARCH_HALF_WIDTH_PX)으로
#                 독립 탐색. 둘 다 찾으면 중점 채택 + 간격(self._white_yellow_gap_px,
#                 DL_LL_YELLOW_GAP_EMA_ALPHA로 EMA) 갱신.
#               ③ 노란선만 찾으면(흰선 실패) → 간격만큼 흰선 위치를 추정해서 중점 계산.
#               ④ 노란선을 못 찾으면(이번 밴드) → [2026-08-10] 좁은 창 하나로 흰선
#                 하나만 보고 간격을 역적용하던 옛 방식은, 실차 15초 지점에서 gap
#                 EMA가 노이즈로 부푼 뒤 그 값 그대로 실제 흰선 위치를 무시하고
#                 차선 밖으로 waypoint를 밀어내는 문제가 확인돼(README §2.18)
#                 재설계했다. 넓은 창(DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX)으로
#                 흰선을 몇 개나 찾았는지로 3분기한다:
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
#   [2026-08-10] main 기본값을 'da'에서 'll'로 전환(요청 반영) — 위 ①~⑤ 재설계 이후
#   실차 재검증 목적. 'll'/'ll_da' 둘 다 여전히 실차 미검증 상태라, SPEED_NORMAL도
#   같이 낮춰뒀다(위 참고). 문제가 생기면 이 값을 'da'로 되돌릴 것.
# [2026-08-13] 'll_da' → 'da'(요청 반영) — ll_da/ll 둘 다 노란선 인식 불안정으로 계속
#   막혀서, 일단 da 자체 검출 품질만 따로 보려고 전환. 아래 DL_DA_SKIP_LL_CLIP과 짝.
DL_CENTER_MODE = 'da'  # 'da' | 'll_da' | 'll'

# [2026-08-13] DL_CENTER_MODE='da' 테스트 전용 — "da가 자체적으로 잘 검출된다"는
#   가정하에 ll(차선) 기반으로 옆 차선을 잘라내는 _clip_da_by_ll() 단계를 통째로
#   건너뛴다(요청 반영). True면 largest-component만 남긴 da_mask를 클리핑 없이 그대로
#   센터라인 계산에 쓴다 — da_ll_clip_skipped/da_clip_band_virtual 등 기존 "클리핑
#   건너뜀" 디버그 표시를 그대로 재사용해 visualize()에도 반영된다(detect() 참고).
#   DL_CENTER_MODE='ll'에는 영향 없음 — 그쪽은 이 플래그와 무관하게 항상 클리핑 적용.
#   ★주의★ 클리핑을 끄면 옆 차선 침범 가드가 없어지므로, da 세그멘테이션 품질 자체만
#   보는 실험용이지 이 상태로 실주행에 쓰라는 뜻이 아니다. 테스트 끝나면 True로 남겨두지
#   말 것(또는 다시 DL_CENTER_MODE로 전환할 때 같이 False로 되돌릴 것).
# [2026-08-17] False로 되돌렸다가(위 §주석 "복구" 시도) 실차 확인 결과 이 클리핑
#   (_clip_da_by_ll(), ll 라인 기준으로 da를 잘라 그중 ref_x에 가장 가까운 구간만 남기는
#   동작)이 의도한 동작이 아니라는 게 확인돼 다시 True로 되돌림 — da 단독 테스트 설정
#   유지.
DL_DA_SKIP_LL_CLIP = True

# ── [2026-08-14] da 안전마진(차량 폭) 침식 — DL_CENTER_MODE='da' 전용 ──
#   실차에서 "장애물 회피 중 앞코가 장애물 뒷꽁지를 긁는다"는 보고가 나와 원인을 짚어보니,
#   _da_slice_centers_windowed()가 da 중심선을 뽑을 때 차량을 폭 0인 점으로 취급해서
#   da 경계(장애물이든 트랙 벽이든)에 바짝 붙여 경로를 뽑고 있었다(README §2.30). 라이다
#   실측 결합안(같은 절)은 라이다-카메라 오프셋 실측이 먼저 필요해 시간이 걸리므로,
#   우선 더 단순한 안 — da 마스크 자체를 차폭(VEHICLE_WIDTH_M)+여유만큼 침식(erosion)해서
#   중심선이 da 경계에서 최소한 이만큼은 떨어지도록 강제한다(ROS2 Nav2의 costmap inflation과
#   동일한 개념) — 를 실차에서 먼저 테스트해보기로 했다(요청 반영, "안 되면 수정").
#   ★주의★ 이건 da 세그멘테이션 자체는 그대로 두고 후처리로 마진을 만드는 것뿐이라,
#   DL_PIXELS_PER_METER(설계값, 실측 아님 — 위 주석 참고)의 원거리 외삽/피치 변화 취약성을
#   그대로 물려받는다 — "몇 cm 여유"를 엄밀히 보장하진 못하고 근사치다. 커브가 심한
#   구간에서 침식이 과해 da가 DL_DA_MIN_COMPONENT_AREA 밑으로 꺼지면(=그 프레임 무효
#   처리) 실차에서 확인 후 DL_DA_VEHICLE_MARGIN_M을 낮출 것. 실차 미검증 초기값.
DL_DA_APPLY_VEHICLE_MARGIN = True
DL_DA_VEHICLE_MARGIN_M = 0.05   # ASTAR_VEHICLE_MARGIN_M(라이다/Hybrid A* 쪽)과 동일 관례 — 좌우 마진

# [2026-08-17g] 위 DL_DA_VEHICLE_MARGIN_M(+VEHICLE_WIDTH_M/2)은 좌/우/전/후 모두 같은 반경으로
#   침식하는 등방(isotropic) 원형 커널이라, 방해차량 뒤쪽(=지나간 뒤 재진입하는 da 영역)도
#   좌우와 똑같은 폭만큼만 벌어져 있었다(질문 확인 결과 — _apply_vehicle_margin()이
#   cv2.MORPH_ELLIPSE를 정사각형(반경 동일)으로 만들어 씀). 속도가 높을수록 접근 상대속도가
#   커져 "앞코가 장애물 뒷꽁지를 긁는" 여유가 더 필요하다는 요청 반영 — da 마스크의 세로축
#   (BEV 캔버스 row, DL_BEV_FAR_CROP_ROW 주석 참고: row가 작을수록 원거리=전방, 클수록
#   근거리=차량 쪽)이 곧 진행방향이므로, 세로 반경만 v_mps에 비례해 더 키운다(가로=좌우 폭은
#   DL_DA_VEHICLE_MARGIN_M 그대로). extra_m = min(DL_DA_REAR_MARGIN_REACT_SEC * v_mps,
#   DL_DA_REAR_MARGIN_MAX_M) — REACT_SEC은 "제동/재계획까지 걸리는 시간"의 근사치로 삼은
#   설계값(실측 아님), 단위가 s인 게 자연스럽도록 "속도(m/s) × 시간(s) = 거리(m)"로 뒀다.
#   MAX_M은 da가 통째로 침식돼 사라지는 걸(§2.30, _apply_vehicle_margin() 폴백 참고) 막는 상한.
#   실차 미검증 초기값 — 코너에서 da가 자주 무효 처리되면 REACT_SEC/MAX_M을 낮출 것.
DL_DA_REAR_MARGIN_REACT_SEC = 0.2   # 속도(v_mps)에 비례해 "뒤" 방향 마진을 추가로 늘리는 반응시간(s)
DL_DA_REAR_MARGIN_MAX_M = 0.5       # 위 추가 마진의 상한(m) — 대략 VEHICLE_LENGTH_M(0.64) 이내로 캡

# ── [2026-08-14] 회피 "복귀 유예"(avoid-hold) — DL_CENTER_MODE='da' 전용 (README §2.32) ──
#   위 안전마진 침식은 da에 뚫린 장애물 구멍 주변을 자연스럽게 우회하게 만들 뿐, "언제
#   원래 차선 중앙으로 돌아가도 되는지"는 전혀 모른다. 카메라가 차량 앞코에 달려있어서
#   장애물을 실제로 지나치는 순간 그 장애물이 화면(및 da 구멍)에서 사라지고, da가 그
#   프레임에 바로 원래 폭으로 돌아와 중심선도 즉시 원래 차선 중앙으로 복귀한다. 장애물이
#   정지해 있으면 문제없지만(지금 실차에서 관찰되는 회피는 전부 정지 장애물 기준 —
#   TEST_DISABLE_B2_B3=True라 README §2.30 배경 참고), 장애물이 방해차량처럼 계속
#   주행 중이면 "지나친 그 순간"엔 아직 옆이나 뒤에 바짝 붙어있을 수 있어 너무 이른
#   복귀가 그 차와의 충돌(추월 후 방해차량이 우리 뒤를 들이받는 상황)로 이어질 수
#   있다는 우려가 나왔다(요청 반영, 실차 미검증).
#   대응: perc_obstacle()의 obstacle_front/obstacle_dist(라이다, B2/B3 미션 자체와 무관하게
#   TEST_DISABLE_B2_B3와 상관없이 매 틱 갱신됨)를 근거로, 장애물이 AVOID_HOLD_TRIGGER_DIST_M
#   안으로 마지막으로 들어왔던 시점부터 AVOID_HOLD_SEC초 동안은 da를 raw centroid 그대로
#   쓰지 않고 ll(차선)로 "지금 차선 하나"만 남기도록 강제로 자른다 — 예전부터 있던
#   _clip_da_by_ll() 클리핑(DL_DA_SKIP_LL_CLIP=True로 평소엔 꺼둔 것)을 이 창에서만
#   되살리는 방식(track_drive.py _update_avoid_hold()/perception/dl_lane.py detect() 참고).
#   장애물이 시야에서 사라진 직후에도 몇 초간은 옆 차선으로 안 새고 지금 차선 폭 안에서만
#   주행해, 급하게 원래 차선 중앙으로 꺾여 들어가는 걸 늦추는 목적.
#   ★주의★ 둘 다 실차 미검증 초기값이다.
AVOID_HOLD_TRIGGER_DIST_M = 1.5   # 이 거리(m) 안으로 장애물이 들어오면 "회피 중"으로 본다

# [2026-08-15] avoid-hold 개선 — 가변 유예시간 + 거리기반 조기 해제 + da 연속성 보조트리거 +
#   방향 힌트 + 안전판(avoid_hold_improvement_proposal.md "1차 적용 결정" 적용1~4). 기존
#   고정 AVOID_HOLD_SEC(=2.0, 아래 AVOID_HOLD_SEC_BASE로 대체)만으로는 "짧으면 방해차량에
#   위험, 길면 정지장애물엔 낭비"라는 트레이드오프를 하나의 숫자로 감내할 수밖에 없었다 —
#   설계 배경/시나리오별 사이드이펙트/대비책은 위 proposal 문서, 실제 시나리오 비교는
#   같은 문서가 링크한 다이어그램 참고. ★ 아래 값 중 실측이 필요한 건 track_drive/
#   track_drive/avoid_hold_measurement_todo.md에 측정 절차와 함께 정리해뒀다 — 실차
#   테스트 전에 그 문서부터 볼 것. DEBUG_VIZ_AVOID_HOLD(아래 "5. 디버깅 ON/OFF" 참고) 창이
#   이 값들과 지금 상태를 실시간으로 같이 보여준다 ★

# [적용1] 유예시간 계산(track_drive.py _update_avoid_hold()) —
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

# [적용1] 조기 해제(같은 함수) — obstacle_front=False가 RELEASE_CONFIRM_FRAMES 연속 +
#   마지막으로 obstacle_front=True였을 때의 obstacle_dist가 RELEASE_DIST_M 이상이면
#   hold_sec을 다 채우기 전에도 즉시 해제한다. "안 보임=멀어짐"이 아니라 "마지막으로 봤을
#   때 이미 멀었음"으로 조건화해 라이다 사각지대와 실제 이탈을 최소한이나마 구분한다
#   (문제6 대비책) — 그래도 둘을 완전히 못 가르는 건 여전한 한계(proposal 문서 참고).
AVOID_HOLD_RELEASE_DIST_M = 2.0        # TRIGGER_DIST_M(1.5)보다 크게 둬 히스테리시스 확보. ★실측 필요★
AVOID_HOLD_RELEASE_CONFIRM_FRAMES = 4  # SIG_CONFIRM_FRAMES(3)/VEHICLE_TRIGGER_FRAMES(5) 등과
                                        #   동일 관례 — 순간적 미검출(사각지대/flicker)로
                                        #   조기해제가 새는 것 방지.

# [적용2] da 연속성 보조 트리거(perception/dl_lane.py DLSlideWindow) — 라이다 사각지대
#   (문제3) 보완. 이번 프레임 da_chosen_area_px가 직전 프레임 대비 이 배율 이상 급증하면
#   (=방금까지 뚫려있던 구멍이 갑자기 메워짐) "뭔가 방금 시야에서 사라졌을 수 있다"는
#   신호로 보고, 라이다 obstacle_front 트리거와 OR로 결합한다(단독 트리거로는 안 씀 —
#   세그멘테이션 자체가 흔들리는 프레임에서 오발동할 수 있어서, 문제3 대비책).
AVOID_HOLD_DA_AREA_JUMP_RATIO = 1.4    # ★실측 필요★

# [적용3] 방향 힌트(track_drive.py _update_avoid_hold()가 매 틱 계산 → perc_lane()이
#   DL 백엔드로 전달 → perception/dl_lane.py _clip_da_by_ll()) — TargetPassing.choose_side()
#   가 반환한 side(-1/0/+1, lane_offset과 동일한 "우측+" 부호규약)를, _clip_da_by_ll()의
#   "ll도 잔상도 없는" 최후수단 가상경계 폴백에서만 기준점을 이만큼(px) 안전한 쪽으로
#   미리 기울이는 데 쓴다 — 실측/잔상 등 실제 증거가 있는 밴드는 건드리지 않는다(방향
#   힌트가 실패해도 원래 로직으로 조용히 폴백되는 소프트 제약, 문제2 대비책).
AVOID_HOLD_DIR_BIAS_PX = 20.0   # ≈ PASS_OFFSET(80.0, "7. 기타" 절, 실측 기반)의 1/4.
                                 #   ★비율 자체는 실측/재검증 필요★

# [2026-08-18] [적용4] SPEED_AVOID_HOLD_BLOCKED 안전판 삭제 — 실차 테스트에서 "속도 5
#   고정" 증상의 실제 원인으로 확인됨(README §2.43 참고). TEST_DISABLE_B2_B3=True로 실제
#   회피 기동(옆차선 이동)은 꺼져있는데 이 캡만 무관하게 계속 걸려서, 트리거를 풀어줄
#   수단이 없어 무한정 고정되는 구조였다. avoid_hold 타이머/DA 클리핑 방향 편향(적용3,
#   AVOID_HOLD_DIR_BIAS_PX 위 참고)은 그대로 유지 — 요청 반영(속도캡 소비부만 제거).

# [2026-08-10] DL_CENTER_MODE='ll' 내부에서 실제 밴드 중심 계산 알고리즘을 고르는
#   2차 스위치 — 같은 날 두 사람이 독립적으로 서로 다른 재설계를 했다(origin/main
#   병합 시 두 구현이 정면으로 겹쳐 병합 커밋에서 "둘 다 남기고 전환 가능하게" 하기로
#   결정, 아래 README §2.19 참고).
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
#   [2026-08-12] DL_LL_VELOCITY_EMA_ALPHA/DL_LL_VELOCITY_MAX_PX/DL_LL_BAND_ANCHOR_ALPHA
#   (아래 'lr' 섹션에 있음)도 이제 'yw'가 같이 쓴다 — main 기본값인 'yw'
#   (_ll_yellow_white_centers())엔 원래 §2.23 탐색창 확장(widen)만 있고 'lr'
#   (_ll_slice_centers())의 속도예측+프레임간 앵커링이 빠져 있던 공백을 메웠다
#   (README §2.27). 새 상수를 따로 만들지 않고 재사용한 이유는 둘 다 "밴드 간
#   이동 속도를 추적해 탐색창을 미리 옮기고, 직전 프레임 그 밴드 위치로 당긴다"는
#   동일한 물리적 개념이라서다.
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

# [2026-08-12] _pick_open_run()의 프레임 간 히스테리시스(직전 프레임 채택 위치와 가장
#   가까운 open run을 우선)는 정적이라, 빠른 S자에서 실제 열린 구간 위치가 그 사이 크게
#   이동하면 뒤처질 수 있다. da/ll 모드에 적용한 것과 동일한 원리로 밴드 간 이동 속도를
#   EMA 추적해(_corridor_slice_centers()) prefer_x를 "직전 위치 + 예측 이동량"으로
#   미리 옮긴다(README §2.27). corridor는 좌/우 두 갈래가 아니라 "열린 구간 하나"만
#   추적하므로 da처럼 스칼라 하나면 된다. 실차 미검증 초기값.
DL_CORRIDOR_VELOCITY_EMA_ALPHA = 0.3
DL_CORRIDOR_VELOCITY_MAX_PX = 40.0

# DL_CENTER_MODE='ll'일 때 쓰는 ll 중점 채택 임계값.
# 밴드 내 ll 픽셀수가 이 미만이면(노란선/흰선 각각 판정) "이 밴드는 그게 안 보임" 처리.
#   DL_MIN_PIXELS(=40, da용)보다 낮은 이유: ll은 da처럼 면을 채우는 마스크가 아니라 가는
#   선이라 같은 밴드 안에 있는 픽셀수 자체가 원래 훨씬 적다. 실차 미검증 초기값.
DL_LL_SIDE_MIN_PIXELS = 15

# [2026-08-07] _ll_yellow_white_centers()가 노란선/흰선을 찾을 때 보는 탐색창 반경(px).
#   좌/우 분리 기준점 하나로 밴드를 절반씩(왼쪽 전체/오른쪽 전체, 보통 수백 px) 나눠 그
#   안 전체 픽셀로 무게중심을 내면, 그 "반쪽"이 넓다 보니 옆 차선 선이나 반사광이 반쪽
#   어디에 있든 평균에 섞여 들어가는 문제가 있다(다중 후보 오탐). 참고:
#   github.com/junhyukch7/Advanced-Lane-Detection의 슬라이딩 윈도우가 폭 120px(반경
#   60px)짜리 좁은 창만 보는 것에서 착안 — 창 밖의 무관한 픽셀이 애초에 평균 계산에 안
#   들어오게 예상 위치 중심의 좁은 창만 보도록 한다. 실차 미검증 초기값(참고 프로젝트와
#   동일하게 60으로 시작) — 급커브에서 밴드 간 실제 선 이동량이 이 값보다 크면 창이
#   선을 놓치고 추적이 끊길 수 있으니, 그런 구간에서 검출 밴드 비율이 뚝 떨어지면 이
#   값을 키울 것.
DL_LL_SEARCH_HALF_WIDTH_PX = 60.0

# [2026-08-10] DL_CENTER_MODE='ll' 재설계 — "좌/우 흰선 두 개를 독립 추적"에서 "노란
#   중앙선 + (차선 판정에 따라) 한쪽 흰색 경계선"으로 바꿨다(실제 도로 구조가 편도 1
#   차로 기준 흰-노-흰이라, 옛 모델은 노란선이 있는 쪽엔 애초에 흰선 탐색이 거의 항상
#   실패하는 구조적 문제가 있었음 — 실차 영상에서 white_bands가 계속 0~1/8이었던 원인).
#   노란선 대비 흰색 경계선까지의 간격(px) 러닝 추정치 self._white_yellow_gap_px의 초기값
#   /EMA 계수. 둘 다 찾은 밴드에서만 이 계수로 갱신한다(_ll_yellow_white_centers() 참고).
#   [2026-08-10] 초기값을 실측 기반으로 교체 — 흰-노 간격 실측 0.4m을
#   DL_PIXELS_PER_METER(200px/m)로 환산해 80px(=0.4*200)로 잡았다. 처음엔
#   DL_LL_SEARCH_HALF_WIDTH_PX(60) 근방으로 대충 잡았던 값이 우연히 비슷했을 뿐,
#   이제는 그 실측값으로 대체된 것 — 다만 DL_PIXELS_PER_METER 자체가 "설계값(실측
#   아님)"이라(위 주석 참고), 이 200px/m 환산이 실제로 맞는지는 별도 확인 필요.
#   DEBUG_VIZ_DL_LANE에서 정상 구간의 gap 표시값이 80px 근방으로 수렴하는지 보고
#   재조정할 것 — 크게 벗어나면 DL_PIXELS_PER_METER 쪽 오차를 의심할 것.
DL_LL_YELLOW_GAP_INIT_PX = 80.0
DL_LL_YELLOW_GAP_EMA_ALPHA = 0.1

# [2026-08-10] gap EMA 상하한 클램프 — 실차 영상(15초 지점)에서 노란선이 죽기 전
#   노이즈(글레어 등)로 몇 프레임 큰 |흰선-노란선| 값이 잡히면서 EMA가 161px까지
#   부풀었고, 그 직후 노란선이 아예 안 잡히기 시작해 "둘 다 찾았을 때만 갱신"되는
#   이 값이 부푼 채로 얼어붙어버린 게 확인됐다(실측 40cm=80px의 두 배). 그 상태로
#   한쪽 선 없는 밴드의 위치 추정에 이 부푼 gap을 그대로 썼더니 waypoint가 실제
#   흰선을 넘어 차선 밖까지 밀려나 급조향(15초 우회전)으로 이어졌다. 실측값(80px)
#   근방으로 상하한을 걸어 어떤 노이즈가 껴도 이만큼은 안 부풀게 막는다 — 여유폭은
#   경험적으로 잡은 값(실측 미세조정 여지 있음), DEBUG_VIZ_DL_LANE의 gap 표시값이
#   이 범위 끝에 계속 붙어있으면 실제 트랙 폭이 이 범위 밖일 수 있다는 뜻이니 재조정할 것.
DL_LL_YELLOW_GAP_MIN_PX = 50.0
DL_LL_YELLOW_GAP_MAX_PX = 110.0

# [2026-08-10] 노란선이 이번 밴드에서 안 보일 때 흰선을 찾는 탐색창 반경(px) —
#   DL_LL_SEARCH_HALF_WIDTH_PX(60, 노란/흰 각각 하나씩 좁게 찾는 창)와 별개로,
#   "노란선 없을 때 양쪽 흰선이 몇 개나 보이는지"를 세야 하므로 그보다 훨씬 넓게
#   잡는다 — 좌우 흰선이 각각 gap(최대 DL_LL_YELLOW_GAP_MAX_PX=110)만큼 떨어져
#   있을 수 있으므로 그보다 여유를 더 둔 값. cur_yellow(기준점) 중심으로 이 반경
#   안의 흰선 connected component를 전부 찾아(_ll_line_centers() 재사용) 개수로
#   3분기한다(_ll_yellow_white_centers() 참고): 2개=양쪽 다 보임(중점 채택),
#   1개=한쪽만 보임(어느 쪽인지 실측 위치로 판정 후 gap만큼 안쪽으로 재구성),
#   0개=잔상. 실측 미검증 — 실제 트랙 폭 기준으로 재조정할 것.
DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX = 150.0

# ── DL_LL_ALGO='lr'(좌/우 흰선 독립 슬라이딩 윈도우, _ll_slice_centers()) 전용 튜닝값 ──
#   [2026-08-10 병합] 'yw'(위)가 main 기본이 되면서 병합 충돌로 지워질 뻔했으나, 두
#   알고리즘을 둘 다 살리고 DL_LL_ALGO로 전환 가능하게 하기로 해서 복원했다(README
#   §2.19). 원래 주석: 밴드 내 좌/우 ll 중점을 채택하기 위한 두 선 사이 거리(px)
#   허용범위 — 범위 밖이면(반대쪽 밴드의 다른 차선을 잘못 짝지은 경우 등) 그 밴드는
#   버린다(da 폴백 없음). 실측 라인 간격이 75~80px로 나와 하한을 100→50으로 낮춘 이력
#   있음(2026-08-07) — 여전히 넓게 열려있으니 visualize()의 밴드별 실측 폭 표시로 좁힐 것.
#   실차 미검증 초기값.
DL_LL_WIDTH_MIN_PX = 50
DL_LL_WIDTH_MAX_PX = 200
# 좌/우 둘 다 찾아 실측 폭이 나온 밴드에서만 self._ll_half_width(차로 반폭 러닝
#   추정치)를 이 계수로 EMA 갱신한다. classic_cv 백엔드의 LANE_WIDTH_EMA_ALPHA(=0.1,
#   hough_lane.py)와 동일한 관례.
DL_LL_WIDTH_EMA_ALPHA = 0.1

# [2026-08-10] _ll_slice_centers()(DL_LL_ALGO='lr') 적응형 탐색창 — 밴드 간 실제 선 이동량이
#   DL_LL_SEARCH_HALF_WIDTH_PX보다 크면 창이 선을 놓치는 문제(위 주석 "급커브에서...
#   추적이 끊길 수 있으니" 참고) 대응. 두 갈래로 완화한다:
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
PP_LOOKAHEAD_CURVATURE_GAIN = 224.8  # 직전 프레임 curvature가 클수록(코너) lookahead를 줄이는 게인 — 100.0→224.8(그리드서치)
PP_LOOKAHEAD_MIN_PX = 62.61        # 코너에서 lookahead가 줄어들 수 있는 하한 — 40.0→62.61(그리드서치)
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
# [2026-08-17c] 위 문제는 이제 아래 PP_STRAIGHT_ALPHA가 직진 확정 중에만 담당하도록
#   분리했다(pure_pursuit.py의 "직진(A)/코너+S자(B) 2상태 분기" — is_straight일 때만
#   PP_STRAIGHT_ALPHA, 그 외엔 이 값을 쓴다). 그래서 이 값은 이제 "코너/S자 등 직진이
#   아닐 때"의 필터로 그 의미가 좁혀졌고, 그 조건에서는 진동 억제보다 반응성이 우선이라
#   0.35 → 0.8로 다시 올렸다(대화 중 코너90도/S자커브 합성 시뮬레이션 그리드서치 상위권
#   — corner_lag/scurve_amp_ratio 기준). 직진 잡음 억제는 더 이상 이 값이 아니라
#   PP_STRAIGHT_ALPHA + PP_STRAIGHT_DEADZONE_PX + 아래 편향감지(bias EMA) 조합이 담당한다.
#   실차 미검증 — 화이트박스 합성 시뮬레이션 결과이므로 실차 재검증 필수.
# [2026-08-17h] 0.8→0.5244(그리드서치, PP_WHEELBASE_PX 상향과 함께 재탐색된 조합).
PP_ALPHA = 0.5244                  # 프레임간 조향각 저역통과 필터(1=필터없음, 0=반응없음) — "코너/S자"(비직진) 상태 전용
PP_MIN_LOOKAHEAD_PX = 86.95        # curvature 분모(ld) 바닥값 — 노이즈 증폭 방지용. PP_LOOKAHEAD_MIN_PX와 다른 값이니 헷갈리지 말 것. 90.0→86.95(그리드서치)
# [2026-08-12] 6.0 → 15.0. 직진 진동 대응 세 번째 레버 — 원래 값이 중앙 부근 잔떨림을
#   죽이기엔 너무 작아서(픽셀 몇 개짜리 노이즈도 그대로 통과) 직진에서도 매 프레임 미세한
#   조향이 나갔던 것으로 추정. LANE_DEADZONE(구 PID 전용, 40px)보다는 여전히 훨씬 작게
#   유지 — Pure Pursuit 목표점은 이미 lookahead 앞 실제 경로점이라 그만큼 크게 죽이면
#   완만한 커브 진입까지 무시하게 된다(pure_pursuit.py 상단 주석 참고). 실차 미검증.
# [2026-08-13] 15.0 → 5.0(요청 반영, PP_WHEELBASE_PX를 67→40으로 줄여 조향 반응 자체가
#   약해진 만큼 데드존도 같이 줄임 — 노란선 흔들림 대응) 했었으나, [2026-08-17] PP_ALPHA와
#   같은 이유로 12.0으로 절충 원복.
# [2026-08-17c] PP_ALPHA와 같은 이유로 의미가 "코너/S자(비직진) 상태 전용"으로 좁혀졌다 —
#   직진 잡음 억제는 PP_STRAIGHT_DEADZONE_PX가 담당하므로, 이 값은 이제 코너/S자 추종
#   반응성 쪽으로 다시 낮췄다(12.0 → 6.0, 그리드서치 상위권 조합). 실차 미검증.
# [2026-08-17h] 6.0→4.445(재증속 후 재실행한 그리드서치, pp_tune_gridsearch.py).
PP_DX_DEADZONE_PX = 4.445          # 이 이하 픽셀오차는 0으로 죽여 중앙 부근 잔떨림 제거 — "코너/S자"(비직진) 상태 전용

# [2026-08-17] 명시적 "직진 모드"(README §0.5.9) — 지금까지의 진동 억제(PP_ALPHA/
#   PP_DX_DEADZONE_PX)는 전부 "연속값을 더 세게 누르는" 방식이라 코너 반응성과 항상
#   트레이드오프였다. probe_curvature(코너 판단 신호, pure_pursuit.py control())가 연속
#   PP_STRAIGHT_CONFIRM_FRAMES 프레임 동안 이 값 미만이면 "직진 확정" 상태로 보고, 그
#   동안만 데드존을 PP_STRAIGHT_DEADZONE_PX로 넓힌다 — 코너 중엔 이 조건 자체가 안
#   걸리므로 코너 추종 감도는 그대로다. 셋 다 실차 미검증 첫 추정치.

# [2026-08-17e] 0.001은 너무 타이트했다 — 실차 녹화(카카오톡 영상, 2026-08-17 14:00)에서
#   육안으론 명백한 직진 구간이 계속 "커브대응"(주황)으로 표시됨을 확인. probe_curvature=
#   2*sin(alpha)/probe_ld 공식상 0.001은 probe_ld=90~150px(PP_LOOKAHEAD_BASE_PX~MAX_PX,
#   일반 주행 중 실제 쓰이는 범위) 구간에서 dx 약 5~11px만 넘어도 넘는 값이다 — 이 정도
#   dx 흔들림은 이 파일 위쪽 min_lookahead_px 주석이 직접 든 실측 예("ld=42px, dx=3px
#   (육안으론 거의 직진)여도 curvature≈0.0034")보다도 작아, 세그멘테이션의 프레임 간
#   서브픽셀~픽셀 단위 잡음만으로도 상시 초과했을 가능성이 높다. 그 실측 예(0.0034)가
#   "거의 직진"으로 불렸다는 걸 기준 삼아 0.0035로 3.5배 완화 — 해제는 여전히 즉시(단일
#   프레임 디바운스 없음)라 실제 코너 진입 반응성엔 영향 없다(위 PP_STRAIGHT_CONFIRM_FRAMES
#   주석 참고). 실차 재검증 필요.
# [2026-08-17h] 셋 다 pp_tune_gridsearch.py 재실행(SPEED_NORMAL=15.0) 결과로 교체.
PP_STRAIGHT_CURVATURE_EPS = 0.003283  # 이 미만이면 "사실상 직진" — 0.0035→0.003283(그리드서치)
PP_STRAIGHT_CONFIRM_FRAMES = 10       # 연속 이 프레임 수만큼 유지돼야 직진 확정(20Hz 기준 0.5초, 5→10). 해제는 즉시(디바운스 없음) — 코너 진입 반응이 늦어지면 안 되므로
PP_STRAIGHT_DEADZONE_PX = 21.64       # 직진 확정 중에만 적용하는 넓은 데드존 — PP_DX_DEADZONE_PX보다 커야 함. 20.0→21.64(그리드서치)

# [2026-08-17c] "직진(A)/코너+S자(B) 2상태 분기" 확장 — 원래 직진모드는 PP_DX_DEADZONE_PX
#   하나만 넓혔는데(위), PP_ALPHA(저역통과)도 직진과 코너/S자가 원하는 값이 정반대라는 게
#   대화 중 시뮬레이션으로 확인됐다(직진: 강한 필터가 유리, 코너/S자: 약한 필터가 유리).
#   그래서 직진 확정 중에는 PP_ALPHA 대신 이 값을 쓴다(pure_pursuit.py control()의
#   filter_alpha 분기). PP_WHEELBASE_PX는 두 상태 그리드서치에서 공통으로 25.0을
#   선호해 굳이 분리하지 않았다. 실차 미검증 — 화이트박스 합성 시뮬레이션 추정치.
# [2026-08-17h] 0.4→0.5096(그리드서치 재실행) — 더 이상 "PP_ALPHA(0.8)보다 낮게"가 아니게
#   됐다(PP_ALPHA 자체도 0.5244로 낮아짐, 둘이 거의 같은 값) — 직진/코너 필터 차등 자체는
#   유지되지만 격차가 좁혀졌다는 뜻.
PP_STRAIGHT_ALPHA = 0.5096          # 직진 확정 중 조향각 저역통과 필터

# [2026-08-17c] "코너 탈출 직후 살짝 틀어진 채 직진 진입" 대응 — 위 PP_STRAIGHT_DEADZONE_PX
#   (넓은 데드존)는 곡률만 보고 확정되므로, 곡률은 0인데 dx만 한쪽으로 계속 쏠려있는
#   잔여 오프셋을 영원히 못 지울 위험이 있다(대화 중 시뮬레이션으로 재현). raw dx(데드존
#   적용 전)의 EMA가 PP_STRAIGHT_DEADZONE_PX 폭을 넘으면 노이즈가 아니라 진짜 편향으로
#   보고 곡률 조건과 무관하게 직진확정을 해제한다(pure_pursuit.py __init__/control() 상단
#   주석 참고). 실차 미검증 첫 구현 — 편향 회복이 너무 느리면 이 값을 올릴 것(반응은
#   빨라지지만 노이즈에도 더 민감해짐).
# [2026-08-17h] 0.15→0.06785(그리드서치 재실행) — 편향 감지가 더 느려짐(반응은 느려지지만
#   노이즈에는 덜 민감).
PP_STRAIGHT_BIAS_EMA_ALPHA = 0.06785

# ── 차량 물리 상수 ──
# [2026-08-14] 옛 이름 LQR_WHEELBASE_M → WHEELBASE_M. LQR 컨트롤러 제거로 "LQR 전용"이
#   아니라 EncoderPoseEstimator(localization/pose_estimator.py)가 쓰는 일반 차량 상수임을
#   반영한 이름 변경 — 값 자체는 그대로(실측 유지).
WHEELBASE_M = 0.335         # 실측값(2026-08-06, 줄자로 앞바퀴-뒷바퀴 축간거리 실측 — LQR 브랜치에서
                             #   이식). planner/hybrid_astar.py의 wheelbase 기본값(같은 차량이므로
                             #   반드시 같은 값)과 일치시킬 것 — 재실측 시 둘 다 갱신.

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

# [2026-08-11] 라바콘 실차 테스트 중엔 라이다 창만 보고 싶다는 요청으로, 아래 DEBUG_VIZ_LIDAR만
#   켜고 나머지는 전부 잠시 끔. 다른 디버그창이 다시 필요하면(예: 차선 인식 디버깅) 개별적으로
#   다시 True로 되돌릴 것 — 서로 독립적인 스위치라 다른 항목엔 영향 없음.
DEBUG_VIZ_LIDAR    = False   # 라이다 BEV 장애물 감지 디버그 창 (track_drive.py)
DEBUG_VIZ_LAVACON  = False  # 라바콘 트리거 좌우 클러스터 BEV 디버그 창 (track_drive.py)
DEBUG_PLANNER      = False  # Hybrid A* OccupancyGrid 디버그 창 (track_drive.py, USE_HYBRID_ASTAR_FOR_B3=True일 때만 의미있음)
DEBUG_VIZ_STEER    = False  # 조향 컨트롤러(직전값유지/현재값반영) 한글 디버그 창 (track_drive.py)
DEBUG_VIZ_VESC     = False  # VESC 실측속도(/vesc_speed_erpm) 연동 상태(수신중/끊김/미수신) 디버그 창
                             #   (track_drive.py, 2026-08-06 LQR 브랜치에서 이식)

DEBUG_VIZ_DL_LANE    = True  # 차선 — 기본 백엔드('dl') 디버그 창 (perception/dl_lane.py)
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
# [2026-08-13] 신호등(S0/S2) 디버그 강화 — ROI 좌표/HoughCircles 원(후보 전체+선택된 4개)/
#   현재 인식 상태(정지·직진·좌회전·미검출)를 창 하나에 다 보여주도록 확장
#   (perception/traffic_signal.py detect_s2()). 요청에 따라 기본 True로 켜둠 — 다른 항목과
#   달리(위 2026-08-11 라바콘 테스트 메모 참고) 이 스위치는 독립적으로 True 유지할 것.
DEBUG_VIZ_SIGNAL     = False   # 신호등 ROI/HoughCircles 디버그 창 (perception/traffic_signal.py)
# DEBUG_LOG_SIGNAL: 신호등 전용 상세 진단 로그. 전역 DEBUG_LOG(0.5초 주기 요약 [SIG] 한 줄)와는
#   별개로, 이 플래그가 켜지면 S0/S2 상태에서 매 프레임 "왜 못 잡았는지"(원 개수 부족/배치 불량/
#   밝기 대비 부족 등) 원인을 자세히 찍는다 — DEBUG_LOG를 꺼도 이것만 켜서 신호등만 디버깅 가능.
#   (track_drive.py perc_signal())
DEBUG_LOG_SIGNAL     = True
DEBUG_VIZ_YOLO_CONE  = False  # 라바콘 YOLO 검출 박스 디버그 창 (perception/yolo_cone.py)
# [2026-08-15] avoid-hold(§2.32) 전용 상태창 — 지금 유예가 걸려있는지/왜 걸렸는지/방향
#   힌트/조기해제 진행상황을 한곳에 모아 보여주고, 실측 안 된 파라미터 값도 항상 같이
#   띄워서 "이 숫자 아직 지어낸 값"이라는 걸 상기시킨다(track_drive.py
#   _debug_viz_avoid_hold(), avoid_hold_measurement_todo.md 참고). 다른 회피 관련 창
#   (lidar_bev 등)과 별개로 언제든 독립적으로 켜고 끌 수 있다.
DEBUG_VIZ_AVOID_HOLD = True
#   [2026-08-11] smooth-imu-yaw-rate 브랜치(0c0d88b)에서 수동 포팅 — 라바콘 실차 테스트 중
#   라이다 창과 함께 켜 두고 나머지는 꺼둔 상태(요청 반영).


# #############################################################
# 6. 미션 State / 실차 테스트 범위 제한
# #############################################################
START_STATE     = MissionState.S1_LANE_FOLLOW
ENABLE_BEHAVIOR = False   # S1에서 라바콘/장애물/추월 Behavior를 켤지 여부(최상위 스위치)
#   [2026-08-11] 라바콘(B1) 실차 테스트를 위해 True로 켬. TEST_FORCE_BEHAVIOR=True와 함께
#   있으면 S2 교차로 없이도 시작부터 라바콘 단독 테스트 가능. B2/B3까지 실차 테스트 범위를
#   넓힐 준비가 되기 전까지는 TEST_DISABLE_B2_B3=True로 B2/B3 발동 자체는 계속 막아둔 상태.

# ── 실차 테스트 범위 제한 ──
#   지금 단계에서 실차로 검증 가능한 건 딱 세 가지: ①신호등 인식 후 출발(S0)
#   ②차선주행(S1) ③라바콘 주행(B1). 나머지(S2 교차로/S3 지름길)는 아직 실차
#   미검증(좌회전 각도·속도 placeholder)이라 테스트 중 의도치 않게 발동하면 위험할
#   수 있어 아래 플래그로 강제로 꺼둔다. → 좌회전 튜닝 끝나면 False로 되돌릴 것.
TEST_DISABLE_INTERSECTION = True
#   True: 정지선을 감지해도 감속→S2_INTERSECTION 전환을 아예 안 함(차선주행만 계속).
#   False: 원래대로 정지선 감지 시 감속 후 S2로 정상 전환.
# [2026-08-11] B2/B3 실차 테스트 시작 — True → False. 라바콘(B1) 격리 테스트는 이 값과
#   무관(B1엔 트리거 조건이 없음, apply_behavior_override() 참고)하니 그대로 True 둬도
#   B1은 계속 검증 가능하다.
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

# [2026-08-11] B2(정지 장애물) Hybrid A* 대안(USE_HYBRID_ASTAR_FOR_B2) 삭제.
#   대신 _da_avoidance_failed() 게이트 + TargetPassing(실측 기반 하드코딩)로 대체
#   — 구조화된 2차선 환경에서 검색 기반 계획은 과한 방식이라는 결론(README §4/§5.1)에
#   따름. B3(방해차량, 동적)는 여전히 USE_HYBRID_ASTAR_FOR_B3로 Hybrid A* 대안을 쓴다
#   (아래, 819행 부근).

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

# ── 라바콘/장애물/방해차량/신호등 트리거 ──
LAVACON_DONE_FRAMES = 80      # 우측콘 미검출이 연속 N프레임(20Hz→약 4초) 쌓이면 Phase 전환(디바운스)
LAVACON_TRIGGER_FRAMES = 5    # (YOLO 콘 검출 AND 좌우 라이다 클러스터 동시검출)이 연속 N프레임
                               #   쌓이면 B1_LAVACON 진입 확정. [2026-08-07] 카메라(YOLO)+라이다
                               #   이중확인으로 강화 — 값 자체는 기존 그대로 유지.

# ── 라바콘 카메라 이중확인 (perception/yolo_cone.py, YOLOv8n ONNX) ──
#   perc_lavacon_trigger()가 기존 라이다 좌우 클러스터 판정에 "카메라로도 콘이 보이는가"를
#   AND로 추가한다 — 라이다 단독 클러스터 판정은 벽 모서리 등에서 오검출 여지가 있어서,
#   실제로 콘(cone) 클래스가 화면에 잡힐 때만 진입을 인정하도록 이중화한다.
#   [2026-08-11] smooth-imu-yaw-rate 브랜치(0c0d88b)에서 수동 포팅.
YOLO_CONE_INPUT_SIZE = 640     # cone_best_n.onnx export 시 imgsz와 반드시 일치시킬 것
YOLO_CONE_CONF_THRESHOLD = 0.5 # 이 신뢰도 이상인 검출만 인정(모델이 nms=True로 export돼 좌표 디코딩은 불필요)
YOLO_CONE_MODEL_PATH = None    # None이면 yolo_ros/cone_best_n.onnx(형제 디렉터리)를 자동으로 찾음(perception/yolo_cone.py 참고)

SAFETY_DIST      = 5.0        # B2(고정장애물) 발동 거리(m)
OVERTAKE_TRIGGER = 6.5        # B3(방해차량) 발동 거리(m)
VEHICLE_TRIGGER_FRAMES = 5    # 라이다 단독검출 연속 N프레임이면 B3_VEHICLE 진입 확정
SIG_CONFIRM_FRAMES = 3        # 신호등(직진/좌회전) 판정이 연속 N프레임 유지돼야 확정(20Hz→0.15s)

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

# ── 신호등(S0/S2 공용 4구, perception/traffic_signal.py) ──
# [2026-08-13] 실차 랩 캡처(lap_001/frame_000055.png, 640x480)에서 신호등이 실제로 찍힌
#   프레임을 찾아 재튜닝. 기존 값(T,B=0.08,0.28 / L,R=0.04,0.78 / R=15~25)은 이 프레임에서
#   신호등을 아예 못 찾았다(circle_count=0) — ROI 자체가 신호등 실제 위치(native px 기준
#   대략 x=194~291, y=81~89)와 안 맞았던 것으로 보인다. 아래 값은 그 프레임에서
#   HoughCircles가 4개를 안정적으로 찾고 shape_ok()까지 통과하는 걸 확인한 값.
#   ROI를 실측 박스에 딱 맞추지 않고 일부러 여유 있게(러프하게) 잡았다 — 정지 위치/각도가
#   매번 픽셀 단위로 똑같지 않을 것이므로. 다만 얼마나 넉넉하게 잡을지는 트레이드오프였다:
#     - 타이트(185x72px): frame_000055 성공, 무작위 25프레임 오탐 0/25
#     - 러프(320x144px): frame_000055 성공하지만(pick_best_4가 노이즈 속에서도 골라냄),
#       무작위 40프레임 중 4개가 우연히 shape_ok를 통과하는 오탐 발생(4/40, 약 10%) —
#       ROI가 넓을수록 원이 더 많이 잡히고 pick_best_4()가 "적당히 나란한 4개"를 실제
#       신호등이 아닌 조합에서도 찾아버릴 수 있다.
#     - 지금 값(262x110px, 중간): frame_000055 성공, 무작위 40프레임 오탐 1/40 — 이 정도가
#       "너무 타이트하지도, 오탐이 늘지도 않는" 절충점으로 판단.
#   주의: 프레임 한 장 기준이라 다른 거리/각도(S0 vs S2)에서는 또 안 맞을 수 있다 —
#   실차에서 DEBUG_VIZ_SIGNAL/DEBUG_LOG_SIGNAL 켜고 재확인 필요.
SIG4_ROI_T, SIG4_ROI_B = 0.07, 0.30
SIG4_ROI_L, SIG4_ROI_R = 0.18, 0.50  # R을 0.58→0.50으로 축소: 그 우측 벽 패널 이음새가 원으로 오검출되던 걸 제외(2026-08-13)
SIG4_MIN_RADIUS, SIG4_MAX_RADIUS = 9, 26
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
    'speed10': dict(
        PP_LOOKAHEAD_BASE_PX=78.61, PP_LOOKAHEAD_SPEED_GAIN=1.476, PP_LOOKAHEAD_MAX_PX=263.7,
        PP_WHEELBASE_PX=49.39, PP_ALPHA=0.7678, PP_MIN_LOOKAHEAD_PX=63.26, PP_DX_DEADZONE_PX=1.626,
        PP_LOOKAHEAD_CURVATURE_GAIN=446.4, PP_LOOKAHEAD_MIN_PX=41.66,
        PP_STRAIGHT_CURVATURE_EPS=0.008086, PP_STRAIGHT_CONFIRM_FRAMES=5, PP_STRAIGHT_DEADZONE_PX=5.222,
        PP_STRAIGHT_ALPHA=0.9083, PP_STRAIGHT_BIAS_EMA_ALPHA=0.5997,
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
        PP_WHEELBASE_PX=49.17, PP_ALPHA=0.2258, PP_MIN_LOOKAHEAD_PX=52.63, PP_DX_DEADZONE_PX=10.56,
        PP_LOOKAHEAD_CURVATURE_GAIN=490.4, PP_LOOKAHEAD_MIN_PX=34.59,
        PP_STRAIGHT_CURVATURE_EPS=0.01194, PP_STRAIGHT_CONFIRM_FRAMES=7, PP_STRAIGHT_DEADZONE_PX=2.582,
        PP_STRAIGHT_ALPHA=0.8493, PP_STRAIGHT_BIAS_EMA_ALPHA=0.434,
        # [2026-08-18 배포 직후 수정] 위 speed10과 동일 버그(14.17 > SPEED_NORMAL 12.5) —
        # SPEED_NORMAL*0.7로 완화.
        SPEED_CORNER_MIN=8.75, CORNER_SIGN_EMA_ALPHA=1.0, LANE_LOOKAHEAD_REF=405.9,
        SPEED_ACCEL_STEP=1.084, CORNER_HOLD_DECAY_LO=0.8675, CORNER_HOLD_DECAY_HI=0.9038,
        CORNER_MIN_RADIUS_PX=460.3, CORNER_MIN_SPEED_SCALE=0.126,
        PATH_EMA_ALPHA=0.5028, DL_STABLE_FRAME_MIN=7, DL_STABLE_JUMP_MAX=10.55,
        SPEED_NORMAL=12.5,
    ),
    'speed15': dict(
        PP_LOOKAHEAD_BASE_PX=81.12, PP_LOOKAHEAD_SPEED_GAIN=1.225, PP_LOOKAHEAD_MAX_PX=278.0,
        PP_WHEELBASE_PX=49.97, PP_ALPHA=0.9349, PP_MIN_LOOKAHEAD_PX=62.77, PP_DX_DEADZONE_PX=1.887,
        PP_LOOKAHEAD_CURVATURE_GAIN=508.0, PP_LOOKAHEAD_MIN_PX=45.14,
        PP_STRAIGHT_CURVATURE_EPS=0.00574, PP_STRAIGHT_CONFIRM_FRAMES=5, PP_STRAIGHT_DEADZONE_PX=11.35,
        PP_STRAIGHT_ALPHA=0.411, PP_STRAIGHT_BIAS_EMA_ALPHA=0.7405,
        # [2026-08-18 배포 직후 수정] 실차 테스트 중 발견 — 그리드서치 원값 15.77이
        # SPEED_NORMAL(15.0)보다 커서 max(SPEED_CORNER_MIN, SPEED_NORMAL*(1-...)) 공식상
        # 코너감속 경로가 완전히 죽어있었다(§0.5.10과 동일 실패모드). 실차에서 관찰된
        # "속도 5 고정" 증상 자체의 원인은 아니었음(그건 LL_DEGRADED/LANE_STALE/
        # AVOID_HOLD_BLOCKED 캡 — 전부 5.0 — 이 원인으로 추정, 인식 불안정 쪽 별도 확인
        # 필요) — 다만 이 버그도 별개로 진짜였고 방치하면 실제 코너에서 무감속 위험이
        # 있어 SPEED_NORMAL*0.7(=10.5)로 완화.
        SPEED_CORNER_MIN=10.5, CORNER_SIGN_EMA_ALPHA=0.7895, LANE_LOOKAHEAD_REF=477.6,
        SPEED_ACCEL_STEP=1.476, CORNER_HOLD_DECAY_LO=0.9196, CORNER_HOLD_DECAY_HI=0.9215,
        CORNER_MIN_RADIUS_PX=678.3, CORNER_MIN_SPEED_SCALE=0.345,
        PATH_EMA_ALPHA=0.6403, DL_STABLE_FRAME_MIN=5, DL_STABLE_JUMP_MAX=35.35,
        SPEED_NORMAL=15.0,
    ),
    'speed17_5': dict(
        PP_LOOKAHEAD_BASE_PX=82.53, PP_LOOKAHEAD_SPEED_GAIN=1.162, PP_LOOKAHEAD_MAX_PX=253.4,
        PP_WHEELBASE_PX=50.0, PP_ALPHA=0.6642, PP_MIN_LOOKAHEAD_PX=56.15, PP_DX_DEADZONE_PX=1.694,
        PP_LOOKAHEAD_CURVATURE_GAIN=478.8, PP_LOOKAHEAD_MIN_PX=30.73,
        PP_STRAIGHT_CURVATURE_EPS=0.002449, PP_STRAIGHT_CONFIRM_FRAMES=5, PP_STRAIGHT_DEADZONE_PX=10.34,
        PP_STRAIGHT_ALPHA=0.75, PP_STRAIGHT_BIAS_EMA_ALPHA=0.4548,
        SPEED_CORNER_MIN=15.61, CORNER_SIGN_EMA_ALPHA=0.4682, LANE_LOOKAHEAD_REF=314.7,
        SPEED_ACCEL_STEP=1.518, CORNER_HOLD_DECAY_LO=0.8894, CORNER_HOLD_DECAY_HI=0.9731,
        CORNER_MIN_RADIUS_PX=551.4, CORNER_MIN_SPEED_SCALE=0.06394,
        PATH_EMA_ALPHA=0.6743, DL_STABLE_FRAME_MIN=4, DL_STABLE_JUMP_MAX=32.57,
        SPEED_NORMAL=17.5,
    ),
    'speed20': dict(
        PP_LOOKAHEAD_BASE_PX=80.62, PP_LOOKAHEAD_SPEED_GAIN=1.353, PP_LOOKAHEAD_MAX_PX=243.3,
        PP_WHEELBASE_PX=49.36, PP_ALPHA=0.7727, PP_MIN_LOOKAHEAD_PX=88.88, PP_DX_DEADZONE_PX=1.898,
        PP_LOOKAHEAD_CURVATURE_GAIN=526.2, PP_LOOKAHEAD_MIN_PX=44.0,
        PP_STRAIGHT_CURVATURE_EPS=0.001976, PP_STRAIGHT_CONFIRM_FRAMES=10, PP_STRAIGHT_DEADZONE_PX=6.18,
        PP_STRAIGHT_ALPHA=0.5738, PP_STRAIGHT_BIAS_EMA_ALPHA=0.4795,
        # [주의] speed_corner_min이 다른 속도보다 훨씬 낮음(5.818) — §8 상단 경고의
        # "14.05로 트랙 이탈" 사례와는 반대 방향(과감속 쪽)이라 그 실패모드 재현
        # 가능성은 낮지만, 코너 대응이 다른 속도 프리셋보다 약할 수 있다.
        SPEED_CORNER_MIN=5.818, CORNER_SIGN_EMA_ALPHA=0.1089, LANE_LOOKAHEAD_REF=458.9,
        SPEED_ACCEL_STEP=1.576, CORNER_HOLD_DECAY_LO=0.9285, CORNER_HOLD_DECAY_HI=0.9612,
        CORNER_MIN_RADIUS_PX=185.1, CORNER_MIN_SPEED_SCALE=0.3435,
        PATH_EMA_ALPHA=0.7997, DL_STABLE_FRAME_MIN=10, DL_STABLE_JUMP_MAX=21.91,
        SPEED_NORMAL=20.0,
    ),
    'speed22_5': dict(
        PP_LOOKAHEAD_BASE_PX=82.36, PP_LOOKAHEAD_SPEED_GAIN=1.23, PP_LOOKAHEAD_MAX_PX=283.5,
        PP_WHEELBASE_PX=49.29, PP_ALPHA=0.7039, PP_MIN_LOOKAHEAD_PX=56.8, PP_DX_DEADZONE_PX=1.904,
        PP_LOOKAHEAD_CURVATURE_GAIN=442.0, PP_LOOKAHEAD_MIN_PX=46.28,
        PP_STRAIGHT_CURVATURE_EPS=0.002509, PP_STRAIGHT_CONFIRM_FRAMES=2, PP_STRAIGHT_DEADZONE_PX=4.695,
        PP_STRAIGHT_ALPHA=0.4509, PP_STRAIGHT_BIAS_EMA_ALPHA=0.6679,
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
        PP_WHEELBASE_PX=49.99, PP_ALPHA=0.7777, PP_MIN_LOOKAHEAD_PX=73.1, PP_DX_DEADZONE_PX=1.989,
        PP_LOOKAHEAD_CURVATURE_GAIN=517.9, PP_LOOKAHEAD_MIN_PX=49.07,
        PP_STRAIGHT_CURVATURE_EPS=0.001982, PP_STRAIGHT_CONFIRM_FRAMES=5, PP_STRAIGHT_DEADZONE_PX=6.251,
        PP_STRAIGHT_ALPHA=0.6336, PP_STRAIGHT_BIAS_EMA_ALPHA=0.3725,
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
PP_TUNE_ACTIVE_PRESET = 'speed15'   # None / 'speed10' / 'speed12_5' / 'speed15' / 'speed17_5' / 'speed20' / 'speed22_5' / 'speed25'
if PP_TUNE_ACTIVE_PRESET is not None:
    globals().update(PP_TUNE_PRESETS[PP_TUNE_ACTIVE_PRESET])
