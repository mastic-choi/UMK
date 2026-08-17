#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# pp_tune_gridsearch.py — Pure Pursuit + 코너감속 파라미터 화이트박스 시뮬레이션
# 튜닝 도구 (2026-08-17)
#
# ── 왜 필요한가 ──
#   실차 회피 영상(2026-08-17 14:00)에서 "직진을 커브로 오판" 문제를 잡던 중,
#   현재 PP_* 튜닝값들이 SPEED_NORMAL=3.0(저속) 재튜닝 시점("대화 중 코너90도/S자
#   합성 시뮬레이션 그리드서치" — config.py PP_ALPHA/PP_DX_DEADZONE_PX 주석 참고)
#   결과라는 걸 확인했다. 그 시뮬레이션 자체는 코드로 저장되지 않고 대화 중에만
#   존재했던 것으로 보인다. SPEED_NORMAL이 이제 10.0으로 올랐고(§config.py
#   2026-08-17f) 나중에 더 올릴 계획이라, 이번엔 그 시뮬레이션 방법론을 코드로
#   남겨 재사용 가능하게 만들고 속도 10/20(모터단위)에서 다시 튜닝한다.
#
# ── [2026-08-17 2차] 코너 감속 로직까지 폐루프에 통합 ──
#   1차 버전은 시나리오 내내 speed_units를 상수로 고정해서 돌렸다 — 실제
#   track_drive.py._lane_drive()가 하는 "코너에서 조향 신호 기반으로 목표속도를
#   깎는" 로직(turn_for_speed 3제곱 감속식 + _corner_radius_speed_scale() 회전반경
#   기반 추가감속 + SPEED_ACCEL_STEP 가속램프 + CORNER_HOLD_DECAY 코너직후 감속유지)
#   이 전혀 반영 안 됐던 것. 조향(PP_*)과 감속(SPEED_*/CORNER_*)은 서로 커플링돼
#   있다 — 조향이 세지면 corner_signal이 커져 더 세게 감속하고, 감속하면 다음 틱
#   lookahead가 줄어 조향이 또 달라진다. 이번 버전은 simulate()가 _lane_drive()의
#   감속 공식을 그대로 재현해 매 틱 목표속도를 다시 계산하고, 그 값으로 물리
#   전진 + 다음 틱 lookahead 기준(speed_for_lookahead)까지 같이 갱신한다. 그리드서치
#   대상도 PP_*(조향) + SPEED_CORNER_MIN/CORNER_SIGN_EMA_ALPHA/LANE_LOOKAHEAD_REF/
#   SPEED_ACCEL_STEP/CORNER_HOLD_DECAY_LO·HI/CORNER_MIN_RADIUS_PX/
#   CORNER_MIN_SPEED_SCALE(감속)까지 22개 전부로 넓혔다.
#
# ── [2026-08-17 3차] 실제 트랙 도면 곡률 반영 + 경로 스무딩(PATH_EMA_ALPHA)까지 폐루프 통합 ──
#   2차 버전까지 남아있던 구멍 두 개를 메운다.
#   1) [지오메트리] 기존 3개 시나리오의 커브 반경(90도커브 1.2m, S자커브 1.3m/55·55도)은
#      전부 가정값이었다. 사용자가 트랙 도면(국민대 자율주행 경진대회, 2026-08-17 제공)을
#      공유해줘서, 그 도면 실측값 — 좌상단/좌하단 코너("curve mode") 반지름 R1,950mm
#      동일, 우측 연결부("zgzg mode")는 서로 다른 반지름 4개짜리 원호 R1,615/1,530/
#      1,595/1,470mm — 로 직진→커브→직진→지그재그→직진→커브→직진이 이어지는 연속 경로
#      '실전트랙'을 추가했다(build_track_path(), TRACK_CORNER_RADIUS_M/
#      TRACK_ZIGZAG_RADII_M). 코너1(90도)+지그재그(net 180도)+코너2(90도)=360도라 한
#      바퀴를 도는 닫힌 루프로 앞뒤가 맞는다(도면이 실제로 폐루프 트랙인 것과 일치).
#      각 구간에 모드 라벨(long_straight/curve/zigzag)과 "전환구간"(모드 경계 앞뒤
#      TRANSITION_ZONE_M) 마스크를 같이 만들어(_build_path_meta()) simulate()가 모드별
#      cte뿐 아니라 "직진에서 코너로, 코너에서 지그재그로 전환되는 순간의 cte"를 따로
#      집계하게 했다 — 기존 3개는 각각 그 모드 하나만 독립적으로 겪었어서 이 "전환
#      손실"(정상상태 커브 추종은 괜찮은데 전환 찰나에만 유독 튀는 조합)이 한 번도 채점
#      대상이 아니었다. score()가 이 전환구간 cte에 특히 더 무거운 가중치를 준다.
#      실전트랙 시나리오는 한 바퀴 전체라 기존 3개보다 훨씬 길어서(약 23m) MAX_SIM_T도
#      12초→30초로 늘렸다(저속에서 완주 전에 타임아웃돼 부당하게 벌점먹는 것 방지).
#      [알려진 한계] 도면에 반지름은 있지만 각 원호가 몇 도씩 도는지는 안 나와있다 —
#      위/아래 직선이 반대방향(오벌이라 net 180도 필요)이라는 기하학적 제약만 확실하고,
#      ZIGZAG_WIGGLE_DEG(지그재그 굴곡 진폭)는 그 제약 안에서 고른 임의 설계값이다.
#   2) [경로 스무딩] simulate()가 매 틱 새로 합성한 픽셀 웨이포인트(fitted_path, 실제
#      디버깅창의 노란선에 해당)를 지금까진 pure_pursuit에 그대로 넘겼는데, 실제 코드
#      (perception/lane_util.py._update_path())는 이걸 바로 안 쓰고 직전 틱의 스무딩된
#      경로(self.path, 디버깅창의 자홍색/보라색 선 — pure_pursuit이 실제로 추종하는 값)와
#      웨이포인트 인덱스별로 PATH_EMA_ALPHA 가중 EMA 블렌딩한 뒤에야 넘긴다. 이 블렌딩은
#      픽셀 좌표를 그대로 섞을 뿐 그 사이 차량이 이동한 만큼은 보정하지 않는다 — config.py의
#      PATH_EMA_ALPHA 튜닝 이력(0.25→0.45→0.75→0.25)이 반복해서 기록하는 "코너 추종
#      지연"이 바로 이 근사에서 나온다. simulate()도 동일하게 smoothed_path 상태를 매 틱
#      갱신해 pp.control()에 넘기도록 바꾸고, path_ema_alpha를 그리드서치 대상에 추가해
#      (PATH_KEYS) 총 23개 파라미터를 함께 탐색한다.
#
# ── [2026-08-17 4차] Optuna(TPE) 탐색 모드 추가 ──
#   run_search()의 랜덤서치는 23차원 공간에서 매번 완전히 새로 뽑은 점을 평가할 뿐,
#   지금까지의 평가 결과가 다음 시도에 전혀 반영되지 않는다(고분산 — 위 3차 문단 참고).
#   Optuna TPE 샘플러로 같은 evaluate()/score()를 objective로 재사용하는
#   run_optuna_search()/--optuna를 추가했다(아래 "Optuna(TPE) 탐색" 섹션 참고).
#   --seed-file로 이전 탐색이 찾은 best_params들을 study.enqueue_trial()로 먼저
#   평가시켜, 새 탐색이 이전 결과를 버리지 않고 그 주변부터 이어서 찾게 한다 — 실제로
#   1차(랜덤서치 20000샘플) → 2차(Optuna, 1차 시드) → 아래 5차(Optuna, 1~2차 시드+
#   디바운스 2개 추가)로 매번 이전 결과를 이어받으며 진행했다. 진행 기록/실제 수치는
#   track_drive/track_drive/새로운_조향테스트.md 참고.
#
# ── [2026-08-17 5차] 코너 프리뷰 신호의 N프레임 디바운스(DL_STABLE_FRAME_MIN/
#    DL_STABLE_JUMP_MAX) 반영 — 23개 → 25개 ──
#   simulate()의 lookahead_ema(코너 프리뷰 신호, _lane_drive()의 turn_preview가 소비)는
#   지금까지 매 틱 새 far_offset_px를 그냥 고정 0.5/0.5 EMA로만 흘려보냈다. 그런데 실제
#   perception/lane_util.py._debounce()는 그 앞단에서 별도 필터를 하나 더 거친다 —
#   근거리 offset이 직전 "후보"와 DL_STABLE_JUMP_MAX(px) 이내로 DL_STABLE_FRAME_MIN
#   프레임 연속으로 비슷해야만 그 프레임의 (offset, lookahead) 묶음이 "확정"으로
#   승격되고, 그 확정된 lookahead만 track_drive.py의 0.5/0.5 EMA에 들어간다 — 이건
#   PATH_EMA_ALPHA(self.path 자체의 프레임간 블렌딩)와는 완전히 다른, 별개의
#   스무딩 단계다. 이번 세션에서 사용자가 "빼먹은 파라미터 있는지 확인해달라"고
#   요청해 코드 대조 중 발견한 진짜 누락 — SPEED_LL_DEGRADED류(센서 죽음 인프라
#   미시뮬레이션)처럼 의도적으로 뺀 게 아니라 그냥 놓쳤던 것이다. simulate()에
#   lane_util.py._debounce()와 동일한 confirmed/pending/pending_count 상태를
#   추가해 재현하고(근거리 offset 안정성으로 게이트, 원거리 lookahead를 확정값으로
#   승격), stable_frame_min/stable_jump_max를 DEBOUNCE_KEYS로 그리드서치에 추가한다.
#   이미 찾아둔 1~3차 best_params들은 이 두 값이 아예 없던 상태에서 나온 결과라
#   config.py 현재값(2, 20.0)을 채워 25차원 시드로 확장해 재탐색을 이어간다(값을
#   버리지 않고 Optuna seed로 재사용 — 4차 탐색 방법론 참고).
#
# ── 방법론(화이트박스 합성 시뮬레이션 — 실측 아님) ──
#   controller/pure_pursuit.py의 PurePursuitController를 코드 수정 없이 그대로
#   가져다 쓴다(알고리즘 자체는 건드리지 않고 파라미터만 바꿔가며 평가). 직진/
#   90도커브/S자커브 3종 기준경로(월드 좌표, 미터)를 만들고, 매 제어틱(0.05s,
#   control_loop()과 동일 주기)마다
#     1) 차량의 현재 위치/헤딩 기준으로 기준경로를 차량 상대좌표로 변환
#     2) DL_PIXELS_PER_METER(200px/m)로 픽셀 변환 + 세그멘테이션 잡음(가우시안,
#        기본 표준편차 1.5px — config.py min_lookahead_px 주석의 실측 예
#        "dx=3px가 육안상 거의 직진"과 같은 자릿수)을 얹어 실제 ROI 픽셀
#        웨이포인트 배열(가까운점→먼점, PATH_N_WAYPOINTS=12개)을 합성
#     3) PurePursuitController.control()을 그대로 호출해 조향각(도)을 얻고
#     4) track_drive.py._lane_drive()/_corner_radius_speed_scale()과 동일한
#        공식으로 이번 틱 목표속도(모터단위)를 다시 계산하고(가속램프 포함)
#     5) 실측 WHEELBASE_M(0.335m, PP_WHEELBASE_PX와는 다른 값 — 아래 주의 참고)
#        기반 바이시클 모델로 그 속도만큼 차량을 실제로 전진시킨다
#   는 폐루프 시뮬레이션을 돌려, 기준경로 대비 횡편차(cross-track error)/조향
#   잔떨림/평균속도를 채점한다.
#
#   ★ 중요한 주의 — PP_WHEELBASE_PX ≠ WHEELBASE_M ★
#   PP_WHEELBASE_PX(현재 25.0)는 "곡률→조향각" 변환에 쓰이는 튜닝된 게인이지
#   차량의 실제 물리 축거가 아니다(config.py 주석: "튜닝값 — 조향을 더 ..."로
#   67.0(실측 기반 계산값)에서 낮춘 것). 즉 PurePursuitController 내부는
#   PP_WHEELBASE_PX를 쓰지만, 시뮬레이터가 차량을 실제로 움직일 때는 반드시
#   WHEELBASE_M(실측 축거)*바이시클 모델을 써야 한다 — 이 둘을 같은 값으로
#   섞으면 "튜닝된 게인이 물리적으로도 맞다"고 잘못 가정하게 되어 그리드서치가
#   자기 자신과 짜고 치는 꼴이 된다. _corner_radius_speed_scale()의 회전반경
#   계산은 실제 코드처럼 PP_WHEELBASE_PX(pp.wheelbase_px)를 그대로 쓴다 — 이건
#   물리 반경이 아니라 "그 함수가 보는" 반경이라 실제 코드와 똑같이 맞춰야 한다.
#
# ── 실차 미검증 명시 ──
#   이 도구가 찾아내는 "최적 파라미터"는 전부 화이트박스 합성 시뮬레이션
#   결과다. 실측이 아닌 값(DL_PIXELS_PER_METER=200px/m 자체가 설계값, 트랙
#   곡률 반경/노이즈 표준편차는 이 스크립트의 가정)에 의존하므로, 여기서 나온
#   조합은 반드시 실차에서 재검증할 것 — 기존 PP_* 값들과 동일한 원칙
#   (config.py 전역에 "실차 미검증" 표기가 붙는 것과 같은 이유).
#
# ── 속도 단위 ──
#   PurePursuitController.control(speed=...)의 speed는 "모터 단위"(drive()가
#   ±100으로 클립하는 값, m/s 아님) — lookahead 스케일링에만 쓰인다. 시뮬레이터가
#   차량을 실제로 전진시킬 때 쓰는 물리 속도(m/s)는 METERS_PER_SPEED_UNIT=0.1347
#   (config.py, speed=5~10 구간 실측 회귀)로 변환한다. speed=20은 이 회귀 구간
#   (5~10) 밖으로 외삽한 값이라 결과 해석 시 감안할 것 — README 6.5절 참고.
#   시나리오의 "speed=N"은 이제 SPEED_NORMAL(순항 목표속도) 역할이다 — 실제
#   순간속도는 코너에서 그보다 낮게 동적으로 깎인다(_lane_drive() 재현).
#
# ── VESC/IMU 미시뮬레이션 ──
#   실차의 _speed_for_lookahead()는 VESC 실측이 살아있으면 그걸 쓰지만, 이
#   시뮬레이터는 VESC/IMU를 흉내내지 않는다 — 실제 코드의 "VESC 죽었을 때" 폴백
#   경로(self._prev_speed 사용, imu_curvature_px=None)를 그대로 탄다. 즉 이
#   시뮬레이션은 "센서가 없거나 신뢰 못 할 때"의 동작을 검증하는 셈이고, VESC/IMU가
#   붙었을 때의 실제 거동은 여기서 다루지 않는다.
#
# ── 사용법 ──
#   numpy가 있는 환경에서:
#     python3 pp_tune_gridsearch.py --samples 3000 --speeds 10 20
#     python3 pp_tune_gridsearch.py --sensitivity --speeds 5 10 15 20 25
#=============================================
import argparse
import json
import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controller.pure_pursuit import PurePursuitController  # noqa: E402
import config as cfg  # noqa: E402


# ─────────────────────────── 물리/스케일 상수 (config.py에서 그대로 가져옴) ───────────────────────────
PX_PER_M = cfg.DL_PIXELS_PER_METER          # 200.0
WHEELBASE_M = cfg.WHEELBASE_M               # 0.335 (실측 — PP_WHEELBASE_PX와 혼동 금지, 위 주석 참고)
MPS_PER_UNIT = cfg.METERS_PER_SPEED_UNIT    # 0.1347 (speed=5~10 실측 회귀)
ANGLE_MAX_DEG = cfg.ANGLE_MAX               # 80.0 — 안전 클램프, 그리드서치 대상 아님(고정)

ROI_W_PX = 585.0                            # README §2.35 실측: BEV 캔버스 실측폭
ROI_DEPTH_M = cfg.DL_BEV_FAR_LIMIT_M        # 0.7 — 원거리 크롭 한계(전방 가시거리 근사)
ROI_H_PX = ROI_DEPTH_M * PX_PER_M           # 140px — roi_h 근사(정확한 BEV 기하 대신 근사, 주석 참고)

N_WAYPOINTS = 12                            # lane_util.PATH_N_WAYPOINTS와 동일
# [알려진 한계] 1.5px는 "적당히 보수적인" 가정일 뿐 실측이 아니다 — 실제로 돌려보니
# 이 표준편차에서는 baseline(eps=0.001)조차 직진 시나리오에서 직진태그=100%가 나온다
# (즉 이 노이즈 수준만으로는 실차 영상에서 봤던 "직진을 커브로 오판" 실패모드가
# 재현되지 않는다 — 실제 세그멘테이션 잡음이 이보다 크거나, 순수 잡음이 아니라 지속적인
# 잔여 편향(bias)이 섞여있을 가능성). eps 자체를 이 시뮬레이션만으로 확정하지 말고,
# 이 한계를 감안해 실차 재검증할 것.
NOISE_STD_PX = 1.5                          # 세그멘테이션 프레임간 잡음 가정(표준편차, px)
DT = 0.05                                   # control_loop() 주기와 동일
# [2026-08-17 3차] 원래 12.0 — '실전트랙'(한 바퀴, 약 23m)이 저속(speed_norm≈10)에서
# 12초로는 완주 전에 타임아웃돼 부당하게 미완주 페널티를 먹는다. 기존 3개 시나리오는
# 원래도 훨씬 일찍 끝나 영향 없음. 30.0으로 늘림.
# [2026-08-17~18 재실측] METERS_PER_SPEED_UNIT이 재실측으로 0.1347→0.0848(약 63%)로
# 낮아지면서 같은 speed_norm이라도 물리 속도가 그만큼 느려져 완주 시간이 늘었다 —
# baseline speed=10에서 실전트랙 소요=28.3s로 30초 한계에 거의 닿는 걸 확인, 코너감속이
# 더 걸리는 조합은 30초를 넘겨 부당하게 미완주 판정될 수 있다. 0.1347 기준으로 캘리브레이션한
# 30.0을 METERS_PER_SPEED_UNIT 비율로 그대로 스케일링 — 나중에 이 상수가 또 바뀌어도 자동으로
# 맞춰지도록 하드코딩 대신 수식으로 둔다.
MAX_SIM_T = 30.0 * (0.1347 / cfg.METERS_PER_SPEED_UNIT)
DIVERGE_CTE_M = 1.0                         # 이 이상 벗어나면 "발산"으로 보고 그 즉시 종료+페널티
PATH_DS = 0.02                              # 기준경로 점 간격(m) — build_paths()/build_track_path()
                                             # 둘 다 이 값을 기본값으로 쓴다(아래 NEAREST_WINDOW_M을
                                             # 포인트 개수로 환산할 때도 동일 값을 가정).
# [2026-08-17 3차] '실전트랙'은 폐루프라 직선 길이를 짧게 압축한 만큼(실제 길이가 아니라
# "전환 관찰에 충분한" 근사값을 쓰기로 함, build_track_path() 참고) 경로상 멀리 떨어진
# 두 구간(예: 출발 직후 첫 직진과 마지막 바퀴 진입 직전 커브)이 월드좌표상으로는 가까워질
# 수 있다. _nearest_point()가 매 틱 경로 전체에서 전역 최근접점을 찾으면, 차량이 살짝만
# 벗어나도 "가장 가까운 점"이 지금 실제로 달리고 있는 구간이 아니라 공간적으로만 가까운
# 엉뚱한(경로상 훨씬 뒤/앞의) 구간으로 튀어버려 cte가 가짜로 폭증하는 버그가 있었다(실제로
# 재현: 첫 직진 구간에서 60cm대 cte로 잘못 집계됨). 같은 이유로 _local_path_px()의
# "센서가 보는 웨이포인트" 합성도 forward/lateral 마스킹만으론 부족해서(실제로 재현: 출발
# 직후 조향이 즉시 ±45도로 폭주 — 두 번째 커브 구간 점이 섞여 들어간 것) 동일 원리로
# 창을 씌운다. 실차 컨트롤러/카메라도 전역 최근접이나 도로 반대편을 뚫어보는 게 아니라
# "지금까지 따라온 지점 근처"만 보는 게 정상이므로, _nearest_point()/_local_path_px() 둘 다
# 직전 틱 인덱스 기준 탐색창을 공유해 쓴다 — NEAREST_WINDOW_M은 정상 주행 중 겪을 수 있는
# cte/ROI_DEPTH_M보다 넉넉히 크고, 트랙의 서로 다른 구간 사이 최소 간격보다는 작게 잡은
# 값(임의 설계값).
NEAREST_WINDOW_M = 1.5

# ─────────────────────────── 실제 트랙 도면 실측값 (2026-08-17, 사용자 제공 도면) ───────────────────────────
# 국민대 자율주행 경진대회 트랙 도면 기준. 좌상단/좌하단 코너("curve mode", 도면 파란색)는
# 반지름 R1,950mm로 동일. 우측 연결부("zgzg mode", 도면 보라색)는 단순 커브가 아니라
# 서로 다른 반지름 4개짜리 원호가 이어지는 S자형 굴곡(R1,615/1,530/1,595/1,470mm, 도면
# 라벨 그대로)이다 — 기존 3개 시나리오의 가정 반경(90도커브 1.2m, S자커브 1.3m)보다 실제
# 곡률 난이도를 반영한다.
# [알려진 한계] 도면에 반지름은 명시돼 있지만 각 원호가 몇 도씩 도는지는 표기돼 있지
# 않다. 위/아래 직선이 서로 반대방향(오벌 트랙이므로 net 180도 방향전환 필요)이라는
# 기하학적 제약만 확실하고, 4개 원호에 그 180도를 정확히 어떻게 나눠 배분하는지는 도면만
# 으로 알 수 없다 — ZIGZAG_WIGGLE_DEG는 "지그재그처럼 좌우로 흔드는 느낌"을 내기 위한
# 임의 설계값이지 도면 실측치가 아니다. 실차 재검증 시 이 부분을 특히 감안할 것.
TRACK_CORNER_RADIUS_M = 1.95
TRACK_ZIGZAG_RADII_M = (1.615, 1.530, 1.595, 1.470)
ZIGZAG_WIGGLE_DEG = 20.0                    # 위 "알려진 한계" 참고 — 임의 가정값
TRANSITION_ZONE_M = 0.4                     # 모드 경계 앞뒤 이 거리(m) 안쪽을 "전환구간"으로
                                             # 별도 채점(모드 전환 시 손실 파악용) — 임의 설계값


# ─────────────────────────── 기준경로 생성 (월드 좌표, 미터, 표준 수학각) ───────────────────────────
def _straight(cursor, length, ds):
    x, y, th = cursor
    n = max(1, int(length / ds))
    pts = []
    for _ in range(n):
        x += ds * math.cos(th)
        y += ds * math.sin(th)
        pts.append((x, y))
    return pts, (x, y, th)


def _arc(cursor, radius, turn_deg, direction, ds):
    """direction: +1=좌회전(ccw), -1=우회전(cw). radius(m), turn_deg(회전각, 항상 양수로 줄 것)."""
    x, y, th = cursor
    total = math.radians(abs(turn_deg))
    dtheta = ds / radius
    n = max(1, int(total / dtheta))
    dtheta = total / n
    pts = []
    for _ in range(n):
        th += direction * dtheta
        x += ds * math.cos(th)
        y += ds * math.sin(th)
        pts.append((x, y))
    return pts, (x, y, th)


def build_paths(ds=PATH_DS):
    """직진/90도커브/S자커브(가정 지오메트리, 개별 실패모드 디버깅용) + 실전트랙(사용자
    제공 트랙 도면 실측 곡률, 연속 경로 — 실제 그리드서치의 주 채점 대상) 4종. 앞의 3개
    커브 반경(1.2~1.3m)은 실측이 아니라 VEHICLE_WIDTH_M(0.31)/LANE_WIDTH_M(0.4) 스케일에서
    합리적으로 가정한 값이고, 실전트랙은 TRACK_CORNER_RADIUS_M/TRACK_ZIGZAG_RADII_M(도면
    실측)을 쓴다 — build_track_path() 참고.

    반환: (paths, path_meta). path_meta는 '실전트랙'에 대해서만 모드/전환구간 정보를
    담는다(다른 3개는 키 자체가 없음)."""
    paths = {}

    pts, _ = _straight((0.0, 0.0, math.pi / 2), 6.0, ds)
    paths['직진'] = pts

    cur = (0.0, 0.0, math.pi / 2)
    pts_a, cur = _straight(cur, 1.5, ds)
    pts_b, cur = _arc(cur, 1.2, 90.0, +1, ds)
    pts_c, cur = _straight(cur, 1.5, ds)
    paths['90도커브'] = pts_a + pts_b + pts_c

    cur = (0.0, 0.0, math.pi / 2)
    pts_a, cur = _straight(cur, 1.0, ds)
    pts_b, cur = _arc(cur, 1.3, 55.0, +1, ds)
    pts_c, cur = _arc(cur, 1.3, 55.0, -1, ds)
    pts_d, cur = _straight(cur, 1.0, ds)
    paths['S자커브'] = pts_a + pts_b + pts_c + pts_d

    paths = {k: np.array(v) for k, v in paths.items()}

    track_pts, track_segments = build_track_path(ds=ds)
    paths['실전트랙'] = track_pts
    mode_arr, transition_mask = _build_path_meta(track_segments, len(track_pts), ds)
    path_meta = {'실전트랙': {'segments': track_segments, 'mode_arr': mode_arr,
                             'transition_mask': transition_mask}}

    return paths, path_meta


def build_track_path(ds=PATH_DS, lead_in_m=2.0, straight_mid_m=3.0):
    """직진→커브(R1.95,90도)→직진→지그재그(4원호, 실측반경)→직진→커브(R1.95,90도)→직진
    순서로 이어지는 연속 경로 — 트랙 도면(위 TRACK_* 상수 주석 참고)의 곡률을 그대로 쓰되,
    직선 길이는 도면 실측이 아니라 "각 모드에서 정상상태에 도달하고 전환을 관찰하기에
    충분한" 근사값이다(실제 길이를 맞추는 게 목적이 아니라 각 모드가 순서대로 이어지는
    과정 자체를 보는 게 목적). 코너1(90도)+지그재그(net 180도)+코너2(90도)=360도라 방향이
    제자리로 돌아오는 닫힌 루프가 된다(도면이 실제로 폐루프 트랙인 것과 일치).

    반환: (pts, segments). segments는 (start_idx, end_idx_exclusive, mode_name) 리스트로,
    simulate()가 모드별/전환구간별 성능을 따로 채점할 수 있게 해준다."""
    pts, segments = [], []
    cur = (0.0, 0.0, math.pi / 2)

    def _add(new_pts, mode):
        start = len(pts)
        pts.extend(new_pts)
        segments.append((start, len(pts), mode))

    p, cur = _straight(cur, lead_in_m, ds)
    _add(p, 'long_straight')
    p, cur = _arc(cur, TRACK_CORNER_RADIUS_M, 90.0, +1, ds)
    _add(p, 'curve')
    p, cur = _straight(cur, straight_mid_m, ds)
    _add(p, 'long_straight')

    # 지그재그: net 180도 회전(오벌 트랙 진행방향 반전) 제약 안에서 굴곡을 준다.
    # turn=[W+90, W, W+90, W], sign=[+,-,+,-] 패턴이면 W값과 무관하게
    # net=(W+90)-W+(W+90)-W=180도가 항상 성립한다 — W(ZIGZAG_WIGGLE_DEG)는 그 안에서
    # "얼마나 좌우로 흔드는가"만 조절하는 임의 가정값(위 TRACK_* 주석 참고).
    w = ZIGZAG_WIGGLE_DEG
    zig_turns = (w + 90.0, w, w + 90.0, w)
    zig_signs = (+1, -1, +1, -1)
    for r, turn, sign in zip(TRACK_ZIGZAG_RADII_M, zig_turns, zig_signs):
        p, cur = _arc(cur, r, turn, sign, ds)
        _add(p, 'zigzag')

    p, cur = _straight(cur, straight_mid_m, ds)
    _add(p, 'long_straight')
    p, cur = _arc(cur, TRACK_CORNER_RADIUS_M, 90.0, +1, ds)
    _add(p, 'curve')
    p, cur = _straight(cur, lead_in_m, ds)
    _add(p, 'long_straight')

    return np.array(pts), segments


def _build_path_meta(segments, n_pts, ds):
    """segments → (모드 라벨 배열, 전환구간 마스크). simulate()가 매 틱 가장 가까운
    경로 인덱스(_nearest_point())로 이 두 배열을 조회해 모드별/전환구간별 cte를 따로
    집계한다."""
    mode_arr = np.empty(n_pts, dtype=object)
    for s, e, name in segments:
        mode_arr[s:e] = name

    zone_pts = max(1, int(TRANSITION_ZONE_M / ds))
    transition_mask = np.zeros(n_pts, dtype=bool)
    for i in range(1, len(segments)):
        b = segments[i][0]  # 이전 구간 끝 == 새 구간 시작
        lo, hi = max(0, b - zone_pts), min(n_pts, b + zone_pts)
        transition_mask[lo:hi] = True

    return mode_arr, transition_mask


# ─────────────────────────── 폐루프 시뮬레이션 ───────────────────────────
def _local_path_px(world_pts, vx, vy, vth, rng, prev_idx=None, window_pts=None):
    """세계좌표 기준경로를 차량 상대좌표로 변환 → ROI 픽셀 웨이포인트(가까운점→먼점,
    N_WAYPOINTS개, 잡음 포함)로 합성. 실제 시스템의 _fit_and_sample_path()
    (균등 간격 재샘플링)와 동일한 발상.

    [2026-08-17 3차, 중대 버그 수정] forward/lateral 마스킹만으로는 "경로상 지금 차량이
    있는 위치 근처"라는 보장이 없다 — 열린 단일경로(기존 3개 시나리오)에선 문제가
    없었지만, '실전트랙'처럼 폐루프라 서로 다른 구간이 월드좌표상 가까워질 수 있는
    경로에서는 지금 실제로 달리는 구간이 아니라 공간적으로만 가까운 엉뚱한(예: 복귀
    커브) 구간의 점이 forward∈[0,ROI_DEPTH_M] 조건을 우연히 만족해 "센서가 본
    웨이포인트"로 섞여 들어갈 수 있다(실제로 재현: 출발 직후 첫 직진 구간에서 조향이
    즉시 ±45도로 폭주 — 원인 추적 결과 두 번째 커브 구간의 점이 합성 경로에 끼어든
    것으로 확인). 실제 카메라도 같은 도로 표면 반대편 구간을 뚫어볼 수 없으므로,
    prev_idx/window_pts가 주어지면 _nearest_point()와 동일하게 차량의 현재 경로상
    진행 위치 근처로 후보점을 먼저 제한한다."""
    if prev_idx is not None and window_pts is not None:
        lo = max(0, prev_idx - window_pts)
        hi = min(len(world_pts), prev_idx + window_pts)
        world_pts = world_pts[lo:hi]

    dx = world_pts[:, 0] - vx
    dy = world_pts[:, 1] - vy
    forward = dx * math.cos(vth) + dy * math.sin(vth)
    lateral = dx * math.sin(vth) - dy * math.cos(vth)

    mask = (forward >= 0.0) & (forward <= ROI_DEPTH_M)
    if not np.any(mask):
        return None

    fwd_v = forward[mask]
    lat_v = lateral[mask]
    order = np.argsort(fwd_v)
    fwd_v, lat_v = fwd_v[order], lat_v[order]

    sample_fwd = np.linspace(0.0, min(fwd_v[-1], ROI_DEPTH_M), N_WAYPOINTS)
    sample_lat = np.interp(sample_fwd, fwd_v, lat_v)

    x_px = ROI_W_PX / 2.0 + sample_lat * PX_PER_M + rng.normal(0.0, NOISE_STD_PX, N_WAYPOINTS)
    y_px = ROI_H_PX - sample_fwd * PX_PER_M
    return list(zip(x_px.tolist(), y_px.tolist()))


def _nearest_point(world_pts, vx, vy, prev_idx=None, window_pts=None):
    """가장 가까운 기준경로 점의 (인덱스, 거리). 인덱스는 '실전트랙'에서 지금 차량이
    어느 모드(mode_arr)/전환구간(transition_mask) 위에 있는지 조회하는 데도 쓰인다.

    prev_idx/window_pts가 주어지면 전체 경로가 아니라 그 주변 구간에서만 찾는다 — 폐루프
    트랙은 경로상 멀리 떨어진 두 구간이 월드좌표상으론 가까울 수 있어(위 NEAREST_WINDOW_M
    주석 참고), 전역 탐색을 쓰면 차량이 살짝만 벗어나도 엉뚱한 구간으로 튀는 앨리어싱이
    생긴다 — 실제 컨트롤러도 "지금까지 따라온 지점 근처"만 보지 전역 최근접을 쓰지 않는다."""
    if prev_idx is None or window_pts is None:
        lo, hi = 0, len(world_pts)
    else:
        lo = max(0, prev_idx - window_pts)
        hi = min(len(world_pts), prev_idx + window_pts + 1)
    seg = world_pts[lo:hi]
    d2 = (seg[:, 0] - vx) ** 2 + (seg[:, 1] - vy) ** 2
    local_idx = int(np.argmin(d2))
    idx = lo + local_idx
    return idx, math.sqrt(float(d2[local_idx]))


def _corner_radius_speed_scale(corner_signal_deg, wheelbase_px, corner_min_radius_px, corner_min_speed_scale):
    """track_drive.py._corner_radius_speed_scale()과 동일 공식. wheelbase_px는
    PP_WHEELBASE_PX(pp.wheelbase_px) — 실제 코드가 물리 축거가 아니라 이 게인으로
    반경을 역산하므로 그대로 맞춘다(위 모듈 주석 참고)."""
    curvature = math.tan(math.radians(corner_signal_deg)) / wheelbase_px
    if curvature == 0.0:
        return 1.0
    radius = abs(1.0 / curvature)
    if radius >= corner_min_radius_px:
        return 1.0
    return max(corner_min_speed_scale, radius / corner_min_radius_px)


def simulate(pp, world_pts, speed_norm, sp, rng, meta=None):
    """speed_norm: 이 시나리오의 SPEED_NORMAL(모터단위, 순항 목표속도).
    sp: 감속+경로스무딩+디바운스 관련 파라미터 dict(SPEED_KEYS/PATH_KEYS/DEBOUNCE_KEYS 참고)
    — track_drive.py의 _lane_drive() 코너 감속, perception/lane_util.py._update_path()의
    PATH_EMA_ALPHA 경로 스무딩, _debounce()의 N프레임 확정 게이트를 그대로 재현한다.
    meta: build_paths()가 '실전트랙'에 대해서만 채워주는 {'mode_arr','transition_mask'} —
    있으면 모드별/전환구간별 cte도 추가로 집계한다(없으면 기존 3개 시나리오처럼 집계 없이
    진행)."""
    vx, vy, vth = float(world_pts[0, 0]), float(world_pts[0, 1]), math.pi / 2
    pp.reset()

    prev_speed = 0.0        # track_drive.py __init__의 self._prev_speed=0.0과 동일 출발
    corner_signal = 0.0     # self._corner_signal
    corner_hold = 0.0       # self._corner_hold
    lookahead_ema = 0.0     # self.lane_lookahead(EMA, dl_lane far_ref 근사)
    smoothed_path = None    # lane_util.py self.path 초기상태([])와 동일 — 첫 유효 프레임 그대로 채택
    path_ema_alpha = sp['path_ema_alpha']
    # lane_util.py._debounce()의 self._confirmed/_pending/_pending_count와 동일 역할 —
    # (근거리 offset, 원거리 lookahead) 튜플을 근거리 offset 안정성으로 게이트한다
    # (아래 [디바운스] 블록 참고). stable_frame_min/stable_jump_max는 DEBOUNCE_KEYS.
    debounce_confirmed = None
    debounce_pending = None
    debounce_pending_count = 0
    stable_frame_min = sp['stable_frame_min']
    stable_jump_max = sp['stable_jump_max']

    ctes, steers, straight_flags, speeds = [], [], [], []
    mode_hits, transition_ctes = [], []
    path_end = world_pts[-1]
    n_ticks = int(MAX_SIM_T / DT)
    completed = False
    prev_idx = 0                            # 위 NEAREST_WINDOW_M 주석 참고 — 폐루프 앨리어싱 방지
    window_pts = max(1, int(NEAREST_WINDOW_M / PATH_DS))

    for _ in range(n_ticks):
        fitted_path = _local_path_px(world_pts, vx, vy, vth, rng, prev_idx=prev_idx, window_pts=window_pts)
        vehicle_xy = (ROI_W_PX / 2.0, ROI_H_PX)
        # _speed_for_lookahead(): VESC 미시뮬레이션이라 항상 self._prev_speed 폴백 경로.
        if fitted_path is None:
            # lane_util.py._update_path(None)과 동일 — smoothed_path 갱신 없이 마지막 값 유지.
            steer_deg = pp.prev_steer_deg
        else:
            # lane_util.py._update_path(): 웨이포인트 개수가 같으면 인덱스별 EMA 블렌딩,
            # 아니면(첫 프레임 등) 새 경로 그대로 채택 — 픽셀 좌표를 그대로 블렌딩하므로
            # 그 사이 차량이 움직인 만큼은 보정되지 않는다(실제 코드와 동일한 근사 — 이게
            # config.py PATH_EMA_ALPHA 주석이 반복해서 기록하는 "코너 추종 지연"의 근본
            # 원인이다). fitted_path(디버깅창 노란선)는 그대로 두고, pure_pursuit이 실제로
            # 추종하는 건 이 스무딩된 smoothed_path(디버깅창 자홍색/보라색 선)다.
            if smoothed_path is None or len(smoothed_path) != len(fitted_path):
                smoothed_path = fitted_path
            else:
                a = path_ema_alpha
                smoothed_path = [
                    (a * nx + (1.0 - a) * px, a * ny + (1.0 - a) * py)
                    for (px, py), (nx, ny) in zip(smoothed_path, fitted_path)
                ]
            steer_deg = pp.control(smoothed_path, vehicle_xy, speed=prev_speed, imu_curvature_px=None)
            # lookahead_ema(코너 프리뷰 신호)는 실제 코드처럼 스무딩 이전의 원시 far_center
            # 기반이다 — 실제 self.lane_lookahead는 self.path(스무딩됨)가 아니라 슬라이스
            # 그룹평균(피팅 이전 원시값)에서 나온다. 스무딩된 값을 쓰면 두 신호가 같은
            # 지연을 공유해버려 실제와 달라진다.
            # [디바운스, 2026-08-17 5차] 그런데 실제 lane_util.py는 이 원시값을 바로 안 쓰고
            # _debounce()를 한 번 더 거친다 — 근거리 offset이 직전 "후보"와 stable_jump_max
            # (px) 이내로 stable_frame_min 프레임 연속으로 비슷해야만 그 프레임의
            # (offset, lookahead) 묶음이 확정으로 승격된다(lane_util.py:747-778과 동일 로직).
            # 근거리 점(fitted_path[0])을 offset 역할로, 원거리 점(fitted_path[-1])을
            # lookahead 역할로 대응시켰다 — 시뮬레이터엔 실제 슬라이드윈도우의 별도
            # near_center 계산이 없어서 합성 경로의 양 끝점으로 근사한 것.
            near_offset_px = fitted_path[0][0] - ROI_W_PX / 2.0
            far_offset_px_raw = fitted_path[-1][0] - ROI_W_PX / 2.0
            candidate = (near_offset_px, far_offset_px_raw)
            if debounce_confirmed is None:
                debounce_confirmed = candidate
                debounce_pending = candidate
                debounce_pending_count = stable_frame_min
            else:
                same_flow = abs(near_offset_px - debounce_pending[0]) <= stable_jump_max
                if same_flow:
                    debounce_pending_count += 1
                else:
                    debounce_pending = candidate
                    debounce_pending_count = 1
                if debounce_pending_count >= stable_frame_min:
                    debounce_confirmed = debounce_pending
            far_offset_px = debounce_confirmed[1]
            lookahead_ema = 0.5 * lookahead_ema + 0.5 * far_offset_px

        # ── track_drive.py._lane_drive() 코너 감속 재현 ──
        corner_signal = (sp['corner_sign_ema_alpha'] * steer_deg
                          + (1.0 - sp['corner_sign_ema_alpha']) * corner_signal)
        turn_now = min(1.0, abs(corner_signal) / ANGLE_MAX_DEG)
        turn_preview = min(1.0, abs(lookahead_ema) / sp['lane_lookahead_ref'])
        is_straight = bool(pp.is_straight)
        turn_for_speed = 0.0 if is_straight else max(turn_now, turn_preview * 0.3)
        target_speed = max(sp['speed_corner_min'], speed_norm * (1.0 - 0.90 * turn_for_speed ** 3))
        corner_radius_scale = 1.0 if is_straight else _corner_radius_speed_scale(
            corner_signal, pp.wheelbase_px, sp['corner_min_radius_px'], sp['corner_min_speed_scale'])
        target_speed = max(sp['speed_corner_min'], target_speed * corner_radius_scale)

        speed_ratio = min(1.0, prev_speed / speed_norm) if speed_norm > 1e-6 else 0.0
        corner_decay = sp['corner_hold_decay_lo'] + (sp['corner_hold_decay_hi'] - sp['corner_hold_decay_lo']) * speed_ratio
        corner_hold = max(turn_now, corner_hold * corner_decay)
        accel_step = sp['speed_accel_step'] * max(0.25, 1.0 - corner_hold)
        if target_speed > prev_speed + accel_step:
            target_speed = prev_speed + accel_step
        prev_speed = target_speed

        # steer_deg는 "우측+"(control()의 alpha=atan2(dx,dy), dx>0=목표가 이미지 오른쪽일 때
        # 양수) 관례다. 월드 프레임 th는 표준 수학각(ccw+)이라, 오른쪽으로 꺾을수록(양의
        # steer) 헤딩은 시계방향=th가 "감소"해야 한다 — 부호를 안 뒤집으면 좌/우가 반대로
        # 시뮬레이션되어(첫 구현에서 실제로 발생: 커브가 좌회전인데 차가 우회전해 발산)
        # 그리드서치 전체가 무의미해진다.
        v_mps = MPS_PER_UNIT * prev_speed
        steer_rad = math.radians(steer_deg)
        vth -= (v_mps / WHEELBASE_M) * math.tan(steer_rad) * DT
        vx += v_mps * math.cos(vth) * DT
        vy += v_mps * math.sin(vth) * DT

        idx, cte = _nearest_point(world_pts, vx, vy, prev_idx=prev_idx, window_pts=window_pts)
        prev_idx = idx
        ctes.append(cte)
        steers.append(steer_deg)
        straight_flags.append(is_straight)
        speeds.append(prev_speed)
        if meta is not None:
            mode_hits.append(meta['mode_arr'][idx])
            if meta['transition_mask'][idx]:
                transition_ctes.append(cte)

        if cte > DIVERGE_CTE_M:
            break
        if math.hypot(vx - path_end[0], vy - path_end[1]) < 0.15:
            completed = True
            break
    else:
        completed = math.hypot(vx - path_end[0], vy - path_end[1]) < 0.3

    ctes = np.array(ctes) if ctes else np.array([DIVERGE_CTE_M])
    steers = np.array(steers) if steers else np.array([0.0])
    speeds = np.array(speeds) if speeds else np.array([0.0])
    sign_changes = int(np.sum(np.diff(np.sign(steers)) != 0)) if len(steers) > 1 else 0
    osc_per_sec = sign_changes / max(len(steers) * DT, 1e-6)

    # 워밍업(≈1초, PP_STRAIGHT_CONFIRM_FRAMES 디바운스가 처음 채워지는 구간) 이후만 봐야
    # "계속 직진으로 안 잡히는" 문제를 제대로 측정한다 — 시작 직후 몇 틱은 아직 확정될
    # 시간이 없었을 뿐이라 여기서 빼지 않으면 모든 조합이 부당하게 벌점을 먹는다.
    warmup = int(1.0 / DT)
    tail = straight_flags[warmup:] if len(straight_flags) > warmup else straight_flags
    straight_frac = float(np.mean(tail)) if tail else 0.0

    result = {
        'cte_rms': float(np.sqrt(np.mean(ctes ** 2))),
        'cte_max': float(np.max(ctes)),
        'steer_rms': float(np.sqrt(np.mean(steers ** 2))),
        'osc_per_sec': osc_per_sec,
        'straight_frac': straight_frac,
        'avg_speed': float(np.mean(speeds)),
        'elapsed_s': len(ctes) * DT,
        'completed': completed,
    }

    if meta is not None:
        # ctes/mode_hits/transition_ctes는 매 틱 동시에 append되므로(위 루프) 길이가
        # 항상 일치한다 — 중간에 발산으로 break해도 둘 다 같은 시점에 잘린다.
        modes_arr = np.array(mode_hits) if mode_hits else np.array([], dtype=object)
        by_mode = {}
        for m in ('long_straight', 'curve', 'zigzag'):
            sel = modes_arr == m
            if np.any(sel):
                sel_ctes = ctes[sel]
                by_mode[m] = {'cte_rms': float(np.sqrt(np.mean(sel_ctes ** 2))),
                              'cte_max': float(np.max(sel_ctes))}
        result['by_mode'] = by_mode
        if transition_ctes:
            t_arr = np.array(transition_ctes)
            result['transition_cte_rms'] = float(np.sqrt(np.mean(t_arr ** 2)))
            result['transition_cte_max'] = float(np.max(t_arr))
        else:
            result['transition_cte_rms'] = 0.0
            result['transition_cte_max'] = 0.0

    return result


# ─────────────────────────── 채점 ───────────────────────────
# 가중치는 이 스크립트만의 설계값(실측/이론적 근거 없음) — 직진은 "잔떨림 억제"를,
# 커브/S자는 "경로를 얼마나 잘 따라가는가(횡편차)"를 우선하되, elapsed_s에 작은
# 페널티를 줘서 "코너에서 무조건 기어가면 이긴다"는 퇴화해를 막는다(이 튜닝을
# 시작한 이유 자체가 "나중에 속도를 더 올리고 싶다"였으므로, 과도한 감속도
# 나쁜 해로 취급).
def score(results):
    s = results['직진']
    c = results['90도커브']
    w = results['S자커브']

    penalty = 0.0
    for r in (s, c, w):
        if not r['completed']:
            penalty += 1000.0

    # (1.0 - straight_frac)*10: 애초에 이번 튜닝을 시작한 이유("직진을 커브로 오판")를
    # 직접 벌점화한다 — steer_rms/osc/cte만으론 "라벨이 맞았는지"가 안 잡혀서 따로 추가.
    straight_score = (s['steer_rms'] * 3.0 + s['osc_per_sec'] * 2.0 + s['cte_rms'] * 50.0
                      + (1.0 - s['straight_frac']) * 10.0)
    curve_score = c['cte_rms'] * 80.0 + c['cte_max'] * 40.0 + c['elapsed_s'] * 0.5
    scurve_score = w['cte_rms'] * 80.0 + w['cte_max'] * 40.0 + w['elapsed_s'] * 0.5
    total = straight_score + curve_score + scurve_score + penalty

    # '실전트랙'(실측 곡률, 직진→커브→직진→지그재그→직진→커브→직진 연속 경로) — 있으면
    # 종합 추종성능에 더해 "모드 전환구간"(코너/지그재그로 진입·이탈하는 순간) cte에 특히
    # 무거운 가중치를 준다. 이게 이 시나리오를 추가한 이유(전환되는 과정에서의 손실 파악)
    # 그 자체를 직접 채점하는 항이다 — 정상상태 추종은 괜찮은데 전환 찰나에만 유독 튀는
    # 조합을 curve_score/scurve_score만으론 못 잡아낸다.
    t = results.get('실전트랙')
    if t is not None:
        if not t['completed']:
            total += 1000.0
        total += t['cte_rms'] * 80.0 + t['cte_max'] * 40.0
        total += t.get('transition_cte_rms', 0.0) * 150.0 + t.get('transition_cte_max', 0.0) * 60.0

    return total


# ─────────────────────────── 파라미터 공간 (baseline = 현재 config.py 값) ───────────────────────────
# PurePursuitController 생성자로 그대로 들어가는 키(조향) — evaluate()가 이 이름으로 필터링.
PP_CTOR_KEYS = (
    'lookahead_base_px', 'lookahead_speed_gain', 'lookahead_max_px', 'wheelbase_px',
    'alpha', 'min_lookahead_px', 'dx_deadzone_px', 'lookahead_curvature_gain',
    'lookahead_min_px', 'straight_curvature_eps', 'straight_confirm_frames',
    'straight_deadzone_px', 'straight_alpha', 'straight_bias_ema_alpha',
)
# track_drive.py._lane_drive()의 코너 감속 로직에 쓰이는 키(감속) — simulate()가 직접 읽음.
SPEED_KEYS = (
    'speed_corner_min', 'corner_sign_ema_alpha', 'lane_lookahead_ref',
    'speed_accel_step', 'corner_hold_decay_lo', 'corner_hold_decay_hi',
    'corner_min_radius_px', 'corner_min_speed_scale',
)
# perception/lane_util.py._update_path()의 프레임간 경로 스무딩(PATH_EMA_ALPHA) — evaluate()가
# SPEED_KEYS와 합쳐서 simulate()에 sp['path_ema_alpha']로 넘긴다.
PATH_KEYS = (
    'path_ema_alpha',
)
# perception/lane_util.py._debounce()의 N프레임 확정 게이트(DL_STABLE_FRAME_MIN/
# DL_STABLE_JUMP_MAX) — PATH_EMA_ALPHA와는 별개의 스무딩 단계(위 [2026-08-17 5차] 주석
# 참고). evaluate()가 SPEED_KEYS/PATH_KEYS와 합쳐서 넘긴다.
DEBOUNCE_KEYS = (
    'stable_frame_min', 'stable_jump_max',
)

BASELINE = dict(
    lookahead_base_px=cfg.PP_LOOKAHEAD_BASE_PX,
    lookahead_speed_gain=cfg.PP_LOOKAHEAD_SPEED_GAIN,
    lookahead_max_px=cfg.PP_LOOKAHEAD_MAX_PX,
    wheelbase_px=cfg.PP_WHEELBASE_PX,
    alpha=cfg.PP_ALPHA,
    min_lookahead_px=cfg.PP_MIN_LOOKAHEAD_PX,
    dx_deadzone_px=cfg.PP_DX_DEADZONE_PX,
    lookahead_curvature_gain=cfg.PP_LOOKAHEAD_CURVATURE_GAIN,
    lookahead_min_px=cfg.PP_LOOKAHEAD_MIN_PX,
    straight_curvature_eps=cfg.PP_STRAIGHT_CURVATURE_EPS,
    straight_confirm_frames=cfg.PP_STRAIGHT_CONFIRM_FRAMES,
    straight_deadzone_px=cfg.PP_STRAIGHT_DEADZONE_PX,
    straight_alpha=cfg.PP_STRAIGHT_ALPHA,
    straight_bias_ema_alpha=cfg.PP_STRAIGHT_BIAS_EMA_ALPHA,
    speed_corner_min=cfg.SPEED_CORNER_MIN,
    corner_sign_ema_alpha=cfg.CORNER_SIGN_EMA_ALPHA,
    lane_lookahead_ref=cfg.LANE_LOOKAHEAD_REF,
    speed_accel_step=cfg.SPEED_ACCEL_STEP,
    corner_hold_decay_lo=cfg.CORNER_HOLD_DECAY_LO,
    corner_hold_decay_hi=cfg.CORNER_HOLD_DECAY_HI,
    corner_min_radius_px=cfg.CORNER_MIN_RADIUS_PX,
    corner_min_speed_scale=cfg.CORNER_MIN_SPEED_SCALE,
    path_ema_alpha=cfg.PATH_EMA_ALPHA,
    stable_frame_min=cfg.DL_STABLE_FRAME_MIN,
    stable_jump_max=cfg.DL_STABLE_JUMP_MAX,
)

# (lo_factor, hi_factor) — baseline 대비 탐색 범위 배율. 정수/이산값(straight_confirm_frames)과
# 절대범위가 필요한 decay 두 값은 별도 처리(ABSOLUTE_RANGES).
RANGE_FACTORS = dict(
    lookahead_base_px=(0.7, 1.4),
    lookahead_speed_gain=(0.4, 1.8),
    lookahead_max_px=(0.8, 1.3),
    wheelbase_px=(0.5, 2.0),
    alpha=(0.4, 1.0 / max(cfg.PP_ALPHA, 1e-6) * 0.95 if cfg.PP_ALPHA < 0.95 else 1.0),
    min_lookahead_px=(0.7, 1.3),
    dx_deadzone_px=(0.3, 3.0),
    lookahead_curvature_gain=(0.3, 2.5),
    lookahead_min_px=(0.5, 1.8),
    straight_curvature_eps=(0.3, 3.0),
    straight_deadzone_px=(0.3, 2.0),
    straight_alpha=(0.4, 1.0 / max(cfg.PP_STRAIGHT_ALPHA, 1e-6) * 0.95 if cfg.PP_STRAIGHT_ALPHA < 0.95 else 1.0),
    straight_bias_ema_alpha=(0.4, 2.5),
    speed_corner_min=(0.4, 1.6),
    corner_sign_ema_alpha=(0.3, 3.0),
    lane_lookahead_ref=(0.5, 2.0),
    speed_accel_step=(0.3, 4.0),
    corner_min_radius_px=(0.4, 2.0),
    corner_min_speed_scale=(0.3, 2.5),
    # baseline(0.25) 기준 [0.075, 0.8] — 실차 튜닝 이력(config.py: 0.25→0.45→0.75→0.25)이
    # 실제로 오갔던 범위(0.2~0.75)를 커버한다.
    path_ema_alpha=(0.3, 3.2),
    # baseline(20.0px) 기준 [6, 60]px — DL_SLICE_OUTLIER_MAX(40px)류 다른 픽셀 임계값들과
    # 비슷한 자릿수 안에서 넓게 잡은 범위(임의 설계값, 실측 아님).
    stable_jump_max=(0.3, 3.0),
)
# 배율이 아니라 절대 범위가 필요한 파라미터 — (0,1) 안에 있어야 하고 baseline*factor로
# 다루면 방향에 따라 1을 넘거나 뒤집힐 수 있어서 따로 뺐다.
ABSOLUTE_RANGES = dict(
    corner_hold_decay_lo=(0.80, 0.95),
    corner_hold_decay_hi=(0.90, 0.99),
)
CONFIRM_FRAMES_CHOICES = [2, 3, 5, 7, 10]
STABLE_FRAME_MIN_CHOICES = [1, 2, 3, 4, 5, 7, 10]  # DL_STABLE_FRAME_MIN(기본 2)과 동일 자릿수


def _candidate_range(name):
    if name in ABSOLUTE_RANGES:
        return ABSOLUTE_RANGES[name]
    lo_f, hi_f = RANGE_FACTORS[name]
    base = BASELINE[name]
    lo, hi = base * lo_f, base * hi_f
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def sample_params(rng):
    p = {}
    for k, base in BASELINE.items():
        if k == 'straight_confirm_frames':
            p[k] = int(rng.choice(CONFIRM_FRAMES_CHOICES))
            continue
        if k == 'stable_frame_min':
            p[k] = int(rng.choice(STABLE_FRAME_MIN_CHOICES))
            continue
        lo, hi = _candidate_range(k)
        val = rng.uniform(lo, hi)
        if k in ('alpha', 'straight_alpha', 'path_ema_alpha'):
            val = float(np.clip(val, 0.05, 1.0))
        p[k] = float(val)
    # corner_hold_decay_hi가 lo보다 낮게 뽑히면(별개 절대범위라 드물게 역전 가능)
    # 원래 설계 의도("고속일수록 느리게 회복")가 깨지므로 스왑해 방지.
    if p['corner_hold_decay_hi'] < p['corner_hold_decay_lo']:
        p['corner_hold_decay_lo'], p['corner_hold_decay_hi'] = p['corner_hold_decay_hi'], p['corner_hold_decay_lo']
    return p


def evaluate(params, speed_norm, paths, path_meta, rng):
    pp_kwargs = {k: params[k] for k in PP_CTOR_KEYS}
    sp_kwargs = {k: params[k] for k in SPEED_KEYS}
    sp_kwargs.update({k: params[k] for k in PATH_KEYS})
    sp_kwargs.update({k: params[k] for k in DEBOUNCE_KEYS})
    pp = PurePursuitController(angle_max_deg=ANGLE_MAX_DEG, **pp_kwargs)
    results = {name: simulate(pp, pts, speed_norm, sp_kwargs, rng, meta=path_meta.get(name))
               for name, pts in paths.items()}
    return score(results), results


def run_search(speed_norm, n_samples, seed, paths, path_meta):
    rng = np.random.default_rng(seed)
    baseline_score, baseline_results = evaluate(BASELINE, speed_norm, paths, path_meta, rng)

    best_score, best_params, best_results = baseline_score, dict(BASELINE), baseline_results
    for i in range(n_samples):
        params = sample_params(rng)
        s, results = evaluate(params, speed_norm, paths, path_meta, rng)
        if s < best_score:
            best_score, best_params, best_results = s, params, results

    return {
        'speed_norm': speed_norm,
        'baseline_score': baseline_score,
        'baseline_results': baseline_results,
        'best_score': best_score,
        'best_params': best_params,
        'best_results': best_results,
    }


def sweep_param_vs_speed(param_name, speeds, grid_n, repeats, paths, path_meta):
    """다른 파라미터는 전부 BASELINE에 고정하고 param_name 하나만 촘촘히 스윕해,
    speed별로 그 파라미터의 1차원 최적값을 찾는다(partial-dependence 근사 —
    파라미터 간 상호작용은 무시).

    [왜 조인트 랜덤서치 argmin 대신 이 방식인가] run_search()의 argmin은 25차원
    비볼록 공간에서 뽑은 단일 샘플이라, 같은 속도라도 시드만 바꾸면 전혀 다른
    조합이 "1등"으로 나올 만큼 분산이 크다(거의 동점인 국소 최적점이 많음).
    그런 고분산 추정치를 이어붙여 speed에 대한 추세선을 피팅하면 우연을 추세로
    착각하기 쉽다(실제로 speed=10/20 두 점만으로 봤을 때 PP_STRAIGHT_CURVATURE_EPS가
    방향이 뒤집혔던 게 그 예). 한 번에 파라미터 하나만 바꾸면 비교가 훨씬
    저분산이라 speed에 따른 진짜 추세를 보기에 더 적합하다."""
    if param_name == 'straight_confirm_frames':
        candidates = CONFIRM_FRAMES_CHOICES
    else:
        lo, hi = _candidate_range(param_name)
        candidates = np.linspace(lo, hi, grid_n)

    best_per_speed = []
    for sp in speeds:
        cand_scores = []
        for v in candidates:
            params = dict(BASELINE)
            params[param_name] = int(v) if param_name == 'straight_confirm_frames' else float(v)
            total = 0.0
            for rep in range(repeats):
                rng = np.random.default_rng(hash((param_name, sp, v, rep)) & 0xFFFFFFFF)
                s, _ = evaluate(params, sp, paths, path_meta, rng)
                total += s
            cand_scores.append(total / repeats)
        best_idx = int(np.argmin(cand_scores))
        best_per_speed.append(float(candidates[best_idx]))
    return best_per_speed


def fit_linear(speeds, values):
    speeds_a, values_a = np.array(speeds, dtype=float), np.array(values, dtype=float)
    slope, intercept = np.polyfit(speeds_a, values_a, 1)
    pred = slope * speeds_a + intercept
    ss_res = float(np.sum((values_a - pred) ** 2))
    ss_tot = float(np.sum((values_a - np.mean(values_a)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return slope, intercept, r2


def run_sensitivity(speeds, grid_n, repeats, paths, path_meta):
    print(f"[민감도 분석] 파라미터 1개씩만 스윕(다른 값은 BASELINE 고정), speed={speeds}, "
          f"grid_n={grid_n}, repeats={repeats}\n")
    fits = {}
    for name in BASELINE:
        best_vals = sweep_param_vs_speed(name, speeds, grid_n, repeats, paths, path_meta)
        slope, intercept, r2 = fit_linear(speeds, best_vals)
        fits[name] = {'speeds': speeds, 'best_vals': best_vals,
                      'slope': slope, 'intercept': intercept, 'r2': r2}
        vals_str = ', '.join(f'{v:.4g}' for v in best_vals)
        print(f"  {name:26s} baseline={BASELINE[name]:<8.4g} @speeds→[{vals_str}]  "
              f"fit: y={slope:.5g}*speed+{intercept:.5g}  R²={r2:.2f}")
    return fits


# ─────────────────────────── Optuna(TPE) 탐색 — 랜덤서치의 대안 ───────────────────────────
# [2026-08-17 4차] run_search()의 랜덤서치는 23차원 공간에서 매번 완전히 새로 뽑은 점을
# 평가할 뿐, 지금까지의 평가 결과가 다음 시도에 전혀 반영되지 않는다. Optuna의 TPE
# (Tree-structured Parzen Estimator) 샘플러는 지금까지 본 (params→score) 관측치로 "좋은
# 점 주변"과 "나쁜 점 주변"의 분포를 추정해 다음 시도를 그 사이 유망한 영역에서 고른다 —
# 같은 evaluate()/score()를 objective로 그대로 재사용하므로 랜덤서치와 결과가 직접
# 비교 가능하다.
def _params_from_trial(trial):
    """optuna Trial → BASELINE과 동일한 키 구조의 params dict. sample_params()와 동일한
    범위(_candidate_range()/CONFIRM_FRAMES_CHOICES)를 그대로 재사용해 랜덤서치와 동일한
    탐색공간에서 TPE가 다음 시도를 고르게 한다."""
    p = {}
    for k in BASELINE:
        if k == 'straight_confirm_frames':
            p[k] = trial.suggest_categorical(k, CONFIRM_FRAMES_CHOICES)
            continue
        if k == 'stable_frame_min':
            p[k] = trial.suggest_categorical(k, STABLE_FRAME_MIN_CHOICES)
            continue
        lo, hi = _candidate_range(k)
        if k in ('alpha', 'straight_alpha', 'path_ema_alpha'):
            lo, hi = max(lo, 0.05), min(hi, 1.0)
        p[k] = trial.suggest_float(k, lo, hi)
    if p['corner_hold_decay_hi'] < p['corner_hold_decay_lo']:
        p['corner_hold_decay_lo'], p['corner_hold_decay_hi'] = p['corner_hold_decay_hi'], p['corner_hold_decay_lo']
    return p


def run_optuna_search(speed_norm, n_trials, seed, paths, path_meta, seed_points=None):
    """run_search()(랜덤서치) 대신 Optuna TPE로 같은 25차원 공간을 탐색.

    seed_points(다른 실행이 찾은 best_params dict들의 리스트)가 있으면 study.enqueue_trial()로
    맨 처음 트라이얼들로 먼저 평가한다 — TPE는 이 결과들도 관측치에 포함해 그 주변부터
    탐색하므로, 기존 랜덤서치가 찾은 값을 버리지 않고 이어서 개선하는 효과가 있다.
    seed_points가 이전(4차 이전) 라운드에서 온 것이면 stable_frame_min/stable_jump_max
    키가 없을 수 있는데, 그 경우 enqueue_trial()에 전달하기 전에 BASELINE 기본값으로
    채워 넣어야 한다(호출부 책임 — main()의 --seed-file 로딩부 참고)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rng = np.random.default_rng(seed)  # run_search()와 동일 — 전체 트라이얼이 한 스트림을 공유

    def objective(trial):
        params = _params_from_trial(trial)
        s, results = evaluate(params, speed_norm, paths, path_meta, rng)
        trial.set_user_attr('results', results)
        return s

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.enqueue_trial(dict(BASELINE))
    for sp in (seed_points or []):
        # sp.get(k, BASELINE[k]): 4차(디바운스) 이전 라운드가 남긴 시드는
        # stable_frame_min/stable_jump_max 키가 없을 수 있다 — 없으면 BASELINE(=config.py
        # 현재값)로 채워 넣는다. 매번 손으로 JSON을 갱신하지 않아도 구버전 시드 파일을
        # 그대로 재사용할 수 있게 하는 안전장치.
        study.enqueue_trial({k: sp.get(k, BASELINE[k]) for k in BASELINE})
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # BASELINE은 항상 첫 enqueue_trial이라 study.trials[0]이 그 결과다 — run_search()가
    # baseline_score/results를 별도 재평가 없이 첫 evaluate() 호출 결과 그대로 쓰는 것과 동일하게,
    # 여기서도 최적화 도중 실제로 쓰인(같은 rng 스트림의) 결과를 그대로 재사용한다.
    baseline_trial = study.trials[0]
    baseline_score = baseline_trial.value
    baseline_results = baseline_trial.user_attrs['results']

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_results = best_trial.user_attrs['results']

    return {
        'speed_norm': speed_norm,
        'baseline_score': baseline_score,
        'baseline_results': baseline_results,
        'best_score': best_trial.value,
        'best_params': best_params,
        'best_results': best_results,
        'n_trials': n_trials,
    }


def fmt_results(results):
    lines = []
    for name in ('직진', '90도커브', 'S자커브', '실전트랙'):
        if name not in results:
            continue
        r = results[name]
        lines.append(
            f"    {name:6s}: cte_rms={r['cte_rms']*100:5.1f}cm cte_max={r['cte_max']*100:5.1f}cm "
            f"steer_rms={r['steer_rms']:5.2f}도 osc={r['osc_per_sec']:4.1f}/s "
            f"직진태그={r['straight_frac']*100:5.1f}% avg_speed={r['avg_speed']:5.2f} "
            f"소요={r['elapsed_s']:4.1f}s 완주={'Y' if r['completed'] else 'N'}"
        )
        if 'by_mode' in r:
            mode_str = '  '.join(
                f"{m}:cte_rms={v['cte_rms']*100:.1f}cm/max={v['cte_max']*100:.1f}cm"
                for m, v in r['by_mode'].items()
            )
            lines.append(
                f"           └ 모드별 {mode_str}  전환구간cte_rms={r['transition_cte_rms']*100:.1f}cm "
                f"max={r['transition_cte_max']*100:.1f}cm"
            )
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=int, default=400, help='속도별 랜덤 샘플 수(기본 400)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--speeds', type=float, nargs='+', default=[10.0, 20.0])
    ap.add_argument('--sensitivity', action='store_true',
                     help='조인트 랜덤서치 대신 파라미터별 1D 속도 민감도 분석(추세선 피팅용)')
    ap.add_argument('--grid-n', type=int, default=21, help='--sensitivity의 파라미터별 그리드 촘촘함')
    ap.add_argument('--repeats', type=int, default=3, help='--sensitivity의 후보값별 반복 평가 횟수(잡음 평균)')
    ap.add_argument('--optuna', action='store_true',
                     help='랜덤서치 대신 Optuna(TPE) 베이지안 최적화로 탐색 — optuna 패키지 필요')
    ap.add_argument('--trials', type=int, default=2000, help='--optuna의 트라이얼 수(기본 2000)')
    ap.add_argument('--seed-file', type=str, default=None,
                     help='--optuna 전용: 이전 탐색이 찾은 best_params dict 리스트가 담긴 JSON 파일 — '
                          'study.enqueue_trial()로 맨 앞 트라이얼로 먼저 평가해 이어서 탐색한다')
    args = ap.parse_args()

    paths, path_meta = build_paths()

    if args.sensitivity:
        return run_sensitivity(args.speeds, args.grid_n, args.repeats, paths, path_meta)

    if args.optuna:
        seed_points = []
        if args.seed_file:
            with open(args.seed_file) as f:
                seed_points = json.load(f)
        print(f"[Optuna] TPE 베이지안 최적화, trials={args.trials}, seed_points={len(seed_points)}개 "
              f"(기존 탐색 결과에서 이어감)\n")
        all_out = {}
        for sp in args.speeds:
            print(f"===== speed={sp} (SPEED_NORMAL 역할, 모터unit, ≈{sp*MPS_PER_UNIT:.2f}m/s"
                  f"{' — 5~10 회귀범위 밖 외삽' if sp > 10 else ''}) =====")
            out = run_optuna_search(sp, args.trials, args.seed, paths, path_meta, seed_points)
            all_out[sp] = out
            print(f"  baseline score={out['baseline_score']:.2f}")
            print(fmt_results(out['baseline_results']))
            print(f"  best({args.trials} trials) score={out['best_score']:.2f}")
            print(fmt_results(out['best_results']))
            print("  best_params:")
            for k, v in out['best_params'].items():
                print(f"    {k} = {v:.4g}" if isinstance(v, float) else f"    {k} = {v}")
            print()
        return all_out

    print(f"[설정] ROI_W={ROI_W_PX}px ROI_H≈{ROI_H_PX:.0f}px(DL_BEV_FAR_LIMIT_M={ROI_DEPTH_M}m 기준) "
          f"noise_std={NOISE_STD_PX}px WHEELBASE_M={WHEELBASE_M} MPS_PER_UNIT={MPS_PER_UNIT}")
    print(f"[baseline] SPEED_NORMAL=3.0 재튜닝 시점 config.py 현재값: {BASELINE}\n")

    all_out = {}
    for sp in args.speeds:
        print(f"===== speed={sp} (SPEED_NORMAL 역할, 모터unit, ≈{sp*MPS_PER_UNIT:.2f}m/s"
              f"{' — 5~10 회귀범위 밖 외삽' if sp > 10 else ''}) =====")
        out = run_search(sp, args.samples, args.seed, paths, path_meta)
        all_out[sp] = out
        print(f"  baseline score={out['baseline_score']:.2f}")
        print(fmt_results(out['baseline_results']))
        print(f"  best({args.samples}샘플) score={out['best_score']:.2f}")
        print(fmt_results(out['best_results']))
        print("  best_params:")
        for k, v in out['best_params'].items():
            print(f"    {k} = {v:.4g}" if isinstance(v, float) else f"    {k} = {v}")
        print()

    return all_out


if __name__ == '__main__':
    main()
