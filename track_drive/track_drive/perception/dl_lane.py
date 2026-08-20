#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# dl_lane.py — TwinLiteNet(ONNX Runtime) 기반 딥러닝 차선인식 백엔드.
#
# [2026-08-13] 모델을 twinlitenetplus_medium_v2.onnx(bootstrap_v2 1016장 기준,
# §2.14/§2.18)에서 fine-tune 저장소 v1.2.0(`models/twinlitenetplus_kmu_v1.2.0.onnx`)로
# 교체했다 — bootstrap_v2(1016) + lap_005(지그재그 주행 보강 2430) = 3446장으로
# 재학습한 최신 결과물(fine-tune 저장소 PROGRESS.md §2.26/§2.27), 사람 GT 1016장 기준
# da IoU 0.945→0.957 / ll IoU 0.577→0.599(v1.0.0 대비, README 참고). 입출력 텐서
# 이름('images'/'da'/'ll'), 전처리(letterbox 없이 리사이즈 → BGR→RGB → /255, mean/std
# 정규화 없음), 출력 형식((1,2,H,W) raw logit, softmax 후 채널1=foreground), 입력
# 해상도(640x384)는 medium_v2와 동일 — DL_INPUT_H 변경 없음. 이 .onnx도 가중치를
# 외부 데이터 파일(`twinlitenetplus_kmu_v1.2.0.onnx.data`, 같은 디렉터리에 있어야 함 —
# onnx 파일 내부에 상대경로로 박혀 있어 파일명을 바꾸면 로드가 깨진다)로 분리해
# export됐다(fine-tune 저장소 원본 파일명 `best.onnx`(+`.onnx.data`)를 이 레포용으로
# `twinlitenetplus_kmu_v1.2.0.onnx`로 리네임하면서, onnx 내부 external-data location도
# 새 파일명에 맞게 재작성함 — 원본 그대로 리네임만 하면 로드가 깨지는 버그가 fine-tune
# 저장소 쪽에도 있었음, PROGRESS.md §2.27 참고). best.onnx(원조)/
# twinlitenetplus_small_bootstrap_v2.onnx(이전 fine-tune)/twinlitenetplus_medium_v2.onnx
# (직전 버전)는 롤백/비교용으로 그대로 남겨뒀다(이제 기본 경로로는 안 쓰임).
# ⚠️ 이 모델은 정적 이미지/ROI 커버리지 기준으로만 검증됐고 실차 주행 테스트는 아직
# 안 됨(fine-tune 저장소 README "실차에 바로 쓰기 전에 반드시 실제 주행 테스트로
# 검증할 것" 참고) — 처음 투입 시 hough 백엔드로 즉시 폴백 가능한 상태로 테스트할 것.
# hough_lane.HoughLaneDetector / perc_floor.LaneDetector와 동일하게
#   detect(frame) -> (lane_valid, lane_offset, lane_lookahead, lane_center, path, debug_img)
# 인터페이스를 구현하므로 track_drive.py의 perc_lane()은 수정 없이 그대로 재사용된다
# (LANE_DETECTOR_BACKEND 플래그로 세 백엔드 중 하나를 고른다). path는 밴드(row 구간)별
# 중심점(config.DL_CENTER_MODE로 고르는 세 알고리즘 중 하나 — 아래 "밴드별 중심 계산"
# 참고)에
# lane_util._fit_and_sample_path()로 선형보간해 만든 명시적 경로(ROI 픽셀좌표
# 웨이포인트, 가까운점→먼점) — controller/pure_pursuit.py가 조향각 계산에 직접 사용한다.
#
# ── 실시간 전략: 제어루프와 분리된 백그라운드 스레드 ──
#   control_loop()는 0.05s(20Hz) 고정 타이머라 조향 명령(drive(), ANGLE_RATE_MAX 등)이
#   여기에 실려 나간다. DL 추론 1회 지연시간은 하드웨어/모델에 따라 들쭉날쭉한데, 이걸
#   perc_lane() 안에서 동기 호출하면 지연이 그대로 조향 발행 주기에 전파된다. 그렇다고
#   "N프레임마다 1번 추론"처럼 고정 스킵 비율을 정하면, 실측 전에는 하드웨어가 빠른 경우엔
#   손해고 느린 경우엔 여전히 밀린다. 그래서 별도 데몬 스레드가 "최신 프레임 하나만 들고
#   있다가 다 되면 그걸 처리"하는 방식(더블 버퍼)으로 자기 페이스껏 추론하고,
#   DLLaneDetector.detect()는 그 시점까지 계산된 최신 결과를 논블로킹으로 즉시 반환한다.
#   카메라가 추론보다 빠르면 중간 프레임은 자연히 버려지고("최신 마스크 재사용"), 추론이
#   더 빠르면 사실상 매 프레임 처리된다 — 하드웨어 성능에 자동으로 적응한다.
#   디바운스(STABLE_FRAME_MIN/JUMP_MAX 상당)는 제어루프 틱이 아니라 "새 추론이 끝난 시점"
#   기준으로 걸어야 원래 의미(연속 몇 *프레임*이 안정적이었는가)가 유지되므로, 워커 스레드
#   안(DLSlideWindow.detect() 내부, SlideWindow._debounce() 재사용)에서만 적용된다.
#
# ── [2026-08-10] 밴드별 중심 계산: config.DL_CENTER_MODE가 고르는 세 가지 서로 다른
#   알고리즘 — 상세 설계 근거는 각 함수 docstring/config.py DL_CENTER_MODE 주석 참고 ──
#   'da'    : da(주행가능영역) 무게중심(_slice_centers(), cv2.moments)을 밴드별 중심으로
#             바로 쓴다(main 기본값). 덩어리 선택은 _largest_da_component() —
#             ①시드(차량 위치와 맞닿은 덩어리) → ②연속성(직전 프레임과 가장 가까운
#             덩어리) → ③면적순위(최후 폴백) 순, 면적 상한 체크는 없다(실차 검증 결과
#             면적만으로 da를 거르는 방식 자체가 불신뢰였음 — da가 옆 차선과 붙는 문제는
#             이제 _clip_da_by_ll()이 ll 잔상(decay)/가상경계로 전담한다).
#   'll_da' : "corridor" 알고리즘(_corridor_slice_centers()) — ll로 도로 폭 자체를
#             규정하고(1번째~3번째 선을 도로 경계로), 그 안에서 da로 장애물 회피용 열린
#             공간을 찾는다. "자기 차선 하나"를 전제로 한 largest-component/클리핑은
#             건너뛴다.
#   'll'    : ll을 흰선/노란선으로 분리(_split_ll_by_yellow())한 뒤, 노란 중앙선 +
#             (차선 판정에 따른) 한쪽 흰색 경계선을 추적한다(_ll_yellow_white_centers()).
#             da 폴백은 없다 — 대신 노란/흰 간격 기반 재구성 + 잔상으로 저신뢰
#             추정한다(config.py DL_CENTER_MODE 'll' 주석 참고).
#   'da'/'ll' 두 모드는 da 파편화 대응(_largest_da_component())/옆 차선
#   클리핑(_clip_da_by_ll())을 공유한다('ll_da'=corridor는 둘 다 건너뜀).
#   [2026-08-18] ll 프레임 단위 sanity check(DL_LL_SANITY_MIN_RATIO)는 삭제됨 — ll을
#   더 이상 안 쓰기로 확정, lane_valid/path_ok 모두 da 중심점 유무로만 판정.
#=============================================
import argparse
import os
import time
import threading
from collections import deque

import cv2
import numpy as np

from .lane_util import SlideWindow
from ..kr_text import put_text_kr

try:
    import onnxruntime as ort
except ImportError as _e:
    ort = None
    _ORT_IMPORT_ERROR = _e
else:
    _ORT_IMPORT_ERROR = None

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


# ── 모델 입출력 스펙 (fine-tune 저장소 twinlitenetplus_kmu_v1.2.0.onnx 기준 —
#   onnxruntime InferenceSession.get_inputs()/get_outputs()로 직접 확인함, 2026-08-12,
#   medium_v2와 입출력 스펙 동일) ──
#   images 텐서 (batch,3,384,640) float32 NCHW. 전처리는 letterbox 없이 640x384로 그냥
#   리사이즈(harrylal 원본 blobFromImage 방식 그대로 유지) → BGR→RGB → /255.0
#   (mean/std 정규화 없음).
DL_INPUT_W = 640
DL_INPUT_H = 384
DL_INPUT_NAME = 'images'
DL_OUTPUT_NAMES = ('da', 'll')

# ── 세그멘테이션 입력은 절대 자르지 않는다 ──
#   원본 리포(blobFromImage)와 동일하게 raw 프레임 전체를 그대로 640x384로 리사이즈해서
#   모델에 넣는다(추가 크롭 없음). 관심영역은 "모델에 들어가기 전"이 아니라 "모델에서 나온
#   세그멘테이션 결과(da/ll)를 원본 프레임 크기로 되돌린 뒤" 잘라서 쓴다 — 아래
#   DL_ROI_Y0/Y1 참고.
#
# da/ll 둘 다 (batch,2,384,640) raw logit. 채널축 softmax 후 채널1이 foreground 확률.
# DL_FG_THRESHOLD(이진화 임계값), DL_ROI_Y0/Y1(원본 480행 기준 절대 픽셀, 실차 실측값)는
# config.py에 있다 — 실차 테스트 중 값을 바꾸려면 이 파일이 아니라 config.py를 고칠 것.
from ..config import DL_FG_THRESHOLD, DL_LL_FG_THRESHOLD, DL_ROI_Y0, DL_ROI_Y1

# ── BEV(원근변환) — 2026-08-05 bev_point_picker.py로 실측 캘리브레이션 ──
#   DL 백엔드는 원래 원근(perspective) 픽셀 스케일 그대로 da/ll 중심선을 뽑았다(위
#   "SlideWindow moments" 섹션 주석 참고) — 카메라에 가까운 픽셀과 먼 픽셀이 나타내는
#   실제 거리(m/px)가 달라서(원근 압축), 같은 실제 곡률도 화면 위치마다 다른 곡률로
#   보이고 PIXELS_PER_METER(config.py)도 애초에 상수로 정의가 안 되는 문제가 있었다.
#   좌/우 백선을 근거리(BL/BR)~1m지점(TL/TR)에서 직접 찍어(bev_point_picker.py) 얻은
#   4점으로 이 왜곡을 제거한다.
#     실측: 폭 W=0.8m(LANE_WIDTH_M*2, config.py 기존 실측 — 두 지점 모두 같은 두
#           백선이므로 폭은 근거리/원거리 공통), 길이 L=1.0m(TL~BL 직접 실측, 2026-08-05)
#   DL_USE_BEV/DL_PIXELS_PER_METER와 실측 원점(DL_BEV_SRC_PX_RAW, 원본 프레임 절대좌표)은
#   config.py에 있다 — 여기서는 그 절대좌표를 ROI 기준 상대좌표로만 변환한다(DL_ROI_Y0만큼
#   빼기. x는 원본 그대로 — ROI가 가로는 안 자름). 순서는 TL(좌상/먼왼쪽)→TR(우상/먼오른쪽)→
#   BR(우하/가까운오른쪽)→BL(좌하/가까운왼쪽).
#   [주의] LANE_DETECTOR_BACKEND='dl'(config.py 기본값) 자체가 "실차 트랙 전체 조건에서
#   아직 미검증" 상태고, BEV로 좌표계를 바꾸면 DL_DA_MIN_COMPONENT_AREA/DL_SLICE_OUTLIER_MAX/
#   DL_STABLE_JUMP_MAX 등 원근 픽셀 스케일 기준으로 잡힌 튜닝값들의 "픽셀당 의미"가 전부
#   바뀐다(아래 캔버스 자동계산 결과 면적이 원래 ROI의 약 1.95배).
from ..config import DL_USE_BEV, DL_BEV_SRC_PX_RAW, DL_PIXELS_PER_METER, DL_BEV_FAR_LIMIT_M
DL_BEV_SRC_PX = DL_BEV_SRC_PX_RAW - np.float32([0, DL_ROI_Y0])

# ── [2026-08-05] 캔버스 크기를 손으로 정하지 않고 "ROI 전체가 어디까지 매핑되는지"
#   역산해서 자동으로 딱 맞춘다. 처음엔 640x400을 임의로 잡았는데, 실측 4점(근거리~1m)이
#   ROI 전체(DL_ROI_Y0:Y1, 640px 폭) 중 일부만 커버하다 보니 그 바깥으로 워프되는 영역이
#   생각보다 넓어서 위/아래/양옆에 안 쓰는 검은 여백이 크게 남았다(디버그 창에서 실측 확인).
#   방법: ①실측 4점→목적 사각형(W=0.8m*스케일, H=1.0m*스케일)으로 1차 M을 구하고,
#   ②그 M으로 ROI 네 모서리가 어디로 매핑되는지 계산해서 바운딩박스를 구한 뒤,
#   ③목적점 전체를 그 바운딩박스의 좌상단이 (0,0)에 오도록 평행이동 — 이러면 ROI 전체가
#   여백 없이 캔버스에 꽉 찬다.
#   단, 이렇게 해도 위(원거리)/아래(근거리) 폭이 서로 다른 "사다리꼴" 모양 자체는 없어지지
#   않는다 — 카메라 화각이 고정이라 원거리일수록 같은 화면폭이 더 넓은 실제거리를 담기
#   때문에(원근투영의 기본 성질), 사다리꼴 아래쪽 양 모서리에 남는 검은 삼각형은 "잘못
#   잡은 여백"이 아니라 "그 위치엔 애초에 대응하는 도로 데이터가 없다"는 뜻이다.
DL_ROI_W_PX = 640  # 원본 카메라 프레임 폭(TwinLiteNetEngine.infer_raw()가 업샘플링하는 크기)
_dl_roi_h_px = DL_ROI_Y1 - DL_ROI_Y0
_dl_block_w = 0.8 * DL_PIXELS_PER_METER
_dl_block_h = 1.0 * DL_PIXELS_PER_METER
_dl_block_dst = np.float32([[0, 0], [_dl_block_w, 0], [_dl_block_w, _dl_block_h], [0, _dl_block_h]])
_dl_M0 = cv2.getPerspectiveTransform(DL_BEV_SRC_PX, _dl_block_dst)

_dl_roi_corners = np.float32([
    [0, 0], [DL_ROI_W_PX - 1, 0], [DL_ROI_W_PX - 1, _dl_roi_h_px - 1], [0, _dl_roi_h_px - 1]
]).reshape(-1, 1, 2)
_dl_mapped_corners = cv2.perspectiveTransform(_dl_roi_corners, _dl_M0).reshape(-1, 2)
_dl_min_xy = _dl_mapped_corners.min(axis=0)
_dl_max_xy = _dl_mapped_corners.max(axis=0)

DL_BEV_CANVAS_W = int(np.ceil(_dl_max_xy[0] - _dl_min_xy[0])) + 1
DL_BEV_CANVAS_H = int(np.ceil(_dl_max_xy[1] - _dl_min_xy[1])) + 1
DL_BEV_DST_PX = _dl_block_dst - _dl_min_xy  # 목적점을 캔버스 원점 기준으로 평행이동

# [2026-08-17] BEV 캔버스에서 "차량 중심" x좌표 — 지금까지 조향/시각화 전부
# roi_w/2.0(캔버스 폭의 단순 절반)을 차량 중심으로 가정했는데, DL_BEV_SRC_PX 4점(좌/우
# 백선을 실측해 손으로 찍은 점)이 카메라 광축 기준으로 좌우 대칭이란 보장이 없어서
# roi_w/2.0이 실제 차량 위치와 어긋날 수 있다. 사다리꼴의 "가까운 변"(BR-BL, 차량 바로
# 앞)의 중점을 DL_BEV_SRC_PX→DL_BEV_DST_PX와 동일한 워프에 통과시켜, 그 변환이 실제로
# 만들어내는 캔버스 좌표를 차량 중심으로 쓴다. DL_BEV_SRC_PX 순서는 TL,TR,BR,BL이므로
# 가까운 변은 인덱스 2(BR)/3(BL).
_dl_bottom_mid_src = ((DL_BEV_SRC_PX[2] + DL_BEV_SRC_PX[3]) / 2.0).reshape(1, 1, 2)
DL_BEV_VEHICLE_CENTER_X = float(
    cv2.perspectiveTransform(
        _dl_bottom_mid_src, cv2.getPerspectiveTransform(DL_BEV_SRC_PX, DL_BEV_DST_PX)
    )[0, 0, 0]
)

# [2026-08-06] 원거리 크롭 행(row) 계산 — config.py의 DL_BEV_FAR_LIMIT_M 주석 참고.
#   block 좌표계에서 근거리 기준점(BL/BR)은 y=_dl_block_h(=1.0m*px/m)이고, 캔버스로 옮기면
#   위 DL_BEV_DST_PX와 같은 평행이동(-_dl_min_xy)을 받는다. 거기서 DL_BEV_FAR_LIMIT_M(m)
#   만큼 위(원거리 방향, y 감소)로 올라간 행이 크롭 경계 — 그 행 "위"(더 먼 부분)를 버린다.
#   DL_PIXELS_PER_METER/DL_BEV_SRC_PX_RAW는 그대로라 캘리브레이션 왜곡 없이 순수하게
#   "얼마나 먼 데까지 볼지"만 제한한다(위 config.py 주석 참고).
_dl_near_canvas_y = _dl_block_h - _dl_min_xy[1]
DL_BEV_FAR_CROP_ROW = max(0, int(round(_dl_near_canvas_y - DL_BEV_FAR_LIMIT_M * DL_PIXELS_PER_METER)))

# ── SlideWindow moments 로직 재사용을 위한 DL 전용 튜닝값 ──
#   classic 파이프라인은 BEV로 워프된 ROI px 스케일이고, DL은 원본 카메라 프레임 px
#   스케일(BEV 없음, 위 DL_ROI_Y0:Y1로 자른 640폭 대역)이라 픽셀당 의미가 달라 값을 따로
#   둔다 — 알고리즘 자체는 lane_util.py의 MOMENT_*/LANE_SLICE_*/STABLE_* 와 동일(이름만
#   DL_ 접두어). 이제 좌/우 두 갈래가 아니라 da 중심선 "한 갈래"에만 적용된다.
#   전부 config.py에 있다(DL_N_SLICES/MIN_PIXELS/NEAR_SLICES/FAR_SLICES/SLICE_OUTLIER_MAX/
#   SLICE_FIT_MIN/STABLE_FRAME_MIN/STABLE_JUMP_MAX/DA_MIN_COMPONENT_AREA/
#   LL_SANITY_MIN_RATIO/LL_CLIP_MARGIN_PX, DEBUG_VIZ_DL_LANE, YELLOW_LOWER/UPPER,
#   FPS_LOG_PERIOD_SEC) — 실차 테스트 중 값을 바꾸려면 이 파일이 아니라 config.py를 고칠 것.
from ..config import (
    DL_N_SLICES, DL_MIN_PIXELS, DL_NEAR_SLICES, DL_FAR_SLICES,
    DL_SLICE_OUTLIER_MAX, DL_SLICE_FIT_MIN,
    DL_STABLE_FRAME_MIN, DL_STABLE_JUMP_MAX,
    # [2026-08-19] 근접 밴드 이상치 오판 방지(1차) + hold 타임아웃(2차) — README 참고
    DL_NEAR_HOLD_MAX_FRAMES,
    DL_DA_MIN_COMPONENT_AREA,
    DL_DA_SEED_ROWS_PX, DL_DA_SEED_HALF_WIDTH_PX,
    # [2026-08-12] DL_CENTER_MODE='da' 밴드 중심 탐색창(prior)+속도예측+앵커링 — README §2.27
    DL_DA_SEARCH_HALF_WIDTH_PX, DL_DA_SEARCH_WIDEN_STEP_PX, DL_DA_SEARCH_WIDEN_MAX_PX,
    DL_DA_VELOCITY_EMA_ALPHA, DL_DA_VELOCITY_MAX_PX, DL_DA_BAND_ANCHOR_ALPHA,
    DL_LL_CLIP_MARGIN_PX,
    DL_LL_DECAY_ALPHA, DL_LL_DECAY_MIN_VALUE,
    DL_CENTER_MODE, DL_LL_ALGO, DL_LL_SIDE_MIN_PIXELS, DL_DA_SKIP_LL_CLIP,
    DL_LL_SEARCH_HALF_WIDTH_PX,
    # [2026-08-14] da 안전마진(차량 폭) 침식 — README §2.30
    # [2026-08-17g] 방해차량 "뒤" 방향 속도비례 추가마진 — config.py DL_DA_REAR_MARGIN_* 주석 참고
    DL_DA_APPLY_VEHICLE_MARGIN, DL_DA_VEHICLE_MARGIN_M, VEHICLE_WIDTH_M,
    DL_DA_SIDE_MARGIN_M,
    DL_DA_REAR_MARGIN_REACT_SEC, DL_DA_REAR_MARGIN_MAX_M,
    # [2026-08-15] avoid-hold 개선 적용2(da 연속성 보조트리거)/적용3(방향 힌트) — README §2.32,
    # avoid_hold_improvement_proposal.md
    AVOID_HOLD_DA_AREA_JUMP_RATIO, AVOID_HOLD_DIR_BIAS_PX,
    # [2026-08-20] da 근접 컷(obstacle-cut, ENABLE_OBSTACLE_CUT) — README §2.5x 참고
    LANE_WIDTH_M, OBSTACLE_CUT_NEAR_M, OBSTACLE_CUT_LANE_HALF_WIDTH_PX, OBSTACLE_CUT_MIN_REMAIN_PX,
    # DL_LL_ALGO='yw'(팀원 작성, main 기본) 전용
    DL_LL_YELLOW_GAP_INIT_PX, DL_LL_YELLOW_GAP_EMA_ALPHA,
    DL_LL_YELLOW_GAP_MIN_PX, DL_LL_YELLOW_GAP_MAX_PX,
    DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX,
    # DL_LL_ALGO='lr'(이지유 작성) 전용
    DL_LL_WIDTH_MIN_PX, DL_LL_WIDTH_MAX_PX, DL_LL_WIDTH_EMA_ALPHA,
    # [2026-08-12] 아래 셋은 이제 'yw'/'lr' 둘 다 쓴다(config.py DL_LL_ALGO 주석 참고)
    DL_LL_VELOCITY_EMA_ALPHA, DL_LL_VELOCITY_MAX_PX,
    DL_LL_SEARCH_WIDEN_STEP_PX, DL_LL_SEARCH_WIDEN_MAX_PX, DL_LL_BAND_ANCHOR_ALPHA,
    DL_CORRIDOR_LINE_MIN_PIXELS, DL_CORRIDOR_LINE_MERGE_PX,
    DL_CORRIDOR_WIDTH_MIN_PX, DL_CORRIDOR_WIDTH_MAX_PX, DL_CORRIDOR_MIN_PASSABLE_PX,
    DL_CORRIDOR_VELOCITY_EMA_ALPHA, DL_CORRIDOR_VELOCITY_MAX_PX,
    DL_LL_YELLOW_VOTE_RATIO, DL_LL_YELLOW_MIN_AREA,
    DEBUG_VIZ_DL_LANE, YELLOW_LOWER, YELLOW_UPPER, FPS_LOG_PERIOD_SEC,
    DL_DEBUG_HISTORY_LEN,
)

# ── [2026-08-14] da 안전마진(차량 폭) 침식 커널 — README §2.30 "da 안전마진 설계 논의" ──
#   VEHICLE_WIDTH_M(실측 차폭)의 절반 + 마진(여유)을 DL_PIXELS_PER_METER로 픽셀 환산해
#   반경으로 쓴다. DL_CENTER_MODE='da'에서만 쓰인다(detect() 참고).
#   [2026-08-19] 좌우(가로, rx)와 전후(세로, ry 기본값)를 서로 다른 설정값에서 뽑도록
#   분리 — 예전엔 이 둘이 같은 DL_DA_VEHICLE_MARGIN_M 하나를 공유해서, "방해차량 옆
#   여유만 조금 늘리고 싶다"는 요청을 반영하려면 전후 기본값까지 같이 딸려 올라갔다.
#   이제 DL_DA_SIDE_MARGIN_M(좌우 전용)과 DL_DA_VEHICLE_MARGIN_M(전후 기본값 전용, 아래
#   REAR_MARGIN_*이 여기에 속도비례로 얹힘)을 독립적으로 조정할 수 있다.
_DL_DA_MARGIN_PX = int(round((VEHICLE_WIDTH_M / 2.0 + DL_DA_VEHICLE_MARGIN_M) * DL_PIXELS_PER_METER))
_DL_DA_SIDE_MARGIN_PX = int(round((VEHICLE_WIDTH_M / 2.0 + DL_DA_SIDE_MARGIN_M) * DL_PIXELS_PER_METER))
# [2026-08-17g] 세로(진행방향) 반경만 v_mps에 비례해 늘려서 "뒤" 쪽 여유를 속도에 맞게
#   더 벌린다 — config.py DL_DA_REAR_MARGIN_REACT_SEC/MAX_M 주석 참고. 속도에 따라 매
#   프레임 달라지므로(다른 커널들과 달리) 모듈 임포트 시 한 번만 만들어두지 못하고
#   _apply_vehicle_margin()이 호출될 때마다 새로 만든다 — cv2.getStructuringElement는
#   이 크기(수십 px)에서는 무시할 만큼 가볍다.
def _dl_da_margin_kernel(v_mps):
    if not DL_DA_APPLY_VEHICLE_MARGIN or (_DL_DA_MARGIN_PX <= 0 and _DL_DA_SIDE_MARGIN_PX <= 0):
        return None
    extra_m = min(DL_DA_REAR_MARGIN_REACT_SEC * max(float(v_mps), 0.0), DL_DA_REAR_MARGIN_MAX_M)
    ry = _DL_DA_MARGIN_PX + int(round(extra_m * DL_PIXELS_PER_METER))
    rx = _DL_DA_SIDE_MARGIN_PX
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(rx, 0) * 2 + 1, max(ry, 0) * 2 + 1))

# [2026-08-10] visualize()가 result 패널 맨 위에 그리는 모드 배너 색을 한곳에서 관리.
# [2026-08-11] 원래는 여기 색을 _build_params_panel()의 배너와도 맞췄었는데, 그 함수를
# 지우면서(파라미터 텍스트 목록이 config.py를 보면 알 수 있는 고정값 위주라 화면만
# 차지했음, show_debug_windows() 주석 참고) 지금은 참조하는 곳이 이 하나뿐이다.
DL_MODE_COLORS = {
    'da':    (170, 60, 0),    # 진한 파랑
    'll':    (0, 130, 0),     # 진한 초록
    'll_da': (130, 0, 130),   # 진한 자홍
}
DL_MODE_COLOR_DEFAULT = (60, 60, 60)  # 위 셋에 없는 값(오타 등) 대비 폴백


DL_MODEL_FILENAME = 'twinlitenetplus_kmu_v1.2.0.onnx'


def _default_model_path():
    """모델 가중치 파일(twinlitenetplus_kmu_v1.2.0.onnx) 기본 경로.
    1순위: colcon install된 share 디렉터리(share/track_drive/models/<파일명>)
    2순위: 소스트리에서 직접 실행 중일 때(개발 중, colcon build 전) — track_drive 패키지 디렉터리 기준 상대경로
    같은 디렉터리의 <파일명>.data(외부 데이터 파일)도 같이 있어야 한다 — onnx 파일 안에
    상대경로로 박혀 있어 둘 중 하나만 옮기면 로드가 깨진다.
    """
    if get_package_share_directory is not None:
        try:
            share_dir = get_package_share_directory('track_drive')
            candidate = os.path.join(share_dir, 'models', DL_MODEL_FILENAME)
            if os.path.isfile(candidate):
                return candidate
        except Exception:
            pass
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_dir, 'models', DL_MODEL_FILENAME)


class TwinLiteNetEngine:
    """ONNX Runtime으로 TwinLiteNet 모델을 로드하고 전처리/추론/후처리(softmax)를 담당."""

    def __init__(self, model_path=None, providers=None, logger=None):
        if ort is None:
            raise ImportError(
                'onnxruntime이 설치돼 있지 않습니다. Jetson 보드에서는 TensorRT/CUDA를 지원하는 '
                'onnxruntime-gpu(또는 jetson 전용 빌드)를 설치해야 합니다. '
                f'원래 import 에러: {_ORT_IMPORT_ERROR}'
            )

        self.model_path = model_path or _default_model_path()
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f'TwinLiteNetPlus 가중치 파일을 찾을 수 없습니다: {self.model_path}\n'
                'fine-tune 저장소의 outputs/models/best.onnx(v1.2.0, medium config)를 '
                'twinlitenetplus_kmu_v1.2.0.onnx(+.onnx.data)로 리네임해서 이 경로에 두세요 '
                '(리네임 시 onnx 내부 external-data location도 새 파일명에 맞게 재작성해야 함 — '
                'fine-tune 저장소 PROGRESS.md §2.27 참고).'
            )
        self._logger = logger

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Jetson은 코어 수가 적어 onnxruntime 기본 스레드풀이 ROS2 콜백 스레드와 경쟁할 수
        # 있으므로 세션 내부 스레드 수를 제한한다.
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        available = set(ort.get_available_providers())
        if providers is None:
            # [2026-08-14 실차 확인] 원래는 TensorRT EP > CUDA EP > CPU EP 순이었다 —
            # 그 순서 자체는 이전 모델(twinlitenetplus_medium_v2.onnx)에서 TensorRT가
            # 정상 동작해서 붙인 우선순위였다(아래 model_path FileNotFoundError 메시지가
            # 안내하는 v1.2.0 모델로 교체되기 전 기준). 그런데 오늘 새로 교체된
            # twinlitenetplus_kmu_v1.2.0.onnx는 trt_cache가 전혀 없는 상태에서(교체 직후라
            # 당연함) TensorRT 엔진을 처음부터 빌드하는데, 실측(standalone 스크립트로 직접
            # infer_raw() 호출) 결과 4분 넘게도 첫 추론 한 번이 안 끝났다 — yolo_cone.py의
            # cone_best_n.onnx가 겪은 것과 같은 부류의 문제다(그쪽은 TRT가 그 모델 자체를
            # 아예 못 빌드해서 ~456초 뒤에야 조용히 CUDA로 자동 폴백, yolo_cone.py
            # YoloConeEngine.__init__ 주석 참고). xydrive는 재출발/재테스트마다 프로세스를
            # 새로 띄우므로(xydrive 함수가 매번 kill -9 후 재실행), 이 지연이 매번
            # 반복되면 사실상 추론이 한 번도 안 끝난 채로 계속 재시작만 되는 상태가 된다
            # (실측 재현됨 — DA/LL 디버그 창이 안 뜨고 lane_valid가 계속 False였던 원인).
            # CUDAExecutionProvider로 강제해보니 로드 0.3초, 첫 추론 0.9초, 이후 ~5.7fps로
            # da_prob이 정상 범위(최대 0.99, DL_FG_THRESHOLD=0.5 초과 비율 27%)로 나와
            # 모델 자체는 문제 없음을 확인했다 — 그래서 이 모델도 cone과 동일하게 CUDA를
            # 우선한다. TensorRT는 교집합에서 아예 제외한다(교집합에 넣어두면 provider
            # 리스트에 남아 다음 로드 때 다시 그 긴 빌드를 시도할 여지가 있음 — 완전히
            # 배제하는 게 cone과 같은 방식). 이 모델용 trt_cache가 나중에 실제로 완성되고
            # (수 분 이상 켜둔 채 기다려서) TensorRT가 더 빠르다는 게 실측되면 그때
            # 되돌릴 것 — 지금은 "매번 멈춰있는 것보다 확실히 도는 것"을 우선한다.
            priority = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            providers = [p for p in priority if p in available] or ['CPUExecutionProvider']

        provider_options = []
        for p in providers:
            if p == 'TensorrtExecutionProvider':
                # 최초 실행 시 TensorRT 엔진 빌드에 수십초~수분이 걸릴 수 있어 캐시를 켜서
                # 두 번째 실행부터는 빌드를 건너뛴다. fp16은 속도를 위해 켜뒀다 — 정확도
                # 저하가 실측으로 문제되면 끌 것(실차 미검증 트레이드오프).
                cache_dir = os.path.join(os.path.dirname(self.model_path), 'trt_cache')
                os.makedirs(cache_dir, exist_ok=True)
                provider_options.append({
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': cache_dir,
                    'trt_fp16_enable': True,
                })
            else:
                provider_options.append({})

        self.session = ort.InferenceSession(
            self.model_path, sess_options=sess_options,
            providers=providers, provider_options=provider_options,
        )
        # get_providers()는 세션에 등록된 provider들을 요청 우선순위 그대로 반환한다
        # (그래프의 각 노드가 실제 어느 EP에서 실행됐는지까지는 보장하지 않음).
        self.active_provider = self.session.get_providers()[0]
        self._log(f'TwinLiteNet ONNX 세션 로드 완료 | 최우선 provider={self.active_provider} '
                   f'(요청순위={providers})')

        self._latency_ema = None

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(f'[dl_lane] {msg}')
        else:
            print(f'[dl_lane] {msg}')

    def preprocess(self, bgr_frame):
        resized = cv2.resize(bgr_frame, (DL_INPUT_W, DL_INPUT_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]   # HWC -> NCHW
        return resized, np.ascontiguousarray(blob)

    @staticmethod
    def softmax_fg(logits_2ch):
        """(2,H,W) raw logit -> 채널1의 softmax foreground 확률 (H,W). onnxruntime 없이도
        단위 테스트 가능하도록 순수 numpy 함수로 분리."""
        m = np.max(logits_2ch, axis=0, keepdims=True)
        exp = np.exp(logits_2ch - m)
        prob = exp / np.sum(exp, axis=0, keepdims=True)
        return prob[1]

    def infer_raw(self, bgr_frame):
        """입력 : 임의 크기(H,W) BGR 프레임
        출력 : (bgr_frame, da_prob, ll_prob) — 셋 다 입력 프레임과 같은 (H,W) 좌표계.
          da_prob, ll_prob — 모델은 항상 고정 해상도(360,640)로 내놓지만, 여기서 원본
          프레임 크기로 다시 업샘플링해서 반환한다(bilinear — 이진화 전 확률맵 상태에서
          업샘플링해야 마스크 경계가 블록처럼 뭉개지지 않는다). 이렇게 해야 DL_ROI_Y0/Y1
          같은 관심영역을 (STOPLINE_ROI 등 다른 ROI들과 동일하게) 원본 프레임 절대
          픽셀좌표로 그대로 쓸 수 있다.
        """
        t0 = time.perf_counter()
        h, w = bgr_frame.shape[:2]
        _, blob = self.preprocess(bgr_frame)
        da_out, ll_out = self.session.run(list(DL_OUTPUT_NAMES), {DL_INPUT_NAME: blob})
        da_prob = self.softmax_fg(da_out[0])
        ll_prob = self.softmax_fg(ll_out[0])
        if (h, w) != (DL_INPUT_H, DL_INPUT_W):
            da_prob = cv2.resize(da_prob, (w, h), interpolation=cv2.INTER_LINEAR)
            ll_prob = cv2.resize(ll_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        dt = time.perf_counter() - t0
        self._latency_ema = dt if self._latency_ema is None else 0.8 * self._latency_ema + 0.2 * dt
        return bgr_frame, da_prob, ll_prob

    @property
    def fps(self):
        if not self._latency_ema:
            return 0.0
        return 1.0 / self._latency_ema


class DLSlideWindow(SlideWindow):
    """lane_util.SlideWindow에서 "임의의 이진마스크를 세로로 N등분해 구간별 moments
    중심을 구하고(_slice_centers), 이상치를 제거하고(_reject_outliers), 근거리/원거리
    평균을 내고(_group_mean), 선형보간으로 경로를 만들고/샘플링하고(_fit_and_sample_path),
    프레임 간 스파이크를 걸러내는(_debounce)" 범용 유틸리티들만 재사용한다.

    좌/우 ll(차선) 라인을 따로 찾아 폭을 추정해 중점을 계산하던 옛 4단계 폴백(양쪽→
    한쪽→노랑→무효, classic_cv용 calc_center() 방식)과 순수 da 무게중심 방식을 거쳐,
    config.DL_CENTER_MODE로 셋 중 하나를 고른다 — [2026-08-10] 세 모드가 이제 서로 완전히
    다른 알고리즘이다(모듈 상단 주석에 상세). 'da'는 da 무게중심(_slice_centers(),
    cv2.moments)을 밴드별 중심으로 바로 쓴다. 'll_da'는 "corridor" 알고리즘 —
    ll로 도로 폭 자체를 규정하고 그 안에서 da로 열린 공간(장애물 회피)을 찾는다
    (_corridor_slice_centers()). 'll'은 ll을 흰선/노란선으로 분리해
    (_split_ll_by_yellow()) 노란 중앙선 + 한쪽 흰색 경계선을 추적한다
    (_ll_yellow_white_centers()). 'da' 모드만 여전히 calc_center()를 호출하지 않고 detect()
    안에서 직접 조립하는 예전 구조를 따르고, 'll_da'/'ll'은 각자 자기 완결적인 파이프라인
    이다. da 파편화 대응(_largest_da_component())/옆 차선 클리핑(_clip_da_by_ll())은
    'da'/'ll' 두 모드에서만 쓰인다('ll_da'=corridor는 건너뜀). 프레임 단위 ll sanity
    check는 모드와 무관하게 항상 적용된다(모듈 상단 주석 참고).
    """

    def __init__(self):
        super().__init__(
            n_slices=DL_N_SLICES, min_pixels=DL_MIN_PIXELS,
            near_slices=DL_NEAR_SLICES, far_slices=DL_FAR_SLICES,
            slice_outlier_max=DL_SLICE_OUTLIER_MAX, slice_fit_min=DL_SLICE_FIT_MIN,
            stable_frame_min=DL_STABLE_FRAME_MIN, stable_jump_max=DL_STABLE_JUMP_MAX,
        )
        self.ll_coverage = 0.0     # 최근 프레임 ROI 내 ll foreground 비율(sanity check/디버그용)
        # [2026-08-17n] 경로(self.path) 갱신 허용 여부 전용 디바운스 상태 — 아래
        # _debounce_path_ok() 참고. self.stable_frame_min(=DL_STABLE_FRAME_MIN, __init__
        # 인자로 이미 받음)을 그대로 재사용해 offset 디바운스(self._confirmed 등)와 같은
        # "N프레임 연속 확인" 강도를 쓴다.
        self._path_ok_confirmed = None
        self._path_ok_pending = None
        self._path_ok_pending_count = 0
        self.path_ok = False
        self.da_mask_roi = None    # 시각화용(가장 큰 덩어리만 남긴 이후의 da 마스크 — 실제 waypoint 추출에 쓰는 것)
        self.da_mask_all_roi = None  # 시각화용(이진화 직후, 덩어리 선택/ll클리핑 전 da 전체 — visualize()가 파란색으로 그림)
        self.ll_mask_roi = None    # 시각화용
        self.vis = None            # 시각화용 결과 이미지
        self.centerline = []       # 밴드별 중심점(원본 관측점, 길이 self.n_slices) — DL_CENTER_MODE에 따라 da 단독 또는 ll 우선/da 폴백
        self.ll_band_used = []     # 이번 프레임 각 밴드가 ll 기반으로 채택됐는지(길이 self.n_slices bool, 'da' 모드에선 항상 전부 False) — visualize() 색상 구분용
        self.da_fallback_used = False  # 이번 프레임 da가 직전 채택 덩어리와의 근접성이 아니라 면적순위 차선책으로 골라졌는지 — visualize() 색상 구분용
        self.da_ll_clip_skipped = False  # 이번 프레임 ll 클리핑이 유효 밴드를 너무 줄여 건너뛰었는지 — visualize() 구분용
        self.avoid_hold_active = False  # [2026-08-14] 이번 프레임 avoid-hold(§2.32)로 DL_DA_SKIP_LL_CLIP을 무시하고 ll 클리핑을 강제했는지 — visualize() 구분용
        self.avoid_hold_dir_hint = 0    # [2026-08-15] 적용3 — 이번 프레임 track_drive.py가 넘긴 방향 힌트(-1/0/+1) — visualize() 구분용
        self._prev_da_chosen_area_px = 0    # [2026-08-15] 적용2 — 직전 프레임 da_chosen_area_px(급증 감지용)
        self.da_area_jump_detected = False  # [2026-08-15] 적용2 — 이번 프레임 da_chosen_area_px가 직전 대비 AVOID_HOLD_DA_AREA_JUMP_RATIO 이상 급증했는지 — DLLaneDetector가 그대로 읽어 track_drive.py에 노출
        self.da_ll_virtual_clip_used = False  # [2026-08-07] 이번 프레임 _clip_da_by_ll()이 ll/잔상 없이 가상경계(기대 차로폭)로 클리핑한 밴드가 있었는지 — visualize() 구분용
        self.da_largest_mask_roi = None  # 면적 1위 덩어리(차선책을 썼다면 그 사유가 된, 상한 초과로 버려진 덩어리) — fallback일 때 원래 색으로 같이 그리기용
        self.da_largest_area_px = 0  # 면적 1위 덩어리의 절대 픽셀 면적(채택 여부 무관) — DL_DA_MAX_AREA_PX 실측 튜닝용
        self._prev_da_centroid = None  # [2026-08-07] 직전 프레임에 채택된 da 덩어리의 중심(cx,cy) — _largest_da_component()가 이번 프레임 후보를 "가장 가까운 것"으로 고르는 기준. 무효 프레임 뒤엔 None으로 리셋(옛 위치에 계속 붙잡히지 않도록)
        self.da_chosen_area_px = 0   # 실제로 채택돼 waypoint 추출에 쓰인 덩어리의 면적(무효 프레임엔 0)
        self.da_seed_width_px = 0   # [2026-08-10] 시드 위치(ROI 최하단 중앙, 차량 위치)에서 찾은 덩어리의 bounding box 가로폭(px) — 실제 채택/면적통과 여부와 무관하게 항상 기록(시드 위치에 아무것도 없으면 0). 너비 기반 선택 로직으로 바꿀지 판단하기 위한 실측용 — 아직 판단 로직에는 안 쓰임(_debug_viz_steer() 참고)
        # [2026-08-12] DL_CENTER_MODE='da' 밴드 중심 탐색창(_da_slice_centers_windowed())
        # 전용 상태 — _ll_left/right_velocity와 동일한 원리(밴드 간 이동 속도 EMA로
        # 다음 밴드 탐색창을 미리 옮기고, 못 찾은 밴드는 그 속도로 dead-reckoning),
        # da는 좌/우 두 갈래가 아니라 중심선 "한 갈래"라 값이 하나씩만 필요하다.
        # README §2.27 참고.
        self._da_velocity = 0.0
        self._da_prev_band_x = [None] * self.n_slices
        # [2026-08-19] 근접 밴드 이상치 오판(2차 안전판) — _reject_outliers() protect_indices로
        # 근접 밴드를 검사에서 뺐어도(1차) 진짜 순간 미검출 등으로 비면, 이 카운터가
        # DL_NEAR_HOLD_MAX_FRAMES에 도달할 때까지만 _da_prev_band_x[i]로 대신 채운다.
        # detect()/_reject_outliers() 주석, config.py DL_NEAR_HOLD_MAX_FRAMES 참고.
        self._near_hold_streak = [0] * self.n_slices
        self.near_band_held = [False] * self.n_slices   # 이 프레임 그 근접 밴드가 hold로 채워졌는지 — visualize() 구분용
        self.near_band_stale = False   # 근접 밴드가 DL_NEAR_HOLD_MAX_FRAMES 넘게 안 잡혀 hold도 포기했는지 — track_drive.py가 SPEED_LANE_STALE 캡에 씀
        self._white_yellow_gap_px = DL_LL_YELLOW_GAP_INIT_PX  # [2026-08-10] 노란 중앙선-흰색 경계선 간격 러닝 추정치(px, DL_LL_ALGO='yw' 전용) — 둘 다 찾은 밴드에서 EMA 갱신, 한쪽만 찾았을 때 반대쪽 위치 추정에 씀(_ll_yellow_white_centers() 참고).
        # [2026-08-12] _ll_yellow_white_centers()(DL_LL_ALGO='yw') 밴드 간 속도예측 +
        # 프레임 간 앵커링 상태 — 아래 _ll_left/right_velocity(DL_LL_ALGO='lr')와 동일한
        # 원리를 노란선/흰선 각각에 적용한다(원래 'lr'에만 있고 'yw'엔 §2.23 탐색창
        # 확장만 있던 공백을 메움, README §2.27).
        self._yw_yellow_velocity = 0.0
        self._yw_white_velocity = 0.0
        self._yw_prev_band_yellow = [None] * self.n_slices
        self._yw_prev_band_white = [None] * self.n_slices
        self._ll_half_width = (DL_LL_WIDTH_MIN_PX + DL_LL_WIDTH_MAX_PX) / 4.0  # [2026-08-07] ll 좌/우 독립 슬라이딩 윈도우의 차로 반폭 러닝 추정치(px, DL_LL_ALGO='lr' 전용) — 양쪽 다 찾은 밴드에서 EMA 갱신, 편측만 찾았을 때 반대쪽 위치 추정에 씀(_ll_slice_centers() 참고).
        # [2026-08-10 병합] _clip_da_by_ll()의 가상경계 최후수단은 DL_LL_ALGO에 맞는 반폭
        # 추정치를 써야 하므로(위 두 값 중 어느 쪽이 실제로 갱신되고 있는지는 DL_LL_ALGO가
        # 결정) _ll_active_half_width()를 통해서만 읽는다 — README §2.19 참고.
        self._ll_decay_mask = None   # [2026-08-07] ll 잔상(decay) 누적 마스크(float32, roi shape) — detect()가 매 프레임 갱신, _clip_da_by_ll() 전용(centerline 추출엔 안 씀). None이면 첫 프레임이라 detect()에서 새로 할당.
        self.lane_side = None  # [2026-08-10] DL_CENTER_MODE='ll' 전용 — 'left'/'right'/None(아직 미판정). 근거리 노란선이 seed(차량 위치) 기준 왼쪽이면 'right'(우측차선 주행중), 오른쪽이면 'left' — _ll_yellow_white_centers() 참고
        self.ll_degraded = False  # [2026-08-10] 이번 프레임 'll' 모드가 노란/흰선 중 하나를 저신뢰 추정(간격 재구성 또는 잔상)으로 메운 밴드가 하나라도 있었는지 — track_drive.py _lane_drive()가 SPEED_LL_DEGRADED 강제용으로 읽음, _debug_viz_steer()도 표시
        self.ll_band_degraded = []  # [2026-08-10] 밴드별(길이 n_slices) 위 저신뢰 추정 여부 — visualize() 색 구분용
        self.ll_band_case = []  # [2026-08-10] 밴드별(길이 n_slices) _ll_yellow_white_centers()가 이번 밴드에 실제로 어떤 분기를 탔는지 — 'Y+W'(둘다 정상)/'Y+gap'(노란만, 흰 추정)/'2W'(노란없음, 양쪽흰선)/'1W:L'|'1W:R'(노란없음, 한쪽흰선)/'LOST'(둘다없음, 잔상). visualize()가 밴드별 텍스트로 그려서 "지금 SW가 어느 분기로 주행중인지" 실차에서 바로 보이게 함

        # [2026-08-10] DL_CENTER_MODE='ll_da'(corridor 알고리즘) 전용 상태 (모듈 상단
        # "corridor" 주석, config.py DL_CENTER_MODE 주석 참고).
        #   _corridor_prev_open_x : 밴드별(길이 n_slices) 직전 프레임에 채택한 열린 구간
        #     중심의 절대 ROI x좌표. 다음 프레임 _pick_open_run()의 prefer_x로 넘겨서,
        #     폭이 비슷한 두 열린 구간 사이를 매 프레임 오가는 flip-flop을 막는
        #     히스테리시스로 쓴다(없으면 None).
        #   corridor_bounds : 밴드별(길이 n_slices) 이번 프레임 채택된 (left_bound,
        #     right_bound) 절대 ROI x좌표 — 시각화 전용, sanity check를 통과한 밴드만
        #     채워지고 나머지는 None.
        self._corridor_prev_open_x = [None] * self.n_slices
        self.corridor_bounds = [None] * self.n_slices
        # [2026-08-12] 밴드 간 열린구간 이동 속도(px/밴드) EMA — _pick_open_run()의
        # prefer_x를 "직전 프레임 위치 그대로"가 아니라 "그 위치 + 예측 이동량"으로
        # 미리 옮겨서, 빠른 S자에서 정적 히스테리시스가 뒤처지는 걸 완화한다(da/ll에
        # 적용한 것과 동일한 원리 — README §2.27). corridor는 좌/우 두 갈래가 아니라
        # "열린 구간 하나"만 추적하므로 스칼라 하나면 된다.
        self._corridor_velocity = 0.0

        # [2026-08-10] DL_CENTER_MODE='ll'(흰선/노란선 분리) 전용 부가 정보 — 전부
        # visualize()용이고 경로/조향 계산에는 안 쓰인다(노란선은 아직 stateless 디버그
        # 표시만, 모듈 상단 "'ll'" 주석 참고).
        self.ll_white_mask_roi = None    # 흰선으로 확정된 ll 컴포넌트만 남은 마스크(_ll_yellow_white_centers()/_ll_slice_centers()의 실제 입력, DL_LL_ALGO 참고)
        self.ll_yellow_mask_roi = None   # 노란선으로 확정된 ll 컴포넌트만 남은 마스크
        self.yellow_band_centers = []    # ll_yellow_mask_roi를 밴드별 무게중심(_slice_centers(), 탐색창 없는 stateless 방식)으로 뽑은 결과 — 길이 self.n_slices
        self.ll_search_windows = []      # _ll_yellow_white_centers()(노란/흰 탐색창) 또는 _ll_slice_centers()(좌/우 탐색창)가 이번 프레임에 훑은 좌표(밴드별, DL_LL_ALGO 참고) — visualize()가 사각형으로 그림

        # [2026-08-10] _ll_slice_centers() 적응형 탐색창(속도 예측)/밴드별 프레임 간
        # 앵커링 전용 상태 (config.py DL_LL_VELOCITY_*/DL_LL_BAND_ANCHOR_ALPHA 주석 참고).
        #   _ll_left_velocity/_right_velocity : 그 사이드의 밴드 간 이동 속도(px/밴드)
        #     러닝 EMA — 다음 밴드 탐색창을 선을 따라 미리 옮기는 데 씀. 미검출 밴드가
        #     이어지는 동안에도 이 속도로 계속 dead-reckoning 이동시킨다.
        #   _ll_prev_band_left/right : 밴드별(길이 n_slices) 직전 프레임에 그 밴드에서
        #     실제로 찾은 x좌표. 없으면(아직 한 번도 못 찾음) None. 찾았을 때만 갱신되고
        #     못 찾은 프레임엔 이전 값을 그대로 들고 있는다(self._ll_half_width와 동일한
        #     관례 — 잠깐 안 보인다고 즉시 잊지 않음).
        self._ll_left_velocity = 0.0
        self._ll_right_velocity = 0.0
        self._ll_prev_band_left = [None] * self.n_slices
        self._ll_prev_band_right = [None] * self.n_slices
        self.ll_band_anchor_left = [None] * self.n_slices   # 디버그 시각화 전용 스냅샷(_ll_slice_centers() 참고)
        self.ll_band_anchor_right = [None] * self.n_slices
        self.offset_sparkline_img = None  # [2026-08-11] show_debug_windows()가 'dl_lane' 창 맨 아래에 같이 붙이는 offset 스파크라인(visualize()가 매 프레임 다시 그림). 예전엔 여기에 파라미터 텍스트 패널도 같이 들어있었으나 삭제(아래 _offset_history 주석 참고)

        # [2026-08-10] 디버그 전용 — "이 밴드가 왜 이렇게 채택/거부됐는지" 근거를 그때
        # 그때 계산만 하고 버리지 않고 남겨서 visualize()가 사람이 읽을 수 있는 태그로
        # 보여준다. 알고리즘 자체(_ll_slice_centers()/_clip_da_by_ll())의 판단 결과에는
        # 전혀 영향 없음.
        self.ll_band_reason = [None] * self.n_slices     # 'B'=양쪽검출/채택 'X'=양쪽검출됐지만폭이상해거부 'L'=왼쪽만 'R'=오른쪽만 '-'=둘다없음. DL_CENTER_MODE='ll' 전용(_ll_slice_centers()가 채움)
        self.da_clip_band_virtual = [None] * self.n_slices  # True=이 밴드는 _clip_da_by_ll()이 가상경계(②)로 잘랐음, False=실측/잔상 ll(①)로 잘랐음. 'da'/'ll' 공통(_clip_da_by_ll()이 채움, 'll_da'=corridor는 클리핑을 안 하므로 항상 None)
        # [2026-08-17] avoid-hold ll클리핑이 밴드별로 실제 몇 px를 잘라냈는지 실차에서
        # 바로 확인할 수 있게 — da_clip_band_virtual(①/②중 뭘 썼는지)만으론 "발동 여부"만
        # 보이고 "얼마나 깎였는지"는 안 보여서 추가. _clip_da_by_ll()이 채움, 클리핑을
        # 건너뛴 프레임(DL_DA_SKIP_LL_CLIP=True)엔 전부 None으로 리셋됨(visualize() 참고).
        self.da_clip_cut_left_x = [None] * self.n_slices   # 이 밴드에서 왼쪽 경계로 잘라낸 x좌표(px, ROI 좌표계). None=왼쪽은 안 잘림
        self.da_clip_cut_right_x = [None] * self.n_slices  # 오른쪽 동일
        self.da_clip_bias_px = [None] * self.n_slices       # avoid-hold 적용3(AVOID_HOLD_DIR_BIAS_PX)이 가상경계 기준점을 이 밴드에서 실제로 얼마나(부호 포함, px) 밀었는지. 실측/잔상(①) 밴드는 항상 None

        # [2026-08-20] da 근접 컷(_clip_da_by_obstacle(), ENABLE_OBSTACLE_CUT) — 이번 프레임
        # 실제로 컷이 적용됐는지/어느 열(px) 범위를 잘랐는지. visualize() 오버레이 + track_drive.py
        # 디버그 창(_debug_viz_obstacle_cut())이 getattr(self.lane_detector._slide, ...)로 조회.
        self.obstacle_cut_active = False
        self.obstacle_cut_col_range = None

        # [2026-08-10] 최근 DL_DEBUG_HISTORY_LEN 프레임의 offset(디바운스 이후 최종값)을
        # 들고 있다가 [2026-08-11] 'dl_lane' 창 맨 아래에 스파크라인으로 그린다(예전엔
        # 별도 'dl_lane_params' 창이었음) — README §2.12에서
        # 문제됐던 "S자로 좌우 왔다갔다" 같은 프레임 간 흔들림은 순간값 텍스트만으론
        # 눈으로 판단하기 어려워서, 최근 추세를 한눈에 보려는 목적. deque(maxlen=...)라
        # 오래된 값은 자동으로 밀려나 별도 정리 로직이 필요 없다.
        self._offset_history = deque(maxlen=DL_DEBUG_HISTORY_LEN)

        # DL_USE_BEV=True일 때만 쓰는 워프 행렬. 상수라 매 프레임 다시 안 만들고 한 번만 계산.
        self._bev_M = (
            cv2.getPerspectiveTransform(DL_BEV_SRC_PX, DL_BEV_DST_PX)
            if DL_USE_BEV else None
        )

    def _bev_warp(self, roi_img, nearest=False):
        """cropped ROI(da/ll 확률맵 float32, yellow 이진마스크, 디버그용 raw_bgr) 하나를
        DL_BEV_CANVAS 크기로 원근변환한다. da/ll은 이진화 전(float 확률맵) 상태로 넣어야
        경계가 계단처럼 뭉개지지 않는다(INTER_LINEAR로 워프 후 detect()에서 이진화).
        이미 이진값인 마스크(yellow)는 중간값이 생기지 않도록 nearest=True로 호출한다.
        classic_cv(CameraProcessor)와 달리 워프 전 소스를 사다리꼴 밖으로 미리 마스킹하지
        않는다 — da/ll은 이미 세그멘테이션으로 배경이 지워진 깨끗한 마스크라 classic_cv가
        겪은 "배경 텍스처가 대각선으로 늘어붙는" 문제(원본 컬러/엣지 기반이라 생김)가 원천적으로
        없고, 오히려 사다리꼴(두 백선 사이)만 남기면 백선 바깥 도로까지 포함해야 하는
        da/ll 정보를 불필요하게 잘라내게 된다."""
        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.warpPerspective(
            roi_img, self._bev_M, (DL_BEV_CANVAS_W, DL_BEV_CANVAS_H),
            flags=flags, borderValue=0
        )

    def _largest_da_component(self, da_mask):
        """da 마스크에서 연결 덩어리 하나만 남기고 나머지(급커브 등에서 생기는 파편)는
        지운다. 덩어리가 DL_DA_MIN_COMPONENT_AREA보다 작으면(사실상 안 보임) 빈 마스크를
        반환한다.

        [2026-08-10] 예전엔 가장 큰 덩어리가 DL_DA_MAX_AREA_PX(절대 픽셀수)를 넘으면
        outlier로 버리고 다음 후보를 시도했다(ㅓ교차로에서 옆 갈래까지 붙는 경우, 맨바닥
        오검출 등을 "비정상적으로 넓다"는 신호로 걸러내려는 목적). **이 상한을 완전히
        없앴다** — 실차 검증 결과 면적 임계값만으로 da를 판단하는 방식 자체가 신뢰할 수
        없었고(파라미터를 계속 바꿔봐도 개선 안 됨), 지금은 아래 ①시드/②연속성이 "어느
        덩어리가 내 차선인가"를 면적과 무관하게 물리적/시간적 근거로 판단하므로 굳이
        크기로 다시 거를 필요가 없다는 판단.
          ★ 대가: da가 옆 차선과 실제로 이어붙었는데 그 융합된 덩어리가 시드 위치와
          맞닿아 있으면, 이 함수는 더 이상 그걸 막지 않는다 ★ — 그 방어는 이제 전적으로
          `_clip_da_by_ll()`(ll 잔상 + 가상경계, §2.14)이 담당한다. da 선택 단계에서
          면적으로 다시 거르지 않는 대신, 클리핑 단계에서 옆 차선 쪽을 잘라내는 구조로
          역할을 옮긴 것.

        [2026-08-06] 하한(DL_DA_MIN_COMPONENT_AREA)은 그대로 유지한다 — 이건 "비정상적으로
        크다"가 아니라 "사실상 안 보인다"를 걸러내는 노이즈 필터라 성격이 다르다.

        [2026-08-07] 차선책을 "면적 내림차순"만으로 고르면, 실제로는 계속 같은 차선을
        보고 있는데도 두 덩어리 크기가 비슷해 프레임마다 순위가 뒤집히는 것만으로
        채택 대상이 바뀌어 지금 따라가던 경로가 불필요하게 흔들리는 문제가 있었다
        (실측 재현됨). 그래서 순위보다 "연속성"을 우선한다 — self._prev_da_centroid
        (직전 프레임에 실제로 채택된 덩어리의 중심)와 가장 가까운 덩어리를 최우선
        후보로 고정하고, 그 후보의 면적이 DL_DA_MIN_COMPONENT_AREA 이상이면 순위와
        무관하게 바로 채택한다. 무효 프레임(빈 마스크 반환) 뒤에는
        self._prev_da_centroid를 None으로 리셋해, 한참 뒤에 엉뚱한 위치의 덩어리가
        "옛 중심과 가장 가깝다"는 이유만으로 잘못 이어붙는 것을 막는다.

        [2026-08-07] 위 "직전 중심과 가장 가까운 덩어리"는 어디까지나 *과거 판단*에
        기대는 방식이라, 만약 직전 프레임에 이미 엉뚱한 덩어리를 채택했다면(예:
        교차로에서 다른 갈래로 잘못 넘어감) 그 뒤로도 "그때 그 위치와 가장 가깝다"는
        이유만으로 계속 틀린 채로 이어질 수 있다(드리프트가 스스로 교정되지 않음).
        이를 보강하기 위해 *이번 프레임만 놓고 봐도 검증 가능한* 물리적 신호를
        최우선으로 추가했다 — "차량이 실제로 서 있는 위치"(ROI 최하단 중앙, 카메라/BEV
        캔버스가 차량 중심선에 맞춰 캘리브레이션돼 있다는 전제)와 실제로 맞닿은 덩어리가
        있으면 그걸 무조건 최우선으로 채택한다(`cv2.floodFill`을 새로 돌릴 필요 없이,
        이미 계산된 `labels`에서 시드 영역의 라벨만 조회하면 된다 — CCL 결과 재사용).
        이 신호는 매 프레임 독립적으로 "차와 물리적으로 붙어있는가"만 보므로, 직전
        프레임의 오판에 영향받지 않고 스스로 교정된다. 시드 위치에 유효한 덩어리가
        없을 때만(근거리가 가려짐 등) 기존 연속성→면적순위 순서로 넘어간다."""
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(da_mask, connectivity=8)
        self.da_fallback_used = False
        self.da_largest_mask_roi = None
        self.da_largest_area_px = 0
        self.da_chosen_area_px = 0
        self.da_seed_width_px = 0
        if num <= 1:
            self._prev_da_centroid = None
            return np.zeros_like(da_mask)

        areas = stats[1:, cv2.CC_STAT_AREA]
        comp_centroids = centroids[1:]  # centroids[0]은 배경 — stats[1:]와 동일하게 인덱스 정렬
        order = np.argsort(areas)[::-1]  # 큰 덩어리부터 — 차선책(연속성 후보 탈락 시) 폴백용
        min_area = DL_DA_MIN_COMPONENT_AREA

        largest_label = 1 + int(order[0])
        self.da_largest_mask_roi = np.where(labels == largest_label, np.uint8(255), np.uint8(0))
        # [2026-08-06] 실측 튜닝용 — 가장 큰 덩어리 면적(채택 여부와 무관)을 항상 기록해둔다.
        # DEBUG_VIZ_STEER 창(track_drive.py _debug_viz_steer())이 이 값을 그대로 읽어서
        # "직선 구간 da 면적이 실제로 몇 px인지" 실측하는 용도로 쓴다.
        self.da_largest_area_px = int(areas[order[0]])

        def _choose(idx, fallback):
            label = 1 + int(idx)
            self.da_fallback_used = fallback
            self.da_chosen_area_px = int(areas[idx])  # 실제로 채택돼 waypoint 추출에 쓰이는 면적
            self._prev_da_centroid = (float(comp_centroids[idx][0]), float(comp_centroids[idx][1]))
            return np.where(labels == label, np.uint8(255), np.uint8(0))

        # ① 시드(seed) 기반 최우선 후보 — ROI 최하단 중앙(차량 위치)과 물리적으로
        # 맞닿은 덩어리가 있으면 과거 판단(연속성/면적순위)보다 우선 채택한다.
        h, w = da_mask.shape
        seed_x = w / 2.0
        sy0 = max(0, h - DL_DA_SEED_ROWS_PX)
        sx0 = int(np.clip(seed_x - DL_DA_SEED_HALF_WIDTH_PX, 0, w))
        sx1 = int(np.clip(seed_x + DL_DA_SEED_HALF_WIDTH_PX, 0, w))
        if sx1 > sx0:
            seed_fg = labels[sy0:h, sx0:sx1]
            seed_fg = seed_fg[seed_fg > 0]
            if seed_fg.size:
                seed_vals, seed_counts = np.unique(seed_fg, return_counts=True)
                seed_idx = int(seed_vals[np.argmax(seed_counts)]) - 1  # 시드 영역에서 가장 많이 나온 라벨
                seed_area = int(areas[seed_idx])
                # [2026-08-10] 실측용 — 시드에 걸린 덩어리의 bounding box 가로폭을 채택/면적
                # 통과 여부와 무관하게 항상 기록해둔다. 너비 기반 판단(면적 대신, 팀원
                # lrkdms의 4a6bff6 착안 — 면적은 "가로 폭"과 "세로 길이"가 뭉뚱그려져서
                # 옆 차선 융합 같은 순수 가로 방향 문제를 잘 못 잡는다는 지적)으로 바꿀지
                # 결정하기 전에, 실제 값 분포부터 관찰하려는 목적 — 아직 선택 로직에는
                # 전혀 안 쓴다. _debug_viz_steer()(steer_debug 창)가 이 값을 그대로 읽는다.
                self.da_seed_width_px = int(stats[seed_idx + 1, cv2.CC_STAT_WIDTH])
                if seed_area >= min_area:
                    return _choose(seed_idx, fallback=False)
                # 시드 위치 덩어리가 너무 작음(사실상 노이즈) — 아래 ②(연속성)/③(면적순위)로 이동

        # ② 직전에 채택한 덩어리가 있으면 그 중심과 가장 가까운 덩어리를 다음 우선
        # 후보로 고정(연속성 유지) — 최소 면적만 넘으면 순위와 무관하게 바로 채택한다.
        if self._prev_da_centroid is not None:
            px, py = self._prev_da_centroid
            dists = np.hypot(comp_centroids[:, 0] - px, comp_centroids[:, 1] - py)
            nearest_idx = int(np.argmin(dists))
            nearest_area = int(areas[nearest_idx])
            if nearest_area >= min_area:
                return _choose(nearest_idx, fallback=False)
            # 근접 후보가 너무 작음 — 아래 ③(면적순위 차선책)으로 이동

        # ③ 시드/연속성 둘 다 못 쓴 경우의 최후 폴백 — 면적 내림차순(사실상 1위 그대로)
        if int(areas[order[0]]) >= min_area:
            fallback = self._prev_da_centroid is not None
            return _choose(int(order[0]), fallback=fallback)

        self._prev_da_centroid = None
        return np.zeros_like(da_mask)

    def _apply_vehicle_margin(self, da_mask, v_mps=0.0):
        """[2026-08-14] da 마스크를 차폭(VEHICLE_WIDTH_M)+여유(DL_DA_VEHICLE_MARGIN_M)만큼
        침식(erosion)해서, 중심선이 da 경계(장애물이든 트랙 벽이든 무관)에서 최소 이만큼은
        떨어지도록 강제한다 — ROS2 Nav2의 costmap inflation과 같은 개념(README §2.30 "da
        안전마진 설계 논의"). 지금까지 중심선 계산(`_da_slice_centers_windowed()` 등)은
        차량을 폭 0인 점으로 취급해왔는데, 이게 "장애물 회피 중 앞코가 장애물 뒷꽁지를
        긁는다"는 실차 보고의 원인으로 지목됐다.

        [2026-08-17g] v_mps(현재 속도, m/s)가 높을수록 세로(진행방향, "뒤" 포함) 반경을
        추가로 늘린다 — `_dl_da_margin_kernel()`/config.py DL_DA_REAR_MARGIN_* 참고. 좌우
        반경은 속도와 무관하게 그대로다.

        ★주의★ 이건 순수 카메라 픽셀(DL_PIXELS_PER_METER, config.py에 "설계값(실측 아님)"
        이라고 명시된 값) 기반 근사치다 — 라이다 실측으로 대신하는 안(같은 README 절)이
        더 정확하지만 라이다-카메라 오프셋 실측이 먼저 필요해 시간이 걸리므로, 우선 이
        단순한 안을 실차에서 먼저 테스트해본다(요청 반영, 안 되면 수정 예정).

        침식으로 da가 통째로 비면(좁은 커브 등에서 과침식) 원본을 그대로 반환한다 —
        `_largest_da_component()`의 "차선책" 폴백들과 같은 원칙: 마진 없이라도 주행하는
        게 프레임이 무효 처리돼 멈추는 것보다 낫다는 판단."""
        kernel = _dl_da_margin_kernel(v_mps)
        if kernel is None:
            return da_mask
        eroded = cv2.erode(da_mask, kernel)
        return eroded if np.any(eroded) else da_mask

    def _da_slice_centers_windowed(self, da_mask, ref_x):
        """[2026-08-12] DL_CENTER_MODE='da' 밴드 중심 계산 — 탐색창(prior) + 밴드 간
        속도예측 + 프레임 간 앵커링. `_slice_centers()`(무보정 cv2.moments, 밴드 전체
        폭)를 대체한다.

        **왜 필요한가**: `_largest_da_component()`(시드/연속성, 위)는 "어느 덩어리를
        볼지"만 정한다 — 그 덩어리가 옆 차선/여백까지 과검출로 넓어져도(S자 커브에서
        특히 잦음, README §2.1/§2.2) 막지 않는다(§2.16에서 면적 상한 자체를 없앴으므로
        더더욱). `_slice_centers()`는 그 넓은 덩어리의 밴드 전체 폭을 그대로
        `cv2.moments`에 넣으므로, 과검출된 영역이 무게중심을 그쪽으로 끌어당긴다.
        Mobileye(클로소이드+칼만 — 과거 상태로 예측/추적), openpilot(프레임 간 hidden
        state), drivable-area 연구(공간 prior로 억제)가 공통으로 쓰는 "탐색을 예측
        위치 근방으로 제한" 아이디어를, 이미 `_ll_slice_centers()`(DL_LL_ALGO='lr',
        §2.20)가 검증 방향을 잡아둔 그대로 da에도 적용한다 — da는 좌/우 두 갈래가
        아니라 중심선 "한 갈래"라 더 단순하다.

        밴드마다: ①직전 프레임 그 밴드 위치(`self._da_prev_band_x[i]`)와 이번 프레임
        전파값을 `DL_DA_BAND_ANCHOR_ALPHA`로 가중평균해 탐색창 중심(anchor)을 잡고,
        ②`DL_DA_SEARCH_HALF_WIDTH_PX`(연속 미검출 시 `DL_DA_SEARCH_WIDEN_STEP_PX`씩
        확장, 상한 `DL_DA_SEARCH_WIDEN_MAX_PX`) 반경 안에서만 `cv2.moments`를 구한다
        — 창 밖(과검출된 옆차선 등) 픽셀은 애초에 무게중심 계산에 안 들어간다.
        찾으면 밴드 간 이동량으로 속도 EMA(`DL_DA_VELOCITY_EMA_ALPHA`, 클램프
        `DL_DA_VELOCITY_MAX_PX`)를 갱신하고 다음 밴드 기준을 "찾은 위치+속도"로 미리
        옮긴다. 못 찾으면 그 속도로 dead-reckoning만 하고 그 밴드는 무효.
        픽셀수 임계값은 da 전용 `DL_MIN_PIXELS`를 그대로 재사용한다(da 전체 밴드에
        쓰던 것과 동일 기준 — 창이 좁아졌다고 임계값 자체를 낮추면 안 됨).

          입력 : da_mask — (roi_h, roi_w) uint8 이진마스크(_largest_da_component() 통과 후,
                 옆 차선 클리핑 전/후 무관하게 동일 인터페이스)
                 ref_x   — 첫(근거리) 밴드의 탐색 기준 x좌표. 보통 직전 프레임 lane_center.
          출력 : centers — 길이 self.n_slices. 채택되면 (y_center, cx), 아니면 None
                 (`_reject_outliers()`가 그대로 이어받을 수 있게 `_slice_centers()`와
                 동일한 반환 형식).

        알려진 한계(실차 미검증):
        - 초기 창(`DL_DA_SEARCH_HALF_WIDTH_PX=100`)이 실제 도로 폭보다 좁으면 정상
          코너까지 놓칠 수 있다 — `DEBUG_VIZ_DL_LANE`에서 검출 밴드 수(`ll_bands`류와
          별개로, `centerline`에서 None이 아닌 개수)가 이전보다 줄면 이 값을 키울 것.
        - `DL_DA_BAND_ANCHOR_ALPHA`가 "도로 곡률이 프레임 간 급격히 안 변한다"는
          가정에 기대므로, 급조향 중이거나 프레임레이트가 낮으면 과거 위치로 창을
          잘못 당길 수 있다(`_ll_slice_centers()`와 동일한 한계).
        - `_largest_da_component()`가 매 프레임 다른 덩어리를 고르면(연속성이 깨지는
          경우) 이 창의 앵커/속도도 같이 흔들릴 수 있다 — 덩어리 선택 자체의 안정성이
          이 창의 안정성의 전제 조건이다."""
        h, w = da_mask.shape
        slice_h = h // self.n_slices
        centers = [None] * self.n_slices

        cur_x = ref_x
        base_win = DL_DA_SEARCH_HALF_WIDTH_PX
        miss_streak = 0
        last_i = last_x = None

        for i in range(self.n_slices):
            y_high = h - i * slice_h
            y_low = 0 if i == self.n_slices - 1 else h - (i + 1) * slice_h
            y_center = (y_low + y_high) / 2.0

            prev_x = self._da_prev_band_x[i]
            anchor_x = (
                cur_x if prev_x is None else
                (1 - DL_DA_BAND_ANCHOR_ALPHA) * cur_x + DL_DA_BAND_ANCHOR_ALPHA * prev_x
            )
            win = min(base_win + miss_streak * DL_DA_SEARCH_WIDEN_STEP_PX, DL_DA_SEARCH_WIDEN_MAX_PX)
            x0, x1 = int(np.clip(anchor_x - win, 0, w)), int(np.clip(anchor_x + win, 0, w))

            cx = None
            if x1 > x0:
                M = cv2.moments(da_mask[y_low:y_high, x0:x1], binaryImage=True)
                if M['m00'] >= self.min_pixels:
                    cx = x0 + M['m10'] / M['m00']

            if cx is not None:
                if last_i is not None and i > last_i:
                    raw_v = (cx - last_x) / (i - last_i)
                    raw_v = float(np.clip(raw_v, -DL_DA_VELOCITY_MAX_PX, DL_DA_VELOCITY_MAX_PX))
                    a = DL_DA_VELOCITY_EMA_ALPHA
                    self._da_velocity = (1 - a) * self._da_velocity + a * raw_v
                last_i, last_x = i, cx
                self._da_prev_band_x[i] = cx
                centers[i] = (y_center, cx)
                cur_x = cx + self._da_velocity
                miss_streak = 0
            else:
                cur_x = cur_x + self._da_velocity
                miss_streak += 1

        return centers

    def _ll_active_half_width(self):
        """[2026-08-10 병합] DL_LL_ALGO에 따라 실제로 갱신되고 있는 차로 반폭 러닝
        추정치를 골라 반환한다 — 'yw'(팀원 작성)는 self._white_yellow_gap_px를,
        'lr'(이지유 작성)는 self._ll_half_width를 각각 자기 알고리즘 안에서만 EMA
        갱신하므로, 둘 다 살리기로 한 이상(README §2.19) _clip_da_by_ll()의 가상경계
        최후수단처럼 "지금 모드에서 신뢰할 수 있는 반폭"이 필요한 공용 소비처는 이
        헬퍼 하나만 거치게 한다 — 안 쓰는 쪽 알고리즘의 값(초기값에 멈춰있거나 갱신
        안 됨)을 잘못 참조하는 실수를 막기 위함."""
        return self._white_yellow_gap_px if DL_LL_ALGO == 'yw' else self._ll_half_width

    def _clip_da_by_ll(self, da_mask, ll_mask, ref_x, direction_hint=0):
        """da_mask에서 ll(차선) 라인을 경계로 "내 차선 바깥"에 해당하는 픽셀을 지운다.
        da는 점선 틈으로 옆 차선 da와 하나의 덩어리로 이어붙는 실패모드가 있는데, 이 경우
        _largest_da_component()의 "가장 큰 덩어리" 기준만으로는 옆 차선까지 통째로 살아남는다.
        ll(차선 마킹)이 실제로 보이는 한 그 선을 경계로 반대편을 잘라내면 이 문제를 막을 수
        있다.
        밴드(_slice_centers와 동일한 n_slices 분할)마다 독립적으로 자른다 — 한 번에 직선
        기준으로 자르면 커브에서 밴드마다 달라지는 ll 위치를 못 따라간다. ref_x는 "내 차선이
        어디쯤인가"의 기준점으로, 근거리(아래) 밴드부터 원거리(위) 밴드로 올라가며 방금
        잘라낸 da 밴드의 실제 중심으로 갱신한다(커브를 따라 기준점도 같이 휘어지게).
          ★ cur_ref는 아래 ①(실측 또는 잔상) 클리핑을 실제로 했을 때만 갱신한다 ★
          — 근거 없이 다음 밴드 기준을 흔들지 않기 위함(오염 전파 방지, 아래 참고).

        [2026-08-07] 실차 캡처(전체 프레임 ll_cov=0.022, ll_bands=0/8)로 확인된 실패모드:
        ll이 프레임 전체에서 거의 안 보이는 구간에서는 da 자체도 두 차선을 구분하는
        내부 경계 없이 뭉텅하게 하나로 나온다("얇은 목으로 이어붙는다"는 기존 가정과
        달리, 침식(erosion)으로 끊을 만한 구조 자체가 da 마스크 안에 없었다 — da 모델이
        그 프레임에서 애초에 두 차선을 시각적으로 구분 못 한 것). 이 경우 매달릴 수 있는
        근거가 이 프레임엔 전혀 없으므로, 두 단계 방어를 추가했다:
        ① `ll_mask` 인자 자체를 호출부(detect())에서 "잔상(decay)" 처리된 마스크로 바꿔
           받는다 — 최근 몇 프레임 동안 확실했던 ll 픽셀을 감쇠 가중치로 들고 있다가 이번
           프레임 ll이 비어도 그 잔상을 여기서는 여전히 "보이는 것"처럼 쓴다(자세한 감쇠
           로직은 DL_LL_DECAY_ALPHA 주석, detect() 참고). 이 함수 자체는 마스크가 이번
           프레임 실측인지 잔상인지 모르고 그냥 받은 대로 쓴다 — 자연스럽게 재사용된다.
        ② 잔상마저 없는 밴드(ll_cols가 완전히 빔)는 최후 수단으로 **증거 없이** 기대
           차로 반폭(self._ll_active_half_width() — DL_LL_ALGO='yw'면
           self._white_yellow_gap_px, 'lr'이면 self._ll_half_width를 재사용, 둘 다
           각자 알고리즘이 관리하는 러닝 추정치)만큼 cur_ref 양옆을 강제로 자른다
           ("가상 경계"). 픽셀 근거는 없지만
           "차로폭은 대략 이 정도"라는 기하학적 사전지식이, 무근거 병합(da가 옆 차선까지
           안 잘린 채 남는 것)보다는 안전하다는 판단이다. ①(실측/잔상 클리핑)과 달리
           cur_ref는 갱신하지 않는다 — 실측 근거 없는 추정을 다음 밴드로 계속 누적시키지
           않기 위해서다.
        classic_cv 백엔드의 "한쪽 차선만 검출" 폴백(lane_util.SlideWindow.calc_center())과
        같은 "차로폭 기반 추정" 원칙을 여기 클리핑에도 적용한 것.

          입력 : da_mask — (roi_h, roi_w) uint8 이진마스크
                 ll_mask — 동일 shape 이진마스크. 호출부가 실측/잔상 어느 쪽을 넣어도 무방.
                 ref_x   — 첫(근거리) 밴드의 기준 x좌표. 보통 직전 프레임 lane_center.
                 direction_hint — [2026-08-15] avoid-hold 적용3(config.py AVOID_HOLD_DIR_BIAS_PX
                   주석). -1/0/+1, lane_offset과 동일한 "우측+" 부호규약. self.avoid_hold_active
                   이고 0이 아닐 때만, 아래 ②(가상경계 — 실측/잔상이 전혀 없는 최후수단)
                   분기에서 기준점을 이 방향으로 AVOID_HOLD_DIR_BIAS_PX만큼 미리 기울인다.
                   ①(실측/잔상 있음) 분기는 건드리지 않는다 — 실제 증거가 항상 힌트보다
                   우선한다(문제2 대비책, 실측 없이 방향만으로 결정하지 않음).
          출력 : (clipped, virtual_used) — clipped는 da_mask에서 ll(또는 가상) 경계 밖
                 픽셀만 0으로 지운 복사본(shape 동일), virtual_used는 이번 호출에서 ②
                 (가상경계)가 한 밴드라도 발동했는지(bool) — visualize() 디버그 표시용.
        """
        h, w = da_mask.shape
        slice_h = h // self.n_slices
        clipped = da_mask.copy()
        cur_ref = ref_x
        virtual_used = False  # 이번 호출에서 ②(가상경계)가 한 밴드라도 발동했는지 — visualize() 디버그 표시용
        # [2026-08-10] 밴드별로 ①(실측/잔상)/②(가상경계) 중 뭐가 발동했는지 기록 —
        # virtual_used(프레임 전체 요약)만으론 "몇 번째 밴드가 가상경계였는지"를 알 수
        # 없어서, visualize()가 밴드별 틱으로 표시할 수 있게 여기서 채운다. 알고리즘
        # 판단에는 안 쓰이는 순수 디버그 부가정보.
        self.da_clip_band_virtual = [None] * self.n_slices
        # [2026-08-17] 밴드별 실제 컷 위치(px) — visualize()가 "①/②중 뭘 썼는지"뿐 아니라
        # "그래서 몇 px가 잘렸는지"까지 보여줄 수 있게 매 호출 리셋 후 채운다.
        self.da_clip_cut_left_x = [None] * self.n_slices
        self.da_clip_cut_right_x = [None] * self.n_slices
        self.da_clip_bias_px = [None] * self.n_slices

        for i in range(self.n_slices):
            y_high = h - i * slice_h
            y_low = 0 if i == self.n_slices - 1 else h - (i + 1) * slice_h

            ll_cols = np.nonzero(np.any(ll_mask[y_low:y_high, :] > 0, axis=0))[0]
            if ll_cols.size:
                left_cols = ll_cols[ll_cols < cur_ref]
                right_cols = ll_cols[ll_cols > cur_ref]
                if left_cols.size:
                    cut = min(w, int(left_cols.max()) + DL_LL_CLIP_MARGIN_PX)
                    clipped[y_low:y_high, :cut] = 0
                    self.da_clip_cut_left_x[i] = cut
                if right_cols.size:
                    cut = max(0, int(right_cols.min()) - DL_LL_CLIP_MARGIN_PX)
                    clipped[y_low:y_high, cut:] = 0
                    self.da_clip_cut_right_x[i] = cut

                band_da_cols = np.nonzero(np.any(clipped[y_low:y_high, :] > 0, axis=0))[0]
                if band_da_cols.size:
                    cur_ref = float(np.mean(band_da_cols))
                self.da_clip_band_virtual[i] = False
            else:
                # ② ll도 잔상도 없음 — 최후 수단: 기대 차로 반폭 기준 가상 경계로 강제 클리핑.
                half_width = self._ll_active_half_width()
                # [2026-08-15] avoid-hold 적용3 — 실측 근거가 전혀 없는 이 최후수단에서만,
                # 방향 힌트가 있으면(avoid_hold 활성 중 + direction_hint != 0) 기준점을
                # 그 방향으로 미리 살짝 기울인다. 근거 없는 추정에 또 다른 근거 없는
                # 추정(힌트)을 더하는 것뿐이라 위험이 없진 않지만, 힌트 자체는 라이다
                # 실측(obstacle_y)에서 나온 값이라 "아무 근거 없는 것"보다는 낫다는 판단.
                biased_ref = cur_ref
                bias_px = 0.0
                if self.avoid_hold_active and direction_hint:
                    bias_px = direction_hint * AVOID_HOLD_DIR_BIAS_PX
                    biased_ref = cur_ref + bias_px
                lcut = int(np.clip(biased_ref - half_width, 0, w))
                rcut = int(np.clip(biased_ref + half_width, 0, w))
                clipped[y_low:y_high, :lcut] = 0
                clipped[y_low:y_high, rcut:] = 0
                self.da_clip_cut_left_x[i] = lcut
                self.da_clip_cut_right_x[i] = rcut
                if bias_px:
                    self.da_clip_bias_px[i] = bias_px
                virtual_used = True
                self.da_clip_band_virtual[i] = True
                # cur_ref는 갱신하지 않는다 — 실측 근거 없는 추정이라 그대로 다음 밴드로 넘김
                # (biased_ref로 방향만 살짝 튼 것도 마찬가지 — 다음 밴드는 원래 cur_ref 기준으로
                # 다시 판단한다).

        return clipped, virtual_used

    def _clip_da_by_obstacle(self, da_mask, obstacle_y_m, confirmed):
        """[2026-08-20] da 근접 컷(ENABLE_OBSTACLE_CUT) — `_clip_da_by_ll()`(위)의
        가상경계(②)와 `_apply_vehicle_margin()`에 이은 이 파일의 세 번째 "근거(픽셀)
        없이 강제로 da를 클리핑"하는 함수. 다만 여기서는 근거가 픽셀이 아니라
        라이다+YOLO로 확정된 외부 신호(`confirmed`)다.

        [설계 배경] 장애물/방해차량을 da 안전마진(`_apply_vehicle_margin()`, §2.30)의
        국소 침식만으로 피하게 두면 반응이 장애물 바로 앞에서만 완만하게 걸린다.
        Pure Pursuit lookahead를 늘려서 더 멀리부터 보게 하는 안도 검토했으나
        curvature=2·sin(α)/ld 공식상 ld(lookahead 거리)가 커질수록 같은 횡편차라도
        곡률 추정이 오히려 희석되는 역효과만 확인돼(README §2.5x) 폐기했다. 대신
        "차량↔장애물 사이 구간의 da를 장애물 쪽 절반만 통째로 잘라 갈림길을
        뚜렷하게 만드는" 방식으로 전환했다 — 이러면 Pure Pursuit이 평소 코너/분기를
        따라가듯 자연스럽게 이른 조향을 낸다.

        [컷의 먼 경계를 obstacle_dist로 계산하지 않는 이유] 이 함수가 호출되는
        시점엔(트리거 확정 조건, config.py OBSTACLE_CUT_TRIGGER_X_MAX_M=1.0 참고)
        장애물까지 실측거리가 항상 DL_BEV_FAR_LIMIT_M(0.7m, da BEV 캔버스의 표현
        한계)보다 가깝거나 비슷하다 — 즉 da 안에는 애초에 "장애물보다 먼 행(row)"이
        따로 존재하지 않는다. 그래서 컷의 먼 경계는 그냥 캔버스 자체의 끝(row 0,
        detect()에서 이미 DL_BEV_FAR_CROP_ROW로 크롭된 상태)으로 두고, 가까운
        경계만 차량 바로 앞 고정값(OBSTACLE_CUT_NEAR_M)으로 잡는다 — 그 사이는
        전부 컷 대상.

        ★부호규약 주의(실차 미검증, 반드시 저속에서 먼저 확인)★ obstacle_y_m은
        perc_obstacle_cut_trigger()가 라이다로 잰 값으로, 이 저장소 관례상 +가 좌측
        이다(TargetPassing.choose_side()가 "obstacle_y>0(좌측) → +1(우측 통과)"로
        쓰는 것과 동일 부호). da BEV 캔버스는 이미지 좌표계라 x가 클수록 화면
        오른쪽(=물리적 우측)이므로, 장애물이 좌측(obstacle_y_m>0)이면 캔버스에서
        작은 x(왼쪽) 절반을 잘라야 한다 — 반대로 자르면 열린 쪽이 아니라 장애물
        쪽으로 조향하게 되는 치명적 버그이니 실차 첫 테스트에서 반드시 확인할 것.

        입력 : da_mask — (roi_h, roi_w) uint8 이진마스크
               obstacle_y_m — 장애물 횡위치(m, +좌측). None이면 컷 안 함.
               confirmed — 위 트리거(라이다 AND YOLO, 디바운스 통과)가 True로
                           확정했는지. False면 그대로 반환(컷 없음).
        출력 : clipped da_mask(원본과 shape 동일). 부수효과로 self.obstacle_cut_active/
               self.obstacle_cut_col_range를 이번 프레임 상태로 갱신(visualize() 디버그용).
        """
        self.obstacle_cut_active = False
        self.obstacle_cut_col_range = None
        if not confirmed or obstacle_y_m is None:
            return da_mask

        h, w = da_mask.shape
        vehicle_x = self.vehicle_center_x
        half_width_px = (OBSTACLE_CUT_LANE_HALF_WIDTH_PX if OBSTACLE_CUT_LANE_HALF_WIDTH_PX is not None
                          else LANE_WIDTH_M * DL_PIXELS_PER_METER)
        near_row_px = int(np.clip(h - OBSTACLE_CUT_NEAR_M * DL_PIXELS_PER_METER, 0, h))

        if obstacle_y_m > 0:   # 장애물 좌측(라이다 +y=좌측) → 좌측(작은 x) 절반 클리핑
            x0, x1 = int(np.clip(vehicle_x - half_width_px, 0, w)), int(np.clip(vehicle_x, 0, w))
        else:                  # 장애물 우측 → 우측(큰 x) 절반 클리핑
            x0, x1 = int(np.clip(vehicle_x, 0, w)), int(np.clip(vehicle_x + half_width_px, 0, w))
        if x1 <= x0 or near_row_px <= 0:
            return da_mask

        # [안전장치] 클리핑 후 열린(반대) 쪽에 da가 최소폭 이상 남는지 확인 —
        # 안 남으면(da가 그 구간에서 통째로 비면) pure_pursuit.control()의
        # "path 없으면 직전 조향각 유지(held)" 폴백이 걸려, 회피가 가장 필요한
        # 순간 조향이 오히려 얼어붙는다(세션 초반에 다룬 그 문제) — 이럴 땐 컷을
        # 포기하고 원본을 그대로 둔다("차선책" 원칙, _apply_vehicle_margin()과 동일).
        open_region = da_mask[0:near_row_px, :]
        open_cols = np.nonzero(np.any(open_region > 0, axis=0))[0]
        open_cols = open_cols[(open_cols < x0) | (open_cols >= x1)]
        if open_cols.size == 0 or (open_cols.max() - open_cols.min()) < OBSTACLE_CUT_MIN_REMAIN_PX:
            return da_mask

        clipped = da_mask.copy()
        clipped[0:near_row_px, x0:x1] = 0
        self.obstacle_cut_active = True
        self.obstacle_cut_col_range = (x0, x1)
        return clipped

    def _split_ll_by_yellow(self, ll_mask, yellow_roi):
        """ll_mask(흰/노랑 구분 없는 차선 이진마스크)를 커넥티드 컴포넌트 단위로
        흰선/노란선 마스크로 나눈다. DL_CENTER_MODE='ll' 전용(_ll_yellow_white_centers()가
        노란선/흰선을 각각 따로 입력받도록 하기 위함, config.py DL_CENTER_MODE 주석 참고).

        ll 픽셀 자체엔 색 정보가 없으므로(모듈 상단 "TwinLiteNet의 ll 출력은 흰/노랑을
        구분하지 않아" 주석 참고), 픽셀 하나하나를 yellow_roi로 지우는 대신 ll의
        커넥티드 컴포넌트(점선 한 조각/실선 한 덩어리) 단위로 "이 덩어리 안에 노란
        픽셀이 DL_LL_YELLOW_VOTE_RATIO 이상 겹치는가"를 투표해 덩어리 전체를 한
        색으로 확정한다. 픽셀 단위로 빼면 dash 가장자리(HSV가 못 잡는 안티앨리어싱
        경계)가 지저분하게 흰색 잔여물로 남는데, 컴포넌트째로 넘기면 그 경계까지
        깔끔하게 갈린다.

          입력 : ll_mask    — (roi_h, roi_w) uint8 이진마스크(0/255)
                 yellow_roi — 같은 shape의 HSV 기반 노란색 이진마스크(0/255,
                              detect()에서 ll_mask와 동일한 크롭/BEV 좌표계로 이미 정렬됨)
          출력 : (ll_white, ll_yellow) — 둘 다 ll_mask와 같은 shape/dtype.
                 컴포넌트 하나는 반드시 둘 중 하나에만 속한다(합치면 ll_mask와 동일).
        """
        num, labels, stats, _ = cv2.connectedComponentsWithStats(ll_mask, connectivity=8)
        ll_white = ll_mask.copy()
        ll_yellow = np.zeros_like(ll_mask)

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < DL_LL_YELLOW_MIN_AREA:
                continue   # 너무 작은 덩어리는 투표 생략, 흰선 쪽에 그대로 둠(이후 CCA/픽셀수 필터가 처리)
            comp = (labels == i)
            yellow_overlap = np.count_nonzero(yellow_roi[comp])
            if yellow_overlap / area >= DL_LL_YELLOW_VOTE_RATIO:
                ll_yellow[comp] = 255
                ll_white[comp] = 0

        return ll_white, ll_yellow

    def _ll_line_centers(self, ll_band):
        """ll_band(2D 이진 밴드 슬라이스, _slice_centers/_clip_da_by_ll과 동일한 y_low:y_high
        구간)에서 서로 다른 물리적 선들을 구분해 x중심 좌표를 왼쪽부터 정렬해 반환한다.
        DL_CENTER_MODE='ll_da'(corridor) 전용 — _ll_yellow_white_centers()는 "노란선 +
        한쪽 흰선" 딱 2선만 다루는 반면, 이건 밴드 안의 선을 개수 제한 없이 전부 찾아 순서만
        매긴다(1번째/3번째를 고르는 건 호출측 _corridor_slice_centers()가 한다). 흰/노랑을
        가리지 않은 원본 ll_mask를 그대로 받는다 — 노란 중앙선도 "2번째 선"으로 그대로
        세야 corridor의 1/3번째 선 인덱싱이 맞는다.

        connectedComponentsWithStats로 성분을 찾고 DL_CORRIDOR_LINE_MIN_PIXELS 미만인
        작은 잡음은 버린다. 점선 틈으로 같은 선이 이 밴드 안에서 둘로 쪼개져 서로 다른
        component가 되는 경우가 있어(밴드 경계와 점선 간격이 우연히 겹칠 때), 정렬 후
        인접한 두 x가 DL_CORRIDOR_LINE_MERGE_PX보다 가까우면 하나로 합친다 — 안 그러면
        실제로는 3개인 선이 4개로 잘못 세어져 1/3번째 선택이 틀어진다.
        """
        num, _, stats, centroids = cv2.connectedComponentsWithStats(ll_band, connectivity=8)
        xs = sorted(
            float(centroids[i][0])
            for i in range(1, num)
            if stats[i, cv2.CC_STAT_AREA] >= DL_CORRIDOR_LINE_MIN_PIXELS
        )
        merged = []
        for x in xs:
            if merged and x - merged[-1] < DL_CORRIDOR_LINE_MERGE_PX:
                merged[-1] = (merged[-1] + x) / 2.0
            else:
                merged.append(x)
        return merged

    def _pick_open_run(self, open_cols, prefer_x):
        """open_cols(1D bool 배열, corridor 폭만큼의 컬럼별 "da가 있는가")에서 연속 True
        구간(run)들을 찾아 DL_CORRIDOR_MIN_PASSABLE_PX(차량 실폭 기반) 이상인 것만 후보로
        삼는다. prefer_x(band-local 좌표 — 없으면 None)가 있으면 그 값에 가장 가까운 run을
        고른다 — 이게 "장애물을 피해간 방향을 다음 프레임도 유지"하는 히스테리시스라, 폭이
        비슷한 두 열린 구간(예: 장애물 좌/우) 사이를 매 프레임 오가는 flip-flop을 막는다.
        [2026-08-12] prefer_x는 이제 호출부(_corridor_slice_centers())가 "직전 프레임 이
        밴드 값"과 "이번 프레임 밴드 간 속도예측 값"을 블렌드해서 넘긴다(README §2.27) —
        이 함수 자체는 여전히 "받은 좌표에 가장 가까운 run"만 고르는 순수 선택 로직이라
        바뀐 게 없다. prefer_x가 없으면(첫 프레임 등) 가장 넓은 run을 고른다. 통과 가능한
        run이 하나도 없으면 None(이 밴드는 완전히 막힘 — 무효 처리).
        """
        n = len(open_cols)
        runs = []
        start = None
        for x in range(n):
            if open_cols[x] and start is None:
                start = x
            elif not open_cols[x] and start is not None:
                runs.append((start, x))
                start = None
        if start is not None:
            runs.append((start, n))

        runs = [(s, e) for s, e in runs if (e - s) >= DL_CORRIDOR_MIN_PASSABLE_PX]
        if not runs:
            return None
        if prefer_x is None:
            return max(runs, key=lambda r: r[1] - r[0])
        return min(runs, key=lambda r: abs((r[0] + r[1]) / 2.0 - prefer_x))

    def _corridor_slice_centers(self, da_mask, ll_mask):
        """DL_CENTER_MODE='ll_da'(corridor 알고리즘)일 때만 호출된다(모듈 상단 주석
        참고). 밴드마다 ll에서 왼쪽부터 정렬된 선 목록을 얻어 1번째~3번째 선 사이를
        도로(전체 트랙, 양쪽 차로 폭 — 2번째 선인 중앙분리선은 그냥 지나침)로 규정하고,
        그 x범위 안에서만 da를 봐서 실제 열린(장애물 없는) 구간을 찾아 그 중심을 밴드
        중심으로 삼는다.

        밴드에서 검출된 선이 3개 미만이거나(점선 틈 등) corridor 폭이 정상범위
        (DL_CORRIDOR_WIDTH_MIN_PX~MAX_PX) 밖이면 그 밴드는 그냥 드롭한다(None) — 'da'/
        'll' 모드처럼 da 무게중심으로 대체하지 않는다: corridor 경계 자체가 ll에서
        나오므로 ll이 불충분한 순간엔 "도로 폭이 얼마인지" 판단할 근거가 없다(config.py
        DL_CENTER_MODE 주석 — 실측상 이런 밴드는 드물다는 전제).

        입력 : da_mask, ll_mask — 동일 shape의 (roi_h, roi_w) uint8 이진마스크. da_mask는
               largest-component 선택/ll클리핑을 거치지 않은 원본(self.da_mask_all_roi)이어야
               한다 — corridor는 장애물이 도로를 좌우로 쪼갤 수 있다고 보므로, 한쪽만 남기는
               largest-component/clip을 적용하면 지나갈 수 있는 쪽을 통째로 잃는다. ll_mask는
               흰/노랑을 가리지 않은 원본이어야 한다(_ll_line_centers() 참고 — 노란 중앙선도
               2번째 선으로 세야 함).
        출력 : (results, used) — 길이 self.n_slices. results[i] : 채택되면 (y_center, cx),
               아니면 None. used[i] : results[i]가 채택됐는지(bool) — 시각화용.
        부수효과 : self.corridor_bounds[i]를 sanity check를 통과한 밴드에 한해 갱신하고
               (시각화용), self._corridor_prev_open_x[i]를 이번 프레임 채택된 열린 구간
               중심(절대 좌표)으로 갱신한다(다음 프레임 히스테리시스 기준점).
        """
        h, w = da_mask.shape
        slice_h = h // self.n_slices
        results = [None] * self.n_slices
        used = [False] * self.n_slices

        # [2026-08-12] 밴드 간 속도예측(README §2.27) — 정적 히스테리시스(직전 프레임
        # 그 밴드의 값만 봄)만으로는 빠른 S자에서 실제 열린구간 위치가 그 사이 크게
        # 이동하면 뒤처진다. cur_x는 "이번 프레임 안에서 여기까지 채택해온 위치를
        # 속도만큼 앞서 예측한 값"(da/ll의 cur_x와 동일 원리), last_i/last_x는 이번
        # 프레임에서만 유효한 속도 계산용 지역 상태 — 프레임 경계를 넘어 재사용하면
        # 밴드 인덱스가 롤오버돼 음수 gap이 나온다.
        cur_x = None
        last_i = last_x = None

        for i in range(self.n_slices):
            y_high = h - i * slice_h
            y_low = 0 if i == self.n_slices - 1 else h - (i + 1) * slice_h
            y_center = (y_low + y_high) / 2.0

            ll_band = ll_mask[y_low:y_high, :]
            line_xs = self._ll_line_centers(ll_band)
            if len(line_xs) < 3:
                continue  # 선 부족 — 도로 폭 판단 근거 없음, 이 밴드는 드롭

            left_bound, right_bound = line_xs[0], line_xs[2]
            width = right_bound - left_bound
            if not (DL_CORRIDOR_WIDTH_MIN_PX < width < DL_CORRIDOR_WIDTH_MAX_PX):
                continue  # 비정상 폭 — 선 오검출/오정렬 가능성, 신뢰 불가

            self.corridor_bounds[i] = (left_bound, right_bound)

            lb = int(np.clip(left_bound, 0, w))
            rb = int(np.clip(right_bound, 0, w))
            if rb <= lb:
                continue
            da_band = da_mask[y_low:y_high, lb:rb]
            open_cols = np.any(da_band > 0, axis=0)

            # [2026-08-12] prefer_x = "직전 프레임 이 밴드 값"과 "이번 프레임 속도예측
            # 값"을 blend — 하나만 있으면 그거 그대로, 둘 다 없으면(첫 프레임 등) None
            # (기존과 동일하게 _pick_open_run()이 가장 넓은 run을 고른다). 직전 프레임
            # 값만 쓰던 예전 방식은 그 밴드가 한 번도 채택된 적 없으면(cur_x도 없는
            # 첫 시도) 방향 힌트가 전혀 없어 flip-flop에 더 취약했다.
            prev_x = self._corridor_prev_open_x[i]
            if prev_x is not None and cur_x is not None:
                predicted_x = 0.5 * prev_x + 0.5 * cur_x
            elif prev_x is not None:
                predicted_x = prev_x
            else:
                predicted_x = cur_x
            prefer_x = (predicted_x - lb) if predicted_x is not None else None
            run = self._pick_open_run(open_cols, prefer_x)
            if run is None:
                continue  # corridor 안에 지나갈 폭이 있는 열린 구간이 없음 — 완전히 막힘

            cx = lb + (run[0] + run[1]) / 2.0
            results[i] = (y_center, cx)
            used[i] = True
            self._corridor_prev_open_x[i] = cx

            if last_i is not None and i > last_i:
                raw_v = (cx - last_x) / (i - last_i)
                raw_v = float(np.clip(raw_v, -DL_CORRIDOR_VELOCITY_MAX_PX, DL_CORRIDOR_VELOCITY_MAX_PX))
                a = DL_CORRIDOR_VELOCITY_EMA_ALPHA
                self._corridor_velocity = (1 - a) * self._corridor_velocity + a * raw_v
            last_i, last_x = i, cx
            cur_x = cx + self._corridor_velocity

        return results, used

    def _ll_yellow_white_centers(self, ll_white_mask, ll_yellow_mask, ref_x):
        """[2026-08-10] DL_CENTER_MODE='ll' && DL_LL_ALGO='yw'(main 기본값)일 때만
        호출된다 — DL_LL_ALGO='lr'이면 대신 _ll_slice_centers()가 호출된다(둘 다
        살리고 config.py DL_LL_ALGO로 전환하도록 병합, README §2.19 참고). 원래
        "좌/우 흰선 두 개를 독립 추적"하던 _ll_slice_centers() 하나만 있었는데, 실제
        도로는 편도 1차로 기준 흰-노-흰 구조라 노란선이 있는 쪽엔 애초에 흰선이
        없어서(노란선은 흰선 마스크에서 이미 제외됨, _split_ll_by_yellow() 참고)
        그쪽 탐색이 거의 항상 실패하는 구조적 문제가 있었다(실차 영상에서 검출 밴드가
        계속 0~1/8이었던 원인) — 이 함수는 그 문제에 대응해 새로 작성됐다. **노란
        중앙선 + (내 차선에 맞는) 한쪽 흰색 경계선**만 추적한다.

        ① 차선 판정(self.lane_side): 근거리(가장 아래)부터 밴드를 훑다가 노란선을
        처음 찾은 밴드에서, 그 x좌표가 seed(ref_x, 차량 위치) 기준 왼쪽이면 "우측차선
        주행중"(흰 경계선은 오른쪽에서 탐색), 오른쪽이면 "좌측차선 주행중"(왼쪽에서
        탐색)으로 확정한다. 이번 프레임에 노란선을 한 번도 못 찾으면 직전 프레임의
        self.lane_side를 그대로 쓴다(둘 다 없으면 'right'를 임의 기본값으로 시작 —
        실차 미검증).

        ② 밴드마다 노란선/흰선을 각각 좁은 창(DL_LL_SEARCH_HALF_WIDTH_PX, cur_yellow/
        cur_white 중심)에서 독립적으로 찾는다. 둘 다 찾으면 그 중점을 채택하고, 이때의
        실측 간격(|흰선-노란선|)으로 self._white_yellow_gap_px(러닝 추정치,
        DL_LL_YELLOW_GAP_EMA_ALPHA로 EMA)를 갱신한다 — cur_yellow/cur_white는 각자
        실제로 찾았을 때만 갱신(독립 슬라이딩 윈도우 원칙은 유지).

        ③ 노란선만 찾고 흰선을 못 찾으면(좁은 창 기준) → 노란선 위치 + (차선 방향)×
        gap으로 흰선 위치를 추정해서 중점을 계산한다. 근거 없는 추정이므로 이 밴드는
        self.ll_band_degraded[i]=True로 표시한다. 태그 'Y+gap'.

        ④ [2026-08-10 재설계, README §2.18] 노란선을 이번 밴드에서 못 찾으면 →
        예전엔 좁은 창(cur_white 중심) 하나로 흰선 하나만 찾아 gap을 역적용했는데,
        실차 15초 지점에서 gap EMA가 노이즈로 161px까지 부푼 뒤(정상 실측치는
        80px) 노란선이 아예 안 잡히기 시작해 그 부푼 값이 그대로 얼어붙었고, 그
        값으로 흰선 위치를 무시한 채 waypoint를 실제 흰선 너머 차선 밖으로 밀어내
        급조향(우회전)으로 이어지는 게 확인됐다. 이제 넓은 창
        (DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX)에서 _ll_line_centers()로 흰선
        컴포넌트를 전부 찾아 개수로 3분기한다:
          - 케이스1(태그 '2W', 흰선 2개 이상 — 가장 왼쪽/오른쪽을 채택): 두 실측
            위치의 중점을 그대로 중앙선으로 쓴다. gap 추정치에 안 기대는 가장 신뢰
            높은 재구성이라, cur_yellow도 이 중점으로 갱신해 다음 밴드 탐색창이
            따라가게 한다.
          - 케이스2(태그 '1W:L'/'1W:R', 흰선 1개): 그 흰선이 기준점(cur_yellow)
            대비 왼쪽/오른쪽인지 **매 밴드 새로 실측 판정**한 뒤(self.lane_side/
            side_sign은 프레임당 한 번만 고정돼 stale할 수 있어 안 씀), 그 방향
            으로 gap(클램프됨, DL_LL_YELLOW_GAP_MIN/MAX_PX)만큼 차선 안쪽으로
            당겨 중앙선을 재구성한다.
          - 케이스3(태그 'LOST', 흰선 0개): 새로 추정할 근거가 전혀 없으므로 직전
            까지 추적하던 cur_yellow/cur_white 위치를 그대로 "잔상"으로 써서
            중점을 만든다(cur_* 갱신 안 함 — 아무 근거 없이 같은 값을 우려먹는
            것이므로 새 오염원이 되지 않게).

        ③④ 전부 self.ll_degraded(프레임 단위, 이번 프레임에 degraded 밴드가 하나라도
        있으면 True)를 세우고, track_drive.py _lane_drive()가 이를 보고 속도를
        SPEED_LL_DEGRADED로 강제 제한한다(요청 반영: "안 보이면 잔상 주행 + 속도 5").
        밴드별로 어느 분기를 탔는지는 self.ll_band_case(visualize()가 밴드 점 옆
        텍스트 + 상단 요약 줄로 그림)에 남는다.

        [2026-08-10] ②의 노란/흰 탐색창(DL_LL_SEARCH_HALF_WIDTH_PX)에 _ll_slice_centers()
        (DL_LL_ALGO='lr')에서 쓰던 "연속 미검출 시 반경 확장" 메커니즘을 이식했다 —
        급조향 후 직진 복귀 구간에서 ref_x(탐색창 seed)가 디바운스로 지연돼 있는 동안
        실제 위치가 좁은 창 밖으로 나가 계속 놓치는 문제(README §2.20/§2.22, 사용자
        보고) 대응. 노란/흰 각각 독립적으로 연속 미검출 횟수(yellow_miss_streak/
        white_miss_streak)를 세서 그 사이드의 탐색창 반경을 DL_LL_SEARCH_WIDEN_STEP_PX
        씩 넓히고(DL_LL_SEARCH_WIDEN_MAX_PX 상한), 다시 찾으면 기본 반경으로 리셋한다.
        특히 **흰선 쪽에 실질적 효과**가 크다 — 노란선은 이미 못 찾으면 곧바로 ④(150px
        광역 3분기)로 넘어가지만, 흰선은 노란선이 계속 잡히는 동안(②/③ 경로)
        `cur_white`가 실제로 찾았을 때만 갱신되고 못 찾으면 그 자리에 멈춰 있어서
        (③ "Y+gap" 근거 없는 추정으로 계속 빠짐) 재포착 수단이 전혀 없었다 — 이제는
        흰선 탐색창도 넓어지며 실제 위치를 다시 잡을 기회를 얻는다. 노란선 쪽은 ④가
        이미 훨씬 넓은(150px) 안전망이라 이 확장의 실효는 "아주 살짝 벗어나서 ④까지
        갈 필요 없는" 경우를 줄이는 정도다.

          입력 : ll_white_mask, ll_yellow_mask — (roi_h, roi_w) uint8 이진마스크(동일 shape)
                 ref_x — 첫(근거리) 밴드의 seed/기준 x좌표. 보통 직전 프레임 lane_center.
          출력 : (results, used) — 둘 다 길이 self.n_slices.
                 results[i] : 항상 (y_center, cx) — ②~④(케이스1~3 포함) 중 하나로
                              채워짐(cur_yellow/cur_white가 한 번도 안 잡힌 첫 프레임
                              극초반 제외).
                 used[i]    : ②(정상 검출)로 채워졌는지(bool) — 디버그 시각화용
                              (③④=degraded는 False).

        알려진 한계:
        - [2026-08-10 일부 완화] 탐색창이 좁아서(②③) 급커브에서 밴드 간 실제 선
          이동량이 반경보다 크면 추적이 끊길 수 있다(노란/흰 각각 독립) — 위 확장
          메커니즘으로 연속 미검출 시 반경이 넓어지긴 하지만, 실차 미검증이라 급커브에서
          실제로 놓치지 않고 따라가는지, 넓어진 창이 오히려 옆 차선/반사광을 잘못
          무는지는 확인 필요.
        - 케이스3(LOST)이 여러 프레임 연속되면 cur_yellow/cur_white가 실제 위치와
          점점 벌어질 수 있다 — 실차 미검증, 잔상이 몇 프레임까지 안전한지 확인할 것.
        - 케이스1에서 흰선이 3개 이상 잡히면(점선 파편/노이즈) 가장 왼쪽/오른쪽만
          채택하는데, 그중 하나가 실제로는 옆 차선/노이즈일 경우 중점이 틀어질 수
          있다 — 아직 폭 sanity check 없음.
        - ③번(노란선만 찾음)은 여전히 self.lane_side/side_sign(프레임당 한 번 고정)에
          기대므로, lane_side 오판 시(예: 교차로) 이 분기만 틀어질 수 있다."""
        h, w = ll_white_mask.shape
        slice_h = h // self.n_slices
        results = [None] * self.n_slices
        used = [False] * self.n_slices
        self.ll_band_degraded = [False] * self.n_slices
        self.ll_band_case = ['?'] * self.n_slices  # visualize()가 밴드별로 그림 — "지금 SW가 어느 분기로 주행중인지"
        # 디버그 시각화용 — 밴드별 노란/흰 탐색창 좌표 + 실제로 찾았는지.
        # 알고리즘 자체엔 전혀 쓰이지 않는 부가 정보.
        self.ll_search_windows = []

        base_win = DL_LL_SEARCH_HALF_WIDTH_PX
        cur_yellow = ref_x
        # side_sign: +1이면 흰 경계선이 노란선보다 오른쪽(=나는 우측차선 주행중),
        # -1이면 왼쪽(=좌측차선). 직전 프레임 확정값을 시작점으로 쓰고, 이번 프레임
        # 근거리에서 노란선을 새로 찾으면 그걸로 갱신한다.
        side_sign = -1.0 if self.lane_side == 'left' else 1.0
        cur_white = ref_x + side_sign * self._white_yellow_gap_px
        lane_side_locked = False
        # [2026-08-10] 탐색창 확장(위 docstring 참고) — 노란/흰 각각 독립적으로 연속
        # 미검출 횟수를 센다. 찾으면 0으로 리셋, 못 찾으면 +1 → 다음 밴드 그 사이드
        # 탐색창 반경이 그만큼 넓어진다.
        yellow_miss_streak = 0
        white_miss_streak = 0
        # [2026-08-12] 밴드 간 속도예측용 — 이번 프레임 안에서만 유효한 "마지막으로
        # 실제 찾은 밴드" 기록(_ll_slice_centers()와 동일 원리, README §2.27). 매
        # 호출(=매 프레임)마다 새로 시작해야 밴드 인덱스가 프레임 경계를 넘어 롤오버되는
        # 걸 막는다.
        last_yellow_i = last_yellow_x = None
        last_white_i = last_white_x = None

        for i in range(self.n_slices):
            y_high = h - i * slice_h
            y_low = 0 if i == self.n_slices - 1 else h - (i + 1) * slice_h
            y_center = (y_low + y_high) / 2.0

            win_y = min(base_win + yellow_miss_streak * DL_LL_SEARCH_WIDEN_STEP_PX, DL_LL_SEARCH_WIDEN_MAX_PX)
            win_w = min(base_win + white_miss_streak * DL_LL_SEARCH_WIDEN_STEP_PX, DL_LL_SEARCH_WIDEN_MAX_PX)

            # [2026-08-12] 밴드별 프레임 간 앵커링 — 직전 프레임에 이 밴드(같은
            # y위치)에서 실제로 찾았던 위치가 있으면 이번 프레임 내 전파값과 가중평균해
            # 탐색창 중심으로 쓴다(_ll_slice_centers()와 동일 원리, README §2.27) — band 0
            # 검출 오차가 위 밴드로 그대로 누적 전파되는 걸 막는다.
            prev_y = self._yw_prev_band_yellow[i]
            anchor_yellow = (
                cur_yellow if prev_y is None else
                (1 - DL_LL_BAND_ANCHOR_ALPHA) * cur_yellow + DL_LL_BAND_ANCHOR_ALPHA * prev_y
            )
            prev_w = self._yw_prev_band_white[i]
            anchor_white = (
                cur_white if prev_w is None else
                (1 - DL_LL_BAND_ANCHOR_ALPHA) * cur_white + DL_LL_BAND_ANCHOR_ALPHA * prev_w
            )

            yx = None
            yx0, yx1 = int(np.clip(anchor_yellow - win_y, 0, w)), int(np.clip(anchor_yellow + win_y, 0, w))
            if yx1 > yx0:
                M_y = cv2.moments(ll_yellow_mask[y_low:y_high, yx0:yx1], binaryImage=True)
                if M_y['m00'] >= DL_LL_SIDE_MIN_PIXELS:
                    yx = yx0 + M_y['m10'] / M_y['m00']
            if yx is not None:
                # [2026-08-12] 속도예측 — 밴드 간 이동량(px/밴드)을 EMA로 추적해뒀다가
                # 다음 밴드 탐색창을 "찾은 위치" 그대로가 아니라 "그 위치 + 예측 이동량"
                # 으로 미리 옮긴다(미검출 밴드가 이어지는 동안엔 이 속도로 계속
                # dead-reckoning, 아래 else 분기 참고).
                if last_yellow_i is not None and i > last_yellow_i:
                    raw_v = (yx - last_yellow_x) / (i - last_yellow_i)
                    raw_v = float(np.clip(raw_v, -DL_LL_VELOCITY_MAX_PX, DL_LL_VELOCITY_MAX_PX))
                    a = DL_LL_VELOCITY_EMA_ALPHA
                    self._yw_yellow_velocity = (1 - a) * self._yw_yellow_velocity + a * raw_v
                last_yellow_i, last_yellow_x = i, yx
                self._yw_prev_band_yellow[i] = yx
                cur_yellow = yx + self._yw_yellow_velocity
                yellow_miss_streak = 0
            else:
                cur_yellow = cur_yellow + self._yw_yellow_velocity
                yellow_miss_streak += 1

            # ① 이번 프레임 첫(근거리) 유효 노란선으로 차선 판정을 확정한다. yx(이번
            # 밴드 실측값)를 기준으로 판정한다 — cur_yellow는 위에서 이미 다음 밴드용
            # 속도 예측이 더해진 값이라 "이번 밴드의 실제 위치"로 쓰면 안 된다.
            if yx is not None and not lane_side_locked:
                new_sign = 1.0 if yx < ref_x else -1.0
                if new_sign != side_sign:
                    side_sign = new_sign
                    cur_white = yx + side_sign * self._white_yellow_gap_px
                self.lane_side = 'right' if side_sign > 0 else 'left'
                lane_side_locked = True

            wx = None
            wx0, wx1 = int(np.clip(anchor_white - win_w, 0, w)), int(np.clip(anchor_white + win_w, 0, w))
            if wx1 > wx0:
                M_w = cv2.moments(ll_white_mask[y_low:y_high, wx0:wx1], binaryImage=True)
                if M_w['m00'] >= DL_LL_SIDE_MIN_PIXELS:
                    wx = wx0 + M_w['m10'] / M_w['m00']
            if wx is not None:
                if last_white_i is not None and i > last_white_i:
                    raw_v = (wx - last_white_x) / (i - last_white_i)
                    raw_v = float(np.clip(raw_v, -DL_LL_VELOCITY_MAX_PX, DL_LL_VELOCITY_MAX_PX))
                    a = DL_LL_VELOCITY_EMA_ALPHA
                    self._yw_white_velocity = (1 - a) * self._yw_white_velocity + a * raw_v
                last_white_i, last_white_x = i, wx
                self._yw_prev_band_white[i] = wx
                cur_white = wx + self._yw_white_velocity
                white_miss_streak = 0
            else:
                cur_white = cur_white + self._yw_white_velocity
                white_miss_streak += 1

            if yx is not None and wx is not None:
                # ② 둘 다 찾음 — 정상 검출. gap EMA는 상하한을 클램프한다(아래 참고).
                results[i] = (y_center, (yx + wx) / 2.0)
                used[i] = True
                self.ll_band_case[i] = 'Y+W'
                alpha = DL_LL_YELLOW_GAP_EMA_ALPHA
                new_gap = (1 - alpha) * self._white_yellow_gap_px + alpha * abs(wx - yx)
                self._white_yellow_gap_px = float(
                    np.clip(new_gap, DL_LL_YELLOW_GAP_MIN_PX, DL_LL_YELLOW_GAP_MAX_PX)
                )
            elif yx is not None:
                # ③ 노란선만 찾음 — 간격으로 흰선 위치 추정.
                est_wx = yx + side_sign * self._white_yellow_gap_px
                results[i] = (y_center, (yx + est_wx) / 2.0)
                self.ll_band_degraded[i] = True
                self.ll_band_case[i] = 'Y+gap'
            else:
                # ④ 노란선을 이번 밴드에서 못 찾음 — [2026-08-10] 좁은 창(cur_white
                # 중심) 하나로 흰선을 찾아 간격을 역적용하던 옛 방식은, gap EMA가
                # 부풀면 실제 흰선 위치와 무관하게 waypoint를 차선 밖으로 밀어내는
                # 문제가 실차에서 확인됐다(config.py DL_LL_YELLOW_GAP_MIN/MAX_PX
                # 주석 참고). 넓은 창(DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX)에서
                # 흰선을 몇 개 찾았는지로 3분기한다 — _ll_line_centers()는 로컬
                # 슬라이스 좌표를 반환하므로 wx0w를 다시 더해 원래 좌표로 되돌린다.
                wide_win = DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX
                wx0w = int(np.clip(cur_yellow - wide_win, 0, w))
                wx1w = int(np.clip(cur_yellow + wide_win, 0, w))
                lines = []
                if wx1w > wx0w:
                    lines = [
                        wx0w + lx for lx in
                        self._ll_line_centers(ll_white_mask[y_low:y_high, wx0w:wx1w])
                    ]
                gap = float(
                    np.clip(self._white_yellow_gap_px, DL_LL_YELLOW_GAP_MIN_PX, DL_LL_YELLOW_GAP_MAX_PX)
                )

                if len(lines) >= 2:
                    # 케이스1: 양쪽 흰선을 다 찾음 — 두 실측 위치의 중점을 그대로
                    # 채택한다(간격 추정치에 안 기대는 가장 신뢰도 높은 재구성).
                    left_x, right_x = lines[0], lines[-1]
                    center = (left_x + right_x) / 2.0
                    cur_white = right_x if side_sign > 0 else left_x
                    cur_yellow = center  # 다음 밴드 탐색창 연속성용(실측 2개 기반이라 신뢰 가능)
                    results[i] = (y_center, center)
                    self.ll_band_case[i] = '2W'
                elif len(lines) == 1:
                    # 케이스2: 흰선 하나만 찾음 — 어느 쪽인지는 self.lane_side(프레임당
                    # 한 번 고정, stale할 수 있음) 대신 이번 밴드 실측 위치를 기준점
                    # (cur_yellow)과 비교해 매 밴드 새로 판정한다.
                    wx_found = lines[0]
                    found_sign = 1.0 if wx_found >= cur_yellow else -1.0
                    center = wx_found - found_sign * gap
                    cur_white = wx_found
                    cur_yellow = center
                    results[i] = (y_center, center)
                    self.ll_band_case[i] = '1W:R' if found_sign > 0 else '1W:L'
                else:
                    # 케이스3: 흰선도 하나도 못 찾음 — 잔상(직전 위치 유지, cur_* 갱신 안 함).
                    results[i] = (y_center, (cur_yellow + cur_white) / 2.0)
                    self.ll_band_case[i] = 'LOST'
                self.ll_band_degraded[i] = True

            self.ll_search_windows.append((y_low, y_high, yx0, yx1, yx, wx0, wx1, wx))

        self.ll_degraded = any(self.ll_band_degraded)
        return results, used

    def _ll_slice_centers(self, ll_mask, ref_x):
        """[2026-08-10] DL_CENTER_MODE='ll' && DL_LL_ALGO='lr'일 때만 호출된다('ll_da'는
        corridor 알고리즘(_corridor_slice_centers())으로 교체돼 더 이상 이 함수를 안 쓴다.
        DL_LL_ALGO='yw'(main 기본값)면 대신 _ll_yellow_white_centers()가 호출된다 — 두
        알고리즘을 병합 때 둘 다 살리고 전환 가능하게 하기로 해서 이 함수가 남아있다,
        README §2.19 참고). ll_mask(흰선만
        담긴 이진마스크, _split_ll_by_yellow() 참고)를
        _slice_centers()와 동일한 n_slices 밴드로 나눠, **좌/우 라인을 각각 독립적인
        슬라이딩 윈도우로 추적**한다 — 참고 프로젝트
        (github.com/junhyukch7/Advanced-Lane-Detection)의 `slidingWindow()`가 좌/우를
        따로 두 번 호출해 서로 무관하게 창을 옮기는 것과 동일한 원칙.

        [2026-08-07] 기존에는 좌/우를 한 밴드 안에서 같이 판정해서, 한쪽이라도 실패하면
        (한쪽 창에 픽셀이 모자라거나, 둘 다 찾았어도 폭이 비정상이면) 그 밴드 전체를
        버렸다 — 반대쪽 선은 멀쩡히 보이는데도 같이 버려지는 게 낭비였다(예: 한쪽
        차선이 반사/가려짐으로 몇 밴드 끊겨도 반대쪽은 계속 잘 보이는 실제 상황).
        이번에 좌/우 창(cur_left/cur_right)을 완전히 독립적으로 갱신하도록 바꿨다 —
        왼쪽 창은 왼쪽에서 뭔가 찾았을 때만(오른쪽 결과와 무관하게) 갱신하고, 오른쪽도
        마찬가지다. 밴드별 최종 중심점은 세 갈래로 결정한다:
          1. 양쪽 다 찾고 두 중심 간 거리가 실측 차로폭 범위(DL_LL_WIDTH_MIN_PX~MAX_PX)
             안이면 → 중점을 채택하고, 이때의 실측 폭으로 self._ll_half_width(차로
             반폭 러닝 추정치, DL_LL_WIDTH_EMA_ALPHA로 EMA 갱신)를 업데이트한다.
          2. 한쪽만 찾았으면(또는 양쪽 다 찾았지만 폭이 비정상이라 서로 못 믿을 때는
             제외) → 찾은 쪽 위치에서 self._ll_half_width만큼 반대쪽으로 밀어 중심을
             추정한다(lane_util.SlideWindow.calc_center()의 "한쪽 차선만 검출" 폴백과
             동일한 원칙 — classic_cv 백엔드가 이미 쓰던 패턴을 ll에도 적용).
          3. 양쪽 다 못 찾았으면 → None.
        좌/우 각자의 탐색창은 여전히 좁은 고정폭(DL_LL_SEARCH_HALF_WIDTH_PX 반경)만
        본다 — ROI 폭 전체(수백 px)를 반씩 나눠 보던 옛 방식(2026-08-07 이전)은 옆
        차선/반사광이 반쪽 어디에 있든 섞여 들어가는 문제가 있었다.

        [2026-08-10] 탐색창을 두 방향으로 "적응형"으로 바꿨다(config.py DL_LL_VELOCITY_*/
        DL_LL_SEARCH_WIDEN_* 주석 참고) — 아래 "알려진 한계" 1번 대응:
          ①속도 예측 — 그 사이드에서 실제로 찾은 밴드들 사이의 x 이동량(밴드 간
            간격으로 나눈 px/밴드)을 self._ll_left_velocity/_right_velocity로 EMA
            추적한다. 다음 밴드 탐색창 중심은 "마지막으로 찾은 위치"가 아니라 "그
            위치 + 예측 이동량"으로 미리 옮겨서, 창이 곡선을 따라 먼저 움직이게
            한다. 미검출 밴드가 이어지는 동안에도 이 속도로 계속 dead-reckoning
            이동시킨다(멈춰서 있지 않음).
          ②탐색창 확장 — 그 사이드가 연속으로 못 찾을 때마다 탐색창 반경을
            넓혀(DL_LL_SEARCH_WIDEN_STEP_PX씩, DL_LL_SEARCH_WIDEN_MAX_PX 상한)
            재포착 기회를 늘리고, 다시 찾으면 기본 반경(DL_LL_SEARCH_HALF_WIDTH_PX)
            으로 리셋한다.
        속도 EMA는 self에 영속(프레임 간 유지)되지만, "밴드 간 간격" 계산에 쓰는
        마지막 검출 밴드 인덱스/위치는 이번 프레임 안에서만 의미가 있어 매 호출마다
        지역변수로 새로 시작한다(프레임 경계를 넘어 간격을 계산하면 밴드 인덱스가
        롤오버돼 음수 gap이 나옴).

        [2026-08-10] 프레임 간 밴드별 앵커링도 추가했다(config.py
        DL_LL_BAND_ANCHOR_ALPHA 주석 참고) — 기존엔 band 0(근거리)만 직전 프레임 전체
        확정 lane_center(ref_x)에 앵커링되고 그 위 밴드는 전부 이번 프레임 안에서만
        전파되던 방식이라, band 0 검출이 노이즈로 틀어지면 그 오차가 위 밴드까지
        누적 전파됐다. self._ll_prev_band_left/right[i]에 밴드별로 "직전 프레임에 그
        밴드(같은 y위치)에서 실제로 찾은 위치"를 따로 기억해뒀다가, 이번 프레임 그
        밴드의 탐색창 중심을 (이번 프레임 내 전파값, 직전 프레임 그 밴드 값)의
        가중평균(DL_LL_BAND_ANCHOR_ALPHA)으로 잡는다 — 도로 곡률이 프레임 간
        급격히 안 변한다는 가정에 기대어, band 0의 오차가 위로 그대로 번지지 않고 각
        밴드 고유의 과거 위치 쪽으로 당겨지게 한다. 밴드값은 실제로 찾았을 때만
        갱신하고 못 찾은 프레임엔 이전 값을 그대로 들고 있는다(self._ll_half_width와
        동일한 관례 — 잠깐 안 보여도 즉시 잊지 않음).

          입력 : ll_mask — (roi_h, roi_w) uint8 이진마스크(da_mask와 동일 shape/좌표계)
                 ref_x   — 첫(근거리) 밴드의 좌/우 초기 중심 x좌표. 보통 직전 프레임 lane_center.
          출력 : (results, used) — 둘 다 길이 self.n_slices.
                 results[i] : 채택되면 (y_center, cx), 아니면 None(da로 폴백해야 함을 뜻함)
                 used[i]    : results[i]가 ll 기반으로 채택됐는지(양쪽/편측 무관, bool) — 디버그 시각화용

        알려진 한계:
        - [2026-08-10 일부 완화] 속도예측+탐색창 확장을 추가했지만 둘 다 실차
          미검증이다 — 급커브에서 창이 실제로 선을 놓치지 않고 따라가는지, 확장된
          창이 오히려 옆 차선/반사광을 잘못 물지는 않는지 확인 필요.
        - 편측 폴백(위 2번)이 여러 밴드 연속으로 이어지면, self._ll_half_width가 그 사이
          갱신되지 않아(양쪽 다 찾은 밴드에서만 갱신) 오래된 추정치를 계속 쓰게 된다 —
          실차 미검증, 편측 검출이 긴 구간에서 추정 중심이 실제와 얼마나 벌어지는지
          확인할 것.
        - 밴드별 앵커링은 "도로 곡률이 프레임 간 급격히 안 변한다"는 가정에 기댄다 —
          차량이 급조향 중이거나 카메라 프레임레이트가 낮으면 이 가정이 깨져 오히려
          과거 위치로 창을 잘못 당길 수 있다. 실차 미검증."""
        h, w = ll_mask.shape
        slice_h = h // self.n_slices
        results = [None] * self.n_slices
        used = [False] * self.n_slices
        # 디버그 시각화용 — 밴드별 좌/우 탐색창 좌표 + 실제로 찾았는지(visualize()가
        # 사각형/색으로 그림). 알고리즘 자체엔 전혀 쓰이지 않는 부가 정보.
        self.ll_search_windows = []
        # [2026-08-10] 디버그 시각화 전용 스냅샷 — 이번 프레임이 실제로 앵커로 쓴 "직전
        # 프레임 밴드별 위치"를 self._ll_prev_band_left/right가 이 루프 안에서 곧바로
        # 덮어써지기 전에 따로 떠둔다. visualize()가 이 스냅샷을 마젠타 점으로 찍어서
        # "이번 프레임이 어디를 앵커로 당겨졌는지"를 실제 검출 결과와 나란히 비교할 수
        # 있게 한다(self._ll_prev_band_left/right 자체는 알고리즘 상태라 루프 도중 계속
        # 갱신되므로 디버그용으로 못 씀).
        self.ll_band_anchor_left = list(self._ll_prev_band_left)
        self.ll_band_anchor_right = list(self._ll_prev_band_right)

        cur_left = ref_x - self._ll_half_width
        cur_right = ref_x + self._ll_half_width
        base_win = DL_LL_SEARCH_HALF_WIDTH_PX
        left_miss_streak = 0
        right_miss_streak = 0
        # 이번 프레임 안에서만 유효한 "마지막으로 실제 찾은 밴드" 기록 — 밴드 간
        # 간격(gap)을 구해 속도(px/밴드)를 계산하는 데만 쓴다(위 docstring 참고).
        last_left_i = last_left_x = None
        last_right_i = last_right_x = None

        for i in range(self.n_slices):
            y_high = h - i * slice_h
            y_low = 0 if i == self.n_slices - 1 else h - (i + 1) * slice_h
            band = ll_mask[y_low:y_high, :]
            y_center = (y_low + y_high) / 2.0

            # 밴드별 프레임 간 앵커링 — 직전 프레임에 이 밴드에서 실제로 찾았던 위치가
            # 있으면 이번 프레임 내 전파값(cur_left/right)과 가중평균해서 탐색창
            # 중심으로 쓴다.
            prev_l = self._ll_prev_band_left[i]
            anchor_left = (
                cur_left if prev_l is None else
                (1 - DL_LL_BAND_ANCHOR_ALPHA) * cur_left + DL_LL_BAND_ANCHOR_ALPHA * prev_l
            )
            prev_r = self._ll_prev_band_right[i]
            anchor_right = (
                cur_right if prev_r is None else
                (1 - DL_LL_BAND_ANCHOR_ALPHA) * cur_right + DL_LL_BAND_ANCHOR_ALPHA * prev_r
            )

            win_l = min(base_win + left_miss_streak * DL_LL_SEARCH_WIDEN_STEP_PX, DL_LL_SEARCH_WIDEN_MAX_PX)
            win_r = min(base_win + right_miss_streak * DL_LL_SEARCH_WIDEN_STEP_PX, DL_LL_SEARCH_WIDEN_MAX_PX)

            lx = None
            lx0, lx1 = int(np.clip(anchor_left - win_l, 0, w)), int(np.clip(anchor_left + win_l, 0, w))
            if lx1 > lx0:
                M_l = cv2.moments(band[:, lx0:lx1], binaryImage=True)
                if M_l['m00'] >= DL_LL_SIDE_MIN_PIXELS:
                    lx = lx0 + M_l['m10'] / M_l['m00']

            rx = None
            rx0, rx1 = int(np.clip(anchor_right - win_r, 0, w)), int(np.clip(anchor_right + win_r, 0, w))
            if rx1 > rx0:
                M_r = cv2.moments(band[:, rx0:rx1], binaryImage=True)
                if M_r['m00'] >= DL_LL_SIDE_MIN_PIXELS:
                    rx = rx0 + M_r['m10'] / M_r['m00']

            if lx is not None:
                if last_left_i is not None and i > last_left_i:
                    raw_v = (lx - last_left_x) / (i - last_left_i)
                    raw_v = float(np.clip(raw_v, -DL_LL_VELOCITY_MAX_PX, DL_LL_VELOCITY_MAX_PX))
                    a = DL_LL_VELOCITY_EMA_ALPHA
                    self._ll_left_velocity = (1 - a) * self._ll_left_velocity + a * raw_v
                last_left_i, last_left_x = i, lx
                self._ll_prev_band_left[i] = lx
                cur_left = lx + self._ll_left_velocity  # 왼쪽 창은 왼쪽 결과만으로 독립 갱신 — 오른쪽 성패와 무관, 다음 밴드 예상 위치까지 미리 반영
                left_miss_streak = 0
            else:
                cur_left = cur_left + self._ll_left_velocity  # 못 찾아도 속도만큼 계속 예측 이동(dead-reckoning)
                left_miss_streak += 1

            if rx is not None:
                if last_right_i is not None and i > last_right_i:
                    raw_v = (rx - last_right_x) / (i - last_right_i)
                    raw_v = float(np.clip(raw_v, -DL_LL_VELOCITY_MAX_PX, DL_LL_VELOCITY_MAX_PX))
                    a = DL_LL_VELOCITY_EMA_ALPHA
                    self._ll_right_velocity = (1 - a) * self._ll_right_velocity + a * raw_v
                last_right_i, last_right_x = i, rx
                self._ll_prev_band_right[i] = rx
                cur_right = rx + self._ll_right_velocity  # 오른쪽 창은 오른쪽 결과만으로 독립 갱신 — 왼쪽 성패와 무관
                right_miss_streak = 0
            else:
                cur_right = cur_right + self._ll_right_velocity
                right_miss_streak += 1

            if lx is not None and rx is not None:
                width = rx - lx
                if DL_LL_WIDTH_MIN_PX < width < DL_LL_WIDTH_MAX_PX:
                    results[i] = (y_center, (lx + rx) / 2.0)
                    used[i] = True
                    alpha = DL_LL_WIDTH_EMA_ALPHA
                    self._ll_half_width = (1 - alpha) * self._ll_half_width + alpha * (width / 2.0)
                    self.ll_band_reason[i] = 'B'
                else:
                    # 폭이 비정상 — 반대 차선을 잘못 짝지었을 가능성, 양쪽 다 못 믿으므로 무효
                    self.ll_band_reason[i] = 'X'
            elif lx is not None:
                results[i] = (y_center, lx + self._ll_half_width)
                used[i] = True
                self.ll_band_reason[i] = 'L'
            elif rx is not None:
                results[i] = (y_center, rx - self._ll_half_width)
                used[i] = True
                self.ll_band_reason[i] = 'R'
            else:
                self.ll_band_reason[i] = '-'

            self.ll_search_windows.append((y_low, y_high, lx0, lx1, lx, rx0, rx1, rx))

        return results, used

    def _debounce_path_ok(self, path_ok_raw):
        """self.path(자홍색 경로, pure_pursuit이 그대로 쓰는 값) 갱신을 허용해도 되는지를
        lane_valid(근접 밴드 필수, offset/lane_center용)와 별도로 판단한다.

        [2026-08-17n, README §2.36 후속] 원래는 lane_valid 하나로 self.path 갱신까지
        같이 막았다(§2.36 "룩어헤드 점프" 크래시 수정) — 내부 self.path와 외부
        track_drive.py의 self.lane_path가 서로 다른 조건으로 얼고 풀리면 그 틈에 내부만
        몰래 흘러가다 한 틱에 점프하는 문제였다. 그런데 lane_valid는 근접 밴드
        (near_center)가 반드시 있어야 하는데, 급커브 진입처럼 "근접만 일시적으로 안
        보이고 원거리는 있는" 상황(실차 재현: 좌회전 진입에서 근접 밴드가 몇 초간
        비어 그동안 경로 전체가 필요 이상으로 얼어붙음, 카카오톡 영상 2026-08-17
        15:44)에서 원거리 정보를 아예 못 쓰게 만들어버린다는 게 뒤늦게 드러났다.

        그래서 "경로 갱신 허용"은 근접 OR 원거리 둘 중 하나만 있어도(+ll sanity 통과)
        허용하는 별도의 raw 신호로 만들고, 여기서 offset과 동일한 강도(stable_frame_min
        연속 확인)로 디바운스한다 — 2026-08-10에 고쳤던 "밴드 판정이 프레임마다
        흔들려 조향이 그 흔들림을 그대로 흡수하는" 문제가 되돌아오지 않게 하기 위함.
        detect()가 이 디바운스된 값(self.path_ok)으로 내부 _update_path() 호출을
        가드하고, DLLaneDetector._worker()가 그대로 읽어 밖으로 노출하면
        track_drive.py도 같은 값으로 self.lane_path 갱신을 가드한다(perc_lane() 참고) —
        내부/외부가 항상 같은 신호를 보므로 §2.36 크래시가 재발할 여지가 없다."""
        if self._path_ok_confirmed is None:
            self._path_ok_confirmed = path_ok_raw
            self._path_ok_pending = path_ok_raw
            self._path_ok_pending_count = self.stable_frame_min
            return self._path_ok_confirmed

        if path_ok_raw == self._path_ok_pending:
            self._path_ok_pending_count += 1
        else:
            self._path_ok_pending = path_ok_raw
            self._path_ok_pending_count = 1

        if self._path_ok_pending_count >= self.stable_frame_min:
            self._path_ok_confirmed = self._path_ok_pending

        return self._path_ok_confirmed

    def detect(self, raw_bgr, da_prob, ll_prob, yellow_mask, avoid_hold=False, direction_hint=0, v_mps=0.0,
               obstacle_y_m=None, obstacle_cut_confirmed=False):
        """입력 : raw_bgr — 원본 카메라 프레임 그대로의 (H,W,3) BGR(크롭/리사이즈 없음)
                 da_prob, ll_prob — 위와 같은 (H,W) float32 foreground 확률(모델은 360행
                   고정이지만 TwinLiteNetEngine.infer_raw()가 이미 원본 크기로 업샘플링해서 줌)
                 yellow_mask — 위와 같은 (H,W) uint8 이진마스크(HSV 기반, da/ll과 무관)
                 avoid_hold — [2026-08-14] True면 DL_DA_SKIP_LL_CLIP=True(평소 테스트 설정)를
                   무시하고 _clip_da_by_ll()을 강제로 돌린다(§2.32, DL_CENTER_MODE='da' 전용
                   — 다른 모드는 이 인자와 무관하게 항상 클리핑 적용). track_drive.py의
                   _update_avoid_hold()가 라이다 obstacle_front/dist(+da 연속성, 적용2)로
                   판단해 넘긴다.
                 v_mps — [2026-08-17g] 현재 실측 속도(m/s). DL_CENTER_MODE='da'의
                   _apply_vehicle_margin()이 방해차량 "뒤" 방향 추가 마진 계산에만 쓴다
                   (config.py DL_DA_REAR_MARGIN_* 참고) — 그 외 로직에는 영향 없다.
                 direction_hint — [2026-08-15] 적용3(avoid_hold_improvement_proposal.md).
                   track_drive.py TargetPassing.choose_side()가 반환한 -1/0/+1(lane_offset과
                   동일한 "우측+" 부호규약) — _clip_da_by_ll()의 실측/잔상이 전혀 없는
                   최후수단(가상경계) 폴백에서만 기준점을 이 방향으로 살짝 기울이는 데 쓴다.
                 obstacle_y_m, obstacle_cut_confirmed — [2026-08-20] da 근접 컷
                   (_clip_da_by_obstacle(), ENABLE_OBSTACLE_CUT). track_drive.py의
                   perc_obstacle_cut_trigger()가 라이다 AND YOLO로 확정한 값을
                   set_obstacle()을 거쳐 넘겨준다. DL_CENTER_MODE=='da' 전용.
          출력 : lane_valid, offset, lookahead, lane_center, path — 기존 SlideWindow.calc_center()와
                 동일한 계약(같은 4-tuple+path 형태)이지만, 계산은 da 중심선 기준으로 직접 한다.
          내부에서 DL_ROI_Y0:DL_ROI_Y1(원본 프레임 절대 픽셀)만 잘라서 da 중심선을 뽑는다.
        """
        self.avoid_hold_active = bool(avoid_hold)
        self.avoid_hold_dir_hint = int(direction_hint)
        h, _ = ll_prob.shape
        y0 = max(0, min(DL_ROI_Y0, h))
        y1 = max(y0, min(DL_ROI_Y1, h))

        ll_roi = ll_prob[y0:y1]
        da_roi = da_prob[y0:y1]
        yellow_roi = yellow_mask[y0:y1]
        vis_roi = raw_bgr[y0:y1]

        # [2026-08-05] DL_USE_BEV=True면 여기서 원근왜곡을 제거한다 — da/ll은 이진화 전
        # (float 확률맵) 상태로 워프해야 경계가 계단처럼 뭉개지지 않는다. 이후의 이진화/
        # largest_component/clip_by_ll/slice_centers는 전부 마스크 shape만 보는 범용
        # 로직이라 좌표계가 원근이든 BEV든 그대로 재사용된다(self.roi_h/roi_w가 da_mask.shape
        # 를 따라가므로 자동으로 캔버스 크기로 바뀐다).
        if DL_USE_BEV:
            ll_roi = self._bev_warp(ll_roi)
            da_roi = self._bev_warp(da_roi)
            yellow_roi = self._bev_warp(yellow_roi, nearest=True)
            vis_roi = self._bev_warp(vis_roi)
            # [2026-08-06] 원거리 크롭 — DL_BEV_FAR_CROP_ROW보다 위(더 먼) 행은 버린다
            # (config.py DL_BEV_FAR_LIMIT_M 주석 참고). 네 배열 전부 같은 행을 자르므로
            # 이후 좌표계는 계속 서로 일치한다.
            if DL_BEV_FAR_CROP_ROW > 0:
                ll_roi = ll_roi[DL_BEV_FAR_CROP_ROW:]
                da_roi = da_roi[DL_BEV_FAR_CROP_ROW:]
                yellow_roi = yellow_roi[DL_BEV_FAR_CROP_ROW:]
                vis_roi = vis_roi[DL_BEV_FAR_CROP_ROW:]

        # ll은 da(DL_FG_THRESHOLD)보다 높은 임계값을 쓴다 — BEV 워프가 원거리일수록 확률맵
        # 경계 blur를 더 크게 확대해서, 낮은 임계값으로는 원거리 ll이 실제보다 두껍게 잡힌다
        # (config.py DL_LL_FG_THRESHOLD 주석 참고).
        ll_mask = (ll_roi >= DL_LL_FG_THRESHOLD).astype(np.uint8) * 255
        self.ll_coverage = float(np.count_nonzero(ll_mask)) / ll_mask.size if ll_mask.size else 0.0

        # [2026-08-07] ll 잔상(decay) — _clip_da_by_ll() 전용 입력. 이번 프레임 ll이
        # 순간적으로 끊겨도(반사/모션블러 등) 직전까지 확실했던 픽셀을 감쇠 가중치로 몇
        # 프레임 더 "보이는 것"처럼 들고 있는다 — DL_LL_DECAY_ALPHA(매 프레임 곱해지는
        # 감쇠율)만큼 값이 줄다가 DL_LL_DECAY_MIN_VALUE 밑으로 내려가면 자연히 "안 보임"
        # 취급된다(별도 리셋 로직 없이 곱셈 감쇠만으로 N프레임 뒤 자동 소멸). centerline
        # 추출(_ll_yellow_white_centers()/_ll_slice_centers(), DL_LL_ALGO 참고)에는 이 잔상을 안 쓴다 — waypoint 자체를 과거 위치로 밀면
        # 더 위험하고, 클리핑은 "울타리" 역할이라 약간 stale해도 안전하다는 판단
        # (_clip_da_by_ll() 클래스 docstring 참고). shape이 안 맞으면(첫 프레임/캔버스
        # 크기 변경 등) 새로 0으로 할당한다.
        if self._ll_decay_mask is None or self._ll_decay_mask.shape != ll_mask.shape:
            self._ll_decay_mask = np.zeros(ll_mask.shape, dtype=np.float32)
        self._ll_decay_mask = np.maximum(
            ll_mask.astype(np.float32), self._ll_decay_mask * DL_LL_DECAY_ALPHA
        )
        ll_mask_for_clip = (self._ll_decay_mask >= DL_LL_DECAY_MIN_VALUE).astype(np.uint8) * 255

        # [2026-08-07] ll 흰선/노란선 분리(_split_ll_by_yellow() 참고) — DL_CENTER_MODE='ll'
        # 전용 _ll_yellow_white_centers()(DL_LL_ALGO='yw')는 노란선/흰선을 각각 분리된
        # 마스크(ll_yellow_mask/ll_white_mask)로 따로 추적하고, _ll_slice_centers()
        # (DL_LL_ALGO='lr')는 흰선 마스크만 쓴다(중앙 노란 점선이 좌/우 트래킹에 안
        # 섞이게). 'll_da'(corridor)는 이 분리 결과를 안 쓰고 흰/노랑 안 가린 원본
        # ll_mask를 그대로 쓴다(_ll_line_centers() 참고 — 노란 중앙선도 "2번째 선"으로
        # 세야 하므로).
        ll_white_mask, ll_yellow_mask = self._split_ll_by_yellow(ll_mask, yellow_roi)
        self.ll_white_mask_roi = ll_white_mask
        self.ll_yellow_mask_roi = ll_yellow_mask

        da_mask = (da_roi >= DL_FG_THRESHOLD).astype(np.uint8) * 255
        # 덩어리 선택(_largest_da_component)/ll 클리핑 전, 이진화 직후의 da 전체 — 아래에서
        # da_mask가 선택/클리핑된 결과로 재대입되기 전에 따로 남겨둔다(visualize()가 "모델이
        # 주행가능하다고 본 전체"를 파란색으로, 실제 채택분(self.da_mask_roi)을 그 위에
        # 초록/주황/청록으로 겹쳐 그려서 "전체 중 실제로 뭘 골랐는지" 한눈에 비교 가능하게 함).
        self.da_mask_all_roi = da_mask

        # _slice_centers()가 self.vis/self.roi_h를 참조하므로(DEBUG_VIZ_LANE 디버그
        # 사각형 — classic_cv 백엔드용 플래그지만 SlideWindow 공용 코드라 여기도 거친다),
        # 아래에서 클리핑 전/후 시험 호출을 하기 전에 미리 채워둔다. da_mask.shape는 이후
        # largest_component/clip을 거쳐도 안 바뀌므로 지금 확정해도 안전하다.
        self.roi_h, self.roi_w = da_mask.shape
        # [2026-08-17] roi_w/2.0(캔버스 단순 절반) 대신 BEV 사다리꼴의 실제 기하로 구한
        # 차량 중심(DL_BEV_VEHICLE_CENTER_X, 위 모듈 상단 주석 참고)을 쓴다. DL_USE_BEV=False면
        # 워프 자체가 없어 저 상수가 의미 없으므로 그때만 roi_w/2.0로 되돌아간다.
        self.vehicle_center_x = DL_BEV_VEHICLE_CENTER_X if DL_USE_BEV else self.roi_w / 2.0
        self.vis = vis_roi.copy()
        self.ll_mask_roi = ll_mask

        # [2026-08-10] 세 모드가 이제 서로 완전히 다른 알고리즘이다(모듈 상단 주석,
        # config.py DL_CENTER_MODE 주석 참고) — 'll_da'(corridor)만 완전히 별도 경로다.
        if DL_CENTER_MODE == 'll_da':
            # corridor는 "자기 차선 하나"를 전제로 한 largest-component/ll클리핑을
            # 건너뛰고 클리핑 전 원본 da(da_mask_all_roi)를 그대로 쓴다 — 장애물이
            # 도로를 좌/우로 쪼갤 때 largest-component는 지나갈 수 있는 작은 덩어리를
            # 통째로 버리고, clip_da_by_ll은 ref_x 기준 한쪽 차로만 남겨 반대 차로 da까지
            # 잘라내 버려서 corridor 취지(양쪽 차로를 동시에 보고 그 안에서 고른다)와
            # 정반대다. 이 모드 전용이 아닌 디버그 필드('da'/'ll' 모드의
            # largest-component/clip 결과를 나타내는 것들)는 의미가 없으므로 명시적으로
            # 리셋해 visualize()가 엉뚱한 태그를 그리지 않게 한다.
            self.da_fallback_used = False
            self.da_largest_mask_roi = None
            self.da_largest_area_px = 0
            self.da_chosen_area_px = 0
            self.da_ll_clip_skipped = False
            self.da_ll_virtual_clip_used = False
            self.corridor_bounds = [None] * self.n_slices
            self.ll_search_windows = []
            self.da_clip_band_virtual = [None] * self.n_slices
            self.ll_band_reason = [None] * self.n_slices
            self.ll_band_degraded = [False] * self.n_slices
            self.ll_band_case = ['?'] * self.n_slices
            self.ll_degraded = False
            da_mask = self.da_mask_all_roi
            merged_centers, self.ll_band_used = self._corridor_slice_centers(da_mask, ll_mask)
        else:
            # 급커브 파편화 대응: 가장 큰 덩어리만 남긴다(모듈 상단 주석 참고). ll
            # 클리핑보다 먼저 해야 한다 — 이 시점엔 da가 아직 하나의 연결된 덩어리라
            # "가장 큰 덩어리"가 곧 도로 전체를 뜻하지만, 클리핑을 먼저 하면 밴드마다
            # 독립적으로 좌우를 잘라 인접 밴드끼리 남은 x범위가 안 겹치는 경우가 생겨(ll
            # 경계가 밴드마다 조금씩 다르게 잡히면 흔함) 마스크가 밴드별로 끊기고, 그 뒤
            # largest_da_component가 "가장 큰 덩어리"로 폭 넓은 밴드 하나만 통째로
            # 골라버려 도로 모양이 아니라 네모난 밴드 하나만 남는 문제가 생긴다(실측으로
            # 확인됨).
            da_mask = self._largest_da_component(da_mask)

            # [2026-08-15] avoid-hold 적용2(문제3 라이다 사각지대 보완, config.py
            # AVOID_HOLD_DA_AREA_JUMP_RATIO 주석) — da_chosen_area_px가 직전 프레임 대비
            # 급증했으면(=방금까지 뚫려있던 구멍이 갑자기 메워짐) "뭔가 방금 시야에서
            # 사라졌을 수 있다"는 보조 신호로 쓴다. track_drive.py _update_avoid_hold()가
            # 라이다 obstacle_front와 OR로만 결합한다(세그멘테이션 자체가 흔들리는
            # 프레임에서 이 신호 단독으로는 오발동할 수 있어서 — 문제3 대비책).
            prev_area = self._prev_da_chosen_area_px
            self.da_area_jump_detected = bool(
                prev_area > 0 and self.da_chosen_area_px >= prev_area * AVOID_HOLD_DA_AREA_JUMP_RATIO)
            self._prev_da_chosen_area_px = self.da_chosen_area_px

            # 옆 차선 침범 대응: ll(잔상 포함) 또는 가상경계로 그 바깥(옆 차선 쪽) da를
            # 잘라낸다(모듈 상단 주석, _clip_da_by_ll() docstring 참고). ref_x는 직전
            # 프레임 확정 lane_center — 아직 없으면(첫 프레임) ROI 중앙을 기준으로
            # 시작한다. 이 클리핑이 밴드 간 연결을 끊어도 상관없다 — 아래
            # _slice_centers()는 밴드별로 독립적으로 moments를 구하므로 전역 연결성이
            # 필요 없다(그래서 여기선 largest_da_component를 다시 돌리지 않는다).
            # ll_mask_for_clip(잔상 합성본)을 넘긴다 — 원본 ll_mask/ll_white_mask는
            # 아래 centerline 추출에만 쓴다.
            #
            # [2026-08-06] 클리핑 결과가 fit 가능한 최소 밴드 수(DL_SLICE_FIT_MIN)에 못
            # 미치면 클리핑을 버리고 클리핑 전 da로 되돌린다("차선책",
            # _largest_da_component()의 면적상한 폴백과 같은 원칙). S자 연속 커브에서
            # 원거리 ll이 DL_LL_FG_THRESHOLD를 올려도 여전히 두껍게 잡히면
            # _clip_da_by_ll()이 여러 밴드를 통째로 깎아버려 da가 "작게 검출된" 것처럼
            # 보이는 경우가 실측으로 확인됨 — da 자체는 멀쩡한데 ll 클리핑이 지워버린
            # 것이므로, 이럴 땐 클리핑 없는(=옆 차선 침범 위험은 있지만 최소한 주행은
            # 하는) da를 쓰는 편이 self.path가 무한정 얼어붙어 완전정지하는 것보다
            # 낫다는 판단. self.da_ll_clip_skipped로 표시해 visualize()가 구분 표시한다.
            ref_x = self._confirmed[3] if self._confirmed is not None else da_mask.shape[1] / 2.0
            # [2026-08-13] DL_CENTER_MODE='da' + DL_DA_SKIP_LL_CLIP=True(config.py 주석
            # 참고) — "da 자체가 잘 검출된다" 가정하에 이 클리핑 단계를 아예 건너뛴다.
            # 기존 "클리핑을 버리고 되돌린" 경우(da_ll_clip_skipped)와 화면상 표시가
            # 같아지도록(청록 오버레이 + 클리핑 틱 없음) 같은 필드를 그대로 재사용한다 —
            # 이유는 다르지만("밴드 부족으로 버림" vs "테스트를 위해 애초에 안 함") 둘 다
            # "이번 프레임 da_mask는 클리핑 안 된 largest-component 그대로"라는 결과는
            # 동일하다.
            # [2026-08-14] avoid_hold(§2.32)가 True면 위 스킵을 무시하고 클리핑을 강제
            # 되살리던 예외였으나, [2026-08-17] da 단독 검출 테스트 중엔 노란선/ll을 완전히
            # 무시하고 싶다는 요청으로 이 예외를 없앴다 — avoid_hold 중에도 항상 스킵.
            # avoid_hold_active 자체(§2.32 유예 타이머)는 ENABLE_BEHAVIOR와 무관하게 계속
            # 갱신되므로, 이 예외를 살려두면 라이다가 전방 장애물을 잡을 때마다 테스트 중에도
            # 조용히 클리핑이 되살아나 "노란선 무시"가 깨졌었다. da 클리핑을 다시 켜고 싶으면
            # DL_DA_SKIP_LL_CLIP=False로 되돌리거나(모든 프레임 클리핑 적용) 이 예외를 복원할 것.
            if DL_CENTER_MODE == 'da' and DL_DA_SKIP_LL_CLIP:
                self.da_ll_virtual_clip_used = False
                self.da_ll_clip_skipped = True
                self.da_clip_band_virtual = [None] * self.n_slices
                self.da_clip_cut_left_x = [None] * self.n_slices
                self.da_clip_cut_right_x = [None] * self.n_slices
                self.da_clip_bias_px = [None] * self.n_slices
            else:
                clipped, self.da_ll_virtual_clip_used = self._clip_da_by_ll(
                    da_mask, ll_mask_for_clip, ref_x, direction_hint=direction_hint)
                clipped_valid = sum(1 for c in self._slice_centers(clipped, 0, (0, 255, 0)) if c is not None)
                self.da_ll_clip_skipped = clipped_valid < self.slice_fit_min
                da_mask = da_mask if self.da_ll_clip_skipped else clipped
                if self.da_ll_clip_skipped:
                    # 이번 클리핑 자체가 통째로 버려졌으니(위 "차선책" 주석) 밴드별
                    # ①/②(실측/가상경계) 태그도 "실제로 적용 안 된" 시도 결과라 그대로 보여주면
                    # 오해를 살 수 있다 — 전부 비워서 visualize()가 아무 것도 안 그리게 한다.
                    self.da_clip_band_virtual = [None] * self.n_slices

            # 밴드별 중심점 — DL_CENTER_MODE='da'면 da 무게중심(_slice_centers(),
            # cv2.moments)을 그대로 쓴다. [2026-08-10] 한때 좌우 경계 중점
            # (_slice_edge_midpoints())으로 바꿔봤는데(갓길 등 여백이 비대칭일 때
            # 무게중심이 그쪽으로 쏠리는 문제를 없애려는 목적), 실차 주행 결과 오히려
            # 더 나빠졌다("S자로 좌우 왔다갔다") — 왼쪽/오른쪽 끝 열 딱 2개 값만 보다
            # 보니, 세그멘테이션 노이즈로 마스크 가장자리에 픽셀 하나만 튀어도 경계가
            # 그만큼 통째로 밀려서 매 프레임 크게 흔들렸던 것으로 보인다(README §2.12
            # "알려진 한계"에 이미 이 위험을 적어뒀었음 — 실차에서 그대로 재현됨).
            # 무게중심은 반대로 "여백 쏠림"이라는 자체 문제가 있지만(픽셀 밀도 가중
            # 평균이라 비대칭 여백에 끌려감), 노이즈 픽셀 하나의 영향이 전체 평균에
            # 희석되어 프레임 간 흔들림은 훨씬 적다 — 지금은 "쏠림이 있지만 안정적인"
            # 쪽을 "안 쏠리지만 흔들리는" 쪽보다 우선한 것. 여백 쏠림 자체는
            # ①시드/②연속성(위 _largest_da_component() 참고)이 어느 정도 완화해준다는
            # 판단도 있다.
            # [2026-08-12] 무보정 전체 밴드 폭 무게중심(_slice_centers())을 탐색창
            # 버전(_da_slice_centers_windowed())으로 교체 — S자 커브에서 da가 과검출돼도
            # (§2.1/§2.2, §2.16에서 면적 상한 자체를 없앤 뒤로는 더욱) 창 밖 픽셀은
            # 애초에 무게중심 계산에 안 들어가게 한다. 무게중심(moments) 자체는 그대로
            # 쓴다 — 위에서 이미 "여백 쏠림보다 안정성 우선"이라고 정리한 결론은 바뀌지
            # 않았고, 이번 변경은 "그 무게중심을 볼 범위를 예측 위치 근방으로 좁힌다"는
            # 별개의 축이다(README §2.27).
            # 'll'이면 DL_LL_ALGO로 실제 추적 알고리즘을 고른다(둘 다 da 폴백 없음 —
            # 모듈 상단/config.py DL_CENTER_MODE/DL_LL_ALGO 주석, README §2.19 참고).
            #   'yw'(main 기본) : 노란 중앙선 + 한쪽 흰색 경계선을 _ll_yellow_white_centers()로.
            #   'lr'(이지유)    : 좌/우 흰선을 각각 독립 슬라이딩 윈도우로 _ll_slice_centers()로.
            # 안 쓰는 쪽 알고리즘의 디버그 필드(ll_band_case 계열/ll_band_reason)는 이전
            # 프레임(또는 DL_LL_ALGO를 실행 중 바꾼 경우) 값이 남아 visualize()가 엉뚱한
            # 태그를 그리지 않도록 매 프레임 중립값으로 리셋한다.
            if DL_CENTER_MODE == 'll':
                if DL_LL_ALGO == 'lr':
                    merged_centers, self.ll_band_used = self._ll_slice_centers(ll_white_mask, ref_x)
                    self.ll_band_degraded = [False] * len(merged_centers)
                    self.ll_band_case = ['?'] * len(merged_centers)
                    self.ll_degraded = False
                else:
                    merged_centers, self.ll_band_used = self._ll_yellow_white_centers(
                        ll_white_mask, ll_yellow_mask, ref_x
                    )
                    self.ll_band_reason = [None] * self.n_slices
            else:
                # [2026-08-20] da 근접 컷 — ll 클리핑 이후, 차폭 안전마진 침식 이전에
                # 적용한다(_apply_vehicle_margin()이 그 위에 다시 침식을 걸어주므로 레이어
                # 순서가 자연스럽게 쌓인다). ENABLE_OBSTACLE_CUT=False면 obstacle_cut_confirmed가
                # 항상 False라 여기서 사실상 아무 일도 안 한다(그대로 반환).
                da_mask = self._clip_da_by_obstacle(da_mask, obstacle_y_m, obstacle_cut_confirmed)

                # [2026-08-14] 중심선 계산에만 안전마진을 적용한다 — self.da_mask_roi(아래,
                # 디버그 시각화용)는 침식 전 원본을 그대로 담아야 "da가 실제로 어디까지
                # 검출됐는지"와 "마진 때문에 얼마나 물러났는지"를 구분해서 볼 수 있다.
                margined_da_mask = self._apply_vehicle_margin(da_mask, v_mps)
                merged_centers = self._da_slice_centers_windowed(margined_da_mask, ref_x)
                self.ll_band_used = [False] * len(merged_centers)
                self.ll_search_windows = []
                self.ll_band_reason = [None] * self.n_slices
                self.ll_band_degraded = [False] * len(merged_centers)
                self.ll_band_case = ['?'] * len(merged_centers)
                self.ll_degraded = False

        self.da_mask_roi = da_mask
        # [2026-08-19] da 모드에서만 근접 밴드를 이상치 검사에서 보호한다(①) —
        # lane_util.py _reject_outliers() protect_indices 주석 참고. 'll'/'ll_da'는
        # 이 문제의 전제(회피 중 원거리 여러 밴드가 함께 휘는 것)와 무관해 기존과
        # 동일하게 전부 검사한다.
        protect_near = range(self.near_slices) if DL_CENTER_MODE == 'da' else None
        self.centerline = self._reject_outliers(merged_centers, protect_indices=protect_near)

        if DL_CENTER_MODE == 'da':
            # [2026-08-19] ②(2차 안전판) — ①을 통과하고도(진짜 순간 미검출 등) 근접
            # 밴드가 비면, 마지막으로 실제 찾았던 위치(_da_prev_band_x[i], 못 찾은
            # 프레임엔 안 갱신되므로 자연히 "마지막 확인값"을 들고 있다)로
            # DL_NEAR_HOLD_MAX_FRAMES 프레임까지만 대신 채운다. 그 이상 넘기면 포기하고
            # (기존 _fit_and_sample_path()의 np.interp 공간적 hold로 자연 폴백)
            # near_band_stale=True로 노출한다 — config.py DL_NEAR_HOLD_MAX_FRAMES,
            # track_drive.py _lane_drive() 참고.
            self.near_band_stale = False
            slice_h = self.roi_h // self.n_slices
            for i in range(self.near_slices):
                if self.centerline[i] is not None:
                    self._near_hold_streak[i] = 0
                    self.near_band_held[i] = False
                    continue
                if (self._da_prev_band_x[i] is not None
                        and self._near_hold_streak[i] < DL_NEAR_HOLD_MAX_FRAMES):
                    y_high = self.roi_h - i * slice_h
                    y_low = 0 if i == self.n_slices - 1 else self.roi_h - (i + 1) * slice_h
                    self.centerline[i] = ((y_low + y_high) / 2.0, self._da_prev_band_x[i])
                    self._near_hold_streak[i] += 1
                    self.near_band_held[i] = True
                else:
                    self.near_band_held[i] = False
                    if self._near_hold_streak[i] >= DL_NEAR_HOLD_MAX_FRAMES:
                        self.near_band_stale = True
        else:
            self.near_band_held = [False] * self.n_slices
            self.near_band_stale = False

        # 중앙 노란 점선의 밴드별 무게중심 — 탐색창 없는 stateless 방식(_slice_centers(),
        # lane_util.py 참고)으로 뽑는다. 이미 _split_ll_by_yellow()가 컴포넌트 단위로
        # 노란선만 분리해뒀으므로, 좌/우처럼 다른 선과 헷갈릴 걱정이 없어 슬라이딩
        # 윈도우(탐색창 추적)가 굳이 필요 없다 — 점선이라 밴드 간 끊김이 잦은데, 탐색창을
        # 쓰면 끊긴 동안 커브를 돌았을 때 창이 다음 조각을 놓칠 위험만 커진다. 모드와
        # 무관하게 항상 계산한다(경로 계산엔 아직 안 쓰고, visualize() 디버그 표시용).
        self.yellow_band_centers = self._slice_centers(
            ll_yellow_mask, 0, (0, 255, 255), min_pixels=DL_LL_SIDE_MIN_PIXELS
        )

        # 노란 중앙선(lane_side 판정용) — 이제 밴드별로 나눌 필요 없이 hough_lane.py와
        # 동일하게 ROI 전체의 단순 평균 위치 하나만 뽑는다(경로 계산에는 안 쓰임).
        ys, xs = np.nonzero(yellow_roi)
        self.yellow_centers = [(float(np.mean(ys)), float(np.mean(xs)))] if len(xs) else []

        near_center = self._group_mean(self.centerline, self.near_slices, True)
        far_center = self._group_mean(self.centerline, self.far_slices, False)

        # [2026-08-18] ll sanity check(ll_coverage >= DL_LL_SANITY_MIN_RATIO) 삭제 — ll(차선)을
        # 더 이상 안 쓰기로 확정(요청 반영, SPEED_LL_DEGRADED 삭제와 같은 사유, README §2.42
        # 참고). da 근접 중심점 유무만으로 판정한다.
        lane_valid = near_center is not None

        offset = lookahead = 0.0
        if lane_valid:
            offset = near_center - self.vehicle_center_x
            far_ref = far_center if far_center is not None else near_center
            lookahead = far_ref - self.vehicle_center_x
        lane_center = self.vehicle_center_x + offset

        # 명시적 경로(웨이포인트) — da 밴드 중심점을 선형보간으로 이어 만든다. 유효 밴드가
        # 2개 미만이면 fitted_path가 None이 되고, 그 경우 _update_path()가 self.path를
        # 갱신하지 않아(직전 프레임 값 유지) offset/lane_offset과 동일한 "무효 프레임엔
        # 마지막 값 유지" 원칙을 따른다. 유효할 때도 그대로 대입하지 않고 직전 경로와
        # EMA 블렌딩한다(lane_util.PATH_EMA_ALPHA 주석 참고) — 조향이 매 프레임 새로
        # 피팅된 경로에 과민하게 반응하는 걸 막기 위함.
        # [2026-08-17][중대 버그 수정, §2.36] 원래 lane_valid(근접 밴드 필수)로 이 갱신을
        # 가드했다 — fitted_path는 근접 없이 원거리만으로도 만들어지는데(2점 이상이면
        # 충분) lane_valid는 그걸 몰라서, 내부 self.path가 외부 track_drive.py의
        # self.lane_path(lane_valid로 얼림)와 다른 조건으로 계속 흘러가다 유효 판정이
        # 복귀하는 순간 한 틱 만에 점프해 룩어헤드가 튀는 형태로 벽 충돌까지 재현됐었다
        # (카카오톡 녹화, 2026-08-17 13:03) — 그래서 lane_valid로 가드했었다.
        # [2026-08-17n 후속, §2.36 재발] 그런데 lane_valid를 그대로 쓰면 "근접만 일시
        # 안 보이고 원거리는 있는" 상황(급커브 진입)에서 원거리 정보를 아예 못 써서
        # 경로가 필요 이상으로 오래 얼어붙는 게 실차로 확인됐다(카카오톡 녹화,
        # 2026-08-17 15:44 — 좌회전 진입에서 근접 밴드만 몇 초간 비었는데 경로 전체가
        # 그동안 멈춤). self.path_ok(아래, _debounce_path_ok() 참고)는 근접 OR 원거리
        # 둘 중 하나만 있어도 통과시키면서도 offset과 같은 강도로 디바운스해 예전
        # 크래시 원인(내부/외부 조건 불일치)은 재발하지 않게 한다 — DLLaneDetector가
        # 이 값을 그대로 밖으로 노출해 track_drive.py의 self.lane_path도 같은 값으로
        # 가드한다(perc_lane() 참고), 이제 lane_valid는 offset/lane_center 전용이다.
        # [2026-08-18] ll sanity check(ll_coverage >= DL_LL_SANITY_MIN_RATIO) 삭제 — ll을
        # 더 이상 안 쓰기로 확정(위 lane_valid와 동일 사유, README §2.42 참고). near/far
        # 중심점 조건만 남아 오히려 완화되는 방향(path가 얼어붙는 빈도가 줄어듦).
        path_ok_raw = (near_center is not None or far_center is not None)
        self.path_ok = self._debounce_path_ok(path_ok_raw)

        fitted_path = self._fit_and_sample_path(
            [c for c in self.centerline if c is not None]
        )
        if self.path_ok:
            self._update_path(fitted_path)

        lane_valid, offset, lookahead, lane_center = self._debounce(
            lane_valid, offset, lookahead, lane_center
        )

        # [2026-08-10] 디버그 스파크라인용 — 실제로 조향에 쓰이는 디바운스 이후 최종
        # offset을 남긴다(디바운스 전 순간값이 아님 — 그건 조향에 안 쓰이므로 흔들림을
        # 봐도 실제 주행 흔들림과 무관할 수 있음).
        self._offset_history.append(offset)

        self.visualize(offset)

        return lane_valid, offset, lookahead, lane_center, self.path

    def visualize(self, offset):
        """da 전체(파랑, self.da_mask_all_roi)/실제 채택 da(초록/면적상한 차선책이면 주황/
        ll클리핑 건너뜀이면 청록)/ll(흰선=흰색, 노란선=노랑) 반투명 오버레이 + (DL_CENTER_MODE
        가 'll'일 때) 좌/우 슬라이딩 윈도우 탐색창 + ('ll_da'=corridor일 때) corridor 경계
        틱 + da 중심선 관측점 + 피팅된 경로 + offset/lane_center 텍스트를 self.vis에
        그려 넣기만 한다. ★ 여기서
        cv2.imshow()/cv2.waitKey()를 호출하면 안 된다 ★ 이 메서드는 DLLaneDetector의
        백그라운드 추론 스레드에서 호출되는데, OpenCV HighGUI(특히 GTK 백엔드)는 스레드
        세이프하지 않아서 메인 스레드(다른 디버그 창들이 이미 거기서 cv2.imshow/waitKey를
        부르고 있음)와 다른 스레드가 동시에 GUI를 건드리면 초반 몇 초는 멀쩡하다가 GTK
        이벤트루프가 통째로 멈추는 형태로 실차에서 재현됐다(freeze). 그래서 그리기
        (cv2.rectangle/circle/putText/addWeighted, 창 없이 이미지 버퍼에만 작동)만
        여기서 하고, 실제 창 표시는 DLLaneDetector.show_debug_windows()가 메인 스레드
        (perc_lane() 호출 시점)에서 담당한다."""
        if self.vis is None:
            return

        if DEBUG_VIZ_DL_LANE:
            overlay = self.vis.copy()
            # 모델이 "주행가능하다"고 본 da 전체(덩어리 선택/ll클리핑 전, self.da_mask_all_roi)를
            # 먼저 파란색으로 깔고, 그 위에 실제로 waypoint 추출에 쓰인 부분(self.da_mask_roi,
            # 아래 초록/주황/청록)을 덧그린다 — "모델이 본 전체 vs 실제로 채택한 부분"을 한
            # 화면에서 바로 비교할 수 있게 한다. da_mask_roi는 항상 da_mask_all_roi의 부분집합이라
            # (largest-component 선택 + ll 클리핑으로 줄어들기만 함) 겹치는 픽셀은 뒤에 그리는
            # 초록/주황/청록이 그대로 덮어써서 보인다.
            if self.da_mask_all_roi is not None:
                overlay[self.da_mask_all_roi > 0] = (255, 0, 0)  # 파랑 — da 전체(모델 원본 판단)
            # 차선책(최댓값 덩어리가 면적 상한에 걸려 그다음 덩어리를 대신 쓴 프레임)은
            # 주황, ll 클리핑을 건너뛴 프레임(클리핑하면 밴드가 너무 줄어드는 경우)은
            # 청록으로 표시해 초록(정상)과 구분한다 — _largest_da_component()/detect()
            # 주석 참고. 두 상황이 겹치면(둘 다 발동) 주황을 우선 표시한다.
            if self.da_fallback_used:
                da_color = (0, 140, 255)      # 주황
            elif self.da_ll_clip_skipped:
                da_color = (255, 255, 0)      # 청록
            else:
                da_color = (0, 200, 0)        # 초록(정상)
            # 차선책을 쓴 프레임엔 실제로 버려진 면적 1위 덩어리도 원래색(초록)으로 같이
            # 그려서, "원래대로였다면 뭘 골랐을지"와 "실제로 채택한 차선책"을 한 화면에서
            # 바로 비교할 수 있게 한다. 서로 다른 connected component라 픽셀이 겹치지
            # 않으므로 어느 순서로 칠해도 서로를 덮어쓰지 않는다.
            if self.da_fallback_used and self.da_largest_mask_roi is not None:
                overlay[self.da_largest_mask_roi > 0] = (0, 200, 0)
            if self.da_mask_roi is not None:
                overlay[self.da_mask_roi > 0] = da_color  # 주행가능영역(실제 채택분)
            # ll을 흰선/노란선 분리 결과(_split_ll_by_yellow())대로 실제 색과 맞춰
            # 칠한다 — "이 라인이 지금 흰선/노란선 중 뭘로 인식되고 있는지"를 색만
            # 보고 바로 알 수 있게.
            if self.ll_white_mask_roi is not None:
                overlay[self.ll_white_mask_roi > 0] = (255, 255, 255)  # 흰선
            if self.ll_yellow_mask_roi is not None:
                overlay[self.ll_yellow_mask_roi > 0] = (0, 255, 255)   # 노란선
            cv2.addWeighted(overlay, 0.35, self.vis, 0.65, 0, dst=self.vis)

            # [2026-08-20] da 근접 컷(_clip_da_by_obstacle(), ENABLE_OBSTACLE_CUT) — 실차
            # 주행 중 "지금 자르고 있는지"를 한눈에 보려는 목적으로, 마젠타 반투명 채움 +
            # 굵은 "CUT" 라벨을 실제 자른 열(px) 범위에 그린다(윤곽선만이던 이전 버전보다
            # 더 눈에 띄게 — 운전하면서 흘끗 봐도 알아채야 해서). da_clip_band_virtual
            # (①/② 틱, ll 클리핑용)과는 별개 오버레이 — 근접 컷은 밴드 단위가 아니라 한
            # 사각형이라 별도로 그린다. 컷이 안 걸린 프레임에도(꺼짐/트리거 전) 좌상단에
            # "OBSTACLE CUT: enabled/off" 한 줄은 항상 띄워, 기능 자체가 켜져 있는지부터
            # 확인할 수 있게 한다.
            cut_status = 'enabled(대기)' if ENABLE_OBSTACLE_CUT else 'off'
            cv2.putText(self.vis, f'OBSTACLE CUT: {cut_status}', (8, self.roi_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            if self.obstacle_cut_active and self.obstacle_cut_col_range is not None:
                x0, x1 = self.obstacle_cut_col_range
                near_row_px = int(np.clip(self.roi_h - OBSTACLE_CUT_NEAR_M * DL_PIXELS_PER_METER, 0, self.roi_h))
                cut_fill = self.vis.copy()
                cv2.rectangle(cut_fill, (x0, 0), (x1, near_row_px), (255, 0, 255), -1)
                cv2.addWeighted(cut_fill, 0.4, self.vis, 0.6, 0, dst=self.vis)
                cv2.rectangle(self.vis, (x0, 0), (x1, near_row_px), (255, 0, 255), 2)
                label_y = min(self.roi_h - 10, near_row_px + 18)
                cv2.putText(self.vis, 'CUT', (x0 + 4, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(self.vis, 'OBSTACLE CUT: ACTIVE', (8, self.roi_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 0, 255), 2, cv2.LINE_AA)

            # corridor(DL_CENTER_MODE='ll_da') 전용: 밴드별로 채택된 corridor 경계
            # (1/3번째 ll 선)를 자홍색 세로 틱으로 표시 — sanity check를 통과해
            # self.corridor_bounds[i]가 채워진 밴드만 그려진다(선 부족/폭 이상으로
            # 드롭된 밴드는 틱이 안 보임 = 그 밴드가 왜 무효인지 바로 알 수 있다).
            if DL_CENTER_MODE == 'll_da':
                slice_h = self.roi_h // self.n_slices
                for i, bounds in enumerate(self.corridor_bounds):
                    if bounds is None:
                        continue
                    y_high = self.roi_h - i * slice_h
                    y_low = 0 if i == self.n_slices - 1 else self.roi_h - (i + 1) * slice_h
                    for bx in bounds:
                        bx_i = int(np.clip(bx, 0, self.roi_w - 1))
                        cv2.line(self.vis, (bx_i, y_low), (bx_i, y_high), (255, 0, 255), 2)

            # [2026-08-10] da/ll 클리핑(_clip_da_by_ll(), 'da'/'ll' 공통, 'll_da'=corridor는
            # 클리핑 자체를 안 해서 self.da_clip_band_virtual이 전부 None)이 밴드별로
            # ①(실측/잔상 ll)/②(가상경계)중 뭘 썼는지를 화면 왼쪽 끝 세로 띠에 작은 틱으로
            # 표시한다 — 초록=①(근거 있음), 주황=②(근거 없이 기대 차로폭으로 강제 클리핑,
            # `self._ll_active_half_width()` 기반). 프레임 전체 요약 태그([LL_VIRTUAL])는
            # "이번 프레임에 한 번이라도 발동했는지"만 알려주는데, 이 틱은 정확히 몇 번째
            # 밴드(=화면 어느 높이)에서 발동했는지 바로 보여준다. 클리핑 자체가 통째로
            # 버려진 프레임([LL_CLIP_SKIP])은 detect()가 이 리스트를 전부 None으로 비워둬서
            # 틱이 하나도 안 그려진다(오해 방지).
            slice_h = self.roi_h // self.n_slices
            for i, is_virtual in enumerate(self.da_clip_band_virtual):
                if is_virtual is None:
                    continue
                y_high = self.roi_h - i * slice_h
                y_low = 0 if i == self.n_slices - 1 else self.roi_h - (i + 1) * slice_h
                color = (0, 140, 255) if is_virtual else (0, 220, 0)
                cv2.rectangle(self.vis, (0, y_low), (5, max(y_high - 1, y_low)), color, -1)

            # [2026-08-17] 위 왼쪽 끝 틱은 "①/②중 뭘 썼는지"만 보여주고 "실제로 몇 px가
            # 잘렸는지"는 안 보여준다 — avoid-hold ll클리핑 마진(DL_LL_CLIP_MARGIN_PX)/
            # 방향 힌트 바이어스(AVOID_HOLD_DIR_BIAS_PX)가 튜닝값대로 동작하는지 실차에서
            # 바로 확인할 수 있게, 밴드가 실제로 잘려나간 x좌표에 세로선을 그린다(틱과 같은
            # 색 규약: 초록=①, 주황=②). avoid-hold 바이어스가 적용된 밴드는 그 옆에 부호
            # 포함 px를 텍스트로 같이 찍는다.
            for i, (lx, rx) in enumerate(zip(self.da_clip_cut_left_x, self.da_clip_cut_right_x)):
                if lx is None and rx is None:
                    continue
                y_high = self.roi_h - i * slice_h
                y_low = 0 if i == self.n_slices - 1 else self.roi_h - (i + 1) * slice_h
                color = (0, 140, 255) if self.da_clip_band_virtual[i] else (0, 220, 0)
                if lx is not None:
                    cv2.line(self.vis, (lx, y_low), (lx, max(y_high - 1, y_low)), color, 1)
                if rx is not None:
                    cv2.line(self.vis, (rx, y_low), (rx, max(y_high - 1, y_low)), color, 1)
                bias = self.da_clip_bias_px[i]
                if bias:
                    tx = (rx if rx is not None else lx) + 4
                    cv2.putText(
                        self.vis, f'{bias:+.0f}px', (tx, int((y_low + y_high) / 2) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1
                    )

            # [2026-08-10 병합] 탐색창 시각화 — DL_LL_ALGO에 따라 self.ll_search_windows
            # 튜플의 의미가 다르다(둘 다 8-tuple이지만 'lr'은
            # (y_low,y_high,lx0,lx1,lx,rx0,rx1,rx), 'yw'는
            # (y_low,y_high,yx0,yx1,yx,wx0,wx1,wx)) — 그려주는 로직도 완전히 분기한다.
            if DL_LL_ALGO == 'lr':
                # 좌/우 슬라이딩 윈도우 탐색창(_ll_slice_centers()가 이번 프레임에 훑은
                # 범위) — 찾았으면 초록, 못 찾았으면(창 안에 픽셀 부족) 회색 테두리로
                # 구분한다. 연속 미검출로 탐색창이 확장된 상태(DL_LL_SEARCH_WIDEN_STEP_PX
                # 적용분, `_ll_slice_centers()` 참고)면 기본 반경(DL_LL_SEARCH_HALF_WIDTH_PX)
                # 보다 눈에 띄게 넓은 사각형 자체가 이미 "이 밴드는 몇 프레임째 못 찾고
                # 있다"는 신호라 별도 표시 없이도 폭으로 드러난다 — 다만 놓치기 쉬우니
                # 확장된 창은 주황 테두리로 강조한다(찾았든 못 찾았든). 밴드별 실측
                # 차로폭(rx-lx) 텍스트는 DL_LL_WIDTH_MIN_PX~MAX_PX 튜닝용 — 범위 안이면
                # 초록(채택), 밖이면 빨강(그 밴드는 버려짐)으로 색을 나눈다.
                widen_epsilon = 1.0  # np.clip 반올림 오차 흡수용 여유
                for i, (y_low, y_high, lx0, lx1, lx, rx0, rx1, rx) in enumerate(self.ll_search_windows):
                    l_widened = (lx1 - lx0) > 2 * DL_LL_SEARCH_HALF_WIDTH_PX + widen_epsilon
                    r_widened = (rx1 - rx0) > 2 * DL_LL_SEARCH_HALF_WIDTH_PX + widen_epsilon
                    l_color = (0, 140, 255) if l_widened else ((0, 255, 0) if lx is not None else (120, 120, 120))
                    r_color = (0, 140, 255) if r_widened else ((0, 255, 0) if rx is not None else (120, 120, 120))
                    cv2.rectangle(self.vis, (lx0, y_low), (lx1, max(y_high - 1, y_low)), l_color, 1)
                    cv2.rectangle(self.vis, (rx0, y_low), (rx1, max(y_high - 1, y_low)), r_color, 1)
                    if lx is not None and rx is not None:
                        width = rx - lx
                        in_range = DL_LL_WIDTH_MIN_PX < width < DL_LL_WIDTH_MAX_PX
                        cv2.putText(
                            self.vis, f'{width:.0f}px', (int(rx1) + 4, int((y_low + y_high) / 2) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 255, 0) if in_range else (0, 0, 255), 1
                        )
                    # [2026-08-10] 밴드별 프레임 간 앵커링(DL_LL_BAND_ANCHOR_ALPHA) 디버그 —
                    # 이번 프레임이 실제로 앵커로 끌어당긴 "직전 프레임 그 밴드 위치"를
                    # 마젠타 점으로 찍는다. 이 점이 이번 프레임 실제 검출(lx/rx, 사각형
                    # 중심 부근)과 많이 벌어져 있으면 앵커링이 오히려 창을 엉뚱한 쪽으로
                    # 당기고 있다는 뜻이니 DL_LL_BAND_ANCHOR_ALPHA를 낮출 근거가 된다.
                    y_c = int((y_low + y_high) / 2)
                    anchor_l = self.ll_band_anchor_left[i] if i < len(self.ll_band_anchor_left) else None
                    if anchor_l is not None:
                        cv2.drawMarker(self.vis, (int(np.clip(anchor_l, 0, self.roi_w - 1)), y_c),
                                        (255, 0, 255), markerType=cv2.MARKER_SQUARE, markerSize=6, thickness=1)
                    anchor_r = self.ll_band_anchor_right[i] if i < len(self.ll_band_anchor_right) else None
                    if anchor_r is not None:
                        cv2.drawMarker(self.vis, (int(np.clip(anchor_r, 0, self.roi_w - 1)), y_c),
                                        (255, 0, 255), markerType=cv2.MARKER_SQUARE, markerSize=6, thickness=1)
                    # [2026-08-10] 밴드별 채택 근거 태그 — 'B'=양쪽검출/채택, 'X'=양쪽검출됐지만
                    # 폭(DL_LL_WIDTH_MIN~MAX_PX) 밖이라 거부, 'L'/'R'=편측만 검출(반대쪽은
                    # self._ll_half_width로 추정), '-'=양쪽 다 못 찾음. 왼쪽 사각형 바로
                    # 왼편에 붙여서, 색(초록/회색 테두리)만으론 안 보이던 "왜 이 색인지"를
                    # 글자로 확인.
                    reason = self.ll_band_reason[i] if i < len(self.ll_band_reason) else None
                    if reason is not None:
                        reason_color = {
                            'B': (0, 255, 0), 'L': (255, 255, 0), 'R': (255, 255, 0),
                            'X': (0, 0, 255), '-': (120, 120, 120),
                        }.get(reason, (255, 255, 255))
                        cv2.putText(
                            self.vis, reason, (max(0, lx0 - 14), y_c + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, reason_color, 1
                        )
            else:
                # 노란선/흰선 탐색창(_ll_yellow_white_centers()가 이번 프레임에 훑은 범위)
                # — 노란 창은 노란색, 흰 창은 흰색 테두리로 찾았을 때 표시하고, 못
                # 찾았으면(창 안에 픽셀 부족) 회색으로 구분. [2026-08-10] 연속 미검출로
                # 그 사이드 반경이 기본값(DL_LL_SEARCH_HALF_WIDTH_PX)보다 넓어진 상태면
                # ('lr' 모드와 동일한 표시 관례) 주황 테두리로 강조한다 — 특히 흰 창이
                # 자주 주황으로 뜨면 급조향 복귀 구간에서 확장이 실제로 발동하고 있다는
                # 뜻. 두 창 사이의 실측 간격(px, self._white_yellow_gap_px 실측/재구성용)
                # 텍스트를 같이 찍는다.
                widen_epsilon = 1.0  # np.clip 반올림 오차 흡수용 여유
                for (y_low, y_high, yx0, yx1, yx, wx0, wx1, wx) in self.ll_search_windows:
                    y_widened = (yx1 - yx0) > 2 * DL_LL_SEARCH_HALF_WIDTH_PX + widen_epsilon
                    w_widened = (wx1 - wx0) > 2 * DL_LL_SEARCH_HALF_WIDTH_PX + widen_epsilon
                    y_color = (0, 140, 255) if y_widened else ((0, 255, 255) if yx is not None else (120, 120, 120))
                    w_color = (0, 140, 255) if w_widened else ((255, 255, 255) if wx is not None else (120, 120, 120))
                    cv2.rectangle(self.vis, (yx0, y_low), (yx1, max(y_high - 1, y_low)), y_color, 1)
                    cv2.rectangle(self.vis, (wx0, y_low), (wx1, max(y_high - 1, y_low)), w_color, 1)
                    if yx is not None and wx is not None:
                        gap = abs(wx - yx)
                        cv2.putText(
                            self.vis, f'{gap:.0f}px',
                            (int(max(yx1, wx1)) + 4, int((y_low + y_high) / 2) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1
                        )

            # 중앙 노란 점선의 밴드별 무게중심(detect()에서 self._slice_centers()로
            # 뽑아둔 self.yellow_band_centers) — 아직 경로 계산엔 안 쓰지만, 어디를
            # "노란 중앙선"으로 보고 있는지 확인할 수 있게 노란 다이아몬드로 표시한다.
            for c in self.yellow_band_centers:
                if c is None:
                    continue
                y, cx = c
                pt = (int(np.clip(cx, 0, self.roi_w - 1)), int(y))
                cv2.drawMarker(self.vis, pt, (0, 255, 255), markerType=cv2.MARKER_DIAMOND,
                               markerSize=10, thickness=2)

        # 밴드별 중심점 — DL_CENTER_MODE='ll'&&DL_LL_ALGO='yw'에서 정상 검출(②)된 밴드는
        # 흰색, 저신뢰 추정(③④⑤, self.ll_band_degraded)된 밴드는 주황으로 구분해서 어느
        # 밴드가 간격 재구성/잔상에 의존했는지 한눈에 보이게 한다. 'lr'/'da' 모드에선
        # ll_band_degraded가 항상 전부 False라 전부 주황('da'는 기존 방식과 동일 표시,
        # 'lr'은 이 개념 자체가 없어 그냥 무시). 'll_da'=corridor 모드에선 ll_band_used가
        # "이 밴드가 corridor로 채택됐는지"를 뜻하므로 흰색=채택, 주황은 애초에 나오지
        # 않는다.
        # draw_centers()는 단일 색만 지원해 여기선 직접 그린다.
        pts = [
            (int(np.clip(cx, 0, self.roi_w - 1)), int(y))
            for c in self.centerline if c is not None
            for (y, cx) in [c]
        ]
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(self.vis, p1, p2, (0, 255, 255), 2)
        for i, c in enumerate(self.centerline):
            if c is None:
                continue
            y, cx = c
            pt = (int(np.clip(cx, 0, self.roi_w - 1)), int(y))
            ll_used = i < len(self.ll_band_used) and self.ll_band_used[i]
            cv2.circle(self.vis, pt, 4, (255, 255, 255) if ll_used else (0, 140, 255), -1)
            # [2026-08-10] "지금 SW가 3분기(2W/1W/LOST) 중 뭘로 주행중인지" 실차에서
            # 바로 보이게, 밴드별 분기 태그(_ll_yellow_white_centers() 참고)를 점 옆에
            # 그린다. DL_CENTER_MODE='ll'&&DL_LL_ALGO='yw'에서만 의미 있는 값이라 그때만
            # 그린다('lr'로 돌 때는 detect()가 이 리스트를 전부 '?'로 리셋해두므로, 안
            # 걸러내면 밴드마다 의미 없는 "?"만 찍혀 화면이 지저분해진다).
            if DL_CENTER_MODE == 'll' and DL_LL_ALGO == 'yw' and i < len(self.ll_band_case):
                cv2.putText(
                    self.vis, self.ll_band_case[i], (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255) if not ll_used else (255, 255, 255), 1
                )
        self.draw_path(self.path)  # 피팅된 최종 경로(자홍색)

        # [2026-08-17] 이 빨간 세로선은 단순 "화면 중앙"이 아니다 — track_drive.py
        # _lane_steer()가 pure_pursuit에 넘기는 vehicle_x 기본값(self.vehicle_center_x,
        # vehicle_x=0.0을 명시로 넘기는 라바콘 등 일부 예외 제외)과 정확히 같은 값이라, 이
        # 줄이 곧 "차량이 지금 자기 위치라고 믿는 x좌표" 그 자체다. 자홍색 경로(self.path)가
        # 이 줄에서 얼마나 벗어나 있는지가 pure_pursuit이 보는 실제 횡편차와 같다는 걸
        # 라벨로 명시해, "이게 그냥 화면 중앙 참고선"이라고 오해하지 않게 한다.
        # [같은 날 후속 수정] vehicle_center_x는 더 이상 roi_w//2(캔버스 단순 절반)가
        # 아니라 BEV 사다리꼴 실측 기하로 구한 실제 차량 중심이다(모듈 상단
        # DL_BEV_VEHICLE_CENTER_X 주석 참고) — 그 값을 그대로 이 줄의 위치로 쓴다.
        vehicle_center_x_i = int(round(self.vehicle_center_x))
        cv2.line(self.vis, (vehicle_center_x_i, 0), (vehicle_center_x_i, self.roi_h), (0, 0, 255), 1)
        cv2.putText(
            self.vis, 'vehicle_x', (vehicle_center_x_i + 4, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
        )
        lane_center = self.vehicle_center_x + offset
        ll_band_count = sum(1 for u in self.ll_band_used if u)
        tags = ''
        if self.da_fallback_used:
            tags += ' [FALLBACK]'
        if self.da_ll_clip_skipped:
            tags += ' [LL_CLIP_SKIP]'
        if self.da_ll_virtual_clip_used:
            tags += ' [LL_VIRTUAL]'
        if self.avoid_hold_active:
            dir_tag = {1: 'R', -1: 'L', 0: '-'}[self.avoid_hold_dir_hint]
            tags += f' [AVOID_HOLD dir:{dir_tag}]'
        if self.ll_degraded:
            tags += ' [LL_DEGRADED]'
        if any(self.near_band_held[:self.near_slices]):
            tags += ' [NEAR_HELD]'
        if self.near_band_stale:
            tags += ' [NEAR_STALE]'
        # 모드마다 밴드 카운트가 뜻하는 바가 달라서 라벨/부가정보를 따로 붙인다 —
        # 'll_da'(corridor)는 corridor로 채택된 밴드 수, 'll'은 DL_LL_ALGO에 따라 서로
        # 다른 부가정보(아래), 'da'는 항상 0(ll 미사용, 기존 방식과 동일 표시).
        branch_summary_line = None  # DL_LL_ALGO='yw'일 때만 채워짐 — MODE 배너 아래 별도 줄로 그림(아래)
        if DL_CENTER_MODE == 'll_da':
            extra = f'corridor_bands:{ll_band_count}/{self.n_slices}'
        elif DL_CENTER_MODE == 'll':
            yellow_band_count = sum(1 for c in self.yellow_band_centers if c is not None)
            if DL_LL_ALGO == 'lr':
                # [2026-08-10] Lvel/Rvel(px/밴드) — _ll_slice_centers()가 추적하는 좌/우
                # 속도 예측 EMA 실측값. 급커브 구간에서 이 값이 DL_LL_VELOCITY_MAX_PX
                # 근처에 계속 붙어있으면 클램프가 실제 곡률을 못 따라간다는 뜻이니 그
                # 값을 올릴 근거가 된다.
                extra = (
                    f'white_bands:{ll_band_count}/{self.n_slices} '
                    f'yellow_bands:{yellow_band_count}/{self.n_slices} '
                    f'lane_w_est:{self._ll_half_width * 2:.0f}px '
                    f'Lvel:{self._ll_left_velocity:+.1f} Rvel:{self._ll_right_velocity:+.1f}'
                )
            else:
                degraded_count = sum(1 for d in self.ll_band_degraded if d)
                extra = (
                    f'ok_bands:{ll_band_count}/{self.n_slices} '
                    f'degraded:{degraded_count}/{self.n_slices} '
                    f'yellow_bands:{yellow_band_count}/{self.n_slices} '
                    f'gap:{self._white_yellow_gap_px:.0f}px side:{self.lane_side}'
                )
                # [2026-08-10] "지금 SW가 어느 분기로 주행중인지" 한눈에 보이게, 노란선
                # 없을 때의 3분기(케이스1=2W/케이스2=1W:L·1W:R/케이스3=LOST) + 기존
                # 분기(Y+W/Y+gap) 밴드 수를 따로 한 줄 더 찍는다. 밴드별 태그는 위
                # centerline 점 옆에도 그려지지만(개별 밴드 확인용), 이건 이번 프레임
                # 전체가 어느 분기에 몰려있는지 한눈에 보기 위한 요약.
                case_counts = {}
                for c in self.ll_band_case:
                    case_counts[c] = case_counts.get(c, 0) + 1
                branch_summary_line = ' '.join(f'{k}:{v}' for k, v in sorted(case_counts.items()))
        else:
            extra = f'll_bands:{ll_band_count}/{self.n_slices}'

        # [2026-08-10] 지금 어느 DL_CENTER_MODE로 주행 중인지 한눈에 보이게 result 패널
        # 맨 위에 색 배너를 깐다 — 기존엔 하단 텍스트 줄 안에 `mode:xx`로만 섞여 있어서
        # 다른 정보 사이에서 놓치기 쉬웠다(da/ll/ll_da를 실차에서 A/B로 계속 바꿔가며
        # 테스트할 예정이라 지금 뭘 보고 있는지 착각하면 튜닝값을 엉뚱한 모드에 반영하는
        # 사고로 이어짐). 모든 다른 오버레이보다 나중에(맨 위에) 그려서 절대 안 가려지게
        # 한다.
        mode_color = DL_MODE_COLORS.get(DL_CENTER_MODE, DL_MODE_COLOR_DEFAULT)
        cv2.rectangle(self.vis, (0, 0), (self.roi_w - 1, 22), mode_color, -1)
        cv2.putText(
            self.vis, f'MODE: {DL_CENTER_MODE.upper()}', (8, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            self.vis,
            f'offset:{offset:+.1f} center:{lane_center:.1f} ll_cov:{self.ll_coverage:.3f} '
            f'mode:{DL_CENTER_MODE} {extra}{tags}',
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )
        # [2026-08-10 병합] DL_LL_ALGO='yw'일 때만 채워지는 밴드 분기 요약 — MODE 배너
        # 도입으로 원래 (10,40)에 그리던 걸 위 상태 줄이 차지하게 돼서 그 아래(60)로
        # 한 줄 밀었다(README §2.19).
        if branch_summary_line is not None:
            cv2.putText(
                self.vis, f'branch: {branch_summary_line}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )

        # [2026-08-11] 예전엔 여기서 _build_params_panel()이 'dl_lane_params'라는 별도
        # 창을 만들어 지금 DL_CENTER_MODE의 튜닝값(대부분 config 고정값, 일부만 런닝
        # 추정치)을 텍스트로 늘어놓고 그 아래 offset 스파크라인을 붙였다 — 그런데 실차
        # 테스트 중 매 프레임 봐야 하는 건 "차선이 흔들리고 있는가"(스파크라인)뿐이고
        # 나머지 텍스트는 코드/config.py를 보면 알 수 있는 값이라 화면만 차지했다.
        # 텍스트 목록은 통째로 지우고, 스파크라인만 'dl_lane' 창(result/da/ll/yellow
        # vconcat) 맨 아래에 같이 붙이도록 show_debug_windows()로 넘긴다 — 폭은 그
        # 패널들과 같은 self.vis.shape[1](= self.roi_w)로 맞춘다.
        self.offset_sparkline_img = self._build_offset_sparkline(self.vis.shape[1])

    def _build_offset_sparkline(self, width, height=70):
        """[2026-08-10] 최근 self._offset_history(최대 DL_DEBUG_HISTORY_LEN프레임, 디바운스
        이후 최종 offset)를 선 그래프로 그린다. README §2.12 "S자로 좌우 왔다갔다" 같은
        프레임 간 흔들림은 순간 텍스트 값만 봐서는 "지금 떨고 있다"는 걸 알아채기 어려운데,
        최근 값을 이어서 그리면 진폭이 그대로 보인다. y축 스케일은 고정하지 않고 이번
        창(window) 안의 |offset| 최댓값에 맞춰 자동으로 잡는다 — 고정 스케일이면 조용한
        구간에서는 그래프가 거의 평평해 보여서 미세한 흔들림을 놓치기 쉽다(대신 "지금 이
        그래프가 몇 px 스케일인지"는 우측 하단 max|.| 텍스트로 항상 같이 보여준다)."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)
        cv2.putText(panel, 'offset history (debounced, px)', (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1, cv2.LINE_AA)

        hist = list(self._offset_history)
        graph_top = 18
        graph_h = height - graph_top - 16
        zero_y = graph_top + graph_h // 2
        cv2.line(panel, (0, zero_y), (width, zero_y), (80, 80, 80), 1)

        if len(hist) >= 2:
            max_abs = max(1.0, max(abs(v) for v in hist))
            n = len(hist)
            pts = [
                (int(idx / (n - 1) * (width - 1)), int(zero_y - (v / max_abs) * (graph_h / 2)))
                for idx, v in enumerate(hist)
            ]
            for p1, p2 in zip(pts, pts[1:]):
                cv2.line(panel, p1, p2, (0, 255, 255), 1)
            cv2.putText(
                panel, f'now:{hist[-1]:+.1f} max|.|:{max_abs:.1f} n={n}',
                (6, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
            )
        return panel


class DLLaneDetector:
    """HoughLaneDetector/perc_floor.LaneDetector와 동일한
        detect(frame) -> (lane_valid, lane_offset, lane_lookahead, lane_center, path, debug_img)
    인터페이스를 제공하는 DL(TwinLiteNet) 백엔드. track_drive.py의 perc_lane()은 수정 없이
    그대로 재사용된다(roi_w/yellow_centers 속성도 _update_lane_side() 호환용으로 노출).

    추론은 별도 데몬 스레드에서 자기 페이스로 돌고, detect()는 항상 논블로킹으로 최신
    결과를 반환한다 — 실시간 전략에 대한 설계 근거는 모듈 상단 주석 참고.
    """

    def __init__(self, model_path=None, providers=None, logger=None):
        self.engine = TwinLiteNetEngine(model_path=model_path, providers=providers, logger=logger)
        self._slide = DLSlideWindow()
        self._logger = logger

        # [2026-08-17] 첫 프레임 처리 전까지만 쓰이는 초기 플레이스홀더 — _worker()가 매
        # 추론마다 self._slide.roi_w(실제 da_mask/path 좌표계 폭)로 덮어쓴다. 예전엔 이
        # DL_INPUT_W(640, 모델 입력 고정폭)가 계속 유지돼서 _lane_steer()/_update_lane_side()가
        # 잘못된 폭으로 vehicle_x/차선판정 기준을 잡는 버그가 있었다(위 _worker() 주석 참고).
        self.roi_w = DL_INPUT_W          # _update_lane_side()가 참조하는 정규화 분모
        self.yellow_centers = []         # _update_lane_side() 호환용, 워커가 매 추론마다 갱신

        default_center = DL_INPUT_W / 2.0
        # [2026-08-17] _lane_steer()/_update_lane_side()가 roi_w/2.0 대신 참조하는 실제
        # 차량 중심 x좌표 — DLSlideWindow.vehicle_center_x와 동일한 값을 워커가 매 추론마다
        # 동기화한다(아래 _worker() 참고). 첫 프레임 전까지는 이 플레이스홀더를 쓴다.
        self.vehicle_center_x = DL_BEV_VEHICLE_CENTER_X if DL_USE_BEV else default_center
        self._lock = threading.Lock()
        self._latest_frame = None
        # [2026-08-14] avoid-hold(§2.32) — track_drive.py의 perc_lane()이 매 틱
        # set_avoid_hold()로 갱신하고, _worker()가 다음 추론 때 _latest_frame과 같은
        # 락으로 같이 읽어서 DLSlideWindow.detect()에 넘긴다(config.py
        # AVOID_HOLD_TRIGGER_DIST_M/AVOID_HOLD_SEC_* 주석 참고).
        self._latest_avoid_hold = False
        # [2026-08-15] 적용3 — set_avoid_hold()가 side도 같이 받아 저장해두면 _worker()가
        # 다음 추론 때 avoid_hold와 같은 락으로 같이 읽어 DLSlideWindow.detect()에 넘긴다.
        self._latest_avoid_hold_side = 0
        # [2026-08-17g] 현재 속도(m/s) — track_drive.py가 매 틱 set_speed()로 갱신하면
        # _worker()가 avoid_hold와 같은 락으로 같이 읽어 DLSlideWindow.detect()에 넘긴다
        # (da 안전마진의 속도비례 "뒤" 마진 계산용, config.py DL_DA_REAR_MARGIN_* 참고).
        self._latest_v_mps = 0.0
        # [2026-08-15] 적용2 — _worker()가 매 추론 후 DLSlideWindow.da_area_jump_detected를
        # 여기로 복사해둔다. result_seq/yellow_centers와 동일한 관례로, 단순 bool 대입이라
        # GIL 하에서 원자적이므로 별도 락 없이 읽는다(track_drive.py _update_avoid_hold()가
        # getattr(self.lane_detector, 'da_area_jump', False)로 조회).
        self.da_area_jump = False
        # [2026-08-17n] self._slide.path_ok(§2.36 재발 수정)를 그대로 복사해 노출 —
        # da_area_jump와 동일한 관례(getattr로 조회, GIL 하에서 원자적이라 락 불필요).
        # track_drive.py perc_lane()이 getattr(self.lane_detector, 'path_ok', valid)로
        # 읽어 self.lane_path 갱신을 가드한다 — hough/classic_cv처럼 이 속성이 없는
        # 백엔드는 getattr 기본값(valid)으로 조용히 폴백해 기존 동작 그대로 유지된다.
        self.path_ok = False
        # [2026-08-20] da 근접 컷(_clip_da_by_obstacle(), ENABLE_OBSTACLE_CUT) — track_drive.py가
        # 매 틱 set_obstacle()로 갱신하면 _worker()가 다음 추론 때 avoid_hold와 같은 락으로
        # 같이 읽어 DLSlideWindow.detect()에 넘긴다.
        self._latest_obstacle_y = None
        self._latest_obstacle_cut_confirmed = False
        self._latest_result = (False, 0.0, 0.0, default_center, [], None)
        # 디버그 창에 띄울 최근 프레임(초록/빨강 오버레이가 이미 그려진 vis, da/ll 원본 마스크).
        # 워커 스레드가 여기 값만 갱신하고, 실제 cv2.imshow()는 show_debug_windows()가
        # 메인 스레드에서만 호출한다(스레드 간 GUI 호출 혼용 방지 — 아래 _worker()/
        # show_debug_windows() 주석 참고).
        self._latest_debug = (None, None, None, None, None)   # (vis, da_mask_roi, ll_mask_roi, ll_yellow_mask_roi, offset_sparkline_img)
        # [2026-08-11] "이번 틱이 새로 나온 추론 결과인지" 구분용 카운터 — _worker()가 추론
        # 한 번을 끝내고 _latest_result를 갱신할 때마다만 1씩 증가한다(detect()가 몇 번
        # 호출됐는지가 아니라 "실제로 새 결과가 몇 번 나왔는지"를 센다). track_drive.py의
        # perc_lane()이 매 틱 이 값을 직전 틱 값과 비교해서, 변화가 없는 상태가 얼마나
        # 오래(초 단위) 지속되는지로 LANE_STALE_SEC 초과 여부(lane_stale)를 판정한다 —
        # detect()가 항상 논블로킹으로 최신 결과를 즉시 반환하는 구조라(모듈 상단 주석),
        # 추론이 20Hz control_loop()보다 느려 매 틱 같은 결과가 재사용되는 "정상" 상황과
        # 추론/카메라가 완전히 죽어 몇 초씩 안 갱신되는 "고장" 상황을 이 카운터 없이는
        # 구분할 방법이 없었다(둘 다 겉으로는 "같은 값이 계속 나옴"으로 동일하게 보임).
        # 단순 파이썬 int라 GIL 하에서 단일 읽기/증가가 원자적이므로 별도 락 없이
        # self.yellow_centers/self.roi_w와 동일한 관례로 읽는다.
        self.result_seq = 0
        self._stopped = False
        self._last_fps_log_t = time.time()

        self._thread = threading.Thread(target=self._worker, name='dl_lane_infer', daemon=True)
        self._thread.start()

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(f'[dl_lane] {msg}')

    def set_avoid_hold(self, active, side=0):
        """track_drive.py의 perc_lane()이 매 틱 호출 — avoid-hold(§2.32) 상태와
        [2026-08-15 적용3] 방향 힌트(side, -1/0/+1)를 다음 추론에 반영한다. 단순 대입이라
        별도 검증 없이 그대로 저장한다."""
        with self._lock:
            self._latest_avoid_hold = bool(active)
            self._latest_avoid_hold_side = int(side)

    def set_speed(self, v_mps):
        """[2026-08-17g] track_drive.py가 매 틱 호출 — 현재 실측 속도(m/s)를 다음 추론에
        반영한다(set_avoid_hold()와 동일 관례). 단순 대입이라 별도 검증 없이 저장한다."""
        with self._lock:
            self._latest_v_mps = float(v_mps)

    def set_obstacle(self, y_m, confirmed):
        """[2026-08-20] track_drive.py의 perc_lane()이 매 틱 호출 — da 근접 컷
        (_clip_da_by_obstacle(), ENABLE_OBSTACLE_CUT) 트리거 상태를 다음 추론에
        반영한다(set_avoid_hold()/set_speed()와 동일 관례). 단순 대입이라 별도 검증
        없이 저장한다. y_m=None이면 장애물 횡위치 정보 없음(confirmed도 무시됨)."""
        with self._lock:
            self._latest_obstacle_y = None if y_m is None else float(y_m)
            self._latest_obstacle_cut_confirmed = bool(confirmed)

    def _worker(self):
        """추론 워커 — 이 스레드 안에서는 절대 cv2.imshow()/cv2.waitKey()를 호출하지 않는다.
        OpenCV HighGUI(GTK 백엔드)가 스레드 세이프하지 않아서, 메인 스레드(다른 디버그
        창들이 이미 거기서 imshow/waitKey를 부름)와 여기서 동시에 GUI를 건드리면 실차에서
        "몇 초는 정상 동작하다가 화면이 그대로 멈춰버리는" 형태의 프리즈가 재현됐다.
        그려진 이미지 버퍼만 _latest_debug에 넘겨두고, 실제 창 표시는 show_debug_windows()가
        메인 스레드(perc_lane() 호출 시점)에서 전담한다."""
        while not self._stopped:
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
                avoid_hold = self._latest_avoid_hold
                avoid_hold_side = self._latest_avoid_hold_side
                v_mps = self._latest_v_mps
                obstacle_y = self._latest_obstacle_y
                obstacle_cut_confirmed = self._latest_obstacle_cut_confirmed
            if frame is None:
                time.sleep(0.005)
                continue

            try:
                raw_bgr, da_prob, ll_prob = self.engine.infer_raw(frame)
                yellow_mask = cv2.inRange(
                    cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV), YELLOW_LOWER, YELLOW_UPPER
                )
                lane_valid, offset, lookahead, lane_center, path = self._slide.detect(
                    raw_bgr, da_prob, ll_prob, yellow_mask,
                    avoid_hold=avoid_hold, direction_hint=avoid_hold_side, v_mps=v_mps,
                    obstacle_y_m=obstacle_y, obstacle_cut_confirmed=obstacle_cut_confirmed,
                )
                debug_img = self._slide.vis
            except Exception as e:
                self._log(f'추론 실패, 이번 프레임 스킵: {e}')
                continue

            with self._lock:
                self.yellow_centers = self._slide.yellow_centers
                # [2026-08-17 버그 수정] self.roi_w를 __init__에서 DL_INPUT_W(640, 모델 입력
                # 고정폭)로 하드코딩해뒀었는데, path/yellow_centers는 전부 self._slide.roi_w
                # (BEV 캔버스 실측폭, DL_USE_BEV=True 기준 585 — da_mask.shape에서 나옴) 좌표계다.
                # _lane_steer()의 vehicle_x = self.lane_detector.roi_w/2.0가 이 640을 그대로
                # 썼던 탓에, 실제 차량 기준점(292.5)이 아니라 320을 기준으로 dx를 계산해 매
                # 프레임 27.5px(≈13.75cm) 만큼 조향이 한쪽으로 밀리는 상시 편향이 있었다(raw
                # 카메라 데이터셋(lap_005) 재생 검증 결과 평균 -24°/최대 -38° 좌편향 확인,
                # 실차 미검증이었던 이유는 이 값이 조용히 항상 틀려서 "차선을 못 맞추는" 것과
                # "진동"이 뒤섞여 보였을 가능성). _slide.roi_w는 첫 프레임부터 항상 실제
                # da_mask 폭으로 정확하므로, 매 프레임 그대로 동기화한다 — _update_lane_side()도
                # 같은 self.roi_w를 쓰므로 이 수정 하나로 조향/차선판정 둘 다 고쳐진다.
                self.roi_w = self._slide.roi_w
                self.vehicle_center_x = self._slide.vehicle_center_x
                self.da_area_jump = self._slide.da_area_jump_detected  # [2026-08-15] 적용2
                self.path_ok = self._slide.path_ok  # [2026-08-17n] §2.36 재발 수정 참고
                self._latest_result = (lane_valid, offset, lookahead, lane_center, path, debug_img)
                self._latest_debug = (
                    self._slide.vis, self._slide.da_mask_roi, self._slide.ll_mask_roi,
                    self._slide.ll_yellow_mask_roi, self._slide.offset_sparkline_img,
                )
                # 이번 while 루프 반복이 예외 없이 여기까지 왔다 = 실제로 새 추론 결과가
                # 나왔다는 뜻이므로 여기서만 올린다(위 except: continue 경로는 못 지나감).
                self.result_seq += 1

            now = time.time()
            if now - self._last_fps_log_t >= FPS_LOG_PERIOD_SEC:
                self._log(f'DL 추론 FPS≈{self.engine.fps:.1f} (provider={self.engine.active_provider})')
                self._last_fps_log_t = now

    def detect(self, frame):
        if frame is not None:
            with self._lock:
                self._latest_frame = frame
        with self._lock:
            return self._latest_result

    def show_debug_windows(self, lookahead_xy=None, lookahead_px=None, v_mps=None,
                            steer_deg_raw=None, steer_deg_final=None):
        """da(초록)/ll(흰선=흰색·노란선=노랑) 오버레이 + (모드에 따라) 좌우 슬라이딩
        윈도우/corridor 경계가 그려진 result에 da/ll/노란선 원본 이진마스크를 위→아래로
        이어붙여 창 하나(`dl_lane`)로 띄운다 — result/da/ll/yellow 순서로 세로 스택.
        [2026-08-06] 예전엔 3개 별도 창(dl_lane_result/da/ll)이었는데, 창이 흩어져 있으면
        서로 다른 위치에 배치해야 해서 실차 테스트 중 한눈에 비교하기 불편하다는 피드백으로
        하나로 합쳤다. [2026-08-07] ll을 흰선/노란선으로 분리(_split_ll_by_yellow())하면서
        노란선만 따로 보이는 패널을 추가했다(result 패널에 이미 색으로 겹쳐 그려지긴 하지만,
        da/ll 전체 위에 옅게 깔린 것보다 노란선만 100% 불투명하게 보이는 게 dash가 끊기는지
        확인하기 더 쉽다). da/ll/yellow는 원래 1채널 이진마스크라 result(3채널 BGR)와 그대로
        못 이어붙이므로 BGR로 변환 후 vconcat한다 — 넷 다 같은 ROI에서 나온 동일 shape(BEV
        캔버스 크기)이라 폭이 항상 맞는다. [2026-08-10 도입, 2026-08-11 정리] 맨 아래에
        offset 디바운스 스파크라인(`DLSlideWindow._build_offset_sparkline()`, 폭을 이
        패널들과 같은 self.roi_w로 맞춰서 만듦)을 한 장 더 붙인다 — 원래 여기 붙던 config
        튜닝값 텍스트 목록(대부분 고정값이라 코드/config.py를 보면 알 수 있음)은 화면만
        차지해서 지웠고, 실제로 매 프레임 눈으로 봐야 하는 스파크라인만 남겼다.

        lookahead_xy(있으면) : pure_pursuit.PurePursuitController가 직전에 계산한 look-ahead
        목표점, (x, y) — self.lane_path와 같은 da ROI 픽셀좌표계라 result 패널에 좌표 변환
        없이 그대로 찍을 수 있다. 속도가 오르면(lookahead_speed_gain) 이 점이 위로(원거리로)
        멀어지는 걸 한눈에 보려는 디버그용 — 실제 판단 로직에는 영향 없다. pure_pursuit은
        detect()가 만든 self.path를 한 제어 틱(0.05s) 뒤에 소비하므로, 여기 찍히는 점은
        엄밀히는 "이번에 그려진 result"가 아니라 "직전 틱까지 계산된 최신 목표점"이다(한
        프레임 이내 오차, 디버깅 목적엔 무시 가능). path가 아직 없으면(첫 프레임) 호출측이
        None을 넘기고, 이 경우 마커를 그리지 않는다.

        v_mps(있으면) : 현재 실측 속도(m/s, track_drive.py self.v_mps) — [2026-08-17g]
        맨 아래 노란선(yellow) 전용 패널을 없애고 그 자리에 이 값을 표시한다(요청 반영:
        "yellow 부분을 제거하고 현 속도를 거기에 표시"). 노란선 자체는 result 패널에 이미
        색으로 겹쳐 그려지므로 정보 손실은 없다. [2026-08-19] 직진/커브대응 2상태 분기
        (pure_pursuit의 명시적 직진 모드, README §0.5.9) 자체를 제거하면서 이 패널이
        같이 표시하던 상태 텍스트도 함께 뺐다(요청 반영) — 이제 컨트롤러는 항상
        "커브대응" 파라미터 하나만 쓴다.

        steer_deg_raw/steer_deg_final(있으면) : [2026-08-19, 요청 반영] pure_pursuit의
        wheelbase 부스트(controller/pure_pursuit.py PP_WHEELBASE_BOOST_* 참고) "전" 1차
        조향각(last_pre_boost_steer_deg)과 "후"(필터+클램프까지 다 거쳐 실제로 차에 나가는
        prev_steer_deg)를 ll 패널 상단에 같이 찍는다 — 부스트가 지금 실제로 얼마나 개입
        중인지 화면만 보고 바로 비교할 수 있게. lookahead_xy와 동일하게 _lane_steer()가
        이번 틱에 아직 안 돌았을 때는 직전 틱 값(0.05s 이내 오차, 무시 가능). 둘 다
        None이면(호출측이 안 넘기거나 pure_pursuit이 아직 없는 초기 프레임) 아무것도
        안 그린다.
        ★ 반드시 메인 스레드(ROS 콜백/타이머가 도는 스레드)에서만 호출할 것 ★ — 워커
        스레드가 cv2.imshow를 직접 부르지 않는 이유는 _worker()/DLSlideWindow.visualize()
        주석 참고. track_drive.py의 perc_lane()이 detect() 직후 이 메서드를 호출한다
        (다른 백엔드는 이 메서드가 없으므로 getattr(..., lambda: None)()로 안전하게 건너뜀)."""
        if not DEBUG_VIZ_DL_LANE:
            return
        with self._lock:
            vis, da_mask, ll_mask, ll_yellow_mask, sparkline = self._latest_debug
        if vis is None:
            return
        if lookahead_xy is not None:
            lx, ly = int(round(lookahead_xy[0])), int(round(lookahead_xy[1]))
            cv2.drawMarker(vis, (lx, ly), (0, 255, 255), markerType=cv2.MARKER_CROSS,
                            markerSize=14, thickness=2)
            cv2.circle(vis, (lx, ly), 6, (0, 255, 255), 2)
            if lookahead_px is not None:
                cv2.putText(
                    vis, f'ld:{lookahead_px:.0f}px', (lx + 8, ly - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
                )
        da_bgr = cv2.cvtColor(da_mask, cv2.COLOR_GRAY2BGR)
        ll_bgr = cv2.cvtColor(ll_mask, cv2.COLOR_GRAY2BGR)
        # [2026-08-19] ll 패널 상단에 조향각 원본(부스트 전)/최종(부스트+필터+클램프,
        # 실제로 나가는 값)을 같이 표시 — 위 show_debug_windows() docstring 참고. 값이
        # 다르면(=지금 부스트가 개입 중) 최종값을 주황으로 강조해 한눈에 띄게 한다.
        if steer_deg_raw is not None and steer_deg_final is not None:
            boosted = abs(steer_deg_final - steer_deg_raw) > 1e-3
            final_color = (0, 140, 255) if boosted else (255, 255, 255)  # 주황=부스트 개입중
            steer_text = f'조향 원본:{steer_deg_raw:+.1f}도 -> 최종:{steer_deg_final:+.1f}도'
            put_text_kr(ll_bgr, steer_text, (5, 24), font_size=18, color_bgr=final_color,
                        fallback=f'raw:{steer_deg_raw:+.1f}deg -> final:{steer_deg_final:+.1f}deg')
        # [2026-08-17g] 예전엔 여기 노란선(ll_yellow_mask) 전용 패널이 있었다(노란선만
        # 100% 불투명하게 보여 dash 끊김 확인용, ll_yellow_mask.shape 크기). 요청 반영으로
        # 제거하고 같은 크기 패널에 속도를 표시한다 — 노란선 자체는 result 패널 오버레이에
        # 이미 보이므로 이 패널이 없어져도 정보 손실은 없다. [2026-08-19] 직진/커브대응
        # 상태 텍스트는 그 2상태 분기 자체를 제거하면서 함께 뺐다(요청 반영).
        speed_bgr = np.zeros((*ll_yellow_mask.shape, 3), dtype=np.uint8)
        if v_mps is not None:
            speed_text = f'speed: {v_mps:+.2f} m/s'
            put_text_kr(speed_bgr, speed_text, (10, 8), font_size=22, color_bgr=(255, 255, 255),
                        fallback=f'speed:{v_mps:+.2f}m/s')
        for label, panel in (('result', vis), ('da', da_bgr), ('ll', ll_bgr), ('speed', speed_bgr)):
            cv2.putText(panel, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        stack = [vis, da_bgr, ll_bgr, speed_bgr]
        if sparkline is not None:
            stack.append(sparkline)
        cv2.imshow('dl_lane', cv2.vconcat(stack))
        cv2.waitKey(1)

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=2.0)


def _run_benchmark(source, n_frames, model_path):
    """오프라인 FPS 벤치마크. onnxruntime/모델 파일이 있는 실기기(Jetson)에서 직접 돌려서
    실측치를 확인하는 용도 — 이 저장소를 준비한 개발 머신에는 onnxruntime/GPU가 없어 여기서는
    실행할 수 없었다.
    사용: python3 -m track_drive.dl_lane --source 0 (카메라 인덱스) 또는 --source video.mp4"""
    engine = TwinLiteNetEngine(model_path=model_path)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f'소스를 열 수 없습니다: {source}')

    n = 0
    t0 = time.perf_counter()
    while n < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        engine.infer_raw(frame)
        n += 1
    elapsed = time.perf_counter() - t0
    cap.release()
    print(f'{n} frames in {elapsed:.2f}s -> {(n / elapsed if elapsed > 0 else 0.0):.1f} FPS '
          f'(provider={engine.active_provider})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TwinLiteNet ONNX 추론 FPS 벤치마크')
    parser.add_argument('--source', default='0', help='카메라 인덱스 또는 비디오 파일 경로')
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
    _run_benchmark(args.source, args.frames, args.model)
