#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# dl_lane.py — TwinLiteNet(ONNX Runtime) 기반 딥러닝 차선인식 백엔드.
#
# https://github.com/harrylal/TwinLiteNet-onnxruntime 의 사전학습 가중치(models/best.onnx)를
# 그대로 사용한다. hough_lane.HoughLaneDetector / perc_floor.LaneDetector와 동일하게
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
#   'll'    : ll을 흰선/노란선으로 분리(_split_ll_by_yellow())한 뒤, 흰선만으로 좌/우
#             독립 슬라이딩 윈도우(_ll_slice_centers())를 추적한다. da 폴백은 없다(둘 다
#             못 찾은 밴드는 무효).
#   'da'/'ll' 두 모드는 da 파편화 대응(_largest_da_component())/옆 차선
#   클리핑(_clip_da_by_ll())을 공유한다('ll_da'=corridor는 둘 다 건너뜀). ll 프레임 단위
#   sanity check(DL_LL_SANITY_MIN_RATIO 미만이면 이번 프레임 무효)는 세 모드 모두 적용.
#=============================================
import argparse
import os
import time
import threading

import cv2
import numpy as np

from .lane_util import SlideWindow

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


# ── 모델 입출력 스펙 (harrylal/TwinLiteNet-onnxruntime 리포 기준 — 이미 검증됨, 재검증 불필요) ──
#   images 텐서 (1,3,360,640) float32 NCHW. 전처리는 letterbox 없이 640x360으로 그냥
#   리사이즈(원본 리포의 blobFromImage와 동일) → BGR→RGB → /255.0 (mean/std 정규화 없음).
DL_INPUT_W = 640
DL_INPUT_H = 360
DL_INPUT_NAME = 'images'
DL_OUTPUT_NAMES = ('da', 'll')

# ── 세그멘테이션 입력은 절대 자르지 않는다 ──
#   원본 리포(blobFromImage)와 동일하게 raw 프레임 전체를 그대로 640x360으로 리사이즈해서
#   모델에 넣는다(추가 크롭 없음). 관심영역은 "모델에 들어가기 전"이 아니라 "모델에서 나온
#   세그멘테이션 결과(da/ll)를 원본 프레임 크기로 되돌린 뒤" 잘라서 쓴다 — 아래
#   DL_ROI_Y0/Y1 참고.
#
# da/ll 둘 다 (1,2,360,640) raw logit. 채널축 softmax 후 채널1이 foreground 확률.
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
    DL_DA_MIN_COMPONENT_AREA,
    DL_DA_SEED_ROWS_PX, DL_DA_SEED_HALF_WIDTH_PX,
    DL_LL_SANITY_MIN_RATIO, DL_LL_CLIP_MARGIN_PX,
    DL_LL_DECAY_ALPHA, DL_LL_DECAY_MIN_VALUE,
    DL_CENTER_MODE, DL_LL_SIDE_MIN_PIXELS, DL_LL_WIDTH_MIN_PX, DL_LL_WIDTH_MAX_PX,
    DL_LL_SEARCH_HALF_WIDTH_PX, DL_LL_WIDTH_EMA_ALPHA,
    DL_CORRIDOR_LINE_MIN_PIXELS, DL_CORRIDOR_LINE_MERGE_PX,
    DL_CORRIDOR_WIDTH_MIN_PX, DL_CORRIDOR_WIDTH_MAX_PX, DL_CORRIDOR_MIN_PASSABLE_PX,
    DL_LL_YELLOW_VOTE_RATIO, DL_LL_YELLOW_MIN_AREA,
    DEBUG_VIZ_DL_LANE, YELLOW_LOWER, YELLOW_UPPER, FPS_LOG_PERIOD_SEC,
)


def _default_model_path():
    """모델 가중치 파일(best.onnx) 기본 경로.
    1순위: colcon install된 share 디렉터리(share/track_drive/models/best.onnx)
    2순위: 소스트리에서 직접 실행 중일 때(개발 중, colcon build 전) — track_drive 패키지 디렉터리 기준 상대경로
    """
    if get_package_share_directory is not None:
        try:
            share_dir = get_package_share_directory('track_drive')
            candidate = os.path.join(share_dir, 'models', 'best.onnx')
            if os.path.isfile(candidate):
                return candidate
        except Exception:
            pass
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_dir, 'models', 'best.onnx')


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
                f'TwinLiteNet 가중치 파일을 찾을 수 없습니다: {self.model_path}\n'
                'https://github.com/harrylal/TwinLiteNet-onnxruntime 의 models/best.onnx를 '
                '내려받아 이 경로에 두세요.'
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
            # 요구사항: TensorRT EP > CUDA EP > CPU EP 순, 실제 존재하는 provider와 교집합만.
            priority = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
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
    (_split_ll_by_yellow()) 흰선만으로 좌/우 슬라이딩 윈도우를 추적한다
    (_ll_slice_centers()). 'da' 모드만 여전히 calc_center()를 호출하지 않고 detect()
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
        self.da_mask_roi = None    # 시각화용(가장 큰 덩어리만 남긴 이후의 da 마스크 — 실제 waypoint 추출에 쓰는 것)
        self.da_mask_all_roi = None  # 시각화용(이진화 직후, 덩어리 선택/ll클리핑 전 da 전체 — visualize()가 파란색으로 그림)
        self.ll_mask_roi = None    # 시각화용
        self.centerline = []       # 밴드별 중심점(원본 관측점, 길이 self.n_slices) — DL_CENTER_MODE에 따라 da 단독 또는 ll 우선/da 폴백
        self.ll_band_used = []     # 이번 프레임 각 밴드가 ll 기반으로 채택됐는지(길이 self.n_slices bool, 'da' 모드에선 항상 전부 False) — visualize() 색상 구분용
        self.da_fallback_used = False  # 이번 프레임 da가 직전 채택 덩어리와의 근접성이 아니라 면적순위 차선책으로 골라졌는지 — visualize() 색상 구분용
        self.da_ll_clip_skipped = False  # 이번 프레임 ll 클리핑이 유효 밴드를 너무 줄여 건너뛰었는지 — visualize() 구분용
        self.da_ll_virtual_clip_used = False  # [2026-08-07] 이번 프레임 _clip_da_by_ll()이 ll/잔상 없이 가상경계(기대 차로폭)로 클리핑한 밴드가 있었는지 — visualize() 구분용
        self.da_largest_mask_roi = None  # 면적 1위 덩어리(차선책을 썼다면 그 사유가 된, 상한 초과로 버려진 덩어리) — fallback일 때 원래 색으로 같이 그리기용
        self.da_largest_area_px = 0  # 면적 1위 덩어리의 절대 픽셀 면적(채택 여부 무관) — DL_DA_MAX_AREA_PX 실측 튜닝용
        self._prev_da_centroid = None  # [2026-08-07] 직전 프레임에 채택된 da 덩어리의 중심(cx,cy) — _largest_da_component()가 이번 프레임 후보를 "가장 가까운 것"으로 고르는 기준. 무효 프레임 뒤엔 None으로 리셋(옛 위치에 계속 붙잡히지 않도록)
        self.da_chosen_area_px = 0   # 실제로 채택돼 waypoint 추출에 쓰인 덩어리의 면적(무효 프레임엔 0)
        self.da_seed_width_px = 0   # [2026-08-10] 시드 위치(ROI 최하단 중앙, 차량 위치)에서 찾은 덩어리의 bounding box 가로폭(px) — 실제 채택/면적통과 여부와 무관하게 항상 기록(시드 위치에 아무것도 없으면 0). 너비 기반 선택 로직으로 바꿀지 판단하기 위한 실측용 — 아직 판단 로직에는 안 쓰임(_debug_viz_steer() 참고)
        self._ll_half_width = (DL_LL_WIDTH_MIN_PX + DL_LL_WIDTH_MAX_PX) / 4.0  # [2026-08-07] ll 좌/우 독립 슬라이딩 윈도우의 차로 반폭 러닝 추정치(px) — 양쪽 다 찾은 밴드에서 EMA 갱신, 편측만 찾았을 때 반대쪽 위치 추정에 씀(_ll_slice_centers() 참고). _clip_da_by_ll()의 가상경계 최후수단에도 재사용.
        self._ll_decay_mask = None   # [2026-08-07] ll 잔상(decay) 누적 마스크(float32, roi shape) — detect()가 매 프레임 갱신, _clip_da_by_ll() 전용(centerline 추출엔 안 씀). None이면 첫 프레임이라 detect()에서 새로 할당.

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

        # [2026-08-10] DL_CENTER_MODE='ll'(흰선/노란선 분리) 전용 부가 정보 — 전부
        # visualize()용이고 경로/조향 계산에는 안 쓰인다(노란선은 아직 stateless 디버그
        # 표시만, 모듈 상단 "'ll'" 주석 참고).
        self.ll_white_mask_roi = None    # 흰선으로 확정된 ll 컴포넌트만 남은 마스크(_ll_slice_centers()의 실제 입력)
        self.ll_yellow_mask_roi = None   # 노란선으로 확정된 ll 컴포넌트만 남은 마스크
        self.yellow_band_centers = []    # ll_yellow_mask_roi를 밴드별 무게중심(_slice_centers(), 탐색창 없는 stateless 방식)으로 뽑은 결과 — 길이 self.n_slices
        self.ll_search_windows = []      # _ll_slice_centers()가 이번 프레임에 훑은 좌/우 탐색창 좌표(밴드별) — visualize()가 사각형으로 그림

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

    def _clip_da_by_ll(self, da_mask, ll_mask, ref_x):
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
           차로 반폭(self._ll_half_width — _ll_slice_centers()가 관리하는 것과 같은
           러닝 추정치)만큼 cur_ref 양옆을 강제로 자른다("가상 경계"). 픽셀 근거는 없지만
           "차로폭은 대략 이 정도"라는 기하학적 사전지식이, 무근거 병합(da가 옆 차선까지
           안 잘린 채 남는 것)보다는 안전하다는 판단이다. ①(실측/잔상 클리핑)과 달리
           cur_ref는 갱신하지 않는다 — 실측 근거 없는 추정을 다음 밴드로 계속 누적시키지
           않기 위해서다.
        classic_cv 백엔드의 "한쪽 차선만 검출" 폴백(lane_util.SlideWindow.calc_center())과
        같은 "차로폭 기반 추정" 원칙을 여기 클리핑에도 적용한 것.

          입력 : da_mask — (roi_h, roi_w) uint8 이진마스크
                 ll_mask — 동일 shape 이진마스크. 호출부가 실측/잔상 어느 쪽을 넣어도 무방.
                 ref_x   — 첫(근거리) 밴드의 기준 x좌표. 보통 직전 프레임 lane_center.
          출력 : (clipped, virtual_used) — clipped는 da_mask에서 ll(또는 가상) 경계 밖
                 픽셀만 0으로 지운 복사본(shape 동일), virtual_used는 이번 호출에서 ②
                 (가상경계)가 한 밴드라도 발동했는지(bool) — visualize() 디버그 표시용.
        """
        h, w = da_mask.shape
        slice_h = h // self.n_slices
        clipped = da_mask.copy()
        cur_ref = ref_x
        virtual_used = False  # 이번 호출에서 ②(가상경계)가 한 밴드라도 발동했는지 — visualize() 디버그 표시용

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
                if right_cols.size:
                    cut = max(0, int(right_cols.min()) - DL_LL_CLIP_MARGIN_PX)
                    clipped[y_low:y_high, cut:] = 0

                band_da_cols = np.nonzero(np.any(clipped[y_low:y_high, :] > 0, axis=0))[0]
                if band_da_cols.size:
                    cur_ref = float(np.mean(band_da_cols))
            else:
                # ② ll도 잔상도 없음 — 최후 수단: 기대 차로 반폭 기준 가상 경계로 강제 클리핑.
                lcut = int(np.clip(cur_ref - self._ll_half_width, 0, w))
                rcut = int(np.clip(cur_ref + self._ll_half_width, 0, w))
                clipped[y_low:y_high, :lcut] = 0
                clipped[y_low:y_high, rcut:] = 0
                virtual_used = True
                # cur_ref는 갱신하지 않는다 — 실측 근거 없는 추정이라 그대로 다음 밴드로 넘김.

        return clipped, virtual_used

    def _split_ll_by_yellow(self, ll_mask, yellow_roi):
        """ll_mask(흰/노랑 구분 없는 차선 이진마스크)를 커넥티드 컴포넌트 단위로
        흰선/노란선 마스크로 나눈다. DL_CENTER_MODE='ll' 전용(_ll_slice_centers()가
        흰선만 입력받도록 하기 위함, config.py DL_CENTER_MODE 주석 참고).

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
        DL_CENTER_MODE='ll_da'(corridor) 전용 — _ll_slice_centers()는 "ref_x 기준 정확히
        좌/우 2선"만 다루는 반면, 이건 밴드 안의 선을 개수 제한 없이 전부 찾아 순서만
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
        삼는다. prefer_x(band-local 좌표, 직전 프레임에 이 밴드에서 채택했던 위치 — 없으면
        None)가 있으면 그 값에 가장 가까운 run을 고른다 — 이게 "장애물을 피해간 방향을
        다음 프레임도 유지"하는 히스테리시스라, 폭이 비슷한 두 열린 구간(예: 장애물 좌/우)
        사이를 매 프레임 오가는 flip-flop을 막는다. prefer_x가 없으면(첫 프레임 등) 가장
        넓은 run을 고른다. 통과 가능한 run이 하나도 없으면 None(이 밴드는 완전히 막힘 —
        무효 처리).
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

            prev_x = self._corridor_prev_open_x[i]
            prefer_x = (prev_x - lb) if prev_x is not None else None
            run = self._pick_open_run(open_cols, prefer_x)
            if run is None:
                continue  # corridor 안에 지나갈 폭이 있는 열린 구간이 없음 — 완전히 막힘

            cx = lb + (run[0] + run[1]) / 2.0
            results[i] = (y_center, cx)
            used[i] = True
            self._corridor_prev_open_x[i] = cx

        return results, used

    def _ll_slice_centers(self, ll_mask, ref_x):
        """[2026-08-10] DL_CENTER_MODE='ll'일 때만 호출된다('ll_da'는 corridor 알고리즘
        (_corridor_slice_centers())으로 교체돼 더 이상 이 함수를 안 쓴다). ll_mask(흰선만
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

          입력 : ll_mask — (roi_h, roi_w) uint8 이진마스크(da_mask와 동일 shape/좌표계)
                 ref_x   — 첫(근거리) 밴드의 좌/우 초기 중심 x좌표. 보통 직전 프레임 lane_center.
          출력 : (results, used) — 둘 다 길이 self.n_slices.
                 results[i] : 채택되면 (y_center, cx), 아니면 None(da로 폴백해야 함을 뜻함)
                 used[i]    : results[i]가 ll 기반으로 채택됐는지(양쪽/편측 무관, bool) — 디버그 시각화용

        알려진 한계:
        - 탐색창이 좁은 만큼, 급커브에서 밴드 간 실제 선 이동량이
          DL_LL_SEARCH_HALF_WIDTH_PX보다 크면 그 라인의 창이 선을 놓치고 이후 밴드까지
          이전 위치에 멈춰 서게 된다(추적 이탈, 좌/우 독립이라 한쪽만 이탈해도 반대쪽은
          영향 없음).
        - 편측 폴백(2번)이 여러 밴드 연속으로 이어지면, self._ll_half_width가 그 사이
          갱신되지 않아(양쪽 다 찾은 밴드에서만 갱신) 오래된 추정치를 계속 쓰게 된다 —
          실차 미검증, 편측 검출이 긴 구간에서 추정 중심이 실제와 얼마나 벌어지는지
          확인할 것."""
        h, w = ll_mask.shape
        slice_h = h // self.n_slices
        results = [None] * self.n_slices
        used = [False] * self.n_slices
        # 디버그 시각화용 — 밴드별 좌/우 탐색창 좌표 + 실제로 찾았는지(visualize()가
        # 사각형/색으로 그림). 알고리즘 자체엔 전혀 쓰이지 않는 부가 정보.
        self.ll_search_windows = []

        cur_left = ref_x - self._ll_half_width
        cur_right = ref_x + self._ll_half_width
        win = DL_LL_SEARCH_HALF_WIDTH_PX

        for i in range(self.n_slices):
            y_high = h - i * slice_h
            y_low = 0 if i == self.n_slices - 1 else h - (i + 1) * slice_h
            band = ll_mask[y_low:y_high, :]
            y_center = (y_low + y_high) / 2.0

            lx = None
            lx0, lx1 = int(np.clip(cur_left - win, 0, w)), int(np.clip(cur_left + win, 0, w))
            if lx1 > lx0:
                M_l = cv2.moments(band[:, lx0:lx1], binaryImage=True)
                if M_l['m00'] >= DL_LL_SIDE_MIN_PIXELS:
                    lx = lx0 + M_l['m10'] / M_l['m00']
                    cur_left = lx  # 왼쪽 창은 왼쪽 결과만으로 독립 갱신 — 오른쪽 성패와 무관

            rx = None
            rx0, rx1 = int(np.clip(cur_right - win, 0, w)), int(np.clip(cur_right + win, 0, w))
            if rx1 > rx0:
                M_r = cv2.moments(band[:, rx0:rx1], binaryImage=True)
                if M_r['m00'] >= DL_LL_SIDE_MIN_PIXELS:
                    rx = rx0 + M_r['m10'] / M_r['m00']
                    cur_right = rx  # 오른쪽 창은 오른쪽 결과만으로 독립 갱신 — 왼쪽 성패와 무관

            if lx is not None and rx is not None:
                width = rx - lx
                if DL_LL_WIDTH_MIN_PX < width < DL_LL_WIDTH_MAX_PX:
                    results[i] = (y_center, (lx + rx) / 2.0)
                    used[i] = True
                    alpha = DL_LL_WIDTH_EMA_ALPHA
                    self._ll_half_width = (1 - alpha) * self._ll_half_width + alpha * (width / 2.0)
                # 폭이 비정상 — 반대 차선을 잘못 짝지었을 가능성, 양쪽 다 못 믿으므로 무효
            elif lx is not None:
                results[i] = (y_center, lx + self._ll_half_width)
                used[i] = True
            elif rx is not None:
                results[i] = (y_center, rx - self._ll_half_width)
                used[i] = True

            self.ll_search_windows.append((y_low, y_high, lx0, lx1, lx, rx0, rx1, rx))

        return results, used

    def detect(self, raw_bgr, da_prob, ll_prob, yellow_mask):
        """입력 : raw_bgr — 원본 카메라 프레임 그대로의 (H,W,3) BGR(크롭/리사이즈 없음)
                 da_prob, ll_prob — 위와 같은 (H,W) float32 foreground 확률(모델은 360행
                   고정이지만 TwinLiteNetEngine.infer_raw()가 이미 원본 크기로 업샘플링해서 줌)
                 yellow_mask — 위와 같은 (H,W) uint8 이진마스크(HSV 기반, da/ll과 무관)
          출력 : lane_valid, offset, lookahead, lane_center, path — 기존 SlideWindow.calc_center()와
                 동일한 계약(같은 4-tuple+path 형태)이지만, 계산은 da 중심선 기준으로 직접 한다.
          내부에서 DL_ROI_Y0:DL_ROI_Y1(원본 프레임 절대 픽셀)만 잘라서 da 중심선을 뽑는다.
        """
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
        # 추출(_ll_slice_centers)에는 이 잔상을 안 쓴다 — waypoint 자체를 과거 위치로 밀면
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
        # 전용 좌/우 슬라이딩 윈도우(_ll_slice_centers())는 흰선만 담긴 ll_white_mask를
        # 본다. 중앙 노란 점선(ll_yellow_mask)은 경로 계산에 안 섞고 밴드별 무게중심만
        # 뽑아 visualize() 디버그 표시용으로만 쓴다(추후 "도로 중앙" 힌트로 확장 예정).
        # 'll_da'(corridor)는 이 분리 결과를 안 쓰고 흰/노랑 안 가린 원본 ll_mask를 그대로
        # 쓴다(_ll_line_centers() 참고 — 노란 중앙선도 "2번째 선"으로 세야 하므로).
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
            clipped, self.da_ll_virtual_clip_used = self._clip_da_by_ll(da_mask, ll_mask_for_clip, ref_x)
            clipped_valid = sum(1 for c in self._slice_centers(clipped, 0, (0, 255, 0)) if c is not None)
            self.da_ll_clip_skipped = clipped_valid < self.slice_fit_min
            da_mask = da_mask if self.da_ll_clip_skipped else clipped

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
            # 'll'이면 흰선만 담긴 ll_white_mask로 _ll_slice_centers()를 돌려 da 폴백
            # 없이 그대로 쓴다(모듈 상단/config.py DL_CENTER_MODE 주석 참고).
            if DL_CENTER_MODE == 'll':
                merged_centers, self.ll_band_used = self._ll_slice_centers(ll_white_mask, ref_x)
            else:
                merged_centers = self._slice_centers(da_mask, 0, (0, 255, 0))
                self.ll_band_used = [False] * len(merged_centers)
                self.ll_search_windows = []

        self.da_mask_roi = da_mask
        self.centerline = self._reject_outliers(merged_centers)

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

        # ll sanity check: da 중심선이 있어도 ll(차선) 신호가 거의 안 보이면(모션블러 등으로
        # 세그멘테이션이 통째로 깨진 경우) 이번 프레임을 무효 처리한다 — da 커버리지만으로는
        # 못 거르는 실패모드를 ll로 보강(모듈 상단 "da를 경로의 주 신호로" 주석 참고).
        lane_valid = near_center is not None and self.ll_coverage >= DL_LL_SANITY_MIN_RATIO

        offset = lookahead = 0.0
        if lane_valid:
            offset = near_center - self.roi_w / 2.0
            far_ref = far_center if far_center is not None else near_center
            lookahead = far_ref - self.roi_w / 2.0
        lane_center = self.roi_w / 2.0 + offset

        # 명시적 경로(웨이포인트) — da 밴드 중심점을 선형보간으로 이어 만든다. 유효 밴드가
        # 2개 미만이면 fitted_path가 None이 되고, 그 경우 _update_path()가 self.path를
        # 갱신하지 않아(직전 프레임 값 유지) offset/lane_offset과 동일한 "무효 프레임엔
        # 마지막 값 유지" 원칙을 따른다. 유효할 때도 그대로 대입하지 않고 직전 경로와
        # EMA 블렌딩한다(lane_util.PATH_EMA_ALPHA 주석 참고) — 조향이 매 프레임 새로
        # 피팅된 경로에 과민하게 반응하는 걸 막기 위함.
        fitted_path = self._fit_and_sample_path(
            [c for c in self.centerline if c is not None]
        )
        self._update_path(fitted_path)

        lane_valid, offset, lookahead, lane_center = self._debounce(
            lane_valid, offset, lookahead, lane_center
        )

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

        if DEBUG_VIZ_DL_LANE and self.da_mask_roi is not None:
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
            overlay[self.da_mask_roi > 0] = da_color       # 주행가능영역(실제 채택분)
            # ll을 흰선/노란선 분리 결과(_split_ll_by_yellow())대로 실제 색과 맞춰
            # 칠한다 — "이 라인이 지금 흰선/노란선 중 뭘로 인식되고 있는지"를 색만
            # 보고 바로 알 수 있게.
            if self.ll_white_mask_roi is not None:
                overlay[self.ll_white_mask_roi > 0] = (255, 255, 255)  # 흰선
            if self.ll_yellow_mask_roi is not None:
                overlay[self.ll_yellow_mask_roi > 0] = (0, 255, 255)   # 노란선
            cv2.addWeighted(overlay, 0.35, self.vis, 0.65, 0, dst=self.vis)

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

            # 좌/우 슬라이딩 윈도우 탐색창(_ll_slice_centers()가 이번 프레임에 훑은
            # 범위, DL_CENTER_MODE='ll'에서만 채워짐) — 찾았으면 초록, 못 찾았으면(창
            # 안에 픽셀 부족) 회색 테두리로 구분. 밴드별 실측 차로폭(rx-lx) 텍스트는
            # DL_LL_WIDTH_MIN_PX~MAX_PX 튜닝용 — 범위 안이면 초록(채택), 밖이면
            # 빨강(그 밴드는 버려짐)으로 색을 나눈다.
            for (y_low, y_high, lx0, lx1, lx, rx0, rx1, rx) in self.ll_search_windows:
                cv2.rectangle(self.vis, (lx0, y_low), (lx1, max(y_high - 1, y_low)),
                              (0, 255, 0) if lx is not None else (120, 120, 120), 1)
                cv2.rectangle(self.vis, (rx0, y_low), (rx1, max(y_high - 1, y_low)),
                              (0, 255, 0) if rx is not None else (120, 120, 120), 1)
                if lx is not None and rx is not None:
                    width = rx - lx
                    in_range = DL_LL_WIDTH_MIN_PX < width < DL_LL_WIDTH_MAX_PX
                    cv2.putText(
                        self.vis, f'{width:.0f}px', (int(rx1) + 4, int((y_low + y_high) / 2) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 255, 0) if in_range else (0, 0, 255), 1
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

        # 밴드별 중심점 — DL_CENTER_MODE='ll'에서 ll 기반으로 채택된 밴드는 흰색, da로
        # 폴백한 밴드는 노란색으로 구분해서(_ll_slice_centers()/detect() 병합 로직 참고)
        # 어느 밴드가 ll을 못 써서 da로 대체됐는지 한눈에 보이게 한다('da' 모드에선
        # ll_band_used가 항상 전부 False라 전부 노란색 — 기존 draw_centers() 색과 동일해
        # 시각적으로 하위호환. 'll' 모드에선 애초에 da 폴백이 없어 그려지는 점은 전부
        # 흰색이고, ll이 부족해 무효 처리된 밴드는 centerline이 None이라 아예 안 그려짐.
        # 'll_da'=corridor 모드에선 ll_band_used가 "이 밴드가 corridor로 채택됐는지"를
        # 뜻하므로 흰색=채택, 노란색은 애초에 나오지 않는다).
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
            cv2.circle(self.vis, pt, 4, (255, 255, 255) if ll_used else (0, 255, 255), -1)
        self.draw_path(self.path)  # 피팅된 최종 경로(자홍색)

        cv2.line(self.vis, (self.roi_w // 2, 0), (self.roi_w // 2, self.roi_h), (0, 0, 255), 1)
        lane_center = self.roi_w / 2.0 + offset
        ll_band_count = sum(1 for u in self.ll_band_used if u)
        tags = ''
        if self.da_fallback_used:
            tags += ' [FALLBACK]'
        if self.da_ll_clip_skipped:
            tags += ' [LL_CLIP_SKIP]'
        if self.da_ll_virtual_clip_used:
            tags += ' [LL_VIRTUAL]'
        # 모드마다 밴드 카운트가 뜻하는 바가 달라서 라벨/부가정보를 따로 붙인다 —
        # 'll_da'(corridor)는 corridor로 채택된 밴드 수, 'll'은 흰선 채택 밴드 수 +
        # 노란선 채택 밴드 수 + 현재 러닝 차로폭 추정치, 'da'는 항상 0(ll 미사용, 기존
        # 방식과 동일 표시).
        if DL_CENTER_MODE == 'll_da':
            extra = f'corridor_bands:{ll_band_count}/{self.n_slices}'
        elif DL_CENTER_MODE == 'll':
            yellow_band_count = sum(1 for c in self.yellow_band_centers if c is not None)
            extra = (
                f'white_bands:{ll_band_count}/{self.n_slices} '
                f'yellow_bands:{yellow_band_count}/{self.n_slices} '
                f'lane_w_est:{self._ll_half_width * 2:.0f}px'
            )
        else:
            extra = f'll_bands:{ll_band_count}/{self.n_slices}'
        cv2.putText(
            self.vis,
            f'offset:{offset:+.1f} center:{lane_center:.1f} ll_cov:{self.ll_coverage:.3f} '
            f'mode:{DL_CENTER_MODE} {extra}{tags}',
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )


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

        self.roi_w = DL_INPUT_W          # _update_lane_side()가 참조하는 정규화 분모
        self.yellow_centers = []         # _update_lane_side() 호환용, 워커가 매 추론마다 갱신

        default_center = DL_INPUT_W / 2.0
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_result = (False, 0.0, 0.0, default_center, [], None)
        # 디버그 창에 띄울 최근 프레임(초록/빨강 오버레이가 이미 그려진 vis, da/ll 원본 마스크).
        # 워커 스레드가 여기 값만 갱신하고, 실제 cv2.imshow()는 show_debug_windows()가
        # 메인 스레드에서만 호출한다(스레드 간 GUI 호출 혼용 방지 — 아래 _worker()/
        # show_debug_windows() 주석 참고).
        self._latest_debug = (None, None, None, None)   # (vis, da_mask_roi, ll_mask_roi, ll_yellow_mask_roi)
        self._stopped = False
        self._last_fps_log_t = time.time()

        self._thread = threading.Thread(target=self._worker, name='dl_lane_infer', daemon=True)
        self._thread.start()

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(f'[dl_lane] {msg}')

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
            if frame is None:
                time.sleep(0.005)
                continue

            try:
                raw_bgr, da_prob, ll_prob = self.engine.infer_raw(frame)
                yellow_mask = cv2.inRange(
                    cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV), YELLOW_LOWER, YELLOW_UPPER
                )
                lane_valid, offset, lookahead, lane_center, path = self._slide.detect(
                    raw_bgr, da_prob, ll_prob, yellow_mask
                )
                debug_img = self._slide.vis
            except Exception as e:
                self._log(f'추론 실패, 이번 프레임 스킵: {e}')
                continue

            with self._lock:
                self.yellow_centers = self._slide.yellow_centers
                self._latest_result = (lane_valid, offset, lookahead, lane_center, path, debug_img)
                self._latest_debug = (
                    self._slide.vis, self._slide.da_mask_roi, self._slide.ll_mask_roi,
                    self._slide.ll_yellow_mask_roi,
                )

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

    def show_debug_windows(self, lookahead_xy=None, lookahead_px=None):
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
        캔버스 크기)이라 폭이 항상 맞는다.

        lookahead_xy(있으면) : pure_pursuit.PurePursuitController가 직전에 계산한 look-ahead
        목표점, (x, y) — self.lane_path와 같은 da ROI 픽셀좌표계라 result 패널에 좌표 변환
        없이 그대로 찍을 수 있다. 속도가 오르면(lookahead_speed_gain) 이 점이 위로(원거리로)
        멀어지는 걸 한눈에 보려는 디버그용 — 실제 판단 로직에는 영향 없다. pure_pursuit은
        detect()가 만든 self.path를 한 제어 틱(0.05s) 뒤에 소비하므로, 여기 찍히는 점은
        엄밀히는 "이번에 그려진 result"가 아니라 "직전 틱까지 계산된 최신 목표점"이다(한
        프레임 이내 오차, 디버깅 목적엔 무시 가능). path가 아직 없거나(첫 프레임)
        STEERING_CONTROLLER='lqr'이면 호출측이 None을 넘기고, 이 경우 마커를 그리지 않는다.
        ★ 반드시 메인 스레드(ROS 콜백/타이머가 도는 스레드)에서만 호출할 것 ★ — 워커
        스레드가 cv2.imshow를 직접 부르지 않는 이유는 _worker()/DLSlideWindow.visualize()
        주석 참고. track_drive.py의 perc_lane()이 detect() 직후 이 메서드를 호출한다
        (다른 백엔드는 이 메서드가 없으므로 getattr(..., lambda: None)()로 안전하게 건너뜀)."""
        if not DEBUG_VIZ_DL_LANE:
            return
        with self._lock:
            vis, da_mask, ll_mask, ll_yellow_mask = self._latest_debug
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
        # 노란선 패널만 실제 노란색(BGR 0,255,255)으로 칠해서 흰/회색인 다른 패널과
        # 바로 구분되게 한다 — 단순 GRAY2BGR이면 da/ll처럼 흰색 마스크로 보여서
        # "이게 노란선 전용 패널"이라는 게 라벨 텍스트 말곤 안 보임.
        yellow_bgr = np.zeros((*ll_yellow_mask.shape, 3), dtype=np.uint8)
        yellow_bgr[ll_yellow_mask > 0] = (0, 255, 255)
        for label, panel in (('result', vis), ('da', da_bgr), ('ll', ll_bgr), ('yellow', yellow_bgr)):
            cv2.putText(panel, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow('dl_lane', cv2.vconcat([vis, da_bgr, ll_bgr, yellow_bgr]))
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
