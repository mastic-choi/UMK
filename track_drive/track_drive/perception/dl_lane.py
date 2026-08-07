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
# 중심점(config.DL_CENTER_MODE='da'면 da 무게중심, 'll_da'면 ll이 확실한 밴드는 ll 좌/우
# 중점 그 외는 da 무게중심으로 폴백, 'll'이면 ll이 확실한 밴드만 쓰고 그 외는 무효 —
# 아래 "밴드별 중심 계산" 참고)에
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
# ── 밴드별 중심 계산: 'da' 단독 / 'll(차선)+da' 하이브리드 / 'll' 단독 — config.DL_CENTER_MODE ──
#   DL_CENTER_MODE='da'(main 기본값):
#   da(주행가능영역)를 "주행 가능한 영역 하나의 덩어리"로 보고
#   그 가로 중심을 밴드(행 구간)별로 바로 뽑는다 — 좌/우 ll을 따로 찾아 폭을 추정해
#   중점을 계산하는 간접적인 방식보다 단순하고, 가는 선 하나가 반사/그림자로 끊기는 것보다
#   넓은 덩어리가 노이즈에 더 안정적이라는 판단.
#   DL_CENTER_MODE='ll_da'/'ll': da 무게중심은 "주행 가능한 영역"이지 "차로 중앙"이 아니므로,
#   갓길 등 여백이 넓은 구간에서 무게중심이 여백 쪽으로 쏠려 경로가 차로 중앙을 벗어나는
#   문제가 실측으로 확인됐다. ll(차선 자체, 두 백선)은 여백 크기와 무관하게 "선이 실제로
#   있는 위치"만 가리키므로, 밴드마다 좌/우 ll이 둘 다 신뢰할 만큼 보이면 그 중점을
#   채택한다(DLSlideWindow._ll_slice_centers()). 'll_da'는 ll이 부족한 밴드만 da 중심으로
#   개별 폴백하고(프레임 전체 무효화 아님), 'll'은 그 폴백 없이 해당 밴드를 그냥 무효(None)로
#   둔다 — da가 섞여 여백 쪽으로 쏠리는 걸 완전히 배제하고 싶을 때 쓰는 모드
#   (config.py DL_CENTER_MODE 주석 참고).
#   세 모드 공통: 급커브에서 da 마스크가 파편화되는 실패모드에 대응해 ConnectedComponents로
#   가장 큰 덩어리 하나만 남기고(DL_DA_MIN_COMPONENT_AREA 미만이면 그 프레임은 무효 처리)
#   나머지 파편에 중심선이 끌려가지 않게 막는다(_largest_da_component() 참고). 또한 da가
#   점선 틈으로 옆 차선 da와 하나의 덩어리로 이어붙는 실패모드에 대해서는
#   DLSlideWindow._clip_da_by_ll()이 ll 라인이 보이는 구간에서 그 바깥(옆 차선 쪽) da
#   픽셀을 밴드별로 잘라내고 나서 _largest_da_component()를 적용한다(da 자체의 방어선 —
#   'da' 모드에서도 동작). ll은 그 외에 프레임 단위 sanity check로도 쓰인다 — ROI 내
#   커버리지가 DL_LL_SANITY_MIN_RATIO 미만이면(사실상 차선 신호가 전혀 없는 프레임 —
#   모션블러 등) da가 뭘 내놓든 이번 프레임을 무효 처리한다. 디버그 시각화(빨강 반투명
#   오버레이)에도 그대로 쓴다.
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
    DL_DA_MIN_COMPONENT_AREA, DL_DA_MAX_AREA_PX,
    DL_LL_SANITY_MIN_RATIO, DL_LL_CLIP_MARGIN_PX,
    DL_CENTER_MODE, DL_LL_SIDE_MIN_PIXELS, DL_LL_WIDTH_MIN_PX, DL_LL_WIDTH_MAX_PX,
    DL_LL_SEARCH_HALF_WIDTH_PX,
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
    config.DL_CENTER_MODE로 둘 중 하나를 고른다 — 'da'는 da 무게중심을 밴드별 중심으로
    바로 쓰고, 'll_da'는 밴드마다 "ll이 확실하면 ll 중점을, 아니면 da 무게중심을" 쓰는
    하이브리드다(모듈 상단 주석 참고). 어느 모드든 좌/우 두 갈래를 따로 다룰 필요가 없어
    calc_center()의 프레임 단위 4단계 분기와는 다르고, 그래서 여전히 calc_center()를
    호출하지 않고 detect() 안에서 직접 조립한다. da는 여백(갓길 등)이 넓으면 무게중심이
    차로 중앙에서 벗어나는 약점이 있지만 ll보다 끊기지 않는다는 장점이 있어, 'll_da'
    모드에서 ll이 부족한 밴드(점선 틈, 마모, 반사, 편측 가려짐)의 안전망으로 쓰인다. 그
    외에 da가 옆 차선까지 이어붙었을 때 그 경계를 잘라내는 방어선(_clip_da_by_ll())과
    프레임 단위 sanity check로는 모드와 무관하게 항상 ll을 쓴다(모듈 상단 주석 참고).
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
        self.da_largest_mask_roi = None  # 면적 1위 덩어리(차선책을 썼다면 그 사유가 된, 상한 초과로 버려진 덩어리) — fallback일 때 원래 색으로 같이 그리기용
        self.da_largest_area_px = 0  # 면적 1위 덩어리의 절대 픽셀 면적(채택 여부 무관) — DL_DA_MAX_AREA_PX 실측 튜닝용
        self._prev_da_centroid = None  # [2026-08-07] 직전 프레임에 채택된 da 덩어리의 중심(cx,cy) — _largest_da_component()가 이번 프레임 후보를 "가장 가까운 것"으로 고르는 기준. 무효 프레임 뒤엔 None으로 리셋(옛 위치에 계속 붙잡히지 않도록)
        self.da_chosen_area_px = 0   # 실제로 채택돼 waypoint 추출에 쓰인 덩어리의 면적(무효 프레임엔 0)

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

        가장 큰 덩어리가 DL_DA_MAX_AREA_PX(절대 픽셀수)를 넘을 만큼 크면 그 덩어리는
        outlier로 버린다. 정상적인 자기 차선 폭이라면 이 정도로 넓을 수 없는데, 실측으로
        확인된 두 실패모드가 여기 해당한다:
          ① ㅓ교차로 등 분기에서 da가 옆 갈래까지 하나로 이어붙는 경우
          ② 차선(백선) 자체가 없는 맨바닥을 통째로 주행가능영역으로 오검출하는 경우
        두 경우 모두 원인은 다르지만 "정상보다 비정상적으로 넓다"는 신호는 공통이라
        같은 임계값 하나로 같이 걸러낸다.
        [2026-08-06] 원래는 마스크 전체 대비 비율(DL_DA_MAX_AREA_RATIO=0.6, DL_USE_BEV
        캔버스 크기가 바뀌어도 재계산 없이 유효하다는 장점)로 잡았는데, 실차에서 직선
        구간 da 면적을 직접 실측해 절대 픽셀값(DL_DA_MAX_AREA_PX)으로 교체했다 — 비율은
        "대충 이 정도면 비정상적으로 넓다"는 추정이었지만, 절대값은 "정상 직선 구간에서
        실제로 관찰되는 면적"에 직접 근거하므로 더 정확하다. 대신 원거리 크롭
        (DL_BEV_FAR_LIMIT_M, §2.5) 등으로 캔버스 크기 자체가 바뀌면 이 값도 다시
        실측해야 한다(비율 방식과 달리 자동으로 안 따라감).

        [2026-08-06] 가장 큰 덩어리를 버린 뒤 곧바로 빈 마스크(=이번 프레임 무효)로
        처리하지 않고, 그다음으로 큰 덩어리부터 순서대로 [MIN_COMPONENT_AREA,
        MAX_AREA_PX] 범위 안에 드는 것을 찾아 대신 채택한다("차선책"). S자 연속
        커브 구간에서 실측으로 확인된 문제: 첫 덩어리가 옆 차로/노면 반사와 붙어
        면적 상한에 걸리는 프레임이 길게 이어지면, 예전 방식(그냥 무효 처리)은
        self.path가 몇 프레임짜리 튐이 아니라 사실상 무한정 정지 상태로 얼어붙어
        조향이 점점 벌어진 stale 경로를 계속 따라가다 turn_for_speed가 포화되어
        실차 속도가 바닥값까지 떨어지는(=사실상 정지) 결과로 이어졌다. 두 번째로 큰
        덩어리가 범위 안에 들면 자기 차선일 가능성이 높으므로(같은 프레임에서 옆
        차로 덩어리와 분리돼 있었다는 뜻) 이걸로 대체하는 편이 "몇 초씩 정지"보다
        낫다 — self.da_fallback_used로 이번 프레임이 차선책을 썼는지 표시해
        visualize()가 다른 색으로 구분해 그린다(디버깅용, 실제 판단 로직에는
        영향 없음). 이때 버려진 면적 1위 덩어리 자체도 self.da_largest_mask_roi에
        따로 남겨서, visualize()가 "원래(가장 큰) da"와 "실제로 채택한 차선책 da"를
        동시에(각각 원래색/차선책색으로) 그릴 수 있게 한다.

        [2026-08-07] 차선책을 "면적 내림차순"만으로 고르면, 실제로는 계속 같은 차선을
        보고 있는데도 두 덩어리 크기가 비슷해 프레임마다 순위가 뒤집히는 것만으로
        채택 대상이 바뀌어 지금 따라가던 경로가 불필요하게 흔들리는 문제가 있었다
        (실측 재현됨). 그래서 순위보다 "연속성"을 우선한다 — self._prev_da_centroid
        (직전 프레임에 실제로 채택된 덩어리의 중심)와 가장 가까운 덩어리를 최우선
        후보로 고정하고, 그 후보의 면적이 [MIN_COMPONENT_AREA, MAX_AREA_PX] 범위
        안이면 순위와 무관하게 바로 채택한다. 이 근접 후보가 범위를 벗어났을 때만
        (교차로에서 실제로 다른 갈래로 넘어갔거나, 따라가던 덩어리가 사실상 사라진
        경우) 기존 면적 내림차순 차선책으로 넘어간다. 무효 프레임(빈 마스크 반환)
        뒤에는 self._prev_da_centroid를 None으로 리셋해, 한참 뒤에 엉뚱한 위치의
        덩어리가 "옛 중심과 가장 가깝다"는 이유만으로 잘못 이어붙는 것을 막는다."""
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(da_mask, connectivity=8)
        self.da_fallback_used = False
        self.da_largest_mask_roi = None
        self.da_largest_area_px = 0
        self.da_chosen_area_px = 0
        if num <= 1:
            self._prev_da_centroid = None
            return np.zeros_like(da_mask)

        areas = stats[1:, cv2.CC_STAT_AREA]
        comp_centroids = centroids[1:]  # centroids[0]은 배경 — stats[1:]와 동일하게 인덱스 정렬
        order = np.argsort(areas)[::-1]  # 큰 덩어리부터 — 차선책(연속성 후보 탈락 시) 폴백용
        min_area = DL_DA_MIN_COMPONENT_AREA
        max_area = DL_DA_MAX_AREA_PX

        largest_label = 1 + int(order[0])
        self.da_largest_mask_roi = np.where(labels == largest_label, np.uint8(255), np.uint8(0))
        # [2026-08-06] 실측 튜닝용 — 가장 큰 덩어리 면적(채택 여부와 무관)을 항상 기록해둔다.
        # DEBUG_VIZ_STEER 창(track_drive.py _debug_viz_steer())이 이 값을 그대로 읽어서
        # "직선 구간 da 면적이 실제로 몇 px인지" 실측하는 용도로 쓴다 — DL_DA_MAX_AREA_PX를
        # 이 실측값 기반으로 잡기 위함(config.py 주석 참고).
        self.da_largest_area_px = int(areas[order[0]])

        def _choose(idx, fallback):
            label = 1 + int(idx)
            self.da_fallback_used = fallback
            self.da_chosen_area_px = int(areas[idx])  # 실제로 채택돼 waypoint 추출에 쓰이는 면적
            self._prev_da_centroid = (float(comp_centroids[idx][0]), float(comp_centroids[idx][1]))
            return np.where(labels == label, np.uint8(255), np.uint8(0))

        # 직전에 채택한 덩어리가 있으면 그 중심과 가장 가까운 덩어리를 최우선 후보로 고정
        # (연속성 유지) — 범위 안이면 면적 순위와 무관하게 바로 채택한다.
        if self._prev_da_centroid is not None:
            px, py = self._prev_da_centroid
            dists = np.hypot(comp_centroids[:, 0] - px, comp_centroids[:, 1] - py)
            nearest_idx = int(np.argmin(dists))
            nearest_area = int(areas[nearest_idx])
            if min_area <= nearest_area <= max_area:
                return _choose(nearest_idx, fallback=False)
            # 근접 후보가 범위를 벗어남(상한 초과/하한 미만) — 아래 면적 순위 차선책으로 이동

        for rank, idx in enumerate(order):
            area = int(areas[idx])
            if area < min_area:
                break  # 내림차순 정렬이라 이후는 전부 더 작다 — 더 볼 필요 없음
            if area <= max_area:
                # 직전 연속 후보가 없어서(첫 프레임 등) 면적 1위를 그대로 쓴 경우만
                # fallback=False — 그 외(연속 후보가 있었는데 범위를 벗어나 여기로
                # 온 경우, 또는 1위가 상한에 걸려 2번째 이하를 쓴 경우)는 전부 차선책.
                fallback = (rank > 0) or (self._prev_da_centroid is not None)
                return _choose(idx, fallback=fallback)
            # 이 덩어리는 outlier(면적 상한 초과) — 다음으로 큰 덩어리를 시도

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
        잘라낸 da 밴드의 실제 중심으로 갱신한다(커브를 따라 기준점도 같이 휘어지게). 밴드 안에
        ll이 한쪽만 보이면 그쪽만 자르고, 양쪽 다 안 보이면(가려짐/마모 등) 이번 밴드는 자르지
        않고 da를 그대로 둔다 — ll이 확실할 때만 개입한다("da를 경로의 주 신호로, ll은 보강" —
        모듈 상단 주석 참고).
          ★ cur_ref는 반드시 "이번 밴드에서 ll이 실제로 보여서 클리핑 근거가 있었을 때만"
          갱신한다(아래 if ll_cols.size 안에서만 재계산) ★ — ll이 안 보이는 밴드(점선 틈 등)는
          da가 옆 차선까지 안 잘린 채 그대로 남아있을 수 있는데, 그 밴드의 컬럼 평균을 그대로
          다음(더 먼) 밴드의 기준점으로 넘기면 오염된 기준이 근거리→원거리로 계속 누적(cascade)
          된다 — 한 프레임 안에서 점선 틈 하나가 그 위 모든 밴드의 좌/우 판정을 연쇄적으로
          틀어지게 만드는 실패모드가 실측으로 확인됨(여러 밴드가 "같은 방향으로" 같이 밀리면
          _reject_outliers()의 leave-one-out 추세 검사도 못 잡아낸다 — 이상치 하나가 아니라
          추세 자체가 휜 것처럼 보이기 때문). ll이 안 보인 밴드는 기준점을 갱신하지 않고 직전
          확정 기준을 그대로 들고 다음 밴드로 넘어가서 오염 전파를 그 밴드 하나로 막는다.
          입력 : da_mask, ll_mask — 동일 shape의 (roi_h, roi_w) uint8 이진마스크
                 ref_x           — 첫(근거리) 밴드의 기준 x좌표. 보통 직전 프레임 lane_center.
          출력 : da_mask에서 ll 경계 밖 픽셀만 0으로 지운 복사본(shape 동일).
        """
        h, w = da_mask.shape
        slice_h = h // self.n_slices
        clipped = da_mask.copy()
        cur_ref = ref_x

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

        return clipped

    def _ll_slice_centers(self, ll_mask, ref_x):
        """DL_CENTER_MODE='ll_da' 또는 'll'일 때 호출된다. ll_mask(차선 이진마스크)를
        _slice_centers()와 동일한 n_slices 밴드로 나눠, 밴드마다 좌/우 각각 **좁은
        고정폭 탐색창**(DL_LL_SEARCH_HALF_WIDTH_PX 반경) 안에서만 무게중심(cv2.moments)을
        구한다. 양쪽 다 DL_LL_SIDE_MIN_PIXELS 이상이고 두 중심 간 거리가 실측 차로폭
        범위(DL_LL_WIDTH_MIN_PX~DL_LL_WIDTH_MAX_PX) 안에 들 때만 그 중점을 이번 밴드의
        결과로 채택한다 — 한쪽만 보이거나 폭이 비정상이면(반대 차선을 잘못 짝짓는 등)
        신뢰할 수 없다고 보고 None을 반환한다. 이 None을 detect()가 'll_da'에서는 그
        밴드만 da로 폴백시키는 신호로, 'll'에서는 그대로 무효 밴드로 쓰는 신호로 쓴다.

        [2026-08-07] 원래는 좌/우를 나누는 기준점(cur_ref) 하나로 밴드 전체를 반씩(왼쪽
        전체/오른쪽 전체) 나눠 그 안 전체 픽셀로 무게중심을 냈다. ROI 폭이 넓으면 그
        "반쪽"도 수백 px라, 옆 차선 선이나 반사광이 그 반쪽 어디에 있든 평균에 그대로
        섞여 들어가는 문제가 있었다(지난 대화의 "ll 다중 후보 선택 문제"). 참고 프로젝트
        (github.com/junhyukch7/Advanced-Lane-Detection)의 슬라이딩 윈도우처럼, 좌/우 각각
        예상 위치를 중심으로 좁은 창(고정폭)만 보게 바꿔 창 밖의 무관한 픽셀이 애초에
        평균 계산에 안 들어오게 했다. 좌/우 각자의 중심(cur_left/cur_right)은
        _clip_da_by_ll()과 같은 원칙으로 근거리→원거리로 올라가며 갱신하되, ★ 이번
        밴드에서 실제로 채택(양쪽 다 신뢰됨)됐을 때만 갱신한다 ★ — 그래야 ll이 부족해
        이번 밴드를 못 쓴 경우, 그 밴드의 (있을 수도 있는) 애매한 위치가 다음 밴드의
        좌/우 탐색창 기준으로 오염되어 누적되는 걸 막는다. 좌/우 초기 위치는 ref_x(직전
        프레임 확정 lane_center) 기준 ±(기대 차로폭/2)로 잡는다.

          입력 : ll_mask — (roi_h, roi_w) uint8 이진마스크(da_mask와 동일 shape/좌표계)
                 ref_x   — 첫(근거리) 밴드의 좌/우 분리 기준 x좌표. 보통 직전 프레임 lane_center.
          출력 : (results, used) — 둘 다 길이 self.n_slices.
                 results[i] : 채택되면 (y_center, cx), 아니면 None(da로 폴백해야 함을 뜻함)
                 used[i]    : results[i]가 ll 기반으로 채택됐는지(bool) — 디버그 시각화용

        알려진 한계: 탐색창이 좁아진 만큼, 급커브에서 밴드 간 실제 선 이동량이
        DL_LL_SEARCH_HALF_WIDTH_PX보다 크면 창이 선을 놓치고 그 밴드부터 계속 이전
        위치에 멈춰 서게 된다(추적 이탈) — 실차 미검증, 급커브 구간에서 ll_bands 비율이
        갑자기 뚝 떨어지는지 확인할 것."""
        h, w = ll_mask.shape
        slice_h = h // self.n_slices
        results = [None] * self.n_slices
        used = [False] * self.n_slices

        half_lane = (DL_LL_WIDTH_MIN_PX + DL_LL_WIDTH_MAX_PX) / 4.0  # 기대 차로폭의 절반 — 좌/우 초기 위치 추정용
        cur_left = ref_x - half_lane
        cur_right = ref_x + half_lane
        win = DL_LL_SEARCH_HALF_WIDTH_PX

        for i in range(self.n_slices):
            y_high = h - i * slice_h
            y_low = 0 if i == self.n_slices - 1 else h - (i + 1) * slice_h
            band = ll_mask[y_low:y_high, :]
            y_center = (y_low + y_high) / 2.0

            lx0, lx1 = int(np.clip(cur_left - win, 0, w)), int(np.clip(cur_left + win, 0, w))
            rx0, rx1 = int(np.clip(cur_right - win, 0, w)), int(np.clip(cur_right + win, 0, w))
            if lx1 <= lx0 or rx1 <= rx0:
                continue  # 탐색창이 화면 밖으로 완전히 밀려남(추적 이탈) — 스킵, cur_left/right 유지

            M_l = cv2.moments(band[:, lx0:lx1], binaryImage=True)
            M_r = cv2.moments(band[:, rx0:rx1], binaryImage=True)

            if M_l['m00'] < DL_LL_SIDE_MIN_PIXELS or M_r['m00'] < DL_LL_SIDE_MIN_PIXELS:
                continue  # 한쪽 이상 안 보임 — 이 밴드는 da로 폴백(cur_left/right도 갱신 안 함)

            lx = lx0 + M_l['m10'] / M_l['m00']
            rx = rx0 + M_r['m10'] / M_r['m00']
            width = rx - lx
            if not (DL_LL_WIDTH_MIN_PX < width < DL_LL_WIDTH_MAX_PX):
                continue  # 폭이 비정상 — 반대 차선을 잘못 짝지었을 가능성, da로 폴백

            cx = (lx + rx) / 2.0
            results[i] = (y_center, cx)
            used[i] = True
            cur_left = lx
            cur_right = rx

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

        # 급커브 파편화 대응: 가장 큰 덩어리만 남긴다(모듈 상단 주석 참고). ll 클리핑보다
        # 먼저 해야 한다 — 이 시점엔 da가 아직 하나의 연결된 덩어리라 "가장 큰 덩어리"가
        # 곧 도로 전체를 뜻하지만, 클리핑을 먼저 하면 밴드마다 독립적으로 좌우를 잘라
        # 인접 밴드끼리 남은 x범위가 안 겹치는 경우가 생겨(ll 경계가 밴드마다 조금씩
        # 다르게 잡히면 흔함) 마스크가 밴드별로 끊기고, 그 뒤 largest_da_component가
        # "가장 큰 덩어리"로 폭 넓은 밴드 하나만 통째로 골라버려 도로 모양이 아니라
        # 네모난 밴드 하나만 남는 문제가 생긴다(실측으로 확인됨).
        da_mask = self._largest_da_component(da_mask)

        # 옆 차선 침범 대응: ll 라인이 보이는 구간에서는 그 바깥(옆 차선 쪽) da를 잘라낸다
        # (모듈 상단 주석 참고). ref_x는 직전 프레임 확정 lane_center — 아직 없으면(첫
        # 프레임) ROI 중앙을 기준으로 시작한다. 이 클리핑이 밴드 간 연결을 끊어도 상관없다
        # — 아래 _slice_centers()는 밴드별로 독립적으로 moments를 구하므로 전역 연결성이
        # 필요 없다(그래서 여기선 largest_da_component를 다시 돌리지 않는다).
        #
        # [2026-08-06] 클리핑 결과가 fit 가능한 최소 밴드 수(DL_SLICE_FIT_MIN)에 못
        # 미치면 클리핑을 버리고 클리핑 전 da로 되돌린다("차선책", _largest_da_component()의
        # 면적상한 폴백과 같은 원칙). S자 연속 커브에서 원거리 ll이 DL_LL_FG_THRESHOLD를
        # 올려도 여전히 두껍게 잡히면 _clip_da_by_ll()이 여러 밴드를 통째로 깎아버려
        # da가 "작게 검출된" 것처럼 보이는 경우가 실측으로 확인됨 — da 자체는 멀쩡한데
        # ll 클리핑이 지워버린 것이므로, 이럴 땐 클리핑 없는(=옆 차선 침범 위험은 있지만
        # 최소한 주행은 하는) da를 쓰는 편이 self.path가 무한정 얼어붙어 완전정지하는
        # 것보다 낫다는 판단. self.da_ll_clip_skipped로 표시해 visualize()가 구분 표시한다.
        ref_x = self._confirmed[3] if self._confirmed is not None else da_mask.shape[1] / 2.0
        clipped = self._clip_da_by_ll(da_mask, ll_mask, ref_x)
        clipped_valid = sum(1 for c in self._slice_centers(clipped, 0, (0, 255, 0)) if c is not None)
        self.da_ll_clip_skipped = clipped_valid < self.slice_fit_min
        da_mask = da_mask if self.da_ll_clip_skipped else clipped

        self.da_mask_roi = da_mask

        # 밴드별 중심점 — DL_CENTER_MODE='da'면 da 무게중심을 그대로 쓰고(_slice_centers는
        # 색상/의미에 상관없이 "임의의 이진마스크를 세로로 N등분해 구간별 moments 중심을
        # 구하는" 범용 로직), 'll_da'면 밴드마다 ll이 신뢰되면 ll 중점을 우선 채택하고
        # 그 외 밴드만 da 중심으로 폴백하며, 'll'은 그 da 폴백 없이 ll이 신뢰되는 밴드만
        # 쓰고 나머지는 무효(None)로 둔다(모듈 상단 주석 참고) — None은 이후
        # _reject_outliers/_fit_and_sample_path가 원래부터 걸러내는 값이라 별도 처리가
        # 필요 없다. ll_mask는 클리핑 전 원본을 쓴다 — _clip_da_by_ll()의 클리핑은 da를
        # 깎아내기 위한 것이지 ll 자체의 좌/우 라인 위치를 바꾸는 게 아니므로 그대로
        # 재사용해도 무방하다. ref_x는 위에서 _clip_da_by_ll()에 쓴 것과 같은 기준점
        # (직전 프레임 확정 lane_center, 없으면 ROI 중앙)을 그대로 재사용한다.
        raw_da_centers = self._slice_centers(da_mask, 0, (0, 255, 0))
        if DL_CENTER_MODE == 'll_da':
            raw_ll_centers, self.ll_band_used = self._ll_slice_centers(ll_mask, ref_x)
            merged_centers = [
                ll_c if ll_c is not None else da_c
                for ll_c, da_c in zip(raw_ll_centers, raw_da_centers)
            ]
        elif DL_CENTER_MODE == 'll':
            merged_centers, self.ll_band_used = self._ll_slice_centers(ll_mask, ref_x)
        else:
            merged_centers = raw_da_centers
            self.ll_band_used = [False] * len(raw_da_centers)
        self.centerline = self._reject_outliers(merged_centers)

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
        ll클리핑 건너뜀이면 청록)/ll(빨강) 반투명 오버레이 + da 중심선 관측점 + 피팅된 경로 +
        offset/lane_center 텍스트를 self.vis에
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
            overlay[self.ll_mask_roi > 0] = (0, 0, 220)    # 차선 = 빨강 (da 위에 덧그림)
            cv2.addWeighted(overlay, 0.35, self.vis, 0.65, 0, dst=self.vis)

        # 밴드별 중심점 — DL_CENTER_MODE='ll_da'에서 ll 기반으로 채택된 밴드는 흰색, da로
        # 폴백한 밴드는 노란색으로 구분해서(_ll_slice_centers()/detect() 병합 로직 참고)
        # 어느 밴드가 ll을 못 써서 da로 대체됐는지 한눈에 보이게 한다('da' 모드에선
        # ll_band_used가 항상 전부 False라 전부 노란색 — 기존 draw_centers() 색과 동일해
        # 시각적으로 하위호환. 'll' 모드에선 애초에 da 폴백이 없어 그려지는 점은 전부
        # 흰색이고, ll이 부족해 무효 처리된 밴드는 centerline이 None이라 아예 안 그려짐).
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
        cv2.putText(
            self.vis,
            f'offset:{offset:+.1f} center:{lane_center:.1f} ll_cov:{self.ll_coverage:.3f} '
            f'mode:{DL_CENTER_MODE} ll_bands:{ll_band_count}/{self.n_slices}{tags}',
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
        self._latest_debug = (None, None, None)   # (vis, da_mask_roi, ll_mask_roi)
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
                self._latest_debug = (self._slide.vis, self._slide.da_mask_roi, self._slide.ll_mask_roi)

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
        """da(초록)/ll(빨강) 오버레이가 그려진 result에 da/ll 원본 이진마스크를 위→아래로
        이어붙여 창 하나(`dl_lane`)로 띄운다 — result/da/ll 순서로 세로 스택.
        [2026-08-06] 예전엔 3개 별도 창(dl_lane_result/da/ll)이었는데, 창이 흩어져 있으면
        서로 다른 위치에 배치해야 해서 실차 테스트 중 한눈에 비교하기 불편하다는 피드백으로
        하나로 합쳤다. da/ll은 원래 1채널 이진마스크라 result(3채널 BGR)와 그대로 못
        이어붙이므로 BGR로 변환 후 vconcat한다 — result/da/ll 모두 같은 ROI에서 나온
        동일 shape(BEV 캔버스 크기)이라 폭이 항상 맞는다.

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
            vis, da_mask, ll_mask = self._latest_debug
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
        for label, panel in (('result', vis), ('da', da_bgr), ('ll', ll_bgr)):
            cv2.putText(panel, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow('dl_lane', cv2.vconcat([vis, da_bgr, ll_bgr]))
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
