#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# dl_lane.py — TwinLiteNet(ONNX Runtime) 기반 딥러닝 차선인식 백엔드.
#
# https://github.com/harrylal/TwinLiteNet-onnxruntime 의 사전학습 가중치(models/best.onnx)를
# 그대로 사용한다. hough_lane.HoughLaneDetector / perc_floor.LaneDetector와 동일하게
#   detect(frame) -> (lane_valid, lane_offset, lane_lookahead, lane_center, path, debug_img)
# 인터페이스를 구현하므로 track_drive.py의 perc_lane()은 수정 없이 그대로 재사용된다
# (LANE_DETECTOR_BACKEND 플래그로 세 백엔드 중 하나를 고른다). path는 da(주행가능영역)
# 마스크의 행(row)별 중심선에 lane_util._fit_and_sample_path()로 다항식을 피팅해 만든
# 명시적 경로(ROI 픽셀좌표 웨이포인트, 가까운점→먼점) — controller/pure_pursuit.py가
# 조향각 계산에 직접 사용한다.
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
# ── 양 옆 흰 차선(ll)만 경계로, 노란 중앙선은 완전히 무시 ──
#   기존엔 da(주행가능영역) 덩어리 하나의 가로 중심을 그대로 경로로 썼는데, da를 ll 경계로
#   클리핑하는 과정(_clip_da_by_ll, 폐기됨)에서 ll이 흰/노랑을 구분 못 하다 보니 도로 중앙의
#   노란 점선까지 "넘으면 안 되는 경계"로 오인해 차량이 항상 한쪽 차선에만 갇히는 문제가
#   있었다(노란선을 넘어 다녀도 되는 코스인데도). 그래서 da 기반 방식을 걷어내고, classic
#   파이프라인(lane_util.SlideWindow)과 같은 "ll을 좌/우로 나눠 슬라이딩윈도우(구간별
#   moments)로 각각 찾고 중점을 잇는" 방식으로 되돌리되, ll에서 노란색 위치(HSV 기반)를
#   미리 지워 "흰색 차선만" 좌/우 경계로 쓴다(YELLOW_LOWER/UPPER,
#   DL_YELLOW_EXCLUDE_MARGIN_PX 참고) — 노란 중앙선은 지나가도 아무 영향이 없고, 차량은
#   양쪽 흰 실선 사이 전체 폭에서 자유롭게 주행한다.
#   da_prob은 모델이 한 번의 forward pass로 어차피 같이 뽑아내므로 계산 낭비는 없지만,
#   이제 경로 계산에는 안 쓰고 디버그 오버레이(초록 영역, 참고용)에만 쓴다.
#   좌/우 조합 폴백(양쪽 검출→중점, 한쪽만 검출→lane_width로 반대쪽 추정)은
#   DLSlideWindow._combine_left_right()/_build_centerline_points()가 담당하며, classic과
#   달리 "양쪽 다 안 보이면 노란선으로 폴백"하는 4단계는 의도적으로 빼서 노란색이 경로
#   계산에 전혀 관여하지 않게 한다.
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

# da/ll 둘 다 (1,2,360,640) raw logit. 채널축 softmax 후 채널1이 foreground 확률.
DL_FG_THRESHOLD = 0.5   # 이진화 임계값(요구사항에 명시된 값)

# ── 세그멘테이션 입력은 절대 자르지 않는다 ──
#   원본 리포(blobFromImage)와 동일하게 raw 프레임 전체를 그대로 640x360으로 리사이즈해서
#   모델에 넣는다(추가 크롭 없음). 관심영역은 "모델에 들어가기 전"이 아니라 "모델에서 나온
#   세그멘테이션 결과(da/ll)를 원본 프레임 크기로 되돌린 뒤" 잘라서 쓴다 — 아래
#   DL_ROI_Y0/Y1 참고.

# ── 세그멘테이션 결과에서 좌/우 차선 중심을 뽑을 관심영역 ──
#   da/ll은 모델 고정 해상도(360행)로 나오지만, TwinLiteNetEngine.infer_raw()가 이걸
#   원본 카메라 프레임 크기(640x480)로 다시 업샘플링해서 돌려주므로, 여기 Y0/Y1은
#   (STOPLINE_ROI 등 프로젝트의 다른 ROI들과 동일하게) 원본 480행 기준 절대 픽셀값이다.
#   실차 실측값.
DL_ROI_Y0 = 250
DL_ROI_Y1 = 390

# ── SlideWindow moments 로직 재사용을 위한 DL 전용 튜닝값 ──
#   classic 파이프라인은 BEV로 워프된 ROI px 스케일이고, DL은 원본 카메라 프레임 px
#   스케일(BEV 없음, 위 DL_ROI_Y0:Y1로 자른 640폭 대역)이라 픽셀당 의미가 달라 값을 따로
#   둔다 — 알고리즘 자체는 lane_util.py의 MOMENT_*/LANE_SLICE_*/STABLE_* 와 동일(이름만
#   DL_ 접두어). 이제 좌/우 두 갈래가 아니라 da 중심선 "한 갈래"에만 적용된다.
#   실차 미검증 튜닝값.
DL_N_SLICES = 8              # 밴드를 classic(5)보다 세분화해 커브 추종을 촘촘하게 한다.
DL_MIN_PIXELS = 20           # 얇은 흰 ll 라인 기준(da 덩어리가 아니라 classic의 ll 기준(15)에
                              # 가깝게 낮춤) — 밴드 하나가 대부분 비어있으면 신뢰하지 않는다.
DL_NEAR_SLICES = 2
DL_FAR_SLICES = 2
DL_SLICE_OUTLIER_MAX = 60    # 640px 스케일(≈classic BEV ROI 폭의 2배)이라 허용폭도 비례해서 키움
DL_SLICE_FIT_MIN = 3

DL_STABLE_FRAME_MIN = 3      # "새 추론이 끝난 시점" 기준 연속 횟수(제어루프 틱 아님)
DL_STABLE_JUMP_MAX = 30      # 640px 스케일에 맞춰 classic(15px)의 약 2배

# da 파편화 대응(디버그 시각화 전용) — 경로/조향 계산은 이제 아래 ll 흰색 좌/우 라인만
# 쓰고 da는 안 쓴다(모듈 상단 "양 옆 흰 차선만 경계로" 주석 참고). da 자체는 모델이
# 한 번의 forward pass로 어차피 같이 뽑아내므로, 디버그 오버레이(초록 영역)를 깔끔하게
# 보여주는 참고용으로만 남겨둔다 — ConnectedComponents로 가장 큰 덩어리만 남긴다.
DL_DA_MIN_COMPONENT_AREA = 800

# ll(흰색 전용, 노란색 제외) sanity check 임계값 — ROI 내 흰 차선 foreground 비율이 이
# 미만이면 이번 프레임을 무효 처리한다(모션블러 등으로 세그멘테이션이 통째로 깨진 경우,
# 또는 흰 차선이 전혀 안 보이고 노란색만 보이는 경우 — 노란색은 더 이상 경계로 안 쓰므로
# 이때도 무효). 가는 선 하나가 640x140 ROI에서 차지하는 넓이 자체가 원래 작아서(정상
# 상태에서도 대략 1%대) 문턱을 낮게 잡았다. 실차 미검증 튜닝값.
DL_LL_SANITY_MIN_RATIO = 0.005

DEBUG_VIZ_DL_LANE = True   # lane_util.DEBUG_VIZ_LANE과 동일한 패턴의 디버그 창 on/off 스위치

# ── 노란 중앙선을 차선 경계에서 제외 ──
#   TwinLiteNet의 ll 출력은 흰/노랑을 구분하지 않는다. 그대로 두면 도로 중앙의 노란 점선까지
#   "경계"로 취급돼 차량이 그 선을 넘지 못하고 한쪽 차선에만 갇히는 문제가 있었다(실측으로
#   확인됨). 아래 HSV 임계값(hough_lane.py와 동일)으로 노란색 위치를 찾아 ll에서 지우고
#   "흰색 차선만" 좌/우 경계로 쓴다(DLSlideWindow.detect() 참고). track_drive.py의
#   _update_lane_side()(현재 어느 차선에 있는지 판정용)도 이 값을 그대로 재사용한다.
YELLOW_LOWER = np.array([15, 80, 80])
YELLOW_UPPER = np.array([40, 255, 255])
DL_YELLOW_EXCLUDE_MARGIN_PX = 10  # 노란선 두께+안티앨리어싱 여유(px) — 이만큼 팽창시켜 제외
_YELLOW_EXCLUDE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_RECT, (DL_YELLOW_EXCLUDE_MARGIN_PX * 2 + 1, DL_YELLOW_EXCLUDE_MARGIN_PX * 2 + 1)
)

# ── 좌/우 흰 차선 폭(白-白, px) — DL 원본 카메라 스케일(BEV 없음) ──
#   classic 파이프라인(lane_util.py)의 lane_width_min/max(180~400)는 BEV로 워프된 스케일이라
#   그대로 못 쓴다. DL ROI는 원본 카메라 프레임(640폭)을 그대로 쓰므로 근거리 밴드 기준
#   흰-흰 전체 폭이 훨씬 크게 잡힐 수 있다. 실차 미검증 추정치 — 실측 후 반드시 재조정할 것.
DL_LANE_WIDTH_MIN  = 200.0
DL_LANE_WIDTH_MAX  = 640.0
DL_LANE_WIDTH_INIT = 400.0

FPS_LOG_PERIOD_SEC = 5.0   # 워커 스레드 FPS/provider 로그 주기


def _default_model_path():
    """모델 가중치 파일(best.onnx) 기본 경로.
    1순위: colcon install된 share 디렉터리(share/track_drive/models/best.onnx)
    2순위: 소스트리에서 직접 실행 중일 때(개발 중, colcon build 전) — 이 파일 기준 상대경로
    """
    if get_package_share_directory is not None:
        try:
            share_dir = get_package_share_directory('track_drive')
            candidate = os.path.join(share_dir, 'models', 'best.onnx')
            if os.path.isfile(candidate):
                return candidate
        except Exception:
            pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best.onnx')


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
    평균을 내고(_group_mean), 다항식으로 경로를 피팅/샘플링하고(_fit_and_sample_path),
    프레임 간 스파이크를 걸러내는(_debounce)" 범용 유틸리티들을 재사용한다.

    ll(차선) 마스크에서 HSV로 검출한 노란 중앙선 위치를 지워 "흰색 차선만" 남긴 뒤,
    그 흰 마스크를 좌/우로 나눠 각각 슬라이딩윈도우(구간별 moments)로 찾고 중점을 잇는다
    (classic 파이프라인 lane_util.SlideWindow.calc_center()와 같은 골격이지만, 노란선
    폴백 없이 좌/우 흰 차선만 쓴다 — _combine_left_right()/_build_centerline_points()
    참고). da(주행가능영역)는 더 이상 경로 계산에 안 쓰고 디버그 시각화 참고용으로만
    남겨둔다(모듈 상단 "양 옆 흰 차선만 경계로" 주석 참고).
    """

    def __init__(self):
        super().__init__(
            n_slices=DL_N_SLICES, min_pixels=DL_MIN_PIXELS,
            near_slices=DL_NEAR_SLICES, far_slices=DL_FAR_SLICES,
            slice_outlier_max=DL_SLICE_OUTLIER_MAX, slice_fit_min=DL_SLICE_FIT_MIN,
            stable_frame_min=DL_STABLE_FRAME_MIN, stable_jump_max=DL_STABLE_JUMP_MAX,
            lane_width_init=DL_LANE_WIDTH_INIT, lane_width_min=DL_LANE_WIDTH_MIN,
            lane_width_max=DL_LANE_WIDTH_MAX,
        )
        self.ll_coverage = 0.0       # 최근 프레임 ROI 내 흰색 ll(노란색 제외) foreground 비율
        self.da_mask_roi = None      # 시각화 전용(더 이상 경로 계산에는 안 쓰임)
        self.ll_mask_roi = None      # 시각화용(흰색 전용 ll 마스크 — 실제 좌/우 경계로 쓰는 것)
        self.yellow_mask_roi = None  # 시각화용(무시되는 노란 영역 참고 표시)

    def _largest_da_component(self, da_mask):
        """da 마스크에서 가장 큰 연결 덩어리 하나만 남기고 나머지(급커브 등에서 생기는
        파편)는 지운다. 덩어리가 DL_DA_MIN_COMPONENT_AREA보다 작으면(사실상 안 보임)
        빈 마스크를 반환한다. 경로 계산에는 더 이상 안 쓰고 디버그 오버레이 전용."""
        num, labels, stats, _ = cv2.connectedComponentsWithStats(da_mask, connectivity=8)
        if num <= 1:
            return np.zeros_like(da_mask)
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[best, cv2.CC_STAT_AREA] < DL_DA_MIN_COMPONENT_AREA:
            return np.zeros_like(da_mask)
        return np.where(labels == best, np.uint8(255), np.uint8(0))

    def _combine_left_right(self):
        """근거리(near)/원거리(far) 좌/우 흰 차선 그룹평균으로 lane_valid/offset/lookahead/
        lane_center를 계산한다. lane_util.SlideWindow.calc_center()와 동일한 1~3단계
        폴백(양쪽 검출→중점, 한쪽만 검출→lane_width로 반대쪽 추정)을 쓰지만, 4단계
        (노란선 폴백)는 의도적으로 뺐다 — 요구사항: 노란색은 완전히 무시하고, 양쪽 흰
        차선을 못 찾으면 그냥 무효 처리한다(옆 차선을 넘어서까지 노란선에 기대 주행하지
        않는다)."""
        near_left  = self._group_mean(self.left_centers,  self.near_slices, True)
        far_left   = self._group_mean(self.left_centers,  self.far_slices,  False)
        near_right = self._group_mean(self.right_centers, self.near_slices, True)
        far_right  = self._group_mean(self.right_centers, self.far_slices,  False)

        lane_valid = False
        near_center = far_center = None

        if near_left is not None and near_right is not None:
            width = near_right - near_left
            if self.lane_width_min < width < self.lane_width_max:
                alpha = 0.1
                self.lane_width = (1 - alpha) * self.lane_width + alpha * width
                near_center = (near_left + near_right) / 2.0
                far_center = (
                    (far_left if far_left is not None else near_left) +
                    (far_right if far_right is not None else near_right)
                ) / 2.0
                lane_valid = True
            # width가 비정상이면 반사 등으로 한쪽이 오검출됐을 가능성이 높은데 어느 쪽이
            # 맞는지 판단 근거가 없으므로 이번 프레임은 무효 처리한다(classic과 동일 원칙).
        elif near_left is not None:
            far_ref = far_left if far_left is not None else near_left
            near_center = near_left + self.lane_width / 2.0
            far_center = far_ref + self.lane_width / 2.0
            lane_valid = True
        elif near_right is not None:
            far_ref = far_right if far_right is not None else near_right
            near_center = near_right - self.lane_width / 2.0
            far_center = far_ref - self.lane_width / 2.0
            lane_valid = True
        # 흰 차선을 양쪽 다 못 찾으면 여기서 끝 — 노란선 폴백 없음(요구사항: 노란색 무시).

        offset = lookahead = 0.0
        if lane_valid:
            offset = near_center - self.roi_w / 2.0
            lookahead = far_center - self.roi_w / 2.0
        lane_center = self.roi_w / 2.0 + offset

        return lane_valid, offset, lookahead, lane_center

    def _build_centerline_points(self):
        """부모(SlideWindow)의 4단계 폴백 중 1~3단계(양쪽/한쪽 흰 차선)만 쓰고 4단계
        (노란선 폴백)는 뺀 버전 — 요구사항: 노란색은 경로(웨이포인트) 생성에도 전혀
        관여하지 않는다."""
        points = []
        for i in range(self.n_slices):
            l = self.left_centers[i] if i < len(self.left_centers) else None
            r = self.right_centers[i] if i < len(self.right_centers) else None

            if l is not None and r is not None:
                width = r[1] - l[1]
                if not (self.lane_width_min < width < self.lane_width_max):
                    continue
                y_ref, cx = (l[0] + r[0]) / 2.0, (l[1] + r[1]) / 2.0
            elif l is not None:
                y_ref, cx = l[0], l[1] + self.lane_width / 2.0
            elif r is not None:
                y_ref, cx = r[0], r[1] - self.lane_width / 2.0
            else:
                continue

            points.append((y_ref, cx))
        return points

    def detect(self, raw_bgr, da_prob, ll_prob, yellow_mask):
        """입력 : raw_bgr — 원본 카메라 프레임 그대로의 (H,W,3) BGR(크롭/리사이즈 없음)
                 da_prob — (H,W) float32 주행가능영역 foreground 확률. 이제 경로 계산에는
                   안 쓰고 디버그 오버레이(초록 영역)에만 참고용으로 쓴다.
                 ll_prob — 위와 같은 (H,W) float32 차선(ll) foreground 확률(색상 구분 없음).
                 yellow_mask — 위와 같은 (H,W) uint8 이진마스크(HSV 기반, da/ll과 무관) — 이
                   위치를 ll에서 지워 "흰색 차선만" 남기는 데 쓴다(노란 중앙선 완전 무시).
          출력 : lane_valid, offset, lookahead, lane_center, path — 기존 SlideWindow.calc_center()와
                 동일한 계약(같은 4-tuple+path 형태)이지만, classic 파이프라인(lane_util.py)과
                 동일하게 "좌/우 흰 차선을 각각 찾아 그 중점을 경로로" 쓴다.
          내부에서 DL_ROI_Y0:DL_ROI_Y1(원본 프레임 절대 픽셀)만 잘라서 계산한다.
        """
        h, _ = ll_prob.shape
        y0 = max(0, min(DL_ROI_Y0, h))
        y1 = max(y0, min(DL_ROI_Y1, h))

        ll_roi = ll_prob[y0:y1]
        ll_mask = (ll_roi >= DL_FG_THRESHOLD).astype(np.uint8) * 255

        yellow_roi = yellow_mask[y0:y1]
        # 노란 중앙선을 차선 경계에서 제외 — ll은 색상을 구분하지 않으므로, 별도 계산한
        # HSV 노란 마스크를 여기서 지워 "흰색 차선만" 남긴다(모듈 상단 주석 참고). dilate로
        # 노란선 두께+안티앨리어싱 여유를 준다.
        yellow_dilated = cv2.dilate(yellow_roi, _YELLOW_EXCLUDE_KERNEL)
        ll_white_mask = cv2.bitwise_and(ll_mask, cv2.bitwise_not(yellow_dilated))

        self.roi_h, self.roi_w = ll_white_mask.shape
        self.vis = raw_bgr[y0:y1].copy()

        self.ll_coverage = (
            float(np.count_nonzero(ll_white_mask)) / ll_white_mask.size if ll_white_mask.size else 0.0
        )

        # 좌/우 분리 기준점 — 직전 프레임 확정 lane_center(없으면 ROI 중앙). BEV가 없는
        # 원근시점이라 커브에서 도로가 좌우로 치우칠 수 있는데, 프레임마다 직전 확정값을
        # 기준으로 갱신하면 서서히 따라간다(급커브 한 프레임 안에서의 곡률까지는 못 따라감 —
        # classic 파이프라인의 고정 5등분 분리보다는 낫지만 완벽하지 않음, 실차 미검증).
        split_x = int(self._confirmed[3]) if self._confirmed is not None else self.roi_w // 2
        split_x = int(np.clip(split_x, 0, self.roi_w))

        self.left_centers = self._reject_outliers(self._slice_centers(
            ll_white_mask[:, :split_x], 0, (0, 255, 0)
        ))
        self.right_centers = self._reject_outliers(self._slice_centers(
            ll_white_mask[:, split_x:], split_x, (0, 255, 0)
        ))

        # 노란 중앙선 위치 — 경로 계산에는 이제 전혀 안 쓰지만, track_drive.py의
        # _update_lane_side()(현재 어느 차선에 있는지 판정, 추월 방향 결정용)가 여전히
        # 참조하므로 계속 채워둔다.
        ys, xs = np.nonzero(yellow_roi)
        self.yellow_centers = [(float(np.mean(ys)), float(np.mean(xs)))] if len(xs) else []

        lane_valid, offset, lookahead, lane_center = self._combine_left_right()

        # 명시적 경로(웨이포인트) — 유효 밴드가 2개 미만이면 fitted_path가 None이 되고,
        # 그 경우 _update_path()가 self.path를 갱신하지 않아(직전 프레임 값 유지)
        # offset/lane_offset과 동일한 "무효 프레임엔 마지막 값 유지" 원칙을 따른다.
        fitted_path = self._fit_and_sample_path(self._build_centerline_points())
        self._update_path(fitted_path)

        lane_valid, offset, lookahead, lane_center = self._debounce(
            lane_valid, offset, lookahead, lane_center
        )

        # 디버그 시각화용 원본 da 마스크(참고용, 경로 계산에는 안 씀 — 위 docstring 참고)
        da_roi = da_prob[y0:y1]
        da_mask = (da_roi >= DL_FG_THRESHOLD).astype(np.uint8) * 255
        self.da_mask_roi = self._largest_da_component(da_mask)
        self.ll_mask_roi = ll_white_mask
        self.yellow_mask_roi = yellow_roi

        self.visualize(offset)

        return lane_valid, offset, lookahead, lane_center, self.path

    def visualize(self, offset):
        """da(초록, 참고용)/흰 ll(빨강, 실제 경계)/노란(파랑, 무시됨) 반투명 오버레이 +
        좌/우 차선 관측점 + 피팅된 경로 + offset/lane_center 텍스트를 self.vis에 그려
        넣기만 한다. ★ 여기서 cv2.imshow()/cv2.waitKey()를 호출하면 안 된다 ★ 이 메서드는
        DLLaneDetector의 백그라운드 추론 스레드에서 호출되는데, OpenCV HighGUI(특히 GTK
        백엔드)는 스레드 세이프하지 않아서 메인 스레드(다른 디버그 창들이 이미 거기서
        cv2.imshow/waitKey를 부르고 있음)와 다른 스레드가 동시에 GUI를 건드리면 초반 몇
        초는 멀쩡하다가 GTK 이벤트루프가 통째로 멈추는 형태로 실차에서 재현됐다(freeze).
        그래서 그리기(cv2.rectangle/circle/putText/addWeighted, 창 없이 이미지 버퍼에만
        작동)만 여기서 하고, 실제 창 표시는 DLLaneDetector.show_debug_windows()가 메인
        스레드(perc_lane() 호출 시점)에서 담당한다."""
        if self.vis is None:
            return

        if DEBUG_VIZ_DL_LANE:
            overlay = self.vis.copy()
            if self.da_mask_roi is not None:
                overlay[self.da_mask_roi > 0] = (0, 160, 0)      # 참고용 da(더 이상 경로에는 안 씀)
            if self.yellow_mask_roi is not None:
                overlay[self.yellow_mask_roi > 0] = (200, 0, 0)  # 무시되는 노란 중앙선(파랑)
            if self.ll_mask_roi is not None:
                overlay[self.ll_mask_roi > 0] = (0, 0, 220)      # 실제로 쓰는 흰 차선 경계(빨강)
            cv2.addWeighted(overlay, 0.35, self.vis, 0.65, 0, dst=self.vis)

        self.draw_centers(self.left_centers, (0, 255, 255))
        self.draw_centers(self.right_centers, (0, 255, 255))
        self.draw_path(self.path)                          # 피팅된 최종 경로(자홍색)

        cv2.line(self.vis, (self.roi_w // 2, 0), (self.roi_w // 2, self.roi_h), (0, 0, 255), 1)
        lane_center = self.roi_w / 2.0 + offset
        cv2.putText(
            self.vis, f'offset:{offset:+.1f} center:{lane_center:.1f} white_cov:{self.ll_coverage:.3f}',
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

    def show_debug_windows(self):
        """da(초록)/ll(빨강) 오버레이가 그려진 최근 결과를 디버그 창으로 띄운다.
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
        cv2.imshow('dl_lane_da', da_mask)
        cv2.imshow('dl_lane_ll', ll_mask)
        cv2.imshow('dl_lane_result', vis)
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