#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# dl_lane.py — TwinLiteNet(ONNX Runtime) 기반 딥러닝 차선인식 백엔드.
#
# https://github.com/harrylal/TwinLiteNet-onnxruntime 의 사전학습 가중치(models/best.onnx)를
# 그대로 사용한다. hough_lane.HoughLaneDetector / perc_floor.LaneDetector와 동일하게
#   detect(frame) -> (lane_valid, lane_offset, lane_lookahead, lane_center, debug_img)
# 인터페이스를 구현하므로 track_drive.py의 perc_lane()은 수정 없이 그대로 재사용된다
# (LANE_DETECTOR_BACKEND 플래그로 세 백엔드 중 하나를 고른다).
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
#   안(DLSlideWindow.calc_center() 내부)에서만 적용된다.
#
# ── da(주행가능영역) 사용 정책 ──
#   급커브에서 da 마스크가 한두 프레임 파편화되는 실패모드가 이미 확인돼 있어서, 구간별로
#   강하게 게이팅하면 하필 급커브에서 차선 검출 자체를 죽이는 역효과가 난다. 그래서 "프레임
#   전체가 사실상 통째로 깨졌을 때"만 거르는 거친 안전장치로만 쓰고(DA_MIN_COVERAGE_RATIO),
#   그 외에는 디버그 시각화(초록 반투명 오버레이)에만 사용한다.
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

# ── ll(차선) 마스크에서 좌/우 차선 중심을 뽑을 관심영역(모델출력 세로 360행 기준 비율) ──
#   모델이 프레임 전체(천장/하늘 포함)를 세그멘테이션하므로, classic 파이프라인의
#   LANE_ROI_TOP/BOT과 같은 개념으로 원거리(위쪽) 구간을 잘라 천장 반사 등 노이즈를 줄인다.
#   실차 미검증 튜닝값.
DL_ROI_TOP = 0.40
DL_ROI_BOT = 1.0

# ── SlideWindow moments 로직 재사용을 위한 DL 전용 튜닝값 ──
#   classic 파이프라인은 BEV로 워프된 ROI px 스케일이고, DL은 640x360으로 리사이즈된 원본
#   프레임 px 스케일(BEV 없음)이라 픽셀당 의미가 달라 값을 따로 둔다 — 알고리즘 자체는
#   lane_util.py의 MOMENT_*/LANE_SLICE_*/STABLE_* 와 동일(이름만 DL_ 접두어).
#   실차 미검증 튜닝값.
DL_N_SLICES = 5
DL_MIN_PIXELS = 25          # 640px 폭 기준이라 classic(15)보다 살짝 높임
DL_NEAR_SLICES = 2
DL_FAR_SLICES = 2
DL_SLICE_OUTLIER_MAX = 60   # 640px 스케일(≈classic BEV ROI 폭의 2배)이라 허용폭도 비례해서 키움
DL_SLICE_FIT_MIN = 3

DL_STABLE_FRAME_MIN = 3     # "새 추론이 끝난 시점" 기준 연속 횟수(제어루프 틱 아님)
DL_STABLE_JUMP_MAX = 30     # 640px 스케일에 맞춰 classic(15px)의 약 2배

DL_LANE_WIDTH_INIT = 320.0
DL_LANE_WIDTH_MIN = 220.0
DL_LANE_WIDTH_MAX = 480.0

# da 안전장치 임계값 — ROI 내 da foreground 비율이 이 미만이면 이번 프레임의 ll 결과를
# 아예 안 믿는다(모션블러 등으로 세그멘테이션이 통째로 깨진 경우). 급커브 한두 프레임
# 파편화 정도로는 안 걸리도록 낮게 잡았다.
DA_MIN_COVERAGE_RATIO = 0.05

DEBUG_VIZ_DL_LANE = True   # lane_util.DEBUG_VIZ_LANE과 동일한 패턴의 디버그 창 on/off 스위치

# ── 색상기반 노란 중앙선 보조 검출 (lane_side 판정용) ──
#   TwinLiteNet의 ll 출력은 흰/노랑을 구분하지 않는다. track_drive.py의 _update_lane_side()가
#   yellow_centers에 의존하므로, hough_lane.py와 동일한 HSV 임계값으로 별도 계산해 채워준다.
#   ※ 이건 DL 모델의 출력이 아니라 명시적으로 추가하는 classic-CV 보조 컴포넌트다.
YELLOW_LOWER = np.array([15, 80, 80])
YELLOW_UPPER = np.array([40, 255, 255])

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
        """입력 : 임의 크기 BGR 프레임
        출력 : (resized_bgr, da_prob, ll_prob)
          resized_bgr — 640x360 BGR(디버그 시각화용, 마스크들과 같은 좌표계)
          da_prob, ll_prob — (360,640) float32 foreground 확률맵
        """
        t0 = time.perf_counter()
        resized_bgr, blob = self.preprocess(bgr_frame)
        da_out, ll_out = self.session.run(list(DL_OUTPUT_NAMES), {DL_INPUT_NAME: blob})
        da_prob = self.softmax_fg(da_out[0])
        ll_prob = self.softmax_fg(ll_out[0])
        dt = time.perf_counter() - t0
        self._latency_ema = dt if self._latency_ema is None else 0.8 * self._latency_ema + 0.2 * dt
        return resized_bgr, da_prob, ll_prob

    @property
    def fps(self):
        if not self._latency_ema:
            return 0.0
        return 1.0 / self._latency_ema


class DLSlideWindow(SlideWindow):
    """lane_util.SlideWindow(구간별 moments + 이상치제거 + 디바운스)를 그대로 재사용하되,
    입력을 BEV 워프된 흰/노랑 마스크 대신 TwinLiteNet의 ll(차선) 마스크로 바꾼 버전.
    _slice_centers()는 원래도 색상에 의존하지 않는 범용 로직(임의의 이진마스크를 세로로
    N등분해 구간별 무게중심을 구함)이라, classic이 흰색마스크를 좌/우 절반으로 잘라 넣던
    것과 똑같이 ll 마스크를 폭 절반으로 잘라 넣으면 그대로 동작한다. ll은 흰/노랑을
    구분하지 않으므로 yellow_centers는 별도 HSV 색상마스크로 채운다(lane_side 판정용,
    calc_center()의 '흰 차선 둘 다 놓쳤을 때만 노랑 사용' 폴백에도 그대로 활용됨).
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
        self.da_coverage = 0.0     # 최근 프레임 ROI 내 da foreground 비율(디버그/게이팅용)
        self.da_mask_roi = None    # 시각화용
        self.ll_mask_roi = None    # 시각화용

    def detect(self, resized_bgr, da_prob, ll_prob, yellow_mask):
        """입력 : resized_bgr — (DL_INPUT_H,DL_INPUT_W,3) BGR
                 da_prob, ll_prob — (DL_INPUT_H,DL_INPUT_W) float32 foreground 확률
                 yellow_mask — (DL_INPUT_H,DL_INPUT_W) uint8 이진마스크(HSV 기반, ll과 무관)
          출력 : lane_valid, offset, lookahead, lane_center — SlideWindow.calc_center()와 동일 계약
        """
        h, _ = ll_prob.shape
        y0, y1 = int(h * DL_ROI_TOP), int(h * DL_ROI_BOT)

        da_roi = da_prob[y0:y1]
        da_mask = (da_roi >= DL_FG_THRESHOLD).astype(np.uint8) * 255
        self.da_coverage = float(np.count_nonzero(da_mask)) / da_mask.size if da_mask.size else 0.0

        ll_roi = ll_prob[y0:y1]
        ll_mask = (ll_roi >= DL_FG_THRESHOLD).astype(np.uint8) * 255
        yellow_roi = yellow_mask[y0:y1]

        # da 안전장치: ll(DL 출력)의 신뢰도만 깎는다 — yellow_roi는 DL과 무관한 독립
        # classic-CV 신호라 여기서 같이 지우지 않는다(모듈 상단 "da 사용 정책" 주석 참고).
        # per-slice가 아니라 마스크 전체를 통째로 비워서, calc_center()의 기존 폴백 경로
        # (양쪽 실패 → 노랑 → 무효)를 그대로 타게 한다 — 디바운스 상태/시각화가 이 판단과
        # 항상 일관되게 유지된다.
        if self.da_coverage < DA_MIN_COVERAGE_RATIO:
            ll_mask = np.zeros_like(ll_mask)

        self.da_mask_roi = da_mask
        self.ll_mask_roi = ll_mask
        self.roi_h, self.roi_w = ll_mask.shape
        self.vis = resized_bgr[y0:y1].copy()

        half = self.roi_w // 2
        self.left_centers = self._reject_outliers(self._slice_centers(
            ll_mask[:, :half], 0, (0, 255, 0)
        ))
        self.right_centers = self._reject_outliers(self._slice_centers(
            ll_mask[:, half:], half, (0, 255, 0)
        ))
        self.yellow_centers = self._reject_outliers(self._slice_centers(
            yellow_roi, 0, (0, 180, 255)
        ))

        # calc_center() 내부에서 _debounce() + visualize()까지 처리한다(부모 클래스 그대로).
        return self.calc_center()

    def visualize(self, offset):
        """da(초록)/ll(빨강) 반투명 오버레이 + 좌/우/노랑 중심점 + offset/lane_center 텍스트를
        self.vis에 그려 넣기만 한다. ★ 여기서 cv2.imshow()/cv2.waitKey()를 호출하면 안 된다 ★
        이 메서드는 calc_center()를 통해 DLLaneDetector의 백그라운드 추론 스레드에서 호출되는데,
        OpenCV HighGUI(특히 GTK 백엔드)는 스레드 세이프하지 않아서 메인 스레드(다른 디버그
        창들이 이미 거기서 cv2.imshow/waitKey를 부르고 있음)와 다른 스레드가 동시에 GUI를
        건드리면 초반 몇 초는 멀쩡하다가 GTK 이벤트루프가 통째로 멈추는 형태로 실차에서
        재현됐다(freeze). 그래서 그리기(cv2.rectangle/circle/putText/addWeighted, 창 없이
        이미지 버퍼에만 작동)만 여기서 하고, 실제 창 표시는 DLLaneDetector.show_debug_windows()가
        메인 스레드(perc_lane() 호출 시점)에서 담당한다."""
        if self.vis is None:
            return

        if DEBUG_VIZ_DL_LANE and self.da_mask_roi is not None:
            overlay = self.vis.copy()
            overlay[self.da_mask_roi > 0] = (0, 200, 0)    # 주행가능영역 = 초록
            overlay[self.ll_mask_roi > 0] = (0, 0, 220)    # 차선 = 빨강 (da 위에 덧그림)
            cv2.addWeighted(overlay, 0.35, self.vis, 0.65, 0, dst=self.vis)

        self.draw_centers(self.left_centers, (0, 255, 255))
        self.draw_centers(self.right_centers, (0, 255, 255))
        self.draw_centers(self.yellow_centers, (0, 165, 255))

        cv2.line(self.vis, (self.roi_w // 2, 0), (self.roi_w // 2, self.roi_h), (0, 0, 255), 1)
        lane_center = self.roi_w / 2.0 + offset
        cv2.putText(
            self.vis, f'offset:{offset:+.1f} center:{lane_center:.1f} da_cov:{self.da_coverage:.2f}',
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )


class DLLaneDetector:
    """HoughLaneDetector/perc_floor.LaneDetector와 동일한
        detect(frame) -> (lane_valid, lane_offset, lane_lookahead, lane_center, debug_img)
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
        self._latest_result = (False, 0.0, 0.0, default_center, None)
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
                resized_bgr, da_prob, ll_prob = self.engine.infer_raw(frame)
                yellow_mask = cv2.inRange(
                    cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV), YELLOW_LOWER, YELLOW_UPPER
                )
                lane_valid, offset, lookahead, lane_center = self._slide.detect(
                    resized_bgr, da_prob, ll_prob, yellow_mask
                )
                debug_img = self._slide.vis
            except Exception as e:
                self._log(f'추론 실패, 이번 프레임 스킵: {e}')
                continue

            with self._lock:
                self.yellow_centers = self._slide.yellow_centers
                self._latest_result = (lane_valid, offset, lookahead, lane_center, debug_img)
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
