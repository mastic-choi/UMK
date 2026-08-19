#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# yolo_signal.py — YOLOv8n(ONNX Runtime) 기반 신호등 배경판(rig) 위치 검출.
#
# [2026-08-19, 파일럿] yolo_cone.py와 거의 동일한 구조(모델 로드/스레드/논블로킹
# detect())를 그대로 따르되, 딱 하나 다르다 — yolo_cone은 "화면에 콘이 있는지"
# 이진값만 필요해서 좌표를 원본 스케일로 안 돌렸지만, 여기서는 traffic_signal.py의
# detect_s2()가 배경판 "위치"(ROI 크롭 좌표)를 그대로 받아써야 하므로 반드시
# 원본 프레임 픽셀 좌표로 되돌린다.
#
# 하이브리드 설계: 점등 색상 판정(빨강/초록/좌회전)은 여전히 traffic_signal.py의
# circle_brightness()/shape_ok()/pick_best_4()가 담당한다. 이 모듈은 "배경판이 어디
# 있는지"만 찾아 detect_s2()의 roi_boxes 후보 리스트에 HSV 자동크롭(_board_candidates())
# 대신(또는 함께) 얹어주는 역할만 한다 — 원 4개 판정 로직은 무수정.
#
# signal_board_best_n.pt(yolo_ros/, YOLOv8n 파인튜닝, 클래스: {0: 'signal_board'})를
# Colab에서 `model.export(format='onnx', imgsz=640, opset=12, simplify=True, nms=True)`로
# 변환한 yolo_ros/signal_board_best_n.onnx를 그대로 쓴다(datasets/yolo_signal_pilot/
# train_pilot.md 참고). nms=True export라 여기서는 신뢰도 임계값 필터링만 한다.
#
# [주의, 2026-08-19] 지금 존재하는 학습 데이터는 datasets/yolo_signal_pilot(15장,
# 스모크테스트 전용)뿐이라 config.py YOLO_SIGNAL_ENABLE 기본값은 False다 — 실데이터로
# 재학습하기 전엔 켜지 말 것(yolo_signal_detection_plan.md 참고).
#=============================================
import os
import threading
import time

import cv2
import numpy as np

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

from ..config import (
    YOLO_SIGNAL_INPUT_SIZE, YOLO_SIGNAL_CONF_THRESHOLD, YOLO_SIGNAL_MODEL_PATH,
    DEBUG_VIZ_YOLO_SIGNAL, FPS_LOG_PERIOD_SEC,
)


def _default_model_path():
    """signal_board_best_n.onnx 기본 경로 — yolo_cone.py _default_model_path()와
    동일한 배포 구조 근거(형제 디렉터리 yolo_ros/, realpath로 --symlink-install
    심볼릭 링크 해소) 그대로 적용."""
    if YOLO_SIGNAL_MODEL_PATH:
        return YOLO_SIGNAL_MODEL_PATH

    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # .../track_drive/track_drive
    ros_pkg_root = os.path.dirname(package_dir)                                # .../track_drive (ROS 패키지 루트)
    src_dir = os.path.dirname(ros_pkg_root)                                    # .../src (또는 저장소 루트)
    return os.path.join(src_dir, 'yolo_ros', 'signal_board_best_n.onnx')


class YoloSignalEngine:
    """ONNX Runtime으로 YOLOv8n(NMS 포함 export) 신호등 배경판 검출 모델을 로드하고 추론한다."""

    def __init__(self, model_path=None, providers=None, logger=None):
        if ort is None:
            raise ImportError(
                'onnxruntime이 설치돼 있지 않습니다. '
                f'원래 import 에러: {_ORT_IMPORT_ERROR}'
            )

        self.model_path = model_path or _default_model_path()
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f'YOLO 신호등 검출 가중치 파일을 찾을 수 없습니다: {self.model_path}\n'
                'datasets/yolo_signal_pilot/train_pilot.md대로 학습한 best.onnx를 '
                'signal_board_best_n.onnx로 리네임해 yolo_ros/에 두세요.'
            )
        self._logger = logger

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # yolo_cone.py YoloConeEngine과 동일한 이유(Jetson 코어 수 제한, 이미 dl_lane+
        # yolo_cone 두 모델이 동시에 도는 상태라 세 번째 모델도 가볍게 잡아둠).
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        available = set(ort.get_available_providers())
        if providers is None:
            # yolo_cone.py의 cone_best_n.onnx가 TensorRT 빌드 실패(§주석 참고, ~456초 지연 뒤
            # CUDA로 조용히 폴백)를 겪은 것과 같은 부류의 문제가 nms=True export 모델 전반에
            # 있을 수 있어 TensorRT는 처음부터 제외하고 CUDA로 바로 간다. 실차에서 TensorRT가
            # 이 모델도 정상 동작하는 게 확인되면 그때 우선순위 재검토.
            priority = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            providers = [p for p in priority if p in available] or ['CPUExecutionProvider']

        provider_options = []
        for p in providers:
            if p == 'TensorrtExecutionProvider':
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
        self.active_provider = self.session.get_providers()[0]
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        self._log(f'YOLO 신호등 검출 ONNX 세션 로드 완료 | 최우선 provider={self.active_provider} '
                   f'(요청순위={providers})')

        self._latency_ema = None

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(f'[yolo_signal] {msg}')
        else:
            print(f'[yolo_signal] {msg}')

    def preprocess(self, bgr_frame):
        resized = cv2.resize(bgr_frame, (YOLO_SIGNAL_INPUT_SIZE, YOLO_SIGNAL_INPUT_SIZE),
                              interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]   # HWC -> NCHW
        return blob

    def infer(self, bgr_frame):
        """입력 : 임의 크기(H,W) BGR 프레임
        출력 : detections — [(x1,y1,x2,y2,conf), ...], **원본 프레임(bgr_frame) 픽셀
               스케일로 리스케일된 좌표**. yolo_cone.py와 달리 detect_s2()가 이 좌표로
               직접 ROI를 크롭해야 하므로 리스케일이 필수다(letterbox 없이 단순 리사이즈라
               종횡비가 640x640 정사각형과 다르면 약간 뒤틀릴 수 있음 — 640x480 카메라
               기준으로는 세로 방향이 약간 늘어나는 정도라 배경판 위치 후보 용도로는
               margin(YOLO_SIGNAL_CROP_MARGIN)이 흡수함).
        모델이 nms=True로 export돼 output0에 이미 NMS 적용된 [x1,y1,x2,y2,conf,cls]가
        나온다 — 여기서는 conf 임계값 필터링 + 리스케일만 한다.
        """
        t0 = time.perf_counter()
        blob = self.preprocess(bgr_frame)
        outputs = self.session.run([self._output_name], {self._input_name: blob})[0]
        dt = time.perf_counter() - t0
        self._latency_ema = dt if self._latency_ema is None else 0.8 * self._latency_ema + 0.2 * dt

        h, w = bgr_frame.shape[:2]
        scale_x = w / YOLO_SIGNAL_INPUT_SIZE
        scale_y = h / YOLO_SIGNAL_INPUT_SIZE

        # outputs shape: (1, N, 6) — N은 이번 프레임 검출 수(패딩된 0행 포함 가능)
        dets = outputs[0]
        detections = []
        for x1, y1, x2, y2, conf, _cls in dets:
            if conf >= YOLO_SIGNAL_CONF_THRESHOLD:
                detections.append((
                    float(x1) * scale_x, float(y1) * scale_y,
                    float(x2) * scale_x, float(y2) * scale_y,
                    float(conf),
                ))

        return detections

    @property
    def fps(self):
        if not self._latency_ema:
            return 0.0
        return 1.0 / self._latency_ema


class YoloSignalDetector:
    """별도 데몬 스레드에서 자기 페이스로 추론하고, detect()는 논블로킹으로 최신
    결과를 반환한다(dl_lane.DLLaneDetector/yolo_cone.YoloConeDetector와 동일한 실시간 전략)."""

    def __init__(self, model_path=None, providers=None, logger=None):
        self.engine = YoloSignalEngine(model_path=model_path, providers=providers, logger=logger)
        self._logger = logger

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_result = []          # detections: [(x1,y1,x2,y2,conf), ...] (원본 프레임 스케일)
        self._latest_debug = None           # 시각화용 vis 프레임
        self._stopped = False
        self._last_fps_log_t = time.time()

        self._thread = threading.Thread(target=self._worker, name='yolo_signal_infer', daemon=True)
        self._thread.start()

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(f'[yolo_signal] {msg}')

    def _worker(self):
        while not self._stopped:
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.005)
                continue

            try:
                detections = self.engine.infer(frame)
                vis = None
                if DEBUG_VIZ_YOLO_SIGNAL:
                    # 그리기만 여기서(스레드 세이프하지 않은 imshow/waitKey는 절대 호출 안 함
                    # — dl_lane.py DLSlideWindow.visualize() 주석과 동일한 이유).
                    vis = frame.copy()
                    for x1, y1, x2, y2, conf in detections:
                        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
                        cv2.putText(vis, f'{conf:.2f}', (p1[0], max(0, p1[1] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(vis, f'n={len(detections)}',
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                self._log(f'추론 실패, 이번 프레임 스킵: {e}')
                continue

            with self._lock:
                self._latest_result = detections
                self._latest_debug = vis

            now = time.time()
            if now - self._last_fps_log_t >= FPS_LOG_PERIOD_SEC:
                self._log(f'YOLO 신호등 추론 FPS≈{self.engine.fps:.1f} (provider={self.engine.active_provider})')
                self._last_fps_log_t = now

    def detect(self, frame):
        """논블로킹: 최신 프레임을 추론 큐에 올리고, 지금까지 계산된 최신 결과를 즉시 반환.
        출력 : detections — [(x1,y1,x2,y2,conf), ...] (원본 프레임 픽셀 스케일, conf 내림차순 아님 —
               정렬은 호출부(traffic_signal.py _board_candidates_yolo())가 필요에 따라 한다)."""
        if frame is not None:
            with self._lock:
                self._latest_frame = frame
        with self._lock:
            return list(self._latest_result)

    def show_debug_windows(self):
        """★ 반드시 메인 스레드에서만 호출할 것 ★ (dl_lane.py/yolo_cone.py와 동일한 이유)."""
        if not DEBUG_VIZ_YOLO_SIGNAL:
            return
        with self._lock:
            vis = self._latest_debug
        if vis is None:
            return
        cv2.imshow('yolo_signal_result', vis)
        cv2.waitKey(1)

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=2.0)
