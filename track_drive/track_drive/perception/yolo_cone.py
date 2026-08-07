#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# yolo_cone.py — YOLOv8n(ONNX Runtime) 기반 라바콘 카메라 검출.
#
# cone_best_n.pt(yolo_ros/, YOLOv8n 파인튜닝, 클래스: {0: 'cone'})를 Colab에서
# `model.export(format='onnx', imgsz=640, opset=12, simplify=True, nms=True)`로
# 변환한 yolo_ros/cone_best_n.onnx를 그대로 쓴다. nms=True로 내보냈기 때문에
# ONNX 그래프 안에 NonMaxSuppression이 이미 포함돼 있어, 여기서는 출력(output0,
# (1,N,6)=[x1,y1,x2,y2,conf,cls])을 신뢰도 임계값으로 거르기만 하면 된다(직접
# NMS/좌표 디코딩 불필요).
#
# perc_lavacon_trigger()의 진입 조건("YOLO 콘 검출 AND 라이다 클러스터 검출")에서
# 화면에 콘이 있는지 없는지 이진값만 필요하므로, 좌표를 원본 프레임 스케일로
# 되돌리는 등의 후처리는 하지 않는다(신뢰도만 보면 됨).
#
# dl_lane.py와 동일하게 추론을 별도 데몬 스레드에서 자기 페이스로 돌리고,
# detect()는 논블로킹으로 최신 결과를 반환한다. cv2.imshow()는 메인 스레드에서만
# 호출해야 한다는 제약도 동일(dl_lane.py DLLaneDetector 주석 참고, GTK 프리즈 재현됨).
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
    YOLO_CONE_INPUT_SIZE, YOLO_CONE_CONF_THRESHOLD, YOLO_CONE_MODEL_PATH,
    DEBUG_VIZ_YOLO_CONE, FPS_LOG_PERIOD_SEC,
)


def _default_model_path():
    """cone_best_n.onnx 기본 경로.
    1순위: YOLO_CONE_MODEL_PATH(config.py)가 절대경로로 지정돼 있으면 그대로.
    2순위: 배포 구조상 yolo_ros/는 track_drive/(ROS2 패키지 루트, package.xml이 있는
      바로 그 폴더)와 형제 디렉터리로 xycar_ws/src/ 아래 나란히 들어간다(CLAUDE.md 참고).
      소스 파일 위치(track_drive/track_drive/perception/yolo_cone.py) 기준으로
      세 단계 위(perception→track_drive 패키지→track_drive ROS 루트) 올라가야
      xycar_ws/src/(또는 이 저장소의 루트)에 닿는다. colcon install share 디렉터리는
      yolo_ros가 정식 ROS2 패키지가 아니라서(package.xml 없음) 조회하지 않는다.
    """
    if YOLO_CONE_MODEL_PATH:
        return YOLO_CONE_MODEL_PATH

    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../track_drive/track_drive
    ros_pkg_root = os.path.dirname(package_dir)                                # .../track_drive (ROS 패키지 루트)
    src_dir = os.path.dirname(ros_pkg_root)                                    # .../src (또는 저장소 루트)
    return os.path.join(src_dir, 'yolo_ros', 'cone_best_n.onnx')


class YoloConeEngine:
    """ONNX Runtime으로 YOLOv8n(NMS 포함 export) 콘 검출 모델을 로드하고 추론한다."""

    def __init__(self, model_path=None, providers=None, logger=None):
        if ort is None:
            raise ImportError(
                'onnxruntime이 설치돼 있지 않습니다. '
                f'원래 import 에러: {_ORT_IMPORT_ERROR}'
            )

        self.model_path = model_path or _default_model_path()
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f'YOLO 콘 검출 가중치 파일을 찾을 수 없습니다: {self.model_path}\n'
                'cone_best_n.pt를 ONNX로 변환해 yolo_ros/cone_best_n.onnx에 두세요.'
            )
        self._logger = logger

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # dl_lane.py의 TwinLiteNet 세션과 동일한 이유(Jetson 코어 수 제한, ROS2 콜백
        # 스레드와의 경쟁 방지)로 스레드 수를 제한한다. 두 모델(da/ll + cone)이 동시에
        # 돌아갈 것이므로 이쪽도 가볍게 잡아둔다.
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        available = set(ort.get_available_providers())
        if providers is None:
            priority = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
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
        self._log(f'YOLO 콘 검출 ONNX 세션 로드 완료 | 최우선 provider={self.active_provider} '
                   f'(요청순위={providers})')

        self._latency_ema = None

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(f'[yolo_cone] {msg}')
        else:
            print(f'[yolo_cone] {msg}')

    def preprocess(self, bgr_frame):
        resized = cv2.resize(bgr_frame, (YOLO_CONE_INPUT_SIZE, YOLO_CONE_INPUT_SIZE),
                              interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]   # HWC -> NCHW
        return blob

    def infer(self, bgr_frame):
        """입력 : 임의 크기(H,W) BGR 프레임
        출력 : (cone_detected, detections) — detections는 [(x1,y1,x2,y2,conf), ...]
               (640x640 입력 스케일 좌표, 640 letterbox 없이 단순 리사이즈라 원본
               프레임과 종횡비가 다르면 좌표가 뒤틀릴 수 있음 — 지금은 "검출 여부"만
               쓰므로 원본 스케일로 되돌리지 않는다. 나중에 좌/우 위치까지 쓰게 되면
               반드시 리스케일 로직을 추가할 것).
        모델이 nms=True로 export돼 output0에 이미 NMS 적용된 [x1,y1,x2,y2,conf,cls]가
        나온다 — 여기서는 conf 임계값 필터링만 한다(클래스는 cone 하나뿐이라 필터 불필요).
        """
        t0 = time.perf_counter()
        blob = self.preprocess(bgr_frame)
        outputs = self.session.run([self._output_name], {self._input_name: blob})[0]
        dt = time.perf_counter() - t0
        self._latency_ema = dt if self._latency_ema is None else 0.8 * self._latency_ema + 0.2 * dt

        # outputs shape: (1, N, 6) — N은 이번 프레임 검출 수(패딩된 0행 포함 가능)
        dets = outputs[0]
        detections = []
        for x1, y1, x2, y2, conf, _cls in dets:
            if conf >= YOLO_CONE_CONF_THRESHOLD:
                detections.append((float(x1), float(y1), float(x2), float(y2), float(conf)))

        return (len(detections) > 0), detections

    @property
    def fps(self):
        if not self._latency_ema:
            return 0.0
        return 1.0 / self._latency_ema


class YoloConeDetector:
    """별도 데몬 스레드에서 자기 페이스로 추론하고, detect()는 논블로킹으로 최신
    결과를 반환한다(dl_lane.DLLaneDetector와 동일한 실시간 전략)."""

    def __init__(self, model_path=None, providers=None, logger=None):
        self.engine = YoloConeEngine(model_path=model_path, providers=providers, logger=logger)
        self._logger = logger

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_result = (False, [])          # (cone_detected, detections)
        self._latest_debug = None                    # 시각화용 vis 프레임
        self._stopped = False
        self._last_fps_log_t = time.time()

        self._thread = threading.Thread(target=self._worker, name='yolo_cone_infer', daemon=True)
        self._thread.start()

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(f'[yolo_cone] {msg}')

    def _worker(self):
        while not self._stopped:
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.005)
                continue

            try:
                cone_detected, detections = self.engine.infer(frame)
                vis = None
                if DEBUG_VIZ_YOLO_CONE:
                    # 그리기만 여기서(스레드 세이프하지 않은 imshow/waitKey는 절대 호출 안 함
                    # — dl_lane.py DLSlideWindow.visualize() 주석과 동일한 이유).
                    scale_x = frame.shape[1] / YOLO_CONE_INPUT_SIZE
                    scale_y = frame.shape[0] / YOLO_CONE_INPUT_SIZE
                    vis = frame.copy()
                    for x1, y1, x2, y2, conf in detections:
                        p1 = (int(x1 * scale_x), int(y1 * scale_y))
                        p2 = (int(x2 * scale_x), int(y2 * scale_y))
                        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
                        cv2.putText(vis, f'cone {conf:.2f}', (p1[0], max(0, p1[1] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(vis, f'detected={cone_detected} n={len(detections)}',
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                self._log(f'추론 실패, 이번 프레임 스킵: {e}')
                continue

            with self._lock:
                self._latest_result = (cone_detected, detections)
                self._latest_debug = vis

            now = time.time()
            if now - self._last_fps_log_t >= FPS_LOG_PERIOD_SEC:
                self._log(f'YOLO 콘 추론 FPS≈{self.engine.fps:.1f} (provider={self.engine.active_provider})')
                self._last_fps_log_t = now

    def detect(self, frame):
        """논블로킹: 최신 프레임을 추론 큐에 올리고, 지금까지 계산된 최신 결과를 즉시 반환.
        출력 : cone_detected(bool) — 카메라에 콘이 하나라도 보이는지."""
        if frame is not None:
            with self._lock:
                self._latest_frame = frame
        with self._lock:
            cone_detected, _detections = self._latest_result
        return cone_detected

    def show_debug_windows(self):
        """★ 반드시 메인 스레드에서만 호출할 것 ★ (dl_lane.py와 동일한 이유)."""
        if not DEBUG_VIZ_YOLO_CONE:
            return
        with self._lock:
            vis = self._latest_debug
        if vis is None:
            return
        cv2.imshow('yolo_cone_result', vis)
        cv2.waitKey(1)

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=2.0)
