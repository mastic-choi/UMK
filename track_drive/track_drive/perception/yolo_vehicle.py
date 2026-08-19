#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# yolo_vehicle.py — YOLOv8n(ONNX Runtime) 기반 방해차량 카메라 이중확인.
#
# yolo_cone.py와 거의 동일한 구조다(같은 실시간 전략: 별도 데몬 스레드에서 자기
# 페이스로 추론, detect()는 논블로킹으로 최신 결과 반환, cv2.imshow()는 메인
# 스레드에서만). 차이는 두 가지뿐이다:
#   ① 모델이 파인튜닝된 전용 모델(cone_best_n.onnx)이 아니라 COCO 사전학습
#      yolov8n.pt를 그대로 ONNX(nms=True)로 내보낸 것 — 클래스가 80개라
#      YOLO_VEHICLE_CLASS_ID('car'=2)로 걸러야 한다(콘 모델은 클래스가 1개뿐이라
#      이 필터가 없었다).
#   ② 그래서 신뢰도가 콘 모델만큼 높지 않다(실측 0.15~0.78, 평균 0.3대) — 배경
#      소품(카트/의자 등)을 다른 클래스로 오탐하는 사례도 확인됨(config.py
#      YOLO_VEHICLE_* 주석 참고). track_drive.py가 이 결과를 라이다 근접판정과
#      AND로 묶고 프레임 디바운스(YOLO_VEHICLE_CONFIRM_FRAMES)까지 걸어서 쓴다 —
#      이 모듈 자체는 "이번 프레임 car가 보이는가"만 답한다.
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

from ..config import (
    YOLO_VEHICLE_INPUT_SIZE, YOLO_VEHICLE_CONF_THRESHOLD, YOLO_VEHICLE_CLASS_ID,
    YOLO_VEHICLE_MODEL_PATH, DEBUG_VIZ_YOLO_VEHICLE, FPS_LOG_PERIOD_SEC,
)


def _default_model_path():
    """yolov8n_car.onnx 기본 경로 — yolo_cone.py _default_model_path()와 동일한
    경로 규칙(realpath로 --symlink-install 심볼릭 링크를 해소한 뒤 세 단계 위로
    올라가 yolo_ros/에 닿음). 그쪽 함수 주석 참고."""
    if YOLO_VEHICLE_MODEL_PATH:
        return YOLO_VEHICLE_MODEL_PATH

    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # .../track_drive/track_drive
    ros_pkg_root = os.path.dirname(package_dir)                                # .../track_drive (ROS 패키지 루트)
    src_dir = os.path.dirname(ros_pkg_root)                                    # .../src (또는 저장소 루트)
    return os.path.join(src_dir, 'yolo_ros', 'yolov8n_car.onnx')


class YoloVehicleEngine:
    """ONNX Runtime으로 YOLOv8n(NMS 포함 export, COCO 80클래스) 모델을 로드하고
    'car' 클래스만 걸러 추론한다."""

    def __init__(self, model_path=None, providers=None, logger=None):
        if ort is None:
            raise ImportError(
                'onnxruntime이 설치돼 있지 않습니다. '
                f'원래 import 에러: {_ORT_IMPORT_ERROR}'
            )

        self.model_path = model_path or _default_model_path()
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f'YOLO 차량 검출 가중치 파일을 찾을 수 없습니다: {self.model_path}\n'
                'yolov8n.pt를 ONNX(nms=True)로 변환해 yolo_ros/yolov8n_car.onnx에 두세요.'
            )
        self._logger = logger

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # dl_lane.py/yolo_cone.py와 동일한 이유(Jetson 코어 수 제한, ROS2 콜백 스레드와의
        # 경쟁 방지)로 스레드 수를 제한한다. 세 모델(da/ll + cone + vehicle)이 동시에
        # 돌아갈 수 있으므로 이쪽도 가볍게 잡아둔다.
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        available = set(ort.get_available_providers())
        if providers is None:
            # [2026-08-19] cone_best_n.onnx(§0 공통주의/yolo_cone.py 참고)와 마찬가지로
            # nms=True export는 그래프 안에 NonMaxSuppression이 들어있어 TensorRT 빌드가
            # 실패하는 것으로 이미 확인된 실패모드가 있다(TRT-16198). 같은 export 방식이라
            # 이 모델도 같은 문제를 겪을 가능성이 높아, 처음부터 TensorRT를 건너뛰고
            # CUDA로 간다 — 실차 미검증(cone 모델만큼 실측 확인은 안 됨).
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
        self._log(f'YOLO 차량 검출 ONNX 세션 로드 완료 | 최우선 provider={self.active_provider} '
                   f'(요청순위={providers})')

        self._latency_ema = None

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(f'[yolo_vehicle] {msg}')
        else:
            print(f'[yolo_vehicle] {msg}')

    def preprocess(self, bgr_frame):
        # yolo_cone.py와 동일하게 letterbox 없이 단순 리사이즈(원본 종횡비 무시) —
        # "검출 여부"만 쓰므로 좌표 왜곡은 지금 용도에 영향 없다.
        resized = cv2.resize(bgr_frame, (YOLO_VEHICLE_INPUT_SIZE, YOLO_VEHICLE_INPUT_SIZE),
                              interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]   # HWC -> NCHW
        return blob

    def infer(self, bgr_frame):
        """입력 : 임의 크기(H,W) BGR 프레임
        출력 : (vehicle_detected, detections) — detections는 [(x1,y1,x2,y2,conf), ...]
               (640x640 입력 스케일 좌표, yolo_cone.py와 동일하게 원본 스케일로 안
               되돌림 — 지금은 "검출 여부"만 쓴다).
        모델이 nms=True로 export돼 output0에 이미 NMS 적용된 [x1,y1,x2,y2,conf,cls]가
        나온다(COCO 80클래스 전체) — 여기서 YOLO_VEHICLE_CLASS_ID('car')만 골라내고
        신뢰도 임계값을 적용한다."""
        t0 = time.perf_counter()
        blob = self.preprocess(bgr_frame)
        outputs = self.session.run([self._output_name], {self._input_name: blob})[0]
        dt = time.perf_counter() - t0
        self._latency_ema = dt if self._latency_ema is None else 0.8 * self._latency_ema + 0.2 * dt

        # outputs shape: (1, N, 6) — N은 이번 프레임 검출 수(패딩된 0행 포함 가능)
        dets = outputs[0]
        detections = []
        for x1, y1, x2, y2, conf, cls in dets:
            if int(cls) == YOLO_VEHICLE_CLASS_ID and conf >= YOLO_VEHICLE_CONF_THRESHOLD:
                detections.append((float(x1), float(y1), float(x2), float(y2), float(conf)))

        return (len(detections) > 0), detections

    @property
    def fps(self):
        if not self._latency_ema:
            return 0.0
        return 1.0 / self._latency_ema


class YoloVehicleDetector:
    """별도 데몬 스레드에서 자기 페이스로 추론하고, detect()는 논블로킹으로 최신
    결과를 반환한다(yolo_cone.YoloConeDetector/dl_lane.DLLaneDetector와 동일한
    실시간 전략)."""

    def __init__(self, model_path=None, providers=None, logger=None):
        self.engine = YoloVehicleEngine(model_path=model_path, providers=providers, logger=logger)
        self._logger = logger

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_result = (False, [])          # (vehicle_detected, detections)
        self._latest_debug = None                    # 시각화용 vis 프레임
        self._stopped = False
        self._last_fps_log_t = time.time()

        self._thread = threading.Thread(target=self._worker, name='yolo_vehicle_infer', daemon=True)
        self._thread.start()

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(f'[yolo_vehicle] {msg}')

    def _worker(self):
        while not self._stopped:
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.005)
                continue

            try:
                vehicle_detected, detections = self.engine.infer(frame)
                vis = None
                if DEBUG_VIZ_YOLO_VEHICLE:
                    # 그리기만 여기서(스레드 세이프하지 않은 imshow/waitKey는 절대 호출 안 함
                    # — dl_lane.py DLSlideWindow.visualize() 주석과 동일한 이유).
                    scale_x = frame.shape[1] / YOLO_VEHICLE_INPUT_SIZE
                    scale_y = frame.shape[0] / YOLO_VEHICLE_INPUT_SIZE
                    vis = frame.copy()
                    for x1, y1, x2, y2, conf in detections:
                        p1 = (int(x1 * scale_x), int(y1 * scale_y))
                        p2 = (int(x2 * scale_x), int(y2 * scale_y))
                        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
                        cv2.putText(vis, f'car {conf:.2f}', (p1[0], max(0, p1[1] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(vis, f'detected={vehicle_detected} n={len(detections)}',
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                self._log(f'추론 실패, 이번 프레임 스킵: {e}')
                continue

            with self._lock:
                self._latest_result = (vehicle_detected, detections)
                self._latest_debug = vis

            now = time.time()
            if now - self._last_fps_log_t >= FPS_LOG_PERIOD_SEC:
                self._log(f'YOLO 차량 추론 FPS≈{self.engine.fps:.1f} (provider={self.engine.active_provider})')
                self._last_fps_log_t = now

    def detect(self, frame):
        """논블로킹: 최신 프레임을 추론 큐에 올리고, 지금까지 계산된 최신 결과를 즉시 반환.
        출력 : vehicle_detected(bool) — 카메라에 'car'가 하나라도 보이는지."""
        if frame is not None:
            with self._lock:
                self._latest_frame = frame
        with self._lock:
            vehicle_detected, _detections = self._latest_result
        return vehicle_detected

    def show_debug_windows(self):
        """★ 반드시 메인 스레드에서만 호출할 것 ★ (yolo_cone.py/dl_lane.py와 동일한 이유)."""
        if not DEBUG_VIZ_YOLO_VEHICLE:
            return
        with self._lock:
            vis = self._latest_debug
        if vis is None:
            return
        cv2.imshow('yolo_vehicle_result', vis)
        cv2.waitKey(1)

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=2.0)
