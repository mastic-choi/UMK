#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# yolo_signal_state.py — YOLOv8n(ONNX Runtime) 기반 신호등 색상상태 검출.
#
# perception/yolo_signal.py("신호등 배경판 위치 탐지" 하이브리드, YOLO_SIGNAL_* config)와는
# 별개 모델/별개 목적이다 — 그쪽은 배경판 "위치"만 찾아 기존 HSV 자동크롭(_board_candidates())
# 대신 traffic_signal.py에 넘겨주고, 점등 색상 판정 자체는 여전히 classical CV
# (circle_brightness/shape_ok/pick_best_4)가 한다. 이 모듈은 배경판 위치와 무관하게
# "지금 어떤 색이 켜져 있는지" 자체를 단일 스테이지 YOLO로 직접 예측한다. 이름이 겹치지
# 않게 클래스/설정 전부 *State* 접두어를 쓴다(YOLO_SIGNAL_STATE_*, config.py 참고).
#
# signal_state_best_n.pt(yolo_ros/, YOLOv8n 파인튜닝, 클래스: {0: 'red',
# 1: 'green_straight', 2: 'green_left'} — datasets/signal_state/classes.txt와 순서 동일)를
# Colab에서 `model.export(format='onnx', imgsz=640, opset=12, simplify=True, nms=True)`로
# 변환한 yolo_ros/signal_state_best_n.onnx를 그대로 쓴다. yolo_cone.py와 동일하게
# nms=True export라 output0=(1,N,6)=[x1,y1,x2,y2,conf,cls]에 이미 NMS가 적용돼 있어,
# 여기서는 클래스별로 신뢰도 임계값을 넘는 것 중 최댓값만 고르면 된다.
#
# traffic_signal.py의 SignalDetector.detect_s2()(Hough Circle 방식)와 반환 시그니처를
# (red_on, straight_on, left_on)로 맞췄다 — 다만 perc_signal()의 실제 주행 판단(FSM 전환)에는
# 아직 연결하지 않았다. track_drive.py는 이 결과를 self.signal_*_on_yolo에 별도로 저장해
# _debug_viz_signal_status() 창에서 기존 Hough 결과와 나란히 보여주기만 한다 — 실차에서
# 정확도를 충분히 확인한 뒤에 perc_signal()의 판단 소스를 교체할지 결정할 것
# (config.py "da 블롭 선택" 항목과 같은 이유: 새 인식기를 실측 검증 없이 바로 주행 판단에
# 연결하지 않는다).
#
# dl_lane.py/yolo_cone.py와 동일하게 추론을 별도 데몬 스레드에서 자기 페이스로 돌리고,
# detect()는 논블로킹으로 최신 결과를 반환한다. cv2.imshow()는 메인 스레드에서만 호출해야
# 한다는 제약도 동일(dl_lane.py DLLaneDetector 주석 참고, GTK 프리즈 재현됨).
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
    YOLO_SIGNAL_STATE_INPUT_SIZE, YOLO_SIGNAL_STATE_CONF_THRESHOLD, YOLO_SIGNAL_STATE_MODEL_PATH,
    YOLO_SIGNAL_STATE_CLASS_NAMES, YOLO_SIGNAL_STATE_MIN_BOX_HEIGHT_PX,
    YOLO_SIGNAL_STATE_MAX_BOX_HEIGHT_PX, DEBUG_VIZ_YOLO_SIGNAL_STATE, FPS_LOG_PERIOD_SEC,
)


def _default_model_path():
    """signal_state_best_n.onnx 기본 경로. yolo_cone.py _default_model_path()와 동일한
    이유(형제 디렉터리 yolo_ros/, realpath로 --symlink-install 심볼릭 링크 해소)로 그대로 가져옴."""
    if YOLO_SIGNAL_STATE_MODEL_PATH:
        return YOLO_SIGNAL_STATE_MODEL_PATH

    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # .../track_drive/track_drive
    ros_pkg_root = os.path.dirname(package_dir)                                # .../track_drive (ROS 패키지 루트)
    src_dir = os.path.dirname(ros_pkg_root)                                    # .../src (또는 저장소 루트)
    return os.path.join(src_dir, 'yolo_ros', 'signal_state_best_n.onnx')


class YoloSignalStateEngine:
    """ONNX Runtime으로 YOLOv8n(NMS 포함 export) 신호등 색상상태 모델을 로드하고 추론한다."""

    def __init__(self, model_path=None, providers=None, logger=None):
        if ort is None:
            raise ImportError(
                'onnxruntime이 설치돼 있지 않습니다. '
                f'원래 import 에러: {_ORT_IMPORT_ERROR}'
            )

        self.model_path = model_path or _default_model_path()
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f'YOLO 신호등 색상상태 검출 가중치 파일을 찾을 수 없습니다: {self.model_path}\n'
                'signal_state_best_n.pt를 ONNX(nms=True)로 변환해 '
                'yolo_ros/signal_state_best_n.onnx에 두세요.'
            )
        self._logger = logger

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # yolo_cone.py와 동일한 이유(Jetson 코어 수 제한, 다른 인식 스레드와의 경쟁 방지)로
        # 스레드 수를 제한한다.
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        available = set(ort.get_available_providers())
        if providers is None:
            # yolo_cone.py의 cone_best_n.onnx(nms=True export)가 TensorRT 빌드 실패(TRT-16198)로
            # 매번 456초 지연 후 CUDA로 자동 폴백하는 문제를 겪었다(2026-08-13 실차 확인) — 같은
            # nms=True export 구조라 이 신호등 모델도 같은 문제를 겪을 가능성이 높아, 처음부터
            # TensorRT를 건너뛰고 CUDA로 간다. 실차에서 TensorRT가 실제로 되는 게 확인되면
            # 그때 우선순위를 조정할 것.
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
        self._log(f'YOLO 신호등 색상상태 검출 ONNX 세션 로드 완료 | 최우선 provider={self.active_provider} '
                   f'(요청순위={providers})')

        self._latency_ema = None

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(f'[yolo_signal_state] {msg}')
        else:
            print(f'[yolo_signal_state] {msg}')

    def preprocess(self, bgr_frame):
        resized = cv2.resize(bgr_frame, (YOLO_SIGNAL_STATE_INPUT_SIZE, YOLO_SIGNAL_STATE_INPUT_SIZE),
                              interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]   # HWC -> NCHW
        return blob

    def infer(self, bgr_frame):
        """입력 : 임의 크기(H,W) BGR 프레임
        출력 : (state_dict, detections)
          state_dict — {'red': (present, conf), 'green_straight': (...), 'green_left': (...)},
            클래스별 신뢰도 임계값 이상 검출 중 최댓값만 남긴 것(없으면 (False, 0.0)).
          detections — [(x1,y1,x2,y2,conf,class_name), ...] 시각화용
            (640 입력 스케일 좌표, letterbox 없이 단순 리사이즈라 원본과 종횡비가 다르면
            좌표가 뒤틀릴 수 있음 — yolo_cone.py와 동일한 제약).
        모델이 nms=True로 export돼 output0에 이미 NMS 적용된 [x1,y1,x2,y2,conf,cls]가
        나온다 — 여기서는 conf 임계값 필터링 + 클래스별 최댓값 선택만 한다.
        원거리/근접 오검출 억제를 위해 bbox 높이가 [YOLO_SIGNAL_STATE_MIN_BOX_HEIGHT_PX,
        YOLO_SIGNAL_STATE_MAX_BOX_HEIGHT_PX] 범위 밖인 검출은 conf와 무관하게 버린다
        (2026-08-23, config.py 해당 상수 주석 참고 — ROI 크롭 대신 탐지 후 필터링을 택한
        이유)."""
        t0 = time.perf_counter()
        blob = self.preprocess(bgr_frame)
        outputs = self.session.run([self._output_name], {self._input_name: blob})[0]
        dt = time.perf_counter() - t0
        self._latency_ema = dt if self._latency_ema is None else 0.8 * self._latency_ema + 0.2 * dt

        # outputs shape: (1, N, 6) — N은 이번 프레임 검출 수(패딩된 0행 포함 가능)
        dets = outputs[0]
        detections = []
        best_by_class = {name: (False, 0.0) for name in YOLO_SIGNAL_STATE_CLASS_NAMES}
        for x1, y1, x2, y2, conf, cls in dets:
            if conf < YOLO_SIGNAL_STATE_CONF_THRESHOLD:
                continue
            box_h = y2 - y1
            if not (YOLO_SIGNAL_STATE_MIN_BOX_HEIGHT_PX <= box_h <= YOLO_SIGNAL_STATE_MAX_BOX_HEIGHT_PX):
                continue
            cls_idx = int(round(cls))
            if not (0 <= cls_idx < len(YOLO_SIGNAL_STATE_CLASS_NAMES)):
                continue
            name = YOLO_SIGNAL_STATE_CLASS_NAMES[cls_idx]
            detections.append((float(x1), float(y1), float(x2), float(y2), float(conf), name))
            if conf > best_by_class[name][1]:
                best_by_class[name] = (True, float(conf))

        return best_by_class, detections

    @property
    def fps(self):
        if not self._latency_ema:
            return 0.0
        return 1.0 / self._latency_ema


class YoloSignalStateDetector:
    """별도 데몬 스레드에서 자기 페이스로 추론하고, detect()는 논블로킹으로 최신
    결과를 반환한다(yolo_cone.YoloConeDetector와 동일한 실시간 전략)."""

    def __init__(self, model_path=None, providers=None, logger=None):
        self.engine = YoloSignalStateEngine(model_path=model_path, providers=providers, logger=logger)
        self._logger = logger

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_state = {name: (False, 0.0) for name in YOLO_SIGNAL_STATE_CLASS_NAMES}
        self._latest_detections = []
        self._latest_debug = None                    # 시각화용 vis 프레임
        self._stopped = False
        self._last_fps_log_t = time.time()

        self._thread = threading.Thread(target=self._worker, name='yolo_signal_state_infer', daemon=True)
        self._thread.start()

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(f'[yolo_signal_state] {msg}')

    def _worker(self):
        while not self._stopped:
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.005)
                continue

            try:
                state, detections = self.engine.infer(frame)
                vis = None
                if DEBUG_VIZ_YOLO_SIGNAL_STATE:
                    # 그리기만 여기서(스레드 세이프하지 않은 imshow/waitKey는 절대 호출 안 함
                    # — dl_lane.py DLSlideWindow.visualize() 주석과 동일한 이유).
                    scale_x = frame.shape[1] / YOLO_SIGNAL_STATE_INPUT_SIZE
                    scale_y = frame.shape[0] / YOLO_SIGNAL_STATE_INPUT_SIZE
                    vis = frame.copy()
                    for x1, y1, x2, y2, conf, name in detections:
                        p1 = (int(x1 * scale_x), int(y1 * scale_y))
                        p2 = (int(x2 * scale_x), int(y2 * scale_y))
                        color = (0, 0, 220) if name == 'red' else (0, 200, 0)
                        cv2.rectangle(vis, p1, p2, color, 2)
                        cv2.putText(vis, f'{name} {conf:.2f}', (p1[0], max(0, p1[1] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    summary = ' '.join(f'{n}={"O" if p else "-"}' for n, (p, _c) in state.items())
                    cv2.putText(vis, summary, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                self._log(f'추론 실패, 이번 프레임 스킵: {e}')
                continue

            with self._lock:
                self._latest_state = state
                self._latest_detections = detections
                self._latest_debug = vis

            now = time.time()
            if now - self._last_fps_log_t >= FPS_LOG_PERIOD_SEC:
                self._log(f'YOLO 신호등 색상상태 추론 FPS≈{self.engine.fps:.1f} (provider={self.engine.active_provider})')
                self._last_fps_log_t = now

    def detect(self, frame):
        """논블로킹: 최신 프레임을 추론 큐에 올리고, 지금까지 계산된 최신 결과를 즉시 반환.
        출력 : (red_on, straight_on, left_on) — traffic_signal.SignalDetector.detect_s2()와
          동일한 우선순위(좌회전 > 직진 > 빨강)로 배타 처리한다(§ config.py
          red_lit/straight_lit/left_lit 주석과 동일한 이유 — 동시에 여러 클래스가 잡히는
          오검출을 대비)."""
        if frame is not None:
            with self._lock:
                self._latest_frame = frame
        with self._lock:
            state = self._latest_state

        left_on = state['green_left'][0]
        straight_on = state['green_straight'][0] and not left_on
        red_on = state['red'][0] and not (straight_on or left_on)
        return red_on, straight_on, left_on

    def show_debug_windows(self):
        """★ 반드시 메인 스레드에서만 호출할 것 ★ (yolo_cone.py와 동일한 이유)."""
        if not DEBUG_VIZ_YOLO_SIGNAL_STATE:
            return
        with self._lock:
            vis = self._latest_debug
        if vis is None:
            return
        cv2.imshow('YOLO_신호등', vis)
        cv2.waitKey(1)

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=2.0)
