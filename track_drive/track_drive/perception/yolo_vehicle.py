#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# yolo_vehicle.py — YOLOv8n(ONNX Runtime) 기반 방해차량 카메라 이중확인.
#
# yolo_cone.py와 거의 동일한 실시간 전략(별도 데몬 스레드에서 자기 페이스로 추론,
# detect()는 논블로킹으로 최신 결과 반환, cv2.imshow()는 메인 스레드에서만)을 쓴다.
# 대회에서 실제 회피 대상인 그 차량 한 대(#46, TRAXXAS 검정/연두) 뒷모습 전용으로
# 파인튜닝한 target_vehicle_best.onnx(nc=1 'target_vehicle', class_id=0,
# `export_onnx_with_nms.py`로 export)를 쓴다 — output0가 NMS 적용된
# [1,N,6]=[x1,y1,x2,y2,conf,cls]라 콘 모델과 동일한 단순 파싱으로 충분하다.
# track_drive.py가 이 결과를 라이다 근접판정과 AND로 묶고 프레임 디바운스
# (YOLO_VEHICLE_CONFIRM_FRAMES)까지 걸어서 쓴다 — 이 모듈 자체는 "이번 프레임
# target_vehicle이 보이는가"만 답한다.
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
    YOLO_VEHICLE_MODEL_PATH, DEBUG_VIZ_YOLO_VEHICLE,
    FPS_LOG_PERIOD_SEC,
)


def _default_model_path():
    """target_vehicle_best.onnx 기본 경로 — yolo_cone.py _default_model_path()와 동일한
    경로 규칙(realpath로 --symlink-install 심볼릭 링크를 해소한 뒤 세 단계 위로
    올라가 yolo_ros/에 닿음). 그쪽 함수 주석 참고."""
    if YOLO_VEHICLE_MODEL_PATH:
        return YOLO_VEHICLE_MODEL_PATH

    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # .../track_drive/track_drive
    ros_pkg_root = os.path.dirname(package_dir)                                # .../track_drive (ROS 패키지 루트)
    src_dir = os.path.dirname(ros_pkg_root)                                    # .../src (또는 저장소 루트)
    return os.path.join(src_dir, 'yolo_ros', 'target_vehicle_best.onnx')


class YoloVehicleEngine:
    """ONNX Runtime으로 YOLOv8n(NMS 포함 export, nc=1 'target_vehicle') 전용 파인튜닝
    모델을 로드하고 추론한다."""

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
                '전용 파인튜닝 모델을 yolo_ros/target_vehicle_best.onnx에 두세요 '
                '(yolo-V8-KMU-xycar 저장소 notebooks/finetune_yolov8_local_rtx.ipynb 참고).'
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
        output0가 NMS 내장 [1,N,6]([x1,y1,x2,y2,conf,cls])라 yolo_cone.py와 동일한
        단순 파싱으로 충분하다(위 모듈 상단 주석 참고)."""
        t0 = time.perf_counter()
        blob = self.preprocess(bgr_frame)
        outputs = self.session.run([self._output_name], {self._input_name: blob})[0]
        dt = time.perf_counter() - t0
        self._latency_ema = dt if self._latency_ema is None else 0.8 * self._latency_ema + 0.2 * dt

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
        self._latest_result = (False, [])          # (vehicle_detected, detections) — vehicle_detected는
                                                     #   박스가 하나라도 있으면 True인 원시값(면적 게이트 전).
                                                     #   면적 임계값은 이 검출기가 아니라 호출부
                                                     #   (track_drive.py perc_obstacle_cut_trigger(), B3)가 건다.
        self._latest_max_area = 0.0                 # 이번 프레임 검출 박스 중 최대 면적(px²,
                                                     #   640 입력 스케일) — get_latest_max_area() 참고
        self._latest_side = None                      # 'L'/'R'/None — get_latest_side() 참고
        self._latest_debug = None                    # 시각화용 vis 프레임
        self._stopped = False
        self._last_fps_log_t = time.time()
        self._logged_infer_error = False  # 추론 예외를 매 프레임 로그하면 로그창이 그걸로
                                           #   도배되므로 최초 1회만 찍고 이후는 조용히 스킵

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
                # 검출 박스 중 최대 면적(px², 640 입력 스케일)만 여기서 계산해 넘긴다 —
                # "면적이 임계값 이상이어야 검출 인정"이라는 최종 판단은 그 임계값
                # (config.py YOLO_VEHICLE_MIN_BOX_AREA_PX_B3)을 아는 track_drive.py
                # perc_obstacle_cut_trigger()가 건다(B1/B2 콘 쪽과 동일 관례).
                max_area = max(((x2 - x1) * (y2 - y1) for x1, y1, x2, y2, _c in detections),
                                default=0.0)
                # 라이다-YOLO 좌우 교차검증(perc_obstacle_cut_trigger() 오검출
                # 방지 로직)용 — 가장 신뢰도 높은 박스의 중심 x가 프레임 가로 중앙(=
                # YOLO_VEHICLE_INPUT_SIZE/2, letterbox 없는 단순 리사이즈라 원본 중앙과
                # 동일 비율 지점) 왼쪽이면 'L', 오른쪽이면 'R'. 전방(비반전) 카메라라
                # 이미지 왼쪽=차량 좌측 그대로 대응 — 라이다 y>0=좌측 규약과 부호만
                # 다를 뿐 같은 실세계 방향.
                side = None
                if detections:
                    best = max(detections, key=lambda d: d[4])  # 최고 신뢰도 박스 하나만 사용
                    cx = (best[0] + best[2]) / 2.0
                    side = 'L' if cx < YOLO_VEHICLE_INPUT_SIZE / 2.0 else 'R'
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
                        cv2.putText(vis, f'target_vehicle {conf:.2f}', (p1[0], max(0, p1[1] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(vis, f'detected={vehicle_detected} n={len(detections)}',
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                if not self._logged_infer_error:
                    self._log(f'추론 실패(이후 반복 로그는 생략, 이번 프레임부터 계속 스킵): {e}')
                    self._logged_infer_error = True
                continue

            with self._lock:
                self._latest_result = (vehicle_detected, detections)
                self._latest_max_area = max_area
                self._latest_side = side
                self._latest_debug = vis

            now = time.time()
            # FPS 로그는 "실제로 뭔가 검출됐을 때만" 찍는다 — 주기 자체는 FPS_LOG_PERIOD_SEC로
            # 둬서 검출이 계속되는 동안에도 매 프레임 로그가 나가진 않게 한다.
            if vehicle_detected and now - self._last_fps_log_t >= FPS_LOG_PERIOD_SEC:
                self._log(f'YOLO 차량 검출됨 n={len(detections)} FPS≈{self.engine.fps:.1f} '
                          f'(provider={self.engine.active_provider})')
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

    def get_latest_max_area(self):
        """이번 프레임 검출 박스 중 최대 면적(px², 640 입력 스케일) — 검출이
        하나도 없었으면 0.0. track_drive.py perc_obstacle_cut_trigger()(B3)가
        YOLO_VEHICLE_MIN_BOX_AREA_PX_B3 임계값과 비교해 최종 검출 인정 여부를 정하고,
        _debug_viz_obstacle_cut()이 그 값을 그대로 띄워 실차에서 임계값을 조정할 때
        참고하게 한다."""
        with self._lock:
            return self._latest_max_area

    def get_latest_side(self):
        """최신 프레임에서 검출된(최고 신뢰도) target_vehicle 박스의 좌우
        위치 — 'L'/'R'(검출 없었으면 None). perc_obstacle_cut_trigger()가 라이다 판정
        (self._obstacle_cut_y 부호)과 이 값을 대조해 방향이 일치할 때만 방해차량으로
        확정하는 오검출 방지 게이트에 쓴다."""
        with self._lock:
            return self._latest_side

    def get_latest_debug_frame(self):
        """최신 시각화 프레임(카메라 원본 + 검출 박스)을 스레드 세이프하게 반환만 한다
        (cv2.imshow는 호출하지 않음) — track_drive.py의 _debug_viz_obstacle_cut()이 이
        프레임을 라이다 BEV 패널과 한 창에 합쳐서 그린다(이 검출기 전용
        'yolo_vehicle_result' 창은 따로 안 띄우고 obstacle_cut_debug 창 하나로 합쳐서,
        욜로판정과 그 판정에 쓰인 라이다 ROI를 한눈에 대조할 수 있게 한다).
        DEBUG_VIZ_YOLO_VEHICLE=False면 _worker()가 애초에 vis를 안 만들어서 항상 None."""
        with self._lock:
            return self._latest_debug

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=2.0)
