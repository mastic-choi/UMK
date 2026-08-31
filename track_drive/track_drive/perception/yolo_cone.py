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
    DEBUG_VIZ_YOLO_CONE, DEBUG_WIN_POS_YOLO_CONE, FPS_LOG_PERIOD_SEC,
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
      realpath를 써야 하는 이유: --symlink-install로 빌드하면 이 파일이 실제로 로드되는
      경로는 xycar_ws/build/track_drive/track_drive/perception/yolo_cone.py이고
      (build/track_drive/track_drive는 src/track_drive/track_drive로 가는 심볼릭 링크) —
      abspath는 이 심볼릭 링크를 풀어주지 않아 세 단계 위로 올라가면 xycar_ws/build로
      잘못 도착한다. realpath로 심볼릭 링크를 먼저 해소해야 xycar_ws/src에 정확히 닿는다.
    """
    if YOLO_CONE_MODEL_PATH:
        return YOLO_CONE_MODEL_PATH

    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # .../track_drive/track_drive
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
            # 이 콘 모델(cone_best_n.onnx, nms=True export)은 TensorrtExecutionProvider가
            # 항상 빌드 실패한다 — "TRT-16198: Layers missing empty tensor support"(export에
            # 포함된 NonMaxSuppression 레이어가 빈 텐서 케이스를 처리 못 함, onnxruntime/
            # TensorRT 버전 조합 이슈로 추정). 문제는 실패 자체가 아니라 실패를 확인하기까지
            # 약 7~8분을 태우고서야 onnxruntime이 조용히 CUDAExecutionProvider로 자동
            # 폴백한다는 것 — 노드를 켤 때마다(재출발/재테스트 포함) 이 지연이 매번
            # 반복된다. dl_lane.py의 TwinLiteNet은 TensorRT가 정상 동작하므로 그쪽
            # priority는 그대로 두고, 이 모델(콘 검출)만 TensorRT를 건너뛰고 바로 CUDA로 간다.
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
        self._latest_result = (False, [])          # (cone_detected, detections) — cone_detected는 박스가
                                                     #   하나라도 있으면 True인 원시값(면적 게이트 전).
                                                     #   면적 임계값은 이 검출기가 아니라 호출부
                                                     #   (track_drive.py, B1/B2 각자)가 건다.
        self._latest_max_area = 0.0                 # 이번 프레임 검출 박스 중 최대 면적(px²,
                                                     #   640 입력 스케일) — get_latest_max_area() 참고
        self._latest_side = None                      # 'L'/'R'/None — get_latest_side() 참고
        self._latest_debug = None                    # 시각화용 vis 프레임
        # 'yolo_cone_result' 창을 처음 띄울 때만 cv2.moveWindow로 위치를 잡기
        #   위한 1회성 가드(DEBUG_WIN_POS_YOLO_CONE 참고, yolo_signal_state.py의 동일 패턴).
        self._dbg_win_positioned = False
        self._stopped = False
        self._last_fps_log_t = time.time()
        self._logged_infer_error = False  # 추론 예외를 매 프레임 로그하면 로그창이 그걸로
                                           #   도배되므로 최초 1회만 찍고 이후는 조용히 스킵

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
                # 검출 박스 중 최대 면적(px², 640 입력 스케일)만 여기서 계산해 넘긴다 —
                # "면적이 임계값 이상이어야 검출 인정"이라는 최종 판단은 B1/B2가 서로 다른
                # 임계값(config.py YOLO_CONE_MIN_BOX_AREA_PX_B1/_B2)을 쓸 수 있어야 해서 이
                # 검출기 레벨(여기는 어느 단계가 부르는지 모름)이 아니라 그 값을 아는
                # track_drive.py perc_lavacon_trigger()/perc_obstacle_cut_trigger()가 각자 건다.
                max_area = max(((x2 - x1) * (y2 - y1) for x1, y1, x2, y2, _c in detections),
                                default=0.0)
                # 라이다-YOLO 좌우 교차검증(perc_obstacle_cut_trigger() B2)용 —
                # yolo_vehicle.py _worker()와 동일 계산(최고 신뢰도 박스 중심 x 기준 L/R).
                side = None
                if detections:
                    best = max(detections, key=lambda d: d[4])
                    cx = (best[0] + best[2]) / 2.0
                    side = 'L' if cx < YOLO_CONE_INPUT_SIZE / 2.0 else 'R'
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
                if not self._logged_infer_error:
                    self._log(f'추론 실패(이후 반복 로그는 생략, 이번 프레임부터 계속 스킵): {e}')
                    self._logged_infer_error = True
                continue

            with self._lock:
                self._latest_result = (cone_detected, detections)
                self._latest_max_area = max_area
                self._latest_side = side
                self._latest_debug = vis

            now = time.time()
            # FPS 로그는 "실제로 뭔가 검출됐을 때만" 찍는다 — 검출 안 될 때도 계속 찍히면
            # 로그창을 채운다.
            if cone_detected and now - self._last_fps_log_t >= FPS_LOG_PERIOD_SEC:
                self._log(f'YOLO 콘 검출됨 n={len(detections)} FPS≈{self.engine.fps:.1f} '
                          f'(provider={self.engine.active_provider})')
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

    def get_latest_max_area(self):
        """이번 프레임 검출 박스 중 최대 면적(px², 640 입력 스케일) — 검출이
        하나도 없었으면 0.0. track_drive.py의 perc_lavacon_trigger()(B1)/
        perc_obstacle_cut_trigger()(B2)가 각자의 YOLO_CONE_MIN_BOX_AREA_PX_B1/_B2 임계값과
        비교해 최종 검출 인정 여부를 정하고, _debug_viz_obstacle_cut()이 그 값을 그대로
        띄워 실차에서 임계값을 조정할 때 참고하게 한다."""
        with self._lock:
            return self._latest_max_area

    def get_latest_side(self):
        """최신 프레임에서 검출된(최고 신뢰도) 콘 박스의 좌우 위치 — 'L'/'R'
        (검출 없었으면 None). yolo_vehicle.py get_latest_side()와 동일 용도 — B2용
        perc_obstacle_cut_trigger() 좌우 교차검증."""
        with self._lock:
            return self._latest_side

    def get_latest_debug_frame(self):
        """최신 시각화 프레임(카메라 원본 + 검출 박스)을 스레드 세이프하게
        반환만 한다(cv2.imshow는 호출하지 않음) — yolo_vehicle.py get_latest_debug_frame()과
        동일 패턴. track_drive.py _debug_viz_obstacle_cut()이 B2(고정장애물=콘) 검증 중엔
        이 프레임을, B3(방해차량) 검증 중엔 yolo_vehicle의 프레임을 같은 라이다 BEV 패널
        옆에 합쳐서 그린다(_active_yolo_stage()로 어느 쪽이 활성인지 판단). show_debug_windows()
        가 띄우는 'yolo_cone_result' 창과는 별개로, DEBUG_VIZ_YOLO_CONE=True일 때만 vis가
        만들어지므로(위 _worker() 참고) 꺼져 있으면 항상 None."""
        with self._lock:
            return self._latest_debug

    def show_debug_windows(self):
        """★ 반드시 메인 스레드에서만 호출할 것 ★ (dl_lane.py와 동일한 이유)."""
        if not DEBUG_VIZ_YOLO_CONE:
            return
        with self._lock:
            vis = self._latest_debug
        if vis is None:
            return
        # 화면을 너무 많이 차지해서 표시 직전에만 아주 작게 축소한다(원본 해상도 vis
        # 자체·get_latest_debug_frame()이 돌려주는 프레임·검출 좌표 계산에는 영향 없음 —
        # 순전히 이 창의 표시 크기만 줄이는 것).
        small = cv2.resize(vis, (160, 120), interpolation=cv2.INTER_AREA)
        if not self._dbg_win_positioned:
            cv2.namedWindow('yolo_cone_result', cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow('yolo_cone_result', *DEBUG_WIN_POS_YOLO_CONE)
            self._dbg_win_positioned = True
        cv2.imshow('yolo_cone_result', small)
        cv2.waitKey(1)

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=2.0)
