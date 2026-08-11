# 젯슨 새 보드 세팅 체크리스트

컴퓨팅보드를 엔비디아 젯슨으로 교체하면서 처음부터 다시 설치할 때 필요한 항목 정리
(2026-08-11 작성). 물리적 차체(카메라 위치, 축거, VESC 게인 등)가 그대로라면
`track_drive/config.py`의 실측 튜닝값은 재사용 가능 — 여기서 다루는 건 소프트웨어/드라이버
설치와 장치 파일(udev) 재설정 쪽이다.

## 0. 젯슨 베이스

- [ ] JetPack 플래싱(SDK Manager) — 이 프로젝트는 **ROS2 Humble** 기준(개발 PC도
      `/opt/ros/humble`)이므로 Ubuntu 22.04 계열이 나오는 JetPack 버전으로 플래싱할 것.
- [ ] ROS2 Humble + colcon 설치
- [ ] **Docker + ROS1(noetic) 컨테이너 + `ros1_bridge`** — 인지/판단 노드가 다 떠도 이게
      없으면 모터가 안 움직인다. `/xycar_motor`가 ROS1↔ROS2 브릿지를 거쳐야 실제 구동됨
      (`track_drive/README.md` §공통 주의사항).

## 1. 이 저장소 4개 패키지 pull

`~/xycar_ws/src/`에 그대로 배치:

- `track_drive/`
- `yolo_ros/` — 실제로는 `cone_best_n.onnx` 파일 경로만 필요하다. YOLO ROS 노드 자체는
  `track_drive/launch/track_drive.launch.py`에서 이미 제거됨(인지는 카메라+라이다만 사용).
- `xycar_device/` (xycar_cam, xycar_imu, xycar_lidar, xycar_msgs, xycar_ultrasonic)

`xycar_application`/`xycar_simulator`는 자율주행 실행엔 불필요(연습용 데모 앱들) — 안 붙여도 됨.

## 2. 저장소 밖 ROS2 패키지 (별도 설치 필요)

- [ ] **usb_cam** — apt(`ros-humble-usb-cam`) 또는 소스 빌드. `xycar_cam` 패키지는 저장소에
      있지만 launch/params 래퍼일 뿐 실제 드라이버가 아니므로 이것과는 별개로 필요.
- [ ] **cv_bridge** — apt(`ros-humble-cv-bridge`). `track_drive.py`/`manual_drive_collector.py`가
      직접 사용(`camera_raw_recorder.py`만 cv_bridge 없이 numpy로 직접 파싱하도록 우회돼 있음).
- [ ] `build-essential`/`cmake` — `xycar_lidar`에 YDLidar-SDK가 vendored돼 있어 cmake로
      빌드된다.

## 3. Python 패키지

- [ ] `onnxruntime` — **일반 pip 버전 말고 Jetson용 wheel(CUDA/TensorRT execution provider
      포함)을 설치할 것.** `perception/dl_lane.py`, `perception/yolo_cone.py` 둘 다 onnxruntime
      기반. TensorRT provider는 최초 1회 실행 시 엔진 빌드에 수십초~수분 걸릴 수 있다는 게
      README에 명시돼 있음(`track_drive/track_drive/README.md` §2.24) — 처음 켰을 때 응답이
      없어도 죽은 게 아닐 수 있으니 당황하지 말 것.
- [ ] `opencv-python`, `numpy`, `scipy`(LQR 컨트롤러·라바콘 인지에서 사용)
- [ ] `Pillow` + `sudo apt install fonts-nanum` — 디버그 창 한글 텍스트 렌더링용
      (`track_drive/kr_text.py`)
- [ ] `python-serial` — `xycar_imu` 패키지 의존성

## 4. 하드웨어 연결 + udev 별칭 (젯슨 새로 잡으면 재설정 필요)

- [ ] 전방 카메라 → `/dev/videoCAM`
- [ ] YDLiDAR → `/dev/ttyLIDAR`
- [ ] SparkFun 9DoF Razor IMU M0 → `/dev/ttyIMU` (57600bps)
- [ ] VESC — ROS1쪽 `vesc_driver` + 변환 노드(`track_drive/launch/vesc_speed_bridge.py`)는
      이 ROS2 워크스페이스 **밖**(ROS1 noetic_ws)에 별도로 설치해야 한다
      (`track_drive/track_drive/README.md` §7 참고).
- [ ] IMU 펌웨어 — 보드 자체 펌웨어는 이미 준비돼 있어 새로 만들 필요 없음.
      `xycar_device/xycar_imu/firmware/Razor_AHRS/Razor_AHRS.ino`를 Arduino IDE(보드: SparkFun
      9DoF Razor IMU M0)로 플래싱만 하면 됨.

## 5. 모델 가중치 — 이미 저장소에 커밋돼 있음 (git pull만 하면 됨, 별도 다운로드 불필요)

- `track_drive/track_drive/models/best.onnx`
- `track_drive/track_drive/models/twinlitenetplus_small_bootstrap_v2.onnx` +
  `twinlitenetplus_small_bootstrap_v2.onnx.data` (반드시 같은 폴더에 있어야 로드됨 — onnx
  파일 내부에 데이터 파일명이 상대경로로 박혀 있음)
- `yolo_ros/cone_best_n.onnx`

## 6. ⚠️ 알려진 문제 — 이전 전에 확인할 것

`track_drive/setup.py`의 `console_scripts`에 등록돼 있지만 **소스 파일 자체가 없는 엔트리
포인트가 있다**:

- `joystick_teleop.py`
- `speed_test.py`
- `stationary_noise_calib.py`

자율주행 메인 파이프라인(`track_drive` 실행파일, `track_drive.launch.py`)엔 이 셋이 안
쓰이므로 당장 막히진 않는다. 다만 조이스틱 수동조작/속도테스트/정지노이즈 캘리브레이션을
젯슨에서 쓸 계획이면, `ros2 run track_drive joystick_teleop` 등을 실행하는 순간
`ModuleNotFoundError`로 즉시 죽는다(2026-08-11, `camera_raw_recorder.py`가 똑같은 상태였던 걸
발견하고 복원하면서 같이 확인됨 — `track_drive/track_drive/README.md`에 기록 없음, 별도 복원
필요).

## 7. 빌드 순서 요약

1. 4개 패키지 배치 → usb_cam/cv_bridge 등 외부 ROS2 패키지 설치 → onnxruntime(Jetson
   GPU)/opencv/scipy/Pillow/fonts-nanum 설치
2. `colcon build` (xycar_lidar cmake 빌드라 시간 걸림)
3. udev 규칙 재설정(videoCAM/ttyLIDAR/ttyIMU)
4. IMU 펌웨어 플래싱
5. Docker(ROS1) + `ros1_bridge` 컨테이너 + `vesc_speed_bridge.py` 배치
6. `ros2 launch track_drive track_drive.launch.py`로 최종 확인
