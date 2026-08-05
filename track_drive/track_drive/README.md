# track_drive 테스트 가이드

`track_drive.py`는 하나의 노드 안에서 신호등/차선/라바콘/장애물/추월을 전부 처리하는 2중 FSM 구조입니다
(`MissionState` S0~S4 + `BehaviorState`/`Phase`). 인지 모듈은 `perception/`, 조향/회피 제어는 `controller/`,
Hybrid A* 경로계획(B2 대안, 보존용)은 `planner/`에 모여 있습니다.

## ⚙️ 실차 테스트 중 값을 바꾸려면 → `config.py` 하나만 보세요

**튜닝 파라미터, 디버그 on/off, 미션 state 관련 플래그가 전부 [`config.py`](config.py) 한 파일에
모여 있습니다.** `track_drive.py`를 포함한 프로젝트 거의 모든 모듈이 `from .config import *`
(또는 `from ..config import ...`)로 이 값을 가져다 씁니다 — 즉 `config.py`만 고치면 그 값을 쓰는
모든 파일에 동시에 반영됩니다. 개별 모듈 파일(`track_drive.py`, `perception/dl_lane.py` 등)을
헤집을 필요가 없습니다. `config.py` 안은 아래 순서로 구성돼 있습니다:

1. 차선인식 백엔드 선택 (`LANE_DETECTOR_BACKEND`)
2. 차량 속도/조향 기본값
3. 차선인식(`dl_lane.py`) 세부 튜닝 — BEV 적용여부 포함
4. 조향 컨트롤러 선택 (`STEERING_CONTROLLER`, Pure Pursuit/LQR 각 파라미터)
5. 디버깅 ON/OFF (모든 `DEBUG_VIZ_*`)
6. 미션 State / 실차 테스트 범위 제한 (`START_STATE`, `TEST_*`, 바퀴 카운트 등)
7. 기타 튜닝 파라미터(PID, 트리거, 회피, 신호등, 정지선, 단위환산 등)

**`config.py`에 없는 값들:** OpenCV 알고리즘 내부값(Canny/Hough 임계값, 블러 커널 크기 등)처럼
"행동을 튜닝한다"기보다 "그 함수 내부 구현"에 가까운 상수, 그리고 현재 기본 백엔드가 아닌
`hough`/`classic_cv` 차선인식·Hybrid A* B2 대안의 세부 알고리즘 상수는 원래 파일에 그대로
남아있습니다. 아래 각 절의 참조 표시(`파일:줄번호`)를 따라가면 됩니다.

실행 명령은 공통입니다:
```bash
ros2 launch track_drive track_drive.launch.py
```

## 공통 주의사항

- **테스트 종료는 항상 `Ctrl+C`로**, launch가 "process has finished cleanly"까지 뜨는 걸 확인하고 끄세요.
  `Ctrl+Z`(정지)나 터미널 강제종료는 `usb_cam_node_exe` 등이 좀비로 남아 `/dev/video0`를 붙잡고, 다음 실행에서
  카메라가 아예 안 잡히는 원인이 됩니다. 증상 발생 시:
  ```bash
  sudo fuser -k /dev/video0                      # 카메라만 문제일 때 가장 빠른 해결
  ps -eo pid,stat,cmd | awk '$2 ~ /^T/ && /usb_cam|ros2 launch|xycar_lidar|imu_node|track_drive/ {print $1}' | xargs -r kill -9   # 좀비 전체 정리
  ```
- `SPEED_NORMAL`([config.py:73](config.py#L73))을 `0.0`으로 두지 마세요. `_lane_drive()`에서 나눗셈
  분모로도 쓰여서 `ZeroDivisionError`로 노드가 죽습니다. 저속 테스트는 `5.0`~`10.0` 같은 작은 양수를 쓰세요.
- 실제 모터 구동은 ROS2 노드만으로 안 됩니다. 대회/실차 기준 **Docker(ROS1) 컨테이너 + `ros1_bridge`**가 같이
  떠 있어야 합니다 (`/xycar_motor`는 `Float32MultiArray([angle, speed])`로 브릿지됨 — 구형 `XycarMotor` 커스텀
  메시지는 `ros1_bridge`가 매핑을 못 함). 체크리스트: ①도커 컨테이너 기동 ②`ros1_bridge` 프로세스 기동 확인.
- **모든 디버그 창 스위치는 `config.py`의 "5. 디버깅 ON/OFF" 절 한곳에 모여 있습니다** — 예전엔
  `track_drive.py`에 `DEBUG_VIZ`/`DEBUG_VIZ_LANE`이라는 죽은 플래그(정의만 있고 어디서도 안 쓰임)가
  있어서 실제 스위치(각 `perception/*.py`)와 헷갈리기 쉬웠는데, `config.py`로 통합하면서 이 죽은
  플래그들은 삭제했습니다. 아래 표가 실제 스위치 위치입니다.

| 기능 | 디버그 창 ON/OFF 스위치 (`config.py`) |
|---|---|
| 신호등(S0/S2 공용) | `DEBUG_VIZ_SIGNAL` |
| 차선 — 기본 백엔드(`dl`) | `DEBUG_VIZ_DL_LANE` |
| 차선 — 대안 백엔드(`hough`) | `DEBUG_VIZ_HOUGH_LANE` |
| 차선 — 대안 백엔드(`classic_cv`) | `DEBUG_VIZ_LANE` |
| 정지선(백엔드와 무관하게 항상 동작) | `DEBUG_VIZ_STOPLINE` |
| 라이다 BEV(장애물) | `DEBUG_VIZ_LIDAR` |
| 라이다 BEV(라바콘 트리거) | `DEBUG_VIZ_LAVACON` |
| 조향 컨트롤러 상태 | `DEBUG_VIZ_STEER` |

> 이 프로젝트는 YOLO(yolo_ros)를 사용하지 않습니다 — 모든 인지는 카메라(차선/신호등)와 라이다(장애물/라바콘)만으로 수행합니다.

---

## 0. 차선인식 백엔드 선택

`perc_lane()`은 `self.lane_detector.detect(frame) -> (valid, offset, lookahead, lane_center, path, debug_img)`
인터페이스에만 의존하는 pluggable 구조라, 셋 중 하나를 자유롭게 골라 끼울 수 있습니다
([config.py:67](config.py#L67) `LANE_DETECTOR_BACKEND`).

| 값 | 구현 | 상태 |
|---|---|---|
| `'dl'` | `perception/dl_lane.py`의 `DLLaneDetector` (TwinLiteNet ONNX 세그멘테이션, 별도 스레드 추론) | **현재 기본값** |
| `'hough'` | `perception/hough_lane.py`의 `HoughLaneDetector` | 대안, 실차 라바콘 테스트까지 검증됨. `'dl'` 초기화 실패 시 자동 폴백 |
| `'classic_cv'` | `perception/lane_util.py`(`CameraProcessor`+`SlideWindow`) + `perception/perc_floor.py`(`LaneDetector`) 조립 | 보존용, 현재 라이브 미검증 |

`'dl'` 선택 시 `onnxruntime` 미설치나 `models/best.onnx` 부재 등으로 초기화가 실패하면 `_build_lane_detector()`
([track_drive.py:303-323](track_drive.py#L303))가 에러를 로깅하고 자동으로 `'hough'`로 폴백합니다(조용히
무시하지 않음) — 노드 시작 로그에 `DL 차선인식 백엔드 초기화 실패, hough로 폴백합니다` 가 찍혔는지 꼭 확인하세요.

---

## 0.5 조향 컨트롤러 선택 (Pure Pursuit / LQR)

차선 추종(`_lane_steer()`, [track_drive.py:1141](track_drive.py#L1141))도 백엔드처럼 pluggable합니다.
`self.pure_pursuit`과 `self.lqr` 둘 다 미리 생성해두고([track_drive.py:198-221](track_drive.py#L198),
튜닝값은 전부 `config.py`의 `PP_*`/`LQR_*`에서 가져옴), `STEERING_CONTROLLER`([config.py:176](config.py#L176))
값에 따라 어느 쪽을 쓸지 고릅니다. `_lane_drive()`(S1/S3 차선주행)와 S2 진입 전 감속 구간이 전부 이
하나를 거치므로, 플래그만 바꾸면 차선 추종 전체가 전환됩니다.

```python
STEERING_CONTROLLER = 'pure_pursuit'  # 'pure_pursuit' | 'lqr'
```

| 값 | 구현 | 상태 |
|---|---|---|
| `'pure_pursuit'` | `controller/pure_pursuit.py`의 `PurePursuitController` (기하학적, 속도/커브 적응형 lookahead) | **현재 기본값** |
| `'lqr'` | `controller/lqr.py`의 `LQRController` (횡오차/헤딩오차 2-state 운동학 LQR) | 신규, 실차 미검증 |

**주의:** `pure_pursuit.control()`은 속도적응형 lookahead 때문에 `speed=` 인자를 받지만, `lqr.control()`은
자체 `speed_gain` 튜닝값을 쓰고 그 인자가 없습니다 — 그래서 `_lane_steer()`가 컨트롤러별로 분기해서
호출합니다([track_drive.py:1158-1160](track_drive.py#L1158)). 두 컨트롤러를 직접 갖다 붙일 일이 있으면
이 시그니처 차이를 꼭 확인하세요.

**LQR 튜닝 파라미터** (`config.py:198` `LQR_*`, 전부 실차 미검증):
| 파라미터 | 의미 | 비고 |
|---|---|---|
| `LQR_SPEED_GAIN=120.0` | 속도 대응값(클수록 반응 커짐) | pure_pursuit에 대응 없는 신규 튜닝값 — 실차에서 최우선으로 건드릴 값 |
| `LQR_R_STEER=1.0` | 조향각 가중치(올릴수록 조향 억제, 지그재그 완화) | `Q_LATERAL`/`Q_HEADING` 건드리기 전에 먼저 조정 |
| `LQR_Q_LATERAL=1.0` / `LQR_Q_HEADING=1.0` | 횡오차/헤딩오차 비중 | lateral 비중↑ → 중앙복귀 서두름(오버슈트 위험), heading 비중↑ → 각도부터 맞추고 천천히 복귀 |
| `LQR_WHEELBASE_GAIN=50.0` | 조향 강도 | pure_pursuit의 `PP_WHEELBASE_PX`와 같은 역할의 튜닝 게인 |
| `LQR_ALPHA=0.5` | 프레임간 저역통과 필터 | 반응이 느리면 올리고, 잔떨림이 있으면 낮출 것 |
| `LQR_HEADING_PROBE_PX`/`LQR_MIN_PATH_PX=65.0` | 노이즈 방지 안전장치 | 조향이 자꾸 직전값 유지로 빠지면 낮출 것 |

**디버그 방법:**
- 창: `DEBUG_VIZ_STEER = True`(기본값, [config.py:218](config.py#L218)) → `steer_debug` 창에서
  지금 어느 컨트롤러가 쓰이는지, 이번 프레임이 "직전값 유지"(주황)인지 "현재값 반영"(초록)인지, 조향각과
  (`lqr`일 때) 횡오차 `e_y`/헤딩오차 `e_psi`까지 한글로 보여줍니다. cv2 기본폰트가 한글을 못 그려서
  `kr_text.py`의 PIL 기반 렌더러를 씁니다 — 한글 폰트가 없는 환경이면 영문 fallback으로 표시됩니다.

**알려진 한계:**
- LQR은 아직 실차 튜닝 전입니다. 처음 켤 때는 저속에서, 언제든 사람이 개입할 수 있는 상태로 테스트하세요.
- `localization/pose_estimator.py`의 `EncoderPoseEstimator`가 `self.pose_estimator`로 준비돼 있지만
  ([track_drive.py:249](track_drive.py#L249)), 실제 엔코더 ROS 토픽이 아직 확인 전이라 **어떤 콜백도
  갱신하지 않는 미배선 상태**입니다. LQR은 지금 이 pose 추정기를 쓰지 않고 `speed_gain`(튜닝 게인)으로
  대신합니다 — 엔코더 토픽이 확인되면 실제 m/s를 `set_speed_gain()`에 매 프레임 넣어주는 식으로 전환할 것.

### 0.5.1 코너 진입 시 회전반경 기반 감속 (`pure_pursuit` 전용)

`_lane_drive()`([track_drive.py:1121](track_drive.py#L1121))는 조향각 크기(`turn_now`)와 lookahead
편차(`turn_preview`)로 이미 코너에서 감속하는데, 여기에 **ROS2 Nav2의 Regulated Pure Pursuit**과 같은
방식의 감속을 추가했습니다([track_drive.py:1104](track_drive.py#L1104) `_corner_radius_speed_scale()`).

`pure_pursuit.control()`이 조향각을 계산하며 이미 구하는 `curvature`를 `self.pure_pursuit.last_curvature`로
저장해두고(`controller/pure_pursuit.py`), 회전반경(`1/curvature`)이 `CORNER_MIN_RADIUS_PX`
([config.py:87](config.py#L87), 기본 250px)보다 작아지면(=코너가 타이트해지면) 그 비율만큼
목표속도를 깎습니다 — Nav2의 `curvatureConstraint()`와 동일한 공식
(`속도 *= max(CORNER_MIN_SPEED_SCALE, 반경/CORNER_MIN_RADIUS_PX)`). 기존 `turn_now`/`turn_preview` 기반
감속을 대체하는 게 아니라, 둘 중 더 낮은 속도를 쓰는 **추가 안전판**입니다.

`PIXELS_PER_METER`가 미실측이라 반경도 미터가 아니라 픽셀 단위입니다. `CORNER_MIN_RADIUS_PX=250px`는
`PP_LOOKAHEAD_BASE_PX(90)~PP_LOOKAHEAD_MAX_PX(150)`(둘 다 config.py) 범위에서 alpha 20~30도짜리 코너의
반경을 역산해 다소 이르게(보수적으로) 개입하도록 잡은 추정치일 뿐 실차 미검증입니다.

`STEERING_CONTROLLER='lqr'`일 때는 적용되지 않습니다 — LQR은 curvature가 아니라 횡오차/헤딩오차
상태로 도는 별개 모델이라 이 반경 개념 자체가 안 맞습니다.

**왜 추가했나:** 짧은 lookahead에서 조향이 얼어붙던 버그를 고친 뒤에도, 코너처럼 회전반경이 급격히
작아지는 구간은 여전히 픽셀 노이즈에 민감합니다. 회전반경이 작아질수록 미리 속도를 낮춰두면, 같은
각도 오차라도 프레임당 실제로 밀리는 거리가 줄고 인지/제어 루프가 반응할 시간을 더 벌 수 있어
진동 억제에도 도움이 됩니다.

### 0.5.2 코너 진입 시 lookahead 축소 (curvature 기반, `pure_pursuit` 전용)

Lee, Lee & Moon, *"Frequency Shaping-Based Control Framework for Reducing Motion Sickness in
Autonomous Vehicles"* (Sensors 2025, 25, 819)의 "가변 LAD(Look-Ahead Distance)" 아이디어를
반영했습니다. 그 논문은 GPS+사전 지도로 "지금 위치에서 LAD만큼 앞의 실제 커브 반경"을 미리 조회해
LAD를 정하는데, 이 프로젝트는 그런 전역 지도/위치추정이 없어 그대로는 못 씁니다. 대신 **직전 프레임에
이미 계산해둔 `self.last_curvature`**로 반응형 버전을 구현했습니다([controller/pure_pursuit.py:34](controller/pure_pursuit.py#L34)
`lookahead_curvature_gain`/`lookahead_min_px`(생성자 파라미터명, 실제 값은 config.py의
`PP_LOOKAHEAD_CURVATURE_GAIN`/`PP_LOOKAHEAD_MIN_PX`), [153](controller/pure_pursuit.py#L153) `control()`) —
직전 프레임이 타이트한 코너였으면 이번 프레임 lookahead를 속도 기반 값보다 낮춰서(최소
`PP_LOOKAHEAD_MIN_PX=40px`까지) 더 촘촘하게 추종하고, 직진이면(curvature≈0) 원래 속도 기반 값을 그대로
씁니다. 한 프레임 지연된 반응이라 논문만큼 정교하진 않습니다.

**주의 — `PP_MIN_LOOKAHEAD_PX`와 별개입니다:** lookahead를 줄이는 하한(`PP_LOOKAHEAD_MIN_PX=40`)과, curvature
분모를 바닥까는 하한(`PP_MIN_LOOKAHEAD_PX=90`, 짧은 lookahead 얼어붙음 버그 수정에서 쓰인 값)은 이름이
비슷하지만 완전히 다른 역할입니다(둘 다 config.py에 있음). 후자를 코너에서도 고정 90으로 두면 curvature
계산의 ld가 거의 항상 90으로 다시 눌려서 lookahead 축소 효과가 무효화되므로,
`ld = max(hypot(dx,dy), min(min_lookahead_px, lookahead_px))`로 — lookahead가 의도적으로 줄어든
프레임에는 그 줄어든 값 자체를 바닥으로 쓰도록 함께 수정했습니다([controller/pure_pursuit.py:195](controller/pure_pursuit.py#L195)).

**알려진 한계:**
- `PP_LOOKAHEAD_CURVATURE_GAIN=100.0`, `PP_LOOKAHEAD_MIN_PX=40px`(둘 다 config.py) 모두 추정치입니다
  (`curvature=0.01`, 반경≈100px짜리 중간 코너에서 배율 0.5가 되도록 잡음). `steer_debug` 창이나 CLI
  로그로 관찰하며 튜닝이 필요합니다.
- `STEERING_CONTROLLER='lqr'`일 때는 적용되지 않습니다 — `lqr.py`는 curvature 개념 자체가 없습니다.

---

## 1. 신호등 (S0 출발 / S2 교차로) — 통합 4구 신호등

**(2026-08 규정 변경)** S0(출발)도 더 이상 3구 신호가 아니라 S2(교차로)와 동일한 4구 신호등을 재사용합니다.
`SignalDetector.detect_s2()`([perception/traffic_signal.py:102](perception/traffic_signal.py#L102)) 하나로
양쪽 상태를 다 처리합니다:
- S0: 초록(직진 위치)만 점등 → 출발
- S2: 초록만 점등 → 직진 / 초록+빨강 동시 점등 → 좌회전

두 상태 모두 `SIG_CONFIRM_FRAMES`([config.py:318](config.py#L318), 기본 3프레임) 연속 확정돼야
`signal_straight_confirmed`/`signal_left_confirmed`로 승격되는 디바운스가 걸려 있습니다(단발성 오검출 방지).

**수정할 곳:** `config.py:230` `START_STATE`
```python
START_STATE = MissionState.S0_WAIT_GREEN   # 신호(출발) 테스트
# 또는
START_STATE = MissionState.S2_INTERSECTION # 신호(교차로) 테스트 — 시작하자마자 정지 상태로 대기
```
S2로 시작할 땐 `TEST_DISABLE_INTERSECTION` 값과 무관하게 무조건 S2에서 시작합니다(이 플래그는
S1→S2 전환 경로만 막는 것이라 `START_STATE` 자체를 바꾸는 것과는 별개입니다). S2는 신호(직진/좌회전)를
인식할 때까지 `ang=0, spd=0`으로 계속 정지하는 게 정상 동작이니, 안 움직인다고 바로 버그로 보지 말고
로그의 `[SIG]` 값부터 확인하세요.

**디버그 방법:**
- 창: `DEBUG_VIZ_SIGNAL = True`([config.py:224](config.py#L224)) → `signal4_roi` 창 하나(S0/S2 공용,
  더 이상 `signal_roi`/`signal4_roi`로 나뉘어 있지 않음).
- CLI 로그: `DEBUG_LOG=True`면 S0든 S2든 0.5초마다 `[SIG]` 줄을 찍습니다(`roi=`, `circles=`, `reason=`,
  `bright=`) — 원 검출이 어느 단계(개수 부족/배치 불량/밝기 대비 부족)에서 막혔는지 터미널만으로 바로
  보입니다. 같이 찍히는 `[SENS] sig=` 줄에는 `R/L/S` 원시 판정값과 `confirmS(n/3)`/`confirmL(n/3)` 디바운스
  카운터도 함께 나옵니다.

**알려진 한계(실차 미검증):**
- `find_circles()`(Hough Circle)가 원을 4개 미만으로 찾으면 그 프레임은 무조건 인식 실패 — 폴백 없음
  (4개 **초과**로 잡히는 경우는 `pick_best_4()`가 배치가 맞는 4개 조합을 골라 완화합니다,
  [perception/traffic_signal.py:71](perception/traffic_signal.py#L71)).
- ROI(`SIG4_ROI_*`)와 반지름 범위(`SIG4_MIN/MAX_RADIUS=15~25px`, 전부 config.py)가 S0/S2 공용 고정값이라,
  카메라 각도·정지 위치가 튜닝 당시와 다르면 신호등이 ROI 밖이거나 반지름 범위 밖이라 아예 못 잡을 수 있음.
- 색상(Hue)을 직접 보지 않고 **위치(좌→우=빨강/노랑/좌회전/직진) + 밝기 대비**로만 판정 — 밝은 반사광이
  ROI에 섞이면 오탐 가능.

---

## 2. 라인트래킹 (차선주행, S1)

**수정할 곳:** `config.py:230` `START_STATE`, `config.py:246` `TEST_FORCE_BEHAVIOR`
```python
START_STATE = MissionState.S1_LANE_FOLLOW
TEST_FORCE_BEHAVIOR = False   # 라바콘 등 Behavior 없이 순수 차선주행 PID만 보고 싶을 때
```
`TEST_DISABLE_INTERSECTION = True`(기본값)면 정지선을 밟아도 S2로 안 새고 차선주행을 계속합니다.

**디버그 방법 (기본 백엔드 `dl` 기준 — 백엔드별 차이는 "0. 차선인식 백엔드 선택" 참고):**
- 창: `DEBUG_VIZ_DL_LANE = True`([config.py:220](config.py#L220)) → da(주행가능영역, 초록)/ll(차선, 빨강)
  반투명 오버레이 + da 중심선 관측점 + 피팅된 경로(웨이포인트) + `offset` 표시.
- CLI 로그: `[LANE] lane=편차px(검출여부) side=R/L차선 obs=... lava=...`.

**알려진 한계:**
- (`dl` 백엔드) da(주행가능영역) 마스크의 중심선을 경로 계산의 주 신호로 쓰고, ll(차선 세그멘테이션)은
  "거의 안 보이면 이번 프레임 무효 처리"하는 sanity check로만 사용합니다
  ([perception/dl_lane.py:401-498](perception/dl_lane.py#L401)의 `detect()` 참고). 노란 중앙선은
  `lane_side`(주행 차선 판정)에만 쓰이고 경로 계산 자체에는 관여하지 않습니다.
- `_s1_lane_follow()`가 `self.lane_valid`를 확인하지 않고 `_lane_drive()`를 호출함(`_s3_shortcut()`은 확인함,
  [track_drive.py:1001](track_drive.py#L1001)). 카메라가 순간적으로 차선을 놓쳐도 마지막 유효 offset으로
  계속 조향하니, 실차 테스트 시 차선 이탈 구간에서 주의 깊게 볼 것. (아직 미수정)
- (`classic_cv` 대안 백엔드) `perception/lane_util.py`의 CLAHE+adaptiveThreshold 기반 흰색 검출은
  "보존용"으로 유지 중이며 현재 라이브 미검증입니다.

---

## 3. 라바콘 (B1_LAVACON)

**수정할 곳:** `config.py:230` `START_STATE`, `config.py:246` `TEST_FORCE_BEHAVIOR`
```python
START_STATE = MissionState.S1_LANE_FOLLOW
TEST_FORCE_BEHAVIOR = True    # S2를 거치지 않고 시작부터 Behavior(라바콘부터) 강제 활성화
```
`self.phase`는 기본이 `Phase.LAVACON`([track_drive.py:144](track_drive.py#L144))이라 따로 안 건드려도 됩니다.
`TEST_DISABLE_B2_B3 = True`(기본값)면 라바콘 구간이 끝나도 B2/B3로 안 넘어가고 그냥 일반 차선주행으로
돌아오니, 라바콘만 격리 테스트하기 좋습니다.

라바콘 진입은 **라이다 좌우 클러스터 동시검출**이 `LAVACON_TRIGGER_FRAMES(5프레임)` 연속 유지돼야
확정됩니다(`perc_lavacon_trigger()`, 라이다 단독 판단 — 카메라/YOLO 이중확인 없음).

**디버그 방법:**
- CLI 로그: `trigL=본선카운트/기준(L{좌클러스터}R{우클러스터})` — 좌/우 중 어느 쪽을 못 잡는지 바로 구분됨.
  추가로 `[LAVA-ROI] L pts=... run=... R pts=... run=...` 줄에서 ROI 안에 잡힌 점 개수(pts)와 그중 최대
  연속 묶음 길이(run, 2 이상이어야 클러스터로 인정)까지 확인 가능.
- 창: `DEBUG_VIZ_LAVACON = True`([config.py:216](config.py#L216)) → `lavacon_bev` 창(트리거 ROI와 좌/우 클러스터를
  시각으로 확인).

**알려진 한계:**
- `LAVACON_DONE_FRAMES=80`(우측 콘 연속 미검출 시 구간 종료 판정)이 실차 미검증 값.

---

## 4. 사물회피 (B2_OBSTACLE, 고정장애물)

> **코드 주석 주의:** `track_drive.py`에는 아직 "★재설계 예정(placeholder)" 식 주석이 여러 곳 남아있지만
> (예: [track_drive.py:1356](track_drive.py#L1356)), 이건 오래된 주석이고 **실제 구현은 이미 교체됐습니다.**
> 지금은 `controller/obstacle_avoidance.py`의 `TargetPassing`이 붙어 있어, 감속 후 대기가 아니라
> **SHIFT → ALONGSIDE → RETURN 3단계로 실제 옆차선 통과 기동**을 수행합니다. 코드 주석이 구현을 못
> 따라간 상태이니, 동작을 파악할 때는 이 문서와 `controller/obstacle_avoidance.py`를 기준으로 보세요.

**수정할 곳:** `config.py:230` `START_STATE`, `config.py:242` `TEST_DISABLE_B2_B3`, `config.py:246` `TEST_FORCE_BEHAVIOR`, `track_drive.py:144` `self.phase`
```python
START_STATE = MissionState.S1_LANE_FOLLOW
TEST_DISABLE_B2_B3 = False     # B2 트리거 검사를 켜야 함
TEST_FORCE_BEHAVIOR = True
self.phase = Phase.FIXED_OBSTACLE   # __init__ 안의 self.phase 초기값을 임시로 변경 (격리 테스트용)
```
정상 흐름은 라바콘(B1) 완료 후 자동으로 `Phase.FIXED_OBSTACLE`로 넘어가는 것이라, 이 기능만 격리
테스트하려면 `__init__`의 `self.phase = Phase.LAVACON`을 위처럼 임시로 바꾸면 됩니다.

**동작 방식** (`TargetPassing`, [controller/obstacle_avoidance.py:49](controller/obstacle_avoidance.py#L49)):
1. **IDLE** — 전방 장애물이 감지되면 `choose_side()`로 통과 방향을 정합니다: ①타겟이 없는 차선 쪽(규정
   1순위) ②정면이라 못 가리면 비어있는 쪽(`left_clear`/`right_clear`) ③둘 다 비었으면 노란선 건너편.
   양쪽 다 막히면 `status='blocked'`(흰 실선 밖으로 안 나가고 서행하며 재시도).
2. **SHIFT** — 목표 횡오프셋(`PASS_OFFSET=100px`)까지 서서히 이동(`LATERAL_ALPHA_OUT`).
3. **ALONGSIDE** — 장애물이 안 보이는 상태가 `CLEAR_FRAMES_TO_RETURN`(6프레임) 유지되면 RETURN으로.
4. **RETURN** — 원 차선으로 복귀(`LATERAL_ALPHA_BACK`, SHIFT보다 빠르게) — 목표 오프셋이 5px 미만이면 완료.

`USE_HYBRID_ASTAR_FOR_B2 = True`([config.py:256](config.py#L256))로 바꾸면 위 방식 대신
Hybrid A* + OccupancyGrid + Stanley(`planner/`, `_handle_fixed_obstacle_astar()`) 경로계획 대안을 씁니다
— 비교/보존용이며 기본은 `False`.

**디버그 방법:**
- CLI 로그: `obs=검출여부(거리m,폭m,fixed/vehicle)`. `status='blocked'`가 되면 `[B2] 양쪽 통과 불가 —
  서행 후 재시도` 경고 로그가 뜹니다.

**알려진 한계:**
- `PASS_OFFSET=100px`(config.py)가 실측 차선 폭 대신 쓰는 자리표시값입니다(차선 폭 실측 후 교체 예정,
  [controller/obstacle_avoidance.py:31](controller/obstacle_avoidance.py#L31) 주석 참고).
- `LATERAL_ALPHA_OUT/BACK`, `MIN_GAP_M`, `CENTER_DEADZONE_M` 등 수렴 속도·안전거리 파라미터 다수가
  실차 미검증 튜닝값입니다.
- 좌우 선택은 카메라/YOLO 이중확인 없이 라이다 `obstacle_y` + `lane_side`만으로 판단 — 콘·차량 구분이 없어
  고정장애물(콘/박스류)도 동일한 로직으로 회피 방향이 잡힙니다.

---

## 5. 차량회피/추월 (B3_VEHICLE)

**수정할 곳:** `config.py:230` `START_STATE`, `config.py:242` `TEST_DISABLE_B2_B3`, `config.py:246` `TEST_FORCE_BEHAVIOR`, `track_drive.py:144` `self.phase`
```python
START_STATE = MissionState.S1_LANE_FOLLOW
TEST_DISABLE_B2_B3 = False
TEST_FORCE_BEHAVIOR = True
self.phase = Phase.VEHICLE     # 격리 테스트용 임시 변경
```
B2와 마찬가지로 별도 노드 전환 없이 `self.phase = Phase.VEHICLE`만 바꾸면 격리 테스트할 수 있습니다.
진입 조건은 **라이다 단독** — 전방 장애물 + 거리 < `OVERTAKE_TRIGGER=6.5m`가 `VEHICLE_TRIGGER_FRAMES(5프레임)`
연속 유지되면 확정됩니다(`perc_vehicle_trigger()`).

B2와 동일한 `TargetPassing`을 `moving=True`로 재사용합니다
([track_drive.py:177](track_drive.py#L177), `self.vehicle_controller`). 차이는 방해차량이 차선을 오가므로
SHIFT 단계에서 타겟이 내가 지나가려는 쪽으로 넘어오는 상태가 `SWITCH_FRAMES`(8프레임) 연속되면, 반대쪽이
비어 있는 경우에 한해 통과 방향을 바꾸는 재평가 로직(`_target_cuts_in()`)이 추가로 걸린다는 점입니다.
반대쪽도 막혀 있으면 RETURN으로 물러나 원 차선에서 재시도합니다.

**디버그 방법:**
- CLI 로그: `trigV=본선카운트/기준`.

**알려진 한계:**
- B2와 동일하게 카메라/YOLO 이중확인이 없어 콘·차량 구분 없이 라이다 근접만으로 트리거되므로, 콘이
  남아있는 상태에서도 거리 조건만 맞으면 B3로 오인 진입할 수 있음(Phase 순서가 지켜지는 정상 흐름에서는
  라바콘 구간을 먼저 통과한 뒤라 위험이 적지만, 격리 테스트 시에는 주의).
- `SWITCH_FRAMES`로 조절하는 방향 재전환 로직도 실차 미검증.

---

## 6. 실측값 기록 (캘리브레이션)

실차/트랙에서 직접 측정해 코드 상수로 반영한 값들의 근거를 한곳에 모아둡니다. **실측값**과
(값은 채워져 있지만 측정한 게 아니라 우리가 고른) **설계값**을 구분해서 표기합니다 — 헷갈리면
다음 사람이 "이미 실측됐다"고 착각하고 재검증을 건너뛸 수 있어서입니다.

### 6.1 도로/차량 치수 (2026-08-04 실측)

| 상수 | 값 | 근거 |
|---|---|---|
| `LANE_WIDTH_M` (config.py) | 0.4m | 흰선~흰선(도로 전체폭) 80cm 실측, 노란 중앙선이 정중앙임을 확인 → 차선 1개 폭 = 80/2 = 40cm |
| `VEHICLE_WIDTH_M` (config.py) | 0.31m | xycar 본체 실측: 세로64cm × 가로31cm × 높이20cm |
| 고정장애물(고장난 차량) 실측 크기 | 가로20cm×세로41cm×높이16cm | 방해차량과의 라이다 폭 분류 기준(`OBSTACLE_VEHICLE_WIDTH_M=0.24`, config.py) 산출 근거 |
| 방해차량 실측 크기 | 가로28cm×세로54cm×높이19cm | 위와 동일. `OBSTACLE_VEHICLE_WIDTH_M = (0.20+0.28)/2 = 0.24` |

### 6.2 라이다 장착 각도 보정 (2026-07-22 재실측)

| 상수 | 값 | 근거 |
|---|---|---|
| `LIDAR_ANGLE_OFFSET_DEG` (config.py) | 80.0 | 차량 정면에 사람을 세우고 라이다 BEV 디버그 창(각도/인덱스 컴퍼스)으로 확인 — 자기가림 마스크를 끄고 보니 정면 클러스터가 인덱스 90이 아니라 80에서 찍힘 |

### 6.3 DL 백엔드 BEV 캘리브레이션 (2026-08-05 실측)

`perception/dl_lane.py`의 `DL_USE_BEV` 실험적 경로에 쓰이는 값 — 현재 `True`로 실차 검증 중입니다
(검증 전까지는 기본 `False`였다가, 그 이후 실차에서 계속 켜둔 채 테스트하고 있는 상태입니다).
`bev_point_picker.py`로 라이브 카메라에서 직접 클릭해 픽셀좌표를 얻었다(방법: 좌/우 백선을 근거리
지점과 1m 지점에서 각각 찍음).

- **실측 픽셀좌표** (원본 640×480 프레임 기준, `config.py`의 `DL_BEV_SRC_PX_RAW`):
  - TL(좌상/먼왼쪽) = (246, 257)
  - TR(우상/먼오른쪽) = (455, 257)
  - BR(우하/가까운오른쪽) = (635, 333)
  - BL(좌하/가까운왼쪽) = (60, 333)
  - (`perception/dl_lane.py`가 이 절대좌표에서 `DL_ROI_Y0`만큼 뺀 ROI-상대좌표 `DL_BEV_SRC_PX`를 따로 계산해서 씀)
- **실측 실제 거리**:
  - 폭 W = 0.8m (좌/우 백선 간격 — §6.1 `LANE_WIDTH_M`과 동일 근거, 근거리/1m 지점 모두 같은 두 백선이므로 공통 적용)
  - 길이 L = 1.0m (TL~BL 실측, 근거리 지점과 1m 지점 사이 거리)
- **설계값(실측 아님, 캘리브레이션 계산 시 우리가 고른 값)**:
  - `DL_PIXELS_PER_METER = 200.0` (config.py, px/m 해상도 — 임의 선택)
- `DL_BEV_CANVAS_W`/`DL_BEV_CANVAS_H`는 손으로 정하는 값이 아니라 `perception/dl_lane.py`가 위 4점 +
  `DL_PIXELS_PER_METER`로부터 "ROI 전체가 여백 없이 들어가는 캔버스 크기"를 매 프로세스 시작 시
  자동으로 역산합니다([perception/dl_lane.py:137-138](perception/dl_lane.py#L137)).
- **주의**: 카메라 마운트가 바뀌면(재장착, 진동 등) `DL_BEV_SRC_PX_RAW`(config.py) 4점은 무효가 되므로
  재측정 필요. `bev_point_picker.py`로 재측정 가능.

### 6.4 아직 미실측 (플레이스홀더로 남아있는 값)

| 상수 | 위치 | 상태 |
|---|---|---|
| `METERS_PER_SPEED_UNIT` | config.py | 0.0 (미실측) — 모터 속도단위 1당 m/s. 측정법: `SPEED_NORMAL`로 5초 직진 후 이동거리÷5÷`SPEED_NORMAL` |
| `PIXELS_PER_METER` (전역) | config.py | 0.0 (미실측) — `DL_USE_BEV`가 실차 검증돼 기본으로 전환되면 §6.3의 `DL_PIXELS_PER_METER`로 채울 것 |
| `PP_MIN_LOOKAHEAD_PX`/`PP_WHEELBASE_PX`/`PP_LOOKAHEAD_BASE_PX` 등 | config.py | 전부 실차 미검증 튜닝값(추정/역산치일 뿐 실측 아님) |
