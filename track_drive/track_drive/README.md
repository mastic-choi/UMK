# track_drive 테스트 가이드

`track_drive.py`는 하나의 노드 안에서 신호등/차선/라바콘/장애물/추월을 전부 처리하는 2중 FSM 구조입니다
(`MissionState` S0~S4 + `BehaviorState`/`Phase`). 인지 모듈은 `perception/`, 조향/회피 제어는 `controller/`,
Hybrid A* 경로계획(B2 대안, 보존용)은 `planner/`에 모여 있습니다.

## 🏁 대회 규정 요약 (국민대 자율주행 경진대회, 2026년 9회 — 본선 경주)

> 출처: `2026년-9회대회-경주진행방법-7월29일자버전-1.pdf`(자이트론, 2026-07-29 버전). **본선 주행 경기(트랙
> 미션) 규정만** 정리했습니다 — 주차 대회는 별도 경기라 제외했습니다. 대회 진행 방식과 미션은 **대회 직전까지
> 변경될 수 있다고 원문에 명시**되어 있으니, 최종 확인은 항상 원본 공지로 할 것. 아래 규정이 `MissionState`
> S0~S4 FSM(위 인트로 참고)과 `Phase`(라바콘→고정장애물→방해차량) 순서가 왜 그렇게 설계됐는지의 근거입니다.

### 한눈에 보기
- 차량: RC카 기반 자율주행 차량 (카메라 등 센서로 인식, ROS2 기반 SW)
- 목표: 정해진 트랙을 **3바퀴** 자율주행하며 각 구간의 미션을 통과하고 결승선을 통과
- 성적: `총 주행시간 = 순수 주행시간 + 벌초(penalty seconds)` — 시간이 짧을수록 좋은 성적
- 경주는 오전 1회, 오후 1회 총 2회 진행되며 **둘 중 더 좋은 기록**으로 순위를 매김
- 3바퀴 총 주행시간이 **4분(벌초 미포함)** 을 넘으면 실격

### 미션 순서 (전체 흐름)
```
신호등 인식 출발
   → 라바콘(러버콘) 구간 주행
   → 차선인식 주행 (차선 준수)
   → 고정장애물 회피 주행
   → 방해차량(앞차) 추월 주행
   → [트랙을 3바퀴 반복 주행, 이 중 2바퀴째 또는 3바퀴째 한 번만 "지름길" 선택 가능]
   → 결승선 통과 (3바퀴 완주 후) → 경주 종료
```

### 트랙 구조
- 트랙은 직사각형 순환 코스(반시계/시계 방향 하나의 루프)이며, 코스 중간에 트랙을 좌우로 나누는 **분기 구간
  (지름길)** 이 있음(트랙 전체에 분기는 이 한 곳뿐).
- 트랙 상 구간 배치(대략 시계 순서):
  1. **출발 지점**: 4구 신호등 ↔ Gate(통과감지장치) 사이
  2. **라바콘 구간**: 출발 직후 코너 부근, 지그재그로 배치된 라바콘 사이를 충돌 없이 통과
  3. **차선 주행 구간**: 두 개의 실선(바깥쪽 경계) 사이를 벗어나지 않고 주행 (점선은 그냥 참고선, 1/2차선 구분 없음)
  4. **고정 장애물 회피 구간**: 고장난 차량 모형 등 정지된 장애물을 충돌 없이 회피
  5. **방해차량 추월 구간**: 저속으로 주행하는 방해차량을 추월
  6. **신호등/지름길 분기점**: 트랙 중앙을 가로지르는 지름길 진입 여부를 결정하는 4구 신호등(좌회전 화살표
     포함)이 있음
  7. **결승선(Gate, 통과감지장치)**: 출발 지점 근처, 한 바퀴마다 통과 → 3바퀴 주행 시 총 3번 통과하면 경주 종료
- 트랙을 총 **3바퀴** 주행해야 하며, **2바퀴째 또는 3바퀴째(마지막 바퀴) 중 한 번, 한 번만** 트랙 중앙의
  지름길을 통과할 수 있음. 나머지 바퀴는 바깥쪽 정규 코스로 주행.

### 진행 인력 / 셋업
- 대회 당일 **차량에는 반드시 커버를 장착**하고 참가.
- 팀원 2명 역할 분담: **팀원1**은 노트북을 들고 심판 옆에서 차량 SW를 원격 기동(`ros2 launch ...` 형태로
  심판이 화면을 볼 수 있어야 함), **팀원2**는 트랙에서 차량을 따라다니며 벌점 상황 등에서 차량 터치 등 필요
  조치를 수행.

### 출발 절차 (신호등 인식 출발, `MissionState.S0_WAIT_GREEN`)
- 출발 지점: Gate(통과감지장치)와 4구 신호등 사이. 심판이 신호등을 빨간불→파란불로 전환하는 순간부터
  주행시간(랩타임) 측정 시작(중간 노란불 대기 없음).
- **파란불 전 출발(신호위반)**: 10초 벌초 + 재출발 기회. 재출발에서도 빨간불에 출발하면 추가 10초 벌초 +
  처음 출발 대기 위치로 재이동 후 재주행(`벌초 = 10초 + 10초 + 재위치 소요시간`).
- 파란불로 바뀌었는데 출발 못 하면 차량 자세/위치를 전후좌우 **20cm 이내**로 손 조정 가능.
- 파란불 전환 후 **1분 이내** 미출발 시 **실격**.

### 주행 중 정지 금지 / 차량 터치 벌점
- 주행시간 측정 시작 후 차량이 멈춘 채 **1분 이내**에 재개하지 못하면 **실격**.
- 주행 중 사람이 차량에 손을 대면 **1회당 5초 벌초** (원칙적으로 심판 지시가 있을 때만). **한 바퀴 기준
  15회(60초) 초과 시 실격.**

### 라바콘 구간 (`Phase.LAVACON` / `BehaviorState.B1_LAVACON`)
- 라바콘 충돌 시 **개당 3초 벌초** (충돌해도 정상 경로 주행 중이면 별도 조치 불필요).
- 경로를 완전히 이탈하면 이탈 위치(앞바퀴가 이탈 발생 라바콘 위치 또는 그 뒤)로 옮겨 재주행 — **너무 앞으로
  옮기면 안 됨**(부당하게 유리한 위치 금지).
- **라바콘 구간을 1분 이내에 통과하지 못하면 실격.**

### 차선 준수 주행 (`_lane_drive()`/`_lane_steer()`)
- 양쪽 바깥쪽 **실선**을 벗어나지 않고 주행. 실선-실선 사이 어디로 주행해도 무방(점선/1·2차선 구분 없음).
- **차선 이탈 판정 조건** (하나라도 해당하면 이탈위치로 옮겨 재주행):
  1. 앞바퀴 2개가 동시에 실선 밖으로 나갈 때
  2. 좌/우 바퀴 2개가 실선 밖 + 동시에 카메라(차량 중앙부)도 실선 밖(차량 절반 이상 이탈)
  3. 좌/우 바퀴 2개가 실선 밖으로 나가서 **90cm 이상** 그 상태로 주행할 때
- 코너 안쪽 컷을 막기 위해 일부 코너 안쪽에 **주행 방해용 고정 장애물**이 배치됨 — 여기 걸려 멈추면 빨리
  판단해 터치(5초 벌초)로 옮겨야 함.

### 고정 장애물 회피 (`Phase.FIXED_OBSTACLE` / `BehaviorState.B2_OBSTACLE`)
- 고장난 차량 모형 등 정지 장애물을 충돌 없이 회피. 이 구간의 차선 이탈/터치는 위 일반 규정과 동일 적용
  (별도 예외 없음).

### 방해차량 추월 (`Phase.VEHICLE` / `BehaviorState.B3_VEHICLE`)
- 방해차량 1대가 저속으로 1·2차선을 오가며 주행.
- **추월 중에는 실선 이탈을 차선 이탈로 안 봄.** 추월 후 최대한 빨리 실선 안쪽 복귀 필요 — 복귀 후 방해차량과
  간격이 **90cm 이상**이면 오히려 차선 이탈로 판정.
- **추월 방향**: 방해차량이 주행 "안 하는" 차선 쪽으로만 추월 가능(같은 쪽 추월 시도는 차선이탈로 간주).
- 뒤에서 추돌(가해) / 추돌당함(피해) 모두 각각 **10초 벌초**. 피해 시엔 50cm 이내로 앞으로 옮길 수 있음.

### 지름길 분기 신호 (`MissionState.S2_INTERSECTION`, 1바퀴 주행 후)
- 트랙 중앙 분기점 4구 신호등(좌회전 화살표 포함)에서 **좌회전(지름길) 신호**가 켜지면 지름길 진입 가능.
- 좌회전(지름길) 신호는 **2바퀴째 시작 또는 3바퀴째(마지막 바퀴) 시작 중 랜덤으로 딱 한 번만** 등장 —
  나머지는 항상 초록만 점등(직진 확정). 즉 지름길 선택 기회는 3바퀴 중 **정확히 한 번**.
  - `signal_left_confirmed`(config.py `SIG_CONFIRM_FRAMES` 디바운스)가 이 좌회전 화살표 점등을 인식.

### 시간 제한 및 실격 사유 총정리
| 상황 | 조치 / 벌점 |
|---|---|
| 빨간불에 출발 | 10초 벌초 + 재출발 기회 |
| 재출발에서도 빨간불에 출발 | 추가 10초 벌초 + 원위치 이동 후 재주행 (총 10+10초 + 재위치 시간) |
| 파란불 전환 후 1분 내 미출발 | **실격** |
| 주행 중 정지 후 1분 내 미재개 | **실격** |
| 주행 중 차량 터치 1회 | 5초 벌초 (한 바퀴 기준 15회/60초 초과 시 **실격**) |
| 라바콘 충돌 | 개당 3초 벌초 |
| 라바콘 구간 1분 내 미통과 | **실격** |
| 차선 이탈 (조건 3가지 중 하나 해당) | 이탈위치로 이동 후 재주행 |
| 방해차량 추돌(가해/피해 모두) | 각 10초 벌초 |
| 3바퀴 총 주행시간이 4분(벌초 제외) 초과 | 주행 중단 및 **실격** |
| 결승선(Gate) 3회 통과 (3바퀴 완주) | 경주 종료 |

### 성적 산출
`총 주행시간 = 순수 주행시간 + 벌초 합계`. 오전/오후 각 1회 진행 → **두 기록 중 더 좋은(짧은) 시간**을
최종 성적으로 채택, 짧은 순으로 순위 결정.

---

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

> **(2026-08-06)** `DEBUG_VIZ_LIDAR` 기본값을 `True`→`False`로 변경했습니다. 필요할 때만 켜서 쓰세요.

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

**[2026-08-05, LQR 브랜치에서 이식] 단위버그 수정 — 픽셀/미터 두 모드:** 원래 `Q=diag(1,1)`이 `e_y`(px,
O(1~100))와 `e_psi`(rad, O(0.01~0.5))를 같은 가중치로 취급해서, 미세한 오차에도 조향각이 클램프(±100°)까지
튀는 버그가 실차 영상에서 확인됐다(`e_y=-40px` 하나만 넣어도 클램프 전 raw 조향각이 1234°). `LANE_DETECTOR_BACKEND=='dl'
and DL_USE_BEV`이면 `track_drive.py`가 자동으로 `DL_PIXELS_PER_METER`를 넘겨 `e_y`를 미터로 환산하는
**미터 모드**로 동작(같은 `e_y=-40px`에서 raw 조향각이 10.2°로 정상화됨) — 그 외 백엔드에서는 `None`이
넘어가 기존 픽셀 게인 방식(**레거시 모드**)으로 자동 폴백한다(`controller/lqr.py` 상단 주석 참고).

**LQR 튜닝 파라미터** (`config.py` `LQR_*`, 전부 실차 미검증):
| 파라미터 | 의미 | 비고 |
|---|---|---|
| `LQR_R_STEER=1.0` | 조향각 가중치(올릴수록 조향 억제, 지그재그 완화) | `Q_LATERAL`/`Q_HEADING` 건드리기 전에 먼저 조정 |
| `LQR_Q_LATERAL=1.0` / `LQR_Q_HEADING=1.0` | 횡오차/헤딩오차 비중 | lateral 비중↑ → 중앙복귀 서두름(오버슈트 위험), heading 비중↑ → 각도부터 맞추고 천천히 복귀 |
| `LQR_ALPHA=0.5` | 프레임간 저역통과 필터 | 반응이 느리면 올리고, 잔떨림이 있으면 낮출 것 |
| `LQR_WHEELBASE_M=0.26` / `LQR_SPEED_MPS=1.0` | [미터 모드] 실측 축거(m) / 속도 추정치(m/s) | `wheelbase_m`은 줄자 실측 가능. `speed_mps`는 엔코더 연동 전 임시값 — 실차 최우선 튜닝 대상 |
| `LQR_HEADING_PROBE_M`/`LQR_MIN_PATH_M=0.3` | [미터 모드] 헤딩오차 참조거리 / 최소 경로길이(m) | 노이즈 방지 안전장치 |
| `LQR_WHEELBASE_GAIN=50.0` / `LQR_SPEED_GAIN=120.0` | [레거시 모드 전용] 조향 강도 / 속도 대응값 | `pixels_per_meter=None`일 때만 사용 |
| `LQR_HEADING_PROBE_PX`/`LQR_MIN_PATH_PX=65.0` | [레거시 모드 전용] 노이즈 방지 안전장치 | 조향이 자꾸 직전값 유지로 빠지면 낮출 것 |

**디버그 방법:**
- 창: `DEBUG_VIZ_STEER = True`(기본값, [config.py:218](config.py#L218)) → `steer_debug` 창에서
  지금 어느 컨트롤러가 쓰이는지, 이번 프레임이 "직전값 유지"(주황)인지 "현재값 반영"(초록)인지, 조향각과
  (`lqr`일 때) 횡오차 `e_y`/헤딩오차 `e_psi`까지 한글로 보여줍니다(디버그 표시는 미터 모드에서도 항상 픽셀
  단위). cv2 기본폰트가 한글을 못 그려서 `kr_text.py`의 PIL 기반 렌더러를 씁니다 — 한글 폰트가 없는 환경이면
  영문 fallback으로 표시됩니다.

**알려진 한계:**
- LQR은 아직 실차 튜닝 전입니다. 처음 켤 때는 저속에서, 언제든 사람이 개입할 수 있는 상태로 테스트하세요.
- `localization/pose_estimator.py`의 `EncoderPoseEstimator`가 `self.pose_estimator`로 준비돼 있지만
  ([track_drive.py:249](track_drive.py#L257)), 실제 엔코더 ROS 토픽이 아직 확인 전이라 **어떤 콜백도
  갱신하지 않는 미배선 상태**입니다. 미터 모드도 지금은 이 pose 추정기를 쓰지 않고 `LQR_SPEED_MPS`(튜닝
  임시값)로 대신합니다 — 엔코더 토픽이 확인되면 실제 m/s를 `set_speed_mps()`(레거시 모드면 `set_speed_gain()`)에
  매 프레임 넣어주는 식으로 전환할 것.

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
프레임에는 그 줄어든 값 자체를 바닥으로 쓰도록 함께 수정했습니다([controller/pure_pursuit.py:228](controller/pure_pursuit.py#L228)).

**버그 — 자기순환 lookahead 락업 (2026-08-06 디버그 영상에서 발견, 같은 날 수정):** 위 구현은 damp
근거로 **직전 프레임에 이미 계산된** `self.last_curvature`를 그대로 재사용했습니다. 이게 자기순환
피드백을 만듭니다 — lookahead가 한번 짧아지면(`ld`↓), `curvature = 2·sin(α)/ld` 공식상 ld가 작을수록
같은 픽셀 단위 dx도 더 크게 증폭되어 curvature가 커지고, 그 커진 curvature가 `self.last_curvature`로
저장돼 다음 프레임 lookahead를 또 줄여버립니다. 한번 이 루프에 걸리면 실제 경로가 직진으로 돌아와도
lookahead가 `PP_LOOKAHEAD_MIN_PX(40px)` 근처에 계속 눌려있어 절대 원래 lookahead(90px+)로 복귀하지
못합니다.

- **실차 증상**: `dl_lane` 디버그 창을 녹화한 영상(2026-08-06 14:05)에서, `offset:+1.5px`(차선 중앙에
  거의 붙어있는, 육안으로도 직진 구간)인 프레임에서 조향각이 `ang=-65.7°`로 15초 넘게 고정. 같은 구간
  오버레이의 lookahead 표시(`ld:`)가 매 프레임 40px(하한)에 눌려있었음 — 조향기가 경로 전체가 아니라
  차량 바로 앞 40px짜리 구간만 보고 있었다는 뜻.
- **고침**: damp 판단을 "직전에 실제로 쓴 lookahead에서 나온 curvature"가 아니라, **매 프레임 댐핑
  적용 전 고정 기준 lookahead(`probe_px`)로 새로 계산한 `probe_curvature`**로 바꿨습니다
  ([controller/pure_pursuit.py:195](controller/pure_pursuit.py#L195)). probe는 항상 같은 기준 거리에서
  다시 재므로 "지금 경로가 진짜 코너인지"를 매 프레임 독립적으로 재평가해 순환을 끊습니다. 코너
  감속(0.5.1)이 참조하는 `last_curvature`는 여전히 실제로 사용된(댐핑된) lookahead 기준으로 계산해
  "지금 얼마나 세게 도는지"라는 원래 의미를 그대로 유지합니다.

**알려진 한계:**
- `PP_LOOKAHEAD_CURVATURE_GAIN=100.0`, `PP_LOOKAHEAD_MIN_PX=40px`(둘 다 config.py) 모두 추정치입니다
  (`curvature=0.01`, 반경≈100px짜리 중간 코너에서 배율 0.5가 되도록 잡음). `steer_debug` 창이나 CLI
  로그로 관찰하며 튜닝이 필요합니다.
- `STEERING_CONTROLLER='lqr'`일 때는 적용되지 않습니다 — `lqr.py`는 curvature 개념 자체가 없습니다.
- probe도 근거리 밴드 자체가 구조적으로(노이즈가 아니라 매 프레임 일관되게) 옆으로 치우쳐 있으면 여전히
  큰 curvature를 재현합니다(자기순환은 끊었지만 "근거리 세그멘테이션이 원래 부정확한" 문제 자체를
  고치진 않음) — da/ll 세그멘테이션 정확도 쪽(2.1/2.2절)이 근본 대응입니다.

### 0.5.3 진동을 코너로 오인해 매번 감속하는 문제 (`pure_pursuit` 전용, 2026-08-06)

`pure_pursuit`은 구조상 차선 중앙 부근에서 좌우로 조금씩 진동("와리가리", `PATH_EMA_ALPHA`/`PP_ALPHA`
주석에서 여러 차례 다룬 문제)하는데, 실차에서 이 진동이 있을 때마다 속도가 팍팍 줄었다 늘었다 하는
증상이 확인됐습니다.

**원인:** `_lane_drive()`의 코너 감속 두 갈래(3제곱 `turn_for_speed` 감속, `_corner_radius_speed_scale()`)
가 전부 "이번 틱 `abs(ctrl_angle)`"(혹은 그로부터 나온 curvature)을 그대로 코너 판단 신호로 썼습니다.
진동은 부호만 계속 바뀔 뿐인데, `abs()`를 취하면 부호 정보가 사라져서 진동의 매 스윙이 "지금 급하게
꺾고 있다"는 신호로 그대로 들어갑니다 — 실제 코너와 구분이 안 됐던 것. 게다가 `_corner_hold`(가속
제한용 peak-hold, 감쇠만 하고 절대 즉시 안 내려감)가 `max(turn_now, ...)`로 매 진동 스윙마다 다시
갱신되니, 속도가 깎인 뒤 다시 올라오는 것까지 늦어져 체감상 더 심하게 느껴졌습니다.

**수정:** 코너 판단 신호를 `self.ctrl_angle`의 **signed(부호를 유지한) EMA**(`self._corner_signal`,
`CORNER_SIGN_EMA_ALPHA=0.15`, [config.py](config.py))로 바꿨습니다([track_drive.py:1202](track_drive.py#L1202)
`_lane_drive()`, [1185](track_drive.py#L1185) `_corner_radius_speed_scale()`). 핵심은 **abs()를 EMA
"이후"에 적용**하는 것 — 진동처럼 부호가 계속 바뀌면 signed EMA 단계에서 서로 상쇄돼 0 근처로
수렴하고(그 다음 abs를 취해도 여전히 작음), 실제 코너처럼 한 방향으로 계속 꺾이면 EMA가 실제 각도로
수렴합니다(부호가 안 바뀌니 상쇄될 게 없음). `turn_for_speed`·`_corner_radius_speed_scale`·`_corner_hold`
전부 이 하나의 신호를 공유하므로 세 군데를 따로 고칠 필요 없이 한 번에 해결됩니다.

시뮬레이션(파이썬, 20Hz 가정)으로 확인한 결과:
- 매틱 부호가 바뀌는 진동(±25°): `turn_now` 최대 0.0375, 평균 0.02 — 사실상 무시됨.
- 4틱마다 반전하는 더 느린 진동(±35°): `turn_now` 최대 0.17 — 예전 방식(고정 0.35)의 절반 이하.
- 한 방향으로 유지되는 실제 코너(+40°): 20틱(≈1초)만에 `turn_now`가 0.38까지 수렴, 60틱에 0.40(정상
  포화값)에 도달 — 실제 코너는 여전히 정상적으로 감지·감속됩니다.

**알려진 한계:**
- `CORNER_SIGN_EMA_ALPHA=0.15`(시정수 ≈0.33초)는 실차 미검증 첫 추정치입니다. 값을 낮추면 진동 상쇄는
  더 잘 되지만 실제 코너 진입 감속도 그만큼 늦어집니다 — `turn_preview`(원거리 lookahead 기반 예측감속,
  이 신호와 무관하게 독립적으로 동작)가 어느 정도 보완하지만, 너무 낮추면 급코너에서 감속이 늦어 위험할
  수 있습니다. 실차에서 진동 주기와 코너 반응 속도를 같이 보며 조정할 것.
- 실제 조향 출력(`self.ctrl_angle`, 서보에 나가는 값)은 그대로입니다 — 이 수정은 "속도 계획이 진동에
  덜 민감해지는 것"이지 진동 자체(pure_pursuit의 근본 특성)를 없애지 않습니다. 진동 자체를 줄이려면
  `PP_ALPHA`/`PATH_EMA_ALPHA`/`PP_DX_DEADZONE_PX`(전부 config.py) 쪽을 볼 것.

### 0.5.4 VESC 실측속도를 lookahead 계산에 반영 (`pure_pursuit` 전용, 2026-08-06)

`pure_pursuit`의 속도 적응형 lookahead(`PP_LOOKAHEAD_SPEED_GAIN`)는 원래 "실제로 얼마나 빨리 달리고
있는가"를 봐야 하는데, 지금까지는 `self._prev_speed`(직전 **명령**속도, 모터단위)를 근사치로 썼습니다.
§7에서 이식한 VESC 실측속도(`self.v_mps`)가 이제 있으니, 이걸 쓰도록 바꿨습니다
([track_drive.py:1222](track_drive.py#L1222) `_speed_for_lookahead()`, [1265](track_drive.py#L1265) 호출부).

**동작:** VESC가 살아있으면(최근 `VESC_STALE_SEC` 이내 수신 + `abs(v_mps) >= VESC_MIN_SPEED_MPS`)
`v_mps`를 `METERS_PER_SPEED_UNIT`(§6.5 실측 회귀)로 나눠 "명령속도와 같은 단위"로 역환산해서 씁니다 —
그러면 `PP_LOOKAHEAD_SPEED_GAIN`/`PP_LOOKAHEAD_BASE_PX` 등 기존에 명령속도 스케일로 튜닝된 값을 그대로
재사용할 수 있습니다(단위를 바꾸면 게인도 전부 재튜닝해야 하므로 일부러 이렇게 함). VESC가 안 살아있으면
예전처럼 `self._prev_speed`로 폴백합니다.

**왜 도움이 되나:** 모터 데드존/가속 지연/슬립 때문에 "명령≠실제"인 구간(§2.3에서 다룬 코너 급감속
직후 등)에서, 예전엔 명령값 기준으로 lookahead를 계산해 실제 속도와 안 맞을 수 있었습니다. 실측값을
쓰면 그 구간에서도 lookahead가 실제 주행 상태를 반영합니다.

**알려진 한계:**
- `VESC_MIN_SPEED_MPS=0.05`(config.py, §7에서 이미 LQR용으로 쓰던 값을 공유 — 이름을
  `LQR_MIN_SPEED_MPS`에서 `VESC_MIN_SPEED_MPS`로 일반화했습니다)가 그대로 재사용됩니다.
- ROS1 `vesc_speed_bridge.py`가 안 떠 있으면 항상 폴백 경로만 타므로, 실제 개선 효과는 그 브리지가
  실차에서 살아있을 때만 발생합니다 — §7의 `vesc_debug` 창으로 확인할 것.
- 실차 미검증 — 폴백/실측 전환이 자주 일어나면(예: VESC 메시지가 간헐적으로 끊기는 경우) lookahead가
  단위 사이를 오가며 미세하게 들쭉날쭉할 수 있습니다. 실측해보고 문제되면 전환 시 저역통과를 추가로
  걸 것.

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
- 창: `DEBUG_VIZ_DL_LANE = True`([config.py:220](config.py#L220)) → da(주행가능영역, 초록 — 면적 상한에
  걸려 차선책 덩어리를 대신 쓴 프레임은 **주황**, ll 클리핑을 건너뛴 프레임은 **청록**)/ll(차선, 빨강)
  반투명 오버레이 + 밴드별 중심점(ll로 채택된 밴드는 **흰색**, ll이 부족해 da로 폴백한 밴드는 **노란색**
  — §2.4) + 피팅된 경로(웨이포인트) + `offset` 텍스트(각각 `[FALLBACK]`/`[LL_CLIP_SKIP]` 태그와
  `ll_bands:N/전체밴드수` 추가).
- CLI 로그: `[LANE] lane=편차px(검출여부) side=R/L차선 obs=... lava=...`.

### 2.1 da 면적 상한 대응 — 차선책(fallback) 덩어리 채택 (2026-08-06)

**증상:** S자 연속 커브 구간에서, da 최대 연결덩어리가 옆 차로/노면 반사와 붙어
`DL_DA_MAX_AREA_PX`(면적 상한, §6.4·§2.6)에 계속 걸리는 프레임이 길게 이어지면 — 디버그 창엔 여전히
경로(웨이포인트)가 그려지는데도 실차가 완전히 멈춰버리는 현상이 실측으로 확인됨.

**원인:** 예전 `_largest_da_component()`([perception/dl_lane.py](perception/dl_lane.py))는 최댓값
덩어리가 면적 상한을 넘으면 그 프레임을 곧바로 무효 처리(빈 마스크 반환)했다. `_update_path()`는
무효 프레임엔 `self.path`를 갱신하지 않고 직전 경로를 그대로 유지하는데(원래 "1~2프레임짜리 튐 방지"
목적), 이 무효 상태가 여러 프레임 연속되면 `self.path`가 사실상 무한정 옛 카메라 좌표에 얼어붙는다.
그런데 `_s1_lane_follow()`는 `self.lane_valid`를 확인하지 않고 매 주기 `_lane_drive()`를 호출하므로
(아래 "알려진 한계" 참고), 이 stale 경로를 계속 pure_pursuit/LQR에 먹인다 — 차는 계속 움직이는데
경로가 안 따라오니 조향이 점점 커지고(`turn_now`→1.0 포화), `_lane_drive()`의 3제곱 감속식이
`target_speed`를 바닥값 `SPEED_NORMAL*0.15`까지 계속 눌러버린다. §6.5 실측 결과 이 바닥값(실측
`SPEED_NORMAL=8.0` 기준 1.2)이 실차 구동 최소치(§6.5에서 추정한 데드존 ≈1.4)보다 낮아서, 명령은
계속 나가는데 실제로는 안 움직이는 상태로 굳어버린 것. (이 바닥값 자체는 이후 §2.3에서 별도로 고쳤다.)

**수정:** 가장 큰 덩어리가 면적 상한에 걸리면 곧장 무효 처리하지 않고, 그다음으로 큰 덩어리부터
순서대로 `[DL_DA_MIN_COMPONENT_AREA, DL_DA_MAX_AREA_PX]` 범위 안에 드는 걸 찾아 **대신 채택**한다
(`_largest_da_component()`, `self.da_fallback_used` 플래그로 표시). 같은 프레임에서 다른 덩어리로
분리돼 있었다는 건 그게 자기 차선일 가능성이 높다는 뜻이라, "몇 초씩 정지"보다 낫다는 판단. 어느
덩어리도 범위 안에 없으면(전부 상한 초과 혹은 전부 너무 작음) 기존과 동일하게 무효 처리한다 — 동작이
완전히 바뀐 게 아니라 "무효 처리 전에 한 번 더 시도"가 추가된 것.

**알려진 한계(이 수정 자체):** 차선책 채택 기준이 여전히 같은 면적 비율 임계값이라, 옆 차로 덩어리가
우연히 그 범위 안에 들면(자기 차선과 비슷한 크기로 잘려 보이는 경우) 잘못된 덩어리를 고를 수 있음 —
아직 실차 S자 구간 재검증 전. `DEBUG_VIZ_DL_LANE`의 주황 오버레이로 실제로 자기 차선을 골랐는지
확인할 것.

### 2.2 S자 커브 원거리 ll 두께 과다 검출 + da "작게 검출" 대응 (2026-08-06)

§2.1을 반영한 뒤에도 S자 연속 커브에서 여전히 정지 현상이 재현됐는데, 이번엔 da가 면적 상한(너무 큼)이
아니라 **너무 작게** 잡혀 `DL_DA_MIN_COMPONENT_AREA` 미만으로 걸러지는 경우였다. 원인을 실측 디버그
오버레이로 추적한 결과, 카메라에서 먼 커브 구간의 ll(차선)이 실제 선 두께보다 눈에 띄게 두껍게 잡히는
것을 확인 — `DL_USE_BEV`(BEV 원근보정) 특유의 실패모드였다.

**원인 (BEV 워프의 원거리 확대):** `DLSlideWindow._bev_warp()`는 da/ll을 이진화하기 **전**(float 확률맵)
상태로 원근변환한다(계단 현상 방지 목적, [perception/dl_lane.py](perception/dl_lane.py) 주석 참고).
호모그래피 성질상 카메라에서 먼 지점일수록 원근압축을 되돌리기 위해 더 크게 확대해야 하는데, 이때 모델
출력의 확률 0.5 근방 애매한 경계(blur)도 같이 확대된다. 근거리는 원래 확률이 뚜렷해 영향이 적지만,
원거리는 이 blur가 커져서 이진화(`DL_FG_THRESHOLD=0.5`) 후 ll이 실제보다 두껍게 잡힌다. 이 두꺼운 ll이
`_clip_da_by_ll()`에서 "차선 바깥" 판정 경계로 쓰이면서, 원거리 밴드의 da를 필요 이상으로 깎아낸다 —
da 자체(세그멘테이션)는 멀쩡한데 ll 클리핑이 지워버려서 마치 "da가 작게 검출된 것"처럼 보이는 것.
이렇게 여러 밴드가 무효화되면 §2.1과 동일한 경로 동결 → 조향 포화 → 속도 바닥값 정체로 이어진다.

**수정 (2단계):**
1. **ll 전용 이진화 임계값 상향** (`DL_LL_FG_THRESHOLD=0.7`, [config.py](config.py) — 기존 `da`용
   `DL_FG_THRESHOLD=0.5`와 분리). blur로 번진 저확률 가장자리를 미리 잘라내 원거리 ll이 두꺼워지는
   정도를 줄인다. `da`는 대회 요구사항에 명시된 `0.5`를 그대로 유지(건드리지 않음).
2. **ll 클리핑 결과가 너무 부실하면 클리핑을 건너뜀** (`detect()`, [perception/dl_lane.py](perception/dl_lane.py)).
   ①을 적용해도 여전히 클리핑 후 유효 밴드 수가 `DL_SLICE_FIT_MIN`(fit 최소 밴드 수) 미만이면, 클리핑
   전(=옆 차선 침범 방지가 없는 원본) da로 되돌려서 쓴다. §2.1과 같은 원칙 — "완전 무효화(→ 정지)"보다는
   "덜 안전하지만 주행 가능한 신호"를 우선한다. `self.da_ll_clip_skipped` 플래그로 표시.

**알려진 한계(이 수정 자체):**
- `DL_LL_FG_THRESHOLD=0.7`은 실차 미검증 첫 추정치. 너무 높으면 원거리 ll이 아예 안 보여서(반대로
  `_clip_da_by_ll()`이 "ll 안 보임 → 클리핑 생략"으로 넘어가는 것과 결과적으로 비슷해짐) 옆 차선 침범
  방지 효과 자체가 약해질 수 있음 — `DEBUG_VIZ_DL_LANE`의 빨강(ll) 오버레이로 원거리/근거리 선 두께가
  비슷해지는 지점을 찾아 조정할 것.
- ②(클리핑 건너뛰기)이 발동하면 그 프레임은 옆 차선 침범 방지가 아예 꺼진다 — 정지보다는 낫지만
  ㅓ교차로처럼 옆 차로 da가 실제로 붙어있는 상황에서 하필 같이 발동하면(이론상 가능, 실측 미확인) 옆
  차로로 새는 조향이 나올 수 있음. 청록 오버레이가 잦게 뜨는 구간이 있으면 원인(①로 부족한 건지, 다른
  구조적 문제인지)을 더 파야 함.

**알려진 한계 (기존, 미수정):**
- `_s1_lane_follow()`가 `self.lane_valid`를 확인하지 않고 `_lane_drive()`를 호출함(`_s3_shortcut()`은 확인함,
  [track_drive.py:1001](track_drive.py#L1001)). 카메라가 순간적으로 차선을 놓쳐도 마지막 유효 offset으로
  계속 조향하니, 실차 테스트 시 차선 이탈 구간에서 주의 깊게 볼 것. 위 2.1 증상의 근본 원인 중 하나이기도
  해서, S1에도 S3처럼 무효 지속시간 기반 폴백을 추가하는 게 다음 후보. (아직 미수정)
- (`classic_cv` 대안 백엔드) `perception/lane_util.py`의 CLAHE+adaptiveThreshold 기반 흰색 검출은
  "보존용"으로 유지 중이며 현재 라이브 미검증입니다.

### 2.3 코너 감속 하한이 모터 데드존보다 낮아 정지하는 문제 (2026-08-06)

§2.1/§2.2는 "da/ll 인식이 깨져서 경로가 얼어붙는" 경로로 정지에 이르는 문제였는데, 별도로 **인식이
멀쩡해도** 코너가 이어지면 같은 결과(정지)에 이를 수 있다는 게 확인됐다 — `_lane_drive()`([track_drive.py:1207](track_drive.py#L1207))의
코너 감속식(3제곱 `turn_for_speed` 감속 + `_corner_radius_speed_scale()`)이 둘 다 목표속도를
`SPEED_NORMAL*0.15`(=1.2, 두 곳에 하드코딩)까지 낮게 깎을 수 있었는데, §6.5에서 추정한 모터 데드존
(≈1.4)보다 낮다. 급커브가 잠깐이 아니라 좀 이어지면 목표속도가 이 바닥에 눌린 채 유지되고, 그러면
실차는 거의/전혀 못 움직이면서도 조향각 계산에 쓰이는 lookahead/오프셋은 차가 안 움직이니 안 바뀌어
감속이 안 풀리는 정지 상태로 굳는다 — §2.1/§2.2의 "경로 동결로 인한 정지"와 증상은 같지만 원인(인식
정상, 순수 속도계획 하한값 문제)은 다르다.

**수정:** 두 곳의 `SPEED_NORMAL*0.15` 하드코딩을 이름 있는 상수 `SPEED_CORNER_MIN`([config.py:90](config.py#L90))
으로 빼고, 데드존(≈1.4)보다 확실히 위인 값으로 올렸다([track_drive.py:1226](track_drive.py#L1226),
[1230](track_drive.py#L1230)). `_run_passing()`의 "양쪽 통과 불가 서행" 폴백([track_drive.py:1528](track_drive.py#L1528))도
같은 `SPEED_NORMAL*0.15` 패턴을 쓰지만 B2/B3는 `TEST_DISABLE_B2_B3=True`로 현재 꺼져 있어 이번엔
건드리지 않았다 — B2/B3 실차 검증 시 같이 정리할 것. 처음엔 3.0으로 올렸다가, 이후 최고속도
상향(아래 §2.4)과 함께 5.0으로 재상향(요청 반영) — 데드존 대비 여유를 더 두었다.

**알려진 한계:** `SPEED_CORNER_MIN=5.0`은 데드존 추정치(≈1.4)에 여유를 둔 값일 뿐 실차 미검증. §6.5가
이미 지적했듯 데드존 자체가 2점(speed 5, 10)짜리 외삽 추정이라, 실제 데드존이 다르면 이 값도 다시
맞출 필요가 있음.

### 2.4 최고속도/코너 최저속도 재설정 (`SPEED_NORMAL` 8.0 → 25.0, `SPEED_CORNER_MIN` 3.0 → 5.0, 2026-08-06)

요청에 따라 직진 최고속도(`SPEED_NORMAL`)를 8.0에서 25.0으로, 코너 최저속도(`SPEED_CORNER_MIN`, §2.3)를
3.0에서 5.0으로 올렸다([config.py:73](config.py#L73), [90](config.py#L90)).

**주의 — 실차 재검증 필요:**
- §6.5의 `METERS_PER_SPEED_UNIT` 회귀는 `speed=5`/`10` 두 점만 실측한 것이라, `SPEED_NORMAL=25`는 측정
  범위 밖(2.5배) 외삽이다. 실제 m/s·제동거리·코너 반응이 그 선형식대로 나올지 실차에서 다시 확인할 것.
- `PP_LOOKAHEAD_SPEED_GAIN`(=4.0, [config.py:257](config.py#L257)) 등 `pure_pursuit`이 `speed`(§0.5.4에서
  VESC 실측값 기반으로 바뀌었지만, 명령속도와 같은 단위로 역환산해서 넣으므로 스케일 자체는 그대로)를
  직접 입력받는 게인들도 최고값이 8→25로 커진 만큼 lookahead가 더 크게 튈 수 있다(다만
  `PP_LOOKAHEAD_MAX_PX=150`으로 클램프는 되므로 값이 무한정 커지진 않음) — §0.5 문서에 있는
  진동/오버슈트 증상이 심해지는지 관찰할 것.
- 가속 제한(`SPEED_ACCEL_STEP=0.85`/주기)은 그대로라, 0→25까지 도달하는 데 이전보다 더 오래 걸린다
  (약 29주기 ≈ 1.5초, 20Hz 기준) — 가속 자체는 안전 방향이라 값을 안 건드렸지만, 체감상 가속이 느리게
  느껴지면 `SPEED_ACCEL_STEP`을 같이 올릴 것.

### 2.5 원거리 크롭 (`DL_BEV_FAR_LIMIT_M`) + da 전체/채택분 구분 시각화 (2026-08-06)

**원거리 크롭**: BEV 캔버스는 "ROI 전체가 여백 없이 들어가도록" 자동 확장되는데(§6.3), 그 결과 da/ll
처리가 실측 캘리브레이션 지점(TL/TR, 1.0m)보다 더 먼 영역(외삽, 근거리 기준점으로부터 약 1.30m까지)
까지 포함하고 있었다는 걸 계산으로 확인했다. 실측 재측정(픽셀 좌표 재클릭) 없이, **이미 정확한**
`DL_PIXELS_PER_METER` 스케일을 그대로 이용해 근거리 기준점으로부터 `DL_BEV_FAR_LIMIT_M`(=0.7m,
[config.py:182](config.py#L182))보다 먼 캔버스 행을 워프 직후에 잘라낸다
([perception/dl_lane.py:480](perception/dl_lane.py#L480)). `DL_BEV_SRC_PX_RAW`/`DL_PIXELS_PER_METER`
자체는 그대로 두므로(캘리브레이션 안 건드림) 스케일 왜곡 없이 "얼마나 먼 데까지 볼지"만 제한한다 —
1.0m를 0.7m로 그냥 바꿔치기하면 안 되는 이유(캘리브레이션 왜곡)와 이 방식을 택한 이유는
`perception/dl_lane.py`의 `DL_BEV_FAR_CROP_ROW` 계산부 주석 참고. 원거리 ll 두께 과다검출(§2.2)
문제에도 도움이 될 것으로 기대 — 가장 blur가 심한 먼 영역 자체를 이제 안 본다.

**da 전체/채택분 구분 시각화**: `DEBUG_VIZ_DL_LANE` 오버레이에서 이제 모델이 "주행가능하다"고 판단한
da 전체(덩어리 선택/ll클리핑 전, `self.da_mask_all_roi`)를 **파란색**으로 먼저 깔고, 실제로 waypoint
추출에 쓰인 부분(`self.da_mask_roi`, 기존과 동일하게 초록/주황(면적상한 차선책)/청록(ll클리핑 건너뜀))을
그 위에 덧그린다([perception/dl_lane.py:590](perception/dl_lane.py#L590) `visualize()`). 채택분은
항상 전체의 부분집합이라 겹치는 픽셀은 초록/주황/청록이 그대로 덮어써 보이고, "모델이 본 전체 중 실제로
얼마나/어느 부분을 골랐는지"를 한 화면에서 바로 비교할 수 있다.

**알려진 한계:**
- `DL_BEV_FAR_LIMIT_M=0.7`은 실차 미검증 값. `DEBUG_VIZ_DL_LANE` 창에서 크롭 경계(파란/초록 영역이
  갑자기 끝나는 지점)가 원하는 위치에 오는지 확인할 것.
- 캔버스 높이가 줄어든 만큼(298→178px, 약 60%) `DL_N_SLICES`(8밴드)당 픽셀 수도 줄어든다 —
  `DL_MIN_PIXELS`/`DL_DA_MIN_COMPONENT_AREA`/`DL_DA_MAX_AREA_PX` 등 절대 픽셀 임계값들의
  "픽셀당 의미"가 이 크롭 이전과 달라졌을 수 있다. 아직 재검증하지 않았으니 da가 이유 없이 무효 처리되는
  빈도가 늘면 이쪽을 먼저 볼 것.

### 2.6 da 면적 상한을 실측 절대값으로 교체 (`DL_DA_MAX_AREA_PX`, 2026-08-06)

`_largest_da_component()`의 면적 상한 판단(§2.1)이 원래 "마스크 전체 대비 비율"(`DL_DA_MAX_AREA_RATIO`,
기본 0.6 — DL_USE_BEV 캔버스 크기가 바뀌어도 재계산 없이 유효하다는 장점으로 택한 값, §6.4)이었는데,
"이 정도면 비정상적으로 넓다"는 대충의 추정치였다. 실차에서 `_debug_viz_steer()`(`DEBUG_VIZ_STEER` 창)로
**직선 구간의 실제 da 면적**을 실측할 수 있게 됐으므로, 그 실측값 기반 절대 픽셀값(`DL_DA_MAX_AREA_PX`)
으로 교체했다(요청 반영) — 판단 로직 자체("이 값보다 크면 outlier로 버리고 그다음 크기 덩어리를
시도")는 `_largest_da_component()`에 이미 있던 그대로다(§2.1의 "차선책" 폴백), 비교 기준값의 근거만
추정 → 실측으로 바뀐 것.

**바뀐 것:**
- `config.py`: `DL_DA_MAX_AREA_RATIO`(비율) → `DL_DA_MAX_AREA_PX`(절대 픽셀수)로 이름·단위 교체.
- `perception/dl_lane.py`: `_largest_da_component()`가 매 프레임 `self.da_largest_area_px`(면적 1위
  덩어리, 채택 여부 무관)와 `self.da_chosen_area_px`(실제 채택된 덩어리, 무효 프레임엔 0)를 기록
  ([perception/dl_lane.py:388](perception/dl_lane.py#L388)).
- `track_drive.py`: `_debug_viz_steer()`(`steer_debug` 창)의 DA 면적 표시를 비율(%)에서 위 두 절대
  픽셀값 + `DL_DA_MAX_AREA_PX` 대비 퍼센트로 교체 — 이 창의 `DA largest:`가 그대로 실측값 후보다
  ([track_drive.py:1301](track_drive.py#L1301)).

**실측 방법:** 실차를 직선 구간에 놓고 `steer_debug` 창의 `DA largest:` 값을 몇 프레임 관찰 → 그 정상
범위의 대표값(여유를 약간 둔 상한)을 `DL_DA_MAX_AREA_PX`에 대입.

**실측값 (2026-08-06):** 직선 구간 3프레임 — 13,349px / 13,361px / 12,946px(평균 13,219px, 최대
13,361px). 여유를 두고 `DL_DA_MAX_AREA_PX = 13700`으로 설정 — 원래 플레이스홀더였던 62,478(옛 비율
0.6을 캔버스 크기로 그냥 환산한 값)보다 4.5배 이상 작다. 즉 그 플레이스홀더는 사실상 outlier
판정이 거의 안 걸리는 값이었고, 이번 실측으로 판정이 실제로 유효하게 작동하는 범위로 바로잡혔다.

**알려진 한계:**
- 3프레임(직선 구간 한 번)만 관찰한 값이라 표본이 적다 — 다른 직선 구간(조명/노면이 다른 곳)에서도
  비슷한 범위인지 추가로 관찰해볼 것. 완만한 커브에서도 정상적으로 da 면적이 다소 늘 수 있는데
  13,700이 그런 정상 범위까지 outlier로 잘못 거를 만큼 타이트한지도 같이 확인할 것.
- 비율 방식과 달리 캔버스 크기가 또 바뀌면(예: `DL_BEV_FAR_LIMIT_M` 재조정) 이 값도 같이 재측정해야
  한다 — 캔버스 크기(585×178px) 불변 가정 하의 값이라는 걸 기억할 것.

---

### 2.7 밴드별 중심 계산 모드 스위치 — `da` 단독 vs `ll`(차선)+`da` 하이브리드 (`DL_CENTER_MODE`, 2026-08-06)

지금까지 밴드(row 구간)별 중심점은 항상 da(주행가능영역) 무게중심이었다. 그런데 da는 "주행 가능한
영역"이지 "차로 중앙"이 아니라서, 갓길 등 여백이 넓은 구간에서 무게중심이 여백 쪽으로 쏠려 경로가
차로 중앙을 벗어나는 문제가 실측으로 확인됐다. ll(차선 자체, 두 백선)은 여백 크기와 무관하게
"선이 실제로 있는 위치"만 가리키므로 이 문제에서 자유롭지만, 아직 실차 전 구간에서 검증되지
않았다 — 그래서 기존 da 단독 방식을 남겨두고, `config.py`의 `DL_CENTER_MODE` 하나로 두 방식을
재시작만으로 전환해 실차에서 A/B 비교할 수 있게 했다.

- `DL_CENTER_MODE = 'da'`(main 기본값): 기존과 동일 — 밴드별 중심을 da 무게중심으로만 계산.
- `DL_CENTER_MODE = 'll_da'`(이 브랜치 기본값): 밴드마다 좌/우 ll이 둘 다 신뢰할 만하면
  (`DL_LL_SIDE_MIN_PIXELS` 이상 픽셀 + 두 선 간격이 `DL_LL_WIDTH_MIN_PX`~`DL_LL_WIDTH_MAX_PX`
  범위) 그 중점을 채택하고, 그 외 밴드(점선 틈/마모/반사/편측 가려짐)만 da 무게중심으로
  개별 폴백한다(`DLSlideWindow._ll_slice_centers()`, [perception/dl_lane.py](perception/dl_lane.py)).
  da 파편화 대응(`_largest_da_component`)/옆 차선 클리핑(`_clip_da_by_ll`)/ll sanity check는
  두 모드에서 동일하게 적용된다.

**디버그 시각화:** `DEBUG_VIZ_DL_LANE` 창에서 밴드별 중심점이 ll 채택 시 흰색, da 폴백 시 노란색
(`'da'` 모드에선 항상 노란색)으로 표시되고, 좌상단 텍스트에 `mode:`와 `ll_bands:N/전체밴드수`가
같이 뜬다.

**알려진 한계:**
- `DL_LL_SIDE_MIN_PIXELS=15`, `DL_LL_WIDTH_MIN_PX=100`, `DL_LL_WIDTH_MAX_PX=220`([config.py](config.py))은
  실측 차로폭(0.8m@200px/m=160px 기준 ±40% 여유)으로 잡은 첫 추정치, 실차 미검증. 너무 좁으면 정상 밴드도
  da로 자주 폴백해 `'ll_da'`의 효과가 약해지고, 너무 넓으면 반대 차선을 잘못 짝짓는 밴드를 걸러내지
  못할 수 있음 — `'ll_da'`로 전환 후 여러 직선/커브 구간에서 `ll_bands` 비율과 흰/노랑 점 분포를 보고
  조정할 것.
- `'ll_da'`는 밴드 단위 폴백이라 한 프레임 안에서 근거리는 ll, 원거리는 da처럼 신호 출처가 섞일 수
  있다 — 두 신호의 좌표계(둘 다 같은 BEV ROI 픽셀좌표)는 같지만, 실제 정확도 특성이 달라 그 경계에서
  미세한 경로 꺾임이 생길 가능성이 있음(이론상, 실측 미확인). 밴드별 흰/노란 점 색으로 출처 전환이
  잦은지, 전환 지점에서 경로가 부자연스럽게 꺾이는지 확인할 것.
- 노란 중앙선은 여전히 `lane_side`(주행 차선 판정)에만 쓰이고 경로 계산 자체에는 관여하지 않는다.

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

### 6.4 DL da/ll 튜닝값 BEV 재계산 (2026-08-05, `dl_lane_BEV_파라미터_변경사유.md` 이식)

`816d283 BEV 반영` 커밋에서 §6.3의 원근→BEV 좌표계 전환이 있었는데, `DLSlideWindow`의 da(주행가능영역)/
ll(차선) 마스크를 다루는 픽셀 기준 튜닝값 4개는 **원근 640×140px ROI 스케일로 잡힌 옛값 그대로 남아있었다.**
원근 좌표계는 깊이(화면 위치)에 따라 같은 실제 거리가 다른 픽셀 수로 보이는 왜곡된 좌표계라 애초에 고정된
m/px 환산이 없었는데, BEV(585×298px 캔버스, 면적비 옛 ROI 대비 **×1.9456**, `DL_PIXELS_PER_METER=200px/m`)로
바뀌면서 "픽셀당 의미"가 생겼는데도 재계산이 안 된 상태였다.

**문제였던 것**: 옛 `DL_SLICE_OUTLIER_MAX=60px`는 반차로폭(`LANE_WIDTH_M=0.4m`=80px)의 75%나 돼서, 사실상
옆 차선에 걸친 점도 "이상치 아님"으로 통과시켰다 — 교차로 진입부 등에서 점선으로 나뉜 옆 차선의 da가 한
덩어리로 이어붙어 중심선이 그쪽으로 끌려가는("차선을 뺏기는") 실패모드의 원인 중 하나.

| 상수 (config.py) | 옛값 (원근 스케일) | 새값 (BEV 스케일) | 근거 |
|---|---|---|---|
| `DL_DA_MIN_COMPONENT_AREA` | 800 | **1560** | 옛 ROI(89,600px²) 대비 비율(0.893%)을 새 캔버스(174,330px²)에서 유지 |
| `DL_SLICE_OUTLIER_MAX` | 60 | **40** | 반차로폭(0.4m=80px)의 1/2 |
| `DL_STABLE_JUMP_MAX` | 30 | **20** | 반차로폭의 1/4 (한 프레임 만에 옆 차선만큼 튀는 걸 "그럴듯한 변화"로 받아들이면 안 됨) |
| `DL_LL_CLIP_MARGIN_PX` | 15 | **8** | 실측 라인 두께 2.5cm(=5px @200px/m) + 세그멘테이션 경계 흔들림(1~2px) 여유 |

적용 후 `DEBUG_VIZ_DL_LANE` 오버레이로 교차로 진입 구간(좌회전 전용 차선 분기점)을 실제로 녹화/확인해 da
마스크가 더 이상 옆 차선으로 번지지 않는지 검증할 것 — **여전히 실차 미검증 추정치**입니다. 참고로 "da 면적이
통째로 크게 튀는" 실패모드(ㅓ교차로에서 옆 갈래까지 하나로 이어붙는 경우, 차선 없는 맨바닥을 통째로
오검출하는 경우)는 위 4개와 별개로 `DL_DA_MAX_AREA_RATIO`(마스크 전체 대비 면적 비율 상한, 기본 0.6)가
잡아낸다 — 이쪽은 "중심선이 서서히 옆으로 새는" 경우를 잡는 역할로 서로 보완 관계. (2026-08-06:
`DL_DA_MAX_AREA_RATIO`는 이후 §2.6에서 실측 절대 픽셀값 `DL_DA_MAX_AREA_PX`로 교체됐습니다 — 비율
방식이었던 이유·교체 근거는 그쪽 참고.)

### 6.5 속도 단위 ↔ m/s 환산 (`METERS_PER_SPEED_UNIT`, 2026-08-06 실측)

`drive()`가 발행하는 "모터 속도단위"(±100 클립)와 실제 속도(m/s)의 환산값입니다. 정지 상태에서 출발하면
목표속도까지 즉시 도달하지 않고(소프트웨어 가속 제한 `SPEED_ACCEL_STEP` + 모터/바퀴 관성) 서서히
가속하므로, 단순히 "거리÷시간"으로 나누면 가속 구간이 섞여 정속 속도를 과소평가합니다. 그래서 같은
속도값을 서로 다른 두 주행시간으로 측정해(각 2점) 회귀로 "정속 구간 기울기(m/s)"와 "가속 때문에 못 간
거리(오프셋)"를 분리했습니다 — 등가속 후 정속으로 가정하면 `거리 = v_정속×(시간 − 가속시간/2)`이므로
두 점을 지나는 직선의 기울기가 정속 속도, x절편의 2배가 가속시간입니다.

**실측 원시값:**

| speed 파라미터 | 주행시간(s) | 이동거리(m) |
|---|---|---|
| 5 | 3 | 1.04 |
| 5 | 6 | 2.50 |
| 10 | 3 (2회 평균) | 2.18 (=（2.30+2.06)/2) |
| 10 | 5 | 4.50 |

**회귀 결과:**

| speed 파라미터 | 정속 속도(m/s) | 가속 구간(참고, s) |
|---|---|---|
| 5 | **≈0.487** | ≈1.73 |
| 10 | **≈1.16** | ≈2.24 |

두 점(5, 0.487)/(10, 1.16)을 잇는 직선의 기울기는 `(1.16−0.487)/(10−5) ≈ 0.1347 m/s/unit` —
`METERS_PER_SPEED_UNIT`([config.py:417](config.py#L417))에 이 값을 채웠습니다.

**주의(2점 회귀의 한계):** 이 직선을 그대로 역산하면 x절편(속도=0이 되는 지점)이 **speed≈1.4**로
나옵니다 — 즉 그 아래로는 모터가 사실상 못 움직이는 데드존일 가능성이 있다는 뜻인데, 딱 2개 speed
값(5, 10)만 측정한 상태라 저속 구간을 실측 없이 외삽한 추정일 뿐입니다. `APPROACH_SPEED=2.0`
([config.py:99](config.py#L99))이 이 추정 데드존(≈1.4)에 가까워서, §2.1에서 다룬 "da 면적 상한 →
경로 정지 → 속도 바닥값(`SPEED_NORMAL*0.15`=1.2)"이 정확히 이 데드존 부근이라는 게 우연이 아닐
수 있습니다 — `SPEED_NORMAL`(8.0)이나 `APPROACH_SPEED`(2.0) 자체를 낮은 speed 값 몇 개로 추가
실측해서 이 데드존 추정을 검증/보정할 필요가 있습니다. (코너 감속 하한 자체는 §2.3에서
`SPEED_CORNER_MIN=3.0`으로 데드존 위로 올렸습니다 — `APPROACH_SPEED`는 아직 그대로입니다.)

### 6.6 아직 미실측 (플레이스홀더로 남아있는 값)

| 상수 | 위치 | 상태 |
|---|---|---|
| `PIXELS_PER_METER` (전역) | config.py | 0.0 (미실측) — `DL_USE_BEV`가 실차 검증돼 기본으로 전환되면 §6.3의 `DL_PIXELS_PER_METER`로 채울 것 |
| `PP_MIN_LOOKAHEAD_PX`/`PP_WHEELBASE_PX`/`PP_LOOKAHEAD_BASE_PX` 등 | config.py | 전부 실차 미검증 튜닝값(추정/역산치일 뿐 실측 아님) |

### 6.7 `LQR_WHEELBASE_M` 실측값 반영 (0.26 → 0.335, 2026-08-06, LQR 브랜치에서 이식)

`config.py`의 `LQR_WHEELBASE_M`이 "★실측 필요★" 플레이스홀더(0.26)로 남아있었는데, `LQR` 브랜치가
2026-08-06 같은 차량으로 줄자 실측한 값(0.335)을 갖고 있었습니다 — `planner/hybrid_astar.py`의
`wheelbase` 기본값이 이미 이 값(0.335, "같은 차량이므로 stanley.py와 반드시 같은 값을 써야 한다"는
주석과 함께)으로 갱신돼 있던 것과 같은 실측치입니다. `LQR_WHEELBASE_M`만 그 갱신에서 빠져 있었던
것으로 보여, `LQR` 브랜치 값으로 맞췄습니다(`controller/lqr.py`의 생성자 기본값도 동일하게 갱신 —
실제로는 `track_drive.py`가 항상 `LQR_WHEELBASE_M`을 명시적으로 넘기므로 이 기본값 자체는 안 쓰이지만,
문서 목적상 실측 전 플레이스홀더로 오해되지 않도록 같이 맞춤).

### 6.8 `PP_WHEELBASE_PX`를 물리 기반 값으로 계산 (80.0 → 67.0, 2026-08-06)

`controller/pure_pursuit.py`의 `PP_WHEELBASE_PX`(곡률→조향각 게인, `steer=atan(curvature*wheelbase_px)`)는
"실제 축거리 대신 쓰는 튜닝값"이라는 주석과 함께 80.0으로 하드코딩돼 있었다. 새로 실측값이 생긴 건
아니지만, **기존 두 실측/설계값을 조합해 계산**할 수 있다는 걸 확인했다:

- `LANE_DETECTOR_BACKEND='dl'`(기본값) + `DL_USE_BEV=True`(기본값)에서는 `self.lane_path`가
  `config.DL_PIXELS_PER_METER`(=200px/m, §6.3 — BEV 캔버스 정의상 정확한 스케일) 좌표계로 만들어진다.
- `LQR_WHEELBASE_M = 0.335m`(§6.7, 줄자 실측)는 pure_pursuit이 쓰는 것과 동일한 차량의 실제 축거리다.

따라서 `PP_WHEELBASE_PX = LQR_WHEELBASE_M * DL_PIXELS_PER_METER = 0.335 * 200 = 67.0`으로, "임의
튜닝값"이 아니라 물리적으로 근거 있는 값으로 대체할 수 있다(`config.py`, `controller/pure_pursuit.py`
생성자 기본값도 문서 목적상 동일하게 갱신 — §6.7의 `LQR_WHEELBASE_M`과 같은 패턴).

**★ 실차 재검증 필요 ★:** 80.0은 그 자체로 실차에서 "이 정도 조향 반응이 적당하더라"고 경험적으로
맞춰졌을 가능성이 있다 — lookahead 근사, BEV 워프 오차, 세그멘테이션 노이즈 등 다른 근사 오차를
상쇄해온 값일 수 있어서, 67.0로 바꾸면 같은 curvature에도 조향각이 더 작게(atan 인자↓) 나와 코너링이
더 완만해질 수 있다. 실차에서 코너 추종이 둔해지면 이 값을 다시 올리되, 그때는 "물리 기반 값에서
실차 튜닝으로 벗어난 것"임을 주석에 남길 것.

## 7. VESC 실측 속도 연동 (ROS1, 2026-08-06 LQR 브랜치에서 이식)

`LQR` 브랜치가 main과 갈라진 뒤 독자적으로 진행한 작업 중, main에 없던 실차 연동 하나를 가져왔습니다 —
**구동모터의 실제 회전속도(VESC 홀센서 기반)를 ROS2 쪽에서 받아오는 것**. `localization/pose_estimator.py`는
진작에 준비돼 있었지만(`EncoderPoseEstimator`), "이 로봇에 엔코더 토픽이 있는지 확인 전이라 미배선"
상태로 남아있었는데 — 그 확인이 이번에 됐습니다.

**구조:**
1. 이 로봇엔 별도 엔코더가 없고, VESC 드라이버(ROS1, `vesc_driver`)가 `/sensors/core`
   (`vesc_msgs/VescStateStamped`)로 모터 회전속도(ERPM)를 발행합니다.
2. `vesc_msgs`가 이 ROS2 워크스페이스엔 빌드돼 있지 않아(2026-08-06 실차 확인) `ros1_bridge`가 이
   커스텀 메시지를 그대로 못 넘깁니다.
3. 그래서 ROS1쪽에 작은 변환 노드([launch/vesc_speed_bridge.py](launch/vesc_speed_bridge.py))를 하나 더
   띄워 `state.speed`(ERPM) 값 하나만 표준 메시지(`std_msgs/Float32`)로 `/vesc_speed_erpm`에 다시
   뿌립니다 — 표준 메시지라 `ros1_bridge`가 별도 빌드 없이 자동으로 브리지해줍니다.
4. ROS2쪽 `track_drive.py`의 `cb_vesc()`가 이 토픽을 구독해 `VESC_SPEED_TO_ERPM_GAIN`(=4614.0,
   `vesc.yaml`의 `speed_to_erpm_gain` 실측값, [config.py](config.py))로 나눠 `self.v_mps`(m/s)로 변환합니다.

**이 값을 쓰는 곳 두 군데** (`control_loop()`, [track_drive.py](track_drive.py)):
- `self.lqr.set_speed_mps(self.v_mps)` — `VESC_MIN_SPEED_MPS`(=0.05) 이상일 때만 갱신합니다. 정지
  상태(v≈0)에서 그대로 넣으면 LQR의 상태전이행렬 B가 퇴화(조향이 상태에 영향을 못 미치는 것으로
  계산됨)하므로, 그 미만이면 직전 게인을 유지합니다.
- `self.pose_estimator.update(self.v_mps, math.radians(self.ctrl_angle), 0.05)` — 매 주기 갱신. 이제
  `EncoderPoseEstimator(wheelbase_m=LQR_WHEELBASE_M)`로 축거도 실측값이 물려 있어(§6.7), pose 추정이
  플레이스홀더 없이 동작합니다.

**배포 방법 (ROS1쪽, 이 워크스페이스 바깥):**
[launch/vesc_speed_bridge.py](launch/vesc_speed_bridge.py) 자체는 ROS1 노드라 이 ROS2 워크스페이스
안에서는 실행되지 않습니다 — noetic_ws 안 기존 패키지의 `scripts/`에 넣거나 새 패키지를 만들어
`rosrun`으로 띄우세요(파일 상단 주석에 상세 절차 있음). `[launch/manual_drive.launch.py](launch/manual_drive.launch.py)`는
수동주행/카메라 단독 테스트용 ROS2 launch로 이번에 같이 이식했습니다 — VESC 연동 자체와는 독립적입니다.

**확인 방법:** `DEBUG_VIZ_VESC=True`([config.py](config.py))면 `vesc_debug` 창이 뜹니다 —
빨강(NEVER_RECEIVED, 브리지 노드 미실행/토픽명 불일치/`ros1_bridge` 미전달 의심), 주황(STALE, 브리지·VESC
드라이버가 죽었을 가능성), 초록(LIVE, 정상) 세 상태를 색으로 바로 구분합니다.

**포팅하지 않은 것:** `LQR` 브랜치의 다른 커밋들(`변경사항`/`speed변경` 등)은 main이 그 이후 독자적으로
더 발전시킨 부분(perception 리팩토링, 코너 감속·`pure_pursuit` 락업 수정 등)과 겹치거나 낡은 flat 파일
구조를 그대로 갖고 있어 이식하지 않았습니다 — VESC 연동 파일 3개(`cb_vesc`/`_debug_viz_vesc`/구독 wiring,
`launch/vesc_speed_bridge.py`, `launch/manual_drive.launch.py`)와 `LQR_WHEELBASE_M` 실측값만 선별해서
가져왔습니다.
