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

> **(2026-08-11)** `DEBUG_VIZ_LIDAR`(`lidar_bev` 창) 정리 — 각도 컴퍼스(8방향 i-라벨)/자기가림
> 경계선(MASK_LO/HI)/포인트별 인덱스 숫자를 지웠습니다. 셋 다 `LIDAR_ANGLE_OFFSET_DEG`(§6.2)·
> `BODY_LO`/`BODY_HI` 값을 맞추던 캘리브레이션용이었는데 둘 다 2026-07-22 최종 확정된 뒤로는
> 실차 테스트 화면에 노이즈만 더했습니다. 지금은 거리 링 + 감지 ROI 박스(청록/초록/주황) + 포인트
> + 좌상단 상태 텍스트만 남아있습니다.

> **(2026-08-11 수정)** 이전엔 "이 프로젝트는 YOLO를 사용하지 않는다"고 적혀 있었는데, 부분적으로만
> 맞습니다 — 신호등/차선/고정장애물(B2)/방해차량(B3)은 여전히 YOLO 없이 카메라+라이다만 씁니다.
> 다만 라바콘(B1) 진입 트리거만은 `perception/yolo_cone.py`(YOLOv8n, `yolo_ros/cone_best_n.onnx`)로
> 카메라 이중확인을 추가했습니다(§3 참고). 이 작업은 원래 `smooth-imu-yaw-rate` 브랜치(커밋
> `0c0d88b`)에 있었는데 그 브랜치가 메인 라인에 머지되지 않아 한동안 "없는 것"처럼 보였을 뿐,
> 이번에 현재 브랜치로 수동 포팅했습니다.

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

`'dl'` 선택 시 `onnxruntime` 미설치나 `models/twinlitenetplus_small_bootstrap_v2.onnx`(.data)
부재 등으로 초기화가 실패하면 `_build_lane_detector()`
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

**[2026-08-10] 회귀(regression) 발견 및 복원:** 사용자가 "조향이 30도 이상 바뀌었다 직진으로
돌아오면 차선인식이 흔들리며 잘 안 되는 경향"을 보고해 원인을 조사하던 중, 바로 이 수정이
같은 날 뒤에 올라온 커밋 `80aefe3`("디버그창 적용" — 커밋 메시지상 조향 로직과 무관, `_debug_viz_steer()`
캔버스 레이아웃 변경이 주 목적)에서 **`self._corner_signal`/`turn_now`/`_corner_radius_speed_scale()`
세 군데가 전부 이 절의 "수정" 이전 상태(순간값 `abs(ctrl_angle)`/`self.pure_pursuit.last_curvature`)로
조용히 되돌아가 있는 걸 발견했다** — `git show 80aefe3 -- track_drive.py`로 확인, 디버그 캔버스
작업과 무관한 diff가 같은 커밋에 실수로 섞여 들어간 것으로 보인다. `track_drive.py`(`__init__`의
`self._corner_signal` 초기화, `_corner_radius_speed_scale()`, `_lane_drive()`)를 이 절에 기록된
원래 수정대로 복원했다 — 코드 자체는 이 절의 diff와 동일, 새로 설계한 건 아님. 급조향 후 진동이
매 스윙마다 급코너로 오인돼 속도가 흔들리고, 그 흔들림이 차선인식 쪽 흔들림(§2.17/§2.18의 탐색창
민감도)과 겹쳐 보였을 가능성이 있다 — **아직 실차 재검증 전**이니 복원 후 급조향→직진 복귀 구간에서
개선 여부를 확인할 것.

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

### 0.5.5 IMU 실측 curvature로 코너 감쇠 보강 (`pure_pursuit` 전용, 2026-08-06)

§0.5.2의 코너 진입 lookahead 감쇠는 지금까지 `probe_curvature`(비전 경로에서 뽑은 "아직 안 가본 앞쪽이
얼마나 휘었는지" 추정치) 하나에만 의존했습니다. IMU가 이번에 살아나면서(§8), "차량이 지금 실제로 얼마나
돌고 있는지"를 자이로로 직접 재서 이 판단을 보강하도록 바꿨습니다.

**구조** (`_imu_curvature_px()`, [track_drive.py:1241](track_drive.py#L1241)):
1. `cb_imu()`가 `msg.angular_velocity.z`(yaw rate, rad/s)를 `self.imu_yaw_rate`로, 수신 시각을
   `self._imu_t`로 저장합니다(기존엔 `orientation`만 쓰고 각속도는 버리고 있었음).
2. `kappa_m = imu_yaw_rate_ema / v_mps`(VESC 실측속도, §7)로 실제 curvature(1/m)를 구하고,
   `DL_PIXELS_PER_METER`(=200px/m)로 나눠 `pure_pursuit`이 쓰는 픽셀 curvature 단위로 맞춥니다 —
   이 환산은 `PP_WHEELBASE_PX`(§6.8)와 동일하게 `dl+BEV` 조합에서만 유효합니다.
   - **[2026-08-06 같은 날 보완] `imu_yaw_rate` 저역통과 추가.** `probe_curvature`는 경로 위 여러
     점을 누적한 값이라 어느 정도 스무딩이 걸려있는데, 자이로 순간값(`imu_yaw_rate`)은 그런 스무딩이
     없었습니다. 바로 아래 3번처럼 두 값 중 절댓값이 큰 쪽을 그대로 채택하는 구조라, 스무딩이 없는
     쪽이 노이즈 스파이크 한 프레임만으로 감쇠를 확 눌러버릴 위험이 있어서 — `IMU_YAW_RATE_EMA_ALPHA`
     (=0.3, config.py, `PP_ALPHA`/`CORNER_SIGN_EMA_ALPHA`와 동일한 관례)로 저역통과한
     `self._imu_yaw_rate_ema`를 대신 씁니다(`_imu_curvature_px()`, [track_drive.py:1246](track_drive.py#L1246)).
     IMU/VESC가 죽어있는 동안엔 이 EMA도 갱신을 건너뛰고 그대로 얼어있습니다(held 프레임에
     `last_curvature`를 안 건드리는 것과 같은 원칙) — 다시 살아나면 몇 프레임 안에 자연 수렴합니다.
3. `controller/pure_pursuit.py`의 `control()`이 `probe_curvature`와 이 `imu_curvature_px` 중
   **절댓값이 더 큰 쪽**으로 감쇠를 겁니다 — 비전이 못 본 코너를 IMU가 잡아내는 경우(또는 그 반대)를
   놓치지 않기 위한 보수적 선택입니다. 부호는 안 맞춥니다(`abs()`로만 쓰여서 실차 미검증인 IMU 부호규약이
   실제 조향에 영향을 못 줍니다).

**가드 — IMU/VESC 둘 다 살아있을 때만 반영:** `LANE_DETECTOR_BACKEND=='dl' and DL_USE_BEV`가 아니거나,
IMU가 죽어있거나(`IMU_STALE_SEC=0.5` 이상 미수신), VESC가 죽어있으면(§7 `VESC_STALE_SEC`/
`VESC_MIN_SPEED_MPS` 동일 기준, `_vesc_live()`로 통합) `_imu_curvature_px()`가 `None`을 반환하고
`pure_pursuit`은 기존처럼 `probe_curvature` 단독 판단으로 자동 폴백합니다 — 코드가 항상 들어가 있어도
센서 중 하나라도 안 살아있으면 동작이 예전과 완전히 동일합니다.

**현재 상태 (2026-08-06):** VESC가 지금 실차에서 안 잡히는 상태라(§7) `_imu_curvature_px()`가 항상
`None`을 반환 중 — 즉 이 기능은 지금 당장은 아무 영향이 없습니다(의도된 동작). VESC 연결만 복구되면
코드를 더 안 건드려도 자동으로 활성화됩니다.

**확인 방법:** `DEBUG_VIZ_STEER=True`(기본값)의 `steer_debug` 창에 `pure_pursuit` 모드일 때
`lookahead/curvature` 값과 `IMU curvature:` 줄이 추가로 표시됩니다 — 회색 `미반영(IMU/VESC 확인)`이면
아직 폴백 중, 초록색 숫자가 뜨면 실제로 이번 프레임 감쇠 판단에 IMU 값이 반영된 것입니다.

**알려진 한계:**
- IMU 각속도 부호규약(z축 +가 좌/우 중 어느 쪽인지)이 실차 미검증입니다. 위에서 설명한 대로 `abs()`로만
  쓰여서 조향 자체엔 영향이 없지만, 나중에 다른 용도로 부호를 쓰게 되면 먼저 검증할 것.
- `probe_curvature`와 (저역통과된) `imu_curvature_px`를 여전히 단순 `max(abs, abs)`로만 합칩니다 —
  실제로 어느 쪽이 더 신뢰할 만한지 가중치를 다르게 주는 것(칼만 필터 등)은 아직 안 함. 실차 데이터
  쌓이면 재검토.
- `IMU_YAW_RATE_EMA_ALPHA=0.3`도 다른 값들처럼 실차 미검증 추정치입니다. VESC 복구 후 `steer_debug`의
  `IMU curvature` 값이 여전히 프레임마다 들쭉날쭉하면 낮추고, 코너 반응이 눈에 띄게 늦으면 올릴 것.

### 0.5.6 속도를 올리면 진동이 심해지는 문제 — lookahead 상한이 속도 증가를 못 따라감 (`pure_pursuit` 전용, 2026-08-07)

**실차 증상:** `SPEED_NORMAL`이 5였을 때는 진동이 거의 없었는데, 20까지 올리자 진동이 눈에 띄게 심해짐.

**원인:** `speed_lookahead_px = PP_LOOKAHEAD_BASE_PX + PP_LOOKAHEAD_SPEED_GAIN*speed`
([controller/pure_pursuit.py:203](controller/pure_pursuit.py#L203))는 속도가 오를수록 lookahead도 같이
늘려서 안정성을 유지하는 설계인데(§0.5.2 위쪽 pure_pursuit.py 상단 주석: "gain=4.0은 SPEED_NORMAL=5
기준(90+4*5=110)"), 이 값이 `PP_LOOKAHEAD_MAX_PX`(당시 150)에 클램프됩니다. `SPEED_NORMAL`이 이후
25까지 오르면서(§0.5 도입부, `config.py` `SPEED_NORMAL` 주석 "8.0 → 25.0") 이론상 필요한 lookahead는
`90+4*25=190`인데 150에서 막혀, 속도 15 이상부터는 lookahead가 더 이상 안 늘어나고 고정돼 있었습니다.
Pure Pursuit은 lookahead(`ld`)가 짧을수록 `curvature = 2*sin(alpha)/ld` 공식에서 같은 픽셀오차(dx)도
더 크게 증폭시키므로(§0.5.2에서 이미 한 번 겪은 문제), 속도만 오르고 lookahead가 못 따라가면 고속에서
조향이 과민해져 진동으로 이어집니다.

**수정:** `PP_LOOKAHEAD_MAX_PX`를 150 → 190(`SPEED_NORMAL=25`를 그대로 대입한 값)으로 올림
([config.py:306](config.py#L306) 주변). `PP_LOOKAHEAD_SPEED_GAIN`/`PP_LOOKAHEAD_BASE_PX`는 그대로라
저속 동작은 변화 없고, 속도가 15 이상으로 오를 때 lookahead가 다시 정상적으로 계속 늘어납니다.

**알려진 한계:**
- 190은 공식을 그대로 대입한 값일 뿐 실차 재검증 전입니다. 그래도 고속에서 진동이 남으면 다음으로
  `PP_ALPHA`(현재 0.5, 조향각 프레임간 저역통과)를 낮추는 쪽을 볼 것 — §0.5.3에서 언급한 "진동 자체를
  줄이는" 레버 중 하나입니다.
- `SPEED_NORMAL`이 25보다 더 오르면 이 상한도 같이 재계산해야 합니다(`90+4*speed` 공식 유지).

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
  — §2.4) + 피팅된 경로(웨이포인트) + `offset` 텍스트(각각 `[FALLBACK]`/`[LL_CLIP_SKIP]`/`[LL_VIRTUAL]`
  (§2.14) 태그와 `ll_bands:N/전체밴드수` 추가).
- CLI 로그: `[LANE] lane=편차px(검출여부) side=R/L차선 obs=... lava=...`.

**2026-08-07 da/ll 파이프라인 정비 요약:** 하루 동안 da(주행가능영역)/ll(차선) 검출→waypoint 추출
전 구간을 손봤다. 상세 원인/수정/한계는 각 절 참고, 여기서는 흐름만:
- **da 덩어리 선택**(§2.8, §2.11): 후보 선택 기준을 면적 순위 → 직전 프레임과의 근접성(연속성) →
  **시드(차량 위치와 맞닿은 덩어리, 최우선)** 순으로 발전시켰다. **면적 임계값(`DL_DA_MIN/MAX_AREA_PX`)
  만으로 최선의 da를 고르는 방식은 실차 검증 결과 포기** — 지금은 위치/연속성 근거가 있을 때만 통과
  기준으로 쓰고, 근거 없는 순수 면적순위는 최후 폴백으로만 남아있다.
- **da 밴드 중심**(§2.7, §2.12): 무게중심(픽셀 밀도 가중 평균) → **좌우 경계 중점**(Voronoi 근사)으로
  바꿔, 갓길 등 여백이 넓을 때 중심이 그쪽으로 쏠리는 문제를 없앴다.
- **ll 밴드 추출**(§2.9, §2.13): "반쪽 전체를 보는 무게중심" → **좁은 고정폭 슬라이딩 윈도우** →
  **좌/우 완전 독립 추적**(한쪽만 보여도 반대쪽을 러닝 차로폭으로 추정)으로 발전시켰다.
- **경로 생성**(§2.10): 2차 다항식 피팅+외삽 → **구간별 선형보간**(`np.interp`)으로 바꿔, 저차수
  피팅이 노이즈로 튀어 근거리(조향에 가장 큰 영향) 경로가 휘는 문제를 없앴다.
- **da-옆차선 병합 방어**(§2.14): ll이 프레임 전체에서 안 보이는 경우(da 자체가 두 차선을 구분 못 하고
  통째로 나옴 — 실차로 확인, **침식(erosion)은 이 경우엔 안 통해서 폐기**) **ll 잔상(decay) + 기대
  차로폭 기반 가상경계**로 대응한다.
- **ll 이진화 임계값**(§2.15): 실차 영상을 다시 보니 "가끔 끊김"이 아니라 **ll이 상시 거의 안
  잡히는** 상태였다 — `DL_LL_FG_THRESHOLD` 0.7 → 0.5로 인하. 이걸로도 부족하면 이진화 자체를
  없애고 확률값 가중치로 계산하는 방식을 검토 중
  ([ll_probability_weighting_proposal.md](ll_probability_weighting_proposal.md), 미구현 제안).
- 전 구간이 아직 **실차 재검증 전** — 튜닝값 대부분이 이번에 처음 잡은 초기 추정치다.

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

### 2.8 da 차선책 선택 기준 — "면적 순위" → "직전 채택 덩어리와의 근접성" (2026-08-07)

**증상:** §2.1의 차선책 로직(`_largest_da_component()`)은 상한 초과 시 그다음으로 **큰** 덩어리부터
순서대로 범위 안에 드는 걸 골랐다. 이 방식은 두 덩어리(자기 차선 vs 옆 차로/반사)의 면적이 비슷할 때
프레임마다 대소 순위가 뒤집히는 것만으로도 채택 대상이 계속 바뀌어, 실제로는 같은 차선을 계속 보고
있는데도 경로가 불필요하게 흔들리는 문제가 있었다.

**수정:** 차선책 선택 기준을 면적 순위에서 "연속성"으로 바꿨다 — `_largest_da_component()`가
`self._prev_da_centroid`(직전 프레임에 실제로 채택된 덩어리의 중심)를 들고 다니다가, 이번 프레임엔
그 중심과 **가장 가까운** 덩어리를 최우선 후보로 고정한다. 이 근접 후보의 면적이
`[DL_DA_MIN_COMPONENT_AREA, DL_DA_MAX_AREA_PX]` 범위 안이면 순위와 무관하게 바로 채택하고, 범위를
벗어났을 때만(교차로에서 실제로 다른 갈래로 넘어갔거나 따라가던 덩어리가 사실상 사라진 경우) 기존
면적 내림차순 차선책으로 넘어간다. 무효 프레임(빈 마스크) 뒤에는 `_prev_da_centroid`를 `None`으로
리셋해, 한참 뒤 엉뚱한 위치의 덩어리가 "옛 중심과 가장 가깝다"는 이유만으로 잘못 이어붙는 걸 막는다.
`self.da_fallback_used` 플래그(디버그 시각화 주황색)의 의미도 "면적 1위가 아님"에서 "이번 프레임엔
근접 연속 후보 대신 면적순위 차선책을 썼음"으로 바뀌었다.

**알려진 한계:** 근접성 판단이 덩어리 중심(centroid) 간 단순 유클리드 거리라, 급격한 커브에서 프레임 간
실제 차로 이동량이 크면(카메라 프레임레이트 대비) 다른 차로 덩어리가 "더 가까운" 것으로 잘못 골라질
가능성이 이론상 있음 — 실차 재검증 전. `DEBUG_VIZ_DL_LANE`의 주황 오버레이(차선책 발동 여부)로
전환이 잦은 구간이 있는지 확인할 것.

### 2.9 ll 좌/우 탐색을 "반쪽 전체"에서 "좁은 고정폭 창"으로 (`DL_LL_SEARCH_HALF_WIDTH_PX`, 2026-08-07)

**증상:** `DLSlideWindow._ll_slice_centers()`(`DL_CENTER_MODE='ll_da'|'ll'`에서만 쓰임)가 밴드를 좌/우
분리 기준점(`cur_ref`) 하나로 반씩(왼쪽 전체/오른쪽 전체) 나눠 그 반쪽 안의 모든 픽셀로 무게중심을
냈다. ROI 폭이 넓으면 그 "반쪽"도 수백 px라, 옆 차선 선·반사광 등 무관한 픽셀이 반쪽 어디에 있든
평균에 그대로 섞여 들어가는 구조적 문제가 있었다("여러 ll 후보 중 뭘 선택할지" 문제의 원인).

**참고한 아이디어:** [github.com/junhyukch7/Advanced-Lane-Detection](https://github.com/junhyukch7/Advanced-Lane-Detection)의
고전적 슬라이딩 윈도우는 좌/우 각각 폭 120px(반경 60px)짜리 좁은 창만 보고, 그 창에서 찾은 평균
위치로 다음 창을 옮기는 방식이라 창 밖의 무관한 픽셀이 애초에 안 보인다.

**수정:** `_ll_slice_centers()`가 좌/우 각각 `cur_left`/`cur_right`(예상 위치)를 따로 들고 다니며, 그
위치 ±`DL_LL_SEARCH_HALF_WIDTH_PX`(초기값 60, [config.py](config.py))짜리 좁은 창 안에서만
`cv2.moments`를 구하도록 바꿨다([perception/dl_lane.py](perception/dl_lane.py) `_ll_slice_centers()`).
좌/우 초기 위치는 `ref_x`(직전 프레임 확정 lane_center) 기준 ±(기대 차로폭/2)로 잡고, 이번 밴드에서
실제로 채택(양쪽 다 신뢰됨)됐을 때만 `cur_left`/`cur_right`를 갱신한다(기존 `cur_ref` 갱신 원칙과 동일).

**알려진 한계:** 탐색창이 좁아진 만큼, 급커브에서 밴드 간 실제 선 이동량이
`DL_LL_SEARCH_HALF_WIDTH_PX`보다 크면 창이 선을 놓치고 그 밴드부터 추적이 끊긴다 — 실차 미검증.
`DEBUG_VIZ_DL_LANE`에서 급커브 구간의 `ll_bands` 비율이 갑자기 뚝 떨어지면 이 값을 키울 것.

### 2.10 경로 생성을 "2차 다항식 피팅+외삽"에서 "구간별 선형보간"으로 (`_fit_and_sample_path`, 2026-08-07)

**증상:** `lane_util.SlideWindow._fit_and_sample_path()`(dl/classic_cv 백엔드 공용)는 유효 밴드 점이
3개 이상이면 2차 다항식 하나로 전 구간을 피팅하고, 실측 데이터가 하나도 없는 근거리(`roi_h`, 차량
바로 앞 — 조향에 가장 큰 영향을 주는 구간)까지 그 곡선으로 외삽했다. 밴드 3~5개짜리 저차수 피팅이라
노이즈로 계수가 조금만 튀어도 외삽 구간이 크게 휘어질 수 있었다(`PATH_EXTRAPOLATE_MARGIN` 클램프는
안전판일 뿐 흔들림 자체를 줄이지 못함).

**참고한 아이디어:** 같은 [Advanced-Lane-Detection](https://github.com/junhyukch7/Advanced-Lane-Detection)
레포가 정확히 같은 이유("다항식 보간법은 고차항으로 갈수록 오차가 커질 가능성이 있어")로 고차 다항식
대신 슬라이딩 윈도우 점들을 직선으로 잇는 선형보간을 쓴다.

**수정:** `np.polyfit`(2차) 대신 `np.interp`로 밴드 점들을 구간별 선형보간하도록 바꿨다
([perception/lane_util.py](perception/lane_util.py) `_fit_and_sample_path()`). `np.interp`는 데이터
범위 밖에서 곡선으로 외삽하지 않고 가장 가까운 실측점의 x를 그대로 유지(hold)하므로, 근거리에 실측
밴드가 없어도 값이 곡선으로 튈 수가 없다 — 외삽 자체가 없어져 클램프보다 근본적으로 안전하다. 더 이상
쓰이지 않는 `PATH_EXTRAPOLATE_MARGIN` 상수는 제거했다.

**알려진 한계:** 곡선 구간에서 부드러운 다항식 곡선 대신 각진(piecewise-linear) 경로가 나올 수 있다 —
`PP_ALPHA`(조향각 저역통과)가 이미 완충하지만, 실차에서 커브 진입 시 조향이 미세하게 계단식으로
반응하는지 확인할 것. 실차 미검증.

### 2.11 da 덩어리 선택에 "시드(seed) 기반" 최우선 후보 추가 (`_largest_da_component`, 2026-08-07)

**증상:** §2.8의 "직전 채택 덩어리와의 근접성" 선택은 어디까지나 *과거 판단*에 기대는 방식이다. 만약
직전 프레임에 이미 엉뚱한 덩어리를 채택했다면(예: 교차로에서 다른 갈래로 잘못 넘어감), 그 뒤로도 "그때
그 위치와 가장 가깝다"는 이유만으로 계속 틀린 채로 이어질 수 있다 — 드리프트가 스스로 교정되지 않는다.

**참고한 아이디어:** 사용자 제안 — "본선 DA 필터링"에 Region Growing(Flood Fill)/CCL을 써서 차량의
현재 위치(이미지 최하단 중앙)를 시드로 삼아 그 위치와 물리적으로 연결된 덩어리만 골라내는 방식.

**수정:** `_largest_da_component()`에 시드 기반 최우선 후보를 추가했다
([perception/dl_lane.py](perception/dl_lane.py) `_largest_da_component()`). `cv2.floodFill`을 새로
돌릴 필요 없이, 이미 계산된 `connectedComponentsWithStats()`의 `labels`에서 시드 영역
(ROI 최하단 `DL_DA_SEED_ROWS_PX`행 × ROI 중앙 `DL_DA_SEED_HALF_WIDTH_PX`반경, 둘 다 신규
[config.py](config.py))의 라벨만 조회한다(CCL 결과 재사용). 그 영역과 맞닿은 덩어리가 있고 면적이
유효 범위 안이면 순위/연속성과 무관하게 최우선 채택한다 — "차량이 실제로 서 있는 위치"라는 매 프레임
독립적인 물리 신호라, 직전 프레임의 오판에 영향받지 않고 스스로 교정된다. 우선순위는 ①시드 →
②직전 프레임과의 근접성(§2.8) → ③면적 내림차순 차선책(§2.1) 순.

**알려진 한계:** 카메라/BEV 캔버스가 차량 중심선에 맞춰 캘리브레이션돼 있다는 전제다 — 카메라
마운트가 비대칭이면 시드 x좌표(`ROI 폭/2`)도 다시 잡아야 한다. `DL_DA_SEED_HALF_WIDTH_PX`가 너무
넓으면 옆 차선까지 시드에 걸려 잘못 채택될 수 있고, 너무 좁으면 시드 영역이 자주 비어 ②/③ 폴백만
타게 된다 — 실차 미검증, `DEBUG_VIZ_DL_LANE`으로 확인 후 조정할 것.

### 2.12 da 밴드 중심을 "무게중심"에서 "좌우 경계 중점"으로 (`_slice_edge_midpoints`, 2026-08-07)

**증상:** da 밴드 중심을 `cv2.moments`로 낸 무게중심(픽셀 밀도 가중 평균)으로 계산해왔는데, 갓길 등
한쪽 여백이 넓으면 무게중심이 그 여백 쪽으로 쏠려 실제 도로 폭의 "정중앙"에서 벗어나는 문제가
있었다(§2.7에서 이미 다룬, `ll_da`/`ll` 모드가 생긴 이유이기도 한 그 문제).

**참고한 아이디어:** 사용자 제안 — Voronoi 다이어그램(좌우 경계에서 등거리인 점들로 궤적을 만드는 것)
개념. 실제 Voronoi/skeletonization을 계산하는 대신, 밴드마다 "가장 왼쪽 픽셀 열 + 가장 오른쪽 픽셀
열의 중점"만 써도 같은 효과(밀도가 아니라 순수 "폭의 중앙"을 잡음)를 훨씬 싸게 얻을 수 있다는 아이디어.

**수정:** `lane_util.SlideWindow`에 `_slice_edge_midpoints()`를 새로 추가하고
([perception/lane_util.py](perception/lane_util.py)), da 밴드 중심 계산(`DLSlideWindow.detect()`의
`raw_da_centers`)을 기존 `_slice_centers()`(무게중심)에서 이걸로 교체했다
([perception/dl_lane.py](perception/dl_lane.py) `detect()`). da 전용으로만 적용했다 — ll/좌우차선처럼
원래 얇은 선은 두 방식의 차이가 미미해 그대로 `_slice_centers()`를 쓴다.

**알려진 한계:** 좌우 끝 픽셀 열만 보므로, 노이즈로 튄 픽셀 하나가 밴드 반대쪽 끝에 찍히면(반사광 등)
경계가 그만큼 밀려날 수 있다 — 무게중심 방식은 그런 노이즈 한 점의 영향이 평균으로 희석되는 반면 이
방식은 그렇지 않다. `_reject_outliers()`가 그런 밴드를 걸러내는 안전판이긴 하지만, 실차에서 반사광이
잦은 구간에 특히 취약한지 확인할 것 — 실차 미검증.

### 2.13 ll 좌/우 추적을 완전히 독립된 슬라이딩 윈도우로 (`_ll_slice_centers`, 2026-08-07)

**증상:** `_ll_slice_centers()`(§2.9에서 좁은 탐색창으로 바꾼 그 함수)는 좌/우를 한 밴드 안에서 같이
판정했다 — 한쪽 창에 픽셀이 모자라거나, 양쪽 다 찾았어도 두 중심 간 거리가 비정상이면 그 밴드
전체를 버렸다. 실제로는 한쪽 차선이 반사/가려짐으로 몇 밴드 끊겨도 반대쪽은 계속 잘 보이는 경우가
흔한데, 그럴 때도 멀쩡한 쪽 정보까지 같이 버려지는 게 낭비였다.

**참고한 아이디어:** 사용자 제안 — "ll 추출은 딥러닝, 그 이후는 슬라이딩 윈도우 적용". 참고 프로젝트
(github.com/junhyukch7/Advanced-Lane-Detection)의 `slidingWindow()`는 좌/우를 아예 별개로 두 번
호출해서, 각 라인의 창이 서로 무관하게 자기 페이스로 다음 위치를 찾는다 — 정확히 이 "좌/우 독립"
구조가 지금 코드에 없던 부분이었다.

**수정:** `_ll_slice_centers()`가 좌/우 창(`cur_left`/`cur_right`)을 완전히 독립적으로 갱신하도록
바꿨다([perception/dl_lane.py](perception/dl_lane.py) `_ll_slice_centers()`) — 왼쪽 창은 왼쪽에서
찾았을 때만, 오른쪽 창은 오른쪽에서 찾았을 때만 갱신한다. 밴드별 중심점 결정도 세 갈래로 늘렸다:
① 양쪽 다 찾고 폭이 정상 범위면 중점 채택, ② 한쪽만 찾았으면(또는 양쪽 다 찾았지만 폭이 비정상이라
서로 못 믿을 때) 찾은 쪽에서 러닝 차로 반폭 추정치(`self._ll_half_width`, 신규 `DL_LL_WIDTH_EMA_ALPHA`
로 EMA 갱신, [config.py](config.py))만큼 반대쪽으로 밀어 추정, ③ 양쪽 다 못 찾으면 무효. ②는
`lane_util.SlideWindow.calc_center()`가 classic_cv 백엔드에서 이미 쓰던 "한쪽 차선만 검출" 폴백과
같은 원칙을 ll에도 적용한 것이다.

**알려진 한계:** 편측 폴백이 여러 밴드 연속으로 이어지면 `self._ll_half_width`가 그 사이 갱신되지
않아(양쪽 다 찾은 밴드에서만 갱신) 오래된 추정치를 계속 쓰게 된다 — 실차 미검증, 편측 검출이 긴
구간에서 추정 중심이 실제와 얼마나 벌어지는지 `DEBUG_VIZ_DL_LANE`으로 확인할 것.

### 2.14 da가 옆 차선과 완전히 병합될 때의 방어 — ll 잔상 + 가상경계 (`_clip_da_by_ll`, 2026-08-07)

**증상:** 실차 캡처(`ll_cov:0.022`, `ll_bands:0/8` — ll이 프레임 전체에서 거의 안 보임)로 da가 옆
차선과 완전히 하나로 병합된 사례를 확인했다. 처음엔 "이어붙은 지점이 얇은 다리일 것"이라 침식
(erosion)으로 끊는 방법을 검토했는데, 실제 캡처를 보니 이어붙은 부분이 얇지 않고 뭉텅하게 넓었다 —
da 모델 자체가 그 프레임에서 두 차선을 시각적으로 구분하지 못하고 통째로 하나의 주행가능영역으로
내놓은 것이었다. 침식으로 끊을 구조(얇은 다리) 자체가 마스크 안에 없어 그 방향은 포기했다.

**참고한 아이디어:** 사용자 제안 — ① 시계열 메모리(Temporal Decay): ll이 순간적으로 끊겨도 이전
프레임의 확실했던 픽셀을 감쇠 가중치로 유지해 클리핑 근거로 계속 쓴다. ② 기하학적 추론(가상 차선
투영): 증거가 전혀 없을 때 알려진 차로폭 기준으로 강제 경계를 가정한다.

**수정:**
- **① ll 잔상(decay)** — `detect()`가 매 프레임 `self._ll_decay_mask`(float32)를
  `max(이번 프레임 ll_mask, 직전 잔상 × DL_LL_DECAY_ALPHA)`로 갱신하고, `DL_LL_DECAY_MIN_VALUE`
  이상인 픽셀만 `ll_mask_for_clip`로 이진화해 `_clip_da_by_ll()`에 넘긴다(둘 다 신규
  [config.py](config.py)). `DL_LL_DECAY_ALPHA=0.8`이면 대략 3~4프레임 뒤 자연 소멸한다. **centerline
  추출(`_ll_slice_centers`)에는 이 잔상을 안 쓴다** — waypoint 자체를 과거 위치로 밀면 더 위험하고,
  클리핑은 "울타리" 역할이라 약간 stale해도 안전하다는 판단.
- **② 가상경계** — `_clip_da_by_ll()`의 밴드별 루프에서 잔상마저 없는 밴드(`ll_cols`가 완전히 빔)는
  기대 차로 반폭(`self._ll_half_width` — §2.13에서 추가한 ll 슬라이딩 윈도우의 러닝 추정치를 재사용)
  만큼 `cur_ref` 양옆을 증거 없이 강제로 자른다. `cur_ref`는 이 가상경계로는 갱신하지 않는다(실측
  없는 추정을 다음 밴드로 누적시키지 않기 위해).
- 어느 쪽이 발동했는지 `self.da_ll_virtual_clip_used`(가상경계 발동 시 `[LL_VIRTUAL]` 태그)로
  `visualize()`에 표시한다([perception/dl_lane.py](perception/dl_lane.py)).

**알려진 한계:**
- 둘 다 실차 미검증 초기값. `DL_LL_DECAY_ALPHA`가 너무 높으면(오래 남음) 실제 경계 이동(커브 등)을
  못 따라가는 잔상을 계속 쓰게 되고, 너무 낮으면 잔상 효과가 거의 없어진다.
- 가상경계는 `self._ll_half_width`가 'da' 모드에서는 절대 갱신되지 않는다는 점을 유의할 것 —
  `_ll_slice_centers()`(EMA 갱신 주체)가 `ll_da`/`ll` 모드에서만 호출되므로, `DL_CENTER_MODE='da'`로
  주행 중이면 이 값은 항상 config.py의 초기 상수(`(DL_LL_WIDTH_MIN_PX+MAX_PX)/4`)로 고정된다 —
  실제 차로폭과 차이가 크면 가상경계도 그만큼 부정확해진다.
- 가상경계가 여러 밴드 연속으로 계속 발동하면(장시간 ll 블랙아웃) 결국 `_clip_da_by_ll()` 자체가
  아니라 그 상위의 `da_ll_clip_skipped`(유효 밴드 부족 시 클리핑 전체 되돌림, §2.1 인접 주석)
  안전판이 개입할 수 있다 — 이 경우 가상경계 자체는 무력화되고 이전과 동일하게 무클리핑 da로
  돌아간다는 것도 감안할 것.

### 2.15 ll 이진화 임계값 인하 — 0.7 → 0.5 (`DL_LL_FG_THRESHOLD`, 2026-08-07)

**증상:** §2.14 작업 중 실차 영상(변경 전 촬영분 5개, `dl_lane` 디버그 창)을 프레임 샘플링해서
훑어보니, `ll_bands:0/8`/`ll_cov` 0.03 미만이 거의 전 구간에서 관찰됐다 — "가끔 순간적으로 끊긴다"는
전제(§2.14의 잔상/가상경계 설계 근거)와 달리 **애초에 ll이 거의 항상 안 잡히는 상태**였다. §2.14의
잔상(decay)은 "평소엔 잘 잡히다 잠깐 끊기는" 상황을 전제로 설계했는데, 이 정도로 ll이 상시 약하면
잔상이 쌓일 실측 자체가 부족해 가상경계가 예외적 최후수단이 아니라 사실상 주력으로 작동하게 될
위험이 있었다.

**원인 추정:** `DL_LL_FG_THRESHOLD=0.7`(da의 `DL_FG_THRESHOLD=0.5`보다 높음, §2.2의 원거리 blur
대응 목적으로 올려둔 값)이 blur 방지 범위를 넘어 **정상 신뢰도(0.5~0.7)의 실제 차선 픽셀까지 통째로
걸러내고 있었을 가능성**. 모델은 차선을 약하게라도(0.5대 확률로) 맞히고 있는데 0.7 문턱을 못 넘어
전부 무효 처리됐다는 뜻.

**수정:** `DL_LL_FG_THRESHOLD`를 0.7 → 0.5(`DL_FG_THRESHOLD`와 동일)로 낮췄다([config.py](config.py)).
구조는 그대로— 단순 값 조정.

**알려진 한계:** 이 값을 낮추면 §2.2에서 다뤘던 "원거리 ll이 blur로 두껍게 잡히는" 문제가 다시 나타날
수 있다 — `_clip_da_by_ll()`이 그만큼 da를 과하게 깎아낼 위험. 실차 재검증 전 —
`DEBUG_VIZ_DL_LANE`에서 `ll_cov`가 정상 범위로 올라오는지, 원거리 ll 두께가 다시 과해지진 않는지
같이 확인할 것. 이걸로도 흔들림이 안 잡히면 **이진화 자체를 없애고 확률값 가중치로 계산하는 방식**을
검토 중 — 자세한 설계는 [ll_probability_weighting_proposal.md](ll_probability_weighting_proposal.md)
참고(아직 미구현, 제안 단계).

### 2.16 da 롤백(경계중점→무게중심, 면적상한 제거) + `ll_da`/`ll` 모드를 서로 다른 알고리즘으로 교체 (2026-08-10)

**증상:** §2.12에서 도입한 da 경계 중점(`_slice_edge_midpoints`)으로 실차 주행해보니 "S자로 좌우 왔다갔다"
하는 심한 흔들림이 발생 — 가장자리 노이즈 픽셀 하나에 경계 전체가 밀리는 게 원인으로 추정(§2.12
"알려진 한계"에 이미 이 위험을 적어뒀었음). 또한 §2.11에서 추가한 시드/연속성 기반 선택에서, 실차
검증 결과 "면적 임계값만으로 da를 고르는 방식" 자체가 파라미터를 계속 바꿔도 개선되지 않는다는 게
재확인됨(CLAUDE.md에 이미 기록된 교훈).

**수정 — 'da' 모드:**
- 밴드 중심을 `_slice_edge_midpoints()`(경계 중점) → `_slice_centers()`(무게중심)로 롤백.
- `_largest_da_component()`에서 면적 **상한**(`DL_DA_MAX_AREA_PX`) 체크를 완전히 제거 — 하한
  (`DL_DA_MIN_COMPONENT_AREA`, "사실상 안 보임" 노이즈 필터)만 유지. ①시드 → ②연속성 → ③면적순위
  우선순위 자체는 그대로 유지하되, 세 단계 모두 이제 하한만 통과하면 채택한다. **대가**: da가 옆
  차선과 실제로 붙어도 이 함수는 더 이상 막지 않는다 — 그 방어는 전적으로 `_clip_da_by_ll()`(ll
  잔상+가상경계, §2.14)이 담당하는 구조로 역할이 옮겨갔다.

**수정 — `ll_da`/`ll` 모드를 완전히 다른 소스의 알고리즘으로 교체:**
세 모드가 이제 서로 알고리즘 자체가 다르다(공유 코드는 da 파편화 대응/클리핑뿐이고 그마저
`ll_da`=corridor는 건너뜀).
- **`ll_da` → "corridor" 알고리즘** (팀원 이지유 작성, 원본 커밋 `991f91e`): ll로 도로 폭 자체를
  규정한다 — 밴드마다 ll을 왼쪽부터 정렬해 1번째~3번째 선(2번째=중앙분리선, 그냥 지나침)을 도로
  경계(전체 트랙, 양쪽 차로 폭)로 삼고, 그 범위 안에서만 da를 봐서 실제 열린(장애물 없는) 구간을
  찾아 중심으로 쓴다(`_ll_line_centers()`/`_pick_open_run()`/`_corridor_slice_centers()`). 장애물이
  한쪽을 막으면 자연히 반대쪽 열린 구간으로 경로가 붙어 회피를 겸한다 — 직전 프레임 위치에 가장
  가까운 열린 구간을 우선하는 히스테리시스로 flip-flop을 막는다. "자기 차선 하나"를 전제로 한
  `_largest_da_component()`/`_clip_da_by_ll()`은 건너뛰고 클리핑 전 원본 da를 그대로 쓴다. 신규
  튜닝값(전부 실차 미검증): `DL_CORRIDOR_LINE_MIN_PIXELS`(12), `DL_CORRIDOR_LINE_MERGE_PX`(15),
  `DL_CORRIDOR_WIDTH_MIN/MAX_PX`(190~450), `DL_CORRIDOR_MIN_PASSABLE_PX`(80, 차량 실폭 기반).
- **`ll` → 흰선/노란선 분리** (팀원 yunyunsung 작성, 원본 커밋 `d586aff`): ll을 커넥티드 컴포넌트
  단위로 HSV 노란색 겹침 비율 투표해 흰선/노란선으로 분리하고(`_split_ll_by_yellow()`, 픽셀 단위로
  빼는 것보다 dash 가장자리가 깔끔함), 좌/우 슬라이딩 윈도우(`_ll_slice_centers()`)는 흰선 마스크만
  본다 — 중앙 노란 점선이 좌/우 트래킹에 안 섞인다. 노란선은 아직 상태 없는(stateless) 밴드별
  무게중심만 디버그 표시용으로 뽑아둔다(추후 "도로 중앙" 힌트로 확장 예정). 신규 튜닝값:
  `DL_LL_YELLOW_VOTE_RATIO`(0.35), `DL_LL_YELLOW_MIN_AREA`(10). 겸사겸사
  `DL_LL_WIDTH_MIN/MAX_PX`도 100~220 → 50~200으로 넓힘(녹화 영상 실측 라인 간격이 75~80px로
  기존 하한보다 작게 나왔음).
- 디버그 시각화도 갈아엎었다 — ll이 흰/노랑 실제 색으로 표시되고, `ll` 모드에선 좌/우 탐색창 +
  밴드별 실측 차로폭(범위 통과 여부 색상 구분), `ll_da`(corridor)에선 corridor 경계(1/3번째 선)
  자홍색 틱, `dl_lane` 창에 노란선 전용 패널이 4번째로 추가됐다. 하단 텍스트도 모드별로
  `corridor_bands:`/`white_bands:`+`yellow_bands:`+`lane_w_est:`로 달라진다.

**알려진 한계:** `ll_da`/`ll` 둘 다 원래 서로 다른 브랜치에서 독립적으로 개발된 코드를 오늘 세션의
da 파이프라인(시드 선택/ll 잔상+가상경계) 위에 이식한 것이라, **아직 실차로 전혀 검증 안 됨** —
튜닝값 전부 초기 추정치. `ll_da`(corridor)는 `DL_CORRIDOR_WIDTH_MIN/MAX_PX`가 실제 트랙 폭(양쪽
차로)과 맞는지부터 확인해야 하고, `ll`은 §2.15에서 낮춘 `DL_LL_FG_THRESHOLD=0.5`와 조합했을 때
`DL_LL_YELLOW_VOTE_RATIO`가 여전히 적절한지 재확인이 필요하다. [2026-08-10] 이후 §2.17에서 main
기본값을 'll'로 전환했다 — 아래 참고.

### 2.17 `ll` 모드 재설계 — "좌/우 흰선 독립 추적" → "노란 중앙선 + 한쪽 흰선" (2026-08-10)

**증상:** §2.16의 `ll` 모드(좌/우 흰선 두 개를 독립 슬라이딩 윈도우로 추적)를 실차에서 돌려보니
`white_bands`가 영상 전체에서 계속 0~1/8이었다. 원인은 구조적이었다 — 실제 도로는 편도 1차로 기준
흰(왼쪽 경계)-노(중앙 분리선)-흰(오른쪽 경계) 구성인데, 노란선은 `_split_ll_by_yellow()`가 이미
흰선 마스크에서 제외해두므로, 차량이 지금 있는 차선 기준으로 노란선이 있는 쪽엔 애초에 "흰선"이
탐색될 수가 없다. 좌/우 흰선이 "둘 다" 잡히길 기다리는 §2.16 모델은 이 구조에서 거의 항상 실패할
수밖에 없었다.

**수정:** 사용자 설계 지시(4개 규칙)를 반영해 `DLSlideWindow._ll_yellow_white_centers()`로 완전히
재설계했다(옛 `_ll_slice_centers()`는 삭제):

1. **차선 판정(`self.lane_side`)**: 근거리 밴드에서 처음 찾은 노란선이 seed(차량 위치, ROI 중앙)
   기준 왼쪽에 있으면 "우측차선 주행중"(흰 경계선을 오른쪽에서 탐색), 오른쪽에 있으면 "좌측차선
   주행중"(왼쪽에서 탐색).
2. 밴드마다 노란선/흰선을 각각 좁은 창(`DL_LL_SEARCH_HALF_WIDTH_PX`)으로 독립 탐색. **둘 다 찾으면**
   중점 채택 + 간격(`self._white_yellow_gap_px`, `DL_LL_YELLOW_GAP_EMA_ALPHA`로 EMA) 갱신.
3. **노란선만** 찾으면 → 간격만큼 흰선 위치를 추정해서 중점 계산(저신뢰).
4. **흰선만** 찾으면 → 간격만큼 노란선 위치를 역으로 추정(3번의 대칭, 저신뢰).
5. **둘 다** 못 찾으면 → 직전까지 추적하던 위치를 그대로 "잔상"으로 써서 중점을 만든다(저신뢰).

3~5번(저신뢰 추정)은 `self.ll_band_degraded`/`self.ll_degraded`로 표시되고, `track_drive.py`
`_lane_drive()`가 이번 프레임에 하나라도 있으면 속도를 신규 `SPEED_LL_DEGRADED`(5.0)로 강제
제한한다(가/감속 모두 즉시 적용, 코너 감속과 같은 관례). `_debug_viz_steer()`(`steer_debug` 창)에
`LL 차선:{lane_side} {정상/저신뢰}` 줄이 추가로 뜬다. `_clip_da_by_ll()`의 가상경계도 옛
`self._ll_half_width` 대신 `self._white_yellow_gap_px`를 재사용하도록 같이 바꿨다(같은 "차로
반폭류" 물리량 대체).

이제 안 쓰는 `DL_LL_WIDTH_MIN_PX`/`MAX_PX`/`DL_LL_WIDTH_EMA_ALPHA`는 제거하고
`DL_LL_YELLOW_GAP_INIT_PX`/`DL_LL_YELLOW_GAP_EMA_ALPHA`로 교체했다.

**추가 반영(같은 요청):** main 기본값을 `DL_CENTER_MODE='da'` → **`'ll'`**로, `SPEED_NORMAL`을
25.0 → **15.0**으로 낮췄다 — 재설계된 `'ll'`을 실차에서 검증하기 위한 전환이라, 문제가 생기면
`DL_CENTER_MODE`를 `'da'`로 되돌릴 것.

**알려진 한계:**
- 전부 실차 미검증 초기값(`DL_LL_YELLOW_GAP_INIT_PX=80`, `SPEED_LL_DEGRADED=5.0` 등).
- ⑤(잔상)가 여러 프레임 연속되면 `cur_yellow`/`cur_white`가 실제 위치와 점점 벌어질 수 있다 —
  몇 프레임까지 안전한지 확인 필요.
- `lane_side` 오판(교차로 등에서 노란선을 반대쪽 것으로 잘못 짝지음) 시 흰선 탐색 방향 자체가
  틀어진다 — 아직 별도 sanity check 없음.
- 노란선이 아예 없는 트랙(중앙분리선 없이 흰선만 있는 구간 등)에서는 이 모델 자체가 성립하지
  않는다 — 그런 구간이 있다면 `'da'`로 되돌리거나 별도 처리가 필요.

---

### 2.18 gap EMA 폭주 방지 + "노란선 없음" 3분기 재설계 + 밴드별 분기 시각화 (2026-08-10)

**증상 (실차 영상 `08_10 오후 1.03.02.mov` 분석, §2.17 배포 직후):** `mode:ll`로 주행한 영상에서
15초 지점에 조향각이 급격히 +59~87도까지 튀며 우회전하는 게 확인됐다. 프레임을 0.5초 단위로
뜯어보니 `offset`이 13~15.5초 사이 +25.3 → +47.4 → +87.4로 계속 커지는데, 같은 구간 내내
`ok_bands:0/8`(정상 검출 밴드 0개) — 즉 실제 신뢰할 근거 없이 waypoint가 계속 같은 방향으로
밀려나고 있었다. `DEBUG_VIZ_DL_LANE` 상 흰-노 간격 표시값이 **161px**였는데, 같은 날 실측한
정상값은 **80px**(흰-노 간격 실측 0.4m × `DL_PIXELS_PER_METER` 200px/m) — `self._white_yellow_gap_px`
EMA가 실제값의 2배 가까이 부풀어 있었다.

**원인:** `self._white_yellow_gap_px`는 노란/흰 둘 다 찾은 밴드에서만 EMA로 갱신되는데(§2.17 2번),
노란선이 완전히 안 잡히기 시작하기 직전 몇 프레임에서 노이즈(글레어 등 추정)로 큰 `|흰선-노란선|`
값이 섞여 EMA가 161px까지 부풀었고, 그 직후 노란선이 아예 안 잡히면서 **갱신 자체가 멈춰 부푼 값이
그대로 얼어붙었다**. 옛 ④번 분기("흰선만 찾음 → 간격으로 노란선 역추정")는 좁은 창
(`cur_white=ref_x+side_sign*gap`) 하나로만 흰선을 찾았는데, 부푼 gap 때문에 이 창이 실제 흰선보다
훨씬 바깥쪽에 위치해 흰선을 놓쳤고(`ll` 원본 마스크엔 흰선이 뚜렷이 보이는데도), 결국 ⑤번(잔상)
분기로 떨어져 `ref_x + gap/2`(≈실제 흰선을 넘어선 위치)를 그대로 waypoint로 썼다.

**수정 (`DLSlideWindow._ll_yellow_white_centers()`, `dl_lane.py`):**

1. **gap EMA 클램프**: `DL_LL_YELLOW_GAP_MIN_PX`(50) / `MAX_PX`(110) — 실측값(80px) 근방으로
   상하한을 걸어 노이즈가 껴도 이 이상 부풀지 못하게 막았다.
2. **"노란선 없음" 분기를 3분기로 재설계**(사용자 지시): 예전 ④/⑤(좁은 창 하나로 흰선 탐색 →
   간격 역적용/잔상)를 버리고, `DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX`(150) 넓은 창에서
   `_ll_line_centers()`(corridor 모드가 쓰던 다중 선 검출 재사용)로 흰선 컴포넌트를 전부 찾아
   개수로 분기한다:
   - **케이스1(태그 `2W`, 흰선 2개 이상)**: 가장 왼쪽/오른쪽 두 실측 위치의 중점을 그대로
     중앙선으로 채택 — 간격 추정치에 안 기대는 가장 신뢰 높은 재구성.
   - **케이스2(태그 `1W:L`/`1W:R`, 흰선 1개)**: 기준점(`cur_yellow`) 대비 좌/우를 **매 밴드
     실측으로 새로 판정**(프레임당 한 번 고정되는 `self.lane_side`는 stale할 수 있어 안 씀)한 뒤,
     그 방향으로 (클램프된) gap만큼 안쪽으로 당겨 재구성.
   - **케이스3(태그 `LOST`, 흰선 0개)**: 기존과 동일하게 잔상.
3. **밴드별 분기 시각화**(요청 반영 — "SW가 어떤 걸 생각해서 주행하는지"): `visualize()`가 각
   밴드 중심점 옆에 태그(`Y+W`/`Y+gap`/`2W`/`1W:L`/`1W:R`/`LOST`)를 텍스트로 그리고,
   `dl_lane` 창 상단에 이번 프레임 전체 분기 개수 요약(`branch: 2W:3 LOST:5` 형태)을 한 줄
   추가로 찍는다(`self.ll_band_case`, 길이 `n_slices`).

**알려진 한계:**
- `DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX=150`/클램프 범위(50~110)는 실측 40cm(80px) 기반
  추정이지 직접 튜닝된 값은 아니다 — 실차에서 `branch:` 요약 줄이 `2W`/`1W` 위주로 정상 분포하는지
  보고 재조정할 것.
- 케이스1에서 넓은 창 안에 흰선이 3개 이상 잡히면(점선 파편/노이즈/옆 차선) 가장 왼쪽/오른쪽만
  채택하는데, 그중 하나가 실제로는 무관한 물체일 경우 중점이 틀어질 수 있다 — 아직 폭
  sanity check 없음.
- 옛 ③번(노란선만 찾고 흰선 실패)은 이번에 손 안 댐 — 여전히 `self.lane_side`/`side_sign`
  (프레임당 한 번 고정)에 기대므로, lane_side 오판 시 이 분기만 따로 틀어질 수 있다.
- 47초 지점(사용자가 "노란/흰 둘 다 보이는데 왼쪽에 찍힘"으로 지목한 프레임)은 재확인 결과
  `ll_cov:0.000`으로 `ll`/`yellow` 원본 마스크가 완전히 비어있었다(그 구간 raw 카메라에 심한
  글레어/노출과다 확인) — 즉 실제로는 "둘 다 검출된" 상황이 아니라 완전 미검출 상태의 잔상
  (케이스3/⑤)이었다. `mode:ll`이 세그멘테이션 자체가 실패하는 강한 역광/글레어 조건에는 아직
  전혀 대응하지 못한다는 뜻 — 별도 과제로 남겨둠(예: 노출 보정, 시간적 잔상 신뢰도 감쇠 등).

### 2.19 두 `ll` 알고리즘을 `origin/main` 병합 시 둘 다 살리고 `DL_LL_ALGO` 스위치로 공존 (2026-08-10)

**배경:** 팀원(mastic-choi)이 `origin/main`에 §2.17(`_ll_yellow_white_centers()`, 노란+흰선
짝짓기)/§2.18(gap EMA 클램프+3분기 재설계) 커밋을 올린 것과 거의 같은 시간에, 다른 브랜치
(`이지유`)에서 아래 §2.20/§2.21(당시엔 §2.17/§2.18로 잘못 번호가 겹쳤었음 — 이 절에서 바로잡음)
작업으로 원래의 `_ll_slice_centers()`(좌/우 흰선 독립 슬라이딩 윈도우)를 강화하고 있었다. 두
작업 모두 `DL_CENTER_MODE='ll'`의 밴드 중심 계산 로직 자체를 건드려서, `git merge origin/main`
시 `dl_lane.py`에 충돌 9곳이 났다(핵심은 옛 `_ll_slice_centers()` 자리에 각자 다른 함수를
써넣은 것 — 팀원은 그 함수를 아예 `_ll_yellow_white_centers()`로 개명/재작성했고, `이지유`
브랜치는 같은 이름 그대로 확장했음). 사용자에게 해결 방향을 물어본 결과 "둘 다 살리고
mode전환으로 바꿀 수 있게" 요청받아 아래처럼 병합했다.

**수정 — `DL_LL_ALGO` 2차 스위치 도입** ([config.py](config.py) `DL_CENTER_MODE` 바로 아래):
`DL_CENTER_MODE='ll'`일 때 실제 밴드 중심 계산 알고리즘을 고르는 신규 스위치.
`'yw'`(main 기본값, 팀원 작성) = `_ll_yellow_white_centers()`, `'lr'`(이지유 작성) =
`_ll_slice_centers()`. `DL_CENTER_MODE`는 그대로 두고 이 값만 바꿔서 A/B 비교할 수 있다.

**수정 — `dl_lane.py` 충돌 해소 (함수 자체는 둘 다 원본 그대로, 주변 배선만 변경):**
- **`_ll_yellow_white_centers()`(팀원)와 `_ll_slice_centers()`(이지유)를 완전히 별개
  메서드로 나란히 유지** — 서로 코드를 섞지 않았다(각자 실차로 어느 정도 검증/튜닝된
  로직을 잘못 건드려 깨뜨릴 위험을 피하려는 목적). `detect()`의 `DL_CENTER_MODE == 'll'`
  분기 안에서 `DL_LL_ALGO`로 둘 중 하나만 호출한다 — 안 쓰는 쪽의 디버그 상태
  (`ll_band_case`/`ll_band_degraded`/`ll_degraded` vs `ll_band_reason`)는 매 프레임
  중립값으로 리셋해 `visualize()`가 지난 프레임(또는 반대 알고리즘)의 잔여 태그를
  잘못 그리지 않게 했다.
- **`self._ll_half_width`(이지유)와 `self._white_yellow_gap_px`(팀원)를 `__init__`에
  둘 다 복원** — `git merge`가 두 초기화 줄을 "같은 줄을 서로 다르게 고침"으로 보지
  않고(베이스 대비 팀원 쪽만 그 줄을 바꾸고 이지유 쪽은 그대로 둔 것으로 판단해) 조용히
  팀원 쪽만 남기고 이지유 쪽을 지워버렸다 — 병합 자체는 깨끗했지만(충돌 표시 없음)
  `_ll_slice_centers()`가 여전히 참조하는 값이라 그대로 뒀으면 `AttributeError`로
  터졌을 것. `_clip_da_by_ll()`의 가상경계 최후수단처럼 "지금 모드에서 신뢰할 수 있는
  차로 반폭"이 필요한 공용 소비처는 신규 헬퍼 `_ll_active_half_width()`
  (`DL_LL_ALGO`로 둘 중 하나를 골라 반환) 하나만 거치게 바꿨다.
- **`visualize()`의 탐색창 그리기도 `DL_LL_ALGO`로 분기** — `self.ll_search_windows`가
  두 알고리즘 다 8-tuple이지만(우연히 구조가 같음) 원소의 의미가 다르다(`lr`:
  `lx/rx`=좌/우 흰선, `yw`: `yx/wx`=노랑/흰선). `'lr'`이면 위젠 강조/밴드 앵커
  마커/채택근거 태그(B/X/L/R/-)를, `'yw'`이면 노랑·흰 창 색 구분/간격 텍스트를 그린다.
  하단 요약 텍스트도 `DL_LL_ALGO`별로 다른 지표(`lr`: `Lvel`/`Rvel`, `yw`:
  `degraded`/`gap`/`side` + 밴드 분기 요약 줄)를 보여준다 — §2.21에서 추가한 MODE
  배너가 팀원의 "branch: ..." 요약 줄과 같은 좌표(10,40)를 두고 겹쳐 그려지고 있던
  것도 이번에 발견해서 branch 요약 줄을 (10,60)으로 내렸다.
- **`_params_panel_lines()`도 `DL_LL_ALGO`로 분기** — `'lr'`이면 §2.20의
  `DL_LL_WIDTH_*`/`DL_LL_VELOCITY_*`/`DL_LL_SEARCH_WIDEN_*`/`DL_LL_BAND_ANCHOR_ALPHA`를,
  `'yw'`이면 `DL_LL_YELLOW_GAP_*`/`DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX`를 보여준다
  (병합 전엔 `'yw'` 쪽 값이 아예 패널에 없었다 — `이지유` 브랜치에서 만든 기능이라
  팀원 커밋을 몰랐던 것뿐).

**수정 — `config.py` 튜닝값 복원:** 위와 같은 이유로 `DL_LL_WIDTH_MIN_PX`/
`DL_LL_WIDTH_MAX_PX`/`DL_LL_WIDTH_EMA_ALPHA`(`'lr'` 전용)가 자동 병합 중 조용히
사라져서 복원했다. `README.md`/`config.py`/`track_drive.py`는 `dl_lane.py`와 달리
실제로 완전히 자동 병합됐다(git이 충돌로 표시하지 않음).

**알려진 한계:**
- 두 알고리즘을 각자 원본 그대로 유지하는 방향을 택해서, `'lr'`이 갖고 있던 "옆 차선
  차선인지 판단 없이 좌/우 흰선을 그냥 독립 추적한다"는 구조적 약점(§2.17에서 팀원이
  `'yw'`로 재설계한 바로 그 이유)은 `'lr'`을 선택하면 여전히 그대로 남아있다 — main
  기본값이 `'yw'`인 이유이기도 하다. `'lr'`은 §2.20/§2.21에서 실차 미검증인 채로
  이번 병합에 들어왔다.
- README 절 번호가 같은 날 두 브랜치에서 독립적으로 §2.17/§2.18을 재사용해 겹쳤던 걸
  이번에 §2.19~§2.21로 정리했다 — 혹시 다른 곳(커밋 메시지, 이슈 등)에서 옛 번호로
  이 작업을 참조했다면 어긋날 수 있다.
- `_ll_active_half_width()`처럼 "지금 모드에 맞는 값 하나만 골라주는" 패턴을 이번에
  급하게 하나 추가했는데, 앞으로 두 알고리즘이 공유해야 할 상태가 더 생기면 (예:
  둘 다 참조하는 러닝 추정치가 늘어나면) 이 방식이 계속 늘어나는 게 맞는지, 아니면
  아예 알고리즘별 상태를 작은 클래스/네임스페이스로 묶는 리팩터가 나을지 판단이
  필요하다 — 지금은 딱 하나뿐이라 굳이 안 함.

### 2.20 `ll` 슬라이딩 윈도우에 적응형 탐색창 + 밴드별 프레임 간 앵커링 추가 (`_ll_slice_centers`, 2026-08-10)

**배경:** 앞으로 `ll` 기반 차선인식으로 완전히 갈아엎을 예정이라, 그 기반이 될
`_ll_slice_centers()`(§2.13)를 먼저 강화했다. §2.13 docstring에 이미 적혀있던 "알려진 한계" 두
가지 — ① 탐색창이 좁은 고정폭(`DL_LL_SEARCH_HALF_WIDTH_PX`=60px)이라 급커브에서 밴드 간 실제 선
이동량이 그보다 크면 창이 선을 놓치고 이후 밴드까지 이전 위치에 멈춰 선다, ② band 0(근거리)만 직전
프레임 확정 `lane_center`에 앵커링되고 그 위 밴드는 전부 이번 프레임 안에서만 전파되어 band 0의
오차가 위 밴드로 누적 전파된다 — 를 대응했다.

**수정:**
- **① 적응형 탐색창** — 두 갈래로 완화.
  - *속도 예측*: 그 사이드(좌/우 독립)에서 실제로 찾은 밴드들 사이의 x 이동량을 밴드 간격으로 나눈
    px/밴드 값을 `self._ll_left_velocity`/`_right_velocity`로 EMA 추적(신규
    `DL_LL_VELOCITY_EMA_ALPHA`=0.3, [config.py](config.py))한다. 다음 밴드 탐색창 중심을 "마지막으로
    찾은 위치"가 아니라 "그 위치 + 예측 이동량"으로 미리 옮긴다 — 미검출 밴드가 이어지는 동안에도
    이 속도로 계속 dead-reckoning 이동시켜 창이 멈춰 서 있지 않게 한다. 노이즈로 속도 추정이 튀는
    걸 막기 위해 `DL_LL_VELOCITY_MAX_PX`(40px)로 클램프한다.
  - *탐색창 확장*: 그 사이드가 연속으로 못 찾을 때마다 탐색창 반경을
    `DL_LL_SEARCH_WIDEN_STEP_PX`(15px)씩 넓혀(`DL_LL_SEARCH_WIDEN_MAX_PX`=120px 상한) 재포착
    기회를 늘리고, 다시 찾으면 기본 반경(60px)으로 리셋한다.
- **② 밴드별 프레임 간 앵커링** — `self._ll_prev_band_left`/`_prev_band_right`(길이 n_slices)에
  밴드마다 "직전 프레임에 그 밴드(같은 y위치)에서 실제로 찾은 위치"를 따로 기억해뒀다가, 이번
  프레임 그 밴드의 탐색창 중심을 (이번 프레임 내 전파값, 직전 프레임 그 밴드 값)의 가중평균
  (신규 `DL_LL_BAND_ANCHOR_ALPHA`=0.35)으로 잡는다 — 도로 곡률이 프레임 간 급격히 안 변한다는
  가정에 기대어 band 0의 오차가 위로 그대로 번지지 않게 한다. 밴드값은 실제로 찾았을 때만 갱신하고
  못 찾은 프레임엔 이전 값을 그대로 들고 있는다(`self._ll_half_width`와 동일한 관례).
- 속도 EMA(`self._ll_left/right_velocity`)는 프레임 간 영속하지만, "밴드 간 간격" 계산에 쓰는
  마지막 검출 밴드 인덱스/위치(`last_left_i`/`last_left_x` 등)는 매 호출(=매 프레임)마다 지역변수로
  새로 시작한다 — 프레임 경계를 넘어 간격을 계산하면 밴드 인덱스가 롤오버돼 음수 gap이 나오기
  때문.
- `self.ll_search_windows`(디버그 시각화 튜플 구조)는 그대로 유지해 `show_debug_windows()` 쪽
  변경은 없다.

**알려진 한계:** 전부 실차 미검증 초기값. 급커브에서 창이 실제로 선을 놓치지 않고 따라가는지,
확장된 창이 오히려 옆 차선/반사광을 잘못 무는지, 밴드별 앵커링이 급조향/저프레임레이트 상황에서
오히려 과거 위치로 창을 잘못 당기지는 않는지 확인 필요 — `DL_CENTER_MODE='ll'`로 전환해 A/B 비교할
것. §2.13 "알려진 한계"에 있던 편측 폴백 시 `self._ll_half_width` 미갱신 문제는 이번 수정 범위 밖
(아직 미해결).

### 2.21 튜닝 파라미터 패널(`dl_lane_params` 창) + 슬라이딩 윈도우 디버그 보강 (2026-08-10)

**배경:** §2.20 튜닝값을 실차에서 바꿔가며 확인하려면 매번 config.py를 열어 지금 이 모드에서
실제로 쓰이는 값이 뭔지 찾아야 했다 — 실차 옆에서 노트북으로 여러 파일을 오가는 건 번거롭다.
"내가 수치를 바꿔가며 성능을 개선할 수 있는 파라미터와 그 수치"를 디버그 창에서 바로 보고 싶다는
요청.

**수정:**
- **`dl_lane_params` 창 신규 추가** — `DLSlideWindow._params_panel_lines()`가 지금
  `DL_CENTER_MODE`에서 실제로 영향을 주는 값만 골라 "이름=현재값" 텍스트로 뽑고
  (`_build_params_panel()`이 렌더링), `DLLaneDetector.show_debug_windows()`가 기존 `dl_lane`
  창과 별개로 띄운다. 공용값(모드 무관, `DL_N_SLICES`/`DL_FG_THRESHOLD`/`DL_LL_FG_THRESHOLD`/
  `DL_DA_MIN_COMPONENT_AREA`/`DL_SLICE_FIT_MIN`/`DL_SLICE_OUTLIER_MAX`/`DL_STABLE_FRAME_MIN`/
  `DL_STABLE_JUMP_MAX`/`DL_LL_SANITY_MIN_RATIO`) → da/ll 클리핑값('da'/'ll' 공통,
  `DL_LL_CLIP_MARGIN_PX`/`DL_LL_DECAY_ALPHA`/`DL_LL_DECAY_MIN_VALUE`/`DL_DA_SEED_ROWS_PX`/
  `DL_DA_SEED_HALF_WIDTH_PX`) → 모드 전용값(`ll`이면 §2.13/§2.20의 `DL_LL_*` 전부,
  `ll_da`면 `DL_CORRIDOR_*`) → 러닝 추정치(참고용, `self._ll_half_width*2`/`_ll_left_velocity`/
  `_ll_right_velocity`) 순으로 좁혀서 보여준다 — config.py 전체를 다 보여주면 지금 안 쓰이는
  값까지 섞여 오히려 헷갈리기 때문. `dl_lane`의 result/da/ll/yellow 패널과 같은 `vconcat`으로
  합치지 않고 별도 창(고정폭 420px)으로 뒀다 — 텍스트 패널까지 마스크 패널 폭(ROI가 좁은
  트랙에선 수백 px 미만일 수 있음)에 맞추면 글자가 잘리기 쉬워서다.
- **슬라이딩 윈도우 디버그 보강**(`DLSlideWindow.visualize()`, `DL_CENTER_MODE='ll'` 전용) —
  §2.20에서 추가한 새 상태를 시각적으로 확인할 수 있게 세 가지를 더 그린다:
  ① 탐색창이 기본 반경(`DL_LL_SEARCH_HALF_WIDTH_PX`)보다 넓어진 밴드(연속 미검출로
  `DL_LL_SEARCH_WIDEN_STEP_PX`만큼 확장된 상태)는 사각형 테두리를 주황으로 강조.
  ② 밴드별 프레임 간 앵커링(`DL_LL_BAND_ANCHOR_ALPHA`)이 이번 프레임에 실제로 끌어당긴
  "직전 프레임 그 밴드 위치"를 마젠타 사각 마커로 표시 — 이 점이 실제 검출 위치와 많이
  벌어지면 앵커링이 창을 엉뚱한 쪽으로 당기고 있다는 신호. ③ result 패널 하단 텍스트에
  `Lvel`/`Rvel`(좌/우 속도 예측 EMA, px/밴드)을 추가 — 값이 `DL_LL_VELOCITY_MAX_PX` 근처에
  계속 붙어있으면 클램프가 실제 곡률을 못 따라간다는 뜻.

**후속 수정(같은 날, 사용자 피드백 반영):**
- **밴드별 채택 근거 태그** (`self.ll_band_reason`, `_ll_slice_centers()`가 채움) —
  `DL_CENTER_MODE='ll'`에서 밴드마다 왜 그 결과가 나왔는지를 왼쪽 탐색창 옆에 한 글자로
  찍는다: `B`=양쪽 검출+채택(초록), `X`=양쪽 다 검출됐지만 폭이 `DL_LL_WIDTH_MIN~MAX_PX`
  밖이라 거부(빨강), `L`/`R`=편측만 검출해 `self._ll_half_width`로 반대쪽 추정(청록),
  `-`=양쪽 다 못 찾음(회색). 기존엔 사각형 색(초록/회색 테두리)만으로 "찾았는지"는
  보였지만 "왜 이 색인지"(특히 X — 양쪽 다 찾았는데 거부된 경우)는 구분이 안 됐다.
- **da/ll 클리핑 밴드별 틱** (`self.da_clip_band_virtual`, `_clip_da_by_ll()`가 채움) —
  §2.14에서 추가한 ①실측/잔상 클리핑과 ②가상경계(증거 없이 `self._ll_half_width`로 강제
  클리핑) 중 이번 프레임에 어느 밴드가 어느 쪽이었는지를 화면 왼쪽 끝 세로 띠에 초록(①)/
  주황(②) 틱으로 표시한다. 기존 `[LL_VIRTUAL]` 태그는 "이번 프레임에 한 번이라도 가상경계가
  발동했는지"만 알려줘서 정확히 몇 번째(=어느 높이) 밴드인지는 알 수 없었다. 클리핑 자체가
  통째로 버려진 프레임(`da_ll_clip_skipped`, `[LL_CLIP_SKIP]`)에는 `detect()`가 이 리스트를
  전부 `None`으로 비워서 틱이 안 그려지게 했다 — 실제로 적용 안 된 클리핑 시도 결과를
  보여주면 오해를 살 수 있어서다.
- **offset 스파크라인** (`DLSlideWindow._build_offset_sparkline()`) — 최근
  `DL_DEBUG_HISTORY_LEN`(신규 config.py, 기본 90프레임) 프레임의 디바운스 이후 최종
  offset을 `self._offset_history`(`deque(maxlen=...)`)에 쌓아 `dl_lane_params` 창 하단에
  선 그래프로 이어붙인다. §2.12 "S자로 좌우 왔다갔다" 같은 프레임 간 흔들림은 순간값
  텍스트만으론 "지금 떨고 있다"를 알아채기 어려운데, 최근 추세를 그래프로 보면 진폭이
  바로 보인다. y축 스케일은 고정하지 않고 창(window) 안 `|offset|` 최댓값에 맞춰 자동
  조정(우측 하단 `max|.|` 텍스트로 지금 스케일이 몇 px인지 항상 같이 표시) — 고정
  스케일이면 조용한 구간에서 그래프가 납작해져 미세한 흔들림을 놓치기 쉬워서다.
- **모드 배너** — `dl_lane`의 result 패널 맨 위와 `dl_lane_params` 맨 위에 지금
  `DL_CENTER_MODE`를 색 배너("MODE: DA"/"MODE: LL"/"MODE: LL_DA")로 크게 표시한다.
  모듈 상단 `DL_MODE_COLORS`(da=파랑/ll=초록/ll_da=자홍) 딕셔너리를 두 곳이 공유해서 색이
  항상 일치한다 — 기존엔 하단 텍스트 줄 안에 `mode:xx`로만 섞여 있어 다른 정보 사이에서
  놓치기 쉬웠고, 앞으로 `da`/`ll`/`ll_da`를 실차에서 계속 바꿔가며 A/B 테스트할 예정이라
  지금 뭘 보고 있는지 착각하면 튜닝값을 엉뚱한 모드에 반영하는 사고로 이어질 수 있다는
  우려에서 추가했다. `visualize()`에서는 다른 모든 오버레이보다 나중에(맨 위에) 그려서
  절대 가려지지 않게 했다.

**알려진 한계:** 파라미터 패널은 현재 프레임 기준 스냅샷이라, config.py를 고치고 노드를
재시작하기 전까지는 반영되지 않는다(런타임 hot-reload 아님) — 원래 이 저장소 파라미터들이 다
그렇다(모듈 로드 시 상수로 import). 마젠타 앵커 마커는 밴드별 위치라 화면이 복잡한 트랙에서는
사각형/텍스트와 겹쳐 잘 안 보일 수 있다 — 실차 확인 필요. offset 스파크라인의 y축 자동
스케일은 순간적인 이상치(outlier) 한 프레임 때문에 나머지 구간이 납작해 보이게 만들 수 있다
— 실차 확인 후 필요하면 고정 스케일이나 percentile 클램프로 바꿀 것.

**[2026-08-11 후속] `dl_lane_params` 창 자체는 삭제했습니다.** 실차 테스트 중 매 프레임 실제로
봐야 하는 건 offset 스파크라인(차선이 흔들리고 있는가)뿐이고, 나머지 텍스트 목록은 대부분 config
고정값(일부 러닝 추정치 포함)이라 코드/config.py를 보면 알 수 있는데 화면만 차지한다는 판단이었습니다.
`DLSlideWindow._params_panel_lines()`/`_build_params_panel()`을 지우고, `_build_offset_sparkline()`
결과만 `dl_lane` 창(result/da/ll/yellow) 맨 아래에 같은 폭(`self.roi_w`)으로 이어붙이도록
`show_debug_windows()`를 바꿨습니다 — 창 하나가 줄어서 `DEBUG_VIZ_DL_LANE` 하나로 스파크라인까지
같이 켜지고 꺼집니다. 모드 배너는 `dl_lane`의 result 패널에만 남아있고(`dl_lane_params` 쪽 배너는
그 창과 함께 사라짐), `DL_MODE_COLORS`는 이제 그 한 곳에서만 참조합니다.

### 2.22 조향 경로(`self.lane_path`)가 디바운스를 우회하던 비대칭 수정 (`perc_lane`, 2026-08-10)

**배경:** 급조향(30도 이상) 후 직진 복귀 구간에서 차선인식/조향이 흔들린다는 실차 보고를
조사하던 중(§2.19 병합과는 별개 조사), `perc_lane()`(`track_drive.py`)에서 `self.lane_offset`
과 `self.lane_path`가 서로 다른 안정성 기준으로 갱신되고 있는 걸 발견했다.

**원인:** `self.lane_offset`은 `valid`로 감싸져 있는데, 이 `valid`는 `DLSlideWindow.detect()`
안에서 `_debounce()`(`lane_util.py:747`)를 거친 값이다 — 새 후보가 `DL_STABLE_JUMP_MAX=20px`
이내로 `DL_STABLE_FRAME_MIN=3`프레임 연속 유지돼야만 반영되는 3프레임 안정성 검증을 통과한
값. 반면 실제 조향에 쓰이는 `self.lane_path`는 `if path:`(경로 리스트가 비어있지 않은지만)로만
갱신 조건을 걸었다 — 원래 주석엔 "lane_offset의 '무효 프레임엔 직전 값 유지' 폴백과 동일"이라고
적혀 있었지만 실제로는 그 안정성 검증(`valid`)을 아예 안 거치고 있었다. 그 결과 `steer_debug`
창에서 보이는 `offset`/`lane_center`는 비교적 안정적으로 보여도, 정작 조향각을 만드는 경로는
밴드 판정이 프레임마다 흔들릴 때(§2.20 "알려진 한계"의 탐색창 seed 지연 문제와 겹치면 특히
심함 — `ref_x`가 여러 프레임 지연된 상태에서 좁은 탐색창이 실제 위치를 놓치면 밴드별 분기가
`Y+W`/`1W:L`/`LOST` 등으로 매 프레임 요동친다) 그 흔들림을 거의 그대로 흡수해 조향에
전달하고 있었다.

**수정:** `perc_lane()`(`track_drive.py:417` 부근)의 경로 갱신 조건을 `if path:` →
`if valid and path:`로 바꿨다 — `lane_offset`과 동일한 3프레임 안정성 검증을 `lane_path`
갱신에도 적용한다. `valid=False`(무효/불안정 프레임)엔 직전 경로를 그대로 유지한다 —
`_lane_steer()`(`track_drive.py:1348` 부근)가 이미 "경로가 비어있으면 직전 조향각 유지"
폴백을 갖고 있어서, 갱신을 더 보수적으로 막아도 새로운 실패모드(조향각이 갑자기 사라짐 등)
는 생기지 않는다.

**알려진 한계:** `valid`가 `_debounce()`를 거치므로, 이 수정은 경로 갱신을 그만큼 더 지연시킨다
— 즉 §2.20에서 지목한 "탐색창 seed가 지연값을 쓴다"는 근본 원인(②-1) 자체를 없애는 게 아니라,
그 지연으로 생긴 흔들림이 조향까지 전파되는 경로 하나를 막는 것이다(offset과 같은 지연을
공유하게 만든 것뿐). 실제 코너 진입처럼 정말로 빠르게 바뀌어야 하는 구간에서도 이 3프레임
지연이 조향 반응성을 살짝 늦출 수 있다 — 실차에서 급코너 반응 속도와 흔들림 감소 효과를
같이 보며 `DL_STABLE_FRAME_MIN`/`DL_STABLE_JUMP_MAX` 조정 필요 여부를 판단할 것. 아직
실차 미검증.

### 2.23 ②-1(탐색창 seed 지연) 완화 — `DL_STABLE_FRAME_MIN` 3→2 + 흰/노란 탐색창 확장 이식 (2026-08-10)

**배경:** §2.22와 같은 조사 흐름 — 급조향 후 직진 복귀 구간 흔들림의 원인 ②-1(`ref_x`가
디바운스로 지연된 값이라, 좁은 탐색창이 빠르게 변하는 실제 위치를 놓친다)에 직접 대응.

**수정 1 — `DL_STABLE_FRAME_MIN` 3 → 2** ([config.py:244](config.py)): `ref_x`(§2.20/§2.22에서
설명한 대로 `_ll_yellow_white_centers()`/`_ll_slice_centers()`의 탐색창 seed)가 확정되기까지
필요한 연속 안정 프레임 수를 줄여서, 값이 빨리 바뀌는 급조향 복귀 구간에서 `ref_x`가 실제
위치를 따라잡는 속도를 높인다. `DL_STABLE_JUMP_MAX=20px` 체크는 그대로 남아있어 노이즈
필터링이 완전히 없어지진 않는다.

**수정 2 — `_ll_yellow_white_centers()`(DL_LL_ALGO='yw', main 기본값)에 §2.20의 "연속
미검출 시 탐색창 확장" 메커니즘 이식** ([dl_lane.py](perception/dl_lane.py) `_ll_yellow_white_centers()`):
노란(`cur_yellow`)/흰(`cur_white`) 각각 독립적으로 연속 미검출 횟수를 세서 그 사이드의
탐색창 반경을 `DL_LL_SEARCH_WIDEN_STEP_PX`씩 넓히고(`DL_LL_SEARCH_WIDEN_MAX_PX` 상한),
다시 찾으면 기본 반경(`DL_LL_SEARCH_HALF_WIDTH_PX`)으로 리셋한다 — §2.20에서 `_ll_slice_centers()`
(DL_LL_ALGO='lr')에 처음 넣었던 것과 동일한 패턴. **흰선 쪽에 실질적 효과가 크다** — 노란선은
이미 못 찾으면 즉시 150px 광역 3분기(④번, `DL_LL_NO_YELLOW_SEARCH_HALF_WIDTH_PX`)로 넘어가는
안전망이 있었지만, 흰선은 노란선이 계속 잡히는 동안(②/③ 경로) `cur_white`가 실제로 찾았을
때만 갱신되고 못 찾으면 그 자리에 완전히 멈춰있어서(③ "Y+gap" 근거 없는 추정으로만 빠짐)
재포착 수단이 아예 없었다. `visualize()`의 'yw' 탐색창 표시에도 확장된 창을 주황 테두리로
강조하는 걸 `_ll_slice_centers()`와 동일하게 추가했다(`DL_CENTER_MODE='ll'` 결과 패널).

합성 마스크로 직접 검증: 노란선은 seed와 거의 일치하고 흰선만 seed 대비 고정 편향이 있는
시나리오에서, 편향 40px는 2밴드 만에, 65px는 3밴드 만에 정상 검출('Y+W')로 복귀했다(확장
전이었다면 8밴드 내내 'Y+gap'으로 남았을 상황). 단 확장 상한(`DL_LL_SEARCH_WIDEN_MAX_PX=120`)
을 넘는 편향(예: 90px 이상, 초기 반경 60 기준 실질 탐지 가능 거리 최대 약 180px)은 이 확장만
으로는 못 따라잡는다 — 그런 경우는 여전히 다음 프레임 재확정(위 수정 1)에 기대야 한다.

**알려진 한계:** 두 수정 다 실차 미검증. `DL_STABLE_FRAME_MIN=2`는 노이즈 스파이크가 확정값으로
승격되기까지 필요한 프레임이 하나 줄어든 것이라, 원래보다 약간 더 노이즈에 민감해질 수 있다.
탐색창 확장은 §2.20에 이미 적어둔 것과 같은 한계(넓어진 창이 옆 차선/반사광을 잘못 물 위험)
가 노란/흰 양쪽에 그대로 적용된다 — 노란선은 원래도 ④ 안전망이 있어 확장 자체의 한계 노출은
제한적이지만, 흰선 쪽은 이번에 새로 생긴 경로라 실차에서 오탐 여부를 특히 확인할 것.

### 2.24 control_loop(20Hz)와 DL 추론 스레드 주기 불일치 대응 — `result_seq` + `LANE_STALE_SEC` (2026-08-11)

**배경:** `dl_lane.py` 모듈 상단 주석에 적힌 대로 `DLLaneDetector.detect()`는 항상 논블로킹으로
`self._latest_result`를 즉시 반환한다(추론이 별도 스레드에서 자기 페이스껏 도는 설계 — 동기
호출로 바꾸면 추론 지연이 조향 발행 주기에 그대로 전파되므로 의도적으로 피함). 그런데 이
"최신값 재사용" 구조에는 결과 자체에 타임스탬프/버전이 없어서, **추론이 20Hz보다 느려 몇 틱
동안 같은 값을 재사용하는 정상 상황**과 **추론 스레드가 예외로 멎었거나 카메라 토픽이
끊겨서 몇 초씩 아예 안 갱신되는 고장 상황**을 구분할 방법이 없었다. VESC(`_vesc_t`/
`VESC_STALE_SEC`)·IMU(`_imu_t`/`IMU_STALE_SEC`)엔 이미 있는 "죽으면 감지" 가드가 차선인식
쪽엔 없었던 것 — CLAUDE.md에 적힌 "센서 죽음은 크래시가 아니라 조용한 고정값" 패턴이 비전
파이프라인에도 그대로 적용되는데 방어가 없는 상태였다.

**수정:**
- `DLLaneDetector`에 `result_seq`(int) 카운터를 추가([dl_lane.py](perception/dl_lane.py) `__init__`).
  `_worker()`가 예외 없이 추론을 끝내고 `_latest_result`를 갱신할 때마다만 1 증가한다 —
  `detect()` 호출 횟수가 아니라 "실제로 새 결과가 몇 번 나왔는지"를 센다.
- `track_drive.py`의 `perc_lane()`이 매 틱 이 값을 직전 틱과 비교해서, 값이 안 바뀐 채로
  `LANE_STALE_SEC`(config.py, VESC/IMU와 동일하게 기본 0.5) 이상 지속되면 `self.lane_stale
  = True`로 표시한다. `result_seq`가 없는 백엔드(hough/classic_cv — 매 틱 동기 계산이라
  애초에 "재사용"이 없음)는 `getattr` 폴백으로 이 판정 자체를 건너뛰고 항상 fresh 취급한다.
- `_lane_drive()`가 `lane_stale`이면 목표속도를 `SPEED_LANE_STALE`(config.py, 기본 5.0)로
  제한한다. **속도를 서서히 깎는 방식이 아니라, "코너가 아닌데도 이 정도로 감속됐다"는
  부자연스러움 자체를 사람이 알아챌 수 있는 신호로 쓰려는 의도**(요청 반영) — `SPEED_CORNER_MIN`
  보다 낮추지 않아서 일반적인 코너 감속과 감속량이 비슷하게 겹치지만, `[LANE]` 디버그
  로그(`_print_debug()`)에 `stale=1`이 같이 찍히므로 "코너 감속인지 stale 감속인지"는
  로그로 바로 구분된다.
- 조향(`self.ctrl_angle`)은 이 수정과 무관하게 그대로 둔다 — `self.lane_path`가 멈춰있어도
  `pure_pursuit.control()`이 같은 입력에 같은 출력을 내는 안정된 고정점이라(EMA도
  `0.7*x+0.3*x=x`로 안 움직임) 발산 위험이 없고, 굳이 조향까지 건드릴 이유가 없다는 판단.

**알려진 한계:**
- `LANE_STALE_SEC`는 처음 VESC/IMU와 값만 맞춰 0.5로 넣었으나, DL 추론 1회가 20Hz 주기(0.05s)
  안에 못 끝나는 게 오히려 정상인 데다 TensorRT provider 최초 실행 시 엔진 빌드에 수십초~
  수분이 걸릴 수 있다는 경고까지 있어(`TwinLiteNetEngine.__init__` 참고) 근거가 약했다 —
  **[2026-08-11 후속] 2.0으로 상향.** 일반적인 세그멘테이션 프레임타임의 10배 이상 여유를
  두어 정상 지연을 고장으로 오판할 위험을 줄이면서도, 실제 고장 시 2초 안에는 감지되게 한
  절충점(여전히 실차 미검증). 실차에서 `FPS_LOG_PERIOD_SEC` 로그로 실제 추론 주기를 확인한
  뒤 그 주기의 몇 배 정도로 더 정밀하게 재조정할 것 — 너무 크게 두면 진짜 고장 감지가
  그만큼 늦어진다는 트레이드오프는 여전히 남아있다.
- `SPEED_LANE_STALE=5.0`이 `SPEED_CORNER_MIN`과 같은 값이라, 연속 코너 구간에서 우연히
  이 상태에 들어가면 로그를 안 보는 이상 코너 감속과 육안으로는 구분이 안 될 수 있다 —
  값을 더 낮춰 구분을 뚜렷하게 할지는 실차에서 stale이 실제로 얼마나 자주 발동하는지
  보고 판단할 것.
- `img_front is None`(카메라 프레임이 아예 한 번도 안 온 경우)도 `lane_stale=True`로
  잡지만, 카메라 토픽이 "받았다가 끊긴" 경우는 `self.img_front`가 마지막 프레임을 계속
  들고 있어 `None`이 되지 않는다 — 이 경우는 `result_seq`가 안 늘어나는 것으로만 잡힌다
  (DL 추론이 같은 정지 프레임을 계속 새로 돌더라도 입력이 똑같으면 출력도 똑같을 것이므로
  실질적으로는 문제없이 잡히지만, 엄밀히 "카메라 나이"를 직접 보는 방식은 아니다).

---

### 2.25 DL 세그멘테이션 모델을 자체 fine-tune 결과물로 교체 (`twinlitenetplus_small_bootstrap_v2.onnx`, 2026-08-11)

**배경:** 지금까지 `'dl'` 백엔드는 harrylal/TwinLiteNet-onnxruntime의 사전학습 가중치
(`models/best.onnx`)를 그대로 썼다. `fine-tune` 저장소(별도 작업 디렉터리)에서 TwinLiteNetPlus
(small)를 bootstrap_v2 데이터셋(사람 라벨 134장 기반)으로 fine-tune한 결과물이 나왔고,
`fine-tune/scripts/compare_bootstrap_v2_da.py`의 74장 사람 GT 정량비교에서 구모델 대비
da 과다포함(§2.12에서 확인된 커브 구간 편향)이 줄어든 것으로 확인됨.

**수정:**
- `fine-tune/outputs/onnx/twinlitenetplus_small_bootstrap_v2.onnx` + 외부 데이터 파일
  `twinlitenetplus_small_bootstrap_v2.onnx.data`를 이 저장소의
  `track_drive/track_drive/models/`로 복사해 커밋함 — 실차에 올리려면 저장소를 pull하면
  같이 딸려온다. 두 파일 다 같은 디렉터리에 있어야 로드된다(onnx 파일 내부에 데이터
  파일명이 상대경로로 박혀 있음, `strings`로 확인).
- `perception/dl_lane.py`: `_default_model_path()`가 가리키는 파일명을 `best.onnx` →
  `twinlitenetplus_small_bootstrap_v2.onnx`로 변경. `DL_INPUT_H`를 360 → 384로 변경
  (onnxruntime `get_inputs()/get_outputs()`로 새 모델의 입력이 `(batch,3,384,640)`,
  출력이 `(batch,2,384,640)`×2(`da`,`ll`)임을 직접 확인함 — 폭(640)·텐서 이름(`images`/
  `da`/`ll`)·전처리(letterbox 없이 리사이즈 → BGR→RGB → /255, mean/std 정규화 없음)는
  구모델과 동일해서 그 외 코드는 손 안 댐).
- `best.onnx`는 롤백/비교용으로 저장소에 그대로 남겨뒀다(기본 경로로는 더 이상 안 쓰임).

**알려진 한계:**
- 이 교체는 아직 **실차 미검증**이다 — `compare_bootstrap_v2_da.py`의 정량비교는 개발
  머신에서 정적 이미지 74장 기준으로만 확인됐고, 실제 트랙 주행(다른 조명/각도/속도)에서
  da/ll 품질이 실제로 개선됐는지는 실차 테스트로 확인해야 한다.
- `DL_INPUT_H`가 360→384로 늘어 프레임당 추론 연산량이 소폭(약 6.7%) 늘었다 — Jetson에서
  FPS가 유의미하게 떨어지면 `FPS_LOG_PERIOD_SEC` 로그로 확인 후 판단할 것.
- ll(차선) 출력의 정량비교는 위 스크립트에 없다(da만 비교함) — ll 품질도 구모델과 같거나
  나은지는 별도로 실차/육안 확인이 필요하다.

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

라바콘 진입은 **(YOLO 카메라 콘 검출 AND 라이다 좌우 클러스터 동시검출)**이
`LAVACON_TRIGGER_FRAMES(5프레임)` 연속 유지돼야 확정됩니다(`perc_lavacon_trigger()`).
`perception/yolo_cone.py`(YOLOv8n ONNX, `yolo_ros/cone_best_n.onnx`)가 별도 스레드에서
카메라 프레임에 실제 cone 클래스가 보이는지 확인하고, 라이다 단독 클러스터 판정(벽 모서리
등에서 오검출 여지가 있음)에 AND 조건으로 얹습니다. **YOLO 검출기 초기화 실패**(onnxruntime
미설치, onnx 파일 없음 등, `self.yolo_cone_detector is None`)**시엔 자동으로 라이다 단독
판정으로 폴백**합니다 — 카메라 이중확인이 안 되는 게 라바콘 전체가 안 켜지는 것보다 낫다는
판단입니다.

**[2026-08-11] 조향도 라인주행과 완전히 동일한 파라미터/컨트롤러를 씁니다.** 예전엔
`ctrl_angle = lavacon_offset * LAVACON_KP`(라바콘 전용 P게인, 라이다 offset 평균 하나만
보고 계산)였는데, 이제는 `perc_lavacon()`이 채택된 보로노이 중심선 정점들을 x(전방)
오름차순으로 정렬한 뒤 `self.lane_path`와 동일한 픽셀 스케일(`DL_PIXELS_PER_METER=200px/m`)로
변환해 `self.lavacon_path`에 담아두고, `_handle_lavacon()`이 `_lane_steer(path=self.lavacon_path,
vehicle_x=0.0)`로 라인주행(`_lane_drive()`)과 **완전히 같은 함수**(`STEERING_CONTROLLER`로
고른 Pure Pursuit/LQR 컨트롤러 인스턴스, 같은 `PP_*`/`LQR_*` 게인)를 그대로 호출합니다.
`LAVACON_KP`는 이제 안 쓰여서 config.py에서 삭제했습니다. Pure Pursuit/LQR 둘 다
"1m=200px, x=오른쪽+, 전방=이미지 위쪽" 스케일로 실측 축거를 캘리브레이션해뒀기 때문에
(controller/pure_pursuit.py·lqr.py 상단 주석 참고) 이 변환이 물리적으로 맞지만, **Voronoi
정점을 그냥 x순으로 정렬만 한 것이라 매끄러운 스플라인이 아니고, 지그재그가 심한 구간에서는
경로가 거칠 수 있습니다 — 실차 미검증.**

**디버그 방법:**
- CLI 로그: `trigL=본선카운트/기준(L{좌클러스터}R{우클러스터}Y{YOLO검출})` — 좌/우/카메라 중 어느
  쪽을 못 잡는지 바로 구분됨. 추가로 `[LAVA-ROI] L pts=... run=... R pts=... run=...` 줄에서 ROI
  안에 잡힌 점 개수(pts)와 그중 최대 연속 묶음 길이(run, 2 이상이어야 클러스터로 인정)까지 확인 가능.
- 창: `DEBUG_VIZ_LAVACON = True`([config.py:216](config.py#L216)) → `lavacon_bev` 창(트리거 ROI,
  좌/우 클러스터, `YOLO cone=` 검출 상태를 시각으로 확인). **[2026-08-11]** 조향에 실제로 쓰이는
  경로(`self._lavacon_path_m`, `perc_lavacon()`이 채운 보로노이 정점 → x오름차순 정렬 결과)도
  노란 점(정점 하나하나)+선(Pure Pursuit/LQR이 그대로 걷는 꺾은선)으로 같은 창에 겹쳐 그림 —
  트리거 판정용 ROI(좁은 0.3~0.5m)와는 별개로, 조향 경로 계산용 ROI(0~4m)에서 나온 결과라는
  점에 주의(`perception/perc_lavacon.py` 참고).
- 창: `DEBUG_VIZ_YOLO_CONE = True` → `yolo_cone_result` 창(카메라 프레임 위에 콘 검출 박스/신뢰도
  표시, `perception/yolo_cone.py`).

**알려진 한계:**
- `LAVACON_DONE_FRAMES=80`(우측 콘 연속 미검출 시 구간 종료 판정)이 실차 미검증 값.
- `YOLO_CONE_CONF_THRESHOLD=0.5`/`YOLO_CONE_INPUT_SIZE=640`이 실차 미검증 초기값.
- 조향에 쓰는 `self.lavacon_path`가 Voronoi 정점을 단순 정렬한 것이라 스플라인 피팅된
  `self.lane_path`보다 거칠 수 있음 — Pure Pursuit/LQR 자체는 라인주행에서 검증됐지만, 이
  입력(라바콘 경로)과의 조합은 실차 미검증.

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

**진입 게이트 — `_da_avoidance_failed()` (2026-08-11 추가):** B2 트리거가 걸렸다고 바로
`TargetPassing`이 켜지지 않고 먼저 `_da_avoidance_failed()`를 봅니다. `False`면(= da 기반 경로가
알아서 피하고 있다고 믿을 수 있으면) 개입 없이 Mission의 lane-follow 출력을 그대로 둡니다.
`True`(회피 실패)일 때만 아래 `TargetPassing`이 override합니다. 실패 조건은 OR로 두 개:
1. **경로 끊김/불안정** — `self.lane_valid`/`self.lane_stale`(오늘도 실제로 동작하는 신호).
2. **da가 장애물을 반영했다는 근거 없음** — da 세그멘테이션이 아직 차선표시만 학습돼 있고 장애물
   인지가 전혀 없어서, 지금은 이 조건이 **항상 참**입니다(`track_drive.py` `_da_avoidance_failed()`의
   `da_unaware_of_obstacle` 참고). 그래서 오늘 기준 실질 동작은 이 게이트를 넣기 전과 동일하게
   "B2 트리거만 걸리면 매번 TargetPassing"입니다 — da가 장애물 인지형으로 바뀌는 날 그 한 줄만
   실제 판단 로직으로 바꾸면 자동으로 전환되도록 미리 분리해둔 것입니다.

**동작 방식** (`TargetPassing`, [controller/obstacle_avoidance.py:49](controller/obstacle_avoidance.py#L49)) —
"실측 기반 하드코딩" 폴백입니다(위 게이트가 열렸을 때만 동작). Hybrid A*(검색 기반)를 여기 쓰지 않기로
한 이유는 구조화된 2차선 환경에서 검색은 과한 방식이라는 결론(§5.1과 동일)과 같습니다:
1. **IDLE** — 전방 장애물이 감지되면 `choose_side()`로 통과 방향을 정합니다: ①타겟이 없는 차선 쪽(규정
   1순위) ②정면이라 못 가리면 비어있는 쪽(`left_clear`/`right_clear`) ③둘 다 비었으면 노란선 건너편.
   양쪽 다 막히면 `status='blocked'`(흰 실선 밖으로 안 나가고 서행하며 재시도).
2. **SHIFT** — 목표 횡오프셋(`PASS_OFFSET=80px`, 실측 `LANE_WIDTH_M` 기반)까지 서서히 이동(`LATERAL_ALPHA_OUT`).
3. **ALONGSIDE** — 장애물이 안 보이는 상태가 `CLEAR_FRAMES_TO_RETURN`(6프레임) 유지되면 RETURN으로.
4. **RETURN** — 원 차선으로 복귀(`LATERAL_ALPHA_BACK`, SHIFT보다 빠르게) — 목표 오프셋이 5px 미만이면 완료.

> **[2026-08-11] Hybrid A* 대안(`USE_HYBRID_ASTAR_FOR_B2`) 삭제됨.** 위 게이트로 이미 "da가
> 알아서 처리 vs 하드코딩 폴백" 구조로 정리됐기 때문에, 검색 기반 경로계획을 B2에 따로
> 남겨둘 이유가 없어졌습니다. 동적 장애물(B3)엔 Hybrid A*가 여전히 남아있습니다 — §5.1 참고.

**디버그 방법:**
- CLI 로그: `obs=검출여부(거리m,폭m,fixed/vehicle)`. `status='blocked'`가 되면 `[B2] 양쪽 통과 불가 —
  서행 후 재시도` 경고 로그가 뜹니다.

**알려진 한계:**
- ~~`PASS_OFFSET=100px`가 실측 차선 폭 대신 쓰는 자리표시값~~ → **2026-08-11 해결**: `LANE_WIDTH_M=0.4m`
  (§6.1 실측) × `DL_PIXELS_PER_METER=200px/m` = `80px`로 교체했습니다(config.py). 수렴 속도·차선 내
  실제 여유폭까지 반영된 값은 아니라 실차에서 미세조정은 필요할 수 있습니다.
- `LATERAL_ALPHA_OUT/BACK`, `MIN_GAP_M`, `CENTER_DEADZONE_M` 등 수렴 속도·안전거리 파라미터 다수가
  실차 미검증 튜닝값입니다.
- 좌우 선택은 카메라/YOLO 이중확인 없이 라이다 `obstacle_y` + `lane_side`만으로 판단 — 콘·차량 구분이 없어
  고정장애물(콘/박스류)도 동일한 로직으로 회피 방향이 잡힙니다.
- `_da_avoidance_failed()`의 `da_unaware_of_obstacle` 조건이 아직 하드코딩 `True`입니다 — da 세그멘테이션
  모델이 장애물을 인지하도록 바뀌기 전까지는 게이트가 사실상 항상 열려 있는 상태(=매번 TargetPassing)라는
  뜻입니다. 실차 동작엔 영향 없지만, da가 장애물 인지형이 됐는데 이 줄을 안 바꾸면 계속 옛 동작(TargetPassing
  상시개입)으로 남아있다는 점을 잊지 말 것.

### 4.1 정적/동적 교차확인 로그 (`_cross_check_obstacle_motion()`, 2026-08-11)

정적(B2)/동적(B3) 분류는 여전히 **Phase 순서**(라바콘→고정장애물→방해차량, 순차 미션 설계)가 기준이고
실시간으로 바꿔타지 않습니다 — 대회 규정상 미션이 트랙 위 고정 순서/구간으로 보장되므로 원래 이걸로
충분하다는 전제이고, 실시간 속도추정을 판단 기준으로 승격하면 라이다 노이즈에 더 취약해질 위험이 있다고
판단해 로그만 남기는 선에서 그쳤습니다(`controller/obstacle_avoidance.py` 상단 주석과 동일 전제).

`apply_behavior_override()`가 B2/B3 진입 시마다 `_cross_check_obstacle_motion(tag)`를 먼저 호출합니다:
- `target_speed_est = self.v_mps(VESC 실측) + self.obstacle_rate(라이다 접근율, perc_obstacle()에서
  이미 계산)`. 타겟이 정지해 있으면 자차가 다가가는 속도만큼 `obstacle_rate`가 음수가 돼서 합이 0에
  가깝고, 자차와 같은 속도로 달리면 `obstacle_rate≈0`이라 합이 `v_mps`에 가까워집니다.
- `|target_speed_est| < OBSTACLE_STATIC_SPEED_TH_MPS`(0.3, 설계값)면 "정지처럼 보임".
- B2인데 안 정지처럼 보이면, B3인데 정지처럼 보이면 각각 경고 로그(`throttle_duration_sec=1.0`).
- VESC가 죽어있으면(`_vesc_live()`) `v_mps`를 못 믿으므로 아예 판단을 건너뜁니다.

**알려진 한계:**
- `OBSTACLE_STATIC_SPEED_TH_MPS=0.3`은 실차 미검증 설계값입니다.
- `target_speed_est` 근사는 타겟이 라이다 방사방향(자차 정면)과 거의 나란히 움직인다는 전제입니다 —
  타겟이 옆으로 가로지르면(횡방향 성분) 오차가 커집니다. 자차가 회전 중이면 더 부정확해질 수 있음.
- 로그만 남기고 B2/B3 판단 자체는 바꾸지 않습니다 — 실차에서 이 경고가 자주 뜨면 그때 Phase 기준
  게이팅을 재검토할 근거로 쓸 것.

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
- `[B3] Phase는 방해차량인데 obstacle_rate 기준 타겟이 정지해있는 것처럼 보임` 경고가 뜨면 아래
  "알려진 한계"의 오인 진입(콘을 방해차량으로 착각) 의심 상황입니다 — §4.1 참고.

**알려진 한계:**
- B2와 동일하게 카메라/YOLO 이중확인이 없어 콘·차량 구분 없이 라이다 근접만으로 트리거되므로, 콘이
  남아있는 상태에서도 거리 조건만 맞으면 B3로 오인 진입할 수 있음(Phase 순서가 지켜지는 정상 흐름에서는
  라바콘 구간을 먼저 통과한 뒤라 위험이 적지만, 격리 테스트 시에는 주의). §4.1의 교차확인 로그가 이걸
  실차에서 잡아내는 용도로 걸려있지만, 실제로 B2/B3 판단을 바꿔타지는 않습니다(로그만).
- `SWITCH_FRAMES`로 조절하는 방향 재전환 로직도 실차 미검증.

### 5.1 Hybrid A* 대안 — 동적 장애물용 (`USE_HYBRID_ASTAR_FOR_B3`, 2026-08-11)

B2에도 원래 같은 자리에 Hybrid A* 대안(`_handle_fixed_obstacle_astar()`, `USE_HYBRID_ASTAR_FOR_B2`)이
있었지만 2026-08-11에 삭제했습니다(§4 참고 — `_da_avoidance_failed()` 게이트 + `TargetPassing`
하드코딩 폴백으로 대체). B3는 그 방식을 그대로 재사용할 수 없어서 애초에 다르게 설계했습니다 — B2
방식은 "replan 시점에 그리드/goal을 딱 한 번 만들고 최대 20제어주기(1초) 또는 goal 도달까지 재사용"하는
구조인데, 방해차량은 그 1초 사이에 차선을 넘어올 수 있어 그대로 쓰면 위험합니다. 그래서 B3 전용으로
"그리드/충돌검사는 매틱, 전체 재탐색(A* 실행)은 트리거 기반"으로 설계했습니다
(`track_drive.py` `_handle_overtake_astar()`).

**재탐색 트리거 (하나라도 걸리면 발동, `_handle_overtake_astar()`):**
1. **경로 무효화** — 남은 waypoint를 매틱 최신 그리드로 재검사(`_path_blocked()`, `collision()` 5점
   투영이라 그리드 생성보다도 쌈)해서 걸리면 주기 무시하고 즉시.
2. **타겟 진입** — `TargetPassing._target_cuts_in()`과 동일 조건(통과중인 방향으로 타겟이
   `SWITCH_FRAMES` 연속 넘어옴)이면 즉시.
3. **주기적** — `ASTAR_B3_REPLAN_TICKS`(기본 4틱=0.2s)마다 최소 한 번.

goal의 좌우 방향은 `make_goal()`을 새로 손대지 않고, 이미 검증 방향성이 있는
`TargetPassing.choose_side()`를 그대로 호출해서 받습니다(`planner/hybrid_astar.py`
`make_goal_by_side()`) — 방향 우선순위(①타겟 없는 차선 ②비어있는 쪽 ③노란선 건너편) 로직을
두 곳에 중복 구현하지 않기 위함입니다.

**탐색 실패 시 — TargetPassing 폴백:** ①②(무효화/진입)로 강제 재탐색했는데도 빈 경로가 나오면
그 프레임은 감속만 하고 재시도하다가, `ASTAR_B3_FAIL_GRACE_TICKS`(기본 3틱)를 넘기면 이미 검증된
`TargetPassing.update()`로 그 통과가 끝날 때까지 위임합니다(`_b3_using_fallback` 래치 — 매틱
astar/TargetPassing을 오가면 진행 중이던 SHIFT/ALONGSIDE 기동이 중간에 끊기기 때문에, 한 번
폴백하면 `_run_passing()`의 `done` 처리에서 통과가 완료될 때까지 풀어주지 않습니다). ③(주기적)만으로
트리거된 재탐색이 실패한 경우는 기존 경로가 아직 안전하다는 뜻(무효화 검사를 이미 통과)이라 그대로
유지합니다.

**로컬 pose:** yaw는 B2와 동일하게 IMU 실측(`self.imu_yaw`) 기준입니다. x/y 적분 속도는 B2가 쓰는
명령속도(`self.ctrl_speed`, 미측정) 대신 VESC 실측(`self.v_mps`, `_vesc_live()`로 생존 가드)이
살아있으면 그걸 우선 씁니다 — B3는 재탐색이 잦아 적분 구간이 B2보다 짧아 드리프트 영향 자체는
작지만, 이미 있는 실측 인프라를 안 쓸 이유가 없습니다.

**차량 풋프린트:** `planner/hybrid_astar.py`의 충돌검사(`collision()`)가 그동안 `vehicle_width=0.45`/
`vehicle_length=0.70`을 하드코딩하고 있었는데, xycar 실측값(`VEHICLE_WIDTH_M=0.31`,
`VEHICLE_LENGTH_M=0.64`, §6.1)과 달랐습니다. 이번에 실측값 + `ASTAR_VEHICLE_MARGIN_M`(설계값,
0.05m 편도 여유)로 교체했고, B2/B3 둘 다 이 값을 공유합니다.

**디버그 방법:**
- `DEBUG_PLANNER=True`(config.py) → `Occupancy_B3` 창(B2의 `Occupancy` 창과 별도 이름 — 두 Phase가
  같은 세션에서 순차로 격리 테스트될 때 헷갈리지 않게 하려고 분리).

**알려진 한계:**
- 시간축을 갖는 진짜 동적 충돌검사가 아닙니다 — 매틱 최신 스냅샷을 트리거 기반으로 재탐색하는
  반응형 근사(receding horizon)입니다. 장애물의 미래 위치를 예측하지 않습니다.
- `ASTAR_B3_REPLAN_TICKS`/`ASTAR_B3_FAIL_GRACE_TICKS`/`ASTAR_VEHICLE_MARGIN_M` 모두 실차 미검증
  설계값입니다.
- 기본값은 `False`(B2와 동일하게 비교/보존용) — B2조차 아직 기본 비활성인 상태라, 이쪽을 실차에서
  켜기 전에 B2부터 검증하는 게 순서상 맞습니다.

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
| `VEHICLE_LENGTH_M` (config.py) | 0.64m | 위와 동일 실측(세로=전후 길이). 2026-08-11까지 config 상수로는 안 쓰이고 있었음 — `planner/hybrid_astar.py`의 충돌검사 풋프린트(`vehicle_width`/`vehicle_length`)가 실측 대신 0.45/0.70 하드코딩 추정치를 쓰고 있던 걸 이번에 이 값 + `ASTAR_VEHICLE_MARGIN_M`(설계값, 0.05m)으로 교체 |
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

### 7.1 출발/가속 중 순간 정지 — 배터리 전압 강하 의심 (2026-08-07)

**실차 증상:** `ros2 run` 직후(정지→출발)와 가속 도중, 차량이 갑자기 멈췄다가 다시 움직이는 증상이
반복 재현됨. `vesc_debug`는 초록(LIVE)이었고, `ctrl_speed`(spd)는 이 구간 내내 계속 25로 발행되고
있었는데 `v_mps`(VESC 실측)만 순간적으로 0으로 떨어졌다 회복됨.

**원인 추정:** `v_mps`가 `vesc_debug` LIVE 상태에서 0으로 떨어졌다는 건(§7 "확인 방법") 센서가 죽어서
값이 얼어있는 게 아니라 실제로 바퀴가 멈췄다는 뜻이고, 동시에 `spd`가 계속 25로 나가고 있었다는 건
FSM/판단 로직(`_change_state()` 등 `SPEED_STOP`을 세팅하는 지점, [track_drive.py:930](track_drive.py#L930)
등)이 전혀 개입하지 않았다는 뜻이다. 정지가 재현된 두 시점(재출발 직후, 가속 도중) 모두 모터 전류
요구량이 가장 큰 구간과 일치해서, 배터리 전압이 순간적으로 처지며 ESC/VESC의 저전압·과전류 보호(LVC)가
트립됐다가 부하가 빠지면(속도 0→전류 감소) 전압이 회복돼 다시 도는 패턴으로 추정된다. `SPEED_NORMAL`이
최근 5→25로 오르며(§2) 가속 시 전류 피크도 같이 커졌을 것으로 보여 시점상 들어맞는다.

**소프트웨어 완화책:** 근본 해결은 배터리 점검/교체지만, 우선 가속 램프를 더 완만하게 해서 전류 피크를
낮추는 쪽으로 대응했다 — `SPEED_ACCEL_STEP`을 0.85 → 0.4로 낮춤([config.py:93](config.py#L93) 주변,
0→`SPEED_NORMAL`(25) 도달 시간이 20Hz 기준 약 1.5초 → 약 3.1초로 늘어남).

**알려진 한계:**
- 실차 미검증 — 완화책이 실제로 정지 빈도를 줄이는지 다음 주행에서 확인 필요.
- 근본 원인이 배터리라면 `SPEED_ACCEL_STEP`을 아무리 낮춰도 정상 주행 중(가속이 끝난 뒤 정속 구간,
  혹은 코너 탈출 재가속) 전류 피크에서 여전히 트립될 수 있다 — 반복되면 배터리 잔량/전압을 부하 상태에서
  직접 측정하거나 완충 배터리로 교체해 재현 여부를 볼 것.
- VESC/ESC 쪽에 자체 LVC·과전류 임계값 설정이 있다면(VESC Tool 등) 그쪽을 낮추는 것도 대안이지만, 이
  워크스페이스 밖(하드웨어 설정) 영역이라 여기서는 다루지 않는다.

---

## 8. IMU 센서 연동 (SparkFun 9DoF Razor IMU M0, 2026-08-06)

### 8.1 하드웨어 상태 — 지금까지 고장, 이번에 수리

`/imu` 토픽은 처음부터 `track_drive.py`가 구독하고 있었지만(`cb_imu()`), **IMU 하드웨어 자체가 고장나
있어서** 실제로는 한 번도 살아있었던 적이 없었습니다. 이번에 하드웨어를 수리했고, 사용 중인 보드는
[SparkFun 9DoF Razor IMU M0 (SKU: SEN-14001)](https://learn.sparkfun.com/tutorials/9dof-razor-imu-m0-hookup-guide/all)
입니다. 펌웨어는 SparkFun 가이드가 안내하는 방식대로 다시 올릴 예정입니다.

**펌웨어를 처음부터 새로 만들 필요 없음 — 이미 이 보드 전용으로 준비돼 있었습니다.** 로봇의 기존
`~/xycar_ws/src/xycar_device/xycar_imu` 패키지(버전관리 밖에 있던 워크스페이스)를 확인한 결과,
`firmware/Razor_AHRS/Razor_AHRS.ino`에 정확히 `#define HW__VERSION_CODE 14001 // SparkFun "9DoF Razor
IMU M0" version "SEN-14001"`로 지금 쓰는 보드용 설정이 이미 돼 있었습니다 — 이 `.ino`를 Arduino
IDE(보드: SparkFun 9DoF Razor IMU M0)로 그대로 올리면 됩니다.

### 8.2 `xycar_imu` 패키지를 이 저장소로 편입

기존엔 이 IMU 드라이버 패키지가 실차의 `~/xycar_ws`(버전관리 안 됨)에만 있어서, 실차를 다시 세팅하거나
다른 사람이 이어받을 때 그대로 사라질 위험이 있었습니다. 그래서 **`~/xycar_ws/src/xycar_device/xycar_imu`
전체를 이 저장소의 [`xycar_device/xycar_imu/`](../../xycar_device/xycar_imu)로 그대로 복사해 버전관리에
편입**했습니다(빌드 산출물 `__pycache__`/`*.pyc`는 제외, `package.xml`/`setup.py`/`config/`/`launch/`/
`firmware/`/`src/` 등 ament_python 빌드에 필요한 파일은 전부 포함).

**배포 방법 (실차):** 이 저장소를 pull한 뒤 `xycar_device/xycar_imu`를 `~/xycar_ws/src/xycar_device/`에
그대로 붙여넣고 `colcon build --packages-select xycar_imu` → 다시 source하면 끝입니다.

**편입하면서 발견해 고친 버그 (2개, 둘 다 같은 원인):** `xycar_imu` 패키지의 launch 파일 두 개가 존재하지
않는 설정파일을 가리키고 있었습니다.
- `launch/xycar_imu.launch.py`, `launch/xycar_imu_viewer.launch.py` 둘 다 `config/imu.yaml`을
  로드하려 했는데, 실제로는 그런 파일이 없고(`config/`엔 `razor.yaml`/`razor_diags.yaml`/`xycar_imu.yaml`
  세 개뿐, `razor.yaml`과 `xycar_imu.yaml`은 내용이 완전히 동일함) — `imu.yaml`로 오타가 난 것으로 보입니다.
  이 상태로 `imu_node`를 띄우면 파라미터 파일을 못 찾아 시작 직후 죽습니다. 두 파일 모두 `"xycar_imu.yaml"`을
  가리키도록 고쳤습니다.
- (`launch/razor-pub-diags.launch.py` → `razor_diags.yaml`, `launch/xycar_imu_and_display.launch.py` →
  `xycar_imu.yaml`은 원래부터 정확히 존재하는 파일을 가리키고 있어 손대지 않았습니다.)

**아직 안 한 것 — 다음 단계:** [`launch/track_drive.launch.py`](launch/track_drive.launch.py)의
`imu_include`가 여전히 주석 처리돼 있습니다(`# imu_include,  # S0->S1 테스트 단계에서 비활성화...`) —
IMU 하드웨어를 고치고 `xycar_imu` 패키지를 빌드해도 이 줄을 살리지 않으면 `track_drive` 노드는 여전히
`/imu`를 못 받습니다. 펌웨어 플래싱 + 패키지 빌드가 실차에서 확인되면 이 주석을 해제할 것.

**시리얼 포트:** `config/xycar_imu.yaml`의 `port: /dev/ttyIMU`(57600bps)로 고정돼 있습니다. 이 udev
별칭이 지금 로봇에도 잡혀 있는지(보드가 새 걸로 바뀌었으니) 실차에서 확인이 필요합니다.

### 8.3 `track_drive.py`가 IMU를 쓰는 곳

`cb_imu()`가 `/imu`(`sensor_msgs/Imu`) 메시지에서 두 값을 뽑습니다:
- `self.imu_yaw` (orientation 쿼터니언 → yaw, 원래부터 있던 값) — 바퀴 카운트(아래 8.4)와 S3 지름길,
  B3(방해차량) Hybrid A* 대안 `_handle_overtake_astar()`의 Stanley 헤딩(`_yaw_delta()`)이 씁니다.
- `self.imu_yaw_rate` (`angular_velocity.z`, 이번에 추가) + `self._imu_t`(수신 시각) — §0.5.5
  `pure_pursuit` 코너 감쇠 보강 전용. `IMU_STALE_SEC=0.5`(config.py) 이상 안 들어오면 죽었다고 봅니다.

### 8.4 이번에 발견한 버그 — 바퀴 카운트가 항상 0에 멈추는 문제

**증상:** 실차에서 아무리 주행해도 CLI 로그의 `[LAP] 1/3 바퀴 누적=+0도/...`가 계속 0에 멈춰 있었음.

**원인:** `_update_lap()`([track_drive.py:832](track_drive.py#L832))은 **휠 회전이 아니라 IMU yaw
누적만으로** 바퀴 수를 셉니다 — `self.imu_yaw`의 프레임간 차이를 계속 더하는 구조라, 독립적인
휠 기반 폴백이 전혀 없습니다. IMU 하드웨어가 죽어있던 동안엔 `self.imu_yaw`가 초기값 `0.0`에서 한 번도
안 바뀌었으니(`/imu`를 아예 못 받음) 프레임간 차이가 항상 `0`이라 `_yaw_accum`도 영원히 `0`으로
찍힌 것 — 코드 버그가 아니라 **§8.1의 하드웨어 고장이 그대로 드러난 증상**이었습니다.

**대응:** 이번 IMU 수리(§8.1)로 `/imu`가 실제로 들어오기 시작하면 자동으로 해결됩니다 — `_update_lap()`
자체는 손대지 않았습니다. 다만 §8.2의 "아직 안 한 것"(`imu_include` 주석 해제)까지 끝나야 실제로
`/imu`가 `track_drive` 노드에 도달합니다.

**확인 방법:** 실차 주행 중 CLI 로그의 `[LAP] n/3 바퀴 누적=...도` 값이 주행하는 동안 계속 늘어나는지
확인하세요. 여전히 `0`에 멈춰 있다면 ①`imu_include` 주석 해제했는지 ②`xycar_imu` 노드 시작 로그에러
③`/dev/ttyIMU` 포트 순으로 확인할 것.

**알려진 한계:** IMU가 나중에 다시 죽어도(§7 VESC의 `vesc_debug` 창처럼) 바로 알아챌 디버그 창이 없습니다
— `_imu_t`(§8.3)로 생존 체크 인프라 자체는 §0.5.5에서 만들었지만, 아직 전용 디버그 창(`imu_debug`
같은)까지는 안 만들었습니다. 지금은 `[LAP] 누적=`이 안 늘어나는 것으로 간접 확인해야 합니다.
