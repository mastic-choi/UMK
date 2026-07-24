# track_drive 테스트 가이드

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu opencv-python


`track_drive.py`는 하나의 노드 안에서 신호등/차선/라바콘/장애물/추월을 전부 처리하는 2중 FSM 구조입니다
(`MissionState` S0~S4 + `BehaviorState`/`Phase`). 기능별로 **부분만 골라서** 테스트하려면 파일 상단의
"개발/테스트 플래그" 블록([track_drive.py:142-176](track_drive.py#L142))만 조합해서 바꾸면 됩니다.

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
- `SPEED_NORMAL`([track_drive.py:71](track_drive.py#L71))을 `0.0`으로 두지 마세요. `_lane_drive()`에서 나눗셈
  분모로도 쓰여서 `ZeroDivisionError`로 노드가 죽습니다. 저속 테스트는 `5.0`~`10.0` 같은 작은 양수를 쓰세요.
- 실제 모터 구동은 ROS2 노드만으로 안 됩니다. 대회/실차 기준 **Docker(ROS1) 컨테이너 + `ros1_bridge`**가 같이
  떠 있어야 합니다 (`/xycar_motor`는 `Float32MultiArray([angle, speed])`로 브릿지됨 — 구형 `XycarMotor` 커스텀
  메시지는 `ros1_bridge`가 매핑을 못 함). 체크리스트: ①도커 컨테이너 기동 ②`ros1_bridge` 프로세스 기동 확인.
- `DEBUG_VIZ`([track_drive.py:147](track_drive.py#L147))는 **죽은 플래그입니다.** 실제로 신호등 디버그 창을
  켜는 스위치는 `traffic_signal.py`의 `DEBUG_VIZ_SIGNAL`입니다. 마찬가지로 `track_drive.py`의
  `DEBUG_VIZ_LANE`도 죽은 플래그이고, 실제 차선 디버그 창 스위치는 `lane_util.py`의 `DEBUG_VIZ_LANE`
  (별개의 변수, 이름만 같음)입니다. 헷갈리지 않게 아래 기능별 표에 실제 스위치 위치를 정리해뒀습니다.

| 기능 | 디버그 창 ON/OFF 스위치 |
|---|---|
| 신호등 | `traffic_signal.py:23` `DEBUG_VIZ_SIGNAL` |
| 차선 | `lane_util.py:38` `DEBUG_VIZ_LANE` |
| 라이다 BEV(장애물) | `track_drive.py:149` `DEBUG_VIZ_LIDAR` (이건 정상 연결됨) |
| 라이다 BEV(라바콘 트리거) | `track_drive.py:150` `DEBUG_VIZ_LAVACON` |

> 이 프로젝트는 YOLO(yolo_ros)를 사용하지 않습니다 — 모든 인지는 카메라(차선/신호등)와 라이다(장애물/라바콘)만으로 수행합니다.

---

## 1. 신호등 (S0 출발 / S2 교차로)

**수정할 곳:** `track_drive.py:143`
```python
START_STATE = MissionState.S0_WAIT_GREEN   # 3구 신호(출발) 테스트
# 또는
START_STATE = MissionState.S2_INTERSECTION # 4구 신호(교차로) 테스트 — 시작하자마자 정지 상태로 대기
```
S2로 시작할 땐 `TEST_DISABLE_INTERSECTION` 값과 무관하게 무조건 S2에서 시작합니다(이 플래그는
S1→S2 전환 경로만 막는 것이라 `START_STATE` 자체를 바꾸는 것과는 별개입니다). S2는 4구 신호(직진/좌회전)를
인식할 때까지 `ang=0, spd=0`으로 계속 정지하는 게 정상 동작이니, 안 움직인다고 바로 버그로 보지 말고
로그의 `sig=`/`[SIG-S0]` 값부터 확인하세요.

**디버그 방법:**
- 창: `traffic_signal.py:23` `DEBUG_VIZ_SIGNAL = True` → `signal_roi`(S0) / `signal4_roi`(S2) 창.
- CLI 로그: S0 상태일 때 `_print_debug()`가 0.5초마다 `[SIG-S0]` 줄을 찍습니다
  (`roi=`, `circles=`, `reason=`, `bright=`, `margin=`) — 원 검출이 어느 단계(개수 부족/배치 불량/
  밝기 대비 부족)에서 막혔는지 터미널만으로 바로 보입니다.

**알려진 한계(실차 미검증):**
- `find_circles()`(Hough Circle)가 원 개수를 **정확히** 3개(S0)/4개(S2)로 요구하고, 배치 검사까지 실패하면
  그 프레임은 무조건 인식 실패 — 디바운스/폴백 없음([traffic_signal.py:53](traffic_signal.py#L53) 주석 참고).
- ROI(`SIG_ROI_*`)와 반지름 범위(`SIG_MIN/MAX_RADIUS=15~25px`)가 고정값이라, 카메라 각도·정지 위치가
  튜닝 당시와 다르면 신호등이 ROI 밖이거나 반지름 범위 밖이라 아예 못 잡을 수 있음.
- 색상(Hue)을 직접 보지 않고 **위치(좌→우=빨강/노랑/초록) + 밝기 대비**로만 판정 — 밝은 반사광이 ROI에
  섞이면 오탐 가능. S0의 색 판정은 단일 프레임 즉시 트리거라(`_s0_wait_green()`) 디바운스도 없음.

---

## 2. 라인트래킹 (차선주행, S1)

**수정할 곳:** `track_drive.py:143`, `166`
```python
START_STATE = MissionState.S1_LANE_FOLLOW
TEST_FORCE_BEHAVIOR = False   # 라바콘 등 Behavior 없이 순수 차선주행 PID만 보고 싶을 때
```
`TEST_DISABLE_INTERSECTION = True`(기본값)면 정지선을 밟아도 S2로 안 새고 차선주행을 계속합니다.

**디버그 방법:**
- 창: `lane_util.py:38` `DEBUG_VIZ_LANE = True` → `lane_bev`(BEV 변환), `lane_white`/`lane_yellow`(색 마스크),
  `lane_result`(슬라이딩윈도우 피팅 결과 + `offset` 표시).
- CLI 로그: `[LANE] lane=편차px(검출여부) obs=... lava=...`.

**알려진 한계:**
- (2026-07-21 업데이트) 흰색 검출은 더 이상 단순 HSV 고정 임계값이 아닙니다. 지금은
  `lane_util.py:81-145` — Gray→CLAHE(지역 대비 향상)→**Top-Hat 모폴로지**(31×31 커널, 넓은 영역에
  걸친 균일한 반사광은 눌러주고 국소적으로 튀는 밝은 부분만 남김)→threshold(20)→세로 성분 강조
  커널(3×10)→Connected Components 면적 필터(20~1500px²)로 재구성됐습니다. 노란색도 HSV 범위를
  좁히고(`[18,120,120]~[35,255,255]`) 면적(20~1000)·폭(<40px) 필터를 추가했습니다([lane_util.py:106-165](lane_util.py#L106)).
  넓고 균일한 반사광 오검출 문제는 이 구조로 상당히 개선될 것으로 보이나, Top-Hat은 "넓은 면적"의 밝기
  변화를 누르는 방식이라 **가늘고 긴 반사(예: 금속 난간의 얇은 하이라이트 줄)**는 여전히 차선처럼 남을
  수 있습니다. Top-Hat 커널 크기·threshold(20)·면적 범위 전부 실차 미검증 값이라 실측 튜닝 필요.
- `_s1_lane_follow()`가 `self.lane_valid`를 확인하지 않고 `_lane_drive()`를 호출함(`_s3_shortcut()`은 확인함,
  [track_drive.py:725](track_drive.py#L725)). 카메라가 순간적으로 차선을 놓쳐도 마지막 유효 offset으로
  계속 조향하니, 실차 테스트 시 차선 이탈 구간에서 주의 깊게 볼 것. (아직 미수정)

---

## 3. 라바콘 (B1_LAVACON)

**수정할 곳:** `track_drive.py:143`, `166`
```python
START_STATE = MissionState.S1_LANE_FOLLOW
TEST_FORCE_BEHAVIOR = True    # S2를 거치지 않고 시작부터 Behavior(라바콘부터) 강제 활성화
```
`self.phase`는 기본이 `Phase.LAVACON`([track_drive.py:248](track_drive.py#L248))이라 따로 안 건드려도 됩니다.
`TEST_DISABLE_B2_B3 = True`(기본값)면 라바콘 구간이 끝나도 B2/B3로 안 넘어가고 그냥 일반 차선주행으로
돌아오니, 라바콘만 격리 테스트하기 좋습니다.

라바콘 진입은 **라이다 좌우 클러스터 동시검출**이 `LAVACON_TRIGGER_FRAMES(5프레임)` 연속 유지돼야
확정됩니다(`perc_lavacon_trigger()`, 라이다 단독 판단 — 카메라/YOLO 이중확인 없음).

**디버그 방법:**
- CLI 로그: `trigL=본선카운트/기준(L{좌클러스터}R{우클러스터})` — 좌/우 중 어느 쪽을 못 잡는지 바로 구분됨.
  추가로 `[LAVA-ROI] L pts=... run=... R pts=... run=...` 줄에서 ROI 안에 잡힌 점 개수(pts)와 그중 최대
  연속 묶음 길이(run, 2 이상이어야 클러스터로 인정)까지 확인 가능.
- 창: `track_drive.py:150` `DEBUG_VIZ_LAVACON = True` → `lavacon_bev` 창(트리거 ROI와 좌/우 클러스터를
  시각으로 확인).

**알려진 한계:**
- `LAVACON_DONE_FRAMES=80`(우측 콘 연속 미검출 시 구간 종료 판정)이 실차 미검증 값.

---

## 4. 사물회피 (B2_OBSTACLE, 고정장애물) — ★재설계 예정 placeholder

**수정할 곳:** `track_drive.py:143`, `161`, `166`, `248`
```python
START_STATE = MissionState.S1_LANE_FOLLOW
TEST_DISABLE_B2_B3 = False     # B2 트리거 검사를 켜야 함
TEST_FORCE_BEHAVIOR = True
self.phase = Phase.FIXED_OBSTACLE   # __init__ 안의 self.phase 초기값을 임시로 변경 (격리 테스트용)
```
정상 흐름은 라바콘(B1) 완료 후 자동으로 `Phase.FIXED_OBSTACLE`로 넘어가는 것이라, 이 기능만 격리
테스트하려면 `__init__`의 `self.phase = Phase.LAVACON`을 위처럼 임시로 바꾸면 됩니다(YOLO 노드 전환 같은
추가 조치는 필요 없음).

**디버그 방법:**
- CLI 로그: `obs=검출여부(거리m,좌/우/중앙,fixed/vehicle)`.

**알려진 한계:**
- `decide_target_lane()`의 좌우 회피 방향은 라이다 기반 `obstacle_side`(`perc_obstacle()`이 전방 ROI
  포인트의 횡위치 EMA로 산출)를 씁니다. 카메라/YOLO 이중확인이 없어 콘·차량 구분 없이 "뭔가 있으면" 방향만
  판단하는 대신, 고정장애물(콘/박스류)도 동일하게 회피 방향이 잡힙니다. `_ema_y` 데드존(`SIDE_DEADZONE`)
  안에 있으면 `'center'`로 판정되어 회피 방향이 정해지지 않으니, 실차 테스트 시 이 경계값 튜닝이 필요합니다.
- 실제 회피 궤적 자체도 "감지되면 감속하고 버티다가 사라지면 복귀"하는 placeholder입니다
  ([track_drive.py:990](track_drive.py#L990) 주석 참고). 실제 회피 기동 아님.

---

## 5. 차량회피/추월 (B3_VEHICLE) — ★재설계 예정 placeholder

**수정할 곳:** `track_drive.py:143`, `161`, `166`, `248`
```python
START_STATE = MissionState.S1_LANE_FOLLOW
TEST_DISABLE_B2_B3 = False
TEST_FORCE_BEHAVIOR = True
self.phase = Phase.VEHICLE     # 격리 테스트용 임시 변경
```
B2와 마찬가지로 별도 노드 전환 없이 `self.phase = Phase.VEHICLE`만 바꾸면 격리 테스트할 수 있습니다.
진입 조건은 **라이다 단독** — 전방 장애물 + 거리 < `OVERTAKE_TRIGGER=6.5m`가 `VEHICLE_TRIGGER_FRAMES(5프레임)`
연속 유지되면 확정됩니다(`perc_vehicle_trigger()`).

**디버그 방법:**
- CLI 로그: `trigV=본선카운트/기준`.

**알려진 한계:**
- B2와 동일하게 실제 추월 궤적은 "감속하고 버티다가 차량 사라지면 종료"하는 placeholder입니다
  ([track_drive.py:1060](track_drive.py#L1060) 주석 참고). `decide_target_lane()`을 매 프레임 재호출하는데
  B2는 진입 시 한 번만 호출 — 두 Behavior 간 동작 방식이 비대칭이라는 점도 참고.
- 카메라/YOLO 이중확인이 없어 콘·차량 구분 없이 라이다 근접만으로 트리거되므로, 콘이 남아있는 상태에서도
  거리 조건만 맞으면 B3로 오인 진입할 수 있음(Phase 순서가 지켜지는 정상 흐름에서는 라바콘 구간을 먼저
  통과한 뒤라 위험이 적지만, 격리 테스트 시에는 주의).
