# track_drive 아키텍처 레퍼런스

`track_drive.py`는 신호등 인식/차선주행/라바콘/장애물회피/추월을 하나의 노드에서 처리하는
2중 FSM 구조입니다(`MissionState` + `BehaviorState`/`Phase`). 인지 모듈은 `perception/`,
조향/회피 제어는 `controller/`, Hybrid A*(B2 대안, 현재 미사용·보존용)는 `planner/`에
모여 있습니다.

이 문서는 루트 README(설치/실행)를 보완하는 **내부 아키텍처 레퍼런스**로, "지금 시스템이
어떻게 동작하는가"만 다룹니다. 개발 중 시행착오·날짜별 수정 이력은 `git log`에서 확인하세요.

실행 명령:
```bash
ros2 launch track_drive track_drive.launch.py
```

튜닝 파라미터, 디버그 on/off, 미션 관련 플래그는 전부 [`config.py`](config.py) 한 파일에
모여 있고 대부분의 모듈이 `from .config import *`로 가져다 씁니다. 값을 찾거나 바꿀 땐
여기부터 확인하세요.

---

## 목차
1. [대회 규정 요약](#1-대회-규정-요약)
2. [노드 실행 구조](#2-노드-실행-구조)
3. [미션 상태 머신 (MissionState)](#3-미션-상태-머신-missionstate)
4. [비헤이비어 FSM (BehaviorState / Phase)](#4-비헤이비어-fsm-behaviorstate--phase)
5. [지름길(좌회전) 파이프라인](#5-지름길좌회전-파이프라인)
6. [인지 스택](#6-인지-스택)
7. [조향 컨트롤러 (Pure Pursuit)](#7-조향-컨트롤러-pure-pursuit)
8. [VESC / IMU 연동](#8-vesc--imu-연동)
9. [디버그 시각화](#9-디버그-시각화)
10. [알려진 한계](#10-알려진-한계)
11. [캘리브레이션/실측값](#11-캘리브레이션실측값)

---

## 1. 대회 규정 요약

> 출처: `2026년-9회대회-경주진행방법-7월29일자버전-1.pdf`(자이트론, 2026-07-29 버전). 본선 주행
> 경기(트랙 미션) 규정만 정리. 대회 직전까지 변경될 수 있다고 원문에 명시돼 있으니 최종 확인은
> 항상 원본 공지로 할 것. 아래 규정이 `MissionState`/`Phase` 순서 설계의 근거입니다.

### 한눈에 보기
- 차량: RC카 기반 자율주행 차량(카메라 등 센서 인식, ROS2 기반 SW).
- 목표: 정해진 트랙을 **3바퀴** 자율주행하며 각 구간 미션을 통과하고 결승선 통과.
- 성적: `총 주행시간 = 순수 주행시간 + 벌초(penalty seconds)` — 짧을수록 좋음.
- 오전 1회 + 오후 1회, **둘 중 더 좋은 기록**으로 순위.
- 3바퀴 총 주행시간이 **4분**(벌초 미포함)을 넘으면 실격.

### 미션 순서
```
신호등 인식 출발
  → 라바콘(러버콘) 구간 주행           [B1]
  → 차선인식 주행(차선 준수)
  → 고정장애물 회피 주행               [B2]
  → 방해차량(앞차) 추월 주행           [B3]
  → [3바퀴 반복 — 2바퀴째 또는 3바퀴째 중 한 번만 "지름길" 선택 가능]
  → 결승선 통과(3바퀴 완주) → 경주 종료
```

### 트랙 구조
- 직사각형 순환 코스 1개 루프. 코스 중간에 트랙을 좌우로 나누는 분기 구간(지름길)이 트랙
  전체에서 한 곳만 있음.
- 구간 배치(시계 순서): ① 출발 지점(4구 신호등 ↔ Gate) → ② 라바콘 구간(지그재그 배치) →
  ③ 차선 주행 구간(실선-실선 사이, 점선은 참고선) → ④ 고정 장애물 회피 → ⑤ 방해차량 추월 →
  ⑥ 신호등/지름길 분기점(좌회전 화살표 포함 4구 신호등) → ⑦ 결승선(Gate, 한 바퀴마다 통과).
- 총 3바퀴, **2바퀴째 또는 3바퀴째 중 한 번만** 지름길 통과 가능. 나머지는 정규 코스.

### 출발 절차 (`MissionState.S0_SIGNAL`)
- 심판이 신호등을 빨강→파랑으로 전환하는 순간부터 랩타임 측정 시작(중간 노랑 대기 없음).
- 파란불 전 출발(신호위반): 10초 벌초 + 재출발. 재출발도 위반 시 추가 10초 + 재위치 이동 후
  재주행.
- 파란불 전환 후 **1분 이내** 미출발 시 실격.

### 주행 중 정지 / 차량 터치
- 주행 중 정지 후 **1분 이내** 재개 못 하면 실격.
- 사람이 차량을 터치하면 1회당 5초 벌초(원칙적으로 심판 지시 시). 한 바퀴 기준 15회(60초)
  초과 시 실격.

### 라바콘 구간 (`Phase.LAVACON` / `BehaviorState.B1_LAVACON`)
- 충돌 시 개당 3초 벌초. 경로 완전 이탈 시 이탈 위치(그 이후로는 안 됨)로 옮겨 재주행.
- 구간을 1분 이내에 통과하지 못하면 실격.

### 차선 준수 주행
- 양쪽 바깥쪽 실선을 벗어나지 않고 주행(실선 사이 어디든 무방, 점선/1·2차선 구분 없음).
- 이탈 판정(하나라도 해당 시 이탈위치로 재주행):
  1. 앞바퀴 2개 동시에 실선 밖
  2. 좌/우 바퀴 2개 실선 밖 + 카메라(차량 중앙부)도 실선 밖(절반 이상 이탈)
  3. 좌/우 바퀴 2개가 실선 밖으로 90cm 이상 유지 주행
- 코너 안쪽 컷 방지용 고정 장애물이 일부 코너 안쪽에 배치됨.

### 고정 장애물 회피 (`Phase.OBSTACLE_ZONE` / `BehaviorState.B2_OBSTACLE`)
- 정지 장애물을 충돌 없이 회피. 차선 이탈/터치 판정은 위 일반 규정과 동일.

### 방해차량 추월 (`Phase.OBSTACLE_ZONE` / `BehaviorState.B3_VEHICLE`)
- 방해차량 1대가 저속으로 1·2차선을 오가며 주행.
- 추월 중 실선 이탈은 차선이탈로 안 봄. 추월 후 최대한 빨리 복귀 필요(복귀 후 방해차량과
  90cm 이상 간격이면 오히려 이탈 판정).
- 추월은 방해차량이 없는 차선 쪽으로만 가능.
- 추돌(가해/피해 모두) 각 10초 벌초. 피해 시 50cm 이내로 앞당겨 이동 가능.

### 지름길 분기 신호 (`MissionState.S0_SIGNAL` 재진입, 1바퀴 주행 후)
- 트랙 중앙 분기점 4구 신호등에서 좌회전(지름길) 신호가 켜지면 진입 가능.
- 좌회전 신호는 2바퀴째 또는 3바퀴째 시작 중 랜덤으로 딱 한 번만 등장 — 나머지는 항상
  직진(초록) 확정. 즉 지름길 선택 기회는 3바퀴 중 정확히 한 번.

### 시간 제한 / 실격 사유 총정리
| 상황 | 조치 / 벌점 |
|---|---|
| 빨간불 출발 | 10초 벌초 + 재출발 |
| 재출발도 빨간불 | 추가 10초 벌초 + 원위치 재주행 |
| 파란불 전환 후 1분 내 미출발 | **실격** |
| 정지 후 1분 내 미재개 | **실격** |
| 차량 터치 1회 | 5초 벌초(바퀴당 15회/60초 초과 시 **실격**) |
| 라바콘 충돌 | 개당 3초 벌초 |
| 라바콘 구간 1분 내 미통과 | **실격** |
| 차선 이탈 | 이탈위치로 재주행 |
| 방해차량 추돌(가해/피해 모두) | 각 10초 벌초 |
| 3바퀴 총 주행시간 4분(벌초 제외) 초과 | **실격** |
| 결승선 3회 통과(3바퀴 완주) | 경주 종료 |

성적 산출: `총 주행시간 = 순수 주행시간 + 벌초 합계`. 오전/오후 중 더 짧은 기록으로 순위 결정.

---

## 2. 노드 실행 구조

3개 ROS2 패키지:
- `track_drive/` — 메인 제어 노드(FSM/인지/제어 전부).
- `yolo_ros/` — 코드 패키지 아님. `.onnx`/`.pt` 가중치 + TensorRT 캐시만 보관하고,
  `track_drive/perception/*.py`가 상대경로로 로드.
- `xycar_device/` — 벤더 드라이버 패키지(`xycar_imu` 등).

`track_drive.py`의 `control_loop()`는 20Hz(`create_timer(0.05, ...)`)로 돌며 매 틱 순서는:

```
perceive_all()
  → _update_lap()
  → run_mission_fsm()
  → [디버그 시각화 — 전부 기본 OFF]
  → if mission_state == S1_LANE_FOLLOW and _behavior_enabled and _shortcut_exit_kick_cnt <= 0:
        run_behavior_fsm() + apply_behavior_override()
    else:
        behavior_state = B0_NORMAL
  → pose_estimator.update()
  → drive(ctrl_angle, ctrl_speed)
  → [디버그 시각화/로그]
```

`xydrive.py`는 별도의 최소 스모크테스트 노드로, `xycar_motor`에 `angle=0, speed=DEFAULT_SPEED
(5.0)`을 20Hz로 그대로 publish — `track_drive.py`를 전혀 거치지 않고 모터 드라이버 배선만
검증하는 용도.

---

## 3. 미션 상태 머신 (MissionState)

```python
class MissionState(Enum):
    S0_SIGNAL      = 0  # 4구 신호등 판단 (출발선/교차로 공용)
    S1_LANE_FOLLOW = 1  # 차선주행 (B1/B2/B3는 이 상태 안에서 처리)
    S4_FINISH      = 4  # 종료 (레이스 완주 판정은 사람 심판이 함 — 코드가 이 상태로 전이하지 않음)
```

(`S3_SHORTCUT`은 삭제됨 — 좌회전은 별도 상태 없이 S0_SIGNAL 진입 램프 + S1 안의 시간/거리
트리거로 처리한다. §5 참고.)

### `_s0_signal()`
- `_checker_ramp_dist is not None` → `_do_checker_ramp_turn()` 진행 중(좌회전 진입 램프, §5).
- 아니고 `_s2_commit_dist is not None` → 정상 주행하며 거리를 누적, `checker_pillar_trigger`
  대기(§5). `CHECKER_PILLAR_LIDAR_TIMEOUT_SEC=5.0`초 안에 트리거가 안 뜨면 좌회전을 포기하고
  직진으로 처리.
- 둘 다 아니면 정지 상태로 신호 대기.

### `_s1_lane_follow()`
- `signal_left_confirmed` → `_change_state(S0_SIGNAL)` + `_s2_commit_dist=0.0`(좌회전 커밋
  시작).
- `signal_straight_confirmed` → S1 유지, `_behavior_enabled=True`로 켜고
  `phase=Phase.LAVACON` + `_b2_passed`/`_b3_passed`/`_lavacon_engaged=False`로 리셋 —
  이게 B1~B3의 유일한 재무장(rearm) 지점 중 하나(다른 하나는 좌회전 램프 완료 시, 아래).
- 이후 B1/장애물 가드를 거쳐, 해당 없으면 `_lane_drive()`(일반 차선주행).

### `_do_checker_ramp_turn()` 완료 시
좌회전 램프가 끝나면 즉시:
- `_behavior_enabled=True`, `phase=Phase.LAVACON`, b2/b3/lavacon 플래그 리셋(재무장)
- `_change_state(S1_LANE_FOLLOW)` — 신호 대기 없이 곧장 차선주행 복귀
- `_shortcut_exit_dist=0.0` 시작(지름길 탈출 킥 트리거용, §5)

### `S4_FINISH`
코드상 존재는 하지만(`ctrl_angle,ctrl_speed = 0, SPEED_STOP`) 아무것도 이 상태로 전이시키지
않는다 — 레이스 완주 판정은 사람 심판의 외부 타이머로 이뤄지고, 코드는 관여하지 않는다.

---

## 4. 비헤이비어 FSM (BehaviorState / Phase)

```python
class BehaviorState(Enum):
    B0_NORMAL   = 0  # 차선주행 출력 그대로
    B1_LAVACON  = 1  # 라바콘 구간
    B2_OBSTACLE = 2  # 고정장애물 회피
    B3_VEHICLE  = 3  # 방해차량 추월

class Phase(Enum):
    LAVACON       = 0
    OBSTACLE_ZONE = 1  # 고정장애물 + 방해차량 통합 구간(예전 FIXED_OBSTACLE/VEHICLE)
    DONE          = 2  # B2+B3 둘 다 완료 — 이후 계속 B0로 일반 차선주행
```

`run_behavior_fsm()`이 `Phase`를 순차 진행(라바콘 → 장애물구간 → 완료, 우선순위 판단 불필요).
정적/동적 장애물 구분은 Phase가 아니라 매 프레임 `obstacle_type`(라이다 실측 폭 기반)으로
판단한다.

### B1 (Phase.LAVACON, 라바콘)
- **진입**: `lavacon_trigger`(라이다 좌우 클러스터 동시검출 AND YOLO 콘 검출,
  `perc_lavacon_trigger()`) → `_lavacon_engaged=True` → `behavior_state=B1_LAVACON` →
  `apply_behavior_override()`가 `_handle_lavacon()` 호출.
- **조향**: 진입 확정 순간(rising edge) `LAVACON_KICK_ENABLED=True`면 고정
  `LAVACON_KICK_ANGLE_DEG=-30.0`을 `LAVACON_KICK_DURATION_S=0.4`초 강제 유지. 이후
  `LAVACON_STEER_MODE_DA_PUSH=True`로 `_lavacon_steer_da_push()`(da 경로 + 침범한 콘
  반대쪽으로 밀기)가 담당:
  - 안전마진: 좌 `LAVACON_PUSH_SAFETY_MARGIN_L_M=0.35`, 우 `_R_M=0.23`
  - push 세기: `LAVACON_PUSH_GAIN=1.35`
  - ROI: `LAVACON_PUSH_LON_MIN=-0.1`~`LON_MAX=0.25`(좌측만 `+0.1` 확장), `LAT_LIMIT=1.0`
- **속도**: 일반 S1 주행과 동일한 `_update_speed()` 공유 — B1 전용 고정 속도 없음.
- **탈출**: `lavacon_done`(넓은 ROI로 좌우 모두 비었음, `perc_lavacon.py`)이
  `LAVACON_DONE_FRAMES=40`(약 2초) 연속 유지되면 `phase=Phase.OBSTACLE_ZONE`으로 전환.

### B2/B3 (Phase.OBSTACLE_ZONE, 고정장애물/방해차량)
- `behavior_state`는 항상 `B0_NORMAL`로 유지된다 — 실제 회피는 `ENABLE_OBSTACLE_CUT=True`
  da-근접-컷(da near-cut)이 담당하며, `behavior_state`와 무관하게 항상 켜져 있다.
- 메커니즘: `perc_obstacle_cut_trigger()`(라이다+YOLO 이중확인) → `obstacle_cut_active` →
  `_clip_da_by_obstacle()`(`dl_lane.py`)가 da 경로를 장애물 쪽에서 잘라냄. 속도는
  `SPEED_PRE_OBSTACLE_CAP=8.0`으로 사전 제한.
- `run_behavior_fsm()`은 `obstacle_cut_active`의 진입/이탈만 관찰해 B2('fixed')/B3('vehicle')
  태그를 붙이고 `_mark_behavior_passed()`를 호출하는 역할만 한다.
- **트랙 순서 = B2가 먼저.** `_b3_armed()` = `_b2_passed and (now - _b2_passed_t) >=
  B2_TO_B3_DELAY_SEC(3.0)` — B2 완료 후 3초 동안은 차량형 검출도 강제로 'B2'/콘 분류로
  취급한다(방금 지나온 고정장애물을 B3로 오분류하는 것을 막는 가드).
- `Phase.DONE`: B2+B3 둘 다 완료되면 `behavior_state=B0_NORMAL`로 일반 차선주행만 계속.

---

## 5. 지름길(좌회전) 파이프라인

1. `signal_left_confirmed` → `S0_SIGNAL` 진입, `_s2_commit_dist=0.0`.
2. 정상 주행하며 거리 누적. `perc_checker_pillar()`(라이다 좌우 기둥쌍 게이트) →
   `checker_pillar_trigger` → `_begin_checker_ramp_turn()`. (5초 타임아웃 시 포기하고 직진
   처리, §3)
3. **진입 램프**: `_checker_turn_ramp_angle()` — `CHECKER_TURN_RAMP_START_ANGLE=0`에서
   `END_ANGLE=-25.0`까지 `CHECKER_TURN_RAMP_DIST_M=2.5`m에 걸쳐 `'smoothstep'` 곡선으로
   전개. 속도는 `TURN_SPEED=12.0`.
4. 램프 완료 시 즉시 B1 재무장 + `S1_LANE_FOLLOW` 복귀 + `_shortcut_exit_dist=0.0` 시작(§3).
5. 정상 S1 주행. `_s1_lane_follow()`가 `_shortcut_exit_dist`를 누적하다가
   `SHORTCUT_EXIT_DIST_M=5.5`에 도달하면 `_begin_shortcut_exit_kick()`.
6. **탈출 킥**(진입 램프와는 별개의, 시간 기반 메커니즘): `_do_shortcut_exit_kick()`이 고정
   `SHORTCUT_EXIT_KICK_ANGLE_DEG=-20.0`을 `SHORTCUT_EXIT_KICK_DURATION_S=0.5`초 동안
   `TURN_SPEED`로 유지. 이 동안 `control_loop()`는 비헤이비어 FSM 게이트 전체를 꺼서 다른
   로직이 이 조향을 덮어쓰지 못하게 한다.
7. 킥 완료는 Phase/B1~B3 상태를 건드리지 않는다 — 지름길은 결승선으로 곧장 이어지므로, 킥
   이후 더 이상 B1/B2/B3가 나오지 않는 것을 전제로 한다.

---

## 6. 인지 스택

### 차선/DA(주행가능영역)
`LANE_DETECTOR_BACKEND='dl'`(기본값) → `DLLaneDetector`(`perception/dl_lane.py`)가
`twinlitenetplus_kmu_v1.2.0.onnx`(TwinLiteNet+ 파인튜닝, da+ll 듀얼 세그멘테이션, 별도
추론 스레드)를 사용. `'dl'` 초기화 실패 시 `'hough'`(`HoughLaneDetector`)로 자동 폴백.
`'classic_cv'`는 보존용으로 남아있으나 현재 라이브 미검증.

`_largest_da_component()`가 da 커넥티드컴포넌트 중 "내 차선"을 고를 때, 우선순위는
① 시드(차량 위치와 맞닿은 덩어리) → ② 시간적 연속성(직전 프레임과 가장 가까운 덩어리) →
③ 면적 범위(`DL_DA_MIN_COMPONENT_AREA`/`DL_DA_MAX_AREA_PX`, 위 둘 다 없을 때만 쓰는 최후
폴백). 면적만으로 판단하는 방식은 실차 주행으로 검증했을 때 통하지 않았다.

ll(차선) 인식이 프레임 전체에서 실패해 da가 두 차선을 뭉텅하게 병합하는 경우, 얇은 다리
구조가 없어 침식(erosion)으로 못 끊어낸다 — 대신 ll 잔상(decay, 최근 확실했던 픽셀을 감쇠
유지)과 기대 차로폭 기반 가상경계(`_clip_da_by_ll()`)로 대응한다.

### YOLO 모델
`perception/yolo_*.py`가 `yolo_ros/*.onnx`에서 상대경로로 로드(가중치는 이 저장소에
`cone_best_n.onnx`만 포함, 나머지는 루트 README 안내대로 별도 다운로드):
- `cone_best_n.onnx` — 라바콘(B1) 검출
- `target_vehicle_best.onnx` — 방해차량(B3) 후방 검출
- `signal_state_best_n.onnx` — 신호등 보드 위치 + 색상 상태 동시 예측(YOLOv8 파인튜닝)

`_active_yolo_stage()`가 현재 `Phase`에 필요한 모델만 가동해 연산량을 줄인다. YOLO는 신호등/
B1 진입/B2·B3 트리거 확인용으로만 쓰이고, 차선/DA 자체는 YOLO를 쓰지 않는다.

### 신호등 인식
옛 HSV/Hough Circle 기반 로직은 삭제됐고, `yolo_signal_state.py` 단일 모델
(`signal_state_best_n.onnx`)이 보드 위치+색상 상태를 함께 예측한다. `SIG_CONFIRM_FRAMES=5`
프레임 연속 확인 후 확정. `signal_left_confirmed`는 초록+빨강 동시 점등(좌회전/지름길 옵션),
`signal_straight_confirmed`는 직진 신호.

### 라이다 트리거
- `perc_lavacon_trigger()` — B1(라바콘) 진입 트리거(좌우 클러스터 동시검출)
- `perc_checker_pillar()` — 지름길 진입 게이트(좌우 기둥쌍 검출)
- `perc_obstacle_cut_trigger()` — B2/B3 실제 회피 트리거(라이다+YOLO 이중확인)

---

## 7. 조향 컨트롤러 (Pure Pursuit)

차선 추종(`_lane_steer()`)은 `PurePursuitController`(`controller/pure_pursuit.py`,
기하학적·속도/커브 적응형 lookahead) 하나로 고정. 튜닝값은 전부 `config.py`의 `PP_*`.
`_lane_drive()`(S1 차선주행)와 `_handle_lavacon()`(B1, 조향 파라미터를 차선주행과 동일하게
씀)이 전부 이 컨트롤러를 거친다. LQR 컨트롤러는 제거됨(보존 안 함).

핵심 파라미터(현재값, `config.py`):
- `PP_WHEELBASE_PX = 49.64` — 그리드서치(`pp_tune_gridsearch.py`)로 재튜닝된 값(물리 기반
  계산값 67.0이 아님, §11 참고)
- `PP_TUNE_ACTIVE_PRESET = 'speed15'` — `SPEED_NORMAL=15.0`에 맞춰 그리드서치로 구한
  프리셋 세트(`PP_ALPHA`/`PP_LD_FLOOR_PX`/`PP_DX_DEADZONE_PX` 등 일괄 적용)
- `SPEED_NORMAL = 15.0`, `SPEED_CORNER_MIN = 8.0`
- 코너 진입 시 회전반경 기반 감속 + curvature 기반 lookahead 축소가 기본 동작에 포함돼
  있음(둘 다 pure_pursuit 전용)

라바콘 push 중(B1)에는 `PP_WHEELBASE_PX_LAVACON=20.0` 등 라바콘 전용 파라미터 세트를 별도로
쓴다(`config.py` 하단 라바콘 전용 절 참고).

---

## 8. VESC / IMU 연동

### VESC (실측 속도)
구동모터 실회전속도(VESC 홀센서)를 ROS1 변환 노드(`vesc_speed_bridge.py`, 워크스페이스
바깥 noetic_ws에서 별도 실행)가 `std_msgs/Float32`(`/vesc_speed_erpm`)로 publish →
`ros1_bridge`로 수신 → `cb_vesc()`가 `VESC_SPEED_TO_ERPM_GAIN=4614.0`(vesc.yaml 실측값)로
나눠 `self.v_mps`로 변환. 속도 적응형 lookahead / `EncoderPoseEstimator`에 사용.
생존 확인은 `DEBUG_VIZ_VESC` → `vesc_debug` 창(빨강 NEVER_RECEIVED/주황 STALE/초록 LIVE).

### IMU (SparkFun 9DoF Razor IMU M0)
`xycar_imu` 패키지(`xycar_device/xycar_imu/`)가 `/imu`를 publish. `cb_imu()`가
`imu_yaw`(바퀴 카운트/랩 카운트용)와 `imu_yaw_rate`+`_imu_t`(코너 감쇠 보강용,
`IMU_STALE_SEC=0.5`)를 뽑는다. `_update_lap()`은 휠 인코더가 아니라 IMU yaw 누적으로 바퀴를
센다 — IMU가 죽어있으면(센서 하드웨어 장애, 코드 버그 아님) `imu_yaw`가 안 바뀌어 랩 카운트가
0에 고정된 채 조용히 멈춘다. 에러 로그 없이 "값이 안 움직인다"로만 드러나므로 `[LAP] 누적=`
로그로 간접 확인.

두 센서 모두 `_*_t` 타임스탬프 + `*_STALE_SEC` 가드 패턴(`_vesc_live()` 등)으로 생존을
체크한다 — 콜백이 안 불려도 크래시가 아니라 값이 직전값에 고정되는 형태로만 나타나므로, 새
센서를 추가할 때도 이 패턴을 따를 것.

---

## 9. 디버그 시각화

모든 `DEBUG_VIZ_*` 플래그는 기본 `False`(경기 준비 상태) — 창 코드는 남아있고 `config.py`
"5. 디버깅 ON/OFF" 절에서 opt-in.

| 기능 | 스위치 |
|---|---|
| 신호등 | `DEBUG_VIZ_SIGNAL` |
| 차선 — `dl` 백엔드 | `DEBUG_VIZ_DL_LANE` |
| 차선 — `hough` 백엔드 | `DEBUG_VIZ_HOUGH_LANE` |
| 차선 — `classic_cv` 백엔드 | `DEBUG_VIZ_LANE` |
| 정지선 | `DEBUG_VIZ_STOPLINE` |
| 라이다 BEV(장애물) | `DEBUG_VIZ_LIDAR` |
| 라이다 BEV(라바콘) | `DEBUG_VIZ_LAVACON` |
| da 근접 컷(B2/B3) | `DEBUG_VIZ_OBSTACLE_CUT` |
| 좌회전 통합 | `DEBUG_VIZ_LEFT_TURN` |
| VESC 생존 | `DEBUG_VIZ_VESC` |
| 조향 컨트롤러 상태 | `DEBUG_VIZ_STEER` |

---

## 10. 알려진 한계

- **라이다 장착 각도 드리프트 의심, 미해결.** `LIDAR_ANGLE_OFFSET_DEG=80.0`은 2026-07-22
  실측값(정면에 사람을 세우고 컴퍼스로 확인)이지만, 이후 `measure_lidar_camera_offset.py`로
  재확인했을 때 실제 최근접점이 인덱스 80이 아니라 88~89(약 6.5도 차이)에서 찍힌 단발
  관찰이 있었다. 마운트가 미세하게 틀어졌을 가능성 — `config.py`는 아직 갱신 안 됨. 이 값을
  바꾸면 B1/B2/B3 좌우 판정 전부에 영향이 크므로 재측정 시 반드시 재검증할 것.
- **`speed=20` 이상은 `METERS_PER_SPEED_UNIT` 상수를 못 쓴다.** 5~15 구간에서 재측정한
  `METERS_PER_SPEED_UNIT≈0.0848`은 이 구간 반복측정 기반 회귀값이고, `speed=20` 부근부터
  이미 비선형(견인력 포화) 구간에 들어섰을 가능성이 있어 단일 상수로 표현할 수 없다.
  `speed=25`는 하드웨어 신뢰성 문제(배터리/ESC/모터 의심)로 측정 자체가 보류 중.
  (`실제속도측정.md` §0.1 참고)
- **출발 초반 "틱틱거림/힘딸림"은 미해결.** 배터리 LVC 트립 가설(`SPEED_ACCEL_STEP` 완화)과
  정지마찰 체류시간 가설(`SPEED_KICK_START` 도입) 둘 다 시도했으나 실차 재검증 결과 개선
  체감이 없었고, 오히려 출발 성공률이 급락한 세션도 있었다. 소프트웨어 레버는 이미 적용된
  상태이므로, 원인은 하드웨어(배터리 내부저항 열화/ESC 보호임계값/기계적 걸림/커넥터
  접촉불량) 쪽일 가능성이 높다고 재평가됨 — 정상 배터리팩 교체 A/B 비교가 다음 점검 순서.
- **da/BEV 관련 다수 파라미터가 실차 미검증 추정치.** `OBSTACLE_CUT_TRIGGER_Y_HALF_M`,
  `OBSTACLE_CUT_HALF_WIDTH_SCALE_FIXED`, `B2_TO_B3_DELAY_SEC`, `LAVACON_KICK_ANGLE_DEG`의
  부호(방향), `SHORTCUT_EXIT_DIST_M`/`SHORTCUT_EXIT_KICK_*` 등은 코드 리뷰 수준 또는
  체감 기반 추정치로 설정된 뒤 아직 완전히 실차로 검증되지 않았다. 대회 전 최종 실차
  주행에서 각 값이 기대대로 동작하는지 재확인 필요.
- **B2가 항상 B3보다 먼저 나온다는 트랙 순서 전제.** `_b3_armed()`의 3초 지연 가드와
  좌우 교차검증 veto 로직 모두 "B2가 B3보다 항상 먼저"라는 트랙 순서 전제 위에서 동작한다.
  전제가 실제 트랙과 다르면(예: 배치가 바뀌면) 오분류 가드가 반대로 오작동할 수 있다.
  현재 코드는 이 전제를 최종값으로 채택한 상태(과거 한 차례 반전 후 다시 원복됨).
- **IMU `/imu` 연동은 launch 파일에서 여전히 주석 처리돼 있을 수 있다.** `xycar_imu`
  패키지 자체는 저장소에 편입돼 있으나, `launch/track_drive.launch.py`의 `imu_include`가
  주석 처리된 상태라면 빌드해도 `/imu`가 노드에 안 들어온다 — 실차 배포 시 반드시 확인.

---

## 11. 캘리브레이션/실측값

**실측값**(직접 측정)과 **설계값**(값은 있지만 임의로 고른 것)을 구분합니다 — 헷갈리면 이미
검증된 값으로 착각하고 재검증을 건너뛸 수 있습니다.

### 도로/차량 치수 (실측)
| 상수 | 값 | 근거 |
|---|---|---|
| `LANE_WIDTH_M` | 0.4m | 흰선~흰선(도로 전체폭) 80cm 실측, 노란 중앙선이 정중앙 → 차선 1개 폭 40cm |
| `VEHICLE_WIDTH_M` | 0.31m | xycar 본체 실측(세로64×가로31×높이20cm) |
| `VEHICLE_LENGTH_M` | 0.64m | 위와 동일 실측 |
| `OBSTACLE_VEHICLE_WIDTH_M` | 0.24m | 고정장애물(20×41×16cm)/방해차량(28×54×19cm) 실측폭 평균 `(0.20+0.28)/2` |
| `WHEELBASE_M` | 0.335m | 줄자로 앞바퀴-뒷바퀴 축간거리 실측(옛 이름 `LQR_WHEELBASE_M`). `planner/hybrid_astar.py`의
wheelbase 기본값과 반드시 같은 값을 유지할 것 |

### 라이다
`LIDAR_ANGLE_OFFSET_DEG=80.0` — 실측값(§10 "알려진 한계"에 드리프트 의심 미해결 사항 있음).

### DL 백엔드 BEV 캘리브레이션
`bev_point_picker.py`로 라이브 카메라에서 직접 클릭한 4점(원본 640×480 기준):
TL(246,257)/TR(455,257)/BR(635,333)/BL(60,333), 실측 W=0.8m(좌우 백선 간격)/L=1.0m.
`DL_PIXELS_PER_METER=200.0`은 **설계값**(임의 선택, 실측 아님) — 목적 캔버스를 1m=200px로
정의. 카메라 재장착/진동 시 `bev_point_picker.py`로 재측정 필요.

da/ll 튜닝값(BEV 좌표계 기준, 반차로폭/실측 라인두께로 재산정): `DL_DA_MIN_COMPONENT_AREA=1560`,
`DL_SLICE_OUTLIER_MAX=40`, `DL_STABLE_JUMP_MAX=20`, `DL_LL_CLIP_MARGIN_PX=8`.

### 속도 단위 ↔ m/s 환산
`METERS_PER_SPEED_UNIT=0.0848` — `measure_speed_calibration.py`(줄자+시간차 실측)로
speed 5/10/15를 각 3~4회 반복 측정한 원점고정 회귀. 데드존은 사실상 0. **`speed≥20`에는
쓸 수 없음**(§10 참고). VESC(`v_mps`) 실측 대조 결과 항상 실제보다 5~11% 높게 보고됨 —
`VESC_SPEED_TO_ERPM_GAIN` 재보정 여부는 미결정.

`VESC_SPEED_TO_ERPM_GAIN=4614.0` — VESC 드라이버 `vesc.yaml`의 `speed_to_erpm_gain` 값
그대로(실차 확인).

### 아직 미실측인 값
| 상수 | 상태 |
|---|---|
| `PIXELS_PER_METER`(전역) | 0.0(미실측) — `DL_USE_BEV` 검증 완료 시 `DL_PIXELS_PER_METER`로 채울 것 |
| `PP_MIN_LOOKAHEAD_PX`/`PP_WHEELBASE_PX`/`PP_LOOKAHEAD_BASE_PX` 등 | 그리드서치로 재튜닝된 값이지 물리 실측값 아님 |

### 조향 컨트롤러 물리 기반 계산
`WHEELBASE_M(0.335) * DL_PIXELS_PER_METER(200) = 67.0`이 물리 기반 `PP_WHEELBASE_PX` 계산값
이지만, 현재 실사용값은 그리드서치로 재튜닝된 `49.64`다(§7 참고) — 두 값이 다른 것은 의도된
것이며, 물리 기반 값이 "정답"이라고 착각해 되돌리지 말 것.
