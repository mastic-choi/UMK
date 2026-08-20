# 상태전환 로직 실차 개별 테스트 체크리스트 (2026-08-20)

**상태: 진행 중.** `mission_overlay_restructure_proposal.md`(override→overlay 구조 전환)
논의는 잠시 보류 — 그 전에 지금 코드(특히 오늘 저녁 병합된 `_s0_signal()` 통합,
VESC거리 기반 좌회전)가 실차에서 각 구간별로 실제로 도는지부터 순서대로 확인한다.
구조 개편 논의는 여기서 실측/문제를 모은 뒤 이어서 진행.

## 공통 준비

- `config.py` 현재값: `ENABLE_BEHAVIOR=True`, `TEST_DISABLE_INTERSECTION=False`,
  `TEST_DISABLE_B2_B3=False`, `TEST_FORCE_BEHAVIOR=False` — 전체 파이프라인이 이미 다
  켜진 상태. 아래 항목을 하나씩 검증하다가 특정 구간이 자꾸 다른 구간과 얽혀 원인
  파악이 어려우면, 해당 항목의 "격리하려면" 메모대로 임시로 다시 좁혀도 됨(테스트
  끝나면 원복 잊지 말 것).
- `DEBUG_LOG=True`(기본) — 0.5초마다 `[mission_state|behavior_state|phase]` 헤더로
  현재 상태가 터미널에 찍힌다. 각 항목 테스트 때 이 로그와 아래 지정한 디버그창을
  같이 켜고 영상/로그를 남겨두면 나중에 다시 판독 가능.
- 항목마다 "☐ 결과"란에 통과/실패와 관찰 내용을 짧게 채워나갈 것 — 이 문서가 그대로
  실측 기록이 된다.

---

## 1. 출발 — `S0_SIGNAL` 최초 진입 → 직진 커밋 → `S1` + Behavior 활성화

**무엇이 바뀌었나:** 정지 판정이 별도 타이머 없이 `_s0_signal()` 진입 시 기본값(신호
미확정=정지)으로 처리되도록 바뀜(`APPROACH_TIME` 감속시퀀스 폐지). 커밋구간 거리적산도
`_speed_mps_fallback()`으로 일반화됨 — 둘 다 이번 병합 이후 실차 미검증.

**확인할 것:**
- 출발선에서 신호 미확정 동안 완전정지 유지되는지(오검출로 조기 출발 안 하는지).
- 직진 초록 확정 후 `S2_COMMIT_DIST_M`(≈1m)만큼 비전 무시하고 그냥 직진하다가 `S1`로
  넘어가는지 — 이 구간이 출발선처럼 짧고 직선인 곳에서도 위화감 없이 지나가는지.
- `S1` 진입과 동시에 `_behavior_enabled=True`로 라바콘부터 바로 시작되는지.

**디버그창:** `DEBUG_VIZ_SIGNAL=True`(기본 켜짐), `DEBUG_LOG_SIGNAL=True`(기본 켜짐).
필요하면 `DEBUG_VIZ_SIGNAL_DETAIL=True`로 후보탐색 과정까지.

☐ 결과:

---

## 2. `B1_LAVACON`

**무엇이 바뀌었나:** 이 구간 자체는 이번 병합의 영향을 안 받음(박스 스택 페어링,
`LAVACON_STEER_MODE_DA_PUSH` 등은 그대로) — 다만 `ENABLE_BEHAVIOR`가 최근에야 다시
`True`로 복원됐으니(§ 이전 대화) 게이팅이 실제로 잘 열리는지부터 확인.

**확인할 것:**
- 좌우 라이다 클러스터 동시검출로 `B1` 진입(latch)되는지, 우측 콘 연속 미검출로
  정상 종료돼 `Phase.OBSTACLE_ZONE`으로 넘어가는지.

**디버그창:** `DEBUG_VIZ_LAVACON=True`, 필요시 `DEBUG_VIZ_LAVACON_SHOW_PATH=True`,
`DEBUG_VIZ_YOLO_CONE=True`(YOLO 콘 이중확인 확인용).

☐ 결과:

---

## 3. `B2_OBSTACLE` (고정장애물)

**확인할 것:**
- `obstacle_type` 판정(라이다 실측 폭)이 고정장애물을 제대로 'fixed'로 분류하는지.
- `_da_avoidance_failed()`가 지금 `da_unaware_of_obstacle=True`로 하드코딩돼 있어
  **항상 TargetPassing으로 override**한다는 점 확인(da 신뢰 여부는 다음 단계 논의
  대상, 지금은 override가 정상 동작인지만 본다).
- SHIFT→ALONGSIDE→RETURN 완주 후 `_mark_behavior_passed('B2')` → 정상적으로 `B0`
  복귀하는지.

**디버그창:** `DEBUG_VIZ_LIDAR=True`(장애물 거리/타입 확인), `DEBUG_VIZ_OBSTACLE_CUT=True`
(기본 켜짐 — B2 override와 별개 메커니즘이니 서로 안 헷갈리게 주의).

☐ 결과:

---

## 4. `B3_VEHICLE` (이동장애물/방해차량)

**무엇이 바뀌었나:** YOLO 방해차량 검출기가 전용 파인튜닝 모델(`target_vehicle_best.onnx`)
로 교체됨(오늘 22:12 커밋, 실차 미검증) — `YOLO_VEHICLE_CLASS_ID`/`CONF_THRESHOLD` 등도
같이 바뀜.

**확인할 것:**
- YOLO가 방해차량을 실제로 잡는지(오탐/미탐 둘 다 확인), 라이다 폭 기준과 AND 결합된
  `vehicle_trigger`가 정상 디바운스되는지.
- `_cross_check_obstacle_motion('B3')` 경고 로그가 남발되지 않는지(방해차량인데
  정지로 오판되면 경고 뜸 — 뜨면 `obstacle_rate` 노이즈 의심).
- `USE_HYBRID_ASTAR_FOR_B3=False`(기본)이라 TargetPassing 경로로만 도는지 확인.

**디버그창:** `DEBUG_VIZ_YOLO_VEHICLE=True`(기본 켜짐, 오늘 추가), `DEBUG_VIZ_LIDAR=True`.

☐ 결과:

---

## 5. `S0_SIGNAL` 재진입 (교차로, 매 랩)

**⚠️ 확정 안 된 설계 질문과 직결 — 일단 "지금 코드가 실제로 이렇게 동작하는지"만 확인,
정지가 맞는지 여부 자체는 `mission_overlay_restructure_proposal.md`에서 이어서 결정.**

**확인할 것:**
- `signal_board_confirmed`(정지선 아님, 신호등 보드 인식)로 `S1`→`S0_SIGNAL` 전환이
  트리거되는지, `SIGNAL_REENTRY_COOLDOWN`(3초) 동안 재진입 오검출 억제가 도는지.
- 재진입 즉시 신호값이 리셋되고(`_change_state`) 다시 완전정지하는지.
- 직진/좌회전 신호에 따라 각각 §1과 동일한 커밋구간 → `S1`복귀 / 좌회전 진입으로
  갈라지는지.

**디버그창:** §1과 동일.

☐ 결과: (여기서 관찰된 내용이 "매랩 정지가 맞는지" 결정에 직접 참고자료가 됨 —
반드시 기록)

---

## 6. 좌회전 진입 (`S0_SIGNAL`→`S3_SHORTCUT`)

**무엇이 바뀌었나:** 이번 병합으로 IMU yaw closed-loop → **VESC 적분거리
(`TURN_DIST_M`=1.0m, open-loop) 로 롤백됨** — IMU 무관하게 고정조향각(`TURN_ANGLE`=-60°)
+ 고정거리로 회전 종료 판정. `TURN_DIST_M`/`TURN_ANGLE`/`TURN_SPEED` 전부 "실차 미검증
초기값"이라고 주석에 명시돼 있음 — 이번이 사실상 첫 실차 검증.

**확인할 것:**
- 1m 이동 동안 -60° 조향을 유지했을 때 실제로 지름길 방향으로 무난하게 꺾이는지
  (너무 짧게/길게 도는지 육안 확인 → `TURN_DIST_M`/`TURN_ANGLE` 조정 근거 수집).
- VESC가 죽어있을 때(`_speed_mps_fallback()`이 명령속도로 폴백) 이동거리 적분이
  터무니없이 틀어지지 않는지.

**디버그창:** `DEBUG_VIZ_VESC=True`(실측 v_mps 확인), `DEBUG_VIZ_IMU=True`(기본 켜짐,
비교용 — 지금은 종료판정에 안 쓰지만 참고로 같이 봄).

☐ 결과:

---

## 7. `S3_SHORTCUT` 주행 (헤딩홀드 구간 포함)

**확인할 것:**
- 진입 초반 `_lane_drive()` 비전주행 → `SHORTCUT_VISION_CUTOFF_T` 이후 IMU 헤딩홀드
  전환이 매끄러운지(전환 시점에 조향이 튀지 않는지).
- `_shortcut_end()`(정지선 감지 또는 `SHORTCUT_MAX_T` 초과)가 실제 합류부에서 제 타이밍에
  걸리는지 — 너무 일찍/늦게 걸리면 `SHORTCUT_MIN_T`/`SHORTCUT_MAX_T` 재조정 필요.

**디버그창:** `DEBUG_VIZ_DL_LANE=True`(기본 켜짐), `DEBUG_VIZ_STOPLINE=True`.

☐ 결과:

---

## 8. 진출 좌회전 (`S3_SHORTCUT`→`S1_LANE_FOLLOW`)

**무엇이 바뀌었나:** §6과 동일하게 VESC거리 기반(`TURN_EXIT_DIST_M`=1.0m,
`TURN_EXIT_ANGLE`=-60°)으로 롤백. 종료 후 `S1` 재진입 시 Behavior는 켜지지 않음(주석
명시 — 지름길 진출 후엔 라바콘부터 다시 시작하지 않는다는 뜻인지 확인 필요).

**확인할 것:**
- §6과 동일한 관찰 + "진출 후 정말 Behavior 없이 순수 차선주행만 하는지"(의도인지
  재확인 — `_change_state`의 `S1_LANE_FOLLOW` 진입부에 `_behavior_enabled=True` 설정이
  없음, `S0_SIGNAL`의 직진 커밋 경로에만 있음).

☐ 결과:

---

## 9. 완주 (`S4_FINISH`)

**보류 — 삭제 여부 논의 중**(직전 대화 참고: 삭제 시 완주 후 정지를 어떻게 처리할지
미정). 지금 구조 그대로는 `TOTAL_LAPS` 초과 시 완전정지가 정상 동작인지만 가볍게
확인해두고, 이 항목의 결론은 구조개편 논의로 넘김.

☐ 결과:

---

## 다음 단계

위 9개 항목 실차 테스트 완료 후 `mission_overlay_restructure_proposal.md`로 돌아가서:
- §5 결과를 근거로 "교차로 재진입 시에도 완전정지가 맞는지" 결정.
- §9(`S4_FINISH`) 삭제 여부/대체 방식 결정.
- 전체 override→overlay 구조 개편 논의 재개.
