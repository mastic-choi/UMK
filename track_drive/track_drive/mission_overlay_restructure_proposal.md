# Mission FSM → S1 상시주행 + 오버레이 구조 전환 (미구현, 2026-08-20)

**상태: 계획 단계 — 코드 수정 시작 전, 사용자와 항목별로 맞춰보는 중.**
`da_based_b2b3_proposal.md`(2026-08-15)에서 미결정으로 남겨뒀던 "TargetPassing(override)을
버리고 da를 신뢰한다"는 B/C안이 이번 요청으로 사실상 재점화된 것 — 그 문서의 "다음 단계"와
이어서 읽을 것.

## 배경 — 사용자가 설명한 새 흐름 (원문 정리)

> S0_WAIT_GREEN 이후 초록불 들어오면 S1 전환 → 직진하다 라바콘 트리거 만나면 B1, 끝나면 S1
> 복귀 → (이후 장애물 검출은 B2 고정장애물) 고정장애물 만나면 B2 회피, 종료 후 S1 복귀 →
> 차량 검출해서 이동장애물 만나면 B3, 이후 다시 S1 복귀. **다만 B가 켜질 때 이전 코드처럼
> 조향을 덮어쓰는 게 아니라, 기존 S1 주행 위에 "컷"을 만드는 식의 추가 로직만 수행.**
> 이후 다시 신호등 인식 → 직진이면 이전 단계 반복, 좌회전이면 좌회전 진입로직 후
> S1라인팔로우, 좌회전 종료로직 후 S1라인팔로우로 복귀.

## 현재 코드와 비교했을 때 핵심 변화 포인트

1. **B1/B2/B3가 override → overlay(cut) 방식으로 바뀐다.**
   지금 `apply_behavior_override()`(`track_drive.py:2854`)는 `_handle_lavacon()` /
   `_handle_fixed_obstacle()` / `_handle_overtake()`를 호출해서 **`ctrl_angle`/`ctrl_speed`를
   통째로 다시 계산·대입**한다. `_s1_lane_follow()`도 B1 확정(`_lavacon_engaged`) 또는
   `_obstacle_active`/`_overtake_active`면 **`_lane_drive()` 자체를 건너뛴다**
   (`:1790-1793`). 즉 지금은 "S1 주행이 멈추고 B가 대신 조향한다" 구조.

   사용자가 말한 새 구조는 이미 저장소에 절반 구현돼 있다 — **`ENABLE_OBSTACLE_CUT`
   ("da 근접 컷") 메커니즘**(`perception/dl_lane.py` `_clip_da_by_obstacle()`,
   `set_obstacle()`)이 정확히 이 패턴이다: `track_drive.py`는 트리거 판정만 하고
   (`perc_obstacle_cut_trigger()` → `_update_obstacle_cut_hold()`), 실제 "컷"은
   **DL 백엔드에 넘겨서 da 마스크 자체를 깎는다** — 그러면 `_lane_drive()`가 평소와
   똑같이 돌면서 이미 깎인 da 위에서 알아서 피해간다. `ctrl_angle`을 어디서도 직접
   대입하지 않는다.

   → **제안**: B2(고정장애물)/B3(이동장애물)를 `TargetPassing`/`_handle_fixed_obstacle`/
   `_handle_overtake` override 대신, obstacle-cut과 같은 "da를 깎아서 S1이 알아서
   피하게" 방식으로 흡수. `da_based_b2b3_proposal.md`의 B안이 이미 이 방향을
   제안했었고(`_da_avoidance_failed()`를 실제 조건으로 바꾸자는 것), obstacle-cut이
   그 실측/구현을 먼저 끝내놓은 셈이라 합칠 여지가 커 보인다.

2. **좌회전(S2 커밋 이후 / S3 끝)도 같은 원리로 "S1 위 오버레이"가 될 수 있다.**
   지금 `_do_left_turn()`은 S2/S3 핸들러 안에서 매 틱 `_lane_drive()`를 아예 안 부르고
   `ctrl_angle`/`ctrl_speed`를 고정값(`TURN_ANGLE`/`TURN_SPEED`)으로 직접 대입하는
   open/closed-loop 전용 루틴이다. 사용자 설명대로면 좌회전도 "진입 로직 → (S1
   라인팔로우가 계속 돌아가는 채로) → 종료 로직"으로 바뀌어야 하는지, 아니면 신호
   대기(완전정지)처럼 "차선을 따를 게 없는 구간"이라 지금처럼 별도 전용 루틴으로
   남아야 하는지 **불확실** — 아래 열린 질문 참고.

3. **B1(라바콘)은 애초에 da를 안 쓴다.** `Phase` enum 주석(`config.py:50-56`)에 이미
   "라바콘은 완전히 다른 조향 로직으로 갈아타는 것이라 da 신뢰 논리가 적용 안 된다"고
   명시돼 있다. 즉 B1까지 "cut" 방식으로 통일할지, B1은 지금처럼 override로 남기고
   B2/B3만 cut으로 바꿀지도 확인이 필요하다.

## 열린 질문 (코드 손대기 전에 정해야 할 것들)

1. **Mission State(S0~S4) 자체를 줄이나, 그대로 두고 내부 로직만 바꾸나?**
   - A. S0(신호대기)만 별도 상태로 남기고, S1/S2/S3 구분을 없애 사실상
     "S1 상시주행 + 오버레이(신호정지/라바콘컷/장애물컷/차량컷/좌회전)"로 통합.
   - B. S0~S4 상태는 그대로 두되, 각 상태 **내부**에서 override 방식이던 부분(B1/B2/B3,
     좌회전)만 "S1 주행 위 오버레이"로 바꾼다 — 예를 들어 S2/S3에서도 좌회전 구간
     빼고는 지금처럼 `_lane_drive()` 기반.
   - C. B2/B3만 cut으로 바꾸고 좌회전은 지금 구조(전용 open-loop 루틴) 그대로 유지.
2. **좌회전 중에도 `_lane_drive()`(비전 차선인식)를 실제로 돌릴 수 있나?**
   지금 좌회전 구간(교차로 분기 직후, 지름길 진출부)은 차선 자체가 아직 안 보이거나
   애매한 지점이라 일부러 비전을 끄고 IMU 헤딩홀드/고정조향을 썼다(`_s2_intersection`
   커밋구간 주석, `_s3_shortcut` `SHORTCUT_VISION_CUTOFF_T` 주석 참고). "오버레이"로
   바꾼다고 해도 이 구간에서 비전을 신뢰할 근거가 새로 생긴 게 아니라면, 좌회전은
   내용상 지금처럼 비전을 배제한 전용 로직으로 남아야 할 수도 있다.
3. **B2/B3를 cut으로 바꾸면 `Phase`/`TargetPassing`/`obstacle_controller`/
   `vehicle_controller`(현재 회피궤적 생성기)는 폐기하나, 폴백으로 남기나?**
   `da_based_b2b3_proposal.md` 결론은 "즉시 삭제하지 않고 da 실패 시 폴백으로 남긴다"
   였다 — 이번에도 같은 원칙으로 갈지 확인 필요.
4. **B1(라바콘)도 cut으로 편입하나, override로 유지하나?** (위 3번 항목)
5. **속도 제어는?** 사용자 설명은 조향("컷") 위주였는데, 지금 override 핸들러들은
   속도도 같이 덮어쓴다(`APPROACH_SPEED`, 감속 등). cut 방식으로 가면 속도는 여전히
   S1의 코너감속(`_corner_radius_speed_scale` 등) 로직에 맡기는 건지, 별도 속도
   보정(예: `da_based_b2b3_proposal.md` 2번 항목의 "target_speed_est 기반 부스트")이
   필요한지 정해야 한다.
6. **`RESET_PHASE_EACH_LAP`/`_mark_behavior_passed`/`Phase.DONE`처럼 "한 바퀴 안에서
   순서를 추적"하던 장치들은 오버레이 구조에서도 그대로 필요한가?** cut 방식은 매
   프레임 독립 판단이라 원래 이 추적이 왜 필요했는지(커밋형 기동 방지)가 없어지는데,
   그래도 "이미 지나친 장애물을 다시 컷 안 하게" 같은 이유로 최소한의 상태는 남아야
   할 수 있다.

## 다음 단계

정답을 먼저 정하지 않고, 위 열린 질문을 하나씩 짚어가며 확정되는 대로 이 문서에
반영 → 확정된 부분부터 순서대로 구현. 우선순위 제안(동의 필요):

1. 큰 틀(질문 1) 먼저 — Mission State를 얼마나 남길지부터.
2. B2/B3 cut 전환 (obstacle-cut과 통합 여부, 질문 3) — 이미 절반 구현된 메커니즘이라
   가장 착수가 쉬움.
3. B1(라바콘) cut 여부 (질문 4).
4. 좌회전 처리 방식 (질문 2) — 비전 신뢰 여부에 달려있어 가장 실차 검증이 필요한 부분.
5. 속도(질문 5), 순서추적 상태(질문 6) 정리.
