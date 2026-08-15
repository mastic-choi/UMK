# da 기반 회피로 formal B2/B3(TargetPassing)를 대체하는 방안 (미구현, 2026-08-15)

**상태: "해결 방향 B안"(Phase 통합)은 2026-08-15에 구현 완료 — 실차 미검증
(README §2.34 참고). 나머지(`_da_avoidance_failed()` 실제 조건 교체, 속도부스트, B3에
같은 게이트 이식)는 여전히 미구현.** avoid-hold 개선(§2.33) 작업 중 나온
후속 논의 — "지금 da 안전마진(§2.30)+avoid-hold(§2.32/§2.33)로 이미 회피가 되고 있는데,
`TargetPassing`(B2/B3 정식 상태기계)이 굳이 따로 필요한가?"를 검토한 기록.
`avoid_hold_improvement_proposal.md`와 성격은 같지만 주제가 avoid-hold 자체가 아니라
"B2/B3 미션을 어떤 아키텍처로 처리할 것인가"라 별도 문서로 뗐다.

## 배경 — 세 가지 반박과 정리

`avoid_hold_improvement_proposal.md`에서 "TargetPassing 없인 B3를 못 채운다"고 정리했던
근거 세 가지를, 이 트랙이 **2차선뿐**이라는 전제로 다시 검토했다.

### 1. 방향 — 대부분 저절로 해결됨 (조건부)

도로가 2차선이고 장애물이 한쪽을 막으면, da는 원래 하나로 이어진 도로에 장애물 자리만
구멍 난 형태가 된다. `_apply_vehicle_margin()`(차폭 침식, §2.30)까지 걸리면 막힌 차선
쪽에 남는 좁은 여유가 침식으로 완전히 사라져 **열린 차선 하나만 남는다** — 그러면
`_largest_da_component()`가 뭘 고르든 선택지가 하나뿐이라 방향이 저절로 정해진다.
밴드별 무게중심 계산도 "픽셀이 있는 쪽으로 쏠린다"는 성질상 자연히 열린 쪽으로 붙는다.

**단, 두 전제가 깨지면 다시 애매해진다:**
- 침식량이 부족하면(`DL_DA_VEHICLE_MARGIN_M` 실측 미검증) 막힌 차선에 "간신히 통과
  가능해 보이는" 여유가 남아 다시 두 갈래로 보일 수 있음.
- 방해차량이 차선을 오가는 전환 순간(규정상 "1·2차선을 오간다")엔 양쪽 다 걸치거나
  애매한 모양이 나올 수 있음 — 이건 avoid-hold 적용4(`choose_side()`가 0을 반환하면
  강제 감속)로 이미 커버해둔 정확히 그 상황이다.

### 2. 속도 — `target_speed_est` 기반 부스트로 구현 가능 (미구현)

`target_speed_est`(=`v_mps + obstacle_rate`, 이미 계산돼 있음)가 타겟의 실제 속도
추정치이므로, 옆을 지나는 동안 이 값이 낮으면 목표속도를 올리고 이미 충분히 앞서면
그대로 두는 로직으로 `TargetPassing.alongside_speed`(고정값)를 대체할 수 있다 — 오히려
고정값보다 나을 수 있다. `obstacle_rate` 노이즈(특히 최근접 통과 순간)에 상한/스무딩이
필요하고, 다른 감속 캡(`SPEED_CORNER_MIN` 등)과 같은 `min()`/`max()` 캡 패턴으로
얹으면 된다. **아직 미구현.**

### 3. 복귀 타이밍 — 절반만 맞음

"추월 후 da가 원래대로 돌아오면 끝"은 정확히 avoid-hold를 만든 이유였던 문제와
같다 — 카메라가 차 앞코에 달려있어 장애물을 지나친 순간(아직 옆/뒤에 붙어있는 순간)
그 장애물이 화면에서 사라지고 da가 즉시 원래 폭으로 돌아온다. `TargetPassing.RETURN`
처럼 횡위치를 명시적으로 수렴시키는 별도 상태는 필요 없지만(da 중심선이 열리면 조향은
자연히 따라감), **언제** 복귀를 허용할지는 여전히 라이다 거리 게이팅(avoid-hold)이
필요하다 — 이미 구현돼 있음.

**종합**: 방향/타이밍은 이미 구현된 것(da 침식 + avoid-hold)으로 대부분 커버되고,
속도만 미구현 상태. `TargetPassing`의 정교한 3단계(SHIFT/ALONGSIDE/RETURN) 없이도 B3를
실질적으로 커버할 수 있어 보인다는 쪽으로 결론이 기울었다.

---

## 중요 발견 — 이 전환은 이미 코드에 반쯤 설계돼 있었다

`_handle_fixed_obstacle()`(B2)를 다시 보니, **정확히 이 전환을 위한 게이트가 이미
있었다**:

```python
def _handle_fixed_obstacle(self):
    if not self._da_avoidance_failed():
        # da 기반 경로(Mission의 lane-follow 출력)가 알아서 피하고 있다고 신뢰 —
        # TargetPassing으로 덮어쓰지 않고 이번 틱은 그냥 둔다.
        return
    self._run_passing(self.obstacle_controller, 'B2', done_next_phase=Phase.VEHICLE)

def _da_avoidance_failed(self):
    path_broken = (not self.lane_valid) or self.lane_stale
    da_unaware_of_obstacle = True  # ★da가 장애물 인지형이 되면 실제 조건으로 교체
    return path_broken or da_unaware_of_obstacle
```

2026-08-11 당시엔 "da 세그멘테이션이 아직 장애물을 인지 못 한다"는 전제로
`da_unaware_of_obstacle`을 하드코딩 `True`로 박아둬 **B2가 항상 TargetPassing으로
폴백**하게 해뒀다 — 주석에 "da가 장애물 인지형이 되는 날 이 한 줄만 실제 판단 로직으로
교체하면 된다"고 명시까지 돼 있다. §2.30(da 안전마진 침식)과 §2.32/§2.33(avoid-hold)이
바로 그 "장애물 인지형 da"를 만든 작업이었으니, **이 조건을 갱신할 시점이 이미 왔다.**
(B3 쪽 `_handle_overtake()`엔 이런 게이트 자체가 없다 — B2에서 먼저 검증한 뒤 B3에
이식하는 순서가 자연스러워 보인다.)

## 남는 문제 — 그러면 `Phase`(FIXED_OBSTACLE→VEHICLE→DONE)는 누가 넘기나?

**바로 이 지점이 이번 질문의 핵심이다.** `Phase`가 `FIXED_OBSTACLE`에서 `VEHICLE`로,
`VEHICLE`에서 `DONE`으로 넘어가는 유일한 경로는 `_run_passing()`의 `done_next_phase`뿐이다
— 즉 **오직 TargetPassing이 실제로 한 번 SHIFT→ALONGSIDE→RETURN을 완주해야만** Phase가
전진한다. `_da_avoidance_failed()`를 실제 조건으로 바꿔 "da를 신뢰"하기 시작하면
`_handle_fixed_obstacle()`이 `_run_passing()`을 아예 안 부르므로, **`Phase`가
`FIXED_OBSTACLE`에 영원히 멈춘다.**

이게 왜 문제냐면, `run_behavior_fsm()`이 `Phase.VEHICLE`일 때만 `vehicle_trigger`를
본다(`elif self.phase == Phase.VEHICLE: ... behavior_state = B3_VEHICLE if
self.vehicle_trigger ...`) — Phase가 멈추면 실제 방해차량을 만나도 그 트랙 진입 자체가
`Phase.FIXED_OBSTACLE` 브랜치(B2용 트리거 조건)로 계속 걸린다. 지금은 어차피 두 브랜치
다 "da 신뢰 시 아무것도 안 함"으로 수렴하니 **조향에는 영향이 없지만**, 위 2번(속도
부스트)을 "Phase.VEHICLE일 때만" 같은 식으로 얹으면 방해차량을 만나도 그 로직이 영영
안 켜지는 조용한 버그가 된다.

### 왜 원래 `Phase`가 필요했는가

`controller/obstacle_avoidance.py` 상단 주석: "정적/동적 판별을 위한 속도추정은 하지
않는다 — 상위 Phase가 이미 FIXED_OBSTACLE/VEHICLE로 구분해주기 때문이다." —
`TargetPassing`은 한번 시작하면 끝까지 실행하는 **커밋된(committed) 기동**이라, 그
기동을 고를 때 쓰는 정적/동적 분류가 실시간 속도추정(노이즈에 취약)이면 잘못된 기동을
통째로 커밋할 위험이 있었다. 그래서 "트랙 순서는 고정돼 있다"는 가정(Phase)에 기대는
쪽을 택했다.

**da+avoid-hold+속도부스트 방식은 이 위험 자체가 다르다.** 커밋된 기동이 없다 —
매 프레임 `target_speed_est`로 속도를 "연속적으로" 조정할 뿐이라, 순간적으로 값이
튀어도 그 프레임만 속도가 살짝 흔들리는 정도지 "잘못된 기동을 끝까지 실행"하는 위험이
없다. 즉 **원래 Phase가 막아주려던 위험이, 커밋형 기동을 없애면서 같이 사라진다.**
`Phase.LAVACON`은 예외다 — 라바콘은 회피의 "정도" 조정이 아니라 완전히 다른 조향
로직(`_handle_lavacon()`, da를 아예 안 씀)으로 갈아타는 것이라 이 논리가 적용 안 되고,
지금처럼 명시적 게이트가 계속 필요하다.

### 해결 방향 (3안, 미결정)

**A안 — Phase 전진 조건을 TargetPassing 완료와 분리한다.** avoid-hold 자신의
"장애물이 가까웠다가 멀어짐" 감지(적용1의 조기해제 조건과 같은 신호)를 재사용해
"기록장애물 하나를 통과했다"고 판단하면 `Phase.FIXED_OBSTACLE → VEHICLE`을 직접
넘긴다. Phase 개념 자체는 유지하되, 그 전진 주체를 TargetPassing에서 avoid-hold로
옮기는 것.
- 장점: 기존 Phase 기반 구조/로그/`RESET_PHASE_EACH_LAP`를 그대로 씀.
- 단점: "장애물 하나 통과 = obstacle_front가 한 번 True였다가 False로 돌아옴"이라는
  새 판정 기준이 또 하나 필요하고, 이것도 라바콘 잔재물 등에 오검출될 여지가 있다
  (avoid_hold_improvement_proposal.md 문제3과 같은 성격의 리스크).

**B안 — `FIXED_OBSTACLE`/`VEHICLE`을 하나로 합친다.** 두 Phase를 구분할 실익이
없어졌으므로(회피 기동 자체가 동일, `_handle_fixed_obstacle()`의 2026-08-11 주석도
이미 "회피 기동 자체는 고정물이든 차량이든 동일"이라고 명시) `OBSTACLE_ZONE` 하나로
합치고, 정적/동적 구분이 필요한 곳(속도 부스트)은 Phase가 아니라 `perc_obstacle()`이
이미 매 프레임 계산해두는 `obstacle_type`(폭 기반, `OBSTACLE_VEHICLE_WIDTH_M`
임계값 — 이미 센서 기반, Phase와 무관)과 `target_speed_est`로 그때그때 판단한다.
- 장점: Phase-stuck 버그 자체가 구조적으로 안 생김(전진시킬 게 하나로 줄어듦). 정적/
  동적 구분을 라이다 실측(폭)으로 하므로 트랙 순서 가정에 덜 의존.
  `_da_avoidance_failed()`/`da_unaware_of_obstacle` 갱신과 자연스럽게 같이 간다.
- 단점: `obstacle_type` 오분류(README §2.30 실측 메모에 "폭 0.35m 차량이 거리에 따라
  fixed/vehicle로 갈렸다"는 사례가 이미 있음) 위험을 그대로 안는다 — 다만 이건
  "판단"이 아니라 "부스트 세기 조절"에만 쓰이므로(2번 항목과 동일 논리) 오분류의
  대가가 TargetPassing 시절보다 훨씬 작다.

**C안 — Phase는 그대로 두되 사실상 무력화한다.** `Phase.FIXED_OBSTACLE`에 멈춰
있어도 상관없도록, 속도부스트 등 새 로직을 Phase 조건 없이 항상 켜둔다(지금
avoid-hold가 `TEST_DISABLE_B2_B3`/Phase와 무관하게 항상 도는 것과 동일 원칙).
사실상 B안과 결론은 같지만 enum/FSM 구조 자체는 안 건드리는 최소변경.

**추천(미확정) → 채택됨**: B안(또는 C안, 사실상 동급) — Phase가 막아주려던 위험(노이즈 낀
분류가 커밋형 기동을 잘못 고르는 것)이 애초에 커밋형 기동을 없애면서 사라졌으니, 그
위험을 막던 장치(Phase 세분화)도 같이 걷어내는 게 일관적이다. A안은 "Phase를 유지해야
할 이유"가 딱히 남아있지 않은데 그 개념을 지키려고 새 판정 로직을 하나 더 만드는
셈이라 우선순위가 낮다.

**[2026-08-15] B안 구현 완료** — README §2.34 참고. `Phase.FIXED_OBSTACLE`/`VEHICLE`을
`Phase.OBSTACLE_ZONE` 하나로 합치고, 정적/동적 구분은 `obstacle_type`(라이다 실측 폭)로
매 프레임 판단하도록 `run_behavior_fsm()`을 재작성했다. Phase 전진은
`_mark_behavior_passed()`가 B2/B3 둘 다 완료됐는지 추적해서 결정한다(순서 무관).
`_da_avoidance_failed()`는 아직 안 건드림 — 아래 "다음 단계"는 그대로 유효하다.

---

## 다음 단계 (미착수)

착수 순서 제안:
1. B2에서 먼저 `_da_avoidance_failed()`의 `da_unaware_of_obstacle`을 실제 조건으로
   교체(예: `not DL_DA_APPLY_VEHICLE_MARGIN`처럼 "안전마진이 꺼져있으면만 신뢰 안 함"
   정도의 최소 조건)하고 실차에서 B2만 먼저 검증.
2. 그 검증과 함께 Phase 처리(B/C안 중 택1)도 같이 반영 — B2 하나만으로는 Phase가
   막히는지 확인이 안 됨(막힌 채로도 B2까진 정상 동작하는 것처럼 보일 수 있어서, 반드시
   그다음 방해차량 조우까지 실차에서 재현해봐야 함).
3. 속도부스트(2번) 구현 + B3에도 같은 `_da_avoidance_failed()`류 게이트 이식.
4. `TargetPassing`/`_run_passing()`은 즉시 삭제하지 않고 "da가 실패로 판단될 때의
   폴백"으로 계속 남겨둔다(이미 그렇게 설계돼 있음, B2 코드가 그 패턴) — da 경로가
   실차에서 예상보다 불안정하면 언제든 되돌아갈 수 있게.

**착수 조건**: avoid-hold(§2.33)의 1차 실차 검증이 먼저 끝나야 한다 — da 기반 회피
자체의 신뢰도가 아직 실차 미확인 상태에서 그 위에 "TargetPassing까지 걷어내기"를
얹는 건 순서가 안 맞는다.
