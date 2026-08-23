# TODO: BEV 근접 밴드 폭 편향 보정

## 증상
DA 디버그창(lavacon_bev/dl_lane) 기준, 실제로는 직선 구간인데 근접 밴드(차량 바로 앞,
BEV 마름모꼴 시작부분)에서 경로(magenta centerline)가 한쪽으로 밀려 보임. 사용자가
스크린샷으로 확인(2026-08-24).

## 원인 (확인됨, dl_lane.py:145-148 기존 주석과 일치)
BEV(IPM) 워프는 카메라 화각이 고정이라 원근투영상 사다리꼴 아래쪽(근접) 양 모서리에
"애초에 대응하는 도로 데이터가 없는" 검은 삼각형 사각지대가 생긴다 — 이건 원근투영의
근본 성질이라 없앨 수 없음(기존 주석이 이미 명시).

문제는 이게 근접 밴드에서 **비대칭적으로** da를 잘라내서, 실제 도로는 직선인데도 그
밴드의 visible 폭 중심이 한쪽으로 편향된다는 것. 기존 안전장치(`near_band_stale`/hold,
`_reject_outliers`의 `protect_indices`)는 "근접 밴드가 아예 안 잡힘"만 방어하고, "잡히긴
하는데 편향된 값"은 못 거른다 — da가 매 프레임 계속 뭔가 있으니 hold/stale이 발동을 안
하고 편향값이 그대로 센터라인 피팅에 들어감.

## 합의된 해결 방향 (AskUserQuestion으로 확정, 2026-08-24)
행(슬라이스)별로 visible da 폭을 기대 차선폭(LANE_WIDTH_M)과 비교해, 비정상적으로
좁으면(=모서리 사각지대에 잘렸다는 신호) 그 행의 centerline x를 `vehicle_center_x`
쪽으로 보정(반은 hold/보정값, 반은 원값 블렌드).

## 구현 지점 (수정, 2026-08-24)
`_slice_edge_midpoints()`는 2026-08-12부로 실차 오검출("S자 좌우 왔다갔다")로 폐기되고
**da 모드는 실제로 `dl_lane.py` `_da_slice_centers_windowed()`(cv2.moments 탐색창 기반)를
쓰고 있었음** — 위 원안은 죽은 함수를 대상으로 잡은 오류. 구현은 다음으로 변경:
- `_da_slice_centers_windowed()`에서 cx 계산 직후(`da_lane.py` cx 계산 블록), 근접 밴드
  (`i < self.near_slices`)에 한해 탐색창과 무관한 **전체 행**의 nonzero 컬럼 폭을 따로
  구해 기대폭과 비교, 좁으면 `self.vehicle_center_x` 쪽으로 블렌드.
- 새 config 상수 `DL_DA_NEAR_WIDTH_MIN_RATIO=0.7`, `DL_DA_NEAR_WIDTH_BLEND_MAX=0.5`
  (둘 다 실차 미검증 추정치).
- 블렌드: `bw = min(1 - visible_w/min_w, 1) * DL_DA_NEAR_WIDTH_BLEND_MAX`,
  `cx = cx*(1-bw) + vehicle_center_x*bw`.

## 검증 방법
- `python3 -m py_compile` 문법 체크까지만 여기서 가능(ROS2 툴체인 없음).
- 실차에서 lavacon_bev/dl_lane 디버그창으로 직선 구간 근접 밴드가 안 밀리는지 확인 —
  코너 진입 시 정상적으로 휘는 건 유지되는지도 같이 확인할 것(과보정 주의).

## 상태
구현 완료(2026-08-24, `_da_slice_centers_windowed()` + config 상수 2개), py_compile 통과.
실차 검증 아직 안 됨 — lavacon_bev/dl_lane 디버그창으로 직선 구간 근접 밴드 확인 필요.
