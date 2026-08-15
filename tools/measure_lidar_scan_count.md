# `measure_lidar_scan_count.py` — 라이다 스캔 포인트 개수(n) 실측 도구

**상태: 2026-08-15 작성, 아직 실차 미실행.** `local_costmap_proposal.md` 선행작업
1순위("라이다 인덱스↔각도 공식 버그 수정") 착수 전에 먼저 확인해야 하는 전제(n이
정말 항상 500인지)를 자동으로 재는 도구.

## 왜 필요한가

`perc_obstacle()`/`perc_lavacon_trigger()`(track_drive.py)의 라이다 인덱스→각도
변환이 아직 `m = min(len(ranges), 360)`으로 **n=360을 가정한 옛 공식**을 쓴다.
2026-08-14에 `tools/measure_lidar_camera_offset.py`로 실측하다가 이 라이다가 실제로
한 바퀴에 **n=500** 포인트를 찍는다는 게 우연히 발견됐다(README §2.30) — 정면
(index≈80) 근처에서는 옛 공식도 우연히 거의 맞지만, 정면에서 멀어질수록(=장애물이
옆에 있을수록) 각도 오차가 커진다. 좌/우 장애물 판정, 라바콘 좌우 클러스터 판정
등 "정면이 아닌 각도"를 보는 모든 라이다 판정에 실차 배포 이후 계속 영향을 줬을
가능성이 있다.

프로덕션 코드를 고치기 전에, **n이 이 라이다에서 정말 항상 500으로 일정한지**부터
확인해야 한다 — 스캔마다 흔들리면 상수로 못 박고 매번 `len(ranges)`를 읽어써야
하기 때문.

## 도구가 하는 일

- `/scan`(`sensor_msgs/LaserScan`)을 지정한 시간(기본 10초) 동안 구독해서, 매
  메시지의 `len(ranges)`를 전부 기록한다.
- 스캔 주기(Hz)도 같이 계산해 보여준다(참고용).
- 관찰된 n값 분포(히스토그램)를 출력하고, **n이 하나로 고정돼 있으면 OK**, 여러
  값이 섞여 있으면 경고를 띄운다.
- 마지막에 `perc_obstacle()`/`perc_lavacon_trigger()`에 적용할 수정 공식(옛
  공식 vs 새 공식)을 그대로 출력한다 — `tools/measure_lidar_camera_offset.py`의
  이미 고쳐진 `_index_to_deg()`와 동일한 공식.

라이다 드라이버 노드만 떠 있으면 되고(`track_drive` 전체 launch 불필요), 클릭이나
수동 입력 없이 그냥 구독만 하는 순수 읽기 도구라 사람이 할 일이 거의 없다 — 라이다를
켜두고 실행만 하면 됨(가능하면 차를 실제로 이동시키며/정지 상태에서 각각 한 번씩
돌려보는 것도 참고가 될 수 있음, 필수는 아님).

## 사용법

```bash
# 1) 라이다 드라이버만 띄운다(track_drive 전체 launch 불필요)
# 2) 실행
python3 tools/measure_lidar_scan_count.py
# 측정 시간을 바꾸고 싶으면
python3 tools/measure_lidar_scan_count.py --duration 20
# 토픽 이름이 다르면
python3 tools/measure_lidar_scan_count.py --scan-topic /scan
```

## 출력 예시 (가상)

```
=== 라이다 스캔 포인트 개수(n) 실측 — 10초간 /scan 구독 ===

총 스캔 수: 82  (약 8.2Hz)
n(=len(ranges)) 분포:
  n= 500 : 82회 (100.0%)

=== 결과 ===
OK — n이 이번 측정 내내 500으로 일정합니다. ...

track_drive.py에 적용할 수정 방향 (perc_obstacle()/perc_lavacon_trigger() 둘 다):
  ...
```

## 알려진 한계 / 다음 단계

- 이번 측정 세션 안에서의 일관성만 확인한다 — **재부팅했을 때도, 다른 라이다
  개체(교체 시)에서도 n=500인지는 별도로 다시 확인 필요**.
- n이 일정하다고 확인돼도, 이 도구가 프로덕션 코드를 직접 고쳐주지는 않는다 —
  출력된 수정 공식을 `track_drive.py`의 두 함수에 실제로 반영하는 건 별도 작업.
  특히 `perc_obstacle()`은 각도 배열(`self._obs_cos`/`_obs_sin`)을 `hasattr`
  가드로 최초 1회만 캐싱하므로, n이 스캔마다 안 바뀐다는 전제가 이 코드에도
  깔려있다는 점을 같이 고려해서 고칠 것.
- 이 확인이 끝나야 `tools/measure_lidar_camera_offset.py`(라이다-카메라 오프셋
  실측, 선행작업 2순위)의 결과도 더 신뢰할 수 있다 — 순서를 바꾸지 말 것.
