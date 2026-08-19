# =============================================================
# perc_lavacon.py — 좌/우 라이다 콘 클러스터 중앙 추종 경로 생성 모듈
#
# [사용법] track_drive.py 에서 import 하여 호출:
#     from perc_lavacon import process_lavacon
#     offset, done, path_m = process_lavacon(self.lidar_ranges)
#
# [2026-08-19] 보로노이(scipy.spatial.Voronoi) 방식 폐기, 클러스터 중앙 추종으로 교체
#   구 방식은 콘 점군 전체로 보로노이 다이어그램을 계산해 그 정점(vertex)들을 중심선으로
#   썼다 — 좌우 콘 개수가 비대칭이어도 위상적으로 그럴듯한 골격이 나온다는 장점은 있었지만,
#   실차 라바콘 테스트를 앞두고 "일반 da(주행가능영역) 중앙 추종 주행과 구조를 통일하자"는
#   요청(경로생성만 라바콘 클러스터 기반으로 바꾸고, 그 뒤(Pure Pursuit 조향)는 그대로 재사용)에
#   따라 훨씬 단순하고 실차에서 검증하기 쉬운 방식으로 교체했다:
#     1) 좌(y>0)/우(y<0) 콘 후보 점을 원본 라이다 인덱스(각도) 인접성으로 묶어(perc_obstacle()/
#        perc_lavacon_trigger()와 동일한 클러스터링 패턴) 콘 하나당 중심점(centroid) 하나로 압축.
#     2) 차량 정면 기준선(y=0, 전방 x축) 위에서 가까운 좌측 콘부터, 그때 아직 안 쓰인 우측
#        콘 중 유클리드 거리가 가장 가까운 것을 찾아 짝짓는다 (_pair_nearest()).
#     3) 각 페어의 중점(midpoint)이 경로점 하나 — 즉 "좌우 차선처럼 인식한 콘 사이를 잇는
#        중앙선"이다. da 중앙 추종과 동일하게 "좌/우 경계 → 중앙 경로점" 구조를 그대로 따름.
#   출력 형식(offset/done/path_m)과 좌표 약속은 구 방식과 완전히 동일하게 유지했으므로
#   track_drive.py의 호출부(perc_lavacon(), _handle_lavacon())는 변경 불필요 —
#   Pure Pursuit(controller/pure_pursuit.py)가 그대로 이 경로를 追従한다.
#
# [2026-08-19 후속] 페어링을 "좌/우 각각 x 정렬 후 같은 순번끼리 묶기"에서 "기준선(y=0)
#   위에서 가까운 좌측 콘부터, 실제 유클리드 거리로 가장 가까운 우측 콘을 찾는" 최근접
#   이웃 방식으로 교체 — dl_lane.py의 vehicle_center_x(차량이 자기 위치/정면이라 믿는
#   기준선, 그 선 기준으로 좌우 차선을 판단) 개념을 라이다 콘 페어링에도 그대로 적용한
#   것. 순번 기반은 좌우 콘 개수/간격이 비대칭이면(급커브 등) i번째끼리 묶인 두 콘이
#   서로 물리적으로 가깝다는 보장이 없어 중점이 트랙 중앙에서 벗어날 수 있었다.
#
# [라이다 좌표 약속] (track_drive.py 재실측 기준, 2026-07-22 확정)
#   · 360칸, 인덱스 = 각도(도), 반시계 방향
#   · ★인덱스 0이 정면이 아니다★ — 라이다 장착 각도가 80도 어긋나 있어
#     "보정후각도 = 인덱스 - LIDAR_ANGLE_OFFSET_DEG" 로 빼줘야 한다.
#     보정 후: 0 = 정면 / 90 = 좌측 / 180 = 정후방 / 270 = 우측
#   · 인덱스 215~304 는 차체 자기가림 구간 → 항상 무효 처리
#   · 직교좌표 변환: x = r·cos(보정후각도) (전방+), y = r·sin(보정후각도) (좌측+)
#   ※ 이 두 상수(LIDAR_ANGLE_OFFSET_DEG / BODY_LO,HI)는 track_drive.py의 동일 상수와
#     반드시 값을 일치시킬 것. 2026-06-19 구 규약(오프셋 0, 마스크 99~262)을 쓰던 시절
#     코드가 남아 실제 좌측 콘이 마스크에 지워지고 우측 콘이 좌측으로 반전 해석되어
#     done 이 항상 True 가 되는 버그가 있었다.
#
# [부호 약속] (track_drive.py 제어팀 합의와 동일)
#   · lavacon_offset > 0 : 중심선이 차량 기준 '우측'에 있음 → 우조향
#   · y(좌측+) 기준으로는 중심선 y평균이 음수일 때 offset이 양수
#     → offset = -mean(y) 로 부호 반전하여 계산한다.
# =============================================================
import math
import numpy as np

# [2026-08-07] LIDAR_ANGLE_OFFSET_DEG를 이 파일에 별도 상수로 하드코딩해뒀던 게
#   config.py와 값이 어긋날 수 있는 위험이었다(위 "2026-06-19 구 규약" 버그가 정확히
#   이 종류의 비동기화로 생겼었음) — config.py를 단일 소스로 삼아 여기서도 그대로
#   가져다 쓰도록 고쳤다. 값 자체(80.0)는 바뀌지 않았다.
from ..config import LIDAR_ANGLE_OFFSET_DEG

# ─────────────────────────────────────────────
# 튜닝 상수 (track_drive.py 의 실측 ROI 값과 일치시킴)
# ─────────────────────────────────────────────
BODY_LO, BODY_HI = 215, 305     # 차체 가림 인덱스 구간 [215, 304] 마스킹 경계 (305는 미포함)
                                 # (config.py엔 중앙화돼 있지 않음 — perc_obstacle()/
                                 #  perc_lavacon_trigger()도 각 함수 안에 동일 값을 로컬로
                                 #  들고 있는 게 이 프로젝트의 기존 관례라 그대로 따름)
LON_MIN          = 0.0          # 콘 후보 점의 전방 최소거리 (m) — 차체 바로 앞 반사 배제
CONE_LON_MAX     = 4.0          # 콘 후보 점의 전방 최대거리 (m) — 벽/원거리 잡음 배제
CONE_LAT_LIMIT   = 2.5          # 콘 후보 점의 횡방향 한계 (m)
OFFSET_CLAMP     = 0.8          # 편차 물리한계 (m) — 콘 사이 폭 초과값은 오검출로 간주
OFFSET_GAIN      = 1.0          # y평균 → offset 스케일 계수 (제어팀 LAVACON_KP와 별도, 여기선 1:1)
# [2026-08-19] 좌우 클러스터를 몇 쌍(gate)까지 경로점으로 쓸지 — 가까운 것부터 훑어서
#   자르므로 "가장 가까운 포인트들 위주로 경로생성" 요구사항을 그대로 만족한다.
#   [2026-08-19 후속: 최근접 이웃 페어링 교체] 예전엔 이 값을 늘리면 원거리에서 좌우 개수가
#   어긋나 엉뚱한 페어링(예: 좌1 vs 우4)이 섞일 위험이 있었는데, 아래 _pair_nearest()가
#   순번이 아니라 실제 유클리드 거리로 짝을 찾으므로 그 위험은 사실상 사라졌다 — 그래도
#   너무 크게 잡으면 관련 없는 먼 콘까지 억지로 짝지어질 수 있어 기존 값(6)을 그대로 유지.
MAX_GATES        = 6


def _pair_nearest(left_cx, left_cy, right_cx, right_cy, max_gates):
    """차량 정면 기준선(y=0, 전방 x축 — dl_lane.py의 vehicle_center_x 빨간 세로선과 같은
    역할, "차량이 자기 위치/정면이라고 믿는 선") 위를 가까운 좌측 콘부터 훑으면서, 그 순간
    아직 안 쓰인 우측 콘 중 유클리드 거리가 가장 가까운 것을 짝으로 찾는다.

    [2026-08-19 교체] 이전엔 좌/우를 각각 x(전방거리)로만 정렬해 같은 순번끼리(i번째로
    가까운 좌 ↔ i번째로 가까운 우) 묶었다 — 좌우 콘 개수/간격이 비대칭이면(급커브 등)
    순번이 서로 다른 물리적 위치를 가리켜 중점이 트랙 중앙에서 벗어나는 문제가 있었다.
    이제는 "기준선 위에서 가까운 좌측 콘부터, 그때 실제로 가장 가까운 우측 콘"을 찾는
    최근접 이웃 방식이라 좌우 개수/간격이 어긋나도 물리적으로 가까운 콘끼리 묶인다.
    """
    order = np.argsort(left_cx)
    left_cx, left_cy = left_cx[order], left_cy[order]

    right_used = np.zeros(len(right_cx), dtype=bool)
    mid_x, mid_y = [], []
    for i in range(min(len(left_cx), max_gates)):
        lx, ly = left_cx[i], left_cy[i]
        d2 = (right_cx - lx) ** 2 + (right_cy - ly) ** 2
        d2 = np.where(right_used, np.inf, d2)
        j = int(np.argmin(d2))
        if not np.isfinite(d2[j]):
            break   # 우측 콘을 이미 다 써버림 — 더 짝지을 상대가 없음
        right_used[j] = True
        mid_x.append((lx + right_cx[j]) / 2.0)
        mid_y.append((ly + right_cy[j]) / 2.0)

    mid_x = np.array(mid_x, dtype=np.float32)
    mid_y = np.array(mid_y, dtype=np.float32)

    # 가까운 좌측 콘부터 순서대로 짝지었지만, 우측 파트너가 항상 그만큼 가깝다는 보장은
    # 없으므로(먼 우측 콘과 억지로 묶였을 수 있음) 최종 경로점은 x 오름차순으로 한 번 더
    # 정렬해 확실히 한다.
    if len(mid_x) > 0:
        o2 = np.argsort(mid_x)
        mid_x, mid_y = mid_x[o2], mid_y[o2]
    return mid_x, mid_y


def _cluster_cone_side(px, py, pidx):
    """같은 편(좌 또는 우) 콘 후보 점들을 원본 라이다 인덱스(각도) 인접성으로 묶어
    콘 하나당 중심점(centroid) 하나로 압축한다.

    perc_obstacle()의 전방 타겟 그룹핑(fidx/groups, np.diff(fidx)>1로 분할)과
    perc_lavacon_trigger()의 _has_cluster()와 동일한 클러스터링 방식을 재사용한 것 —
    인덱스가 서로 붙어있으면(gap<=1) 공간적으로도 인접한 점(같은 콘의 여러 빔 반사)으로 본다.
    콘 하나가 원거리에서 라이다 빔 1개만 맞히는 경우도 흔해(각도 분해능 1도, 4m에서 콘
    폭이 차지하는 각도는 수 도 이내) 최소 포인트 수 제한은 두지 않는다 — 단일 점도
    유효한 콘 하나로 인정.
    """
    if len(pidx) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    order = np.argsort(pidx)
    idx_s, px_s, py_s = pidx[order], px[order], py[order]
    splits = np.where(np.diff(idx_s) > 1)[0] + 1
    groups = np.split(np.arange(len(idx_s)), splits)
    cx = np.array([float(np.mean(px_s[g])) for g in groups], dtype=np.float32)
    cy = np.array([float(np.mean(py_s[g])) for g in groups], dtype=np.float32)
    return cx, cy


def process_lavacon(lidar_ranges):
    """
    2D 라이다 1스캔(360점)으로부터 라바콘 트랙 중심 편차 + 조향용 경로를 계산한다.

    입력 : lidar_ranges — 길이 360의 거리 배열 (list 또는 np.ndarray)
                          인덱스 0 = 정면, 인덱스 = 각도(도), 반시계
    출력 : (lavacon_offset, lavacon_done, path_m) 튜플
           · lavacon_offset (float) : 중심 편차 [-0.8, +0.8], 양수 = 우조향(디버그/로깅용)
           · lavacon_done   (bool)  : 좌/우 모두 콘 미검출 = 라바콘 구간 종료 신호
           · path_m (list[(float,float)]) : 좌/우 콘 클러스터를 차량 정면 기준선(y=0)
             기준 최근접 이웃으로 페어링한 중점들, x(전방) 오름차순, (x, y) 라이다 미터
             좌표(x=전방+, y=좌측+). 조향 실계산은 이걸 쓴다. 한쪽이라도 콘이 안 보이면
             빈 리스트.
    """
    # ── 0) 입력 유효성 검사 : None 이거나 비어 있으면 즉시 안전 폴백 ──
    if lidar_ranges is None:
        return (0.0, True, [])

    # ── 1) 전처리 : NumPy 배열화 + 무효값(inf/nan/음수/0) 제거 + 차체 마스킹 ──
    ranges = np.asarray(lidar_ranges, dtype=np.float32).copy()  # 원본 훼손 방지 복사
    n = len(ranges)
    if n == 0:
        return (0.0, True, [])

    ranges[~np.isfinite(ranges)] = 0.0     # inf / nan → 0.0 (무효 표시)
    ranges[ranges <= 0.0] = 0.0            # 0 이하 거리 → 무효

    # 차체 자기가림 구간(인덱스 215~304)을 0.0으로 마스킹 → 전방(0~214, 305~359)만 사용
    if n > BODY_LO:
        ranges[BODY_LO:min(BODY_HI, n)] = 0.0

    # ── 2) 극좌표 → 직교좌표 변환 (x: 전방+, y: 좌측+) ──
    # 인덱스 0이 정면이 아니므로 LIDAR_ANGLE_OFFSET_DEG 만큼 빼서 영점을 보정한다.
    deg = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False) - math.radians(LIDAR_ANGLE_OFFSET_DEG)
    x = ranges * np.cos(deg)    # 종방향(전방거리) 성분
    y = ranges * np.sin(deg)    # 횡방향 성분 — y > 0 좌측, y < 0 우측

    # ── 3) 콘 후보 필터링 : 전방 ROI 안의 유효 점만 남김 (원본 인덱스도 함께 보존) ──
    # 벽·원거리 구조물처럼 트랙과 무관한 점이 섞이면 클러스터링/페어링이 트랙 밖으로
    # 왜곡되므로, 콘이 존재할 수 있는 영역으로 제한한다.
    cone_mask = (ranges > 0.0) & (x > LON_MIN) & (x < CONE_LON_MAX) & (np.abs(y) < CONE_LAT_LIMIT)
    idx_all = np.arange(n)
    px, py, pidx = x[cone_mask], y[cone_mask], idx_all[cone_mask]  # 콘 후보 점들의 (x, y, 원본 인덱스)

    # ── 4) 종료 판정 : 좌·우 어느 쪽에도 콘이 없어야 라바콘 구간 끝 ──
    # 구 로직은 "우측(y<0) 콘 소멸"만 봤는데, 콘 간격이 유동적인 코스에서는
    # 우측에 넓은 틈이 하나만 있어도 곧바로 '구간 끝'으로 오판한다.
    # 한쪽 줄이라도 보이면 아직 구간 안이라고 보는 편이 안전하다.
    # (디바운스는 상위 FSM(_handle_lavacon)의 LAVACON_DONE_FRAMES에서 수행)
    has_left  = bool(np.any(py > 0.0))
    has_right = bool(np.any(py < 0.0))
    lavacon_done = not (has_left or has_right)

    # ── 5) 좌/우 콘 클러스터링 : 점군 → 콘 단위 중심점(centroid) ──
    # da(주행가능영역) 중앙 추종과 동일한 구조 — "좌 경계 / 우 경계를 먼저 인식하고
    # 그 중앙을 경로로 삼는다"를 그대로 따르되, 경계 소스만 차선 대신 콘 클러스터로 바뀐 것.
    left_cx,  left_cy  = _cluster_cone_side(px[py > 0.0], py[py > 0.0], pidx[py > 0.0])
    right_cx, right_cy = _cluster_cone_side(px[py < 0.0], py[py < 0.0], pidx[py < 0.0])

    if len(left_cx) == 0 or len(right_cx) == 0:
        # 한쪽 콘이 아예 안 보이면(구간 시작/끝, 일시적 가림 등) 이번 프레임은 경로 생성을
        # 보류한다 — pure_pursuit.control()이 빈 경로를 받으면 직전 조향각을 그대로 유지한다.
        return (0.0, lavacon_done, [])

    # ── 6) 최근접 이웃 페어링 : 차량 정면 기준선(y=0) 위에서 가까운 좌측 콘부터, 그때
    #   실제로 가장 가까운 우측 콘을 찾아 짝짓는다 — 상세 근거는 _pair_nearest() 참고.
    mid_x, mid_y = _pair_nearest(left_cx, left_cy, right_cx, right_cy, MAX_GATES)
    if len(mid_x) == 0:
        return (0.0, lavacon_done, [])

    # ── 7) 편차 계산 : 경로점 y좌표 평균 → 부호 반전 → 클램프 (디버그/로깅용) ──
    # y는 좌측+ 이므로, 중심선이 우측(y평균 < 0)에 있으면
    # 우조향(+)이 필요 → offset = -mean(y) 로 부호를 뒤집는다.
    mean_y = float(np.mean(mid_y))
    lavacon_offset = -mean_y * OFFSET_GAIN

    # 물리한계 클램프 : 콘 사이 폭을 넘는 값은 오검출(벽 등)로 보고 잘라냄
    lavacon_offset = float(np.clip(lavacon_offset, -OFFSET_CLAMP, OFFSET_CLAMP))

    path_m = list(zip(mid_x.tolist(), mid_y.tolist()))

    return (lavacon_offset, lavacon_done, path_m)


# ─────────────────────────────────────────────
# 간단 자가 테스트 (ROS 없이 로직 검증용)
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # 가상 시나리오 : 좌측 콘 줄 y=+2.0 m, 우측 콘 줄 y=-1.0 m 인 직선 트랙
    # → 중심선은 y ≈ +0.5 (좌측) → offset ≈ -0.5 (좌조향) 기대
    # ※ 인덱스 0이 정면이 아니므로 "인덱스 = 실제각 + LIDAR_ANGLE_OFFSET_DEG" 로 배치한다.
    test = np.zeros(360, dtype=np.float32)

    def put(true_deg, lateral_y):
        """실제각 true_deg 방향에 y=lateral_y 가 되도록 콘 하나를 놓는다."""
        idx = int(round(true_deg + LIDAR_ANGLE_OFFSET_DEG)) % 360
        test[idx] = lateral_y / math.sin(math.radians(true_deg))

    for t in (30, 45, 60):      # 좌측 줄 (y = +2.0)
        put(t, 2.0)
    for t in (-20, -35, -50):   # 우측 줄 (y = -1.0)
        put(t, -1.0)

    off, done, path_m = process_lavacon(test)
    print(f'offset={off:+.3f} (음수=좌조향 기대), done={done} (False 기대), path_m={path_m}')
