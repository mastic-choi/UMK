# =============================================================
# perc_lavacon.py — 좌/우 라이다 콘 클러스터 중앙 추종 경로 생성 모듈
#
# [사용법] track_drive.py 에서 import 하여 호출:
#     from perc_lavacon import process_lavacon
#     offset, done, path_m = process_lavacon(self.lidar_ranges)
#
# [경로생성 방식 변천사, 전부 2026-08-19]
#   1) 보로노이(scipy.spatial.Voronoi) — 콘 점군 전체로 다이어그램을 계산해 정점(vertex)들을
#      중심선으로 썼다. 좌우 콘 개수가 비대칭이어도 위상적으로 그럴듯한 골격이 나온다는
#      장점은 있었지만, 정점 개수/위치가 프레임마다 크게 흔들려 실차 테스트에서 경로가
#      계속 튀는 문제로 폐기.
#   2) 클러스터링 + 순번 매칭 — 좌/우 콘 후보 점을 라이다 인덱스 인접성으로 묶어(콘 하나당
#      중심점 하나) 좌/우 각각 x(전방거리) 오름차순으로 정렬 후 같은 순번끼리 묶었다.
#      좌우 콘 개수/간격이 비대칭이면(급커브 등) i번째끼리 묶인 두 콘이 물리적으로
#      가깝다는 보장이 없어 중점이 트랙 중앙을 벗어나는 문제로 폐기.
#   3) 클러스터링 + 최근접 이웃(유클리드) 매칭(`_pair_nearest`, `_cluster_cone_side`,
#      한때 채택됐다 폐기) — 순번 대신 기준선(y=0) 위에서 가까운 좌측 콘부터 실제 유클리드
#      거리가 가장 가까운 우측 콘을 찾아 짝지었다. 그런데 유클리드 거리만 보면 급커브에서
#      트랙을 가로지르는 두 콘(예: 내 차로 안쪽 콘과 그 너머 바깥쪽 콘)이 실제로는 다른
#      경계에 속하는데도 "물리적으로 가깝다"는 이유만으로 잘못 짝지어질 위험이 있다는
#      지적(사용자)에 따라 폐기.
#   4) [현재] 박스 스택 페어링(`_pick_boxed_centers`) — perc_lavacon_trigger()의 진입
#      트리거 박스(전방 0.3~0.5m 좁은 구간에서 좌우 클러스터 유무만 봄)와 완전히 같은
#      발상으로 돌아간다: 그 박스와 같은 폭(BOX_LON_WIDTH)의 박스를 전방으로 쭉 쌓아
#      올리고, 각 박스 "안에서만" 좌(y>0)/우(y<0) 각 1점(차량에서 가장 가까운 점)을 뽑아
#      그 중점을 그 박스의 경로점으로 삼는다. 짝짓기 후보 자체가 같은 좁은 종방향 구간
#      안으로 국한되므로 3)의 "먼 트랙 구간 콘과 잘못 짝지어지는" 문제가 구조적으로
#      불가능해진다. 박스를 전방 순서대로 훑으므로 결과가 자연히 x 오름차순이라 별도
#      정렬도 불필요.
#   출력 형식(offset/done/path_m)과 좌표 약속은 매 단계 동일하게 유지했으므로
#   track_drive.py의 호출부(perc_lavacon(), _handle_lavacon())는 변경 불필요 —
#   Pure Pursuit(controller/pure_pursuit.py)가 그대로 이 경로를 추종한다.
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
CONE_LAT_LIMIT   = 1.8          # 콘 후보 점의 횡방향 한계 (m) — [2026-08-19] 2.5→1.8로 축소(요청 반영).
                                 # track_drive.py _draw_lavacon_bev()가 이 값을 그대로 import해서
                                 # lavacon_bev 창에 좌우 경계선으로 그린다 — 값을 또 바꾸면 그 선도 같이 움직인다.
OFFSET_CLAMP     = 0.8          # 편차 물리한계 (m) — 콘 사이 폭 초과값은 오검출로 간주
OFFSET_GAIN      = 1.0          # y평균 → offset 스케일 계수 (제어팀 LAVACON_KP와 별도, 여기선 1:1)

# [2026-08-19] 박스 스택 페어링 파라미터 — track_drive.py perc_lavacon_trigger()의 진입
#   트리거 박스와 반드시 같은 폭을 유지할 것(BOX_LON_START=LON_MIN, BOX_LON_WIDTH=
#   LON_MAX-LON_MIN, 트리거 쪽은 0.3~0.5). perc_lavacon.py와 perc_lavacon_trigger()가
#   서로 import하지 않는 기존 관례상 값을 복사해서 들고 있다(위 BODY_LO/HI 주석 참고) —
#   한쪽만 바꾸면 반드시 다른 쪽도 같이 바꿀 것.
BOX_LON_START    = 0.3          # 첫 박스 시작 지점(전방, m) — 차체 바로 앞 반사 배제
BOX_LON_WIDTH    = 0.2          # 박스 1개의 종방향 폭(m)


def _pick_boxed_centers(x, y, ranges, lon_start, lon_width, lon_max, lat_limit):
    """차량 전방을 lon_width 폭의 박스로 lon_start부터 lon_max까지 쭉 쌓아올리고, 각 박스
    "안에서만" 좌(y>0)/우(y<0) 각 1점(그 박스 안에서 차량과의 라이다 반사거리가 가장 짧은
    점)을 뽑아 중점을 그 박스의 경로점으로 삼는다.

    [2026-08-19] 콘 클러스터링 + 유클리드 최근접 이웃 매칭(구버전, 폐기) 대신 이 방식을
    쓰는 이유: 유클리드 거리만으로 좌/우를 짝지으면, 급커브에서 트랙을 가로지르는 두 콘이
    실제로는 서로 다른 경계(내 차로 안쪽 vs 그 너머 바깥쪽)에 속하는데도 "물리적으로
    가깝다"는 이유만으로 잘못 짝지어질 위험이 있다(사용자 지적) — 각 박스가 좁은 종방향
    구간으로 짝짓기 후보 자체를 국한시키므로 그런 오매칭이 구조적으로 불가능해진다.
    박스를 전방 순서대로 훑으므로 결과가 자연히 x(전방) 오름차순이라 별도 정렬도 불필요.
    한쪽 콘만 있는 박스는 건너뛴다(그 프레임 하나 때문에 전체 경로를 막지 않기 위해) —
    perc_lavacon_trigger()의 트리거 박스와 동일하게, "박스 안에 뭔가 있는지"만으로
    판단하고 콘 개수/간격 자체에는 가정을 두지 않는다.
    """
    mid_x, mid_y = [], []
    n_boxes = max(0, int(math.floor((lon_max - lon_start) / lon_width + 1e-6)))
    for i in range(n_boxes):
        b_lo = lon_start + i * lon_width
        b_hi = b_lo + lon_width
        box_mask = (ranges > 0.0) & (x >= b_lo) & (x < b_hi) & (np.abs(y) < lat_limit)
        left_idx = np.where(box_mask & (y > 0.0))[0]
        right_idx = np.where(box_mask & (y < 0.0))[0]
        if left_idx.size == 0 or right_idx.size == 0:
            continue
        li = left_idx[int(np.argmin(ranges[left_idx]))]
        ri = right_idx[int(np.argmin(ranges[right_idx]))]
        mid_x.append((float(x[li]) + float(x[ri])) / 2.0)
        mid_y.append((float(y[li]) + float(y[ri])) / 2.0)
    return mid_x, mid_y


def process_lavacon(lidar_ranges):
    """
    2D 라이다 1스캔(360점)으로부터 라바콘 트랙 중심 편차 + 조향용 경로를 계산한다.

    입력 : lidar_ranges — 길이 360의 거리 배열 (list 또는 np.ndarray)
                          인덱스 0 = 정면, 인덱스 = 각도(도), 반시계
    출력 : (lavacon_offset, lavacon_done, path_m) 튜플
           · lavacon_offset (float) : 중심 편차 [-0.8, +0.8], 양수 = 우조향(디버그/로깅용)
           · lavacon_done   (bool)  : 좌/우 모두 콘 미검출 = 라바콘 구간 종료 신호
           · path_m (list[(float,float)]) : 전방으로 쌓아올린 박스(BOX_LON_WIDTH 폭)마다
             좌/우 최근접 1점씩의 중점, x(전방) 오름차순(박스를 순서대로 훑으므로 자연히
             정렬됨), (x, y) 라이다 미터 좌표(x=전방+, y=좌측+). 조향 실계산은 이걸 쓴다.
             유효한 박스가 하나도 없으면 빈 리스트.
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

    # ── 3) 종료 판정용 콘 후보 필터링 : 전방 넓은 ROI 안의 유효 점만 남김 ──
    # 벽·원거리 구조물처럼 트랙과 무관한 점을 배제하기 위해 콘이 존재할 수 있는 영역으로
    # 제한한다. 아래 5)의 박스 스택 탐색과는 별개 — 종료 판정은 넓은 범위로 봐야
    # 콘 간격이 벌어진 순간에 "아직 구간 안"인데도 종료로 오판하지 않는다.
    cone_mask = (ranges > 0.0) & (x > LON_MIN) & (x < CONE_LON_MAX) & (np.abs(y) < CONE_LAT_LIMIT)
    py_cone = y[cone_mask]

    # ── 4) 종료 판정 : 좌·우 어느 쪽에도 콘이 없어야 라바콘 구간 끝 ──
    # 구 로직은 "우측(y<0) 콘 소멸"만 봤는데, 콘 간격이 유동적인 코스에서는
    # 우측에 넓은 틈이 하나만 있어도 곧바로 '구간 끝'으로 오판한다.
    # 한쪽 줄이라도 보이면 아직 구간 안이라고 보는 편이 안전하다.
    # (디바운스는 상위 FSM(_handle_lavacon)의 LAVACON_DONE_FRAMES에서 수행)
    has_left  = bool(np.any(py_cone > 0.0))
    has_right = bool(np.any(py_cone < 0.0))
    lavacon_done = not (has_left or has_right)

    # ── 5) 박스 스택 페어링 : 전방으로 쌓아올린 박스마다 좌/우 최근접 1점의 중점 ──
    # 상세 근거는 _pick_boxed_centers() 참고. CONE_LON_MAX/CONE_LAT_LIMIT를 그대로
    # 재사용해 박스 탐색 범위를 위 종료판정 ROI와 동일한 한계 안으로 맞춘다.
    mid_x, mid_y = _pick_boxed_centers(x, y, ranges, BOX_LON_START, BOX_LON_WIDTH,
                                        CONE_LON_MAX, CONE_LAT_LIMIT)
    if not mid_x:
        # 유효한 박스가 하나도 없음(콘 미검출 등) — 이번 프레임은 경로 생성을 보류한다.
        # pure_pursuit.control()이 빈 경로를 받으면 직전 조향각을 그대로 유지한다.
        return (0.0, lavacon_done, [])

    # ── 6) 편차 계산 : 경로점 y좌표 평균 → 부호 반전 → 클램프 (디버그/로깅용) ──
    # y는 좌측+ 이므로, 중심선이 우측(y평균 < 0)에 있으면
    # 우조향(+)이 필요 → offset = -mean(y) 로 부호를 뒤집는다.
    mean_y = float(np.mean(mid_y))
    lavacon_offset = -mean_y * OFFSET_GAIN

    # 물리한계 클램프 : 콘 사이 폭을 넘는 값은 오검출(벽 등)로 보고 잘라냄
    lavacon_offset = float(np.clip(lavacon_offset, -OFFSET_CLAMP, OFFSET_CLAMP))

    path_m = list(zip(mid_x, mid_y))

    return (lavacon_offset, lavacon_done, path_m)


# ─────────────────────────────────────────────
# 간단 자가 테스트 (ROS 없이 로직 검증용)
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # 가상 시나리오 : 두 "게이트"(좌우 콘 쌍)를 서로 다른 박스에 하나씩 배치.
    #   게이트1(전방 0.4m): 좌 y=+1.5, 우 y=-1.0 → 중점 y=+0.25 → offset≈-0.25(좌조향)
    #   게이트2(전방 1.0m): 좌 y=+1.0, 우 y=-1.0 → 중점 y=0.0  → offset≈0(직진)
    # 박스 폭(BOX_LON_WIDTH=0.2m)보다 좁게 좌우를 같은 x에 둬야 같은 박스에 페어링된다.
    test = np.zeros(360, dtype=np.float32)

    def put_xy(fx, fy):
        """전방 fx(m), 좌측+ fy(m) 위치에 콘 하나를 놓는다 (x=fx, y=fy 직교좌표 → 극좌표 변환)."""
        r = math.hypot(fx, fy)
        true_deg = math.degrees(math.atan2(fy, fx))
        idx = int(round(true_deg + LIDAR_ANGLE_OFFSET_DEG)) % 360
        test[idx] = r

    put_xy(0.4, 1.5)   # 게이트1 좌
    put_xy(0.4, -1.0)  # 게이트1 우
    put_xy(1.0, 1.0)   # 게이트2 좌
    put_xy(1.0, -1.0)  # 게이트2 우

    off, done, path_m = process_lavacon(test)
    print(f'offset={off:+.3f} (~-0.125 기대: 두 게이트 중점 y평균의 부호반전), '
          f'done={done} (False 기대), path_m={path_m} '
          f'(~[(0.4,0.25),(1.0,0.0)] 기대)')
