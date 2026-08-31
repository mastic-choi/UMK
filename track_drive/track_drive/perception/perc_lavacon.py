# =============================================================
# perc_lavacon.py — 좌/우 라이다 콘 클러스터 중앙 추종 경로 생성 모듈
#
# [사용법] track_drive.py 에서 import 하여 호출:
#     from perc_lavacon import process_lavacon
#     offset, done, path_m, boxes = process_lavacon(self.lidar_ranges, prev_boxes)
#     (prev_boxes는 직전 호출이 반환한 boxes를 self.*에 들고 있다가 그대로 다시 넘긴다 —
#      LAVACON_TEMPORAL_EMA_ENABLED가 꺼져 있으면 안 넘겨도(None) 무방)
#
# [경로 생성 방식] 차량 전방을 BOX_LON_WIDTH 폭의 박스로 쭉 쌓아 올리고, 각 박스
#   "안에서만" 좌(y>0)/우(y<0) 각 1점(차량에서 가장 가까운 점)을 뽑아 좌/우 독립
#   바운더리 시퀀스로 들고 있는다(`_pick_boxed_sides`). 짝짓기 후보 자체를 같은 좁은
#   종방향 구간으로 국한시켜, 급커브에서 트랙을 가로지르는 먼 콘끼리 잘못 짝짓는 문제를
#   구조적으로 막는다. 좌/우 배정은 기본적으로 라인 연속성(직전 박스에서 확정된 좌/우
#   점과 유클리드 최근접, `_assign_by_continuity`, `LAVACON_LINE_CONTINUITY_ENABLED`)으로
#   하는데, 급커브에서 물리적으로 "오른쪽 라인"이던 콘이 차량 헤딩 기준 y>0(왼쪽)으로
#   넘어가도 고정 y=0 경계보다 정확하게 같은 라인으로 계속 배정하기 위해서다. 양쪽 다
#   검출된 박스는 두 점의 중점을 경로점으로 쓰고, 한쪽만 검출된 박스는
#   (`LAVACON_SPARSE_FALLBACK_ENABLED`, 기본 False) 직전까지 양쪽 다 검출됐던 박스들의
#   좌우 반폭 EMA만큼 검출된 쪽 반대로 밀어 중심선을 추정한다(`_build_path`). 라이다가
#   한 프레임 튀는 문제(반사 노이즈 등)에는 박스별 좌/우 포인트("라바콘 차선")에 거는
#   프레임간 EMA(`_blend_boxes_temporal`, 기본 꺼짐, `LAVACON_TEMPORAL_EMA_ENABLED`)로
#   대응한다 — waypoint(중점) 자체가 아니라 그 재료인 좌/우 포인트 쪽에 건다. 박스 스택
#   자체를 우회하는 da_push 조향 모드(`LAVACON_STEER_MODE_DA_PUSH`, track_drive.py
#   `_lavacon_steer_da_push()`)도 있다 — 켜지면 이 파일의 박스 스택 경로(path_m)는
#   lavacon_done 판정에만 쓰이고 조향엔 `nearest_cone_lateral()`이 쓰인다.
#
#   출력 형식(offset/done/path_m)과 좌표 약속은 유지되어 Pure Pursuit
#   (controller/pure_pursuit.py)이 그대로 이 경로를 추종한다. 반환 튜플엔 boxes도
#   포함된다(프레임간 EMA용 상태 — track_drive.py perc_lavacon() 호출부 참고).
#
# [라이다 좌표 약속] (track_drive.py 재실측 기준)
#   · 360칸, 인덱스 = 각도(도), 반시계 방향
#   · ★인덱스 0이 정면이 아니다★ — 라이다 장착 각도가 80도 어긋나 있어
#     "보정후각도 = 인덱스 - LIDAR_ANGLE_OFFSET_DEG" 로 빼줘야 한다.
#     보정 후: 0 = 정면 / 90 = 좌측 / 180 = 정후방 / 270 = 우측
#   · 인덱스 215~304 는 차체 자기가림 구간 → 항상 무효 처리
#   · 직교좌표 변환: x = r·cos(보정후각도) (전방+), y = r·sin(보정후각도) (좌측+)
#   ※ 이 두 상수(LIDAR_ANGLE_OFFSET_DEG / BODY_LO,HI)는 track_drive.py의 동일 상수와
#     반드시 값을 일치시킬 것 — 어긋나면 좌/우 콘이 반전 해석되어 done이 항상 True가
#     되는 등의 버그로 이어진다.
#
# [부호 약속] (track_drive.py 제어팀 합의와 동일)
#   · lavacon_offset > 0 : 중심선이 차량 기준 '우측'에 있음 → 우조향
#   · y(좌측+) 기준으로는 중심선 y평균이 음수일 때 offset이 양수
#     → offset = -mean(y) 로 부호 반전하여 계산한다.
# =============================================================
import math
import numpy as np

# 라이다 각도 보정값 — config.py가 단일 소스(track_drive.py와 반드시 일치해야 함).
from ..config import LIDAR_ANGLE_OFFSET_DEG
# sparse 박스(한쪽만 검출) 반폭 추정 폴백 스위치/게인 — config.py가 단일 소스.
from ..config import LAVACON_SPARSE_FALLBACK_ENABLED, LAVACON_HALFWIDTH_EMA_ALPHA
# 박스별 좌/우 바운더리 포인트("라바콘 차선")에 거는 프레임간 EMA 스위치/게인 —
# waypoint(중점) 자체가 아니라 그 재료인 좌/우 포인트에 건다(_blend_boxes_temporal 참고).
from ..config import LAVACON_TEMPORAL_EMA_ENABLED, LAVACON_TEMPORAL_EMA_ALPHA
# 박스 안 후보점을 y부호(고정 중앙선)로 가르는 대신 직전 박스의 같은 라인과의 최근접
# 연속성으로 배정할지 여부 — _pick_boxed_sides() 참고.
from ..config import LAVACON_LINE_CONTINUITY_ENABLED, LAVACON_LINE_TRACK_MAX_JUMP_M

# ─────────────────────────────────────────────
# 튜닝 상수 (track_drive.py 의 실측 ROI 값과 일치시킴)
# ─────────────────────────────────────────────
BODY_LO, BODY_HI = 215, 305     # 차체 가림 인덱스 구간 [215, 304] 마스킹 경계 (305는 미포함)
                                 # (config.py엔 중앙화돼 있지 않음 — perc_obstacle()/
                                 #  perc_lavacon_trigger()도 각 함수 안에 동일 값을 로컬로
                                 #  들고 있는 게 이 프로젝트의 기존 관례라 그대로 따름)
LON_MIN          = 0.0          # 콘 후보 점의 전방 최소거리 (m) — 차체 바로 앞 반사 배제
CONE_LON_MAX     = 4.0          # 콘 후보 점의 전방 최대거리 (m) — 벽/원거리 잡음 배제
CONE_LAT_LIMIT   = 0.5          # 콘 후보 점의 횡방향 한계 (m) — track_drive.py
                                 #   perc_lavacon_trigger()의 트리거 ROI LAT_MAX와 같은 값으로
                                 #   맞춰야 한다(두 값은 항상 같이 바꿀 것). track_drive.py
                                 #   _draw_lavacon_bev()가 이 값을 그대로 import해서 lavacon_bev
                                 #   창에 좌우 경계선(흰 선)으로 그린다.
OFFSET_CLAMP     = 0.8          # 편차 물리한계 (m) — 콘 사이 폭 초과값은 오검출로 간주
OFFSET_GAIN      = 1.0          # y평균 → offset 스케일 계수 (제어팀 LAVACON_KP와 별도, 여기선 1:1)

# 종료 판정 ROI — track_drive.py perc_lavacon_trigger()의 진입 트리거 박스(LON_MIN/LON_MAX/
#   LAT_MAX)와 동일 크기: "진입 때 본 것과 같은 크기의 박스에 1개도 안 찍히면 종료"로 판정.
#   CONE_LON_MAX/CONE_LAT_LIMIT는 박스 스택 경로(path_m) 탐색 범위로 별도로 쓰인다(용도가
#   다름). 값은 perc_lavacon_trigger()의 LON_MIN/LON_MAX/LAT_MAX를 복사한 것이라 두 값은
#   항상 같이 바꿀 것(위 BODY_LO/HI 주석과 동일 관례).
EXIT_LON_MIN     = -0.1
EXIT_LON_MAX     = 0.3
EXIT_LAT_LIMIT   = 0.75

# 박스 스택 페어링 파라미터 — track_drive.py perc_lavacon_trigger()의 진입 트리거 박스와
#   반드시 같은 폭을 유지할 것(BOX_LON_START=LON_MIN, BOX_LON_WIDTH=LON_MAX-LON_MIN).
#   perc_lavacon.py와 perc_lavacon_trigger()가 서로 import하지 않는 기존 관례상 값을
#   복사해서 들고 있다(위 BODY_LO/HI 주석 참고) — 한쪽만 바꾸면 반드시 다른 쪽도 같이 바꿀 것.
BOX_LON_START    = 0.3          # 첫 박스 시작 지점(전방, m) — 차체 바로 앞 반사 배제
BOX_LON_WIDTH    = 0.4          # 박스 1개의 종방향 폭(m) — 실측 콘 간격(≈0.4m)에 맞춰, 박스
                                 #   하나에 콘 하나가 안정적으로 들어오게 한다. lavacon_bev/
                                 #   lavacon_ema_bev의 파란 격자도 이 값을 따라간다
                                 #   (track_drive.py가 LAVACON_BOX_LON_WIDTH로 import).

# 특정 물리 위치(우측 아주 가까운 지점, 차량 자체/마운트 반사로 추정)에서 반복적으로
#   찍히는 유령 라이다 점을 무효화한다 — 실측 좌표(x≈-0.05/y≈-0.075) 근처의 좁은 원형
#   구역(반경 6cm) 안의 라이다 점만 무효 처리한다. 반경을 넓게 잡으면 그 근처의 진짜
#   라바콘까지 같이 지워버릴 수 있으니 최소한만 잡았다 — 다른 위치에서도 유령 점이
#   재발하면 근본 원인(각도 오프셋, 자기가림 경계, 멀티패스)부터 확인할 것.
#   ※ 코드만 고쳐서는 실행 중이던 노드에 반영 안 됨 — 재시작 후에 확인할 것.
GHOST_POINT_X_M      = -0.05
GHOST_POINT_Y_M      = -0.075
GHOST_POINT_RADIUS_M = 0.06


def _lidar_to_xy(lidar_ranges):
    """라이다 1스캔을 전처리(무효값/자기가림 마스킹)하고 직교좌표(x=전방+, y=좌측+)로
    변환한다. process_lavacon()과 nearest_cone_lateral()이 똑같이 필요로 하는 전처리라
    공용 함수로 뺐다 — 두 곳이 각자 복사해서 들고 있으면 마스킹 범위나 각도보정 상수가
    한쪽만 바뀌는 사고가 재발할 수 있다.
    입력이 무효(None/빈 배열)면 (None, None, None)을 반환한다."""
    if lidar_ranges is None:
        return None, None, None
    ranges = np.asarray(lidar_ranges, dtype=np.float32).copy()  # 원본 훼손 방지 복사
    n = len(ranges)
    if n == 0:
        return None, None, None

    ranges[~np.isfinite(ranges)] = 0.0     # inf / nan → 0.0 (무효 표시)
    ranges[ranges <= 0.0] = 0.0            # 0 이하 거리 → 무효
    if n > BODY_LO:
        ranges[BODY_LO:min(BODY_HI, n)] = 0.0   # 차체 자기가림 구간 마스킹

    deg = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False) - math.radians(LIDAR_ANGLE_OFFSET_DEG)
    x = ranges * np.cos(deg)    # 종방향(전방거리) 성분
    y = ranges * np.sin(deg)    # 횡방향 성분 — y > 0 좌측, y < 0 우측

    ghost_mask = (x - GHOST_POINT_X_M) ** 2 + (y - GHOST_POINT_Y_M) ** 2 < GHOST_POINT_RADIUS_M ** 2
    ranges[ghost_mask] = 0.0    # 유령 점 무효화 — 위 GHOST_POINT_* 주석 참고

    return x, y, ranges


def nearest_cone_lateral(lidar_ranges, lon_min, lon_max, lat_limit, return_range=False, lon_max_l=None):
    """da(주행가능영역) 경로를 그대로 조향에 쓰고, 콘이 안전마진 안으로 들어왔을 때만
    그만큼 옆으로 미는 방식(track_drive.py `_lavacon_steer_da_push` 참고) 전용 — 박스
    스택 페어링(`_pick_boxed_sides`/`_build_path`)처럼 좌우 콘을 짝짓거나 경로 전체를
    재구성하지 않는다. 지정한 좁은 ROI(lon_min~lon_max, ±lat_limit) 안에서 좌(y>0)/
    우(y<0) 각각 라이다 반사거리가 가장 가까운 점 1개의 y좌표만 반환한다 — "정밀한
    경로"가 아니라 "얼마나 침범했는지"만 필요하므로 이거면 충분하고, 라이다가 듬성하게/
    노이즈 섞여 검출돼도 한 점만 있으면 바로 동작한다.

    return_range=True면 push 판정(y, 라인 침범 여부)은 그대로 두고 디버그 표시용으로
    range/x도 같이 반환한다: (left_y, right_y, left_range, right_range, left_x, right_x)
    6-튜플. False(기본)면 (left_y, right_y) 2-튜플만 반환.

    출력 : (left_y, right_y) — 각각 없으면 None, 있으면 라이다 미터 좌표(좌측+).
    """
    x, y, ranges = _lidar_to_xy(lidar_ranges)
    if x is None:
        return (None, None, None, None, None, None) if return_range else (None, None)

    # lon_max_l — 좌측만 전방 ROI를 다르게 주고 싶을 때(config.py LAVACON_PUSH_LON_MAX_L
    # 참고). None이면 lon_max와 동일하게 동작한다.
    roi_common = (ranges > 0.0) & (x > lon_min) & (np.abs(y) < lat_limit)
    left_lon_max = lon_max if lon_max_l is None else lon_max_l
    left_idx = np.where(roi_common & (x < left_lon_max) & (y > 0.0))[0]
    right_idx = np.where(roi_common & (x < lon_max) & (y < 0.0))[0]
    left_i = left_idx[int(np.argmin(ranges[left_idx]))] if left_idx.size > 0 else None
    right_i = right_idx[int(np.argmin(ranges[right_idx]))] if right_idx.size > 0 else None
    left_y = float(y[left_i]) if left_i is not None else None
    right_y = float(y[right_i]) if right_i is not None else None
    if not return_range:
        return left_y, right_y
    left_range = float(ranges[left_i]) if left_i is not None else None
    right_range = float(ranges[right_i]) if right_i is not None else None
    left_x = float(x[left_i]) if left_i is not None else None
    right_x = float(x[right_i]) if right_i is not None else None
    return left_y, right_y, left_range, right_range, left_x, right_x


def _assign_by_continuity(box_idx, x, y, ranges, prev_left_xy, prev_right_xy, max_jump_m):
    """한 박스 안 후보점 인덱스(box_idx)를, 직전 박스에서 확정됐던 좌/우 점(prev_left_xy/
    prev_right_xy)과의 유클리드 최근접 연속성으로 좌/우에 배정한다.

    y부호(y>0=좌/y<0=우, 차량 헤딩 기준 고정 중앙선)로 먼저 가르는 방식은 급커브에서
    물리적으로 계속 "오른쪽 라인"이던 콘이 차량 헤딩 기준 y>0(수학적으로는 왼쪽) 쪽으로
    넘어갈 때 그 점을 왼쪽 라인으로 잘못 편입시킨다. 이 함수는 그 대신 "직전까지
    오른쪽이던 점과 물리적으로 가장 가까운 후보가 이번에도 오른쪽일 가능성이 높다"는
    연속성 가정으로 배정한다.

    한 박스에 후보가 여러 개면 좌 배정을 먼저 확정하고 그 점을 후보군에서 제거한 뒤 우를
    고른다(같은 점이 좌/우 양쪽에 동시 배정되는 것을 구조적으로 막음) — 순서 자체(좌 우선)
    가 결과를 크게 좌우하진 않는다, 어차피 서로 다른 이전 앵커에 대한 최근접이라 겹칠
    일이 드물기 때문. `max_jump_m`보다 먼 점만 남으면(그 라인 콘이 그 박스에 없거나 너무
    많이 꺾인 경우) 연속 배정을 포기한다.

    직전 앵커가 아예 없는 경우(그 라인이 아직 한 번도 검출 안 됐거나, 방금 연속 배정에
    실패한 경우)엔 기존 y부호 방식으로 폴백한다 — 처음 게이트에 진입하는 순간처럼 아직
    "그 라인이 어디인지" 알 근거가 없을 때는 차량 헤딩 기준 좌/우 구분이 최선의 추정이기
    때문. 이때도 그 절반 안에서는 기존과 동일하게 라이다 최근접점(ranges 최소)을 고른다.
    """
    if box_idx.size == 0:
        return None, None
    remaining = list(box_idx)

    def _pop_nearest_to(ref_xy):
        if ref_xy is None or not remaining:
            return None
        d = [math.hypot(x[j] - ref_xy[0], y[j] - ref_xy[1]) for j in remaining]
        k = int(np.argmin(d))
        if d[k] > max_jump_m:
            return None
        j = remaining.pop(k)
        return (float(x[j]), float(y[j]))

    left_xy = _pop_nearest_to(prev_left_xy)
    right_xy = _pop_nearest_to(prev_right_xy)

    if left_xy is None and remaining:
        cand = [j for j in remaining if y[j] > 0.0]
        if cand:
            j = cand[int(np.argmin(ranges[cand]))]
            left_xy = (float(x[j]), float(y[j]))
            remaining.remove(j)
    if right_xy is None and remaining:
        cand = [j for j in remaining if y[j] < 0.0]
        if cand:
            j = cand[int(np.argmin(ranges[cand]))]
            right_xy = (float(x[j]), float(y[j]))
            remaining.remove(j)

    return left_xy, right_xy


def _pick_boxed_sides(x, y, ranges, lon_start, lon_width, lon_max, lat_limit,
                       line_continuity_enabled=LAVACON_LINE_CONTINUITY_ENABLED,
                       max_jump_m=LAVACON_LINE_TRACK_MAX_JUMP_M):
    """차량 전방을 lon_width 폭의 박스로 lon_start부터 lon_max까지 쭉 쌓아올리고, 각 박스
    "안에서만" 좌/우 각 1점을 뽑는다. `line_continuity_enabled`가 꺼져 있으면(기본) 좌
    (y>0)/우(y<0) 고정 경계로 나눈 뒤 그 박스 안에서 차량과의 라이다 반사거리가 가장 짧은
    점을 고른다. 켜져 있으면 `_assign_by_continuity()`로 직전 박스의 같은 라인과의 최근접
    연속성으로 배정한다(급커브에서 y부호가 뒤집히는 문제 대응 — 그 함수 docstring 참고).

    각 박스가 좁은 종방향 구간으로 짝짓기 후보 자체를 국한시키므로, 급커브에서 트랙을
    가로지르는 두 콘이 서로 다른 경계에 속하는데도 "물리적으로 가깝다"는 이유만으로
    잘못 짝지어지는 문제가 구조적으로 불가능해진다.

    박스마다 (left_xy 또는 None, right_xy 또는 None)을 반환해, 양쪽 유무 판단과 중점
    계산은 호출부(`_build_path`)로 넘긴다 — 왼쪽만 모으면 왼쪽 바운더리 시퀀스,
    오른쪽만 모으면 오른쪽 바운더리 시퀀스가 되는 셈이라 "좌우 독립 라인"을 별도
    자료구조 없이 얻는다. 박스 폭이 이미 좁아서(BOX_LON_WIDTH, 실측 콘 간격 기준) 같은
    사이드 안에서도 인접 박스끼리 물리적으로 가깝다는 보장이 있으므로 사이드 내부
    정렬/매칭 로직은 필요 없다(박스 순번이 곧 종방향 순서).

    검출이 하나도 없는 박스도 (None, None)으로 그대로 남겨, 반환 리스트 길이가 항상
    n_boxes로 고정되게 한다 — 박스 위치가 config 상수(lon_start, lon_width, lon_max)로만
    정해지는 고정 그리드라 인덱스 i는 프레임이 바뀌어도 항상 "차량 기준 같은 종방향
    구간"을 가리킨다. 이 안정적인 인덱스가 있어야 호출부에서 좌/우 포인트 각각에
    프레임간 EMA(`_blend_boxes_temporal`)를 인덱스 정렬해서 걸 수 있다.
    """
    boxes = []
    n_boxes = max(0, int(math.floor((lon_max - lon_start) / lon_width + 1e-6)))
    prev_left_xy, prev_right_xy = None, None   # line_continuity_enabled일 때만 갱신/사용
    for i in range(n_boxes):
        b_lo = lon_start + i * lon_width
        b_hi = b_lo + lon_width
        box_mask = (ranges > 0.0) & (x >= b_lo) & (x < b_hi) & (np.abs(y) < lat_limit)

        if line_continuity_enabled:
            box_idx = np.where(box_mask)[0]
            left_xy, right_xy = _assign_by_continuity(
                box_idx, x, y, ranges, prev_left_xy, prev_right_xy, max_jump_m)
            if left_xy is not None:
                prev_left_xy = left_xy
            if right_xy is not None:
                prev_right_xy = right_xy
        else:
            left_idx = np.where(box_mask & (y > 0.0))[0]
            right_idx = np.where(box_mask & (y < 0.0))[0]
            left_xy = None
            right_xy = None
            if left_idx.size > 0:
                li = left_idx[int(np.argmin(ranges[left_idx]))]
                left_xy = (float(x[li]), float(y[li]))
            if right_idx.size > 0:
                ri = right_idx[int(np.argmin(ranges[right_idx]))]
                right_xy = (float(x[ri]), float(y[ri]))

        boxes.append((left_xy, right_xy))
    return boxes


def _blend_boxes_temporal(boxes, prev_boxes, alpha):
    """"라바콘 차선"(박스별 좌/우 바운더리 포인트)에 프레임간 EMA를 건다 — 최종 waypoint
    (중점)를 직접 EMA하지 않고, 그 waypoint의 재료인 좌/우 포인트 쪽에 거는 이유는 중점을
    직접 EMA하면 그 프레임의 검출 유무(양쪽/한쪽/폴백 추정)가 섞여 들어간 값이 다음
    프레임 기준값이 되어버려 원인 추적이 어려워지기 때문 — 좌/우를 각각 독립적으로
    EMA하면 "그 사이드가 실제로 어디 있(었)는지"만 부드러워지고, 중점/반폭 폴백은 매
    프레임 그 스무딩된 좌/우 값으로 그대로 다시 계산한다(_build_path, 변경 없음).

    prev_boxes가 없거나(첫 호출) 길이가 다르면(설정 변경 등 비정상 상황) 블렌딩할 기준이
    없으므로 이번 프레임 값(boxes)을 그대로 쓴다. 인덱스 i에서 이번 프레임에 검출이
    없으면(None) 그 사이드는 "안 보인다"를 그대로 인정하고 직전값을 이어붙이지 않는다
    (사라진 콘을 잔상으로 계속 믿고 가면 더 위험할 수 있다는 판단) — 반대로 직전 프레임에
    없었는데 이번에 새로 검출됐으면 블렌딩할 과거가 없으므로 이번 프레임 값을 그대로 쓴다.
    """
    if prev_boxes is None or len(prev_boxes) != len(boxes):
        return list(boxes)

    def _blend_point(cur, prev):
        if cur is None or prev is None:
            return cur
        return (alpha * cur[0] + (1.0 - alpha) * prev[0],
                alpha * cur[1] + (1.0 - alpha) * prev[1])

    return [(_blend_point(l, pl), _blend_point(r, pr))
            for (l, r), (pl, pr) in zip(boxes, prev_boxes)]


def _build_path(boxes, halfwidth_ema_alpha, sparse_fallback_enabled):
    """`_pick_boxed_sides()`가 반환한 박스별 (left_xy, right_xy)로부터 경로점을 만든다.

    - 양쪽 다 있는 박스: 기존과 동일하게 두 점의 중점.
    - 한쪽만 있는 박스: `sparse_fallback_enabled`일 때만, 직전까지 양쪽 다 검출됐던
      박스들의 좌우 반폭 EMA만큼 검출된 쪽 반대로 밀어 중심선을 추정한다(반폭 EMA가
      아직 없으면, 즉 지금까지 양쪽 다 검출된 박스가 하나도 없었으면 추정 근거가 없으므로
      스킵). `sparse_fallback_enabled`가 False면(기본값) 한쪽만 있는 박스는 그냥 스킵된다.
    - 양쪽 다 없는 박스: 스킵(`_pick_boxed_sides()`가 이런 박스도 (None, None)으로 채워서
      반환하므로 — 프레임간 EMA 인덱스 정렬용).
    """
    mid_x, mid_y = [], []
    hw_ema = None
    for left_xy, right_xy in boxes:
        if left_xy is None and right_xy is None:
            continue
        if left_xy is not None and right_xy is not None:
            cx = (left_xy[0] + right_xy[0]) / 2.0
            cy = (left_xy[1] + right_xy[1]) / 2.0
            hw = abs(left_xy[1] - right_xy[1]) / 2.0
            hw_ema = hw if hw_ema is None else (
                halfwidth_ema_alpha * hw + (1.0 - halfwidth_ema_alpha) * hw_ema)
            mid_x.append(cx)
            mid_y.append(cy)
            continue

        if not sparse_fallback_enabled or hw_ema is None:
            continue  # 반폭 추정 근거가 아직 없거나 폴백이 꺼져 있으면 이 박스는 스킵

        known_xy = left_xy if left_xy is not None else right_xy
        side_sign = 1.0 if left_xy is not None else -1.0   # 검출된 쪽: 왼쪽(+)/오른쪽(-)
        est_y = known_xy[1] - side_sign * hw_ema
        mid_x.append(known_xy[0])
        mid_y.append(est_y)

    return mid_x, mid_y


def process_lavacon(lidar_ranges, prev_boxes=None):
    """
    2D 라이다 1스캔(360점)으로부터 라바콘 트랙 중심 편차 + 조향용 경로를 계산한다.

    입력 : lidar_ranges — 길이 360의 거리 배열 (list 또는 np.ndarray)
                          인덱스 0 = 정면, 인덱스 = 각도(도), 반시계
           prev_boxes — 직전 호출이 반환한 boxes(아래 참고). 첫 호출이거나 프레임간
                          EMA를 안 쓰면 None으로 두면 된다(기본값). 호출부(track_drive.py
                          perc_lavacon())가 self.* 에 들고 있다가 매 틱 그대로 다시
                          넘겨주는 함수형 상태 스레딩 — process_lavacon() 자체는 상태를
                          갖지 않는다(__main__ 자가 테스트에서도 그대로 재사용 가능).
    출력 : (lavacon_offset, lavacon_done, path_m, boxes) 튜플
           · lavacon_offset (float) : 중심 편차 [-0.8, +0.8], 양수 = 우조향(디버그/로깅용)
           · lavacon_done   (bool)  : 좌/우 모두 콘 미검출 = 라바콘 구간 종료 신호
           · path_m (list[(float,float)]) : 전방으로 쌓아올린 박스(BOX_LON_WIDTH 폭)마다
             좌/우 최근접 1점씩의 중점(양쪽 다 검출된 박스) 또는 반폭 추정 중심선(한쪽만
             검출 + LAVACON_SPARSE_FALLBACK_ENABLED), x(전방) 오름차순(박스를 순서대로
             훑으므로 자연히 정렬됨), (x, y) 라이다 미터 좌표(x=전방+, y=좌측+). 조향
             실계산은 이걸 쓴다. 유효한 박스가 하나도 없으면 빈 리스트.
           · boxes (list[(xy_or_None, xy_or_None)]) : 박스별 좌/우 포인트 —
             LAVACON_TEMPORAL_EMA_ENABLED면 프레임간 EMA가 이미 반영된 값(_blend_boxes_temporal
             참고), 아니면 이번 프레임 raw 값. path_m은 항상 이 boxes로부터 계산된다.
             다음 호출의 prev_boxes로 그대로 넘기면 된다 — 콘 미검출 등으로 유효한 박스가
             하나도 없어도(위 path_m이 빈 리스트여도) None이 아니라 이 값을 반환한다.
    """
    # ── 0~2) 입력 유효성 검사 + 전처리 + 극좌표→직교좌표 변환 (_lidar_to_xy 공용) ──
    x, y, ranges = _lidar_to_xy(lidar_ranges)
    if x is None:
        return (0.0, True, [], None)

    # ── 3) 종료 판정용 콘 후보 필터링 : 진입 트리거와 동일 크기 ROI 안의 유효 점만 남김 ──
    # "진입 때 본 것과 같은 크기의 박스에 1개도 안 찍히면 종료"로 판정하기 위해
    # EXIT_LON_MIN/MAX/LAT_LIMIT(perc_lavacon_trigger() 트리거 박스와 동일 크기, 위 상수
    # 선언부 주석 참고)를 쓴다.
    cone_mask = (ranges > 0.0) & (x > EXIT_LON_MIN) & (x < EXIT_LON_MAX) & (np.abs(y) < EXIT_LAT_LIMIT)
    py_cone = y[cone_mask]

    # ── 4) 종료 판정 : 좌·우 어느 쪽에도 콘이 없어야 라바콘 구간 끝 ──
    # 콘 간격이 유동적인 코스에서는 한쪽(예: 우측)에 넓은 틈이 하나만 있어도 곧바로
    # '구간 끝'으로 오판할 수 있으므로, 한쪽 줄이라도 보이면 아직 구간 안이라고 본다.
    # (디바운스는 상위 FSM(_handle_lavacon)의 LAVACON_DONE_FRAMES에서 수행 — 1초 이상 연속
    # 유지돼야 확정, config.py LAVACON_DONE_FRAMES 주석 참고)
    has_left  = bool(np.any(py_cone > 0.0))
    has_right = bool(np.any(py_cone < 0.0))
    lavacon_done = not (has_left or has_right)

    # ── 5) 박스 스택 페어링 : 전방으로 쌓아올린 박스마다 좌/우 최근접 1점 추출 →
    #        (기본 꺼짐) 프레임간 EMA로 "라바콘 차선"(좌/우 포인트) 스무딩 → 중점 계산,
    #        한쪽만 검출된 박스는 반폭 EMA 추정으로 폴백(기본 꺼짐) ──
    # 상세 근거는 _pick_boxed_sides()/_blend_boxes_temporal()/_build_path() 참고.
    # CONE_LON_MAX/CONE_LAT_LIMIT를 그대로 재사용해 박스 탐색 범위를 위 종료판정 ROI와
    # 동일한 한계 안으로 맞춘다.
    boxes_raw = _pick_boxed_sides(x, y, ranges, BOX_LON_START, BOX_LON_WIDTH,
                                   CONE_LON_MAX, CONE_LAT_LIMIT,
                                   LAVACON_LINE_CONTINUITY_ENABLED, LAVACON_LINE_TRACK_MAX_JUMP_M)
    boxes = (_blend_boxes_temporal(boxes_raw, prev_boxes, LAVACON_TEMPORAL_EMA_ALPHA)
             if LAVACON_TEMPORAL_EMA_ENABLED else boxes_raw)
    mid_x, mid_y = _build_path(boxes, LAVACON_HALFWIDTH_EMA_ALPHA, LAVACON_SPARSE_FALLBACK_ENABLED)
    if not mid_x:
        # 유효한 박스가 하나도 없음(콘 미검출 등) — 이번 프레임은 경로 생성을 보류한다.
        # pure_pursuit.control()이 빈 경로를 받으면 직전 조향각을 그대로 유지한다.
        # boxes는 그래도 반환한다 — 다음 프레임 prev_boxes로 이어져야(전부 None이라도)
        # _blend_boxes_temporal()의 길이 비교가 안전하게 계속 성립한다.
        return (0.0, lavacon_done, [], boxes)

    # ── 6) 편차 계산 : 경로점 y좌표 평균 → 부호 반전 → 클램프 (디버그/로깅용) ──
    # y는 좌측+ 이므로, 중심선이 우측(y평균 < 0)에 있으면
    # 우조향(+)이 필요 → offset = -mean(y) 로 부호를 뒤집는다.
    mean_y = float(np.mean(mid_y))
    lavacon_offset = -mean_y * OFFSET_GAIN

    # 물리한계 클램프 : 콘 사이 폭을 넘는 값은 오검출(벽 등)로 보고 잘라냄
    lavacon_offset = float(np.clip(lavacon_offset, -OFFSET_CLAMP, OFFSET_CLAMP))

    path_m = list(zip(mid_x, mid_y))

    return (lavacon_offset, lavacon_done, path_m, boxes)


# ─────────────────────────────────────────────
# 간단 자가 테스트 (ROS 없이 로직 검증용)
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # 가상 시나리오 : 두 "게이트"(좌우 콘 쌍)를 서로 다른 박스에 하나씩 배치.
    #   y값은 CONE_LAT_LIMIT(현재 0.5m)보다 확실히 안쪽으로 잡아야 "< lat_limit" 경계에
    #   걸치는 취약한 테스트가 되지 않는다.
    #   게이트1(전방 0.4m): 좌 y=+0.8, 우 y=-0.5 → 중점 y=+0.15 → offset 기여 -0.15
    #   게이트2(전방 1.0m): 좌 y=+0.6, 우 y=-0.6 → 중점 y=0.0   → offset 기여 0.0
    #   → offset ≈ -(0.15+0.0)/2 = -0.075(좌조향)
    # 박스 폭(BOX_LON_WIDTH, 현재 0.4m)보다 좁게 좌우를 같은 x에 둬야 같은 박스에 페어링된다.
    test = np.zeros(360, dtype=np.float32)

    def put_xy(fx, fy):
        """전방 fx(m), 좌측+ fy(m) 위치에 콘 하나를 놓는다 (x=fx, y=fy 직교좌표 → 극좌표 변환)."""
        r = math.hypot(fx, fy)
        true_deg = math.degrees(math.atan2(fy, fx))
        idx = int(round(true_deg + LIDAR_ANGLE_OFFSET_DEG)) % 360
        test[idx] = r

    put_xy(0.4, 0.8)   # 게이트1 좌
    put_xy(0.4, -0.5)  # 게이트1 우
    put_xy(1.0, 0.6)   # 게이트2 좌
    put_xy(1.0, -0.6)  # 게이트2 우

    off, done, path_m, boxes = process_lavacon(test)
    print(f'offset={off:+.3f} (~-0.075 기대: 두 게이트 중점 y평균의 부호반전), '
          f'done={done} (False 기대), path_m={path_m} '
          f'(~[(0.4,0.15),(1.0,0.0)] 기대)')

    # 시나리오 2 : sparse fallback — 박스0은 양쪽(y=+1.0/-1.0, 반폭=1.0), 박스1은 왼쪽만
    # (y=+1.2). LAVACON_SPARSE_FALLBACK_ENABLED=False(기본)면 박스1은 스킵되고,
    # True면 박스0에서 얻은 반폭(1.0)만큼 오른쪽으로 밀어 y=+0.2로 추정한다.
    boxes2 = [((0.4, 1.0), (0.4, -1.0)), ((0.6, 1.2), None)]
    mx_off, my_off = _build_path(boxes2, halfwidth_ema_alpha=0.3, sparse_fallback_enabled=False)
    mx_on,  my_on  = _build_path(boxes2, halfwidth_ema_alpha=0.3, sparse_fallback_enabled=True)
    print(f'sparse_fallback=False: path=({mx_off},{my_off}) (박스1 스킵, [(0.4,0.0)] 기대)')
    print(f'sparse_fallback=True : path=({mx_on},{my_on}) (박스1 반폭 추정, '
          f'[(0.4,0.0),(0.6,0.2)] 기대)')

    # 시나리오 3 : 프레임간 EMA — 직전 프레임엔 박스0 좌=y+1.0이었는데 이번 프레임엔
    # 라이다가 튀어 y+2.0으로 잡혔다고 가정(우측은 안 튐, y-1.0 그대로). alpha=0.5면
    # 블렌딩 결과는 0.5*2.0+0.5*1.0=1.5로 튐이 절반만 반영돼야 한다.
    prev_boxes3 = [((0.4, 1.0), (0.4, -1.0))]
    cur_boxes3  = [((0.4, 2.0), (0.4, -1.0))]
    blended3 = _blend_boxes_temporal(cur_boxes3, prev_boxes3, alpha=0.5)
    print(f'temporal_ema: blended={blended3} (좌 y=1.5로 절반만 반영 기대, 우는 안 튀었으니 -1.0 그대로)')

    # prev_boxes가 None(첫 호출)이거나 길이가 다르면 블렌딩 없이 이번 프레임 값을 그대로 씀.
    blended3_first = _blend_boxes_temporal(cur_boxes3, None, alpha=0.5)
    print(f'temporal_ema(첫 호출, prev=None): blended={blended3_first} (raw 그대로, y=2.0 기대)')

    # 시나리오 4 : nearest_cone_lateral — da+push 모드용 최소 신호. 박스 그리드/페어링 없이
    # ROI 안에서 좌/우 각각 최근접 1점의 y만 뽑는다. 좌 y=+0.4, 우 y=-0.3 콘 하나씩만 배치.
    test4 = np.zeros(360, dtype=np.float32)
    r = math.hypot(0.5, 0.4)
    idx = int(round(math.degrees(math.atan2(0.4, 0.5)) + LIDAR_ANGLE_OFFSET_DEG)) % 360
    test4[idx] = r
    r = math.hypot(0.5, -0.3)
    idx = int(round(math.degrees(math.atan2(-0.3, 0.5)) + LIDAR_ANGLE_OFFSET_DEG)) % 360
    test4[idx] = r
    ly, ry = nearest_cone_lateral(test4, lon_min=0.2, lon_max=1.5, lat_limit=1.0)
    print(f'nearest_cone_lateral: left_y={ly} right_y={ry} (~+0.4/-0.3 기대)')

    # 시나리오 5 : 좌/우 배정 — y부호 vs 라인 연속성. 급커브에서 오른쪽 라인 콘이 차량
    # 헤딩 기준 y>0(수학적으로는 왼쪽)으로 넘어가는 상황 — 직전 박스에서 오른쪽=(0.4,-0.2),
    # 왼쪽=(0.4,0.9)이 확정된 상태에서, 다음 박스엔 오른쪽 라인이 이어진 (0.8,0.2)
    # (y가 양수로 넘어감)와 왼쪽 라인이 이어진 (0.8,0.95) 두 후보만 있다고 하자.
    x5 = np.array([0.8, 0.8])
    y5 = np.array([0.2, 0.95])   # idx0=오른쪽 라인이 이어진 점(y가 양수로 넘어감), idx1=왼쪽 라인이 이어진 점
    ranges5 = np.hypot(x5, y5)
    box_idx5 = np.array([0, 1])
    prev_left5, prev_right5 = (0.4, 0.9), (0.4, -0.2)

    left_by_sign = list(box_idx5[y5[box_idx5] > 0.0])
    right_by_sign = list(box_idx5[y5[box_idx5] < 0.0])
    print(f'y부호만 쓰면: left 후보={left_by_sign} right 후보={right_by_sign} '
          f'(오른쪽 후보 0개 기대 — 둘 다 y>0이라 오른쪽 라인이 이 박스에서 통째로 안 잡힘)')

    l_cont, r_cont = _assign_by_continuity(box_idx5, x5, y5, ranges5, prev_left5, prev_right5, max_jump_m=0.6)
    print(f'line_continuity: left={l_cont} right={r_cont} '
          f'(기대: left=(0.8,0.95) right=(0.8,0.2) — 둘 다 y>0인데도 각자 직전 라인과 이어져서 올바르게 배정)')
