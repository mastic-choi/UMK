# =============================================================
# perc_lavacon.py — 보로노이 다이어그램 기반 라바콘 중심선 추정 모듈
#
# [사용법] track_drive.py 에서 import 하여 호출 (half_width_hint는 프레임마다 그대로 이어받아 넘길 것):
#     from perc_lavacon import process_lavacon, HALF_WIDTH_DEFAULT
#     self._lavacon_half_width = HALF_WIDTH_DEFAULT   # __init__에서 1회
#     offset, done, self._lavacon_half_width = process_lavacon(self.lidar_ranges, self._lavacon_half_width)
#
# [알고리즘 개요]
#   곡선 구간에서는 안쪽 콘과 바깥쪽 콘의 개수가 크게 비대칭이라
#   단순 1:1 중점 페어링은 실패한다.
#   보로노이 다이어그램(scipy.spatial.Voronoi)은 점군 사이의
#   '위상학적 골격(topological skeleton)'을 자연스럽게 추출하므로,
#   좌/우 콘 개수가 달라도 두 경계 사이의 안전 중심선이
#   보로노이 정점(vertex)들로 나타난다. 이 정점들의 y좌표 평균을
#   횡방향 편차(lavacon_offset)로 사용한다.
#
#   [종료/한쪽 소실 처리] (2026-07-22 실측 변경)
#     실측 결과 코스 끝에서 좌·우 콘이 "동시에" 사라지는 것으로 확인되어, 종료 판정도
#     좌우가 둘 다 안 보일 때만 True가 되도록 바꿨다(예전엔 우측만 사라지면 종료로 봤음).
#     주행 중 한쪽만 일시적으로 안 보이면(가림·노이즈 등), 그동안 좌우가 둘 다 보였던
#     프레임에서 학습해 둔 코스 반폭(half_width_hint)만큼 보이는 쪽 콘에서 안쪽으로 들어간
#     지점을 "반대쪽 콘이 있을 자리"로 추정해서 중심선을 계속 따라가도록 한다.
#
# [라이다 좌표 약속] (track_drive.py 실측 기준, 2026-07-22 재확정)
#   · 360칸, 인덱스 = 각도(도), 반시계 방향
#   · 2026-06-19에 "인덱스 0 = 정면"으로 확정했었으나, 2026-07-22 1차 실측(차량 정면에 사람을
#     세우고 디버그 BEV로 확인)에서 인덱스 90이 정면으로 재확인되었고, 같은 날 2차 재실측에서
#     실제로는 인덱스 80이 정면인 것으로 재확정되었다. 아래처럼 LIDAR_ANGLE_OFFSET_DEG(80도)를
#     빼서 보정한 뒤의 각도 기준이 진짜 차량 기준 각도다:
#       (인덱스(도) - LIDAR_ANGLE_OFFSET_DEG) 기준 0 = 정면 / 90 = 좌측 / 180 = 정후방 / 270 = 우측
#   · 인덱스 180~359 는 차체 자기가림 구간 → 항상 무효 처리 (원본 인덱스 기준이라 위 각도 보정과 무관)
#   · 직교좌표 변환: x = r·cos(deg) (전방+), y = r·sin(deg) (좌측+)  [deg는 보정된 각도]
#
# [부호 약속] (track_drive.py 제어팀 합의와 동일)
#   · lavacon_offset > 0 : 중심선이 차량 기준 '우측'에 있음 → 우조향
#   · y(좌측+) 기준으로는 중심선 y평균이 음수일 때 offset이 양수
#     → offset = -mean(y) 로 부호 반전하여 계산한다.
# =============================================================
import math
import numpy as np
from scipy.spatial import Voronoi, QhullError   # 보로노이 다이어그램 + 퇴화 예외
 
# ─────────────────────────────────────────────
# 튜닝 상수 (track_drive.py 의 실측 ROI 값과 일치시킴)
# ─────────────────────────────────────────────
BODY_LO, BODY_HI = 215, 305     # 차체 가림 인덱스 구간 [215, 304] 마스킹 경계 (305는 미포함) — 최종 확정 2026-07-22
LIDAR_ANGLE_OFFSET_DEG = 80.0   # 라이다 장착 각도 보정(재실측 2026-07-22, i80=정면) — track_drive.py의 동일 상수와 반드시 일치시킬 것
BODY_MASK_ENABLED = True        # 최종 확정(2026-07-22) — track_drive.py와 동일하게 유지
LON_MIN, LON_MAX = 0.0, 4.0     # 보로노이 정점 종방향(전방) 관심영역 (m)
LAT_LIMIT        = 2.5          # 보로노이 정점 횡방향(좌우) 관심영역 한계 (m)
CONE_LON_MAX     = 4.0          # 콘 후보 점의 전방 최대거리 (m) — 벽/원거리 잡음 배제
CONE_LAT_LIMIT   = 2.5          # 콘 후보 점의 횡방향 한계 (m)
OFFSET_CLAMP     = 0.8          # 편차 물리한계 (m) — 콘 사이 폭 초과값은 오검출로 간주
OFFSET_GAIN      = 1.0          # y평균 → offset 스케일 계수 (제어팀 LAVACON_KP와 별도, 여기선 1:1)
MIN_POINTS       = 4            # 보로노이 계산 최소 점수 (4점 미만이면 다이어그램 불가/무의미)
MIN_SIDE_PTS     = 2            # 좌/우 "콘이 있다"로 인정할 최소 점수(단일 반사점 노이즈 배제, trigger의 CLUSTER_MIN_PTS와 동일 관례)
# ── 한쪽 콘만 보일 때 반대쪽을 추정하기 위한 코스 반폭(half_width) 학습값 ──
#   실차 실측 전 임시값이므로 실측 후 DEFAULT/MIN/MAX 재조정 필요.
HALF_WIDTH_DEFAULT = 0.5        # 반폭 초기 추정치(m) — 첫 호출부터 좌우가 다 보이기 전까지 이 값을 씀
HALF_WIDTH_MIN     = 0.2        # 반폭 EMA 갱신 시 허용 최소값(m) — 벗어나면 오검출로 보고 갱신 무시
HALF_WIDTH_MAX     = 1.2        # 반폭 EMA 갱신 시 허용 최대값(m)
WIDTH_EMA_ALPHA    = 0.2        # 반폭 추정 EMA 계수(클수록 최근 프레임에 민감)
 
 
def process_lavacon(lidar_ranges, half_width_hint=HALF_WIDTH_DEFAULT, debug=None):
    """
    2D 라이다 1스캔(360점)으로부터 라바콘 트랙 중심 편차를 계산한다.

    입력 : lidar_ranges — 길이 360의 거리 배열 (list 또는 np.ndarray)
                          인덱스 = 각도(도), LIDAR_ANGLE_OFFSET_DEG로 정면 보정
           half_width_hint — 지금까지 학습된 코스 반폭 추정치(m). 좌우가 둘 다 보이는
                          프레임에서 자동 갱신되며, 한쪽만 보일 때 반대쪽 콘 위치를
                          추정하는 데 쓰인다. 호출자는 이전 호출의 반환값을 그대로
                          다음 호출에 넘겨야 한다(최초 호출은 HALF_WIDTH_DEFAULT).
           debug — dict를 넘기면(예: {}) 이번 프레임의 중간 계산값을 그 dict에 채워 넣는다
                          (offset이 왜 그렇게 나왔는지 진단용). 기본 None이면 아무 오버헤드 없음.
                          채워지는 키는 아래 _finish() 참고.
    출력 : (lavacon_offset, lavacon_done, half_width_hint) 튜플
           · lavacon_offset (float) : 중심 편차 [-0.8, +0.8], 양수 = 우조향
           · lavacon_done   (bool)  : 좌우 콘이 동시에 미검출 = 라바콘 구간 종료 신호
           · half_width_hint(float) : 다음 호출에 그대로 넘겨줄 갱신된 반폭 추정치
    """
    dbg = {
        'branch': 'none',        # 'none'(좌우 다 소실) / 'both'(정상 보로노이) / 'one_side'(반대쪽 추정)
        'reason': '',            # 이번 프레임에 offset=0(직진)이 된 이유(해당시)
        'total_pts': 0, 'left_pts': 0, 'right_pts': 0,
        'near_left_y': None, 'near_right_y': None,
        'width_now': None, 'half_width_before': half_width_hint,
        'vert_total': 0, 'vert_kept': 0,
        'mean_y': None, 'offset_raw': None,
    }

    def _finish(offset, done, hw):
        if debug is not None:
            dbg['offset'] = offset
            dbg['done'] = done
            dbg['half_width_after'] = hw
            debug.clear()
            debug.update(dbg)
        return (offset, done, hw)

    # ── 0) 입력 유효성 검사 : None 이거나 비어 있으면 즉시 안전 폴백 ──
    if lidar_ranges is None:
        dbg['reason'] = 'lidar_ranges=None'
        return _finish(0.0, True, half_width_hint)

    # ── 1) 전처리 : NumPy 배열화 + 무효값(inf/nan/음수/0) 제거 + 차체 마스킹 ──
    ranges = np.asarray(lidar_ranges, dtype=np.float32).copy()  # 원본 훼손 방지 복사
    n = len(ranges)
    if n == 0:
        dbg['reason'] = 'ranges 길이 0'
        return _finish(0.0, True, half_width_hint)

    ranges[~np.isfinite(ranges)] = 0.0     # inf / nan → 0.0 (무효 표시)
    ranges[ranges <= 0.0] = 0.0            # 0 이하 거리 → 무효

    # 차체 자기가림 구간을 0.0으로 마스킹
    if BODY_MASK_ENABLED and n > BODY_LO:
        ranges[BODY_LO:min(BODY_HI, n)] = 0.0

    # ── 2) 극좌표 → 직교좌표 변환 (x: 전방+, y: 좌측+) ──
    # 인덱스가 각도(도)이지만 0번이 정면이 아니라 실측상 정면 기준으로 어긋나 있으므로
    # LIDAR_ANGLE_OFFSET_DEG를 빼서 영점을 보정한다.
    deg = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False) - math.radians(LIDAR_ANGLE_OFFSET_DEG)
    x = ranges * np.cos(deg)    # 종방향(전방거리) 성분
    y = ranges * np.sin(deg)    # 횡방향 성분 — y > 0 좌측, y < 0 우측

    # ── 3) 콘 후보 필터링 : 전방 ROI 안의 유효 점만 남김 ──
    # 벽·원거리 구조물처럼 트랙과 무관한 점이 보로노이에 섞이면
    # 골격이 트랙 밖으로 왜곡되므로, 콘이 존재할 수 있는 영역으로 제한한다.
    cone_mask = (ranges > 0.0) & (x > LON_MIN) & (x < CONE_LON_MAX) & (np.abs(y) < CONE_LAT_LIMIT)
    px = x[cone_mask]           # 콘 후보 점들의 x (전방)
    py = y[cone_mask]           # 콘 후보 점들의 y (좌측+)

    # ── 4) 좌/우 존재 판정 (단일 반사점 노이즈 배제 위해 최소 MIN_SIDE_PTS개 요구) ──
    left_mask  = py > 0.0
    right_mask = py < 0.0
    left_pts  = int(np.count_nonzero(left_mask))
    right_pts = int(np.count_nonzero(right_mask))
    has_left  = left_pts  >= MIN_SIDE_PTS
    has_right = right_pts >= MIN_SIDE_PTS
    dbg['total_pts'] = int(len(px))
    dbg['left_pts']  = left_pts
    dbg['right_pts'] = right_pts

    # ── 5) 종료 판정 : 좌우 콘이 동시에 사라져야 라바콘 구간 끝 ──
    # (2026-07-22 실측 변경: 예전엔 우측만 사라지면 종료로 봤으나, 지금 코스는 좌우가 같이 사라짐.
    #  디바운스는 상위 FSM(_handle_lavacon)의 LAVACON_DONE_FRAMES에서 수행)
    if not has_left and not has_right:
        dbg['reason'] = f'좌우 둘 다 미검출(left_pts={left_pts},right_pts={right_pts} < MIN_SIDE_PTS={MIN_SIDE_PTS})'
        return _finish(0.0, True, half_width_hint)

    # ── 6) 좌우 둘 다 보이는 정상 프레임 : 코스 반폭 학습 + 보로노이 중심선 계산 ──
    if has_left and has_right:
        dbg['branch'] = 'both'
        # 반폭(half_width_hint) 갱신: 가까운 좌/우 콘 y평균 차이의 절반을 이번 프레임 반폭으로
        # 보고 EMA로 스무딩. 튀는 값(오검출)은 HALF_WIDTH_MIN~MAX 범위로 걸러 반영 안 함.
        near_left_y  = float(np.mean(py[left_mask]))
        near_right_y = float(np.mean(py[right_mask]))
        width_now = (near_left_y - near_right_y) / 2.0
        dbg['near_left_y'], dbg['near_right_y'], dbg['width_now'] = near_left_y, near_right_y, width_now
        if HALF_WIDTH_MIN <= width_now <= HALF_WIDTH_MAX:
            half_width_hint = (1.0 - WIDTH_EMA_ALPHA) * half_width_hint + WIDTH_EMA_ALPHA * width_now

        # 보로노이 다이어그램 계산 — 콘 점군을 시드로 하면, 좌/우 콘 경계 '사이의 등거리 능선'이
        # 보로노이 간선/정점으로 나타난다 = 트랙의 안전 중심선(골격). 좌우 콘 개수가 비대칭이어도
        # (급커브) 위상적으로 올바른 골격이 나온다.
        if len(px) < MIN_POINTS:
            dbg['reason'] = f'전체 콘 후보점 {len(px)}개 < MIN_POINTS={MIN_POINTS} (보로노이 계산 불가)'
            return _finish(0.0, False, half_width_hint)
        try:
            vor = Voronoi(np.column_stack((px, py)))
        except (QhullError, ValueError) as e:
            # 점들이 일직선상에 놓이는 등 퇴화(degenerate) 입력이면 Qhull이 실패함
            # → 편차 0(직진) 유지, 구간은 계속(False)으로 두어 다음 프레임에 재시도
            dbg['reason'] = f'Voronoi 퇴화 입력 실패: {e.__class__.__name__}'
            return _finish(0.0, False, half_width_hint)

        # 보로노이 정점 필터링 : 차량 전방의 주행 가능 영역 내 정점만 채택
        #   · 전방 0 ~ 4 m (LON_MIN < vx < LON_MAX)
        #   · 좌우 ±2.5 m (|vy| < LAT_LIMIT)
        verts = vor.vertices                    # (M, 2) 보로노이 정점 배열
        dbg['vert_total'] = int(len(verts))
        if len(verts) == 0:
            dbg['reason'] = '보로노이 정점 자체가 0개'
            return _finish(0.0, False, half_width_hint)   # 정점이 아예 없으면 직진 유지

        vx, vy = verts[:, 0], verts[:, 1]
        keep = (vx > LON_MIN) & (vx < LON_MAX) & (np.abs(vy) < LAT_LIMIT)
        sel_y = vy[keep]                        # 채택된 중심선 정점들의 y좌표(좌측+)
        dbg['vert_kept'] = int(len(sel_y))
        if len(sel_y) == 0:
            # ROI 안에 중심선 정점이 하나도 없음 → 이번 프레임은 조향 판단 보류(직진)
            dbg['reason'] = f'정점 {len(verts)}개 중 ROI(전방0~{LON_MAX}m,좌우±{LAT_LIMIT}m) 안에 남은 게 0개'
            return _finish(0.0, False, half_width_hint)

        # 편차 계산 : 중심선 정점 y좌표 평균 → 부호 반전 → 클램프
        # y는 좌측+ 이므로, 중심선이 우측(y평균 < 0)에 있으면 우조향(+)이 필요
        # → offset = -mean(y) 로 부호를 뒤집는다.
        mean_y = float(np.mean(sel_y))
        dbg['mean_y'] = mean_y
        offset_raw = -mean_y * OFFSET_GAIN
        dbg['offset_raw'] = offset_raw
        offset = float(np.clip(offset_raw, -OFFSET_CLAMP, OFFSET_CLAMP))
        dbg['reason'] = 'OK(보로노이 정상 계산)' + (' [CLAMP 포화]' if abs(offset_raw) > OFFSET_CLAMP else '')
        return _finish(offset, False, half_width_hint)

    # ── 7) 한쪽만 보이는 프레임 : 반대쪽 콘을 half_width_hint로 추정 ──
    # 보이는 쪽 콘들의 y평균에서, 학습된 반폭만큼 안쪽(중심 방향)으로 들어간 지점을
    # "지금 따라가야 할 중심선"으로 본다 (반대쪽 콘이 그 위치에 있다고 가정하는 것과 동일).
    dbg['branch'] = 'one_side'
    visible_y = py[left_mask] if has_left else py[right_mask]
    mean_visible_y = float(np.mean(visible_y))
    if has_left:
        dbg['near_left_y'] = mean_visible_y
    else:
        dbg['near_right_y'] = mean_visible_y
    center_y_est = mean_visible_y - math.copysign(half_width_hint, mean_visible_y)
    dbg['mean_y'] = center_y_est
    offset_raw = -center_y_est * OFFSET_GAIN
    dbg['offset_raw'] = offset_raw
    offset = float(np.clip(offset_raw, -OFFSET_CLAMP, OFFSET_CLAMP))
    dbg['reason'] = f'{"좌" if has_left else "우"}측만 검출 → half_width={half_width_hint:.2f}m로 반대쪽 추정'
    return _finish(offset, False, half_width_hint)
 
 
# ─────────────────────────────────────────────
# 간단 자가 테스트 (ROS 없이 로직 검증용)
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # 주의: 원본 라이다 인덱스는 LIDAR_ANGLE_OFFSET_DEG만큼 정면과 어긋나 있으므로,
    # "차량 기준 목표각(target_deg)"에 오프셋을 더한 실제 인덱스에 테스트 값을 채운다.
    offset_i = int(LIDAR_ANGLE_OFFSET_DEG)

    def make_scan(left_degs=(), left_dist=2.0, right_degs=(), right_dist=1.0):
        scan = np.zeros(360, dtype=np.float32)
        for target_deg in left_degs:
            i = (target_deg + offset_i) % 360
            scan[i] = left_dist / math.sin(math.radians(target_deg))
        for target_deg in right_degs:
            i = (target_deg + offset_i) % 360
            scan[i] = -right_dist / math.sin(math.radians(target_deg))  # sin<0 구간이라 r>0 되도록 부호 반전
        return scan

    # 1) 좌측 콘 1.2m·우측 콘 0.8m, 양쪽 다 보이는 직선 트랙 → 중심선 y≈+0.2 → offset≈-0.2(좌조향) 기대
    #    반폭(width_now)≈1.0m → HALF_WIDTH_DEFAULT(0.5)에서 EMA로 갱신되는 것까지 같이 확인
    scan_both = make_scan(left_degs=(40, 55, 70), left_dist=1.2, right_degs=(290, 305, 320), right_dist=0.8)
    off, done, hw = process_lavacon(scan_both, HALF_WIDTH_DEFAULT)
    print(f'[양쪽 다 보임 ] offset={off:+.3f}(좌조향 기대,음수) done={done} half_width={hw:.3f}(갱신됐는지 확인, 기본값=0.5)')

    # 2) 우측 콘만 순간적으로 안 보임(좌측만 보임) → half_width_hint로 우측을 추정해서 계속 주행해야 함
    scan_left_only = make_scan(left_degs=(40, 55, 70), left_dist=1.0)
    off2, done2, hw2 = process_lavacon(scan_left_only, hw)
    print(f'[좌측만 보임  ] offset={off2:+.3f} done={done2}(False 기대) half_width={hw2:.3f}')

    # 3) 좌우 콘이 동시에 소실 → 구간 종료(done=True) 기대
    scan_none = np.zeros(360, dtype=np.float32)
    off3, done3, hw3 = process_lavacon(scan_none, hw2)
    print(f'[둘다 안 보임 ] offset={off3:+.3f} done={done3}(True 기대) half_width={hw3:.3f}')
