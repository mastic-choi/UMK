# =============================================================
# perc_lavacon.py — 보로노이 다이어그램 기반 라바콘 중심선 추정 모듈
#
# [사용법] track_drive.py 에서 import 하여 호출:
#     from perc_lavacon import process_lavacon
#     offset, done = process_lavacon(self.lidar_ranges)
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
# [라이다 좌표 약속] (track_drive.py 실측 기준, 2026-07-22 재확정)
#   · 360칸, 인덱스 = 각도(도), 반시계 방향
#   · 2026-06-19에 "인덱스 0 = 정면"으로 확정했었으나, 2026-07-22 실측(차량 정면에 사람을
#     세우고 디버그 BEV로 확인)에서 그 클러스터가 좌측 90도로 찍히는 오류를 발견 → 실제로는
#     라이다 인덱스 원점이 차량 정면 기준 90도 어긋나 있었다. 아래처럼 LIDAR_ANGLE_OFFSET_DEG(90도)를
#     빼서 보정한 뒤의 각도 기준이 진짜 차량 기준 각도다:
#       (인덱스(도) - LIDAR_ANGLE_OFFSET_DEG) 기준 0 = 정면 / 90 = 좌측 / 180 = 정후방 / 270 = 우측
#   · 인덱스 99~262 는 차체 자기가림 구간 → 항상 무효 처리 (원본 인덱스 기준이라 위 각도 보정과 무관)
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
BODY_LO, BODY_HI = 99, 263      # 차체 가림 인덱스 구간 [99, 262] 마스킹 경계 (263은 미포함)
LIDAR_ANGLE_OFFSET_DEG = 90.0   # 라이다 장착 각도 보정(실측 2026-07-22) — track_drive.py의 동일 상수와 반드시 일치시킬 것
LON_MIN, LON_MAX = 0.0, 4.0     # 보로노이 정점 종방향(전방) 관심영역 (m)
LAT_LIMIT        = 2.5          # 보로노이 정점 횡방향(좌우) 관심영역 한계 (m)
CONE_LON_MAX     = 4.0          # 콘 후보 점의 전방 최대거리 (m) — 벽/원거리 잡음 배제
CONE_LAT_LIMIT   = 2.5          # 콘 후보 점의 횡방향 한계 (m)
OFFSET_CLAMP     = 0.8          # 편차 물리한계 (m) — 콘 사이 폭 초과값은 오검출로 간주
OFFSET_GAIN      = 1.0          # y평균 → offset 스케일 계수 (제어팀 LAVACON_KP와 별도, 여기선 1:1)
MIN_POINTS       = 4            # 보로노이 계산 최소 점수 (4점 미만이면 다이어그램 불가/무의미)
 
 
def process_lavacon(lidar_ranges):
    """
    2D 라이다 1스캔(360점)으로부터 라바콘 트랙 중심 편차를 계산한다.
 
    입력 : lidar_ranges — 길이 360의 거리 배열 (list 또는 np.ndarray)
                          인덱스 0 = 정면, 인덱스 = 각도(도), 반시계
    출력 : (lavacon_offset, lavacon_done) 튜플
           · lavacon_offset (float) : 중심 편차 [-0.8, +0.8], 양수 = 우조향
           · lavacon_done   (bool)  : 우측(y<0) 콘 미검출 = 라바콘 구간 종료 신호
    """
    # ── 0) 입력 유효성 검사 : None 이거나 비어 있으면 즉시 안전 폴백 ──
    if lidar_ranges is None:
        return (0.0, True)
 
    # ── 1) 전처리 : NumPy 배열화 + 무효값(inf/nan/음수/0) 제거 + 차체 마스킹 ──
    ranges = np.asarray(lidar_ranges, dtype=np.float32).copy()  # 원본 훼손 방지 복사
    n = len(ranges)
    if n == 0:
        return (0.0, True)
 
    ranges[~np.isfinite(ranges)] = 0.0     # inf / nan → 0.0 (무효 표시)
    ranges[ranges <= 0.0] = 0.0            # 0 이하 거리 → 무효
 
    # 차체 자기가림 구간(인덱스 99~262)을 0.0으로 마스킹 → 전방(0~98, 263~359)만 사용
    if n > BODY_LO:
        ranges[BODY_LO:min(BODY_HI, n)] = 0.0
 
    # ── 2) 극좌표 → 직교좌표 변환 (x: 전방+, y: 좌측+) ──
    # 인덱스가 각도(도)이지만 0번이 정면이 아니라 실측상 정면 기준 90도 어긋나 있으므로
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
 
    # ── 4) 종료 판정 : 우측(y<0) 콘 소멸 = 라바콘 구간 끝 ──
    # (코스 특성상 우측 콘이 먼저 사라짐 — 디바운스는 상위 FSM(_s1_lavacon)에서 수행)
    lavacon_done = not bool(np.any(py < 0.0))
 
    # ── 5) 안전 폴백 : 유효 점이 4개 미만이면 보로노이 계산 불가 → 직진 + 종료 신호 ──
    if len(px) < MIN_POINTS:
        return (0.0, True)
 
    # ── 6) 보로노이 다이어그램 계산 ──
    # 콘 점군을 시드로 하면, 좌/우 콘 경계 '사이의 등거리 능선'이
    # 보로노이 간선/정점으로 나타난다 = 트랙의 안전 중심선(골격).
    # 좌우 콘 개수가 비대칭이어도(급커브) 위상적으로 올바른 골격이 나온다.
    points = np.column_stack((px, py))      # (N, 2) 형태로 결합
    try:
        vor = Voronoi(points)
    except (QhullError, ValueError):
        # 점들이 일직선상에 놓이는 등 퇴화(degenerate) 입력이면 Qhull이 실패함
        # → 편차 0(직진) 유지, 구간은 계속(False)으로 두어 다음 프레임에 재시도
        return (0.0, False)
 
    # ── 7) 보로노이 정점 필터링 : 차량 전방의 주행 가능 영역 내 정점만 채택 ──
    # 보로노이 정점은 트랙 바깥 먼 곳에도 생기므로(무한 간선 근처),
    # '지금 따라가야 할 중심선' 조각만 남긴다:
    #   · 전방 0 ~ 4 m (LON_MIN < vx < LON_MAX)
    #   · 좌우 ±2.5 m (|vy| < LAT_LIMIT)
    verts = vor.vertices                    # (M, 2) 보로노이 정점 배열
    if len(verts) == 0:
        return (0.0, lavacon_done)          # 정점이 아예 없으면 직진 유지
 
    vx = verts[:, 0]
    vy = verts[:, 1]
    keep = (vx > LON_MIN) & (vx < LON_MAX) & (np.abs(vy) < LAT_LIMIT)
    sel_y = vy[keep]                        # 채택된 중심선 정점들의 y좌표(좌측+)
 
    if len(sel_y) == 0:
        # ROI 안에 중심선 정점이 하나도 없음 → 이번 프레임은 조향 판단 보류(직진)
        return (0.0, lavacon_done)
 
    # ── 8) 편차 계산 : 중심선 정점 y좌표 평균 → 부호 반전 → 클램프 ──
    # y는 좌측+ 이므로, 중심선이 우측(y평균 < 0)에 있으면
    # 우조향(+)이 필요 → offset = -mean(y) 로 부호를 뒤집는다.
    mean_y = float(np.mean(sel_y))
    lavacon_offset = -mean_y * OFFSET_GAIN
 
    # 물리한계 클램프 : 콘 사이 폭을 넘는 값은 오검출(벽 등)로 보고 잘라냄
    lavacon_offset = float(np.clip(lavacon_offset, -OFFSET_CLAMP, OFFSET_CLAMP))
 
    return (lavacon_offset, lavacon_done)
 
 
# ─────────────────────────────────────────────
# 간단 자가 테스트 (ROS 없이 로직 검증용)
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # 가상 시나리오 : 좌측 콘 2 m, 우측 콘 1 m 거리에 배치된 직선 트랙
    # → 중심선은 y ≈ +0.5 (좌측) → offset ≈ -0.5 (좌조향) 기대
    # 주의: 원본 라이다 인덱스는 LIDAR_ANGLE_OFFSET_DEG(90도)만큼 정면과 어긋나 있으므로,
    # "차량 기준 목표각(target_deg)"에 오프셋을 더한 실제 인덱스에 테스트 값을 채운다.
    test = np.zeros(360, dtype=np.float32)
    offset_i = int(LIDAR_ANGLE_OFFSET_DEG)
    # 좌측 콘들 (차량 기준 목표각 40~70도 부근, y ≈ +2.0 라인 근사)
    for target_deg in (40, 55, 70):
        i = (target_deg + offset_i) % 360
        test[i] = 2.0 / math.sin(math.radians(target_deg))   # y = r·sin(target_deg) = 2.0 이 되도록 역산
    # 우측 콘들 (차량 기준 목표각 290~320도 부근, y ≈ -1.0 라인 근사)
    for target_deg in (290, 305, 320):
        i = (target_deg + offset_i) % 360
        test[i] = -1.0 / math.sin(math.radians(target_deg))  # y = r·sin(target_deg) = -1.0 (sin<0 이라 r>0)
    off, done = process_lavacon(test)
    print(f'offset={off:+.3f} (음수=좌조향 기대), done={done}')
