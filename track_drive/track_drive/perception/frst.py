"""Fast Radial Symmetry Transform (Loy, G., & Zelinsky, A. 2002,
"A fast radial symmetry transform for detecting points of interest", ECCV 2002).

[2026-08-18] 신호등(perception/traffic_signal.py) 원 탐지 엔진 후보로 도입. 각 픽셀의
그래디언트 방향으로 radius만큼 떨어진 위치에 "투표"를 누적해, 그 위치를 중심으로 그래디언트가
방사형으로 대칭인지를 점수화한다 — Hough Circle Transform처럼 뚜렷한 원형 엣지가 필요 없고
그래디언트 방향 통계만 맞으면 반응하기 때문에, 모션블러로 원 경계가 뭉개진 상황에 더 강건하다
(lap_001 실차 랩 캡처로 확인 — frame_000052에서 Hough는 후보 크롭 안의 천장조명과 진짜 램프가
섞인 4개 조합을 잘못 통과시켰는데, FRST는 같은 크롭에서 섞임 없이 진짜 4개 램프만 찾았다).

주의: 이 넓은 범위(전체 프레임)에서 단독으로 돌리면 바닥 무늬/천장 조명 등에서 오탐이 여전히
난다는 것도 같은 실측으로 확인됐다 — 그래서 config.py SIG4_CIRCLE_ENGINE 주석에 적어뒀듯,
전체화면 단독 탐색이 아니라 흰 배경판 후보 크롭(SIG4_BOARD_*) "안에서만" 돌리는 용도로 쓴다.
C++ 원본 구현(github.com/Xonxt/frst)을 참고해 표준 (row,col)=(y,x) 좌표계로 옮기고 numpy로
벡터화했다."""

import cv2
import numpy as np


def frst(gray, radius, alpha=2.0, std_factor=0.25, mode='bright'):
    """단일 반지름(scale)에 대한 FRST 대칭도 맵을 반환한다. mode='bright'면 어두운 배경 위
    밝은 원형(신호등 램프처럼)에, 'dark'면 그 반대에, 'both'면 둘 다에 반응한다."""
    gray = gray.astype(np.float64)
    gy, gx = np.gradient(gray)
    gnorm = np.sqrt(gx * gx + gy * gy)

    h, w = gray.shape
    pad = radius
    O_n = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float64)
    M_n = np.zeros_like(O_n)

    ys, xs = np.nonzero(gnorm > 1e-6)
    gn = gnorm[ys, xs]
    dx = np.round((gx[ys, xs] / gn) * radius).astype(np.int64)
    dy = np.round((gy[ys, xs] / gn) * radius).astype(np.int64)

    if mode in ('bright', 'both'):
        px, py = xs + dx + pad, ys + dy + pad
        np.add.at(O_n, (py, px), 1.0)
        np.add.at(M_n, (py, px), gn)
    if mode in ('dark', 'both'):
        px, py = xs - dx + pad, ys - dy + pad
        np.add.at(O_n, (py, px), -1.0)
        np.add.at(M_n, (py, px), -gn)

    O_n = np.abs(O_n)
    o_max = O_n.max()
    if o_max > 0:
        O_n = O_n / o_max
    M_n = np.abs(M_n)
    m_max = M_n.max()
    if m_max > 0:
        M_n = M_n / m_max

    S = (O_n ** alpha) * M_n

    ksize = int(np.ceil(radius / 2))
    if ksize % 2 == 0:
        ksize += 1
    S = cv2.GaussianBlur(S, (ksize, ksize), radius * std_factor)

    return S[pad:pad + h, pad:pad + w]


def frst_multiscale(gray, radii, alpha=2.0, std_factor=0.25, mode='bright'):
    """radii(우리 램프 반지름 후보군, SIG4_MIN/MAX_RADIUS 범위)마다 frst()를 계산해, 픽셀별로
    가장 강한 반응을 낸 스케일을 그 픽셀의 '추정 반지름'으로 함께 반환한다.
    반환: (combined_symmetry_map, best_radius_map)"""
    best, best_r = None, None
    for r in radii:
        s = frst(gray, r, alpha=alpha, std_factor=std_factor, mode=mode)
        if best is None:
            best, best_r = s.copy(), np.full(s.shape, r, dtype=np.int32)
        else:
            mask = s > best
            best = np.where(mask, s, best)
            best_r = np.where(mask, r, best_r)
    return best, best_r


def find_peaks(sym_map, radius_map, max_peaks, min_dist, rel_threshold):
    """대칭도 맵에서 국소최대(local maxima) 상위 max_peaks개를 (x,y,r) 리스트로 반환한다 —
    cv2.HoughCircles() 반환 형식과 맞춰서 shape_ok()/pick_best_4()를 그대로 재사용하기 위함.
    NMS(비최대 억제)로 min_dist 이내 중복 피크를 제거하고, 프레임 최댓값의 rel_threshold
    미만인 약한 피크는 버린다(이 값이 너무 낮으면 후보가 항상 max_peaks까지 꽉 차서 노이즈까지
    다 들어온다 — config.py SIG4_FRST_REL_THRESHOLD 주석 참고, 실측으로 재튜닝된 값)."""
    m = sym_map.copy()
    thresh = m.max() * rel_threshold
    peaks = []
    while len(peaks) < max_peaks:
        idx = np.unravel_index(np.argmax(m), m.shape)
        val = m[idx]
        if val < thresh:
            break
        y, x = idx
        r = int(radius_map[y, x])
        peaks.append((int(x), int(y), r))
        y0, y1 = max(0, y - min_dist), y + min_dist
        x0, x1 = max(0, x - min_dist), x + min_dist
        m[y0:y1, x0:x1] = -1
    return peaks
