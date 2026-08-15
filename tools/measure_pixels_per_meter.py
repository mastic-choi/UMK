#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# measure_pixels_per_meter.py — DL_PIXELS_PER_METER(BEV 픽셀↔미터 환산) 재검증 도구.
#
# ── 왜 필요한가 ──
# config.py의 DL_PIXELS_PER_METER=200.0에는 "설계값(실측 아님) — 목적 캔버스를
# 1m=200px 스케일로 만든다"라고 명시돼 있다. 이 값은 실제로 잰 게 아니라 목적
# 캔버스 크기를 정하려고 "그냥 고른" 숫자다 — da 안전마진(_apply_vehicle_margin),
# 라이다-카메라 오프셋 등 여러 곳이 이 값을 "1px가 실제로 몇 m인지"의 근거로 쓰고
# 있어서, 실제로 맞는지 확인이 필요하다.
#
# ── 왜 근거리에서는 self-consistent 한지, 그런데도 재검증이 필요한지 ──
# BEV 호모그래피(_dl_M0)는 실측 캘리브레이션 사각형(가로 0.8m×세로 1.0m,
# README §6.3)을 "0.8*PPM × 1.0*PPM" px 크기의 목적 사각형으로 매핑해서 만든다.
# 그 목적 사각형 안(=원래 캘리브레이션한 근거리 ~1m까지)에서는 변환→역변환에 쓰는
# PPM이 서로 상쇄되어, PPM 값 자체가 뭐든 근거리 결과는 이미 자체 일관적(self-
# consistent)이다 — 그래서 근거리에서는 이 도구로 재봐도 오차가 잘 안 보일 수 있다.
# 문제는 그 사각형 **밖**(원거리 외삽, config.py DL_BEV_FAR_LIMIT_M 참고)이다 —
# 원근 왜곡·카메라 마운트 각도·피치 변화가 커지는 원거리일수록 "PPM이 실제로도
# 똑같이 맞는다"는 보장이 없다. 그래서 이 도구는 **여러 거리에서 반복 측정**해서
# 거리에 따라 오차가 커지는지(=원거리 외삽 취약성이 실제로 있는지) 보는 용도로 쓸 것.
#
# ── 원리 ──
# 실제 길이를 아는 물체(줄자, 1m 막대 등)를 바닥에 놓고 양 끝점을 카메라 BEV
# 화면에서 클릭한다. BEV 변환(현재 DL_PIXELS_PER_METER 가정)으로 계산된 두 점
# 사이 거리(measured_m)를 실측 길이(known_distance_m)와 비교해서, 얼마나
# 차이나는지와 그 비율만큼 PPM을 어떻게 보정해야 하는지 계산한다.
#
# ── 이 파일은 ROS2 빌드/노드 없이 그냥 실행합니다 ──
#   python3 measure_pixels_per_meter.py
# 카메라는 cv2.VideoCapture로 직접 연다(usb_cam_node_exe/ROS2 안 거침).
# track_drive 패키지를 import하지 않는다(필요한 상수는 config.py에서 그대로
# 복사해왔음 — 값이 바뀌면 이 파일도 같이 갱신할 것).
#
# ── 사용법 ──
#   1) 길이를 아는 물체(예: 줄자로 정확히 1m 표시한 막대/테이프)를 준비한다.
#   2) python3 measure_pixels_per_meter.py [--known-distance-m 1.0] [--samples 3]
#   3) 물체를 바닥에 놓고(첫 샘플은 차량 정면 가까운 곳부터), 안내에 따라 양 끝점을
#      각각 클릭 + Enter/Space로 확정 — 한 샘플에 두 번 클릭한다.
#   4) 다음 샘플은 더 먼 위치에 놓고 반복(--samples만큼) — 거리별 오차 추이를 보려는
#      목적이니 가능하면 서로 다른 거리에서 잴 것.
#   5) 마지막에 샘플별/전체 보정 배율과, config.py에 붙여넣을 코드를 출력한다.
#=============================================
import argparse
import sys
import time

import cv2
import numpy as np

# ── config.py와 반드시 동기화 유지 (이 값들이 바뀌면 여기도 같이 고칠 것) ──
DL_ROI_Y0, DL_ROI_Y1 = 250, 390
DL_BEV_SRC_PX_RAW = np.float32([
    [246, 257],  # TL(좌상/먼왼쪽)
    [455, 257],  # TR(우상/먼오른쪽)
    [635, 333],  # BR(우하/가까운오른쪽)
    [60,  333],  # BL(좌하/가까운왼쪽)
])
DL_PIXELS_PER_METER = 200.0


def build_camera_bev():
    """dl_lane.py의 _dl_M0과 동일 정의 — measure_lidar_camera_offset.py와 동일 함수."""
    src = DL_BEV_SRC_PX_RAW - np.float32([0, DL_ROI_Y0])
    block_w = 0.8 * DL_PIXELS_PER_METER   # 실측 폭 0.8m (README §6.3)
    block_h = 1.0 * DL_PIXELS_PER_METER   # 실측 길이 1.0m (근거리~1m 지점, README §6.3)
    dst = np.float32([[0, 0], [block_w, 0], [block_w, block_h], [0, block_h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return M, block_w, block_h


def raw_px_to_cam_xy(px, py, M, block_w, block_h):
    """원본 프레임(640x480) 절대 픽셀 좌표 -> 카메라 BEV 기준 (forward_m, lateral_m).
    현재 config.py DL_PIXELS_PER_METER 값을 그대로 가정해서 미터로 환산한다 — 이
    도구가 검증하려는 게 바로 이 가정이 맞는지다."""
    pt = np.float32([[[px, py - DL_ROI_Y0]]])
    xb, yb = cv2.perspectiveTransform(pt, M)[0, 0]
    forward_m = (block_h - yb) / DL_PIXELS_PER_METER
    lateral_m = (xb - block_w / 2.0) / DL_PIXELS_PER_METER
    return float(forward_m), float(lateral_m)


def _to_640x480(frame):
    """measure_lidar_camera_offset.py의 동일 함수 참고 — 종횡비 왜곡 방지 center-crop."""
    h, w = frame.shape[:2]
    if (w, h) == (640, 480):
        return frame
    target_ar = 640.0 / 480.0
    src_ar = w / float(h)
    if src_ar > target_ar:
        new_w = int(round(h * target_ar))
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    elif src_ar < target_ar:
        new_h = int(round(w / target_ar))
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h, :]
    return cv2.resize(frame, (640, 480))


def capture_camera_point(cap, M, bw, bh, prompt_label):
    """카메라 라이브 뷰에서 한 점을 클릭+확정 받아 (forward_m, lateral_m)으로 반환.
    q를 누르면 None. measure_lidar_camera_offset.py의 capture_camera_point()와
    동일 UX(클릭=미리보기, Enter/Space=확정)이되, cap을 매번 새로 열지 않고 재사용한다
    (한 샘플에 두 점을 연달아 찍어야 해서)."""
    clicked = {}

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked['pt'] = (x, y)

    win = f'camera - {prompt_label} (q=cancel)'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    result = None
    warned_resize = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03)
                continue
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                if not warned_resize:
                    print(f'  [주의] 카메라가 {frame.shape[1]}x{frame.shape[0]}로 열렸습니다 '
                          f'— 640x480으로 center-crop+리사이즈해서 표시합니다.')
                    warned_resize = True
                frame = _to_640x480(frame)
            vis = frame.copy()
            cv2.rectangle(vis, (0, DL_ROI_Y0), (vis.shape[1] - 1, DL_ROI_Y1), (0, 255, 255), 1)
            pts = DL_BEV_SRC_PX_RAW.astype(np.int32)
            cv2.polylines(vis, [pts], True, (255, 0, 255), 1)
            cv2.putText(vis, f'{prompt_label}   click=pick   ENTER/SPACE=confirm   q=cancel',
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            if 'pt' in clicked:
                cv2.circle(vis, clicked['pt'], 6, (0, 0, 255), -1)
                fwd, lat = raw_px_to_cam_xy(clicked['pt'][0], clicked['pt'][1], M, bw, bh)
                cv2.putText(vis, f'forward={fwd:.3f}m lateral={lat:.3f}m  [ENTER/SPACE=confirm]',
                            (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0), 2, cv2.LINE_AA)
                result = (fwd, lat)
            cv2.imshow(win, vis)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'):
                result = None
                break
            if k in (13, 10, 32) and result is not None:
                break
    finally:
        cv2.destroyWindow(win)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--camera-index', type=int, default=0)
    ap.add_argument('--known-distance-m', type=float, default=1.0,
                     help='두 끝점 사이 실측(줄자) 거리(m)')
    ap.add_argument('--samples', type=int, default=3, help='서로 다른 거리에서 반복할 횟수')
    args = ap.parse_args()

    print(f'=== DL_PIXELS_PER_METER 재검증 ({args.samples}회, 매회 실측거리 '
          f'{args.known_distance_m:.3f}m 가정) ===')
    print('길이를 아는 물체(줄자로 정확히 잰 막대/테이프)의 양 끝점을 클릭합니다.')
    print('첫 샘플은 가까운 곳(BEV 캘리브레이션 범위 안, ~1m 이내)부터, 그 다음은')
    print('점점 먼 곳에 놓고 반복하면 "원거리일수록 오차가 커지는지"를 볼 수 있습니다.\n')

    M, bw, bh = build_camera_bev()
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f'카메라(index={args.camera_index})를 열 수 없습니다 — 다른 프로세스가 '
              f'점유 중이거나(usb_cam_node_exe가 떠 있으면 먼저 끌 것) 인덱스가 '
              f'다를 수 있습니다(--camera-index로 다른 값 시도).')
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ratios = []
    try:
        for i in range(args.samples):
            print(f'--- 샘플 {i+1}/{args.samples} ---')
            label = input('  이 샘플을 대략 몇 m 지점에 놓았는지(기록용, 그냥 엔터로 생략 가능): ').strip()

            a = capture_camera_point(cap, M, bw, bh, '끝점 A 클릭')
            if a is None:
                print('  취소됨 — 이 샘플 건너뜀')
                continue
            b = capture_camera_point(cap, M, bw, bh, '끝점 B 클릭')
            if b is None:
                print('  취소됨 — 이 샘플 건너뜀')
                continue

            measured_m = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            ratio = args.known_distance_m / measured_m if measured_m > 1e-6 else float('nan')
            depth_tag = f' (~{label}m 지점)' if label else ''
            print(f'  A=(fwd={a[0]:.3f}, lat={a[1]:.3f})  B=(fwd={b[0]:.3f}, lat={b[1]:.3f})')
            print(f'  측정거리={measured_m:.3f}m  실측(줄자)거리={args.known_distance_m:.3f}m'
                  f'  보정배율={ratio:.3f}{depth_tag}\n')
            ratios.append((label, measured_m, ratio))
    finally:
        cap.release()

    if not ratios:
        print('측정된 샘플이 없습니다. 종료합니다.')
        return 1

    ratio_vals = [r for (_l, _m, r) in ratios]
    ratio_mean = float(np.mean(ratio_vals))
    ratio_std = float(np.std(ratio_vals))
    suggested_ppm = DL_PIXELS_PER_METER * ratio_mean

    print('=== 결과 ===')
    for label, measured_m, ratio in ratios:
        tag = f'~{label}m' if label else '(거리 미기록)'
        print(f'  {tag:>10s}: 측정거리={measured_m:.3f}m  보정배율={ratio:.3f}')
    print(f'\n보정배율 평균={ratio_mean:.3f} (표준편차={ratio_std:.3f})')
    if ratio_std > 0.05:
        print('★주의★ 샘플 간 보정배율 표준편차가 5% 넘게 큽니다 — 거리별로 오차가')
        print('        다르다는 뜻(원거리 외삽 취약성, README §2.30/§2.2)일 가능성이')
        print('        높습니다. 이 경우 PPM 상수 하나로는 다 못 고치고, 근거리용/')
        print('        원거리용을 따로 두거나 거리 함수로 보정하는 걸 고려해야 합니다.')

    print(f'\n현재 DL_PIXELS_PER_METER = {DL_PIXELS_PER_METER:.1f}')
    print(f'제안 DL_PIXELS_PER_METER = {suggested_ppm:.1f}  (= 현재값 × 보정배율 평균)')
    print('\nconfig.py에 붙여넣을 코드:')
    print(f'# [실측 재검증 — 날짜/측정자/샘플 거리대 채울 것]')
    print(f'DL_PIXELS_PER_METER = {suggested_ppm:.1f}   # 이전 200.0(설계값) → 실측 보정')
    print('\n★주의★ 이 값을 실제로 config.py에 반영하면 DL_DA_VEHICLE_MARGIN_M 침식 반경,')
    print('  PASS_OFFSET, PP_WHEELBASE_PX 등 DL_PIXELS_PER_METER를 참조하는 다른 상수들의')
    print('  픽셀 환산도 전부 같이 바뀐다 — 반영 전 어디서 쓰이는지 한 번 더 확인할 것.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
