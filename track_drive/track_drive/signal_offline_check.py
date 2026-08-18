#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# signal_offline_check.py — 실차 랩 캡처(예: ~/Downloads/lap_001)를 오프라인으로 재생하며
# SignalDetector.detect_s2()만 반복 호출해 신호등 인식 코드를 바꿀 때마다 이 노트북에서
# 바로 재검증하는 도구 (2026-08-18).
#
# ── 왜 필요한가 ──
# 이 개발환경엔 ROS2 툴체인이 없어(CLAUDE.md) 실시간 카메라/노드 실행으로 검증이 불가능
# 하다. README §1.1/§1.2/§1.3의 "lap_001 70프레임 검증"도 실제로는 이 방식(사전에 실차에서
# 저장해둔 랩 캡처 PNG 시퀀스를 detect_s2()에 그대로 먹이는 오프라인 재생)으로 했던 것이라,
# 그 방법을 스크립트로 남겨 코드 수정 → 재실행 → 결과 비교를 반복하기 쉽게 한다.
#
# ── GUI 없이도 디버그 창을 저장하는 이유 ──
# cv2.imshow는 디스플레이가 있어야 동작한다(자동화 실행 시 에러 위험). --save-viz를 쓰면
# cv2.imshow를 캡처용 스텁으로 "이 스크립트 프로세스 안에서만" 임시 교체해(운영 코드인
# traffic_signal.py는 건드리지 않음) signal4_roi/signal4_board_search 창 내용을 그대로
# PNG로 저장한다. --show는 실제 imshow로 화면에 띄워 직접 눈으로 넘겨보는 모드 — 사람이
# 이 노트북 앞에서 돌릴 때 쓰면 된다.
#
# ── 사용 예 ──
#   python3 signal_offline_check.py --start 20 --end 89 --save-viz /tmp/sig_check
#   python3 signal_offline_check.py --start 20 --end 89 --show
#   python3 signal_offline_check.py --start 20 --end 89 --engine frst --csv /tmp/sig_frst.csv
import argparse
import csv
import glob
import os
import sys

import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../track_drive/track_drive
_PKG_ROOT = os.path.dirname(_THIS_DIR)                     # .../track_drive (setup.py가 있는 쪽)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from track_drive import config as cfg               # noqa: E402  (sys.path 조작 뒤 import)
import track_drive.perception.traffic_signal as ts   # noqa: E402  DEBUG_VIZ_SIGNAL/엔진을 모듈에 직접 패치하려고 모듈째로 import
from track_drive.perception.traffic_signal import SignalDetector  # noqa: E402

_CAPTURED_WINDOWS = {}


def _stub_imshow(name, img):
    _CAPTURED_WINDOWS[name] = img


def _stub_waitkey(_delay=1):
    return -1


def _default_lap_dir():
    return os.path.expanduser('~/Downloads/lap_001')


def _list_frames(frames_dir, start, end):
    paths = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
    if not paths:
        raise SystemExit(f'프레임을 못 찾음: {frames_dir}/frame_*.png')

    def idx_of(p):
        return int(os.path.splitext(os.path.basename(p))[0].split('_')[-1])

    picked = [(idx_of(p), p) for p in paths
              if (start is None or idx_of(p) >= start) and (end is None or idx_of(p) <= end)]
    if not picked:
        raise SystemExit(f'--start/--end 범위에 해당하는 프레임이 없음 ({frames_dir})')
    return picked


def _category(chosen_idx, n_boxes):
    if chosen_idx < 0:
        return 'FAIL'
    return 'fallback' if chosen_idx == n_boxes - 1 else 'auto'


def _state_label(red, straight, left):
    if left:
        return 'LEFT'
    if straight:
        return 'STR'
    if red:
        return 'RED'
    return '---'


def main():
    ap = argparse.ArgumentParser(
        description='lap_001 등 실차 랩 캡처를 오프라인 재생하며 detect_s2()를 반복 검증')
    ap.add_argument('--frames-dir', default=_default_lap_dir(),
                     help='frame_NNNNNN.png들이 있는 디렉토리 (기본: ~/Downloads/lap_001)')
    ap.add_argument('--start', type=int, default=None, help='시작 프레임 번호(포함)')
    ap.add_argument('--end', type=int, default=None, help='끝 프레임 번호(포함)')
    ap.add_argument('--engine', choices=['hough', 'frst'], default=None,
                     help='config.py의 SIG4_CIRCLE_ENGINE 대신 이번 실행만 강제 지정')
    ap.add_argument('--save-viz', default=None,
                     help='이 디렉토리에 프레임별 signal4_roi/signal4_board_search PNG 저장(헤드리스)')
    ap.add_argument('--show', action='store_true',
                     help='실제 imshow 창으로 표시(디스플레이 필요). 창 포커스 상태에서 q로 중단')
    ap.add_argument('--delay-ms', type=int, default=200, help='--show일 때 프레임 사이 대기(ms)')
    ap.add_argument('--csv', default=None, help='프레임별 결과를 이 경로에 CSV로 저장')
    ap.add_argument('--quiet', action='store_true', help='프레임별 한 줄 로그를 생략하고 요약만 출력')
    args = ap.parse_args()

    if args.engine:
        ts.SIG4_CIRCLE_ENGINE = args.engine

    want_viz = bool(args.save_viz or args.show)
    ts.DEBUG_VIZ_SIGNAL = want_viz
    if args.save_viz:
        os.makedirs(args.save_viz, exist_ok=True)
        cv2.imshow = _stub_imshow
        cv2.waitKey = _stub_waitkey

    frames = _list_frames(args.frames_dir, args.start, args.end)
    print(f'[signal_offline_check] {args.frames_dir} | {len(frames)}프레임 '
          f'| engine={ts.SIG4_CIRCLE_ENGINE} '
          f'| autocrop={cfg.SIG4_AUTOCROP_ENABLE} '
          f'| board S<={cfg.SIG4_BOARD_SAT_MAX} V={cfg.SIG4_BOARD_V_MIN}~{cfg.SIG4_BOARD_V_MAX}')

    detector = SignalDetector()
    rows = []
    n_auto = n_fallback = n_fail = 0

    for idx, path in frames:
        frame = cv2.imread(path)
        if frame is None:
            print(f'[{idx:06d}] 이미지 로드 실패: {path}')
            continue

        _CAPTURED_WINDOWS.clear()
        red, straight, left = detector.detect_s2(frame)

        n_boxes = len(detector.s2_board_candidates)
        chosen = detector.s2_chosen_idx
        cat = _category(chosen, n_boxes)
        if cat == 'auto':
            n_auto += 1
        elif cat == 'fallback':
            n_fallback += 1
        else:
            n_fail += 1

        reason = detector.s2_reject_reason or 'OK'
        state = _state_label(red, straight, left)
        row = dict(frame=idx, category=cat, chosen_idx=chosen, n_boxes=n_boxes, state=state,
                   circles=detector.s2_circle_count, reason=reason,
                   bright=detector.s2_brightness, lit=detector.s2_lit,
                   roi_px=detector.s2_roi_px)
        rows.append(row)

        if not args.quiet:
            print(f'[{idx:06d}] {cat:8s} chosen=#{chosen}/{n_boxes - 1} state={state:4s} '
                  f'circles={detector.s2_circle_count} reason={reason} '
                  f'bright={row["bright"]} lit={[int(v) for v in row["lit"]]}')

        if args.save_viz:
            roi_img = _CAPTURED_WINDOWS.get('signal4_roi')
            board_img = _CAPTURED_WINDOWS.get('signal4_board_search')
            if roi_img is not None:
                cv2.imwrite(os.path.join(args.save_viz, f'{idx:06d}_roi.png'), roi_img)
            if board_img is not None:
                cv2.imwrite(os.path.join(args.save_viz, f'{idx:06d}_board.png'), board_img)

        if args.show:
            key = cv2.waitKey(args.delay_ms) & 0xFF
            if key == ord('q'):
                print('사용자 중단(q)')
                break

    if args.show:
        cv2.destroyAllWindows()

    total = n_auto + n_fallback + n_fail
    print('─' * 60)
    print(f'요약: 총 {total}프레임 | 자동크롭 성공 {n_auto} | 고정폴백 성공 {n_fallback} | '
          f'전부 실패 {n_fail}')
    print(f'  자동크롭 성공 프레임: {[r["frame"] for r in rows if r["category"] == "auto"]}')
    print(f'  고정폴백 성공 프레임: {[r["frame"] for r in rows if r["category"] == "fallback"]}')

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader()
            w.writerows(rows)
        print(f'CSV 저장: {args.csv}')


if __name__ == '__main__':
    main()
