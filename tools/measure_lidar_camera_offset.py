#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# measure_lidar_camera_offset.py — 라이다 원점 ↔ 카메라 BEV 원점 평행이동 오프셋 실측 도구.
#
# ── 왜 필요한가 ──
# perc_obstacle()(track_drive.py)은 라이다 실측 거리(m)로 장애물 위치를 알고,
# dl_lane.py의 BEV는 카메라 화면을 실측 캘리브레이션(DL_BEV_SRC_PX_RAW, README §6.3)으로
# "1m=200px" 좌표계로 바꾼다. 문제는 이 둘이 **서로 다른 원점을 각자 독립적으로
# "차량 중심이겠거니" 가정**하고 있을 뿐, 라이다가 실제로 카메라 BEV 좌표계 기준 어디에
#있는지(전후/좌우 평행이동)는 한 번도 실측한 적이 없다는 것 — LIDAR_ANGLE_OFFSET_DEG
# (config.py, README §6.2)는 "각도(회전)"만 보정했고, 평행이동 오프셋은 아예 없다.
#
# 이 오프셋이 있어야 "라이다가 실측한 장애물 상자(거리+폭, m)"를 카메라 BEV da 마스크
# 위에 정확히 겹쳐 그릴 수 있다 — da 마진(안전여유) 계산을 카메라 픽셀→미터 환산(원거리
# 외삽/피치 변화에 취약, config.py DL_PIXELS_PER_METER 주석 "설계값, 실측 아님" 참고)에만
# 의존하지 않고, 라이다의 실측 거리/폭으로 대신할 수 있게 하려는 목적.
#
# ── 원리 ──
# 같은 물체(상자·라바콘 등)를 특정 위치에 두고, 카메라 BEV에서 본 위치(forward_m,
# lateral_m)와 라이다에서 본 위치(x_m, y_m)를 각각 재서 그 차이(dx, dy)를 구한다.
# 여러 위치(가운데/좌/우)에서 반복해 평균·표준편차를 내면 신뢰도가 올라간다 —
# 표준편차가 크게 튀면 오프셋이 아니라 DL_PIXELS_PER_METER 자체(캘리브레이션 스케일)를
# 의심해볼 신호도 된다.
#
# ── 이 파일은 ROS2 빌드/노드 없이 그냥 실행합니다 ──
#   python3 measure_lidar_camera_offset.py
# track_drive 패키지를 import하지 않는다(관련 상수는 config.py에서 그대로 복사해
# 왔음 — 값이 바뀌면 이 파일도 같이 갱신할 것, 아래 "config.py와 반드시 동기화" 절 참고).
# 카메라는 cv2.VideoCapture로 직접 연다(usb_cam_node_exe/ROS2 안 거침). 라이다는
# rclpy가 설치돼 있고 /scan이 이미 발행 중이면(라이다 드라이버 노드만 떠 있으면 됨 —
# track_drive 전체를 launch할 필요 없음) 자동으로 짧게 구독해서 읽고, rclpy가 없거나
# 응답이 없으면 수동 입력(터미널에 raw range/index 두 숫자만 입력)으로 자동 폴백한다.
#
# ── 사용법 ──
#   1) 라이다 드라이버만 띄워둔다 (예: ros2 launch xycar_lidar ... 또는 팀에서 쓰는 launch
#      파일 중 라이다만 포함된 것 — track_drive 노드 전체를 켤 필요는 없다).
#   2) python3 measure_lidar_camera_offset.py 실행.
#   3) 물체(상자 등)를 차량 앞 정면에 놓는다(먼저 "가운데"부터 — 좌우 오차가 0이어야
#      정상이라 sanity check가 된다). 카메라 창이 뜨면 물체가 바닥에 닿는 지점을 클릭.
#   4) rclpy 자동모드면 몇 초 안에 자동으로 라이다값을 잡는다. 수동모드면
#      `ros2 topic echo /scan --once`로 물체 방향의 range(m)/그 인덱스를 확인해 입력.
#   5) 물체를 좌/우로 옮겨가며 --samples 만큼(기본 3회) 반복.
#   6) 마지막에 평균 오프셋과 config.py에 붙여넣을 코드 블록을 출력한다.
#=============================================
import argparse
import math
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
LIDAR_ANGLE_OFFSET_DEG = 80.0


def build_camera_bev():
    """dl_lane.py가 쓰는 것과 동일한 1차 호모그래피(M0) — perc_lane()이 실제로 쓰는
    캔버스 자동확장/원거리크롭 로직은 시각화 용도라 여기선 필요 없다(좌표 변환 자체는
    이 M0 하나로 충분, dl_lane.py의 _dl_M0과 동일 정의)."""
    src = DL_BEV_SRC_PX_RAW - np.float32([0, DL_ROI_Y0])
    block_w = 0.8 * DL_PIXELS_PER_METER   # 실측 폭 0.8m (README §6.3)
    block_h = 1.0 * DL_PIXELS_PER_METER   # 실측 길이 1.0m (근거리~1m 지점, README §6.3)
    dst = np.float32([[0, 0], [block_w, 0], [block_w, block_h], [0, block_h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return M, block_w, block_h


def raw_px_to_cam_xy(px, py, M, block_w, block_h):
    """원본 프레임(640x480) 절대 픽셀 좌표 -> 카메라 BEV 기준 (forward_m, lateral_m).
    forward_m: 근거리 기준선(BL/BR, 대략 차량 앞범퍼 부근)에서 얼마나 먼지(+ = 전방).
    lateral_m: 캘리브레이션 폭(0.8m)의 중앙 기준 좌우 위치(+ = TR/BR 쪽, 원본 이미지 우측)."""
    pt = np.float32([[[px, py - DL_ROI_Y0]]])
    xb, yb = cv2.perspectiveTransform(pt, M)[0, 0]
    forward_m = (block_h - yb) / DL_PIXELS_PER_METER
    lateral_m = (xb - block_w / 2.0) / DL_PIXELS_PER_METER
    return float(forward_m), float(lateral_m)


def _index_to_deg(index, n):
    """라이다 인덱스 -> 정면(0°) 기준 각도(도).

    [2026-08-14, 실차 확인 후 수정] `LIDAR_ANGLE_OFFSET_DEG`는 이름과 달리 실제로는
    "도(degree)"가 아니라 **정면일 때의 라이다 인덱스**로 실측된 값이다(README §6.2 —
    사람을 정면에 세우고 인덱스 컴퍼스로 확인했더니 인덱스 80에서 찍혔다는 절차 그대로).
    이 도구가 처음 썼던 `index*(360/n) - LIDAR_ANGLE_OFFSET_DEG` 공식은 "인덱스 한 칸 =
    1도"라는 가정이 깔려 있는데, 실차 캡처(`index=122/500`, 2026-08-14)로 이 라이다가
    한 바퀴에 **n=500** 포인트를 찍는다는 게 확인됐다 — 인덱스 한 칸은 실제로 360/500=0.72도.
    옛 공식은 정면(index=80) 근처에서는 우연히 거의 맞지만, 정면에서 멀어질수록(index가
    80에서 멀어질수록) 오차가 커진다 — 오프셋 측정 3샘플의 표준편차가 비정상적으로 컸던
    (dx 표준편차 0.7m) 원인으로 추정된다.

    수정한 공식: `80`을 각도가 아니라 "정면 인덱스"로 다뤄서
    `deg = (index - 정면인덱스) * (360/n)`으로 계산한다 — index=80일 때만 옛 공식과 결과가
    같고(정확히 그 지점에서 캘리브레이션됐으므로), 그 외에는 달라진다.

    ★주의★ `track_drive.py`의 `perc_obstacle()`/`perc_lavacon_trigger()`는 아직
    `m=min(n,360)` + 옛 공식을 그대로 쓰고 있다 — 이 도구만 먼저 고쳤고, 프로덕션 코드
    쪽 수정은 아직 안 함(README §2.30에 메모, 다음 단계로 남겨둠)."""
    return (index - LIDAR_ANGLE_OFFSET_DEG) * (360.0 / n)


def lidar_index_to_xy(range_m, index, n):
    """라이다 (range, index) -> (x_m, y_m). n = 그 스캔의 실제 len(ranges) — 절대
    360으로 가정하지 말 것(_index_to_deg() 주석 참고, 이 차량은 500으로 확인됨)."""
    deg = _index_to_deg(index, n)
    rad = math.radians(deg)
    x = range_m * math.cos(rad)   # 전방(+)
    y = range_m * math.sin(rad)   # 좌측(+) — perc_obstacle()과 동일 부호 관례
    return x, y


def _to_640x480(frame):
    """[2026-08-14, 실차 확인 후 2차 수정] 단순 cv2.resize(640,480)만 하면 원본 종횡비가
    4:3이 아닐 때(예: 흔한 16:9 1280x720) 가로/세로가 서로 다른 비율로 눌려서(짓눌림) 실제
    트랙 폭이 왜곡된다 — DL_BEV_SRC_PX_RAW 사다리꼴(640x480 기준 고정 좌표)이 눌린 실제
    차선보다 상대적으로 넓게 보이는 증상으로 나타남(실차에서 확인). 리사이즈 전에 먼저
    가운데를 4:3으로 잘라내(center crop) 종횡비를 맞춘 뒤 리사이즈하면 왜곡 없이 스케일만
    바뀐다 — 다만 이건 "usb_cam_node_exe가 실제로 잡는 640x480 모드와 광학적으로 완전히
    동일한 크롭"이라는 보장은 없는 근사치다(그 모드가 센서를 어떻게 크롭/스케일하는지 정확히
    모름) — 최종 확인은 실제 라이다-카메라 오프셋 값이 안정적으로 나오는지로 할 것."""
    h, w = frame.shape[:2]
    if (w, h) == (640, 480):
        return frame
    target_ar = 640.0 / 480.0
    src_ar = w / float(h)
    if src_ar > target_ar:            # 원본이 더 넓다(16:9 등) -> 좌우를 잘라 4:3으로
        new_w = int(round(h * target_ar))
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    elif src_ar < target_ar:          # 원본이 더 좁다(세로가 김) -> 위아래를 잘라 4:3으로
        new_h = int(round(w / target_ar))
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h, :]
    return cv2.resize(frame, (640, 480))


def capture_camera_point(camera_index):
    """cv2.VideoCapture로 카메라를 직접 열어(ROS2 미경유), 클릭한 지점을
    (forward_m, lateral_m)으로 반환. q를 누르면 None."""
    M, bw, bh = build_camera_bev()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'카메라(index={camera_index})를 열 수 없습니다 — 다른 프로세스가 '
                            f'점유 중이거나(usb_cam_node_exe가 떠 있으면 먼저 끌 것) 인덱스가 '
                            f'다를 수 있습니다(--camera-index로 다른 값 시도).')

    # [버그 수정] DL_BEV_SRC_PX_RAW/DL_ROI_Y0/Y1은 전부 "640x480 프레임" 기준 절대 픽셀
    # 좌표다(usb_cam_node_exe가 그 해상도로 발행하도록 맞춰져 있어서 track_drive.py가 받는
    # /usb_cam/image_raw/front는 항상 640x480). 근데 cv2.VideoCapture()는 해상도를 지정 안
    # 하면 드라이버 기본값(웹캠마다 다름, 보통 640x480보다 훨씬 큼 — 1280x720 등)으로 열려서,
    # 그 상태로 250~390행에 ROI 박스를 그리면 "화면 상단 근처"만 가리키게 된다 — 카메라가
    # 도로를 보도록 아래로 기울어 장착돼 있으면 화면 상단은 천장/벽 쪽이라, 딱 이 증상(ROI가
    # 천장을 가리킴)으로 나타난다. CAP_PROP으로 요청은 해보되(드라이버가 안 들어줄 수 있음),
    # 실제 읽힌 프레임을 강제로 640x480 리사이즈해서 좌표계를 항상 맞춘다.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    _warned_resize = [False]

    clicked = {}

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked['pt'] = (x, y)

    win = 'camera - click the object base point (q=cancel)'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    print('  카메라 창에서 물체가 바닥에 닿는 지점을 클릭하세요.')
    print('  -> 클릭하면 forward/lateral 값이 미리보기로 뜹니다(다시 클릭하면 위치 갱신 가능).')
    print('  -> 그 상태에서 카메라 창을 클릭해 포커스를 준 뒤 Enter 또는 Space를 눌러야 확정됩니다')
    print('     (그냥 클릭만 하면 안 넘어갑니다 — 잘못 찍었을 때 다시 클릭해서 고칠 수 있게')
    print('     확정을 한 단계 더 둔 것입니다). q=취소')

    result = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03)
                continue
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                if not _warned_resize[0]:
                    print(f'  [주의] 카메라가 {frame.shape[1]}x{frame.shape[0]}로 열렸습니다 '
                          f'(기대값 640x480) — cv2.CAP_PROP 요청을 드라이버가 무시한 것으로 '
                          f'보입니다. 가운데를 4:3으로 crop한 뒤 640x480으로 리사이즈해서 '
                          f'표시합니다(종횡비 왜곡 방지) — 그래도 실제 usb_cam 640x480 모드와 '
                          f'완전히 같다는 보장은 없으니, 화면에서 보라색 사다리꼴이 실제 차선폭과')
                    print('  비슷하게 보이는지 육안으로 한 번 더 확인하세요.')
                    _warned_resize[0] = True
                frame = _to_640x480(frame)
            vis = frame.copy()
            cv2.rectangle(vis, (0, DL_ROI_Y0), (vis.shape[1] - 1, DL_ROI_Y1), (0, 255, 255), 1)
            pts = DL_BEV_SRC_PX_RAW.astype(np.int32)
            cv2.polylines(vis, [pts], True, (255, 0, 255), 1)
            cv2.putText(vis, 'click=pick point   ENTER/SPACE=confirm   q=cancel',
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            if 'pt' in clicked:
                cv2.circle(vis, clicked['pt'], 6, (0, 0, 255), -1)
                fwd, lat = raw_px_to_cam_xy(clicked['pt'][0], clicked['pt'][1], M, bw, bh)
                cv2.putText(vis, f'forward={fwd:.3f}m lateral={lat:.3f}m  [press ENTER/SPACE to confirm]',
                            (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0), 2, cv2.LINE_AA)
                result = (fwd, lat)
            cv2.imshow(win, vis)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'):
                result = None
                break
            # Enter는 백엔드/OS별로 코드가 13(CR)/10(LF)로 다르게 들어올 수 있어 둘 다 받는다.
            if k in (13, 10, 32) and result is not None:  # Enter/Space로 확정
                break
    finally:
        cap.release()
        cv2.destroyWindow(win)
    return result


def capture_lidar_point_auto(scan_topic, front_window_deg, timeout_s):
    """rclpy로 /scan을 짧게 구독해서, 정면 부채꼴(LIDAR_ANGLE_OFFSET_DEG 중심
    ±front_window_deg) 안에서 가장 가까운 유효 포인트를 (x_m, y_m)으로 반환.
    rclpy가 없거나 timeout_s 안에 메시지를 못 받으면 None(수동 입력으로 폴백)."""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan
    except ImportError:
        print('  (rclpy 없음 — 수동 입력으로 진행합니다)')
        return None

    result = {}

    class _Probe(Node):
        def __init__(self):
            super().__init__('measure_lidar_camera_offset_probe')
            # [2026-08-14, 실차 확인 후 수정] xycar_lidar_node가 rclcpp::SensorDataQoS()
            # (BEST_EFFORT)로 퍼블리시하는데, create_subscription에 정수 depth만 주면
            # 기본 QoS(RELIABLE)로 구독돼 "New publisher discovered ... incompatible QoS"
            # 경고와 함께 메시지를 전혀 못 받는다(자동모드가 항상 타임아웃 -> 수동 폴백으로만
            # 동작하던 원인). 발행 쪽과 동일한 qos_profile_sensor_data로 맞춘다.
            self.sub = self.create_subscription(
                LaserScan, scan_topic, self.cb, qos_profile_sensor_data)

        def cb(self, msg):
            n = len(msg.ranges)
            if n == 0:
                return
            best = None
            for i, r in enumerate(msg.ranges):
                if not math.isfinite(r) or r <= 0.0:
                    continue
                deg = _index_to_deg(i, n)
                deg = (deg + 180.0) % 360.0 - 180.0  # -180~180으로 정규화
                if abs(deg) <= front_window_deg:
                    if best is None or r < best[0]:
                        best = (r, i, n)
            if best is not None:
                result['hit'] = best

    rclpy.init(args=[])
    node = _Probe()
    t0 = time.time()
    try:
        while time.time() - t0 < timeout_s and 'hit' not in result:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if 'hit' not in result:
        print(f'  ({timeout_s:.0f}초 안에 {scan_topic}에서 정면 유효 포인트를 못 받음 — 수동 입력으로 진행합니다)')
        return None

    r, idx, n = result['hit']
    x, y = lidar_index_to_xy(r, idx, n)
    print(f'  [자동] 라이다 정면 최근접점: range={r:.3f}m index={idx}/{n} -> x={x:.3f}m y={y:.3f}m')
    return (x, y)


def _flush_stdin():
    """[2026-08-14, 실차 확인 후 수정] 터미널에 포커스가 있는 동안(카메라 창 Enter 확정
    직후 ~ 라이다 자동탐지 3초 대기 사이) 무심코 누른 키(특히 Enter)가 stdin 버퍼에 그대로
    쌓여 있으면, 바로 다음에 뜨는 input() 프롬프트가 타이핑할 틈도 없이 그 버퍼링된 빈
    줄을 즉시 삼켜버려 `float('')` -> ValueError -> '입력 취소'로 이어지는 문제가 있었다
    (증상: range 프롬프트가 뜨자마자 바로 취소됨). 수동 입력 시작 직전에 항상 버퍼를 비운다
    — stdin이 tty가 아니면(리다이렉션 등) 조용히 무시."""
    try:
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


def capture_lidar_point_manual():
    _flush_stdin()
    print('  수동 입력 모드 — 다른 터미널에서 `ros2 topic echo /scan --once` 로 확인하세요.')
    print('  (물체 방향의 range값과 그 인덱스를 찾으면 됩니다 — 대략 정면이면 인덱스가')
    print(f'   {LIDAR_ANGLE_OFFSET_DEG:.0f} 근처)')
    print('  ★주의★ ranges 배열 길이(n)를 360이라고 가정하지 마세요 — 이 라이다는 실측')
    print('  결과 n=500으로 확인됐습니다(2026-08-14). echo 출력에서 ranges 배열 길이를')
    print('  직접 세어 입력하세요(예: 500).')
    try:
        r = float(input('  range (m): ').strip())
        idx = int(input('  index: ').strip())
        n = input('  ranges 배열 길이 (엔터=이전 실측값 500 사용): ').strip()
        n = int(n) if n else 500
    except (ValueError, EOFError):
        print('  입력 취소')
        return None
    return lidar_index_to_xy(r, idx, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--camera-index', type=int, default=0)
    ap.add_argument('--scan-topic', default='/scan')
    ap.add_argument('--samples', type=int, default=3, help='다른 위치에서 반복 측정할 횟수')
    ap.add_argument('--front-window-deg', type=float, default=20.0,
                     help='자동 라이다 모드에서 "정면"으로 볼 각도 범위(±deg)')
    ap.add_argument('--lidar-timeout', type=float, default=3.0)
    ap.add_argument('--manual-lidar', action='store_true', help='rclpy 자동측정을 쓰지 않고 항상 수동 입력')
    args = ap.parse_args()

    print(f'=== 라이다-카메라 오프셋 실측 ({args.samples}회 반복) ===')
    print('물체를 차량 앞에 두고(첫 샘플은 정면 권장), 매 회차 다른 좌우 위치로 옮겨가며 진행하세요.\n')

    pairs = []  # (cam_fwd, cam_lat, lidar_x, lidar_y)
    for i in range(args.samples):
        print(f'--- 샘플 {i+1}/{args.samples} ---')
        try:
            cam = capture_camera_point(args.camera_index)
        except RuntimeError as e:
            print(f'  카메라 오류: {e}')
            break
        if cam is None:
            print('  취소됨 — 이 샘플 건너뜀')
            continue

        lidar = None
        if not args.manual_lidar:
            lidar = capture_lidar_point_auto(args.scan_topic, args.front_window_deg, args.lidar_timeout)
        if lidar is None:
            lidar = capture_lidar_point_manual()
        if lidar is None:
            print('  라이다 값 없음 — 이 샘플 건너뜀')
            continue

        cam_fwd, cam_lat = cam
        lidar_x, lidar_y = lidar
        dx, dy = lidar_x - cam_fwd, lidar_y - cam_lat
        print(f'  카메라(forward={cam_fwd:.3f}m, lateral={cam_lat:.3f}m)  '
              f'라이다(x={lidar_x:.3f}m, y={lidar_y:.3f}m)  Δ=({dx:.3f}, {dy:.3f})\n')
        pairs.append((cam_fwd, cam_lat, lidar_x, lidar_y))

    if not pairs:
        print('측정된 샘플이 없습니다. 종료합니다.')
        return 1

    dxs = [lx - cf for cf, cl, lx, ly in pairs]
    dys = [ly - cl for cf, cl, lx, ly in pairs]
    dx_mean, dy_mean = float(np.mean(dxs)), float(np.mean(dys))
    dx_std, dy_std = float(np.std(dxs)), float(np.std(dys))

    print('=== 결과 ===')
    print(f'샘플 수: {len(pairs)}')
    print(f'LIDAR_TO_CAM_DX_M = {dx_mean:.3f}  (표준편차 {dx_std:.3f})')
    print(f'LIDAR_TO_CAM_DY_M = {dy_mean:.3f}  (표준편차 {dy_std:.3f})')
    if dx_std > 0.05 or dy_std > 0.05:
        print('★주의★ 표준편차가 5cm 넘게 큽니다 — 오프셋이 아니라 클릭 오차나')
        print('        DL_PIXELS_PER_METER 캘리브레이션 자체를 의심해볼 필요가 있습니다.')

    print('\nconfig.py에 붙여넣을 코드:')
    print('# [실측 라이다-카메라 BEV 오프셋 — 날짜/측정자 채울 것]')
    print(f'LIDAR_TO_CAM_DX_M = {dx_mean:.3f}   # 라이다 원점 -> 카메라 BEV 근거리기준선, 전방(+) 오프셋(m)')
    print(f'LIDAR_TO_CAM_DY_M = {dy_mean:.3f}   # 라이다 원점 -> 카메라 BEV 근거리기준선, 좌(+) 오프셋(m)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
