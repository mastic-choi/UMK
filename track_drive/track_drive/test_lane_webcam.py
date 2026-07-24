#!/usr/bin/env python3
"""Standalone webcam smoke-test for PolyLaneNetDetector -- no ROS required.

Run this on a dev machine (before porting to the xycar+ROS node) to visually
confirm a checkpoint (e.g. tusimple_resnet34/model_2695.pt) tracks lanes
reasonably: reads frames from a local webcam and feeds them straight into
lane_util.PolyLaneNetDetector.detect(), which draws the decoded polynomials
itself via its DEBUG_VIZ_LANE path (window "polylanenet_result").

Usage:
    python3 test_lane_webcam.py --weights /path/to/model_2695.pt
"""
import argparse

import cv2

from lane_util import PolyLaneNetDetector, DEFAULT_WEIGHTS_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='cv2.VideoCapture index')
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS_PATH, help='PolyLaneNet checkpoint (.pt/.pth)')
    args = parser.parse_args()

    if args.weights is None:
        raise SystemExit(
            'No checkpoint found. Place model_2695.pt at '
            'track_drive/track_drive/weights/polylanenet.pth, or pass --weights <path>.'
        )

    detector = PolyLaneNetDetector(weights_path=args.weights)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f'Could not open camera index {args.camera}')

    print('Press q (with the video window focused) to quit.')
    while True:
        ok, frame = cap.read()
        if not ok:
            print('Failed to read frame from camera.')
            break

        valid, offset, lookahead, lane_center, _ = detector.detect(frame)
        print(f'valid={valid} offset={offset:+7.1f}px lookahead={lookahead:+7.1f}px center={lane_center:7.1f}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
