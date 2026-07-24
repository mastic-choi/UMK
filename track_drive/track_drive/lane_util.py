"""
PolyLaneNet-based lane perception utility.

Replaces the previous OpenCV pipeline (BEV homography, adaptive-threshold
color masking, moments-based slice tracking, outlier rejection) with the
deep polynomial regression approach from:

    Tabelini et al., "PolyLaneNet: Lane Estimation via Deep Polynomial
    Regression" (arXiv:2004.10924).

Core idea (paper Eq. 1): a CNN backbone + single FC head directly regresses,
in one forward pass over the raw camera frame, M_MAX lane-marking candidates,
each a 3rd-order polynomial

        p(y) = a*y^3 + b*y^2 + c*y + d                              (Eq. 1, K=3)

together with a confidence score c_j and a vertical validity range for that
polynomial. The paper's Eq. 2 additionally lists a single global horizon `h`
shared across all lanes -- but the officially released weights (which this
module is built to load directly, e.g. `tusimple_resnet34/model_2695.pt`)
never actually expose that as its own output slot: `models.py`'s `decode()`
only sigmoids the confidence column, and `share_top_y` -- the mechanism meant
to make one lane's "lower" bound track the shared horizon -- is dead-code
even in the reference repo. What every release actually predicts, and what
`lane_dataset.py`'s own visualization code consumes, is a *per-candidate*
`(lower, upper)` pair used directly as that polynomial's valid y-domain. This
module follows that real, checkpoint-compatible layout:

    per lane candidate: (c_j, lower_j, upper_j, a_j, b_j, c_j, d_j)  -- 7 values
    FC head output     : M_MAX * 7 values, no separate global output

There is no segmentation, clustering, anchor shifting, or curve-fitting here:
the lanes returned by this module ARE the network's raw polynomial outputs,
only gated by the confidence threshold from the paper (Sec. IV-B).

`x`/`y` polynomial coordinates are normalized to [0, 1] (fraction of frame
width/height), exactly as they were during training -- pixel coordinates are
only reconstructed at the very end, for control/visualization.
"""

import os
from dataclasses import dataclass
from typing import List

import cv2  # only used for input resize/color-convert and optional debug drawing
import numpy as np
import torch
import torch.nn as nn
from torchvision import models as tv_models

# ----------------------------------------------------------------------------
# PolyLaneNet architecture constants
# ----------------------------------------------------------------------------
POLY_ORDER = 3                               # K in Eq. 1 -- paper's chosen (default) polynomial degree
NUM_COEFFS = POLY_ORDER + 1                  # (a, b, c, d) per lane candidate
M_MAX = 5                                    # max simultaneous lane candidates the FC head predicts
                                              #   (paper: real scenes have M<=4 annotated lanes; the official
                                              #   tusimple_resnet34 checkpoint uses 5 slots -- kept identical
                                              #   here so those released weights load without reshaping)
OUTPUTS_PER_LANE = 3 + NUM_COEFFS            # c_j, lower, upper, + 4 poly coeffs = 7 (matches official repo)
TOTAL_OUTPUTS = M_MAX * OUTPUTS_PER_LANE     # no extra global-h slot -- see module docstring

CONF_THRESHOLD = 0.5                         # c_j < 0.5 candidates are discarded at inference (paper Sec. IV-B)

NET_INPUT_W, NET_INPUT_H = 640, 360          # network input resolution (paper's TuSimple training setup)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LOOKAHEAD_Y_NORM = 0.6                       # normalized y sampled for the "far" (lookahead) offset

# Off by default: the vehicle's Jetson normally runs this node headless (no attached
# display / no X session over SSH), and cv2.imshow() there either errors or hangs.
# Set XYCAR_LANE_DEBUG_VIZ=1 in the environment to see the "polylanenet_result" window
# during on-desk/dev-machine testing (e.g. test_lane_webcam.py).
DEBUG_VIZ_LANE = os.environ.get('XYCAR_LANE_DEBUG_VIZ', '0') == '1'


def _imshow(window_name, image):
    """cv2.imshow guarded against a missing display -- logs once and disables
    further attempts instead of taking down the control loop mid-drive."""
    global DEBUG_VIZ_LANE
    if not DEBUG_VIZ_LANE or image is None:
        return
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
    except cv2.error as exc:
        print(f'[lane_util] disabling debug visualization (no display available?): {exc}')
        DEBUG_VIZ_LANE = False


def _resolve_weights_dir():
    """Prefer the ROS2-installed share directory (colcon build copies `weights/`
    there via setup.py's data_files); fall back to the path next to this source
    file for dev runs (symlink-install, or a plain `python3 test_lane_webcam.py`
    with no ROS2 environment sourced at all)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share_weights = os.path.join(get_package_share_directory('track_drive'), 'weights')
        if os.path.isdir(share_weights):
            return share_weights
    except Exception:
        pass  # not running under a sourced ROS2 install (e.g. plain dev-machine test script)
    return os.path.join(os.path.dirname(__file__), 'weights')


# Official released checkpoint (e.g. tusimple_resnet34/model_2695.pt) expected as
# weights/polylanenet.{pt,pth} -- either extension is accepted since torch.save()
# doesn't care and the official releases ship as .pt. Falls back to None (random-init
# backbone -- inference runs but produces meaningless lanes) until a checkpoint is
# placed there or a path is passed explicitly to PolyLaneNetDetector.
_WEIGHTS_DIR = _resolve_weights_dir()
DEFAULT_WEIGHTS_PATH = next(
    (p for ext in ('.pt', '.pth') if os.path.isfile(p := os.path.join(_WEIGHTS_DIR, f'polylanenet{ext}'))),
    None,
)


@dataclass
class LaneCandidate:
    """One decoded lane-marking candidate: a 3rd-order polynomial p(y) = a*y^3 + b*y^2 + c*y + d,
    valid on the normalized y-range [lower, upper] (lower is nearer the horizon/top of frame,
    upper is nearer the camera/bottom of frame -- matches the official repo's own convention).

    `coeffs` is ordered (a, b, c, d) -- highest degree first -- matching the
    reference implementation's use of `np.polyval`, not the paper's
    increasing-order a_k indexing (they are the same 4 numbers, just reversed).
    """
    coeffs: np.ndarray   # shape (4,): (a, b, c, d)
    lower: float         # normalized y where this lane's predicted extent begins (near horizon)
    upper: float         # normalized y where this lane's predicted extent ends (near camera / bottom)
    confidence: float    # c_j in [0, 1], already passed the CONF_THRESHOLD gate in decode_predictions()

    def evaluate_x(self, y_norm):
        """p(y) via Horner's method. `y_norm` may be a scalar or an array, both normalized to [0, 1]."""
        a, b, c, d = self.coeffs
        y = np.asarray(y_norm, dtype=np.float32)
        return ((a * y + b) * y + c) * y + d

    def to_pixel_points(self, img_w, img_h, y_lo=None, y_hi=None, num_points=50):
        """Sample the polynomial on [y_lo, y_hi] (normalized, defaults to this lane's own [lower, upper])
        and project to pixel space."""
        y_lo = self.lower if y_lo is None else y_lo
        y_hi = self.upper if y_hi is None else y_hi
        ys = np.linspace(y_lo, y_hi, num_points, dtype=np.float32)
        xs = self.evaluate_x(ys)
        pts = np.stack([xs * img_w, ys * img_h], axis=1).astype(np.int32)
        # drop points the polynomial extrapolated outside the visible frame
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < img_w)]
        return pts


def _build_backbone(backbone, pretrained):
    """torchvision's pretrained-weights API changed across versions; support both."""
    if backbone == 'resnet34':
        ctor, weights_enum = tv_models.resnet34, getattr(tv_models, 'ResNet34_Weights', None)
    elif backbone == 'resnet50':
        ctor, weights_enum = tv_models.resnet50, getattr(tv_models, 'ResNet50_Weights', None)
    else:
        raise NotImplementedError(f'Unsupported backbone: {backbone}')

    if weights_enum is not None:
        return ctor(weights=weights_enum.IMAGENET1K_V1 if pretrained else None)
    return ctor(pretrained=pretrained)


class _OutputLayer(nn.Module):
    """Verbatim structure of the official repo's `OutputLayer` (lib/models.py).

    Kept name-for-name (`regular_outputs_layer` / `extra_outputs_layer`) so a
    released checkpoint's state_dict -- which stores exactly these submodule
    names -- loads via plain `load_state_dict(strict=True)`, no key remapping.
    """

    def __init__(self, fc, num_extra=0):
        super().__init__()
        self.regular_outputs_layer = fc
        self.num_extra = num_extra
        if num_extra > 0:
            self.extra_outputs_layer = nn.Linear(fc.in_features, num_extra)

    def forward(self, x):
        regular_outputs = self.regular_outputs_layer(x)
        extra_outputs = self.extra_outputs_layer(x) if self.num_extra > 0 else None
        return regular_outputs, extra_outputs


class PolyLaneNet(nn.Module):
    """Backbone + FC head producing the raw TOTAL_OUTPUTS regression vector.

    The resnet instance is stored as `self.model` (not e.g. `self.backbone`)
    and its head wrapped in `_OutputLayer`, both purely so this module's
    state_dict layout matches the officially released checkpoints exactly.

    No decoding (sigmoid/thresholding) happens here -- that is decode_predictions()'s
    job, kept separate so the network stays a plain regressor.
    """

    def __init__(self, backbone='resnet34', pretrained=True, m_max=M_MAX, extra_outputs=0):
        super().__init__()
        self.m_max = m_max
        total_outputs = m_max * OUTPUTS_PER_LANE

        self.model = _build_backbone(backbone, pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, total_outputs)
        self.model.fc = _OutputLayer(self.model.fc, extra_outputs)

    def forward(self, x):
        regular_outputs, _ = self.model(x)
        return regular_outputs  # raw logits; caller must run decode_predictions()


def decode_predictions(raw_output, m_max=M_MAX, conf_threshold=CONF_THRESHOLD):
    """Parse one FC-layer output vector into confidence-filtered lane candidates.

    Implements the "Network Outputs Extraction" + "Confidence Filtering" steps:
    reshapes the flat vector into M_MAX blocks of (c_j, lower, upper, a, b, c, d),
    sigmoids only the confidence column (matching the official `decode()` in
    lib/models.py -- lower/upper/coeffs are left as raw regression outputs,
    never squashed at inference), and strictly discards any candidate with
    c_j < conf_threshold.

    Args:
        raw_output: 1-D tensor of length `m_max * OUTPUTS_PER_LANE`
                    (i.e. a single image's network output, batch dim already removed).
    Returns:
        A confidence-filtered list of LaneCandidate (never containing
        c_j < conf_threshold entries).
    """
    expected = m_max * OUTPUTS_PER_LANE
    if raw_output.shape[-1] != expected:
        raise ValueError(f'expected {expected} outputs for m_max={m_max}, got {raw_output.shape[-1]}')

    lane_block = raw_output.view(m_max, OUTPUTS_PER_LANE)
    confidences = torch.sigmoid(lane_block[:, 0])
    lowers = lane_block[:, 1]
    uppers = lane_block[:, 2]
    coeffs = lane_block[:, 3:]

    candidates = []
    for j in range(m_max):
        c_j = confidences[j].item()
        if c_j < conf_threshold:
            continue  # strict filtering: below-threshold candidates are dropped entirely, not zeroed
        candidates.append(LaneCandidate(
            coeffs=coeffs[j].detach().cpu().numpy().astype(np.float32),
            lower=float(lowers[j].item()),
            upper=float(uppers[j].item()),
            confidence=c_j,
        ))
    return candidates


class PolyLaneNetDetector:
    """PolyLaneNet-only replacement for the previous CameraProcessor + SlideWindow pipeline.

    Consumes the raw front-camera BGR frame directly -- no ROI crop, no BEV
    warp, no color thresholding, no moments/outlier-rejection clustering. The
    lanes are read straight off the network's polynomial outputs.
    """

    def __init__(self, weights_path=None, backbone='resnet34', m_max=M_MAX,
                 conf_threshold=CONF_THRESHOLD, device=None):
        self.m_max = m_max
        self.conf_threshold = conf_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = PolyLaneNet(backbone=backbone, pretrained=weights_path is None, m_max=m_max)
        if weights_path is not None:
            checkpoint = torch.load(weights_path, map_location=self.device)
            # official train.py saves {'model': state_dict, 'optimizer': ..., 'epoch': ...};
            # also accept a bare state_dict for checkpoints saved some other way.
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def _preprocess(self, frame_bgr):
        """Resize + ImageNet-normalize. This is the *only* preprocessing PolyLaneNet needs."""
        resized = cv2.resize(frame_bgr, (NET_INPUT_W, NET_INPUT_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return (tensor - self._mean) / self._std

    @torch.inference_mode()
    def infer(self, frame_bgr) -> List[LaneCandidate]:
        """Run the network on one frame and return decoded, confidence-filtered candidates."""
        x = self._preprocess(frame_bgr)
        raw_output = self.model(x)[0]  # drop batch dim -> (TOTAL_OUTPUTS,)
        return decode_predictions(raw_output, self.m_max, self.conf_threshold)

    def detect(self, frame_bgr):
        """
        Drop-in entry point for the perception pipeline (mirrors the previous
        CameraProcessor.processor() + SlideWindow.detect() contract).

        Returns: (lane_valid, lane_offset, lane_lookahead, lane_center, vis)
        """
        if frame_bgr is None:
            return False, 0.0, 0.0, NET_INPUT_W / 2.0, None

        img_h, img_w = frame_bgr.shape[:2]
        candidates = self.infer(frame_bgr)
        vis = frame_bgr.copy() if DEBUG_VIZ_LANE else None

        center_x_norm = 0.5
        near_y, far_y = 1.0, LOOKAHEAD_Y_NORM

        # Ego-lane = the candidates immediately left/right of bottom-center,
        # following the paper's ego-lane definition (Sec. IV-D): "the lane
        # markings that are closer to the middle of the bottom part of the
        # image", one to the left and one to the right.
        left = right = None
        best_left_x, best_right_x = -np.inf, np.inf
        for cand in candidates:
            x_near = float(cand.evaluate_x(near_y))
            if x_near < center_x_norm and x_near > best_left_x:
                left, best_left_x = cand, x_near
            elif x_near >= center_x_norm and x_near < best_right_x:
                right, best_right_x = cand, x_near

        def offset_px(y_norm):
            xs = [c.evaluate_x(y_norm) for c in (left, right) if c is not None]
            return (float(np.mean(xs)) - center_x_norm) * img_w if xs else None

        near_offset = offset_px(near_y)

        if near_offset is None:
            _imshow('polylanenet_result', vis)
            return False, 0.0, 0.0, img_w / 2.0, vis

        far_offset = offset_px(far_y)
        if far_offset is None:
            far_offset = near_offset
        lane_center = img_w / 2.0 + near_offset

        if DEBUG_VIZ_LANE and vis is not None:
            self._draw_debug(vis, candidates, left, right, img_w, img_h)
        _imshow('polylanenet_result', vis)

        return True, near_offset, far_offset, lane_center, vis

    @staticmethod
    def _draw_debug(vis, candidates, left, right, img_w, img_h):
        """Render each decoded polynomial directly -- pure visualization of network output, no re-fitting."""
        for cand in candidates:
            color = (0, 255, 255)
            if cand is left:
                color = (0, 255, 0)
            elif cand is right:
                color = (0, 0, 255)
            pts = cand.to_pixel_points(img_w, img_h)  # defaults to this lane's own [lower, upper]
            for p1, p2 in zip(pts[:-1], pts[1:]):
                cv2.line(vis, tuple(p1), tuple(p2), color, 2)
            if len(pts):
                cv2.putText(vis, f'{cand.confidence:.2f}', tuple(pts[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.line(vis, (img_w // 2, 0), (img_w // 2, img_h), (255, 255, 255), 1)
