"""Synthetic dye-on-towel scenes with exactly known geometry."""

from __future__ import annotations

import cv2
import numpy as np

W, H = 1280, 960
TOWEL_X = (500, 700)  # columns [500, 700)
TOWEL_TOP = 150
BASE_ROW = 850  # dye rows are [BASE_ROW - h, BASE_ROW); the reservoir is below
MARKER_PX = 120
MM_PER_PX = 0.25  # markers are "30 mm"
MARKER_MM = MARKER_PX * MM_PER_PX

BLUE = (190, 90, 30)  # BGR
TOWEL = (240, 240, 240)
BACKGROUND = (200, 205, 210)

# Continuous coordinates: pixel (r, c) covers [c-0.5, c+0.5] x [r-0.5, r+0.5].
BASE_POINT = (600.0, BASE_ROW - 0.5)
TIP_POINT = (600.0, TOWEL_TOP - 0.5)
ROI = [(490.0, 140.0), (710.0, 140.0), (710.0, 860.0), (490.0, 860.0)]


def _marker(dictionary: int, marker_id: int) -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(dictionary)
    return cv2.cvtColor(cv2.aruco.generateImageMarker(d, marker_id, MARKER_PX), cv2.COLOR_GRAY2BGR)


def scene(
    dye_height_px: int,
    markers: bool = True,
    texture: bool = False,
    front_spike: int = 0,
    seed: int = 0,
) -> np.ndarray:
    img = np.full((H, W, 3), BACKGROUND, np.uint8)
    if texture:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 25, (H // 8, W // 8)).astype(np.float32)
        noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC)
        img = np.clip(img.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
    x0, x1 = TOWEL_X
    img[TOWEL_TOP:900, x0:x1] = TOWEL
    img[BASE_ROW:900, x0:x1] = BLUE  # reservoir, below the base
    if dye_height_px > 0:
        img[BASE_ROW - dye_height_px:BASE_ROW, x0:x1] = BLUE
    if front_spike:
        img[BASE_ROW - dye_height_px - front_spike:BASE_ROW, 590:600] = BLUE
    if markers:
        for marker_id, (x, y) in {0: (120, 120), 1: (980, 680)}.items():
            q = 30
            img[y - q:y + MARKER_PX + q, x - q:x + MARKER_PX + q] = 255
            img[y:y + MARKER_PX, x:x + MARKER_PX] = _marker(cv2.aruco.DICT_4X4_50, marker_id)
    return img


def bump(img: np.ndarray, dx: float, dy: float, angle_deg: float) -> np.ndarray:
    """Simulate a knocked camera: rotate about the center and shift."""
    m = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, 1.0)
    m[:, 2] += (dx, dy)
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
