"""Synthetic field seen by a tilted camera: two plots with irregular discoloured patches.

Ground coordinates are 1 px = 1 mm, so true areas in mm² are polygon areas in ground px.
"""

from __future__ import annotations

import cv2
import numpy as np

GROUND_W, GROUND_H = 1600, 1200
MARKER_MM = 100.0
IMAGE_W, IMAGE_H = 1280, 960

GRASS = (60, 140, 70)
PATCH = (60, 170, 205)  # yellow-brown (BGR)

PLOTS = {
    "plot1": [(150.0, 200.0), (750.0, 200.0), (750.0, 1000.0), (150.0, 1000.0)],
    "plot2": [(850.0, 200.0), (1450.0, 200.0), (1450.0, 1000.0), (850.0, 1000.0)],
}
CENTRES = {"plot1": (450.0, 600.0), "plot2": (1150.0, 600.0)}

# Ground -> image: a camera on a pole looking down at an angle.
GROUND_CORNERS = np.float32([[0, 0], [GROUND_W, 0], [GROUND_W, GROUND_H], [0, GROUND_H]])
IMAGE_CORNERS = np.float32([[210, 150], [1070, 140], [1250, 930], [30, 945]])
TILT = cv2.getPerspectiveTransform(GROUND_CORNERS, IMAGE_CORNERS)


def blob(center, radius: float, n: int = 720) -> np.ndarray:
    """Irregular closed outline; its area and shape scale with ``radius``."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = radius * (1 + 0.25 * np.sin(3 * theta) + 0.1 * np.cos(5 * theta + 1))
    return np.c_[center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)]


def true_area(radius: float) -> float:
    return float(cv2.contourArea(blob((0, 0), radius, 4000).astype(np.float32)))


def ground(radii: dict[str, float], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((GROUND_H, GROUND_W, 3), GRASS, np.float32)
    noise = cv2.resize(rng.normal(0, 18, (GROUND_H // 10, GROUND_W // 10)).astype(np.float32),
                       (GROUND_W, GROUND_H), interpolation=cv2.INTER_CUBIC)
    img = np.clip(img + noise[..., None] * np.float32([0.5, 1.0, 0.5]), 0, 255).astype(np.uint8)
    for name, radius in radii.items():
        if radius > 0:
            poly = np.round(blob(CENTRES[name], radius) * 16).astype(np.int32)
            cv2.fillPoly(img, [poly], PATCH, lineType=cv2.LINE_8, shift=4)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = int(MARKER_MM)
    for marker_id, (x, y) in enumerate([(30, 30), (GROUND_W - 130, 30),
                                        (GROUND_W - 130, GROUND_H - 130), (30, GROUND_H - 130)]):
        img[y - 20:y + side + 20, x - 20:x + side + 20] = 255
        marker = cv2.aruco.generateImageMarker(d, marker_id, side)
        img[y:y + side, x:x + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return img


def photograph(ground_img: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(ground_img, TILT, (IMAGE_W, IMAGE_H), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(90, 110, 100))
