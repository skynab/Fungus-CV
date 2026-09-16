"""Synthetic curved plant stem with moss climbing it, with exactly known arc lengths."""

from __future__ import annotations

import cv2
import numpy as np

from . import synthetic as syn

W, H = syn.W, syn.H
STEM_HALF = 12  # stem is 24 px wide
MOSS_HALF = 15  # moss sleeve is 30 px wide, covering the stem where it grows
SOIL_Y = 880

BACKGROUND = (225, 228, 230)
SOIL = (40, 55, 75)
STEM = (40, 75, 120)  # brown (BGR)
MOSS = (60, 185, 70)  # green


def bezier(p0, p1, p2, p3, n=20000) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


class Stem:
    """Centerline starting at the soil line (s = 0), heading up and curving."""

    def __init__(self, sway: float = 0.0):
        base = (640.0, float(SOIL_Y))
        self.curve = bezier(base, (640, 620), (380 + sway, 520), (520 + sway, 170))
        seg = np.linalg.norm(np.diff(self.curve, axis=0), axis=1)
        self.s = np.r_[0.0, np.cumsum(seg)]
        self.length = float(self.s[-1])
        self.base = base

    def at(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, float)
        x = np.interp(s, self.s, self.curve[:, 0])
        y = np.interp(s, self.s, self.curve[:, 1])
        tangent = np.gradient(self.curve, axis=0)
        tx = np.interp(s, self.s, tangent[:, 0])
        ty = np.interp(s, self.s, tangent[:, 1])
        norm = np.hypot(tx, ty)
        return np.c_[x, y], np.c_[-ty / norm, tx / norm]

    def sleeve(self, s0: float, s1: float, half: float) -> np.ndarray:
        """Polygon of the band ``half`` px either side of the centerline for s in [s0, s1]."""
        s = np.linspace(s0, s1, max(2, int(s1 - s0) * 2))
        pts, normals = self.at(s)
        left, right = pts + normals * half, pts - normals * half
        return np.round(np.vstack([left, right[::-1]]) * 16).astype(np.int32)  # 4-bit subpixel


def scene(stem: Stem, moss_length: float) -> np.ndarray:
    img = np.full((H, W, 3), BACKGROUND, np.uint8)
    cv2.fillPoly(img, [stem.sleeve(0, stem.length, STEM_HALF)], STEM, lineType=cv2.LINE_8,
                 shift=4)
    if moss_length > 0:
        cv2.fillPoly(img, [stem.sleeve(0, moss_length, MOSS_HALF)], MOSS, lineType=cv2.LINE_8,
                     shift=4)
    img[SOIL_Y:] = SOIL
    for marker_id, (x, y) in {0: (120, 120), 1: (980, 680)}.items():
        q = 30
        img[y - q:y + syn.MARKER_PX + q, x - q:x + syn.MARKER_PX + q] = 255
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        img[y:y + syn.MARKER_PX, x:x + syn.MARKER_PX] = cv2.cvtColor(
            cv2.aruco.generateImageMarker(d, marker_id, syn.MARKER_PX), cv2.COLOR_GRAY2BGR)
    return img


def color_mask(img: np.ndarray, bgr) -> np.ndarray:
    return np.all(img == np.asarray(bgr, np.uint8), axis=-1)


ROI = [(300.0, 100.0), (900.0, 100.0), (900.0, SOIL_Y + 20.0), (300.0, SOIL_Y + 20.0)]
