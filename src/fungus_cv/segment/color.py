"""Segmentation by HSV color thresholds, e.g. blue dye on a white paper towel."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

HsvBounds = tuple[tuple[int, int, int], tuple[int, int, int]]


def _kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


@dataclass
class ColorThresholdSegmenter:
    ranges: list[HsvBounds]
    open_px: int = 3
    close_px: int = 7
    min_blob_area_px: int = 50
    name: str = field(default="color", init=False)

    @classmethod
    def from_config(cls, cfg) -> ColorThresholdSegmenter:
        return cls(
            ranges=[(tuple(r.lower), tuple(r.upper)) for r in cfg.hsv_ranges],
            open_px=cfg.open_px,
            close_px=cfg.close_px,
            min_blob_area_px=cfg.min_blob_area_px,
        )

    def segment(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lower, upper in self.ranges:
            mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        if self.open_px > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(self.open_px))
        if self.close_px > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(self.close_px))
        if self.min_blob_area_px > 0:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            keep = np.zeros(n, bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_blob_area_px
            mask = keep[labels].astype(np.uint8)
        return mask.astype(bool)

    def describe(self) -> dict:
        return {
            "method": self.name,
            "ranges": [[list(lo), list(hi)] for lo, hi in self.ranges],
            "open_px": self.open_px,
            "close_px": self.close_px,
            "min_blob_area_px": self.min_blob_area_px,
        }


def hsv_range_from_samples(
    hsv_pixels: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    hue_margin: int = 5,
    sv_margin: int = 25,
) -> list[HsvBounds]:
    """Suggest HSV ranges covering sampled pixels (Nx3, OpenCV HSV).

    Hue is circular (0 and 179 are both red), so hues are rotated to center the samples
    before taking percentiles; a range crossing 0 is split in two.
    """
    px = np.asarray(hsv_pixels, dtype=np.float64).reshape(-1, 3)
    if len(px) == 0:
        raise ValueError("no pixels sampled")
    angles = px[:, 0] * (2 * np.pi / 180)
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    center = (mean_angle * 180 / (2 * np.pi)) % 180
    shifted = (px[:, 0] - center + 90) % 180  # samples now centered on 90

    h_lo = int(np.floor(np.percentile(shifted, low_pct))) - hue_margin
    h_hi = int(np.ceil(np.percentile(shifted, high_pct))) + hue_margin
    s_lo, v_lo = (int(np.percentile(px[:, i], low_pct)) - sv_margin for i in (1, 2))
    s_hi, v_hi = (int(np.percentile(px[:, i], high_pct)) + sv_margin for i in (1, 2))
    s_lo, v_lo = max(0, s_lo), max(0, v_lo)
    s_hi, v_hi = min(255, s_hi), min(255, v_hi)

    if h_hi - h_lo >= 179:
        return [((0, s_lo, v_lo), (179, s_hi, v_hi))]
    offset = int(round(center)) - 90
    lo, hi = h_lo + offset, h_hi + offset
    if lo < 0:
        return [((0, s_lo, v_lo), (hi, s_hi, v_hi)), ((lo + 180, s_lo, v_lo), (179, s_hi, v_hi))]
    if hi > 179:
        return [((lo, s_lo, v_lo), (179, s_hi, v_hi)), ((0, s_lo, v_lo), (hi - 180, s_hi, v_hi))]
    return [((lo, s_lo, v_lo), (hi, s_hi, v_hi))]
