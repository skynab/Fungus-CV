"""Reference geometry drawn once on the reference frame, and measurements against it."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ANNOTATIONS_NAME = "annotations.json"


@dataclass
class Annotations:
    """Clicked once on the reference frame; valid for every aligned frame.

    ``base`` is where growth starts (e.g. the waterline or soil line), ``tip`` is the far end
    of the reference object (top of the towel or stem), ``roi`` is a polygon around the
    object; only target pixels inside it are measured.
    """

    base: tuple[float, float]
    tip: tuple[float, float]
    roi: list[tuple[float, float]]
    image_size: tuple[int, int]  # (width, height) of the reference frame
    reference_file: str = ""

    def __post_init__(self) -> None:
        if len(self.roi) < 3:
            raise ValueError("roi needs at least 3 points")
        if math.dist(self.base, self.tip) < 1:
            raise ValueError("base and tip must be different points")

    @property
    def axis_length_px(self) -> float:
        return math.dist(self.base, self.tip)

    def roi_mask(self, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(np.array(self.roi)).astype(np.int32)], 1)
        return mask.astype(bool)

    def save(self, path: Path) -> None:
        data = asdict(self)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Annotations:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            base=tuple(data["base"]),
            tip=tuple(data["tip"]),
            roi=[tuple(p) for p in data["roi"]],
            image_size=tuple(data["image_size"]),
            reference_file=data.get("reference_file", ""),
        )


@dataclass
class ExtentMeasurement:
    target_px: int  # target pixels inside the ROI
    extent_px: float  # front position along the axis (percentile across the width)
    extent_max_px: float  # furthest point of the front
    extent_fraction: float  # extent / axis length
    front_width_px: int  # number of 1-px strips across the axis that contain target
    roi_area_px: int
    coverage_fraction: float  # target area / ROI area


def measure_extent(
    mask: np.ndarray, ann: Annotations, front_percentile: float = 50.0
) -> ExtentMeasurement:
    """How far the target reaches from ``base`` toward ``tip``.

    The region is cut into 1-px strips parallel to the axis. In each strip the front is the
    far edge of the furthest target pixel. ``extent_px`` is the ``front_percentile`` of those
    per-strip fronts (50 = median front height across the width, 100 = highest point) and
    ``extent_max_px`` is the highest. Pixels behind the base (e.g. dye in the reservoir)
    count as 0.
    """
    roi = ann.roi_mask(mask.shape)
    inside = mask & roi
    ys, xs = np.nonzero(inside)
    roi_area = int(roi.sum())
    length = ann.axis_length_px
    if len(xs) == 0:
        return ExtentMeasurement(0, 0.0, 0.0, 0.0, 0, roi_area, 0.0)

    base = np.asarray(ann.base, np.float64)
    u = (np.asarray(ann.tip, np.float64) - base) / length  # along the axis
    n = np.array([-u[1], u[0]])  # across the axis
    dx, dy = xs - base[0], ys - base[1]
    along = dx * u[0] + dy * u[1]
    across = dx * n[0] + dy * n[1]
    # A pixel is a unit square around its integer center; its far edge along the axis is
    # half its projected size beyond the center.
    along_edge = np.clip(along + 0.5 * (abs(u[0]) + abs(u[1])), 0, None)

    strips = np.floor(across).astype(np.int64)
    order = np.argsort(strips, kind="stable")
    strips, along_edge = strips[order], along_edge[order]
    starts = np.flatnonzero(np.r_[True, strips[1:] != strips[:-1]])
    fronts = np.maximum.reduceat(along_edge, starts)

    extent = float(np.percentile(fronts, front_percentile))
    return ExtentMeasurement(
        target_px=int(len(xs)),
        extent_px=extent,
        extent_max_px=float(fronts.max()),
        extent_fraction=extent / length,
        front_width_px=int(len(fronts)),
        roi_area_px=roi_area,
        coverage_fraction=len(xs) / roi_area if roi_area else 0.0,
    )


def extent_uncertainty_mm(
    extent_px: float,
    mm_per_px: float,
    scale_se_mm_per_px: float,
    align_rms_px: float | None,
) -> float:
    """Combined standard uncertainty of an extent in mm (independent terms in quadrature).

    - scale: uncertainty of mm/px from the marker edges, times the extent
    - pixel: front position quantized to whole pixels (uniform, sd = 1/sqrt(12) px)
    - alignment: residual registration error between this frame and the reference

    Segmentation uncertainty (where exactly the color threshold puts the edge) is not
    included; estimate it by re-running with slightly different thresholds.
    """
    scale_term = extent_px * scale_se_mm_per_px
    pixel_term = mm_per_px / math.sqrt(12)
    align_term = (align_rms_px or 0.0) * mm_per_px
    return math.sqrt(scale_term**2 + pixel_term**2 + align_term**2)
