"""Reference geometry drawn once on the reference frame, and measurements against it."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from fungus_cv.measure.path import ExtentMeasurement, Polyline, measure_along_path

ANNOTATIONS_NAME = "annotations.json"

__all__ = ["Annotations", "ExtentMeasurement", "measure_extent", "extent_uncertainty_mm"]


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
    # Optional polygon on a neutral surface that never changes (white/grey card), used by
    # analysis.lighting.method: patch.
    reference_patch: list[tuple[float, float]] | None = None
    # Optional line from base to tip following a curved object, clicked in `fungus annotate`
    # (base and tip included). Used by analysis.measure.mode: path.
    path: list[tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if len(self.roi) < 3:
            raise ValueError("roi needs at least 3 points")
        if math.dist(self.base, self.tip) < 1:
            raise ValueError("base and tip must be different points")
        if self.reference_patch is not None and len(self.reference_patch) < 3:
            raise ValueError("reference_patch needs at least 3 points")

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
            reference_patch=[tuple(p) for p in data["reference_patch"]]
            if data.get("reference_patch") else None,
            path=[tuple(p) for p in data["path"]] if data.get("path") else None,
        )

    def polyline(self) -> Polyline:
        """The clicked path, or the straight base -> tip axis if none was clicked."""
        return Polyline(self.path if self.path else [self.base, self.tip])

    def patch_mask(self, shape: tuple[int, int]) -> np.ndarray | None:
        if not self.reference_patch:
            return None
        mask = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(np.array(self.reference_patch)).astype(np.int32)], 1)
        return mask.astype(bool)


def measure_extent(
    mask: np.ndarray, ann: Annotations, front_percentile: float = 50.0
) -> ExtentMeasurement:
    """How far the target reaches from ``base`` toward ``tip`` along the straight axis.

    The region is cut into 1-px strips parallel to the axis. In each strip the front is the
    far edge of the furthest target pixel. ``extent_px`` is the ``front_percentile`` of those
    per-strip fronts (50 = median front height across the width, 100 = highest point) and
    ``extent_max_px`` is the highest. Pixels behind the base (e.g. dye in the reservoir)
    count as 0. See `measure_along_path` for curved objects.
    """
    return measure_along_path(mask, Polyline([ann.base, ann.tip]), ann.roi_mask(mask.shape),
                              front_percentile)


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
