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

__all__ = ["Annotations", "ExtentMeasurement", "Plot", "measure_extent",
           "extent_uncertainty_mm", "plots_hull"]


def _polygon_mask(points, shape) -> np.ndarray:
    """Pixels whose centres lie inside the polygon.

    ``cv2.fillPoly`` also includes every pixel the outline touches, which inflates areas by
    about half a pixel around the whole perimeter (~1% for a typical field plot).
    """
    from matplotlib.path import Path as MplPath

    h, w = shape[:2]
    poly = np.asarray(points, np.float64)
    mask = np.zeros((h, w), bool)
    x0, y0 = np.maximum(np.floor(poly.min(axis=0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(poly.max(axis=0)).astype(int) + 1, (w, h))
    if x1 <= x0 or y1 <= y0:
        return mask
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = MplPath(poly).contains_points(np.c_[xx.ravel(), yy.ravel()])
    mask[y0:y1, x0:x1] = inside.reshape(yy.shape)
    return mask


def _points(data) -> list[tuple[float, float]] | None:
    return [tuple(p) for p in data] if data else None


@dataclass
class Plot:
    """A named region measured on its own (e.g. one field plot or one plant).

    ``base``/``tip``/``path`` are optional: without them only area, coverage and colour are
    measured; with them extent is measured too.
    """

    name: str
    polygon: list[tuple[float, float]]
    base: tuple[float, float] | None = None
    tip: tuple[float, float] | None = None
    path: list[tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"plot {self.name!r} needs at least 3 polygon points")
        if (self.base is None) != (self.tip is None):
            raise ValueError(f"plot {self.name!r}: give both base and tip, or neither")
        if self.base is not None and math.dist(self.base, self.tip) < 1:
            raise ValueError(f"plot {self.name!r}: base and tip must be different points")

    @property
    def has_axis(self) -> bool:
        return self.base is not None

    def mask(self, shape) -> np.ndarray:
        return _polygon_mask(self.polygon, shape)

    def polyline(self, follow_path: bool = True) -> Polyline:
        if not self.has_axis:
            raise ValueError(f"plot {self.name!r} has no base/tip")
        return Polyline(self.path if (follow_path and self.path) else [self.base, self.tip])


@dataclass
class Annotations:
    """Clicked once on the reference frame; valid for every aligned frame.

    ``base`` is where growth starts (e.g. the waterline or soil line), ``tip`` is the far end
    of the reference object (top of the towel or stem), ``roi`` is a polygon around the
    object; only target pixels inside it are measured.

    For fields, ``plots`` lists named regions measured separately; ``base``/``tip`` may then
    be omitted and ``roi`` should enclose all plots.
    """

    base: tuple[float, float] | None
    tip: tuple[float, float] | None
    roi: list[tuple[float, float]]
    image_size: tuple[int, int]  # (width, height) of the reference frame
    reference_file: str = ""
    # Optional polygon on a neutral surface that never changes (white/grey card), used by
    # analysis.lighting.method: patch.
    reference_patch: list[tuple[float, float]] | None = None
    # Optional line from base to tip following a curved object, clicked in `fungus annotate`
    # (base and tip included). Used by analysis.measure.mode: path.
    path: list[tuple[float, float]] | None = None
    plots: list[Plot] | None = None

    def __post_init__(self) -> None:
        if len(self.roi) < 3:
            raise ValueError("roi needs at least 3 points")
        if (self.base is None) != (self.tip is None):
            raise ValueError("give both base and tip, or neither")
        if self.base is None and not self.plots:
            raise ValueError("base and tip are required unless plots are defined")
        if self.base is not None and math.dist(self.base, self.tip) < 1:
            raise ValueError("base and tip must be different points")
        if self.reference_patch is not None and len(self.reference_patch) < 3:
            raise ValueError("reference_patch needs at least 3 points")
        if self.plots:
            names = [p.name for p in self.plots]
            if len(names) != len(set(names)):
                raise ValueError(f"plot names must be unique, got {names}")

    @property
    def axis_length_px(self) -> float:
        return math.dist(self.base, self.tip)

    def roi_mask(self, shape: tuple[int, int]) -> np.ndarray:
        return _polygon_mask(self.roi, shape)

    def measured_plots(self) -> list[Plot]:
        """The plots to measure: the defined plots, or the whole region as plot ``main``."""
        if self.plots:
            return list(self.plots)
        return [Plot("main", list(self.roi), self.base, self.tip, self.path)]

    def save(self, path: Path) -> None:
        data = asdict(self)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Annotations:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        plots = None
        if data.get("plots"):
            plots = [Plot(name=pl["name"], polygon=_points(pl["polygon"]),
                          base=tuple(pl["base"]) if pl.get("base") else None,
                          tip=tuple(pl["tip"]) if pl.get("tip") else None,
                          path=_points(pl.get("path")))
                     for pl in data["plots"]]
        return cls(
            base=tuple(data["base"]) if data.get("base") else None,
            tip=tuple(data["tip"]) if data.get("tip") else None,
            roi=_points(data["roi"]),
            image_size=tuple(data["image_size"]),
            reference_file=data.get("reference_file", ""),
            reference_patch=_points(data.get("reference_patch")),
            path=_points(data.get("path")),
            plots=plots,
        )

    def polyline(self) -> Polyline:
        """The clicked path, or the straight base -> tip axis if none was clicked."""
        if self.base is None:
            raise ValueError("these annotations have no base/tip (field plots only)")
        return Polyline(self.path if self.path else [self.base, self.tip])

    def patch_mask(self, shape: tuple[int, int]) -> np.ndarray | None:
        if not self.reference_patch:
            return None
        return _polygon_mask(self.reference_patch, shape)


def plots_hull(plots: list[Plot], margin: float = 0.0) -> list[tuple[float, float]]:
    """Convex hull around all plot polygons, e.g. as the overall region of interest."""
    pts = np.concatenate([np.asarray(p.polygon, np.float32) for p in plots])
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
    if margin:
        centre = hull.mean(0)
        direction = hull - centre
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        hull = hull + direction / np.where(norm > 0, norm, 1) * margin
    return [tuple(map(float, p)) for p in hull]


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
