"""Measuring along a (possibly curved) path such as a plant stem's centerline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree


class Polyline:
    """A path densely resampled by arc length, for projecting pixels onto it."""

    def __init__(self, points, spacing: float = 0.25):
        pts = np.asarray(points, np.float64).reshape(-1, 2)
        keep = np.r_[True, np.any(np.abs(np.diff(pts, axis=0)) > 1e-9, axis=1)]
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("a path needs at least two distinct points")
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.r_[0.0, np.cumsum(seg)]
        self.points = pts
        self.length = float(cum[-1])
        self.spacing = spacing
        s = np.arange(0.0, self.length, spacing)
        self.s = np.r_[s, self.length] if s[-1] < self.length else s
        self.dense = np.c_[np.interp(self.s, cum, pts[:, 0]), np.interp(self.s, cum, pts[:, 1])]
        tangents = np.gradient(self.dense, axis=0)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        self.tangents = tangents / np.where(norms > 0, norms, 1)
        self._tree = cKDTree(self.dense)

    def project(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Arc-length position ``along``, signed offset ``across`` and local unit tangent.

        Points beyond either end continue along the end tangent (``along`` < 0 before the
        start, > length past the end).
        """
        pts = np.c_[np.asarray(xs, np.float64), np.asarray(ys, np.float64)]
        _, idx = self._tree.query(pts)
        t = self.tangents[idx]
        off = pts - self.dense[idx]
        along = self.s[idx] + np.einsum("ij,ij->i", off, t)
        across = off[:, 1] * t[:, 0] - off[:, 0] * t[:, 1]
        return along, across, t

    def point_at(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and unit tangent at arc length ``s`` (clamped to the path)."""
        s = float(np.clip(s, 0.0, self.length))
        x = np.interp(s, self.s, self.dense[:, 0])
        y = np.interp(s, self.s, self.dense[:, 1])
        k = int(np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1))
        return np.array([x, y]), self.tangents[k]

    def smoothed(self, window_px: float) -> Polyline:
        """Savitzky-Golay smoothing over ``window_px`` of arc length; endpoints stay put.

        Pixel-traced centerlines zig-zag; their length is overestimated by up to ~8% (the
        8-connected staircase), so smooth before measuring lengths.
        """
        n = len(self.dense)
        win = int(round(window_px / self.spacing)) | 1
        win = min(win, n - 1 if (n - 1) % 2 else n - 2)
        if win < 5:
            return self
        xs = savgol_filter(self.dense[:, 0], win, 2, mode="interp")
        ys = savgol_filter(self.dense[:, 1], win, 2, mode="interp")
        xs[0], ys[0] = self.dense[0]
        xs[-1], ys[-1] = self.dense[-1]
        # Resample coarser before rebuilding so the new tangents are not noisy.
        step = max(1, int(round(1.0 / self.spacing)))
        idx = np.r_[np.arange(0, n - 1, step), n - 1]
        return Polyline(np.c_[xs[idx], ys[idx]], self.spacing)


@dataclass
class ExtentMeasurement:
    target_px: int  # target pixels counted (inside the region and corridor)
    extent_px: float  # front position along the path (percentile across the width)
    extent_max_px: float  # furthest point of the front
    extent_fraction: float  # extent / path length
    front_width_px: int  # number of 1-px strips alongside the path that contain target
    roi_area_px: int
    coverage_fraction: float  # target area / region area
    path_length_px: float = 0.0
    covered_length_px: float = 0.0  # length of path with target beside it
    reference_px: int | None = None  # reference object (e.g. stem) pixels, if segmented
    reference_covered_fraction: float | None = None  # target on reference / reference


def _fronts(along_edge: np.ndarray, across: np.ndarray) -> np.ndarray:
    strips = np.floor(across).astype(np.int64)
    order = np.argsort(strips, kind="stable")
    strips, along_edge = strips[order], along_edge[order]
    starts = np.flatnonzero(np.r_[True, strips[1:] != strips[:-1]])
    return np.maximum.reduceat(along_edge, starts)


def measure_along_path(
    mask: np.ndarray,
    path: Polyline,
    roi: np.ndarray,
    front_percentile: float = 50.0,
    corridor_px: float | None = None,
    reference_mask: np.ndarray | None = None,
) -> ExtentMeasurement:
    """How far the target reaches along ``path``, measured by arc length from its start.

    The region is cut into 1-px strips running alongside the path. In each strip the front
    is the far edge of the furthest target pixel; ``extent_px`` is the ``front_percentile``
    of those fronts (50 = median across the width, 100 = furthest point). Target behind the
    start counts as 0. ``corridor_px`` ignores target further than that from the path (e.g.
    moss on the ground beside a stem).
    """
    inside = mask & roi
    roi_area = int(roi.sum())
    length = path.length
    ys, xs = np.nonzero(inside)
    along = across = t = np.empty(0)
    if len(xs):
        along, across, t = path.project(xs, ys)
        if corridor_px is not None:
            keep = np.abs(across) <= corridor_px
            along, across, t = along[keep], across[keep], t[keep]

    reference_px = reference_frac = None
    if reference_mask is not None:
        ref = reference_mask & roi
        ry, rx = np.nonzero(ref)
        if len(rx):
            r_along, r_across, _ = path.project(rx, ry)
            ok = r_along >= -0.5
            if corridor_px is not None:
                ok &= np.abs(r_across) <= corridor_px
            ref_sel = np.zeros_like(ref)
            ref_sel[ry[ok], rx[ok]] = True
            reference_px = int(ok.sum())
            reference_frac = (float((ref_sel & mask).sum() / reference_px)
                              if reference_px else 0.0)
        else:
            reference_px, reference_frac = 0, 0.0

    if len(along) == 0:
        return ExtentMeasurement(0, 0.0, 0.0, 0.0, 0, roi_area, 0.0, length, 0.0,
                                 reference_px, reference_frac)

    # A pixel is a unit square around its integer center; its far edge along the path is
    # half its projected size beyond the center.
    along_edge = np.clip(along + 0.5 * (np.abs(t[:, 0]) + np.abs(t[:, 1])), 0, None)
    fronts = _fronts(along_edge, across)
    extent = float(np.percentile(fronts, front_percentile))

    on_path = along[(along >= 0) & (along < length)]
    covered = float(len(np.unique(np.floor(on_path).astype(np.int64))))
    return ExtentMeasurement(
        target_px=int(len(along)),
        extent_px=extent,
        extent_max_px=float(fronts.max()),
        extent_fraction=extent / length if length else 0.0,
        front_width_px=int(len(fronts)),
        roi_area_px=roi_area,
        coverage_fraction=len(along) / roi_area if roi_area else 0.0,
        path_length_px=length,
        covered_length_px=min(covered, length),
        reference_px=reference_px,
        reference_covered_fraction=reference_frac,
    )
