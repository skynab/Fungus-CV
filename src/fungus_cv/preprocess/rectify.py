"""Perspective correction: warp frames to a top-down view of the marker plane.

Each marker is a square of known size lying in the measured plane. A single plane-to-image
homography is fitted so that *all* detected markers become squares of that size, with each
marker free to sit anywhere and at any angle. No printed layout is needed, so markers on
separate stakes in a field work as well as markers on one sheet.

With one marker the fit is exact for that marker and extrapolates to the rest of the image;
spread several markers around the measured area for a well-constrained correction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from fungus_cv.preprocess.markers import Markers

log = logging.getLogger(__name__)

UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)


def _apply_h(h: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx2 points; also returns the homogeneous w (<= 0 means behind the horizon)."""
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3], p[:, 2]


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


@dataclass
class Rectification:
    matrix: np.ndarray  # 3x3, source image pixels -> rectified pixels
    size: tuple[int, int]  # (width, height) of the rectified image
    px_per_mm: float
    residual_rms_mm: float  # how far marker corners are from perfect squares after the fit
    n_markers: int

    def warp(self, image: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        return cv2.warpPerspective(image, self.matrix, self.size, flags=interpolation,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def unwarp_mask(self, mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        """Map a mask in rectified coordinates back onto the source image."""
        out = cv2.warpPerspective(mask.astype(np.uint8), self.matrix, image_size,
                                  flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out.astype(bool)

    def to_dict(self) -> dict:
        return {
            "matrix": [[round(float(v), 9) for v in row] for row in self.matrix],
            "size": list(self.size),
            "px_per_mm": round(self.px_per_mm, 6),
            "residual_rms_mm": round(self.residual_rms_mm, 4),
            "n_markers": self.n_markers,
        }


def fit_rectification(
    markers: Markers,
    size_mm: float,
    image_size: tuple[int, int],
    max_side_px: int = 6000,
) -> Rectification:
    if not markers:
        raise ValueError("perspective correction needs at least one ArUco marker in view")
    w, h = image_size
    ids = sorted(markers)
    corners = [markers[i].astype(np.float64) for i in ids]

    # Output resolution: keep roughly the camera's own resolution on the marker plane.
    mean_edge_px = np.mean([np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1).mean()
                            for c in corners])
    px_per_mm = float(mean_edge_px / size_mm)
    side = size_mm * px_per_mm

    # Start from the homography of the largest marker, which is usually the best resolved.
    anchor = int(np.argmax([cv2.contourArea(c.astype(np.float32)) for c in corners]))
    h0 = cv2.getPerspectiveTransform(corners[anchor].astype(np.float32),
                                     (UNIT_SQUARE * side).astype(np.float32)).astype(np.float64)

    others = [k for k in range(len(ids)) if k != anchor]

    def unpack(params):
        hm = np.append(params[:8], 1.0).reshape(3, 3)
        poses = params[8:].reshape(-1, 3)
        return hm, poses

    def residuals(params):
        hm, poses = unpack(params)
        res = [(_apply_h(hm, corners[anchor])[0] - UNIT_SQUARE * side).ravel()]
        for (theta, tx, ty), k in zip(poses, others):
            ideal = UNIT_SQUARE * side @ _rot(theta).T + (tx, ty)
            res.append((_apply_h(hm, corners[k])[0] - ideal).ravel())
        return np.concatenate(res)

    h0 = h0 / h0[2, 2]
    init = [h0.ravel()[:8]]
    for k in others:  # initial pose of each other marker: rigid fit of its projected corners
        proj = _apply_h(h0, corners[k])[0]
        src = UNIT_SQUARE * side
        d = proj - proj.mean(0)
        s = src - src.mean(0)
        u, _, vt = np.linalg.svd(s.T @ d)
        r = (u @ vt).T
        theta = float(np.arctan2(r[1, 0], r[0, 0]))
        t = proj.mean(0) - src.mean(0) @ _rot(theta).T
        init.append([theta, t[0], t[1]])
    params0 = np.concatenate([np.ravel(p) for p in init])

    if others:
        # Scale the homography parameters so the solver's steps are well conditioned.
        fit = least_squares(residuals, params0, x_scale="jac", method="lm", max_nfev=5000)
        params = fit.x
    else:
        params = params0
    hm, _ = unpack(params)
    rms_px = float(np.sqrt(np.mean(residuals(params) ** 2) * 2))  # per-corner distance
    rms_mm = rms_px / px_per_mm

    # Output bounds: all image points in front of the camera, limited to a generous area
    # around the markers so a visible horizon (field shots) doesn't create an infinite canvas.
    gx, gy = np.meshgrid(np.linspace(0, w - 1, 64), np.linspace(0, h - 1, 64))
    plane, wz = _apply_h(hm, np.c_[gx.ravel(), gy.ravel()])
    marker_pts = np.concatenate([_apply_h(hm, c)[0] for c in corners])
    centre = marker_pts.mean(0)
    spread = max(float(np.ptp(marker_pts, axis=0).max()), 20 * side)
    ok = (wz > 0) & (np.abs(plane - centre).max(axis=1) < 4 * spread)
    if not ok.any():
        raise ValueError("could not find a valid rectified area; check the markers")
    lo, hi = plane[ok].min(0), plane[ok].max(0)
    out_w, out_h = hi - lo
    shrink = min(1.0, max_side_px / max(out_w, out_h))
    if shrink < 1.0:
        log.warning("rectified image would be %dx%d px; reducing resolution by %.2fx "
                    "(raise analysis.rectify.max_side_px to keep full detail)",
                    int(out_w), int(out_h), 1 / shrink)
    shift = np.array([[shrink, 0, -lo[0] * shrink], [0, shrink, -lo[1] * shrink], [0, 0, 1]])
    matrix = shift @ hm
    size = (max(1, int(np.ceil(out_w * shrink))), max(1, int(np.ceil(out_h * shrink))))
    return Rectification(matrix / matrix[2, 2], size, px_per_mm * shrink, rms_mm, len(ids))
