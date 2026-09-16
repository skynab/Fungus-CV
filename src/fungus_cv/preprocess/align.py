"""Align each frame to the reference frame so annotations and measurements line up."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from fungus_cv.preprocess.markers import Markers, detect_markers
from fungus_cv.quality import to_gray

log = logging.getLogger(__name__)


@dataclass
class Alignment:
    method: str  # markers | ecc | identity | failed
    matrix: np.ndarray  # 2x3, maps frame -> reference coordinates
    rms_px: float | None = None  # residual of marker corners after alignment
    n_points: int = 0
    ecc: float | None = None

    @property
    def scale(self) -> float:
        """Zoom factor of the correction; drifting away from 1 means the camera moved."""
        return float(np.sqrt(abs(np.linalg.det(self.matrix[:, :2]))))

    @property
    def shift_px(self) -> float:
        return float(np.hypot(self.matrix[0, 2], self.matrix[1, 2]))


IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], np.float32)


def align_by_markers(frame_markers: Markers, ref_markers: Markers) -> Alignment | None:
    shared = sorted(set(frame_markers) & set(ref_markers))
    if not shared:
        return None
    src = np.concatenate([frame_markers[i] for i in shared])
    dst = np.concatenate([ref_markers[i] for i in shared])
    # Similarity transform (shift, rotation, uniform zoom): a bumped camera or pot, without
    # letting a few corners bend the image.
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        return None
    moved = src @ matrix[:, :2].T + matrix[:, 2]
    rms = float(np.sqrt(np.mean(np.sum((moved - dst) ** 2, axis=1))))
    return Alignment("markers", matrix.astype(np.float32), rms_px=rms, n_points=len(src))


# ECC correlation below these: alignment is doubtful / unusable.
ECC_POOR = 0.9
ECC_FAILED = 0.6


def align_by_ecc(
    frame: np.ndarray,
    reference: np.ndarray,
    exclude: np.ndarray | None = None,
    max_side: int = 1000,
) -> Alignment | None:
    """Intensity-based alignment (rotation + shift) for scenes without markers.

    ``exclude`` marks pixels to ignore, normally the region where growth happens: a large
    change there (e.g. the dye front) would otherwise pull the alignment off. Runs coarse to
    fine so moderate bumps still converge.
    """
    ref_gray = to_gray(reference).astype(np.float32)
    img_gray = to_gray(frame).astype(np.float32)
    valid = np.full(ref_gray.shape, 255, np.uint8)
    if exclude is not None:
        valid[exclude] = 0

    base = min(1.0, max_side / max(ref_gray.shape))
    warp = np.eye(2, 3, dtype=np.float32)  # in full-resolution coordinates, ref -> frame
    cc = None
    for level in (0.25, 1.0):
        f = base * level
        size = (max(8, int(ref_gray.shape[1] * f)), max(8, int(ref_gray.shape[0] * f)))
        ref_s, img_s = (
            cv2.GaussianBlur(cv2.resize(g, size, interpolation=cv2.INTER_AREA), (0, 0), 1)
            for g in (ref_gray, img_gray)
        )
        mask_s = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
        w = warp.copy()
        w[:, 2] *= f
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        try:
            cc, w = cv2.findTransformECC(ref_s, img_s, w, cv2.MOTION_EUCLIDEAN, criteria,
                                         mask_s, 5)
        except cv2.error as exc:
            log.debug("ECC alignment failed at scale %.2f: %s", f, exc)
            return None
        w[:, 2] /= f
        warp = w
    inverse = cv2.invertAffineTransform(warp)
    method = "ecc" if cc is not None and cc >= ECC_FAILED else "failed"
    return Alignment(method, inverse.astype(np.float32), ecc=float(cc))


def align_frame(
    frame: np.ndarray,
    reference: np.ndarray,
    ref_markers: Markers,
    mode: str = "markers_or_ecc",
    dictionary: str = "DICT_4X4_50",
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, Alignment]:
    """Return the frame warped into reference coordinates, and how that was done."""
    if mode == "none":
        return frame, Alignment("identity", IDENTITY.copy())

    alignment = None
    if mode in ("markers", "markers_or_ecc") and ref_markers:
        alignment = align_by_markers(detect_markers(frame, dictionary), ref_markers)
    if alignment is None and mode in ("ecc", "markers_or_ecc"):
        alignment = align_by_ecc(frame, reference, exclude)
    if alignment is None or alignment.method == "failed":
        return frame, Alignment("failed", IDENTITY.copy(),
                                ecc=alignment.ecc if alignment else None)

    h, w = reference.shape[:2]
    warped = cv2.warpAffine(frame, alignment.matrix, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, alignment
