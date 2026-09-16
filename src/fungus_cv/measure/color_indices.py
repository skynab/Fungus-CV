"""Colour indices for tracking greenness or discolouration of a region without segmenting it.

These are the chromatic coordinates used in plant phenology camera networks (e.g. PhenoCam):

- GCC, green chromatic coordinate = G / (R + G + B)
- RCC, red chromatic coordinate   = R / (R + G + B)
- ExG, excess green               = 2*GCC - RCC - BCC

GCC's 90th percentile (``gcc_p90``) is the usual robust summary: it is less affected by
shadows and dark gaps than the mean. Compute them on lighting-corrected frames.
"""

from __future__ import annotations

import numpy as np

KEYS = ("gcc_mean", "gcc_p90", "rcc_mean", "exg_mean")


def color_indices(image_bgr: np.ndarray, region: np.ndarray) -> dict[str, float]:
    # Exclude pure-black pixels: the empty border that alignment/rectification leaves.
    sel = region & (image_bgr.max(axis=-1) > 0)
    if not sel.any():
        return {k: float("nan") for k in KEYS}
    px = image_bgr[sel].astype(np.float64)
    total = px.sum(axis=1)
    total[total == 0] = 1.0
    b, g, r = px[:, 0] / total, px[:, 1] / total, px[:, 2] / total
    return {
        "gcc_mean": float(g.mean()),
        "gcc_p90": float(np.percentile(g, 90)),
        "rcc_mean": float(r.mean()),
        "exg_mean": float((2 * g - r - b).mean()),
    }
