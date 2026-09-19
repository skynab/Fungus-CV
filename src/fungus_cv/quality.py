"""Cheap per-frame quality metrics, stored with every frame to flag bad shots later."""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def mean_brightness(image: np.ndarray) -> float:
    """Mean gray level, 0-255. Sudden jumps suggest lighting or exposure changed."""
    return float(to_gray(image).mean())


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian. Higher is sharper; a sudden drop suggests blur or refocus."""
    return float(cv2.Laplacian(to_gray(image), cv2.CV_64F).var())


def contrast_normalized_sharpness(image: np.ndarray, region: np.ndarray | None = None) -> float:
    """Laplacian variance divided by intensity variance, optionally within ``region``.

    Dimming or brightening scales both by the same factor, so this only changes when edges
    actually get softer (blur, lost focus) - unlike `sharpness`, which drops in dim light.
    Growth changes the scene itself (a large dark patch raises the contrast without adding
    edges), so compare frames on a ``region`` that doesn't change: outside the measured area.
    """
    gray = to_gray(image).astype(np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    if region is not None and region.shape == gray.shape and region.sum() > 100:
        gray, lap = gray[region], lap[region]
    var = gray.var()
    return float(lap.var() / var) if var > 0 else 0.0
