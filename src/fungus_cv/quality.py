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


def contrast_normalized_sharpness(image: np.ndarray) -> float:
    """Laplacian variance divided by intensity variance.

    Dimming or brightening scales both by the same factor, so this only changes when edges
    actually get softer (blur, lost focus) - unlike `sharpness`, which drops in dim light.
    """
    gray = to_gray(image).astype(np.float64)
    var = gray.var()
    return float(cv2.Laplacian(gray, cv2.CV_64F).var() / var) if var > 0 else 0.0
