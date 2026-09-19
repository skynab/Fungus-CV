"""What a model's training images look like, to notice when it is used on something else.

A model trained on bark photos in spring can produce confident nonsense on a new scene: other
light, other season, a different camera. At training time the colour, brightness, contrast and
texture of every training image are summarised; during analysis each frame's distance from
that summary (Mahalanobis, i.e. in units of the training spread) is recorded, and frames far
outside what the model has seen are flagged ``unfamiliar_input``. It says nothing about
whether the masks are right, only that the model is being asked about something new.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

FEATURES = ("lightness_mean", "lightness_sd", "a_mean", "a_sd", "b_mean", "b_sd",
            "saturation_mean", "texture")
# Flag a frame further out than this multiple of the furthest training image (and at least
# the chi-square 99.9% point for this many features, so tiny datasets aren't over-strict).
MARGIN = 1.5
CHI2_999 = 26.12  # chi2(8 dof) 99.9%: distance^2 limit for normally distributed features


def input_features(bgr: np.ndarray) -> np.ndarray:
    """Eight numbers describing an image's colour, brightness, contrast and texture."""
    small = bgr
    h, w = bgr.shape[:2]
    if max(h, w) > 512:  # the summary doesn't need full resolution
        scale = 512 / max(h, w)
        small = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float64)
    texture = float(np.abs(cv2.Laplacian(gray, cv2.CV_64F)).mean())
    return np.array([lab[:, 0].mean(), lab[:, 0].std(), lab[:, 1].mean(), lab[:, 1].std(),
                     lab[:, 2].mean(), lab[:, 2].std(), float(hsv[..., 1].mean()), texture])


def build_profile(images) -> dict:
    """Summary of the training images, stored in the model card."""
    x = np.array([input_features(img) for img in images], float)
    if len(x) < 2:
        raise ValueError("a profile needs at least two images")
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    # Shrink toward the diagonal: with few images the full covariance is unstable.
    shrink = min(1.0, len(FEATURES) / len(x))
    cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
    cov += np.eye(len(FEATURES)) * 1e-6 * max(float(np.trace(cov)), 1.0)
    profile = {"features": list(FEATURES), "n": len(x), "mean": mean.tolist(),
               "covariance": cov.tolist()}
    distances = [distance(profile, f) for f in x]
    profile["max_training_distance"] = float(max(distances))
    profile["limit"] = float(max(MARGIN * max(distances), math.sqrt(CHI2_999)))
    return profile


def distance(profile: dict, features: np.ndarray) -> float:
    diff = np.asarray(features, float) - np.asarray(profile["mean"], float)
    cov = np.asarray(profile["covariance"], float)
    return float(math.sqrt(max(diff @ np.linalg.solve(cov, diff), 0.0)))


def image_distance(profile: dict, bgr: np.ndarray) -> float:
    return distance(profile, input_features(bgr))


def is_unfamiliar(profile: dict, value: float) -> bool:
    return value > profile["limit"]
