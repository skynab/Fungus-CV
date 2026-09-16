"""Mask agreement metrics."""

from __future__ import annotations

import cv2
import numpy as np


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def boundary_f1(pred: np.ndarray, truth: np.ndarray, tolerance_px: float = 2.0) -> float:
    """F1 of boundary pixels matched within ``tolerance_px``. Sensitive to edge placement,
    which area metrics like IoU barely notice on large objects."""
    bp, bt = _boundary(pred), _boundary(truth)
    if not bp.any() and not bt.any():
        return 1.0
    if not bp.any() or not bt.any():
        return 0.0
    dist_to_t = cv2.distanceTransform((~bt).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_p = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((dist_to_t[bp] <= tolerance_px).mean())
    recall = float((dist_to_p[bt] <= tolerance_px).mean())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def mask_metrics(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> dict:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    if valid is not None:
        pred, truth = pred & valid, truth & valid
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    union = tp + fp + fn
    return {
        "iou": tp / union if union else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if union else 1.0,
        "precision": tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0),
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "boundary_f1_2px": boundary_f1(pred, truth, 2.0),
        "pred_px": tp + fp,
        "truth_px": tp + fn,
    }


def summarize(rows: list[dict], keys=("iou", "dice", "precision", "recall", "boundary_f1_2px")):
    out = {"n": len(rows)}
    for k in keys:
        values = np.array([r[k] for r in rows], float)
        if len(values):
            out[f"{k}_mean"] = float(values.mean())
            out[f"{k}_min"] = float(values.min())
    return out
