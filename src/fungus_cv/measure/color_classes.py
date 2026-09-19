"""Share of a region in each named colour class: whole-field change without segmenting a target.

For discolouration that spreads over a whole field (leaves yellowing, a lawn browning, mould
greying a surface) the question is "what share of the plot is which colour", not "how far has
something grown". Each class is a set of HSV ranges, as for colour segmentation; a pixel
belongs to the first class it matches, so the shares never add up to more than 100 %.

Uncertainty: the class boundaries are a judgement call, so the shares are also computed with
every range narrowed and widened by ``hsv_delta`` (the same rule as segmentation uncertainty).
The spread of the three is treated as rectangular: u = (max - min) / (2*sqrt(3)).

Measure on lighting-corrected frames, or a dimmer day will look like a browner field.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from fungus_cv.segment.color import HsvBounds, shift_ranges


def column(name: str) -> str:
    return f"class_{name}_pct"


def columns(names: list[str]) -> list[str]:
    return [c for name in names for c in (column(name), column(name) + "_unc")]


def label_pixels(hsv: np.ndarray, classes: list[tuple[str, list[HsvBounds]]]) -> np.ndarray:
    """Per pixel: the index of the first matching class, or -1."""
    labels = np.full(hsv.shape[:2], -1, np.int16)
    for k, (_, ranges) in enumerate(classes):
        hit = np.zeros(hsv.shape[:2], bool)
        for lower, upper in ranges:
            hit |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8)) > 0
        labels[hit & (labels < 0)] = k
    return labels


def _shares(hsv: np.ndarray, sel: np.ndarray,
            classes: list[tuple[str, list[HsvBounds]]]) -> np.ndarray:
    labels = label_pixels(hsv, classes)[sel]
    counts = np.bincount(labels[labels >= 0], minlength=len(classes))[:len(classes)]
    return 100.0 * counts / labels.size


def class_shares(image_bgr: np.ndarray, region: np.ndarray,
                 classes: list[tuple[str, list[HsvBounds]]],
                 hsv_delta: tuple[int, int, int] | None = None) -> dict[str, float]:
    """``class_<name>_pct`` and ``class_<name>_pct_unc`` for every class, within ``region``.

    Pure-black pixels (the empty border alignment leaves) are not part of the field.
    """
    if not classes:
        return {}
    sel = region & (image_bgr.max(axis=-1) > 0)
    if not sel.any():
        return {c: math.nan for c in columns([n for n, _ in classes])}
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    base = _shares(hsv, sel, classes)
    if hsv_delta is not None:
        spread = np.stack([base] + [
            _shares(hsv, sel, [(n, shift_ranges(r, hsv_delta, sign)) for n, r in classes])
            for sign in (-1, 1)])
        unc = (spread.max(0) - spread.min(0)) / (2 * math.sqrt(3))
    else:
        unc = np.full(len(classes), math.nan)
    out = {}
    for (name, _), share, u in zip(classes, base, unc):
        out[column(name)] = float(share)
        out[column(name) + "_unc"] = float(u)
    return out
