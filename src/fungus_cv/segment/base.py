"""Common interface so color thresholds, SAM and trained models are interchangeable."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Segmenter(Protocol):
    name: str

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Return a boolean HxW mask of the target for a BGR image."""
        ...

    def describe(self) -> dict:
        """Settings that affect results; stored with every analysis for provenance."""
        ...


def build_segmenter(target_config) -> Segmenter:
    if target_config.method == "color":
        from fungus_cv.segment.color import ColorThresholdSegmenter

        return ColorThresholdSegmenter.from_config(target_config)
    raise ValueError(f"unknown segmentation method {target_config.method!r}")
