"""Common interface so color thresholds, SAM and trained models are interchangeable."""

from __future__ import annotations

from collections.abc import Callable, Iterator
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


class SequenceSegmenter(Protocol):
    """Segments frames in time order, carrying memory between frames (e.g. SAM 2 video)."""

    name: str

    def segment_sequence(
        self,
        frame_files: list[str],
        load: Callable[[int], np.ndarray],
        needed: set[int],
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(index, mask)`` for every index in ``needed``, in any order.

        ``load(i)`` returns frame ``i`` already aligned to the reference frame. The segmenter
        may load other frames too, to build up memory.
        """
        ...

    def describe(self) -> dict:
        ...


def is_sequence_segmenter(segmenter) -> bool:
    return hasattr(segmenter, "segment_sequence")


def segment_with_variants(segmenter, image: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """The mask plus narrower/wider variants used for segmentation uncertainty.

    Segmenters opt in with ``segment_variants(image) -> (mask, variants)``; others give no
    variants. Sequence segmenters instead yield ``(index, mask, variants)`` triples.
    """
    if hasattr(segmenter, "segment_variants"):
        return segmenter.segment_variants(image)
    return segmenter.segment(image), []


def build_segmenter(target_config, experiment_root=None, roi=None, uncertainty=None,
                    prompts_file=None):
    """``roi`` is the annotated region polygon in reference coordinates (used for cropping).

    ``uncertainty`` (an ``UncertaintyConfig`` with ``segmentation`` on) makes the segmenter
    also produce narrower and wider masks for every frame.
    """
    if uncertainty is not None and not uncertainty.segmentation:
        uncertainty = None
    if target_config.method == "color":
        from fungus_cv.segment.color import ColorThresholdSegmenter

        seg = ColorThresholdSegmenter.from_config(target_config.color)
        seg.variant_hsv_delta = tuple(uncertainty.hsv_delta) if uncertainty else None
        return seg
    if target_config.method == "sam2":
        from fungus_cv.segment.sam2 import Sam2VideoSegmenter

        seg = Sam2VideoSegmenter.from_config(target_config.sam2, experiment_root, roi,
                                            prompts_file)
        seg.variant_logit_delta = uncertainty.logit_delta if uncertainty else None
        return seg
    if target_config.method == "model":
        from fungus_cv.segment.trained import TrainedModelSegmenter

        seg = TrainedModelSegmenter.from_config(target_config.model, experiment_root, roi)
        seg.variant_probability_delta = uncertainty.probability_delta if uncertainty else None
        return seg
    raise ValueError(f"unknown segmentation method {target_config.method!r}")
