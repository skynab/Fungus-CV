"""Segmentation: turn an aligned frame into a mask of the target (dye, moss, ...)."""

from fungus_cv.segment.base import Segmenter, build_segmenter

__all__ = ["Segmenter", "build_segmenter"]
