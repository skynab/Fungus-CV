"""Segment frames with a model trained by `fungus train`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fungus_cv.segment.prompts import CropWindow
from fungus_cv.segment.sam2 import _drop_small_blobs


def resolve_model_path(path: str, experiment_root: Path | None) -> Path:
    if not path:
        raise ValueError("analysis.target.model.path is empty; point it at a `fungus train` folder")
    p = Path(path)
    candidates = [p] if p.is_absolute() else [Path(experiment_root or ".") / p, Path.cwd() / p]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    raise FileNotFoundError(f"model folder {path!r} not found (looked in "
                            f"{', '.join(str(c) for c in candidates)})")


def threshold_variants(threshold: float, delta: float) -> list[float]:
    """Stricter and looser probability thresholds, kept inside (0.01, 0.99)."""
    return [min(threshold + delta, 0.99), max(threshold - delta, 0.01)]


@dataclass
class TrainedModelSegmenter:
    model_dir: Path
    device: str = "auto"
    roi: list | None = None
    crop_margin_px: int = 32
    threshold: float | None = None
    tile_px: int = 512
    overlap_px: int = 64
    min_blob_area_px: int = 0
    variant_probability_delta: float | None = None  # for segmentation uncertainty
    name: str = field(default="model", init=False)

    @classmethod
    def from_config(cls, cfg, experiment_root, roi) -> TrainedModelSegmenter:
        return cls(
            model_dir=resolve_model_path(cfg.path, experiment_root),
            device=cfg.device,
            roi=roi if cfg.crop_to_roi else None,
            crop_margin_px=cfg.crop_margin_px,
            threshold=cfg.threshold,
            tile_px=cfg.tile_px,
            overlap_px=cfg.overlap_px,
            min_blob_area_px=cfg.min_blob_area_px,
        )

    def _loaded(self):
        from fungus_cv.learn.infer import load_trained

        return load_trained(self.model_dir, self.device)

    def segment(self, image: np.ndarray) -> np.ndarray:
        return self._segment(image, with_variants=False)[0]

    def segment_variants(self, image: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """The mask, plus masks at the threshold -/+ ``variant_probability_delta``."""
        return self._segment(image, with_variants=self.variant_probability_delta is not None)

    def _segment(self, image: np.ndarray, with_variants: bool):
        from fungus_cv.learn.infer import predict_probabilities

        loaded = self._loaded()
        h, w = image.shape[:2]
        window = (CropWindow.around(self.roi, self.crop_margin_px, w, h) if self.roi
                  else CropWindow.full(w, h))
        prob = predict_probabilities(loaded.net, window.crop(image), loaded.device,
                                     self.tile_px, self.overlap_px)
        t = self.threshold if self.threshold is not None else loaded.threshold
        thresholds = [t]
        if with_variants:
            thresholds += threshold_variants(t, self.variant_probability_delta)
        masks = [_drop_small_blobs(window.paste(prob >= x, h, w), self.min_blob_area_px)
                 for x in thresholds]
        return masks[0], masks[1:]

    def describe(self) -> dict:
        loaded = self._loaded()
        return {
            "method": self.name,
            "model_dir": str(self.model_dir),
            "weights_sha256": loaded.weights_sha256,
            "dataset_fingerprint": loaded.card["dataset"]["fingerprint"],
            "threshold": self.threshold if self.threshold is not None else loaded.threshold,
            "crop": {"roi_margin_px": self.crop_margin_px} if self.roi else None,
            "tile_px": self.tile_px,
            "overlap_px": self.overlap_px,
            "min_blob_area_px": self.min_blob_area_px,
        }
