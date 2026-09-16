"""Compensate lighting changes between frames (a lamp switched on, clouds, a colour cast).

Each colour channel is multiplied by a gain so that a reference region looks the same as in
the reference frame. A change in exposure or light intensity scales camera pixel values (for
gamma-encoded images too, as long as nothing is clipped), so per-channel gains undo overall
brightness changes and colour casts. They cannot undo shadows or glare that affect only
part of the scene.

Reference regions:
- ``patch``: a neutral card or other surface that never changes, marked with `fungus annotate`
  (most reliable).
- ``background``: everything outside the measured region (works if the background is static).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SATURATED = 250


def channel_medians(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    pixels = image[region]
    if len(pixels) < 50:
        raise ValueError("lighting reference region has too few pixels")
    return np.median(pixels.reshape(-1, image.shape[2]), axis=0).astype(np.float64)


@dataclass
class LightingResult:
    gains: tuple[float, float, float]  # per BGR channel, applied to this frame
    saturated_fraction: float  # of reference-region pixels clipped in any channel

    @property
    def max_change(self) -> float:
        return float(max(abs(g - 1.0) for g in self.gains))


class LightingNormalizer:
    def __init__(self, reference: np.ndarray, region: np.ndarray, max_gain: float = 3.0):
        self.region = region
        self.max_gain = max_gain
        self.ref = channel_medians(reference, region)
        if np.any(self.ref < 5):
            raise ValueError("lighting reference region is almost black in the reference frame")

    def apply(self, image: np.ndarray, valid: np.ndarray | None = None
              ) -> tuple[np.ndarray, LightingResult]:
        region = self.region if valid is None else self.region & valid
        current = channel_medians(image, region)
        gains = np.clip(self.ref / np.maximum(current, 1.0), 1 / self.max_gain, self.max_gain)
        saturated = float((image[region] >= SATURATED).any(axis=-1).mean())
        corrected = np.clip(image.astype(np.float32) * gains.astype(np.float32), 0, 255)
        return corrected.astype(np.uint8), LightingResult(
            tuple(round(float(g), 4) for g in gains), round(saturated, 4))
