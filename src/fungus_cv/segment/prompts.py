"""Clicks that tell a promptable model (SAM) what the target is, and crop helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

POSITIVE, NEGATIVE = 1, 0


@dataclass
class FramePrompt:
    """Prompts on one frame, in reference-frame (aligned) pixel coordinates."""

    frame_file: str
    points: list[tuple[float, float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)  # 1 = target, 0 = not target
    box: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1

    def __post_init__(self) -> None:
        if len(self.points) != len(self.labels):
            raise ValueError("points and labels must have the same length")
        if not self.points and self.box is None:
            raise ValueError(f"prompt for {self.frame_file} has no points or box")
        if POSITIVE not in self.labels and self.box is None:
            raise ValueError(f"prompt for {self.frame_file} needs a positive point or a box")


@dataclass
class Prompts:
    frames: list[FramePrompt] = field(default_factory=list)

    def for_file(self, frame_file: str) -> FramePrompt | None:
        return next((p for p in self.frames if p.frame_file == frame_file), None)

    def set(self, prompt: FramePrompt) -> None:
        self.frames = [p for p in self.frames if p.frame_file != prompt.frame_file] + [prompt]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"frames": [asdict(p) for p in self.frames]}, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Prompts:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        frames = []
        for f in data.get("frames", []):
            frames.append(FramePrompt(
                frame_file=f["frame_file"],
                points=[tuple(p) for p in f.get("points", [])],
                labels=[int(v) for v in f.get("labels", [])],
                box=tuple(f["box"]) if f.get("box") else None,
            ))
        return cls(frames)


@dataclass(frozen=True)
class CropWindow:
    """Integer pixel window [x0, x1) x [y0, y1) inside an image."""

    x0: int
    y0: int
    x1: int
    y1: int

    @classmethod
    def full(cls, width: int, height: int) -> CropWindow:
        return cls(0, 0, width, height)

    @classmethod
    def around(cls, polygon, margin: int, width: int, height: int) -> CropWindow:
        pts = np.asarray(polygon, float)
        x0 = max(0, int(np.floor(pts[:, 0].min())) - margin)
        y0 = max(0, int(np.floor(pts[:, 1].min())) - margin)
        x1 = min(width, int(np.ceil(pts[:, 0].max())) + 1 + margin)
        y1 = min(height, int(np.ceil(pts[:, 1].max())) + 1 + margin)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("crop window is empty")
        return cls(x0, y0, x1, y1)

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y0:self.y1, self.x0:self.x1]

    def to_crop(self, points) -> list[tuple[float, float]]:
        return [(x - self.x0, y - self.y0) for x, y in points]

    def box_to_crop(self, box) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = box
        return (x0 - self.x0, y0 - self.y0, x1 - self.x0, y1 - self.y0)

    def paste(self, crop_mask: np.ndarray, height: int, width: int) -> np.ndarray:
        full = np.zeros((height, width), bool)
        full[self.y0:self.y1, self.x0:self.x1] = crop_mask
        return full
