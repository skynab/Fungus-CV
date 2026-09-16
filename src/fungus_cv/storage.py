"""Experiment folder layout, image writing and the ``frames.csv`` metadata log."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from fungus_cv.config import ExperimentConfig, default_config_yaml, load_config

CONFIG_NAME = "config.yaml"
FRAMES_DIR = "frames"
FRAMES_CSV = "frames.csv"
LOG_NAME = "capture.log"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(ts: datetime) -> str:
    """``2026-09-20T14:05:00.123Z``"""
    return ts.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def frame_filename(ts: datetime, camera: str, ext: str) -> str:
    """Sortable, Windows-safe name, e.g. ``2026-09-20T14-05-00.123Z_cam0.png``."""
    stamp = iso_utc(ts).replace(":", "-")
    return f"{stamp}_{camera}.{ext.lstrip('.').lower()}"


@dataclass
class FrameRecord:
    timestamp_utc: str
    camera: str
    status: str  # "ok" or "failed"
    file: str = ""  # relative to the experiment folder
    sha256: str = ""
    width: int | None = None
    height: int | None = None
    source: str = "capture"  # capture | import:<how the timestamp was found>
    scheduled_utc: str = ""
    lag_s: float | None = None
    mean_brightness: float | None = None
    sharpness: float | None = None
    camera_settings: dict = field(default_factory=dict)
    notes: str = ""


FRAME_FIELDS = [f.name for f in fields(FrameRecord)]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_image(image: np.ndarray, fmt: str, jpeg_quality: int = 95) -> bytes:
    ext = {"jpg": ".jpg", "png": ".png", "tiff": ".tiff"}[fmt]
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if fmt == "jpg" else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise OSError(f"failed to encode image as {fmt}")
    return buf.tobytes()


def write_atomic(path: Path, data: bytes) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written image."""
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Experiment:
    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"{self.config_path} not found; create an experiment with `fungus init {root}`"
            )
        self.config: ExperimentConfig = load_config(self.config_path)
        self.frames_dir.mkdir(exist_ok=True)

    @classmethod
    def create(cls, root: Path, name: str | None = None) -> Experiment:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        config_path = root / CONFIG_NAME
        if config_path.exists():
            raise FileExistsError(f"{config_path} already exists")
        config_path.write_text(default_config_yaml(name or root.name), encoding="utf-8")
        return cls(root)

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_NAME

    @property
    def frames_dir(self) -> Path:
        return self.root / FRAMES_DIR

    @property
    def frames_csv(self) -> Path:
        return self.root / FRAMES_CSV

    @property
    def log_path(self) -> Path:
        return self.root / LOG_NAME

    def free_disk_mb(self) -> float:
        return shutil.disk_usage(self.root).free / 1e6

    def unique_frame_path(self, ts: datetime, camera: str, ext: str) -> Path:
        path = self.frames_dir / frame_filename(ts, camera, ext)
        n = 1
        while path.exists():
            path = self.frames_dir / f"{path.stem.split('~')[0]}~{n}{path.suffix}"
            n += 1
        return path

    def save_bytes(self, data: bytes, ts: datetime, camera: str, ext: str) -> Path:
        path = self.unique_frame_path(ts, camera, ext)
        write_atomic(path, data)
        return path

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def append_frame(self, record: FrameRecord) -> None:
        new_file = not self.frames_csv.exists()
        row = asdict(record)
        row["camera_settings"] = json.dumps(record.camera_settings, sort_keys=True)
        with open(self.frames_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def read_frames(self) -> list[dict]:
        if not self.frames_csv.exists():
            return []
        with open(self.frames_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["camera_settings"] = json.loads(row["camera_settings"] or "{}")
        return rows
