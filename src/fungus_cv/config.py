"""Experiment configuration (one ``config.yaml`` per experiment folder)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_UNIT_SECONDS = {
    "": 1, "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]*)")


def parse_duration(value: float | int | str) -> float:
    """Parse ``90``, ``"30s"``, ``"5m"``, ``"1h30m"`` or ``"2 days"`` into seconds."""
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = value.strip().replace(" ", "")
        parts = _DURATION_PART.findall(text)
        if not parts or "".join(n + u for n, u in parts) != text:
            raise ValueError(f"cannot parse duration {value!r} (examples: 30s, 5m, 2h, 1h30m)")
        seconds = 0.0
        for number, unit in parts:
            if unit.lower() not in _UNIT_SECONDS:
                raise ValueError(f"unknown time unit {unit!r} in {value!r}")
            seconds += float(number) * _UNIT_SECONDS[unit.lower()]
    if seconds <= 0:
        raise ValueError("duration must be positive")
    return seconds


# A camera control is "auto" (camera decides), "lock" (let auto settle once when the
# camera opens, then freeze that value for the rest of the run) or a fixed number.
ControlSetting = float | Literal["auto", "lock"]

Backend = Literal["auto", "any", "dshow", "msmf", "avfoundation", "v4l2", "gstreamer"]


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "cam0"
    index: int = 0
    backend: Backend = "auto"
    width: int | None = 1920
    height: int | None = 1080
    fourcc: str | None = "MJPG"
    exposure: ControlSetting = "lock"
    white_balance: ControlSetting = "lock"
    focus: ControlSetting = "lock"
    gain: float | None = None
    warmup_frames: int = Field(10, ge=0)
    settle_seconds: float = Field(1.0, ge=0)

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", v):
            raise ValueError("camera name may only contain letters, digits, '_' and '-'")
        return v

    @field_validator("fourcc")
    @classmethod
    def _fourcc_len(cls, v: str | None) -> str | None:
        if v is not None and len(v) != 4:
            raise ValueError("fourcc must be exactly 4 characters, e.g. MJPG")
        return v


class CaptureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interval: float = 60.0
    start_at: datetime | None = None
    duration: float | None = None
    max_frames: int | None = Field(None, ge=1)
    image_format: Literal["png", "jpg", "tiff"] = "png"
    jpeg_quality: int = Field(95, ge=1, le=100)
    keep_camera_open: bool | None = None
    retries: int = Field(3, ge=0)
    retry_delay: float = Field(2.0, ge=0)
    min_free_disk_mb: int = Field(500, ge=0)
    keep_awake: bool = True

    @field_validator("interval", "duration", mode="before")
    @classmethod
    def _parse_durations(cls, v):
        return None if v is None else parse_duration(v)

    @field_validator("start_at")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            v = v.astimezone()  # naive times are local wall-clock time
        return v.astimezone(timezone.utc)

    @property
    def camera_stays_open(self) -> bool:
        """Short intervals keep the camera open; long ones release it between shots."""
        if self.keep_camera_open is not None:
            return self.keep_camera_open
        return self.interval <= 120


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    cameras: list[CameraConfig] = Field(default_factory=lambda: [CameraConfig()])
    capture: CaptureConfig = Field(default_factory=CaptureConfig)

    @model_validator(mode="after")
    def _unique_cameras(self) -> ExperimentConfig:
        names = [c.name for c in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError(f"camera names must be unique, got {names}")
        if not self.cameras:
            raise ValueError("at least one camera is required")
        return self

    def camera(self, name: str) -> CameraConfig:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise KeyError(f"no camera named {name!r} (have: {[c.name for c in self.cameras]})")


def load_config(path: Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ExperimentConfig.model_validate(data)


def default_config_yaml(name: str) -> str:
    return f"""\
# Fungus-CV experiment configuration
name: {name}
description: ""

cameras:
  - name: cam0            # used in file names; letters, digits, _ and -
    index: 0              # run `fungus cameras` to see what is connected
    backend: auto         # auto | dshow | msmf (Windows) | avfoundation (macOS) | v4l2 (Linux)
    width: 1920
    height: 1080
    fourcc: MJPG          # needed by many USB webcams for full resolution; set to null to skip
    # Camera controls: auto | lock | <number>
    #   lock = let the camera auto-adjust once when it opens, then freeze that value.
    #   Freezing exposure / white balance / focus avoids fake "changes" between frames.
    #   Not every camera/OS supports every control; check `fungus preview` and the
    #   camera_settings column in frames.csv.
    exposure: lock
    white_balance: lock
    focus: lock
    gain: null
    warmup_frames: 10     # frames discarded after opening, before saving
    settle_seconds: 1.0   # time for auto exposure to settle before locking

capture:
  interval: 30s           # e.g. 5s, 10m, 2h, 1h30m
  start_at: null          # optional ISO time; keeps a fixed schedule if you restart
  duration: null          # e.g. 3h, 14d; null = run until stopped
  max_frames: null        # number of capture rounds; null = unlimited
  image_format: png       # png (lossless, recommended) | jpg | tiff
  jpeg_quality: 95
  keep_camera_open: null  # null = automatic (open if interval <= 2 min)
  retries: 3
  retry_delay: 2.0
  min_free_disk_mb: 500
  keep_awake: true        # stop the computer sleeping while capturing
"""
