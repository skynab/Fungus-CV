"""Experiment configuration (one ``config.yaml`` per experiment folder)."""

from __future__ import annotations

import json
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


class HsvRange(BaseModel):
    """Inclusive OpenCV HSV bounds: H 0-179, S 0-255, V 0-255."""

    model_config = ConfigDict(extra="forbid")

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    @model_validator(mode="after")
    def _check(self) -> HsvRange:
        for lo, hi, top in zip(self.lower, self.upper, (179, 255, 255)):
            if not (0 <= lo <= hi <= top):
                raise ValueError(f"invalid HSV range {self.lower} - {self.upper}")
        return self


class ColorClass(BaseModel):
    """A named colour for whole-field change: every plot gets the share of it over time."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_]+$")  # becomes the column class_<name>_pct
    hsv_ranges: list[HsvRange] = Field(min_length=1)


class ColorTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Default: blue dye. Use `fungus pick-color` to measure ranges from your own images.
    hsv_ranges: list[HsvRange] = Field(
        default_factory=lambda: [HsvRange(lower=(95, 60, 30), upper=(135, 255, 255))]
    )
    open_px: int = Field(3, ge=0)  # removes specks smaller than this
    close_px: int = Field(7, ge=0)  # fills small holes and gaps
    min_blob_area_px: int = Field(50, ge=0)


class Sam2TargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "facebook/sam2.1-hiera-small"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    prompts_file: str = "prompts.json"
    # SAM works at ~1024 px. Cropping to the annotated region first gives it more pixels
    # on the object, so mask edges are more precise.
    crop_to_roi: bool = True
    crop_margin_px: int = Field(32, ge=0)
    mask_threshold: float = 0.0  # on SAM's logits; >0 = tighter masks, <0 = looser
    min_blob_area_px: int = Field(0, ge=0)


class ModelTargetConfig(BaseModel):
    """A segmentation model trained with `fungus train`."""

    model_config = ConfigDict(extra="forbid")

    path: str = ""  # model folder; absolute, or relative to the experiment or current folder
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    crop_to_roi: bool = True
    crop_margin_px: int = Field(32, ge=0)
    threshold: float | None = Field(None, gt=0, lt=1)  # None = value tuned during training
    class_name: str | None = None  # multi-class model: which class; None = its first class
    tile_px: int = Field(512, ge=64)
    overlap_px: int = Field(64, ge=0)
    min_blob_area_px: int = Field(0, ge=0)


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["color", "sam2", "model"] = "color"
    color: ColorTargetConfig = Field(default_factory=ColorTargetConfig)
    sam2: Sam2TargetConfig = Field(default_factory=Sam2TargetConfig)
    model: ModelTargetConfig = Field(default_factory=ModelTargetConfig)

    def selected(self) -> dict:
        """Only the active method's settings; used for the results settings hash."""
        return {"method": self.method,
                self.method: getattr(self, self.method).model_dump(mode="json")}


class MarkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dictionary: str = "DICT_4X4_50"
    # Edge length of the black marker square as printed, in mm. Measure the print with a
    # ruler. Required for mm results; without it results are in pixels only.
    size_mm: float | None = Field(None, gt=0)


class RectifyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Warp frames to a top-down view of the marker plane (needs markers.size_mm).
    enabled: bool = False
    max_side_px: int = Field(6000, ge=256)


class LightingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # none | patch (neutral card marked with `fungus annotate`) | background (outside the region)
    method: Literal["none", "patch", "background"] = "none"
    flag_change: float = Field(0.3, gt=0)  # flag frames whose gains differ from 1 by more


class ReferenceConfig(BaseModel):
    """Optional per-frame segmentation of the reference object (e.g. the plant stem)."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["none", "color", "sam2", "model"] = "none"
    color: ColorTargetConfig = Field(default_factory=ColorTargetConfig)
    sam2: Sam2TargetConfig = Field(
        default_factory=lambda: Sam2TargetConfig(prompts_file="reference_prompts.json"))
    model: ModelTargetConfig = Field(default_factory=ModelTargetConfig)

    def selected(self) -> dict:
        if self.method == "none":
            return {"method": "none"}
        return {"method": self.method,
                self.method: getattr(self, self.method).model_dump(mode="json")}


class MeasureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # axis: straight line base -> tip.  path: along a curved centerline (e.g. a stem).
    mode: Literal["axis", "path"] = "axis"
    # For mode path: the line clicked in `fungus annotate`, or the centerline of the
    # reference object segmented in every frame (follows a stem that bends or grows).
    path_source: Literal["annotation", "reference"] = "annotation"
    corridor_px: float | None = Field(None, gt=0)  # ignore target further from the path
    smooth_px: float | None = Field(None, gt=0)  # centerline smoothing; None = automatic


class UncertaintyConfig(BaseModel):
    """How much segmentation settings are nudged to estimate where the edge could be.

    Each frame is also segmented with a narrower and a wider setting; the spread of the
    resulting measurements is treated as a rectangular distribution (GUM type B).
    """

    model_config = ConfigDict(extra="forbid")

    segmentation: bool = True
    hsv_delta: tuple[int, int, int] = (4, 20, 20)  # color: H, S, V bounds moved in and out
    probability_delta: float = Field(0.1, gt=0, lt=0.5)  # model: threshold +/- this
    logit_delta: float = Field(1.0, gt=0)  # sam2: mask_threshold +/- this (logits)


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: str | None = None  # default: the camera with the most frames
    align: Literal["markers_or_ecc", "markers", "ecc", "none"] = "markers_or_ecc"
    markers: MarkerConfig = Field(default_factory=MarkerConfig)
    rectify: RectifyConfig = Field(default_factory=RectifyConfig)
    lighting: LightingConfig = Field(default_factory=LightingConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    measure: MeasureConfig = Field(default_factory=MeasureConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    # Named colours (e.g. healthy, yellowing, brown): the share of each plot in each, per frame.
    color_classes: list[ColorClass] = Field(default_factory=list)
    # Front position across the object's width: 50 = median front, 100 = highest point.
    front_percentile: float = Field(50.0, ge=0, le=100)
    save_masks: bool = True
    save_overlays: bool = True

    @model_validator(mode="after")
    def _unique_classes(self) -> AnalysisConfig:
        names = [c.name for c in self.color_classes]
        if len(names) != len(set(names)):
            raise ValueError(f"colour class names must be unique: {names}")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    cameras: list[CameraConfig] = Field(default_factory=lambda: [CameraConfig()])
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

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

analysis:
  camera: null            # which camera's frames to analyze; null = the one with most frames
  align: markers_or_ecc   # line frames up with the first: markers_or_ecc | markers | ecc | none
  markers:
    dictionary: DICT_4X4_50   # must match the printed sheet (`fungus markers`)
    size_mm: null         # black square edge as printed, measured with a ruler, e.g. 30.0
  rectify:                # correct a camera that isn't square-on to the measured plane
    enabled: false        # true: warp to a top-down view using the markers (run annotate after)
    max_side_px: 6000
  lighting:               # compensate brightness / colour changes between frames
    method: none          # none | patch (neutral card, marked in annotate) | background
    flag_change: 0.3      # flag frames needing more than a 30% correction
  target:
    method: color         # color | sam2 | model (sam2 and model need: pip install -e ".[sam]")
    color:
      hsv_ranges:         # OpenCV HSV (H 0-179). Measure yours with `fungus pick-color`
        - lower: [95, 60, 30]
          upper: [135, 255, 255]
      open_px: 3          # remove specks
      close_px: 7         # fill small holes
      min_blob_area_px: 50
    sam2:
      model: facebook/sam2.1-hiera-small  # -tiny | -small | -base-plus | -large (slowest, best)
      device: auto        # auto | cuda | mps | cpu
      prompts_file: prompts.json  # created by `fungus prompt`
      crop_to_roi: true   # give SAM more pixels on the object (more precise edges)
      crop_margin_px: 32
      mask_threshold: 0.0 # >0 tighter masks, <0 looser
      min_blob_area_px: 0
    model:
      path: ""            # folder written by `fungus train`, e.g. ../../models/moss-bark-v1
      device: auto
      crop_to_roi: true
      crop_margin_px: 32
      threshold: null     # null = threshold tuned on validation data during training
      class_name: null    # a model with several classes (e.g. stem, moss): which one
      tile_px: 512        # large frames are processed in overlapping tiles
      overlap_px: 64
      min_blob_area_px: 0
  reference:              # optional: segment the reference object (e.g. stem) in every frame
    method: none          # none | color | sam2 | model (same settings blocks as target)
    color:
      hsv_ranges:
        - lower: [5, 80, 40]
          upper: [25, 255, 200]
      open_px: 3
      close_px: 7
      min_blob_area_px: 50
    sam2:
      model: facebook/sam2.1-hiera-small
      device: auto
      prompts_file: reference_prompts.json  # `fungus prompt --reference`
      crop_to_roi: true
      crop_margin_px: 32
      mask_threshold: 0.0
      min_blob_area_px: 0
    model:
      path: ""
      device: auto
      crop_to_roi: true
      crop_margin_px: 32
      threshold: null
      class_name: null    # e.g. stem, from the same model as the target's moss
      tile_px: 512
      overlap_px: 64
      min_blob_area_px: 0
  measure:
    mode: axis            # axis: straight base->tip | path: along a curved object (stem)
    path_source: annotation  # annotation: line clicked in annotate | reference: centerline
                             # of the reference object in each frame (needs reference.method)
    corridor_px: null     # ignore target further than this from the path (null = whole region)
    smooth_px: null       # centerline smoothing length; null = about twice the stem width
  uncertainty:            # segmentation uncertainty: re-segment each frame narrower and wider
    segmentation: true    # adds *_seg_unc columns and includes them in extent_mm_unc
    hsv_delta: [4, 20, 20]    # color: HSV bounds moved in and out by this much
    probability_delta: 0.1    # model: probability threshold +/- this
    logit_delta: 1.0          # sam2: mask_threshold +/- this
  color_classes: []       # named colours for whole-field change, e.g. healthy / brown;
                          # `fungus pick-color EXP --class brown` adds one
  front_percentile: 50    # front across the width: 50 = median, 100 = highest point
  save_masks: true
  save_overlays: true
"""


def replace_hsv_ranges_in_yaml(text: str, ranges, section: str = "target") -> str:
    """Swap ``analysis.<section>.color.hsv_ranges`` in config text, keeping all other lines.

    ``section`` is ``target`` or ``reference``. Configs without that structure fall back to
    the first ``hsv_ranges:`` block.
    """
    lines = text.splitlines(keepends=True)
    try:
        i = _find_key_line(lines, f"analysis.{section}.color.hsv_ranges")
    except KeyError:
        i = next((k for k, line in enumerate(lines) if re.match(r"^\s*hsv_ranges:", line)), None)
        if i is None:
            raise ValueError("no hsv_ranges: entry found in config") from None
    indent = len(lines[i]) - len(lines[i].lstrip(" "))
    end = i + 1
    while end < len(lines):
        stripped = lines[end].strip()
        current = len(lines[end]) - len(lines[end].lstrip())
        if stripped and current <= indent:
            break
        end += 1
    pad = " " * (indent + 2)
    block = [f"{' ' * indent}hsv_ranges:\n"]
    for lower, upper in ranges:
        block.append(f"{pad}- lower: [{', '.join(map(str, lower))}]\n")
        block.append(f"{pad}  upper: [{', '.join(map(str, upper))}]\n")
    new_text = "".join(lines[:i] + block + lines[end:])
    yaml.safe_load(new_text)  # never write a config that no longer parses
    return new_text


def set_color_class_in_yaml(text: str, name: str, ranges) -> str:
    """Add colour class ``name`` to ``analysis.color_classes`` (or replace its ranges),
    keeping every other line of the config as it is."""
    ColorClass(name=name, hsv_ranges=[HsvRange(lower=lo, upper=hi) for lo, hi in ranges])
    current = (yaml.safe_load(text) or {}).get("analysis", {}).get("color_classes") or []
    entries = [dict(c) for c in current if c.get("name") != name]
    entries.append({"name": name, "hsv_ranges": [
        {"lower": list(lo), "upper": list(hi)} for lo, hi in ranges]})
    order = [c.get("name") for c in current]
    if name in order:  # keep its place: the first matching class wins, so order matters
        entries.insert(order.index(name), entries.pop())

    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    try:
        i = _find_key_line(lines, "analysis.color_classes")
        indent = len(lines[i]) - len(lines[i].lstrip(" "))
        end = i + 1
        while end < len(lines):
            stripped = lines[end].strip()
            current_indent = len(lines[end]) - len(lines[end].lstrip(" "))
            if stripped and current_indent <= indent and not stripped.startswith("- "):
                break
            end += 1
        while end > i + 1 and not lines[end - 1].strip():  # keep blank lines that follow
            end -= 1
    except KeyError:
        try:
            i = end = _find_key_line(lines, "analysis.front_percentile")
            indent = len(lines[i]) - len(lines[i].lstrip(" "))
        except KeyError:
            a = _find_key_line(lines, "analysis")
            i = end = a + 1
            nxt = next((ln for ln in lines[a + 1:] if ln.strip()), "  x")
            indent = max(len(nxt) - len(nxt.lstrip(" ")), 2)
    pad = " " * indent
    block = [f"{pad}color_classes:\n"]
    for entry in entries:
        block.append(f"{pad}  - name: {entry['name']}\n")
        block.append(f"{pad}    hsv_ranges:\n")
        for r in entry["hsv_ranges"]:
            block.append(f"{pad}      - lower: [{', '.join(map(str, r['lower']))}]\n")
            block.append(f"{pad}        upper: [{', '.join(map(str, r['upper']))}]\n")
    new_text = "".join(lines[:i] + block + lines[end:])
    yaml.safe_load(new_text)  # never write a config that no longer parses
    return new_text


def _find_key_line(lines: list[str], dotted_key: str) -> int:
    """Index of the line holding ``dotted_key``, following YAML indentation."""
    keys = dotted_key.split(".")
    start, parent_indent = 0, -1
    for depth, key in enumerate(keys):
        child_indent = None
        found = None
        for i in range(start, len(lines)):
            raw = lines[i].rstrip("\n")
            stripped = raw.lstrip(" ")
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(raw) - len(stripped)
            if indent <= parent_indent:
                break
            if child_indent is None:
                child_indent = indent
            if indent == child_indent and re.match(rf"{re.escape(key)}\s*:", stripped):
                found = i
                break
        if found is None:
            raise KeyError(dotted_key)
        if depth == len(keys) - 1:
            return found
        start, parent_indent = found + 1, child_indent
    raise KeyError(dotted_key)


def _yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./+-]+", text) and \
            yaml.safe_load(text) == text:  # plain word that YAML reads back as the same string
        return text
    return json.dumps(text)


def set_yaml_value(text: str, dotted_key: str, value) -> str:
    """Set one scalar, e.g. ``capture.interval``, keeping every other line and comment.

    Raises ``KeyError`` if the key is not in the text (add it by editing the file instead).
    """
    lines = text.splitlines(keepends=True)
    found = _find_key_line(lines, dotted_key)
    raw = lines[found].rstrip("\n")
    match = re.match(r"^(\s*[^:#]+:[ \t]*)(.*?)([ \t]+#.*)?$", raw)
    comment = match.group(3) or ""
    prefix = match.group(1) if match.group(1).endswith((" ", "\t")) else match.group(1) + " "
    new_line = prefix + _yaml_scalar(value)
    if comment:
        new_line = new_line.ljust(len(match.group(1)) + len(match.group(2))) + comment
    lines[found] = new_line + ("\n" if lines[found].endswith("\n") else "")
    new_text = "".join(lines)
    yaml.safe_load(new_text)  # never produce a file that no longer parses
    return new_text
