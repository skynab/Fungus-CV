"""Demo experiments: realistic synthetic time-lapses to try every feature without a camera.

A paper towel strip dipped in blue dye, photographed from the front, with two ArUco markers
for scale. The dye rises by capillary wicking, h = k·√(t − t_lag), with a wet front that
fades over a few millimetres, slightly uneven across the strip, a little camera shake, a slow
drift in the room light and sensor noise. Because the true height of every frame is known,
the demo also writes hand measurements (the truth plus a ruler's reading error) and a
validation suite, so `validate`, `validate-suite`, `sensitivity`, `study` and `power` all have
something real to work on.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from fungus_cv.measure.geometry import Annotations
from fungus_cv.quality import mean_brightness, sharpness
from fungus_cv.storage import Experiment, FrameRecord, encode_image, iso_utc, sha256_bytes

W, H = 1280, 960
TOWEL_X = (500, 700)
TOWEL_TOP = 150
BASE_ROW = 850
MARKER_PX = 120
MM_PER_PX = 0.25
MARKER_MM = MARKER_PX * MM_PER_PX
BLUE = np.array((190, 90, 30), np.float32)  # BGR
TOWEL = np.array((238, 238, 236), np.float32)
BACKGROUND = np.array((200, 205, 210), np.float32)
START = datetime(2026, 9, 20, 14, 0, tzinfo=timezone.utc)


@dataclass
class DemoRun:
    """The truth behind a demo experiment, for checking what the analysis finds."""

    root: Path
    k_mm_per_sqrt_min: float
    lag_min: float
    minutes: list[float]
    heights_mm: list[float]
    files: list[str] = field(default_factory=list)


def _marker(marker_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_PX),
                        cv2.COLOR_GRAY2BGR).astype(np.float32)


def render(height_mm: float, rng: np.random.Generator, light: float = 1.0,
           shake_px: float = 1.5, front_mm: float = 3.0) -> np.ndarray:
    """One photo: the towel with dye up to ``height_mm`` (median across the strip)."""
    img = np.empty((H, W, 3), np.float32)
    img[:] = BACKGROUND
    # A little texture everywhere, so image matching has something to hold on to.
    grain = cv2.resize(rng.normal(0, 6, (H // 16, W // 16)).astype(np.float32), (W, H),
                       interpolation=cv2.INTER_CUBIC)
    img += grain[..., None]
    x0, x1 = TOWEL_X
    img[TOWEL_TOP:900, x0:x1] = TOWEL
    rows = np.arange(H, dtype=np.float32)[:, None]
    cols = np.arange(x0, x1, dtype=np.float32)[None, :]
    # The front is slightly uneven across the strip, and fades over front_mm.
    wobble = 1.2 * np.sin(cols / 23.0) + 0.6 * np.sin(cols / 7.0 + height_mm)
    front_row = BASE_ROW - (height_mm + wobble) / MM_PER_PX
    fade = max(front_mm / MM_PER_PX, 1.0)
    wet = np.clip((rows - front_row) / fade + 0.5, 0, 1)  # 1 = fully wet
    wet[BASE_ROW:900] = 1.0  # the reservoir below the waterline
    band = slice(TOWEL_TOP, 900)
    strip = img[band, x0:x1]
    w = wet[band][..., None]
    img[band, x0:x1] = strip * (1 - w) + BLUE * w + (strip - TOWEL) * 0.3
    for marker_id, (mx, my) in {0: (120, 120), 1: (980, 680)}.items():
        q = 30
        img[my - q:my + MARKER_PX + q, mx - q:mx + MARKER_PX + q] = 255
        img[my:my + MARKER_PX, mx:mx + MARKER_PX] = _marker(marker_id)
    img *= light
    img += rng.normal(0, 2.0, img.shape).astype(np.float32)  # sensor noise
    shift = rng.normal(0, shake_px, 2)
    matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    img = cv2.warpAffine(img, matrix, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return np.clip(img, 0, 255).astype(np.uint8)


def dye_experiment(root: Path, frames: int = 30, every_min: float = 2.0,
                   k: float = 12.0, lag_min: float = 0.5, seed: int = 0,
                   name: str | None = None, start: datetime = START) -> DemoRun:
    """Write a complete demo experiment: photos, frames.csv, config, annotations and truth."""
    root = Path(root)
    exp = Experiment.create(root, name or root.name)
    text = exp.config_path.read_text(encoding="utf-8")
    text = text.replace("size_mm: null", f"size_mm: {MARKER_MM}")
    text = text.replace("interval: 30s", f"interval: {int(every_min * 60)}s")
    exp.config_path.write_text(text, encoding="utf-8")
    exp = Experiment(root)
    rng = np.random.default_rng(seed)
    run = DemoRun(root=root, k_mm_per_sqrt_min=k, lag_min=lag_min, minutes=[], heights_mm=[])
    for i in range(frames):
        minute = i * every_min
        height = k * math.sqrt(max(minute - lag_min, 0.0))
        height = min(height, (BASE_ROW - TOWEL_TOP) * MM_PER_PX - 5)  # the strip's top
        light = 1.0 - 0.06 * math.sin(i / max(frames - 1, 1) * math.pi)  # the room dims a bit
        image = render(height, rng, light)
        when = start + timedelta(minutes=minute, seconds=float(rng.uniform(0, 0.4)))
        data = encode_image(image, "png")
        path = exp.save_bytes(data, when, "cam0", "png")
        exp.append_frame(FrameRecord(
            timestamp_utc=iso_utc(when), camera="cam0", status="ok",
            file=exp.relative(path), sha256=sha256_bytes(data), width=W, height=H,
            source="demo", mean_brightness=round(mean_brightness(image), 2),
            sharpness=round(sharpness(image), 2), notes="synthetic demo frame"))
        run.minutes.append(minute)
        run.heights_mm.append(height)
        run.files.append(exp.relative(path))
    Annotations(base=(600.0, BASE_ROW - 0.5), tip=(600.0, TOWEL_TOP - 0.5),
                roi=[(490.0, 140.0), (710.0, 140.0), (710.0, 860.0), (490.0, 860.0)],
                image_size=(W, H)).save(root / "annotations.json")
    _write_truth(run)
    return run


def _write_truth(run: DemoRun, ruler_sd_mm: float = 0.3, every: int = 3) -> None:
    """The true heights, and hand measurements of every ``every``-th frame as a person with a
    ruler would read them (the truth plus a reading error)."""
    rng = np.random.default_rng(12345)
    with open(run.root / "demo_truth.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "minute", "true_height_mm"])
        for file, minute, height in zip(run.files, run.minutes, run.heights_mm):
            writer.writerow([Path(file).name, minute, round(height, 4)])
    with open(run.root / "hand_measurements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "value", "value_unc", "observer", "notes"])
        for i, (file, height) in enumerate(zip(run.files, run.heights_mm)):
            if i % every == 0 and height > 0:
                reading = round((height + rng.normal(0, ruler_sd_mm)) * 2) / 2  # 0.5 mm marks
                writer.writerow([Path(file).name, reading, ruler_sd_mm, "demo",
                                 "truth + ruler reading error"])
    suite = {"name": f"demo-{run.root.name}", "cases": [{
        "name": run.root.name, "experiment": ".", "checks": [{
            "type": "measurements", "hand": "hand_measurements.csv", "metric": "extent_mm",
            "expect": {"bias": {"abs_max": 1.0, "max_drift": 0.2},
                       "rmse": {"max": 2.0}, "within_2u": {"min": 0.7}}}]}]}
    (run.root / "suite.yaml").write_text(
        "# Validation suite for the demo: `fungus validate-suite suite.yaml`\n"
        + yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")


def dye_study(root: Path, replicates: int = 3, frames: int = 20, every_min: float = 2.0,
              control_k: float = 10.0, treated_k: float = 13.0, spread: float = 0.08,
              seed: int = 0) -> tuple[Path, list[DemoRun]]:
    """Replicate demo experiments in two conditions (treated wicks faster) and a study file."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    runs, entries = [], []
    for condition, k in (("control", control_k), ("treated", treated_k)):
        for i in range(1, replicates + 1):
            name = f"{condition}{i}"
            this_k = k * float(rng.normal(1.0, spread))  # replicates differ a little
            run = dye_experiment(root / name, frames=frames, every_min=every_min, k=this_k,
                                 seed=int(rng.integers(1_000_000)), name=name,
                                 start=START + timedelta(days=len(runs)))
            runs.append(run)
            entries.append({"path": name, "condition": condition})
    study = {"name": "demo-study", "metric": "extent_mm", "model": "sqrt_lag",
             "also_fit": ["sqrt", "power"], "reference": "control",
             "params": ["k", "time_to_20"],
             "time_unit": "min", "bootstrap": 200, "experiments": entries}
    path = root / "study.yaml"
    path.write_text("# Demo study: `fungus analyze` each replicate, then "
                    "`fungus study study.yaml`\n" + yaml.safe_dump(study, sort_keys=False),
                    encoding="utf-8")
    return path, runs
