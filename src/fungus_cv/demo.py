"""Demo experiments: realistic synthetic time-lapses to try every feature without a camera.

Two demos, one per way of telling the program what to measure:

- ``dye_experiment`` (Set Up Measurement, colour thresholds). A paper towel strip dipped in
  blue dye, photographed from the front, with two ArUco markers for scale and a neutral grey
  card for the lighting correction. The dye rises by capillary wicking, h = k·√(t − t_lag),
  with a wet front that fades over a few millimetres, slightly uneven across the strip, a
  little camera shake, a slow drift in the room light and sensor noise.
- ``colony_experiment`` (SAM Prompts, SAM 2). A mould colony spreading over an agar plate,
  photographed from above every few hours for six days: a fuzzy off-white rim around an
  older, sporulating grey-green centre, growing faster in one direction. The colony has no
  colour of its own to threshold on, which is what SAM 2 is for; the experiment is set to
  SAM 2 and left for you to click on.

Because the truth of every frame is known, both demos also write hand measurements (the truth
plus a ruler's reading error), a validation suite and a room-conditions log, so `validate`,
`validate-suite`, `sensitivity`, `spread`, `covariates`, `study` and `power` all have
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
# A neutral grey card propped in shot, as a careful experimenter would: it is what the
# `patch` lighting correction and the Neutral patch tool measure the room light with.
CARD = np.array((160, 160, 160), np.float32)
CARD_BOX = (170, 560, 430, 790)  # x0, y0, x1, y1
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
    cx0, cy0, cx1, cy1 = CARD_BOX
    img[cy0:cy1, cx0:cx1] = CARD
    cv2.rectangle(img, (cx0, cy0), (cx1 - 1, cy1 - 1), (120.0, 120.0, 120.0), 2)
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
    _write_truth(root, run.files, run.minutes, run.heights_mm, "true_height_mm", "extent_mm")
    _write_conditions(exp, start, minutes=frames * every_min, every_min=1.0, seed=seed)
    return run


def _write_truth(root: Path, files: list[str], minutes: list[float], values: list[float],
                 column: str, metric: str, ruler_sd_mm: float = 0.3, every: int = 3) -> None:
    """The true values, hand measurements of every ``every``-th frame as a person with a
    ruler would read them (the truth plus a reading error), and a validation suite."""
    rng = np.random.default_rng(12345)
    with open(root / "demo_truth.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "minute", column])
        for file, minute, value in zip(files, minutes, values):
            writer.writerow([Path(file).name, minute, round(value, 4)])
    with open(root / "hand_measurements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "value", "value_unc", "observer", "notes"])
        for i, (file, value) in enumerate(zip(files, values)):
            if i % every == 0 and value > 0:
                reading = round((value + rng.normal(0, ruler_sd_mm)) * 2) / 2  # 0.5 mm marks
                writer.writerow([Path(file).name, reading, ruler_sd_mm, "demo",
                                 "truth + ruler reading error"])
    suite = {"name": f"demo-{root.name}", "cases": [{
        "name": root.name, "experiment": ".", "checks": [{
            "type": "measurements", "hand": "hand_measurements.csv", "metric": metric,
            "expect": {"bias": {"abs_max": 1.0, "max_drift": 0.2},
                       "rmse": {"max": 2.0}, "within_2u": {"min": 0.7}}}]}]}
    (root / "suite.yaml").write_text(
        "# Validation suite for the demo: `fungus validate-suite suite.yaml`\n"
        + yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")


def _write_conditions(exp: Experiment, start: datetime, minutes: float, every_min: float,
                      seed: int, day_cycle: bool = False) -> None:
    """A thermometer/hygrometer logging next to the camera: ``room_logger.csv`` in the
    logger's own format, imported as `fungus covariates import` would, so the report's
    Conditions tab has something to show."""
    from fungus_cv.analyze import covariates

    rng = np.random.default_rng(seed + 777)
    path = exp.root / "room_logger.csv"
    steps = int(minutes / every_min) + 2
    drift = np.cumsum(rng.normal(0, 0.03, steps))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Temperature (°C)", "Humidity (%RH)"])
        for i in range(steps):
            minute = (i - 1) * every_min  # from just before the first photo
            cycle = math.sin(2 * math.pi * (minute / 1440 - 0.3)) if day_cycle else 0.0
            temp = 21.5 + 1.5 * cycle + drift[i] + rng.normal(0, 0.05)
            humidity = 48 - 4 * cycle - 2 * drift[i] + rng.normal(0, 0.3)
            when = start + timedelta(minutes=minute)
            writer.writerow([when.isoformat(timespec="seconds"), round(temp, 2),
                             round(humidity, 1)])
    covariates.import_log(exp, path)


# --- the SAM 2 demo: a mould colony on an agar plate ----------------------------------------

PLATE_CENTER = (600, 480)
PLATE_MM = 90.0
COLONY_MM_PER_PX = 0.125  # a closer view than the towel: the markers are 15 mm here
INOCULUM = (PLATE_CENTER[0] - 60.0, float(PLATE_CENTER[1]))  # off centre: room to go right
AGAR = np.array((150, 196, 218), np.float32)  # pale amber (BGR)
TABLE = np.array((70, 72, 80), np.float32)
MYCELIUM = np.array((232, 236, 238), np.float32)  # fuzzy off-white rim
SPORES = np.array((118, 146, 120), np.float32)  # older, sporulating grey-green centre


def _colony(radius_mm: float, fast_deg: float, anisotropy: float, t: float) -> np.ndarray:
    """Soft (0..1) colony of mean radius ``radius_mm`` around the inoculum: lobed, and reaching
    further toward ``fast_deg`` (0 = right, 90 = up) by ``anisotropy`` (fastest/slowest)."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xs - INOCULUM[0], INOCULUM[1] - ys  # y up
    theta = np.arctan2(dy, dx)
    r = np.hypot(dx, dy) * COLONY_MM_PER_PX
    a = (anisotropy - 1) / (anisotropy + 1)
    lobes = (0.05 * np.sin(5 * theta + 0.7) + 0.035 * np.sin(9 * theta + 2.1)
             + 0.02 * np.sin(17 * theta + 0.3 + 0.1 * t))  # the edge keeps its shape
    reach = radius_mm * (1 + a * np.cos(theta - math.radians(fast_deg))) * (1 + lobes)
    return np.clip((reach - r) / 0.6 + 0.5, 0, 1)  # the hyphae fade out over ~0.6 mm


def render_colony(radius_mm: float, rng: np.random.Generator, light: float = 1.0,
                  shake_px: float = 1.0, fast_deg: float = 0.0, anisotropy: float = 1.4,
                  t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """One photo from above: the plate with a colony of mean radius ``radius_mm``. Returns the
    photo and the colony's true mask (before the camera shake)."""
    img = np.empty((H, W, 3), np.float32)
    img[:] = TABLE
    grain = cv2.resize(rng.normal(0, 7, (H // 12, W // 12)).astype(np.float32), (W, H),
                       interpolation=cv2.INTER_CUBIC)
    img += grain[..., None]
    plate_px = PLATE_MM / 2 / COLONY_MM_PER_PX
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xs - PLATE_CENTER[0], ys - PLATE_CENTER[1])
    inside = d < plate_px
    shade = 1 - 0.08 * (d / plate_px) ** 2  # the agar darkens a little toward the wall
    img[inside] = AGAR * shade[inside][:, None] + grain[inside][:, None] * 0.4
    wall = (d >= plate_px - 6) & (d < plate_px + 4)
    img[wall] = img[wall] * 0.7 + 255 * 0.3  # the dish wall catches the light
    soft = _colony(radius_mm, fast_deg, anisotropy, t)
    core = _colony(max(radius_mm - 9.0, 0.0), fast_deg, anisotropy, t) * 0.85
    fuzz = cv2.resize(rng.normal(0, 1, (H // 3, W // 3)).astype(np.float32), (W, H),
                      interpolation=cv2.INTER_LINEAR)
    colony = MYCELIUM * (1 - core[..., None]) + SPORES * core[..., None]
    colony = colony + (fuzz * 10)[..., None]  # a cottony texture
    img = img * (1 - soft[..., None]) + colony * soft[..., None]
    img[640:900, 60:250] = CARD
    for marker_id, (mx, my) in {0: (80, 80), 1: (1080, 760)}.items():
        q = 20
        img[my - q:my + MARKER_PX + q, mx - q:mx + MARKER_PX + q] = 255
        img[my:my + MARKER_PX, mx:mx + MARKER_PX] = _marker(marker_id)
    img *= light
    img += rng.normal(0, 2.0, img.shape).astype(np.float32)
    shift = rng.normal(0, shake_px, 2)
    matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    img = cv2.warpAffine(img, matrix, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return np.clip(img, 0, 255).astype(np.uint8), soft >= 0.5


@dataclass
class ColonyRun:
    """The truth behind the SAM 2 demo."""

    root: Path
    speed_mm_per_h: float  # growth of the mean radius
    lag_h: float
    anisotropy: float  # fastest / slowest direction
    fast_deg: float
    hours: list[float]
    radius_mm: list[float]  # equivalent radius, from the true area
    area_mm2: list[float]
    files: list[str] = field(default_factory=list)


def colony_experiment(root: Path, frames: int = 25, every_h: float = 6.0,
                      speed: float = 0.22, lag_h: float = 12.0, r0: float = 2.5,
                      anisotropy: float = 1.4, fast_deg: float = 0.0, seed: int = 0,
                      name: str | None = None, start: datetime = START) -> ColonyRun:
    """Write the SAM 2 demo: photos, frames.csv, a config set to SAM 2, annotations (scale,
    base, tip and the plate as the region) and truth. It has no prompts: clicking the colony
    on the SAM Prompts page is the part of the workflow it is there to try."""
    from fungus_cv.config import set_yaml_value

    root = Path(root)
    exp = Experiment.create(root, name or root.name)
    text = exp.config_path.read_text(encoding="utf-8")
    text = text.replace("size_mm: null", f"size_mm: {MARKER_PX * COLONY_MM_PER_PX}")
    text = text.replace("interval: 30s", f"interval: {int(every_h * 3600)}s")
    text = set_yaml_value(text, "analysis.target.method", "sam2")
    exp.config_path.write_text(text, encoding="utf-8")
    exp = Experiment(root)
    rng = np.random.default_rng(seed)
    run = ColonyRun(root=root, speed_mm_per_h=speed, lag_h=lag_h, anisotropy=anisotropy,
                    fast_deg=fast_deg, hours=[], radius_mm=[], area_mm2=[])
    for i in range(frames):
        hour = i * every_h
        radius = r0 + speed * max(hour - lag_h, 0.0)  # colonies grow at a steady speed
        light = 1.0 - 0.04 * math.sin(2 * math.pi * hour / 24)  # daylight in the room
        image, truth = render_colony(radius, rng, light, fast_deg=fast_deg,
                                     anisotropy=anisotropy, t=hour / 24)
        when = start + timedelta(hours=hour, seconds=float(rng.uniform(0, 2)))
        data = encode_image(image, "png")
        path = exp.save_bytes(data, when, "cam0", "png")
        exp.append_frame(FrameRecord(
            timestamp_utc=iso_utc(when), camera="cam0", status="ok",
            file=exp.relative(path), sha256=sha256_bytes(data), width=W, height=H,
            source="demo", mean_brightness=round(mean_brightness(image), 2),
            sharpness=round(sharpness(image), 2), notes="synthetic SAM 2 demo frame"))
        area = float(truth.sum()) * COLONY_MM_PER_PX ** 2
        run.hours.append(hour)
        run.area_mm2.append(area)
        run.radius_mm.append(math.sqrt(area / math.pi))
        run.files.append(exp.relative(path))
    inner = PLATE_MM / 2 / COLONY_MM_PER_PX - 12  # just inside the dish wall
    roi = [(PLATE_CENTER[0] + inner * math.cos(a), PLATE_CENTER[1] + inner * math.sin(a))
           for a in np.linspace(0, 2 * math.pi, 32, endpoint=False)]
    # Base = the inoculation point, tip = the dish wall in the fast direction: extent_mm is
    # then how far the fastest edge has reached.
    Annotations(base=INOCULUM, tip=(PLATE_CENTER[0] + inner, INOCULUM[1]), roi=roi,
                image_size=(W, H)).save(root / "annotations.json")
    _write_truth(root, run.files, [h * 60 for h in run.hours], run.radius_mm,
                 "true_equivalent_radius_mm", "equivalent_radius_mm")
    _write_conditions(exp, start, minutes=frames * every_h * 60, every_min=30.0, seed=seed,
                      day_cycle=True)
    return run


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
