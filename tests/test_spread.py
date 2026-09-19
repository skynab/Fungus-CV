"""Arrival-time maps and spread speed by direction, on patches with a known shape."""

import csv
import json
import math
from datetime import timedelta

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from fungus_cv.analyze.pipeline import MEASUREMENT_FIELDS
from fungus_cv.analyze.spread import arrival_times, spread
from fungus_cv.cli import app
from fungus_cv.measure.geometry import Annotations, Plot
from fungus_cv.storage import Experiment, iso_utc

from .test_pipeline import T0

W, H = 400, 300
MM_PER_PX = 0.5


def ellipse_experiment(exp: Experiment, n=10, speed_x=6.0, speed_y=2.0, noise_frame=None):
    """A patch that grows 3x faster left-right than up-down (px per minute)."""
    results = exp.root / "results"
    (results / "masks" / "h1").mkdir(parents=True)
    (results / "runs").mkdir()
    (results / "runs" / "h1.json").write_text(json.dumps({"scale": {"mm_per_px": MM_PER_PX}}))
    bed = [(0, 0), (W - 1, 0), (W - 1, H - 1), (0, H - 1)]
    Annotations(base=None, tip=None, roi=bed, image_size=(W, H),
                plots=[Plot("bed", bed)]).save(exp.root / "annotations.json")
    rows = []
    for i in range(n):
        mask = np.zeros((H, W), np.uint8)
        cv2.ellipse(mask, (200, 150), (int(10 + speed_x * i), int(8 + speed_y * i)), 0, 0, 360,
                    255, -1)
        if i == noise_frame:
            mask[10:20, 10:20] = 255  # a one-frame speck far away
        path = results / "masks" / "h1" / f"f{i:02d}.png"
        cv2.imwrite(str(path), mask)
        rows.append({"timestamp_utc": iso_utc(T0 + timedelta(minutes=i)),
                     "frame_file": f"frames/f{i:02d}.png", "camera": "cam0", "plot": "bed",
                     "settings_hash": "h1", "coverage_pct": float(mask.mean() / 2.55),
                     "flags": "", "mask_file": path.relative_to(exp.root).as_posix()})
    with open(results / "measurements.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MEASUREMENT_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return Experiment(exp.root)


def by_angle(result):
    return {round(s.angle_deg): s for s in result.sectors}


def test_arrival_times_need_persistence():
    a = np.zeros((1, 4), bool)
    frames = [a.copy() for _ in range(5)]
    for f in frames[1:]:
        f[0, 0] = True  # covered from t=1 on
    frames[2][0, 1] = True  # a one-frame blip
    for f in frames[3:]:
        f[0, 2] = True
    arrival = arrival_times(frames, [0, 1, 2, 3, 4], persistence=2)
    assert arrival[0, 0] == 1 and arrival[0, 2] == 3
    assert math.isnan(arrival[0, 1]) and math.isnan(arrival[0, 3])
    once = arrival_times(frames, [0, 1, 2, 3, 4], persistence=1)
    assert once[0, 1] == 2


def test_anisotropic_spread_is_measured_by_direction(experiment):
    exp = ellipse_experiment(experiment, noise_frame=4)
    result = spread(exp, time_unit="min")
    assert result.unit == "mm" and result.time_unit == "min"
    assert result.origin == pytest.approx((200, 150), abs=1)
    sectors = by_angle(result)
    # 6 px/min left-right and 2 px/min up-down, at 0.5 mm/px.
    for angle in (0, 180):
        assert sectors[angle].speed == pytest.approx(3.0, rel=0.1)
        assert sectors[angle].r2 > 0.98
    for angle in (90, 270):
        assert sectors[angle].speed == pytest.approx(1.0, rel=0.15)
    assert result.anisotropy == pytest.approx(3, rel=0.2)
    assert result.fastest.angle_deg in (0, 180)
    # The one-frame speck is never "reached"; the map starts at t = 0 in the middle.
    assert math.isnan(result.arrival[15, 15]) and result.arrival[150, 200] == 0
    for name in ("arrival_time.npy", "spread_by_direction.csv", "spread.json",
                 "spread_map.png"):
        assert (exp.root / "results" / "spread" / "bed" / name).exists(), name
    saved = np.load(exp.root / "results" / "spread" / "bed" / "arrival_time.npy")
    assert saved.shape == (H, W)


def test_even_spread_has_no_preferred_direction(experiment):
    exp = ellipse_experiment(experiment, speed_x=4, speed_y=4)
    result = spread(exp, time_unit="min", sectors=4)
    assert len(result.sectors) == 4
    assert result.anisotropy == pytest.approx(1.0, abs=0.15)
    assert result.area_speed == pytest.approx(4 * MM_PER_PX, rel=0.15)


def test_spread_needs_masks_and_the_cli(experiment):
    exp = ellipse_experiment(experiment, n=6)
    runner = CliRunner()
    out = runner.invoke(app, ["spread", str(exp.root), "--time-unit", "min"])
    assert out.exit_code == 0, out.output
    assert "fastest toward" in out.output and "fastest/slowest" in out.output
    for path in (exp.root / "results" / "masks" / "h1").glob("*.png"):
        path.unlink()
    missing = runner.invoke(app, ["spread", str(exp.root)])
    assert missing.exit_code == 1 and "saved masks" in missing.output
