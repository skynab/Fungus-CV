"""Colour classes: the share of a plot in each named colour, for whole-field change."""

import math

import cv2
import numpy as np
import pytest
import yaml

from fungus_cv.analyze.pipeline import analyze
from fungus_cv.analyze.report import make_report
from fungus_cv.config import ExperimentConfig, set_color_class_in_yaml
from fungus_cv.measure.color_classes import class_shares
from fungus_cv.measure.geometry import Annotations

from . import synthetic as syn
from .test_pipeline import T0, read_measurements
from .test_robustness import dim, set_config, write_frames

GREEN = (60, 170, 70)   # BGR: healthy
YELLOW = (40, 200, 220)
BROWN = (30, 55, 120)
HEALTHY = [((40, 80, 60), (85, 255, 255))]
YELLOWING = [((20, 80, 80), (34, 255, 255))]
BROWNED = [((0, 80, 30), (19, 255, 200))]
CLASSES = [("healthy", HEALTHY), ("yellowing", YELLOWING), ("brown", BROWNED)]


def field(brown_rows: int, yellow_rows: int = 0, texture: bool = True) -> np.ndarray:
    """The towel area as a green field, browning from the bottom up."""
    img = syn.scene(0, texture=False)
    x0, x1 = syn.TOWEL_X
    img[syn.TOWEL_TOP:900, x0:x1] = GREEN
    if yellow_rows:
        img[syn.BASE_ROW - brown_rows - yellow_rows:syn.BASE_ROW - brown_rows, x0:x1] = YELLOW
    if brown_rows:
        img[syn.BASE_ROW - brown_rows:syn.BASE_ROW, x0:x1] = BROWN
    if texture:
        noise = np.random.default_rng(brown_rows).normal(0, 4, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def region_mask() -> np.ndarray:
    mask = np.zeros((syn.H, syn.W), bool)
    x0, x1 = syn.TOWEL_X
    mask[syn.TOWEL_TOP:syn.BASE_ROW, x0:x1] = True
    return mask


def test_shares_match_painted_areas():
    region = region_mask()
    rows = syn.BASE_ROW - syn.TOWEL_TOP
    out = class_shares(field(210, 70), region, CLASSES, hsv_delta=(4, 20, 20))
    assert out["class_brown_pct"] == pytest.approx(100 * 210 / rows, abs=0.2)
    assert out["class_yellowing_pct"] == pytest.approx(100 * 70 / rows, abs=0.2)
    assert out["class_healthy_pct"] == pytest.approx(100 * (rows - 280) / rows, abs=0.2)
    # Clean colours well inside their ranges: moving the bounds changes nothing.
    assert out["class_brown_pct_unc"] < 0.2


def test_first_matching_class_wins_and_shares_never_exceed_100():
    region = region_mask()
    overlapping = [("brown", BROWNED), ("anything", [((0, 0, 0), (179, 255, 255))])]
    out = class_shares(field(210), region, overlapping)
    assert out["class_brown_pct"] + out["class_anything_pct"] == pytest.approx(100)
    assert math.isnan(out["class_brown_pct_unc"])  # no delta, no uncertainty


def test_boundary_colours_carry_uncertainty():
    """A colour right at a class boundary is a judgement call; the uncertainty says so."""
    img = np.zeros((20, 20, 3), np.uint8)
    img[:] = cv2.cvtColor(np.uint8([[[19, 200, 150]]]), cv2.COLOR_HSV2BGR)[0, 0]
    out = class_shares(img, np.ones((20, 20), bool), [("brown", [((0, 80, 30), (18, 255, 200))])],
                       hsv_delta=(4, 20, 20))
    assert out["class_brown_pct"] == 0
    assert out["class_brown_pct_unc"] == pytest.approx(100 / (2 * math.sqrt(3)), rel=0.01)


def test_black_border_is_not_part_of_the_field():
    img = field(0, texture=False)
    region = region_mask()
    img[syn.TOWEL_TOP:syn.TOWEL_TOP + 100] = 0  # left empty by alignment
    assert class_shares(img, region, CLASSES)["class_healthy_pct"] == pytest.approx(100)


def test_yaml_upsert_keeps_comments_and_order(experiment):
    text = experiment.config_path.read_text()
    text = set_color_class_in_yaml(text, "healthy", HEALTHY)
    text = set_color_class_in_yaml(text, "brown", BROWNED)
    text = set_color_class_in_yaml(text, "healthy", [((35, 60, 50), (90, 255, 255))])
    assert "# front across the width" in text  # the rest of the file is untouched
    cfg = ExperimentConfig.model_validate(yaml.safe_load(text))
    assert [c.name for c in cfg.analysis.color_classes] == ["healthy", "brown"]
    assert cfg.analysis.color_classes[0].hsv_ranges[0].lower == (35, 60, 50)
    with pytest.raises(ValueError):
        set_color_class_in_yaml(text, "not a name", HEALTHY)


def test_duplicate_class_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        ExperimentConfig.model_validate({"name": "x", "analysis": {"color_classes": [
            {"name": "a", "hsv_ranges": [{"lower": [0, 0, 0], "upper": [9, 9, 9]}]},
            {"name": "a", "hsv_ranges": [{"lower": [0, 0, 0], "upper": [9, 9, 9]}]}]}})


def test_pipeline_measures_class_shares_through_dimming(experiment):
    """Browning at a steady rate, with the lights dimmed in two frames: the lighting-corrected
    shares follow the truth, and the report fits the rate."""
    rows_total = syn.BASE_ROW - syn.TOWEL_TOP
    brown = [0, 70, 140, 210, 280, 350, 420, 490]
    frames = [field(b) for b in brown]
    for i in (3, 4):
        frames[i] = dim(frames[i])
    write_frames(experiment, frames)
    patch = [(800.0, 100.0), (950.0, 100.0), (950.0, 250.0), (800.0, 250.0)]
    Annotations(syn.BASE_POINT, syn.TIP_POINT,
                [(500.0, 150.0), (700.0, 150.0), (700.0, 850.0), (500.0, 850.0)],
                (syn.W, syn.H), reference_patch=patch).save(experiment.root / "annotations.json")
    text = experiment.config_path.read_text()
    for name, ranges in CLASSES:
        text = set_color_class_in_yaml(text, name, ranges)
    experiment.config_path.write_text(text)
    exp = set_config(experiment, ("size_mm: null", f"size_mm: {syn.MARKER_MM}"),
                     ("method: none", "method: patch"))

    analyze(exp)
    rows = sorted(read_measurements(exp), key=lambda r: r["timestamp_utc"])
    for row, b in zip(rows, brown):
        assert float(row["class_brown_pct"]) == pytest.approx(100 * b / rows_total, abs=1.5)
        assert float(row["class_healthy_pct"]) == pytest.approx(100 - 100 * b / rows_total,
                                                                abs=1.5)
        assert row["class_brown_pct_unc"] != ""

    result = make_report(exp, metric="class_brown_pct", t0=T0, models=["linear"])
    fit = next(f for f in result.fits if f.model == "linear")
    assert fit.value("b") == pytest.approx(100 * 70 / rows_total, rel=0.05)  # % per min
