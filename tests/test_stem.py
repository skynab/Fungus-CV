"""M5: measuring moss along a curved plant stem, and agreement with hand measurements."""

import csv
from datetime import timedelta

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from fungus_cv.analyze.pipeline import AnalysisError, Analyzer, analyze
from fungus_cv.analyze.validate import validate, write_template
from fungus_cv.cli import app
from fungus_cv.measure.centerline import centerline_from_mask, thin
from fungus_cv.measure.geometry import Annotations
from fungus_cv.measure.path import Polyline, measure_along_path
from fungus_cv.storage import Experiment, FrameRecord, encode_image, iso_utc

from . import synthetic as syn
from . import synthetic_stem as ss
from .test_pipeline import T0, read_measurements


def roi_mask():
    m = np.zeros((ss.H, ss.W), np.uint8)
    cv2.fillPoly(m, [np.int32(ss.ROI)], 1)
    return m.astype(bool)


# --- geometry ------------------------------------------------------------------------------


def test_thin_rectangle_gives_centered_line():
    mask = np.zeros((60, 200), bool)
    mask[20:41, 10:190] = True
    skel = thin(mask)
    ys, xs = np.nonzero(skel)
    assert set(np.unique(ys)) <= {29, 30, 31}
    assert xs.max() - xs.min() > 150


def test_smoothing_removes_staircase_length_error():
    t = np.arange(0, 300)
    staircase = np.c_[t, np.floor(t * 0.4)]  # a 0.4-slope line traced on the pixel grid
    true_len = 299 * np.hypot(1, 0.4)
    raw = Polyline(staircase)
    assert raw.length > true_len * 1.03  # pixel tracing overestimates
    assert Polyline(staircase).smoothed(15).length == pytest.approx(true_len, rel=0.005)


def test_projection_on_curve():
    # Quarter circle of radius 100 around (0, 0) from (100, 0) to (0, 100).
    angles = np.linspace(0, np.pi / 2, 400)
    path = Polyline(np.c_[100 * np.cos(angles), 100 * np.sin(angles)])
    assert path.length == pytest.approx(50 * np.pi, rel=1e-3)
    a = np.pi / 6
    along, across, _ = path.project(np.array([110 * np.cos(a)]), np.array([110 * np.sin(a)]))
    assert along[0] == pytest.approx(100 * a, abs=0.2)
    assert abs(across[0]) == pytest.approx(10, abs=0.2)


@pytest.mark.parametrize("sway", [0.0, 60.0])
@pytest.mark.parametrize("moss_len", [0.0, 150.0, 420.0, 700.0])
def test_curved_stem_centerline_and_moss_extent(sway, moss_len):
    stem = ss.Stem(sway)
    img = ss.scene(stem, moss_len)
    moss = ss.color_mask(img, ss.MOSS)
    obj = ss.color_mask(img, ss.STEM) | moss
    path = centerline_from_mask(obj, stem.base)
    assert path.length == pytest.approx(stem.length, rel=0.01)

    m = measure_along_path(moss, path, roi_mask(), reference_mask=obj)
    assert m.extent_px == pytest.approx(moss_len, abs=1.5)
    assert m.covered_length_px == pytest.approx(moss_len, abs=2.0)
    if moss_len:
        expected = moss_len * 2 * ss.MOSS_HALF / (moss_len * 2 * ss.MOSS_HALF +
                                                  (stem.length - moss_len) * 2 * ss.STEM_HALF)
        assert m.reference_covered_fraction == pytest.approx(expected, abs=0.02)


def test_corridor_ignores_target_away_from_path():
    stem = ss.Stem()
    img = ss.scene(stem, 200)
    moss = ss.color_mask(img, ss.MOSS)
    stray = moss.copy()
    stray[200:230, 820:850] = True  # moss on something else, far up and to the side
    path = centerline_from_mask(ss.color_mask(img, ss.STEM) | moss, stem.base)
    loose = measure_along_path(stray, path, roi_mask(), front_percentile=100)
    tight = measure_along_path(stray, path, roi_mask(), front_percentile=100, corridor_px=25)
    assert loose.extent_max_px > 400
    assert tight.extent_max_px == pytest.approx(200, abs=1.5)


def test_annotations_path_roundtrip(tmp_path):
    ann = Annotations((0, 0), (10, 10), [(0, 0), (20, 0), (20, 20)], (30, 30),
                      path=[(0, 0), (8, 2), (10, 10)])
    ann.save(tmp_path / "a.json")
    loaded = Annotations.load(tmp_path / "a.json")
    assert loaded == ann
    assert loaded.polyline().length == pytest.approx(np.hypot(8, 2) + np.hypot(2, 8))


# --- pipeline ------------------------------------------------------------------------------


def hsv_range(bgr, hue=8, sv=40):
    h, s, v = (int(c) for c in cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0])
    lo = [max(0, h - hue), max(0, s - sv), max(0, v - sv)]
    hi = [min(179, h + hue), min(255, s + sv), min(255, v + sv)]
    return lo, hi


def stem_experiment(experiment, moss_lengths, sways, source="reference"):
    stems = [ss.Stem(sw) for sw in sways]
    for minute, (stem, length) in enumerate(zip(stems, moss_lengths)):
        ts = T0 + timedelta(hours=minute * 6)
        path = experiment.save_bytes(encode_image(ss.scene(stem, length), "png"), ts, "cam0",
                                     "png")
        experiment.append_frame(FrameRecord(timestamp_utc=iso_utc(ts), camera="cam0",
                                            status="ok", file=experiment.relative(path)))
    first = stems[0]
    clicked = [tuple(first.at(s)[0][0]) for s in np.linspace(0, first.length, 12)]
    Annotations(first.base, clicked[-1], ss.ROI, (ss.W, ss.H), path=clicked).save(
        experiment.root / "annotations.json")

    moss_lo, moss_hi = hsv_range(ss.MOSS)
    stem_lo, stem_hi = hsv_range(ss.STEM)
    text = experiment.config_path.read_text()
    for old, new in [
        ("size_mm: null", f"size_mm: {syn.MARKER_MM}"),
        ("- lower: [95, 60, 30]\n          upper: [135, 255, 255]",
         f"- lower: {moss_lo}\n          upper: {moss_hi}"),
        ("- lower: [5, 80, 40]\n          upper: [25, 255, 200]",
         f"- lower: {stem_lo}\n          upper: {stem_hi}"),
        ("mode: axis", "mode: path"),
        ("path_source: annotation", f"path_source: {source}"),
    ]:
        assert old in text, old
        text = text.replace(old, new, 1)
    if source == "reference":
        text = text.replace("    method: none          # none | color | sam2 | model (same",
                            "    method: color         # none | color | sam2 | model (same", 1)
    experiment.config_path.write_text(text)
    return Experiment(experiment.root), stems


def test_moss_on_swaying_stem_measured_along_its_centerline(experiment):
    lengths = [0, 90, 210, 360, 520, 690]
    exp, stems = stem_experiment(experiment, lengths, sways=[0, 25, 50, 20, -10, 40])
    summary = analyze(exp)
    assert summary.processed == len(lengths) and summary.failed == 0
    rows = sorted(read_measurements(exp), key=lambda r: r["timestamp_utc"])
    for row, length, stem in zip(rows, lengths, stems):
        assert float(row["extent_mm"]) == pytest.approx(length * syn.MM_PER_PX, abs=0.5)
        assert float(row["axis_length_mm"]) == pytest.approx(stem.length * syn.MM_PER_PX,
                                                             rel=0.01)
        assert float(row["covered_length_pct"]) == pytest.approx(100 * length / stem.length,
                                                                 abs=1.0)
        assert row["reference_covered_pct"] != ""
        assert "centerline_failed" not in row["flags"]
    reference_masks = list((exp.root / "results" / "masks").glob("*/reference/*.png"))
    assert len(reference_masks) == len(lengths)


def test_static_clicked_path(experiment):
    lengths = [100, 300, 500]
    exp, stems = stem_experiment(experiment, lengths, sways=[0, 0, 0], source="annotation")
    analyze(exp)
    rows = sorted(read_measurements(exp), key=lambda r: r["timestamp_utc"])
    for row, length in zip(rows, lengths):
        # 12 clicks approximate the curve with straight chords: a little short on bends.
        assert float(row["extent_mm"]) == pytest.approx(length * syn.MM_PER_PX, abs=2.0)


def test_reference_path_requires_reference_segmentation(experiment):
    exp, _ = stem_experiment(experiment, [100], sways=[0], source="annotation")
    text = exp.config_path.read_text().replace("path_source: annotation",
                                               "path_source: reference", 1)
    exp.config_path.write_text(text)
    with pytest.raises(AnalysisError, match="reference.method"):
        Analyzer(Experiment(exp.root))


# --- validation against hand measurements --------------------------------------------------


def test_validate_against_hand_measurements(experiment, tmp_path):
    lengths = [0, 90, 210, 360, 520, 690]
    exp, _ = stem_experiment(experiment, lengths, sways=[0, 25, 50, 20, -10, 40])
    analyze(exp)

    template = tmp_path / "hand.csv"
    assert write_template(exp, template, count=6) == 6
    rows = list(csv.DictReader(open(template)))
    rng = np.random.default_rng(3)
    # A person measuring with a ruler: truth plus ~0.3 mm reading error, 0.2 mm long on average.
    for row, length in zip(rows, lengths):
        row["value"] = f"{length * syn.MM_PER_PX + 0.2 + rng.normal(0, 0.3):.2f}"
        row["observer"] = "A"
    with open(template, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    a = validate(exp, template)
    assert a.n == 6
    assert a.bias == pytest.approx(-0.2, abs=0.4)
    assert a.limits_of_agreement[0] < a.bias < a.limits_of_agreement[1]
    assert a.pearson_r > 0.999
    assert all(f.exists() for f in a.files)

    out = CliRunner().invoke(app, ["validate", str(exp.root), str(template)])
    assert out.exit_code == 0, out.output
    assert "limits of agreement" in out.output
