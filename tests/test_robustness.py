"""M4: a tilted camera, lighting changes and jumps must not create fake growth."""

import csv
from datetime import timedelta

import cv2
import numpy as np
import pytest

from fungus_cv.analyze.pipeline import Analyzer, analyze
from fungus_cv.analyze.report import detect_jumps, make_report
from fungus_cv.measure.geometry import Annotations
from fungus_cv.preprocess.markers import detect_markers, scale_from_markers
from fungus_cv.preprocess.rectify import fit_rectification
from fungus_cv.storage import Experiment, FrameRecord, encode_image, iso_utc

from . import synthetic as syn
from .test_pipeline import T0, read_measurements

# A strong tilt: the far edge of the scene is much smaller than the near edge.
SRC = np.float32([[0, 0], [syn.W, 0], [syn.W, syn.H], [0, syn.H]])
DST = np.float32([[180, 120], [syn.W - 140, 40], [syn.W + 60, syn.H + 30], [-80, syn.H - 60]])
TILT = cv2.getPerspectiveTransform(SRC, DST)


def tilt(img: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(img, TILT, (syn.W, syn.H), borderMode=cv2.BORDER_REPLICATE)


def project(h: np.ndarray, pts) -> list[tuple[float, float]]:
    p = cv2.perspectiveTransform(np.float64(pts).reshape(-1, 1, 2), h).reshape(-1, 2)
    return [tuple(map(float, q)) for q in p]


def write_frames(exp: Experiment, frames: list[np.ndarray]) -> None:
    for minute, img in enumerate(frames):
        ts = T0 + timedelta(minutes=minute)
        path = exp.save_bytes(encode_image(img, "png"), ts, "cam0", "png")
        exp.append_frame(FrameRecord(timestamp_utc=iso_utc(ts), camera="cam0", status="ok",
                                     file=exp.relative(path), width=syn.W, height=syn.H))


def set_config(exp: Experiment, *pairs: tuple[str, str]) -> Experiment:
    text = exp.config_path.read_text()
    for old, new in pairs:
        assert old in text, old
        text = text.replace(old, new, 1)
    exp.config_path.write_text(text)
    return Experiment(exp.root)


def truth_annotations(h_to_prepared: np.ndarray, size) -> Annotations:
    base, tip = project(h_to_prepared, [syn.BASE_POINT, syn.TIP_POINT])
    return Annotations(base=base, tip=tip, roi=project(h_to_prepared, syn.ROI),
                       image_size=size)


# --- rectification -------------------------------------------------------------------------


def test_rectification_recovers_true_distances():
    tilted = tilt(syn.scene(300, texture=True))
    markers = detect_markers(tilted)
    naive = scale_from_markers(markers, syn.MARKER_MM)
    assert naive.std_mm_per_px / naive.mm_per_px > 0.1  # the tilt is obvious

    rect = fit_rectification(markers, syn.MARKER_MM, (syn.W, syn.H))
    assert rect.residual_rms_mm < 0.1
    rectified = detect_markers(rect.warp(tilted))
    scale = scale_from_markers(rectified, syn.MARKER_MM)
    assert scale.std_mm_per_px / scale.mm_per_px < 0.005
    true_mm = np.hypot(980 - 120, 680 - 120) * syn.MM_PER_PX
    measured = np.linalg.norm(rectified[0].mean(0) - rectified[1].mean(0)) * scale.mm_per_px
    assert measured == pytest.approx(true_mm, rel=0.003)


def test_rectification_bounded_when_horizon_visible():
    top = syn.scene(0)
    extreme = cv2.getPerspectiveTransform(
        SRC, np.float32([[560, 300], [720, 300], [syn.W + 900, syn.H], [-900, syn.H]]))
    view = cv2.warpPerspective(top, extreme, (syn.W, syn.H), borderMode=cv2.BORDER_REPLICATE)
    markers = detect_markers(view)
    if not markers:
        pytest.skip("markers not detectable at this extreme angle")
    rect = fit_rectification(markers, syn.MARKER_MM, (syn.W, syn.H), max_side_px=4000)
    assert max(rect.size) <= 4000


def test_tilted_camera_measured_correctly_only_with_rectification(experiment):
    heights = [60, 120, 180, 240, 300, 360]
    frames = [tilt(syn.scene(h)) for h in heights]
    frames[3] = syn.bump(frames[3], 8, -5, 0.6)  # knocked mid-run
    write_frames(experiment, frames)
    exp = set_config(experiment, ("size_mm: null", f"size_mm: {syn.MARKER_MM}"))

    # Without rectification: annotations in the tilted image, single average scale.
    truth_annotations(TILT, (syn.W, syn.H)).save(exp.root / "annotations.json")
    analyze(exp)
    plain = sorted(read_measurements(exp), key=lambda r: r["timestamp_utc"])
    plain_err = max(abs(float(r["extent_mm"]) - h * syn.MM_PER_PX)
                    for r, h in zip(plain, heights))

    exp = set_config(exp, ("enabled: false", "enabled: true"))
    rect = Analyzer(exp, with_segmenter=False, require_annotations=False).rectification
    truth_annotations(rect.matrix @ TILT, rect.size).save(exp.root / "annotations.json")
    analyze(exp)
    fixed = sorted(read_measurements(exp), key=lambda r: r["timestamp_utc"])
    errors = [abs(float(r["extent_mm"]) - h * syn.MM_PER_PX) for r, h in zip(fixed, heights)]

    assert max(errors) < 1.0, errors
    assert plain_err > 3 * max(errors)


# --- lighting ------------------------------------------------------------------------------


def dim(img: np.ndarray, factor=(0.5, 0.55, 0.6)) -> np.ndarray:
    return np.clip(img.astype(np.float32) * np.float32(factor), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("method", ["background", "patch"])
def test_lights_dimming_does_not_change_measurements(experiment, method):
    heights = [80, 140, 200, 260, 320, 380]
    frames = [syn.scene(h, texture=True, seed=0) for h in heights]
    for i in (3, 4):
        frames[i] = dim(frames[i])  # lights dimmed, with a slight colour shift
    write_frames(experiment, frames)
    patch = [(800.0, 100.0), (950.0, 100.0), (950.0, 250.0), (800.0, 250.0)]
    Annotations(syn.BASE_POINT, syn.TIP_POINT, syn.ROI, (syn.W, syn.H),
                reference_patch=patch).save(experiment.root / "annotations.json")
    # A colour range that needs normal brightness (value >= 120), like a real tuned range.
    exp = set_config(experiment, ("size_mm: null", f"size_mm: {syn.MARKER_MM}"),
                     ("- lower: [95, 60, 30]", "- lower: [95, 60, 120]"))

    analyze(exp)
    uncorrected = sorted(read_measurements(exp), key=lambda r: r["timestamp_utc"])
    assert float(uncorrected[3]["extent_mm"]) < 1.0  # dye vanished: fake "shrinkage"

    exp = set_config(exp, ("method: none", f"method: {method}"))
    analyze(exp)
    rows = sorted(read_measurements(exp), key=lambda r: r["timestamp_utc"])
    for row, h in zip(rows, heights):
        assert float(row["extent_mm"]) == pytest.approx(h * syn.MM_PER_PX, abs=0.5)
    assert float(rows[3]["light_gain_b"]) == pytest.approx(2.0, rel=0.05)
    assert float(rows[3]["light_gain_r"]) == pytest.approx(1 / 0.6, rel=0.05)
    assert "lighting_changed" in rows[3]["flags"]
    assert "lighting_changed" not in rows[2]["flags"]
    # Dimming is not blur, and marker borders that are always white are not "saturated".
    assert not any("blurry" in r["flags"] or "saturated" in r["flags"] for r in rows)


def test_patch_lighting_requires_patch(experiment):
    write_frames(experiment, [syn.scene(50)])
    Annotations(syn.BASE_POINT, syn.TIP_POINT, syn.ROI, (syn.W, syn.H)).save(
        experiment.root / "annotations.json")
    exp = set_config(experiment, ("method: none", "method: patch"))
    from fungus_cv.analyze.pipeline import AnalysisError

    with pytest.raises(AnalysisError, match="reference patch"):
        Analyzer(exp)


# --- jumps ---------------------------------------------------------------------------------


def test_detect_jumps_flags_spike_not_growth():
    t = np.arange(30, dtype=float)
    y = 10 * np.sqrt(t + 1) + np.random.default_rng(0).normal(0, 0.1, t.size)
    unc = np.full(t.size, 0.1)
    assert not detect_jumps(t, y, unc).any()
    y2 = y.copy()
    y2[14] += 5
    jumps = detect_jumps(t, y2, unc)
    assert np.flatnonzero(jumps).tolist() == [14]


def test_report_marks_jumps_and_writes_frame_flags(experiment):
    heights = [int(40 * np.sqrt(m + 1)) for m in range(12)]
    heights[6] += 120  # e.g. a reflection segmented as dye in one frame
    write_frames(experiment, [syn.scene(h) for h in heights])
    Annotations(syn.BASE_POINT, syn.TIP_POINT, syn.ROI, (syn.W, syn.H)).save(
        experiment.root / "annotations.json")
    exp = set_config(experiment, ("size_mm: null", f"size_mm: {syn.MARKER_MM}"))
    analyze(exp)

    kept = make_report(exp, t0=T0)
    assert kept.jumps == 1 and kept.n_used == 12
    dropped = make_report(exp, t0=T0, exclude_jumps=True)
    assert dropped.n_used == 11
    with open(exp.root / "results" / "report" / "frame_flags.csv", newline="") as f:
        flags = list(csv.DictReader(f))
    assert len(flags) == 12 and sum(row["jump"] == "1" for row in flags) == 1


def test_annotations_patch_roundtrip(tmp_path):
    ann = Annotations((1, 2), (3, 4), [(0, 0), (5, 0), (5, 5)], (10, 10),
                      reference_patch=[(6, 6), (8, 6), (8, 8)])
    ann.save(tmp_path / "a.json")
    loaded = Annotations.load(tmp_path / "a.json")
    assert loaded == ann
    assert loaded.patch_mask((10, 10)).sum() > 0
