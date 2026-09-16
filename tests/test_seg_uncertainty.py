"""Segmentation uncertainty: re-segmenting narrower and wider to bound where the edge is."""

import json
import math
from datetime import timedelta

import cv2
import numpy as np
import pytest

from fungus_cv.analyze import pipeline
from fungus_cv.analyze.pipeline import analyze
from fungus_cv.analyze.report import make_report
from fungus_cv.config import ColorTargetConfig, UncertaintyConfig
from fungus_cv.measure.geometry import Annotations, area_uncertainty_mm2, extent_uncertainty_mm
from fungus_cv.segment.base import build_segmenter, segment_with_variants
from fungus_cv.segment.color import ColorThresholdSegmenter, shift_ranges
from fungus_cv.segment.trained import threshold_variants
from fungus_cv.storage import Experiment, FrameRecord, encode_image, iso_utc

from . import synthetic as syn
from .test_pipeline import T0, read_measurements


def soft_scene(height_px: int, ramp_px: int) -> np.ndarray:
    """Dye whose front fades into the towel over ``ramp_px`` rows, like a wet edge."""
    img = syn.scene(height_px)
    x0, x1 = syn.TOWEL_X
    front = syn.BASE_ROW - height_px
    for k in range(ramp_px):
        row = front - 1 - k
        if row < syn.TOWEL_TOP:
            break
        a = 1 - (k + 1) / (ramp_px + 1)  # share of dye colour
        img[row, x0:x1] = np.round(a * np.array(syn.BLUE) + (1 - a) * np.array(syn.TOWEL))
    return img


def build(experiment: Experiment, ramp_px: int, heights=(0, 100, 200, 300),
          uncertainty: bool = True) -> None:
    for i, h in enumerate(heights):
        ts = T0 + timedelta(minutes=i)
        path = experiment.save_bytes(encode_image(soft_scene(h, ramp_px), "png"), ts, "cam0", "png")
        experiment.append_frame(FrameRecord(
            timestamp_utc=iso_utc(ts), camera="cam0", status="ok",
            file=experiment.relative(path), width=syn.W, height=syn.H,
        ))
    Annotations(base=syn.BASE_POINT, tip=syn.TIP_POINT, roi=syn.ROI,
                image_size=(syn.W, syn.H)).save(experiment.root / "annotations.json")
    text = experiment.config_path.read_text().replace("size_mm: null", f"size_mm: {syn.MARKER_MM}")
    if not uncertainty:
        text = text.replace("segmentation: true", "segmentation: false")
    experiment.config_path.write_text(text)


# --- building blocks -----------------------------------------------------------------------


def test_shift_ranges_widens_narrows_and_keeps_open_bounds():
    ranges = [((95, 60, 30), (135, 255, 255))]
    assert shift_ranges(ranges, (4, 20, 20), 1) == [((91, 40, 10), (139, 255, 255))]
    assert shift_ranges(ranges, (4, 20, 20), -1) == [((99, 80, 50), (131, 255, 255))]
    # Red split across the hue seam: the 0 and 179 ends are the seam and stay put.
    red = [((0, 50, 50), (8, 255, 255)), ((172, 50, 50), (179, 255, 255))]
    assert shift_ranges(red, (4, 0, 0), 1) == [((0, 50, 50), (12, 255, 255)),
                                               ((168, 50, 50), (179, 255, 255))]
    # Narrowing past nothing collapses to the middle instead of inverting.
    assert shift_ranges([((100, 60, 30), (104, 70, 255))], (4, 20, 0), -1) == \
        [((102, 65, 30), (102, 65, 255))]


def test_color_variants_nest_inside_each_other():
    rng = np.random.default_rng(1)  # smooth colour blobs, so clean-up doesn't erase them
    image = cv2.resize(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8), (160, 120),
                       interpolation=cv2.INTER_CUBIC)
    seg = ColorThresholdSegmenter.from_config(ColorTargetConfig())
    seg.variant_hsv_delta = (6, 30, 30)
    mask, (narrow, wide) = seg.segment_variants(image)
    assert np.array_equal(mask, seg.segment(image))
    assert not (narrow & ~mask).any() and not (mask & ~wide).any()
    assert narrow.sum() < mask.sum() < wide.sum()


def test_variants_off_gives_none():
    seg = build_segmenter(type("T", (), {"method": "color", "color": ColorTargetConfig()})(),
                          uncertainty=UncertaintyConfig(segmentation=False))
    image = syn.scene(100)
    mask, variants = segment_with_variants(seg, image)
    assert variants == [] and mask.any()


def test_threshold_variants_stay_in_range():
    assert threshold_variants(0.5, 0.1) == pytest.approx([0.6, 0.4])
    assert threshold_variants(0.95, 0.1) == [0.99, pytest.approx(0.85)]
    assert threshold_variants(0.05, 0.1) == [pytest.approx(0.15), 0.01]


def test_uncertainty_formulas_add_in_quadrature():
    base = extent_uncertainty_mm(400, 0.25, 0.0005, 0.5)
    with_seg = extent_uncertainty_mm(400, 0.25, 0.0005, 0.5, segmentation_px=4)
    assert with_seg == pytest.approx(math.hypot(base, 4 * 0.25))
    assert area_uncertainty_mm2(1000, 0.25, 0.001) == pytest.approx(2 * 1000 * 0.25 * 0.001)
    assert area_uncertainty_mm2(1000, 0.25, 0.0, segmentation_px=100) == \
        pytest.approx(100 * 0.0625)


# --- in the pipeline -----------------------------------------------------------------------


def test_soft_front_has_larger_segmentation_uncertainty(experiment, tmp_path):
    build(experiment, ramp_px=30)
    sharp = Experiment.create(tmp_path / "sharp", name="sharp")
    build(sharp, ramp_px=0)
    analyze(Experiment(experiment.root))
    analyze(Experiment(sharp.root))
    soft_rows = read_measurements(experiment)[1:]  # frame 0 has no dye
    sharp_rows = read_measurements(sharp)[1:]
    for soft, hard in zip(soft_rows, sharp_rows):
        # A sharp edge doesn't move when the thresholds do; a fading one does.
        assert float(hard["extent_mm_seg_unc"]) < 0.05
        assert float(soft["extent_mm_seg_unc"]) > 0.5
        assert float(soft["extent_px_seg_unc"]) == \
            pytest.approx(float(soft["extent_mm_seg_unc"]) / syn.MM_PER_PX, abs=0.01)
        assert float(soft["extent_mm_unc"]) >= float(soft["extent_mm_seg_unc"])
        assert float(soft["target_area_mm2_unc"]) > float(hard["target_area_mm2_unc"])
        assert float(soft["coverage_pct_unc"]) > 0


def test_turning_it_off_leaves_columns_empty(experiment):
    build(experiment, ramp_px=30, uncertainty=False)
    analyze(Experiment(experiment.root))
    rows = read_measurements(experiment)
    info = json.loads((experiment.root / "results" / "run_info.json").read_text())
    k, se = info["scale"]["mm_per_px"], info["scale"]["se_mm_per_px"]
    for row in rows:
        assert row["extent_mm_seg_unc"] == "" and row["coverage_pct_unc"] == ""
        expected = extent_uncertainty_mm(float(row["extent_px"]), k, se,
                                         float(row["align_rms_px"] or 0))
        assert float(row["extent_mm_unc"]) == pytest.approx(expected, abs=0.002)
        assert float(row["target_area_mm2_unc"]) == pytest.approx(
            area_uncertainty_mm2(int(row["target_px"]), k, se), abs=0.01)


def test_changing_uncertainty_settings_starts_a_new_run(experiment):
    build(experiment, ramp_px=10)
    first = analyze(Experiment(experiment.root))
    text = experiment.config_path.read_text().replace("hsv_delta: [4, 20, 20]",
                                                      "hsv_delta: [2, 10, 10]")
    experiment.config_path.write_text(text)
    second = analyze(Experiment(experiment.root))
    assert second.settings_hash != first.settings_hash and second.processed == 4


class VariantSequenceSegmenter:
    """A sequence segmenter that yields ``(index, mask, variants)`` like SAM 2 does."""

    name = "fake-seq-variants"

    def __init__(self):
        self.inner = ColorThresholdSegmenter.from_config(ColorTargetConfig())
        self.inner.variant_hsv_delta = (4, 20, 20)

    def describe(self):
        return {"method": self.name}

    def segment_sequence(self, frame_files, load, needed):
        for i in sorted(needed):
            yield (i, *self.inner.segment_variants(load(i)))


def test_sequence_segmenter_variants_reach_the_measurements(experiment, monkeypatch):
    build(experiment, ramp_px=30)
    monkeypatch.setattr(pipeline, "build_segmenter", lambda *a, **k: VariantSequenceSegmenter())
    summary = analyze(Experiment(experiment.root))
    assert summary.processed == 4 and summary.failed == 0
    rows = read_measurements(experiment)[1:]
    assert all(float(r["extent_mm_seg_unc"]) > 0.5 for r in rows)


def test_report_weights_area_fits_by_their_uncertainty(experiment):
    build(experiment, ramp_px=30, heights=(40, 100, 160, 220, 280, 340))
    analyze(Experiment(experiment.root))
    exp = Experiment(experiment.root)
    make_report(exp, metric="target_area_mm2", t0=T0)
    fits = json.loads((exp.root / "results" / "report" / "fits.json").read_text())
    assert fits["weighted_by_uncertainty"] is True
