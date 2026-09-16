import math

import cv2
import numpy as np
import pytest
import yaml

from fungus_cv.analyze.fit import fit_all, fit_model
from fungus_cv.config import ColorTargetConfig, replace_hsv_ranges_in_yaml
from fungus_cv.measure.geometry import Annotations, extent_uncertainty_mm, measure_extent
from fungus_cv.preprocess.align import align_frame
from fungus_cv.preprocess.markers import detect_markers, marker_sheet, scale_from_markers
from fungus_cv.segment.color import ColorThresholdSegmenter, hsv_range_from_samples
from fungus_cv.storage import Experiment

from . import synthetic as syn


def annotations() -> Annotations:
    return Annotations(base=syn.BASE_POINT, tip=syn.TIP_POINT, roi=syn.ROI,
                       image_size=(syn.W, syn.H))


def blue_segmenter() -> ColorThresholdSegmenter:
    return ColorThresholdSegmenter.from_config(ColorTargetConfig())


# --- markers & scale -------------------------------------------------------------------


def test_scale_from_markers_is_accurate():
    found = detect_markers(syn.scene(100))
    assert sorted(found) == [0, 1]
    scale = scale_from_markers(found, syn.MARKER_MM)
    assert scale.mm_per_px == pytest.approx(syn.MM_PER_PX, rel=0.003)
    assert scale.n_edges == 8


def test_marker_sheet_prints_at_true_size():
    dpi = 150
    sheet = marker_sheet([0, 1, 2], size_mm=30, dpi=dpi, page="a4")
    found = detect_markers(cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR))
    assert sorted(found) == [0, 1, 2]
    px_per_mm = 1 / scale_from_markers(found, 30).mm_per_px
    assert px_per_mm == pytest.approx(dpi / 25.4, rel=0.01)


def test_marker_sheet_rejects_too_many():
    with pytest.raises(ValueError):
        marker_sheet(list(range(40)), size_mm=60, page="letter")


# --- extent geometry ---------------------------------------------------------------------


@pytest.mark.parametrize("h", [0, 1, 57, 300, 699])
def test_extent_exact_on_perfect_mask(h):
    mask = np.zeros((syn.H, syn.W), bool)
    x0, x1 = syn.TOWEL_X
    mask[syn.BASE_ROW:900, x0:x1] = True  # reservoir must not count
    if h:
        mask[syn.BASE_ROW - h:syn.BASE_ROW, x0:x1] = True
    m = measure_extent(mask, annotations())
    assert m.extent_px == pytest.approx(h, abs=1e-9)
    assert m.extent_max_px == pytest.approx(h, abs=1e-9)
    assert m.front_width_px == x1 - x0


def test_extent_median_ignores_narrow_spike_but_max_sees_it():
    mask = np.zeros((syn.H, syn.W), bool)
    x0, x1 = syn.TOWEL_X
    mask[syn.BASE_ROW - 200:syn.BASE_ROW, x0:x1] = True
    mask[syn.BASE_ROW - 260:syn.BASE_ROW, 590:600] = True
    m = measure_extent(mask, annotations(), front_percentile=50)
    assert m.extent_px == pytest.approx(200)
    assert m.extent_max_px == pytest.approx(260)


def test_extent_along_diagonal_axis():
    # A band perpendicular to a 45-degree axis: distance along the axis is known.
    size = 400
    yy, xx = np.mgrid[0:size, 0:size]
    along = ((xx - 50) + (yy - 50)) / math.sqrt(2)
    mask = (along >= 0) & (along <= 100)
    ann = Annotations(base=(50, 50), tip=(350, 350),
                      roi=[(0, 0), (399, 0), (399, 399), (0, 399)], image_size=(size, size))
    m = measure_extent(mask, ann)
    assert m.extent_px == pytest.approx(100, abs=1.0)


def test_empty_mask():
    m = measure_extent(np.zeros((syn.H, syn.W), bool), annotations())
    assert (m.target_px, m.extent_px, m.coverage_fraction) == (0, 0.0, 0.0)


def test_uncertainty_combines_terms():
    u = extent_uncertainty_mm(extent_px=400, mm_per_px=0.25, scale_se_mm_per_px=0.0005,
                              align_rms_px=0.4)
    expected = math.sqrt((400 * 0.0005) ** 2 + (0.25 / math.sqrt(12)) ** 2 + (0.4 * 0.25) ** 2)
    assert u == pytest.approx(expected)


def test_annotations_roundtrip(tmp_path):
    ann = annotations()
    ann.save(tmp_path / "a.json")
    assert Annotations.load(tmp_path / "a.json") == ann


def test_annotations_validate():
    with pytest.raises(ValueError):
        Annotations(base=(1, 1), tip=(1, 1), roi=syn.ROI, image_size=(10, 10))


# --- segmentation ------------------------------------------------------------------------


@pytest.mark.parametrize("h", [40, 350])
def test_color_segmentation_measures_synthetic_dye(h):
    mask = blue_segmenter().segment(syn.scene(h))
    assert measure_extent(mask, annotations()).extent_px == pytest.approx(h, abs=0.5)


def test_hsv_range_from_blue_samples_covers_them():
    bgr = np.array([[[190, 90, 30], [200, 100, 40], [180, 80, 25]]], np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    ranges = hsv_range_from_samples(hsv)
    assert len(ranges) == 1
    lo, hi = ranges[0]
    assert all(lo[i] <= px[i] <= hi[i] for px in hsv for i in range(3))


def test_hsv_range_wraps_for_red():
    hsv = np.array([[2, 200, 200], [177, 210, 190], [0, 190, 210], [179, 200, 200]])
    ranges = hsv_range_from_samples(hsv, hue_margin=3)
    assert len(ranges) == 2
    seg = ColorThresholdSegmenter(ranges, open_px=0, close_px=0, min_blob_area_px=0)
    red = cv2.cvtColor(np.array([[[1, 200, 200], [178, 200, 200]]], np.uint8), cv2.COLOR_HSV2BGR)
    green = cv2.cvtColor(np.array([[[60, 200, 200]]], np.uint8), cv2.COLOR_HSV2BGR)
    assert seg.segment(red).all()
    assert not seg.segment(green).any()


def test_replace_hsv_ranges_keeps_comments(experiment):
    text = experiment.config_path.read_text()
    new = replace_hsv_ranges_in_yaml(text, [((1, 2, 3), (4, 5, 6)), ((170, 2, 3), (179, 5, 6))])
    data = yaml.safe_load(new)
    assert data["analysis"]["target"]["color"]["hsv_ranges"] == [
        {"lower": [1, 2, 3], "upper": [4, 5, 6]}, {"lower": [170, 2, 3], "upper": [179, 5, 6]},
    ]
    assert data["analysis"]["target"]["color"]["open_px"] == 3
    assert "# remove specks" in new
    Experiment(experiment.root)  # still a valid config


# --- alignment ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["markers", "ecc"])
def test_alignment_undoes_camera_bump(mode):
    h = 250
    reference = syn.scene(0, markers=mode == "markers", texture=True)
    bumped = syn.bump(syn.scene(h, markers=mode == "markers", texture=True), 14, -9, 1.2)
    ref_markers = detect_markers(reference) if mode == "markers" else {}

    exclude = annotations().roi_mask(reference.shape)
    aligned, alignment = align_frame(bumped, reference, ref_markers, mode=mode, exclude=exclude)
    assert alignment.method == mode
    if mode == "markers":
        assert alignment.rms_px < 0.5
    else:
        assert alignment.ecc > 0.95
    extent = measure_extent(blue_segmenter().segment(aligned), annotations()).extent_px
    assert extent == pytest.approx(h, abs=1.5)

    # Without alignment the same frame measures wrong.
    unaligned = measure_extent(blue_segmenter().segment(bumped), annotations()).extent_px
    assert abs(unaligned - h) > 5


# --- fitting -----------------------------------------------------------------------------


def test_power_fit_recovers_lucas_washburn_exponent():
    t = np.linspace(0.5, 30, 40)
    rng = np.random.default_rng(1)
    y = 12.0 * np.sqrt(t) + rng.normal(0, 0.2, t.size)
    fits = {f.model: f for f in fit_all(t, y)}
    assert fits["power"].params["n"] == pytest.approx(0.5, abs=0.02)
    assert fits["sqrt"].params["k"] == pytest.approx(12.0, rel=0.01)
    assert fits["sqrt"].aic < fits["linear"].aic


def test_logistic_fit():
    t = np.linspace(0, 20, 50)
    y = 80 / (1 + np.exp(-0.6 * (t - 9)))
    fit = fit_model("logistic", t, y)
    assert fit.ok
    assert fit.params["K"] == pytest.approx(80, rel=0.01)
    assert fit.params["t_mid"] == pytest.approx(9, abs=0.1)


def test_fit_needs_enough_points():
    fit = fit_model("logistic", [1, 2, 3], [1, 2, 3])
    assert not fit.ok and "at least" in fit.message


def test_ecc_reports_failure_instead_of_a_wrong_alignment():
    reference = syn.scene(0, markers=False, texture=True, seed=1)
    unrelated = syn.scene(0, markers=False, texture=True, seed=2)  # different background
    _, alignment = align_frame(unrelated, reference, {}, mode="ecc")
    assert alignment.method == "failed"
