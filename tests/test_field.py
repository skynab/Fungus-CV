"""M6: field plots seen at an angle; areas vs true outlines, edge advance, colour indices."""

import csv
from datetime import timedelta

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from fungus_cv.analyze.pipeline import Analyzer, analyze
from fungus_cv.analyze.report import make_report
from fungus_cv.analyze.validate import validate, write_template
from fungus_cv.cli import app
from fungus_cv.measure.color_indices import color_indices
from fungus_cv.measure.geometry import Annotations, Plot, plots_hull
from fungus_cv.storage import Experiment, FrameRecord, encode_image, iso_utc

from . import synthetic_field as sf
from .test_pipeline import T0, read_measurements

RADII = {  # patch "radius" per day, per plot
    "plot1": [60, 90, 120, 150, 180, 210],
    "plot2": [0, 30, 45, 60, 75, 90],
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def project(h, pts):
    return [tuple(map(float, p)) for p in
            cv2.perspectiveTransform(np.float64(pts).reshape(-1, 1, 2), h).reshape(-1, 2)]


def field_experiment(experiment: Experiment, rectify: bool = True) -> Experiment:
    for day in range(len(RADII["plot1"])):
        img = sf.photograph(sf.ground({k: v[day] for k, v in RADII.items()}, seed=day))
        ts = T0 + timedelta(days=day)
        path = experiment.save_bytes(encode_image(img, "png"), ts, "cam0", "png")
        experiment.append_frame(FrameRecord(timestamp_utc=iso_utc(ts), camera="cam0",
                                            status="ok", file=experiment.relative(path)))
    h, s, v = (int(c) for c in cv2.cvtColor(np.uint8([[sf.PATCH]]), cv2.COLOR_BGR2HSV)[0, 0])
    text = experiment.config_path.read_text()
    for old, new in [
        ("size_mm: null", f"size_mm: {sf.MARKER_MM}"),
        ("- lower: [95, 60, 30]\n          upper: [135, 255, 255]",
         f"- lower: [{h - 8}, {s - 60}, {v - 60}]\n          upper: [{h + 8}, 255, 255]"),
        ("enabled: false", f"enabled: {'true' if rectify else 'false'}"),
    ]:
        assert old in text, old
        text = text.replace(old, new, 1)
    experiment.config_path.write_text(text)
    exp = Experiment(experiment.root)

    # "Hand-drawn" plot outlines, mapped into the prepared (rectified) reference frame.
    analyzer = Analyzer(exp, with_segmenter=False, require_annotations=False)
    to_prepared = sf.TILT if analyzer.rectification is None else \
        analyzer.rectification.matrix @ sf.TILT
    w, hgt = analyzer.reference.shape[1], analyzer.reference.shape[0]
    plots = [Plot(name, project(to_prepared, poly)) for name, poly in sf.PLOTS.items()]
    Annotations(None, None, plots_hull(plots, 10), (w, hgt), plots=plots).save(
        exp.root / "annotations.json")
    return exp


def by_plot(rows):
    out = {}
    for r in sorted(rows, key=lambda r: r["timestamp_utc"]):
        out.setdefault(r["plot"], []).append(r)
    return out


def test_plot_areas_match_true_outlines(experiment):
    """M6 acceptance: area error well under 10% against the true outlines."""
    exp = field_experiment(experiment)
    summary = analyze(exp)
    assert summary.processed == 6 and summary.failed == 0
    rows = by_plot(read_measurements(exp))
    assert set(rows) == {"plot1", "plot2"}
    for name, plot_rows in rows.items():
        for row, radius in zip(plot_rows, RADII[name]):
            truth = sf.true_area(radius)
            measured = float(row["target_area_mm2"])
            if radius == 0:
                assert measured < 150
                continue
            # Small patches: edge pixels are a bigger share of the area.
            assert measured == pytest.approx(truth, rel=0.03 if radius >= 60 else 0.06), \
                (name, radius)
            assert float(row["equivalent_radius_mm"]) == pytest.approx(
                np.sqrt(truth / np.pi), rel=0.02)
        plot_area = cv2.contourArea(np.float32(sf.PLOTS[name]))
        assert float(plot_rows[0]["roi_area_mm2"]) == pytest.approx(plot_area, rel=0.01)


def test_without_rectification_areas_are_wrong(experiment):
    exp = field_experiment(experiment, rectify=False)
    analyze(exp)
    rows = by_plot(read_measurements(exp))
    errors = [abs(float(r["target_area_mm2"]) / sf.true_area(rad) - 1)
              for r, rad in zip(rows["plot1"], RADII["plot1"])]
    # One average scale is wrong across a tilted field (rectified errors are under 3%).
    assert max(errors) > 0.06


def test_edge_advance_and_colour_indices(experiment):
    exp = field_experiment(experiment)
    analyze(exp)
    result = make_report(exp, metric="edge_advance_p95_mm", plot="plot1", t0=T0)
    assert result.plot == "plot1"
    assert (exp.root / "results" / "report" / "plot1" / "fits.json").exists()
    flags = read_csv(exp.root / "results" / "report" / "plot1" / "frame_flags.csv")
    advances = [float(r["edge_advance_p95_mm"]) for r in flags]
    assert advances[0] == 0.0
    assert all(b > a for a, b in zip(advances, advances[1:]))
    # Growth of R by 150 mm moves the edge by 150 mm * (0.65 .. 1.35) depending on direction.
    assert 100 < advances[-1] < 220

    rows = by_plot(read_measurements(exp))
    gcc = [float(r["gcc_mean"]) for r in rows["plot1"]]
    assert all(b < a for a, b in zip(gcc, gcc[1:]))  # yellowing lowers greenness

    # plot2 starts with no patch: advance is blank until the patch appears.
    make_report(exp, metric="edge_advance_max_mm", plot="plot2", t0=T0)
    flags2 = read_csv(exp.root / "results" / "report" / "plot2" / "frame_flags.csv")
    assert flags2[0]["edge_advance_max_mm"] == "" and flags2[1]["edge_advance_max_mm"] == "0.0"


def test_report_cli_covers_all_plots(experiment):
    exp = field_experiment(experiment)
    analyze(exp)
    out = CliRunner().invoke(app, ["report", str(exp.root), "--metric", "target_area_mm2"])
    assert out.exit_code == 0, out.output
    assert "[plot1]" in out.output and "[plot2]" in out.output
    for name in ("plot1", "plot2"):
        assert (exp.root / "results" / "report" / name / "target_area_mm2_vs_time.png").exists()


def test_validate_field_areas_with_plot_column(experiment, tmp_path):
    exp = field_experiment(experiment)
    analyze(exp)
    template = tmp_path / "hand.csv"
    assert write_template(exp, template, count=6) == 12  # 6 frames x 2 plots
    rows = read_csv(template)
    day = {ts: i for i, ts in enumerate(sorted({r["timestamp_utc"] for r in rows}))}
    for r in rows:
        r["value"] = f"{sf.true_area(RADII[r['plot']][day[r['timestamp_utc']]]):.1f}"
    with open(template, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    a = validate(exp, template, metric="target_area_mm2", plot="plot1")
    assert a.n == 6 and a.pearson_r > 0.999
    both = validate(exp, template, metric="target_area_mm2")
    assert both.n == 12


def test_colour_indices_values():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:, :5] = (0, 255, 0)  # pure green
    img[:, 5:] = (0, 0, 0)  # empty border, ignored
    region = np.ones((10, 10), bool)
    ci = color_indices(img, region)
    assert ci["gcc_mean"] == pytest.approx(1.0)
    assert ci["exg_mean"] == pytest.approx(2.0)


def test_field_annotations_need_no_base(tmp_path):
    plots = [Plot("a", [(0, 0), (10, 0), (10, 10)]), Plot("b", [(20, 0), (30, 0), (30, 10)])]
    ann = Annotations(None, None, plots_hull(plots), (40, 20), plots=plots)
    ann.save(tmp_path / "a.json")
    loaded = Annotations.load(tmp_path / "a.json")
    assert [p.name for p in loaded.measured_plots()] == ["a", "b"]
    with pytest.raises(ValueError):
        Annotations(None, None, plots_hull(plots), (40, 20))  # neither axis nor plots
    with pytest.raises(ValueError):
        Annotations(None, None, plots_hull(plots), (40, 20), plots=[plots[0], plots[0]])
