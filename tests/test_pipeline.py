"""End to end: synthetic time-lapse -> analyze -> report."""

import csv
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from typer.testing import CliRunner

from fungus_cv.analyze.pipeline import analyze
from fungus_cv.analyze.report import make_report
from fungus_cv.cli import app
from fungus_cv.measure.geometry import Annotations
from fungus_cv.storage import Experiment, FrameRecord, encode_image, iso_utc

from . import synthetic as syn

K_PX = 60.0  # dye height = K_PX * sqrt(minutes)
T0 = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


def build_experiment(experiment: Experiment, minutes=range(0, 11), bump_at=5) -> list[float]:
    heights = []
    for i, minute in enumerate(minutes):
        h = int(round(K_PX * np.sqrt(minute)))
        img = syn.scene(h)
        if i == bump_at:
            img = syn.bump(img, 10, 6, 0.8)
        ts = T0 + timedelta(minutes=minute)
        data = encode_image(img, "png")
        path = experiment.save_bytes(data, ts, "cam0", "png")
        experiment.append_frame(FrameRecord(
            timestamp_utc=iso_utc(ts), camera="cam0", status="ok",
            file=experiment.relative(path), width=syn.W, height=syn.H,
        ))
        heights.append(h)
    Annotations(base=syn.BASE_POINT, tip=syn.TIP_POINT, roi=syn.ROI,
                image_size=(syn.W, syn.H)).save(experiment.root / "annotations.json")
    text = experiment.config_path.read_text().replace("size_mm: null", f"size_mm: {syn.MARKER_MM}")
    experiment.config_path.write_text(text)
    return heights


def read_measurements(experiment):
    with open(experiment.root / "results" / "measurements.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_analyze_measures_known_heights(experiment):
    heights = build_experiment(experiment)
    exp = Experiment(experiment.root)
    summary = analyze(exp)
    assert summary.processed == len(heights) and summary.failed == 0

    rows = read_measurements(exp)
    for row, h in zip(rows, heights):
        assert float(row["extent_mm"]) == pytest.approx(h * syn.MM_PER_PX, abs=0.4)
        assert float(row["extent_mm_unc"]) > 0
        assert row["align_method"] == "markers"
        assert (exp.root / row["mask_file"]).exists()
        assert (exp.root / row["overlay_file"]).exists()
    assert float(rows[5]["align_shift_px"]) > 5  # the bumped frame was corrected
    assert float(rows[0]["axis_length_mm"]) == pytest.approx(700 * syn.MM_PER_PX, rel=0.005)

    run_info = json.loads((exp.root / "results" / "run_info.json").read_text())
    assert run_info["scale"]["mm_per_px"] == pytest.approx(syn.MM_PER_PX, rel=0.003)


def test_analyze_is_incremental_and_archives_on_settings_change(experiment):
    build_experiment(experiment, minutes=range(0, 4))
    exp = Experiment(experiment.root)
    first = analyze(exp)
    again = analyze(exp)
    assert again.processed == 0 and again.skipped_existing == first.processed

    text = exp.config_path.read_text().replace("front_percentile: 50", "front_percentile: 90")
    exp.config_path.write_text(text)
    changed = analyze(Experiment(exp.root))
    assert changed.processed == first.processed
    assert changed.settings_hash != first.settings_hash
    assert len(list((exp.root / "results" / "archive").glob("measurements_*.csv"))) == 1


def test_report_recovers_sqrt_law(experiment):
    build_experiment(experiment, minutes=range(1, 16), bump_at=-1)
    exp = Experiment(experiment.root)
    analyze(exp)
    result = make_report(exp, t0=T0, video=True)
    fits = {f.model: f for f in result.fits}
    assert result.metric == "extent_mm" and result.time_unit == "min"
    assert fits["power"].params["n"] == pytest.approx(0.5, abs=0.02)
    assert fits["sqrt"].params["k"] == pytest.approx(K_PX * syn.MM_PER_PX, rel=0.01)
    assert result.retreats == 0
    names = {p.name for p in result.files}
    assert {"extent_mm_vs_time.png", "quality_checks.png", "fits.json"} <= names


def test_cli_analyze_requires_annotations(experiment):
    build_experiment(experiment, minutes=range(0, 2))
    (experiment.root / "annotations.json").unlink()
    result = CliRunner().invoke(app, ["analyze", str(experiment.root)])
    assert result.exit_code == 1
    assert "annotate" in result.output


def test_cli_analyze_and_report(experiment, tmp_path):
    build_experiment(experiment, minutes=range(0, 8), bump_at=-1)
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", str(experiment.root)])
    assert result.exit_code == 0, result.output
    assert "measured 8" in result.output
    result = runner.invoke(app, ["report", str(experiment.root)])
    assert result.exit_code == 0, result.output
    assert "power" in result.output and "lowest AIC" in result.output

    sheet = tmp_path / "markers.png"
    result = runner.invoke(app, ["markers", str(sheet), "--dpi", "100"])
    assert result.exit_code == 0 and sheet.exists()


def test_report_writes_vector_figures_and_residuals(experiment):
    build_experiment(experiment, minutes=range(1, 14), bump_at=-1)
    exp = Experiment(experiment.root)
    analyze(exp)
    result = make_report(exp, t0=T0, formats=("png", "pdf", "svg"), bootstrap=0)
    names = {p.name for p in result.files}
    for ext in ("png", "pdf", "svg"):
        assert f"extent_mm_vs_time.{ext}" in names and f"quality_checks.{ext}" in names
        assert f"fit_and_residuals.{ext}" in names
    svg = next(p for p in result.files if p.name == "extent_mm_vs_time.svg")
    assert svg.read_text(encoding="utf-8").lstrip().startswith("<?xml")  # vector, not a bitmap
    pdf = next(p for p in result.files if p.name.endswith("fit_and_residuals.pdf"))
    assert pdf.read_bytes().startswith(b"%PDF")

    default = make_report(exp, t0=T0, bootstrap=0)
    assert not any(str(p).endswith((".pdf", ".svg")) for p in default.files)

    with pytest.raises(ValueError, match="unknown figure format"):
        make_report(exp, t0=T0, formats=("eps",), bootstrap=0)


def test_report_cli_rejects_unknown_format(experiment):
    build_experiment(experiment, minutes=range(1, 4), bump_at=-1)
    analyze(Experiment(experiment.root))
    result = CliRunner().invoke(app, ["report", str(experiment.root), "--format", "eps"])
    assert result.exit_code == 1 and "unknown figure format" in result.output
