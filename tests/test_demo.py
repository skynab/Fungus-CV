"""The demo experiments: realistic enough that every command has something to work on."""

import csv
import math

import pytest
from typer.testing import CliRunner

from fungus_cv.analyze.pipeline import analyze
from fungus_cv.analyze.report import make_report
from fungus_cv.analyze.suite import run_suite
from fungus_cv.cli import app
from fungus_cv.demo import dye_experiment, dye_study
from fungus_cv.storage import Experiment


def test_demo_experiment_is_measured_close_to_the_truth(tmp_path):
    run = dye_experiment(tmp_path / "demo", frames=20)
    exp = Experiment(run.root)
    rows = exp.read_frames()
    assert len(rows) == 20 and all(r["sha256"] and r["sharpness"] for r in rows)
    assert (run.root / "annotations.json").exists()
    summary = analyze(exp)
    assert summary.processed == 20 and not summary.flagged.get("blurry")

    result = make_report(exp, models=("sqrt_lag",), bootstrap=100)
    fit = result.fits[0]
    lo, hi = fit.ci95["k"]
    assert lo - 0.2 < run.k_mm_per_sqrt_min < hi + 0.2  # the √t law comes out
    assert fit.reduced_chi2 < 3  # and the reported uncertainties explain the scatter

    with open(run.root / "hand_measurements.csv", newline="") as f:
        hand = list(csv.DictReader(f))
    truth = dict(zip([p.split("/")[-1] for p in run.files], run.heights_mm))
    assert hand and all(abs(float(h["value"]) - truth[h["frame"]]) < 1.5 for h in hand)

    suite = run_suite(run.root / "suite.yaml", require_baseline=False)
    assert suite.passed, [(c.check, c.error, [m.failures for m in c.metrics]) for c in
                          suite.checks]


def test_demo_study_finds_the_faster_condition(tmp_path):
    from fungus_cv.analyze.study import run_study

    path, runs = dye_study(tmp_path / "study", replicates=3, frames=12)
    assert len(runs) == 6 and path.exists()
    for run in runs:
        analyze(Experiment(run.root))
    result = run_study(path)
    comparison = next(c for c in result.comparisons if c["param"] == "k")
    assert comparison["condition"] == "treated" and comparison["diff"] > 1.5
    assert comparison["p"] < 0.05
    assert not math.isnan(next(c for c in result.conditions if c["param"] == "k")["sd"])


def test_demo_cli(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["demo", str(tmp_path / "d"), "--frames", "6"])
    assert result.exit_code == 0 and "fungus analyze" in result.output
    assert len(Experiment(tmp_path / "d").read_frames()) == 6
    again = runner.invoke(app, ["demo", str(tmp_path / "d")])
    assert again.exit_code == 1 and "not empty" in again.output
    study = runner.invoke(app, ["demo", str(tmp_path / "s"), "--study", "--replicates", "2",
                                "--frames", "5"])
    assert study.exit_code == 0 and "fungus study" in study.output
    assert (tmp_path / "s" / "study.yaml").exists()
    sam = runner.invoke(app, ["demo", str(tmp_path / "c"), "--sam", "--frames", "3"])
    assert sam.exit_code == 0 and "fungus prompt" in sam.output
    assert Experiment(tmp_path / "c").config.analysis.target.method == "sam2"


def test_demo_has_room_conditions_for_the_report(tmp_path):
    from fungus_cv.analyze import covariates

    run = dye_experiment(tmp_path / "demo", frames=6)
    exp = Experiment(run.root)
    logged = covariates.load(exp)
    assert logged.names == ["temperature_c", "humidity_pctrh"]
    analyze(exp)
    result = make_report(exp, models=("sqrt_lag",), bootstrap=20,
                         covariates=("temperature_c",))
    assert result.covariates["temperature_c"]["coverage"] > 0.99
    assert any(p.name.endswith("covariates.png") for p in result.files)


def _colony_stand_in():
    """Stands in for SAM 2 (which needs PyTorch and a download): anything on the plate that
    isn't amber agar, i.e. the pale rim or the green centre."""
    from fungus_cv.config import ColorTargetConfig, HsvRange
    from fungus_cv.segment.color import ColorThresholdSegmenter

    return ColorThresholdSegmenter.from_config(ColorTargetConfig(hsv_ranges=[
        HsvRange(lower=(0, 0, 90), upper=(179, 45, 255)),
        HsvRange(lower=(35, 0, 60), upper=(90, 255, 255))]))


def test_sam_demo_is_set_up_for_prompting(tmp_path):
    from fungus_cv.demo import colony_experiment

    run = colony_experiment(tmp_path / "colony", frames=4)
    exp = Experiment(run.root)
    assert exp.config.analysis.target.method == "sam2"
    assert exp.config.analysis.markers.size_mm == pytest.approx(15.0)
    assert (run.root / "annotations.json").exists()
    assert not (run.root / "prompts.json").exists()  # clicking is the user's part
    result = CliRunner().invoke(app, ["analyze", str(run.root)])
    assert result.exit_code == 1 and "prompt" in result.output


def test_sam_demo_spreads_faster_one_way(tmp_path, monkeypatch):
    from fungus_cv.analyze import pipeline
    from fungus_cv.analyze.spread import spread
    from fungus_cv.demo import colony_experiment

    run = colony_experiment(tmp_path / "colony", frames=10, every_h=12)
    monkeypatch.setattr(pipeline, "build_segmenter", lambda *a, **k: _colony_stand_in())
    exp = Experiment(run.root)
    summary = analyze(exp)
    assert summary.processed == 10 and summary.failed == 0
    series = make_report(exp, metric="equivalent_radius_mm", models=("linear",),
                         bootstrap=20)
    assert series.n_used >= 8
    from fungus_cv.analyze.report import load_measurements

    rows = sorted(load_measurements(exp), key=lambda r: r["timestamp_utc"])
    for row, truth in zip(rows, run.radius_mm):
        assert float(row["equivalent_radius_mm"]) == pytest.approx(truth, abs=0.6)
    result = spread(exp)
    assert result.fastest is not None and abs(result.fastest.angle_deg) < 50
    assert result.anisotropy > 1.2


@pytest.mark.parametrize("frames", [3])
def test_demo_is_reproducible(tmp_path, frames):
    a = dye_experiment(tmp_path / "a", frames=frames)
    b = dye_experiment(tmp_path / "b", frames=frames)
    assert [r["sha256"] for r in Experiment(a.root).read_frames()] == \
        [r["sha256"] for r in Experiment(b.root).read_frames()]
