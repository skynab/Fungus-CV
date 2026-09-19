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


@pytest.mark.parametrize("frames", [3])
def test_demo_is_reproducible(tmp_path, frames):
    a = dye_experiment(tmp_path / "a", frames=frames)
    b = dye_experiment(tmp_path / "b", frames=frames)
    assert [r["sha256"] for r in Experiment(a.root).read_frames()] == \
        [r["sha256"] for r in Experiment(b.root).read_frames()]
