"""Frames excluded by hand: stored with reasons, always left out of fits, and reported."""

import csv
import json

import pytest
from typer.testing import CliRunner

from fungus_cv.analyze import exclusions
from fungus_cv.analyze.pipeline import analyze
from fungus_cv.analyze.report import load_series, make_report
from fungus_cv.cli import app
from fungus_cv.storage import Experiment

from .test_pipeline import T0, build_experiment


@pytest.fixture
def analyzed(experiment):
    build_experiment(experiment, minutes=range(1, 10), bump_at=-1)
    exp = Experiment(experiment.root)
    analyze(exp)
    return exp


def test_exclude_include_and_reasons(analyzed):
    exp = analyzed
    frames = [r["file"] for r in exp.read_frames()]
    item = exclusions.exclude(exp, frames[2].split("/")[-1].removesuffix(".png"),
                              "  hand in front of the camera ")
    assert item.frame_file == frames[2] and item.reason == "hand in front of the camera"
    exclusions.exclude(exp, frames[2], "reflection")  # same frame again: replaced
    exclusions.exclude(exp, frames[4], "moved", plot="plot1")
    marked = exclusions.load(exp)
    assert len(marked) == 2
    assert exclusions.reason_for(marked, frames[2], "main") == "reflection"
    assert exclusions.reason_for(marked, frames[4], "main") == ""
    assert exclusions.reason_for(marked, frames[4], "plot1") == "moved"
    with pytest.raises(ValueError, match="reason"):
        exclusions.exclude(exp, frames[1], " ")
    with pytest.raises(ValueError, match="not in frames.csv"):
        exclusions.exclude(exp, "nope.png", "x")
    assert exclusions.include(exp, frames[2]) == 1
    assert [e.frame_file for e in exclusions.load(exp)] == [frames[4]]


def test_reports_leave_excluded_frames_out_even_with_flags_included(analyzed):
    exp = analyzed
    frames = [r["file"] for r in exp.read_frames()]
    before = load_series(exp, exclude_flags=())
    exclusions.exclude(exp, frames[3], "lamp switched on")
    series = load_series(exp, exclude_flags=())
    assert series.use.sum() == before.use.sum() - 1 and not series.use[3]
    assert series.manual[3] == "lamp switched on"

    result = make_report(exp, t0=T0, bootstrap=0)
    assert result.n_excluded == 1
    out = exp.root / "results" / "report"
    with open(out / "frame_flags.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[3]["excluded_by_hand"] == "lamp switched on" and rows[3]["used_in_fits"] == "0"
    fits = json.loads((out / "fits.json").read_text())
    assert fits["excluded_by_hand"] == [{"frame_file": frames[3], "reason": "lamp switched on"}]


def test_exclude_cli(analyzed):
    runner = CliRunner()
    frame = analyzed.read_frames()[1]["file"].split("/")[-1]
    root = str(analyzed.root)
    result = runner.invoke(app, ["exclude", root, frame, "--reason", "condensation"])
    assert result.exit_code == 0 and "Excluded" in result.output
    listed = runner.invoke(app, ["exclude", root])
    assert "condensation" in listed.output
    assert runner.invoke(app, ["exclude", root, frame]).exit_code == 1  # no reason
    removed = runner.invoke(app, ["exclude", root, frame, "--remove"])
    assert "Removed 1" in removed.output
    assert "No frames excluded" in runner.invoke(app, ["exclude", root]).output
