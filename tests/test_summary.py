"""The shareable HTML summary: self-contained, with fits, intervals, exclusions and settings."""

import re
from pathlib import Path

from typer.testing import CliRunner

from fungus_cv.analyze import exclusions
from fungus_cv.analyze.pipeline import analyze
from fungus_cv.analyze.summary import make_summary
from fungus_cv.cli import app
from fungus_cv.demo import dye_experiment
from fungus_cv.storage import Experiment


def test_summary_is_self_contained_and_complete(tmp_path):
    run = dye_experiment(tmp_path / "demo", frames=14)
    exp = Experiment(run.root)
    analyze(exp)
    exclusions.exclude(exp, Path(run.files[5]).name, "towel touched the glass")

    result = make_summary(exp, bootstrap=40, models=("sqrt", "sqrt_lag", "linear"),
                          time_to=(20,))
    page = result.path.read_text(encoding="utf-8")
    assert result.path == run.root / "results" / "summary.html"
    assert not result.problems
    # Everything embedded: no file or web references a reader would lack.
    assert page.count("data:image/png;base64,") >= 2
    assert not re.search(r'(src|href)="(?!data:)', page)
    for text in ("sqrt_lag", "Akaike weight", "95% CI", "time to 20",
                 "towel touched the glass", "Analysis settings", "extent_mm"):
        assert text in page, text
    settings_hash = exp.root.joinpath("results", "run_info.json").read_text()
    assert re.search(r'"settings_hash": "(\w+)"', settings_hash).group(1) in page
    assert "<script" not in page


def test_summary_cli(tmp_path):
    run = dye_experiment(tmp_path / "demo", frames=10)
    analyze(Experiment(run.root))
    out = tmp_path / "share" / "demo.html"
    res = CliRunner().invoke(app, ["summary", str(run.root), "--bootstrap", "20",
                                   "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists() and "1 plot(s)" in res.output


def test_summary_needs_results(tmp_path):
    run = dye_experiment(tmp_path / "demo", frames=3)
    res = CliRunner().invoke(app, ["summary", str(run.root)])
    assert res.exit_code == 1 and "fungus analyze" in res.output
