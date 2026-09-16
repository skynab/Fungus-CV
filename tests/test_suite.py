"""Validation suite: every accuracy check in one run, with bounds and a drift baseline."""

import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from fungus_cv.analyze.suite import Bound, evaluate_bound, load_suite, run_suite, save_baseline
from fungus_cv.cli import app
from fungus_cv.learn.dataset import Dataset
from fungus_cv.storage import Experiment

from . import synthetic as syn
from .test_pipeline import build_experiment


def make_hand_labels(exp: Experiment, heights: list[int], root: Path) -> None:
    """Exact masks and ruler readings, as a careful person would produce them."""
    masks = root / "hand_masks"
    masks.mkdir()
    rows = []
    for frame, h in zip(exp.read_frames(), heights):
        mask = np.zeros((syn.H, syn.W), np.uint8)
        x0, x1 = syn.TOWEL_X
        mask[syn.BASE_ROW - h:900, x0:x1] = 255  # all the blue, reservoir included
        cv2.imwrite(str(masks / f"{Path(frame['file']).stem}.png"), mask)
        rows.append({"frame": Path(frame["file"]).name, "value": h * syn.MM_PER_PX,
                     "value_unc": 0.3})
    with open(root / "hand.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "value", "value_unc"])
        writer.writeheader()
        writer.writerows(rows)


def fake_evaluate(model_dir, dataset, reviewed_only, device, threshold):
    rows = [{"id": "a", "in_training_set": False, "iou": 0.8}]
    stats = {"n": 1, "iou_mean": 0.8, "iou_min": 0.8, "boundary_f1_2px_mean": 0.7}
    return rows, {"all": stats, "not_in_training_set": stats, "threshold": 0.5}


def write_suite(root: Path, **overrides) -> Path:
    spec = {
        "name": "synthetic",
        "cases": [
            {"name": "dye", "experiment": "exp", "checks": [
                {"type": "masks", "masks": "hand_masks", "expect": {
                    "iou_mean": {"min": 0.95, "max_drift": 0.001},
                    "extent_diff_mean": {"abs_max": 0.5}}},
                {"type": "measurements", "hand": "hand.csv", "expect": {
                    "bias": {"abs_max": 0.5}, "within_2u": {"min": 0.9}}},
            ]},
            {"name": "model", "checks": [
                {"type": "model", "model": "models/m", "dataset": "ds",
                 "expect": {"unseen_iou_mean": {"min": 0.75}}},
            ]},
        ],
    }
    spec.update(overrides)
    path = root / "suite.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


@pytest.fixture
def suite_dir(tmp_path):
    exp = Experiment.create(tmp_path / "exp", name="exp")
    heights = build_experiment(exp, minutes=range(0, 5), bump_at=-1)
    make_hand_labels(Experiment(exp.root), heights, tmp_path)
    (tmp_path / "models" / "m").mkdir(parents=True)
    Dataset.create(tmp_path / "ds").save()
    return tmp_path


def by_check(result):
    return {(c.case, c.check): c for c in result.checks}


def test_bounds():
    assert evaluate_bound(0.9, Bound(min=0.8, max=1.0), None) == []
    assert evaluate_bound(-0.7, Bound(abs_max=0.5), None) == ["|value| above 0.5"]
    assert evaluate_bound(0.5, Bound(max_drift=0.01), 0.52) == \
        ["drifted -0.02 from baseline 0.52 (max 0.01)"]
    assert "no baseline" in evaluate_bound(0.5, Bound(max_drift=0.01), None)[0]
    assert evaluate_bound(float("nan"), Bound(min=0), None) == ["no value"]


def test_suite_passes_on_accurate_measurements_and_writes_results(suite_dir):
    path = write_suite(suite_dir)
    result = run_suite(path, evaluate_model=fake_evaluate)
    checks = by_check(result)
    masks = {m.metric: m for m in checks[("dye", "masks-1")].metrics}
    assert masks["iou_mean"].value > 0.97
    # Drift can't be judged without a baseline: that check fails until one is saved.
    assert masks["iou_mean"].status == "fail" and not result.passed
    hand = {m.metric: m for m in checks[("dye", "measurements-2")].metrics}
    assert abs(hand["bias"].value) < 0.5 and hand["within_2u"].value == 1.0
    assert hand["n"].status == "info"
    assert checks[("model", "model-1")].passed
    assert checks[("model", "model-1")].provenance["weights_sha256"] is None
    assert checks[("dye", "masks-1")].provenance["settings_hash"]

    data = json.loads((result.out_dir / "results.json").read_text())
    assert data["provenance"]["fungus_cv_version"]
    assert {c["check"] for c in data["checks"]} == {"masks-1", "measurements-2", "model-1"}
    with open(result.out_dir / "summary.csv", newline="") as f:
        assert any(r["metric"] == "iou_mean" for r in csv.DictReader(f))

    save_baseline(result, path)
    again = run_suite(path, evaluate_model=fake_evaluate)
    assert again.passed and again.baseline_found
    iou = next(m for m in by_check(again)[("dye", "masks-1")].metrics if m.metric == "iou_mean")
    assert iou.drift == pytest.approx(0, abs=1e-9)  # re-analysis is reproducible


def test_suite_catches_drift_from_the_baseline(suite_dir):
    path = write_suite(suite_dir)
    save_baseline(run_suite(path, evaluate_model=fake_evaluate), path)
    baseline = json.loads((suite_dir / "baseline.json").read_text())
    baseline["dye"]["masks-1"]["iou_mean"] += 0.01  # results were better before
    (suite_dir / "baseline.json").write_text(json.dumps(baseline))
    result = run_suite(path, evaluate_model=fake_evaluate)
    assert not result.passed
    iou = next(m for m in by_check(result)[("dye", "masks-1")].metrics if m.metric == "iou_mean")
    assert iou.failures and "drifted" in iou.failures[-1]


def test_one_broken_check_does_not_stop_the_others(suite_dir):
    (suite_dir / "hand.csv").unlink()
    path = write_suite(suite_dir)
    result = run_suite(path, evaluate_model=fake_evaluate)
    checks = by_check(result)
    assert "FileNotFoundError" in checks[("dye", "measurements-2")].error
    assert checks[("dye", "masks-1")].metrics and checks[("model", "model-1")].passed


def test_unknown_metric_in_expect_fails(suite_dir):
    path = write_suite(suite_dir, cases=[{"name": "model", "checks": [
        {"type": "model", "model": "models/m", "dataset": "ds",
         "expect": {"iou_meen": {"min": 0.5}}}]}])
    result = run_suite(path, evaluate_model=fake_evaluate)
    bad = next(m for m in result.checks[0].metrics if m.metric == "iou_meen")
    assert "unknown metric" in bad.failures[0] and not result.passed


def test_suite_file_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"name": "x", "cases": [
        {"name": "a", "checks": [{"type": "masks", "masks": "m"}]}]}))
    with pytest.raises(ValueError, match="need 'experiment'"):
        load_suite(bad)
    bad.write_text(yaml.safe_dump({"name": "x", "cases": [
        {"name": "a", "experiment": "e", "checks": [
            {"type": "masks", "masks": "m", "expect": {"iou_mean": {"minimum": 1}}}]}]}))
    with pytest.raises(ValueError):
        load_suite(bad)


def test_cli_init_run_and_exit_codes(suite_dir, monkeypatch):
    runner = CliRunner()
    template = suite_dir / "template.yaml"
    result = runner.invoke(app, ["validate-suite", str(template), "--init"])
    assert result.exit_code == 0 and load_suite(template).cases

    import fungus_cv.learn.train as train

    monkeypatch.setattr(train, "evaluate_model", fake_evaluate)
    path = write_suite(suite_dir)
    result = runner.invoke(app, ["validate-suite", str(path)])
    assert result.exit_code == 1  # no baseline yet for max_drift
    assert "FAIL  dye / masks-1" in result.output and "no baseline" in result.output
    result = runner.invoke(app, ["validate-suite", str(path), "--save-baseline"])
    assert result.exit_code == 0 and "saved baseline" in result.output
    result = runner.invoke(app, ["validate-suite", str(path)])
    assert result.exit_code == 0, result.output
    assert "3/3 checks passed" in result.output


def test_validate_reports_uncertainty_coverage(suite_dir):
    from fungus_cv.analyze.pipeline import analyze
    from fungus_cv.analyze.validate import validate

    exp = Experiment(suite_dir / "exp")
    analyze(exp)
    a = validate(exp, suite_dir / "hand.csv")
    assert a.within_2u == 1.0 and 0 < a.z_rms < 2
    with open(suite_dir / "hand.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:  # a hand reading 3 mm off is outside any honest uncertainty
        r["value"] = float(r["value"]) + 3
    with open(suite_dir / "hand.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    a = validate(exp, suite_dir / "hand.csv")
    assert a.within_2u == 0.0 and a.z_rms > 3


@pytest.mark.skipif(not os.environ.get("FUNGUS_SUITE"),
                    reason="set FUNGUS_SUITE=path/to/suite.yaml to run the real-data suite")
def test_real_data_suite():
    """Regression check on hand-labeled real data (not in CI: the data isn't in the repo)."""
    result = run_suite(Path(os.environ["FUNGUS_SUITE"]))
    failed = [f"{c.case} / {c.check}: {c.error or [m.metric for m in c.metrics if m.failures]}"
              for c in result.checks if not c.passed]
    assert not failed, f"see {result.out_dir}: {failed}"
