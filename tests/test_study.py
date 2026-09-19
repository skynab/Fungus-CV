"""Studies: per-replicate fits, condition summaries and between-condition tests."""

import csv
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from fungus_cv.analyze.fit import MODELS
from fungus_cv.analyze.pipeline import MEASUREMENT_FIELDS
from fungus_cv.analyze.study import load_study, run_study
from fungus_cv.cli import app
from fungus_cv.storage import Experiment, iso_utc

START = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


def fake_experiment(root, name, K, r, t_mid, seed, hours=48, every_h=1.0, noise=1.0,
                    target_settings=None, delay_h=0.0):
    """An experiment folder with measurements.csv of logistic coverage over time."""
    exp = Experiment.create(root / name, name=name)
    rng = np.random.default_rng(seed)
    results = exp.root / "results"
    (results / "runs").mkdir(parents=True)
    settings_hash = f"hash{seed:04d}"
    settings = {"analysis": {"align": "markers_or_ecc", "rectify": {"enabled": False},
                             "lighting": {"method": "none"},
                             "target": target_settings or {"method": "color"},
                             "measure": {"mode": "path"},
                             "uncertainty": {"segmentation": True, "hsv_delta": [4, 20, 20],
                                             "probability_delta": 0.1, "logit_delta": 1.0},
                             "front_percentile": 50.0},
                "segmenter": {"method": (target_settings or {}).get("method", "color")}}
    (results / "runs" / f"{settings_hash}.json").write_text(json.dumps({"settings": settings}))
    with open(results / "measurements.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MEASUREMENT_FIELDS, restval="")
        writer.writeheader()
        for i, h in enumerate(np.arange(0, hours, every_h)):
            ts = START + timedelta(hours=float(h + delay_h))
            value = MODELS["logistic"].func(h, K, r, t_mid) + rng.normal(0, noise)
            writer.writerow({"timestamp_utc": iso_utc(ts), "frame_file": f"frames/{i}.png",
                             "camera": "cam0", "plot": "main", "settings_hash": settings_hash,
                             "coverage_pct": round(value, 4),
                             "coverage_pct_unc": noise, "flags": ""})
    return exp


def write_study(root, **overrides):
    spec = {
        "name": "moss-test", "metric": "coverage_pct", "model": "logistic",
        "also_fit": ["gompertz"], "reference": "control", "params": ["K", "r"],
        "bootstrap": 100, "time_unit": "h",
        "experiments": [
            {"path": f"{cond}{i}", "condition": cond}
            for cond in ("control", "treated") for i in range(1, 4)
        ],
    }
    spec.update(overrides)
    path = root / "study.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


@pytest.fixture
def study_dir(tmp_path):
    for i in range(1, 4):  # same plateau, treated grows faster
        fake_experiment(tmp_path, f"control{i}", 70 + i, 0.20 + 0.01 * i, 24, seed=i)
        fake_experiment(tmp_path, f"treated{i}", 70 + i, 0.40 + 0.01 * i, 24, seed=10 + i)
    return tmp_path


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_study_recovers_conditions_and_finds_the_rate_difference(study_dir):
    result = run_study(write_study(study_dir))
    assert result.time_unit == "h" and result.metric == "coverage_pct"
    assert all(r.fit is not None for r in result.replicates)
    by = {(c["condition"], c["param"]): c for c in result.conditions}
    assert by[("control", "r")]["mean"] == pytest.approx(0.22, abs=0.02)
    assert by[("treated", "r")]["mean"] == pytest.approx(0.42, abs=0.03)
    assert by[("control", "K")]["n"] == 3
    ci = by[("treated", "r")]
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    assert ci["re_mean"] == pytest.approx(ci["mean"], abs=0.02)

    comp = {c["param"]: c for c in result.comparisons}
    assert set(comp) == {"K", "r"} and comp["r"]["condition"] == "treated"
    assert comp["r"]["diff"] == pytest.approx(0.2, abs=0.03)
    assert comp["r"]["p"] < 0.01 and comp["r"]["p_holm"] >= comp["r"]["p"]
    assert comp["K"]["p"] > 0.05  # plateaus are the same

    totals = {r["model"]: r for r in result.model_selection if r["condition"] == "(all)"}
    assert totals["logistic"]["akaike_weight"] > 0.9

    out = result.out_dir
    for name in ("replicates.csv", "conditions.csv", "comparisons.csv", "model_selection.csv",
                 "data_long.csv", "study.json", "methods.md", "curves.png", "curves.svg",
                 "parameters.png"):
        assert (out / name).exists(), name
    reps = rows(out / "replicates.csv")
    assert len(reps) == 6 and {"r", "r_se", "r_ci_low", "r_ci_high"} <= set(reps[0])
    assert len(rows(out / "data_long.csv")) == 6 * 48
    json.loads((out / "study.json").read_text())  # valid JSON (no NaN)
    methods = (out / "methods.md").read_text()
    assert "Welch" in methods and "control: n = 3" in methods and "Holm" in methods
    assert "logistic" in methods and "hsv" in methods.lower()


def test_all_pairs_without_a_reference(study_dir):
    fake_experiment(study_dir, "dry1", 40, 0.3, 24, seed=21)
    fake_experiment(study_dir, "dry2", 42, 0.3, 24, seed=22)
    spec = yaml.safe_load(write_study(study_dir).read_text())
    spec["reference"] = None
    spec["experiments"] += [{"path": "dry1", "condition": "dry"},
                            {"path": "dry2", "condition": "dry"}]
    (study_dir / "study.yaml").write_text(yaml.safe_dump(spec))
    result = run_study(study_dir / "study.yaml")
    pairs = {(c["condition"], c["versus"]) for c in result.comparisons if c["param"] == "K"}
    assert pairs == {("control", "treated"), ("control", "dry"), ("treated", "dry")}
    k = [c for c in result.comparisons if c["param"] == "K"]
    assert all(c["p_holm"] >= c["p"] for c in k)


def test_per_replicate_start_times(study_dir):
    """A replicate photographed from a later start is shifted back with its own t0."""
    fake_experiment(study_dir, "late1", 72, 0.4, 24, seed=31, delay_h=5.0)
    spec = yaml.safe_load(write_study(study_dir).read_text())
    spec["experiments"] = [{"path": "late1", "condition": "late",
                            "t0": (START + timedelta(hours=5)).isoformat()},
                           {"path": "control1", "condition": "control"}]
    (study_dir / "study.yaml").write_text(yaml.safe_dump(spec))
    result = run_study(study_dir / "study.yaml")
    late = result.replicates[0].fit
    assert late.params["t_mid"] == pytest.approx(24, abs=0.5)
    assert any("only 1 usable replicate" in w for w in result.warnings)
    comp = result.comparisons[0]
    assert np.isfinite(comp["diff"]) and np.isnan(comp["p"])


def test_problems_are_reported_not_hidden(study_dir):
    fake_experiment(study_dir, "short", 70, 0.3, 24, seed=41, hours=3)  # too few frames
    fake_experiment(study_dir, "sam", 70, 0.4, 24, seed=42,
                    target_settings={"method": "sam2"})
    spec = yaml.safe_load(write_study(study_dir).read_text())
    spec["experiments"] += [{"path": "short", "condition": "treated", "replicate": "s"},
                            {"path": "sam", "condition": "treated", "replicate": "sam"},
                            {"path": "missing", "condition": "treated", "replicate": "m"}]
    (study_dir / "study.yaml").write_text(yaml.safe_dump(spec))
    result = run_study(study_dir / "study.yaml")
    text = "\n".join(result.warnings)
    assert "treated/s: logistic fit failed" in text
    assert "treated/sam was analyzed with different target" in text
    assert "treated/m:" in text
    treated_r = next(c for c in result.conditions
                     if c["condition"] == "treated" and c["param"] == "r")
    assert treated_r["n"] == 4  # three good + sam; the short and missing ones are left out
    methods = (result.out_dir / "methods.md").read_text()
    assert "Warnings to resolve" in methods and "not used" in methods


def test_study_file_validation(tmp_path):
    path = write_study(tmp_path, model="cubic")
    with pytest.raises(ValueError, match="unknown model"):
        load_study(path)
    path = write_study(tmp_path, params=["K", "q"])
    with pytest.raises(ValueError, match="no parameter"):
        load_study(path)
    path = write_study(tmp_path, reference="placebo")
    with pytest.raises(ValueError, match="reference condition"):
        load_study(path)
    path = write_study(tmp_path, reference=None, experiments=[
        {"path": "a", "condition": "c", "replicate": "1"},
        {"path": "b", "condition": "c", "replicate": "1"}])
    with pytest.raises(ValueError, match="unique"):
        load_study(path)


def test_cli_init_and_run(study_dir):
    runner = CliRunner()
    template = study_dir / "template.yaml"
    result = runner.invoke(app, ["study", str(template), "--init"])
    assert result.exit_code == 0 and load_study(template).model == "logistic"
    result = runner.invoke(app, ["study", str(write_study(study_dir))])
    assert result.exit_code == 0, result.output
    assert "treated − control" in result.output and "Holm p=" in result.output
    assert "model selection over 6 replicates" in result.output
    bad = runner.invoke(app, ["study", str(study_dir / "nope.yaml")])
    assert bad.exit_code == 1


def test_a_quantity_missing_for_some_replicates(study_dir):
    """max_rate doesn't exist for a sqrt_lag curve; time_to_ a level some never reach."""
    result = run_study(write_study(study_dir, params=["max_rate", "time_to_72"],
                                   bootstrap=0))
    by = {(c["condition"], c["param"]): c for c in result.conditions}
    assert by[("control", "max_rate")]["n"] == 3  # logistic: every replicate has one
    # Plateaus are 71-73, so only some replicates ever reach 72.
    assert by[("control", "time_to_72")]["n"] < 3
    assert any("could not be read off" in w for w in result.warnings)
