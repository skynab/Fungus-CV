"""Environmental covariates: logger import, interpolation, window summaries, meta-regression."""

import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from fungus_cv.analyze import covariates
from fungus_cv.analyze.covariates import Covariates, meta_regression, safe_name
from fungus_cv.cli import app
from fungus_cv.storage import Experiment

from .test_daily import outdoor

BERLIN = ZoneInfo("Europe/Berlin")


def berlin_experiment(tmp_path) -> Experiment:
    exp = Experiment.create(tmp_path / "exp", "exp")
    text = exp.config_path.read_text().replace("timezone: null", "timezone: Europe/Berlin")
    exp.config_path.write_text(text)
    return Experiment(exp.root)


def test_safe_names():
    assert safe_name("Temperature (°C)") == "temperature_c"
    assert safe_name("RH %") == "rh_pct"
    assert safe_name("  ") == "value"


def test_import_iso_log_and_merge(tmp_path):
    exp = berlin_experiment(tmp_path)
    log = tmp_path / "log.csv"
    log.write_text("Timestamp,Temperature (°C),RH %,Serial\n"
                   "2026-06-01T10:00:00Z,20.5,60,A1\n"
                   "2026-06-01T11:00:00Z,21.5,58,A1\n")
    summary = covariates.import_log(exp, log)
    assert summary.columns == ["temperature_c", "rh_pct"]  # the text column is not a covariate
    assert summary.time_format == "iso"
    second = tmp_path / "light.csv"
    second.write_text("time,lux\n2026-06-01T11:00:00Z,900\n2026-06-01T12:00:00Z,1000\n")
    covariates.import_log(exp, second)
    cov = covariates.load(exp)
    assert cov.names == ["temperature_c", "rh_pct", "lux"]
    assert len(cov.times) == 3
    assert math.isnan(cov.values["lux"][0]) and cov.values["lux"][1] == 900


def test_import_day_first_semicolon_decimal_comma_local_time(tmp_path):
    exp = berlin_experiment(tmp_path)
    log = tmp_path / "logger.csv"
    log.write_text("Logger export, site 3\n"
                   "Date;Time;Temp\n"
                   "13.06.2026;14:00;21,5\n"
                   "13.06.2026;15:00;22,0\n")
    covariates.import_log(exp, log)
    cov = covariates.load(exp)
    first = datetime.fromtimestamp(cov.times[0], timezone.utc)
    assert first == datetime(2026, 6, 13, 14, tzinfo=BERLIN)  # local summer time, not UTC
    assert cov.values["temp"].tolist() == [21.5, 22.0]


def test_ambiguous_dates_are_refused(tmp_path):
    exp = berlin_experiment(tmp_path)
    log = tmp_path / "log.csv"
    log.write_text("time,temp\n03/04/2026 10:00,20\n05/04/2026 10:00,21\n")
    with pytest.raises(ValueError, match="day-first or month-first"):
        covariates.import_log(exp, log)
    covariates.import_log(exp, log, time_format="%d/%m/%Y %H:%M")
    cov = covariates.load(exp)
    assert datetime.fromtimestamp(cov.times[1], BERLIN).month == 4


def test_interpolation_and_gaps():
    t = np.array([0.0, 60, 120, 180, 1000, 1060])
    cov = Covariates(t, {"temp": np.array([10.0, 11, 12, 13, 20, 21])})
    assert cov.at("temp", [30, 150]).tolist() == [10.5, 12.5]
    assert math.isnan(cov.at("temp", [500])[0])  # inside a logging gap: unknown
    assert math.isnan(cov.at("temp", [-10])[0]) and math.isnan(cov.at("temp", [2000])[0])
    assert cov.at("temp", [1000])[0] == 20


def test_summaries_are_time_weighted():
    # 10 degrees for one hour, then 30 for three hours, logged unevenly.
    t = np.array([0, 3600 - 1, 3600, 5400, 7200, 10800], float)
    v = np.array([10, 10, 30, 30, 30, 30], float)
    cov = Covariates(t, {"temp": v})
    s = cov.summarize("temp", 0, 10800)
    assert s["mean"] == pytest.approx((10 * 3600 + 30 * 7200) / 10800, rel=0.01)
    assert s["min"] == 10 and s["max"] == 30 and s["coverage"] == pytest.approx(1.0)
    half = cov.summarize("temp", -10800, 10800)
    assert half["coverage"] == pytest.approx(0.5)


def test_meta_regression_recovers_a_slope():
    x = np.array([12.0, 14, 16, 18, 20, 22, 24, 26])
    se = np.full(8, 0.05)
    y = 0.5 + 0.1 * x + np.random.default_rng(0).normal(0, 0.05, 8)
    out = meta_regression(y, se, x)
    assert out["slope"] == pytest.approx(0.1, abs=0.02)
    assert out["ci_low"] < 0.1 < out["ci_high"] and out["p"] < 1e-4 and out["weighted"]
    flat = meta_regression(np.full(8, 2.0) + np.random.default_rng(1).normal(0, 0.05, 8), se, x)
    assert flat["p"] > 0.01


def test_meta_regression_interval_coverage():
    """With few replicates and real between-replicate spread, the 95% interval still covers
    the true slope about 95% of the time (the reason for Knapp-Hartung and t)."""
    rng = np.random.default_rng(2)
    hits = 0
    runs = 1000
    for _ in range(runs):
        k = 6
        x = rng.uniform(10, 25, k)
        se = rng.uniform(0.02, 0.1, k)
        y = 1 + 0.05 * x + rng.normal(0, 0.08, k) + rng.normal(0, se)
        out = meta_regression(y, se, x)
        hits += out["ci_low"] <= 0.05 <= out["ci_high"]
    assert 0.93 <= hits / runs <= 0.995, hits / runs


def test_meta_regression_adjusted_for_condition():
    """Treated replicates happen to be warmer: alone the covariate looks important; adjusted
    for condition it isn't."""
    rng = np.random.default_rng(3)
    x = np.array([15.0, 16, 17, 18, 22, 23, 24, 25])
    groups = ["control"] * 4 + ["treated"] * 4
    y = np.array([1.0] * 4 + [2.0] * 4) + rng.normal(0, 0.05, 8)
    se = np.full(8, 0.05)
    alone = meta_regression(y, se, x)
    adjusted = meta_regression(y, se, x, groups)
    assert alone["p"] < 0.01
    assert adjusted["p"] > 0.05 and abs(adjusted["slope"]) < 0.05


def test_meta_regression_needs_spread_and_replicates():
    assert math.isnan(meta_regression([1, 2], [0.1, 0.1], [3, 4])["slope"])
    assert math.isnan(meta_regression([1, 2, 3], [0.1] * 3, [5, 5, 5])["slope"])
    unweighted = meta_regression([1.0, 2.1, 2.9, 4.2], [math.nan] * 4, [1, 2, 3, 4])
    assert unweighted["slope"] == pytest.approx(1.04, abs=0.01) and not unweighted["weighted"]


def warm_log(exp: Experiment, temp_c: float, days: int = 6) -> None:
    start = datetime(2026, 5, 31, 22, tzinfo=timezone.utc)
    lines = ["timestamp,Temperature (°C),rh"]
    for h in range(days * 24 + 4):
        t = start + timedelta(hours=h)
        daily = 3 * math.sin(2 * math.pi * (t.hour - 9) / 24)
        lines.append(f"{t.isoformat()},{temp_c + daily:.3f},70")
    path = exp.root / "logger.csv"
    path.write_text("\n".join(lines) + "\n")
    covariates.import_log(exp, path)


def test_report_shows_covariates(tmp_path):
    from fungus_cv.analyze.report import make_report

    exp = outdoor(tmp_path / "x", days=4)
    warm_log(exp, 18.0, days=4)
    result = make_report(exp, metric="coverage_pct", models=("linear",), bootstrap=0,
                         hours=(10, 14), covariates=("temperature_c",))
    names = [p.name for p in result.files]
    assert "covariates.png" in names
    summary = result.covariates["temperature_c"]
    assert summary["coverage"] == pytest.approx(1.0)
    # Only 10-14 h is fitted, but the covariate is summarised over the whole fitted span
    # (first to last fitted frame), day and night.
    assert summary["mean"] == pytest.approx(18.0, abs=0.6)
    with pytest.raises(ValueError, match="no covariate"):
        make_report(exp, metric="coverage_pct", models=("linear",), bootstrap=0,
                    covariates=("pressure",))


def test_study_relates_rate_to_temperature(tmp_path):
    from fungus_cv.analyze.study import run_study

    entries = []
    temps = [14.0, 16, 18, 20, 22, 24]
    for i, temp in enumerate(temps):
        name = f"r{i}"
        rate = 0.5 + 0.1 * temp  # growth speeds up by 0.1 %/day per degree
        exp = outdoor(tmp_path / name, rate=rate, days=6, seed=i, name=name)
        warm_log(exp, temp)
        entries.append({"path": name, "condition": "field"})
    study = {"name": "warmth", "metric": "coverage_pct", "model": "linear", "time_unit": "d",
             "hours": "10-14", "daily": "median", "bootstrap": 0,
             "covariates": ["temperature_c"], "experiments": entries}
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(study))
    result = run_study(path)
    effect = next(e for e in result.covariate_effects
                  if e["param"] == "b" and e["covariate"] == "temperature_c"
                  and e["adjusted_for"] == "none")
    assert effect["slope"] == pytest.approx(0.1, abs=0.02)
    assert effect["p"] < 0.001
    assert (result.out_dir / "covariate_effects.csv").exists()
    reps = (result.out_dir / "replicates.csv").read_text()
    assert "temperature_c_mean" in reps
    methods = (result.out_dir / "methods.md").read_text()
    assert "meta-regression" in methods and "Knapp-Hartung" in methods


def test_covariates_cli(tmp_path):
    exp = berlin_experiment(tmp_path)
    log = tmp_path / "log.csv"
    log.write_text("time,temp\n2026-06-01 10:00,20\n2026-06-01 11:00,21\n")
    runner = CliRunner()
    res = runner.invoke(app, ["covariates", "import", str(exp.root), str(log)])
    assert res.exit_code == 0, res.output
    assert "temp" in res.output and "2 rows" in res.output
    res = runner.invoke(app, ["covariates", "list", str(exp.root)])
    assert res.exit_code == 0 and "temp" in res.output and "20" in res.output
    bad = tmp_path / "bad.csv"
    bad.write_text("time,temp\n03/04/2026 10:00,20\n")
    res = runner.invoke(app, ["covariates", "import", str(exp.root), str(bad)])
    assert res.exit_code == 1 and "--time-format" in res.output
