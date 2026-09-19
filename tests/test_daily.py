"""Outdoor runs: an hours-of-day window and one value per day."""

import csv
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from fungus_cv.analyze.report import (
    daily_series,
    in_hours,
    load_series,
    make_report,
    parse_hours,
)
from fungus_cv.analyze.study import run_study
from fungus_cv.cli import app
from fungus_cv.storage import Experiment, iso_utc

BERLIN = ZoneInfo("Europe/Berlin")
FIELDS = ["timestamp_utc", "frame_file", "camera", "plot", "settings_hash", "flags",
          "coverage_pct", "coverage_pct_unc"]


def outdoor(root, rate=2.0, days=8, every_h=1, noise=0.5, seed=0, name="field") -> Experiment:
    """Hourly frames in Berlin. By day the coverage grows at ``rate`` % per day with a little
    noise; at night (and in low sun) the dark frames read far too low and scatter wildly."""
    exp = Experiment.create(root, name)
    text = exp.config_path.read_text().replace("timezone: null", "timezone: Europe/Berlin")
    exp.config_path.write_text(text)
    rng = np.random.default_rng(seed)
    start = datetime(2026, 6, 1, 0, 0, tzinfo=BERLIN)
    results = root / "results"
    results.mkdir()
    with open(results / "measurements.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        for k in range(days * 24 // every_h):
            local = start + timedelta(hours=k * every_h)
            days_in = (local - start).total_seconds() / 86400
            truth = 10 + rate * days_in
            if 9 <= local.hour < 16:
                value = truth + rng.normal(0, noise)
            else:
                value = truth * rng.uniform(0.1, 0.6)  # dark frame
            writer.writerow({
                "timestamp_utc": iso_utc(local.astimezone(timezone.utc)),
                "frame_file": f"frames/{k:04d}.jpg", "camera": "cam0", "plot": "main",
                "settings_hash": "abc", "flags": "", "coverage_pct": round(value, 4),
                "coverage_pct_unc": 0.3})
    return Experiment(root)


def test_parse_and_match_hours():
    assert parse_hours("10-14") == (10, 14)
    assert parse_hours("9:30-15") == (9.5, 15)
    night = parse_hours("20-6")
    assert in_hours(datetime(2026, 1, 1, 23), night) and in_hours(datetime(2026, 1, 1, 3), night)
    assert not in_hours(datetime(2026, 1, 1, 12), night)
    assert in_hours(datetime(2026, 1, 1, 10), (10, 14))
    assert not in_hours(datetime(2026, 1, 1, 14), (10, 14))
    for bad in ("10", "a-b", "10-10", "25-3"):
        with pytest.raises(ValueError):
            parse_hours(bad)


def test_hours_use_the_sites_time_zone(tmp_path):
    exp = outdoor(tmp_path / "x", days=2)
    series = load_series(exp, "coverage_pct", hours=(10, 14))
    used = [datetime.fromisoformat(r["timestamp_utc"].replace("Z", "+00:00"))
            .astimezone(BERLIN).hour for r, u in zip(series.rows, series.use) if u]
    assert sorted(set(used)) == [10, 11, 12, 13]  # Berlin time, not UTC or this computer's
    assert series.outside_hours.sum() == 48 - 8


def test_hours_remove_the_night_bias(tmp_path):
    exp = outdoor(tmp_path / "x")
    everything = make_report(exp, metric="coverage_pct", models=("linear",), bootstrap=0,
                             time_unit="d")
    windowed = make_report(exp, metric="coverage_pct", models=("linear",), bootstrap=0,
                           time_unit="d", hours=(10, 14))
    assert abs(everything.fits[0].params["b"] - 2.0) > 0.5  # night frames drag it down
    assert windowed.fits[0].params["b"] == pytest.approx(2.0, abs=0.05)
    assert windowed.outside_hours == 8 * 24 - 8 * 4


def test_daily_values_and_their_uncertainty(tmp_path):
    exp = outdoor(tmp_path / "x", noise=0.5)
    series = load_series(exp, "coverage_pct", time_unit="d", hours=(9, 16), daily="median")
    assert len(series.t) == 8 and series.daily == "median"
    assert [r["n_frames"] for r in series.rows] == [7] * 8
    # Each day sits at its midday, on the truth.
    for t, y in zip(series.t, series.y):
        assert y == pytest.approx(10 + 2 * t, abs=0.6)
    # Shared frame uncertainty (0.3) and the within-day scatter both count.
    assert np.all(series.unc > 0.3) and np.all(series.unc < 1.0)

    result = make_report(exp, metric="coverage_pct", models=("linear",), bootstrap=0,
                         time_unit="d", hours=(9, 16), daily="median")
    assert result.daily == "median" and result.n_used == 8
    assert result.fits[0].params["b"] == pytest.approx(2.0, abs=0.1)
    out = exp.root / "results" / "report"
    with open(out / "daily.csv", newline="") as f:
        days = list(csv.DictReader(f))
    assert len(days) == 8 and days[0]["frames"] == "7"
    with open(out / "frame_flags.csv", newline="") as f:
        frames = list(csv.DictReader(f))
    assert len(frames) == 8 * 24  # every frame behind the days is still accounted for


def test_single_frame_days_keep_the_reported_uncertainty(tmp_path):
    exp = outdoor(tmp_path / "x", every_h=24, days=5)  # one frame a day, at midnight
    series = load_series(exp, "coverage_pct", daily="mean")
    assert np.allclose(series.unc, 0.3)


def test_daily_uncertainty_is_honest():
    """Over many simulated days, the daily-median uncertainty covers the truth ~95%."""
    from fungus_cv.analyze.report import Series

    rng = np.random.default_rng(1)
    hits, n = 0, 0
    for _ in range(300):
        frames_per_day = 8
        t = np.repeat(np.arange(10.0), frames_per_day) + np.tile(
            np.linspace(0.4, 0.6, frames_per_day), 10)
        truth_day = rng.normal(0, 0.4, 10)  # a shared error per day (e.g. that day's light)
        y = 5.0 + np.repeat(truth_day, frames_per_day) + rng.normal(0, 1.0, t.size)
        unc = np.full(t.size, 0.4)  # the reported shared uncertainty
        zeros = np.zeros(t.size, bool)
        frames = Series("m", "main", [{"overlay_file": ""} for _ in t], "d", 0.0, t, y, unc,
                        zeros, zeros, zeros, [""] * t.size)
        days = daily_series(frames, "median", timezone.utc, n_boot=200,
                            seed=int(rng.integers(1e6)))
        z = (days.y - 5.0) / days.unc
        hits += int((np.abs(z) < 1.96).sum())
        n += z.size
    assert 0.90 < hits / n < 0.98, hits / n


def test_study_with_hours_and_daily(tmp_path):
    entries = []
    for condition, rate in (("shade", 1.0), ("sun", 2.0)):
        for i in range(3):
            name = f"{condition}{i}"
            outdoor(tmp_path / name, rate=rate * (1 + 0.03 * (i - 1)), days=6, seed=i,
                    name=name)
            entries.append({"path": name, "condition": condition})
    study = {"name": "outdoors", "metric": "coverage_pct", "model": "linear",
             "reference": "shade", "time_unit": "d", "hours": "10-14", "daily": "median",
             "bootstrap": 0, "experiments": entries}
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(study))
    result = run_study(path)
    rates = [r.fit.params["b"] for r in result.replicates]
    assert rates[:3] == pytest.approx([0.97, 1.0, 1.03], abs=0.08)
    assert rates[3:] == pytest.approx([1.94, 2.0, 2.06], abs=0.12)
    methods = (result.out_dir / "methods.md").read_text()
    assert "between 10 and 14 h" in methods and "reduced to their median" in methods


def test_report_cli_hours_daily(tmp_path):
    exp = outdoor(tmp_path / "x", days=4)
    res = CliRunner().invoke(app, ["report", str(exp.root), "--metric", "coverage_pct",
                                   "--hours", "10-14", "--daily", "p90", "--bootstrap", "0"])
    assert res.exit_code == 0, res.output
    assert "daily p90" in res.output and "4 days used" in res.output
    assert "outside 10-14 h" in res.output
    bad = CliRunner().invoke(app, ["report", str(exp.root), "--daily", "max"])
    assert bad.exit_code == 1


def test_timezone_is_checked():
    from fungus_cv.config import ExperimentConfig

    assert ExperimentConfig(name="x", timezone="Europe/Berlin").tzinfo() == BERLIN
    assert ExperimentConfig(name="x").tzinfo() is None
    with pytest.raises(ValueError, match="time zone"):
        ExperimentConfig(name="x", timezone="Mars/Olympus")
