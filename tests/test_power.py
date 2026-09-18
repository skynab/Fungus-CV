"""Replicate planning: power of the study's Welch test, checked against theory."""

import math

import numpy as np
import pytest
from scipy import stats
from typer.testing import CliRunner

from fungus_cv.analyze import power
from fungus_cv.cli import app


def student_power(n, effect, sd, alpha=0.05):
    df, nc = 2 * n - 2, effect / (sd * math.sqrt(2 / n))
    crit = stats.t.ppf(1 - alpha / 2, df)
    return stats.nct.sf(crit, df, nc) + stats.nct.cdf(-crit, df, nc)


def test_matches_theory_when_spreads_are_equal():
    # Welch gives up a little power at small n (it doesn't assume equal spreads) and
    # converges to the textbook Student t result as replicates increase.
    for n, tolerance in ((20, 0.02), (40, 0.015)):
        assert power.welch_power(n, 1.0, 1.0, 1.0) == pytest.approx(
            student_power(n, 1.0, 1.0), abs=tolerance)
    assert power.welch_power(5, 1.0, 1.0, 1.0) <= student_power(5, 1.0, 1.0)
    assert power.welch_power(10, 0.0, 1.0, 1.0) == pytest.approx(0.05, abs=0.01)  # alpha


def test_table_needed_n_and_detectable_effect():
    result = power.power_table(effect=1.0, sd_a=1.0, n_max=25)
    powers = [r.power for r in result.rows]
    assert all(b >= a - 0.01 for a, b in zip(powers, powers[1:]))  # more replicates help
    assert result.needed in (17, 18)  # a one-SD difference: about 17 per condition
    smallest = power.detectable_effect(result.needed, 1.0)
    assert smallest == pytest.approx(1.0, rel=0.08)
    assert power.detectable_effect(2, 1.0, comparisons=10) == math.inf or \
        power.detectable_effect(2, 1.0, comparisons=10) > 5

    # More comparisons need more replicates; unequal spreads need more than equal ones.
    assert power.power_table(1.0, 1.0, comparisons=3, n_max=30).needed > result.needed
    assert power.power_table(1.0, 1.0, sd_b=2.0, n_max=70).needed > result.needed
    with pytest.raises(ValueError):
        power.power_table(0.0, 1.0)
    with pytest.raises(ValueError):
        power.power_table(1.0, 0.0)


def pilot(tmp_path):
    from fungus_cv.analyze.study import run_study

    from .test_study import fake_experiment, write_study

    for i in range(1, 4):
        fake_experiment(tmp_path, f"control{i}", 70 + i, 0.20 + 0.01 * i, 24, seed=i)
        fake_experiment(tmp_path, f"treated{i}", 70 + i, 0.30 + 0.02 * i, 24, seed=10 + i)
    path = write_study(tmp_path, params=["r", "max_rate"], bootstrap=0)
    run_study(path)
    return path


def test_spread_from_a_pilot_study(tmp_path):
    path = pilot(tmp_path)
    spread = power.spread_from_study(path, "r")
    assert spread.condition_a == "control" and spread.condition_b == "treated"
    assert spread.n_a == 3 and spread.sd_a == pytest.approx(0.01, rel=0.3)
    assert spread.sd_b == pytest.approx(0.02, rel=0.3)
    assert spread.mean_a == pytest.approx(0.22, abs=0.01)
    with pytest.raises(ValueError, match="was not compared"):
        power.spread_from_study(path, "K")
    from .test_study import write_study

    not_run = tmp_path / "later"
    not_run.mkdir()
    with pytest.raises(FileNotFoundError, match="run `fungus study"):
        power.spread_from_study(write_study(not_run), "r")


def test_power_cli(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["power", "--effect", "1", "--sd", "1", "--n-max", "20",
                                 "--n", "5"])
    assert result.exit_code == 0, result.output
    assert "<- enough" in result.output and "replicates per condition give 80% power" in \
        result.output
    assert "smallest difference found 80% of the time" in result.output

    path = pilot(tmp_path)
    from_pilot = runner.invoke(app, ["power", "--study", str(path), "--param", "r",
                                     "--effect", "10%", "--n-max", "15"])
    assert from_pilot.exit_code == 0, from_pilot.output
    assert "pilot moss.yaml" in from_pilot.output or "pilot study.yaml" in from_pilot.output
    assert "observed difference" in from_pilot.output

    bad = runner.invoke(app, ["power", "--effect", "10%", "--sd", "1"])
    assert bad.exit_code == 1 and "needs --study" in bad.output
    assert runner.invoke(app, ["power", "--effect", "1"]).exit_code == 1


def test_simulation_is_reproducible():
    a = power.welch_power(6, 0.8, 1.0, 1.3, seed=4)
    b = power.welch_power(6, 0.8, 1.0, 1.3, seed=4)
    assert a == b and 0 < a < 1
    assert np.isclose(power.welch_power(1, 1.0, 1.0, 1.0), 0.0)
