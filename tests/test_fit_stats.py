"""Fit uncertainty: AICc and Akaike weights, reduced chi-square, AR(1) errors, bootstrap
intervals, and the lag / asymmetric growth models."""

import math

import numpy as np
import pytest
from scipy import stats
from typer.testing import CliRunner

from fungus_cv.analyze.fit import (
    MODELS,
    best_fit,
    durbin_watson,
    fit_all,
    fit_model,
    format_params,
)
from fungus_cv.cli import app

T = np.linspace(0, 20, 60)
TRUE = {"K": 80.0, "r": 0.6, "t_mid": 9.0}


def logistic_data(noise=1.0, seed=0, ar=0.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, noise, len(T))
    if ar:  # AR(1) errors: neighbouring frames share errors
        for i in range(1, len(e)):
            e[i] = ar * e[i - 1] + math.sqrt(1 - ar**2) * e[i]
    return MODELS["logistic"].func(T, *TRUE.values()) + e


def test_weights_pick_the_true_model_and_sum_to_one():
    y = logistic_data()
    fits = fit_all(T, y, models=("linear", "sqrt", "power", "logistic"), sigma=np.ones(len(T)),
                   errors="iid")
    assert sum(f.akaike_weight for f in fits if f.ok) == pytest.approx(1.0)
    assert best_fit(fits).model == "logistic"
    lg = next(f for f in fits if f.model == "logistic")
    k = 4  # three parameters + the residual variance
    assert lg.aicc == pytest.approx(lg.aic + 2 * k * (k + 1) / (lg.n - k - 1))
    assert lg.bic > lg.aic


def test_reduced_chi2_and_absolute_standard_errors():
    y = logistic_data(noise=1.0)
    right = fit_model("logistic", T, y, sigma=np.ones(len(T)), errors="iid")
    assert 0.6 < right.reduced_chi2 < 1.5 and not right.warnings
    # stderr is the absolute one scaled by the observed scatter.
    for p in right.params:
        assert right.stderr[p] == pytest.approx(
            right.stderr_absolute[p] * math.sqrt(right.reduced_chi2), rel=1e-3)
    too_small = fit_model("logistic", T, y, sigma=np.full(len(T), 0.2), errors="iid")
    assert too_small.reduced_chi2 > 3
    assert any("reduced chi-square" in w for w in too_small.warnings)
    unweighted = fit_model("logistic", T, y)
    assert not unweighted.weighted and math.isnan(unweighted.reduced_chi2)
    assert unweighted.stderr_absolute == {}


def test_bootstrap_interval_contains_truth_and_is_reproducible():
    y = logistic_data()
    a = fit_model("logistic", T, y, bootstrap=300, seed=3, errors="iid")
    b = fit_model("logistic", T, y, bootstrap=300, seed=3, errors="iid")
    assert a.ci95 == b.ci95
    assert a.bootstrap["n_ok"] >= 290 and a.error_model == "iid"
    assert a.bootstrap["method"] == "residual bootstrap"
    for p, true in TRUE.items():
        lo, hi = a.ci95[p]
        assert lo < a.params[p] < hi and lo < true < hi
    assert "[" in format_params(a) and "±" in format_params(a)


def test_correlated_errors_are_modelled_and_widen_the_intervals():
    """Correlated frame errors: treating frames as independent gives far too small errors.

    Coverage of the 95% interval for t_mid over 150 simulated datasets (60 frames, noise sd 2):
    independent errors 0.94 either way; AR(1) 0.5: 0.75 naive vs 0.90 here; AR(1) 0.9: 0.31
    vs 0.71 (few effectively independent frames; the warning says so).
    """
    y = logistic_data(noise=2.0, ar=0.8, seed=5)
    auto = fit_model("logistic", T, y, bootstrap=300, seed=1)
    naive = fit_model("logistic", T, y, errors="iid")
    assert auto.error_model == "ar1" and 0.5 < auto.ar1_phi < 0.95
    assert auto.aicc < naive.aicc
    assert auto.stderr["t_mid"] > 1.5 * naive.stderr["t_mid"]
    assert auto.bootstrap["method"] == "AR(1) sieve bootstrap"
    width = auto.ci95["t_mid"][1] - auto.ci95["t_mid"][0]
    assert width > 1.5 * 2 * 1.96 * naive.stderr["t_mid"]
    assert any("Durbin-Watson" in w for w in naive.warnings)

    independent = fit_model("logistic", T, logistic_data(noise=2.0, seed=5))
    assert independent.error_model == "iid" and independent.ar1_phi == 0
    assert 1.5 < independent.durbin_watson < 2.5
    forced = fit_model("logistic", T, logistic_data(noise=2.0, seed=5), errors="ar1")
    assert forced.error_model == "ar1" and forced.ar1_phi < 0.3


def test_ar1_handles_uneven_intervals():
    """Frames at irregular times: correlation decays with the real gap between them."""
    covered = chosen = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0, 20, 70))
        e = np.zeros(len(t))
        e[0] = rng.normal()
        for i in range(1, len(t)):
            rho = 0.5 ** ((t[i] - t[i - 1]) / 0.28)
            e[i] = rho * e[i - 1] + math.sqrt(1 - rho**2) * rng.normal()
        y = MODELS["logistic"].func(t, *TRUE.values()) + 2 * e
        fit = fit_model("logistic", t, y)
        chosen += fit.error_model == "ar1"
        covered += abs(fit.params["t_mid"] - 9) <= 1.96 * fit.stderr["t_mid"]
    assert chosen >= 30
    assert covered >= 33  # 38/40 at these seeds; treating frames as independent: 27/40


def test_durbin_watson_values():
    assert durbin_watson(np.array([1, -1, 1, -1, 1, -1.0])) > 3
    assert durbin_watson(np.array([1, 1, 1, -1, -1, -1.0])) < 1
    assert math.isnan(durbin_watson(np.zeros(5)))


def test_lag_models_recover_their_parameters():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 30, 80)
    y = MODELS["gompertz"].func(t, 50.0, 4.0, 6.0) + rng.normal(0, 0.3, len(t))
    g = fit_model("gompertz", t, y)
    assert g.ok
    assert g.params["A"] == pytest.approx(50, rel=0.02)
    assert g.params["mu"] == pytest.approx(4, rel=0.05)
    assert g.params["lam"] == pytest.approx(6, abs=0.3)

    y = MODELS["sqrt_lag"].func(t, 12.0, 4.0) + rng.normal(0, 0.2, len(t))
    s = fit_model("sqrt_lag", t, y)
    assert s.params["t_lag"] == pytest.approx(4.0, abs=0.3)
    assert s.params["k"] == pytest.approx(12.0, rel=0.03)

    r = fit_model("richards", T, logistic_data(noise=0.3))
    assert r.ok and r.params["nu"] == pytest.approx(1.0, abs=0.2)


def test_failed_fits_have_no_weight():
    fits = fit_all(T[:4], logistic_data()[:4], models=("linear", "logistic"))
    lg = next(f for f in fits if f.model == "logistic")
    assert not lg.ok and math.isnan(lg.akaike_weight)
    assert next(f for f in fits if f.model == "linear").akaike_weight == 1.0


def test_to_dict_is_valid_json():
    import json

    fit = fit_model("logistic", T, logistic_data(), bootstrap=20)
    json.dumps(fit.to_dict(), allow_nan=False)
    json.dumps(fit_model("logistic", T[:3], T[:3]).to_dict(), allow_nan=False)


def test_welch_helpers_match_scipy():
    from fungus_cv.analyze.study import describe, holm, random_effects, welch

    a, b = [0.81, 0.77, 0.86, 0.79], [0.52, 0.61, 0.49]
    w = welch(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert w["t"] == pytest.approx(ref.statistic) and w["p"] == pytest.approx(ref.pvalue)
    assert w["ci_low"] < w["diff"] < w["ci_high"] and w["hedges_g"] > 2
    assert math.isnan(welch([1.0], [2.0, 3.0])["p"])

    assert holm([0.01, 0.04, 0.03, float("nan")])[:3] == pytest.approx([0.03, 0.06, 0.06])

    d = describe([1.0, 2.0, 3.0])
    assert d["mean"] == 2 and d["sd"] == 1
    assert d["ci_high"] - d["mean"] == pytest.approx(stats.t.ppf(0.975, 2) / math.sqrt(3))

    same = random_effects([5.0, 5.0, 5.0], [1.0, 2.0, 1.0])
    assert same["re_mean"] == 5 and same["tau"] == 0
    spread = random_effects([1.0, 5.0, 9.0], [0.1, 0.1, 0.1])
    assert spread["re_mean"] == pytest.approx(5) and spread["tau"] > 3 and spread["i2"] > 0.9


def test_report_cli_models_and_bootstrap(experiment):
    from .test_pipeline import build_experiment

    build_experiment(experiment, minutes=range(1, 12), bump_at=-1)
    runner = CliRunner()
    assert runner.invoke(app, ["analyze", str(experiment.root)]).exit_code == 0
    result = runner.invoke(app, ["report", str(experiment.root), "--model", "sqrt",
                                 "--model", "sqrt_lag", "--bootstrap", "50"])
    assert result.exit_code == 0, result.output
    assert "sqrt_lag" in result.output and "weight=" in result.output
    assert "50 refits" in result.output and "AICc=" in result.output
    bad = runner.invoke(app, ["report", str(experiment.root), "--model", "cubic"])
    assert bad.exit_code == 1 and "unknown model" in bad.output
