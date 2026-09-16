"""Growth models fitted to a measurement over time, with honest parameter uncertainty."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _linear(t, a, b):
    return a + b * t


def _sqrt(t, k):
    return k * np.sqrt(np.clip(t, 0, None))


def _sqrt_lag(t, k, t_lag):
    return k * np.sqrt(np.clip(t - t_lag, 0, None))


def _power(t, a, n):
    return a * np.power(np.clip(t, 1e-12, None), n)


def _logistic(t, K, r, t_mid):  # noqa: N803 - conventional symbol
    with np.errstate(over="ignore"):
        return K / (1 + np.exp(-r * (t - t_mid)))


def _gompertz(t, A, mu, lam):  # noqa: N803
    """Zwietering et al. (1990): A = plateau, mu = maximum rate, lam = lag time."""
    with np.errstate(over="ignore"):
        return A * np.exp(-np.exp(mu * math.e / A * (lam - t) + 1))


def _richards(t, K, r, t_mid, nu):  # noqa: N803
    """Generalised logistic; nu = 1 is the logistic, nu -> 0 approaches Gompertz."""
    with np.errstate(over="ignore", invalid="ignore"):
        return K / np.power(1 + nu * np.exp(-r * (t - t_mid)), 1 / nu)


@dataclass
class Model:
    name: str
    formula: str
    func: object
    params: tuple[str, ...]
    note: str = ""
    bounds: tuple = (-np.inf, np.inf)


MODELS = {
    "linear": Model("linear", "y = a + b*t", _linear, ("a", "b"), "constant speed"),
    "sqrt": Model("sqrt", "y = k*sqrt(t)", _sqrt, ("k",),
                  "Lucas-Washburn capillary wicking"),
    "sqrt_lag": Model("sqrt_lag", "y = k*sqrt(t - t_lag)", _sqrt_lag, ("k", "t_lag"),
                      "wicking with the start time estimated from the data"),
    "power": Model("power", "y = a*t^n", _power, ("a", "n"),
                   "n = 0.5 is Lucas-Washburn; n = 1 is constant speed"),
    "logistic": Model("logistic", "y = K / (1 + exp(-r*(t - t_mid)))", _logistic,
                      ("K", "r", "t_mid"), "saturating spread (e.g. infection)"),
    "gompertz": Model("gompertz", "y = A*exp(-exp(mu*e/A*(lam - t) + 1))", _gompertz,
                      ("A", "mu", "lam"),
                      "saturating growth with a lag: mu = maximum rate, lam = lag time"),
    "richards": Model("richards", "y = K / (1 + nu*exp(-r*(t - t_mid)))^(1/nu)", _richards,
                      ("K", "r", "t_mid", "nu"),
                      "asymmetric saturating growth; nu = 1 is the logistic",
                      bounds=([-np.inf, -np.inf, -np.inf, 1e-3], [np.inf, np.inf, np.inf, 50])),
}
DEFAULT_MODELS = ("linear", "sqrt", "power", "logistic")


@dataclass
class FitResult:
    model: str
    formula: str
    ok: bool
    params: dict[str, float]
    stderr: dict[str, float]  # scaled by the residual scatter (reduced chi-square)
    n: int
    r2: float = math.nan
    rmse: float = math.nan
    aic: float = math.nan  # K = parameters + 1 (residual variance) [+ 1 for AR(1) phi]
    aicc: float = math.nan  # small-sample AIC; use this to compare models
    bic: float = math.nan
    akaike_weight: float = math.nan  # share of evidence among the models fitted together
    weighted: bool = False
    # Uncertainties taken as exact (only when weighted). Trust these over ``stderr`` only if
    # reduced_chi2 is close to 1.
    stderr_absolute: dict[str, float] = field(default_factory=dict)
    reduced_chi2: float = math.nan  # weighted: ~1 = uncertainties explain the scatter
    durbin_watson: float = math.nan  # ~2 = independent residuals; < 1 = strongly correlated
    error_model: str = "iid"  # iid | ar1 (correlated errors between neighbouring frames)
    ar1_phi: float = 0.0  # correlation between frames one typical interval apart
    ci95: dict[str, list[float]] = field(default_factory=dict)  # bootstrap percentiles
    bootstrap: dict = field(default_factory=dict)  # n, n_ok, method, seed
    warnings: list[str] = field(default_factory=list)
    message: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return json_safe(asdict(self))

    def predict(self, t: np.ndarray) -> np.ndarray:
        return MODELS[self.model].func(np.asarray(t, float), *self.params.values())


def json_safe(value):
    """NaN -> None recursively, so the JSON is valid."""
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _initial_guess(name: str, t: np.ndarray, y: np.ndarray) -> list[float]:
    span = max(float(t.max() - t.min()), 1e-9)
    if name == "linear":
        b, a = np.polyfit(t, y, 1)
        return [a, b]
    if name == "sqrt":
        s = np.sqrt(np.clip(t, 0, None))
        return [float(s @ y / max(s @ s, 1e-12))]
    if name == "sqrt_lag":
        return [*_initial_guess("sqrt", t, y), float(t.min()) - 1e-3 * span]
    if name == "power":
        good = (t > 0) & (y > 0)
        if good.sum() >= 2:
            n, log_a = np.polyfit(np.log(t[good]), np.log(y[good]), 1)
            return [float(np.exp(log_a)), float(n)]
        return [float(y.max()), 0.5]
    if name in ("logistic", "richards"):
        K = float(y.max()) * 1.1 or 1.0  # noqa: N806
        t_mid = float(t[np.argmin(np.abs(y - K / 2))])
        return [K, 4.0 / span, t_mid] + ([1.0] if name == "richards" else [])
    if name == "gompertz":
        A = float(y.max()) * 1.05 or 1.0  # noqa: N806
        slopes = np.diff(y) / np.maximum(np.diff(t), 1e-12)
        i = int(np.argmax(slopes)) if len(slopes) else 0
        mu = float(slopes[i]) if len(slopes) and slopes[i] > 0 else A / span
        lam = float(t[i] - y[i] / mu) if len(slopes) else float(t.min())
        return [A, mu, lam]
    raise KeyError(name)


def durbin_watson(residuals: np.ndarray) -> float:
    e = np.asarray(residuals, float)
    ss = float(e @ e)
    return float(np.sum(np.diff(e) ** 2) / ss) if ss > 0 and len(e) > 1 else math.nan


def _lags(t: np.ndarray) -> np.ndarray:
    """Gaps between frames in units of the typical gap, so AR(1) handles uneven intervals."""
    dt = np.diff(t)
    typical = float(np.median(dt)) if len(dt) and np.median(dt) > 0 else 1.0
    return np.clip(dt / typical, 1e-6, None)


def _innovations(u: np.ndarray, phi: float, lags: np.ndarray) -> tuple[np.ndarray, float]:
    """Whiten standardized residuals under AR(1) errors with correlation ``phi`` per typical
    gap (``phi ** gap`` between two frames). Returns the innovations and sum log(1 - rho^2)."""
    if phi == 0:
        return u, 0.0
    rho = np.minimum(phi**lags, 1 - 1e-9)
    z = np.empty_like(u)
    z[0] = u[0]
    z[1:] = (u[1:] - rho * u[:-1]) / np.sqrt(1 - rho**2)
    return z, float(np.sum(np.log(1 - rho**2)))


@dataclass
class _Solution:
    params: np.ndarray
    jac: np.ndarray  # of the innovations with respect to the parameters
    z: np.ndarray  # innovations (= standardized residuals for independent errors)
    phi: float
    neg2_loglik: float  # up to a constant shared by every model on the same data


def _solve(model: Model, t, y, w, p0, phi: float, lags, max_nfev: int = 2000) -> _Solution:
    lower, upper = model.bounds
    bounded = not (np.all(np.isinf(lower)) and np.all(np.isinf(upper)))
    x0 = np.asarray(p0, float)
    if bounded:
        x0 = np.clip(x0, np.asarray(lower, float) + 1e-9, np.asarray(upper, float) - 1e-9)

    def residuals(p):
        with np.errstate(all="ignore"):
            z, _ = _innovations((y - model.func(t, *p)) / w, phi, lags)
        return np.where(np.isfinite(z), z, 1e12)

    res = least_squares(residuals, x0, method="trf" if bounded else "lm",
                        bounds=model.bounds if bounded else (-np.inf, np.inf),
                        max_nfev=max_nfev)
    z, logdet = _innovations((y - model.func(t, *res.x)) / w, phi, lags)
    n = len(y)
    return _Solution(res.x, res.jac, z, phi,
                     n * math.log(max(float(z @ z), 1e-300) / n) + logdet)


def _solve_ar1(model: Model, t, y, w, p0, lags) -> _Solution:
    """Profile the likelihood over the AR(1) correlation; parameters re-fitted for each."""
    start = {"p": np.asarray(p0, float)}

    def objective(phi):
        sol = _solve(model, t, y, w, start["p"], phi, lags)
        start["p"] = sol.params
        return sol.neg2_loglik

    best = minimize_scalar(objective, bounds=(0.0, 0.98), method="bounded",
                           options={"xatol": 1e-3})
    return _solve(model, t, y, w, start["p"], float(best.x), lags)


def _covariance(sol: _Solution) -> np.ndarray:
    jtj = sol.jac.T @ sol.jac
    try:
        if np.linalg.cond(jtj) > 1e14:
            raise np.linalg.LinAlgError
        return np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        return np.full(jtj.shape, np.inf)


def fit_model(name: str, t, y, sigma=None, bootstrap: int = 0, errors: str = "auto",
              seed: int = 0) -> FitResult:
    """Least-squares fit of one model, weighted by ``sigma`` (per-point standard uncertainty).

    ``errors``: ``iid`` treats frames as independent; ``ar1`` models correlation between
    neighbouring frames as a first-order autoregressive process (generalized least squares,
    as ``gnls`` with ``corAR1`` in R); ``auto`` fits both and keeps the one with lower AICc.
    Ignoring correlated errors makes standard errors and intervals far too narrow.

    ``bootstrap`` > 0 adds percentile 95% intervals from a model-based residual bootstrap:
    innovations are resampled and, for AR(1) errors, turned back into correlated errors.
    """
    if errors not in ("auto", "iid", "ar1"):
        raise ValueError(f"errors must be auto, iid or ar1, not {errors!r}")
    model = MODELS[name]
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    n = len(t)
    p = len(model.params)
    empty = {q: math.nan for q in model.params}
    use_sigma = sigma is not None and bool(np.all(np.asarray(sigma) > 0))
    w = np.asarray(sigma, float)[order] if use_sigma else np.ones(n)
    if n < p + 2:
        return FitResult(name, model.formula, False, empty, dict(empty), n,
                         message=f"need at least {p + 2} points", note=model.note)
    lags = _lags(t)
    try:
        iid = _solve(model, t, y, w, _initial_guess(name, t, y), 0.0, lags)
        if not np.all(np.isfinite(iid.params)):
            raise ValueError("fit did not converge to finite parameters")
        candidates = [(iid, p + 1)]
        if errors != "iid" and n >= p + 5:
            candidates.append((_solve_ar1(model, t, y, w, iid.params, lags), p + 2))
    except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
        return FitResult(name, model.formula, False, empty, dict(empty), n,
                         message=str(exc), note=model.note)

    def aicc(sol, k):
        aic = sol.neg2_loglik + 2 * k
        return aic, aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else math.nan

    if errors == "ar1" and len(candidates) > 1:
        sol, k = candidates[1]
    elif errors == "auto" and len(candidates) > 1:
        sol, k = min(candidates, key=lambda c: (aicc(*c)[1], c[1]))
    else:
        sol, k = candidates[0]
    aic, aicc_value = aicc(sol, k)

    popt = sol.params
    residuals = y - model.func(t, *popt)
    std_resid = residuals / w
    rss = float(residuals @ residuals)
    tss = float(((y - y.mean()) ** 2).sum())
    chi2 = float(sol.z @ sol.z)
    cov_abs = _covariance(sol)
    reduced = chi2 / (n - p) if n > p else math.nan
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.diag(cov_abs) * reduced)
        se_abs = np.sqrt(np.diag(cov_abs))
    result = FitResult(
        model=name, formula=model.formula, ok=True,
        params=dict(zip(model.params, map(float, popt))),
        stderr=dict(zip(model.params, map(float, se))),
        n=n, r2=1 - rss / tss if tss > 0 else math.nan, rmse=math.sqrt(rss / n),
        aic=aic, aicc=aicc_value, bic=sol.neg2_loglik + k * math.log(n),
        weighted=use_sigma, durbin_watson=durbin_watson(std_resid), note=model.note,
        error_model="ar1" if sol.phi > 0 or k == p + 2 else "iid", ar1_phi=sol.phi,
    )
    if use_sigma:
        result.reduced_chi2 = reduced
        result.stderr_absolute = dict(zip(model.params, map(float, se_abs)))
        if reduced > 3:
            result.warnings.append(
                f"reduced chi-square {reduced:.1f}: scatter is larger than the per-frame "
                "uncertainties explain (missing error source, or the model doesn't fit)")
    if not np.all(np.isfinite(se)):
        result.warnings.append("covariance could not be estimated: parameters may be "
                               "unidentifiable from these data")
    if result.error_model == "iid" and result.durbin_watson < 1:
        result.warnings.append(
            f"Durbin-Watson {result.durbin_watson:.2f}: residuals run in long streaks, so "
            "the model probably misses the shape of the curve; standard errors are too small")
    if result.error_model == "ar1" and sol.phi > 0.7:
        result.warnings.append(
            f"strongly correlated frame errors (AR(1) phi {sol.phi:.2f}): few effectively "
            "independent frames, so intervals may still be too narrow. Longer intervals "
            "between photos or more replicates help")

    if bootstrap > 0:
        samples = _residual_bootstrap(model, t, y - residuals, sol, w, lags, bootstrap,
                                      np.random.default_rng(seed))
        result.bootstrap = {"n": bootstrap, "n_ok": len(samples), "seed": seed,
                            "method": "AR(1) sieve bootstrap" if sol.phi > 0
                            else "residual bootstrap"}
        if len(samples) >= max(20, bootstrap // 2):
            lo, hi = np.percentile(np.array(samples), [2.5, 97.5], axis=0)
            result.ci95 = {q: [float(a), float(b)] for q, a, b in zip(model.params, lo, hi)}
        else:
            result.warnings.append(f"only {len(samples)}/{bootstrap} bootstrap refits "
                                   "converged; no intervals")
    return result


def _residual_bootstrap(model: Model, t, fitted, sol: _Solution, w, lags, n_boot: int,
                        rng) -> list[np.ndarray]:
    """Resample innovations, rebuild (correlated) errors, refit with the same error model."""
    n = len(t)
    z = sol.z - sol.z.mean()
    rho = np.minimum(sol.phi**lags, 1 - 1e-9) if sol.phi > 0 else np.zeros(n - 1)
    scale = np.sqrt(1 - rho**2)
    samples = []
    for _ in range(n_boot):
        draw = rng.choice(z, n)
        u = np.empty(n)
        u[0] = draw[0]
        for i in range(1, n):
            u[i] = rho[i - 1] * u[i - 1] + scale[i - 1] * draw[i]
        try:
            boot = _solve(model, t, fitted + u * w, w, sol.params, sol.phi, lags, max_nfev=500)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        if np.all(np.isfinite(boot.params)):
            samples.append(boot.params)
    return samples


def fit_all(t, y, models=DEFAULT_MODELS, sigma=None, bootstrap: int = 0, errors: str = "auto",
            seed: int = 0) -> list[FitResult]:
    fits = [fit_model(name, t, y, sigma, bootstrap, errors, seed) for name in models]
    scores = [f.aicc if math.isfinite(f.aicc) else f.aic for f in fits if f.ok]
    if scores:
        best = min(scores)
        rel = [math.exp(-(s - best) / 2) for s in scores]
        total = sum(rel)
        for f, r in zip((f for f in fits if f.ok), rel):
            f.akaike_weight = r / total
    return fits


def format_params(fit: FitResult) -> str:
    """``k=12.01±0.05 [11.9, 12.1]``: estimate, standard error, bootstrap 95% interval."""
    parts = []
    for name, value in fit.params.items():
        text = f"{name}={value:.4g}±{fit.stderr[name]:.2g}"
        if name in fit.ci95:
            lo, hi = fit.ci95[name]
            text += f" [{lo:.4g}, {hi:.4g}]"
        parts.append(text)
    return ", ".join(parts)


def best_fit(fits: list[FitResult]) -> FitResult | None:
    ok = [f for f in fits if f.ok]
    return max(ok, key=lambda f: f.akaike_weight) if ok else None
