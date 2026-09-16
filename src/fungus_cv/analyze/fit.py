"""Growth models fitted to a measurement over time."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import curve_fit


def _linear(t, a, b):
    return a + b * t


def _sqrt(t, k):
    return k * np.sqrt(np.clip(t, 0, None))


def _power(t, a, n):
    return a * np.power(np.clip(t, 1e-12, None), n)


def _logistic(t, K, r, t_mid):  # noqa: N803 - conventional symbol
    return K / (1 + np.exp(-r * (t - t_mid)))


@dataclass
class Model:
    name: str
    formula: str
    func: object
    params: tuple[str, ...]
    note: str = ""


MODELS = {
    "linear": Model("linear", "y = a + b*t", _linear, ("a", "b"), "constant speed"),
    "sqrt": Model("sqrt", "y = k*sqrt(t)", _sqrt, ("k",),
                  "Lucas-Washburn capillary wicking"),
    "power": Model("power", "y = a*t^n", _power, ("a", "n"),
                   "n = 0.5 is Lucas-Washburn; n = 1 is constant speed"),
    "logistic": Model("logistic", "y = K / (1 + exp(-r*(t - t_mid)))", _logistic,
                      ("K", "r", "t_mid"), "saturating spread (e.g. infection)"),
}


@dataclass
class FitResult:
    model: str
    formula: str
    ok: bool
    params: dict[str, float]
    stderr: dict[str, float]
    n: int
    r2: float = math.nan
    rmse: float = math.nan
    aic: float = math.nan
    message: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in asdict(self).items()}

    def predict(self, t: np.ndarray) -> np.ndarray:
        return MODELS[self.model].func(np.asarray(t, float), *self.params.values())


def _initial_guess(name: str, t: np.ndarray, y: np.ndarray) -> list[float]:
    span = max(float(t.max() - t.min()), 1e-9)
    if name == "linear":
        b, a = np.polyfit(t, y, 1)
        return [a, b]
    if name == "sqrt":
        s = np.sqrt(np.clip(t, 0, None))
        return [float(s @ y / max(s @ s, 1e-12))]
    if name == "power":
        good = (t > 0) & (y > 0)
        if good.sum() >= 2:
            n, log_a = np.polyfit(np.log(t[good]), np.log(y[good]), 1)
            return [float(np.exp(log_a)), float(n)]
        return [float(y.max()), 0.5]
    if name == "logistic":
        K = float(y.max()) * 1.1 or 1.0  # noqa: N806
        t_mid = float(t[np.argmin(np.abs(y - K / 2))])
        return [K, 4.0 / span, t_mid]
    raise KeyError(name)


def fit_model(name: str, t, y, sigma=None) -> FitResult:
    model = MODELS[name]
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    empty = {p: math.nan for p in model.params}
    if n < len(model.params) + 2:
        return FitResult(name, model.formula, False, empty, dict(empty), n,
                         message=f"need at least {len(model.params) + 2} points", note=model.note)
    use_sigma = sigma is not None and np.all(np.asarray(sigma) > 0)
    try:
        popt, pcov = curve_fit(
            model.func, t, y, p0=_initial_guess(name, t, y),
            sigma=np.asarray(sigma, float) if use_sigma else None,
            absolute_sigma=False, maxfev=20000,
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        return FitResult(name, model.formula, False, empty, dict(empty), n,
                         message=str(exc), note=model.note)

    residuals = y - model.func(t, *popt)
    rss = float(residuals @ residuals)
    tss = float(((y - y.mean()) ** 2).sum())
    k = len(popt)
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.diag(pcov))
    return FitResult(
        model=name,
        formula=model.formula,
        ok=True,
        params=dict(zip(model.params, map(float, popt))),
        stderr=dict(zip(model.params, map(float, se))),
        n=n,
        r2=1 - rss / tss if tss > 0 else math.nan,
        rmse=math.sqrt(rss / n),
        aic=n * math.log(max(rss, 1e-300) / n) + 2 * k,
        note=model.note,
    )


def fit_all(t, y, models=("linear", "sqrt", "power", "logistic"), sigma=None) -> list[FitResult]:
    return [fit_model(name, t, y, sigma) for name in models]
