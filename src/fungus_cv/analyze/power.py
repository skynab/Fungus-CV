"""How many replicates does a study need?

`fungus study` compares conditions with Welch's t-test on per-replicate estimates. This
simulates exactly that: replicates are drawn with the spread seen between replicates (typed
in, or taken from a pilot study), the test is run, and the share of simulated studies that
find the difference is the power. With several comparisons the Holm correction is applied as
in the study, conservatively as a Bonferroni threshold for a single comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass
class PowerRow:
    n: int  # replicates per condition
    power: float


@dataclass
class PowerResult:
    effect: float  # difference between the two condition means
    sd_a: float
    sd_b: float
    alpha: float
    comparisons: int
    target: float
    rows: list[PowerRow] = field(default_factory=list)
    source: str = ""  # where the spread came from
    reference_mean: float = math.nan

    @property
    def needed(self) -> int | None:
        """Smallest number of replicates per condition that reaches the target power."""
        return next((r.n for r in self.rows if r.power >= self.target), None)


def welch_power(n: int, effect: float, sd_a: float, sd_b: float, alpha: float = 0.05,
                comparisons: int = 1, sims: int = 20000, seed: int = 0) -> float:
    """Share of simulated studies where Welch's test finds the difference (two-sided)."""
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sd_a, (sims, n))
    b = rng.normal(effect, sd_b, (sims, n))
    va, vb = a.var(axis=1, ddof=1) / n, b.var(axis=1, ddof=1) / n
    se = np.sqrt(va + vb)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (b.mean(axis=1) - a.mean(axis=1)) / se
        df = (va + vb) ** 2 / (va**2 / (n - 1) + vb**2 / (n - 1))
    p = 2 * stats.t.sf(np.abs(t), df)
    threshold = alpha / max(comparisons, 1)  # Holm's strictest step; conservative
    return float(np.mean(p < threshold))


def power_table(effect: float, sd_a: float, sd_b: float | None = None, alpha: float = 0.05,
                comparisons: int = 1, target: float = 0.8, n_max: int = 20,
                sims: int = 20000, seed: int = 0) -> PowerResult:
    if effect == 0:
        raise ValueError("the difference to detect must not be zero")
    if sd_a <= 0 or (sd_b is not None and sd_b <= 0):
        raise ValueError("the spread between replicates must be positive")
    sd_b = sd_a if sd_b is None else sd_b
    result = PowerResult(abs(effect), sd_a, sd_b, alpha, comparisons, target)
    for n in range(2, n_max + 1):
        result.rows.append(PowerRow(n, welch_power(n, abs(effect), sd_a, sd_b, alpha,
                                                   comparisons, sims, seed)))
    return result


def detectable_effect(n: int, sd_a: float, sd_b: float | None = None, alpha: float = 0.05,
                      comparisons: int = 1, target: float = 0.8, sims: int = 20000,
                      seed: int = 0) -> float:
    """Smallest difference found with probability ``target`` using ``n`` replicates each."""
    sd_b = sd_a if sd_b is None else sd_b
    lo, hi = 0.0, 20 * max(sd_a, sd_b)
    if welch_power(n, hi, sd_a, sd_b, alpha, comparisons, sims, seed) < target:
        return math.inf  # e.g. n = 2 with many comparisons
    for _ in range(30):
        mid = (lo + hi) / 2
        if welch_power(n, mid, sd_a, sd_b, alpha, comparisons, sims, seed) >= target:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass
class PilotSpread:
    param: str
    sd_a: float
    sd_b: float
    mean_a: float
    mean_b: float
    condition_a: str
    condition_b: str
    n_a: int
    n_b: int


def spread_from_study(study_file: Path, param: str, condition_a: str | None = None,
                      condition_b: str | None = None) -> PilotSpread:
    """Between-replicate SDs from a pilot study that has already been run."""
    import csv

    from fungus_cv.analyze.study import load_study

    study = load_study(study_file)
    results = Path(study_file).parent / f"{Path(study_file).stem}_results" / "conditions.csv"
    if not results.exists():
        raise FileNotFoundError(f"{results} not found; run `fungus study {study_file}` first")
    with open(results, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["param"] == param]
    if not rows:
        raise ValueError(f"{param!r} was not compared in this study; add it to params")
    by = {r["condition"]: r for r in rows}
    names = list(by)
    a = condition_a or study.reference or names[0]
    b = condition_b or next((c for c in names if c != a), a)
    for name in (a, b):
        if name not in by:
            raise ValueError(f"condition {name!r} is not in {results}")
        if not by[name]["sd"]:
            raise ValueError(f"condition {name!r} has fewer than 2 usable replicates: no SD")
    return PilotSpread(param, float(by[a]["sd"]), float(by[b]["sd"]), float(by[a]["mean"]),
                       float(by[b]["mean"]), a, b, int(by[a]["n"]), int(by[b]["n"]))
