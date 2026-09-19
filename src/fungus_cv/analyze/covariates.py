"""Environmental covariates: temperature, humidity, light... logged next to the camera.

Growth of fungi and moss depends strongly on temperature and moisture, so a difference
between replicates may be the weather rather than the treatment. This module:

- imports a data logger's CSV into the experiment (``covariates.csv``, times in UTC). Loggers
  write times in many ways; ISO times, day-first or month-first dates and separate date and
  time columns are understood, and times without a zone are taken as the experiment's
  ``timezone``. An ambiguous date such as 03/04 is refused rather than guessed;
- reads a covariate at any time (linear interpolation, never across a gap in the log);
- summarises it over a time window (time-weighted mean, min, max and how much of the window
  the log covers), e.g. the part of a run a growth curve was fitted to;
- relates a fitted parameter to a covariate across replicates (`meta_regression`).
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fungus_cv.storage import Experiment, iso_utc, parse_iso_utc

FILE_NAME = "covariates.csv"
TIME_WORDS = ("timestamp", "datetime", "date_time", "time", "date", "zeit", "datum")
DAY_FIRST = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
             "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M")
MONTH_FIRST = ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y %I:%M:%S %p",
               "%m/%d/%Y %I:%M %p", "%m/%d/%y %H:%M:%S", "%m/%d/%y %H:%M")


def safe_name(name: str) -> str:
    """``Temperature (°C)`` -> ``temperature_c``: a column name usable everywhere."""
    text = name.strip().lower().replace("°", "").replace("%", "pct")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "value"


# --- reading a logger file -----------------------------------------------------------------


@dataclass
class ImportSummary:
    path: Path
    rows: int
    columns: list[str]
    first_utc: str
    last_utc: str
    time_format: str
    skipped: list[str] = field(default_factory=list)


def _delimiter(text: str) -> str:
    """The separator that splits the last lines (the data, not a title line) into the same
    number of cells. Tab and semicolon win over comma, which may be a decimal comma."""
    from collections import Counter

    lines = [ln for ln in text.splitlines() if ln.strip()][-30:]
    for candidate in ("\t", ";", ","):
        count, n = Counter(ln.count(candidate) for ln in lines).most_common(1)[0]
        if count >= 1 and n >= 0.7 * len(lines):
            return candidate
    return ","


def _choose_format(samples: list[str]) -> str:
    """ISO if every sample parses as ISO, otherwise a day- or month-first pattern that fits
    every sample; refuses when both fit (every day <= 12)."""
    def fits(fmt: str) -> bool:
        try:
            for s in samples:
                datetime.strptime(s, fmt)
            return True
        except ValueError:
            return False

    try:
        for s in samples:
            datetime.fromisoformat(s.replace("Z", "+00:00"))
        return "iso"
    except ValueError:
        pass
    day = next((f for f in DAY_FIRST if fits(f)), None)
    month = next((f for f in MONTH_FIRST if fits(f)), None)
    if day and month:
        raise ValueError(f"dates like {samples[0]!r} could be day-first or month-first; give "
                         f"the format, e.g. --time-format '{day}' or '{month}'")
    if day or month:
        return day or month
    raise ValueError(f"cannot read times like {samples[0]!r}; give --time-format "
                     "(strptime codes, e.g. '%d/%m/%Y %H:%M')")


def _parse_time(text: str, fmt: str, tz) -> datetime:
    if fmt == "iso":
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    else:
        dt = datetime.strptime(text, fmt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz) if tz is not None else dt.astimezone()
    return dt.astimezone(timezone.utc)


def _number(text: str, decimal_comma: bool) -> float:
    text = text.strip()
    if decimal_comma:
        text = text.replace(".", "").replace(",", ".") if text.count(",") == 1 else text
    try:
        return float(text)
    except ValueError:
        return math.nan


def read_log(path: Path, time_column: str | None = None, time_format: str | None = None,
             columns: list[str] | None = None, tz=None) -> tuple[list[datetime], dict, str]:
    """Times (UTC) and numeric columns of a logger CSV. ``time_column`` may name two columns
    joined by a comma (``Date,Time``)."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    delimiter = _delimiter(text)
    reader = csv.reader(text.splitlines(), delimiter=delimiter)
    rows = [r for r in reader if any(c.strip() for c in r)]
    # Some loggers write a title line first: the header is the first row with 2+ cells.
    while rows and len([c for c in rows[0] if c.strip()]) < 2:
        rows.pop(0)
    if len(rows) < 2:
        raise ValueError(f"{path}: no data rows")
    header = [h.strip() for h in rows[0]]
    data = rows[1:]
    lower = [h.lower() for h in header]
    if time_column:
        parts = [p.strip() for p in time_column.split(",")]
        missing = [p for p in parts if p not in header]
        if missing:
            raise ValueError(f"no column(s) {missing} in {header}")
        time_idx = [header.index(p) for p in parts]
    else:
        date_i = next((i for i, h in enumerate(lower) if h in ("date", "datum")), None)
        time_i = next((i for i, h in enumerate(lower) if h in ("time", "zeit")), None)
        if date_i is not None and time_i is not None:
            time_idx = [date_i, time_i]
        else:
            found = next((i for i, h in enumerate(lower) if any(w in h for w in TIME_WORDS)),
                         0)
            time_idx = [found]
    stamps_text = [" ".join(r[i].strip() for i in time_idx) for r in data
                   if len(r) > max(time_idx)]
    fmt = time_format or _choose_format(stamps_text[:200] + stamps_text[-50:])
    decimal_comma = delimiter == ";"

    wanted = [i for i, h in enumerate(header) if i not in time_idx
              and (not columns or h in columns or safe_name(h) in columns)]
    if columns:
        names = {header[i] for i in wanted} | {safe_name(header[i]) for i in wanted}
        unknown = [c for c in columns if c not in names]
        if unknown:
            raise ValueError(f"no column(s) {unknown} in {header}")
    times, values = [], {safe_name(header[i]): [] for i in wanted}
    for r in data:
        if len(r) <= max(time_idx):
            continue
        try:
            times.append(_parse_time(" ".join(r[i].strip() for i in time_idx), fmt, tz))
        except ValueError:
            continue
        for i in wanted:
            values[safe_name(header[i])].append(_number(r[i], decimal_comma)
                                                if i < len(r) else math.nan)
    # Keep columns that are mostly numbers (text columns such as a logger's serial are not
    # covariates).
    keep = {k: v for k, v in values.items() if np.isfinite(v).sum() >= max(1, 0.5 * len(v))}
    if not keep:
        raise ValueError(f"{path}: no numeric columns besides the time")
    return times, keep, fmt


def import_log(experiment: Experiment, path: Path, time_column: str | None = None,
               time_format: str | None = None, columns: list[str] | None = None,
               prefix: str = "") -> ImportSummary:
    """Merge a logger file into ``covariates.csv`` (a time already there is updated)."""
    times, values, fmt = read_log(path, time_column, time_format, columns,
                                  experiment.config.tzinfo())
    if not times:
        raise ValueError(f"{path}: no rows with a readable time")
    values = {f"{prefix}{k}": v for k, v in values.items()}
    existing = load(experiment)
    table: dict[str, dict[str, float]] = {}
    for i, ts in enumerate(existing.times):
        key = iso_utc(datetime.fromtimestamp(ts, timezone.utc))
        table[key] = {n: existing.values[n][i] for n in existing.names}
    for i, t in enumerate(times):
        row = table.setdefault(iso_utc(t), {})
        for name, column in values.items():
            if math.isfinite(column[i]):
                row[name] = column[i]
    names = list(dict.fromkeys([*existing.names, *values]))
    out = experiment.root / FILE_NAME
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", *names])
        for key in sorted(table):
            row = table[key]
            writer.writerow([key, *("" if not math.isfinite(row.get(n, math.nan))
                                    else repr(float(row[n])) for n in names)])
    return ImportSummary(out, len(times), list(values), iso_utc(min(times)), iso_utc(max(times)),
                         fmt)


# --- using them ----------------------------------------------------------------------------


@dataclass
class Covariates:
    times: np.ndarray  # epoch seconds, sorted
    values: dict[str, np.ndarray]

    @property
    def names(self) -> list[str]:
        return list(self.values)

    def series(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Logged times and values of one covariate (its missing rows dropped)."""
        if name not in self.values:
            raise KeyError(f"no covariate {name!r}; have {self.names}")
        v = self.values[name]
        ok = np.isfinite(v)
        return self.times[ok], v[ok]

    def max_gap(self, name: str) -> float:
        """Longest step interpolation may bridge: 3 typical logging intervals."""
        t, _ = self.series(name)
        return 3 * float(np.median(np.diff(t))) if len(t) > 1 else 0.0

    def at(self, name: str, times) -> np.ndarray:
        """Linear interpolation; NaN outside the log or inside a gap in it."""
        t, v = self.series(name)
        times = np.asarray(times, float)
        out = np.full(times.shape, math.nan)
        if len(t) == 0:
            return out
        if len(t) == 1:
            out[times == t[0]] = v[0]
            return out
        inside = (times >= t[0]) & (times <= t[-1])
        out[inside] = np.interp(times[inside], t, v)
        right = np.clip(np.searchsorted(t, times, side="left"), 1, len(t) - 1)
        gap = (t[right] - t[right - 1]) > self.max_gap(name)
        on_sample = np.isin(times, t)
        out[inside & gap & ~on_sample] = math.nan
        return out

    def summarize(self, name: str, start: float, end: float) -> dict:
        """Time-weighted mean, min and max over [start, end] (epoch seconds), and the share
        of the window the log covers."""
        out = {"mean": math.nan, "min": math.nan, "max": math.nan, "coverage": 0.0}
        t, v = self.series(name)
        if end <= start or len(t) == 0:
            return out
        grid = np.unique(np.concatenate([[start, end], t[(t > start) & (t < end)]]))
        y = self.at(name, grid)
        ok = np.isfinite(y[:-1]) & np.isfinite(y[1:])
        widths = np.diff(grid)
        covered = float(widths[ok].sum())
        out["coverage"] = covered / (end - start)
        if covered > 0:
            area = float((0.5 * (y[:-1] + y[1:]) * widths)[ok].sum())
            finite = y[np.isfinite(y)]
            out.update(mean=area / covered, min=float(finite.min()), max=float(finite.max()))
        return out


def path_for(experiment: Experiment) -> Path:
    return experiment.root / FILE_NAME


def load(experiment: Experiment) -> Covariates:
    path = path_for(experiment)
    if not path.exists():
        return Covariates(np.array([]), {})
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return Covariates(np.array([]), {})
    names = [k for k in rows[0] if k != "timestamp_utc"]
    times = np.array([parse_iso_utc(r["timestamp_utc"]).timestamp() for r in rows])
    order = np.argsort(times)
    values = {n: np.array([float(r[n]) if r[n] not in ("", None) else math.nan
                           for r in rows])[order] for n in names}
    return Covariates(times[order], values)


# --- relating a parameter to a covariate across replicates --------------------------------


def meta_regression(y, se, x, groups: list[str] | None = None) -> dict:
    """Random-effects meta-regression of replicate estimates ``y`` (standard errors ``se``)
    on a covariate ``x``, optionally adjusted for condition (``groups``).

    The between-replicate variance tau^2 is the method-of-moments (DerSimonian-Laird type)
    estimate for meta-regression; the slope's interval uses the Knapp-Hartung variance
    (with the conservative truncation at 1) and a t distribution with k - p degrees of
    freedom, which keeps coverage near 95% with few replicates. Without usable standard
    errors every replicate weighs the same (ordinary least squares).
    """
    from scipy import stats

    y, se, x = (np.asarray(a, float) for a in (y, se, x))
    ok = np.isfinite(y) & np.isfinite(x)
    out = {"n": int(ok.sum()), "slope": math.nan, "slope_se": math.nan,
           "ci_low": math.nan, "ci_high": math.nan, "p": math.nan, "tau": math.nan,
           "df": math.nan, "weighted": False, "x_range": [math.nan, math.nan]}
    y, se, x = y[ok], se[ok], x[ok]
    cols = [np.ones_like(x), x]
    if groups is not None:
        g = [groups[i] for i in np.flatnonzero(ok)]
        for level in sorted(set(g))[1:]:
            cols.append(np.array([1.0 if gi == level else 0.0 for gi in g]))
    design = np.column_stack(cols)
    k, p = design.shape
    if k - p < 1 or np.ptp(x) == 0 or np.linalg.matrix_rank(design) < p:
        return out
    weighted = bool(np.all(np.isfinite(se)) and np.all(se > 0))
    v = se**2 if weighted else np.zeros(k)

    def wls(w):
        xtw = design.T * w
        cov = np.linalg.inv(xtw @ design)
        beta = cov @ xtw @ y
        return beta, cov

    if weighted:
        w = 1 / v
        beta, cov = wls(w)
        resid = y - design @ beta
        q = float(w @ resid**2)
        trace = float(w.sum() - np.trace(cov @ (design.T * w**2) @ design))
        tau2 = max(0.0, (q - (k - p)) / trace) if trace > 0 else 0.0
    else:
        beta, _ = wls(np.ones(k))
        tau2 = float(np.sum((y - design @ beta) ** 2) / (k - p))
    w_star = 1 / (v + tau2) if np.all(v + tau2 > 0) else np.ones(k)
    beta, cov = wls(w_star)
    resid = y - design @ beta
    kh = float(w_star @ resid**2) / (k - p)
    # Unweighted, tau^2 is the residual variance, so this is ordinary least squares already.
    scale = max(1.0, kh) if weighted else 1.0
    slope_se = math.sqrt(cov[1, 1] * scale)
    df = k - p
    tcrit = float(stats.t.ppf(0.975, df))
    slope = float(beta[1])
    tval = slope / slope_se if slope_se > 0 else math.nan
    out.update(slope=slope, slope_se=slope_se, ci_low=slope - tcrit * slope_se,
               ci_high=slope + tcrit * slope_se,
               p=float(2 * stats.t.sf(abs(tval), df)) if math.isfinite(tval) else math.nan,
               tau=math.sqrt(tau2), df=df, weighted=weighted,
               x_range=[float(x.min()), float(x.max())])
    return out
