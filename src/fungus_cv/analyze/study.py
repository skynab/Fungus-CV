"""Studies: fit replicates, summarise conditions and compare them.

The analysis is two-stage, the standard approach for replicated growth curves:

1. Each replicate (one experiment, or one plot of one) is fitted on its own with the same
   model, weighted by the per-frame uncertainties, with independent or AR(1) frame errors
   and bootstrap intervals.
2. Its parameter estimates (e.g. the logistic rate ``r``) are the data for between-condition
   statistics, so the replicate is the unit of inference and frames of one time-lapse are
   never treated as independent samples.

Conditions are summarised by mean, SD and a t-based 95% CI of replicate estimates, and by a
random-effects (DerSimonian-Laird) mean that also uses each replicate's own uncertainty.
Conditions are compared with Welch's t-test, Holm-adjusted within each parameter.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats

from fungus_cv.analyze.fit import MODELS, FitResult, fit_all, json_safe
from fungus_cv.analyze.pipeline import RESULTS_DIR
from fungus_cv.analyze.report import (
    DEFAULT_EXCLUDE,
    INK,
    INK_2,
    SERIES,
    TIME_UNITS,
    Series,
    _style,
    load_series,
    pick_time_unit,
)
from fungus_cv.provenance import software_provenance
from fungus_cv.storage import Experiment, iso_utc, utc_now

log = logging.getLogger(__name__)

# Settings that must match across replicates, or differences could come from the analysis.
COMPARED_SETTINGS = ("align", "rectify", "lighting", "target", "measure", "uncertainty",
                     "front_percentile")


class StudyExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str  # experiment folder, relative to the study file
    condition: str
    replicate: str | None = None  # default: position within its condition
    plot: str | None = None  # for field experiments with several plots
    t0: datetime | None = None  # start of this replicate (e.g. inoculation); default first frame


class Study(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    metric: str | None = None  # default: as in `fungus report`
    model: str  # model whose parameters are compared
    also_fit: list[str] = Field(default_factory=list)  # other models, for model selection only
    time_unit: str = "auto"
    reference: str | None = None  # compare every condition with this one; None = all pairs
    params: list[str] | None = None  # default: every parameter of the model
    exclude_flags: list[str] = Field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    exclude_jumps: bool = False
    bootstrap: int = Field(1000, ge=0)
    errors: str = "auto"  # auto | iid | ar1: are frame errors correlated in time?
    seed: int = 0
    experiments: list[StudyExperiment]

    @model_validator(mode="after")
    def _check(self) -> Study:
        for name in [self.model, *self.also_fit]:
            if name not in MODELS:
                raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
        wrong = [p for p in self.params or [] if p not in MODELS[self.model].params]
        if wrong:
            raise ValueError(f"{self.model} has no parameter(s) {wrong}; it has "
                             f"{list(MODELS[self.model].params)}")
        if self.errors not in ("auto", "iid", "ar1"):
            raise ValueError("errors must be auto, iid or ar1")
        if self.time_unit != "auto" and self.time_unit not in TIME_UNITS:
            raise ValueError(f"time_unit must be auto or one of {list(TIME_UNITS)}")
        conditions = {e.condition for e in self.experiments}
        if self.reference is not None and self.reference not in conditions:
            raise ValueError(f"reference condition {self.reference!r} is not among "
                             f"{sorted(conditions)}")
        keys = [(e.condition, e.replicate) for e in self.experiments if e.replicate is not None]
        if len(keys) != len(set(keys)):
            raise ValueError("replicate names must be unique within a condition")
        return self

    @property
    def compared_params(self) -> list[str]:
        return list(self.params or MODELS[self.model].params)


def load_study(path: Path) -> Study:
    with open(path, encoding="utf-8") as f:
        return Study.model_validate(yaml.safe_load(f) or {})


# --- statistics ----------------------------------------------------------------------------


def describe(values: list[float]) -> dict:
    x = np.asarray(values, float)
    n = len(x)
    out = {"n": n, "mean": math.nan, "sd": math.nan, "se": math.nan, "ci_low": math.nan,
           "ci_high": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    if n == 0:
        return out
    out.update(mean=float(x.mean()), median=float(np.median(x)), min=float(x.min()),
               max=float(x.max()))
    if n >= 2:
        sd = float(x.std(ddof=1))
        se = sd / math.sqrt(n)
        half = float(stats.t.ppf(0.975, n - 1)) * se
        out.update(sd=sd, se=se, ci_low=out["mean"] - half, ci_high=out["mean"] + half)
    return out


def random_effects(values: list[float], ses: list[float]) -> dict:
    """DerSimonian-Laird random-effects mean: weights each replicate by its own uncertainty
    plus the between-replicate variance ``tau^2`` estimated from the data."""
    x, se = np.asarray(values, float), np.asarray(ses, float)
    out = {"re_mean": math.nan, "re_se": math.nan, "re_ci_low": math.nan,
           "re_ci_high": math.nan, "tau": math.nan, "i2": math.nan}
    ok = np.isfinite(x) & np.isfinite(se) & (se > 0)
    x, se = x[ok], se[ok]
    k = len(x)
    if k < 2:
        return out
    w = 1 / se**2
    fixed = float(w @ x / w.sum())
    q = float(w @ (x - fixed) ** 2)
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    w_re = 1 / (se**2 + tau2)
    mean = float(w_re @ x / w_re.sum())
    se_re = math.sqrt(1 / w_re.sum())
    out.update(re_mean=mean, re_se=se_re, re_ci_low=mean - 1.959964 * se_re,
               re_ci_high=mean + 1.959964 * se_re, tau=math.sqrt(tau2),
               i2=max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0)
    return out


def welch(a: list[float], b: list[float]) -> dict:
    """Difference of means (a - b) with Welch's t-test and its 95% CI, plus Hedges' g."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    out = {"diff": math.nan, "ci_low": math.nan, "ci_high": math.nan, "t": math.nan,
           "df": math.nan, "p": math.nan, "hedges_g": math.nan}
    if len(x) == 0 or len(y) == 0:
        return out
    out["diff"] = float(x.mean() - y.mean())
    if len(x) < 2 or len(y) < 2:
        return out
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    if se == 0:
        return out
    df = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    t = out["diff"] / se
    half = float(stats.t.ppf(0.975, df)) * se
    pooled = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1))
                       / (len(x) + len(y) - 2))
    correction = 1 - 3 / (4 * (len(x) + len(y)) - 9)
    out.update(ci_low=out["diff"] - half, ci_high=out["diff"] + half, t=float(t), df=float(df),
               p=float(2 * stats.t.sf(abs(t), df)),
               hedges_g=float(out["diff"] / pooled * correction) if pooled > 0 else math.nan)
    return out


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p_values, float)
    adjusted = np.full(len(p), math.nan)
    idx = [i for i in np.argsort(p) if np.isfinite(p[i])]
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


# --- running a study -----------------------------------------------------------------------


@dataclass
class Replicate:
    condition: str
    replicate: str
    experiment: Path
    plot: str
    series: Series | None = None
    t: np.ndarray | None = None  # in the study's time unit
    fits: list[FitResult] = field(default_factory=list)
    settings_hash: str = ""
    settings: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def fit(self) -> FitResult | None:
        return self.fits[0] if self.fits and self.fits[0].ok else None

    def param_se(self, name: str) -> float:
        """Standard uncertainty of a parameter: from the bootstrap interval when there is one
        (robust to correlated residuals), otherwise the fit's standard error."""
        fit = self.fit
        if fit is None:
            return math.nan
        if name in fit.ci95:
            lo, hi = fit.ci95[name]
            return (hi - lo) / (2 * 1.959964)
        return fit.stderr.get(name, math.nan)


@dataclass
class StudyResult:
    study: Study
    metric: str
    time_unit: str
    replicates: list[Replicate]
    conditions: list[dict]  # one row per condition x parameter
    comparisons: list[dict]  # one row per pair x parameter
    model_selection: list[dict]
    warnings: list[str]
    out_dir: Path
    files: list[Path] = field(default_factory=list)


def _run_settings(experiment_root: Path, settings_hash: str) -> dict:
    path = experiment_root / RESULTS_DIR / "runs" / f"{settings_hash}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("settings", {})


def _comparable(settings: dict) -> dict:
    analysis = settings.get("analysis", {})
    out = {k: analysis.get(k) for k in COMPARED_SETTINGS}
    seg = settings.get("segmenter") or {}
    out["segmenter"] = {k: seg.get(k) for k in ("method", "model", "weights_sha256") if k in seg}
    return out


def run_study(study_path: Path, out_dir: Path | None = None) -> StudyResult:
    study_path = Path(study_path)
    study = load_study(study_path)
    root = study_path.parent
    out_dir = out_dir or root / f"{study_path.stem}_results"
    warnings: list[str] = []

    counters: dict[str, int] = {}
    replicates: list[Replicate] = []
    metric = study.metric
    for spec in study.experiments:
        counters[spec.condition] = counters.get(spec.condition, 0) + 1
        rep = Replicate(spec.condition, spec.replicate or str(counters[spec.condition]),
                        (root / spec.path).resolve(), spec.plot or "")
        replicates.append(rep)
        t0 = spec.t0
        if t0 is not None and t0.tzinfo is None:
            t0 = t0.astimezone()
        try:
            experiment = Experiment(rep.experiment)
            rep.series = load_series(experiment, metric, spec.plot, t0, "s",
                                     tuple(study.exclude_flags), study.exclude_jumps)
        except (FileNotFoundError, ValueError) as exc:
            rep.error = str(exc)
            warnings.append(f"{rep.condition}/{rep.replicate}: {exc}")
            continue
        metric = metric or rep.series.metric  # the first replicate decides for all
        rep.plot = rep.series.plot
        hashes = {r["settings_hash"] for r in rep.series.rows}
        rep.settings_hash = ",".join(sorted(hashes))
        if len(hashes) == 1:
            rep.settings = _run_settings(rep.experiment, rep.series.rows[0]["settings_hash"])

    loaded = [r for r in replicates if r.series is not None]
    if not loaded:
        raise ValueError("no replicate could be loaded: " + "; ".join(warnings))
    time_unit = study.time_unit
    if time_unit == "auto":
        time_unit = pick_time_unit(max(r.series.t[r.series.use].max(initial=0) for r in loaded))

    models = (study.model, *[m for m in study.also_fit if m != study.model])
    for rep in loaded:
        s = rep.series
        rep.t = s.t / TIME_UNITS[time_unit]
        use = s.use
        rep.fits = fit_all(rep.t[use], s.y[use], models=models, sigma=s.sigma,
                           bootstrap=study.bootstrap, errors=study.errors,
                           seed=study.seed)
        if rep.fit is None:
            message = rep.fits[0].message if rep.fits else "no fit"
            warnings.append(f"{rep.condition}/{rep.replicate}: {study.model} fit failed "
                            f"({message}); left out of the statistics")
        else:
            warnings.extend(f"{rep.condition}/{rep.replicate}: {w}" for w in rep.fit.warnings)

    reference_settings = next((r for r in loaded if r.settings), None)
    if reference_settings is not None:
        base = _comparable(reference_settings.settings)
        for rep in loaded:
            if not rep.settings:
                warnings.append(f"{rep.condition}/{rep.replicate}: analysis settings not found "
                                f"(settings hash {rep.settings_hash or 'mixed'})")
                continue
            other = _comparable(rep.settings)
            differ = [k for k in base if base[k] != other[k]]
            if differ:
                warnings.append(
                    f"{rep.condition}/{rep.replicate} was analyzed with different "
                    f"{', '.join(differ)} than {reference_settings.condition}/"
                    f"{reference_settings.replicate}: differences between conditions may come "
                    "from the analysis")

    condition_names = list(dict.fromkeys(r.condition for r in replicates))
    conditions, comparisons = [], []
    for param in study.compared_params:
        values = {c: [r.fit.params[param] for r in loaded if r.condition == c and r.fit]
                  for c in condition_names}
        for c in condition_names:
            reps = [r for r in loaded if r.condition == c and r.fit]
            row = {"condition": c, "param": param, **describe(values[c]),
                   **random_effects(values[c], [r.param_se(param) for r in reps])}
            if row["n"] < 2:
                warnings.append(f"{c}: only {row['n']} usable replicate(s) for {param}; no SD "
                                "or confidence interval")
            conditions.append(row)
        if study.reference is not None:
            pairs = [(c, study.reference) for c in condition_names if c != study.reference]
        else:
            pairs = list(combinations(condition_names, 2))
        rows = [{"param": param, "condition": a, "versus": b, **welch(values[a], values[b])}
                for a, b in pairs]
        for row, p_adj in zip(rows, holm([r["p"] for r in rows])):
            row["p_holm"] = p_adj
        comparisons.extend(rows)

    model_selection = _model_selection(loaded, models)
    result = StudyResult(study, metric, time_unit, replicates, conditions, comparisons,
                         model_selection, warnings, out_dir)
    write_study(result)
    return result


def _model_selection(replicates: list[Replicate], models: tuple[str, ...]) -> list[dict]:
    """Per replicate AICc and Akaike weight of each model, and the total over replicates
    where every model converged (independent replicates: log-likelihoods add up)."""
    rows = []
    complete = [r for r in replicates if r.fits and all(f.ok for f in r.fits)]
    for rep in replicates:
        for f in rep.fits:
            rows.append({"condition": rep.condition, "replicate": rep.replicate, "model": f.model,
                         "ok": f.ok, "aicc": f.aicc, "akaike_weight": f.akaike_weight})
    if complete:
        totals = {m: sum(next(f for f in r.fits if f.model == m).aicc for r in complete)
                  for m in models}
        best = min(totals.values())
        rel = {m: math.exp(-(v - best) / 2) for m, v in totals.items()}
        norm = sum(rel.values())
        for m in models:
            rows.append({"condition": "(all)", "replicate": f"{len(complete)} replicates",
                         "model": m, "ok": True, "aicc": totals[m],
                         "akaike_weight": rel[m] / norm})
    return rows


# --- output --------------------------------------------------------------------------------


def _num(value, digits: int = 6):
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else round(value, digits)
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: _num(v) for k, v in r.items()} for r in rows)


def replicate_rows(result: StudyResult) -> list[dict]:
    params = list(MODELS[result.study.model].params)
    rows = []
    for rep in result.replicates:
        fit = rep.fit
        row = {"condition": rep.condition, "replicate": rep.replicate,
               "experiment": str(rep.experiment), "plot": rep.plot,
               "settings_hash": rep.settings_hash,
               "n_frames": int(rep.series.use.sum()) if rep.series else 0,
               "fit_ok": fit is not None,
               "error": rep.error or (rep.fits[0].message if rep.fits and not fit else "")}
        for p in params:
            row[p] = fit.params[p] if fit else math.nan
            row[f"{p}_se"] = fit.stderr[p] if fit else math.nan
            ci = fit.ci95.get(p, [math.nan, math.nan]) if fit else [math.nan, math.nan]
            row[f"{p}_ci_low"], row[f"{p}_ci_high"] = ci
        row.update({
            "r2": fit.r2 if fit else math.nan, "rmse": fit.rmse if fit else math.nan,
            "aicc": fit.aicc if fit else math.nan,
            "reduced_chi2": fit.reduced_chi2 if fit else math.nan,
            "durbin_watson": fit.durbin_watson if fit else math.nan,
            "warnings": " | ".join(fit.warnings) if fit else "",
        })
        rows.append(row)
    return rows


def write_study(result: StudyResult) -> None:
    out = result.out_dir
    out.mkdir(parents=True, exist_ok=True)
    study = result.study
    files = {
        "replicates": out / "replicates.csv",
        "conditions": out / "conditions.csv",
        "comparisons": out / "comparisons.csv",
        "model_selection": out / "model_selection.csv",
    }
    _write_csv(files["replicates"], replicate_rows(result))
    _write_csv(files["conditions"], result.conditions)
    _write_csv(files["comparisons"], result.comparisons)
    _write_csv(files["model_selection"], result.model_selection)
    result.files = [p for p in files.values() if p.exists()]

    # Long format: every used frame of every replicate, for plotting or modelling elsewhere.
    data_csv = out / "data_long.csv"
    with open(data_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "replicate", "timestamp_utc", "frame_file",
                         f"t_{result.time_unit}", result.metric, f"{result.metric}_unc",
                         "used_in_fit"])
        for rep in result.replicates:
            if rep.series is None:
                continue
            s = rep.series
            for i, row in enumerate(s.rows):
                writer.writerow([rep.condition, rep.replicate, row["timestamp_utc"],
                                 row["frame_file"], _num(float(rep.t[i])), _num(float(s.y[i])),
                                 _num(float(s.unc[i])), int(s.use[i])])
    result.files.append(data_csv)

    summary = {
        "study": study.model_dump(mode="json"), "metric": result.metric,
        "time_unit": result.time_unit, "created_utc": iso_utc(utc_now()),
        "provenance": software_provenance(), "warnings": result.warnings,
        "replicates": [{
            "condition": r.condition, "replicate": r.replicate, "experiment": str(r.experiment),
            "plot": r.plot, "settings_hash": r.settings_hash, "error": r.error,
            "fits": [f.to_dict() for f in r.fits],
        } for r in result.replicates],
        "conditions": result.conditions, "comparisons": result.comparisons,
        "model_selection": result.model_selection,
    }
    study_json = out / "study.json"
    study_json.write_text(json.dumps(json_safe(summary), indent=2), "utf-8")
    result.files.append(study_json)

    methods = out / "methods.md"
    methods.write_text(methods_text(result), "utf-8")
    result.files.append(methods)
    result.files.extend(_figures(result))


def methods_text(result: StudyResult) -> str:
    """A draft methods paragraph with every number a reader needs to reproduce the analysis."""
    study = result.study
    model = MODELS[study.model]
    prov = software_provenance()
    loaded = [r for r in result.replicates if r.series is not None]
    counts = {c: sum(1 for r in loaded if r.condition == c and r.fit)
              for c in dict.fromkeys(r.condition for r in result.replicates)}
    seg_methods = sorted({(r.settings.get("segmenter") or {}).get("method", "?")
                          for r in loaded if r.settings})
    uncertainty = next((r.settings.get("analysis", {}).get("uncertainty")
                        for r in loaded if r.settings), None)
    weighted = all(r.fit is not None and r.fit.weighted for r in loaded if r.fit)
    boot = next((r.fit.bootstrap for r in loaded if r.fit and r.fit.bootstrap), None)
    commit = prov.get("git_commit", "")
    lines = [
        f"# Methods draft: {study.name}",
        "",
        "Generated by Fungus-CV; edit freely. Numbers are filled in from this analysis.",
        "",
        f"Images were analyzed with Fungus-CV {prov['fungus_cv_version']}"
        + (f" (commit {commit[:10]}{', with local changes' if prov.get('git_dirty') else ''})"
           if commit else "")
        + f". The target was segmented by {', '.join(seg_methods) or 'an unrecorded method'}"
        + ", after aligning each frame to the first frame of its time-lapse."
        + (f" Measurement uncertainty per frame combined the marker scale, pixel size, "
           f"alignment residual and segmentation (each frame re-segmented with settings "
           f"narrowed and widened by {_describe_uncertainty(uncertainty)}; the spread was "
           f"treated as a rectangular distribution)." if uncertainty and
           uncertainty.get("segmentation") else ""),
        "",
        f"For each replicate, {result.metric} was modelled over time (in {result.time_unit}) as "
        f"a {model.name} curve, {model.formula}, by {'weighted ' if weighted else ''}"
        "nonlinear least squares"
        + (" with weights from the per-frame uncertainties" if weighted else "")
        + f". Frames flagged {', '.join(study.exclude_flags) or '(none)'} were excluded"
        + ("; so were frames departing from the local trend (jumps)" if study.exclude_jumps
           else "") + ".",
    ]
    fitted = [r.fit for r in loaded if r.fit]
    n_ar1 = sum(f.error_model == "ar1" for f in fitted)
    errors_text = {
        "iid": "Frame errors were treated as independent.",
        "ar1": "Correlation between neighbouring frames was modelled as a first-order "
               "autoregressive process (generalized least squares, correlation per typical "
               "interval between frames, estimated by profile likelihood).",
        "auto": "For each replicate, independent and first-order autoregressive (AR(1)) frame "
                "errors were both fitted by generalized least squares and the one with lower "
                f"AICc was kept (AR(1) in {n_ar1} of {len(fitted)} replicates).",
    }[study.errors]
    lines.append(errors_text)
    if boot:
        lines.append(
            f"Parameter 95% intervals are percentiles of {boot['n']} model-based residual "
            "bootstrap refits (innovations resampled and, for AR(1) errors, rebuilt into "
            f"correlated errors; seed {boot['seed']}).")
    if study.also_fit:
        lines.append(
            f"Model adequacy was compared with {', '.join(study.also_fit)} by AICc summed over "
            "replicates (see model_selection.csv).")
    lines += [
        "",
        "Parameter estimates were then analyzed with the replicate as the experimental unit "
        f"({', '.join(f'{c}: n = {n}' for c, n in counts.items())}). Conditions are summarised "
        "as mean ± SD with t-based 95% confidence intervals; random-effects (DerSimonian-Laird) "
        "means weighting each replicate by its own uncertainty are also reported. "
        + ("Each condition was compared with "
           f"{study.reference}" if study.reference else "All pairs of conditions were compared")
        + " using Welch's t-test; p-values were Holm-adjusted within each parameter, and "
        "effect sizes are Hedges' g.",
        "",
        "Replicates:",
        "",
    ]
    for r in result.replicates:
        status = "" if r.fit else " (not used: " + (r.error or "fit failed") + ")"
        lines.append(f"- {r.condition} / {r.replicate}: {r.experiment.name}"
                     + (f" plot {r.plot}" if r.plot not in ("", "main") else "")
                     + f", settings {r.settings_hash or 'unknown'}{status}")
    if result.warnings:
        lines += ["", "Warnings to resolve before publishing:", ""]
        lines += [f"- {w}" for w in result.warnings]
    return "\n".join(lines) + "\n"


def _describe_uncertainty(u: dict) -> str:
    h, s, v = u.get("hsv_delta", (0, 0, 0))
    return (f"HSV ±({h}, {s}, {v}) for colour, probability ±{u.get('probability_delta')} for "
            f"trained models, logit ±{u.get('logit_delta')} for SAM 2")


def _colors(n: int) -> list[str]:
    import matplotlib

    extra = [matplotlib.colors.to_hex(c) for c in matplotlib.colormaps["tab10"].colors]
    return (SERIES + [c for c in extra if c not in SERIES])[:max(n, 1)]


def _figures(result: StudyResult) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    study = result.study
    conditions = list(dict.fromkeys(r.condition for r in result.replicates))
    colors = dict(zip(conditions, _colors(len(conditions))))
    loaded = [r for r in result.replicates if r.series is not None]
    files = []

    # Curves: replicate data and fits, plus the pointwise mean of each condition's fitted curves.
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _style(ax)
    for c in conditions:
        reps = [r for r in loaded if r.condition == c]
        for r in reps:
            use = r.series.use
            ax.plot(r.t[use], r.series.y[use], "o", ms=2.5, color=colors[c], alpha=0.35,
                    zorder=2)
        fitted = [r for r in reps if r.fit]
        if not fitted:
            continue
        t_end = min(float(r.t[r.series.use].max()) for r in fitted)
        grid = np.linspace(0.0, t_end, 300)
        curves = np.array([r.fit.predict(grid) for r in fitted])
        for curve in curves:
            ax.plot(grid, curve, color=colors[c], lw=1, alpha=0.6, zorder=3)
        ax.plot(grid, curves.mean(axis=0), color=colors[c], lw=2.5, zorder=4,
                label=f"{c} (n = {len(fitted)})")
    ax.set_xlabel(f"Time ({result.time_unit})", color=INK)
    ax.set_ylabel(result.metric, color=INK)
    ax.set_title(f"{study.name}: {study.model} fits per replicate, thick = condition mean",
                 color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    for ext in ("png", "svg"):
        path = result.out_dir / f"curves.{ext}"
        fig.savefig(path)
        files.append(path)
    plt.close(fig)

    # Parameters: each replicate with its bootstrap interval, and the condition mean with its CI.
    params = study.compared_params
    fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params) + 1, 4.2), dpi=150,
                             squeeze=False)
    for ax, param in zip(axes[0], params):
        _style(ax)
        ax.grid(False, axis="x")
        for i, c in enumerate(conditions):
            reps = [r for r in loaded if r.condition == c and r.fit]
            offsets = np.linspace(-0.3, 0.0, len(reps)) if len(reps) > 1 else [-0.15]
            for off, r in zip(offsets, reps):
                v = r.fit.params[param]
                ci = r.fit.ci95.get(param)
                yerr = None if ci is None else [[v - ci[0]], [ci[1] - v]]
                ax.errorbar(i + off, v, yerr=yerr, fmt="o", ms=4, color=colors[c],
                            ecolor=matplotlib.colors.to_rgba(colors[c], 0.45), elinewidth=1.5,
                            zorder=3)
            row = next(x for x in result.conditions if x["condition"] == c and x["param"] == param)
            if row["n"]:  # condition mean and its CI, beside the replicates
                yerr = ([[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]]
                        if row["n"] >= 2 else None)
                ax.errorbar(i + 0.25, row["mean"], yerr=yerr, fmt="D", ms=6, color=INK,
                            ecolor=INK_2, elinewidth=1.5, capsize=3, zorder=4)
        ax.set_xticks(range(len(conditions)), conditions, rotation=30, ha="right")
        ax.set_xlim(-0.6, len(conditions) - 0.4)
        ax.set_title(param, color=INK, loc="left", fontsize=11)
    fig.suptitle(f"{study.model} parameters: dots = replicates (95% bootstrap interval), "
                 "diamond = condition mean (95% CI)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        path = result.out_dir / f"parameters.{ext}"
        fig.savefig(path)
        files.append(path)
    plt.close(fig)
    return files


TEMPLATE = """\
# Fungus-CV study: `fungus study <this file>`
# Each experiment (or field plot) is one replicate. Paths are relative to this file.
name: moss-treatment-2026
description: ""
metric: covered_length_pct   # any column of measurements.csv; null = as in `fungus report`
model: logistic              # parameters of this model are compared between conditions
also_fit: [gompertz, richards]   # only for model selection (model_selection.csv)
time_unit: auto              # auto | s | min | h | d (the same for every replicate)
reference: control           # compare each condition with this one; null = all pairs
params: [K, r]               # null = every parameter of the model
exclude_flags: [align_failed, blurry]
exclude_jumps: false
bootstrap: 1000              # block-bootstrap refits per replicate for 95% intervals
errors: auto                 # frame errors: auto (by AICc) | iid | ar1 (correlated in time)
seed: 0

experiments:
  - path: experiments/moss-control-1
    condition: control
    t0: 2026-09-20T14:05:00   # inoculation time (local); null = first frame
  - path: experiments/moss-control-2
    condition: control
    t0: 2026-09-20T14:20:00
  - path: experiments/moss-treated-1
    condition: treated
    t0: 2026-09-20T14:35:00
  - path: experiments/moss-treated-2
    condition: treated
    t0: 2026-09-20T14:50:00
"""


def write_template(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    Study.model_validate(yaml.safe_load(TEMPLATE))  # the template itself must stay valid
    path.write_text(TEMPLATE, "utf-8")
