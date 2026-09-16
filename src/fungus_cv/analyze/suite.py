"""Validation suite: run every accuracy check against hand-labeled real data in one go.

A suite file lists experiments with hand-drawn masks or hand measurements, and trained
models with labeled test sets, plus the accuracy each must reach. Running it re-analyzes
the experiments with the current code, checks every bound, and compares each number with a
saved baseline so a change to the algorithms can't silently shift results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fungus_cv import __version__
from fungus_cv.storage import iso_utc, utc_now

log = logging.getLogger(__name__)

BASELINE_NAME = "baseline.json"
RESULTS_DIR_NAME = "suite_results"


class Bound(BaseModel):
    """Acceptance limits for one metric. Every limit given must hold."""

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None
    abs_max: float | None = None  # |value| <= abs_max, e.g. for a bias
    max_drift: float | None = None  # |value - baseline| <= max_drift


class MasksCheck(BaseModel):
    """Automatic masks against hand-drawn mask PNGs named like the frames."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["masks"]
    name: str | None = None
    masks: str  # folder of hand masks, relative to the suite file
    expect: dict[str, Bound] = Field(default_factory=dict)


class MeasurementsCheck(BaseModel):
    """Automatic measurements against a hand-measurement CSV (`fungus validate`)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["measurements"]
    name: str | None = None
    hand: str  # CSV with frame, value (and optional plot, value_unc, observer)
    metric: str = "extent_mm"
    plot: str | None = None
    expect: dict[str, Bound] = Field(default_factory=dict)


class ModelCheck(BaseModel):
    """A trained model against a labeled dataset (`fungus evaluate`)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["model"]
    name: str | None = None
    model: str
    dataset: str
    include_unreviewed: bool = False
    device: str = "auto"
    expect: dict[str, Bound] = Field(default_factory=dict)


Check = Annotated[MasksCheck | MeasurementsCheck | ModelCheck, Field(discriminator="type")]


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    experiment: str | None = None  # needed by masks and measurements checks
    analyze: bool = True  # measure new frames with the current code first
    checks: list[Check]

    @model_validator(mode="after")
    def _needs_experiment(self) -> Case:
        if self.experiment is None and any(c.type != "model" for c in self.checks):
            raise ValueError(f"case {self.name!r}: masks and measurements checks need "
                             "'experiment'")
        names = [check_name(c, i) for i, c in enumerate(self.checks)]
        if len(names) != len(set(names)):
            raise ValueError(f"case {self.name!r}: check names must be unique, got {names}")
        return self


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    cases: list[Case]

    @model_validator(mode="after")
    def _unique_cases(self) -> Suite:
        names = [c.name for c in self.cases]
        if len(names) != len(set(names)):
            raise ValueError(f"case names must be unique, got {names}")
        return self


def check_name(check, index: int) -> str:
    return check.name or f"{check.type}-{index + 1}"


def load_suite(path: Path) -> Suite:
    with open(path, encoding="utf-8") as f:
        return Suite.model_validate(yaml.safe_load(f) or {})


# --- results -------------------------------------------------------------------------------


@dataclass
class MetricResult:
    metric: str
    value: float
    baseline: float | None = None
    bound: Bound | None = None
    failures: list[str] = field(default_factory=list)

    @property
    def drift(self) -> float | None:
        if self.baseline is None or math.isnan(self.value) or math.isnan(self.baseline):
            return None
        return self.value - self.baseline

    @property
    def status(self) -> str:
        if self.failures:
            return "fail"
        return "pass" if self.bound is not None else "info"


@dataclass
class CheckResult:
    case: str
    check: str
    type: str
    metrics: list[MetricResult] = field(default_factory=list)
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.error is None and all(m.status != "fail" for m in self.metrics)


@dataclass
class SuiteResult:
    suite: str
    started_utc: str
    checks: list[CheckResult]
    out_dir: Path
    baseline_found: bool
    provenance: dict

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def values(self) -> dict:
        """``{case: {check: {metric: value}}}``, the shape of a baseline file."""
        out: dict = {}
        for c in self.checks:
            if c.error is None:
                out.setdefault(c.case, {})[c.check] = {m.metric: m.value for m in c.metrics}
        return out


def evaluate_bound(value: float, bound: Bound, baseline: float | None,
                   require_baseline: bool = True) -> list[str]:
    """Why ``value`` breaks ``bound`` (empty if it doesn't)."""
    failures = []
    if math.isnan(value):
        return ["no value"]
    if bound.min is not None and value < bound.min:
        failures.append(f"below min {bound.min}")
    if bound.max is not None and value > bound.max:
        failures.append(f"above max {bound.max}")
    if bound.abs_max is not None and abs(value) > bound.abs_max:
        failures.append(f"|value| above {bound.abs_max}")
    if bound.max_drift is not None:
        if baseline is None:
            if require_baseline:
                failures.append(
                    "no baseline to measure drift against (run with --save-baseline)")
        elif abs(value - baseline) > bound.max_drift:
            failures.append(f"drifted {value - baseline:+.4g} from baseline {baseline:.4g} "
                            f"(max {bound.max_drift})")
    return failures


# --- running checks ------------------------------------------------------------------------


def _masks_metrics(experiment, settings_hash: str, masks_dir: Path) -> tuple[dict, list[str]]:
    from fungus_cv.analyze.compare import compare_runs

    if not masks_dir.is_dir():
        raise FileNotFoundError(f"hand mask folder {masks_dir} not found")
    # a = hand, b = automatic, so differences read "automatic - hand" as in `validate`.
    r = compare_runs(experiment, str(masks_dir.resolve()), settings_hash)
    n_hand = sum(1 for _ in masks_dir.glob("*.png"))
    notes = []
    if n_hand > r.n_frames:
        notes.append(f"{n_hand - r.n_frames} hand mask(s) had no matching analyzed frame")
    return {
        "n_frames": float(r.n_frames),
        "iou_mean": r.iou_mean, "iou_median": r.iou_median, "iou_min": r.iou_min,
        "extent_diff_mean": r.extent_diff_mean, "extent_diff_sd": r.extent_diff_sd,
    }, notes + [f"worst frame {r.worst_frame}; extents in {r.extent_unit}"]


def _measurement_metrics(experiment, check: MeasurementsCheck,
                         hand: Path) -> tuple[dict, list[str]]:
    from fungus_cv.analyze.validate import validate

    a = validate(experiment, hand, check.metric, check.plot)
    notes = [f"largest difference {a.worst_diff:+.4g} at {a.worst_frame}"]
    if a.unmatched:
        notes.append(f"{len(a.unmatched)} hand row(s) not matched: {', '.join(a.unmatched[:5])}")
    return {
        "n": float(a.n), "bias": a.bias, "bias_ci_low": a.bias_ci95[0],
        "bias_ci_high": a.bias_ci95[1], "sd_diff": a.sd_diff,
        "loa_low": a.limits_of_agreement[0], "loa_high": a.limits_of_agreement[1],
        "mae": a.mae, "rmse": a.rmse, "pearson_r": a.pearson_r, "slope": a.slope,
        "intercept": a.intercept, "within_2u": a.within_2u, "z_rms": a.z_rms,
    }, notes


def _model_metrics(check: ModelCheck, model_dir: Path, dataset_dir: Path,
                   evaluate=None) -> tuple[dict, list[str], dict]:
    from fungus_cv.learn.dataset import Dataset

    if evaluate is None:
        from fungus_cv.learn.train import evaluate_model as evaluate
    ds = Dataset.open(dataset_dir)
    rows, summary = evaluate(model_dir, ds, not check.include_unreviewed, check.device, None)
    if not rows:
        raise ValueError(f"no items to evaluate in {dataset_dir} (none reviewed?)")
    metrics = {"threshold": float(summary["threshold"])}
    for prefix, key in (("", "all"), ("unseen_", "not_in_training_set")):
        for k, v in summary[key].items():
            metrics[f"{prefix}{k}"] = float(v)
    notes = []
    if not summary["not_in_training_set"]["n"]:
        notes.append("every item was used in training; unseen_* metrics are missing and the "
                     "rest are optimistic")
    weights = model_dir / "model.pt"
    prov = {"model": str(model_dir), "dataset": str(dataset_dir),
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()
            if weights.exists() else None}
    return metrics, notes, prov


def _prepare_experiment(case: Case, root: Path, cache: dict):
    """Open (and optionally analyze) a case's experiment once for all its checks."""
    from fungus_cv.analyze.pipeline import RESULTS_DIR, RUN_INFO_NAME, Analyzer
    from fungus_cv.storage import Experiment

    key = case.name
    if key not in cache:
        path = (root / case.experiment).resolve()
        experiment = Experiment(path)
        if case.analyze:
            analyzer = Analyzer(experiment)
            summary = analyzer.run()
            log.info("%s: measured %d new frame(s) (settings %s)", case.name,
                     summary.processed, analyzer.settings_hash)
            settings_hash = analyzer.settings_hash
        else:
            info_path = path / RESULTS_DIR / RUN_INFO_NAME
            if not info_path.exists():
                raise FileNotFoundError(f"{path} has no analysis yet; set analyze: true")
            settings_hash = json.loads(info_path.read_text(encoding="utf-8"))["settings_hash"]
        cache[key] = (experiment, {"experiment": str(path), "settings_hash": settings_hash})
    return cache[key]


def run_suite(suite_path: Path, out_root: Path | None = None, evaluate_model=None,
              progress=None, require_baseline: bool = True) -> SuiteResult:
    """Run every check. ``evaluate_model`` replaces the model evaluator (used in tests).

    ``require_baseline=False`` (when a baseline is about to be saved) doesn't fail
    ``max_drift`` limits for metrics that have no baseline yet.
    """
    suite_path = Path(suite_path)
    suite = load_suite(suite_path)
    root = suite_path.parent
    baseline_path = root / BASELINE_NAME
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) \
        if baseline_path.exists() else {}
    started = utc_now()
    out_dir = (out_root or root / RESULTS_DIR_NAME) / f"{started:%Y%m%dT%H%M%SZ}"

    results: list[CheckResult] = []
    experiments: dict = {}
    for case in suite.cases:
        for i, check in enumerate(case.checks):
            name = check_name(check, i)
            if progress:
                progress(case.name, name)
            result = CheckResult(case.name, name, check.type)
            try:
                if check.type == "model":
                    values, notes, prov = _model_metrics(
                        check, (root / check.model).resolve(), (root / check.dataset).resolve(),
                        evaluate_model)
                else:
                    experiment, prov = _prepare_experiment(case, root, experiments)
                    if check.type == "masks":
                        values, notes = _masks_metrics(experiment, prov["settings_hash"],
                                                       root / check.masks)
                    else:
                        values, notes = _measurement_metrics(experiment, check,
                                                             root / check.hand)
                result.notes, result.provenance = notes, dict(prov)
            except Exception as exc:  # noqa: BLE001 - one broken case must not stop the suite
                log.exception("%s / %s failed", case.name, name)
                result.error = f"{type(exc).__name__}: {exc}"
                results.append(result)
                continue

            base = baseline.get(case.name, {}).get(name, {})
            for metric, bound in check.expect.items():
                if metric not in values:
                    result.metrics.append(MetricResult(metric, math.nan, None, bound,
                                                       [f"unknown metric; have {sorted(values)}"]))
            for metric, value in values.items():
                bound = check.expect.get(metric)
                b = base.get(metric)
                b = None if b is None else float(b)
                m = MetricResult(metric, float(value), b, bound)
                if bound is not None:
                    m.failures = evaluate_bound(m.value, bound, b, require_baseline)
                result.metrics.append(m)
            results.append(result)

    suite_result = SuiteResult(
        suite=suite.name, started_utc=iso_utc(started), checks=results, out_dir=out_dir,
        baseline_found=bool(baseline), provenance=_provenance(),
    )
    write_results(suite_result)
    return suite_result


def _provenance() -> dict:
    info = {"fungus_cv_version": __version__}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
            text=True, timeout=5)
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=5)
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
            info["git_dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _num(value: float | None):
    return "" if value is None or math.isnan(value) else round(value, 6)


def write_results(result: SuiteResult) -> None:
    result.out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "suite": result.suite, "started_utc": result.started_utc, "passed": result.passed,
        "baseline_found": result.baseline_found, "provenance": result.provenance,
        "checks": [{
            "case": c.case, "check": c.check, "type": c.type, "passed": c.passed,
            "error": c.error, "notes": c.notes, "provenance": c.provenance,
            "metrics": [{
                "metric": m.metric, "value": _num(m.value), "baseline": _num(m.baseline),
                "drift": _num(m.drift), "status": m.status, "failures": m.failures,
                "bound": m.bound.model_dump(exclude_none=True) if m.bound else None,
            } for m in c.metrics],
        } for c in result.checks],
    }
    (result.out_dir / "results.json").write_text(json.dumps(data, indent=2), "utf-8")
    with open(result.out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "check", "metric", "value", "baseline", "drift", "status",
                         "detail"])
        for c in result.checks:
            if c.error:
                writer.writerow([c.case, c.check, "", "", "", "", "error", c.error])
            for m in c.metrics:
                writer.writerow([c.case, c.check, m.metric, _num(m.value), _num(m.baseline),
                                 _num(m.drift), m.status, "; ".join(m.failures)])


def save_baseline(result: SuiteResult, suite_path: Path) -> Path:
    """Store this run's values as the reference later runs are compared with."""
    path = Path(suite_path).parent / BASELINE_NAME
    data = {"_saved_utc": result.started_utc, "_provenance": result.provenance,
            **result.values()}
    path.write_text(json.dumps(data, indent=2, allow_nan=True), "utf-8")
    return path


TEMPLATE = """\
# Fungus-CV validation suite: `fungus validate-suite <this file>`
# Paths are relative to this file. Keep the hand-labeled data under version control (or
# archived) next to it, so the same checks can be re-run after any change to the code.
name: real-data
description: ""

cases:
  - name: dye-real-1
    experiment: experiments/dye-real-1
    analyze: true             # measure new frames with the current code before checking
    checks:
      # Hand-drawn masks, one PNG per frame named like the frame (white = target).
      - type: masks
        masks: hand_masks/dye-real-1
        expect:               # limits: min, max, abs_max, max_drift (vs baseline.json)
          iou_mean: {min: 0.90, max_drift: 0.01}
          iou_min: {min: 0.80}
          extent_diff_mean: {abs_max: 1.0}   # automatic - hand, mm
      # Hand measurements (`fungus validate ... --make-template 20`, then fill in 'value').
      - type: measurements
        hand: hand/dye-real-1.csv
        metric: extent_mm
        expect:
          bias: {abs_max: 1.0, max_drift: 0.2}
          rmse: {max: 2.0}
          within_2u: {min: 0.85}  # the reported uncertainties cover real errors

  # A trained model against a labeled test set it was not trained on.
  - name: moss-model
    checks:
      - type: model
        model: models/moss-bark-v1
        dataset: datasets/moss-bark-test
        expect:
          unseen_iou_mean: {min: 0.75, max_drift: 0.02}
          unseen_boundary_f1_2px_mean: {min: 0.70}
"""


def write_template(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.write_text(TEMPLATE, "utf-8")
    Suite.model_validate(yaml.safe_load(TEMPLATE))  # the template itself must stay valid
