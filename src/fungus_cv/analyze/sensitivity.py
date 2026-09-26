"""How much do the results depend on the settings?

Each variant changes one setting a reasonable person might have chosen differently, re-runs
the analysis into its own folder and compares it with the current settings, frame by frame.
The differences are also expressed in units of the reported uncertainty: a variant that moves
the measurement by much less than 1 u does not change the conclusions, while one that moves it
by several u means the setting has to be justified in the methods.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fungus_cv.analyze.pipeline import RESULTS_DIR, Analyzer
from fungus_cv.analyze.report import DEFAULT_EXCLUDE, load_series
from fungus_cv.provenance import software_provenance
from fungus_cv.storage import Experiment, iso_utc, utc_now

log = logging.getLogger(__name__)

SENSITIVITY_DIR = "sensitivity"


@dataclass
class Variant:
    name: str
    description: str  # what was changed, in words
    changes: dict  # dotted analysis setting -> value


@dataclass
class VariantResult:
    variant: Variant
    n_frames: int = 0
    n_compared: int = 0
    mean_abs_change: float = math.nan
    max_abs_change: float = math.nan
    mean_change: float = math.nan  # signed: a systematic shift
    final_change: float = math.nan  # at the last frame both share
    change_in_uncertainties: float = math.nan  # mean |change| / mean reported uncertainty
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class SensitivityResult:
    metric: str
    plot: str
    baseline_final: float
    baseline_uncertainty: float
    variants: list[VariantResult] = field(default_factory=list)
    out_dir: Path | None = None
    files: list[Path] = field(default_factory=list)

    @property
    def largest(self) -> VariantResult | None:
        done = [v for v in self.variants if v.ok and np.isfinite(v.mean_abs_change)]
        return max(done, key=lambda v: v.mean_abs_change) if done else None


def _set(config, dotted: str, value) -> None:
    node = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)


def _get(config, dotted: str):
    node = config
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


def default_variants(experiment: Experiment) -> list[Variant]:
    """Settings worth questioning, chosen from what this experiment actually uses."""
    analysis = experiment.config.analysis
    variants: list[Variant] = []
    target = analysis.target

    if target.method == "color":
        from fungus_cv.segment.color import shift_ranges

        ranges = [(tuple(r.lower), tuple(r.upper)) for r in target.color.hsv_ranges]
        delta = tuple(analysis.uncertainty.hsv_delta)
        for sign, word in ((1, "wider"), (-1, "narrower")):
            shifted = shift_ranges(ranges, delta, sign)
            variants.append(Variant(
                f"colour_{word}", f"colour thresholds {word} by HSV {delta}",
                {"target.color.hsv_ranges": [{"lower": list(lo), "upper": list(hi)}
                                             for lo, hi in shifted]}))
        variants.append(Variant("no_cleanup", "no morphological clean-up of the mask",
                                {"target.color.open_px": 0, "target.color.close_px": 0}))
    elif target.method == "model":
        base = target.model.threshold
        for value, word in ((0.35, "looser"), (0.65, "stricter")):
            variants.append(Variant(f"threshold_{word}",
                                    f"model probability threshold {value} instead of "
                                    f"{base if base is not None else 'the tuned value'}",
                                    {"target.model.threshold": value}))
    elif target.method == "sam2":
        for value, word in ((-1.0, "looser"), (1.0, "tighter")):
            variants.append(Variant(f"mask_threshold_{word}",
                                    f"SAM 2 mask threshold {value:+g}",
                                    {"target.sam2.mask_threshold": value}))

    if analysis.align != "none":
        other = "ecc" if analysis.align in ("markers", "markers_or_ecc") else "markers"
        variants.append(Variant(f"align_{other}", f"alignment by {other} instead of "
                                f"{analysis.align}", {"align": other}))
    if analysis.lighting.method != "none":
        variants.append(Variant("lighting_off", "no lighting correction",
                                {"lighting.method": "none"}))
    elif (experiment.root / "annotations.json").exists():
        variants.append(Variant("lighting_background", "lighting corrected from the background",
                                {"lighting.method": "background"}))
    # Switching perspective correction changes the prepared frame's size and geometry, so
    # annotations clicked on it would no longer fit and the variant could only fail.
    if analysis.markers.size_mm and not (experiment.root / "annotations.json").exists():
        variants.append(Variant(
            "rectify_off" if analysis.rectify.enabled else "rectify_on",
            "perspective correction " + ("off" if analysis.rectify.enabled else "on"),
            {"rectify.enabled": not analysis.rectify.enabled}))
    if analysis.measure.mode != "path" or analysis.measure.path_source == "annotation":
        for value in (25.0, 75.0):
            variants.append(Variant(f"front_percentile_{int(value)}",
                                    f"front taken at the {int(value)}th percentile across the "
                                    "width instead of the median",
                                    {"front_percentile": value}))
    return variants


def _variant_experiment(experiment: Experiment, variant: Variant) -> Experiment:
    """The same experiment with one setting changed (in memory only)."""
    changed = Experiment(experiment.root)
    changed.config = copy.deepcopy(experiment.config)
    for dotted, value in variant.changes.items():
        if dotted == "target.color.hsv_ranges":
            from fungus_cv.config import HsvRange

            value = [HsvRange(**r) for r in value]
        _set(changed.config.analysis, dotted, value)
    return changed


def _series_values(series) -> dict:
    return {row["frame_file"]: value for row, value, used
            in zip(series.rows, series.y, series.use) if used and np.isfinite(value)}


def run(
    experiment: Experiment,
    metric: str | None = None,
    plot: str | None = None,
    variants: list[Variant] | None = None,
    out_dir: Path | None = None,
    keep_masks: bool = False,
    exclude_flags: tuple[str, ...] = DEFAULT_EXCLUDE,
    progress=None,
) -> SensitivityResult:
    """Re-run the analysis for each variant and compare it with the current settings."""
    out_dir = Path(out_dir) if out_dir else experiment.root / RESULTS_DIR / SENSITIVITY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_series(experiment, metric, plot, exclude_flags=exclude_flags)
    metric, plot = baseline.metric, baseline.plot
    base_values = _series_values(baseline)
    if not base_values:
        raise ValueError(f"no usable {metric} values in the current results")
    unc = baseline.unc[baseline.use]
    mean_unc = float(np.nanmean(unc)) if np.isfinite(unc).any() else math.nan
    result = SensitivityResult(
        metric=metric, plot=plot,
        baseline_final=float(list(base_values.values())[-1]),
        baseline_uncertainty=mean_unc, out_dir=out_dir)

    chosen = variants if variants is not None else default_variants(experiment)
    for n, variant in enumerate(chosen, 1):
        if progress:
            progress(n, len(chosen), variant.name)
        log.info("sensitivity %d/%d: %s", n, len(chosen), variant.description)
        entry = VariantResult(variant)
        variant_dir = out_dir / variant.name
        try:
            changed = _variant_experiment(experiment, variant)
            summary = Analyzer(changed, results_dir=variant_dir).run(force=True)
            entry.n_frames = summary.processed
            series = load_series(changed, metric, plot, exclude_flags=exclude_flags,
                                 results_dir=variant_dir)
            values = _series_values(series)
        except Exception as exc:  # noqa: BLE001 - one bad variant must not stop the rest
            log.exception("variant %s failed", variant.name)
            entry.error = f"{type(exc).__name__}: {exc}"
            result.variants.append(entry)
            continue
        shared = [f for f in base_values if f in values]
        if shared:
            diff = np.array([values[f] - base_values[f] for f in shared])
            entry.n_compared = len(shared)
            entry.mean_abs_change = float(np.abs(diff).mean())
            entry.max_abs_change = float(np.abs(diff).max())
            entry.mean_change = float(diff.mean())
            entry.final_change = float(diff[-1])
            if mean_unc and np.isfinite(mean_unc) and mean_unc > 0:
                entry.change_in_uncertainties = entry.mean_abs_change / mean_unc
        else:
            entry.error = "no frames in common with the current results"
        result.variants.append(entry)
        if not keep_masks:
            for part in ("masks", "overlays"):
                shutil.rmtree(variant_dir / part, ignore_errors=True)

    _write(result, experiment, baseline, out_dir)
    return result


def _write(result: SensitivityResult, experiment: Experiment, baseline, out_dir: Path) -> None:
    rows = [{
        "variant": v.variant.name, "changed": v.variant.description,
        "frames_compared": v.n_compared,
        "mean_abs_change": round(v.mean_abs_change, 6) if np.isfinite(v.mean_abs_change) else "",
        "max_abs_change": round(v.max_abs_change, 6) if np.isfinite(v.max_abs_change) else "",
        "mean_change": round(v.mean_change, 6) if np.isfinite(v.mean_change) else "",
        "change_at_last_frame": round(v.final_change, 6) if np.isfinite(v.final_change) else "",
        "mean_abs_change_in_uncertainties":
            round(v.change_in_uncertainties, 3) if np.isfinite(v.change_in_uncertainties) else "",
        "error": v.error,
    } for v in result.variants]
    csv_path = out_dir / "sensitivity.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result.files.append(csv_path)
    json_path = out_dir / "sensitivity.json"
    json_path.write_text(json.dumps({
        "created_utc": iso_utc(utc_now()), "provenance": software_provenance(),
        "experiment": experiment.config.name, "metric": result.metric, "plot": result.plot,
        "baseline": {"final": result.baseline_final,
                     "mean_uncertainty": None if math.isnan(result.baseline_uncertainty)
                     else result.baseline_uncertainty,
                     "settings": experiment.config.analysis.model_dump(mode="json")},
        "variants": rows,
    }, indent=2), encoding="utf-8")
    result.files.append(json_path)
    result.files.append(_chart(result, baseline, out_dir))


def _chart(result: SensitivityResult, baseline, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fungus_cv.analyze.report import INK, SERIES, _style

    done = [v for v in result.variants if v.ok and np.isfinite(v.mean_abs_change)]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(done) + 2.4), dpi=150)
    _style(ax)
    ax.grid(False, axis="y")
    order = sorted(done, key=lambda v: v.mean_abs_change)
    labels = [v.variant.description for v in order]
    values = [v.mean_abs_change for v in order]
    ax.barh(range(len(order)), values, color=SERIES[0], height=0.5)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks(range(len(order)), labels, fontsize=8)
    unit = " (mm)" if result.metric.endswith("_mm") else ""
    ax.set_xlabel(f"Mean change in {result.metric}{unit} when the setting is changed")
    if np.isfinite(result.baseline_uncertainty) and result.baseline_uncertainty > 0:
        ax.axvline(result.baseline_uncertainty, color=INK, lw=1.5, ls=(0, (6, 3)),
                   label="reported uncertainty (1u)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"How much the settings matter: {result.metric}", loc="left", color=INK,
                 fontsize=11)
    fig.tight_layout()
    path = out_dir / "sensitivity.png"
    fig.savefig(path)
    plt.close(fig)
    return path
