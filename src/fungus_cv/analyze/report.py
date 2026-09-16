"""Plots, model fits and a time-lapse video from ``results/measurements.csv``."""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from fungus_cv.analyze.fit import DEFAULT_MODELS, FitResult, fit_all
from fungus_cv.analyze.pipeline import MEASUREMENTS_NAME, RESULTS_DIR
from fungus_cv.storage import Experiment, iso_utc, parse_iso_utc

log = logging.getLogger(__name__)

TIME_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
DEFAULT_EXCLUDE = ("align_failed", "blurry")

# Reference palette (light mode): neutral ink for data, fixed categorical order for fits.
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


@dataclass
class ReportResult:
    metric: str
    plot: str
    time_unit: str
    t0_utc: str
    n_used: int
    n_excluded: int
    retreats: int
    jumps: int = 0
    fits: list[FitResult] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


def load_measurements(experiment: Experiment, plot: str | None = None) -> list[dict]:
    """Measurement rows sorted by time; only ``plot``'s rows if given."""
    path = experiment.root / RESULTS_DIR / MEASUREMENTS_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `fungus analyze` first")
    with open(path, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["timestamp_utc"])
    for r in rows:
        r["plot"] = r.get("plot") or "main"
    return [r for r in rows if plot is None or r["plot"] == plot]


def list_plots(experiment: Experiment) -> list[str]:
    return sorted({r["plot"] for r in load_measurements(experiment)})


EDGE_METRICS = ("edge_advance_p95_mm", "edge_advance_max_mm",
                "edge_advance_p95_px", "edge_advance_max_px")


def add_edge_advance(experiment: Experiment, rows: list[dict], plot_name: str,
                     skip_flags: tuple[str, ...] = DEFAULT_EXCLUDE) -> None:
    """Add how far the target has spread beyond its first outline in the plot.

    The baseline is the first usable frame (in time) with target in the plot. For each later
    frame, every newly covered pixel's distance to the baseline patch is measured; the 95th
    percentile (robust) and maximum are reported. Values are 0 at the baseline frame and
    blank before it.
    """
    from fungus_cv.analyze.pipeline import annotations_path
    from fungus_cv.measure.geometry import Annotations

    plots = {p.name: p for p in Annotations.load(annotations_path(experiment)).measured_plots()}
    if plot_name not in plots:
        raise ValueError(f"plot {plot_name!r} is not in annotations.json")
    plot = plots[plot_name]
    runs_dir = experiment.root / RESULTS_DIR / "runs"
    scales: dict[str, float | None] = {}
    baseline_dist = region = None
    for r in rows:
        for key in EDGE_METRICS:
            r[key] = ""
        flags = set(filter(None, r["flags"].split(";")))
        if not r.get("mask_file") or flags & set(skip_flags):
            continue
        img = cv2.imread(str(experiment.root / r["mask_file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if region is None:
            region = plot.mask(img.shape)
        target = (img > 127) & region
        if baseline_dist is None:
            if not target.any():
                continue
            baseline_dist = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)
            p95 = mx = 0.0
        else:
            values = baseline_dist[target & (baseline_dist > 0)]
            p95 = float(np.percentile(values, 95)) if len(values) else 0.0
            mx = float(values.max()) if len(values) else 0.0
        h = r["settings_hash"]
        if h not in scales:
            info_path = runs_dir / f"{h}.json"
            info = json.loads(info_path.read_text("utf-8")) if info_path.exists() else {}
            scales[h] = (info.get("scale") or {}).get("mm_per_px")
        r["edge_advance_p95_px"], r["edge_advance_max_px"] = round(p95, 2), round(mx, 2)
        if scales[h]:
            r["edge_advance_p95_mm"] = round(p95 * scales[h], 3)
            r["edge_advance_max_mm"] = round(mx * scales[h], 3)


def pick_time_unit(span_seconds: float) -> str:
    if span_seconds < 2 * 3600:
        return "min"
    if span_seconds < 3 * 86400:
        return "h"
    return "d"


def _float(value: str) -> float:
    return float(value) if value not in ("", None) else math.nan


def count_retreats(y: np.ndarray, unc: np.ndarray) -> np.ndarray:
    """Frames where the front moved back by more than 3 standard uncertainties.

    Dye and infections should only advance; a retreat usually means a segmentation or
    lighting problem in that frame.
    """
    running = np.maximum.accumulate(np.nan_to_num(y, nan=-np.inf))
    tol = 3 * np.nan_to_num(unc, nan=0.0)
    return (running - y) > np.maximum(tol, 1e-9)


def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust line: median of pairwise slopes, so one bad point can't tilt it."""
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok])) if ok.any() else 0.0
    return slope, float(np.median(y - slope * x))


def detect_jumps(t: np.ndarray, y: np.ndarray, unc: np.ndarray, window: int = 3,
                 k: float = 4.0) -> np.ndarray:
    """Frames far off the local trend of their neighbours (a robust Hampel-style test).

    For each frame, a robust line is fitted through up to ``window`` usable frames on each
    side; the frame is a jump if its residual exceeds ``k`` times the larger of the
    neighbours' robust scatter (MAD) and the frame's own measurement uncertainty. The local
    line keeps steady growth from being flagged. The first and last frames are not tested:
    with neighbours on one side only, curvature can't be told apart from a jump.
    """
    n = len(y)
    jumps = np.zeros(n, bool)
    usable = np.flatnonzero(np.isfinite(y))
    for pos, i in enumerate(usable):
        before = usable[max(0, pos - window):pos]
        after = usable[pos + 1:pos + 1 + window]
        if len(before) == 0 or len(after) == 0 or len(before) + len(after) < 3:
            continue
        neighbours = np.r_[before, after]
        slope, intercept = _theil_sen(t[neighbours], y[neighbours])
        resid = y[neighbours] - (slope * t[neighbours] + intercept)
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        u = unc[i] if np.isfinite(unc[i]) else 0.0
        scale = max(mad, u, 1e-9 + 1e-6 * np.nanmax(np.abs(y)))
        jumps[i] = abs(y[i] - (slope * t[i] + intercept)) > k * scale
    return jumps


@dataclass
class Series:
    """One metric over time for one plot, with the frames that go into fits marked."""

    metric: str
    plot: str
    rows: list[dict]
    time_unit: str
    t0_ts: float
    t: np.ndarray  # in time_unit since t0
    y: np.ndarray
    unc: np.ndarray  # standard uncertainty per frame (NaN if the metric has none)
    excluded: np.ndarray
    jump: np.ndarray
    retreat: np.ndarray

    @property
    def use(self) -> np.ndarray:
        return ~self.excluded

    @property
    def span_seconds(self) -> float:
        return float(self.t[-1] - self.t[0]) * TIME_UNITS[self.time_unit] if len(self.t) else 0.0

    @property
    def sigma(self) -> np.ndarray | None:
        """Uncertainties of the used frames, if every one is known and positive."""
        u = self.unc[self.use]
        return u if len(u) and np.all(np.isfinite(u)) and np.all(u > 0) else None


def load_series(
    experiment: Experiment,
    metric: str | None = None,
    plot: str | None = None,
    t0: datetime | None = None,
    time_unit: str = "auto",
    exclude_flags: tuple[str, ...] = DEFAULT_EXCLUDE,
    exclude_jumps: bool = False,
) -> Series:
    plots = list_plots(experiment)
    if plot is None:
        if len(plots) > 1:
            raise ValueError(f"several plots {plots}; choose one")
        plot = plots[0] if plots else "main"
    rows = load_measurements(experiment, plot)
    if not rows:
        raise ValueError(f"no measurements for plot {plot!r}")
    if metric is None:
        for candidate in ("extent_mm", "extent_px", "target_area_mm2", "coverage_pct"):
            if rows[0].get(candidate, "") != "":
                metric = candidate
                break
    if metric in EDGE_METRICS:
        add_edge_advance(experiment, rows, plot, exclude_flags)
    if metric not in rows[0]:
        raise ValueError(f"unknown metric {metric!r}; columns: {list(rows[0])}")
    if time_unit != "auto" and time_unit not in TIME_UNITS:
        raise ValueError(f"unknown time unit {time_unit!r}; use one of {list(TIME_UNITS)}")

    times = np.array([parse_iso_utc(r["timestamp_utc"]).timestamp() for r in rows])
    t0_ts = t0.timestamp() if t0 else times[0]
    if time_unit == "auto":
        time_unit = pick_time_unit(times[-1] - t0_ts)
    t = (times - t0_ts) / TIME_UNITS[time_unit]
    y = np.array([_float(r[metric]) for r in rows])
    unc_column = f"{metric}_unc"
    unc = np.array([_float(r.get(unc_column, "")) for r in rows]) if unc_column in rows[0] \
        else np.full(len(rows), math.nan)

    flags = [set(filter(None, r["flags"].split(";"))) for r in rows]
    excluded = np.array([bool(f & set(exclude_flags)) for f in flags]) | np.isnan(y) | (t < 0)
    jump = detect_jumps(t, np.where(excluded, np.nan, y), unc) & ~excluded
    if exclude_jumps:
        excluded = excluded | jump
    retreat = count_retreats(np.where(excluded, np.nan, y), unc) & ~excluded
    return Series(metric, plot, rows, time_unit, t0_ts, t, y, unc, excluded, jump, retreat)


def make_report(
    experiment: Experiment,
    metric: str | None = None,
    t0: datetime | None = None,
    time_unit: str = "auto",
    exclude_flags: tuple[str, ...] = DEFAULT_EXCLUDE,
    models: tuple[str, ...] = DEFAULT_MODELS,
    video: bool = False,
    fps: int = 10,
    exclude_jumps: bool = False,
    plot: str | None = None,
    bootstrap: int = 1000,
    errors: str = "auto",
    seed: int = 0,
) -> ReportResult:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = list_plots(experiment)
    series = load_series(experiment, metric, plot, t0, time_unit, exclude_flags, exclude_jumps)
    metric, plot, rows, time_unit = series.metric, series.plot, series.rows, series.time_unit
    t, y, unc = series.t, series.y, series.unc
    excluded, jump, retreat, use = series.excluded, series.jump, series.retreat, series.use
    t0_ts = series.t0_ts
    sigma = series.sigma
    fits = fit_all(t[use], y[use], models=models, sigma=sigma, bootstrap=bootstrap,
                   errors=errors, seed=seed)

    out_dir = experiment.root / RESULTS_DIR / "report"
    if plots != ["main"]:
        out_dir = out_dir / plot
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ReportResult(
        metric=metric, plot=plot, time_unit=time_unit,
        t0_utc=iso_utc(datetime.fromtimestamp(t0_ts, timezone.utc)),
        n_used=int(use.sum()), n_excluded=int(excluded.sum()),
        retreats=int(retreat.sum()), jumps=int(jump.sum()), fits=fits,
    )

    # Every frame's flags in one place, so exclusions can be audited.
    flags_csv = out_dir / "frame_flags.csv"
    with open(flags_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "frame_file", f"t_{time_unit}", metric,
                         "pipeline_flags", "jump", "retreat", "used_in_fits"])
        for i, r in enumerate(rows):
            writer.writerow([r["timestamp_utc"], r["frame_file"], round(float(t[i]), 6),
                             r[metric], r["flags"], int(jump[i]), int(retreat[i]),
                             int(use[i])])
    result.files.append(flags_csv)

    # --- main plot: metric over time with fits ---------------------------------------
    label = {"extent_mm": "Extent (mm)", "extent_px": "Extent (px)",
             "coverage_pct": "Coverage (%)", "extent_fraction": "Extent (fraction of axis)",
             "target_area_mm2": "Area (mm²)",
             "extent_max_mm": "Highest point (mm)",
             "covered_length_pct": "Length covered (%)",
             "covered_length_mm": "Length covered (mm)",
             "reference_covered_pct": "Object area covered (%)",
             "equivalent_radius_mm": "Equivalent radius (mm)",
             "edge_advance_p95_mm": "Edge advance, 95th percentile (mm)",
             "edge_advance_max_mm": "Edge advance, maximum (mm)",
             "gcc_mean": "Green chromatic coordinate (mean)",
             "gcc_p90": "Green chromatic coordinate (90th pct)",
             "rcc_mean": "Red chromatic coordinate (mean)",
             "exg_mean": "Excess green (mean)"}.get(metric, metric)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _style(ax)
    if np.isfinite(unc[use]).any():
        ax.errorbar(t[use], y[use], yerr=unc[use], fmt="o", ms=4, color=INK_2, ecolor=GRID,
                    elinewidth=1, capsize=0, label="measured", zorder=3)
    else:
        ax.plot(t[use], y[use], "o", ms=4, color=INK_2, label="measured", zorder=3)
    if excluded.any():
        ax.plot(t[excluded], y[excluded], "x", ms=6, color=INK_2, alpha=0.6,
                label="excluded (flagged)", zorder=3)
    if jump.any():
        ax.plot(t[jump], y[jump], "s", ms=10, mfc="none", mec=INK, mew=1.2,
                label="jump from local trend" + (" (excluded)" if exclude_jumps else ""),
                zorder=4)
    if retreat.any():
        ax.plot(t[retreat], y[retreat], "o", ms=9, mfc="none", mec=INK, mew=1.2,
                label="retreat > 3σ (check frame)", zorder=4)

    if use.sum() > 1:
        grid = np.linspace(max(0.0, t[use].min()), t[use].max(), 300)
        for i, (color, fit) in enumerate(zip(SERIES, fits)):
            if fit.ok:
                # Later models are dashed and drawn on top, so identical fits stay visible.
                ax.plot(grid, fit.predict(grid), color=color, lw=2,
                        ls=(0, (6, 3)) if i % 2 else "-", zorder=2.5 if i % 2 else 2,
                        label=f"{fit.model}  R² {fit.r2:.4f}  w {fit.akaike_weight:.2f}")
    ax.set_xlabel(f"Time since start ({time_unit})", color=INK)
    ax.set_ylabel(label, color=INK)
    where = experiment.config.name + ("" if plot == "main" else f" / {plot}")
    ax.set_title(f"{where}: {label[0].lower() + label[1:]} over time", color=INK,
                 loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    main_png = out_dir / f"{metric}_vs_time.png"
    fig.savefig(main_png)
    plt.close(fig)
    result.files.append(main_png)

    # --- quality control: one measure per panel, shared time axis --------------------
    qc = [("mean_brightness", "Brightness (0-255)"), ("align_shift_px", "Alignment shift (px)"),
          ("align_rms_px", "Alignment residual (px)"), ("coverage_pct", "Coverage (%)")]
    if any(r.get("light_gain_g") not in ("", None) for r in rows):
        qc.insert(1, ("light_gain_g", "Lighting gain (green)"))
    fig, axes = plt.subplots(len(qc), 1, figsize=(8, 8), dpi=150, sharex=True)
    for ax, (col, title) in zip(axes, qc):
        _style(ax)
        values = np.array([_float(r[col]) for r in rows])
        ax.plot(t, values, color=SERIES[0], lw=2)
        ax.set_ylabel(title, color=INK, fontsize=9)
    axes[-1].set_xlabel(f"Time since start ({time_unit})", color=INK)
    axes[0].set_title("Quality checks", color=INK, loc="left", fontsize=12)
    fig.tight_layout()
    qc_png = out_dir / "quality_checks.png"
    fig.savefig(qc_png)
    plt.close(fig)
    result.files.append(qc_png)

    summary = {
        "metric": metric, "plot": plot, "time_unit": time_unit, "t0": result.t0_utc,
        "n_used": result.n_used, "n_excluded": result.n_excluded,
        "excluded_flags": list(exclude_flags), "retreats": result.retreats,
        "jumps": result.jumps, "jumps_excluded": exclude_jumps,
        "weighted_by_uncertainty": sigma is not None,
        "model_selection": "AICc; akaike_weight = relative likelihood among the fitted models",
        "fits": [f.to_dict() for f in fits],
    }
    fits_json = out_dir / "fits.json"
    fits_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result.files.append(fits_json)

    if video:
        mp4 = out_dir / "overlay_timelapse.mp4"
        if write_video([experiment.root / r["overlay_file"] for r in rows if r["overlay_file"]],
                       mp4, fps):
            result.files.append(mp4)
    return result


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def write_video(frames: list[Path], path: Path, fps: int = 10) -> bool:
    frames = [p for p in frames if p.exists()]
    if not frames:
        log.warning("no overlay images to make a video from")
        return False
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        log.warning("could not open a video writer for %s", path)
        return False
    for p in frames:
        img = cv2.imread(str(p))
        if img is not None:
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
    writer.release()
    return True
