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

from fungus_cv.analyze.fit import DEFAULT_MODELS, FitResult, best_fit, fit_all
from fungus_cv.analyze.pipeline import MEASUREMENTS_NAME, RESULTS_DIR
from fungus_cv.storage import Experiment, iso_utc, parse_iso_utc

log = logging.getLogger(__name__)

TIME_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
DEFAULT_EXCLUDE = ("align_failed", "blurry")
# png to look at; pdf and svg keep text and lines sharp at any size, for a paper.
FIGURE_FORMATS = ("png", "pdf", "svg")
DEFAULT_FORMATS = ("png",)

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
    outside_hours: int = 0  # frames left out for being outside the chosen hours
    daily: str | None = None  # the points are days, reduced with this statistic
    files: list[str] = field(default_factory=list)


def load_measurements(experiment: Experiment, plot: str | None = None,
                      results_dir: Path | None = None) -> list[dict]:
    """Measurement rows sorted by time; only ``plot``'s rows if given.

    ``results_dir`` reads another run's results instead of the experiment's own (used by
    `fungus sensitivity`).
    """
    path = (Path(results_dir) if results_dir else experiment.root / RESULTS_DIR) / \
        MEASUREMENTS_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `fungus analyze` first")
    with open(path, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["timestamp_utc"])
    for r in rows:
        r["plot"] = r.get("plot") or "main"
    return [r for r in rows if plot is None or r["plot"] == plot]


def list_plots(experiment: Experiment, results_dir: Path | None = None) -> list[str]:
    return sorted({r["plot"] for r in load_measurements(experiment, results_dir=results_dir)})


EDGE_METRICS = ("edge_advance_p95_mm", "edge_advance_max_mm",
                "edge_advance_p95_px", "edge_advance_max_px")


def add_edge_advance(experiment: Experiment, rows: list[dict], plot_name: str,
                     skip_flags: tuple[str, ...] = DEFAULT_EXCLUDE,
                     results_dir: Path | None = None) -> None:
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
    runs_dir = (Path(results_dir) if results_dir else experiment.root / RESULTS_DIR) / "runs"
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
    manual: list[str] = field(default_factory=list)  # reason a person excluded a frame, or ""
    outside_hours: np.ndarray | None = None  # taken outside the chosen hours of the day
    daily: str | None = None  # the statistic each day was reduced to, if this is daily
    frames: Series | None = None  # for a daily series: the frames it was made from

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


DAILY_STATS = ("median", "mean", "p90")


def parse_hours(text: str) -> tuple[float, float]:
    """``10-14`` or ``9:30-15`` (local hours, start inclusive); ``20-6`` wraps past midnight."""
    def hour(part: str) -> float:
        h, _, m = part.strip().partition(":")
        value = int(h) + (int(m) / 60 if m else 0.0)
        if not 0 <= value <= 24:
            raise ValueError
        return value

    try:
        start, end = (hour(x) for x in text.split("-"))
    except ValueError:
        raise ValueError(f"hours must look like 10-14 or 9:30-15, not {text!r}") from None
    if start == end:
        raise ValueError(f"empty hours window {text!r}")
    return start, end


def in_hours(local: datetime, hours: tuple[float, float]) -> bool:
    h = local.hour + local.minute / 60 + local.second / 3600
    start, end = hours
    return start <= h < end if start < end else (h >= start or h < end)


def load_series(
    experiment: Experiment,
    metric: str | None = None,
    plot: str | None = None,
    t0: datetime | None = None,
    time_unit: str = "auto",
    exclude_flags: tuple[str, ...] = DEFAULT_EXCLUDE,
    exclude_jumps: bool = False,
    results_dir: Path | None = None,
    hours: tuple[float, float] | None = None,
    daily: str | None = None,
) -> Series:
    """One metric over time, with the frames that go into fits marked.

    ``hours`` keeps only frames taken in that window of the local day (the experiment's
    ``timezone``): for outdoor runs, leave out night and low sun. ``daily`` then reduces each
    local day's usable frames to one value (see `daily_series`).
    """
    if daily is not None and daily not in DAILY_STATS:
        raise ValueError(f"daily must be one of {list(DAILY_STATS)}, not {daily!r}")
    plots = list_plots(experiment, results_dir)
    if plot is None:
        if len(plots) > 1:
            raise ValueError(f"several plots {plots}; choose one")
        plot = plots[0] if plots else "main"
    rows = load_measurements(experiment, plot, results_dir)
    if not rows:
        raise ValueError(f"no measurements for plot {plot!r}")
    if metric is None:
        for candidate in ("extent_mm", "extent_px", "target_area_mm2", "coverage_pct"):
            if rows[0].get(candidate, "") != "":
                metric = candidate
                break
    if metric in EDGE_METRICS:
        add_edge_advance(experiment, rows, plot, exclude_flags, results_dir)
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

    from fungus_cv.analyze import exclusions

    flags = [set(filter(None, r["flags"].split(";"))) for r in rows]
    # Frames a person excluded are always left out, whatever the flag settings.
    marked = exclusions.load(experiment)
    manual = [exclusions.reason_for(marked, r["frame_file"], plot) for r in rows]
    excluded = np.array([bool(f & set(exclude_flags)) for f in flags]) | np.isnan(y) | (t < 0)
    excluded |= np.array([bool(m) for m in manual])
    outside = None
    if hours is not None:
        tz = experiment.config.tzinfo()
        outside = np.array([not in_hours(datetime.fromtimestamp(ts, timezone.utc).astimezone(tz),
                                         hours) for ts in times])
        excluded |= outside
    if daily is not None:
        frames = _finish_series(Series(metric, plot, rows, time_unit, t0_ts, t, y, unc, excluded,
                                       np.zeros(len(rows), bool), np.zeros(len(rows), bool),
                                       manual, outside), exclude_jumps)
        return _finish_series(daily_series(frames, daily, experiment.config.tzinfo()),
                              exclude_jumps)
    return _finish_series(Series(metric, plot, rows, time_unit, t0_ts, t, y, unc, excluded,
                                 np.zeros(len(rows), bool), np.zeros(len(rows), bool), manual,
                                 outside), exclude_jumps)


def _finish_series(series: Series, exclude_jumps: bool) -> Series:
    """Mark jumps and retreats among the usable points (and drop jumps if asked)."""
    t, y, unc, excluded = series.t, series.y, series.unc, series.excluded
    jump = detect_jumps(t, np.where(excluded, np.nan, y), unc) & ~excluded
    if exclude_jumps:
        excluded = excluded | jump
    retreat = count_retreats(np.where(excluded, np.nan, y), unc) & ~excluded
    series.excluded, series.jump, series.retreat = excluded, jump, retreat
    return series


def _statistic(values: np.ndarray, stat: str) -> float:
    if stat == "mean":
        return float(np.mean(values))
    if stat == "p90":
        return float(np.percentile(values, 90))
    return float(np.median(values))


def daily_series(frames: Series, stat: str = "median", tz=None, n_boot: int = 400,
                 seed: int = 0) -> Series:
    """One point per local day from the usable frames: the day's ``stat`` (median, mean or
    90th percentile, as phenology cameras use for greenness).

    Its uncertainty adds, in quadrature, the frames' mean reported uncertainty (errors that
    every frame of a day shares) and the standard error of the statistic from the scatter
    within the day (bootstrap over the day's frames; with two frames, sd/sqrt(2)). A day
    with a single frame has no scatter to go on, so only the reported uncertainty counts.
    """
    use = frames.use
    stamps = frames.t0_ts + frames.t * TIME_UNITS[frames.time_unit]
    by_day: dict = {}
    for i in np.flatnonzero(use):
        day = datetime.fromtimestamp(stamps[i], timezone.utc).astimezone(tz).date()
        by_day.setdefault(day, []).append(i)
    rng = np.random.default_rng(seed)
    rows, t, y, unc = [], [], [], []
    for day in sorted(by_day):
        idx = np.array(by_day[day])
        values = frames.y[idx]
        value = _statistic(values, stat)
        if len(idx) >= 3:
            picks = rng.integers(0, len(idx), (n_boot, len(idx)))
            se = float(np.std([_statistic(values[p], stat) for p in picks], ddof=1))
        elif len(idx) == 2:
            se = float(np.std(values, ddof=1) / math.sqrt(2))
        else:
            se = math.nan
        u_frames = frames.unc[idx]
        u_shared = float(np.mean(u_frames)) if np.isfinite(u_frames).all() else math.nan
        parts = [v for v in (u_shared, se) if math.isfinite(v)]
        u = math.sqrt(sum(v * v for v in parts)) if parts else math.nan
        middle = float(np.median(stamps[idx]))
        last = frames.rows[idx[-1]]
        rows.append({
            "timestamp_utc": iso_utc(datetime.fromtimestamp(middle, timezone.utc)),
            "frame_file": f"{day.isoformat()} ({len(idx)} frames)",
            "plot": frames.plot, "flags": "", frames.metric: value,
            f"{frames.metric}_unc": u, "overlay_file": last.get("overlay_file", ""),
            "n_frames": len(idx),
            **{k: last.get(k, "") for k in ("mean_brightness", "align_shift_px",
                                             "align_rms_px", "coverage_pct",
                                             "light_gain_g")},
        })
        t.append((middle - frames.t0_ts) / TIME_UNITS[frames.time_unit])
        y.append(value)
        unc.append(u)
    n = len(rows)
    if not n:
        raise ValueError("no usable frames left to make daily values from")
    return Series(frames.metric, frames.plot, rows, frames.time_unit, frames.t0_ts,
                  np.array(t), np.array(y), np.array(unc), np.zeros(n, bool),
                  np.zeros(n, bool), np.zeros(n, bool), [""] * n, None, stat, frames)


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
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    dpi: int = 150,
    bootstrap: int = 1000,
    errors: str = "auto",
    seed: int = 0,
    results_dir: Path | None = None,
    time_to: tuple[float, ...] = (),
    hours: tuple[float, float] | None = None,
    daily: str | None = None,
) -> ReportResult:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = list_plots(experiment, results_dir)
    series = load_series(experiment, metric, plot, t0, time_unit, exclude_flags, exclude_jumps,
                         results_dir, hours=hours, daily=daily)
    metric, plot, rows, time_unit = series.metric, series.plot, series.rows, series.time_unit
    t, y, unc = series.t, series.y, series.unc
    excluded, jump, retreat, use = series.excluded, series.jump, series.retreat, series.use
    t0_ts = series.t0_ts
    sigma = series.sigma
    fits = fit_all(t[use], y[use], models=models, sigma=sigma, bootstrap=bootstrap,
                   errors=errors, seed=seed, levels=time_to)

    out_dir = (Path(results_dir) if results_dir else experiment.root / RESULTS_DIR) / "report"
    if plots != ["main"]:
        out_dir = out_dir / plot
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ReportResult(
        metric=metric, plot=plot, time_unit=time_unit,
        t0_utc=iso_utc(datetime.fromtimestamp(t0_ts, timezone.utc)),
        n_used=int(use.sum()), n_excluded=int(excluded.sum()),
        retreats=int(retreat.sum()), jumps=int(jump.sum()), fits=fits,
        outside_hours=int((series.frames or series).outside_hours.sum())
        if (series.frames or series).outside_hours is not None else 0,
        daily=series.daily,
    )

    # Every frame's flags in one place, so exclusions can be audited.
    flags_csv = out_dir / "frame_flags.csv"
    with open(flags_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "frame_file", f"t_{time_unit}", metric,
                         "pipeline_flags", "jump", "retreat", "excluded_by_hand",
                         "outside_hours", "used_in_fits"])
        framewise = series.frames or series  # daily: list the frames behind the days
        for i, r in enumerate(framewise.rows):
            outside = framewise.outside_hours is not None and framewise.outside_hours[i]
            writer.writerow([r["timestamp_utc"], r["frame_file"],
                             round(float(framewise.t[i]), 6), r[metric], r["flags"],
                             int(framewise.jump[i]), int(framewise.retreat[i]),
                             framewise.manual[i], int(outside), int(framewise.use[i])])
    result.files.append(flags_csv)
    if series.daily:
        daily_csv = out_dir / "daily.csv"
        with open(daily_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["day", "timestamp_utc", f"t_{time_unit}", f"{metric}_{daily}",
                             f"{metric}_unc", "frames", "jump", "retreat"])
            for i, r in enumerate(rows):
                writer.writerow([r["frame_file"].split(" ")[0], r["timestamp_utc"],
                                 round(float(t[i]), 6), round(float(y[i]), 6),
                                 "" if not math.isfinite(unc[i]) else round(float(unc[i]), 6),
                                 r["n_frames"], int(jump[i]), int(retreat[i])])
        result.files.append(daily_csv)

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
    if metric.startswith("class_") and metric.endswith("_pct"):
        label = f"Share {metric[6:-4]} (%)"
    if series.daily:
        label = f"{label} — daily {daily}"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _style(ax)
    if np.isfinite(unc[use]).any():
        ax.errorbar(t[use], y[use], yerr=unc[use], fmt="o", ms=4, color=INK_2, ecolor=GRID,
                    elinewidth=1, capsize=0, label="measured", zorder=3)
    else:
        ax.plot(t[use], y[use], "o", ms=4, color=INK_2, label="measured", zorder=3)
    if series.daily:  # the frames behind each day, faintly
        frames = series.frames
        ax.plot(frames.t[frames.use], frames.y[frames.use], ".", ms=3, color=GRID,
                label="frames", zorder=1)
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
        best_model = best_fit(fits)
        for i, (color, fit) in enumerate(zip(SERIES, fits)):
            if fit is best_model:
                band = fit.band(grid)
                if band is not None:  # the best model's 95% band, from the bootstrap
                    ax.fill_between(grid, *band, color=color, alpha=0.18, lw=0, zorder=1.5,
                                    label=f"{fit.model} 95% band")
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
    result.files += save_figure(fig, out_dir, f"{metric}_vs_time", formats, dpi)

    # --- quality control: one measure per panel, shared time axis --------------------
    qc = [("mean_brightness", "Brightness (0-255)"), ("align_shift_px", "Alignment shift (px)"),
          ("align_rms_px", "Alignment residual (px)"), ("coverage_pct", "Coverage (%)")]
    if any(r.get("light_gain_g") not in ("", None) for r in rows):
        qc.insert(1, ("light_gain_g", "Lighting gain (green)"))
    fig, axes = plt.subplots(len(qc), 1, figsize=(8, 8), dpi=150, sharex=True)
    for ax, (col, title) in zip(axes, qc):
        _style(ax)
        framewise = series.frames or series
        values = np.array([_float(r.get(col, "")) for r in framewise.rows])
        ax.plot(framewise.t, values, color=SERIES[0], lw=2)
        ax.set_ylabel(title, color=INK, fontsize=9)
    axes[-1].set_xlabel(f"Time since start ({time_unit})", color=INK)
    axes[0].set_title("Quality checks", color=INK, loc="left", fontsize=12)
    fig.tight_layout()
    result.files += save_figure(fig, out_dir, "quality_checks", formats, dpi)

    # --- residuals of the best model, in units of the reported uncertainty ------------
    best = best_fit(fits)
    if best is not None and use.sum() > 2:
        residual = y[use] - best.predict(t[use])
        scaled = np.isfinite(unc[use]).all() and np.all(unc[use] > 0)
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6), dpi=dpi, sharex=True,
                                          height_ratios=[2, 1])
        for ax in (top, bottom):
            _style(ax)
        top.plot(t[use], y[use], "o", ms=4, color=INK_2, label="measured", zorder=3)
        grid = np.linspace(max(0.0, t[use].min()), t[use].max(), 300)
        band = best.band(grid)
        if band is not None:
            top.fill_between(grid, *band, color=SERIES[0], alpha=0.18, lw=0,
                             label="95% band (bootstrap)")
        top.plot(grid, best.predict(grid), color=SERIES[0], lw=2,
                 label=f"{best.model}: {best.formula}")
        top.set_ylabel(label, color=INK)
        top.legend(frameon=False, fontsize=9, labelcolor=INK)
        top.set_title(f"{where}: best model and what it misses", color=INK, loc="left",
                      fontsize=12)
        bottom.axhline(0, color=GRID, lw=2)
        if scaled:
            bottom.plot(t[use], residual / unc[use], "o", ms=4, color=SERIES[1])
            for level in (-2, 2):
                bottom.axhline(level, color=GRID, lw=1, ls=(0, (6, 3)))
            bottom.set_ylabel("Residual / uncertainty", color=INK)
        else:
            bottom.plot(t[use], residual, "o", ms=4, color=SERIES[1])
            bottom.set_ylabel(f"Residual ({time_unit})", color=INK)
        bottom.set_xlabel(f"Time since start ({time_unit})", color=INK)
        fig.tight_layout()
        result.files += save_figure(fig, out_dir, "fit_and_residuals", formats, dpi)

    summary = {
        "metric": metric, "plot": plot, "time_unit": time_unit, "t0": result.t0_utc,
        "n_used": result.n_used, "n_excluded": result.n_excluded,
        "excluded_flags": list(exclude_flags), "retreats": result.retreats,
        "jumps": result.jumps, "jumps_excluded": exclude_jumps,
        "excluded_by_hand": [{"frame_file": r["frame_file"], "reason": m}
                             for r, m in zip((series.frames or series).rows,
                                             (series.frames or series).manual) if m],
        "hours": list(hours) if hours else None,
        "timezone": experiment.config.timezone or "this computer's",
        "daily": daily,
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


def save_figure(fig, out_dir: Path, stem: str, formats=DEFAULT_FORMATS,
                dpi: int = 150) -> list[Path]:
    """Save one figure in each format (png for looking at, pdf/svg for publication)."""
    import matplotlib.pyplot as plt

    paths = []
    for fmt in formats:
        if fmt not in FIGURE_FORMATS:
            raise ValueError(f"unknown figure format {fmt!r}; use {list(FIGURE_FORMATS)}")
        path = Path(out_dir) / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    plt.close(fig)
    return paths


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
