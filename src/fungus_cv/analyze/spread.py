"""Where and how fast does a patch spread? Arrival-time maps and speed by direction.

From the masks of an analysis run, every point gets the time it was first covered for good
(covered in ``persistence`` usable frames in a row, so one noisy frame can't mark it early).
The map shows at a glance where growth started and which way it went.

Speed by direction: from the centre of the first patch, the front is followed toward each of
``sectors`` directions (0° = right, 90° = up in the image; after rectification, the marker
plane). Its distance is the 95th percentile of covered points within a narrow wedge (±5°) of
the direction, robust to specks, and a straight line through time gives the speed with its
standard error. (A wide wedge would mix in the slower neighbouring directions and
underestimate an elongated patch's fastest speed.) Equal speeds mean even
spread; a large ratio between the fastest and slowest direction means the patch grows one
way (slope, wind, water, light).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fungus_cv.analyze.pipeline import annotations_path, results_dir_for
from fungus_cv.analyze.report import (
    DEFAULT_EXCLUDE,
    DEFAULT_FORMATS,
    INK,
    SERIES,
    load_series,
    save_figure,
)
from fungus_cv.measure.geometry import Annotations
from fungus_cv.storage import Experiment

WEDGE_HALF_DEG = 5.0  # the front toward a direction is measured within ± this angle


@dataclass
class SectorSpeed:
    angle_deg: float  # centre of the sector: 0 = right, 90 = up
    speed: float  # front advance per time unit (mm, or px without a scale)
    se: float
    r2: float
    n: int  # frames used in the fit
    distances: list[float] = field(default_factory=list)  # front distance per frame


@dataclass
class SpreadResult:
    plot: str
    time_unit: str
    unit: str  # mm or px
    arrival: np.ndarray  # time each pixel was reached (NaN = never), in time_unit
    origin: tuple[float, float]
    times: list[float]
    sectors: list[SectorSpeed] = field(default_factory=list)
    area_speed: float = math.nan  # growth of the equivalent radius, per time unit
    files: list[Path] = field(default_factory=list)

    @property
    def mean_speed(self) -> float:
        values = [s.speed for s in self.sectors if math.isfinite(s.speed)]
        return float(np.mean(values)) if values else math.nan

    @property
    def anisotropy(self) -> float:
        """Fastest over slowest direction (1 = even spread)."""
        values = [s.speed for s in self.sectors if math.isfinite(s.speed) and s.speed > 0]
        return max(values) / min(values) if len(values) > 1 else math.nan

    @property
    def fastest(self) -> SectorSpeed | None:
        ok = [s for s in self.sectors if math.isfinite(s.speed)]
        return max(ok, key=lambda s: s.speed) if ok else None


def arrival_times(masks, times, persistence: int = 2) -> np.ndarray:
    """First time each pixel is covered in ``persistence`` consecutive frames (NaN = never).

    ``masks`` is an iterable of boolean arrays in time order; they are read one at a time.
    """
    arrival = run = start = None
    for mask, t in zip(masks, times):
        if arrival is None:
            arrival = np.full(mask.shape, np.nan, np.float64)
            run = np.zeros(mask.shape, np.int32)
            start = np.zeros(mask.shape, np.float64)
        begins = mask & (run == 0)
        start[begins] = t
        run = np.where(mask, run + 1, 0)
        settled = (run >= persistence) & np.isnan(arrival)
        arrival[settled] = start[settled]
    if arrival is None:
        raise ValueError("no masks to build an arrival map from")
    return arrival


def _fit_line(t: np.ndarray, d: np.ndarray) -> tuple[float, float, float]:
    """Slope, its standard error and R² of d = a + b·t."""
    if len(t) < 3 or np.ptp(t) == 0:
        return math.nan, math.nan, math.nan
    b, a = np.polyfit(t, d, 1)
    resid = d - (a + b * t)
    dof = len(t) - 2
    s2 = float(resid @ resid) / dof if dof > 0 else math.nan
    se = math.sqrt(s2 / float(((t - t.mean()) ** 2).sum()))
    ss = float(((d - d.mean()) ** 2).sum())
    return float(b), se, 1 - float(resid @ resid) / ss if ss > 0 else math.nan


def spread(
    experiment: Experiment,
    plot: str | None = None,
    camera: str | None = None,
    persistence: int = 2,
    sectors: int = 8,
    exclude_flags: tuple[str, ...] = DEFAULT_EXCLUDE,
    time_unit: str = "auto",
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    out_dir: Path | None = None,
) -> SpreadResult:
    results_dir = results_dir_for(experiment, camera)
    series = load_series(experiment, "coverage_pct", plot, time_unit=time_unit,
                         exclude_flags=exclude_flags, results_dir=results_dir)
    plot = series.plot
    used = [i for i in range(len(series.rows))
            if series.use[i] and series.rows[i].get("mask_file")
            and (experiment.root / series.rows[i]["mask_file"]).exists()]
    if len(used) < 3:
        raise ValueError("need at least 3 usable frames with saved masks "
                         "(analysis.save_masks: true)")
    first = cv2.imread(str(experiment.root / series.rows[used[0]]["mask_file"]),
                       cv2.IMREAD_GRAYSCALE)
    plots = {p.name: p for p in
             Annotations.load(annotations_path(experiment, camera)).measured_plots()}
    region = plots[plot].mask(first.shape)

    def masks():
        for i in used:
            img = cv2.imread(str(experiment.root / series.rows[i]["mask_file"]),
                             cv2.IMREAD_GRAYSCALE)
            yield (img > 127) & region

    times = [float(series.t[i]) for i in used]
    arrival = arrival_times(masks(), times, persistence)
    arrival[~region] = np.nan

    mm_per_px = _scale(results_dir, series.rows[used[0]].get("settings_hash", ""))
    unit, k = ("mm", mm_per_px) if mm_per_px else ("px", 1.0)

    # Origin: the centre of the first patch (the earliest points reached).
    reached = np.isfinite(arrival)
    if not reached.any():
        raise ValueError(f"nothing in plot {plot!r} was ever covered")
    earliest = np.nanmin(arrival)
    ys, xs = np.nonzero(reached & (arrival <= earliest + 1e-9))
    origin = (float(xs.mean()), float(ys.mean()))

    yy, xx = np.nonzero(reached)
    t_reach = arrival[yy, xx]
    dx, dy = xx - origin[0], origin[1] - yy  # y up
    distance = np.hypot(dx, dy) * k
    angle = np.degrees(np.arctan2(dy, dx)) % 360
    width = 360 / sectors
    half = min(width / 2, WEDGE_HALF_DEG)
    result = SpreadResult(plot=plot, time_unit=series.time_unit, unit=unit, arrival=arrival,
                          origin=origin, times=times)
    t_arr = np.array(times)
    for s in range(sectors):
        centre = s * width
        inside = np.abs((angle - centre + 180) % 360 - 180) <= half
        dist_s, reach_s = distance[inside], t_reach[inside]
        fronts = []
        for t in times:
            covered = dist_s[reach_s <= t]
            fronts.append(float(np.percentile(covered, 95)) if len(covered) else 0.0)
        fronts_arr = np.array(fronts)
        growing = fronts_arr > 0
        speed, se, r2 = _fit_line(t_arr[growing], fronts_arr[growing])
        result.sectors.append(SectorSpeed(centre, speed, se, r2, int(growing.sum()), fronts))

    areas = np.array([float((arrival <= t).sum()) * k * k for t in times])
    radius = np.sqrt(areas / math.pi)
    result.area_speed = _fit_line(t_arr[areas > 0], radius[areas > 0])[0]

    out = Path(out_dir) if out_dir else \
        results_dir / "spread" / (plot if plot != "main" else "")
    out.mkdir(parents=True, exist_ok=True)
    _write(result, out, formats, region)
    return result


def _scale(results_dir: Path, settings_hash: str) -> float | None:
    for path in (results_dir / "runs" / f"{settings_hash}.json", results_dir / "run_info.json"):
        if path.exists():
            scale = json.loads(path.read_text(encoding="utf-8")).get("scale") or {}
            if scale.get("mm_per_px"):
                return float(scale["mm_per_px"])
    return None


def _write(result: SpreadResult, out: Path, formats, region: np.ndarray) -> None:
    np.save(out / "arrival_time.npy", result.arrival.astype(np.float32))
    result.files.append(out / "arrival_time.npy")
    with open(out / "spread_by_direction.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["direction_deg", f"speed_{result.unit}_per_{result.time_unit}", "se",
                         "r2", "frames"])
        for s in result.sectors:
            writer.writerow([round(s.angle_deg, 1), _r(s.speed), _r(s.se), _r(s.r2), s.n])
    result.files.append(out / "spread_by_direction.csv")
    (out / "spread.json").write_text(json.dumps({
        "plot": result.plot, "time_unit": result.time_unit, "unit": result.unit,
        "origin_px": list(result.origin), "mean_speed": _r(result.mean_speed),
        "anisotropy": _r(result.anisotropy), "equivalent_radius_speed": _r(result.area_speed),
        "directions": [{"deg": s.angle_deg, "speed": _r(s.speed), "se": _r(s.se),
                        "r2": _r(s.r2), "frames": s.n} for s in result.sectors],
        "note": "0 deg = right, 90 deg = up in the (prepared) image",
    }, indent=2), encoding="utf-8")
    result.files.append(out / "spread.json")
    result.files += _figure(result, out, formats, region)


def _r(value: float):
    return None if value is None or not math.isfinite(value) else round(float(value), 6)


def _figure(result: SpreadResult, out: Path, formats, region: np.ndarray) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Show the reached area with a margin (inside the plot), so a small patch isn't a dot.
    ry, rx = np.nonzero(region)
    ys, xs = np.nonzero(np.isfinite(result.arrival))
    pad = int(0.25 * max(np.ptp(ys), np.ptp(xs), 20))
    y0, y1 = max(ys.min() - pad, ry.min()), min(ys.max() + 1 + pad, ry.max() + 1)
    x0, x1 = max(xs.min() - pad, rx.min()), min(xs.max() + 1 + pad, rx.max() + 1)
    arrival = result.arrival[y0:y1, x0:x1]
    fig = plt.figure(figsize=(11, 5), dpi=150)
    ax = fig.add_subplot(1, 2, 1)
    shown = ax.imshow(np.ma.masked_invalid(arrival), cmap="viridis",
                      extent=(x0, x1, y1, y0), interpolation="nearest")
    ax.contour(np.arange(x0, x1), np.arange(y0, y1), region[y0:y1, x0:x1].astype(float),
               levels=[0.5], colors=[INK], linewidths=1)
    ax.plot(*result.origin, "+", color="white", ms=12, mew=2)
    reach = max((s for s in result.sectors if math.isfinite(s.speed)),
                key=lambda s: s.speed, default=None)
    scale = 0.35 * max(x1 - x0, y1 - y0) / (reach.speed if reach and reach.speed > 0 else 1)
    for s in result.sectors:
        if math.isfinite(s.speed) and s.speed > 0:
            a = math.radians(s.angle_deg)
            ax.annotate("", xy=(result.origin[0] + math.cos(a) * s.speed * scale,
                                result.origin[1] - math.sin(a) * s.speed * scale),
                        xytext=result.origin,
                        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.5})
    cbar = fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"First covered ({result.time_unit})")
    ax.set_title(f"{result.plot}: when each point was reached", loc="left", fontsize=11,
                 color=INK)
    ax.set_xticks([])
    ax.set_yticks([])

    polar = fig.add_subplot(1, 2, 2, projection="polar")
    angles = np.radians([s.angle_deg for s in result.sectors])
    speeds = np.array([s.speed if math.isfinite(s.speed) else 0 for s in result.sectors])
    errors = np.array([s.se if math.isfinite(s.se) else 0 for s in result.sectors])
    polar.bar(angles, speeds, width=2 * math.pi / len(angles) * 0.8, color=SERIES[0],
              alpha=0.75, yerr=errors, ecolor=INK, capsize=2)
    polar.set_title(f"Front speed by direction ({result.unit}/{result.time_unit})\n"
                    f"fastest/slowest = {result.anisotropy:.2f}", fontsize=10, color=INK)
    fig.tight_layout()
    _style_polar(polar)
    return save_figure(fig, out, "spread_map", formats)


def _style_polar(ax) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.tick_params(labelsize=8)

