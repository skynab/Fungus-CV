"""Combine several cameras' views of the same object into one measurement.

A camera sees a stem's length foreshortened: parts leaning toward or away from it look
shorter, so **every view underestimates** an arc length or an extent. With two or more
cameras, the largest value across the views is the closest to the truth (still a lower
bound). Areas and coverage percentages are different: they are not simply underestimated,
so for those the views are reported side by side and averaged only if asked.

Frames from different cameras are matched by time, within a tolerance.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from fungus_cv.analyze.pipeline import RESULTS_DIR, cameras_with_frames, results_dir_for
from fungus_cv.analyze.report import load_measurements
from fungus_cv.storage import Experiment, iso_utc, parse_iso_utc, utc_now

log = logging.getLogger(__name__)

# Metrics where a view can only fall short of the truth (projection shortens them).
LOWER_BOUND_METRICS = ("extent_mm", "extent_max_mm", "extent_px", "axis_length_mm",
                       "covered_length_mm", "equivalent_radius_mm")
METHODS = ("max", "mean", "median")


@dataclass
class CombinedRow:
    timestamp_utc: str
    plot: str
    value: float
    method: str
    n_views: int
    best_camera: str
    spread: float  # largest minus smallest across the views
    per_camera: dict = field(default_factory=dict)


@dataclass
class CombineResult:
    metric: str
    method: str
    cameras: list[str]
    rows: list[CombinedRow] = field(default_factory=list)
    unmatched: int = 0
    out_path: Path | None = None
    note: str = ""

    @property
    def mean_spread(self) -> float:
        values = [r.spread for r in self.rows if r.n_views > 1]
        return sum(values) / len(values) if values else math.nan


def _rows_by_camera(experiment: Experiment, cameras: list[str], metric: str,
                    plot: str | None) -> dict[str, list[dict]]:
    out = {}
    for camera in cameras:
        rows = load_measurements(experiment, plot, results_dir_for(experiment, camera))
        if rows and metric not in rows[0]:
            raise ValueError(f"camera {camera}: no column {metric!r}; analyze it first")
        out[camera] = rows
    return out


def combine(
    experiment: Experiment,
    metric: str = "extent_mm",
    cameras: list[str] | None = None,
    method: str = "max",
    plot: str | None = None,
    tolerance_s: float = 60.0,
    out_path: Path | None = None,
) -> CombineResult:
    """One value per moment from several cameras (default: the largest, see the module note)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}")
    cameras = cameras or cameras_with_frames(experiment)
    if len(cameras) < 2:
        raise ValueError("combining needs at least two cameras with frames")
    per_camera = _rows_by_camera(experiment, cameras, metric, plot)
    reference, *others = cameras
    result = CombineResult(metric=metric, method=method, cameras=list(cameras))
    if method == "max" and metric not in LOWER_BOUND_METRICS:
        result.note = (f"{metric} is not a length, so 'max' is not obviously right here; "
                       "mean or median may suit it better")

    times = {c: [(parse_iso_utc(r["timestamp_utc"]), r) for r in rows]
             for c, rows in per_camera.items()}
    for moment, row in times[reference]:
        group = {reference: row}
        for camera in others:
            nearest = min(times[camera], key=lambda pair: abs((pair[0] - moment).total_seconds()),
                          default=None)
            if nearest and abs((nearest[0] - moment).total_seconds()) <= tolerance_s:
                group[camera] = nearest[1]
        values = {c: float(r[metric]) for c, r in group.items()
                  if r.get(metric) not in ("", None)}
        if not values:
            result.unmatched += 1
            continue
        if len(values) < len(cameras):
            result.unmatched += 1
        ordered = sorted(values.values())
        if method == "max":
            value = ordered[-1]
        elif method == "mean":
            value = sum(ordered) / len(ordered)
        else:
            mid = len(ordered) // 2
            value = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        best = max(values, key=lambda c: values[c])
        result.rows.append(CombinedRow(
            timestamp_utc=row["timestamp_utc"], plot=row.get("plot") or "main", value=value,
            method=method, n_views=len(values), best_camera=best,
            spread=ordered[-1] - ordered[0], per_camera=values))

    result.out_path = Path(out_path) if out_path else \
        experiment.root / RESULTS_DIR / "combined" / f"{metric}_{method}.csv"
    _write(result)
    return result


def _write(result: CombineResult) -> None:
    path = result.out_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp_utc", "plot", result.metric, "method", "n_views", "best_camera",
              "spread"] + [f"{result.metric}_{c}" for c in result.cameras]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        for row in result.rows:
            writer.writerow({
                "timestamp_utc": row.timestamp_utc, "plot": row.plot,
                result.metric: round(row.value, 6), "method": row.method,
                "n_views": row.n_views, "best_camera": row.best_camera,
                "spread": round(row.spread, 6),
                **{f"{result.metric}_{c}": round(v, 6) for c, v in row.per_camera.items()},
            })
    note = path.with_name(path.stem + "_README.txt")
    note.write_text(
        f"{result.metric} combined from cameras {', '.join(result.cameras)} with '"
        f"{result.method}', {iso_utc(utc_now())}.\n\n"
        "A camera shortens whatever leans toward or away from it, so each view underestimates "
        "a length: the largest of the views is the closest to the truth and is still a lower "
        "bound. 'spread' (largest minus smallest view) shows how much the views disagree; if "
        "it is large, the object is far from perpendicular to at least one camera.\n"
        + (f"\nNote: {result.note}\n" if result.note else ""), encoding="utf-8")
