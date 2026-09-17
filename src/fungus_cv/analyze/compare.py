"""Compare the masks of two analysis runs (e.g. color threshold vs SAM 2, or vs hand labels)."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fungus_cv.analyze.pipeline import annotations_path, results_dir_for
from fungus_cv.measure.geometry import Annotations, measure_extent
from fungus_cv.storage import Experiment


@dataclass
class RunInfo:
    run_id: str
    method: str
    updated_utc: str
    n_masks: int
    masks_dir: Path
    mm_per_px: float | None


def list_runs(experiment: Experiment, camera: str | None = None) -> list[RunInfo]:
    """Analysis runs that still have masks on disk, oldest first."""
    results = results_dir_for(experiment, camera)
    runs = []
    for masks_dir in sorted((results / "masks").glob("*/")):
        info_path = results / "runs" / f"{masks_dir.name}.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        target = info.get("settings", {}).get("analysis", {}).get("target", {})
        runs.append(RunInfo(
            run_id=masks_dir.name,
            method=target.get("method", "?"),
            updated_utc=info.get("updated_utc", ""),
            n_masks=sum(1 for _ in masks_dir.glob("*.png")),
            masks_dir=masks_dir,
            mm_per_px=(info.get("scale") or {}).get("mm_per_px"),
        ))
    return sorted(runs, key=lambda r: r.updated_utc)


def resolve_run(experiment: Experiment, run: str, camera: str | None = None) -> RunInfo:
    """A run id, a unique prefix of one, or a folder of mask PNGs (e.g. hand labels)."""
    path = Path(run)
    if path.is_dir():
        return RunInfo(run_id=path.name, method="folder", updated_utc="",
                       n_masks=sum(1 for _ in path.glob("*.png")), masks_dir=path,
                       mm_per_px=None)
    matches = [r for r in list_runs(experiment, camera) if r.run_id.startswith(run)]
    if len(matches) != 1:
        raise ValueError(f"run {run!r} matches {len(matches)} runs; see `fungus runs`")
    return matches[0]


def _load_mask(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if img is None else img > 127


@dataclass
class ComparisonSummary:
    run_a: str
    run_b: str
    n_frames: int
    iou_mean: float
    iou_median: float
    iou_min: float
    worst_frame: str
    extent_diff_mean: float  # b - a
    extent_diff_sd: float
    extent_unit: str
    csv_path: Path


def compare_runs(experiment: Experiment, run_a: str, run_b: str,
                 camera: str | None = None) -> ComparisonSummary:
    a = resolve_run(experiment, run_a, camera)
    b = resolve_run(experiment, run_b, camera)
    ann = Annotations.load(annotations_path(experiment, camera))
    roi = None
    mm_per_px = a.mm_per_px or b.mm_per_px
    unit = "mm" if mm_per_px else "px"
    k = mm_per_px or 1.0

    stems = sorted({p.stem for p in a.masks_dir.glob("*.png")} &
                   {p.stem for p in b.masks_dir.glob("*.png")})
    if not stems:
        raise ValueError(f"runs {a.run_id} and {b.run_id} have no frames in common")

    out_dir = results_dir_for(experiment, camera) / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{a.run_id}_vs_{b.run_id}.csv"
    rows = []
    for stem in stems:
        ma, mb = _load_mask(a.masks_dir / f"{stem}.png"), _load_mask(b.masks_dir / f"{stem}.png")
        if ma is None or mb is None or ma.shape != mb.shape:
            continue
        if roi is None:
            roi = ann.roi_mask(ma.shape)
        ma, mb = ma & roi, mb & roi
        union = int((ma | mb).sum())
        inter = int((ma & mb).sum())
        iou = inter / union if union else 1.0  # both empty = full agreement
        if ann.base is not None:
            ea = measure_extent(ma, ann).extent_px * k
            eb = measure_extent(mb, ann).extent_px * k
        else:  # field plots only: compare areas instead
            ea, eb = float(ma.sum()) * k * k, float(mb.sum()) * k * k
        rows.append({
            "frame": stem, "iou": round(iou, 5),
            "area_a_px": int(ma.sum()), "area_b_px": int(mb.sum()),
            f"extent_a_{unit}": round(ea, 3), f"extent_b_{unit}": round(eb, 3),
            f"extent_diff_{unit}": round(eb - ea, 3),
        })
    if not rows:
        raise ValueError("no comparable mask pairs (unreadable or different sizes)")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ious = np.array([r["iou"] for r in rows])
    diffs = np.array([r[f"extent_diff_{unit}"] for r in rows])
    worst = rows[int(np.argmin(ious))]["frame"]
    return ComparisonSummary(
        run_a=a.run_id, run_b=b.run_id, n_frames=len(rows),
        iou_mean=float(ious.mean()), iou_median=float(np.median(ious)),
        iou_min=float(ious.min()), worst_frame=worst,
        extent_diff_mean=float(diffs.mean()),
        extent_diff_sd=float(diffs.std(ddof=1)) if len(diffs) > 1 else math.nan,
        extent_unit=unit, csv_path=csv_path,
    )
