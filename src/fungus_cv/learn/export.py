"""Fill a dataset from an experiment's analysis run or from existing image/mask pairs."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from fungus_cv.analyze.compare import resolve_run
from fungus_cv.analyze.pipeline import Analyzer
from fungus_cv.learn.dataset import Dataset, Item, safe_id
from fungus_cv.segment.prompts import CropWindow
from fungus_cv.storage import Experiment

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def pick_evenly(n: int, count: int) -> list[int]:
    """``count`` indices spread across ``range(n)``, always including the first and last."""
    if count >= n:
        return list(range(n))
    if count <= 1:
        return [n - 1]
    return sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})


def export_from_run(
    experiment: Experiment,
    dataset: Dataset,
    run: str,
    count: int = 20,
    crop_to_roi: bool = True,
    margin_px: int = 32,
    group: str | None = None,
    replace: bool = False,
) -> list[Item]:
    """Copy aligned frames and a run's masks into the dataset as unreviewed labels.

    Frames are spread evenly over the time-lapse so early (small target) and late (large
    target) stages are both represented.
    """
    run_info = resolve_run(experiment, run)
    analyzer = Analyzer(experiment, with_segmenter=False)
    frames = analyzer.frames
    stems = {p.stem for p in run_info.masks_dir.glob("*.png")}
    candidates = [i for i, r in enumerate(frames) if Path(r["file"]).stem in stems]
    if not candidates:
        raise ValueError(f"run {run_info.run_id} has no masks for this experiment's frames")

    h, w = analyzer.reference.shape[:2]
    window = (CropWindow.around(analyzer.annotations.roi, margin_px, w, h) if crop_to_roi
              else CropWindow.full(w, h))
    group = group or experiment.config.name
    added = []
    for k in pick_evenly(len(candidates), count):
        index = candidates[k]
        row = frames[index]
        stem = Path(row["file"]).stem
        item_id = safe_id(f"{experiment.config.name}__{stem}")
        if dataset.get(item_id) and not replace:
            log.info("skipping %s (already in dataset)", item_id)
            continue
        mask = cv2.imread(str(run_info.masks_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
        image = analyzer.aligned_frame(index)
        added.append(dataset.add(
            window.crop(image), window.crop(mask), item_id, replace=replace,
            group=group,
            source=f"{experiment.root.name}/{row['file']} (run {run_info.run_id}, "
                   f"{run_info.method})",
            timestamp_utc=row["timestamp_utc"],
            crop=[window.x0, window.y0, window.x1, window.y1],
        ))
    dataset.save()
    return added


def add_pairs(
    dataset: Dataset,
    images_dir: Path,
    masks_dir: Path,
    group: str,
    reviewed: bool = False,
) -> list[Item]:
    """Add images with same-named masks (any non-zero mask pixel counts as target)."""
    masks = {p.stem: p for p in Path(masks_dir).iterdir()
             if p.suffix.lower() in IMAGE_EXTENSIONS}
    added = []
    for img_path in sorted(Path(images_dir).iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if img_path.stem not in masks:
            log.warning("no mask for %s; skipped", img_path.name)
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(masks[img_path.stem]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            log.warning("cannot read %s or its mask; skipped", img_path.name)
            continue
        added.append(dataset.add(
            image, mask > 0, f"{group}__{img_path.stem}", group=group,
            source=str(img_path), reviewed=reviewed,
        ))
    dataset.save()
    return added


def dataset_stats(dataset: Dataset) -> dict:
    groups: dict[str, dict] = {}
    for item in dataset.items:
        g = groups.setdefault(item.group, {"items": 0, "reviewed": 0, "target_fraction": []})
        g["items"] += 1
        g["reviewed"] += int(item.reviewed)
        g["target_fraction"].append(float(dataset.load_mask(item).mean()))
    for g in groups.values():
        fractions = g.pop("target_fraction")
        g["mean_target_pct"] = round(100 * float(np.mean(fractions)), 2) if fractions else 0.0
    return groups
