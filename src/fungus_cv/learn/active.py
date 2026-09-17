"""Active learning: label the frames a model (or two methods) is least sure about first.

Labeling evenly spaced frames spends most effort on frames the model already gets right.
These scores find the informative ones instead:

- ``tta_disagreement``: 1 - IoU between the model's mask and its masks for the same image
  flipped left-right and made slightly darker and brighter. A confident model gives the
  same answer; an unsure one changes its mind.
- ``ambiguous_fraction``: pixels with probability between 0.25 and 0.75, as a share of those
  and the predicted target together. High when the model can't tell where the edge is.

Both are relative to the target's size, so a soft edge counts for less on a large target;
compare scores within a dataset rather than reading them as absolute.
- ``disagreement``: 1 - IoU between two masks of the same frame, e.g. the model against a
  SAM run, or SAM against the colour threshold. Needs no trained model, so it also works
  for the very first round of labels.

The ``score`` is the mean of the components available, from 0 (confident) to 1.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fungus_cv.learn.dataset import Dataset, Item
from fungus_cv.storage import Experiment, iso_utc, utc_now

log = logging.getLogger(__name__)

ProbFn = Callable[[np.ndarray], np.ndarray]  # BGR image -> per-pixel target probability
AMBIGUOUS = (0.25, 0.75)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 1.0  # both empty = full agreement


def model_prob_fn(model_dir: Path, device: str = "auto") -> tuple[ProbFn, float]:
    """Probability function and tuned threshold of a model trained with `fungus train`."""
    from fungus_cv.learn.infer import load_trained, predict_probabilities

    loaded = load_trained(Path(model_dir), device)
    config = loaded.card["training"]["config"]
    tile, overlap = config.get("tile_px", 512), config.get("overlap_px", 64)

    def prob(image: np.ndarray) -> np.ndarray:
        return predict_probabilities(loaded.net, image, loaded.device, tile, overlap)

    return prob, loaded.threshold


def tta_probabilities(prob_fn: ProbFn, image: np.ndarray) -> list[np.ndarray]:
    """Probabilities for the image as is, flipped left-right, darker and brighter.

    No upside-down flips: growth direction often matters, as in training.
    """
    darker = np.clip(image.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
    brighter = np.clip(image.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    return [prob_fn(image), np.fliplr(prob_fn(np.ascontiguousarray(np.fliplr(image)))),
            prob_fn(darker), prob_fn(brighter)]


def model_scores(probs: list[np.ndarray], threshold: float) -> dict:
    masks = [p >= threshold for p in probs]
    mean = np.mean(probs, axis=0)
    ambiguous_mask = (mean > AMBIGUOUS[0]) & (mean < AMBIGUOUS[1])
    ambiguous = int(ambiguous_mask.sum())
    relevant = int((ambiguous_mask | (mean >= threshold)).sum())
    floor = max(1, int(0.001 * mean.size))  # a few noisy pixels alone aren't "unsure"
    return {
        "tta_disagreement": 1 - float(np.mean([iou(masks[0], m) for m in masks[1:]])),
        "ambiguous_fraction": ambiguous / max(relevant, floor) if ambiguous >= floor else 0.0,
    }


def combine(components: dict) -> float:
    values = [v for v in components.values() if v is not None]
    return float(np.mean(values)) if values else 0.0


def pick_diverse(scores: list[float], count: int, min_gap: int) -> list[int]:
    """Highest scores first, but no two picks within ``min_gap`` positions (near-identical
    consecutive frames); if that leaves too few, the gap is relaxed."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    chosen: list[int] = []
    for gap in (min_gap, 0):
        for i in order:
            if len(chosen) >= count:
                break
            if i not in chosen and all(abs(i - j) >= max(gap, 1) for j in chosen):
                chosen.append(i)
    return sorted(chosen)


# --- suggesting frames from an experiment --------------------------------------------------


@dataclass
class Suggestion:
    candidates: list[dict]  # every scored frame, with ``selected``
    added: list[Item] = field(default_factory=list)
    csv_path: Path | None = None


def suggest_frames(
    experiment: Experiment,
    dataset: Dataset,
    count: int = 20,
    model_dir: Path | None = None,
    run: str | None = None,
    against: str | None = None,
    max_candidates: int = 200,
    crop_to_roi: bool = True,
    margin_px: int = 32,
    group: str | None = None,
    device: str = "auto",
    prob_fn: ProbFn | None = None,
    threshold: float | None = None,
    camera: str | None = None,
    progress=None,
) -> Suggestion:
    """Score frames and add the ``count`` most useful ones to the dataset (unreviewed).

    The starting mask is ``run``'s mask when given, otherwise the model's prediction.
    ``prob_fn``/``threshold`` replace loading ``model_dir`` (used in tests).
    """
    from fungus_cv.analyze.compare import resolve_run
    from fungus_cv.analyze.pipeline import Analyzer
    from fungus_cv.learn.export import frame_item_id, frame_window, pick_evenly

    if prob_fn is None and model_dir is not None:
        prob_fn, threshold = model_prob_fn(model_dir, device)
    if prob_fn is None and not (run and against):
        raise ValueError("give a trained model (--model), or two runs to compare "
                         "(--run and --against)")
    runs = [resolve_run(experiment, r, camera) for r in (run, against) if r]
    analyzer = Analyzer(experiment, with_segmenter=False, camera=camera)
    window = frame_window(analyzer, crop_to_roi, margin_px)
    frames = analyzer.frames

    def run_mask(info, stem):
        img = cv2.imread(str(info.masks_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        return None if img is None else window.crop(img > 127)

    candidates = []
    for i, row in enumerate(frames):
        stem = Path(row["file"]).stem
        if dataset.get(frame_item_id(experiment, stem)):
            continue
        if any(not (info.masks_dir / f"{stem}.png").exists() for info in runs):
            continue
        candidates.append(i)
    if not candidates:
        raise ValueError("no frames to score (all already in the dataset, or the runs have "
                         "no masks for them)")
    candidates = [candidates[k] for k in pick_evenly(len(candidates), max_candidates)]

    rows, masks = [], {}
    for n, index in enumerate(candidates, 1):
        row = frames[index]
        stem = Path(row["file"]).stem
        image = window.crop(analyzer.aligned_frame(index))
        components: dict = {}
        run_masks = [run_mask(info, stem) for info in runs]
        start_mask = run_masks[0] if runs else None
        if prob_fn is not None:
            probs = tta_probabilities(prob_fn, image)
            components.update(model_scores(probs, threshold))
            predicted = probs[0] >= threshold
            if runs:
                components["disagreement"] = 1 - iou(predicted, run_masks[0])
            else:
                start_mask = predicted
        if len(runs) == 2:
            components["run_disagreement"] = 1 - iou(run_masks[0], run_masks[1])
        masks[index] = start_mask
        rows.append({"frame": row["file"], "timestamp_utc": row["timestamp_utc"],
                     "index": index, "score": combine(components), **components,
                     "selected": False})
        if progress:
            progress(n, len(candidates))

    min_gap = max(1, len(rows) // (2 * max(count, 1)))
    chosen = pick_diverse([r["score"] for r in rows], count, min_gap)
    result = Suggestion(rows)
    group = group or experiment.config.name
    method = "model" if prob_fn is not None and not runs else f"run {runs[0].run_id}"
    for k in chosen:
        r = rows[k]
        r["selected"] = True
        stem = Path(r["frame"]).stem
        scores = {key: round(float(r[key]), 4) for key in r
                  if key in ("tta_disagreement", "ambiguous_fraction", "disagreement",
                             "run_disagreement")}
        result.added.append(dataset.add(
            window.crop(analyzer.aligned_frame(r["index"])), masks[r["index"]],
            frame_item_id(experiment, stem), group=group,
            source=f"{experiment.root.name}/{r['frame']} (suggested; mask from {method})",
            timestamp_utc=r["timestamp_utc"],
            crop=[window.x0, window.y0, window.x1, window.y1],
            priority=round(r["score"], 4), scores=scores,
        ))
    dataset.save()

    out = dataset.root / "suggestions"
    out.mkdir(exist_ok=True)
    result.csv_path = out / f"{experiment.config.name}_{utc_now():%Y%m%dT%H%M%SZ}.csv"
    fields = ["frame", "timestamp_utc", "score", "tta_disagreement", "ambiguous_fraction",
              "disagreement", "run_disagreement", "selected"]
    with open(result.csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: -r["score"]))
    return result


# --- ranking items already in a dataset ----------------------------------------------------


def rank_items(
    dataset: Dataset,
    model_dir: Path | None = None,
    include_reviewed: bool = False,
    device: str = "auto",
    prob_fn: ProbFn | None = None,
    threshold: float | None = None,
    progress=None,
) -> list[Item]:
    """Score items with a model and store ``priority``; returns them most useful first.

    Besides the model's own uncertainty, ``disagreement`` compares its prediction with the
    item's current mask (e.g. a SAM pre-label): big differences are worth a look.
    """
    if prob_fn is None:
        if model_dir is None:
            raise ValueError("ranking needs a trained model")
        prob_fn, threshold = model_prob_fn(model_dir, device)
    items = [i for i in dataset.items if include_reviewed or not i.reviewed]
    for n, item in enumerate(items, 1):
        probs = tta_probabilities(prob_fn, dataset.load_image(item))
        components = model_scores(probs, threshold)
        components["disagreement"] = 1 - iou(probs[0] >= threshold, dataset.load_mask(item))
        item.scores = {k: round(float(v), 4) for k, v in components.items()}
        item.scores["ranked_utc"] = iso_utc(utc_now())
        item.priority = round(combine(components), 4)
        if progress:
            progress(n, len(items))
    dataset.save()
    return sorted(items, key=lambda i: -(i.priority or 0))


def by_priority(items: list[Item]) -> list[Item]:
    """Most useful first; items never scored keep their order at the end."""
    scored = sorted((i for i in items if i.priority is not None), key=lambda i: -i.priority)
    return scored + [i for i in items if i.priority is None]
