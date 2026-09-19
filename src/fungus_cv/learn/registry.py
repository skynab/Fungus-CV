"""Find and describe trained models (folders with a ``model.json`` card and ``model.pt``)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from fungus_cv.learn.infer import CARD_NAME, WEIGHTS_NAME


@dataclass
class ModelInfo:
    path: Path
    name: str
    created_utc: str
    encoder: str
    dataset: str
    fingerprint: str
    n_train: int
    n_val: int
    val_iou: float | None
    threshold: float | None
    has_profile: bool
    evaluations: dict = field(default_factory=dict)  # dataset name -> unseen IoU (or all)
    card: dict = field(default_factory=dict)


def read_model(path: Path) -> ModelInfo:
    path = Path(path)
    card = json.loads((path / CARD_NAME).read_text(encoding="utf-8"))
    data = card.get("dataset", {})
    val = (card.get("validation") or {}).get("summary") or {}
    evaluations = {}
    for csv_path in sorted(path.glob("evaluation_*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        unseen = [r for r in rows if r.get("in_training_set") in ("False", "false", "0")]
        chosen = unseen or rows
        if chosen:
            evaluations[csv_path.stem[len("evaluation_"):]] = {
                "iou": sum(float(r["iou"]) for r in chosen) / len(chosen),
                "n": len(chosen), "unseen": bool(unseen)}
    return ModelInfo(
        path=path, name=path.name, created_utc=card.get("created_utc", ""),
        encoder=(card.get("architecture") or {}).get("encoder", "?"),
        dataset=data.get("name", "?"), fingerprint=data.get("fingerprint", "")[:12],
        n_train=len(data.get("train_items", [])), n_val=len(data.get("val_items", [])),
        val_iou=val.get("iou_mean"), threshold=card.get("threshold"),
        has_profile="input_profile" in card, evaluations=evaluations, card=card)


def find_models(roots) -> list[ModelInfo]:
    """Every model folder under the given folders, newest first."""
    found = {}
    for root in roots:
        root = Path(root)
        candidates = [root] if (root / CARD_NAME).exists() else root.rglob(CARD_NAME)
        for card in candidates:
            folder = card if card.is_dir() else card.parent
            if (folder / WEIGHTS_NAME).exists() and folder.resolve() not in found:
                try:
                    found[folder.resolve()] = read_model(folder)
                except (OSError, ValueError, KeyError):
                    continue
    return sorted(found.values(), key=lambda m: m.created_utc, reverse=True)


def add_profile(model_dir: Path, dataset) -> dict:
    """Give an older model an input profile, from the items it was trained on."""
    from fungus_cv.learn.profile import build_profile

    path = Path(model_dir) / CARD_NAME
    card = json.loads(path.read_text(encoding="utf-8"))
    ids = card.get("dataset", {}).get("train_items") or []
    items = [dataset.get(i) for i in ids]
    items = [i for i in items if i is not None]
    if len(items) < 2:
        raise ValueError(f"only {len(items)} of the model's {len(ids)} training items are in "
                         f"{dataset.root}; point at the dataset it was trained on")
    card["input_profile"] = build_profile(dataset.load_image(i) for i in items)
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card["input_profile"]
