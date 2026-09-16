"""Train a segmentation model on a labeled dataset and write a model folder with a card."""

from __future__ import annotations

import json
import logging
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fungus_cv import __version__
from fungus_cv.learn.dataset import Dataset, Item, split_items
from fungus_cv.learn.infer import CARD_NAME, WEIGHTS_NAME, predict_probabilities
from fungus_cv.learn.metrics import mask_metrics, summarize
from fungus_cv.segment.torch_device import choose_device, import_torch
from fungus_cv.storage import iso_utc, utc_now

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    encoder: str = "resnet34"
    pretrained: bool = True
    patch_px: int = 384
    batch_size: int = 8
    steps: int = 3000
    eval_every: int = 250
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    val_groups: list[str] = field(default_factory=list)
    reviewed_only: bool = True
    # Augmentation. Colour jitter is kept small by default because colour is often the
    # signal (dye, discolouration); raise it if lighting varies a lot between experiments.
    flip_horizontal: bool = True
    flip_vertical: bool = False  # growth direction usually matters (up a stem)
    rotate90: bool = False
    scale_jitter: float = 0.2  # random zoom in [1 - s, 1 + s]
    brightness_jitter: float = 0.15
    contrast_jitter: float = 0.15
    hue_jitter_deg: float = 4.0
    saturation_jitter: float = 0.15
    target_patch_fraction: float = 0.5  # share of patches centred on target pixels
    seed: int = 0
    device: str = "auto"
    tile_px: int = 512
    overlap_px: int = 64


# --- data ---------------------------------------------------------------------------------


class PatchSampler:
    def __init__(self, images: list[np.ndarray], masks: list[np.ndarray], cfg: TrainConfig,
                 rng: np.random.Generator):
        self.images, self.masks, self.cfg, self.rng = images, masks, cfg, rng
        self.target_coords = [np.argwhere(m) for m in masks]

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        cfg, rng = self.cfg, self.rng
        k = int(rng.integers(len(self.images)))
        image, mask = self.images[k], self.masks[k]

        scale = 1.0 + rng.uniform(-cfg.scale_jitter, cfg.scale_jitter)
        size = max(8, int(round(cfg.patch_px / scale)))  # source window before resizing
        h, w = mask.shape
        coords = self.target_coords[k]
        if len(coords) and rng.random() < cfg.target_patch_fraction:
            cy, cx = coords[int(rng.integers(len(coords)))]
        else:
            cy, cx = int(rng.integers(h)), int(rng.integers(w))
        y0 = int(np.clip(cy - size // 2, 0, max(0, h - size)))
        x0 = int(np.clip(cx - size // 2, 0, max(0, w - size)))
        img = image[y0:y0 + size, x0:x0 + size]
        msk = mask[y0:y0 + size, x0:x0 + size].astype(np.uint8)
        if img.shape[0] < size or img.shape[1] < size:  # image smaller than the window
            pad = ((0, size - img.shape[0]), (0, size - img.shape[1]))
            img = np.pad(img, pad + ((0, 0),), mode="reflect")
            msk = np.pad(msk, pad, mode="reflect")
        p = cfg.patch_px
        img = cv2.resize(img, (p, p), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (p, p), interpolation=cv2.INTER_NEAREST)

        if cfg.flip_horizontal and rng.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        if cfg.flip_vertical and rng.random() < 0.5:
            img, msk = img[::-1], msk[::-1]
        if cfg.rotate90:
            turns = int(rng.integers(4))
            img, msk = np.rot90(img, turns), np.rot90(msk, turns)
        return self._jitter(np.ascontiguousarray(img)), np.ascontiguousarray(msk)

    def _jitter(self, bgr: np.ndarray) -> np.ndarray:
        cfg, rng = self.cfg, self.rng
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-1, 1) * cfg.hue_jitter_deg / 2) % 180
        hsv[..., 1] *= 1 + rng.uniform(-cfg.saturation_jitter, cfg.saturation_jitter)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        out = out.astype(np.float32)
        mean = out.mean()
        out = (out - mean) * (1 + rng.uniform(-cfg.contrast_jitter, cfg.contrast_jitter)) + mean
        out *= 1 + rng.uniform(-cfg.brightness_jitter, cfg.brightness_jitter)
        return np.clip(out, 0, 255).astype(np.uint8)


def _batch(sampler: PatchSampler, n: int, torch, device):
    from fungus_cv.learn.unet import IMAGENET_MEAN, IMAGENET_STD

    imgs, msks = zip(*(sampler.sample() for _ in range(n)))
    x = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]).astype(np.float32) / 255
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    y = np.stack(msks).astype(np.float32)[:, None]
    return (torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(device),
            torch.from_numpy(y).to(device))


def _loss(logits, target, torch):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    # Dice over the whole batch: per-patch Dice would heavily penalise tiny stray
    # probabilities on patches that contain no target at all.
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + target.sum() + 1)
    return bce + dice


# --- training ------------------------------------------------------------------------------


@dataclass
class TrainResult:
    out_dir: Path
    best_val_iou: float | None
    threshold: float
    val_metrics: dict
    n_train: int
    n_val: int
    split: str
    seconds: float


def _seed(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate(net, items: list[Item], dataset: Dataset, cfg: TrainConfig, device,
              threshold: float | None = 0.5) -> tuple[list[dict], list[np.ndarray]]:
    torch = import_torch()
    net.eval()
    rows, probs = [], []
    with torch.no_grad():
        for item in items:
            prob = predict_probabilities(net, dataset.load_image(item), device,
                                         cfg.tile_px, cfg.overlap_px)
            probs.append(prob)
            if threshold is not None:
                rows.append({"id": item.id, **mask_metrics(prob >= threshold,
                                                           dataset.load_mask(item))})
    net.train()
    return rows, probs


def tune_threshold(probs: list[np.ndarray], truths: list[np.ndarray]) -> tuple[float, float]:
    """Threshold maximising pooled IoU over validation images."""
    best = (0.5, -1.0)
    for t in np.round(np.arange(0.2, 0.81, 0.05), 2):
        tp = fp = fn = 0
        for p, m in zip(probs, truths):
            pred = p >= t
            tp += int((pred & m).sum())
            fp += int((pred & ~m).sum())
            fn += int((~pred & m).sum())
        iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        if iou > best[1]:
            best = (float(t), iou)
    return best


def train(dataset: Dataset, out_dir: Path, cfg: TrainConfig, progress=None) -> TrainResult:
    torch = import_torch()
    from fungus_cv.learn.unet import ResNetUNet

    out_dir = Path(out_dir)
    if (out_dir / WEIGHTS_NAME).exists():
        raise FileExistsError(f"{out_dir} already contains a model; choose a new folder")
    items = dataset.selected(cfg.reviewed_only)
    if not items:
        hint = " (none are marked reviewed; use `fungus label` or --include-unreviewed)" \
            if cfg.reviewed_only else ""
        raise ValueError(f"dataset has no usable items{hint}")
    train_items, val_items, split = split_items(items, cfg.val_fraction, cfg.val_groups or None)
    log.info("training on %d item(s), validating on %d (%s)", len(train_items),
             len(val_items), split)

    _seed(cfg.seed, torch)
    device, _ = choose_device(cfg.device)
    net = ResNetUNet(cfg.encoder, pretrained=cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.learning_rate,
                                                    total_steps=cfg.steps, pct_start=0.1)
    sampler = PatchSampler([dataset.load_image(i) for i in train_items],
                           [dataset.load_mask(i) for i in train_items], cfg,
                           np.random.default_rng(cfg.seed))

    out_dir.mkdir(parents=True, exist_ok=True)
    best_iou, best_state, history = -1.0, None, []
    started = time.time()
    net.train()
    running = 0.0
    for step in range(1, cfg.steps + 1):
        x, y = _batch(sampler, cfg.batch_size, torch, device)
        loss = _loss(net(x), y, torch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running += float(loss.detach())

        if step % cfg.eval_every == 0 or step == cfg.steps:
            entry = {"step": step, "loss": running / cfg.eval_every if step % cfg.eval_every == 0
                     else running / (step % cfg.eval_every)}
            running = 0.0
            if val_items:
                rows, _ = _evaluate(net, val_items, dataset, cfg, device)
                entry["val_iou"] = summarize(rows)["iou_mean"]
                if entry["val_iou"] > best_iou:
                    best_iou = entry["val_iou"]
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            history.append(entry)
            log.info("step %d/%d  loss %.4f%s", step, cfg.steps, entry["loss"],
                     f"  val IoU {entry['val_iou']:.4f}" if "val_iou" in entry else "")
            if progress:
                progress(entry)

    if best_state is None:  # no validation data: keep the final weights
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    threshold, val_metrics, val_rows = 0.5, {}, []
    if val_items:
        _, probs = _evaluate(net, val_items, dataset, cfg, device, threshold=None)
        truths = [dataset.load_mask(i) for i in val_items]
        threshold, _ = tune_threshold(probs, truths)
        val_rows = [{"id": i.id, **mask_metrics(p >= threshold, t)}
                    for i, p, t in zip(val_items, probs, truths)]
        val_metrics = summarize(val_rows)

    torch.save(best_state, out_dir / WEIGHTS_NAME)
    seconds = time.time() - started
    card = {
        "fungus_cv_version": __version__,
        "created_utc": iso_utc(utc_now()),
        "architecture": {"name": "ResNetUNet", "encoder": cfg.encoder,
                         "input": "BGR->RGB, ImageNet normalisation"},
        "threshold": threshold,
        "dataset": {
            "name": dataset.name, "path": str(dataset.root.resolve()),
            "fingerprint": dataset.fingerprint(items),
            "train_items": [i.id for i in train_items],
            "val_items": [i.id for i in val_items],
            "split": split,
        },
        "validation": {"summary": val_metrics, "per_item": val_rows,
                       "note": "metrics use the tuned threshold; validation data also picked "
                               "the best checkpoint, so confirm on separate data with "
                               "`fungus evaluate`"},
        "training": {"config": asdict(cfg), "history": history, "seconds": round(seconds, 1),
                     "device": str(device), "torch": torch.__version__,
                     "python": platform.python_version(), "platform": platform.platform()},
    }
    (out_dir / CARD_NAME).write_text(json.dumps(card, indent=2), encoding="utf-8")
    return TrainResult(out_dir, best_iou if val_items else None, threshold, val_metrics,
                       len(train_items), len(val_items), split, seconds)


def evaluate_model(model_dir: Path, dataset: Dataset, reviewed_only: bool = True,
                   device: str = "auto", threshold: float | None = None) -> tuple[list, dict]:
    from fungus_cv.learn.infer import load_trained

    loaded = load_trained(model_dir, device)
    t = threshold if threshold is not None else loaded.threshold
    tile = loaded.card["training"]["config"].get("tile_px", 512)
    overlap = loaded.card["training"]["config"].get("overlap_px", 64)
    trained_on = set(loaded.card["dataset"]["train_items"])
    rows = []
    for item in dataset.selected(reviewed_only):
        prob = predict_probabilities(loaded.net, dataset.load_image(item), loaded.device,
                                     tile, overlap)
        rows.append({"id": item.id, "group": item.group, "in_training_set": item.id in trained_on,
                     **mask_metrics(prob >= t, dataset.load_mask(item))})
    unseen = [r for r in rows if not r["in_training_set"]]
    return rows, {"all": summarize(rows), "not_in_training_set": summarize(unseen),
                  "threshold": t}
