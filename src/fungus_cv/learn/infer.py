"""Load a trained model folder and predict masks, tiling large images."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fungus_cv.segment.torch_device import choose_device, import_torch

WEIGHTS_NAME = "model.pt"
CARD_NAME = "model.json"


@dataclass
class LoadedModel:
    net: object
    card: dict
    device: object
    path: Path
    weights_sha256: str

    @property
    def threshold(self) -> float:
        return float(self.card.get("threshold", 0.5))

    @property
    def classes(self) -> list[str]:
        return list(self.card.get("classes") or ["target"])

    def class_index(self, class_name: str | None) -> int:
        name = class_name or self.classes[0]
        if name not in self.classes:
            raise ValueError(f"model {self.path.name} has no class {name!r}; it has "
                             f"{self.classes}")
        return self.classes.index(name)

    def threshold_for(self, class_name: str | None) -> float:
        name = class_name or self.classes[0]
        return float((self.card.get("thresholds") or {}).get(name, self.threshold))


_CACHE: dict[tuple[str, str], LoadedModel] = {}


def load_trained(path: Path, device_pref: str = "auto") -> LoadedModel:
    path = Path(path)
    card_path, weights_path = path / CARD_NAME, path / WEIGHTS_NAME
    # Check the folder before importing torch, so a wrong path says so even without torch.
    if not card_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"{path} is not a trained model folder (needs {CARD_NAME} "
                                f"and {WEIGHTS_NAME}; create one with `fungus train`)")
    torch = import_torch()
    from fungus_cv.learn.unet import ResNetUNet

    device, _ = choose_device(device_pref)
    key = (str(path.resolve()), str(device))
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    cached = _CACHE.get(key)
    if cached and cached.weights_sha256 == digest:
        return cached
    card = json.loads(card_path.read_text(encoding="utf-8"))
    net = ResNetUNet(card["architecture"]["encoder"], pretrained=False,
                     classes=len(card.get("classes") or ["target"]))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.to(device).eval()
    loaded = LoadedModel(net, card, device, path, digest)
    _CACHE[key] = loaded
    return loaded


def _to_tensor(bgr: np.ndarray, torch, device):
    from fungus_cv.learn.unet import IMAGENET_MEAN, IMAGENET_STD

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)


def _pad_to(image: np.ndarray, height: int, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.copyMakeBorder(image, 0, height - h, 0, width - w, cv2.BORDER_REFLECT_101)


def _ceil_to(v: int, divisor: int) -> int:
    return -(-v // divisor) * divisor


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _ramp(tile: int, overlap: int) -> np.ndarray:
    """Blend weights that fade in over the overlap, so tile seams don't show."""
    w = np.ones(tile, np.float32)
    if overlap > 0:
        r = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
        w[:overlap] = r
        w[-overlap:] = r[::-1]
    return w


def predict_probabilities(
    net, image: np.ndarray, device, tile_px: int = 512, overlap_px: int = 64,
    predict_fn=None,
) -> np.ndarray:
    """Per-pixel probability for a BGR image of any size: H x W for a one-class model,
    H x W x classes otherwise.

    ``predict_fn(bgr_tile) -> logits (h, w) or (h, w, classes)`` can replace the network
    (used in tests).
    """
    from fungus_cv.learn.unet import DIVISOR

    torch = import_torch()
    h, w = image.shape[:2]
    tile = _ceil_to(max(tile_px, DIVISOR), DIVISOR)

    def logits_for(patch: np.ndarray) -> np.ndarray:
        ph, pw = patch.shape[:2]
        padded = _pad_to(patch, _ceil_to(ph, DIVISOR), _ceil_to(pw, DIVISOR))
        if predict_fn is not None:
            out = predict_fn(padded)
        else:
            with torch.no_grad():
                out = net(_to_tensor(padded, torch, device))[0].float().cpu().numpy()
            out = out[0] if out.shape[0] == 1 else np.moveaxis(out, 0, -1)
        return out[:ph, :pw]

    if h <= tile and w <= tile:
        return 1.0 / (1.0 + np.exp(-logits_for(image)))

    th, tw = min(tile, h), min(tile, w)
    overlap = min(overlap_px, th // 2, tw // 2)
    total = weight = None
    wy, wx = _ramp(th, overlap if h > th else 0), _ramp(tw, overlap if w > tw else 0)
    blend = np.outer(wy, wx) + 1e-6
    for y in _tile_starts(h, th, overlap):
        for x in _tile_starts(w, tw, overlap):
            logits = logits_for(image[y:y + th, x:x + tw])
            if total is None:  # the first tile says how many classes there are
                total = np.zeros((h, w, *logits.shape[2:]), np.float32)
                weight = np.zeros((h, w), np.float32)
            b = blend if logits.ndim == 2 else blend[..., None]
            total[y:y + th, x:x + tw] += logits * b
            weight[y:y + th, x:x + tw] += blend
    ratio = total / (weight if total.ndim == 2 else weight[..., None])
    return 1.0 / (1.0 + np.exp(-ratio))
