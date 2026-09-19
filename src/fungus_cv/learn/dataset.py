"""Labeled image datasets: a folder of images, binary masks and a ``dataset.json`` index.

::

    datasets/moss-bark/
      dataset.json
      images/<id>.png
      masks/<id>.png      # 0 = background, 255 = target

Masks are plain PNGs, so they can also be edited in other tools (GIMP, CVAT, Label Studio).
Only items marked ``reviewed`` (checked by a person) are used for training by default.

A dataset has one class (``target``) unless more are added: e.g. ``stem`` and ``moss``, which
may overlap (moss grows on the stem). Single-class masks are 0/255. With several classes each
mask pixel stores one bit per class (bit 0 = the first class, ...), up to 8 classes; use
`fungus dataset export-coco` to edit those in other tools.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fungus_cv.storage import iso_utc, utc_now

INDEX_NAME = "dataset.json"
DEFAULT_CLASS = "target"
MAX_CLASSES = 8  # one bit each in an 8-bit mask


@dataclass
class Item:
    id: str
    image: str  # relative to the dataset folder
    mask: str
    group: str = ""  # e.g. the experiment; train/validation splits never share a group
    source: str = ""  # where it came from (experiment + frame, or original file)
    timestamp_utc: str = ""
    crop: list[int] | None = None  # [x0, y0, x1, y1] in the aligned source frame
    reviewed: bool = False
    notes: str = ""
    # Active learning: how useful labeling this item is (higher first) and why.
    priority: float | None = None
    scores: dict = field(default_factory=dict)


@dataclass
class Dataset:
    root: Path
    name: str = ""
    description: str = ""
    items: list[Item] = field(default_factory=list)
    created_utc: str = ""
    classes: list[str] = field(default_factory=lambda: [DEFAULT_CLASS])

    # --- persistence -----------------------------------------------------------------

    @classmethod
    def create(cls, root: Path, name: str | None = None, description: str = "") -> Dataset:
        root = Path(root)
        if (root / INDEX_NAME).exists():
            raise FileExistsError(f"{root / INDEX_NAME} already exists")
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "masks").mkdir(exist_ok=True)
        ds = cls(root=root, name=name or root.name, description=description,
                 created_utc=iso_utc(utc_now()))
        ds.save()
        return ds

    @classmethod
    def open(cls, root: Path) -> Dataset:
        root = Path(root)
        path = root / INDEX_NAME
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; create it with `fungus dataset export`")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            root=root, name=data.get("name", root.name),
            description=data.get("description", ""),
            items=[Item(**item) for item in data.get("items", [])],
            created_utc=data.get("created_utc", ""),
            classes=list(data.get("classes") or [DEFAULT_CLASS]),
        )

    @classmethod
    def open_or_create(cls, root: Path, name: str | None = None) -> Dataset:
        return cls.open(root) if (Path(root) / INDEX_NAME).exists() else cls.create(root, name)

    def save(self) -> None:
        data = {
            "name": self.name,
            "description": self.description,
            "created_utc": self.created_utc,
            "classes": self.classes,
            "items": [asdict(i) for i in self.items],
        }
        tmp = self.root / (INDEX_NAME + ".part")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.root / INDEX_NAME)

    # --- items -----------------------------------------------------------------------

    def get(self, item_id: str) -> Item | None:
        return next((i for i in self.items if i.id == item_id), None)

    # --- classes ---------------------------------------------------------------------

    @property
    def multi_class(self) -> bool:
        return len(self.classes) > 1

    def class_bit(self, class_name: str | None) -> int:
        name = class_name or self.classes[0]
        if name not in self.classes:
            raise KeyError(f"no class {name!r} in this dataset; it has {self.classes}")
        return self.classes.index(name)

    def add_class(self, name: str) -> None:
        """Add a class. The first time, existing 0/255 masks become bit 0 (the first class)."""
        name = name.strip()
        if not name or name in self.classes:
            raise ValueError(f"class {name!r} is empty or already in the dataset")
        if len(self.classes) >= MAX_CLASSES:
            raise ValueError(f"at most {MAX_CLASSES} classes")
        if not self.multi_class:
            for item in self.items:
                if (self.root / item.mask).exists():
                    first = self._read_raw(item) > 127
                    cv2.imwrite(str(self.root / item.mask), first.astype(np.uint8))
        self.classes.append(name)
        self.save()

    def add(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        item_id: str,
        replace: bool = False,
        class_name: str | None = None,
        **fields,
    ) -> Item:
        """Add an image with its mask for ``class_name`` (default: the first class), or with
        every class at once when ``mask`` is H x W x classes."""
        item_id = safe_id(item_id)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"{item_id}: image {image.shape[:2]} and mask {mask.shape[:2]} differ")
        existing = self.get(item_id)
        if existing and not replace:
            raise FileExistsError(f"item {item_id} already exists")
        item = Item(id=item_id, image=f"images/{item_id}.png", mask=f"masks/{item_id}.png",
                    **fields)
        cv2.imwrite(str(self.root / item.image), image)
        if existing is None and (self.root / item.mask).exists():
            (self.root / item.mask).unlink()  # a stale file from a removed item
        if mask.ndim == 3:
            self.write_labels(item, mask)
        else:
            self.write_mask(item, mask, class_name)
        if existing:
            self.items[self.items.index(existing)] = item
        else:
            self.items.append(item)
        return item

    def load_image(self, item: Item) -> np.ndarray:
        img = cv2.imread(str(self.root / item.image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.root / item.image)
        return img

    def _read_raw(self, item: Item) -> np.ndarray:
        m = cv2.imread(str(self.root / item.mask), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(self.root / item.mask)
        return m

    def load_mask(self, item: Item, class_name: str | None = None) -> np.ndarray:
        """The mask of one class (default: the first)."""
        bit = self.class_bit(class_name)
        raw = self._read_raw(item)
        if not self.multi_class:
            return raw > 127
        return ((raw >> bit) & 1).astype(bool)

    def load_labels(self, item: Item) -> np.ndarray:
        """Every class at once: H x W x classes, booleans (classes may overlap)."""
        raw = self._read_raw(item)
        if not self.multi_class:
            return (raw > 127)[..., None]
        return np.stack([((raw >> k) & 1).astype(bool) for k in range(len(self.classes))], -1)

    def write_mask(self, item: Item, mask: np.ndarray, class_name: str | None = None) -> None:
        """Replace one class's mask, keeping the other classes."""
        bit = self.class_bit(class_name)
        path = self.root / item.mask
        mask = np.asarray(mask, bool)
        if not self.multi_class:
            cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
            return
        raw = self._read_raw(item) if path.exists() else np.zeros(mask.shape, np.uint8)
        if raw.shape != mask.shape:
            raise ValueError(f"{item.id}: mask {mask.shape} differs from {raw.shape}")
        raw = (raw & np.uint8(~(1 << bit) & 0xFF)) | (mask.astype(np.uint8) << bit)
        cv2.imwrite(str(path), raw)

    def write_labels(self, item: Item, labels: np.ndarray) -> None:
        labels = np.asarray(labels, bool)
        if labels.shape[-1] != len(self.classes):
            raise ValueError(f"{item.id}: {labels.shape[-1]} class layer(s) for "
                             f"{len(self.classes)} classes")
        if not self.multi_class:
            self.write_mask(item, labels[..., 0])
            return
        raw = np.zeros(labels.shape[:2], np.uint8)
        for k in range(labels.shape[-1]):
            raw |= labels[..., k].astype(np.uint8) << k
        cv2.imwrite(str(self.root / item.mask), raw)

    def selected(self, reviewed_only: bool = True) -> list[Item]:
        return [i for i in self.items if i.reviewed or not reviewed_only]

    def fingerprint(self, items: list[Item] | None = None) -> str:
        """Hash of image and mask contents, so a trained model records exactly its data."""
        h = hashlib.sha256()
        if self.multi_class:  # single-class fingerprints stay what they were
            h.update(json.dumps(self.classes).encode())
        for item in sorted(items if items is not None else self.items, key=lambda i: i.id):
            h.update(item.id.encode())
            for rel in (item.image, item.mask):
                h.update(hashlib.sha256((self.root / rel).read_bytes()).digest())
        return h.hexdigest()


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "item"


# --- splitting -----------------------------------------------------------------------------


def split_items(
    items: list[Item],
    val_fraction: float = 0.2,
    val_groups: list[str] | None = None,
) -> tuple[list[Item], list[Item], str]:
    """Split into train/validation without leakage between near-identical frames.

    - ``val_groups`` given: those groups are validation.
    - Several groups: whole groups go to validation (about ``val_fraction`` of items).
    - One group (a single time-lapse): the last ``val_fraction`` of it in time is validation,
      so validation frames are not sandwiched between near-identical training frames.
    Returns (train, val, description of the strategy).
    """
    if not items:
        raise ValueError("no items to split")
    groups = sorted({i.group for i in items})
    if val_groups:
        unknown = set(val_groups) - set(groups)
        if unknown:
            raise ValueError(f"unknown validation groups {sorted(unknown)}; have {groups}")
        val = [i for i in items if i.group in val_groups]
        train = [i for i in items if i.group not in val_groups]
        how = f"validation groups {sorted(val_groups)}"
    elif len(groups) > 1:
        sizes = {g: sum(1 for i in items if i.group == g) for g in groups}
        target = max(1, round(val_fraction * len(items)))
        chosen, count = [], 0
        # Deterministic: smallest groups first until the target size is reached.
        for g in sorted(groups, key=lambda g: (sizes[g], g)):
            if count >= target or len(chosen) == len(groups) - 1:
                break
            chosen.append(g)
            count += sizes[g]
        val = [i for i in items if i.group in chosen]
        train = [i for i in items if i.group not in chosen]
        how = f"held-out groups {chosen}"
    else:
        ordered = sorted(items, key=lambda i: (i.timestamp_utc, i.id))
        n_val = max(1, round(val_fraction * len(ordered))) if len(ordered) > 1 else 0
        train, val = ordered[: len(ordered) - n_val], ordered[len(ordered) - n_val:]
        how = f"last {n_val} item(s) in time of group {groups[0]!r}"
    if not train:
        raise ValueError("split left no training items; add more labeled images")
    return train, val, how
