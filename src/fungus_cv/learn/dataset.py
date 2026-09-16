"""Labeled image datasets: a folder of images, binary masks and a ``dataset.json`` index.

::

    datasets/moss-bark/
      dataset.json
      images/<id>.png
      masks/<id>.png      # 0 = background, 255 = target

Masks are plain PNGs, so they can also be edited in other tools (GIMP, CVAT, Label Studio).
Only items marked ``reviewed`` (checked by a person) are used for training by default.
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


@dataclass
class Dataset:
    root: Path
    name: str = ""
    description: str = ""
    items: list[Item] = field(default_factory=list)
    created_utc: str = ""

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
        )

    @classmethod
    def open_or_create(cls, root: Path, name: str | None = None) -> Dataset:
        return cls.open(root) if (Path(root) / INDEX_NAME).exists() else cls.create(root, name)

    def save(self) -> None:
        data = {
            "name": self.name,
            "description": self.description,
            "created_utc": self.created_utc,
            "items": [asdict(i) for i in self.items],
        }
        tmp = self.root / (INDEX_NAME + ".part")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.root / INDEX_NAME)

    # --- items -----------------------------------------------------------------------

    def get(self, item_id: str) -> Item | None:
        return next((i for i in self.items if i.id == item_id), None)

    def add(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        item_id: str,
        replace: bool = False,
        **fields,
    ) -> Item:
        item_id = safe_id(item_id)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"{item_id}: image {image.shape[:2]} and mask {mask.shape[:2]} differ")
        existing = self.get(item_id)
        if existing and not replace:
            raise FileExistsError(f"item {item_id} already exists")
        item = Item(id=item_id, image=f"images/{item_id}.png", mask=f"masks/{item_id}.png",
                    **fields)
        cv2.imwrite(str(self.root / item.image), image)
        self.write_mask(item, mask)
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

    def load_mask(self, item: Item) -> np.ndarray:
        m = cv2.imread(str(self.root / item.mask), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(self.root / item.mask)
        return m > 127

    def write_mask(self, item: Item, mask: np.ndarray) -> None:
        cv2.imwrite(str(self.root / item.mask), mask.astype(np.uint8) * 255)

    def selected(self, reviewed_only: bool = True) -> list[Item]:
        return [i for i in self.items if i.reviewed or not reviewed_only]

    def fingerprint(self, items: list[Item] | None = None) -> str:
        """Hash of image and mask contents, so a trained model records exactly its data."""
        h = hashlib.sha256()
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
