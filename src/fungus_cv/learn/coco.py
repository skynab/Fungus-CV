"""Exchange labels with CVAT, Label Studio and other tools through COCO JSON.

Export writes the dataset's images and one annotation per item and class (one COCO category
per dataset class): polygons (read by every tool, but holes in a mask are lost) or ``--rle``
run-length encoding (exact, and what CVAT itself writes for masks). Import reads either form,
including compressed RLE, rasterises it and writes it into the dataset: an item with the same
name is updated, a new image is added.

The run-length code follows COCO's own format (column-major counts, starting with the
background, compressed with its 6-bit variable-length code), so no extra package is needed.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fungus_cv.learn.dataset import Dataset, Item, safe_id

# --- run-length encoding -------------------------------------------------------------------


def rle_counts(mask: np.ndarray) -> list[int]:
    """Run lengths of a mask read column by column, starting with a run of background."""
    flat = np.asarray(mask, bool).flatten(order="F")
    changes = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    edges = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(edges).tolist()
    if flat.size and flat[0]:
        counts = [0] + counts  # COCO always starts with background
    return [int(c) for c in counts]


def rle_encode_string(counts: list[int]) -> str:
    """COCO's compressed form of the counts (as pycocotools writes it)."""
    out = []
    for i, x in enumerate(counts):
        if i > 2:
            x -= counts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def rle_decode_string(text: str) -> list[int]:
    counts: list[int] = []
    p = 0
    while p < len(text):
        x, k, more = 0, 0, True
        while more:
            c = ord(text[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def mask_from_rle(rle: dict) -> np.ndarray:
    height, width = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = rle_decode_string(counts)
    flat = np.zeros(height * width, bool)
    position, value = 0, False
    for run in counts:
        if value:
            flat[position:position + run] = True
        position += run
        value = not value
    return flat.reshape((height, width), order="F")


def mask_to_polygons(mask: np.ndarray, min_area: float = 4.0) -> list[list[float]]:
    """Outer outlines of a mask as COCO polygons (holes are not representable and are lost)."""
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area or len(contour) < 3:
            continue
        polygons.append([float(v) for point in contour[:, 0, :] for v in point])
    return polygons


def mask_from_polygons(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    for poly in polygons:
        points = np.array(poly, np.float64).reshape(-1, 2)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask.astype(bool)


# --- export --------------------------------------------------------------------------------


def export_coco(dataset: Dataset, out_dir: Path, rle: bool = False,
                reviewed_only: bool = False) -> Path:
    """Images in ``out_dir/images/`` and ``out_dir/annotations.json``."""
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for image_id, item in enumerate(dataset.selected(reviewed_only), start=1):
        labels = dataset.load_labels(item)
        h, w = labels.shape[:2]
        name = f"{item.id}.png"
        shutil.copyfile(dataset.root / item.image, out_dir / "images" / name)
        images.append({"id": image_id, "file_name": name, "width": w, "height": h,
                       "fungus_cv": {"reviewed": item.reviewed, "group": item.group,
                                     "source": item.source}})
        for k in range(labels.shape[-1]):  # one annotation per class present
            mask = labels[..., k]
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            entry = {"id": len(annotations) + 1, "image_id": image_id, "category_id": k + 1,
                     "area": int(mask.sum()),
                     "bbox": [int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1,
                              int(np.ptp(ys)) + 1]}
            if rle:
                entry.update(iscrowd=1, segmentation={
                    "size": [h, w], "counts": rle_encode_string(rle_counts(mask))})
            else:
                entry.update(iscrowd=0, segmentation=mask_to_polygons(mask))
            annotations.append(entry)
    coco = {"info": {"description": f"Fungus-CV dataset {dataset.name}"},
            "categories": [{"id": k + 1, "name": name}
                           for k, name in enumerate(dataset.classes)],
            "images": images, "annotations": annotations}
    path = out_dir / "annotations.json"
    path.write_text(json.dumps(coco, indent=1), encoding="utf-8")
    return path


# --- import --------------------------------------------------------------------------------


@dataclass
class ImportResult:
    updated: list[Item] = field(default_factory=list)
    added: list[Item] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def import_coco(dataset: Dataset, coco_json: Path, images_dir: Path | None = None,
                categories: list[str] | None = None, reviewed: bool = False,
                group: str = "imported") -> ImportResult:
    """Merge a COCO file's masks into the dataset.

    Categories map to the dataset's classes by name. A new, empty dataset takes the file's
    categories as its classes. A single-class dataset without those names merges every chosen
    category into its one class (as for a moss-only label set).
    """
    coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    images_dir = Path(images_dir) if images_dir else Path(coco_json).parent / "images"
    names = {c["id"]: c["name"] for c in coco.get("categories", [])}
    wanted = {cid for cid, name in names.items() if not categories or name in categories}
    if categories and not wanted:
        raise ValueError(f"no categories {categories} in the file; it has "
                         f"{sorted(names.values())}")
    chosen = [names[cid] for cid in sorted(wanted)]
    if not dataset.items and dataset.classes == ["target"] and len(chosen) > 1:
        dataset.classes = list(chosen)  # a new dataset takes the file's categories
    by_name = {c: c for c in chosen if c in dataset.classes}
    if len(by_name) < len(chosen) and not dataset.multi_class:
        by_name = {c: dataset.classes[0] for c in chosen}  # merge into the one class
    missing = [c for c in chosen if c not in by_name]
    if missing:
        raise ValueError(f"categories {missing} are not classes of this dataset "
                         f"({dataset.classes}); add them first or choose --category")
    by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_id") in wanted:
            by_image.setdefault(ann["image_id"], []).append(ann)

    result = ImportResult()
    for image in coco.get("images", []):
        h, w = int(image["height"]), int(image["width"])
        labels = np.zeros((h, w, len(dataset.classes)), bool)
        for ann in by_image.get(image["id"], []):
            k = dataset.classes.index(by_name[names[ann["category_id"]]])
            seg = ann.get("segmentation")
            if isinstance(seg, dict):
                if isinstance(seg.get("counts"), list) and seg.get("size") is None:
                    seg = {**seg, "size": [h, w]}
                labels[..., k] |= mask_from_rle(seg)
            elif isinstance(seg, list) and seg:
                labels[..., k] |= mask_from_polygons(seg, h, w)
        file_name = Path(image["file_name"]).name
        item = dataset.get(safe_id(Path(file_name).stem))
        imported = {by_name[c] for c in chosen}
        if item is not None:
            if dataset.load_mask(item).shape != (h, w):
                result.skipped.append(f"{file_name}: size differs from the dataset's image")
                continue
            for name in imported:  # only the imported classes change
                dataset.write_mask(item, labels[..., dataset.classes.index(name)], name)
            item.reviewed = item.reviewed or reviewed
            result.updated.append(item)
            continue
        source = images_dir / image["file_name"]
        if not source.exists():
            source = images_dir / file_name
        picture = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if picture is None:
            result.skipped.append(f"{file_name}: image not found in {images_dir}")
            continue
        if picture.shape[:2] != (h, w):
            result.skipped.append(f"{file_name}: image is {picture.shape[1]}x"
                                  f"{picture.shape[0]}, the file says {w}x{h}")
            continue
        result.added.append(dataset.add(picture, labels, Path(file_name).stem, group=group,
                                        source=f"COCO import from {Path(coco_json).name}",
                                        reviewed=reviewed))
    dataset.save()
    return result
