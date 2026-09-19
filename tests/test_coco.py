"""COCO exchange with labeling tools: exact RLE, polygons, and merging back."""

import json

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from fungus_cv.cli import app
from fungus_cv.learn.coco import (
    export_coco,
    import_coco,
    mask_from_polygons,
    mask_from_rle,
    mask_to_polygons,
    rle_counts,
    rle_decode_string,
    rle_encode_string,
)
from fungus_cv.learn.dataset import Dataset


def ring(h=60, w=80):
    """A mask with a hole, which polygons can't represent."""
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (40, 30), 20, 1, -1)
    cv2.circle(mask, (40, 30), 8, 0, -1)
    return mask.astype(bool)


def test_rle_by_hand_and_round_trip():
    mask = np.array([[0, 1], [1, 1]], bool)  # column by column: 0, 1 | 1, 1
    assert rle_counts(mask) == [1, 3]
    assert rle_encode_string([1, 3]) == "13"
    assert rle_counts(np.ones((2, 2), bool)) == [0, 4]  # starts with (no) background
    rng = np.random.default_rng(1)
    for _ in range(50):
        m = rng.random((17, 23)) < 0.4
        counts = rle_counts(m)
        assert rle_decode_string(rle_encode_string(counts)) == counts
        assert np.array_equal(mask_from_rle({"size": [17, 23],
                                             "counts": rle_encode_string(counts)}), m)
        assert np.array_equal(mask_from_rle({"size": [17, 23], "counts": counts}), m)


def test_matches_pycocotools():
    mask_api = pytest.importorskip("pycocotools.mask")
    rng = np.random.default_rng(2)
    for _ in range(100):
        m = rng.random((31, 29)) < rng.random()
        ref = mask_api.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()
        assert rle_encode_string(rle_counts(m)) == ref


def test_polygons_lose_holes_rle_does_not():
    m = ring()
    back = mask_from_polygons(mask_to_polygons(m), *m.shape)
    assert back[30, 40] and not m[30, 40]  # the hole is filled
    outer = back.sum()
    assert abs(outer - (m.sum() + 201)) < 60  # ~ the ring plus its hole


def dataset_with_rings(root):
    ds = Dataset.create(root)
    for i in range(3):
        image = np.full((60, 80, 3), 60 * i + 40, np.uint8)
        ds.add(image, ring() if i else np.zeros((60, 80), bool), f"item{i}",
               reviewed=i == 2)
    ds.save()
    return ds


def test_export_and_import_back_exactly_with_rle(tmp_path):
    ds = dataset_with_rings(tmp_path / "ds")
    path = export_coco(ds, tmp_path / "coco", rle=True)
    coco = json.loads(path.read_text())
    assert len(coco["images"]) == 3 and len(coco["annotations"]) == 2  # empty mask: none
    assert coco["annotations"][0]["iscrowd"] == 1
    assert (tmp_path / "coco" / "images" / "item1.png").exists()

    fresh = Dataset.create(tmp_path / "fresh")
    result = import_coco(fresh, path, reviewed=True)
    assert len(result.added) == 3 and not result.skipped
    for i in range(3):
        assert np.array_equal(fresh.load_mask(fresh.get(f"item{i}")),
                              ds.load_mask(ds.get(f"item{i}")))
    assert all(item.reviewed for item in fresh.items)

    only_reviewed = export_coco(ds, tmp_path / "rev", reviewed_only=True)
    assert len(json.loads(only_reviewed.read_text())["images"]) == 1


def test_corrections_from_a_tool_update_existing_items(tmp_path):
    ds = dataset_with_rings(tmp_path / "ds")
    path = export_coco(ds, tmp_path / "coco")  # polygons, as most tools use
    coco = json.loads(path.read_text())
    # Someone draws a new square on item0 in CVAT, with a second category we ignore.
    coco["categories"].append({"id": 2, "name": "stem"})
    coco["annotations"] += [
        {"id": 10, "image_id": 1, "category_id": 1, "iscrowd": 0,
         "segmentation": [[5, 5, 25, 5, 25, 25, 5, 25]]},
        {"id": 11, "image_id": 1, "category_id": 2, "iscrowd": 0,
         "segmentation": [[50, 5, 70, 5, 70, 25, 50, 25]]}]
    path.write_text(json.dumps(coco))
    result = import_coco(ds, path, categories=["target"])
    assert len(result.updated) == 3 and not result.added
    fixed = ds.load_mask(ds.get("item0"))
    assert fixed[15, 15] and not fixed[15, 60]  # target square in, stem left out
    assert not ds.get("item0").reviewed  # importing doesn't review on its own
    with pytest.raises(ValueError, match="no categories"):
        import_coco(ds, path, categories=["moss"])


def test_coco_cli(tmp_path):
    dataset_with_rings(tmp_path / "ds")
    runner = CliRunner()
    out = runner.invoke(app, ["dataset", "export-coco", str(tmp_path / "ds"),
                              str(tmp_path / "coco"), "--rle"])
    assert out.exit_code == 0 and "as RLE" in out.output
    back = runner.invoke(app, ["dataset", "import-coco", str(tmp_path / "new"),
                               str(tmp_path / "coco" / "annotations.json"), "--reviewed"])
    assert back.exit_code == 0 and "added 3" in back.output
