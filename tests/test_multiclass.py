"""Several classes in one dataset and one model (e.g. stem and moss, which overlap)."""

import json

import cv2
import numpy as np
import pytest

from fungus_cv.learn.dataset import Dataset

BROWN, GREEN, BACK = (40, 70, 110), (60, 170, 70), (200, 205, 210)


def plant(seed: int, size=128):
    """A stem (vertical bar) with moss patches on it: the moss is part of the stem too."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), BACK, np.uint8)
    stem = np.zeros((size, size), bool)
    x = int(rng.integers(40, 88))
    stem[10:size - 10, x - 8:x + 8] = True
    image[stem] = BROWN
    moss = np.zeros((size, size), np.uint8)
    for _ in range(int(rng.integers(1, 3))):
        cv2.circle(moss, (x + int(rng.integers(-4, 5)), int(rng.integers(25, size - 25))),
                   int(rng.integers(8, 14)), 1, -1)
    moss = moss.astype(bool) & stem
    image[moss] = GREEN
    noise = rng.normal(0, 6, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8), np.stack([stem, moss], -1)


def test_single_class_datasets_are_unchanged(tmp_path):
    ds = Dataset.create(tmp_path / "ds")
    image, labels = plant(0)
    ds.add(image, labels[..., 0], "a")
    ds.save()
    raw = cv2.imread(str(tmp_path / "ds" / "masks" / "a.png"), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(raw)) <= {0, 255}  # still plain 0/255
    assert Dataset.open(tmp_path / "ds").classes == ["target"]
    assert not ds.multi_class


def test_classes_overlap_and_convert_old_masks(tmp_path):
    ds = Dataset.create(tmp_path / "ds")
    image, labels = plant(1)
    ds.add(image, labels[..., 0], "a")  # made before there were two classes
    ds.save()
    ds.classes = ["target"]
    ds.add_class("moss")
    assert ds.classes == ["target", "moss"]
    item = ds.get("a")
    assert np.array_equal(ds.load_mask(item), labels[..., 0])  # 255 became bit 0
    assert not ds.load_mask(item, "moss").any()
    ds.write_mask(item, labels[..., 1], "moss")
    both = ds.load_labels(item)
    assert np.array_equal(both, labels)  # moss pixels are stem pixels too
    ds.write_mask(item, np.zeros_like(labels[..., 0]), "target")
    assert np.array_equal(ds.load_mask(item, "moss"), labels[..., 1])  # other class kept

    reopened = Dataset.open(tmp_path / "ds")
    assert reopened.classes == ["target", "moss"] and reopened.multi_class
    with pytest.raises(KeyError, match="no class"):
        reopened.load_mask(reopened.get("a"), "lichen")
    with pytest.raises(ValueError):
        reopened.add_class("moss")
    image2, labels2 = plant(2)
    new = reopened.add(image2, labels2, "b")  # all classes at once
    assert np.array_equal(reopened.load_labels(new), labels2)


def test_coco_round_trip_keeps_classes(tmp_path):
    from fungus_cv.learn.coco import export_coco, import_coco

    ds = Dataset.create(tmp_path / "ds")
    ds.add_class("moss")
    ds.classes[0] = "stem"
    ds.save()
    for i in range(2):
        image, labels = plant(10 + i)
        ds.add(image, labels, f"p{i}")
    ds.save()
    path = export_coco(ds, tmp_path / "coco", rle=True)
    coco = json.loads(path.read_text())
    assert [c["name"] for c in coco["categories"]] == ["stem", "moss"]
    fresh = Dataset.create(tmp_path / "fresh")
    import_coco(fresh, path)
    assert fresh.classes == ["stem", "moss"]
    for i in range(2):
        assert np.array_equal(fresh.load_labels(fresh.get(f"p{i}")),
                              ds.load_labels(ds.get(f"p{i}")))


def test_one_model_learns_both_classes(tmp_path):
    pytest.importorskip("torchvision")
    from fungus_cv.learn.infer import load_trained
    from fungus_cv.learn.train import TrainConfig, evaluate_model, train
    from fungus_cv.segment.trained import TrainedModelSegmenter

    ds = Dataset.create(tmp_path / "ds")
    ds.classes = ["stem", "moss"]
    ds.save()
    for i in range(10):
        image, labels = plant(100 + i)
        ds.add(image, labels, f"p{i}", reviewed=True, timestamp_utc=f"2026-09-20T00:{i:02d}")
    ds.save()
    cfg = TrainConfig(encoder="resnet18", pretrained=False, patch_px=96, batch_size=4,
                      steps=160, eval_every=40, learning_rate=3e-3, device="cpu")
    result = train(ds, tmp_path / "model", cfg)
    card = json.loads((tmp_path / "model" / "model.json").read_text())
    assert card["classes"] == ["stem", "moss"] and set(card["thresholds"]) == {"stem", "moss"}
    per_class = card["validation"]["per_class"]
    assert per_class["stem"]["iou_mean"] > 0.8, per_class
    assert per_class["moss"]["iou_mean"] > 0.5, per_class
    assert result.n_val >= 1

    test = Dataset.create(tmp_path / "test")
    test.classes = ["stem", "moss"]
    for i in range(3):
        image, labels = plant(500 + i)
        test.add(image, labels, f"t{i}", reviewed=True)
    test.save()
    rows, summary = evaluate_model(tmp_path / "model", test, device="cpu")
    assert {r["class"] for r in rows} == {"stem", "moss"} and len(rows) == 6
    assert summary["per_class"]["stem"]["not_in_training_set"]["iou_mean"] > 0.8

    image, labels = plant(900)
    moss = TrainedModelSegmenter(tmp_path / "model", device="cpu", class_name="moss")
    stem = TrainedModelSegmenter(tmp_path / "model", device="cpu", class_name="stem")
    moss_mask, stem_mask = moss.segment(image), stem.segment(image)
    assert moss_mask.sum() < stem_mask.sum()  # moss is only part of the stem
    assert moss.describe()["class"] == "moss"
    with pytest.raises(ValueError, match="no class"):
        TrainedModelSegmenter(tmp_path / "model", device="cpu", class_name="lichen").segment(
            image)
    assert load_trained(tmp_path / "model", "cpu").classes == ["stem", "moss"]


def test_export_a_run_with_its_reference_masks(experiment, tmp_path):
    from fungus_cv.analyze.compare import list_runs
    from fungus_cv.analyze.pipeline import analyze
    from fungus_cv.learn.export import export_from_run
    from fungus_cv.storage import Experiment

    from .test_pipeline import build_experiment

    build_experiment(experiment, minutes=range(1, 5), bump_at=-1)
    exp = Experiment(experiment.root)
    analyze(exp)
    run = list_runs(exp)[0]
    ref_dir = run.masks_dir / "reference"
    ref_dir.mkdir()
    for path in run.masks_dir.glob("*.png"):  # stand-in "stem": the towel strip
        stem = np.zeros_like(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        stem[150:900, 500:700] = 255
        cv2.imwrite(str(ref_dir / path.name), stem)

    ds = Dataset.create(tmp_path / "ds")
    added = export_from_run(exp, ds, run.run_id, count=3, target_class="dye",
                            reference_class="towel")
    assert ds.classes == ["dye", "towel"] and len(added) == 3
    labels = ds.load_labels(added[-1])
    dye, towel = labels[..., 0], labels[..., 1]
    assert dye.any() and towel.sum() > dye.sum()
    assert not (dye & ~towel).any()  # the target is part of the reference object

    plain = Dataset.create(tmp_path / "plain")
    for p in ref_dir.glob("*.png"):
        p.unlink()
    ref_dir.rmdir()
    with pytest.raises(ValueError, match="no reference masks"):
        export_from_run(exp, plain, run.run_id, count=2, reference_class="towel")


def test_dataset_class_cli(tmp_path):
    from typer.testing import CliRunner

    from fungus_cv.cli import app

    ds = Dataset.create(tmp_path / "ds")
    image, labels = plant(3)
    ds.add(image, labels[..., 0], "a")
    ds.save()
    runner = CliRunner()
    out = runner.invoke(app, ["dataset", "add-class", str(tmp_path / "ds"), "moss",
                              "--rename-first", "stem"])
    assert out.exit_code == 0 and "stem, moss" in out.output
    info = runner.invoke(app, ["dataset", "info", str(tmp_path / "ds")])
    assert "classes: stem, moss" in info.output
    again = runner.invoke(app, ["dataset", "add-class", str(tmp_path / "ds"), "moss"])
    assert again.exit_code == 1
    reopened = Dataset.open(tmp_path / "ds")
    assert np.array_equal(reopened.load_mask(reopened.get("a"), "stem"), labels[..., 0])
