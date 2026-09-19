"""Datasets, splits, metrics, export, tiled inference and a small end-to-end training run."""

import json

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from fungus_cv.analyze.compare import list_runs
from fungus_cv.analyze.pipeline import analyze
from fungus_cv.cli import app
from fungus_cv.learn.dataset import Dataset, Item, split_items
from fungus_cv.learn.export import add_pairs, export_from_run, pick_evenly
from fungus_cv.learn.metrics import boundary_f1, mask_metrics
from fungus_cv.storage import Experiment

from . import synthetic as syn
from .test_pipeline import build_experiment, read_measurements

# --- metrics -----------------------------------------------------------------------------


def square(size=100, x0=20, y0=20, side=40):
    m = np.zeros((size, size), bool)
    m[y0:y0 + side, x0:x0 + side] = True
    return m


def test_mask_metrics_basic():
    truth = square()
    m = mask_metrics(truth, truth)
    assert m["iou"] == m["dice"] == m["precision"] == m["recall"] == m["boundary_f1_2px"] == 1
    shifted = square(x0=30)  # overlap 30x40 of union 50x40
    m = mask_metrics(shifted, truth)
    assert m["iou"] == pytest.approx(30 / 50)
    assert m["precision"] == pytest.approx(0.75)


def test_mask_metrics_empty_cases():
    empty = np.zeros((10, 10), bool)
    assert mask_metrics(empty, empty)["iou"] == 1.0
    assert mask_metrics(square(10, 2, 2, 3), empty)["precision"] == 0.0


def test_boundary_f1_tolerance():
    truth = square()
    one_px = square(x0=21)
    far = square(x0=26)
    assert boundary_f1(one_px, truth, 2.0) > 0.9
    assert boundary_f1(far, truth, 2.0) < boundary_f1(one_px, truth, 2.0)


# --- dataset & splits --------------------------------------------------------------------


def items(groups):
    out = []
    for g, n in groups.items():
        out += [Item(id=f"{g}_{i}", image="", mask="", group=g,
                     timestamp_utc=f"2026-01-01T00:{i:02d}:00Z") for i in range(n)]
    return out


def test_split_holds_out_whole_groups():
    train, val, how = split_items(items({"a": 10, "b": 3, "c": 4}), val_fraction=0.2)
    assert {i.group for i in val} == {"b"}
    assert not {i.group for i in train} & {i.group for i in val}
    assert "held-out" in how


def test_split_single_group_uses_latest_frames():
    train, val, _ = split_items(items({"a": 10}), val_fraction=0.3)
    assert [i.id for i in val] == ["a_7", "a_8", "a_9"]
    assert len(train) == 7


def test_split_explicit_groups_and_errors():
    train, val, _ = split_items(items({"a": 2, "b": 2}), val_groups=["a"])
    assert {i.group for i in val} == {"a"} and {i.group for i in train} == {"b"}
    with pytest.raises(ValueError):
        split_items(items({"a": 2}), val_groups=["zzz"])
    with pytest.raises(ValueError):
        split_items(items({"a": 2}), val_groups=["a"])  # nothing left to train on


def test_dataset_roundtrip_and_fingerprint(tmp_path):
    ds = Dataset.create(tmp_path / "ds")
    img = np.full((20, 30, 3), 100, np.uint8)
    mask = np.pad(square(20, 2, 2, 5), ((0, 0), (0, 10)))  # 20x30, 25 target pixels
    ds.add(img, mask, "exp 1/frame:3", group="g")
    ds.save()
    again = Dataset.open(tmp_path / "ds")
    assert [i.id for i in again.items] == ["exp_1_frame_3"]
    assert again.load_mask(again.items[0]).sum() == 25
    assert again.selected(reviewed_only=True) == []
    fp = again.fingerprint()
    again.write_mask(again.items[0], np.zeros((20, 30), bool))
    assert again.fingerprint() != fp
    with pytest.raises(FileExistsError):
        again.add(img, np.zeros((20, 30), bool), "exp_1_frame_3")
    with pytest.raises(ValueError):
        again.add(img, np.zeros((5, 5), bool), "other")


def test_pick_evenly():
    assert pick_evenly(10, 3) == [0, 4, 9]  # round(4.5) == 4
    assert pick_evenly(3, 10) == [0, 1, 2]
    assert pick_evenly(5, 1) == [4]


def test_add_pairs(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "msk").mkdir()
    for name in ("a", "b"):
        cv2.imwrite(str(tmp_path / "img" / f"{name}.jpg"), np.full((16, 16, 3), 90, np.uint8))
    cv2.imwrite(str(tmp_path / "msk" / "a.png"), square(16, 4, 4, 4).astype(np.uint8))
    ds = Dataset.open_or_create(tmp_path / "ds")
    added = add_pairs(ds, tmp_path / "img", tmp_path / "msk", group="phone", reviewed=True)
    assert [i.id for i in added] == ["phone__a"]  # b has no mask
    assert ds.load_mask(added[0]).sum() == 16


def test_export_from_run_crops_to_region(experiment, tmp_path):
    build_experiment(experiment, minutes=range(0, 6), bump_at=-1)
    exp = Experiment(experiment.root)
    analyze(exp)
    ds = Dataset.create(tmp_path / "ds")
    added = export_from_run(exp, ds, list_runs(exp)[0].run_id, count=3)
    assert len(added) == 3
    item = added[0]
    x0, y0, x1, y1 = item.crop
    image, mask = ds.load_image(item), ds.load_mask(item)
    assert image.shape[:2] == mask.shape == (y1 - y0, x1 - x0)
    assert not item.reviewed and item.group == "test"
    # Re-exporting skips existing items.
    assert export_from_run(exp, Dataset.open(ds.root), list_runs(exp)[0].run_id, count=3) == []

    out = CliRunner().invoke(app, ["dataset", "info", str(ds.root)])
    assert out.exit_code == 0 and "3 item(s), 0 reviewed" in out.output


# --- torch-dependent ---------------------------------------------------------------------


def test_tiled_prediction_matches_whole_image():
    pytest.importorskip("torch")
    from fungus_cv.learn.infer import predict_probabilities

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (300, 450, 3), dtype=np.uint8)

    def pointwise(tile):  # depends only on each pixel, so tiling must not change anything
        return tile[..., 0].astype(np.float32) / 32 - 4

    whole = predict_probabilities(None, image, "cpu", tile_px=1024, predict_fn=pointwise)
    tiled = predict_probabilities(None, image, "cpu", tile_px=128, overlap_px=32,
                                  predict_fn=pointwise)
    assert tiled.shape == (300, 450)
    np.testing.assert_allclose(tiled, whole, atol=1e-5)


def test_train_then_analyze_with_model(experiment, tmp_path):
    """Small CPU run: train on synthetic dye, then measure with `method: model`."""
    pytest.importorskip("torchvision")
    from fungus_cv.learn.train import TrainConfig, evaluate_model, train

    heights = build_experiment(experiment, minutes=range(0, 9), bump_at=-1)
    exp = Experiment(experiment.root)
    analyze(exp)  # color run provides the starting labels
    ds = Dataset.create(tmp_path / "ds")
    export_from_run(exp, ds, list_runs(exp)[0].run_id, count=9)
    for item in ds.items:  # stand-in for a person reviewing them
        item.reviewed = True
    ds.save()

    cfg = TrainConfig(encoder="resnet18", pretrained=False, patch_px=128, batch_size=4,
                      steps=80, eval_every=40, learning_rate=2e-3, device="cpu")
    result = train(ds, tmp_path / "model", cfg)
    assert result.n_val >= 1 and result.val_metrics["iou_mean"] > 0.9
    card = json.loads((tmp_path / "model" / "model.json").read_text())
    assert card["dataset"]["fingerprint"] == ds.fingerprint(ds.selected())
    assert card["input_profile"]["n"] == result.n_train  # what the model has seen
    assert not set(card["dataset"]["train_items"]) & set(card["dataset"]["val_items"])

    rows, summary = evaluate_model(tmp_path / "model", ds, device="cpu")
    assert summary["not_in_training_set"]["n"] == result.n_val

    from fungus_cv.learn.active import rank_items

    ranked = rank_items(ds, tmp_path / "model", include_reviewed=True, device="cpu")
    assert len(ranked) == 9 and all(0 <= i.priority <= 1 for i in ranked)

    text = exp.config_path.read_text().replace("method: color", "method: model", 1)
    text = text.replace('path: ""', f"path: '{tmp_path / 'model'}'").replace(
        "device: auto", "device: cpu")
    exp.config_path.write_text(text)
    summary = analyze(Experiment(exp.root))
    assert summary.processed == len(heights) and summary.failed == 0
    measured = sorted(read_measurements(exp), key=lambda r: r["timestamp_utc"])
    for row, h in zip(measured, heights):
        assert float(row["extent_mm"]) == pytest.approx(h * syn.MM_PER_PX, abs=1.5)
        assert row["model_input_distance"] != ""  # same scene: known, not flagged
        assert "unfamiliar_input" not in row["flags"]


def _tiny_dataset(experiment, tmp_path):
    build_experiment(experiment, minutes=range(0, 7), bump_at=-1)
    exp = Experiment(experiment.root)
    analyze(exp)
    ds = Dataset.create(tmp_path / "ds")
    export_from_run(exp, ds, list_runs(exp)[0].run_id, count=7)
    for item in ds.items:
        item.reviewed = True
    ds.save()
    return ds


def test_resumed_training_matches_an_uninterrupted_run(experiment, tmp_path):
    pytest.importorskip("torchvision")
    import torch

    from fungus_cv.learn.train import CHECKPOINT_NAME, TrainConfig, TrainingCancelled, train

    ds = _tiny_dataset(experiment, tmp_path)
    cfg = TrainConfig(encoder="resnet18", pretrained=False, patch_px=64, batch_size=2,
                      steps=24, eval_every=8, learning_rate=2e-3, device="cpu")
    whole = train(ds, tmp_path / "whole", cfg)

    steps = {"n": 0}

    def stop_after_16():
        steps["n"] += 1
        return steps["n"] > 16

    with pytest.raises(TrainingCancelled, match="resume it"):
        train(ds, tmp_path / "parts", cfg, should_stop=stop_after_16)
    assert (tmp_path / "parts" / CHECKPOINT_NAME).exists()
    with pytest.raises(FileExistsError, match="interrupted run"):
        train(ds, tmp_path / "parts", cfg)
    changed = TrainConfig(**{**cfg.__dict__, "learning_rate": 1e-3})
    with pytest.raises(ValueError, match="learning_rate"):
        train(ds, tmp_path / "parts", changed, resume=True)

    resumed = train(ds, tmp_path / "parts", cfg, resume=True)
    assert not (tmp_path / "parts" / CHECKPOINT_NAME).exists()
    a = torch.load(tmp_path / "whole" / "model.pt", weights_only=True)
    b = torch.load(tmp_path / "parts" / "model.pt", weights_only=True)
    for key in a:
        assert torch.allclose(a[key].float(), b[key].float(), atol=1e-5), key
    assert resumed.threshold == whole.threshold
    card = json.loads((tmp_path / "parts" / "model.json").read_text())
    assert card["training"]["resumed"] is True
    assert [h["step"] for h in card["training"]["history"]] == [8, 16, 24]
    with pytest.raises(FileNotFoundError, match="no interrupted run"):
        train(ds, tmp_path / "nothing", cfg, resume=True)


def test_early_stopping(experiment, tmp_path):
    pytest.importorskip("torchvision")
    from fungus_cv.learn.train import TrainConfig, train

    ds = _tiny_dataset(experiment, tmp_path)
    cfg = TrainConfig(encoder="resnet18", pretrained=False, patch_px=64, batch_size=2,
                      steps=400, eval_every=5, learning_rate=0.0, device="cpu", patience=2)
    result = train(ds, tmp_path / "m", cfg)  # a zero learning rate can never improve
    assert result.stopped_early_at == 15  # first eval sets the best, then 2 without progress
    card = json.loads((tmp_path / "m" / "model.json").read_text())
    assert card["training"]["stopped_early_at_step"] == 15
