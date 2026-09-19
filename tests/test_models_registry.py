"""Model registry and input profiles: noticing when a model is used on something new."""

import json

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from fungus_cv.analyze import pipeline
from fungus_cv.analyze.pipeline import analyze
from fungus_cv.cli import app
from fungus_cv.config import ColorTargetConfig
from fungus_cv.learn import profile as prof
from fungus_cv.learn.dataset import Dataset
from fungus_cv.learn.registry import add_profile, find_models, read_model
from fungus_cv.segment.color import ColorThresholdSegmenter
from fungus_cv.storage import Experiment

from . import synthetic as syn
from .test_pipeline import build_experiment, read_measurements

CROP = (slice(100, 900), slice(450, 750))


def dye_images(n=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        img = syn.scene(40 * i)[CROP].astype(np.float32)  # from no dye to a lot
        img *= rng.uniform(0.95, 1.05)  # the small variation a real run has
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def test_profile_accepts_similar_images_and_flags_different_ones():
    profile = prof.build_profile(dye_images())
    assert profile["n"] == 8 and len(profile["mean"]) == len(prof.FEATURES)
    assert profile["limit"] >= profile["max_training_distance"]

    similar = dye_images(n=3, seed=7)
    assert not any(prof.is_unfamiliar(profile, prof.image_distance(profile, im))
                   for im in similar)

    grass = np.zeros_like(similar[0])
    grass[:] = (40, 150, 60)  # green, as from another use case
    dark = (similar[0] * 0.25).astype(np.uint8)  # the lights went off
    for other in (grass, dark):
        value = prof.image_distance(profile, other)
        assert prof.is_unfamiliar(profile, value), value
    with pytest.raises(ValueError):
        prof.build_profile(dye_images(n=1))


class ProfiledColor(ColorThresholdSegmenter):
    """A colour segmenter standing in for a trained model that has an input profile."""

    def input_distance(self, image):
        value = prof.image_distance(self.profile, image[CROP])
        return value, prof.is_unfamiliar(self.profile, value)


def test_analysis_records_distance_and_flags_unfamiliar_frames(experiment, monkeypatch):
    build_experiment(experiment, minutes=range(0, 6), bump_at=-1)
    exp = Experiment(experiment.root)
    rows = exp.read_frames()
    dark = cv2.imread(str(exp.root / rows[4]["file"]))
    cv2.imwrite(str(exp.root / rows[4]["file"]), (dark * 0.3).astype(np.uint8))

    seg = ProfiledColor.from_config(ColorTargetConfig())
    seg.__class__ = ProfiledColor
    seg.profile = prof.build_profile(dye_images())
    monkeypatch.setattr(pipeline, "build_segmenter", lambda *a, **k: seg)
    analyze(exp)
    measured = sorted(read_measurements(exp), key=lambda r: r["timestamp_utc"])
    assert all(r["model_input_distance"] != "" for r in measured)
    flagged = [i for i, r in enumerate(measured) if "unfamiliar_input" in r["flags"]]
    assert flagged == [4]


def fake_model(folder, created, dataset_name, with_profile=False, evaluations=None):
    folder.mkdir(parents=True)
    card = {"created_utc": created, "architecture": {"encoder": "resnet18"},
            "threshold": 0.5,
            "dataset": {"name": dataset_name, "fingerprint": "abcdef1234567890",
                        "train_items": ["a", "b", "c"], "val_items": ["d"], "split": "x"},
            "validation": {"summary": {"iou_mean": 0.9}},
            "training": {"seconds": 60, "device": "cpu"}}
    if with_profile:
        card["input_profile"] = prof.build_profile(dye_images(3))
    (folder / "model.json").write_text(json.dumps(card))
    (folder / "model.pt").write_bytes(b"weights")
    for name, rows in (evaluations or {}).items():
        lines = ["id,in_training_set,iou"] + [f"{i},{t},{v}" for i, t, v in rows]
        (folder / f"evaluation_{name}.csv").write_text("\n".join(lines))


def test_registry_lists_models_newest_first(tmp_path):
    fake_model(tmp_path / "models" / "moss-v1", "2026-09-01T00:00:00Z", "moss")
    fake_model(tmp_path / "models" / "moss-v2", "2026-09-10T00:00:00Z", "moss",
               with_profile=True,
               evaluations={"moss-test": [("x", "False", 0.8), ("y", "False", 0.7),
                                          ("z", "True", 0.99)]})
    (tmp_path / "models" / "broken").mkdir()
    (tmp_path / "models" / "broken" / "model.json").write_text("{")
    found = find_models([tmp_path / "models"])
    assert [m.name for m in found] == ["moss-v2", "moss-v1"]
    v2 = found[0]
    assert v2.has_profile and not found[1].has_profile
    assert v2.evaluations["moss-test"] == {"iou": pytest.approx(0.75), "n": 2, "unseen": True}
    assert read_model(tmp_path / "models" / "moss-v1").n_train == 3


def test_add_profile_to_an_older_model(tmp_path):
    ds = Dataset.create(tmp_path / "ds")
    for i, image in enumerate(dye_images(4)):
        ds.add(image, np.zeros(image.shape[:2], bool), ["a", "b", "c", "d"][i])
    ds.save()
    fake_model(tmp_path / "m", "2026-09-01T00:00:00Z", "ds")
    profile = add_profile(tmp_path / "m", ds)
    assert profile["n"] == 3  # the three training items, not the validation one
    assert read_model(tmp_path / "m").has_profile
    other = Dataset.create(tmp_path / "other")
    with pytest.raises(ValueError, match="dataset it was trained on"):
        add_profile(tmp_path / "m", other)


def test_models_cli(tmp_path):
    fake_model(tmp_path / "models" / "moss-v1", "2026-09-01T00:00:00Z", "moss")
    runner = CliRunner()
    listed = runner.invoke(app, ["models", "list", str(tmp_path / "models")])
    assert listed.exit_code == 0 and "moss-v1" in listed.output
    assert "[no input profile]" in listed.output
    shown = runner.invoke(app, ["models", "show", str(tmp_path / "models" / "moss-v1")])
    assert shown.exit_code == 0 and "add one with `fungus models profile`" in shown.output
    missing = runner.invoke(app, ["models", "show", str(tmp_path)])
    assert missing.exit_code == 1
    empty = runner.invoke(app, ["models", "list", str(tmp_path / "nothing")])
    assert "No trained models" in empty.output
