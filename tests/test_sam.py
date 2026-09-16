"""SAM 2 integration: prompt/crop helpers, the sequence pipeline path, run comparison.

Tests that load the real model are skipped unless FUNGUS_TEST_SAM=1 (they download
weights and need PyTorch).
"""

import os
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fungus_cv.analyze import pipeline
from fungus_cv.analyze.compare import compare_runs, list_runs
from fungus_cv.analyze.pipeline import analyze
from fungus_cv.cli import app
from fungus_cv.config import ColorTargetConfig
from fungus_cv.segment.color import ColorThresholdSegmenter
from fungus_cv.segment.prompts import CropWindow, FramePrompt, Prompts
from fungus_cv.storage import Experiment

from . import synthetic as syn
from .test_pipeline import build_experiment, read_measurements

# --- prompts & crops ---------------------------------------------------------------------


def test_frame_prompt_validation():
    with pytest.raises(ValueError):
        FramePrompt("a.png")  # nothing
    with pytest.raises(ValueError):
        FramePrompt("a.png", points=[(1, 2)], labels=[0])  # only a negative point
    with pytest.raises(ValueError):
        FramePrompt("a.png", points=[(1, 2)], labels=[1, 0])
    FramePrompt("a.png", box=(0, 0, 5, 5))


def test_prompts_roundtrip_and_replace(tmp_path):
    prompts = Prompts()
    prompts.set(FramePrompt("a.png", points=[(1.5, 2.5), (3, 4)], labels=[1, 0]))
    prompts.set(FramePrompt("b.png", box=(1, 2, 30, 40)))
    prompts.set(FramePrompt("a.png", points=[(9, 9)], labels=[1]))  # replaces the first
    prompts.save(tmp_path / "p.json")
    loaded = Prompts.load(tmp_path / "p.json")
    assert loaded == prompts
    assert [p.frame_file for p in loaded.frames] == ["b.png", "a.png"]
    assert loaded.for_file("a.png").points == [(9, 9)]


def test_crop_window_roundtrip():
    win = CropWindow.around([(100.2, 50.0), (200.0, 50.0), (150.0, 300.7)], margin=10,
                            width=640, height=480)
    assert (win.x0, win.y0, win.x1, win.y1) == (90, 40, 211, 312)
    image = np.arange(480 * 640).reshape(480, 640)
    assert win.crop(image)[0, 0] == image[40, 90]
    assert win.to_crop([(100, 50)]) == [(10, 10)]
    assert win.box_to_crop((90, 40, 100, 50)) == (0, 0, 10, 10)
    full = win.paste(np.ones((win.height, win.width), bool), 480, 640)
    assert full.sum() == win.width * win.height and full[40, 90] and not full[39, 90]


def test_crop_window_clamps_to_image():
    win = CropWindow.around([(0, 0), (639, 479)], margin=50, width=640, height=480)
    assert (win.x0, win.y0, win.x1, win.y1) == (0, 0, 640, 480)


# --- sequence pipeline with a stand-in segmenter -------------------------------------------


class FakeSequenceSegmenter:
    """Behaves like SAM 2 for the pipeline: needs frames in order from a keyframe."""

    name = "fake-seq"

    def __init__(self, keyframe: int = 2):
        self.keyframe = keyframe
        self.inner = ColorThresholdSegmenter.from_config(ColorTargetConfig())
        self.loaded: list[int] = []

    def describe(self):
        return {"method": self.name, "keyframe": self.keyframe}

    def segment_sequence(self, frame_files, load, needed):
        order = list(range(self.keyframe, max(needed) + 1)) + \
            list(range(self.keyframe - 1, min(needed) - 1, -1))
        for i in order:
            self.loaded.append(i)
            mask = self.inner.segment(load(i))
            if i in needed:
                yield i, mask


def use_fake(monkeypatch, fake):
    monkeypatch.setattr(pipeline, "build_segmenter", lambda *a, **k: fake)


def test_sequence_path_measures_all_frames(experiment, monkeypatch):
    heights = build_experiment(experiment, minutes=range(0, 6), bump_at=3)
    fake = FakeSequenceSegmenter(keyframe=2)
    use_fake(monkeypatch, fake)
    exp = Experiment(experiment.root)
    summary = analyze(exp)
    assert summary.processed == 6 and summary.failed == 0
    rows = sorted(read_measurements(exp), key=lambda r: r["timestamp_utc"])
    for row, h in zip(rows, heights):
        assert float(row["extent_mm"]) == pytest.approx(h * syn.MM_PER_PX, abs=0.4)
    assert sorted(fake.loaded) == list(range(6))


def test_sequence_path_is_incremental(experiment, monkeypatch):
    build_experiment(experiment, minutes=range(0, 4))
    use_fake(monkeypatch, FakeSequenceSegmenter(keyframe=1))
    first = analyze(Experiment(experiment.root))
    assert first.processed == 4
    again = analyze(Experiment(experiment.root))
    assert again.processed == 0 and again.skipped_existing == 4


def test_compare_runs_and_cli(experiment, monkeypatch):
    build_experiment(experiment, minutes=range(0, 5), bump_at=-1)
    exp = Experiment(experiment.root)
    analyze(exp)  # color run
    use_fake(monkeypatch, FakeSequenceSegmenter(keyframe=2))
    analyze(Experiment(exp.root))  # "sequence" run with identical masks

    runs = list_runs(exp)
    assert len(runs) == 2 and all(r.n_masks == 5 for r in runs)
    result = compare_runs(exp, runs[0].run_id[:6], runs[1].run_id)
    assert result.n_frames == 5
    assert result.iou_mean == pytest.approx(1.0)
    assert result.extent_diff_mean == pytest.approx(0.0)
    assert result.csv_path.exists()

    out = CliRunner().invoke(app, ["compare", str(exp.root)])
    assert out.exit_code == 0, out.output
    assert "IoU  mean 1.0000" in out.output


def test_compare_against_mask_folder(experiment, tmp_path):
    build_experiment(experiment, minutes=range(0, 3), bump_at=-1)
    exp = Experiment(experiment.root)
    analyze(exp)
    run = list_runs(exp)[0]
    labels = tmp_path / "labels"
    labels.mkdir()
    import cv2

    for mask_path in run.masks_dir.glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask[:, : syn.TOWEL_X[0] + 100] = 0  # label only covers half the towel width
        cv2.imwrite(str(labels / mask_path.name), mask)
    result = compare_runs(exp, run.run_id, str(labels))
    assert 0.4 < result.iou_mean < 0.6


def test_missing_prompts_is_a_clear_error(experiment):
    build_experiment(experiment, minutes=range(0, 2))
    text = experiment.config_path.read_text().replace("method: color", "method: sam2", 1)
    experiment.config_path.write_text(text)
    result = CliRunner().invoke(app, ["analyze", str(experiment.root)])
    assert result.exit_code == 1
    assert "fungus prompt" in result.output


# --- real model ----------------------------------------------------------------------------

needs_sam = pytest.mark.skipif(
    os.environ.get("FUNGUS_TEST_SAM") != "1",
    reason="set FUNGUS_TEST_SAM=1 to run SAM 2 model tests (downloads weights)",
)


@needs_sam
def test_sam2_matches_color_threshold_on_synthetic_dye(experiment):
    """M3 acceptance check: SAM 2 masks agree with the color threshold (IoU > 0.9)."""
    heights = build_experiment(experiment, minutes=range(0, 9), bump_at=4)
    exp = Experiment(experiment.root)
    analyze(exp)

    files = [r["file"] for r in exp.read_frames()]
    Prompts([FramePrompt(files[6], points=[(600.0, 800.0)], labels=[1])]).save(
        exp.root / "prompts.json")
    text = exp.config_path.read_text().replace("method: color", "method: sam2", 1)
    text = text.replace("sam2.1-hiera-small", "sam2.1-hiera-tiny")
    exp.config_path.write_text(text)

    summary = analyze(Experiment(exp.root))
    assert summary.processed == len(heights) and summary.failed == 0
    rows = sorted(read_measurements(exp), key=lambda r: r["timestamp_utc"])
    for row, h in zip(rows, heights):
        assert float(row["extent_mm"]) == pytest.approx(h * syn.MM_PER_PX, abs=0.6)

    runs = list_runs(exp)
    result = compare_runs(exp, runs[0].run_id, runs[1].run_id)
    assert result.iou_min > 0.9
    assert Path(result.csv_path).exists()
