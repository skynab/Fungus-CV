"""Active learning (suggest / rank) and SAM-assisted labeling."""

from datetime import timedelta

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from fungus_cv.analyze.compare import list_runs
from fungus_cv.analyze.pipeline import analyze
from fungus_cv.cli import app
from fungus_cv.learn.active import (
    by_priority,
    combine,
    iou,
    model_scores,
    pick_diverse,
    rank_items,
    suggest_frames,
)
from fungus_cv.learn.dataset import Dataset
from fungus_cv.measure.geometry import Annotations
from fungus_cv.storage import Experiment, FrameRecord, encode_image, iso_utc
from fungus_cv.ui import interactive
from fungus_cv.ui.interactive import combine_masks, edit_labels

from . import synthetic as syn
from .test_pipeline import T0
from .test_seg_uncertainty import soft_scene

SOFT = {3, 9}  # frames whose dye front fades out: a model should be unsure there


def blueness_prob(image: np.ndarray) -> np.ndarray:
    """Stand-in for a trained model: confident on clean dye or towel, unsure in between."""
    b, r = image[..., 0].astype(np.float32), image[..., 2].astype(np.float32)
    return 1 / (1 + np.exp(-(b - r - 80) / 8))


def build(exp: Experiment, n: int = 12) -> None:
    for i in range(n):
        ts = T0 + timedelta(minutes=i)
        img = soft_scene(40 + 40 * i, 60 if i in SOFT else 0)
        path = exp.save_bytes(encode_image(img, "png"), ts, "cam0", "png")
        exp.append_frame(FrameRecord(timestamp_utc=iso_utc(ts), camera="cam0", status="ok",
                                     file=exp.relative(path), width=syn.W, height=syn.H))
    Annotations(base=syn.BASE_POINT, tip=syn.TIP_POINT, roi=syn.ROI,
                image_size=(syn.W, syn.H)).save(exp.root / "annotations.json")
    text = exp.config_path.read_text().replace("size_mm: null", f"size_mm: {syn.MARKER_MM}")
    exp.config_path.write_text(text.replace("save_overlays: true", "save_overlays: false"))


def frame_number(item_id: str, exp: Experiment) -> int:
    files = [r["file"] for r in exp.read_frames()]
    return next(i for i, f in enumerate(files) if f.split("/")[-1].split(".png")[0] in item_id)


# --- scores --------------------------------------------------------------------------------


def test_scores_and_helpers():
    a = np.zeros((10, 10), bool)
    a[:5] = True
    assert iou(a, a) == 1 and iou(np.zeros_like(a), np.zeros_like(a)) == 1
    assert iou(a, ~a) == 0

    confident = [np.where(a, 0.99, 0.01)] * 4
    assert model_scores(confident, 0.5) == {"tta_disagreement": 0.0, "ambiguous_fraction": 0.0}
    unsure = np.full((10, 10), 0.01)
    unsure[:5] = 0.55
    flipped = unsure.copy()
    flipped[:5] = 0.45
    s = model_scores([unsure, flipped, unsure, flipped], 0.5)
    assert s["tta_disagreement"] == pytest.approx(2 / 3) and s["ambiguous_fraction"] == 1.0
    assert combine({"a": 0.2, "b": None, "c": 0.6}) == pytest.approx(0.4)

    assert pick_diverse([0.9, 0.95, 0.1, 0.2, 0.8, 0.3], 2, 3) == [1, 4]
    assert pick_diverse([0.9, 0.95, 0.1], 3, 5) == [0, 1, 2]  # gap relaxed to fill


def test_combine_masks():
    mask = np.array([True, True, False, False])
    proposal = np.array([False, True, True, False])
    assert combine_masks(mask, proposal, "replace").tolist() == proposal.tolist()
    assert combine_masks(mask, proposal, "add").tolist() == [True, True, True, False]
    assert combine_masks(mask, proposal, "subtract").tolist() == [True, False, False, False]
    with pytest.raises(ValueError):
        combine_masks(mask, proposal, "xor")


# --- suggesting frames ---------------------------------------------------------------------


def test_suggest_picks_the_uncertain_frames(experiment, tmp_path):
    build(experiment)
    exp = Experiment(experiment.root)
    ds = Dataset.create(tmp_path / "ds")
    result = suggest_frames(exp, ds, count=2, prob_fn=blueness_prob, threshold=0.5)
    picked = sorted(frame_number(i.id, exp) for i in result.added)
    assert picked == sorted(SOFT)
    for item in result.added:
        assert item.priority > 0 and not item.reviewed
        assert set(item.scores) == {"tta_disagreement", "ambiguous_fraction"}
        assert "mask from model" in item.source
        image = ds.load_image(item)
        assert np.array_equal(ds.load_mask(item), blueness_prob(image) >= 0.5)
    sharp = [r["score"] for r in result.candidates if r["index"] not in SOFT]
    assert max(sharp) < min(i.priority for i in result.added)
    assert result.csv_path.exists() and len(result.csv_path.read_text().splitlines()) == 13

    again = suggest_frames(exp, Dataset.open(ds.root), count=2, prob_fn=blueness_prob,
                           threshold=0.5)
    assert len(again.candidates) == 10  # frames already in the dataset aren't offered again


def test_suggest_from_two_runs_without_a_model(experiment, tmp_path):
    """First round: frames where a tighter and a looser colour threshold disagree most."""
    build(experiment)
    analyze(Experiment(experiment.root))
    text = experiment.config_path.read_text().replace("lower: [95, 60, 30]",
                                                      "lower: [95, 150, 30]", 1)
    experiment.config_path.write_text(text)
    analyze(Experiment(experiment.root))
    exp = Experiment(experiment.root)
    loose, tight = (r.run_id for r in list_runs(exp))
    ds = Dataset.create(tmp_path / "ds")
    result = suggest_frames(exp, ds, count=2, run=loose, against=tight)
    assert sorted(frame_number(i.id, exp) for i in result.added) == sorted(SOFT)
    item = result.added[0]
    assert set(item.scores) == {"run_disagreement"} and f"run {loose}" in item.source
    run_mask = cv2.imread(str(exp.root / "results" / "masks" / loose /
                              f"{item.id.split('__')[1]}.png"), cv2.IMREAD_GRAYSCALE) > 127
    x0, y0, x1, y1 = item.crop
    assert np.array_equal(ds.load_mask(item), run_mask[y0:y1, x0:x1])


def test_suggest_needs_a_model_or_two_runs(experiment, tmp_path):
    build(experiment, n=3)
    with pytest.raises(ValueError, match="--model"):
        suggest_frames(Experiment(experiment.root), Dataset.create(tmp_path / "ds"), run="x")


# --- ranking items -------------------------------------------------------------------------


def test_rank_items_orders_by_usefulness(tmp_path):
    ds = Dataset.create(tmp_path / "ds")
    crop = (slice(100, 900), slice(450, 750))
    sharp, soft = syn.scene(300)[crop], soft_scene(300, 60)[crop]
    ds.add(sharp, blueness_prob(sharp) >= 0.5, "sharp")
    ds.add(soft, blueness_prob(soft) >= 0.5, "soft")
    ds.add(sharp, np.zeros(sharp.shape[:2], bool), "sharp_wrong_label")  # e.g. a bad pre-label
    reviewed = ds.add(soft, blueness_prob(soft) >= 0.5, "soft_reviewed", reviewed=True)
    ds.save()

    ranked = rank_items(ds, prob_fn=blueness_prob, threshold=0.5)
    assert [i.id for i in ranked] == ["sharp_wrong_label", "soft", "sharp"]
    assert ranked[0].scores["disagreement"] == 1.0 and ranked[-1].priority < 0.05
    assert reviewed.priority is None
    reopened = Dataset.open(ds.root)
    assert reopened.get("soft").priority == ranked[1].priority
    assert [i.id for i in by_priority(reopened.items)] == \
        ["sharp_wrong_label", "soft", "sharp", "soft_reviewed"]


# --- the label editor with SAM clicks -----------------------------------------------------


class ScriptedWindow:
    """Replaces OpenCV's window functions: mouse events and keys come from a script."""

    def __init__(self, monkeypatch, script):
        self.script = list(script)
        self.callback = None
        self.shown = []
        monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "destroyWindow", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "imshow", lambda name, img: self.shown.append(img))
        monkeypatch.setattr(cv2, "setMouseCallback",
                            lambda name, cb, *a: setattr(self, "callback", cb))
        monkeypatch.setattr(cv2, "waitKey", self.wait_key)
        self.lines = []
        real_show = interactive._View.show

        def show(view, canvas, lines):
            self.lines.append(lines)
            real_show(view, canvas, lines)

        monkeypatch.setattr(interactive._View, "show", show)

    def wait_key(self, delay=0):
        if delay == 1:  # "running SAM..." redraw
            return -1
        if not self.script:
            return 27
        action = self.script.pop(0)
        if isinstance(action, tuple):
            event, x, y = action
            self.callback(event, x, y, 0, None)
            return 255
        return action if isinstance(action, int) else ord(action)


def square_sam(calls):
    def sam(image, prompt):
        calls.append(prompt)
        mask = np.zeros(image.shape[:2], bool)
        x, y = (int(v) for v in prompt.points[0])
        mask[y - 10:y + 10, x - 10:x + 10] = True
        if 0 in prompt.labels:  # a "not target" click trims the proposal
            mask[y - 10:y, :] = False
        return mask
    return sam


def test_label_editor_sam_clicks_replace_add_subtract(monkeypatch, tmp_path):
    ds = Dataset.create(tmp_path / "ds")
    image = np.full((120, 160, 3), 200, np.uint8)
    start = np.zeros((120, 160), bool)
    start[:, :20] = True
    ds.add(image, start, "low", priority=0.1)
    ds.add(image, start, "high", priority=0.9)
    ds.save()
    calls = []
    down, rdown = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN
    window = ScriptedWindow(monkeypatch, [
        "m", (down, 80, 60), 13,                 # replace with a 20x20 square at (80, 60)
        (down, 40, 30), (rdown, 45, 35), "+",    # add a half square (trimmed by the "not" click)
        (down, 80, 60), "-",                     # subtract the first square again
        "s", 27,
    ])
    counts = edit_labels(ds, by_priority=True, sam=square_sam(calls))
    assert counts["saved"] == 1
    assert "high" in window.lines[0][0] and "priority 0.90" in window.lines[0][0]
    assert any("SAM:" in line for lines in window.lines for line in lines)
    assert [p.labels for p in calls] == [[1], [1], [1, 0], [1]]

    saved = Dataset.open(ds.root)
    item = saved.get("high")
    expected = np.zeros((120, 160), bool)
    expected[30:40, 30:50] = True  # only the kept half of the added square remains
    assert item.reviewed and np.array_equal(saved.load_mask(item), expected)
    assert not saved.get("low").reviewed


def test_label_editor_reports_sam_errors(monkeypatch, tmp_path):
    ds = Dataset.create(tmp_path / "ds")
    ds.add(np.full((60, 80, 3), 200, np.uint8), np.zeros((60, 80), bool), "a")
    ds.save()

    def broken(image, prompt):
        raise RuntimeError("out of memory")

    window = ScriptedWindow(monkeypatch, ["m", (cv2.EVENT_LBUTTONDOWN, 10, 10), 13, 27])
    counts = edit_labels(ds, sam=broken)
    assert counts["saved"] == 0  # Enter without a proposal changes nothing
    assert any("SAM error: out of memory" in line for lines in window.lines for line in lines)


# --- CLI -----------------------------------------------------------------------------------


def test_cli_suggest_and_rank_errors(experiment, tmp_path):
    build(experiment, n=4)
    analyze(Experiment(experiment.root))
    runner = CliRunner()
    run = list_runs(Experiment(experiment.root))[0].run_id
    result = runner.invoke(app, ["dataset", "suggest", str(experiment.root), str(tmp_path / "ds"),
                                 "--run", run, "--against", run, "--count", "2"])
    assert result.exit_code == 0, result.output
    assert "added 2" in result.output and "--by-priority" in result.output
    bad = runner.invoke(app, ["dataset", "suggest", str(experiment.root), str(tmp_path / "ds")])
    assert bad.exit_code == 1 and "--model" in bad.output
    missing = runner.invoke(app, ["dataset", "rank", str(tmp_path / "ds"), "--model",
                                  str(tmp_path / "nope")])
    assert missing.exit_code == 1 and "not a trained model folder" in missing.output
