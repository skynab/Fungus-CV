"""Incremental SAM 2 tracking: resume from saved state instead of re-tracking every frame.

The logic tests use a stand-in stream whose masks depend on its memory, so a resume that
restored the wrong state would give different masks. The real-model test is at the end.
"""

import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from fungus_cv.segment import sam2  # noqa: E402
from fungus_cv.segment.prompts import FramePrompt, Prompts  # noqa: E402
from fungus_cv.segment.sam2 import STATE_ATTRS, Sam2VideoSegmenter, _Stream  # noqa: E402


class FakeStream:
    steps: list[int] = []

    def __init__(self, seg, window, full_hw):
        self.device = "cpu"
        self.full_hw = full_hw
        self.session = SimpleNamespace(**{name: {} for name in STATE_ATTRS})
        self.session.output_dict_per_obj = {0: {"cond_frame_outputs": {},
                                                "non_cond_frame_outputs": {}}}

    def add_prompt(self, step, prompt):
        self.session.point_inputs_per_obj[step] = torch.tensor(float(len(prompt.points)))

    def step(self, step, image):
        FakeStream.steps.append(step)
        memory = self.session.output_dict_per_obj[0]["non_cond_frame_outputs"]
        prompts = sum(float(v) for v in self.session.point_inputs_per_obj.values())
        total = sum(float(o["value"]) for o in memory.values()) + prompts
        value = float(image.mean())
        memory[step] = {"value": torch.tensor(value), "high_res_masks": torch.zeros(64, 64)}
        mask = np.zeros(self.full_hw, bool)
        mask[: int(total + value) % self.full_hw[0]] = True
        return (mask,)

    snapshot = _Stream.snapshot
    restore = _Stream.restore


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(sam2, "_Stream", FakeStream)
    FakeStream.steps = []
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 50, (40, 30, 3), dtype=np.uint8) for _ in range(40)]
    files = [f"frames/f{i:02d}.png" for i in range(40)]
    return files, images


def segmenter(files, state_dir, prompt_frames=(0,)):
    prompts = Prompts([FramePrompt(files[i], [(5.0, 5.0)] * (k + 1), [1] * (k + 1))
                       for k, i in enumerate(prompt_frames)])
    return Sam2VideoSegmenter(model_name="fake", prompts=prompts, state_dir=state_dir)


def track(seg, files, images, needed):
    FakeStream.steps = []
    return dict(seg.segment_sequence(files, lambda i: images[i], set(needed)))


def test_new_frames_resume_and_match_full_tracking(fake, tmp_path):
    files, images = fake
    full = track(segmenter(files[:12], None, (0, 7)), files[:12], images, range(12))

    seg = segmenter(files[:12], tmp_path / "state", (0, 7))
    first = track(seg, files[:12], images, range(6))  # e.g. a run stopped after 6 frames
    assert FakeStream.steps == list(range(6))
    saved = torch.load(tmp_path / "state" / "forward.pt", weights_only=True)
    assert saved["last_index"] == 5 and len(saved["frames"]) == 6
    outputs = saved["state"]["output_dict_per_obj"][0]["non_cond_frame_outputs"]
    assert all("high_res_masks" not in o for o in outputs.values())

    # The rest, including the second prompted frame (7), which comes after the saved state.
    second = track(seg, files[:12], images, range(6, 12))
    assert FakeStream.steps == list(range(6, 12))  # only the new frames were tracked
    assert seg.last_resume == {"resumed_after": 5, "steps": 6}
    for i in range(12):
        assert np.array_equal({**first, **second}[i], full[i]), i


def test_starts_over_when_the_saved_state_does_not_fit(fake, tmp_path, caplog):
    files, images = fake
    seg = segmenter(files, tmp_path / "state")
    track(seg, files[:8], images, range(8))
    caplog.set_level(logging.INFO)

    track(seg, files[:10], images, [3, 9])  # an older frame needs a mask again
    assert FakeStream.steps[0] == 0 and "before the saved state" in caplog.text

    shuffled = files[:2] + ["frames/imported.png"] + files[2:9]
    track(seg, shuffled, images, [9])
    assert FakeStream.steps[0] == 0 and "frames up to the saved state changed" in caplog.text

    other = segmenter(files, tmp_path / "state", prompt_frames=(1,))
    track(other, files[:12], images, [11])
    assert FakeStream.steps[0] == 0 and "prompts changed" in caplog.text

    (tmp_path / "state" / "forward.pt").write_bytes(b"not a checkpoint")
    track(seg, files[:12], images, [11])
    assert FakeStream.steps[0] == 0 and "ignoring saved SAM 2 state" in caplog.text


def test_interrupted_run_resumes_from_the_last_checkpoint(fake, tmp_path):
    files, images = fake
    full = track(segmenter(files, None), files[:30], images, range(30))
    seg = segmenter(files, tmp_path / "state")
    FakeStream.steps = []
    gen = seg.segment_sequence(files[:30], lambda i: images[i], set(range(30)))
    got = dict(next(gen) for _ in range(27))  # e.g. cancelled in the app after 27 frames
    gen.close()
    rest = track(seg, files[:30], images, range(27, 30))
    assert FakeStream.steps == list(range(25, 30))  # from the checkpoint after 25 steps
    for i in range(30):
        assert np.array_equal({**got, **rest}[i], full[i])


def test_without_a_state_dir_nothing_is_saved(fake, tmp_path):
    files, images = fake
    seg = segmenter(files, None)
    track(seg, files[:5], images, range(5))
    track(seg, files[:6], images, [5])
    assert FakeStream.steps == list(range(6))
    assert not any(tmp_path.iterdir())


@pytest.mark.skipif(os.environ.get("FUNGUS_TEST_SAM") != "1",
                    reason="set FUNGUS_TEST_SAM=1 to run SAM 2 model tests")
def test_real_sam2_resume_is_pixel_identical(experiment):
    """Real SAM 2: incremental analysis tracks only new frames, with identical masks,
    including when a later prompted frame comes after the saved state."""
    import cv2

    from fungus_cv.analyze.pipeline import analyze

    from .test_pipeline import build_experiment, read_measurements

    build_experiment(experiment, minutes=range(0, 12), bump_at=4)
    exp = type(experiment)(experiment.root)
    files = [r["file"] for r in exp.read_frames()]
    Prompts([FramePrompt(files[1], [(600.0, 800.0)], [1]),
             FramePrompt(files[8], [(600.0, 780.0)], [1])]).save(exp.root / "prompts.json")
    text = exp.config_path.read_text().replace("method: color", "method: sam2", 1)
    text = text.replace("sam2.1-hiera-small", "sam2.1-hiera-tiny")
    exp.config_path.write_text(text.replace("segmentation: true", "segmentation: false"))

    from fungus_cv.analyze.pipeline import Analyzer
    from fungus_cv.storage import Experiment

    analyzer = Analyzer(Experiment(exp.root))
    seg, load = analyzer.segmenter, analyzer.aligned_frame
    fresh = dict(Sam2VideoSegmenter.segment_sequence(
        type(seg)(**{**{k: getattr(seg, k) for k in ("model_name", "prompts", "crop", "device",
                                                     "mask_threshold", "min_blob_area_px")}}),
        files, load, set(range(12))))

    calls = []
    real_step = _Stream.step

    def counting(self, step, image):
        calls.append(step)
        return real_step(self, step, image)

    _Stream.step = counting
    try:
        part1 = dict(seg.segment_sequence(files, load, set(range(6))))
        calls.clear()
        part2 = dict(seg.segment_sequence(files, load, set(range(6, 12))))
        assert calls == list(range(5, 11))  # steps after the saved state only
        for i in range(12):
            mask = part1[i] if i < 6 else part2[i]
            assert np.array_equal(mask, fresh[i]), f"frame {i} differs"

        # Through the pipeline: two new frames cost two tracking steps.
        summary = analyze(Experiment(exp.root))
        assert summary.processed == 12
        from datetime import timedelta

        from fungus_cv.storage import FrameRecord, encode_image, iso_utc

        from . import synthetic as syn
        from .test_pipeline import T0
        for minute in (12, 13):
            ts = T0 + timedelta(minutes=minute)
            img = syn.scene(int(round(60 * np.sqrt(minute))))
            path = exp.save_bytes(encode_image(img, "png"), ts, "cam0", "png")
            exp.append_frame(FrameRecord(timestamp_utc=iso_utc(ts), camera="cam0", status="ok",
                                         file=exp.relative(path), width=syn.W, height=syn.H))
        calls.clear()
        summary = analyze(Experiment(exp.root))
        assert summary.processed == 2 and len(calls) == 2
    finally:
        _Stream.step = real_step
    rows = read_measurements(exp)
    assert len(rows) == 14
    assert cv2.imread(str(exp.root / rows[-1]["mask_file"])) is not None
