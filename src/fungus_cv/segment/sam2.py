"""SAM 2 video segmentation (Hugging Face ``transformers``), prompted with clicks.

Frames are streamed one at a time, so memory use stays flat for long time-lapses. The
target is tracked forward in time from the earliest prompted frame; frames before it are
tracked in a second stream running backward in time.

Incremental tracking: after each run the forward stream's memory (prompts, recent frames'
mask memories and object pointers) is saved next to the run's masks. The next run resumes
from it, so a new frame costs one frame of tracking however long the time-lapse is. The
resumed masks are identical to re-tracking from the start. A saved state is only used when
the model, crop, prompts, software versions and every frame tracked so far still match;
otherwise tracking starts again from the prompted frame.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fungus_cv.segment.prompts import CropWindow, FramePrompt, Prompts
from fungus_cv.segment.torch_device import (
    choose_device,
    import_torch,
    import_transformers,
)

log = logging.getLogger(__name__)

OBJ_ID = 1
# Per-frame model state older than this many frames is no longer read by SAM 2
# (7 mask memories, 16 object pointers) and is freed during long streams.
KEEP_HISTORY = 20
# Saved tracking state: bump when its layout changes. Full-resolution masks are only used to
# produce each frame's own output, not as memory, so they are left out (~7 MB per state).
STATE_FORMAT = 1
STATE_ATTRS = ("_obj_id_to_idx", "_obj_idx_to_id", "obj_ids", "point_inputs_per_obj",
               "mask_inputs_per_obj", "output_dict_per_obj", "frames_tracked_per_obj",
               "obj_with_new_inputs", "video_height", "video_width")
STATE_DROPPED_OUTPUTS = ("high_res_masks",)
CHECKPOINT_EVERY = 25  # steps; an interrupted run then resumes close to where it stopped

_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


class PromptError(ValueError):
    pass


def load_model(name: str, device_pref: str = "auto"):
    """Load (and cache) the SAM 2 video model and processor. Downloads weights on first use."""
    torch = import_torch()
    import_transformers()
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    device, dtype = choose_device(device_pref)
    key = (name, str(device))
    if key not in _MODEL_CACHE:
        log.info("loading %s on %s (first use downloads the weights)", name, device)
        model = Sam2VideoModel.from_pretrained(name).to(device, dtype=dtype).eval()
        processor = Sam2VideoProcessor.from_pretrained(name)
        _MODEL_CACHE[key] = (model, processor, device, dtype)
    torch.set_grad_enabled(False)
    return _MODEL_CACHE[key]


@dataclass
class Sam2VideoSegmenter:
    model_name: str
    prompts: Prompts
    crop: _RoiCrop | None = None  # None = whole frame
    device: str = "auto"
    mask_threshold: float = 0.0
    min_blob_area_px: int = 0
    prompts_digest: str = ""
    variant_logit_delta: float | None = None  # for segmentation uncertainty
    state_dir: Path | None = None  # where tracking state is saved between runs (None = off)
    name: str = field(default="sam2", init=False)
    last_resume: dict = field(default_factory=dict, init=False)  # what the last run did

    @classmethod
    def from_config(cls, cfg, experiment_root: Path | None, roi,
                    prompts_file: Path | None = None) -> Sam2VideoSegmenter:
        path = Path(prompts_file) if prompts_file else Path(experiment_root or ".") / \
            cfg.prompts_file
        if not path.exists():
            raise PromptError(f"{path.name} not found; run `fungus prompt` to click on the target")
        prompts = Prompts.load(path)
        if not prompts.frames:
            raise PromptError(f"{path.name} has no prompts; run `fungus prompt`")
        crop = None
        if cfg.crop_to_roi and roi is not None:
            crop = _RoiCrop(roi, cfg.crop_margin_px)
        return cls(
            model_name=cfg.model,
            prompts=prompts,
            crop=crop,
            device=cfg.device,
            mask_threshold=cfg.mask_threshold,
            min_blob_area_px=cfg.min_blob_area_px,
            prompts_digest=hashlib.sha256(path.read_bytes()).hexdigest()[:12],
        )

    def describe(self) -> dict:
        return {
            "method": self.name,
            "model": self.model_name,
            "prompts_sha256": self.prompts_digest,
            "crop": self.crop.describe() if self.crop else None,
            "mask_threshold": self.mask_threshold,
            "min_blob_area_px": self.min_blob_area_px,
        }

    # --- single frame (used by the prompt tool for a live preview) ----------------------

    def segment_single(self, image: np.ndarray, prompt: FramePrompt) -> np.ndarray:
        h, w = image.shape[:2]
        window = self._window(w, h)
        stream = _Stream(self, window, (h, w))
        stream.add_prompt(0, prompt)
        return stream.step(0, image)[0]

    # --- sequence ---------------------------------------------------------------------

    def segment_sequence(
        self,
        frame_files: list[str],
        load: Callable[[int], np.ndarray],
        needed: set[int],
    ) -> Iterator[tuple]:
        """Yield ``(index, mask)``, or ``(index, mask, variants)`` when variants are on."""
        index = {f: i for i, f in enumerate(frame_files)}
        keyframes: dict[int, FramePrompt] = {}
        for prompt in self.prompts.frames:
            if prompt.frame_file not in index:
                raise PromptError(
                    f"prompt frame {prompt.frame_file} is not among this camera's frames"
                )
            keyframes[index[prompt.frame_file]] = prompt
        if not needed:
            return
        first_key = min(keyframes)
        reference = load(first_key)
        h, w = reference.shape[:2]
        window = self._window(w, h)

        # Forward in time from the first prompt. Tracking is causal, so resuming from saved
        # state (or re-running it) reproduces the earlier frames' masks exactly.
        last_needed = max(needed)
        forward_needed = [i for i in needed if i >= first_key]
        if forward_needed:
            meta = self._state_meta(frame_files, first_key, keyframes, window, (h, w))
            stream = _Stream(self, window, (h, w))
            start = first_key
            resumed = self._resume(stream, meta, frame_files, first_key, min(forward_needed))
            if resumed is not None:
                start = resumed + 1
            self.last_resume = {"resumed_after": resumed, "steps": last_needed + 1 - start}
            for i in range(start, last_needed + 1):
                step = i - first_key
                if i in keyframes:
                    stream.add_prompt(step, keyframes[i])
                result = stream.step(step, reference if i == first_key else load(i))
                if i in needed:
                    yield (i, *result)
                if (step + 1) % CHECKPOINT_EVERY == 0 or i == last_needed:
                    self._save_state(stream, meta, frame_files, first_key, i)

        # Backward in time for frames before the first prompt (e.g. before the dye arrived).
        before = sorted((i for i in needed if i < first_key), reverse=True)
        if before:
            stream = _Stream(self, window, (h, w))
            for step, i in enumerate(range(first_key, before[-1] - 1, -1)):
                if step == 0:
                    stream.add_prompt(0, keyframes[first_key])
                result = stream.step(step, reference if i == first_key else load(i))
                if i in needed and i != first_key:
                    yield (i, *result)

    def _window(self, width: int, height: int) -> CropWindow:
        if self.crop is not None:
            return self.crop.window(width, height)
        return CropWindow.full(width, height)

    # --- saved tracking state -----------------------------------------------------------

    def _state_path(self) -> Path | None:
        return None if self.state_dir is None else Path(self.state_dir) / "forward.pt"

    def _state_meta(self, frame_files, first_key, keyframes, window, full_hw) -> dict:
        import torch
        import transformers

        return {
            "format": STATE_FORMAT, "model": self.model_name,
            "transformers": str(transformers.__version__), "torch": str(torch.__version__),
            "window": [window.x0, window.y0, window.x1, window.y1], "full_hw": list(full_hw),
            "first_frame": frame_files[first_key],
            "prompts": sorted([frame_files[i], list(map(list, p.points)), list(p.labels),
                               list(p.box) if p.box else None] for i, p in keyframes.items()),
        }

    def _resume(self, stream: _Stream, meta: dict, frame_files, first_key: int,
                first_needed: int) -> int | None:
        """Restore saved state into ``stream``; returns the last frame index it covers."""
        path = self._state_path()
        if path is None or not path.exists():
            return None
        import torch

        try:
            saved = torch.load(path, map_location=stream.device, weights_only=True)
        except Exception as exc:  # noqa: BLE001 - unreadable state just means re-tracking
            log.warning("ignoring saved SAM 2 state %s (%s); tracking from the start", path, exc)
            return None
        last = int(saved.get("last_index", -1))
        why = None
        if saved.get("meta") != meta:
            differ = sorted(k for k in meta if saved.get("meta", {}).get(k) != meta[k])
            why = f"{', '.join(differ)} changed"
        elif saved.get("frames") != frame_files[first_key:last + 1]:
            why = "frames up to the saved state changed (e.g. older photos were imported)"
        elif first_needed <= last:
            why = f"frame {frame_files[first_needed]} is before the saved state"
        if why:
            log.info("SAM 2: tracking from the prompted frame (%s)", why)
            return None
        stream.restore(saved["state"])
        log.info("SAM 2: resuming tracking after %s (%d frames already tracked)",
                 frame_files[last], last - first_key + 1)
        return last

    def _save_state(self, stream: _Stream, meta: dict, frame_files, first_key: int,
                    last_index: int) -> None:
        path = self._state_path()
        if path is None:
            return
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        torch.save({"meta": meta, "last_index": last_index,
                    "frames": frame_files[first_key:last_index + 1],
                    "state": stream.snapshot()}, part)
        os.replace(part, path)


@dataclass
class _RoiCrop:
    """Crop to the annotated region plus a margin; the window depends on the frame size."""

    roi: list
    margin: int

    def window(self, width: int, height: int) -> CropWindow:
        return CropWindow.around(self.roi, self.margin, width, height)

    def describe(self) -> dict:
        return {"roi_margin_px": self.margin}


class _Stream:
    """One streaming SAM 2 session over cropped frames."""

    def __init__(self, seg: Sam2VideoSegmenter, window: CropWindow, full_hw: tuple[int, int]):
        self.seg = seg
        self.window = window
        self.full_hw = full_hw
        self.model, self.processor, self.device, self.dtype = load_model(seg.model_name,
                                                                         seg.device)
        self.session = self.processor.init_video_session(
            inference_device=self.device, dtype=self.dtype
        )
        self.crop_hw = (window.height, window.width)

    def add_prompt(self, step: int, prompt: FramePrompt) -> None:
        kwargs = {}
        if prompt.points:
            kwargs["input_points"] = [[[list(p) for p in self.window.to_crop(prompt.points)]]]
            kwargs["input_labels"] = [[list(prompt.labels)]]
        if prompt.box is not None:
            kwargs["input_boxes"] = [[list(self.window.box_to_crop(prompt.box))]]
        self.processor.add_inputs_to_inference_session(
            inference_session=self.session, frame_idx=step, obj_ids=OBJ_ID,
            original_size=self.crop_hw, **kwargs,
        )

    def snapshot(self) -> dict:
        """What SAM 2 needs to continue tracking: everything but frames and full-size masks."""
        state = {name: getattr(self.session, name) for name in STATE_ATTRS}
        state["output_dict_per_obj"] = {
            obj: {kind: {frame: {k: v for k, v in out.items() if k not in STATE_DROPPED_OUTPUTS}
                         for frame, out in frames.items()}
                  for kind, frames in per_obj.items()}
            for obj, per_obj in self.session.output_dict_per_obj.items()
        }
        return state

    def restore(self, state: dict) -> None:
        for name, value in state.items():
            setattr(self.session, name, value)

    def step(self, step: int, aligned_bgr: np.ndarray) -> tuple:
        """``(mask,)``, or ``(mask, [narrower, wider])`` when variants are on."""
        rgb = cv2.cvtColor(self.window.crop(aligned_bgr), cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, device=self.device, return_tensors="pt")
        pixel_values = inputs.pixel_values[0].to(self.device, dtype=self.dtype)
        out = self.model(inference_session=self.session, frame_idx=step, frame=pixel_values)
        logits = self.processor.post_process_masks(
            [out.pred_masks.float()], original_sizes=[list(self.crop_hw)], binarize=False,
        )[0][0, 0].cpu().numpy()
        self._prune(step)
        t, delta = self.seg.mask_threshold, self.seg.variant_logit_delta
        thresholds = [t] if delta is None else [t, t + delta, t - delta]
        masks = [_drop_small_blobs(self.window.paste(logits > x, *self.full_hw),
                                   self.seg.min_blob_area_px) for x in thresholds]
        return (masks[0],) if delta is None else (masks[0], masks[1:])

    def _prune(self, step: int) -> None:
        """Free per-frame state SAM 2 no longer reads, so long runs don't exhaust memory."""
        frames = self.session.processed_frames or {}
        for idx in [i for i in frames if i < step]:
            del frames[idx]
        for outputs in self.session.output_dict_per_obj.values():
            old = outputs.get("non_cond_frame_outputs", {})
            for idx in [i for i in old if i < step - KEEP_HISTORY]:
                del old[idx]


def _drop_small_blobs(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0 or not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]
