"""SAM 2 video segmentation (Hugging Face ``transformers``), prompted with clicks.

Frames are streamed one at a time, so memory use stays flat for long time-lapses. The
target is tracked forward in time from the earliest prompted frame; frames before it are
tracked in a second stream running backward in time.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fungus_cv.segment.prompts import CropWindow, FramePrompt, Prompts
from fungus_cv.segment.torch_device import choose_device, import_torch

log = logging.getLogger(__name__)

OBJ_ID = 1
# Per-frame model state older than this many frames is no longer read by SAM 2
# (7 mask memories, 16 object pointers) and is freed during long streams.
KEEP_HISTORY = 20

_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


class PromptError(ValueError):
    pass


def load_model(name: str, device_pref: str = "auto"):
    """Load (and cache) the SAM 2 video model and processor. Downloads weights on first use."""
    torch = import_torch()
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
    name: str = field(default="sam2", init=False)

    @classmethod
    def from_config(cls, cfg, experiment_root: Path | None, roi) -> Sam2VideoSegmenter:
        path = Path(experiment_root or ".") / cfg.prompts_file
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

        # Forward in time from the first prompt. Tracking is causal, so re-running it for
        # new frames reproduces the earlier frames' masks exactly.
        last_needed = max(needed)
        if last_needed >= first_key:
            stream = _Stream(self, window, (h, w))
            for step, i in enumerate(range(first_key, last_needed + 1)):
                if i in keyframes:
                    stream.add_prompt(step, keyframes[i])
                result = stream.step(step, reference if i == first_key else load(i))
                if i in needed:
                    yield (i, *result)

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
