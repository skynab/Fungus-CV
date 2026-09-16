"""Frame-by-frame analysis: align -> segment -> measure, written incrementally to CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fungus_cv import __version__
from fungus_cv.measure.geometry import (
    ANNOTATIONS_NAME,
    Annotations,
    extent_uncertainty_mm,
    measure_extent,
)
from fungus_cv.preprocess.align import ECC_POOR, Alignment, align_frame
from fungus_cv.preprocess.lighting import LightingNormalizer, LightingResult
from fungus_cv.preprocess.markers import Scale, detect_markers, scale_from_markers
from fungus_cv.preprocess.rectify import Rectification, fit_rectification
from fungus_cv.quality import mean_brightness, sharpness
from fungus_cv.segment.base import build_segmenter, is_sequence_segmenter
from fungus_cv.storage import Experiment, iso_utc, utc_now

log = logging.getLogger(__name__)

RESULTS_DIR = "results"
MEASUREMENTS_NAME = "measurements.csv"
RUN_INFO_NAME = "run_info.json"

MEASUREMENT_FIELDS = [
    "timestamp_utc", "frame_file", "camera", "settings_hash",
    "align_method", "align_rms_px", "align_ecc", "align_scale", "align_shift_px",
    "target_px", "front_width_px", "extent_px", "extent_max_px", "extent_fraction",
    "extent_mm", "extent_mm_unc", "extent_max_mm", "axis_length_mm",
    "target_area_mm2", "roi_area_mm2", "coverage_pct",
    "mean_brightness", "sharpness", "light_gain_b", "light_gain_g", "light_gain_r",
    "flags", "mask_file", "overlay_file",
]


class AnalysisError(RuntimeError):
    pass


@dataclass
class Prepared:
    image: np.ndarray  # raw frame as captured
    frame: np.ndarray  # aligned, rectified and lighting-corrected: what gets segmented
    alignment: Alignment
    lighting: LightingResult | None


@dataclass
class AnalysisSummary:
    processed: int = 0
    skipped_existing: int = 0
    failed: int = 0
    flagged: Counter = field(default_factory=Counter)
    settings_hash: str = ""


def _fmt(value, digits: int = 4):
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if np.isnan(value) else round(value, digits)
    return value


def choose_camera(experiment: Experiment, rows: list[dict]) -> str:
    wanted = experiment.config.analysis.camera
    if wanted:
        return wanted
    counts = Counter(r["camera"] for r in rows if r["status"] == "ok")
    if not counts:
        raise AnalysisError("no captured frames yet")
    return counts.most_common(1)[0][0]


def frames_for_analysis(experiment: Experiment) -> list[dict]:
    rows = experiment.read_frames()
    camera = choose_camera(experiment, rows)
    frames = [r for r in rows if r["status"] == "ok" and r["camera"] == camera]
    if not frames:
        raise AnalysisError(f"no frames for camera {camera!r}")
    return sorted(frames, key=lambda r: r["timestamp_utc"])


def annotations_path(experiment: Experiment) -> Path:
    return experiment.root / ANNOTATIONS_NAME


def read_image(experiment: Experiment, rel: str) -> np.ndarray:
    image = cv2.imread(str(experiment.root / rel), cv2.IMREAD_COLOR)
    if image is None:
        raise AnalysisError(f"cannot read image {rel}")
    return image


class Analyzer:
    """Prepares frames (align -> rectify -> lighting) and measures them.

    ``require_annotations=False`` is for tools that need prepared frames before the region
    has been drawn (e.g. `fungus annotate` itself).
    """

    def __init__(self, experiment: Experiment, with_segmenter: bool = True,
                 require_annotations: bool = True):
        self.experiment = experiment
        self.cfg = experiment.config.analysis
        self.results_dir = experiment.root / RESULTS_DIR
        self.measurements_path = self.results_dir / MEASUREMENTS_NAME

        frames = frames_for_analysis(experiment)
        self.frames = frames
        self.reference_row = frames[0]
        self.raw_reference = read_image(experiment, self.reference_row["file"])
        raw_h, raw_w = self.raw_reference.shape[:2]
        self.raw_markers = detect_markers(self.raw_reference, self.cfg.markers.dictionary)
        size_mm = self.cfg.markers.size_mm

        # Perspective correction, fitted once on the reference frame.
        self.rectification: Rectification | None = None
        if self.cfg.rectify.enabled:
            if size_mm is None or not self.raw_markers:
                raise AnalysisError("analysis.rectify needs markers.size_mm and at least one "
                                    "marker visible in the reference frame")
            self.rectification = fit_rectification(self.raw_markers, size_mm, (raw_w, raw_h),
                                                   self.cfg.rectify.max_side_px)
            log.info("rectified to %dx%d px, %.3f px/mm, marker fit residual %.3f mm "
                     "(%d markers)", *self.rectification.size, self.rectification.px_per_mm,
                     self.rectification.residual_rms_mm, self.rectification.n_markers)
            if self.rectification.n_markers == 1:
                log.warning("perspective correction from a single marker extrapolates across "
                            "the image; use 2-4 markers spread around the measured area")
        self.reference = self._rectify(self.raw_reference)
        h, w = self.reference.shape[:2]

        ann_path = annotations_path(experiment)
        self.annotations: Annotations | None = None
        if ann_path.exists():
            self.annotations = Annotations.load(ann_path)
            if tuple(self.annotations.image_size) != (w, h):
                message = (
                    f"annotations were drawn on a {self.annotations.image_size} image but the "
                    f"prepared reference frame is {(w, h)} (did rectify settings change?); "
                    "run `fungus annotate` again"
                )
                if require_annotations:
                    raise AnalysisError(message)
                log.warning("ignoring existing annotations: %s", message)
                self.annotations = None
            if self.annotations is not None and self.annotations.reference_file and \
                    self.annotations.reference_file != self.reference_row["file"]:
                log.warning(
                    "annotations were drawn on %s but the reference frame is now %s",
                    self.annotations.reference_file, self.reference_row["file"],
                )
        elif require_annotations:
            raise AnalysisError(
                f"{ann_path.name} not found; run `fungus annotate {experiment.root}` first"
            )

        # Scale from markers in the prepared (possibly rectified) reference.
        self.ref_markers = detect_markers(self.reference, self.cfg.markers.dictionary)
        self.scale: Scale | None = None
        if size_mm is None:
            log.warning("analysis.markers.size_mm is not set; results will be in pixels only")
        elif not self.ref_markers:
            log.warning("no ArUco markers found in the reference frame; results in pixels only")
        else:
            self.scale = scale_from_markers(self.ref_markers, size_mm)
            log.info(
                "scale %.5f mm/px (sd %.5f, %d edges, markers %s)",
                self.scale.mm_per_px, self.scale.std_mm_per_px, self.scale.n_edges,
                self.scale.marker_ids,
            )
            if self.scale.std_mm_per_px / self.scale.mm_per_px > 0.02:
                log.warning(
                    "marker edges disagree by more than 2%%; the camera may not be square-on "
                    "to the markers. Set analysis.rectify.enabled: true to correct the view"
                )
        # Alignment residuals are measured in raw pixels; convert to prepared pixels.
        self.align_px_factor = 1.0
        if self.rectification is not None and self.raw_markers:
            raw_scale = scale_from_markers(self.raw_markers, size_mm)
            self.align_px_factor = self.rectification.px_per_mm * raw_scale.mm_per_px

        # Intensity alignment (raw frame coordinates) ignores the measured region.
        self.align_exclude = None
        if self.annotations is not None:
            roi = self.annotations.roi_mask(self.reference.shape)
            if self.rectification is not None:
                roi = self.rectification.unwarp_mask(roi, (raw_w, raw_h))
            margin = max(15, int(0.02 * max(raw_w, raw_h)))
            self.align_exclude = cv2.dilate(roi.astype(np.uint8),
                                            np.ones((margin, margin), np.uint8)).astype(bool)

        self.lighting: LightingNormalizer | None = None
        if self.cfg.lighting.method != "none":
            self.lighting = self._build_lighting()

        self.ref_brightness = mean_brightness(self.raw_reference)
        self.ref_sharpness = sharpness(self.raw_reference)
        if not with_segmenter:  # e.g. the prompt tool only needs aligned frames
            return
        try:
            self.segmenter = build_segmenter(self.cfg.target, experiment.root,
                                             self.annotations.roi)
        except (ValueError, RuntimeError, OSError) as exc:  # missing prompts, model, torch
            raise AnalysisError(str(exc)) from exc
        analysis_settings = self.cfg.model_dump(mode="json")
        analysis_settings["target"] = self.cfg.target.selected()
        self.settings = {
            "fungus_cv_version": __version__,
            "reference_file": self.reference_row["file"],
            "analysis": analysis_settings,
            "segmenter": self.segmenter.describe(),
            "annotations": json.loads(ann_path.read_text(encoding="utf-8")),
            "rectification": self.rectification.to_dict() if self.rectification else None,
        }
        blob = json.dumps(self.settings, sort_keys=True).encode()
        self.settings_hash = hashlib.sha256(blob).hexdigest()[:12]
        # Masks are kept per settings hash so runs with different methods can be compared.
        self.masks_dir = self.results_dir / "masks" / self.settings_hash

    # --- bookkeeping ---------------------------------------------------------------

    def _existing(self) -> set[str]:
        """Frames already measured with the current settings. Results made with different
        settings are archived, never mixed."""
        if not self.measurements_path.exists():
            return set()
        with open(self.measurements_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        hashes = {r["settings_hash"] for r in rows}
        if hashes and hashes != {self.settings_hash}:
            archive = self.results_dir / "archive"
            archive.mkdir(exist_ok=True)
            old = "-".join(sorted(hashes))
            target = archive / f"measurements_{old}_{utc_now():%Y%m%dT%H%M%SZ}.csv"
            os.replace(self.measurements_path, target)
            log.info("analysis settings changed; previous results archived to %s", target.name)
            return set()
        return {r["frame_file"] for r in rows}

    def _append(self, row: dict) -> None:
        new = not self.measurements_path.exists()
        with open(self.measurements_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MEASUREMENT_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)

    def _write_run_info(self) -> None:
        info = {
            "settings_hash": self.settings_hash,
            "updated_utc": iso_utc(utc_now()),
            "opencv_version": cv2.__version__,
            "scale": self.scale.to_dict() if self.scale else None,
            "reference_markers": sorted(self.ref_markers),
            "rectification": self.rectification.to_dict() if self.rectification else None,
            "settings": self.settings,
        }
        text = json.dumps(info, indent=2)
        (self.results_dir / RUN_INFO_NAME).write_text(text, "utf-8")
        runs = self.results_dir / "runs"
        runs.mkdir(exist_ok=True)
        (runs / f"{self.settings_hash}.json").write_text(text, "utf-8")

    # --- processing ----------------------------------------------------------------

    def run(self, force: bool = False) -> AnalysisSummary:
        self.results_dir.mkdir(exist_ok=True)
        if force and self.measurements_path.exists():
            self.measurements_path.unlink()
        done = self._existing()
        self._write_run_info()
        summary = AnalysisSummary(settings_hash=self.settings_hash)
        pending = [i for i, r in enumerate(self.frames) if r["file"] not in done]
        summary.skipped_existing = len(self.frames) - len(pending)
        if not pending:
            return summary

        if is_sequence_segmenter(self.segmenter):
            results = self._run_sequence(pending, summary)
        else:
            results = self._run_per_frame(pending, summary)
        for row in results:
            self._append(row)
            summary.processed += 1
            for flag in filter(None, row["flags"].split(";")):
                summary.flagged[flag] += 1
        return summary

    def _run_per_frame(self, pending: list[int], summary: AnalysisSummary):
        for i in pending:
            frame_row = self.frames[i]
            try:
                prepared = self._prepare(frame_row)
                mask = self.segmenter.segment(prepared.frame)
            except AnalysisError as exc:
                log.error("%s: %s", frame_row["file"], exc)
                summary.failed += 1
                continue
            yield self._measure(frame_row, prepared, mask)

    def _run_sequence(self, pending: list[int], summary: AnalysisSummary):
        """Memory-based segmenters see frames in order; only ``pending`` rows are written."""
        needed = set(pending)
        cache: dict[int, tuple] = {}

        def load(i: int) -> np.ndarray:
            if i not in cache:
                prepared = self._prepare(self.frames[i])
                if i not in needed:
                    return prepared.frame
                cache[i] = prepared
            return cache[i].frame

        files = [r["file"] for r in self.frames]
        try:
            for i, mask in self.segmenter.segment_sequence(files, load, needed):
                prepared = cache.pop(i) if i in cache else self._prepare(self.frames[i])
                yield self._measure(self.frames[i], prepared, mask)
                needed.discard(i)
                done = len(pending) - len(needed)
                if done % 10 == 0 or not needed:
                    log.info("segmented %d/%d frames", done, len(pending))
        except AnalysisError as exc:
            log.error("sequence segmentation stopped: %s", exc)
        summary.failed += len(needed)

    def aligned_frame(self, index: int) -> np.ndarray:
        """Frame ``index`` exactly as the segmenter sees it (aligned, rectified, lighting)."""
        return self._prepare(self.frames[index]).frame

    def prepared_reference(self) -> np.ndarray:
        return self._prepare(self.reference_row).frame

    def _rectify(self, image: np.ndarray) -> np.ndarray:
        return image if self.rectification is None else self.rectification.warp(image)

    def _build_lighting(self) -> LightingNormalizer:
        method = self.cfg.lighting.method
        shape = self.reference.shape
        if self.annotations is None:
            raise AnalysisError(f"lighting method {method!r} needs `fungus annotate` first")
        if method == "patch":
            region = self.annotations.patch_mask(shape)
            if region is None:
                raise AnalysisError("lighting method 'patch' needs a reference patch; run "
                                    "`fungus annotate` and mark a neutral card")
        else:  # background: everything well outside the measured region
            roi = self.annotations.roi_mask(shape).astype(np.uint8)
            margin = max(15, int(0.02 * max(shape[:2])))
            region = ~cv2.dilate(roi, np.ones((margin, margin), np.uint8)).astype(bool)
        region &= self._rectify(np.ones(self.raw_reference.shape[:2], np.uint8)).astype(bool)
        try:
            return LightingNormalizer(self.reference, region)
        except ValueError as exc:
            raise AnalysisError(str(exc)) from exc

    def _prepare(self, frame_row: dict) -> Prepared:
        image = read_image(self.experiment, frame_row["file"])
        if image.shape != self.raw_reference.shape:
            raise AnalysisError(
                f"size {image.shape} differs from reference {self.raw_reference.shape}")
        aligned, alignment = align_frame(
            image, self.raw_reference, self.raw_markers, self.cfg.align,
            self.cfg.markers.dictionary, exclude=self.align_exclude,
        )
        frame = self._rectify(aligned)
        valid = None
        lighting = None
        if self.lighting is not None:
            ones = np.ones(image.shape[:2], np.uint8)
            if alignment.method not in ("identity", "failed"):
                ones = cv2.warpAffine(ones, alignment.matrix, (ones.shape[1], ones.shape[0]))
            valid = self._rectify(ones).astype(bool)
            try:
                frame, lighting = self.lighting.apply(frame, valid)
            except ValueError as exc:
                raise AnalysisError(f"lighting correction failed: {exc}") from exc
        return Prepared(image, frame, alignment, lighting)

    def _measure(self, frame_row: dict, prepared: Prepared, mask: np.ndarray) -> dict:
        exp = self.experiment
        image, alignment = prepared.image, prepared.alignment
        m = measure_extent(mask, self.annotations, self.cfg.front_percentile)

        flags = []
        if alignment.method == "failed":
            flags.append("align_failed")
        elif (alignment.rms_px is not None and alignment.rms_px > 2) or \
                (alignment.ecc is not None and alignment.ecc < ECC_POOR):
            flags.append("align_poor")
        if m.target_px == 0:
            flags.append("no_target")
        brightness = mean_brightness(image)
        sharp = sharpness(image)
        if self.ref_brightness > 0 and abs(brightness / self.ref_brightness - 1) > 0.2:
            flags.append("brightness_changed")
        if self.ref_sharpness > 0 and sharp < 0.5 * self.ref_sharpness:
            flags.append("blurry")
        light = prepared.lighting
        if light is not None:
            if light.max_change > self.cfg.lighting.flag_change:
                flags.append("lighting_changed")
            if light.saturated_fraction > 0.02:
                flags.append("saturated")

        row = {
            "timestamp_utc": frame_row["timestamp_utc"],
            "frame_file": frame_row["file"],
            "camera": frame_row["camera"],
            "settings_hash": self.settings_hash,
            "align_method": alignment.method,
            "align_rms_px": _fmt(alignment.rms_px, 3),
            "align_ecc": _fmt(alignment.ecc, 4),
            "align_scale": _fmt(alignment.scale, 5),
            "align_shift_px": _fmt(alignment.shift_px, 2),
            "target_px": m.target_px,
            "front_width_px": m.front_width_px,
            "extent_px": _fmt(m.extent_px, 2),
            "extent_max_px": _fmt(m.extent_max_px, 2),
            "extent_fraction": _fmt(m.extent_fraction, 5),
            "coverage_pct": _fmt(100 * m.coverage_fraction, 3),
            "mean_brightness": _fmt(brightness, 2),
            "sharpness": _fmt(sharp, 2),
            "light_gain_b": light.gains[0] if light else "",
            "light_gain_g": light.gains[1] if light else "",
            "light_gain_r": light.gains[2] if light else "",
            "flags": ";".join(flags),
            "extent_mm": "", "extent_mm_unc": "", "extent_max_mm": "", "axis_length_mm": "",
            "target_area_mm2": "", "roi_area_mm2": "", "mask_file": "", "overlay_file": "",
        }
        if self.scale is not None:
            k = self.scale.mm_per_px
            row.update({
                "extent_mm": _fmt(m.extent_px * k, 3),
                "extent_mm_unc": _fmt(extent_uncertainty_mm(
                    m.extent_px, k, self.scale.se_mm_per_px,
                    None if alignment.rms_px is None
                    else alignment.rms_px * self.align_px_factor), 3),
                "extent_max_mm": _fmt(m.extent_max_px * k, 3),
                "axis_length_mm": _fmt(self.annotations.axis_length_px * k, 3),
                "target_area_mm2": _fmt(m.target_px * k * k, 2),
                "roi_area_mm2": _fmt(m.roi_area_px * k * k, 2),
            })

        stem = Path(frame_row["file"]).stem
        if self.cfg.save_masks:
            mask_path = self.masks_dir / f"{stem}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            row["mask_file"] = mask_path.relative_to(exp.root).as_posix()
        if self.cfg.save_overlays:
            from fungus_cv.analyze.overlay import draw_overlay

            overlay = draw_overlay(prepared.frame, mask, self.annotations, m, row)
            overlay_path = self.results_dir / "overlays" / self.settings_hash / f"{stem}.jpg"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
            row["overlay_file"] = overlay_path.relative_to(exp.root).as_posix()
        return row


def analyze(experiment: Experiment, force: bool = False) -> AnalysisSummary:
    return Analyzer(experiment).run(force=force)


def watch(experiment_root: Path, poll_seconds: float = 30.0, on_summary=None) -> None:
    """Analyze new frames as they arrive (e.g. alongside `fungus capture`)."""
    while True:
        experiment = Experiment(experiment_root)  # re-read config each pass
        try:
            summary = analyze(experiment)
            if on_summary and summary.processed:
                on_summary(summary)
        except AnalysisError as exc:
            log.info("waiting: %s", exc)
        time.sleep(poll_seconds)
