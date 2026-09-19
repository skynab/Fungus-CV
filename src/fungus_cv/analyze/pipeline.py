"""Frame-by-frame analysis: align -> segment -> measure, written incrementally to CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fungus_cv import __version__
from fungus_cv.measure import color_classes
from fungus_cv.measure.centerline import CenterlineError, centerline_from_mask
from fungus_cv.measure.color_indices import color_indices
from fungus_cv.measure.geometry import (
    ANNOTATIONS_NAME,
    Annotations,
    area_uncertainty_mm2,
    extent_uncertainty_mm,
)
from fungus_cv.measure.path import ExtentMeasurement, Polyline, measure_along_path
from fungus_cv.preprocess.align import ECC_POOR, Alignment, align_frame
from fungus_cv.preprocess.lighting import LightingNormalizer, LightingResult
from fungus_cv.preprocess.markers import Scale, detect_markers, scale_from_markers
from fungus_cv.preprocess.rectify import Rectification, fit_rectification
from fungus_cv.quality import contrast_normalized_sharpness, mean_brightness, sharpness
from fungus_cv.segment.base import build_segmenter, is_sequence_segmenter, segment_with_variants
from fungus_cv.storage import Experiment, iso_utc, utc_now

log = logging.getLogger(__name__)

RESULTS_DIR = "results"
MEASUREMENTS_NAME = "measurements.csv"
RUN_INFO_NAME = "run_info.json"

MEASUREMENT_FIELDS = [
    "timestamp_utc", "frame_file", "camera", "plot", "settings_hash",
    "align_method", "align_rms_px", "align_ecc", "align_scale", "align_shift_px",
    "target_px", "front_width_px", "extent_px", "extent_max_px", "extent_fraction",
    "path_length_px", "covered_length_px", "covered_length_pct",
    "reference_px", "reference_covered_pct",
    "extent_mm", "extent_mm_unc", "extent_mm_seg_unc", "extent_px_seg_unc",
    "extent_max_mm", "axis_length_mm", "covered_length_mm", "reference_area_mm2",
    "target_area_mm2", "target_area_mm2_unc", "roi_area_mm2", "coverage_pct", "coverage_pct_unc",
    "equivalent_radius_mm",
    "gcc_mean", "gcc_p90", "rcc_mean", "exg_mean",
    "mean_brightness", "sharpness", "light_gain_b", "light_gain_g", "light_gain_r",
    "model_input_distance",
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
    stopped: bool = False


def _fmt(value, digits: int = 4):
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if np.isnan(value) else round(value, digits)
    return value


def choose_camera(experiment: Experiment, rows: list[dict], camera: str | None = None) -> str:
    wanted = camera or experiment.config.analysis.camera
    if wanted:
        return wanted
    counts = Counter(r["camera"] for r in rows if r["status"] == "ok")
    if not counts:
        raise AnalysisError("no captured frames yet")
    return counts.most_common(1)[0][0]


def cameras_with_frames(experiment: Experiment) -> list[str]:
    """Cameras that actually took photos, in the order they appear in the config."""
    have = {r["camera"] for r in experiment.read_frames() if r["status"] == "ok"}
    configured = [c.name for c in experiment.config.cameras if c.name in have]
    return configured + sorted(have - set(configured))


def multi_camera(experiment: Experiment) -> bool:
    return len(experiment.config.cameras) > 1


def results_dir_for(experiment: Experiment, camera: str | None = None) -> Path:
    """Where one camera's results live: ``results/`` for a single camera, and
    ``results/cameras/<name>/`` when the experiment has several, so they never mix."""
    base = experiment.root / RESULTS_DIR
    if camera is None or not multi_camera(experiment):
        return base
    return base / "cameras" / camera


def frames_for_analysis(experiment: Experiment, camera: str | None = None) -> list[dict]:
    rows = experiment.read_frames()
    camera = choose_camera(experiment, rows, camera)
    frames = [r for r in rows if r["status"] == "ok" and r["camera"] == camera]
    if not frames:
        raise AnalysisError(f"no frames for camera {camera!r}")
    return sorted(frames, key=lambda r: r["timestamp_utc"])


def annotations_path(experiment: Experiment, camera: str | None = None) -> Path:
    """``annotations.json``, or ``annotations_<camera>.json`` when the experiment has several
    cameras: each one sees a different scene, so each needs its own base, path and region."""
    if camera and multi_camera(experiment):
        return experiment.root / f"annotations_{camera}.json"
    return experiment.root / ANNOTATIONS_NAME


def prompts_path(experiment: Experiment, prompts_file: str, camera: str | None = None) -> Path:
    """The SAM prompts file for one camera (``prompts_<camera>.json`` with several)."""
    path = Path(prompts_file)
    if camera and multi_camera(experiment):
        return experiment.root / f"{path.stem}_{camera}{path.suffix or '.json'}"
    return experiment.root / prompts_file


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
                 require_annotations: bool = True, results_dir: Path | None = None,
                 camera: str | None = None):
        self.experiment = experiment
        self.cfg = experiment.config.analysis
        self.camera = choose_camera(experiment, experiment.read_frames(), camera)
        # A different results folder keeps trial runs (e.g. `fungus sensitivity`) apart from
        # the experiment's own results. It must stay inside the experiment, because mask and
        # overlay paths are recorded relative to it.
        self.results_dir = (Path(results_dir) if results_dir
                            else results_dir_for(experiment, self.camera))
        self.measurements_path = self.results_dir / MEASUREMENTS_NAME

        frames = frames_for_analysis(experiment, self.camera)
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

        ann_path = annotations_path(experiment, self.camera)
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
        # Judge focus where the scene doesn't change: outside the measured region.
        self.quality_region = None if self.align_exclude is None else ~self.align_exclude
        self.ref_sharpness = contrast_normalized_sharpness(self.raw_reference,
                                                           self.quality_region)
        if not with_segmenter:  # e.g. the prompt tool only needs aligned frames
            return
        measure = self.cfg.measure
        if measure.mode == "path" and measure.path_source == "reference" and \
                self.cfg.reference.method == "none":
            raise AnalysisError("measure.path_source: reference needs analysis.reference.method "
                                "(color, sam2 or model) to segment the stem in each frame")
        self.roi = self.annotations.roi_mask(self.reference.shape)
        self.plots = self.annotations.measured_plots()
        self.plot_masks = {p.name: p.mask(self.reference.shape) for p in self.plots}
        self.static_paths = {p.name: p.polyline(follow_path=measure.mode == "path")
                             for p in self.plots if p.has_axis}
        try:
            self.segmenter = build_segmenter(self.cfg.target, experiment.root,
                                             self.annotations.roi, self.cfg.uncertainty,
                                             prompts_file=self._prompts_file("target"))
            self.reference_segmenter = None
            if self.cfg.reference.method != "none":
                self.reference_segmenter = build_segmenter(
                    self.cfg.reference, experiment.root, self.annotations.roi,
                    prompts_file=self._prompts_file("reference"))
        except (ValueError, RuntimeError, OSError) as exc:  # missing prompts, model, torch
            raise AnalysisError(str(exc)) from exc
        analysis_settings = self.cfg.model_dump(mode="json")
        analysis_settings["camera"] = self.camera
        analysis_settings["target"] = self.cfg.target.selected()
        analysis_settings["reference"] = self.cfg.reference.selected()
        self.settings = {
            "fungus_cv_version": __version__,
            "reference_file": self.reference_row["file"],
            "analysis": analysis_settings,
            "segmenter": self.segmenter.describe(),
            "reference_segmenter": (self.reference_segmenter.describe()
                                    if self.reference_segmenter else None),
            "annotations": json.loads(ann_path.read_text(encoding="utf-8")),
            "rectification": self.rectification.to_dict() if self.rectification else None,
        }
        blob = json.dumps(self.settings, sort_keys=True).encode()
        self.settings_hash = hashlib.sha256(blob).hexdigest()[:12]
        # Masks are kept per settings hash so runs with different methods can be compared.
        self.masks_dir = self.results_dir / "masks" / self.settings_hash
        # Sequence segmenters (SAM 2) resume tracking from state saved with this run's masks.
        for role, seg in (("target", self.segmenter), ("reference", self.reference_segmenter)):
            if seg is not None and hasattr(seg, "state_dir"):
                seg.state_dir = self.masks_dir / "sam2_state" / role

    def _prompts_file(self, role: str) -> Path:
        """SAM prompts for this camera (each camera is prompted separately)."""
        return prompts_path(self.experiment, getattr(self.cfg, role).sam2.prompts_file,
                            self.camera)

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

    @property
    def measurement_fields(self) -> list[str]:
        """The fixed columns, then one share column (and its uncertainty) per colour class."""
        extra = color_classes.columns([c.name for c in self.cfg.color_classes])
        i = MEASUREMENT_FIELDS.index("mean_brightness")
        return MEASUREMENT_FIELDS[:i] + extra + MEASUREMENT_FIELDS[i:]

    def _class_shares(self, frame: np.ndarray, region: np.ndarray) -> dict[str, str]:
        classes = [(c.name, [(tuple(r.lower), tuple(r.upper)) for r in c.hsv_ranges])
                   for c in self.cfg.color_classes]
        unc = self.cfg.uncertainty
        delta = tuple(unc.hsv_delta) if unc.segmentation else None
        return {k: _fmt(v, 3)
                for k, v in color_classes.class_shares(frame, region, classes, delta).items()}

    def _append(self, row: dict) -> None:
        new = not self.measurements_path.exists()
        with open(self.measurements_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.measurement_fields)
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

    def run(self, force: bool = False, progress=None, should_stop=None) -> AnalysisSummary:
        """Measure pending frames.

        ``progress(done, total, stage)`` is called after each frame; ``should_stop()`` is
        checked between frames so a GUI can cancel (finished frames stay saved).
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if force and self.measurements_path.exists():
            self.measurements_path.unlink()
        done = self._existing()
        self._write_run_info()
        summary = AnalysisSummary(settings_hash=self.settings_hash)
        pending = [i for i, r in enumerate(self.frames) if r["file"] not in done]
        summary.skipped_existing = len(self.frames) - len(pending)
        if not pending:
            return summary

        if self.reference_segmenter is not None:
            # Pass 1: the reference object (e.g. stem), saved so pass 2 can measure against it.
            ref_summary = AnalysisSummary()
            for n, (i, _, ref_mask, _) in enumerate(self._iter_masks(self.reference_segmenter,
                                                                  pending, ref_summary), 1):
                if should_stop and should_stop():
                    return summary
                if progress:
                    progress(n, len(pending), "reference")
                path = self._reference_mask_path(i)
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), ref_mask.astype(np.uint8) * 255)
            if ref_summary.failed:
                log.warning("reference segmentation failed for %d frame(s)", ref_summary.failed)

        for i, prepared, mask, variants in self._iter_masks(self.segmenter, pending, summary):
            if should_stop and should_stop():
                summary.stopped = True
                break
            ref_mask = None
            if self.reference_segmenter is not None:
                ref_img = cv2.imread(str(self._reference_mask_path(i)), cv2.IMREAD_GRAYSCALE)
                ref_mask = None if ref_img is None else ref_img > 127
            rows = self._measure(self.frames[i], prepared, mask, ref_mask, variants)
            for row in rows:
                self._append(row)
                for flag in filter(None, row["flags"].split(";")):
                    summary.flagged[flag] += 1
            summary.processed += 1
            if progress:
                progress(summary.processed, len(pending), "measure")
        return summary

    def _reference_mask_path(self, index: int) -> Path:
        return self.masks_dir / "reference" / f"{Path(self.frames[index]['file']).stem}.png"

    def _iter_masks(self, segmenter, pending: list[int], summary: AnalysisSummary):
        """Yield ``(index, prepared frame, mask, variant masks)`` for pending frames."""
        if is_sequence_segmenter(segmenter):
            yield from self._iter_sequence(segmenter, pending, summary)
            return
        for i in pending:
            frame_row = self.frames[i]
            try:
                prepared = self._prepare(frame_row)
                mask, variants = segment_with_variants(segmenter, prepared.frame)
            except AnalysisError as exc:
                log.error("%s: %s", frame_row["file"], exc)
                summary.failed += 1
                continue
            yield i, prepared, mask, variants

    def _iter_sequence(self, segmenter, pending: list[int], summary: AnalysisSummary):
        """Memory-based segmenters see frames in order; only ``pending`` masks are yielded."""
        needed = set(pending)
        cache: dict[int, Prepared] = {}

        def load(i: int) -> np.ndarray:
            if i not in cache:
                prepared = self._prepare(self.frames[i])
                if i not in needed:
                    return prepared.frame
                cache[i] = prepared
            return cache[i].frame

        files = [r["file"] for r in self.frames]
        try:
            for i, mask, *variants in segmenter.segment_sequence(files, load, needed):
                prepared = cache.pop(i) if i in cache else self._prepare(self.frames[i])
                yield i, prepared, mask, variants[0] if variants else []
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

    def _measure_path(self, plot, mask: np.ndarray, ref_mask: np.ndarray | None,
                      flags: list[str]) -> Polyline | None:
        if not plot.has_axis:
            return None
        static = self.static_paths[plot.name]
        measure = self.cfg.measure
        if measure.mode != "path" or measure.path_source != "reference":
            return static
        if ref_mask is None:
            flags.append("no_reference")
            return static
        try:
            # The object includes its target: moss can cover the stem completely.
            return centerline_from_mask((ref_mask | mask) & self.plot_masks[plot.name],
                                        plot.base, measure.smooth_px)
        except CenterlineError as exc:
            log.warning("centerline failed for %s (%s); using the annotated path",
                        plot.name, exc)
            flags.append("centerline_failed")
            return static

    def _measure_mask(self, mask: np.ndarray, path: Polyline | None, region: np.ndarray,
                      reference: np.ndarray | None) -> ExtentMeasurement:
        if path is not None:
            return measure_along_path(mask, path, region, self.cfg.front_percentile,
                                      self.cfg.measure.corridor_px, reference_mask=reference)
        return _area_only(mask, region, reference)

    def _frame_flags(self, prepared: Prepared) -> tuple[list[str], dict]:
        image, alignment = prepared.image, prepared.alignment
        flags = []
        if alignment.method == "failed":
            flags.append("align_failed")
        elif (alignment.rms_px is not None and alignment.rms_px > 2) or \
                (alignment.ecc is not None and alignment.ecc < ECC_POOR):
            flags.append("align_poor")
        brightness = mean_brightness(image)
        sharp = sharpness(image)
        if self.ref_brightness > 0 and abs(brightness / self.ref_brightness - 1) > 0.2:
            flags.append("brightness_changed")
        if self.ref_sharpness > 0 and \
                contrast_normalized_sharpness(image, self.quality_region) < \
                0.5 * self.ref_sharpness:
            flags.append("blurry")
        light = prepared.lighting
        if light is not None:
            if light.max_change > self.cfg.lighting.flag_change:
                flags.append("lighting_changed")
            if light.saturated_increase > 0.02:
                flags.append("saturated")
        info = {
            "align_method": alignment.method,
            "align_rms_px": _fmt(alignment.rms_px, 3),
            "align_ecc": _fmt(alignment.ecc, 4),
            "align_scale": _fmt(alignment.scale, 5),
            "align_shift_px": _fmt(alignment.shift_px, 2),
            "mean_brightness": _fmt(brightness, 2),
            "sharpness": _fmt(sharp, 2),
            "light_gain_b": light.gains[0] if light else "",
            "light_gain_g": light.gains[1] if light else "",
            "light_gain_r": light.gains[2] if light else "",
        }
        return flags, info

    def _measure(self, frame_row: dict, prepared: Prepared, mask: np.ndarray,
                 ref_mask: np.ndarray | None = None,
                 variants: list[np.ndarray] | None = None) -> list[dict]:
        """One row per plot.

        ``variants`` are narrower/wider masks of the same frame; the spread of their
        measurements gives the segmentation uncertainty (the path and region stay fixed).
        """
        exp = self.experiment
        alignment = prepared.alignment
        frame_flags, frame_info = self._frame_flags(prepared)
        # Trained models: how far this frame is from what the model was trained on.
        familiarity = (self.segmenter.input_distance(prepared.frame)
                       if hasattr(self.segmenter, "input_distance") else None)
        if familiarity is not None:
            frame_info["model_input_distance"] = _fmt(familiarity[0], 2)
            if familiarity[1]:
                frame_flags.append("unfamiliar_input")
        stem = Path(frame_row["file"]).stem
        mask_file = overlay_file = ""
        if self.cfg.save_masks:
            mask_path = self.masks_dir / f"{stem}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            mask_file = mask_path.relative_to(exp.root).as_posix()

        rows, drawn = [], []
        for plot in self.plots:
            region = self.plot_masks[plot.name]
            flags = list(frame_flags)
            path = self._measure_path(plot, mask, ref_mask, flags)
            reference = None if ref_mask is None else (ref_mask | mask)
            m = self._measure_mask(mask, path, region, reference)
            seg = _segmentation_spread(m, [self._measure_mask(v, path, region, None)
                                           for v in variants or []])
            if m.target_px == 0:
                flags.append("no_target")

            row = {
                "timestamp_utc": frame_row["timestamp_utc"],
                "frame_file": frame_row["file"],
                "camera": frame_row["camera"],
                "plot": plot.name,
                "settings_hash": self.settings_hash,
                **frame_info,
                "target_px": m.target_px,
                "front_width_px": m.front_width_px if path is not None else "",
                "extent_px": _fmt(m.extent_px, 2) if path is not None else "",
                "extent_max_px": _fmt(m.extent_max_px, 2) if path is not None else "",
                "extent_fraction": _fmt(m.extent_fraction, 5) if path is not None else "",
                "path_length_px": _fmt(m.path_length_px, 2) if path is not None else "",
                "covered_length_px": _fmt(m.covered_length_px, 1) if path is not None else "",
                "covered_length_pct": _fmt(100 * m.covered_length_px / m.path_length_px, 3)
                if path is not None and m.path_length_px else "",
                "reference_px": "" if m.reference_px is None else m.reference_px,
                "reference_covered_pct": "" if m.reference_covered_fraction is None
                else _fmt(100 * m.reference_covered_fraction, 3),
                "coverage_pct": _fmt(100 * m.coverage_fraction, 3),
                "coverage_pct_unc": "" if seg is None or not m.roi_area_px
                else _fmt(100 * seg.target_px / m.roi_area_px, 3),
                "extent_px_seg_unc": _fmt(seg.extent_px, 3)
                if seg is not None and path is not None else "",
                **{k: _fmt(v, 5) for k, v in color_indices(prepared.frame, region).items()},
                **self._class_shares(prepared.frame, region),
                "flags": ";".join(flags),
                "extent_mm": "", "extent_mm_unc": "", "extent_mm_seg_unc": "",
                "extent_max_mm": "", "axis_length_mm": "", "covered_length_mm": "",
                "reference_area_mm2": "", "target_area_mm2": "", "target_area_mm2_unc": "",
                "roi_area_mm2": "", "equivalent_radius_mm": "",
                "mask_file": mask_file, "overlay_file": "",
            }
            if self.scale is not None:
                k = self.scale.mm_per_px
                area = m.target_px * k * k
                row.update({
                    "reference_area_mm2": "" if m.reference_px is None
                    else _fmt(m.reference_px * k * k, 2),
                    "target_area_mm2": _fmt(area, 2),
                    "target_area_mm2_unc": _fmt(area_uncertainty_mm2(
                        m.target_px, k, self.scale.se_mm_per_px,
                        None if seg is None else seg.target_px), 2),
                    "roi_area_mm2": _fmt(m.roi_area_px * k * k, 2),
                    "equivalent_radius_mm": _fmt(float(np.sqrt(area / np.pi)), 3),
                })
                if path is not None:
                    row.update({
                        "extent_mm": _fmt(m.extent_px * k, 3),
                        "extent_mm_unc": _fmt(extent_uncertainty_mm(
                            m.extent_px, k, self.scale.se_mm_per_px,
                            None if alignment.rms_px is None
                            else alignment.rms_px * self.align_px_factor,
                            None if seg is None else seg.extent_px), 3),
                        "extent_mm_seg_unc": "" if seg is None else _fmt(seg.extent_px * k, 3),
                        "extent_max_mm": _fmt(m.extent_max_px * k, 3),
                        "axis_length_mm": _fmt(m.path_length_px * k, 3),
                        "covered_length_mm": _fmt(m.covered_length_px * k, 3),
                    })
            rows.append(row)
            drawn.append((plot, m, row, path))

        if self.cfg.save_overlays:
            from fungus_cv.analyze.overlay import draw_overlay

            overlay = draw_overlay(prepared.frame, mask, drawn, reference_mask=ref_mask)
            overlay_path = self.results_dir / "overlays" / self.settings_hash / f"{stem}.jpg"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
            overlay_file = overlay_path.relative_to(exp.root).as_posix()
            for row in rows:
                row["overlay_file"] = overlay_file
        return rows


@dataclass
class SegmentationSpread:
    """Standard uncertainties (px) from how the mask changes under narrower/wider settings."""

    extent_px: float
    target_px: float


def _segmentation_spread(nominal: ExtentMeasurement,
                         variants: list[ExtentMeasurement]) -> SegmentationSpread | None:
    """The nominal and variant values bound where the edge could be: treated as a
    rectangular distribution over their range, u = (max - min) / (2 sqrt 3)."""
    if not variants:
        return None
    every = [nominal, *variants]

    def spread(values):
        return (max(values) - min(values)) / (2 * math.sqrt(3))

    return SegmentationSpread(extent_px=spread([m.extent_px for m in every]),
                              target_px=spread([m.target_px for m in every]))


def _area_only(mask: np.ndarray, region: np.ndarray,
               reference: np.ndarray | None) -> ExtentMeasurement:
    """Area and coverage for a plot without a base/tip (e.g. a field plot)."""
    target = int((mask & region).sum())
    roi_area = int(region.sum())
    ref_px = ref_frac = None
    if reference is not None:
        ref = reference & region
        ref_px = int(ref.sum())
        ref_frac = float((ref & mask).sum() / ref_px) if ref_px else 0.0
    return ExtentMeasurement(target, 0.0, 0.0, 0.0, 0, roi_area,
                             target / roi_area if roi_area else 0.0,
                             reference_px=ref_px, reference_covered_fraction=ref_frac)


def analyze(experiment: Experiment, force: bool = False,
            camera: str | None = None) -> AnalysisSummary:
    return Analyzer(experiment, camera=camera).run(force=force)


def analyze_all_cameras(experiment: Experiment, force: bool = False,
                        progress=None) -> dict[str, AnalysisSummary | str]:
    """Measure every camera that has frames; a camera that fails is reported, not raised."""
    out: dict[str, AnalysisSummary | str] = {}
    for camera in cameras_with_frames(experiment):
        if progress:
            progress(camera)
        try:
            out[camera] = analyze(experiment, force=force, camera=camera)
        except AnalysisError as exc:
            log.error("camera %s: %s", camera, exc)
            out[camera] = str(exc)
    return out


def watch(experiment_root: Path, poll_seconds: float = 30.0, on_summary=None,
          camera: str | None = None) -> None:
    """Analyze new frames as they arrive (e.g. alongside `fungus capture`)."""
    while True:
        experiment = Experiment(experiment_root)  # re-read config each pass
        try:
            summary = analyze(experiment, camera=camera)
            if on_summary and summary.processed:
                on_summary(summary)
        except AnalysisError as exc:
            log.info("waiting: %s", exc)
        time.sleep(poll_seconds)
