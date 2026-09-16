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
from fungus_cv.preprocess.align import ECC_POOR, align_frame
from fungus_cv.preprocess.markers import Scale, detect_markers, scale_from_markers
from fungus_cv.quality import mean_brightness, sharpness
from fungus_cv.segment import build_segmenter
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
    "mean_brightness", "sharpness", "flags", "mask_file", "overlay_file",
]


class AnalysisError(RuntimeError):
    pass


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
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.cfg = experiment.config.analysis
        self.results_dir = experiment.root / RESULTS_DIR
        self.measurements_path = self.results_dir / MEASUREMENTS_NAME

        frames = frames_for_analysis(experiment)
        self.reference_row = frames[0]
        self.reference = read_image(experiment, self.reference_row["file"])

        ann_path = annotations_path(experiment)
        if not ann_path.exists():
            raise AnalysisError(
                f"{ann_path.name} not found; run `fungus annotate {experiment.root}` first"
            )
        self.annotations = Annotations.load(ann_path)
        h, w = self.reference.shape[:2]
        if tuple(self.annotations.image_size) != (w, h):
            raise AnalysisError(
                f"annotations were drawn on a {self.annotations.image_size} image but the "
                f"reference frame is {(w, h)}; run `fungus annotate` again"
            )
        if self.annotations.reference_file and \
                self.annotations.reference_file != self.reference_row["file"]:
            log.warning(
                "annotations were drawn on %s but the reference frame is now %s",
                self.annotations.reference_file, self.reference_row["file"],
            )

        self.ref_markers = detect_markers(self.reference, self.cfg.markers.dictionary)
        self.scale: Scale | None = None
        if self.cfg.markers.size_mm is None:
            log.warning("analysis.markers.size_mm is not set; results will be in pixels only")
        elif not self.ref_markers:
            log.warning("no ArUco markers found in the reference frame; results in pixels only")
        else:
            self.scale = scale_from_markers(self.ref_markers, self.cfg.markers.size_mm)
            log.info(
                "scale %.5f mm/px (sd %.5f, %d edges, markers %s)",
                self.scale.mm_per_px, self.scale.std_mm_per_px, self.scale.n_edges,
                self.scale.marker_ids,
            )
            if self.scale.std_mm_per_px / self.scale.mm_per_px > 0.02:
                log.warning(
                    "marker edges disagree by more than 2%%; the camera may not be square-on "
                    "to the markers, which makes mm results less accurate"
                )

        # Intensity alignment ignores the measured region (plus a margin), where things change.
        roi = self.annotations.roi_mask(self.reference.shape).astype(np.uint8)
        margin = max(15, int(0.02 * max(w, h)))
        self.align_exclude = cv2.dilate(roi, np.ones((margin, margin), np.uint8)).astype(bool)

        self.segmenter = build_segmenter(self.cfg.target)
        self.ref_brightness = mean_brightness(self.reference)
        self.ref_sharpness = sharpness(self.reference)
        self.settings = {
            "fungus_cv_version": __version__,
            "reference_file": self.reference_row["file"],
            "analysis": self.cfg.model_dump(mode="json"),
            "segmenter": self.segmenter.describe(),
            "annotations": json.loads(ann_path.read_text(encoding="utf-8")),
        }
        blob = json.dumps(self.settings, sort_keys=True).encode()
        self.settings_hash = hashlib.sha256(blob).hexdigest()[:12]
        self.frames = frames

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
            "settings": self.settings,
        }
        (self.results_dir / RUN_INFO_NAME).write_text(json.dumps(info, indent=2), "utf-8")

    # --- processing ----------------------------------------------------------------

    def run(self, force: bool = False) -> AnalysisSummary:
        self.results_dir.mkdir(exist_ok=True)
        if force and self.measurements_path.exists():
            self.measurements_path.unlink()
        done = self._existing()
        self._write_run_info()
        summary = AnalysisSummary(settings_hash=self.settings_hash)
        for frame_row in self.frames:
            if frame_row["file"] in done:
                summary.skipped_existing += 1
                continue
            try:
                row = self.process(frame_row)
            except AnalysisError as exc:
                log.error("%s: %s", frame_row["file"], exc)
                summary.failed += 1
                continue
            self._append(row)
            summary.processed += 1
            for flag in filter(None, row["flags"].split(";")):
                summary.flagged[flag] += 1
        return summary

    def process(self, frame_row: dict) -> dict:
        exp = self.experiment
        image = read_image(exp, frame_row["file"])
        if image.shape != self.reference.shape:
            raise AnalysisError(f"size {image.shape} differs from reference {self.reference.shape}")

        aligned, alignment = align_frame(
            image, self.reference, self.ref_markers, self.cfg.align,
            self.cfg.markers.dictionary, exclude=self.align_exclude,
        )
        mask = self.segmenter.segment(aligned)
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
            "flags": ";".join(flags),
            "extent_mm": "", "extent_mm_unc": "", "extent_max_mm": "", "axis_length_mm": "",
            "target_area_mm2": "", "roi_area_mm2": "", "mask_file": "", "overlay_file": "",
        }
        if self.scale is not None:
            k = self.scale.mm_per_px
            row.update({
                "extent_mm": _fmt(m.extent_px * k, 3),
                "extent_mm_unc": _fmt(extent_uncertainty_mm(
                    m.extent_px, k, self.scale.se_mm_per_px, alignment.rms_px), 3),
                "extent_max_mm": _fmt(m.extent_max_px * k, 3),
                "axis_length_mm": _fmt(self.annotations.axis_length_px * k, 3),
                "target_area_mm2": _fmt(m.target_px * k * k, 2),
                "roi_area_mm2": _fmt(m.roi_area_px * k * k, 2),
            })

        stem = Path(frame_row["file"]).stem
        if self.cfg.save_masks:
            mask_path = self.results_dir / "masks" / f"{stem}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            row["mask_file"] = mask_path.relative_to(exp.root).as_posix()
        if self.cfg.save_overlays:
            from fungus_cv.analyze.overlay import draw_overlay

            overlay = draw_overlay(aligned, mask, self.annotations, m, row)
            overlay_path = self.results_dir / "overlays" / f"{stem}.jpg"
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
