"""Interval capture on a fixed, clock-aligned schedule."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from fungus_cv.capture.camera import Camera, CameraError
from fungus_cv.config import CameraConfig
from fungus_cv.quality import mean_brightness, sharpness
from fungus_cv.storage import Experiment, FrameRecord, encode_image, iso_utc, sha256_bytes

log = logging.getLogger(__name__)


class LowDiskSpace(RuntimeError):
    pass


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass
class CaptureSummary:
    rounds: int = 0
    saved: int = 0
    failed: int = 0
    skipped_slots: int = 0
    stopped_reason: str = ""
    files: list[str] = field(default_factory=list)


CameraFactory = Callable[[CameraConfig], Camera]


class CaptureSession:
    """Takes one picture per camera at ``start + k * interval``.

    Slots are computed from a fixed start time rather than "last shot + interval", so small
    delays never accumulate. If the computer was asleep or busy and slots were missed, they
    are logged and skipped rather than taken late in a burst.
    """

    def __init__(
        self,
        experiment: Experiment,
        clock: SystemClock | None = None,
        camera_factory: CameraFactory = Camera,
    ):
        self.experiment = experiment
        self.config = experiment.config
        self.clock = clock or SystemClock()
        self.cameras = {c.name: camera_factory(c) for c in self.config.cameras}
        self.summary = CaptureSummary()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> CaptureSummary:
        cap = self.config.capture
        interval = timedelta(seconds=cap.interval)
        now = self.clock.now()
        start = cap.start_at or now
        end = start + timedelta(seconds=cap.duration) if cap.duration else None
        slot_index = max(0, math.ceil((now - start) / interval))

        log.info(
            "capture started: %d camera(s), every %ss, start %s%s",
            len(self.cameras), cap.interval, iso_utc(start),
            f", end {iso_utc(end)}" if end else "",
        )
        try:
            while not self._stop.is_set():
                slot = start + slot_index * interval
                if end is not None and slot > end:
                    self.summary.stopped_reason = "duration reached"
                    break
                if not self._wait_until(slot):
                    self.summary.stopped_reason = "stopped"
                    break

                late = self.clock.now() - slot
                if late >= interval:
                    missed = int(late / interval)
                    self.summary.skipped_slots += missed
                    log.warning(
                        "running %.0fs behind schedule (computer asleep or busy?); "
                        "skipping %d slot(s)", late.total_seconds(), missed,
                    )
                    slot_index += missed
                    continue

                self.capture_round(slot)
                slot_index += 1
                if cap.max_frames and self.summary.rounds >= cap.max_frames:
                    self.summary.stopped_reason = "max_frames reached"
                    break
        except LowDiskSpace as exc:
            log.error("%s", exc)
            self.summary.stopped_reason = "low disk space"
        finally:
            self.close()
        log.info(
            "capture finished (%s): %d saved, %d failed, %d slot(s) skipped",
            self.summary.stopped_reason or "interrupted",
            self.summary.saved, self.summary.failed, self.summary.skipped_slots,
        )
        return self.summary

    def capture_round(self, scheduled: datetime | None = None) -> list[FrameRecord]:
        """Capture one image from every camera, recording successes and failures."""
        records = [self._capture_camera(cam, scheduled) for cam in self.cameras.values()]
        self.summary.rounds += 1
        return records

    def close(self) -> None:
        for cam in self.cameras.values():
            cam.close()

    # --- internals -----------------------------------------------------------------

    def _wait_until(self, when: datetime) -> bool:
        """Sleep in short steps so stop requests and Ctrl+C are handled promptly."""
        while not self._stop.is_set():
            remaining = (when - self.clock.now()).total_seconds()
            if remaining <= 0:
                return True
            self.clock.sleep(min(remaining, 0.5))
        return False

    def _check_disk(self) -> None:
        free = self.experiment.free_disk_mb()
        if free < self.config.capture.min_free_disk_mb:
            raise LowDiskSpace(
                f"only {free:.0f} MB free (minimum {self.config.capture.min_free_disk_mb} MB)"
            )

    def _capture_camera(self, cam: Camera, scheduled: datetime | None) -> FrameRecord:
        cap = self.config.capture
        name = cam.config.name
        self._check_disk()

        error = ""
        for attempt in range(cap.retries + 1):
            if attempt:
                self.clock.sleep(cap.retry_delay)
            try:
                cam.open()
                image = cam.read()
                taken = self.clock.now()
                settings = cam.settings()
                break
            except CameraError as exc:
                error = str(exc)
                log.warning("[%s] attempt %d failed: %s", name, attempt + 1, exc)
                cam.close()  # force a clean reopen, e.g. after a USB disconnect
        else:
            record = FrameRecord(
                timestamp_utc=iso_utc(self.clock.now()),
                camera=name,
                status="failed",
                scheduled_utc=iso_utc(scheduled) if scheduled else "",
                notes=error,
            )
            self.experiment.append_frame(record)
            self.summary.failed += 1
            log.error("[%s] giving up on this slot: %s", name, error)
            return record

        if not cap.camera_stays_open:
            cam.close()

        data = encode_image(image, cap.image_format, cap.jpeg_quality)
        path = self.experiment.save_bytes(data, taken, name, cap.image_format)
        record = FrameRecord(
            timestamp_utc=iso_utc(taken),
            camera=name,
            status="ok",
            file=self.experiment.relative(path),
            sha256=sha256_bytes(data),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            source="capture",
            scheduled_utc=iso_utc(scheduled) if scheduled else "",
            lag_s=round((taken - scheduled).total_seconds(), 3) if scheduled else None,
            mean_brightness=round(mean_brightness(image), 2),
            sharpness=round(sharpness(image), 2),
            camera_settings=settings,
            notes="; ".join(cam.warnings),
        )
        self.experiment.append_frame(record)
        self.summary.saved += 1
        self.summary.files.append(record.file)
        log.info(
            "[%s] saved %s (brightness %.0f, sharpness %.0f)",
            name, record.file, record.mean_brightness, record.sharpness,
        )
        return record
