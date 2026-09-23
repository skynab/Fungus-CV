"""The live camera preview, shared by the pages that show it.

One camera can only be read by one program at a time, so there is a single preview for the
whole window: the Camera page frames and focuses with it, the Capture page shows the same
stream while a run is being set up, and a capture takes the camera over from both.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from fungus_cv.quality import mean_brightness, sharpness

log = logging.getLogger(__name__)


class PreviewThread(QThread):
    frame = Signal(object, dict)
    failed = Signal(str)

    def __init__(self, camera_config):
        super().__init__()
        self.camera_config = camera_config
        self._running = True
        # Set when a frame is sent, cleared once the window has drawn it: frames arriving
        # in between are skipped so a slow display never builds up a backlog of images.
        self.waiting_for_display = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        from fungus_cv.capture.camera import Camera, CameraError

        cam = Camera(self.camera_config)
        try:
            cam.open()
        except CameraError as exc:
            self.failed.emit(str(exc))
            return
        try:
            last_settings, settings = 0.0, {}
            while self._running:
                try:
                    image = cam.read()
                except CameraError as exc:
                    self.failed.emit(str(exc))
                    return
                if time.monotonic() - last_settings > 1.0:
                    settings = cam.settings()
                    last_settings = time.monotonic()
                stats = {"brightness": mean_brightness(image), "sharpness": sharpness(image),
                         **{k: settings.get(k) for k in ("exposure", "white_balance",
                                                          "focus", "gain")},
                         "warnings": list(cam.warnings)}
                if not self.waiting_for_display:
                    self.waiting_for_display = True
                    self.frame.emit(image, stats)
                self.msleep(30)
        finally:
            cam.close()


class CameraPreview(QObject):
    """Starts and stops the one preview stream and hands its frames to whoever is showing it.

    ``frame`` carries the image and its statistics; ``running_changed`` lets every page show
    the right button and message even when another page started or stopped the preview.
    """

    frame = Signal(object, dict)
    failed = Signal(str)
    running_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread: PreviewThread | None = None
        self.last_frame: np.ndarray | None = None
        self.last_stats: dict = {}

    @property
    def running(self) -> bool:
        return self.thread is not None

    def start(self, camera_config) -> None:
        self.stop()
        self.thread = PreviewThread(camera_config)
        self.thread.frame.connect(self._frame)
        self.thread.failed.connect(self._failed)
        self.thread.finished.connect(self._finished)
        self.thread.start()
        self.running_changed.emit(True)

    def stop(self) -> None:
        thread, self.thread = self.thread, None
        if thread is None:
            return
        thread.stop()
        if not thread.wait(3000):
            log.warning("the camera preview did not stop in time")
        self.running_changed.emit(False)  # last_frame is kept, to save as a snapshot

    def _frame(self, image: np.ndarray, stats: dict) -> None:
        self.last_frame, self.last_stats = image, stats
        self.frame.emit(image, stats)
        if self.thread is not None:  # every page has drawn it; the next frame may come
            self.thread.waiting_for_display = False

    def _failed(self, message: str) -> None:
        self.stop()
        self.failed.emit(message)

    def _finished(self) -> None:
        """The thread ended on its own (the camera went away)."""
        if self.thread is not None and not self.thread.isRunning():
            self.thread = None
            self.running_changed.emit(False)


def camera_config(state, index: int | None = None, use_experiment: bool = True):
    """The settings to preview with: the experiment's camera, or plain automatic settings.

    ``index`` overrides which device is opened (the Camera page's picker).
    """
    from fungus_cv.config import CameraConfig

    exp = state.experiment
    if use_experiment and exp is not None:
        cameras = exp.config.cameras
        chosen = next((c for c in cameras if c.name == state.camera), cameras[0])
        cfg = chosen.model_copy()
        if index is not None:
            cfg.index = int(index)
        return cfg
    return CameraConfig(index=int(index or 0), exposure="auto", white_balance="auto",
                        focus="auto", warmup_frames=3, settle_seconds=0)
