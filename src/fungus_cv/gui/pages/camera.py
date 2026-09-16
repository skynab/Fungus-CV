"""Live camera preview for framing, focus and checking the camera settings."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.gui.image_view import ImageView
from fungus_cv.gui.qt_util import run_task
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


class CameraPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.thread: PreviewThread | None = None
        self.last_frame: np.ndarray | None = None

        self.device = QComboBox()
        self.device.setMinimumWidth(320)
        self.refresh_btn = QPushButton("Find cameras")
        self.refresh_btn.clicked.connect(self.find_cameras)
        self.use_experiment = QCheckBox("Use the experiment's camera settings")
        self.use_experiment.setChecked(True)
        self.start_btn = QPushButton("Start preview")
        self.start_btn.clicked.connect(self.toggle_preview)
        self.snapshot_btn = QPushButton("Save snapshot")
        self.snapshot_btn.clicked.connect(self.save_snapshot)
        self.grid = QCheckBox("Grid")
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.stats = QLabel("—")
        self.view = ImageView()

        top = QHBoxLayout()
        top.addWidget(QLabel("Camera:"))
        top.addWidget(self.device)
        top.addWidget(self.refresh_btn)
        top.addWidget(self.start_btn)
        top.addWidget(self.snapshot_btn)
        top.addWidget(self.grid)
        top.addStretch()
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.use_experiment)
        layout.addWidget(self.message)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.stats)
        self._found_once = False
        state.busy_changed.connect(self._busy)

    def on_shown(self) -> None:
        if not self._found_once:
            self._found_once = True
            self.find_cameras()

    def _busy(self, capturing: bool) -> None:
        if capturing:
            self.stop_preview()
            self._say("Preview is paused while a capture is running (they would share the "
                      "camera).", warn=False)
        self.start_btn.setEnabled(not capturing)
        self.refresh_btn.setEnabled(not capturing)

    def _say(self, text: str, warn: bool = True) -> None:
        color = "#b00020" if warn else "palette(text)"
        self.message.setText(f"<span style='color:{color}'>{text}</span>" if text else "")

    # --- devices -----------------------------------------------------------------------

    def find_cameras(self) -> None:
        self.stop_preview()  # a camera in use by the preview can't be probed
        self.refresh_btn.setEnabled(False)
        self._say("Checking camera permission…", warn=False)

        def work(progress, should_stop):
            from fungus_cv.capture.camera import list_cameras
            from fungus_cv.capture.permissions import camera_access, camera_names

            access = camera_access(request=True, timeout=120)
            if not access.ok:
                return access, [], camera_names()
            return access, list_cameras(max_index=6), camera_names()

        run_task(work, self._cameras_found, self._failed, pool="io")

    def _cameras_found(self, result) -> None:
        access, found, names = result
        self.refresh_btn.setEnabled(True)
        self.device.clear()
        if not access.ok:
            self._say(access.advice())
            return
        for cam in found:
            label = f"{cam['index']}: {cam['width']}×{cam['height']}"
            if isinstance(names, dict) and cam["index"] in names:
                label += f" — {names[cam['index']]}"
            elif isinstance(names, list) and len(names) == len(found):
                label += f" — {names[found.index(cam)]}"
            self.device.addItem(label, cam["index"])
        if not found:
            from fungus_cv.capture.diagnostics import no_camera_hint

            self._say(f"No cameras found. {no_camera_hint()}")
        else:
            self._say("")
            exp = self.state.experiment
            if exp is not None:
                i = self.device.findData(exp.config.cameras[0].index)
                if i >= 0:
                    self.device.setCurrentIndex(i)

    def _failed(self, message: str) -> None:
        self.refresh_btn.setEnabled(True)
        self._say(message)

    # --- preview -----------------------------------------------------------------------

    def _camera_config(self):
        from fungus_cv.config import CameraConfig

        index = self.device.currentData()
        exp = self.state.experiment
        if self.use_experiment.isChecked() and exp is not None:
            cfg = exp.config.cameras[0].model_copy()
            if index is not None:
                cfg.index = int(index)
            return cfg
        return CameraConfig(index=int(index or 0), exposure="auto", white_balance="auto",
                            focus="auto", warmup_frames=3, settle_seconds=0)

    def toggle_preview(self) -> None:
        if self.thread is not None:
            self.stop_preview()
            return
        if self.device.count() == 0:
            self.find_cameras()
            return
        self.thread = PreviewThread(self._camera_config())
        self.thread.frame.connect(self._show_frame)
        self.thread.failed.connect(self._preview_failed)
        self.thread.finished.connect(self._preview_finished)
        self.thread.start()
        self.start_btn.setText("Stop preview")
        self._say("Opening camera…", warn=False)

    def stop_preview(self) -> None:
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait(3000)
            self.thread = None
        self.start_btn.setText("Start preview")

    def _preview_finished(self) -> None:
        if self.thread is not None and not self.thread.isRunning():
            self.thread = None
            self.start_btn.setText("Start preview")

    def _preview_failed(self, message: str) -> None:
        self._say(f"{message}. Is another app using the camera? Try another camera, or see "
                  "Diagnostics.")

    def _show_frame(self, image: np.ndarray, stats: dict) -> None:
        if self.thread is not None:
            self.thread.waiting_for_display = False
        if self.message.text().startswith("<span style='color:palette(text)'>Opening"):
            self._say("")
        self.last_frame = image
        shown = image
        if self.grid.isChecked():
            shown = image.copy()
            h, w = shown.shape[:2]
            for i in (1, 2):
                cv2.line(shown, (w * i // 3, 0), (w * i // 3, h), (0, 255, 255), 1)
                cv2.line(shown, (0, h * i // 3), (w, h * i // 3), (0, 255, 255), 1)
        self.view.set_image(shown, keep_view=True)
        h, w = image.shape[:2]
        text = (f"{w}×{h}   brightness {stats['brightness']:.0f}   sharpness "
                f"{stats['sharpness']:.0f}   exposure {stats['exposure']}   white balance "
                f"{stats['white_balance']}   focus {stats['focus']}")
        if stats["warnings"]:
            text += "   ⚠ " + "; ".join(stats["warnings"])
        self.stats.setText(text)

    def save_snapshot(self) -> None:
        if self.last_frame is None:
            return
        exp = self.state.experiment
        folder = exp.root if exp is not None else Path.home()
        path = folder / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"
        cv2.imwrite(str(path), self.last_frame)
        self._say(f"Saved {path}", warn=False)

    def shutdown(self) -> None:
        self.stop_preview()
