"""Live camera preview for framing, focus and checking the camera settings."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.gui import theme
from fungus_cv.gui.image_view import ImageView, ViewControls
from fungus_cv.gui.preview import camera_config
from fungus_cv.gui.qt_util import run_task

log = logging.getLogger(__name__)


class CameraPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.preview = state.preview  # shared with the Capture page

        self.device = QComboBox()
        self.device.setMinimumWidth(320)
        self.refresh_btn = QPushButton("Find cameras")
        self.refresh_btn.clicked.connect(self.find_cameras)
        self.use_experiment = QCheckBox("Use the experiment's camera settings")
        self.use_experiment.setChecked(True)
        self.start_btn = QPushButton("Start preview")
        theme.mark_primary(self.start_btn)
        self.start_btn.clicked.connect(self.toggle_preview)
        self.snapshot_btn = QPushButton("Save snapshot")
        self.snapshot_btn.clicked.connect(self.save_snapshot)
        self.grid = QCheckBox("Grid")
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.stats = QLabel("—")
        self.view = ImageView()
        self.view.setFixedHeight(190)
        # Framing is mostly scrolling and fitting, so start in Pan: the wheel then moves the
        # picture (or the page, when it all fits) instead of zooming under the pointer.
        self.view.set_nav_mode(ImageView.PAN)
        self.controls = ViewControls(self.view)

        top = QHBoxLayout()
        top.addWidget(QLabel("Camera:"))
        top.addWidget(self.device)
        top.addWidget(self.refresh_btn)
        top.addWidget(self.start_btn)
        top.addWidget(self.snapshot_btn)
        top.addStretch()
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View:"))
        view_row.addWidget(self.controls)
        view_row.addWidget(self.grid)
        view_row.addStretch()
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.use_experiment)
        layout.addWidget(self.message)
        layout.addLayout(view_row)
        layout.addWidget(self.view)
        layout.addWidget(self.stats)
        layout.addStretch(1)
        self._found_once = False
        state.busy_changed.connect(self._busy)
        self.preview.frame.connect(self._show_frame)
        self.preview.failed.connect(self._preview_failed)
        self.preview.running_changed.connect(self._running_changed)

    def on_shown(self) -> None:
        # Probing the cameras needs them free, so never interrupt a running preview for it.
        if not self._found_once and not self.preview.running:
            self._found_once = True
            self.find_cameras()
        if self.preview.last_frame is not None:  # started on another page
            self._show_frame(self.preview.last_frame, self.preview.last_stats)

    def _busy(self, capturing: bool) -> None:
        if capturing:  # the capture stopped the preview: they would share one camera
            self._say("Preview is paused while a capture is running (they would share the "
                      "camera).", warn=False)
        self.start_btn.setEnabled(not capturing)
        self.refresh_btn.setEnabled(not capturing)

    def _say(self, text: str, warn: bool = True) -> None:
        color = theme.BAD if warn else theme.TEXT
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
        return camera_config(self.state, index=self.device.currentData(),
                             use_experiment=self.use_experiment.isChecked())

    def toggle_preview(self) -> None:
        if self.preview.running:
            self.stop_preview()
            return
        if self.device.count() == 0:
            self.find_cameras()
            return
        self.preview.start(self._camera_config())
        self._say("Opening camera…", warn=False)

    def stop_preview(self) -> None:
        self.preview.stop()

    def _running_changed(self, running: bool) -> None:
        self.start_btn.setText("Stop preview" if running else "Start preview")

    def _preview_failed(self, message: str) -> None:
        self._say(f"{message}. Is another app using the camera? Try another camera, or see "
                  "Diagnostics.")

    def _show_frame(self, image: np.ndarray, stats: dict) -> None:
        if not self.isVisible():
            return  # another page is showing the preview; on_shown catches this one up
        if self.message.text().startswith(f"<span style='color:{theme.TEXT}'>Opening"):
            self._say("")
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
        if self.preview.last_frame is None:
            return
        exp = self.state.experiment
        folder = exp.root if exp is not None else Path.home()
        path = folder / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"
        cv2.imwrite(str(path), self.preview.last_frame)
        self._say(f"Saved {path}", warn=False)

    def shutdown(self) -> None:
        self.stop_preview()  # the window stops it too, in case this page was never built
