"""Open or create an experiment, see its status and edit the most used settings."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.config import ExperimentConfig, parse_duration, set_yaml_value
from fungus_cv.gui import theme


def _combo(options: list[str]) -> QComboBox:
    box = QComboBox()
    box.addItems(options)
    return box


def _set_combo(box: QComboBox, value) -> None:
    text = str(value)
    if box.findText(text) < 0:
        box.addItem(text)
    box.setCurrentText(text)


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return ""
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size and seconds % size == 0:
            return f"{int(seconds // size)}{unit}"
    return f"{seconds:g}s"


class ExperimentPage(QWidget):
    def __init__(self, state, window):
        super().__init__()
        self.state = state
        self.window = window

        # --- no experiment: open / create / recent ------------------------------------------
        self.start_box = QGroupBox("Start")
        new_btn = QPushButton("New experiment…")
        open_btn = QPushButton("Open experiment…")
        new_btn.clicked.connect(window.new_experiment)
        open_btn.clicked.connect(lambda: window.open_experiment())
        self.recent = QListWidget()
        self.recent.setMaximumHeight(130)
        self.recent.itemDoubleClicked.connect(
            lambda item: window.open_experiment(item.data(Qt.UserRole)))
        start_layout = QVBoxLayout(self.start_box)
        buttons = QHBoxLayout()
        buttons.addWidget(new_btn)
        buttons.addWidget(open_btn)
        buttons.addStretch()
        start_layout.addLayout(buttons)
        start_layout.addWidget(QLabel("Recent experiments (double-click to open):"))
        start_layout.addWidget(self.recent)

        # --- summary ------------------------------------------------------------------------
        self.summary_box = QGroupBox("Status")
        self.summary = QLabel()
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary.setWordWrap(True)
        self.summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        folder_btn = QPushButton("Show folder")
        folder_btn.clicked.connect(window.reveal_experiment)
        config_btn = QPushButton("Open config.yaml in editor")
        config_btn.clicked.connect(self._open_config)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._reload)
        summary_layout = QVBoxLayout(self.summary_box)
        summary_layout.addWidget(self.summary, 1)
        row = QHBoxLayout()
        for b in (folder_btn, config_btn, reload_btn):
            row.addWidget(b)
        row.addStretch()
        summary_layout.addLayout(row)

        # --- settings form ------------------------------------------------------------------
        self.settings_box = QGroupBox("Settings (saved to config.yaml; comments are kept)")
        form = QFormLayout(self.settings_box)
        self.f_camera_index = QSpinBox()
        self.f_camera_index.setRange(0, 16)
        self.f_resolution = _combo(["1920x1080", "1280x720", "3840x2160", "640x480"])
        self.f_resolution.setEditable(True)
        self.f_exposure = _combo(["lock", "auto"])
        self.f_exposure.setEditable(True)
        self.f_white_balance = _combo(["lock", "auto"])
        self.f_white_balance.setEditable(True)
        self.f_focus = _combo(["lock", "auto"])
        self.f_focus.setEditable(True)
        self.f_interval = QLineEdit()
        self.f_interval.setPlaceholderText("e.g. 30s, 10m, 2h")
        self.f_duration = QLineEdit()
        self.f_duration.setPlaceholderText("empty = until stopped; e.g. 3h, 14d")
        self.f_format = _combo(["png", "jpg", "tiff"])
        self.f_keep_awake = QCheckBox("Keep the computer awake while capturing")
        self.f_marker_mm = QDoubleSpinBox()
        self.f_marker_mm.setRange(0, 1000)
        self.f_marker_mm.setDecimals(2)
        self.f_marker_mm.setSpecialValueText("not set (results in pixels)")
        self.f_marker_mm.setSuffix(" mm")
        self.f_rectify = QCheckBox("Correct perspective using the markers (top-down view)")
        self.f_lighting = _combo(["none", "patch", "background"])
        self.f_target = _combo(["color", "sam2", "model"])
        self.f_reference = _combo(["none", "color", "sam2", "model"])
        self.f_mode = _combo(["axis", "path"])
        self.f_path_source = _combo(["annotation", "reference"])
        self.f_model_path = QLineEdit()
        self.f_model_path.setPlaceholderText("folder from `fungus train` (for target: model)")

        form.addRow(QLabel("<b>Camera</b>"))
        form.addRow("Camera index", self.f_camera_index)
        form.addRow("Resolution", self.f_resolution)
        form.addRow("Exposure", self.f_exposure)
        form.addRow("White balance", self.f_white_balance)
        form.addRow("Focus", self.f_focus)
        form.addRow(QLabel("<b>Capture</b>"))
        form.addRow("Interval", self.f_interval)
        form.addRow("Duration", self.f_duration)
        form.addRow("Image format", self.f_format)
        form.addRow("", self.f_keep_awake)
        form.addRow(QLabel("<b>Analysis</b>"))
        form.addRow("Marker size", self.f_marker_mm)
        form.addRow("", self.f_rectify)
        form.addRow("Lighting correction", self.f_lighting)
        form.addRow("Find target with", self.f_target)
        form.addRow("Trained model folder", self.f_model_path)
        form.addRow("Reference object (stem)", self.f_reference)
        form.addRow("Measure along", self.f_mode)
        form.addRow("Path from", self.f_path_source)
        self.save_btn = QPushButton("Save settings")
        theme.mark_primary(self.save_btn)
        self.save_btn.clicked.connect(self.save_settings)
        form.addRow("", self.save_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.start_box)
        top = QHBoxLayout()
        top.addWidget(self.summary_box, 1)
        top.addWidget(self.settings_box, 1)
        layout.addLayout(top, 1)

        state.experiment_changed.connect(lambda _: self.refresh())
        state.config_changed.connect(self.refresh)
        state.busy_changed.connect(lambda busy: self.save_btn.setEnabled(not busy))
        self.refresh()

    # --- display ---------------------------------------------------------------------------

    def on_shown(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        self.recent.clear()
        for path in self.state.recent():
            self.recent.addItem(str(path))
            self.recent.item(self.recent.count() - 1).setData(Qt.UserRole, str(path))
        exp = self.state.experiment
        self.summary_box.setVisible(exp is not None)
        self.settings_box.setVisible(exp is not None)
        if exp is None:
            return
        cfg = exp.config
        rows = exp.read_frames()
        ok = [r for r in rows if r["status"] == "ok"]
        cams = sorted({r["camera"] for r in ok})
        lines = [f"<b>{cfg.name}</b>", f"Folder: {exp.root}",
                 f"Frames: {len(ok)} saved, {len(rows) - len(ok)} failed"
                 + (f" (cameras: {', '.join(cams)})" if cams else "")]
        if ok:
            lines.append(f"First: {ok[0]['timestamp_utc']} &nbsp; Last: {ok[-1]['timestamp_utc']}")
        ann = exp.root / "annotations.json"
        lines.append("Measurement set up: " + ("yes" if ann.exists() else
                                               "not yet (use <i>Set Up Measurement</i>)"))
        results = exp.root / "results" / "measurements.csv"
        lines.append("Analysis results: " + ("yes" if results.exists() else "none yet"))
        self.summary.setText("<br>".join(lines))

        cam = cfg.cameras[0]
        self.f_camera_index.setValue(cam.index)
        _set_combo(self.f_resolution, f"{cam.width}x{cam.height}")
        _set_combo(self.f_exposure, cam.exposure)
        _set_combo(self.f_white_balance, cam.white_balance)
        _set_combo(self.f_focus, cam.focus)
        self.f_interval.setText(_fmt_seconds(cfg.capture.interval))
        self.f_duration.setText(_fmt_seconds(cfg.capture.duration))
        _set_combo(self.f_format, cfg.capture.image_format)
        self.f_keep_awake.setChecked(cfg.capture.keep_awake)
        a = cfg.analysis
        self.f_marker_mm.setValue(a.markers.size_mm or 0)
        self.f_rectify.setChecked(a.rectify.enabled)
        _set_combo(self.f_lighting, a.lighting.method)
        _set_combo(self.f_target, a.target.method)
        _set_combo(self.f_reference, a.reference.method)
        _set_combo(self.f_mode, a.measure.mode)
        _set_combo(self.f_path_source, a.measure.path_source)
        self.f_model_path.setText(a.target.model.path)

    # --- saving ----------------------------------------------------------------------------

    def _control(self, text: str):
        text = text.strip()
        try:
            return float(text)
        except ValueError:
            return text

    def settings_updates(self) -> dict:
        width, _, height = self.f_resolution.currentText().lower().partition("x")
        duration = self.f_duration.text().strip()
        updates = {
            "capture.interval": self.f_interval.text().strip(),
            "capture.duration": duration or None,
            "capture.image_format": self.f_format.currentText(),
            "capture.keep_awake": self.f_keep_awake.isChecked(),
            "analysis.markers.size_mm": self.f_marker_mm.value() or None,
            "analysis.rectify.enabled": self.f_rectify.isChecked(),
            "analysis.lighting.method": self.f_lighting.currentText(),
            "analysis.target.method": self.f_target.currentText(),
            "analysis.target.model.path": self.f_model_path.text().strip(),
            "analysis.reference.method": self.f_reference.currentText(),
            "analysis.measure.mode": self.f_mode.currentText(),
            "analysis.measure.path_source": self.f_path_source.currentText(),
        }
        camera = {
            "index": self.f_camera_index.value(),
            "width": int(width), "height": int(height),
            "exposure": self._control(self.f_exposure.currentText()),
            "white_balance": self._control(self.f_white_balance.currentText()),
            "focus": self._control(self.f_focus.currentText()),
        }
        return {"updates": updates, "camera": camera}

    def save_settings(self) -> None:
        exp = self.state.experiment
        if exp is None:
            return
        try:
            parse_duration(self.f_interval.text().strip())
            if self.f_duration.text().strip():
                parse_duration(self.f_duration.text().strip())
            values = self.settings_updates()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid setting", str(exc))
            return
        text = exp.config_path.read_text(encoding="utf-8")
        try:
            for key, value in values["updates"].items():
                text = set_yaml_value(text, key, value)
            text = _set_first_camera(text, values["camera"])
            import yaml

            ExperimentConfig.model_validate(yaml.safe_load(text))
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save settings",
                                f"{exc}\n\nNothing was changed. You can edit config.yaml "
                                "directly instead.")
            return
        exp.config_path.write_text(text, encoding="utf-8")
        self.state.reload()
        self.window.statusBar().showMessage("Settings saved", 4000)

    def _open_config(self) -> None:
        if self.state.experiment is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.state.experiment.config_path)))

    def _reload(self) -> None:
        try:
            self.state.reload()
        except ValueError as exc:
            QMessageBox.warning(self, "config.yaml has an error", str(exc))


def _set_first_camera(text: str, camera: dict) -> str:
    """The camera list is a YAML sequence; edit the first entry's lines in place."""
    import re

    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if re.match(r"^cameras\s*:", line)), None)
    if start is None:
        raise KeyError("cameras")
    first_item = next(i for i in range(start + 1, len(lines))
                      if lines[i].lstrip().startswith("- "))
    item_indent = len(lines[first_item]) - len(lines[first_item].lstrip())
    field_indent = item_indent + 2
    for key, value in camera.items():
        pattern = re.compile(rf"^(\s*(?:- )?){re.escape(key)}\s*:")
        for i in range(first_item, len(lines)):
            line = lines[i]
            indent = len(line) - len(line.lstrip())
            if i > first_item and line.strip() and not line.lstrip().startswith("#") and (
                    indent < field_indent or line.lstrip().startswith("- ")):
                break
            if pattern.match(line):
                rest = re.match(r"^(\s*(?:- )?[^:#]+:[ \t]*)(.*?)([ \t]+#.*)?$",
                                line.rstrip("\n"))
                value_text = ("null" if value is None else
                              repr(value) if isinstance(value, (int, float)) else str(value))
                new = rest.group(1) + value_text
                if rest.group(3):
                    new = new.ljust(len(rest.group(1)) + len(rest.group(2))) + rest.group(3)
                lines[i] = new + "\n"
                break
        else:
            raise KeyError(f"cameras[0].{key}")
    return "".join(lines)


__all__ = ["ExperimentPage", "Path"]
