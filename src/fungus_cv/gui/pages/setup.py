"""Set up the measurement: click the base/path/region or field plots, and pick colours."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.gui.image_view import ImageView
from fungus_cv.gui.qt_util import run_task

TOOLS = [
    ("base", "Base", "Click where growth starts (waterline, soil line)."),
    ("path", "Path to tip", "Click to the tip. For a curved stem, click points along it, "
                            "ending at the tip."),
    ("roi", "Region", "Click the corners of the region to measure."),
    ("plot", "Field plot", "Click a plot's corners, then press Enter (or 'Finish plot')."),
    ("patch", "Neutral patch", "Optional: click the corners of a white/grey card that stays "
                               "in view (for lighting correction)."),
    ("color", "Pick colour", "Drag boxes over the target colour; the pink overlay shows what "
                             "is selected."),
]
COLORS = {"base": (0, 220, 0), "path": (0, 220, 0), "roi": (255, 220, 0),
          "plot": (0, 200, 255), "patch": (255, 160, 0)}


class SetupPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.analyzer = None
        self.frame: np.ndarray | None = None
        self.points: dict[str, list[tuple[float, float]]] = {k: [] for k in
                                                             ("base", "path", "roi", "patch")}
        self.plots: list[tuple[str, list[tuple[float, float]]]] = []
        self.current_plot: list[tuple[float, float]] = []
        self.color_boxes: list[tuple[int, int, int, int]] = []
        self.ranges = []

        self.view = ImageView()
        self.view.clicked.connect(self._clicked)
        self.view.dragged.connect(self._dragged)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.message = QLabel()
        self.message.setWordWrap(True)

        tools_box = QGroupBox("Tools")
        tools_layout = QVBoxLayout(tools_box)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for key, label, _ in TOOLS:
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setProperty("tool", key)
            self.tool_group.addButton(button)
            tools_layout.addWidget(button)
        self.tool_group.buttonClicked.connect(lambda _: self._tool_changed())
        self.tool_group.buttons()[0].setChecked(True)
        undo = QPushButton("Undo last point")
        undo.clicked.connect(self.undo)
        clear = QPushButton("Clear this tool")
        clear.clicked.connect(self.clear_tool)
        self.finish_plot_btn = QPushButton("Finish plot")
        self.finish_plot_btn.clicked.connect(self.finish_plot)
        tools_layout.addWidget(undo)
        tools_layout.addWidget(clear)
        tools_layout.addWidget(self.finish_plot_btn)

        self.plot_list = QListWidget()
        self.plot_list.itemChanged.connect(self._rename_plot)
        plots_box = QGroupBox("Field plots (double-click to rename)")
        QVBoxLayout(plots_box).addWidget(self.plot_list)

        self.color_section = QComboBox()
        self.color_section.addItems(["target", "reference"])
        self.save_colors_btn = QPushButton("Save colour ranges")
        self.save_colors_btn.clicked.connect(self.save_colors)
        self.ranges_label = QLabel("No colour boxes yet.")
        self.ranges_label.setWordWrap(True)
        color_box = QGroupBox("Colour")
        cl = QVBoxLayout(color_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Save for"))
        row.addWidget(self.color_section)
        cl.addLayout(row)
        cl.addWidget(self.ranges_label)
        cl.addWidget(self.save_colors_btn)

        self.save_btn = QPushButton("Save measurement setup")
        self.save_btn.clicked.connect(self.save_annotations)
        self.reload_btn = QPushButton("Reload frame")
        self.reload_btn.clicked.connect(self.load)

        side = QVBoxLayout()
        side.addWidget(tools_box)
        side.addWidget(plots_box)
        side.addWidget(color_box)
        side.addWidget(self.save_btn)
        side.addWidget(self.reload_btn)
        side.addStretch()

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.sliderReleased.connect(self._frame_selected)
        self.frame_label = QLabel()
        center = QVBoxLayout()
        center.addWidget(self.hint)
        center.addWidget(self.view, 1)
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        frame_row.addWidget(self.frame_slider, 1)
        frame_row.addWidget(self.frame_label)
        center.addLayout(frame_row)
        center.addWidget(self.message)

        layout = QHBoxLayout(self)
        layout.addLayout(center, 1)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(270)
        layout.addWidget(side_widget)

        state.experiment_changed.connect(lambda _: self._invalidate())
        state.config_changed.connect(self._invalidate)
        state.camera_changed.connect(lambda _: self._invalidate())
        self._tool_changed()
        self._loaded_for = None

    # --- loading ---------------------------------------------------------------------------

    def _invalidate(self) -> None:
        self._loaded_for = None
        if self.isVisible():
            self.load()

    def on_shown(self) -> None:
        exp = self.state.experiment
        if exp is not None and self._loaded_for != exp.root:
            self.load()
        elif exp is None:
            self.message.setText("Open an experiment first.")

    def load(self) -> None:
        exp = self.state.experiment
        if exp is None:
            return
        self._loaded_for = exp.root
        camera = self.state.camera
        self.message.setText("Preparing the reference frame (aligning, correcting)…")

        def work(progress, should_stop):
            from fungus_cv.analyze.pipeline import Analyzer

            analyzer = Analyzer(exp, with_segmenter=False, require_annotations=False,
                                camera=camera)
            return analyzer, analyzer.reference

        run_task(work, self._loaded, lambda m: self.message.setText(
            f"<span style='color:#b00020'>{m}</span> (capture or import some frames first)"))

    def _loaded(self, result) -> None:
        analyzer, reference = result
        self.analyzer = analyzer
        self.frame = reference
        self.view.set_image(reference)
        n = len(analyzer.frames)
        self.frame_slider.setRange(0, n - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(n > 1)
        self.frame_label.setText(f"1 / {n} (reference)")
        self._load_annotations()
        self.message.setText(
            "Reference frame loaded. Scroll to zoom, Option/Alt-drag to pan, double-click to "
            "fit. Right-click removes the last point.")
        self._redraw()

    def _frame_selected(self) -> None:
        if self.analyzer is None:
            return
        index = self.frame_slider.value()
        self.frame_label.setText(f"{index + 1} / {len(self.analyzer.frames)}"
                                 + (" (reference)" if index == 0 else ""))
        self.message.setText("Loading frame…")

        def work(progress, should_stop):
            return self.analyzer.aligned_frame(index)

        run_task(work, self._show_frame, lambda m: self.message.setText(m))

    def _show_frame(self, image) -> None:
        self.frame = image
        self.view.set_image(image, keep_view=True)
        self.message.setText("Annotations are always saved in reference-frame coordinates; "
                             "later frames are aligned to it.")
        self._update_color_preview()

    def _load_annotations(self) -> None:
        for v in self.points.values():
            v.clear()
        self.plots.clear()
        self.current_plot.clear()
        ann = self.analyzer.annotations if self.analyzer else None
        if ann is not None:
            if ann.base is not None:
                self.points["base"] = [ann.base]
                self.points["path"] = list(ann.path[1:]) if ann.path else [ann.tip]
            if not ann.plots:
                self.points["roi"] = list(ann.roi)
            self.points["patch"] = list(ann.reference_patch or [])
            self.plots = [(p.name, list(p.polygon)) for p in (ann.plots or [])]
        self._refresh_plot_list()

    # --- tools -----------------------------------------------------------------------------

    def tool(self) -> str:
        button = self.tool_group.checkedButton()
        return button.property("tool") if button else "base"

    def _tool_changed(self) -> None:
        key = self.tool()
        self.hint.setText(next(h for k, _, h in TOOLS if k == key))
        self.view.drag_enabled = key == "color"
        self.finish_plot_btn.setVisible(key == "plot")

    def _clicked(self, x: float, y: float, button: int) -> None:
        if self.frame is None:
            return
        key = self.tool()
        if button == Qt.RightButton.value:
            self.undo()
            return
        if key == "color":
            return
        h, w = self.frame.shape[:2]
        x, y = float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))
        if key == "base":
            self.points["base"] = [(x, y)]
            self.tool_group.buttons()[1].setChecked(True)  # next: path
            self._tool_changed()
        elif key == "plot":
            self.current_plot.append((x, y))
        else:
            self.points[key].append((x, y))
        self._redraw()

    def _dragged(self, x0, y0, x1, y1) -> None:
        if self.tool() != "color" or self.frame is None:
            return
        h, w = self.frame.shape[:2]
        xa, xb = sorted((int(np.clip(x0, 0, w - 1)), int(np.clip(x1, 0, w - 1))))
        ya, yb = sorted((int(np.clip(y0, 0, h - 1)), int(np.clip(y1, 0, h - 1))))
        if xb - xa >= 2 and yb - ya >= 2:
            self.color_boxes.append((xa, ya, xb, yb))
            self._update_color_preview()
            self._redraw()

    def undo(self) -> None:
        key = self.tool()
        if key == "color" and self.color_boxes:
            self.color_boxes.pop()
            self._update_color_preview()
        elif key == "plot":
            if self.current_plot:
                self.current_plot.pop()
            elif self.plots:
                self.current_plot = self.plots.pop()[1]
                self._refresh_plot_list()
        elif key in self.points and self.points[key]:
            self.points[key].pop()
        self._redraw()

    def clear_tool(self) -> None:
        key = self.tool()
        if key == "color":
            self.color_boxes.clear()
            self._update_color_preview()
        elif key == "plot":
            self.plots.clear()
            self.current_plot.clear()
            self._refresh_plot_list()
        else:
            self.points[key].clear()
        self._redraw()

    def finish_plot(self) -> None:
        if len(self.current_plot) < 3:
            self.message.setText("A plot needs at least 3 corners.")
            return
        names = {n for n, _ in self.plots}
        n = len(self.plots) + 1
        while f"plot{n}" in names:
            n += 1
        self.plots.append((f"plot{n}", list(self.current_plot)))
        self.current_plot.clear()
        self._refresh_plot_list()
        self._redraw()

    def handle_key(self, event) -> bool:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.tool() == "plot":
            self.finish_plot()
            return True
        undo_modifiers = Qt.ControlModifier | Qt.MetaModifier
        if event.key() == Qt.Key_Z and event.modifiers() & undo_modifiers:
            self.undo()
            return True
        return False

    def _refresh_plot_list(self) -> None:
        self.plot_list.blockSignals(True)
        self.plot_list.clear()
        for name, _ in self.plots:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.plot_list.addItem(item)
        self.plot_list.blockSignals(False)

    def _rename_plot(self, item: QListWidgetItem) -> None:
        row = self.plot_list.row(item)
        new = item.text().strip()
        if 0 <= row < len(self.plots) and new:
            self.plots[row] = (new, self.plots[row][1])
            self._redraw()

    def _redraw(self) -> None:
        v = self.view
        v.clear_overlays()
        line = self.points["base"] + self.points["path"]
        v.add_polyline(line, COLORS["path"])
        v.add_points(self.points["base"], COLORS["base"], 6)
        v.add_points(self.points["path"], COLORS["path"], 4)
        v.add_polyline(self.points["roi"], COLORS["roi"], closed=len(self.points["roi"]) > 2)
        v.add_points(self.points["roi"], COLORS["roi"], 3)
        v.add_polyline(self.points["patch"], COLORS["patch"],
                       closed=len(self.points["patch"]) > 2)
        for name, poly in self.plots:
            v.add_polyline(poly, COLORS["plot"], closed=True)
            v.add_label(name, *np.mean(poly, axis=0))
        v.add_polyline(self.current_plot, COLORS["plot"], dashed=True)
        v.add_points(self.current_plot, COLORS["plot"], 3)
        for xa, ya, xb, yb in self.color_boxes:
            v.add_polyline([(xa, ya), (xb, ya), (xb, yb), (xa, yb)], (255, 255, 0), closed=True,
                           width=1, dashed=True)

    # --- colour ------------------------------------------------------------------------------

    def _update_color_preview(self) -> None:
        from fungus_cv.segment.color import ColorThresholdSegmenter, hsv_range_from_samples

        if self.frame is None or not self.color_boxes:
            self.ranges = []
            self.view.set_mask(None)
            self.ranges_label.setText("No colour boxes yet.")
            return
        import cv2

        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        samples = np.concatenate([hsv[ya:yb, xa:xb].reshape(-1, 3)
                                  for xa, ya, xb, yb in self.color_boxes])
        self.ranges = hsv_range_from_samples(samples)
        self.view.set_mask(ColorThresholdSegmenter(self.ranges).segment(self.frame))
        self.ranges_label.setText("HSV ranges: " + "; ".join(f"{list(lo)}–{list(hi)}"
                                                           for lo, hi in self.ranges))

    def save_colors(self) -> None:
        exp = self.state.experiment
        if exp is None or not self.ranges:
            self.message.setText("Drag boxes over the colour first (Pick colour tool).")
            return
        from fungus_cv.config import replace_hsv_ranges_in_yaml

        section = self.color_section.currentText()
        text = exp.config_path.read_text(encoding="utf-8")
        exp.config_path.write_text(replace_hsv_ranges_in_yaml(text, self.ranges, section),
                                   encoding="utf-8")
        self.state.reload()
        self.message.setText(f"Saved colour ranges for the {section} to config.yaml.")

    # --- saving ------------------------------------------------------------------------------

    def build_annotations(self):
        from fungus_cv.measure.geometry import Annotations, Plot, plots_hull

        if self.analyzer is None or self.frame is None:
            raise ValueError("no reference frame loaded")
        h, w = self.analyzer.reference.shape[:2]
        if self.current_plot:
            raise ValueError("finish the plot you are drawing first (Enter)")
        patch = self.points["patch"] or None
        if patch is not None and len(patch) < 3:
            raise ValueError("the neutral patch needs at least 3 corners")
        file = self.analyzer.reference_row["file"]
        if self.plots:
            plots = [Plot(name, poly) for name, poly in self.plots]
            hull = [(float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1)))
                    for x, y in plots_hull(plots, margin=10)]
            return Annotations(None, None, hull, (w, h), reference_file=file,
                               reference_patch=patch, plots=plots)
        if not self.points["base"]:
            raise ValueError("click the base first (or outline field plots)")
        if not self.points["path"]:
            raise ValueError("click at least the tip after the base")
        if len(self.points["roi"]) < 3:
            raise ValueError("click at least 3 corners of the region")
        line = self.points["base"] + self.points["path"]
        return Annotations(line[0], line[-1], list(self.points["roi"]), (w, h),
                           reference_file=file, reference_patch=patch,
                           path=line if len(line) > 2 else None)

    def save_annotations(self) -> None:
        exp = self.state.experiment
        if exp is None:
            return
        try:
            ann = self.build_annotations()
        except ValueError as exc:
            QMessageBox.information(self, "Not ready to save", str(exc))
            return
        from fungus_cv.analyze.pipeline import annotations_path

        ann.save(annotations_path(exp, self.state.camera))
        self.message.setText("Saved measurement setup (annotations.json). Next: Analyze.")
        self.state.config_changed.emit()
