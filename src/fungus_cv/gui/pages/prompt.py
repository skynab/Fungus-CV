"""Click on the target for SAM 2: the mask previews after each click; prompts are saved per
frame and tracked through the time-lapse by `Analyze`."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.gui import theme
from fungus_cv.gui.image_view import ImageView
from fungus_cv.gui.qt_util import preload_model_modules, run_task
from fungus_cv.segment.torch_device import missing_dependency

SAM_MODELS = ["facebook/sam2.1-hiera-tiny", "facebook/sam2.1-hiera-small",
              "facebook/sam2.1-hiera-base-plus", "facebook/sam2.1-hiera-large"]


class PromptPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.analyzer = None
        self.frame: np.ndarray | None = None
        self.index = 0
        self.points: list[tuple[float, float]] = []
        self.labels: list[int] = []
        self.box: tuple[float, float, float, float] | None = None
        self.mask: np.ndarray | None = None
        self.prompts = None
        self._preview_running = False
        self._preview_pending = False
        self._loaded_for = None
        self._position: tuple | None = None  # (experiment root, frame index) to return to
        self.loading = False

        self.which = QComboBox()
        self.which.addItems(["target", "reference"])
        self.which.setToolTip("reference = the object growth is measured along (e.g. the stem)")
        self.which.currentTextChanged.connect(self._which_changed)
        self.model = QComboBox()
        self.model.addItems(SAM_MODELS)
        self.model.currentTextChanged.connect(lambda _: self._changed())
        self.box_btn = QPushButton("Draw box")
        self.box_btn.setCheckable(True)
        self.box_btn.toggled.connect(lambda on: setattr(self.view, "drag_enabled", on))
        undo = QPushButton("Undo click")
        undo.clicked.connect(self.undo)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        self.save_btn = QPushButton("Save prompt for this frame")
        theme.mark_primary(self.save_btn)
        self.save_btn.clicked.connect(self.save_prompt)
        self.remove_btn = QPushButton("Remove this frame's prompt")
        self.remove_btn.clicked.connect(self.remove_prompt)
        self.use_btn = QPushButton("Use SAM 2 for the target")
        self.use_btn.clicked.connect(self.use_sam)

        self.view = ImageView()
        self.view.clicked.connect(self._clicked)
        self.view.dragged.connect(self._dragged)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(lambda: self.show_frame(self.slider.value()))
        self.frame_label = QLabel()
        self.prompted = QListWidget()
        self.prompted.itemClicked.connect(self._jump)
        self.message = QLabel("Open an experiment first.")
        self.message.setWordWrap(True)

        side = QVBoxLayout()
        opts = QGroupBox("Prompt")
        ol = QVBoxLayout(opts)
        for label, widget in (("For", self.which), ("SAM 2 model", self.model)):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(widget, 1)
            ol.addLayout(row)
        ol.addWidget(QLabel("Left click: target\nRight click: not the target"))
        for w in (self.box_btn, undo, clear, self.save_btn, self.remove_btn):
            ol.addWidget(w)
        side.addWidget(opts)
        side.addWidget(QLabel("Prompted frames"))
        side.addWidget(self.prompted, 1)
        side.addWidget(self.use_btn)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(290)

        center = QVBoxLayout()
        center.addWidget(QLabel("Pick a frame where the target is clearly visible. Each "
                                "prompted frame corrects tracking from there on."))
        center.addWidget(self.view, 1)
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        frame_row.addWidget(self.slider, 1)
        frame_row.addWidget(self.frame_label)
        center.addLayout(frame_row)
        center.addWidget(self.message)
        layout = QHBoxLayout(self)
        layout.addLayout(center, 1)
        layout.addWidget(side_widget)

        state.experiment_changed.connect(lambda _: self._invalidate())
        state.config_changed.connect(self._invalidate)
        state.camera_changed.connect(lambda _: self._invalidate())

    def _which_changed(self, which: str) -> None:
        self.use_btn.setText(f"Use SAM 2 for the {which}")
        self._load_prompts()

    # --- loading ---------------------------------------------------------------------------

    def _invalidate(self) -> None:
        self._loaded_for = None
        if self.isVisible():
            self.load()

    def on_shown(self) -> None:
        exp = self.state.experiment
        if exp is None:
            self.message.setText("Open an experiment first.")
        elif self._loaded_for != exp.root:
            self.load()

    def load(self) -> None:
        exp = self.state.experiment
        if exp is None:
            return
        self._loaded_for = exp.root
        self.loading = True
        camera = self.state.camera
        keep = self._position[1] if self._position and self._position[0] == exp.root else None
        model = exp.config.analysis.target.sam2.model
        if self.model.findText(model) < 0:
            self.model.addItem(model)
        self.model.setCurrentText(model)
        self.message.setText("Preparing frames…")

        def work(progress, should_stop):
            from fungus_cv.analyze.pipeline import Analyzer

            analyzer = Analyzer(exp, with_segmenter=False, require_annotations=False,
                                camera=camera)
            n = len(analyzer.frames)
            index = keep if keep is not None and keep < n else n - 1  # stay where the user was
            return analyzer, index, analyzer.aligned_frame(index)

        run_task(work, self._loaded, self._failed)

    def _loaded(self, result) -> None:
        self.loading = False
        self.analyzer, last, frame = result
        n = len(self.analyzer.frames)
        self.slider.setRange(0, n - 1)
        self.slider.setEnabled(n > 1)
        # Before anything reads them: _load_prompts() shows the current frame again, and an
        # index kept from a longer experiment would be past the end of this one.
        self.index, self.frame = last, frame
        self._load_prompts()
        self._set_frame(last, frame)
        self.message.setText("Click on the target. The magenta mask is SAM's answer.")

    def _failed(self, message: str) -> None:
        self._preview_running = False
        self.loading = False
        self.message.setText(f"<span style='color:#b00020'>{message}</span>")

    def _config_block(self):
        analysis = self.state.experiment.config.analysis
        return getattr(analysis, self.which.currentText()).sam2

    def _prompts_path(self):
        from fungus_cv.analyze.pipeline import prompts_path

        return prompts_path(self.state.experiment, self._config_block().prompts_file,
                            self.state.camera)

    def _load_prompts(self) -> None:
        from fungus_cv.segment.prompts import Prompts

        if self.state.experiment is None:
            return
        path = self._prompts_path()
        self.prompts = Prompts.load(path) if path.exists() else Prompts()
        self._refresh_list()
        if self.analyzer is not None and self.frame is not None:
            self._set_frame(self.index, self.frame)

    def _refresh_list(self) -> None:
        self.prompted.clear()
        files = [r["file"] for r in self.analyzer.frames] if self.analyzer else []
        for p in self.prompts.frames:
            item = QListWidgetItem(p.frame_file.split("/")[-1])
            item.setData(Qt.UserRole, files.index(p.frame_file) if p.frame_file in files else -1)
            self.prompted.addItem(item)

    def _jump(self, item) -> None:
        index = item.data(Qt.UserRole)
        if index is not None and index >= 0:
            self.show_frame(index)

    def show_frame(self, index: int) -> None:
        if self.analyzer is None:
            return
        self.message.setText("Loading frame…")
        analyzer = self.analyzer

        def work(progress, should_stop):
            return index, analyzer.aligned_frame(index)

        run_task(work, self._frame_ready, self._failed)

    def _frame_ready(self, result) -> None:
        self._set_frame(*result)

    def _set_frame(self, index: int, frame: np.ndarray) -> None:
        self.index, self.frame = index, frame
        self._position = (self.state.experiment.root, index)
        self.slider.setValue(index)
        n = len(self.analyzer.frames)
        self.frame_label.setText(f"{index + 1} / {n}")
        self.view.set_image(frame, keep_view=True)
        existing = self.prompts.for_file(self.frame_file) if self.prompts else None
        self.points = list(existing.points) if existing else []
        self.labels = list(existing.labels) if existing else []
        self.box = tuple(existing.box) if existing and existing.box else None
        self.remove_btn.setEnabled(existing is not None)
        self.message.setText("Loaded this frame's saved prompt." if existing else "")
        self.mask = None
        self._changed()

    @property
    def frame_file(self) -> str:
        return self.analyzer.frames[self.index]["file"]

    # --- clicks ----------------------------------------------------------------------------

    def _clicked(self, x: float, y: float, button: int) -> None:
        from fungus_cv.segment.prompts import NEGATIVE, POSITIVE

        if self.frame is None:
            return
        self.points.append((x, y))
        self.labels.append(POSITIVE if button == Qt.LeftButton.value else NEGATIVE)
        self._changed()

    def _dragged(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self.box_btn.setChecked(False)
        self._changed()

    def undo(self) -> None:
        if self.points:
            self.points.pop()
            self.labels.pop()
        elif self.box is not None:
            self.box = None
        self._changed()

    def clear(self) -> None:
        self.points, self.labels, self.box = [], [], None
        self._changed()

    def current_prompt(self):
        from fungus_cv.segment.prompts import FramePrompt

        try:
            return FramePrompt(self.frame_file, list(self.points), list(self.labels), self.box)
        except ValueError:
            return None

    def _changed(self) -> None:
        if self.frame is None:
            return
        self._redraw()
        self.save_btn.setEnabled(self.current_prompt() is not None)
        if self.current_prompt() is None:
            self.mask = None
            self.view.set_mask(None)
            return
        if self._preview_running:
            self._preview_pending = True
        else:
            self._preview()

    def _redraw(self) -> None:
        from fungus_cv.segment.prompts import POSITIVE

        self.view.clear_overlays()
        self.view.add_points([p for p, lab in zip(self.points, self.labels) if lab == POSITIVE])
        self.view.add_points([p for p, lab in zip(self.points, self.labels) if lab != POSITIVE],
                             color=(230, 30, 30))
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            self.view.add_polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], (255, 220, 0),
                                   closed=True)

    # --- preview ---------------------------------------------------------------------------

    def _segmenter(self):
        from fungus_cv.segment.prompts import Prompts
        from fungus_cv.segment.sam2 import Sam2VideoSegmenter, _RoiCrop

        cfg = self._config_block()
        ann = self.analyzer.annotations if self.analyzer else None
        return Sam2VideoSegmenter(
            model_name=self.model.currentText(), prompts=Prompts(), device=cfg.device,
            crop=_RoiCrop(ann.roi, cfg.crop_margin_px) if cfg.crop_to_roi and ann else None,
            mask_threshold=cfg.mask_threshold, min_blob_area_px=cfg.min_blob_area_px)

    def _preview(self) -> None:
        prompt, frame = self.current_prompt(), self.frame
        if prompt is None:
            return
        self._preview_running = True
        self.message.setText("Running SAM 2… (the first time loads the model)")
        preload_model_modules({"sam2"})
        try:
            segmenter = self._segmenter()
        except Exception as exc:  # noqa: BLE001 - shown on the page, not raised at the user
            self._failed(missing_dependency(exc) or f"{type(exc).__name__}: {exc}")
            return
        run_task(lambda p, s: (prompt, segmenter.segment_single(frame, prompt)),
                 self._preview_done, self._failed)

    def _preview_done(self, result) -> None:
        prompt, mask = result
        self._preview_running = False
        if self._preview_pending:
            self._preview_pending = False
            self._preview()
            return
        if prompt != self.current_prompt():
            return
        self.mask = mask
        self.view.set_mask(mask)
        share = 100 * float(mask.mean()) if mask.size else 0.0
        self.message.setText(f"SAM 2 mask covers {share:.1f}% of the frame. Save the prompt "
                             "when it looks right.")

    # --- saving ----------------------------------------------------------------------------

    def save_prompt(self) -> None:
        prompt = self.current_prompt()
        if prompt is None or self.prompts is None:
            return
        self.prompts.set(prompt)
        self.prompts.save(self._prompts_path())
        self._refresh_list()
        self.remove_btn.setEnabled(True)
        method = getattr(self.state.experiment.config.analysis, self.which.currentText()).method
        hint = "" if method == "sam2" else (" Set this object's method to SAM 2 to use it "
                                            "(button below).")
        self.message.setText(f"Saved prompt for {prompt.frame_file.split('/')[-1]} "
                             f"({len(self.prompts.frames)} prompted frame(s)).{hint}")

    def remove_prompt(self) -> None:
        if self.prompts is None or self.prompts.for_file(self.frame_file) is None:
            return
        self.prompts.frames = [p for p in self.prompts.frames if p.frame_file != self.frame_file]
        self.prompts.save(self._prompts_path())
        self._refresh_list()
        self.remove_btn.setEnabled(False)
        self.message.setText("Removed this frame's prompt.")

    def use_sam(self) -> None:
        from fungus_cv.config import ExperimentConfig, set_yaml_value

        exp = self.state.experiment
        if exp is None:
            return
        which = self.which.currentText()
        text = exp.config_path.read_text(encoding="utf-8")
        text = set_yaml_value(text, f"analysis.{which}.method", "sam2")
        text = set_yaml_value(text, f"analysis.{which}.sam2.model", self.model.currentText())
        import yaml

        ExperimentConfig.model_validate(yaml.safe_load(text))
        exp.config_path.write_text(text, encoding="utf-8")
        self.state.reload()
        self.message.setText(f"The {which} is now segmented with SAM 2 "
                             f"({self.model.currentText().split('/')[-1]}). Run Analyze.")

    def handle_key(self, event) -> bool:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.save_btn.isEnabled():
            self.save_prompt()
            return True
        if event.key() == Qt.Key_Backspace:
            self.undo()
            return True
        return False
