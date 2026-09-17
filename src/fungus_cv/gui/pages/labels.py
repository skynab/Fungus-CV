"""Label images for training: add frames (evenly or where a model is least sure), correct the
masks with a brush or SAM clicks, and mark them reviewed."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.gui.image_view import ImageView
from fungus_cv.gui.pages.prompt import SAM_MODELS
from fungus_cv.gui.qt_util import preload_model_modules, run_task

UNDO_LIMIT = 30


class AddFramesDialog(QDialog):
    """Choose how to add frames from the open experiment."""

    def __init__(self, parent, runs: list[str]):
        super().__init__(parent)
        self.setWindowTitle("Add frames to the dataset")
        self.evenly = QRadioButton("Spread evenly over time")
        self.suggest = QRadioButton("Most useful to label (active learning)")
        self.suggest.setChecked(True)
        self.run = QComboBox()
        self.run.addItems(["(none)"] + runs)
        if runs:
            self.run.setCurrentIndex(len(runs))
        self.against = QComboBox()
        self.against.addItems(["(none)"] + runs)
        self.model = QLineEdit()
        self.model.setPlaceholderText("trained model folder (optional)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.count = QSpinBox()
        self.count.setRange(1, 1000)
        self.count.setValue(20)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model, 1)
        model_row.addWidget(browse)
        form = QFormLayout(self)
        form.addRow(self.evenly)
        form.addRow(self.suggest)
        form.addRow("Starting masks from run", self.run)
        form.addRow("Compare with run", self.against)
        form.addRow("Model", model_row)
        form.addRow("How many frames", self.count)
        form.addRow(QLabel("Most useful: frames where the model is unsure, or where the two "
                           "runs disagree (no model needed for the first round)."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Trained model folder")
        if folder:
            self.model.setText(folder)

    def values(self) -> dict:
        def run(box):
            return None if box.currentIndex() == 0 else box.currentText()

        return {"method": "evenly" if self.evenly.isChecked() else "suggest",
                "run": run(self.run), "against": run(self.against),
                "model": self.model.text().strip() or None, "count": self.count.value()}


class LabelsPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.dataset = None
        self.items: list = []  # the items shown, in order
        self.index = -1
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.undo_stack: list[np.ndarray] = []
        self.dirty = False
        self._last_paint: tuple[int, int] | None = None
        self.points: list[tuple[float, float]] = []
        self.labels: list[int] = []
        self.box = None
        self.proposal: np.ndarray | None = None
        self._sam_running = False
        self._sam_pending = False
        self.busy = False

        self.path_label = QLabel("No dataset open.")
        self.path_label.setWordWrap(True)
        open_btn = QPushButton("Open dataset…")
        open_btn.clicked.connect(self._choose_open)
        new_btn = QPushButton("New dataset…")
        new_btn.clicked.connect(self._choose_new)
        self.add_btn = QPushButton("Add frames from experiment…")
        self.add_btn.clicked.connect(self._choose_add)
        self.rank_btn = QPushButton("Rank with model…")
        self.rank_btn.clicked.connect(self._choose_rank)

        self.unreviewed_only = QCheckBox("Unreviewed only")
        self.unreviewed_only.toggled.connect(lambda _: self.refresh_items())
        self.order = QComboBox()
        self.order.addItems(["Most useful first", "Dataset order"])
        self.order.currentTextChanged.connect(lambda _: self.refresh_items())
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Item", "Group", "Reviewed", "Priority"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._table_selected)

        self.view = ImageView()
        self.view.paint_enabled = True
        self.view.stroke.connect(self._stroke)
        self.view.clicked.connect(self._clicked)
        self.view.dragged.connect(self._dragged)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.message = QLabel()
        self.message.setWordWrap(True)

        tools = QGroupBox("Tools")
        tl = QVBoxLayout(tools)
        self.mode_group = QButtonGroup(self)
        self.brush_mode = QRadioButton("Brush (left paints, right erases)")
        self.sam_mode = QRadioButton("SAM clicks (left target, right not)")
        self.brush_mode.setChecked(True)
        for i, b in enumerate((self.brush_mode, self.sam_mode)):
            self.mode_group.addButton(b, i)
            tl.addWidget(b)
        self.mode_group.idToggled.connect(lambda *_: self._mode_changed())
        self.brush = QSlider(Qt.Horizontal)
        self.brush.setRange(1, 150)
        self.brush.setValue(8)
        self.brush_label = QLabel()
        self.brush.valueChanged.connect(lambda v: self.brush_label.setText(f"Brush {v} px"))
        self.brush.valueChanged.emit(self.brush.value())
        tl.addWidget(self.brush_label)
        tl.addWidget(self.brush)
        self.sam_model = QComboBox()
        self.sam_model.addItems(SAM_MODELS)
        self.box_btn = QPushButton("Draw box")
        self.box_btn.setCheckable(True)
        self.box_btn.toggled.connect(self._box_toggled)
        self.replace_btn = QPushButton("Replace mask (Enter)")
        self.replace_btn.clicked.connect(lambda: self.apply_proposal("replace"))
        self.add_mask_btn = QPushButton("Add to mask (+)")
        self.add_mask_btn.clicked.connect(lambda: self.apply_proposal("add"))
        self.subtract_btn = QPushButton("Subtract (−)")
        self.subtract_btn.clicked.connect(lambda: self.apply_proposal("subtract"))
        clear_clicks = QPushButton("Clear clicks")
        clear_clicks.clicked.connect(self.clear_clicks)
        self.sam_widgets = [self.sam_model, self.box_btn, self.replace_btn, self.add_mask_btn,
                            self.subtract_btn, clear_clicks]
        tl.addWidget(self.sam_model)
        for w in self.sam_widgets[1:]:
            tl.addWidget(w)
        self.show_mask = QCheckBox("Show mask (T)")
        self.show_mask.setChecked(True)
        self.show_mask.toggled.connect(lambda _: self._draw_mask())
        tl.addWidget(self.show_mask)
        undo = QPushButton("Undo (Z)")
        undo.clicked.connect(self.undo)
        self.save_btn = QPushButton("Save — mark reviewed (S)")
        self.save_btn.clicked.connect(self.save_item)
        prev_btn = QPushButton("◀ Previous (A)")
        prev_btn.clicked.connect(lambda: self.go(-1))
        next_btn = QPushButton("Next (D) ▶")
        next_btn.clicked.connect(lambda: self.go(1))
        nav = QHBoxLayout()
        nav.addWidget(prev_btn)
        nav.addWidget(next_btn)
        tl.addWidget(undo)
        tl.addWidget(self.save_btn)
        tl.addLayout(nav)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.path_label)
        row = QHBoxLayout()
        row.addWidget(open_btn)
        row.addWidget(new_btn)
        ll.addLayout(row)
        ll.addWidget(self.add_btn)
        ll.addWidget(self.rank_btn)
        row = QHBoxLayout()
        row.addWidget(self.unreviewed_only)
        row.addWidget(self.order, 1)
        ll.addLayout(row)
        ll.addWidget(self.table, 1)
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.info)
        cl.addWidget(self.view, 1)
        cl.addWidget(self.message)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(tools)
        rl.addStretch()
        right.setFixedWidth(250)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.setSizes([440, 700])
        layout = QHBoxLayout(self)
        layout.addWidget(splitter, 1)
        layout.addWidget(right)
        self._mode_changed()
        self._update_buttons()

    # --- datasets --------------------------------------------------------------------------

    def on_shown(self) -> None:
        if self.dataset is None:
            last = self.state.settings.value("last_dataset", "", str)
            if last and (Path(last) / "dataset.json").exists():
                self.open_dataset(Path(last))

    def _choose_open(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open dataset folder (with dataset.json)")
        if folder:
            self.open_dataset(Path(folder))

    def _choose_new(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose an empty folder for the dataset")
        if folder:
            self.create_dataset(Path(folder))

    def open_dataset(self, path: Path) -> None:
        from fungus_cv.learn.dataset import Dataset

        if not self._leave_item():
            return
        try:
            self.dataset = Dataset.open(path)
        except (FileNotFoundError, ValueError) as exc:
            self._error(str(exc))
            return
        self.state.settings.setValue("last_dataset", str(Path(path).resolve()))
        self.index = -1
        self.refresh_items()
        if self.items:
            self.select(0)

    def create_dataset(self, path: Path) -> None:
        from fungus_cv.learn.dataset import Dataset

        try:
            Dataset.create(path)
        except (FileExistsError, OSError) as exc:
            self._error(str(exc))
            return
        self.open_dataset(path)
        self.message.setText("New dataset. Add frames from the open experiment to start.")

    def refresh_items(self, keep_id: str | None = None) -> None:
        from fungus_cv.learn.active import by_priority

        ds = self.dataset
        current = keep_id or (self.items[self.index].id if 0 <= self.index < len(self.items)
                              else None)
        items = [] if ds is None else [i for i in ds.items if not (
            self.unreviewed_only.isChecked() and i.reviewed)]
        if self.order.currentText() == "Most useful first":
            items = by_priority(items)
        self.items = items
        self.table.blockSignals(True)
        self.table.setRowCount(len(items))
        for r, item in enumerate(items):
            cells = [item.id, item.group, "yes" if item.reviewed else "",
                     "" if item.priority is None else f"{item.priority:.2f}"]
            for c, text in enumerate(cells):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.blockSignals(False)
        if ds is not None:
            reviewed = sum(i.reviewed for i in ds.items)
            self.path_label.setText(f"<b>{ds.name}</b> — {len(ds.items)} item(s), {reviewed} "
                                    f"reviewed<br>{ds.root}")
        ids = [i.id for i in items]
        self.index = ids.index(current) if current in ids else -1
        if self.index >= 0:
            self.table.blockSignals(True)
            self.table.selectRow(self.index)
            self.table.blockSignals(False)
        self._update_buttons()

    # --- items -----------------------------------------------------------------------------

    def _table_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows and rows[0].row() != self.index:
            self.select(rows[0].row())

    def _leave_item(self) -> bool:
        """Unsaved edits are saved when moving on, as in `fungus label`."""
        if self.dirty:
            self.save_item()
        return True

    def select(self, index: int) -> None:
        if self.dataset is None or not 0 <= index < len(self.items):
            return
        if index != self.index:
            self._leave_item()
        self.index = index
        item = self.items[index]
        self.image = self.dataset.load_image(item)
        self.mask = self.dataset.load_mask(item)
        self.undo_stack, self.dirty = [], False
        self.clear_clicks()
        self.view.set_image(self.image)
        self._draw_mask()
        self.table.blockSignals(True)
        self.table.selectRow(index)
        self.table.blockSignals(False)
        self._show_info()
        self._update_buttons()

    def go(self, step: int) -> None:
        if self.items:
            self.select(max(0, min(len(self.items) - 1, self.index + step)))

    def _show_info(self) -> None:
        if not 0 <= self.index < len(self.items):
            self.info.setText("")
            return
        item = self.items[self.index]
        parts = [f"<b>{self.index + 1}/{len(self.items)}</b>  {item.id}",
                 "reviewed" if item.reviewed else "<b>not reviewed</b>"]
        if item.priority is not None:
            detail = ", ".join(f"{k} {v:.2f}" for k, v in item.scores.items()
                               if isinstance(v, float))
            parts.append(f"priority {item.priority:.2f} ({detail})")
        if self.dirty:
            parts.append("<i>unsaved</i>")
        self.info.setText("  ·  ".join(parts))

    def save_item(self) -> None:
        if self.dataset is None or self.mask is None or not 0 <= self.index < len(self.items):
            return
        item, position = self.items[self.index], self.index
        self.dataset.write_mask(item, self.mask)
        item.reviewed = True
        self.dataset.save()
        self.dirty = False
        self.message.setText(f"Saved {item.id} (reviewed).")
        self.refresh_items(keep_id=item.id)
        if self.index < 0 and self.items:  # "unreviewed only": go on to the next one
            self.select(min(position, len(self.items) - 1))
        self._show_info()

    # --- brush -----------------------------------------------------------------------------

    def _push_undo(self) -> None:
        self.undo_stack.append(self.mask.copy())
        del self.undo_stack[:-UNDO_LIMIT]

    def _stroke(self, x: float, y: float, button: int, phase: int) -> None:
        if self.mask is None or not self.brush_mode.isChecked():
            return
        value = button == Qt.LeftButton.value
        point = (int(round(x)), int(round(y)))
        if phase == ImageView.STROKE_PRESS:
            self._push_undo()
            self._last_paint = point
        if phase in (ImageView.STROKE_PRESS, ImageView.STROKE_MOVE):
            self.paint_segment(self._last_paint or point, point, value)
            self._last_paint = point
        else:
            self._last_paint = None
            self._show_info()

    def paint_segment(self, start: tuple[int, int], end: tuple[int, int], value: bool) -> None:
        """Paint (or erase) a round brush along a line, so fast strokes leave no gaps."""
        m = self.mask.astype(np.uint8)
        radius = self.brush.value()
        cv2.line(m, start, end, 1 if value else 0, thickness=2 * radius + 1)
        cv2.circle(m, end, radius, 1 if value else 0, -1)
        self.mask = m.astype(bool)
        self.dirty = True
        self._draw_mask()

    def undo(self) -> None:
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.dirty = True
            self._draw_mask()
            self._show_info()

    def _draw_mask(self) -> None:
        self.view.set_mask(self.mask if self.show_mask.isChecked() else None)
        self.view.set_mask(self.proposal, color=(255, 220, 0), alpha=120, layer=1)

    # --- SAM clicks ------------------------------------------------------------------------

    def _mode_changed(self) -> None:
        sam = self.sam_mode.isChecked()
        self.view.paint_enabled = not sam
        self.brush.setEnabled(not sam)
        for w in self.sam_widgets:
            w.setEnabled(sam)
        if not sam:
            self.box_btn.setChecked(False)
            self.clear_clicks()

    def _box_toggled(self, on: bool) -> None:
        self.view.drag_enabled = on

    def _clicked(self, x: float, y: float, button: int) -> None:
        if not self.sam_mode.isChecked() or self.image is None:
            return
        self.points.append((x, y))
        self.labels.append(1 if button == Qt.LeftButton.value else 0)
        self._clicks_changed()

    def _dragged(self, x0, y0, x1, y1) -> None:
        if self.sam_mode.isChecked():
            self.box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self.box_btn.setChecked(False)
            self._clicks_changed()

    def clear_clicks(self) -> None:
        self.points, self.labels, self.box, self.proposal = [], [], None, None
        self.view.clear_overlays()
        if self.image is not None:
            self._draw_mask()

    def _prompt(self):
        from fungus_cv.segment.prompts import FramePrompt

        try:
            item_id = self.items[self.index].id if self.items else "item"
            return FramePrompt(item_id, list(self.points), list(self.labels), self.box)
        except ValueError:
            return None

    def _clicks_changed(self) -> None:
        self.view.clear_overlays()
        self.view.add_points([p for p, lab in zip(self.points, self.labels) if lab == 1])
        self.view.add_points([p for p, lab in zip(self.points, self.labels) if lab != 1],
                             color=(230, 30, 30))
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            self.view.add_polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], (255, 220, 0),
                                   closed=True)
        if self._prompt() is None:
            self.proposal = None
            self._draw_mask()
        elif self._sam_running:
            self._sam_pending = True
        else:
            self._run_sam()

    def _run_sam(self) -> None:
        from fungus_cv.segment.prompts import Prompts
        from fungus_cv.segment.sam2 import Sam2VideoSegmenter

        prompt, image = self._prompt(), self.image
        if prompt is None:
            return
        self._sam_running = True
        self.message.setText("Running SAM 2… (the first time loads the model)")
        preload_model_modules({"sam2"})
        segmenter = Sam2VideoSegmenter(model_name=self.sam_model.currentText(),
                                       prompts=Prompts())
        run_task(lambda p, s: (prompt, segmenter.segment_single(image, prompt)),
                 self._sam_done, self._sam_failed)

    def _sam_done(self, result) -> None:
        prompt, mask = result
        self._sam_running = False
        if self._sam_pending:
            self._sam_pending = False
            self._run_sam()
            return
        if prompt != self._prompt():
            return
        self.proposal = np.asarray(mask, bool)
        self._draw_mask()
        self.message.setText("Yellow = SAM's mask. Replace, add or subtract it.")

    def _sam_failed(self, message: str) -> None:
        self._sam_running = False
        self._error(f"SAM 2: {message}")

    def apply_proposal(self, mode: str) -> None:
        from fungus_cv.ui.interactive import combine_masks

        if self.proposal is None or self.mask is None:
            return
        self._push_undo()
        self.mask = combine_masks(self.mask, self.proposal, mode)
        self.dirty = True
        self.clear_clicks()
        self._show_info()

    # --- adding and ranking ----------------------------------------------------------------

    def _choose_add(self) -> None:
        from fungus_cv.analyze.compare import list_runs

        exp = self.state.experiment
        if exp is None or self.dataset is None:
            self._error("Open an experiment (for its frames) and a dataset first.")
            return
        dialog = AddFramesDialog(self, [r.run_id for r in list_runs(exp)])
        if dialog.exec() == QDialog.Accepted:
            self.add_frames(**dialog.values())

    def add_frames(self, method: str = "suggest", run: str | None = None,
                   against: str | None = None, model: str | None = None,
                   count: int = 20) -> None:
        exp, ds = self.state.experiment, self.dataset
        if exp is None or ds is None:
            self._error("Open an experiment (for its frames) and a dataset first.")
            return
        if model:
            preload_model_modules({"model"})
        self._set_busy(True, "Adding frames…")

        def work(progress, should_stop):
            from fungus_cv.analyze.compare import list_runs
            from fungus_cv.learn.active import suggest_frames
            from fungus_cv.learn.export import export_from_run

            if method == "evenly":
                chosen = run or (list_runs(exp)[-1].run_id if list_runs(exp) else None)
                if chosen is None:
                    raise ValueError("no analysis runs yet: run Analyze first")
                return "evenly", export_from_run(exp, ds, chosen, count=count)
            result = suggest_frames(exp, ds, count=count,
                                    model_dir=Path(model) if model else None,
                                    run=run, against=against)
            return "suggest", result.added

        run_task(work, self._added, self._busy_failed)

    def _added(self, result) -> None:
        how, added = result
        self._set_busy(False)
        self.refresh_items()
        what = "most useful" if how == "suggest" else "evenly spaced"
        self.message.setText(f"Added {len(added)} {what} frame(s). Correct their masks and "
                             "save each one.")
        if added and self.index < 0:
            ids = [i.id for i in self.items]
            if added[0].id in ids:
                self.select(ids.index(added[0].id))

    def _choose_rank(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Trained model folder")
        if folder:
            self.rank(Path(folder))

    def rank(self, model_dir: Path, prob_fn=None, threshold=None) -> None:
        """Score unreviewed items. ``prob_fn``/``threshold`` replace the model (tests)."""
        ds = self.dataset
        if ds is None:
            return
        self._leave_item()
        if prob_fn is None:
            preload_model_modules({"model"})
        self._set_busy(True, "Scoring items with the model…")

        def work(progress, should_stop):
            from fungus_cv.learn.active import rank_items

            return rank_items(ds, model_dir, prob_fn=prob_fn, threshold=threshold)

        run_task(work, self._ranked, self._busy_failed)

    def _ranked(self, ranked) -> None:
        self._set_busy(False)
        self.order.setCurrentText("Most useful first")
        self.refresh_items()
        self.message.setText(f"Scored {len(ranked)} item(s); the most useful are at the top.")
        if self.items:
            self.select(0)

    # --- helpers ---------------------------------------------------------------------------

    def _set_busy(self, busy: bool, text: str = "") -> None:
        self.busy = busy
        if text:
            self.message.setText(text)
        self._update_buttons()

    def _busy_failed(self, message: str) -> None:
        self._set_busy(False)
        self._error(message)

    def _error(self, message: str) -> None:
        self.message.setText(f"<span style='color:#b00020'>{message}</span>")

    def _update_buttons(self) -> None:
        has_ds = self.dataset is not None
        self.add_btn.setEnabled(has_ds and not self.busy)
        self.rank_btn.setEnabled(has_ds and not self.busy)
        self.save_btn.setEnabled(0 <= self.index < len(self.items))

    def handle_key(self, event) -> bool:
        key = event.key()
        actions = {
            Qt.Key_S: self.save_item, Qt.Key_Z: self.undo,
            Qt.Key_A: lambda: self.go(-1), Qt.Key_D: lambda: self.go(1),
            Qt.Key_T: lambda: self.show_mask.toggle(),
            Qt.Key_M: lambda: (self.brush_mode if self.sam_mode.isChecked()
                               else self.sam_mode).setChecked(True),
            Qt.Key_BracketLeft: lambda: self.brush.setValue(self.brush.value() - 2),
            Qt.Key_BracketRight: lambda: self.brush.setValue(self.brush.value() + 2),
        }
        if self.sam_mode.isChecked():
            actions.update({Qt.Key_Return: lambda: self.apply_proposal("replace"),
                            Qt.Key_Enter: lambda: self.apply_proposal("replace"),
                            Qt.Key_Plus: lambda: self.apply_proposal("add"),
                            Qt.Key_Equal: lambda: self.apply_proposal("add"),
                            Qt.Key_Minus: lambda: self.apply_proposal("subtract")})
        if key in actions:
            actions[key]()
            return True
        return False
