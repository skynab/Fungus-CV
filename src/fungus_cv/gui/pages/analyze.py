"""Run the analysis, browse per-frame results and exclude frames by hand."""

from __future__ import annotations

import csv

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.gui.image_view import ImageView
from fungus_cv.gui.qt_util import preload_model_modules, run_task

COLUMNS = [("timestamp_utc", "Time"), ("plot", "Plot"), ("extent_mm", "Extent mm"),
           ("extent_mm_unc", "±"), ("coverage_pct", "Coverage %"),
           ("target_area_mm2", "Area mm²"), ("covered_length_pct", "Length covered %"),
           ("flags", "Flags"), ("excluded", "Excluded by hand")]
VIEWS = ["Overlay", "Aligned frame + mask", "Aligned frame", "Photo as taken"]


def _local_time(iso_utc: str) -> str:
    from fungus_cv.storage import parse_iso_utc

    try:
        return parse_iso_utc(iso_utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return iso_utc


class AnalyzePage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.task = None
        self.rows: list[dict] = []
        self.visible_rows: list[dict] = []
        self.exclusions: list = []
        self._analyzer = None  # for aligned frames; built on first use

        self.run_btn = QPushButton("Run analysis")
        self.run_btn.clicked.connect(self.run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.setEnabled(False)
        self.force = QCheckBox("Re-measure all frames")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.plot_filter = QComboBox()
        self.plot_filter.currentTextChanged.connect(lambda _: self._fill_table())

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._row_selected)
        self.overlay = ImageView()
        self.view_mode = QComboBox()
        self.view_mode.addItems(VIEWS)
        self.view_mode.currentTextChanged.connect(lambda _: self._row_selected())
        self.exclude_btn = QPushButton("Exclude frame…")
        self.exclude_btn.setToolTip("Leave this frame out of fits, with a reason (e.g. a hand "
                                    "in the picture). Reports and studies list it.")
        self.exclude_btn.clicked.connect(self._ask_exclude)
        self.include_btn = QPushButton("Include again")
        self.include_btn.clicked.connect(self.include_selected)
        self.frame_info = QLabel()
        self.frame_info.setWordWrap(True)
        for button in (self.exclude_btn, self.include_btn):
            button.setEnabled(False)

        top = QHBoxLayout()
        top.addWidget(self.run_btn)
        top.addWidget(self.cancel_btn)
        top.addWidget(self.force)
        top.addWidget(self.progress, 1)
        top.addWidget(QLabel("Plot:"))
        top.addWidget(self.plot_filter)
        viewer = QWidget()
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Show:"))
        controls.addWidget(self.view_mode)
        controls.addStretch()
        controls.addWidget(self.exclude_btn)
        controls.addWidget(self.include_btn)
        viewer_layout.addLayout(controls)
        viewer_layout.addWidget(self.overlay, 1)
        viewer_layout.addWidget(self.frame_info)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(viewer)
        splitter.setSizes([640, 560])
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.message)
        layout.addWidget(splitter, 1)

        state.experiment_changed.connect(lambda _: self._experiment_changed())
        state.busy_changed.connect(lambda busy: self.run_btn.setEnabled(
            not busy and self.task is None and state.experiment is not None))

    def on_shown(self) -> None:
        self.load_results()

    def _experiment_changed(self) -> None:
        self._analyzer = None
        self.load_results()

    # --- running ---------------------------------------------------------------------------

    def run(self) -> None:
        exp = self.state.experiment
        if exp is None:
            return
        force = self.force.isChecked()
        analysis = exp.config.analysis
        methods = {analysis.target.method, analysis.reference.method}
        if methods & {"sam2", "model"}:
            self.message.setText("Loading PyTorch…")
            self.repaint()
            preload_model_modules(methods)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.message.setText("Preparing (loading models and the reference frame)…")

        def work(progress, should_stop):
            from fungus_cv.analyze.pipeline import Analyzer
            from fungus_cv.storage import Experiment

            analyzer = Analyzer(Experiment(exp.root))
            return analyzer.run(force=force, progress=lambda d, t, stage: progress((d, t, stage)),
                                should_stop=should_stop)

        self.task = run_task(work, self._done, self._failed, self._progress)

    def cancel(self) -> None:
        if self.task is not None:
            self.task.stop()
            self.message.setText("Cancelling after the current frame…")

    def _progress(self, value) -> None:
        done, total, stage = value
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        what = "segmenting the reference object" if stage == "reference" else "measuring"
        self.message.setText(f"{what.capitalize()}: frame {done} of {total}")

    def _finish(self) -> None:
        self.task = None
        self.progress.setVisible(False)
        self.cancel_btn.setEnabled(False)
        self.run_btn.setEnabled(self.state.experiment is not None and not self.state.capturing)

    def _done(self, summary) -> None:
        self._analyzer = None  # settings may have changed
        self._finish()
        flags = ", ".join(f"{k}: {v}" for k, v in summary.flagged.items()) or "none"
        verb = "Stopped" if summary.stopped else "Done"
        self.message.setText(f"{verb}: measured {summary.processed}, already done "
                             f"{summary.skipped_existing}, failed {summary.failed}. "
                             f"Flags: {flags}.")
        self.load_results()

    def _failed(self, message: str) -> None:
        self._finish()
        self.message.setText(f"<span style='color:#b00020'>{message}</span>")

    # --- results ---------------------------------------------------------------------------

    def load_results(self) -> None:
        from fungus_cv.analyze import exclusions

        exp = self.state.experiment
        self.rows = []
        self.exclusions = exclusions.load(exp) if exp is not None else []
        self.overlay.set_image(None)
        self.overlay.set_mask(None)
        self.run_btn.setEnabled(exp is not None and self.task is None)
        if exp is not None:
            path = exp.root / "results" / "measurements.csv"
            if path.exists():
                with open(path, newline="", encoding="utf-8") as f:
                    self.rows = sorted(csv.DictReader(f), key=lambda r: r["timestamp_utc"])
            elif self.task is None:
                self.message.setText("No results yet. Set up the measurement, then run the "
                                     "analysis.")
        plots = sorted({r.get("plot") or "main" for r in self.rows})
        self.plot_filter.blockSignals(True)
        self.plot_filter.clear()
        self.plot_filter.addItems(["all"] + plots if len(plots) > 1 else plots)
        self.plot_filter.blockSignals(False)
        self._fill_table()

    def _fill_table(self) -> None:
        wanted = self.plot_filter.currentText()
        from fungus_cv.analyze.exclusions import reason_for

        selected = self.selected_row()
        rows = [r for r in self.rows if wanted in ("", "all") or (r.get("plot") or "main") ==
                wanted]
        self.visible_rows = rows
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            row["excluded"] = reason_for(self.exclusions, row["frame_file"],
                                         row.get("plot") or "main")
            for j, (key, _) in enumerate(COLUMNS):
                value = str(row.get(key, "") or "")
                if key == "timestamp_utc":
                    value = _local_time(value)
                item = QTableWidgetItem(value)
                if key == "flags" and row.get("flags"):
                    item.setForeground(Qt.darkRed)
                if row["excluded"]:
                    font = QFont(item.font())
                    font.setStrikeOut(key != "excluded")
                    item.setFont(font)
                    item.setForeground(Qt.gray if key != "excluded" else Qt.darkRed)
                self.table.setItem(i, j, item)
        self.table.blockSignals(False)
        if selected is not None and selected in rows:
            self.table.selectRow(rows.index(selected))

    def selected_row(self) -> dict | None:
        chosen = self.table.selectionModel().selectedRows() if self.table.model() else []
        if not chosen or chosen[0].row() >= len(self.visible_rows):
            return None
        return self.visible_rows[chosen[0].row()]

    def _row_selected(self) -> None:
        exp = self.state.experiment
        row = self.selected_row()
        for button in (self.exclude_btn, self.include_btn):
            button.setEnabled(row is not None)
        if exp is None or row is None:
            return
        self.include_btn.setEnabled(bool(row.get("excluded")))
        mode = self.view_mode.currentText()
        self.overlay.set_mask(None)
        info = [f"{row['frame_file']}"]
        if row.get("excluded"):
            info.append(f"<b>Excluded by hand:</b> {row['excluded']}")
        if row.get("flags"):
            info.append(f"Flags: {row['flags']}")
        self.frame_info.setText("<br>".join(info))
        if mode == "Overlay" and row.get("overlay_file"):
            self.overlay.set_image(cv2.imread(str(exp.root / row["overlay_file"])),
                                   keep_view=True)
        elif mode == "Photo as taken" or mode == "Overlay":
            self.overlay.set_image(cv2.imread(str(exp.root / row["frame_file"])), keep_view=True)
        else:
            self._show_aligned(row, with_mask=mode == "Aligned frame + mask")

    def _show_aligned(self, row: dict, with_mask: bool) -> None:
        exp = self.state.experiment
        self.frame_info.setText(self.frame_info.text() + "<br>Aligning frame…")
        cached = self._analyzer

        def work(progress, should_stop):
            from fungus_cv.analyze.pipeline import Analyzer

            analyzer = cached or Analyzer(exp, with_segmenter=False, require_annotations=False)
            index = next(i for i, r in enumerate(analyzer.frames)
                         if r["file"] == row["frame_file"])
            mask = None
            if with_mask and row.get("mask_file"):
                img = cv2.imread(str(exp.root / row["mask_file"]), cv2.IMREAD_GRAYSCALE)
                mask = None if img is None else img > 127
            return analyzer, row, analyzer.aligned_frame(index), mask

        run_task(work, self._aligned_ready, self._failed)

    def _aligned_ready(self, result) -> None:
        analyzer, row, frame, mask = result
        self._analyzer = analyzer
        if row is not self.selected_row():
            return  # the user moved on
        self.overlay.set_image(frame, keep_view=True)
        self.overlay.set_mask(mask if mask is not None and mask.shape == frame.shape[:2]
                              else None)
        self.frame_info.setText(self.frame_info.text().replace("<br>Aligning frame…", ""))

    # --- excluding frames by hand ------------------------------------------------------------

    def _ask_exclude(self) -> None:
        row = self.selected_row()
        if row is None:
            return
        reason, ok = QInputDialog.getText(
            self, "Exclude frame", "Why leave this frame out of fits? (recorded with the "
            "results)", text=row.get("excluded") or "")
        if ok:
            self.exclude_selected(reason)

    def exclude_selected(self, reason: str, all_plots: bool = True) -> None:
        from fungus_cv.analyze import exclusions

        exp, row = self.state.experiment, self.selected_row()
        if exp is None or row is None:
            return
        try:
            exclusions.exclude(exp, row["frame_file"], reason,
                               "" if all_plots else (row.get("plot") or "main"))
        except ValueError as exc:
            self.message.setText(f"<span style='color:#b00020'>{exc}</span>")
            return
        self.message.setText(f"Excluded {row['frame_file']}: {reason.strip()}")
        self._reload_exclusions()

    def include_selected(self) -> None:
        from fungus_cv.analyze import exclusions

        exp, row = self.state.experiment, self.selected_row()
        if exp is None or row is None:
            return
        exclusions.include(exp, row["frame_file"])
        self.message.setText(f"{row['frame_file']} is included again.")
        self._reload_exclusions()

    def _reload_exclusions(self) -> None:
        from fungus_cv.analyze import exclusions

        self.exclusions = exclusions.load(self.state.experiment)
        self._fill_table()
        self._row_selected()

    def shutdown(self) -> None:
        self.cancel()
