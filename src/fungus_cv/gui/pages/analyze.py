"""Run the analysis and browse per-frame results with overlays."""

from __future__ import annotations

import csv

import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
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
           ("flags", "Flags")]


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

        top = QHBoxLayout()
        top.addWidget(self.run_btn)
        top.addWidget(self.cancel_btn)
        top.addWidget(self.force)
        top.addWidget(self.progress, 1)
        top.addWidget(QLabel("Plot:"))
        top.addWidget(self.plot_filter)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(self.overlay)
        splitter.setSizes([640, 560])
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.message)
        layout.addWidget(splitter, 1)

        state.experiment_changed.connect(lambda _: self.load_results())
        state.busy_changed.connect(lambda busy: self.run_btn.setEnabled(
            not busy and self.task is None and state.experiment is not None))

    def on_shown(self) -> None:
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
        exp = self.state.experiment
        self.rows = []
        self.overlay.set_image(None)
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
        rows = [r for r in self.rows if wanted in ("", "all") or (r.get("plot") or "main") ==
                wanted]
        self.visible_rows = rows
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, (key, _) in enumerate(COLUMNS):
                value = str(row.get(key, "") or "")
                if key == "timestamp_utc":
                    value = _local_time(value)
                item = QTableWidgetItem(value)
                if key == "flags" and row.get("flags"):
                    item.setForeground(Qt.darkRed)
                self.table.setItem(i, j, item)

    def _row_selected(self) -> None:
        exp = self.state.experiment
        selected = self.table.selectionModel().selectedRows()
        if exp is None or not selected:
            return
        row = self.visible_rows[selected[0].row()]
        file = row.get("overlay_file") or row.get("frame_file")
        image = cv2.imread(str(exp.root / file)) if file else None
        self.overlay.set_image(image, keep_view=True)

    def shutdown(self) -> None:
        self.cancel()
