"""Fit growth models and show the report charts."""

from __future__ import annotations

from datetime import datetime

import cv2
from PySide6.QtCore import QDateTime, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.gui.image_view import ImageView
from fungus_cv.gui.qt_util import run_task

METRICS = ["extent_mm", "extent_max_mm", "covered_length_pct", "coverage_pct",
           "target_area_mm2", "equivalent_radius_mm", "edge_advance_p95_mm",
           "edge_advance_max_mm", "reference_covered_pct", "gcc_p90", "gcc_mean", "exg_mean",
           "extent_px"]


class ReportPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.result = None

        options = QGroupBox("Report")
        form = QFormLayout(options)
        self.plot = QComboBox()
        self.metric = QComboBox()
        self.metric.setEditable(True)
        self.custom_t0 = QCheckBox("Set start time (t = 0)")
        self.t0 = QDateTimeEdit(QDateTime.currentDateTime())
        self.t0.setCalendarPopup(True)
        self.t0.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.t0.setEnabled(False)
        self.custom_t0.toggled.connect(self.t0.setEnabled)
        self.time_unit = QComboBox()
        self.time_unit.addItems(["auto", "s", "min", "h", "d"])
        self.include_flagged = QCheckBox("Fit flagged frames too")
        self.exclude_jumps = QCheckBox("Leave jumps out of the fits")
        self.vector = QCheckBox("Also save PDF and SVG (for papers)")
        self.video = QCheckBox("Also make an overlay video")
        self.make_btn = QPushButton("Make report")
        self.make_btn.clicked.connect(self.make)
        self.folder_btn = QPushButton("Open report folder")
        self.folder_btn.clicked.connect(self.open_folder)
        form.addRow("Plot", self.plot)
        form.addRow("Measurement", self.metric)
        form.addRow(self.custom_t0, self.t0)
        form.addRow("Time unit", self.time_unit)
        form.addRow("", self.include_flagged)
        form.addRow("", self.exclude_jumps)
        form.addRow("", self.vector)
        form.addRow("", self.video)
        buttons = QHBoxLayout()
        buttons.addWidget(self.make_btn)
        buttons.addWidget(self.folder_btn)
        form.addRow(buttons)
        self.message = QLabel()
        self.message.setWordWrap(True)

        self.fits = QTableWidget(0, 5)
        self.fits.setHorizontalHeaderLabels(["Model", "Parameters (± SE) [95% CI]", "R²", "AICc",
                                             "Weight"])
        header = self.fits.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.fits.verticalHeader().setVisible(False)
        self.fits.setWordWrap(True)
        self.fits.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.tabs = QTabWidget()
        self.chart = ImageView()
        self.quality = ImageView()
        self.tabs.addTab(self.chart, "Measurement")
        self.tabs.addTab(self.quality, "Quality checks")

        left = QVBoxLayout()
        left.addWidget(options)
        left.addWidget(self.message)
        left.addWidget(QLabel("Model fits (weight = share of evidence; highest fits best)"))
        left.addWidget(self.fits, 1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(500)
        layout = QHBoxLayout(self)
        layout.addWidget(left_widget)
        layout.addWidget(self.tabs, 1)

        state.experiment_changed.connect(lambda _: self.refresh())

    def on_shown(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        exp = self.state.experiment
        self.plot.clear()
        self.metric.clear()
        self.make_btn.setEnabled(False)
        if exp is None:
            self.message.setText("Open an experiment first.")
            return
        try:
            from fungus_cv.analyze.report import list_plots, load_measurements

            results_dir = self.state.results_dir()
            plots = list_plots(exp, results_dir)
            rows = load_measurements(exp, results_dir=results_dir)
        except FileNotFoundError:
            self.message.setText("No analysis results yet: run Analyze first.")
            return
        self.plot.addItems(plots)
        available = [m for m in METRICS if m.startswith("edge_advance")
                     or any(r.get(m, "") != "" for r in rows[:50])]
        self.metric.addItems(available)
        self.make_btn.setEnabled(True)
        self.message.setText("")

    def make(self) -> None:
        exp = self.state.experiment
        if exp is None:
            return
        from fungus_cv.analyze.report import DEFAULT_EXCLUDE, make_report

        kwargs = dict(
            metric=self.metric.currentText() or None,
            plot=self.plot.currentText() or None,
            t0=self.t0.dateTime().toPython().astimezone() if self.custom_t0.isChecked()
            else None,
            time_unit=self.time_unit.currentText(),
            exclude_flags=() if self.include_flagged.isChecked() else DEFAULT_EXCLUDE,
            exclude_jumps=self.exclude_jumps.isChecked(),
            video=self.video.isChecked(),
            formats=("png", "pdf", "svg") if self.vector.isChecked() else ("png",),
            results_dir=self.state.results_dir(),
        )
        self.make_btn.setEnabled(False)
        self.message.setText("Fitting models and drawing charts…")
        run_task(lambda progress, stop: make_report(exp, **kwargs), self._done, self._failed)

    def _failed(self, message: str) -> None:
        self.make_btn.setEnabled(True)
        self.message.setText(f"<span style='color:#b00020'>{message}</span>")

    def _done(self, result) -> None:
        self.result = result
        self.make_btn.setEnabled(True)
        self.message.setText(f"{result.n_used} frames used, {result.n_excluded} excluded, "
                             f"{result.retreats} retreat(s), {result.jumps} jump(s) — see "
                             "frame_flags.csv. Time in " + result.time_unit + ".")
        from fungus_cv.analyze.fit import best_fit, format_derived, format_params

        best = best_fit(result.fits)
        self.fits.setRowCount(len(result.fits))
        for i, fit in enumerate(result.fits):
            params = "failed: " + fit.message if not fit.ok else format_params(fit)
            if fit.ok and format_derived(fit):
                params += "\n" + format_derived(fit)
            if fit.warnings:
                params += "\n⚠ " + "\n⚠ ".join(fit.warnings)
            cells = [fit.model, params, f"{fit.r2:.4f}" if fit.ok else "",
                     f"{fit.aicc:.1f}" if fit.ok else "",
                     f"{fit.akaike_weight:.2f}" + (" best" if fit is best else "")
                     if fit.ok else ""]
            for j, text in enumerate(cells):
                self.fits.setItem(i, j, QTableWidgetItem(text))
        self.fits.resizeRowsToContents()
        for path in result.files:
            name = str(path)
            if name.endswith("_vs_time.png"):
                self.chart.set_image(cv2.imread(name))
            elif name.endswith("quality_checks.png"):
                self.quality.set_image(cv2.imread(name))
        self.tabs.setCurrentIndex(0)

    def open_folder(self) -> None:
        exp = self.state.experiment
        if exp is None:
            return
        folder = exp.root / "results" / "report"
        if self.result is not None and self.result.files:
            folder = self.result.files[0].parent
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


__all__ = ["ReportPage", "datetime", "Qt"]
