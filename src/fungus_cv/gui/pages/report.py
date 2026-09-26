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
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.gui import theme
from fungus_cv.gui.image_view import ImageView
from fungus_cv.gui.qt_util import run_task

METRICS = ["extent_mm", "extent_max_mm", "covered_length_pct", "coverage_pct",
           "target_area_mm2", "equivalent_radius_mm", "edge_advance_p95_mm",
           "edge_advance_max_mm", "reference_covered_pct", "gcc_p90", "gcc_mean", "exg_mean",
           "extent_px"]


SPREAD_HINT = (
    "<b>Nothing here yet.</b> Press <b>Spread map</b> (left) to map when each point was first "
    "covered and how fast the edge moved in each direction.<br><br>It works on the masks of "
    "any Analyze run, whether the target came from colour thresholds, SAM 2 or a trained "
    "model. It says most about a patch spreading over a surface (a colony, a stain); for a "
    "front rising along a strip, the Measurement tab already tells the story.")
SENSITIVITY_HINT = (
    "<b>Nothing here yet.</b> Press <b>Sensitivity check</b> (left) to re-run Analyze with each "
    "questionable setting changed a little (thresholds, alignment, lighting, where the front "
    "is taken) and see how far the result moves.<br><br>It works with any segmentation "
    "method; for SAM 2 it varies the mask threshold. It is slow: one full analysis per "
    "variant.")
CONDITIONS_HINT = (
    "<b>Nothing here yet.</b> Choose a logged condition under <b>Conditions</b> (left) and press "
    "<b>Make report</b> to draw it under the measurement, with its average over the fitted "
    "period.")
NO_CONDITIONS_HINT = (
    "<b>No conditions are logged for this experiment.</b> Temperature, humidity and light "
    "change how fast things grow. Press <b>Import log…</b> next to Conditions to add a data "
    "logger's CSV file, then choose it and press <b>Make report</b>.")


class ResultTab(QStackedWidget):
    """A chart, or until there is one, a note saying what fills it."""

    def __init__(self, hint: str):
        super().__init__()
        self.hint = QLabel(hint)
        self.hint.setWordWrap(True)
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setMargin(40)
        self.view = ImageView()
        self.addWidget(self.hint)
        self.addWidget(self.view)

    def set_hint(self, text: str) -> None:
        self.hint.setText(text)
        self.view.set_image(None)
        self.setCurrentWidget(self.hint)

    def set_image(self, image) -> None:
        self.view.set_image(image)
        self.setCurrentWidget(self.view)

    @property
    def has_image(self) -> bool:
        return self.currentWidget() is self.view


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
        self.hours = QLineEdit()
        self.hours.setPlaceholderText("all day (e.g. 10-14 for outdoors)")
        self.hours.setToolTip("Only frames taken in these local hours (the experiment's "
                              "timezone), to leave out night and low sun.")
        self.daily = QComboBox()
        self.daily.addItems(["every frame", "median", "mean", "p90"])
        self.daily.setToolTip("One value per day from its frames (p90 = 90th percentile, "
                              "usual for greenness).")
        self.covariate = QComboBox()
        self.covariate.setToolTip("Draw a logged condition (temperature, humidity…) under "
                                  "the measurement and summarise it over the fitted period.")
        self.import_btn = QPushButton("Import log…")
        self.import_btn.setToolTip("Add a data logger's CSV file (temperature, humidity, "
                                   "light…) to this experiment.")
        self.import_btn.clicked.connect(self.import_conditions)
        self.include_flagged = QCheckBox("Fit flagged frames too")
        self.exclude_jumps = QCheckBox("Leave jumps out of the fits")
        self.vector = QCheckBox("Also save PDF and SVG (for papers)")
        self.video = QCheckBox("Also make an overlay video")
        self.make_btn = QPushButton("Make report")
        theme.mark_primary(self.make_btn)
        self.make_btn.clicked.connect(self.make)
        self.folder_btn = QPushButton("Open report folder")
        self.folder_btn.clicked.connect(self.open_folder)
        form.addRow("Plot", self.plot)
        form.addRow("Measurement", self.metric)
        form.addRow(self.custom_t0, self.t0)
        form.addRow("Time unit", self.time_unit)
        form.addRow("Hours of the day", self.hours)
        form.addRow("Per day", self.daily)
        conditions = QHBoxLayout()
        conditions.addWidget(self.covariate, 1)
        conditions.addWidget(self.import_btn)
        form.addRow("Conditions", conditions)
        form.addRow("", self.include_flagged)
        form.addRow("", self.exclude_jumps)
        form.addRow("", self.vector)
        form.addRow("", self.video)
        buttons = QHBoxLayout()
        buttons.addWidget(self.make_btn)
        buttons.addWidget(self.folder_btn)
        form.addRow(buttons)
        # More analyses of the same results.
        self.spread_btn = QPushButton("Spread map")
        self.spread_btn.setToolTip("When each point was first covered, and how fast the front "
                                   "moved in each direction (uses coverage).")
        self.spread_btn.clicked.connect(self.make_spread)
        self.sensitivity_btn = QPushButton("Sensitivity check")
        self.sensitivity_btn.setToolTip("Re-measure with each questionable setting changed a "
                                        "little and show how much the result moves. Slow: it "
                                        "re-runs the analysis once per variant.")
        self.sensitivity_btn.clicked.connect(self.make_sensitivity)
        self.summary_btn = QPushButton("Shareable page")
        self.summary_btn.setToolTip("One self-contained HTML page with every plot's figures, "
                                    "fits, intervals, exclusions and settings.")
        self.summary_btn.clicked.connect(self.make_summary)
        more = QHBoxLayout()
        for button in (self.spread_btn, self.sensitivity_btn, self.summary_btn):
            more.addWidget(button)
        form.addRow(more)
        self.extra_buttons = (self.spread_btn, self.sensitivity_btn, self.summary_btn)
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
        # Filled by their own buttons (or a chosen condition), not by Make report.
        self.spread_view = ResultTab(SPREAD_HINT)
        self.sensitivity_view = ResultTab(SENSITIVITY_HINT)
        self.tabs.addTab(self.spread_view, "Spread")
        self.tabs.addTab(self.sensitivity_view, "Sensitivity")
        self.covariate_view = ResultTab(CONDITIONS_HINT)
        self.tabs.addTab(self.covariate_view, "Conditions")

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

        state.experiment_changed.connect(lambda _: self._new_experiment())

    def _new_experiment(self) -> None:
        """Charts of the previous experiment would be mistaken for this one's."""
        self.result = None
        self.fits.setRowCount(0)
        self.chart.set_image(None)
        self.quality.set_image(None)
        self.spread_view.set_hint(SPREAD_HINT)
        self.sensitivity_view.set_hint(SENSITIVITY_HINT)
        self.covariate_view.set_hint(CONDITIONS_HINT)
        self.refresh()

    def on_shown(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        exp = self.state.experiment
        self.plot.clear()
        self.metric.clear()
        self.covariate.clear()
        self.covariate.addItem("(none)")
        self.make_btn.setEnabled(False)
        for button in self.extra_buttons:
            button.setEnabled(False)
        self.import_btn.setEnabled(exp is not None)
        if exp is None:
            self.message.setText("Open an experiment first.")
            return
        from fungus_cv.analyze import covariates

        logged = covariates.load(exp).names
        self.covariate.addItems(logged)
        if not self.covariate_view.has_image:
            self.covariate_view.set_hint(CONDITIONS_HINT if logged else NO_CONDITIONS_HINT)
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
        available += [c for c in (rows[0] if rows else {})  # colour classes
                      if c.startswith("class_") and c.endswith("_pct")]
        self.metric.addItems(available)
        self.make_btn.setEnabled(True)
        for button in self.extra_buttons:
            button.setEnabled(True)
        self.message.setText("")

    def make(self) -> None:
        exp = self.state.experiment
        if exp is None:
            return
        from fungus_cv.analyze.report import DEFAULT_EXCLUDE, make_report, parse_hours

        try:
            hours = parse_hours(self.hours.text()) if self.hours.text().strip() else None
        except ValueError as exc:
            self._failed(str(exc))
            return

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
            hours=hours,
            daily=None if self.daily.currentIndex() == 0 else self.daily.currentText(),
            covariates=() if self.covariate.currentIndex() <= 0
            else (self.covariate.currentText(),),
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
        unit = f"days (daily {result.daily})" if result.daily else "frames"
        outside = (f" ({result.outside_hours} frames outside the hours)"
                   if result.outside_hours else "")
        self.message.setText(f"{result.n_used} {unit} used, {result.n_excluded} excluded"
                             f"{outside}, "
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
            elif name.endswith("covariates.png"):
                self.covariate_view.set_image(cv2.imread(name))
        for name, c in result.covariates.items():
            self.message.setText(self.message.text() + f" {name}: mean {c['mean']:.3g} "
                                 f"(logged for {100 * c['coverage']:.0f}% of the period).")
        self.tabs.setCurrentIndex(0)

    def import_conditions(self) -> None:
        exp = self.state.experiment
        if exp is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Data logger file", str(exp.root),
                                              "Logger files (*.csv *.txt *.tsv);;All files (*)")
        if path:
            self.import_log(path)

    def import_log(self, path) -> None:
        from pathlib import Path

        from fungus_cv.analyze import covariates

        try:
            summary = covariates.import_log(self.state.experiment, Path(path))
        except (OSError, ValueError, KeyError) as exc:
            self._failed(f"Could not import {Path(path).name}: {exc}")
            return
        chosen = self.covariate.currentText()
        self.refresh()
        self.covariate.setCurrentText(chosen if chosen != "(none)" else summary.columns[0])
        self.message.setText(f"Imported {summary.rows} rows of {', '.join(summary.columns)} "
                             f"({summary.first_utc} to {summary.last_utc} UTC). Press Make "
                             "report to draw them.")

    # --- spread, sensitivity, shareable page ---------------------------------------------

    def _busy(self, busy: bool, text: str = "") -> None:
        for button in (self.make_btn, *self.extra_buttons):
            button.setEnabled(not busy)
        if text:
            self.message.setText(text)

    def _extra_failed(self, message: str) -> None:
        self._busy(False)
        self._failed(message)

    def make_spread(self) -> None:
        exp = self.state.experiment
        if exp is None:
            return
        from fungus_cv.analyze.spread import spread

        plot, camera = self.plot.currentText() or None, self.state.camera
        self._busy(True, "Mapping when each point was reached…")
        run_task(lambda progress, stop: spread(exp, plot=plot, camera=camera),
                 self._spread_done, self._extra_failed)

    def _spread_done(self, result) -> None:
        self._busy(False)
        unit = f"{result.unit}/{result.time_unit}"
        fastest = max((s for s in result.sectors if s.speed == s.speed), default=None,
                      key=lambda s: s.speed)
        text = f"Spread: mean front speed {result.mean_speed:.3g} {unit}"
        if fastest is not None:
            text += (f", fastest toward {fastest.angle_deg:.0f}° ({fastest.speed:.3g} ± "
                     f"{fastest.se:.2g}); fastest/slowest {result.anisotropy:.2f} "
                     "(1 = even)")
        self.message.setText(text + ". 0° = right, 90° = up.")
        for path in result.files:
            if str(path).endswith("spread_map.png"):
                self.spread_view.set_image(cv2.imread(str(path)))
                self.tabs.setCurrentWidget(self.spread_view)

    def make_sensitivity(self) -> None:
        exp = self.state.experiment
        if exp is None:
            return
        from fungus_cv.analyze import sensitivity

        metric = self.metric.currentText() or None
        plot = self.plot.currentText() or None
        self._busy(True, "Re-measuring with each setting changed…")
        run_task(lambda progress, stop: sensitivity.run(
                     exp, metric=metric, plot=plot,
                     progress=lambda n, total, name: progress((n, total, name))),
                 self._sensitivity_done, self._extra_failed, self._sensitivity_progress)

    def _sensitivity_progress(self, value) -> None:
        n, total, name = value
        self.message.setText(f"Sensitivity: variant {n} of {total} ({name})…")

    def _sensitivity_done(self, result) -> None:
        self._busy(False)
        largest = result.largest
        text = f"Sensitivity of {result.metric}: {len(result.variants)} variant(s) run"
        if largest is not None:
            text += (f"; the largest mean change, {largest.mean_abs_change:.3g}, came from "
                     f"{largest.variant.description}")
            if largest.change_in_uncertainties == largest.change_in_uncertainties:
                text += (f" ({largest.change_in_uncertainties:.2g}× the reported "
                         "uncertainty)")
        failed = [v.variant.name for v in result.variants if not v.ok]
        if failed:
            text += f". Failed: {', '.join(failed)}"
        self.message.setText(text + ".")
        for path in result.files:
            if str(path).endswith(".png"):
                self.sensitivity_view.set_image(cv2.imread(str(path)))
                self.tabs.setCurrentWidget(self.sensitivity_view)
                break

    def make_summary(self) -> None:
        exp = self.state.experiment
        if exp is None:
            return
        from fungus_cv.analyze.summary import make_summary

        metric = self.metric.currentText() or None
        results_dir = self.state.results_dir()
        self._busy(True, "Fitting every plot and writing the page…")
        run_task(lambda progress, stop: make_summary(exp, metric=metric,
                                                     results_dir=results_dir),
                 self._summary_done, self._extra_failed)

    def _summary_done(self, result) -> None:
        self._busy(False)
        problems = f" Not fitted: {'; '.join(result.problems)}." if result.problems else ""
        self.message.setText(f"Wrote {result.path.name} in {result.path.parent}.{problems}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(result.path)))

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
