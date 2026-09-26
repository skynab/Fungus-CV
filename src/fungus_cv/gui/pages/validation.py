"""Check accuracy: agreement with hand measurements, and the validation suite."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.gui import theme
from fungus_cv.gui.chart import ChartLabel
from fungus_cv.gui.pages.report import METRICS, chosen_metric, fill_metrics
from fungus_cv.gui.qt_util import preload_model_modules, run_task

STATUS_COLORS = {"pass": QColor(theme.GOOD), "fail": QColor(theme.BAD),
                 "error": QColor(theme.BAD), "info": QColor(theme.MUTED)}


def _file_row(line: QLineEdit, title: str, pattern: str) -> QHBoxLayout:
    browse = QPushButton("Browse…")

    def choose():
        path, _ = QFileDialog.getOpenFileName(None, title, line.text(), pattern)
        if path:
            line.setText(path)

    browse.clicked.connect(choose)
    row = QHBoxLayout()
    row.addWidget(line, 1)
    row.addWidget(browse)
    return row


def _num(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.4g}"


class ValidationPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.agreement = None
        self.suite_result = None

        # --- hand measurements -----------------------------------------------------------
        self.hand_csv = QLineEdit()
        self.hand_csv.setPlaceholderText("CSV with frame and value columns")
        self.metric = QComboBox()
        fill_metrics(self.metric, METRICS)
        self.plot = QLineEdit()
        self.plot.setPlaceholderText("all (field experiments: one plot name)")
        self.template_count = QSpinBox()
        self.template_count.setRange(3, 1000)
        self.template_count.setValue(20)
        template_btn = QPushButton("Make template…")
        template_btn.clicked.connect(self._choose_template)
        self.validate_btn = QPushButton("Compare with hand measurements")
        self.validate_btn.clicked.connect(self.validate)
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.agreement_chart = ChartLabel("Bland–Altman plot appears here.")
        hand_box = QGroupBox("Hand measurements (open experiment)")
        hf = QFormLayout(hand_box)
        hf.addRow("Hand CSV", _file_row(self.hand_csv, "Hand measurements", "CSV (*.csv)"))
        hf.addRow("Measurement", self.metric)
        hf.addRow("Plot", self.plot)
        row = QHBoxLayout()
        row.addWidget(QLabel("Frames:"))
        row.addWidget(self.template_count)
        row.addWidget(template_btn)
        hf.addRow("New template", row)
        hf.addRow(self.validate_btn)
        hf.addRow(self.stats)

        # --- suite -----------------------------------------------------------------------
        self.suite_path = QLineEdit()
        self.suite_path.setPlaceholderText("suite.yaml")
        new_suite = QPushButton("New suite…")
        new_suite.clicked.connect(self._choose_new_suite)
        self.save_baseline = QCheckBox("Save these numbers as the baseline")
        self.suite_btn = QPushButton("Run validation suite")
        theme.mark_primary(self.suite_btn)
        self.suite_btn.clicked.connect(self.run_suite)
        self.suite_summary = QLabel()
        self.suite_summary.setWordWrap(True)
        self.suite_table = QTableWidget(0, 7)
        self.suite_table.setHorizontalHeaderLabels(["Case / check", "Metric", "Value",
                                                    "Baseline", "Drift", "Status", "Detail"])
        self.suite_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.suite_table.verticalHeader().setVisible(False)
        self.suite_table.horizontalHeader().setStretchLastSection(True)
        self.only_checked = QCheckBox("Only checked metrics")
        self.only_checked.setChecked(True)
        self.only_checked.toggled.connect(lambda _: self._fill_suite())
        suite_box = QGroupBox("Validation suite (re-run every accuracy check)")
        sl = QVBoxLayout(suite_box)
        row = QHBoxLayout()
        row.addLayout(_file_row(self.suite_path, "Validation suite", "Suite (*.yaml *.yml)"), 1)
        row.addWidget(new_suite)
        sl.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.suite_btn)
        row.addWidget(self.save_baseline)
        row.addStretch()
        row.addWidget(self.only_checked)
        sl.addLayout(row)
        sl.addWidget(self.suite_summary)
        sl.addWidget(self.suite_table, 1)

        self.message = QLabel()
        self.message.setWordWrap(True)
        top = QSplitter(Qt.Horizontal)
        top.addWidget(hand_box)
        top.addWidget(self.agreement_chart)
        top.setSizes([480, 640])
        vertical = QSplitter(Qt.Vertical)
        vertical.addWidget(top)
        vertical.addWidget(suite_box)
        layout = QVBoxLayout(self)
        layout.addWidget(self.message)
        layout.addWidget(vertical, 1)

    # --- hand measurements -----------------------------------------------------------------

    def _choose_template(self) -> None:
        exp = self.state.experiment
        start = str(exp.root / "hand_measurements.csv") if exp else "hand_measurements.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save template", start, "CSV (*.csv)")
        if path:
            self.make_template(Path(path))

    def make_template(self, path: Path) -> None:
        from fungus_cv.analyze.validate import write_template

        exp = self.state.experiment
        if exp is None:
            self._error("Open an analyzed experiment first.")
            return
        try:
            if path.exists():
                raise ValueError(f"{path} already exists")
            n = write_template(exp, path, self.template_count.value(),
                               self.state.results_dir())
        except (OSError, ValueError) as exc:
            self._error(str(exc))
            return
        self.hand_csv.setText(str(path))
        self.message.setText(f"Wrote {path.name} with {n} row(s). Measure those frames by hand "
                             "(e.g. calipers or ImageJ), fill in 'value' (and optionally "
                             "'value_unc'), then compare.")

    def validate(self) -> None:
        exp = self.state.experiment
        if exp is None:
            self._error("Open an analyzed experiment first.")
            return
        hand, metric = Path(self.hand_csv.text()), chosen_metric(self.metric)
        plot = self.plot.text().strip() or None
        results_dir = self.state.results_dir()
        self.validate_btn.setEnabled(False)

        def work(progress, should_stop):
            from fungus_cv.analyze.validate import validate

            return validate(exp, hand, metric, plot, results_dir)

        run_task(work, self._validated, self._validate_failed)

    def _validate_failed(self, message: str) -> None:
        self.validate_btn.setEnabled(True)
        self._error(message)

    def _validated(self, a) -> None:
        self.validate_btn.setEnabled(True)
        self.agreement = a
        lines = [
            f"<b>{a.metric}</b>, n = {a.n} frames",
            f"Bias (automatic − hand): <b>{a.bias:+.4g}</b> (95% CI {a.bias_ci95[0]:+.4g} to "
            f"{a.bias_ci95[1]:+.4g})",
            f"95% limits of agreement: {a.limits_of_agreement[0]:+.4g} to "
            f"{a.limits_of_agreement[1]:+.4g}",
            f"MAE {a.mae:.4g} · RMSE {a.rmse:.4g} · r {a.pearson_r:.4f} · automatic = "
            f"{a.slope:.4f} × hand {a.intercept:+.4g}",
            f"Largest difference {a.worst_diff:+.4g} at {a.worst_frame}",
        ]
        if not math.isnan(a.within_2u):
            lines.append(f"Uncertainty check: {100 * a.within_2u:.0f}% within 2u (expect ~95%), "
                         f"RMS z {a.z_rms:.2f} (expect ~1)")
        if a.unmatched:
            lines.append(f"{len(a.unmatched)} hand row(s) not matched: "
                         f"{', '.join(a.unmatched[:5])}")
        self.stats.setText("<br>".join(lines))
        png = next((f for f in a.files if str(f).endswith(".png")), None)
        self.agreement_chart.set_image(cv2.imread(str(png)) if png else None)
        self.message.setText(f"Paired values and the plot are in {Path(a.files[0]).parent}.")

    # --- suite -----------------------------------------------------------------------------

    def _choose_new_suite(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "New validation suite", "suite.yaml",
                                              "Suite (*.yaml)")
        if path:
            self.new_suite(Path(path))

    def new_suite(self, path: Path) -> None:
        from fungus_cv.analyze.suite import write_template

        try:
            write_template(path)
        except FileExistsError as exc:
            self._error(str(exc))
            return
        self.suite_path.setText(str(path))
        self.message.setText(f"Wrote {path.name}: point it at your experiments, hand labels and "
                             "models (open it in a text editor), then run it.")

    def run_suite(self) -> None:
        from fungus_cv.analyze.suite import load_suite

        path = Path(self.suite_path.text())
        try:
            suite = load_suite(path)
        except (OSError, ValueError) as exc:
            self._error(str(exc))
            return
        if any(c.type == "model" for case in suite.cases for c in case.checks):
            preload_model_modules({"model"})
            import fungus_cv.learn.train  # noqa: F401 - import on the main thread
        import fungus_cv.analyze.suite  # noqa: F401
        save_baseline = self.save_baseline.isChecked()
        self.suite_btn.setEnabled(False)
        self.message.setText("Running the suite (experiments are analyzed first)…")

        def work(progress, should_stop):
            from fungus_cv.analyze import suite as suite_mod

            result = suite_mod.run_suite(path, require_baseline=not save_baseline)
            baseline = suite_mod.save_baseline(result, path) if save_baseline else None
            return result, baseline

        run_task(work, self._suite_done, self._suite_failed)

    def _suite_failed(self, message: str) -> None:
        self.suite_btn.setEnabled(True)
        self._error(message)

    def _suite_done(self, result) -> None:
        result, baseline = result
        self.suite_btn.setEnabled(True)
        self.suite_result = result
        self._fill_suite()
        n_fail = sum(not c.passed for c in result.checks)
        colour = "#14823c" if not n_fail else "#b00020"
        text = (f"<span style='color:{colour}'><b>{len(result.checks) - n_fail}/"
                f"{len(result.checks)} checks passed</b></span>")
        if not result.baseline_found and baseline is None:
            text += " · no baseline yet (tick 'Save these numbers as the baseline' once the " \
                    "results look right)"
        if baseline is not None:
            text += f" · baseline saved to {baseline.name}"
        self.suite_summary.setText(text + f" · results in {result.out_dir}")
        self.message.setText("")

    def _fill_suite(self) -> None:
        result = self.suite_result
        if result is None:
            return
        rows = []
        for c in result.checks:
            name = f"{c.case} / {c.check}"
            if c.error:
                rows.append([name, "", "", "", "", "error", c.error])
            for m in c.metrics:
                if self.only_checked.isChecked() and m.status == "info":
                    continue
                rows.append([name, m.metric, _num(m.value), _num(m.baseline), _num(m.drift),
                             m.status, "; ".join(m.failures)])
        self.suite_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for col, value in enumerate(row):
                item = QTableWidgetItem(value)
                if col == 5:
                    item.setForeground(STATUS_COLORS.get(value, QColor(theme.TEXT)))
                self.suite_table.setItem(r, col, item)
        self.suite_table.resizeColumnsToContents()

    def _error(self, message: str) -> None:
        self.message.setText(f"<span style='color:#b00020'>{message}</span>")
