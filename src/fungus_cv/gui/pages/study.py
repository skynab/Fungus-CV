"""Compare conditions across replicate experiments: edit a study, run it, read the results."""

from __future__ import annotations

import math
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.gui.chart import ChartLabel
from fungus_cv.gui.pages.report import METRICS
from fungus_cv.gui.qt_util import run_task

EXPERIMENT_COLUMNS = [("path", "Experiment folder"), ("condition", "Condition"),
                      ("replicate", "Replicate"), ("plot", "Plot"), ("t0", "Start time (t0)")]


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def _fill(table: QTableWidget, headers: list[str], rows: list[list]) -> None:
    table.clear()
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.setItem(r, c, QTableWidgetItem(_fmt(value)))
    table.resizeColumnsToContents()


def _readonly_table() -> QTableWidget:
    table = QTableWidget()
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    return table


class StudyPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.path: Path | None = None
        self.result = None
        self.task = None

        self.path_label = QLabel("No study open.")
        self.path_label.setWordWrap(True)
        open_btn = QPushButton("Open study…")
        open_btn.clicked.connect(self._choose_open)
        new_btn = QPushButton("New study…")
        new_btn.clicked.connect(self._choose_new)

        self.name = QLineEdit()
        self.metric = QComboBox()
        self.metric.setEditable(True)
        self.metric.addItems(METRICS)
        self.model = QComboBox()
        self.also_fit = QLineEdit()
        self.also_fit.setPlaceholderText("e.g. gompertz, richards (model selection only)")
        self.params = QLineEdit()
        self.reference = QComboBox()
        self.reference.setEditable(True)
        self.time_unit = QComboBox()
        self.time_unit.addItems(["auto", "s", "min", "h", "d"])
        self.errors = QComboBox()
        self.errors.addItems(["auto", "iid", "ar1"])
        self.bootstrap = QSpinBox()
        self.bootstrap.setRange(0, 100000)
        self.bootstrap.setValue(1000)
        self.exclude_jumps = QCheckBox("Leave jumps out of the fits")
        from fungus_cv.analyze.fit import MODELS

        self.model.addItems(list(MODELS))
        self.model.setCurrentText("logistic")
        self.model.currentTextChanged.connect(self._model_changed)
        self._model_changed(self.model.currentText())

        settings = QGroupBox("Study")
        form = QFormLayout(settings)
        form.addRow("Name", self.name)
        form.addRow("Measurement", self.metric)
        form.addRow("Model", self.model)
        form.addRow("Compare parameters", self.params)
        form.addRow("Also fit", self.also_fit)
        form.addRow("Reference condition", self.reference)
        form.addRow("Time unit", self.time_unit)
        form.addRow("Frame errors", self.errors)
        form.addRow("Bootstrap refits", self.bootstrap)
        form.addRow("", self.exclude_jumps)

        self.experiments = QTableWidget(0, len(EXPERIMENT_COLUMNS))
        self.experiments.setHorizontalHeaderLabels([c[1] for c in EXPERIMENT_COLUMNS])
        self.experiments.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.experiments.itemChanged.connect(lambda _: self._conditions_changed())
        add_open = QPushButton("Add open experiment")
        add_open.clicked.connect(self.add_open_experiment)
        add_folder = QPushButton("Add experiment folder…")
        add_folder.clicked.connect(self._choose_experiment)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_selected)
        self.save_btn = QPushButton("Save study")
        self.save_btn.clicked.connect(self.save)
        self.run_btn = QPushButton("Run study")
        self.run_btn.clicked.connect(self.run)
        self.folder_btn = QPushButton("Open results folder")
        self.folder_btn.clicked.connect(self._open_results)
        self.folder_btn.setEnabled(False)
        self.message = QLabel("Open or create a study. Each experiment is one replicate.")
        self.message.setWordWrap(True)

        self.conditions = _readonly_table()
        self.comparisons = _readonly_table()
        self.replicates = _readonly_table()
        self.curves = ChartLabel()
        self.parameters = ChartLabel()
        self.methods = QPlainTextEdit()
        self.methods.setReadOnly(True)
        copy_btn = QPushButton("Copy methods text")
        copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText(
            self.methods.toPlainText()))
        methods_tab = QWidget()
        ml = QVBoxLayout(methods_tab)
        ml.addWidget(self.methods, 1)
        ml.addWidget(copy_btn)
        self.warnings = QListWidget()
        self.warnings.setWordWrap(True)
        self.tabs = QTabWidget()
        for widget, title in ((self.conditions, "Conditions"), (self.comparisons, "Comparisons"),
                              (self.replicates, "Replicates"), (self.curves, "Curves"),
                              (self.parameters, "Parameters"), (methods_tab, "Methods"),
                              (self.warnings, "Warnings")):
            self.tabs.addTab(widget, title)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.path_label)
        row = QHBoxLayout()
        row.addWidget(open_btn)
        row.addWidget(new_btn)
        ll.addLayout(row)
        ll.addWidget(settings)
        ll.addWidget(QLabel("Replicates (double-click a cell to edit; t0 like "
                            "2026-09-20T14:05:00, empty = first photo)"))
        ll.addWidget(self.experiments, 1)
        row = QHBoxLayout()
        for b in (add_open, add_folder, remove):
            row.addWidget(b)
        ll.addLayout(row)
        row = QHBoxLayout()
        for b in (self.save_btn, self.run_btn, self.folder_btn):
            row.addWidget(b)
        ll.addLayout(row)
        ll.addWidget(self.message)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.tabs)
        splitter.setSizes([520, 700])
        QVBoxLayout(self).addWidget(splitter)
        self._conditions_changed()
        self._update_buttons()

    # --- editing ---------------------------------------------------------------------------

    def _model_changed(self, name: str) -> None:
        from fungus_cv.analyze.fit import MODELS

        if name in MODELS:
            self.params.setPlaceholderText(f"empty = all: {', '.join(MODELS[name].params)}")

    def _conditions_changed(self) -> None:
        current = self.reference.currentText()
        names = sorted({self._cell(r, 1) for r in range(self.experiments.rowCount())} - {""})
        self.reference.blockSignals(True)
        self.reference.clear()
        self.reference.addItems(["(all pairs)"] + names)
        self.reference.setCurrentText(current or "(all pairs)")
        self.reference.blockSignals(False)

    def _cell(self, row: int, col: int) -> str:
        item = self.experiments.item(row, col)
        return item.text().strip() if item else ""

    def add_experiment(self, folder: Path, condition: str = "", replicate: str = "") -> None:
        base = self.path.parent if self.path else None
        folder = Path(folder).resolve()
        try:
            shown = str(folder.relative_to(base.resolve())) if base else str(folder)
        except ValueError:
            shown = str(folder)
        r = self.experiments.rowCount()
        self.experiments.insertRow(r)
        for c, value in enumerate([shown, condition, replicate, "", ""]):
            self.experiments.setItem(r, c, QTableWidgetItem(value))
        self._conditions_changed()

    def add_open_experiment(self) -> None:
        if self.state.experiment is not None:
            self.add_experiment(self.state.experiment.root)

    def _choose_experiment(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Experiment folder")
        if folder:
            self.add_experiment(Path(folder))

    def _remove_selected(self) -> None:
        for r in sorted({i.row() for i in self.experiments.selectedIndexes()}, reverse=True):
            self.experiments.removeRow(r)
        self._conditions_changed()

    def spec(self) -> dict:
        def listed(text):
            return [p.strip() for p in text.split(",") if p.strip()]

        reference = self.reference.currentText()
        experiments = []
        for r in range(self.experiments.rowCount()):
            entry = {"path": self._cell(r, 0), "condition": self._cell(r, 1)}
            for c, key in ((2, "replicate"), (3, "plot"), (4, "t0")):
                if self._cell(r, c):
                    entry[key] = self._cell(r, c)
            experiments.append(entry)
        return {
            "name": self.name.text().strip() or "study",
            "metric": self.metric.currentText().strip() or None,
            "model": self.model.currentText(),
            "also_fit": listed(self.also_fit.text()),
            "params": listed(self.params.text()) or None,
            "reference": None if reference in ("", "(all pairs)") else reference,
            "time_unit": self.time_unit.currentText(),
            "errors": self.errors.currentText(),
            "bootstrap": self.bootstrap.value(),
            "exclude_jumps": self.exclude_jumps.isChecked(),
            "experiments": experiments,
        }

    def save(self) -> bool:
        from fungus_cv.analyze.study import Study

        if self.path is None:
            self._choose_new()
            if self.path is None:
                return False
        spec = self.spec()
        try:
            Study.model_validate(spec)
        except ValueError as exc:
            self._error(f"Not saved: {exc}")
            return False
        text = ("# Fungus-CV study (`fungus study` or the Study page). Paths are relative to "
                "this file.\n" + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
        self.path.write_text(text, encoding="utf-8")
        self.message.setText(f"Saved {self.path.name}.")
        return True

    # --- files -----------------------------------------------------------------------------

    def _choose_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open study", "", "Study (*.yaml *.yml)")
        if path:
            self.open_study(Path(path))

    def _choose_new(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "New study file", "study.yaml",
                                              "Study (*.yaml)")
        if path:
            self.new_study(Path(path))

    def new_study(self, path: Path) -> None:
        self.path = Path(path)
        self.path_label.setText(f"<b>{self.path.name}</b> (not saved yet)<br>{self.path}")
        self.name.setText(self.path.stem)
        self.experiments.setRowCount(0)
        self._update_buttons()

    def open_study(self, path: Path) -> None:
        from fungus_cv.analyze.study import load_study

        try:
            study = load_study(path)
        except (OSError, ValueError) as exc:
            self._error(str(exc))
            return
        self.path = Path(path)
        self.path_label.setText(f"<b>{self.path.name}</b><br>{self.path}")
        self.name.setText(study.name)
        self.metric.setCurrentText(study.metric or "")
        self.model.setCurrentText(study.model)
        self.also_fit.setText(", ".join(study.also_fit))
        self.params.setText(", ".join(study.params or []))
        self.time_unit.setCurrentText(study.time_unit)
        self.errors.setCurrentText(study.errors)
        self.bootstrap.setValue(study.bootstrap)
        self.exclude_jumps.setChecked(study.exclude_jumps)
        self.experiments.blockSignals(True)
        self.experiments.setRowCount(0)
        for e in study.experiments:
            r = self.experiments.rowCount()
            self.experiments.insertRow(r)
            values = [e.path, e.condition, e.replicate or "", e.plot or "",
                      e.t0.isoformat() if e.t0 else ""]
            for c, value in enumerate(values):
                self.experiments.setItem(r, c, QTableWidgetItem(value))
        self.experiments.blockSignals(False)
        self._conditions_changed()
        self.reference.setCurrentText(study.reference or "(all pairs)")
        self.message.setText(f"{len(study.experiments)} replicate(s). Run the study to fit "
                             "and compare them.")
        self._update_buttons()

    # --- running ---------------------------------------------------------------------------

    def run(self) -> None:
        if not self.save():
            return
        import fungus_cv.analyze.study  # noqa: F401 - import on the main thread (see qt_util)

        path = self.path
        self.run_btn.setEnabled(False)
        self.message.setText("Fitting every replicate (with bootstrap intervals, this can take "
                             "a minute)…")
        def work(progress, should_stop):
            from fungus_cv.analyze.study import run_study

            return run_study(path)

        self.task = run_task(work, self._done, self._failed)

    def _failed(self, message: str) -> None:
        self.task = None
        self._update_buttons()
        self._error(message)

    def _done(self, result) -> None:
        from fungus_cv.analyze.fit import format_params

        self.task = None
        self.result = result
        self._update_buttons()
        _fill(self.conditions,
              ["Condition", "Parameter", "n", "Mean", "SD", "95% CI low", "95% CI high",
               "Random-effects mean", "Between-replicate SD"],
              [[c["condition"], c["param"], c["n"], c["mean"], c["sd"], c["ci_low"],
                c["ci_high"], c["re_mean"], c["tau"]] for c in result.conditions])
        _fill(self.comparisons,
              ["Parameter", "Condition", "versus", "Difference", "95% CI low", "95% CI high",
               "p", "Holm p", "Hedges' g"],
              [[c["param"], c["condition"], c["versus"], c["diff"], c["ci_low"], c["ci_high"],
                c["p"], c["p_holm"], c["hedges_g"]] for c in result.comparisons])
        _fill(self.replicates, ["Condition", "Replicate", "Parameters", "R²", "Frames", "Note"],
              [[r.condition, r.replicate, format_params(r.fit) if r.fit else "",
                r.fit.r2 if r.fit else None, r.fit.n if r.fit else None,
                r.error or ("" if r.fit else "fit failed")] for r in result.replicates])
        out = result.out_dir
        import cv2

        self.curves.set_image(cv2.imread(str(out / "curves.png")))
        self.parameters.set_image(cv2.imread(str(out / "parameters.png")))
        self.methods.setPlainText((out / "methods.md").read_text(encoding="utf-8"))
        self.warnings.clear()
        self.warnings.addItems(result.warnings or ["No warnings."])
        self.tabs.setTabText(6, f"Warnings ({len(result.warnings)})")
        self.message.setText(f"Done: {result.metric} with {result.study.model}, time in "
                             f"{result.time_unit}. Results in {out}.")
        self.tabs.setCurrentIndex(1)

    def _open_results(self) -> None:
        if self.result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.out_dir)))

    def _update_buttons(self) -> None:
        self.save_btn.setEnabled(self.path is not None)
        self.run_btn.setEnabled(self.path is not None and self.task is None)
        self.folder_btn.setEnabled(self.result is not None)

    def _error(self, message: str) -> None:
        self.message.setText(f"<span style='color:#b00020'>{message}</span>")
