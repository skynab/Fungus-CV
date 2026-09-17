"""Run a scheduled capture with live progress, optionally measuring frames as they arrive."""

from __future__ import annotations

import logging
import time

import cv2
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.config import parse_duration
from fungus_cv.gui.chart import ChartLabel
from fungus_cv.gui.image_view import ImageView
from fungus_cv.gui.pages.experiment import _fmt_seconds
from fungus_cv.gui.pages.report import METRICS
from fungus_cv.gui.qt_util import preload_model_modules, run_task
from fungus_cv.gui.theme import INK_2, SERIES

log = logging.getLogger(__name__)


class CaptureThread(QThread):
    finished_with = Signal(object)  # CaptureSummary
    failed = Signal(str)

    def __init__(self, experiment, keep_awake: bool):
        super().__init__()
        self.experiment = experiment
        self.keep_awake = keep_awake
        self.session = None

    def stop(self) -> None:
        if self.session is not None:
            self.session.stop()

    def run(self) -> None:
        from fungus_cv.capture.power import keep_awake
        from fungus_cv.capture.scheduler import CaptureSession

        try:
            self.session = CaptureSession(self.experiment)
            with keep_awake(self.keep_awake):
                summary = self.session.run()
            self.finished_with.emit(summary)
        except Exception as exc:  # show any failure in the window
            log.exception("capture failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CapturePage(QWidget):
    def __init__(self, state, log_handler=None):
        super().__init__()
        self.state = state
        self.thread: CaptureThread | None = None
        self.started_at = 0.0
        self._last_count = -1
        self.watch_task = None  # a live analysis in progress
        self._watch_again = False  # frames arrived while it ran
        self.watch_summary = None

        settings = QGroupBox("This run")
        form = QFormLayout(settings)
        self.interval = QLineEdit()
        self.duration = QLineEdit()
        self.duration.setPlaceholderText("empty = until stopped")
        self.max_frames = QSpinBox()
        self.max_frames.setRange(0, 10_000_000)
        self.max_frames.setSpecialValueText("no limit")
        self.keep_awake = QCheckBox("Keep the computer awake")
        form.addRow("Interval", self.interval)
        form.addRow("Duration", self.duration)
        form.addRow("Stop after rounds", self.max_frames)
        form.addRow("", self.keep_awake)
        form.addRow(QLabel("These override config.yaml for this run only."))

        self.start_btn = QPushButton("Start capture")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        self.status = QLabel("Open an experiment to capture.")
        self.status.setWordWrap(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.last_image = ImageView()
        self.watch = QCheckBox("Measure new frames as they arrive")
        self.watch.setToolTip("Needs the measurement set up (Set up measurement page).")
        self.watch_metric = QComboBox()
        self.watch_metric.setEditable(True)
        self.watch_metric.addItems(["(automatic)"] + METRICS)
        self.watch_metric.currentTextChanged.connect(lambda _: self.draw_live_chart())
        self.watch_status = QLabel()
        self.watch_status.setWordWrap(True)
        self.live_chart = ChartLabel("Measurements appear here as frames are analyzed.")
        live = QGroupBox("While capturing")
        live_form = QFormLayout(live)
        live_form.addRow(self.watch)
        live_form.addRow("Chart", self.watch_metric)
        live_form.addRow(self.watch_status)

        left = QVBoxLayout()
        left.addWidget(settings)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        left.addLayout(buttons)
        left.addWidget(live)
        left.addWidget(self.status)
        left.addWidget(QLabel("Log"))
        left.addWidget(self.log, 1)
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.last_image, "Latest frame")
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)
        live_layout.addWidget(self.live_chart, 1)
        self.right_tabs.addTab(live_tab, "Live measurement")
        right = QVBoxLayout()
        right.addWidget(self.right_tabs, 1)
        layout = QHBoxLayout(self)
        layout.addLayout(left, 1)
        layout.addLayout(right, 2)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        if log_handler is not None:
            log_handler.record.connect(self._log_line)
        state.experiment_changed.connect(lambda _: self.refresh())
        state.config_changed.connect(self.refresh)
        self.refresh()

    def _log_line(self, text: str, level: int) -> None:
        if self.thread is not None or level >= logging.WARNING:
            self.log.appendPlainText(text)

    def on_shown(self) -> None:
        if self.thread is None:
            self.refresh()

    def refresh(self) -> None:
        exp = self.state.experiment
        self.start_btn.setEnabled(exp is not None and self.thread is None)
        if exp is None:
            self.status.setText("Open an experiment to capture.")
            return
        cap = exp.config.capture
        self.interval.setText(_fmt_seconds(cap.interval))
        self.duration.setText(_fmt_seconds(cap.duration))
        self.max_frames.setValue(cap.max_frames or 0)
        self.keep_awake.setChecked(cap.keep_awake)
        if self.thread is None:
            self.status.setText(f"Ready: {len(exp.read_frames())} frame(s) so far.")
        self._show_latest(force=True)

    # --- run -------------------------------------------------------------------------------

    def start(self) -> None:
        exp = self.state.experiment
        if exp is None or self.thread is not None:
            return
        try:
            exp.config.capture.interval = parse_duration(self.interval.text().strip())
            text = self.duration.text().strip()
            exp.config.capture.duration = parse_duration(text) if text else None
        except ValueError as exc:
            self.status.setText(f"<span style='color:#b00020'>{exc}</span>")
            return
        exp.config.capture.max_frames = self.max_frames.value() or None
        self.start_btn.setEnabled(False)
        self.status.setText("Checking camera permission and opening cameras…")

        def preflight(progress, should_stop):
            from fungus_cv.capture.camera import Camera, CameraError
            from fungus_cv.capture.permissions import camera_access

            access = camera_access(request=True, timeout=120)
            if not access.ok:
                return access.advice()
            problems = []
            for cfg in exp.config.cameras:
                cam = Camera(cfg)
                try:
                    cam.open()
                    cam.read()
                except CameraError as exc:
                    problems.append(f"{cfg.name}: {exc}")
                finally:
                    cam.close()
            return "; ".join(problems)

        run_task(preflight, lambda problem: self._begin(problem),
                 lambda message: self._begin(message), pool="io")

    def _begin(self, problem: str) -> None:
        exp = self.state.experiment
        if problem:
            self.status.setText(f"<span style='color:#b00020'>{problem}</span>")
            self.start_btn.setEnabled(True)
            return
        self.log.clear()
        self.thread = CaptureThread(exp, self.keep_awake.isChecked())
        self.thread.finished_with.connect(self._finished)
        self.thread.failed.connect(self._thread_failed)
        self.state.set_capturing(True)
        self.thread.start()
        self.started_at = time.monotonic()
        self.stop_btn.setEnabled(True)
        self.timer.start(1000)
        self.status.setText("Capturing…")

    def stop(self) -> None:
        if self.thread is not None:
            self.status.setText("Stopping after the current shot…")
            self.thread.stop()

    def _done(self) -> None:
        self.timer.stop()
        if self.thread is not None:
            self.thread.wait(5000)
        self.thread = None
        self.state.set_capturing(False)
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(self.state.experiment is not None)
        self._show_latest(force=True)

    def _finished(self, summary) -> None:
        self._done()
        self.status.setText(f"Finished ({summary.stopped_reason or 'stopped'}): "
                            f"{summary.saved} saved, {summary.failed} failed, "
                            f"{summary.skipped_slots} slot(s) skipped.")

    def _thread_failed(self, message: str) -> None:
        self._done()
        self.status.setText(f"<span style='color:#b00020'>Capture stopped: {message}</span>")

    def _poll(self) -> None:
        if self.thread is None or self.thread.session is None:
            return
        s = self.thread.session.summary
        elapsed = int(time.monotonic() - self.started_at)
        self.status.setText(f"Capturing for {elapsed // 3600:d}:{elapsed // 60 % 60:02d}:"
                            f"{elapsed % 60:02d} — {s.saved} saved, {s.failed} failed, "
                            f"{s.skipped_slots} skipped")
        if s.saved != self._last_count:
            self._last_count = s.saved
            self._show_latest()
            if self.watch.isChecked() and s.saved:
                self.measure_new_frames()

    def _show_latest(self, force: bool = False) -> None:
        exp = self.state.experiment
        if exp is None:
            self.last_image.set_image(None)
            return
        ok = [r for r in exp.read_frames() if r["status"] == "ok"]
        if not ok:
            self.last_image.set_image(None)
            return
        image = cv2.imread(str(exp.root / ok[-1]["file"]))
        if image is not None:
            self.last_image.set_image(image, keep_view=not force)

    # --- live measurement ------------------------------------------------------------------

    def measure_new_frames(self) -> None:
        """Analyze frames not measured yet (incremental), then redraw the live chart."""
        exp = self.state.experiment
        if exp is None:
            return
        if self.watch_task is not None:
            self._watch_again = True
            return
        methods = {exp.config.analysis.target.method, exp.config.analysis.reference.method}
        preload_model_modules(methods)
        root = exp.root
        self.watch_status.setText("Measuring new frames…")

        def work(progress, should_stop):
            from fungus_cv.analyze.pipeline import Analyzer
            from fungus_cv.storage import Experiment

            return Analyzer(Experiment(root)).run()

        self.watch_task = run_task(work, self._watch_done, self._watch_failed)

    def _watch_finished(self) -> None:
        self.watch_task = None
        if self._watch_again:
            self._watch_again = False
            self.measure_new_frames()

    def _watch_done(self, summary) -> None:
        self.watch_summary = summary
        flags = ", ".join(f"{k} {v}" for k, v in summary.flagged.items())
        self.watch_status.setText(
            f"Measured {summary.processed} new frame(s)"
            + (f", {summary.failed} failed" if summary.failed else "")
            + (f" · flags: {flags}" if flags else "") + ".")
        self.draw_live_chart()
        self._watch_finished()

    def _watch_failed(self, message: str) -> None:
        self.watch_status.setText(f"<span style='color:#b00020'>Not measured: {message}</span>")
        self._watch_finished()

    def draw_live_chart(self) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        from fungus_cv.analyze.report import _style, list_plots, load_series

        exp = self.state.experiment
        chosen = self.watch_metric.currentText().strip()
        metric = None if chosen in ("", "(automatic)") else chosen
        if exp is None or not (exp.root / "results" / "measurements.csv").exists():
            return
        try:
            plots = list_plots(exp)
            series = [load_series(exp, metric, plot) for plot in plots]
        except (FileNotFoundError, ValueError) as exc:
            self.watch_status.setText(f"<span style='color:#b00020'>{exc}</span>")
            return
        fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
        _style(ax)
        for i, s in enumerate(series):
            color = SERIES[i % len(SERIES)]
            use = s.use
            label = s.plot if len(series) > 1 else "measured"
            if np.isfinite(s.unc[use]).any():
                ax.errorbar(s.t[use], s.y[use], yerr=s.unc[use], fmt="o-", ms=4, lw=1.2,
                            color=color, ecolor=color, elinewidth=0.8, label=label)
            else:
                ax.plot(s.t[use], s.y[use], "o-", ms=4, lw=1.2, color=color, label=label)
            if (~use).any():
                ax.plot(s.t[~use], s.y[~use], "x", ms=6, color=INK_2,
                        label="excluded" if i == 0 else None)
        ax.set_xlabel(f"Time since first frame ({series[0].time_unit})")
        metric = series[0].metric
        ax.set_ylabel(metric)
        n = int(sum(len(s.rows) for s in series))
        ax.set_title(f"{exp.config.name}: {metric} ({n} measurement(s))", loc="left",
                     fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        self.live_chart.set_figure(fig)

    def shutdown(self) -> None:
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait(10000)
