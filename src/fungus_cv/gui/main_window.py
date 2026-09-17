"""Main window: a sidebar of workflow steps and one page per step."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QWidget,
)

from fungus_cv import __version__
from fungus_cv.gui.pages.analyze import AnalyzePage
from fungus_cv.gui.pages.camera import CameraPage
from fungus_cv.gui.pages.capture import CapturePage
from fungus_cv.gui.pages.doctor import DoctorPage
from fungus_cv.gui.pages.experiment import ExperimentPage
from fungus_cv.gui.pages.labels import LabelsPage
from fungus_cv.gui.pages.prompt import PromptPage
from fungus_cv.gui.pages.report import ReportPage
from fungus_cv.gui.pages.setup import SetupPage
from fungus_cv.gui.pages.study import StudyPage
from fungus_cv.gui.pages.train import TrainPage
from fungus_cv.gui.pages.validation import ValidationPage
from fungus_cv.gui.qt_util import APP_NAME, log_dir, preload_modules, show_error
from fungus_cv.gui.state import AppState


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None, log_handler=None):
        super().__init__()
        preload_modules()
        self.state = state or AppState()
        self.setWindowTitle(APP_NAME)
        self.resize(1320, 860)

        self.nav = QListWidget()
        self.nav.setIconSize(QSize(18, 18))
        self.nav.setFixedWidth(190)
        self.stack = QStackedWidget()
        sections = [
            ("Capture", [
                ("Experiment", ExperimentPage(self.state, self)),
                ("Camera", CameraPage(self.state)),
                ("Capture", CapturePage(self.state, log_handler)),
            ]),
            ("Measure", [
                ("Set up measurement", SetupPage(self.state)),
                ("SAM prompts", PromptPage(self.state)),
                ("Analyze", AnalyzePage(self.state)),
                ("Report", ReportPage(self.state)),
            ]),
            ("Models", [
                ("Labels", LabelsPage(self.state)),
                ("Train models", TrainPage(self.state)),
            ]),
            ("Results", [
                ("Study", StudyPage(self.state)),
                ("Validation", ValidationPage(self.state)),
            ]),
            ("This computer", [
                ("Diagnostics", DoctorPage(self.state)),
            ]),
        ]
        self.pages = []
        self._row_to_page: dict[int, int] = {}
        for section, pages in sections:
            header = QListWidgetItem(section.upper())
            header.setFlags(Qt.NoItemFlags)
            font = header.font()
            font.setPointSizeF(max(8.0, font.pointSizeF() - 2))
            font.setBold(True)
            header.setFont(font)
            self.nav.addItem(header)
            for title, page in pages:
                self._row_to_page[self.nav.count()] = len(self.pages)
                self.nav.addItem(QListWidgetItem(title))
                self.stack.addWidget(page)
                self.pages.append((title, page))
        self.nav.currentRowChanged.connect(self._show_page)
        self.nav.setCurrentRow(1)

        self.nav.setObjectName("nav")
        self.nav.setStyleSheet(
            "#nav { border: none; font-size: 14px; padding-top: 8px; }"
            "#nav::item { padding: 9px 12px; border-radius: 6px; margin: 1px 6px; }"
            "#nav::item:selected { background: palette(highlight); "
            "color: palette(highlighted-text); }")
        central = QWidget()
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.nav)
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        row.addWidget(divider)
        row.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.experiment_label = QLabel("No experiment open")
        self.statusBar().addPermanentWidget(self.experiment_label)
        self.state.experiment_changed.connect(self._experiment_changed)
        self._build_menu()

    # --- navigation ----------------------------------------------------------------------

    def _show_page(self, row: int) -> None:
        if row not in self._row_to_page:
            return  # a section header
        self.stack.setCurrentIndex(self._row_to_page[row])
        page = self.stack.currentWidget()
        if hasattr(page, "on_shown"):
            page.on_shown()

    def go_to(self, title: str) -> None:
        for row, index in self._row_to_page.items():
            if self.pages[index][0] == title:
                self.nav.setCurrentRow(row)

    def _experiment_changed(self, experiment) -> None:
        if experiment is None:
            self.experiment_label.setText("No experiment open")
            self.setWindowTitle(APP_NAME)
        else:
            self.experiment_label.setText(f"{experiment.config.name}  —  {experiment.root}")
            self.setWindowTitle(f"{experiment.config.name} — {APP_NAME}")

    # --- menu ----------------------------------------------------------------------------

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        new_action = QAction("New Experiment…", self, shortcut=QKeySequence.New)
        new_action.triggered.connect(self.new_experiment)
        open_action = QAction("Open Experiment…", self, shortcut=QKeySequence.Open)
        open_action.triggered.connect(self.open_experiment)
        reveal = QAction("Show Experiment Folder", self)
        reveal.triggered.connect(self.reveal_experiment)
        archive = QAction("Archive Experiment…", self)
        archive.setToolTip("Pack settings, annotations and results into one zip with a hash "
                           "manifest, for a data repository.")
        archive.triggered.connect(self.archive_experiment)
        quit_action = QAction("Quit", self, shortcut=QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        for action in (new_action, open_action, reveal, archive):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        logs = QAction("Open Log Folder", self)
        logs.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(
            str(log_dir()))))
        about = QAction(f"About {APP_NAME}", self)
        about.triggered.connect(lambda: QMessageBox.about(
            self, APP_NAME, f"<b>{APP_NAME}</b> {__version__}<br>Time-lapse capture and "
                            "measurement of spreading growth."))
        help_menu.addAction(logs)
        help_menu.addAction(about)

    def new_experiment(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose where to create the experiment folder", str(Path.home()))
        if not folder:
            return
        name, ok = QInputDialog.getText(self, "New experiment", "Experiment name "
                                        "(also the folder name):")
        if not ok or not name.strip():
            return
        path = Path(folder) / name.strip()
        try:
            self.state.create(path, name.strip())
        except (OSError, ValueError) as exc:
            show_error(self, "Could not create experiment", str(exc))
            return
        self.go_to("Experiment")

    def open_experiment(self, path: str | Path | None = None) -> None:
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Open experiment folder",
                                                    str(Path.home()))
            if not path:
                return
        try:
            self.state.open(Path(path))
        except (OSError, ValueError) as exc:
            show_error(self, "Could not open experiment",
                       f"{path}\n\n{exc}\n\nChoose a folder containing config.yaml.")

    def archive_experiment(self, out: Path | None = None, frames: bool | None = None) -> None:
        """Bundle the open experiment; asks where to save it and whether to include photos."""
        from fungus_cv.analyze import archive as archive_mod
        from fungus_cv.gui.qt_util import run_task

        exp = self.state.experiment
        if exp is None:
            show_error(self, "No experiment open", "Open an experiment first.")
            return
        if out is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save bundle", str(Path.home() / f"{exp.root.name}.zip"), "Zip (*.zip)")
            if not path:
                return
            out = Path(path)
        if frames is None:
            answer = QMessageBox.question(
                self, "Include the photos?",
                "Include every photo in the bundle?\n\nWithout them the bundle is small and "
                "still records each photo's hash, so a separate image archive can be checked "
                "against it.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            frames = answer == QMessageBox.Yes
        self.statusBar().showMessage("Writing the bundle…")

        def work(progress, should_stop):
            return archive_mod.build(exp.root, out, frames=frames)

        run_task(work, self._archived, self._archive_failed)

    def _archived(self, result) -> None:
        self.statusBar().showMessage(
            f"Wrote {result.path.name} ({result.bytes_stored / 1e6:.1f} MB, "
            f"{result.frames_included} photo(s) included)", 15000)

    def _archive_failed(self, message: str) -> None:
        self.statusBar().clearMessage()
        show_error(self, "Could not write the bundle", message)

    def reveal_experiment(self) -> None:
        if self.state.experiment is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.state.experiment.root)))

    # --- shutdown ------------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802 - Qt API
        if self.state.capturing:
            answer = QMessageBox.question(
                self, "Capture running",
                "A capture is running. Stop it and quit? Frames already saved are kept.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        for _, page in self.pages:
            if hasattr(page, "shutdown"):
                page.shutdown()
        event.accept()

    def keyPressEvent(self, event):  # noqa: N802
        page = self.stack.currentWidget()
        if hasattr(page, "handle_key") and page.handle_key(event):
            return
        super().keyPressEvent(event)


__all__ = ["MainWindow", "Qt"]
