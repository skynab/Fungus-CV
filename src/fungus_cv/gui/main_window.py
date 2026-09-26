"""Main window: a sidebar of workflow steps and one page per step."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from fungus_cv import __version__
from fungus_cv.gui import theme
from fungus_cv.gui.pages.analyze import AnalyzePage
from fungus_cv.gui.pages.camera import CameraPage
from fungus_cv.gui.pages.capture import CapturePage
from fungus_cv.gui.pages.demo_guide import DemoGuidePage
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

PAGE_INFO = {
    "Experiment": "Status and the most used settings. Settings are written back to "
                  "config.yaml; comments are kept.",
    "Camera": "Live preview for framing and focus, and a check of the camera settings.",
    "Capture": "Scheduled photos, measured as they arrive. Each image is written as soon as "
               "it is taken, so stopping never loses a frame.",
    "Set Up Measurement": "Click the base and tip, outline the region, then drag boxes over "
                          "the colour you are following.",
    "SAM Prompts": "Click the target, right-click what looks similar but isn't. SAM 2 tracks "
                   "the mask forward and backward through the whole time-lapse.",
    "Analyze": "Measure every frame, browse the per-frame results and exclude frames by hand.",
    "Report": "Fit growth models and read the report charts, spread map and sensitivity check.",
    "Labels": "Label images for training: add frames, correct the masks with a brush or SAM "
              "clicks, and mark them reviewed.",
    "Train Models": "Train a segmentation model on a labeled dataset, watch it learn, and "
                    "evaluate it.",
    "Study": "Compare conditions across replicate experiments.",
    "Validation": "Agreement with hand measurements, and the validation suite.",
    "Demo Guide": "Step by step through the two demo experiments: colour thresholds and "
                  "SAM 2 clicks.",
    "Diagnostics": "Camera permission, cameras, display and optional dependencies on this "
                   "computer.",
}


class FadeRule(QWidget):
    """A 1px rule that fades to transparent over 48px at each end (a Nocturne signature)."""

    def __init__(self, orientation=Qt.Horizontal):
        super().__init__()
        self.orientation = orientation
        if orientation == Qt.Horizontal:
            self.setFixedHeight(1)
        else:
            self.setFixedWidth(1)

    def paintEvent(self, event):  # noqa: N802 - Qt API
        horizontal = self.orientation == Qt.Horizontal
        length = self.width() if horizontal else self.height()
        end = QPointF(length, 0) if horizontal else QPointF(0, length)
        gradient = QLinearGradient(QPointF(0, 0), end)
        color = QColor(theme.TEXT)
        color.setAlphaF(0.16)
        clear = QColor(color)
        clear.setAlpha(0)
        fade = min(0.45, 48 / max(length, 1))
        gradient.setColorAt(0, clear)
        gradient.setColorAt(fade, color)
        gradient.setColorAt(1 - fade, color)
        gradient.setColorAt(1, clear)
        QPainter(self).fillRect(self.rect(), gradient)


class StatusDot(QWidget):
    """The live indicator: an accent dot with a soft ring, or a grey one when idle."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(13, 13)
        self.active = False

    def set_active(self, active: bool) -> None:
        self.active = active
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        centre = QPointF(self.width() / 2, self.height() / 2)
        if self.active:
            ring = QColor(theme.ACCENT)
            ring.setAlphaF(0.22)
            painter.setBrush(ring)
            painter.drawEllipse(centre, 6.5, 6.5)
        painter.setBrush(QColor(theme.ACCENT if self.active else theme.NEUTRAL[600]))
        painter.drawEllipse(centre, 3.5, 3.5)


class Chip(QPushButton):
    """A small surface button with an uppercase key and a value, as in the top bar."""

    def __init__(self, key: str, value: str = ""):
        super().__init__()
        self.setObjectName("chip")
        self.key = QLabel(key)
        self.key.setObjectName("chipKey")
        self.key.setFont(theme.label_font(10))
        self.value = QLabel(value)
        self.caret = QLabel()
        self.caret.setPixmap(theme.icon_pixmap("caret-down", theme.MUTED, 10))
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 5, 10, 5)
        row.setSpacing(8)
        for label in (self.key, self.value, self.caret):
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
            row.addWidget(label)

    def sizeHint(self):  # noqa: N802 - Qt API
        return self.layout().sizeHint()

    def minimumSizeHint(self):  # noqa: N802 - Qt API
        return self.layout().sizeHint()


class RailDelegate(QStyledItemDelegate):
    """Draws the short accent mark to the left of the current page."""

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if option.state & QStyle.State_Selected and index.flags() & Qt.ItemIsSelectable:
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.ACCENT))
            rect = option.rect
            painter.drawRoundedRect(QRectF(rect.left() + 2, rect.center().y() - 8, 3, 18), 1.5,
                                    1.5)
            painter.restore()


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None, log_handler=None):
        super().__init__()
        preload_modules()
        self.state = state or AppState()
        self.setWindowTitle(APP_NAME)
        self.resize(1320, 860)

        self.nav = QListWidget()
        self.nav.setObjectName("rail")
        self.nav.setFixedWidth(238)
        self.nav.setItemDelegate(RailDelegate(self.nav))
        self.nav.setFocusPolicy(Qt.NoFocus)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stack = QStackedWidget()
        sections = [
            ("Capture", [
                ("Experiment", ExperimentPage(self.state, self)),
                ("Camera", CameraPage(self.state)),
                ("Capture", CapturePage(self.state, log_handler)),
            ]),
            ("Measure", [
                ("Set Up Measurement", SetupPage(self.state)),
                ("SAM Prompts", PromptPage(self.state)),
                ("Analyze", AnalyzePage(self.state)),
                ("Report", ReportPage(self.state)),
            ]),
            ("Models", [
                ("Labels", LabelsPage(self.state)),
                ("Train Models", TrainPage(self.state)),
            ]),
            ("Results", [
                ("Study", StudyPage(self.state)),
                ("Validation", ValidationPage(self.state)),
            ]),
            ("Help", [
                ("Demo Guide", DemoGuidePage(self.state, self)),
                ("Diagnostics", DoctorPage(self.state)),
            ]),
        ]
        self.pages = []
        self._row_to_page: dict[int, int] = {}
        for section, pages in sections:
            header = QListWidgetItem(section.upper())
            header.setFlags(Qt.NoItemFlags)
            header.setFont(theme.label_font(10))
            header.setTextAlignment(Qt.AlignLeft | Qt.AlignBottom)
            header.setSizeHint(QSize(0, 26 if self.nav.count() == 0 else 42))
            self.nav.addItem(header)
            for title, page in pages:
                self._row_to_page[self.nav.count()] = len(self.pages)
                item = QListWidgetItem(title)
                item.setFont(theme.base_font())
                font = item.font()
                font.setPixelSize(14)
                item.setFont(font)
                item.setSizeHint(QSize(0, 36))
                self.nav.addItem(item)
                self.stack.addWidget(page)
                self.pages.append((title, page))

        # --- page heading ------------------------------------------------------------------
        self.page_title = QLabel()
        self.page_title.setFont(theme.heading_font(26))
        self.page_subtitle = QLabel()
        self.page_subtitle.setObjectName("pageSubtitle")
        head = QWidget()
        head.setObjectName("pagehead")
        head.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        head_layout = QVBoxLayout(head)
        head_layout.setContentsMargins(24, 18, 24, 10)
        head_layout.setSpacing(4)
        head_layout.addWidget(self.page_title)
        head_layout.addWidget(self.page_subtitle)

        page_area = QWidget()
        page_layout = QVBoxLayout(page_area)
        page_layout.setContentsMargins(14, 0, 14, 6)
        # A page taller or wider than the window scrolls, so the window can always be made
        # small enough for the screen (and moved: macOS pins a window taller than the screen).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.stack)
        page_layout.addWidget(scroll)
        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(head)
        main_layout.addWidget(page_area, 1)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        rail_box = QVBoxLayout()
        rail_box.setContentsMargins(0, 10, 0, 10)
        rail_box.addWidget(self.nav)
        body.addLayout(rail_box)
        body.addWidget(FadeRule(Qt.Vertical))
        body.addWidget(main, 1)

        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self._build_topbar())
        column.addWidget(FadeRule(Qt.Horizontal))
        column.addLayout(body, 1)
        self.setCentralWidget(central)

        self.nav.currentRowChanged.connect(self._show_page)
        self.nav.setCurrentRow(1)
        self.state.experiment_changed.connect(self._experiment_changed)
        self.state.busy_changed.connect(self._busy_changed)
        self._build_menu()

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("topbar")
        bar.setFixedHeight(54)
        row = QHBoxLayout(bar)
        row.setContentsMargins(18, 0, 18, 0)
        row.setSpacing(11)

        mark = QLabel()
        mark.setPixmap(theme.icon_pixmap("sparkle", theme.ACCENT, 17))
        brand = QLabel(APP_NAME)
        brand.setObjectName("brand")
        brand.setFont(theme.heading_font(15))
        brand_row = QHBoxLayout()
        brand_row.setSpacing(8)
        brand_row.addWidget(mark)
        brand_row.addWidget(brand)
        row.addLayout(brand_row)
        row.addSpacing(6)

        self.experiment_chip = Chip("Experiment", "none open")
        self.experiment_label = self.experiment_chip.value
        self.experiment_menu = QMenu(self)
        self.experiment_menu.aboutToShow.connect(self._fill_experiment_menu)
        self.experiment_chip.setMenu(self.experiment_menu)
        row.addWidget(self.experiment_chip)

        self.camera_chip = QFrame()
        self.camera_chip.setObjectName("chip")
        self.camera_label = QLabel("Camera")
        self.camera_label.setObjectName("chipKey")
        self.camera_label.setFont(theme.label_font(10))
        self.camera_box = QComboBox()
        self.camera_box.setObjectName("chipCombo")
        self.camera_box.setToolTip("With several cameras, each one is annotated, measured and "
                                   "reported on its own.")
        self.camera_box.currentTextChanged.connect(self._camera_chosen)
        chip_row = QHBoxLayout(self.camera_chip)
        chip_row.setContentsMargins(10, 1, 2, 1)
        chip_row.setSpacing(2)
        chip_row.addWidget(self.camera_label)
        chip_row.addWidget(self.camera_box)
        self.camera_chip.setVisible(False)
        row.addWidget(self.camera_chip)
        row.addStretch(1)

        self.live_dot = StatusDot()
        self.live_label = QLabel("Not capturing")
        self.live_label.setObjectName("live")
        row.addWidget(self.live_dot)
        row.addWidget(self.live_label)
        row.addSpacing(6)

        doctor = QPushButton()
        doctor.setObjectName("iconButton")
        doctor.setIcon(theme.icon("settings", theme.TEXT, 16))
        doctor.setToolTip("Diagnostics")
        doctor.clicked.connect(lambda: self.go_to("Diagnostics"))
        row.addWidget(doctor)
        return bar

    # --- navigation ----------------------------------------------------------------------

    def _show_page(self, row: int) -> None:
        if row not in self._row_to_page:
            return  # a section header
        self.stack.setCurrentIndex(self._row_to_page[row])
        # Size the stack for the page on show only, not for the largest page.
        for index in range(self.stack.count()):
            policy = QSizePolicy.Preferred if index == self.stack.currentIndex() \
                else QSizePolicy.Ignored
            self.stack.widget(index).setSizePolicy(policy, policy)
        self.stack.adjustSize()
        self._update_heading()
        page = self.stack.currentWidget()
        if hasattr(page, "on_shown"):
            page.on_shown()

    def go_to(self, title: str) -> None:
        for row, index in self._row_to_page.items():
            if self.pages[index][0] == title:
                self.nav.setCurrentRow(row)

    def _update_heading(self) -> None:
        title = self.pages[self.stack.currentIndex()][0]
        experiment = self.state.experiment
        if title == "Experiment" and experiment is not None:
            self.page_title.setText(experiment.config.name)
        else:
            self.page_title.setText(title)
        self.page_subtitle.setText(PAGE_INFO.get(title, ""))

    def _busy_changed(self, capturing: bool) -> None:
        self.live_dot.set_active(capturing)
        self.live_label.setText("Capturing" if capturing else "Not capturing")

    def _fill_experiment_menu(self) -> None:
        menu = self.experiment_menu
        menu.clear()
        menu.addAction("New Experiment…", self.new_experiment)
        menu.addAction("Open Experiment…", lambda: self.open_experiment())
        recent = self.state.recent()
        if recent:
            menu.addSeparator()
            current = self.state.experiment.root.resolve() if self.state.experiment else None
            for path in recent[:8]:
                action = menu.addAction(theme.icon("folder", theme.ACCENT if path.resolve() ==
                                                   current else theme.MUTED, 15), path.name)
                action.setToolTip(str(path))
                action.triggered.connect(lambda _=False, p=path: self.open_experiment(p))
        if self.state.experiment is not None:
            menu.addSeparator()
            menu.addAction("Show Experiment Folder", self.reveal_experiment)

    def _camera_chosen(self, name: str) -> None:
        if name:
            self.state.set_camera(name)

    def _experiment_changed(self, experiment) -> None:
        cameras = [c.name for c in experiment.config.cameras] if experiment else []
        self.camera_box.blockSignals(True)
        self.camera_box.clear()
        self.camera_box.addItems(cameras)
        if self.state.camera in cameras:
            self.camera_box.setCurrentText(self.state.camera)
        self.camera_box.blockSignals(False)
        self.camera_chip.setVisible(len(cameras) > 1)
        if experiment is None:
            self.experiment_label.setText("none open")
            self.experiment_chip.setToolTip("")
            self.setWindowTitle(APP_NAME)
        else:
            self.experiment_label.setText(experiment.config.name)
            self.experiment_chip.setToolTip(str(experiment.root))
            self.setWindowTitle(f"{experiment.config.name} — {APP_NAME}")
        self.experiment_chip.updateGeometry()
        self._update_heading()

    # --- menu ----------------------------------------------------------------------------

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        new_action = QAction("New Experiment…", self, shortcut=QKeySequence.New)
        new_action.triggered.connect(self.new_experiment)
        open_action = QAction("Open Experiment…", self, shortcut=QKeySequence.Open)
        open_action.triggered.connect(self.open_experiment)
        reveal = QAction("Show Experiment Folder", self)
        reveal.triggered.connect(self.reveal_experiment)
        demo = QAction("Make Demo Experiment…", self)
        demo.setToolTip("A synthetic dye time-lapse, measured with colour thresholds (the Set "
                        "Up Measurement workflow), to try everything without a camera.")
        demo.triggered.connect(lambda: self.make_demo())
        sam_demo = QAction("Make SAM 2 Demo Experiment…", self)
        sam_demo.setToolTip("A synthetic mould colony on an agar plate, set up for the SAM "
                            "Prompts workflow: click the colony, then Analyze.")
        sam_demo.triggered.connect(lambda: self.make_demo(sam=True))
        archive = QAction("Archive Experiment…", self)
        archive.setToolTip("Pack settings, annotations and results into one zip with a hash "
                           "manifest, for a data repository.")
        archive.triggered.connect(self.archive_experiment)
        quit_action = QAction("Quit", self, shortcut=QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        for action in (new_action, open_action, demo, sam_demo, reveal, archive):
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
        guide = QAction("Demo Guide", self)
        guide.triggered.connect(lambda: self.go_to("Demo Guide"))
        help_menu.addAction(guide)
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

    def make_demo(self, folder: Path | None = None, sam: bool = False) -> None:
        """Create a demo experiment in the background, then open it. ``sam``: the mould
        colony for the SAM Prompts workflow instead of the dye strip."""
        from fungus_cv import demo as demo_mod
        from fungus_cv.gui.qt_util import run_task

        if folder is None:
            parent = QFileDialog.getExistingDirectory(
                self, "Where to put the demo experiment", str(Path.home()))
            if not parent:
                return
            stem = "fungus-sam-demo" if sam else "fungus-demo"
            folder = Path(parent) / stem
            n = 2
            while folder.exists():
                folder = Path(parent) / f"{stem}-{n}"
                n += 1
        make = demo_mod.colony_experiment if sam else demo_mod.dye_experiment
        self.statusBar().showMessage("Making the demo experiment…")
        run_task(lambda p, s: make(folder).root, lambda root: self._demo_ready(root, sam),
                 lambda m: show_error(self, "Could not make the demo", m))

    def _demo_ready(self, root, sam: bool = False) -> None:
        self.open_experiment(root)
        if sam:
            self.go_to("SAM Prompts")
            self.statusBar().showMessage(
                "SAM 2 demo ready: on a frame where the colony is clearly visible, click both its "
                "white rim and its green centre (one click takes only the centre), Save the "
                "prompt, then open Analyze and press Run (Help → Demo Guide has every step).",
                30000)
        else:
            self.statusBar().showMessage("Demo ready: open Analyze and press Run (Help → "
                                         "Demo Guide has every step).", 15000)

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
        self.state.preview.stop()  # the pages share it; it must not outlive the window
        event.accept()

    def keyPressEvent(self, event):  # noqa: N802
        page = self.stack.currentWidget()
        if hasattr(page, "handle_key") and page.handle_key(event):
            return
        super().keyPressEvent(event)


__all__ = ["MainWindow", "Qt"]
