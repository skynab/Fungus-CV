"""Small Qt helpers: image conversion, background workers, app paths and logging."""

from __future__ import annotations

import logging
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QStandardPaths, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMessageBox, QWidget

APP_NAME = "Fungus-CV"
ORG_NAME = "Fungus-CV"


def bgr_to_qimage(image: np.ndarray) -> QImage:
    """Copy a BGR (or grayscale) numpy image into a QImage."""
    if image.ndim == 2:
        h, w = image.shape
        return QImage(np.ascontiguousarray(image).data, w, h, w, QImage.Format_Grayscale8).copy()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def bgr_to_pixmap(image: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(bgr_to_qimage(image))


def log_dir() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
    path = Path(base or Path.home() / f".{APP_NAME.lower()}") / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


class QtLogHandler(logging.Handler, QObject):
    """Forwards log records to a Qt signal so pages can show them."""

    record = Signal(str, int)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - logging API
        try:
            self.record.emit(self.format(record), record.levelno)
        except RuntimeError:  # the Qt object was already deleted during shutdown
            pass


def setup_logging() -> QtLogHandler:
    handler = QtLogHandler()
    file_handler = logging.FileHandler(log_dir() / "fungus-cv.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: "
                                                "%(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(file_handler)
    if sys.stderr is not None:
        root.addHandler(logging.StreamHandler(sys.stderr))
    return handler


# --- background work ---------------------------------------------------------------------


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(object)


class Task(QRunnable):
    """Run ``fn(progress, should_stop)`` on the thread pool; results come back as signals."""

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn
        self.signals = _Signals()
        self._stop = False
        self.setAutoDelete(True)

    def stop(self) -> None:
        self._stop = True

    def should_stop(self) -> bool:
        return self._stop

    def run(self) -> None:
        try:
            result = self.fn(self.signals.progress.emit, self.should_stop)
        except Exception as exc:  # report every failure to the UI instead of dying silently
            logging.getLogger(__name__).error("background task failed:\n%s",
                                              traceback.format_exc())
            try:
                self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            except RuntimeError:
                pass
            return
        try:
            self.signals.done.emit(result)
        except RuntimeError:
            pass


_ANALYSIS_POOL: QThreadPool | None = None
# Signal objects of tasks whose result hasn't reached the main thread yet. The pool deletes a
# finished task, and without this its signals could be freed before the queued result is
# delivered; callbacks without their own Qt object (lambdas) would then silently never run.
_PENDING: set = set()


def analysis_pool() -> QThreadPool:
    """One worker thread for image analysis: tasks run one after another, so two analyses
    never compete for memory or run native image code at the same time."""
    global _ANALYSIS_POOL
    if _ANALYSIS_POOL is None:
        _ANALYSIS_POOL = QThreadPool()
        _ANALYSIS_POOL.setMaxThreadCount(1)
    return _ANALYSIS_POOL


def run_task(fn: Callable, on_done=None, on_failed=None, on_progress=None,
             pool: str = "analysis") -> Task:
    """Run ``fn`` in the background. ``pool="io"`` is for camera/permission checks, which
    may wait on the user and must not hold up analysis."""
    task = Task(fn)
    signals = task.signals
    _PENDING.add(signals)
    if on_done:
        signals.done.connect(on_done)
    if on_failed:
        signals.failed.connect(on_failed)
    if on_progress:
        signals.progress.connect(on_progress)
    signals.done.connect(lambda _: _PENDING.discard(signals))
    signals.failed.connect(lambda _: _PENDING.discard(signals))
    (analysis_pool() if pool == "analysis" else QThreadPool.globalInstance()).start(task)
    return task


def preload_modules() -> None:
    """Import heavy modules on the main thread before any background work starts.

    PySide6 installs an import hook that is not safe when modules are first imported from
    worker threads while other threads run native code (it crashed in testing), so nothing
    heavy should be imported lazily inside a background task.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot  # noqa: F401
    import scipy.optimize  # noqa: F401
    import scipy.signal  # noqa: F401
    import scipy.sparse.csgraph  # noqa: F401
    import scipy.spatial  # noqa: F401
    import scipy.stats  # noqa: F401

    import fungus_cv.analyze.compare  # noqa: F401
    import fungus_cv.analyze.covariates  # noqa: F401
    import fungus_cv.analyze.exclusions  # noqa: F401
    import fungus_cv.analyze.fit  # noqa: F401
    import fungus_cv.analyze.overlay  # noqa: F401
    import fungus_cv.analyze.pipeline  # noqa: F401
    import fungus_cv.analyze.report  # noqa: F401
    import fungus_cv.analyze.sensitivity  # noqa: F401
    import fungus_cv.analyze.spread  # noqa: F401
    import fungus_cv.analyze.study  # noqa: F401
    import fungus_cv.analyze.suite  # noqa: F401
    import fungus_cv.analyze.summary  # noqa: F401
    import fungus_cv.analyze.validate  # noqa: F401
    import fungus_cv.capture.camera  # noqa: F401
    import fungus_cv.capture.diagnostics  # noqa: F401
    import fungus_cv.capture.health  # noqa: F401
    import fungus_cv.capture.permissions  # noqa: F401
    import fungus_cv.capture.power  # noqa: F401
    import fungus_cv.capture.scheduler  # noqa: F401
    import fungus_cv.learn.active  # noqa: F401
    import fungus_cv.learn.dataset  # noqa: F401
    import fungus_cv.learn.export  # noqa: F401
    import fungus_cv.measure.centerline  # noqa: F401
    import fungus_cv.segment.color  # noqa: F401
    import fungus_cv.segment.prompts  # noqa: F401

    if sys.platform == "darwin":
        try:
            import AVFoundation  # noqa: F401
            import Foundation  # noqa: F401
        except ImportError:
            pass


def preload_model_modules(methods: set[str]) -> None:
    """Import PyTorch & co. on the main thread when an analysis will need them."""
    if not methods & {"sam2", "model"}:
        return
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        import fungus_cv.learn.infer  # noqa: F401
        import fungus_cv.learn.unet  # noqa: F401
        import fungus_cv.segment.trained  # noqa: F401
        if "sam2" in methods:
            import transformers  # noqa: F401

            import fungus_cv.segment.sam2  # noqa: F401
    except ImportError:
        pass  # the analysis reports the missing dependency itself

def show_error(parent: QWidget | None, title: str, message: str) -> None:
    QMessageBox.warning(parent, title, message)
