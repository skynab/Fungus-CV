"""Application entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _fix_opencv_qt_env() -> None:
    """opencv-python on Linux points Qt at its own plugins, which breaks PySide6."""
    import cv2

    cv2_dir = str(Path(cv2.__file__).parent)
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        if os.environ.get(key, "").startswith(cv2_dir):
            del os.environ[key]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    _fix_opencv_qt_env()

    from PySide6.QtWidgets import QApplication

    from fungus_cv.gui import theme
    from fungus_cv.gui.main_window import MainWindow
    from fungus_cv.gui.qt_util import APP_NAME, ORG_NAME, setup_logging
    from fungus_cv.gui.state import AppState

    QApplication.setApplicationName(APP_NAME)
    QApplication.setOrganizationName(ORG_NAME)
    app = QApplication.instance() or QApplication(argv)
    theme.apply(app)
    log_handler = setup_logging()
    window = MainWindow(AppState(), log_handler)
    window.show()
    experiment = next((a for a in argv[1:] if not a.startswith("-")), None)
    if experiment:
        window.open_experiment(experiment)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
