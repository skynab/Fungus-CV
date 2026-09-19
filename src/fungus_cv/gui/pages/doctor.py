"""Diagnostics: camera permission, cameras, display, optional dependencies."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.gui import theme
from fungus_cv.gui.qt_util import log_dir, run_task


class DoctorPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.run_btn = QPushButton("Run checks")
        theme.mark_primary(self.run_btn)
        self.run_btn.clicked.connect(lambda: self.run(request=False))
        self.ask_btn = QPushButton("Ask for camera permission")
        self.ask_btn.clicked.connect(lambda: self.run(request=True))
        logs_btn = QPushButton("Open log folder")
        logs_btn.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(log_dir()))))
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Check", "Result", "How to fix"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(True)
        top = QHBoxLayout()
        top.addWidget(self.run_btn)
        top.addWidget(self.ask_btn)
        top.addWidget(logs_btn)
        top.addStretch()
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.message)
        layout.addWidget(self.table, 1)
        self._ran = False

    def on_shown(self) -> None:
        if not self._ran:
            self.run(request=False)

    def run(self, request: bool) -> None:
        self._ran = True
        self.run_btn.setEnabled(False)
        self.ask_btn.setEnabled(False)
        self.message.setText("If macOS asks for camera access, click Allow…" if request
                             else "Checking…")
        probe = not self.state.capturing

        def work(progress, should_stop):
            from fungus_cv.capture.diagnostics import run_checks

            return run_checks(probe_cameras=probe, request_permission=request)

        run_task(work, self._show, self._failed, pool="io")

    def _failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.ask_btn.setEnabled(True)
        self.message.setText(message)

    def _show(self, checks) -> None:
        self.run_btn.setEnabled(True)
        self.ask_btn.setEnabled(True)
        self.table.setRowCount(len(checks))
        colors = {True: QColor(theme.GOOD), False: QColor(theme.BAD), None: None}
        for i, check in enumerate(checks):
            mark = {True: "✓ ", False: "✗ ", None: ""}[check.ok]
            name = QTableWidgetItem(mark + check.name)
            if colors[check.ok] is not None:
                name.setForeground(colors[check.ok])
            self.table.setItem(i, 0, name)
            self.table.setItem(i, 1, QTableWidgetItem(check.detail))
            self.table.setItem(i, 2, QTableWidgetItem(check.fix if check.ok is not True else ""))
        self.table.resizeRowsToContents()
        failed = sum(c.ok is False for c in checks)
        self.message.setText("All checks passed." if not failed else
                             f"{failed} problem(s) found; see 'How to fix'.")


__all__ = ["DoctorPage", "Qt"]
