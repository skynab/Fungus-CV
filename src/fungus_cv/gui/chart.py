"""Matplotlib charts shown in the app (PySide6-Essentials has no Qt Charts module)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

from fungus_cv.gui.qt_util import bgr_to_pixmap


def figure_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[..., 2::-1])


class ChartLabel(QLabel):
    """Shows a rendered chart, scaled to fit while keeping its proportions."""

    def __init__(self, placeholder: str = ""):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap = None
        self.image: np.ndarray | None = None

    def set_image(self, image: np.ndarray | None) -> None:
        self.image = image
        self._pixmap = None if image is None else bgr_to_pixmap(image)
        self._rescale()

    def set_figure(self, fig) -> None:
        import matplotlib.pyplot as plt

        self.set_image(figure_to_bgr(fig))
        plt.close(fig)

    def resizeEvent(self, event):  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            self.setPixmap(QPixmap())
            return
        self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation))

