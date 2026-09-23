"""Zoomable image canvas with overlays and click/drag events in image coordinates."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QToolButton,
    QWidget,
)

from fungus_cv.gui import theme
from fungus_cv.gui.qt_util import bgr_to_pixmap


class ImageView(QGraphicsView):
    """Shows one image. Wheel/trackpad scroll zooms, middle-drag or Alt(Option)+drag pans,
    double-click or the Fit button fits the whole image.

    In ``PAN`` mode (``set_nav_mode``) a left-drag pans and the wheel moves the image up and
    down instead of zooming -- and, when the whole image is on screen already, scrolls the
    page behind it. ``ZOOM`` mode, the default, is the behaviour above; ``ViewControls``
    gives the user a button for each.

    Emits ``clicked(x, y, button)`` and ``dragged(x0, y0, x1, y1)`` in full-resolution pixel
    coordinates (pixel centres are integers), so annotations are independent of zoom. With
    ``paint_enabled``, left/right drags emit ``stroke(x, y, button, phase)`` instead
    (phase: 0 press, 1 move, 2 release), for brushes. Left-dragging one of ``handles``
    (points in image coordinates) emits ``handle_moved(index, x, y, phase)`` instead of a
    click, so callers can let the user move points they placed.
    """

    STROKE_PRESS, STROKE_MOVE, STROKE_RELEASE = 0, 1, 2
    ZOOM, PAN = "zoom", "pan"

    clicked = Signal(float, float, int)
    dragged = Signal(float, float, float, float)
    hovered = Signal(float, float)
    stroke = Signal(float, float, int, int)
    handle_moved = Signal(int, float, float, int)
    nav_mode_changed = Signal(str)

    HANDLE_GRAB_PX = 8  # how close (on screen) a press must be to grab a handle

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor(theme.NEUTRAL[900]))
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        # Pixel (0, 0) covers [-0.5, 0.5]: shift so scene coordinates equal pixel coordinates.
        self._pixmap_item.setOffset(-0.5, -0.5)
        self.scene().addItem(self._pixmap_item)
        self._mask_items: dict[int, QGraphicsPixmapItem] = {}
        self._mask_item = self._mask_layer(0)
        self._overlays = []
        self._press = None
        self._pan = None
        self._nav_mode = self.ZOOM
        self.drag_enabled = False
        self.paint_enabled = False
        self._painting: int | None = None  # mouse button of the stroke in progress
        self._has_image = False
        self._rubber = None
        self._image_rect = QRectF()
        self.handles: list[tuple[float, float]] = []
        self._handle: int | None = None  # index of the handle being dragged
        # Panning is by dragging, so scroll bars only take space; the scene rect is padded
        # (see set_image) so the image can be panned even when it fits in the window.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.fit_button = QToolButton(self)
        self.fit_button.setText("Fit")
        self.fit_button.setToolTip("Fit the whole image in the window (or double-click)")
        self.fit_button.setCursor(Qt.ArrowCursor)
        self.fit_button.clicked.connect(self.fit)
        self.fit_button.hide()

    # --- content -----------------------------------------------------------------------

    def set_image(self, image: np.ndarray | None, keep_view: bool = False) -> None:
        if image is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._has_image = False
            self.fit_button.hide()
            return
        first = not self._has_image
        self._pixmap_item.setPixmap(bgr_to_pixmap(image))
        h, w = image.shape[:2]
        self._image_rect = QRectF(-0.5, -0.5, w, h)
        self.scene().setSceneRect(self._image_rect)
        pad = max(w, h)
        self.setSceneRect(self._image_rect.adjusted(-pad, -pad, pad, pad))
        self._has_image = True
        self.fit_button.show()
        if first or not keep_view:
            self.fit()

    def _mask_layer(self, layer: int) -> QGraphicsPixmapItem:
        if layer not in self._mask_items:
            item = QGraphicsPixmapItem()
            item.setOffset(-0.5, -0.5)
            item.setZValue(1 + 0.1 * layer)
            self.scene().addItem(item)
            self._mask_items[layer] = item
        return self._mask_items[layer]

    def set_mask(self, mask: np.ndarray | None, color=(255, 0, 255), alpha: int = 110,
                 layer: int = 0) -> None:
        """Tint ``mask`` pixels; higher ``layer`` numbers are drawn on top (e.g. a proposal)."""
        item = self._mask_layer(layer)
        if mask is None:
            item.setPixmap(QPixmap())
            return
        rgba = np.zeros((*mask.shape, 4), np.uint8)
        rgba[mask] = (*color, alpha)
        from PySide6.QtGui import QImage

        h, w = mask.shape
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        item.setPixmap(QPixmap.fromImage(qimg))

    def fit(self) -> None:
        if self._has_image:
            self.fitInView(self._image_rect, Qt.KeepAspectRatio)

    def fill(self) -> None:
        """Zoom until the image covers the whole window; the long edge runs off-screen."""
        if self._has_image:
            self.fitInView(self._image_rect, Qt.KeepAspectRatioByExpanding)

    # --- navigation mode ---------------------------------------------------------------

    @property
    def nav_mode(self) -> str:
        return self._nav_mode

    def set_nav_mode(self, mode: str) -> None:
        if mode not in (self.ZOOM, self.PAN) or mode == self._nav_mode:
            return
        self._nav_mode = mode
        self._rest_cursor()
        self.nav_mode_changed.emit(mode)

    def image_fits(self) -> bool:
        """Whether the whole image is on screen, so panning has nowhere to go."""
        if not self._has_image:
            return True
        shown, window = self.transform().mapRect(self._image_rect), self.viewport().rect()
        return shown.width() <= window.width() + 1 and shown.height() <= window.height() + 1

    def _rest_cursor(self) -> None:
        """The cursor for this mode when nothing is being dragged."""
        if self._nav_mode == self.PAN:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.unsetCursor()

    def resizeEvent(self, event):  # noqa: N802 - Qt API
        super().resizeEvent(event)
        hint = self.fit_button.sizeHint()
        self.fit_button.setGeometry(self.width() - hint.width() - 8, 8, hint.width(),
                                    hint.height())

    def _handle_at(self, pos) -> int | None:
        """Index of the handle nearest ``pos`` (viewport pixels) within grab range."""
        best, best_d = None, float(self.HANDLE_GRAB_PX) ** 2
        for i, (x, y) in enumerate(self.handles):
            p = self.mapFromScene(QPointF(x, y))
            d = (p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2
            if d <= best_d:
                best, best_d = i, d
        return best

    def clear_overlays(self) -> None:
        for item in self._overlays:
            self.scene().removeItem(item)
        self._overlays.clear()

    def _pen(self, color, width=2.0, dashed=False) -> QPen:
        pen = QPen(QColor(*color))
        pen.setWidthF(width)
        pen.setCosmetic(True)  # constant on-screen width at any zoom
        if dashed:
            pen.setStyle(Qt.DashLine)
        return pen

    def add_points(self, points, color=(0, 220, 0), radius_px: float = 5) -> None:
        for x, y in points:
            item = self.scene().addEllipse(QRectF(-radius_px, -radius_px, 2 * radius_px,
                                                  2 * radius_px),
                                           self._pen((255, 255, 255), 1), QBrush(QColor(*color)))
            item.setFlag(item.GraphicsItemFlag.ItemIgnoresTransformations)
            item.setPos(x, y)
            item.setZValue(3)
            self._overlays.append(item)

    def add_polyline(self, points, color=(0, 220, 0), closed=False, width=2.0,
                     dashed=False) -> None:
        if len(points) < 2:
            return
        if closed:
            item = self.scene().addPolygon(QPolygonF([QPointF(x, y) for x, y in points]),
                                           self._pen(color, width, dashed))
        else:
            path = QPainterPath(QPointF(*points[0]))
            for x, y in points[1:]:
                path.lineTo(x, y)
            item = self.scene().addPath(path, self._pen(color, width, dashed))
        item.setZValue(2)
        self._overlays.append(item)

    def add_label(self, text: str, x: float, y: float, color=(255, 255, 0)) -> None:
        item = self.scene().addSimpleText(text)
        item.setBrush(QBrush(QColor(*color)))
        item.setFlag(item.GraphicsItemFlag.ItemIgnoresTransformations)
        item.setPos(x, y)
        item.setZValue(4)
        self._overlays.append(item)

    # --- interaction -------------------------------------------------------------------

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        delta = event.angleDelta()
        if self._nav_mode == self.PAN:
            if self.image_fits():
                # Nothing to move inside the picture, so let the page behind scroll instead
                # of swallowing the wheel.
                event.ignore()
                return
            dx, dy = -delta.x(), -delta.y()  # a trackpad also scrolls sideways on its own
            if event.modifiers() & Qt.ShiftModifier:
                dx, dy = dy, 0  # Shift+wheel scrolls sideways
            self._scroll_by(dx, dy)
            return
        factor = 1.25 if delta.y() > 0 else 0.8
        self.scale(factor, factor)

    def _scroll_by(self, dx: float, dy: float) -> None:
        h, v = self.horizontalScrollBar(), self.verticalScrollBar()
        h.setValue(round(h.value() + dx))
        v.setValue(round(v.value() + dy))

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.fit()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton
                and (event.modifiers() & Qt.AltModifier or self._nav_mode == self.PAN)):
            self._pan = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() in (Qt.LeftButton, Qt.RightButton) and self._has_image:
            scene = self.mapToScene(event.position().toPoint())
            if self.paint_enabled:
                self._painting = int(event.button().value)
                self.stroke.emit(scene.x(), scene.y(), self._painting, self.STROKE_PRESS)
                return
            if event.button() == Qt.LeftButton:
                self._handle = self._handle_at(event.position())
                if self._handle is not None:
                    self.setCursor(Qt.ClosedHandCursor)
                    self.handle_moved.emit(self._handle, scene.x(), scene.y(),
                                           self.STROKE_PRESS)
                    return
            self._press = (scene, event.button())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._pan is not None:
            delta = event.position() - self._pan
            self._pan = event.position()
            self._scroll_by(-delta.x(), -delta.y())
            return
        scene = self.mapToScene(event.position().toPoint())
        self.hovered.emit(scene.x(), scene.y())
        if self._painting is not None:
            self.stroke.emit(scene.x(), scene.y(), self._painting, self.STROKE_MOVE)
            return
        if self._handle is not None:
            self.handle_moved.emit(self._handle, scene.x(), scene.y(), self.STROKE_MOVE)
            return
        if self._press is None and not self.paint_enabled:
            if self._handle_at(event.position()) is not None:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self._rest_cursor()
        if self._press is not None and self.drag_enabled:
            start = self._press[0]
            rect = QRectF(start, scene).normalized()
            if self._rubber is None:
                self._rubber = self.scene().addRect(rect, self._pen((255, 255, 0), 1.5, True))
                self._rubber.setZValue(5)
            else:
                self._rubber.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._pan is not None:
            self._pan = None
            self._rest_cursor()
            return
        if self._painting is not None:
            scene = self.mapToScene(event.position().toPoint())
            self.stroke.emit(scene.x(), scene.y(), self._painting, self.STROKE_RELEASE)
            self._painting = None
            return
        if self._handle is not None:
            scene = self.mapToScene(event.position().toPoint())
            index, self._handle = self._handle, None
            self.setCursor(Qt.OpenHandCursor)
            self.handle_moved.emit(index, scene.x(), scene.y(), self.STROKE_RELEASE)
            return
        if self._press is not None:
            start, button = self._press
            end = self.mapToScene(event.position().toPoint())
            self._press = None
            if self._rubber is not None:
                self.scene().removeItem(self._rubber)
                self._rubber = None
            moved = abs(end.x() - start.x()) + abs(end.y() - start.y())
            if self.drag_enabled and moved > 3 / max(self.transform().m11(), 1e-6):
                self.dragged.emit(start.x(), start.y(), end.x(), end.y())
            else:
                self.clicked.emit(start.x(), start.y(), int(button.value))
        super().mouseReleaseEvent(event)


class ViewControls(QWidget):
    """A row of buttons for an :class:`ImageView`: Zoom/Pan mode, Fit and Fill.

    The mouse alone is ambiguous on a live view — the wheel cannot both zoom and scroll — so
    the mode is a button the user can see, and Fill sets the preview up at a glance.
    """

    def __init__(self, view: ImageView, parent=None):
        super().__init__(parent)
        self.view = view
        self.zoom_btn = QPushButton("Zoom")
        self.zoom_btn.setToolTip("Scroll wheel zooms in and out")
        self.pan_btn = QPushButton("Pan")
        self.pan_btn.setToolTip("Scroll wheel moves the picture up and down; drag to move it")
        self.modes = QButtonGroup(self)
        self.modes.setExclusive(True)
        for button, mode in ((self.zoom_btn, ImageView.ZOOM), (self.pan_btn, ImageView.PAN)):
            button.setCheckable(True)
            self.modes.addButton(button)
            button.clicked.connect(lambda _=False, m=mode: self.view.set_nav_mode(m))
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setToolTip("Show the whole picture (or double-click it)")
        self.fit_btn.clicked.connect(self.view.fit)
        self.fill_btn = QPushButton("Fill frame")
        self.fill_btn.setToolTip("Zoom until the picture fills the window; its long edge is "
                                 "cropped")
        self.fill_btn.clicked.connect(self.view.fill)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        for widget in (self.zoom_btn, self.pan_btn, self.fit_btn, self.fill_btn):
            row.addWidget(widget)
        view.nav_mode_changed.connect(self._mode_changed)
        self._mode_changed(view.nav_mode)

    def _mode_changed(self, mode: str) -> None:
        (self.pan_btn if mode == ImageView.PAN else self.zoom_btn).setChecked(True)
