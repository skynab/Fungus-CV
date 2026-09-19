"""Zoomable image canvas with overlays and click/drag events in image coordinates."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from fungus_cv.gui import theme
from fungus_cv.gui.qt_util import bgr_to_pixmap


class ImageView(QGraphicsView):
    """Shows one image. Wheel/trackpad scroll zooms, middle-drag or Alt(Option)+drag pans,
    double-click fits.

    Emits ``clicked(x, y, button)`` and ``dragged(x0, y0, x1, y1)`` in full-resolution pixel
    coordinates (pixel centres are integers), so annotations are independent of zoom. With
    ``paint_enabled``, left/right drags emit ``stroke(x, y, button, phase)`` instead
    (phase: 0 press, 1 move, 2 release), for brushes.
    """

    STROKE_PRESS, STROKE_MOVE, STROKE_RELEASE = 0, 1, 2

    clicked = Signal(float, float, int)
    dragged = Signal(float, float, float, float)
    hovered = Signal(float, float)
    stroke = Signal(float, float, int, int)

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
        self.drag_enabled = False
        self.paint_enabled = False
        self._painting: int | None = None  # mouse button of the stroke in progress
        self._has_image = False
        self._rubber = None

    # --- content -----------------------------------------------------------------------

    def set_image(self, image: np.ndarray | None, keep_view: bool = False) -> None:
        if image is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._has_image = False
            return
        first = not self._has_image
        self._pixmap_item.setPixmap(bgr_to_pixmap(image))
        self.scene().setSceneRect(QRectF(-0.5, -0.5, image.shape[1], image.shape[0]))
        self._has_image = True
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
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

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
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.fit()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier):
            self._pan = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() in (Qt.LeftButton, Qt.RightButton) and self._has_image:
            scene = self.mapToScene(event.position().toPoint())
            if self.paint_enabled:
                self._painting = int(event.button().value)
                self.stroke.emit(scene.x(), scene.y(), self._painting, self.STROKE_PRESS)
                return
            self._press = (scene, event.button())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._pan is not None:
            delta = event.position() - self._pan
            self._pan = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        scene = self.mapToScene(event.position().toPoint())
        self.hovered.emit(scene.x(), scene.y())
        if self._painting is not None:
            self.stroke.emit(scene.x(), scene.y(), self._painting, self.STROKE_MOVE)
            return
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
            self.unsetCursor()
            return
        if self._painting is not None:
            scene = self.mapToScene(event.position().toPoint())
            self.stroke.emit(scene.x(), scene.y(), self._painting, self.STROKE_RELEASE)
            self._painting = None
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
