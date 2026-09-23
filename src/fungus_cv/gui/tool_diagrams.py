"""Little drawings of what to click for each tool on the Set up measurement page.

The page asks for clicks whose meaning is hard to guess from a word ("Base", "Region"), so
each tool shows a sketch of the same clicks on a made-up scene, in the colours the real
overlay uses. They are drawn rather than shipped as pictures, so they follow the theme and
cost nothing to keep in step with the tools.
"""

from __future__ import annotations

import cv2
import numpy as np

from fungus_cv.gui import theme

W, H = 230, 148  # the drawing's size in layout pixels; it is rendered at SCALE times this
SCALE = 2  # drawn larger, shown at W×H, so it stays sharp on a high-density screen

GREEN = (0, 220, 0)  # the tool colours, as in SetupPage.COLORS (RGB)
YELLOW = (255, 220, 0)
CYAN = (0, 200, 255)
ORANGE = (255, 160, 0)
MAGENTA = (255, 0, 255)  # the colour overlay's tint


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


def _hex_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (4, 2, 0))


SURFACE = _hex_bgr(theme.SURFACE)
SOIL = _hex_bgr(theme.NEUTRAL[800])
PLANT = _hex_bgr(theme.NEUTRAL[600])
DISH = _hex_bgr(theme.NEUTRAL[700])
CARD = _hex_bgr(theme.NEUTRAL[300])
INK = _hex_bgr(theme.NEUTRAL[400])


def _p(x: float, y: float) -> tuple[int, int]:
    """A point in layout coordinates, scaled to the drawing."""
    return int(round(x * SCALE)), int(round(y * SCALE))


def _blank() -> np.ndarray:
    return np.full((H * SCALE, W * SCALE, 3), SURFACE, np.uint8)


def _line(img, a, b, color, width=2) -> None:
    cv2.line(img, _p(*a), _p(*b), color, width * SCALE, cv2.LINE_AA)


def _poly(img, points, color, width=2, closed=True) -> None:
    pts = np.array([_p(*q) for q in points], np.int32)
    cv2.polylines(img, [pts], closed, color, width * SCALE, cv2.LINE_AA)


def _dashed(img, points, color, width=1) -> None:
    """A dashed outline, like the rubber band the colour tool draws."""
    closed = [*points, points[0]]
    for a, b in zip(closed, closed[1:]):
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        steps = max(int(length / 5), 1)
        for i in range(0, steps, 2):
            t0, t1 = i / steps, min((i + 1) / steps, 1.0)
            _line(img, (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                  (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1), color, width)


def _tint(img, points, color, alpha=0.45) -> None:
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.array([_p(*q) for q in points], np.int32)], 255)
    img[mask > 0] = (img[mask > 0] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def _click(img, point, rgb, order: int | None = None) -> None:
    """One click: a filled dot ringed in white, as the view draws placed points."""
    centre = _p(*point)
    cv2.circle(img, centre, 5 * SCALE, _bgr(rgb), -1, cv2.LINE_AA)
    cv2.circle(img, centre, 5 * SCALE, (255, 255, 255), max(SCALE // 2, 1), cv2.LINE_AA)
    if order is None:
        return
    # Keep the number outside the shape the clicks outline, and inside the drawing.
    x, y = point
    dx = -14 if x > W * 0.75 else 8
    dy = 16 if y > H * 0.5 else -8
    cv2.putText(img, str(order), _p(x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.34 * SCALE,
                (255, 255, 255), SCALE, cv2.LINE_AA)


def _plant(img) -> tuple[tuple[float, float], tuple[float, float]]:
    """A stem growing out of soil; returns where its base and its tip are."""
    cv2.rectangle(img, _p(0, 116), _p(W, H), SOIL, -1)
    stem = [(112, 116), (110, 94), (117, 70), (112, 46), (119, 24)]
    _poly(img, stem, PLANT, 4, closed=False)
    cv2.ellipse(img, _p(139, 62), _p(18, 8), -25, 0, 360, PLANT, -1, cv2.LINE_AA)  # a leaf
    cv2.ellipse(img, _p(90, 88), _p(16, 7), 20, 0, 360, PLANT, -1, cv2.LINE_AA)
    return stem[0], stem[-1]


def _dish(img, centre=(W / 2, 74), axes=(78, 52)) -> None:
    cv2.ellipse(img, _p(*centre), _p(*axes), 0, 0, 360, DISH, 3 * SCALE, cv2.LINE_AA)


def _base(img) -> None:
    base, _ = _plant(img)
    _click(img, base, GREEN, 1)


def _path(img) -> None:
    base, tip = _plant(img)
    points = [base, (112, 88), (114, 56), tip]
    _poly(img, points, _bgr(GREEN), 1, closed=False)
    for i, point in enumerate(points):
        _click(img, point, GREEN, i + 1)


def _roi(img) -> None:
    _dish(img)
    cv2.ellipse(img, _p(W / 2, 76), _p(34, 24), 0, 0, 360, PLANT, -1, cv2.LINE_AA)
    corners = [(66, 42), (164, 42), (164, 108), (66, 108)]
    _poly(img, corners, _bgr(YELLOW), 1)
    for i, corner in enumerate(corners):
        _click(img, corner, YELLOW, i + 1)


def _plot(img) -> None:
    for x in (14, 128):  # neighbouring plots, to show one is picked out of several
        cv2.rectangle(img, _p(x, 30), _p(x + 88, 116), DISH, 2 * SCALE)
    corners = [(128, 30), (216, 30), (216, 116), (128, 116)]
    _poly(img, corners, _bgr(CYAN), 1)
    for i, corner in enumerate(corners):
        _click(img, corner, CYAN, i + 1)
    cv2.putText(img, "corners, then Enter", _p(14, 138), cv2.FONT_HERSHEY_SIMPLEX,
                0.34 * SCALE, INK, SCALE, cv2.LINE_AA)


def _patch(img) -> None:
    _dish(img, centre=(146, 78), axes=(62, 44))
    corners = [(22, 24), (76, 24), (76, 62), (22, 62)]
    cv2.rectangle(img, _p(*corners[0]), _p(*corners[2]), CARD, -1)
    _poly(img, corners, _bgr(ORANGE), 1)
    for corner in corners:
        _click(img, corner, ORANGE)
    cv2.putText(img, "grey card", _p(20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.34 * SCALE, INK,
                SCALE, cv2.LINE_AA)


def _color(img) -> None:
    _dish(img)
    blob = [(92, 46), (140, 40), (162, 74), (138, 106), (94, 100), (76, 72)]
    cv2.fillPoly(img, [np.array([_p(*q) for q in blob], np.int32)], PLANT, cv2.LINE_AA)
    _tint(img, blob, _bgr(MAGENTA))
    for box in ([(92, 50), (126, 50), (126, 76), (92, 76)],
                [(128, 74), (158, 74), (158, 98), (128, 98)]):
        _dashed(img, box, _bgr(YELLOW), 2)
    cv2.putText(img, "drag boxes, don't click", _p(30, 136), cv2.FONT_HERSHEY_SIMPLEX,
                0.34 * SCALE, INK, SCALE, cv2.LINE_AA)


_DRAW = {"base": _base, "path": _path, "roi": _roi, "plot": _plot, "patch": _patch,
         "color": _color}


def diagram(tool: str) -> np.ndarray | None:
    """A BGR drawing of ``tool``'s clicks, or None if there is none for it."""
    draw = _DRAW.get(tool)
    if draw is None:
        return None
    img = _blank()
    draw(img)
    return img


def pixmap(tool: str):
    """``diagram`` as a QPixmap sized for the layout (W×H), drawn at SCALE for sharpness."""
    from fungus_cv.gui.qt_util import bgr_to_pixmap

    image = diagram(tool)
    if image is None:
        return None
    result = bgr_to_pixmap(image)
    result.setDevicePixelRatio(SCALE)
    return result
