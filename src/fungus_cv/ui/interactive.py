"""Click-based tools. Each shows a scaled image and maps clicks back to full resolution."""

from __future__ import annotations

import cv2
import numpy as np

from fungus_cv.measure.geometry import Annotations
from fungus_cv.segment.color import ColorThresholdSegmenter, hsv_range_from_samples
from fungus_cv.segment.prompts import NEGATIVE, POSITIVE, FramePrompt

MAX_VIEW = (1400, 900)
ENTER_KEYS = (13, 10)
ESC = 27


class Cancelled(Exception):
    pass


class _View:
    """Image scaled to fit the screen, with a magnifier around the cursor."""

    def __init__(self, image: np.ndarray, window: str):
        self.image = image
        self.window = window
        h, w = image.shape[:2]
        self.scale = min(1.0, MAX_VIEW[0] / w, MAX_VIEW[1] / h)
        self.base = cv2.resize(image, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA) if self.scale < 1 else image.copy()
        self.mouse = (0, 0)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def to_full(self, x: int, y: int) -> tuple[float, float]:
        # Center of the clicked view pixel, in full-resolution pixel coordinates.
        return ((x + 0.5) / self.scale - 0.5, (y + 0.5) / self.scale - 0.5)

    def to_view(self, pt) -> tuple[int, int]:
        return (int(round(pt[0] * self.scale)), int(round(pt[1] * self.scale)))

    def magnifier(self, canvas: np.ndarray, zoom: int = 6, radius: int = 20) -> None:
        fx, fy = (int(round(v)) for v in self.to_full(*self.mouse))
        h, w = self.image.shape[:2]
        x0, y0 = max(0, fx - radius), max(0, fy - radius)
        patch = self.image[y0:min(h, fy + radius + 1), x0:min(w, fx + radius + 1)]
        if patch.size == 0:
            return
        big = cv2.resize(patch, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        cx, cy = (fx - x0) * zoom + zoom // 2, (fy - y0) * zoom + zoom // 2
        cv2.drawMarker(big, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, zoom * 3, 1)
        bh, bw = big.shape[:2]
        ch, cw = canvas.shape[:2]
        if bh < ch and bw < cw:
            canvas[ch - bh:, cw - bw:] = big
            cv2.rectangle(canvas, (cw - bw, ch - bh), (cw - 1, ch - 1), (255, 255, 255), 1)

    def show(self, canvas: np.ndarray, lines: list[str]) -> None:
        for i, text in enumerate(lines):
            y = 24 + 24 * i
            cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(self.window, canvas)


def annotate(image: np.ndarray, reference_file: str = "") -> Annotations:
    """Click base, tip, then the region-of-interest polygon."""
    view = _View(image, "fungus annotate")
    points: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):
        view.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(view.to_full(x, y))

    cv2.setMouseCallback(view.window, on_mouse)
    nudges = {ord("i"): (0, -1), ord("k"): (0, 1), ord("j"): (-1, 0), ord("l"): (1, 0)}
    prompts = [
        "1/3  Click the BASE: where growth starts (waterline, soil line)",
        "2/3  Click the TIP: the far end of the object (top of towel/stem)",
    ]
    try:
        while True:
            canvas = view.base.copy()
            if points:
                cv2.circle(canvas, view.to_view(points[0]), 5, (0, 200, 0), -1)
            if len(points) >= 2:
                cv2.line(canvas, view.to_view(points[0]), view.to_view(points[1]), (0, 200, 0), 2)
            roi = points[2:]
            if roi:
                poly = np.array([view.to_view(p) for p in roi], np.int32)
                cv2.polylines(canvas, [poly], len(roi) > 2, (0, 255, 255), 2)
            stage = prompts[len(points)] if len(points) < 2 else (
                f"3/3  Click corners AROUND the object ({len(roi)} so far), Enter when done")
            view.magnifier(canvas)
            last = f"last point ({points[-1][0]:.1f}, {points[-1][1]:.1f})" if points else ""
            view.show(canvas, [stage, "u: undo   r: restart   i/j/k/l: nudge last point 1 px   "
                               "Esc: cancel", last])

            key = cv2.waitKey(20) & 0xFF
            if key == ESC:
                raise Cancelled()
            if key == ord("u") and points:
                points.pop()
            elif key in nudges and points:
                dx, dy = nudges[key]
                points[-1] = (points[-1][0] + dx, points[-1][1] + dy)
            elif key == ord("r"):
                points.clear()
            elif key in ENTER_KEYS and len(roi) >= 3:
                h, w = image.shape[:2]
                return Annotations(
                    base=points[0], tip=points[1], roi=roi, image_size=(w, h),
                    reference_file=reference_file,
                )
    finally:
        cv2.destroyWindow(view.window)


def pick_color(image: np.ndarray, **range_kwargs) -> list:
    """Drag boxes over the target color; the live mask preview shows what is selected."""
    view = _View(image, "fungus pick-color")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    boxes: list[tuple[int, int, int, int]] = []
    drag: list = []

    def on_mouse(event, x, y, flags, param):
        view.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            drag[:] = [view.to_full(x, y)]
        elif event == cv2.EVENT_LBUTTONUP and drag:
            (x0, y0), (x1, y1) = drag[0], view.to_full(x, y)
            drag.clear()
            xa, xb = sorted((int(x0), int(x1)))
            ya, yb = sorted((int(y0), int(y1)))
            if xb - xa >= 2 and yb - ya >= 2:
                boxes.append((xa, ya, xb, yb))

    cv2.setMouseCallback(view.window, on_mouse)
    ranges: list = []
    preview_for: list = []
    preview = view.base
    try:
        while True:
            if boxes != preview_for:  # recompute the mask only when the selection changes
                preview_for = list(boxes)
                ranges, preview = [], view.base
                if boxes:
                    samples = np.concatenate([hsv[ya:yb, xa:xb].reshape(-1, 3)
                                              for xa, ya, xb, yb in boxes])
                    ranges = hsv_range_from_samples(samples, **range_kwargs)
                    mask = ColorThresholdSegmenter(ranges).segment(image)
                    tinted = image.copy()
                    tinted[mask] = (255, 0, 255)
                    shown = cv2.addWeighted(tinted, 0.45, image, 0.55, 0)
                    preview = cv2.resize(shown, (view.base.shape[1], view.base.shape[0]),
                                         interpolation=cv2.INTER_AREA)
            canvas = preview.copy()
            for xa, ya, xb, yb in boxes:
                cv2.rectangle(canvas, view.to_view((xa, ya)), view.to_view((xb, yb)),
                              (0, 255, 255), 1)
            view.magnifier(canvas)
            lines = ["Drag boxes over the TARGET color (only target pixels inside the box)",
                     "magenta = currently selected.  u: undo   Enter: accept   Esc: cancel"]
            if ranges:
                lines.append("ranges: " + "  ".join(f"{lo}-{hi}" for lo, hi in ranges))
            view.show(canvas, lines)

            key = cv2.waitKey(20) & 0xFF
            if key == ESC:
                raise Cancelled()
            if key == ord("u") and boxes:
                boxes.pop()
            elif key in ENTER_KEYS and ranges:
                return ranges
    finally:
        cv2.destroyWindow(view.window)


def prompt_target(image: np.ndarray, frame_file: str, preview=None,
                  existing: FramePrompt | None = None) -> FramePrompt:
    """Click on the target (left) and on things that are not the target (right).

    ``preview(image, prompt) -> mask`` is called after each change to show what the model
    would segment; it may be slow on a CPU.
    """
    view = _View(image, "fungus prompt")
    points: list[tuple[float, float]] = list(existing.points) if existing else []
    labels: list[int] = list(existing.labels) if existing else []
    box: list = [existing.box] if existing and existing.box else []
    state = {"box_mode": False, "drag": None, "dirty": True, "mask": None, "error": ""}

    def on_mouse(event, x, y, flags, param):
        view.mouse = (x, y)
        if state["box_mode"]:
            if event == cv2.EVENT_LBUTTONDOWN:
                state["drag"] = view.to_full(x, y)
            elif event == cv2.EVENT_LBUTTONUP and state["drag"] is not None:
                (x0, y0), (x1, y1) = state["drag"], view.to_full(x, y)
                state["drag"] = None
                if abs(x1 - x0) > 3 and abs(y1 - y0) > 3:
                    box[:] = [(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))]
                    state["box_mode"] = False
                    state["dirty"] = True
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(view.to_full(x, y))
            labels.append(POSITIVE)
            state["dirty"] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append(view.to_full(x, y))
            labels.append(NEGATIVE)
            state["dirty"] = True

    def current() -> FramePrompt | None:
        try:
            return FramePrompt(frame_file, list(points), list(labels), box[0] if box else None)
        except ValueError:
            return None

    cv2.setMouseCallback(view.window, on_mouse)
    try:
        while True:
            prompt = current()
            if state["dirty"]:
                state["dirty"] = False
                state["mask"], state["error"] = None, ""
                if prompt is not None and preview is not None:
                    view.show(view.base.copy(), ["running model..."])
                    cv2.waitKey(1)
                    try:
                        state["mask"] = preview(image, prompt)
                    except Exception as exc:  # show the problem instead of crashing the tool
                        state["error"] = str(exc)[:120]

            canvas = view.base.copy()
            if state["mask"] is not None:
                size = (canvas.shape[1], canvas.shape[0])
                small = cv2.resize(state["mask"].astype(np.uint8), size,
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                tint = canvas.copy()
                tint[small] = (255, 0, 255)
                canvas = cv2.addWeighted(tint, 0.45, canvas, 0.55, 0)
            for (px, py), label in zip(points, labels):
                color = (0, 220, 0) if label == POSITIVE else (0, 0, 255)
                cv2.circle(canvas, view.to_view((px, py)), 6, color, -1)
                cv2.circle(canvas, view.to_view((px, py)), 6, (255, 255, 255), 1)
            if box:
                x0, y0, x1, y1 = box[0]
                cv2.rectangle(canvas, view.to_view((x0, y0)), view.to_view((x1, y1)),
                              (0, 255, 255), 2)
            view.magnifier(canvas)
            lines = [
                "Left click: target   Right click: NOT target   b: draw box   c: clear",
                "u: undo   Enter: save   Esc: cancel   (magenta = model's mask)",
            ]
            if state["box_mode"]:
                lines.append("BOX MODE: drag a box around the target")
            if state["error"]:
                lines.append(f"model error: {state['error']}")
            view.show(canvas, lines)

            key = cv2.waitKey(20) & 0xFF
            if key == ESC:
                raise Cancelled()
            if key == ord("u") and points:
                points.pop()
                labels.pop()
                state["dirty"] = True
            elif key == ord("c"):
                points.clear()
                labels.clear()
                box.clear()
                state["dirty"] = True
            elif key == ord("b"):
                state["box_mode"] = not state["box_mode"]
            elif key in ENTER_KEYS and prompt is not None:
                return prompt
    finally:
        cv2.destroyWindow(view.window)
