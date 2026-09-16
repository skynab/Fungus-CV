"""ArUco markers: detection, pixel-to-mm scale and printable marker sheets."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fungus_cv.quality import to_gray

Markers = dict[int, np.ndarray]  # marker id -> 4x2 float32 corners (clockwise from top-left)


def aruco_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"unknown ArUco dictionary {name!r}, e.g. DICT_4X4_50")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_markers(image: np.ndarray, dictionary: str = "DICT_4X4_50") -> Markers:
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel refinement: without it corners sit ~0.5 px inside the true edge, which
    # biases the scale by about 1 px per marker side.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dictionary(dictionary), params)
    corners, ids, _ = detector.detectMarkers(to_gray(image))
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float32) for i, c in zip(ids.ravel(), corners)}


@dataclass
class Scale:
    mm_per_px: float
    std_mm_per_px: float  # spread between individual marker edges
    se_mm_per_px: float  # standard error of the mean
    n_edges: int
    marker_ids: list[int]

    def to_dict(self) -> dict:
        return {
            "mm_per_px": self.mm_per_px,
            "std_mm_per_px": self.std_mm_per_px,
            "se_mm_per_px": self.se_mm_per_px,
            "n_edges": self.n_edges,
            "marker_ids": self.marker_ids,
        }


def scale_from_markers(markers: Markers, size_mm: float) -> Scale | None:
    """mm per pixel from the edge lengths of every detected marker.

    Assumes the markers lie in the same plane as the object and the camera looks at that
    plane roughly square-on. Different edges disagreeing (large std) suggests a tilted view.
    """
    if not markers:
        return None
    edges = []
    for corners in markers.values():
        for i in range(4):
            edges.append(float(np.linalg.norm(corners[i] - corners[(i + 1) % 4])))
    per_edge = size_mm / np.asarray(edges)
    n = len(per_edge)
    std = float(per_edge.std(ddof=1)) if n > 1 else 0.0
    return Scale(
        mm_per_px=float(size_mm / np.mean(edges)),
        std_mm_per_px=std,
        se_mm_per_px=std / np.sqrt(n),
        n_edges=n,
        marker_ids=sorted(markers),
    )


PAGE_SIZES_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}


def marker_sheet(
    ids: list[int],
    size_mm: float = 30.0,
    dictionary: str = "DICT_4X4_50",
    page: str = "letter",
    dpi: int = 300,
) -> np.ndarray:
    """A printable page of markers with a 100 mm check ruler.

    Print at 100% / "actual size" (no fit-to-page), then measure a marker with a ruler and
    put that measured value in ``analysis.markers.size_mm``.
    """
    px_per_mm = dpi / 25.4
    page_w, page_h = (int(round(v * px_per_mm)) for v in PAGE_SIZES_MM[page])
    sheet = np.full((page_h, page_w), 255, np.uint8)
    dict_ = aruco_dictionary(dictionary)

    margin = int(15 * px_per_mm)
    quiet = int(max(5.0, size_mm * 0.25) * px_per_mm)  # white border the detector needs
    side = int(round(size_mm * px_per_mm))
    step = side + 2 * quiet
    cols = max(1, (page_w - 2 * margin) // step)
    rows = max(1, (page_h - 2 * margin - int(40 * px_per_mm)) // step)
    if len(ids) > cols * rows:
        raise ValueError(f"{len(ids)} markers of {size_mm} mm don't fit; max {cols * rows}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    for n, marker_id in enumerate(ids):
        r, c = divmod(n, cols)
        x = margin + c * step + quiet
        y = margin + r * step + quiet
        sheet[y : y + side, x : x + side] = cv2.aruco.generateImageMarker(dict_, marker_id, side)
        cv2.putText(sheet, f"id {marker_id}", (x, y + side + int(quiet * 0.7)), font,
                    dpi / 300, 0, max(1, dpi // 150))

    # 100 mm ruler with 10 mm ticks to confirm the print scale.
    y = page_h - margin - int(10 * px_per_mm)
    x0 = margin
    x1 = x0 + int(round(100 * px_per_mm))
    thick = max(1, dpi // 100)
    cv2.line(sheet, (x0, y), (x1, y), 0, thick)
    for mm in range(0, 101, 10):
        x = x0 + int(round(mm * px_per_mm))
        cv2.line(sheet, (x, y - int(3 * px_per_mm)), (x, y), 0, thick)
    label = (f"100 mm check ruler | markers {size_mm:g} mm, {dictionary} | "
             "print at 100% (actual size)")
    cv2.putText(sheet, label, (x0, y + int(7 * px_per_mm)), font, dpi / 400, 0,
                max(1, dpi // 150))
    return sheet
