"""Draw masks and measurements on frames for visual checking."""

from __future__ import annotations

import cv2
import numpy as np

from fungus_cv.measure.geometry import Annotations, ExtentMeasurement
from fungus_cv.measure.path import Polyline

MASK_COLOR = (255, 0, 255)  # magenta: stands out against blue dye and green moss
ROI_COLOR = (0, 255, 255)
AXIS_COLOR = (0, 200, 0)
FRONT_COLOR = (0, 0, 255)
REFERENCE_COLOR = (255, 160, 0)


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float) -> None:
    thick = max(1, int(round(scale * 2)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)


def draw_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    ann: Annotations,
    m: ExtentMeasurement,
    row: dict,
    path: Polyline | None = None,
    reference_mask: np.ndarray | None = None,
) -> np.ndarray:
    out = image.copy()
    tint = out.copy()
    tint[mask] = MASK_COLOR
    out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)

    h, w = out.shape[:2]
    lw = max(1, round(max(h, w) / 600))
    roi = np.round(np.array(ann.roi)).astype(np.int32)
    cv2.polylines(out, [roi], True, ROI_COLOR, lw)

    if reference_mask is not None:
        contours, _ = cv2.findContours(reference_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, REFERENCE_COLOR, lw)

    path = path or Polyline([ann.base, ann.tip])
    cv2.polylines(out, [np.round(path.dense[::4]).astype(np.int32)], False, AXIS_COLOR, lw)
    cv2.circle(out, tuple(np.round(path.dense[0]).astype(int)), 3 * lw, AXIS_COLOR, -1)

    # Front: a line across the path at the measured extent.
    half = max(20.0, 0.1 * min(path.length, 400))
    front, tangent = path.point_at(m.extent_px)
    normal = np.array([-tangent[1], tangent[0]])
    p1 = tuple(np.round(front - normal * half).astype(int))
    p2 = tuple(np.round(front + normal * half).astype(int))
    cv2.line(out, p1, p2, FRONT_COLOR, 2 * lw)

    scale = max(0.5, max(h, w) / 1600)
    line_h = int(32 * scale)
    if row.get("extent_mm") != "":
        extent = f"extent {row['extent_mm']} +/- {row['extent_mm_unc']} mm"
    else:
        extent = f"extent {row['extent_px']} px"
    lines = [
        row["timestamp_utc"],
        f"{extent}   coverage {row['coverage_pct']}%   length covered "
        f"{row.get('covered_length_pct', '')}%",
        f"align {row['align_method']}" + (f"  rms {row['align_rms_px']} px"
                                           if row.get("align_rms_px") != "" else ""),
    ]
    if row.get("flags"):
        lines.append(f"FLAGS: {row['flags']}")
    for i, text in enumerate(lines):
        _text(out, text, (10, line_h * (i + 1)), scale)
    return out
