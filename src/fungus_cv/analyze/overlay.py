"""Draw masks and measurements on frames for visual checking."""

from __future__ import annotations

import cv2
import numpy as np

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
    plots: list[tuple],
    reference_mask: np.ndarray | None = None,
) -> np.ndarray:
    """``plots`` holds ``(plot, measurement, row, path or None)`` for every measured plot."""
    out = image.copy()
    tint = out.copy()
    tint[mask] = MASK_COLOR
    out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)

    h, w = out.shape[:2]
    lw = max(1, round(max(h, w) / 600))
    scale = max(0.5, max(h, w) / 1600)
    if reference_mask is not None:
        contours, _ = cv2.findContours(reference_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, REFERENCE_COLOR, lw)

    for plot, m, row, path in plots:
        polygon = np.round(np.array(plot.polygon)).astype(np.int32)
        cv2.polylines(out, [polygon], True, ROI_COLOR, lw)
        if len(plots) > 1:
            x, y = polygon.min(axis=0)
            area = row.get("target_area_mm2", "")
            label = f"{plot.name}: {float(row['coverage_pct']):.1f}%"
            if area != "":
                label += f" ({float(area) / 1e6:.3f} m2)" if float(area) >= 1e5 \
                    else f" ({float(area):.0f} mm2)"
            _text(out, label, (int(x) + 4, int(y) + int(26 * scale)), scale * 0.8)
        if path is None:
            continue
        cv2.polylines(out, [np.round(path.dense[::4]).astype(np.int32)], False, AXIS_COLOR, lw)
        cv2.circle(out, tuple(np.round(path.dense[0]).astype(int)), 3 * lw, AXIS_COLOR, -1)
        # Front: a line across the path at the measured extent.
        half = max(20.0, 0.1 * min(path.length, 400))
        front, tangent = path.point_at(m.extent_px)
        normal = np.array([-tangent[1], tangent[0]])
        p1 = tuple(np.round(front - normal * half).astype(int))
        p2 = tuple(np.round(front + normal * half).astype(int))
        cv2.line(out, p1, p2, FRONT_COLOR, 2 * lw)

    first = plots[0][2]
    lines = [first["timestamp_utc"]]
    if len(plots) == 1:
        row = first
        if row.get("extent_mm", "") != "":
            extent = f"extent {row['extent_mm']} +/- {row['extent_mm_unc']} mm   "
        elif row.get("extent_px", "") != "":
            extent = f"extent {row['extent_px']} px   "
        else:
            extent = ""
        covered = (f"   length covered {row['covered_length_pct']}%"
                   if row.get("covered_length_pct", "") != "" else "")
        lines.append(f"{extent}coverage {row['coverage_pct']}%{covered}")
    align = f"align {first['align_method']}"
    if first.get("align_rms_px", "") != "":
        align += f"  rms {first['align_rms_px']} px"
    lines.append(align)
    if first.get("flags"):
        lines.append(f"FLAGS: {first['flags']}")
    line_h = int(32 * scale)
    for i, text in enumerate(lines):
        _text(out, text, (10, line_h * (i + 1)), scale)
    return out
