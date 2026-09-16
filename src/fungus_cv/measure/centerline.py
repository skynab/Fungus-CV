"""Centerline of an elongated object (a plant stem) from its mask."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from fungus_cv.measure.path import Polyline


class CenterlineError(ValueError):
    pass


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning to a 1-px wide skeleton (pure numpy, no opencv-contrib needed)."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            p1 = img[1:-1, 1:-1]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            b = sum(ring[:-1])
            a = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (p1 == 1) & (b >= 2) & (b <= 6) & (a == 1) & c
            if remove.any():
                p1[remove] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def _component_near(mask: np.ndarray, point) -> np.ndarray:
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        raise CenterlineError("reference mask is empty")
    x, y = (int(round(v)) for v in point)
    h, w = mask.shape
    if 0 <= y < h and 0 <= x < w and labels[y, x]:
        return labels == labels[y, x]
    ys, xs = np.nonzero(labels)
    k = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    return labels == labels[ys[k], xs[k]]


def _longest_path_from(skeleton: np.ndarray, start_xy) -> np.ndarray:
    ys, xs = np.nonzero(skeleton)
    if len(xs) < 2:
        raise CenterlineError("reference object is too small for a centerline")
    index = -np.ones(skeleton.shape, np.int64)
    index[ys, xs] = np.arange(len(xs))
    rows, cols, weights = [], [], []
    h, w = skeleton.shape
    for dy, dx, wt in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5)):
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        ok[ok] = skeleton[ny[ok], nx[ok]]
        a, b = index[ys[ok], xs[ok]], index[ny[ok], nx[ok]]
        rows += [a, b]
        cols += [b, a]
        weights += [np.full(len(a), wt)] * 2
    graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(len(xs), len(xs))).tocsr()
    sx, sy = start_xy
    start = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
    dist, pred = dijkstra(graph, indices=start, return_predecessors=True)
    dist[~np.isfinite(dist)] = -1
    node = int(np.argmax(dist))
    chain = [node]
    while pred[node] >= 0:
        node = int(pred[node])
        chain.append(node)
    chain.reverse()
    return np.c_[xs[chain], ys[chain]].astype(np.float64)


def _extend_to_edge(mask: np.ndarray, end: np.ndarray, direction: np.ndarray,
                    max_px: float) -> np.ndarray:
    """March from ``end`` along ``direction`` until leaving the mask."""
    h, w = mask.shape
    last = end
    for step in np.arange(0.5, max_px, 0.5):
        p = end + direction * step
        x, y = int(round(p[0])), int(round(p[1]))
        if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
            break
        last = p
    # The last inside pixel's center is half a pixel short of the object's edge.
    return last + direction * 0.5


def centerline_from_mask(
    mask: np.ndarray,
    base_point,
    smooth_px: float | None = None,
) -> Polyline:
    """Centerline from ``base_point`` to the far end of the object containing it.

    Steps: take the connected object nearest the base, thin it to a skeleton, follow the
    longest skeleton path from the end nearest the base (side branches such as leaves are
    dropped), extend the far end to the object's edge, start the path at the base point,
    and smooth over roughly the object's width.
    """
    obj = _component_near(mask, base_point)
    ys, xs = np.nonzero(obj)
    pad = 2
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = xs.max() + pad + 1, ys.max() + pad + 1
    crop = obj[y0:y1, x0:x1]
    base = np.asarray(base_point, np.float64)

    skeleton = thin(crop)
    chain = _longest_path_from(skeleton, (base[0] - x0, base[1] - y0)) + (x0, y0)

    half_width = float(np.median(cv2.distanceTransform(crop.astype(np.uint8), cv2.DIST_L2, 5)
                                 [skeleton]))
    width = max(2.0, 2 * half_width)
    tail = chain[-min(len(chain), max(3, int(2 * width))):]
    direction = tail[-1] - tail[0]
    if np.linalg.norm(direction) > 0:
        direction /= np.linalg.norm(direction)
        tip = _extend_to_edge(obj, chain[-1], direction, max_px=3 * width + 5)
        chain = np.vstack([chain, tip])

    # Start exactly at the base; drop skeleton points the base already covers.
    near_base = np.linalg.norm(chain - base, axis=1) < half_width
    first = int(np.argmax(~near_base)) if (~near_base).any() else len(chain) - 1
    points = np.vstack([base, chain[first:]])
    path = Polyline(points)
    return path.smoothed(smooth_px if smooth_px is not None else max(9.0, 2 * width))
