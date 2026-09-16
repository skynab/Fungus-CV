"""Draw the app icon (a petri-dish style mark) and write PNG, .icns (macOS) and .ico files."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def draw(size: int = 1024) -> np.ndarray:
    s = size / 1024
    img = np.zeros((size, size, 4), np.uint8)
    c = size // 2
    # Rounded square background.
    bg = np.zeros((size, size), np.uint8)
    r = int(200 * s)
    cv2.rectangle(bg, (r, int(64 * s)), (size - r, size - int(64 * s)), 255, -1)
    cv2.rectangle(bg, (int(64 * s), r), (size - int(64 * s), size - r), 255, -1)
    for x in (r, size - r):
        for y in (r, size - r):
            cv2.circle(bg, (x, y), r - int(64 * s), 255, -1, cv2.LINE_AA)
    img[bg > 0] = (46, 58, 38, 255)  # dark green (BGRA)
    # Dish.
    cv2.circle(img, (c, c), int(360 * s), (225, 235, 238, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), int(360 * s), (150, 170, 175, 255), int(18 * s), cv2.LINE_AA)
    # Growing patch with a measurement front.
    theta = np.linspace(0, 2 * np.pi, 400)
    rad = 190 * s * (1 + 0.18 * np.sin(3 * theta) + 0.08 * np.cos(5 * theta))
    pts = np.c_[c - 40 * s + rad * np.cos(theta), c + 30 * s + rad * np.sin(theta)]
    cv2.fillPoly(img, [pts.astype(np.int32)], (60, 175, 110, 255), cv2.LINE_AA)
    cv2.line(img, (int(c - 260 * s), int(c - 200 * s)), (int(c + 260 * s), int(c - 200 * s)),
             (40, 40, 220, 255), int(26 * s), cv2.LINE_AA)
    return img


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgba = cv2.cvtColor(draw(), cv2.COLOR_BGRA2RGBA)
    base = Image.fromarray(rgba)
    base.save(out_dir / "icon.png")
    base.save(out_dir / "icon.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    if sys.platform == "darwin" and shutil.which("iconutil"):
        iconset = out_dir / "icon.iconset"
        iconset.mkdir(exist_ok=True)
        for size in (16, 32, 64, 128, 256, 512):
            base.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
            base.resize((2 * size, 2 * size), Image.LANCZOS).save(
                iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out_dir / "icon.icns")],
                       check=True)
        shutil.rmtree(iconset)


if __name__ == "__main__":
    main(Path(__file__).parent / "build")
