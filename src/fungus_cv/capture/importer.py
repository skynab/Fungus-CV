"""Import existing photos (phone, trail camera, Raspberry Pi, ...) into an experiment."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image

from fungus_cv.quality import mean_brightness, sharpness
from fungus_cv.storage import Experiment, FrameRecord, iso_utc, sha256_bytes

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

_EXIF_IFD = 0x8769
_TAG_DATETIME = 306
_TAG_DATETIME_ORIGINAL = 36867
_TAG_OFFSET_TIME_ORIGINAL = 36881
_TAG_SUBSEC_TIME_ORIGINAL = 37521

# e.g. IMG_20260920_140500.jpg, PXL_20260920_140500123.jpg, 2026-09-20 14.05.00.png
_FILENAME_TIME = re.compile(
    r"(?<!\d)(\d{4})[-_]?(\d{2})[-_]?(\d{2})[T_\- ]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})"
)


def _to_utc(naive: datetime, tz: tzinfo | None) -> datetime:
    """Interpret a naive wall-clock time in ``tz``, or in this computer's local zone.

    The local case goes through ``astimezone()`` so daylight saving is resolved per date,
    not with today's offset.
    """
    aware = naive.replace(tzinfo=tz) if tz is not None else naive.astimezone()
    return aware.astimezone(timezone.utc)


def _exif_time(path: Path, tz: tzinfo | None) -> datetime | None:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            sub = exif.get_ifd(_EXIF_IFD)
    except Exception:
        return None
    raw = sub.get(_TAG_DATETIME_ORIGINAL) or exif.get(_TAG_DATETIME)
    if not raw:
        return None
    try:
        ts = datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    subsec = str(sub.get(_TAG_SUBSEC_TIME_ORIGINAL) or "").strip("\x00 ")
    if subsec.isdigit():
        ts = ts.replace(microsecond=int(subsec.ljust(6, "0")[:6]))
    offset = str(sub.get(_TAG_OFFSET_TIME_ORIGINAL) or "").strip("\x00 ")
    if re.fullmatch(r"[+-]\d{2}:\d{2}", offset):
        return datetime.fromisoformat(ts.isoformat() + offset).astimezone(timezone.utc)
    return _to_utc(ts, tz)


def _filename_time(path: Path, tz: tzinfo | None) -> datetime | None:
    match = _FILENAME_TIME.search(path.stem)
    if not match:
        return None
    try:
        ts = datetime(*(int(g) for g in match.groups()))
    except ValueError:
        return None
    return _to_utc(ts, tz)


def image_timestamp(path: Path, timezone_name: str | None = None) -> tuple[datetime, str]:
    """Best available capture time (UTC) and how it was found.

    Order: EXIF DateTimeOriginal, a date/time in the file name, file modification time.
    Times without an explicit offset are interpreted in ``timezone_name`` (default: this
    computer's local time zone).
    """
    tz = ZoneInfo(timezone_name) if timezone_name else None
    if (ts := _exif_time(path, tz)) is not None:
        return ts, "exif"
    if (ts := _filename_time(path, tz)) is not None:
        return ts, "filename"
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc), "mtime"


def find_images(folder: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in Path(folder).glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def import_folder(
    experiment: Experiment,
    folder: Path,
    camera: str = "import",
    timezone_name: str | None = None,
    recursive: bool = False,
) -> list[FrameRecord]:
    """Copy images into ``frames/`` unchanged (keeps EXIF, no re-encoding) and log them.

    Files already in the experiment (same SHA-256) are skipped, so re-running is safe.
    """
    known = {row["sha256"] for row in experiment.read_frames() if row.get("sha256")}
    candidates = []
    for path in find_images(folder, recursive):
        ts, how = image_timestamp(path, timezone_name)
        candidates.append((ts, path, how))
    candidates.sort()

    records = []
    for ts, path, how in candidates:
        data = path.read_bytes()
        digest = sha256_bytes(data)
        if digest in known:
            log.info("skipping %s (already imported)", path.name)
            continue
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            log.warning("skipping %s (not a readable image)", path)
            continue
        dest = experiment.save_bytes(data, ts, camera, path.suffix.lower().lstrip("."))
        record = FrameRecord(
            timestamp_utc=iso_utc(ts),
            camera=camera,
            status="ok",
            file=experiment.relative(dest),
            sha256=digest,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            source=f"import:{how}",
            mean_brightness=round(mean_brightness(image), 2),
            sharpness=round(sharpness(image), 2),
            notes=f"original: {path.name}",
        )
        experiment.append_frame(record)
        known.add(digest)
        records.append(record)
        if how == "mtime":
            log.warning("%s: no EXIF or filename time; used file modification time", path.name)
    return records
