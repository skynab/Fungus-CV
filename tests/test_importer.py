from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fungus_cv.capture.importer import image_timestamp, import_folder
from fungus_cv.storage import Experiment

from . import synthetic as syn


def write_jpeg(path, value=100, exif_time=None, offset=None):
    img = Image.fromarray(np.full((20, 30, 3), value, np.uint8))
    exif = Image.Exif()
    if exif_time:
        sub = exif.get_ifd(0x8769)
        sub[36867] = exif_time
        if offset:
            sub[36881] = offset
    img.save(path, exif=exif)


def test_exif_time_with_offset(tmp_path):
    p = tmp_path / "a.jpg"
    write_jpeg(p, exif_time="2026:09:20 08:00:00", offset="-05:00")
    ts, how = image_timestamp(p)
    assert how == "exif"
    assert ts == datetime(2026, 9, 20, 13, 0, tzinfo=timezone.utc)


def test_exif_time_uses_given_timezone(tmp_path):
    p = tmp_path / "a.jpg"
    write_jpeg(p, exif_time="2026:01:15 08:00:00")
    ts, _ = image_timestamp(p, "America/Chicago")  # CST, UTC-6 in January
    assert ts == datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)


def test_filename_time(tmp_path):
    p = tmp_path / "IMG_20260920_140500.jpg"
    write_jpeg(p)
    ts, how = image_timestamp(p, "UTC")
    assert how == "filename"
    assert ts == datetime(2026, 9, 20, 14, 5, 0, tzinfo=timezone.utc)


def test_mtime_fallback(tmp_path):
    p = tmp_path / "plant.jpg"
    write_jpeg(p)
    _, how = image_timestamp(p)
    assert how == "mtime"


def test_import_orders_by_time_and_skips_duplicates(experiment, tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    write_jpeg(src / "z_first.jpg", value=10, exif_time="2026:09:20 08:00:00", offset="+00:00")
    write_jpeg(src / "a_second.jpg", value=20, exif_time="2026:09:20 09:00:00", offset="+00:00")
    (src / "notes.txt").write_text("not an image")

    records = import_folder(experiment, src, camera="phone")
    assert [r.notes for r in records] == ["original: z_first.jpg", "original: a_second.jpg"]
    assert all(r.source == "import:exif" for r in records)
    assert (experiment.root / records[0].file).read_bytes() == (src / "z_first.jpg").read_bytes()

    again = import_folder(experiment, src, camera="phone")
    assert again == []
    assert len(experiment.read_frames()) == 2


def test_watch_imports_photos_as_they_appear(experiment, tmp_path):
    import os
    import time

    from fungus_cv.capture.importer import find_images, watch_folder

    incoming = tmp_path / "synced"
    incoming.mkdir()
    batches = []

    def drop(name: str, minute: int) -> Path:
        path = incoming / name
        cv2.imwrite(str(path), syn.scene(100 + minute))
        return path

    drop("IMG_20260920_140500.jpg", 1)
    assert watch_folder(experiment.root, incoming, passes=1, settle_seconds=0,
                        on_batch=batches.append) == 1
    assert len(batches) == 1 and batches[0][0].source == "import:filename"

    # A file still being written is left for the next pass; the first one is already old.
    settled = time.time() - 120
    os.utime(incoming / "IMG_20260920_140500.jpg", (settled, settled))
    fresh = drop("IMG_20260920_140600.jpg", 2)
    assert find_images(incoming, settle_seconds=60) == [incoming / "IMG_20260920_140500.jpg"]
    assert watch_folder(experiment.root, incoming, passes=1, settle_seconds=60) == 0
    old = time.time() - 120
    os.utime(fresh, (old, old))
    assert watch_folder(experiment.root, incoming, passes=1, settle_seconds=60) == 1

    # Nothing new: no duplicates, whatever the file name.
    assert watch_folder(experiment.root, incoming, passes=1, settle_seconds=0) == 0
    (incoming / "copy.jpg").write_bytes((incoming / "IMG_20260920_140500.jpg").read_bytes())
    assert watch_folder(experiment.root, incoming, passes=1, settle_seconds=0) == 0
    assert len(Experiment(experiment.root).read_frames()) == 2


def test_watch_survives_a_missing_folder(experiment, tmp_path, caplog):
    from fungus_cv.capture.importer import watch_folder

    caplog.set_level("WARNING")
    assert watch_folder(experiment.root, tmp_path / "not-mounted", passes=1,
                        settle_seconds=0) == 0
    assert "not there" in caplog.text


def test_import_watch_cli(experiment, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from fungus_cv.capture import importer
    from fungus_cv.cli import app

    incoming = tmp_path / "in"
    incoming.mkdir()
    cv2.imwrite(str(incoming / "IMG_20260920_140500.jpg"), syn.scene(120))
    real = importer.watch_folder
    monkeypatch.setattr(importer, "watch_folder",
                        lambda *a, **k: real(*a, **{**k, "passes": 1, "poll_seconds": 0}))
    result = CliRunner().invoke(app, ["import", str(experiment.root), str(incoming), "--watch",
                                      "--settle", "0s"])
    assert result.exit_code == 0, result.output
    assert "imported 1 new photo(s)" in result.output
    assert "Stopped after importing 1" in result.output
