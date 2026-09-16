from datetime import datetime, timezone

import numpy as np
from PIL import Image

from fungus_cv.capture.importer import image_timestamp, import_folder


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
