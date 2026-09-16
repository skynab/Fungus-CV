from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from fungus_cv.cli import app
from fungus_cv.storage import Experiment, FrameRecord, frame_filename, iso_utc, parse_iso_utc

runner = CliRunner()


def test_frame_filename_is_sortable_and_windows_safe():
    ts = datetime(2026, 9, 20, 14, 5, 0, 123000, tzinfo=timezone.utc)
    name = frame_filename(ts, "cam0", "PNG")
    assert name == "2026-09-20T14-05-00.123Z_cam0.png"
    assert ":" not in name


def test_iso_roundtrip():
    ts = datetime(2026, 9, 20, 14, 5, 0, 123000, tzinfo=timezone.utc)
    assert parse_iso_utc(iso_utc(ts)) == ts


def test_same_timestamp_gets_unique_file(experiment):
    ts = datetime(2026, 9, 20, tzinfo=timezone.utc)
    a = experiment.save_bytes(b"a", ts, "cam0", "png")
    b = experiment.save_bytes(b"b", ts, "cam0", "png")
    assert a != b and a.read_bytes() == b"a" and b.read_bytes() == b"b"


def test_frames_csv_roundtrip(experiment):
    experiment.append_frame(
        FrameRecord(timestamp_utc="2026-09-20T00:00:00.000Z", camera="cam0", status="ok",
                    camera_settings={"exposure": -6.0})
    )
    rows = experiment.read_frames()
    assert rows[0]["camera_settings"] == {"exposure": -6.0}


def test_create_refuses_to_overwrite(experiment):
    with pytest.raises(FileExistsError):
        Experiment.create(experiment.root)


def test_cli_init_and_status(tmp_path):
    target = tmp_path / "dye"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0, result.output
    assert (target / "config.yaml").exists()

    result = runner.invoke(app, ["status", str(target)])
    assert result.exit_code == 0, result.output
    assert "No frames yet" in result.output


def test_cli_missing_experiment(tmp_path):
    result = runner.invoke(app, ["status", str(tmp_path / "nope")])
    assert result.exit_code == 1
