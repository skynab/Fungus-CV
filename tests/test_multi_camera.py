"""Several cameras: each is annotated, measured and reported on its own, then combined."""

from datetime import timedelta

import pytest
from typer.testing import CliRunner

from fungus_cv.analyze.combine import combine
from fungus_cv.analyze.pipeline import (
    analyze,
    analyze_all_cameras,
    annotations_path,
    cameras_with_frames,
    prompts_path,
    results_dir_for,
)
from fungus_cv.analyze.report import load_measurements
from fungus_cv.cli import app
from fungus_cv.measure.geometry import Annotations
from fungus_cv.storage import Experiment, FrameRecord, encode_image, iso_utc

from . import synthetic as syn
from .test_pipeline import T0

# The second camera sees the towel at an angle: everything looks shorter (foreshortened).
SQUASH = 0.8


def build_two_cameras(exp: Experiment, heights=(100, 200, 300, 400), offset_s=5):
    text = exp.config_path.read_text().replace("size_mm: null", f"size_mm: {syn.MARKER_MM}")
    marker = text.index("capture:")
    cameras = text[:marker].replace("  - name: cam0", "  - name: cam0", 1)
    cameras += "  - name: cam1\n    index: 1\n"
    exp.config_path.write_text(cameras + text[marker:])
    for i, h in enumerate(heights):
        ts = T0 + timedelta(minutes=i)
        for camera, scale in (("cam0", 1.0), ("cam1", SQUASH)):
            when = ts + timedelta(seconds=offset_s if camera == "cam1" else 0)
            image = syn.scene(int(round(h * scale)))
            path = exp.save_bytes(encode_image(image, "png"), when, camera, "png")
            exp.append_frame(FrameRecord(timestamp_utc=iso_utc(when), camera=camera,
                                         status="ok", file=exp.relative(path),
                                         width=syn.W, height=syn.H))
    reopened = Experiment(exp.root)
    for camera in ("cam0", "cam1"):
        Annotations(base=syn.BASE_POINT, tip=syn.TIP_POINT, roi=syn.ROI,
                    image_size=(syn.W, syn.H)).save(annotations_path(reopened, camera))
    return reopened


@pytest.fixture
def two_cameras(experiment):
    return build_two_cameras(experiment)


def test_each_camera_has_its_own_files_and_results(two_cameras):
    exp = two_cameras
    assert cameras_with_frames(exp) == ["cam0", "cam1"]
    assert annotations_path(exp, "cam1").name == "annotations_cam1.json"
    assert prompts_path(exp, "prompts.json", "cam1").name == "prompts_cam1.json"
    assert results_dir_for(exp, "cam1") == exp.root / "results" / "cameras" / "cam1"

    summaries = analyze_all_cameras(exp)
    assert set(summaries) == {"cam0", "cam1"}
    assert all(s.processed == 4 for s in summaries.values())
    for camera, scale in (("cam0", 1.0), ("cam1", SQUASH)):
        rows = load_measurements(exp, results_dir=results_dir_for(exp, camera))
        assert len(rows) == 4 and {r["camera"] for r in rows} == {camera}
        assert float(rows[-1]["extent_mm"]) == pytest.approx(400 * scale * syn.MM_PER_PX,
                                                             abs=0.5)
    assert not (exp.root / "results" / "measurements.csv").exists()  # nothing mixed together


def test_one_camera_at_a_time(two_cameras):
    exp = two_cameras
    summary = analyze(exp, camera="cam1")
    assert summary.processed == 4
    rows = load_measurements(exp, results_dir=results_dir_for(exp, "cam1"))
    assert {r["camera"] for r in rows} == {"cam1"}
    assert not (exp.root / "results" / "cameras" / "cam0").exists()


def test_combine_takes_the_longest_view(two_cameras):
    exp = two_cameras
    analyze_all_cameras(exp)
    result = combine(exp, metric="extent_mm")
    assert result.cameras == ["cam0", "cam1"] and len(result.rows) == 4
    assert result.unmatched == 0  # frames 5 s apart count as the same moment
    for row, height in zip(result.rows, (100, 200, 300, 400)):
        assert row.value == pytest.approx(height * syn.MM_PER_PX, abs=0.5)
        assert row.best_camera == "cam0" and row.n_views == 2
        assert row.spread == pytest.approx(height * (1 - SQUASH) * syn.MM_PER_PX, abs=0.6)
    assert result.mean_spread > 4
    assert result.out_path.exists()
    note = result.out_path.with_name(result.out_path.stem + "_README.txt").read_text()
    assert "underestimates" in note and "spread" in note

    mean = combine(exp, metric="extent_mm", method="mean")
    assert mean.rows[-1].value < result.rows[-1].value  # averaging keeps the short view
    tight = combine(exp, metric="extent_mm", tolerance_s=1)
    assert tight.unmatched == 4 and all(r.n_views == 1 for r in tight.rows)

    area = combine(exp, metric="target_area_mm2")
    assert "not a length" in area.note


def test_combine_needs_two_cameras(experiment):
    from .test_pipeline import build_experiment

    build_experiment(experiment, minutes=range(0, 3), bump_at=-1)
    with pytest.raises(ValueError, match="at least two cameras"):
        combine(Experiment(experiment.root))


def test_multi_camera_cli(two_cameras):
    exp = two_cameras
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", str(exp.root), "--camera", "all"])
    assert result.exit_code == 0, result.output
    assert "camera cam0:" in result.output and "camera cam1:" in result.output

    report = runner.invoke(app, ["report", str(exp.root), "--camera", "cam1", "--bootstrap", "0"])
    assert report.exit_code == 0, report.output
    assert (exp.root / "results" / "cameras" / "cam1" / "report").is_dir()

    combined = runner.invoke(app, ["combine", str(exp.root), "--metric", "extent_mm"])
    assert combined.exit_code == 0, combined.output
    assert "largest view came from: cam0 4x" in combined.output
    assert "views disagree by" in combined.output

    one = runner.invoke(app, ["analyze", str(exp.root), "--camera", "nope"])
    assert one.exit_code == 1 and "no frames for camera" in one.output
    watching = runner.invoke(app, ["analyze", str(exp.root), "--camera", "all", "--watch"])
    assert watching.exit_code == 1 and "one camera" in watching.output
