"""Reproducibility bundles: what goes in, what stays out, and checking one later."""

import json
import zipfile

import pytest
from typer.testing import CliRunner

from fungus_cv.analyze import archive
from fungus_cv.analyze.pipeline import analyze
from fungus_cv.analyze.report import make_report
from fungus_cv.cli import app
from fungus_cv.storage import Experiment

from .test_pipeline import T0, build_experiment


@pytest.fixture
def analyzed(experiment):
    build_experiment(experiment, minutes=range(0, 5), bump_at=-1)
    exp = Experiment(experiment.root)
    analyze(exp)
    make_report(exp, t0=T0, bootstrap=0)
    from fungus_cv.analyze import exclusions

    exclusions.exclude(exp, exp.read_frames()[2]["file"], "hand in the picture")
    return exp


def names(path):
    with zipfile.ZipFile(path) as zf:
        return set(zf.namelist())


def manifest(path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("manifest.json"))


def test_bundle_holds_settings_results_and_hashes(analyzed, tmp_path):
    result = archive.build(analyzed.root, tmp_path / "dye.zip")
    inside = names(result.path)
    assert {"config.yaml", "frames.csv", "annotations.json", "exclusions.json",
            "results/measurements.csv", "results/run_info.json", "README.md",
            "manifest.json"} <= inside
    assert any(n.startswith("results/report/") for n in inside)
    assert not any(n.startswith("results/masks/") or n.startswith("results/overlays/")
                   for n in inside)
    assert not any(n.startswith("frames/") for n in inside)  # photos are opt-in

    data = manifest(result.path)
    assert data["kind"] == "experiment" and data["format"] == archive.BUNDLE_FORMAT
    frames = data["experiment"]["frames"]
    assert len(frames) == 5 and all(f["sha256"] for f in frames)  # hashes even without photos
    assert result.frames_recorded == 5 and result.frames_included == 0
    env = data["environment"]
    assert env["packages"]["numpy"] and env["fungus_cv_version"]
    assert {e["path"] for e in data["files"]} == inside - {"manifest.json"}


def test_bundle_with_photos_and_masks(analyzed, tmp_path):
    result = archive.build(analyzed.root, tmp_path / "full.zip", frames=True, masks=True)
    inside = names(result.path)
    assert sum(n.startswith("frames/") for n in inside) == 5
    assert any(n.startswith("results/masks/") for n in inside)
    assert result.frames_included == 5


def test_verify_detects_a_changed_file(analyzed, tmp_path):
    path = tmp_path / "dye.zip"
    archive.build(analyzed.root, path)
    assert archive.verify(path).ok

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tampered, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "results/measurements.csv":
                data = data.replace(b"extent_mm", b"extent_XX")
            dst.writestr(name, data)
    check = archive.verify(tampered)
    assert not check.ok and check.changed == ["results/measurements.csv"]

    missing = tmp_path / "missing.zip"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(missing, "w") as dst:
        for name in src.namelist():
            if name != "config.yaml":
                dst.writestr(name, src.read(name))
    assert archive.verify(missing).missing == ["config.yaml"]

    with pytest.raises(ValueError, match="not a Fungus-CV bundle"):
        plain = tmp_path / "plain.zip"
        with zipfile.ZipFile(plain, "w") as zf:
            zf.writestr("a.txt", "hello")
        archive.verify(plain)


def test_study_bundle_includes_every_replicate(tmp_path):
    import yaml

    from fungus_cv.analyze.study import run_study

    from .test_study import fake_experiment, write_study

    for i in (1, 2):
        fake_experiment(tmp_path, f"control{i}", 70, 0.2, 24, seed=i)
        fake_experiment(tmp_path, f"treated{i}", 70, 0.4, 24, seed=10 + i)
    path = write_study(tmp_path, bootstrap=0, experiments=[
        {"path": f"{cond}{i}", "condition": cond} for cond in ("control", "treated")
        for i in (1, 2)])
    run_study(path)
    result = archive.build(path, tmp_path / "study.zip")
    inside = names(result.path)
    assert "study.yaml" in inside and "study_results/conditions.csv" in inside
    assert "study_results/methods.md" in inside
    assert sum(n.endswith("results/measurements.csv") for n in inside) == 4
    data = manifest(result.path)
    assert data["kind"] == "study"
    assert [e["condition"] for e in data["study"]["experiments"]] == \
        ["control", "control", "treated", "treated"]
    assert yaml.safe_load(zipfile.ZipFile(result.path).read("study.yaml"))["name"] == "moss-test"
    assert archive.verify(result.path).ok


def test_model_card_is_bundled_without_its_weights(analyzed, tmp_path):
    model = tmp_path / "models" / "m"
    model.mkdir(parents=True)
    (model / "model.json").write_text('{"threshold": 0.5}')
    (model / "model.pt").write_bytes(b"weights")
    text = analyzed.config_path.read_text().replace("method: color", "method: model", 1)
    analyzed.config_path.write_text(text.replace('path: ""', f"path: '{model}'", 1))

    result = archive.build(analyzed.root, tmp_path / "with_model.zip")
    assert "models/m/model.json" in names(result.path)
    assert "models/m/model.pt" not in names(result.path)
    entry = manifest(result.path)["experiment"]["models"][0]
    assert entry["role"] == "target" and len(entry["weights_sha256"]) == 64
    assert any("weights not included" in n for n in result.notes)

    with_weights = archive.build(analyzed.root, tmp_path / "w.zip", weights=True)
    assert "models/m/model.pt" in names(with_weights.path)


def test_archive_cli(analyzed, tmp_path):
    runner = CliRunner()
    out = tmp_path / "bundle.zip"
    result = runner.invoke(app, ["archive", str(analyzed.root), str(out)])
    assert result.exit_code == 0 and "photo(s) included" in result.output
    check = runner.invoke(app, ["archive", "--verify", str(out)])
    assert check.exit_code == 0 and "Bundle is intact" in check.output
    assert "experiment bundle" in check.output

    out.write_bytes(out.read_bytes()[:-20])  # truncated download
    broken = runner.invoke(app, ["archive", "--verify", str(out)])
    assert broken.exit_code == 1
    assert runner.invoke(app, ["archive"]).exit_code == 1
