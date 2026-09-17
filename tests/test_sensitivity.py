"""Sensitivity: how much do the results move when a debatable setting is changed?"""

import csv
import json

from typer.testing import CliRunner

from fungus_cv.analyze import sensitivity as sens
from fungus_cv.analyze.pipeline import analyze
from fungus_cv.analyze.report import load_measurements
from fungus_cv.cli import app
from fungus_cv.storage import Experiment

from .test_active import build as build_soft
from .test_seg_uncertainty import build as build_sharp


def prepared(experiment, soft: bool):
    (build_soft if soft else build_sharp)(experiment, **({} if soft else {"ramp_px": 0}))
    exp = Experiment(experiment.root)
    analyze(exp)
    return Experiment(experiment.root)


def by_name(result):
    return {v.variant.name: v for v in result.variants}


def test_default_variants_follow_the_settings(experiment):
    exp = prepared(experiment, soft=False)
    names = [v.name for v in sens.default_variants(exp)]
    assert {"colour_wider", "colour_narrower", "no_cleanup", "align_ecc", "rectify_on",
            "front_percentile_25", "front_percentile_75"} <= set(names)
    assert "lighting_background" in names  # not corrected yet: try correcting

    text = exp.config_path.read_text().replace("method: none", "method: background", 1)
    exp.config_path.write_text(text)
    names = [v.name for v in sens.default_variants(Experiment(exp.root))]
    assert "lighting_off" in names and "lighting_background" not in names

    text = exp.config_path.read_text().replace("method: color", "method: sam2", 1)
    exp.config_path.write_text(text)
    names = [v.name for v in sens.default_variants(Experiment(exp.root))]
    assert {"mask_threshold_looser", "mask_threshold_tighter"} <= set(names)
    assert not any(n.startswith("colour_") for n in names)


def test_soft_edges_are_sensitive_to_thresholds_sharp_ones_are_not(experiment, tmp_path):
    soft = prepared(experiment, soft=True)
    chosen = [v for v in sens.default_variants(soft)
              if v.name in ("colour_wider", "colour_narrower")]
    before = len(load_measurements(soft))
    result = sens.run(soft, variants=chosen)
    assert result.metric == "extent_mm" and result.plot == "main"
    changes = by_name(result)
    assert all(v.ok and v.n_compared > 5 for v in changes.values())
    # Averaged over all frames (only some have soft edges): ~0.25 mm, about 1.2u.
    assert changes["colour_wider"].mean_abs_change > 0.15
    assert changes["colour_wider"].change_in_uncertainties > 0.8
    assert result.largest.variant.name.startswith("colour_")

    # The experiment's own results are untouched, and variant masks are cleaned up.
    assert len(load_measurements(Experiment(soft.root))) == before
    variant_dir = soft.root / "results" / "sensitivity" / "colour_wider"
    assert (variant_dir / "measurements.csv").exists()
    assert not (variant_dir / "masks").exists()
    for name in ("sensitivity.csv", "sensitivity.json", "sensitivity.png"):
        assert (result.out_dir / name).exists(), name

    sharp = prepared(Experiment.create(tmp_path / "sharp", name="sharp"), soft=False)
    crisp = by_name(sens.run(sharp, variants=[v for v in sens.default_variants(sharp)
                                              if v.name == "colour_wider"]))
    assert crisp["colour_wider"].mean_abs_change < 0.1
    assert crisp["colour_wider"].mean_abs_change < changes["colour_wider"].mean_abs_change / 3


def test_a_broken_variant_is_reported_not_raised(experiment):
    exp = prepared(experiment, soft=False)
    broken = sens.Variant("stem_mode", "measure along a stem that is not segmented",
                          {"measure.mode": "path", "measure.path_source": "reference"})
    result = sens.run(exp, variants=[broken])
    entry = result.variants[0]
    assert not entry.ok and "reference" in entry.error
    with open(result.out_dir / "sensitivity.csv", newline="") as f:
        assert list(csv.DictReader(f))[0]["error"].startswith("AnalysisError")
    assert result.largest is None


def test_sensitivity_cli(experiment):
    exp = prepared(experiment, soft=True)
    runner = CliRunner()
    listed = runner.invoke(app, ["sensitivity", str(exp.root), "--list"])
    assert listed.exit_code == 0 and "colour_wider" in listed.output

    result = runner.invoke(app, ["sensitivity", str(exp.root), "--variant", "colour_wider",
                                 "--variant", "front_percentile_75"])
    assert result.exit_code == 0, result.output
    assert "mean |change|" in result.output and "Largest effect" in result.output
    assert "u" in result.output and "wrote" in result.output
    data = json.loads((exp.root / "results" / "sensitivity" / "sensitivity.json").read_text())
    assert [v["variant"] for v in data["variants"]] == ["colour_wider", "front_percentile_75"]
    assert data["baseline"]["settings"]["target"]["method"] == "color"

    bad = runner.invoke(app, ["sensitivity", str(exp.root), "--variant", "nope"])
    assert bad.exit_code == 1 and "unknown variant" in bad.output
