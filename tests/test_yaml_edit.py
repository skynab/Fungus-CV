import pytest
import yaml

from fungus_cv.config import ExperimentConfig, default_config_yaml, set_yaml_value


@pytest.mark.parametrize("key, value", [
    ("capture.interval", "5m"),
    ("capture.duration", None),
    ("capture.max_frames", 120),
    ("capture.keep_awake", False),
    ("analysis.markers.size_mm", 30.5),
    ("analysis.rectify.enabled", True),
    ("analysis.lighting.method", "patch"),
    ("analysis.target.method", "sam2"),
    ("analysis.target.sam2.model", "facebook/sam2.1-hiera-tiny"),
    ("analysis.reference.method", "color"),
    ("analysis.measure.mode", "path"),
    ("description", "moss on bark, bench 3: north"),
])
def test_set_value_keeps_comments_and_validates(key, value):
    text = default_config_yaml("x")
    new = set_yaml_value(text, key, value)
    data = yaml.safe_load(new)
    node = data
    for part in key.split("."):
        node = node[part]
    assert node == value
    ExperimentConfig.model_validate(data)
    assert new.count("#") == text.count("#")  # every comment survives
    assert len(new.splitlines()) == len(text.splitlines())


def test_nested_key_is_not_confused_with_same_name_elsewhere():
    text = default_config_yaml("x")
    new = set_yaml_value(text, "analysis.reference.method", "model")
    data = yaml.safe_load(new)
    assert data["analysis"]["reference"]["method"] == "model"
    assert data["analysis"]["target"]["method"] == "color"
    assert data["analysis"]["lighting"]["method"] == "none"


def test_missing_key():
    with pytest.raises(KeyError):
        set_yaml_value(default_config_yaml("x"), "capture.nope", 1)


def test_replace_reference_colour_ranges_only():
    from fungus_cv.config import replace_hsv_ranges_in_yaml

    text = default_config_yaml("x")
    new = replace_hsv_ranges_in_yaml(text, [((10, 20, 30), (40, 50, 60))], section="reference")
    data = yaml.safe_load(new)
    assert data["analysis"]["reference"]["color"]["hsv_ranges"] == [
        {"lower": [10, 20, 30], "upper": [40, 50, 60]}]
    assert data["analysis"]["target"]["color"]["hsv_ranges"][0]["lower"] == [95, 60, 30]
    assert data["analysis"]["reference"]["color"]["open_px"] == 3
    ExperimentConfig.model_validate(data)
