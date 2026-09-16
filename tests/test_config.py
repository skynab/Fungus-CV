from datetime import datetime, timezone

import pytest
import yaml
from pydantic import ValidationError

from fungus_cv.config import (
    CaptureConfig,
    ExperimentConfig,
    default_config_yaml,
    parse_duration,
)


@pytest.mark.parametrize(
    "text, seconds",
    [
        (5, 5.0),
        (2.5, 2.5),
        ("30", 30.0),
        ("30s", 30.0),
        ("5m", 300.0),
        ("5 min", 300.0),
        ("2h", 7200.0),
        ("1h30m", 5400.0),
        ("14d", 14 * 86400.0),
        ("2 days", 2 * 86400.0),
    ],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("bad", ["", "abc", "5 parsecs", "-5s", "0", "5m garbage"])
def test_parse_duration_rejects(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)


def test_default_template_is_valid():
    cfg = ExperimentConfig.model_validate(yaml.safe_load(default_config_yaml("dye")))
    assert cfg.name == "dye"
    assert cfg.capture.interval == 30
    assert cfg.cameras[0].exposure == "lock"


def test_numeric_controls_and_unknown_keys():
    cfg = ExperimentConfig.model_validate(
        {"name": "x", "cameras": [{"name": "top", "exposure": -5, "focus": "auto"}]}
    )
    assert cfg.cameras[0].exposure == -5.0
    assert cfg.cameras[0].focus == "auto"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "x", "cameras": [{"exposur": 1}]})


def test_duplicate_camera_names_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "x", "cameras": [{"name": "a"}, {"name": "a"}]})


def test_camera_name_must_be_filename_safe():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "x", "cameras": [{"name": "top cam/1"}]})


def test_keep_camera_open_auto():
    assert CaptureConfig(interval="10s").camera_stays_open
    assert not CaptureConfig(interval="1h").camera_stays_open
    assert not CaptureConfig(interval="10s", keep_camera_open=False).camera_stays_open


def test_start_at_converted_to_utc():
    cfg = CaptureConfig(start_at="2026-09-20T08:00:00-05:00")
    assert cfg.start_at == datetime(2026, 9, 20, 13, 0, tzinfo=timezone.utc)
