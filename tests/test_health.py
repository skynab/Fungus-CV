"""Unattended runs: heartbeat, health checks, alerts and login services."""

import json
import plistlib
from datetime import timedelta

import pytest
from typer.testing import CliRunner

from fungus_cv.capture import health, service
from fungus_cv.cli import app
from fungus_cv.storage import Experiment, FrameRecord, encode_image, iso_utc, utc_now

from . import synthetic as syn
from .test_pipeline import T0


def add_frames(exp: Experiment, n=4, end=None, brightness=None, sharpness=None, every_s=30):
    end = end or utc_now()
    for i in range(n):
        ts = end - timedelta(seconds=every_s * (n - 1 - i))
        image = syn.scene(100 + 20 * i)
        path = exp.save_bytes(encode_image(image, "png"), ts, "cam0", "png")
        exp.append_frame(FrameRecord(
            timestamp_utc=iso_utc(ts), camera="cam0", status="ok", file=exp.relative(path),
            width=syn.W, height=syn.H,
            mean_brightness=200.0 if brightness is None or i < n - 1 else brightness,
            sharpness=500.0 if sharpness is None or i < n - 1 else sharpness))
    return Experiment(exp.root)


def levels(report):
    return {c.name: c.level for c in report.checks}


@pytest.fixture
def running(experiment):
    exp = add_frames(experiment)
    health.write_heartbeat(exp, state="running", interval_s=30, rounds=4, saved=4,
                           last_frame_utc=iso_utc(utc_now()))
    return Experiment(exp.root)


def test_healthy_run(running):
    report = health.check(running)
    assert report.ok and report.level == "ok"
    assert levels(report)["capture"] == "ok" and levels(report)["frames"] == "ok"
    assert "healthy" in report.summary()
    assert report.heartbeat.pid > 0 and report.heartbeat.host


def test_dead_capture_and_stale_frames(experiment):
    old = utc_now() - timedelta(hours=3)
    exp = add_frames(experiment, end=old)  # capture stopped three hours ago
    health.write_heartbeat(exp, state="running", interval_s=30, saved=4)
    path = health.status_path(exp)
    beat = json.loads(path.read_text())
    beat["updated_utc"] = iso_utc(old)
    path.write_text(json.dumps(beat))

    report = health.check(Experiment(exp.root))
    assert not report.ok and report.level == "problem"
    assert levels(report)["capture"] == "problem"
    assert levels(report)["frames"] == "problem"  # photos stopped too
    assert "looks dead" in report.summary()
    assert any("restart" in c.fix for c in report.problems)


def test_finished_capture_is_not_a_problem(running):
    health.write_heartbeat(running, state="finished", stopped_reason="duration reached",
                           interval_s=30, saved=4)
    report = health.check(running)
    assert levels(report)["capture"] == "ok" and "finished" in report.checks[0].detail


def test_quality_and_schedule_warnings(experiment):
    dark = add_frames(experiment, brightness=5.0)
    health.write_heartbeat(dark, state="running", interval_s=30, saved=4, skipped_slots=3)
    report = health.check(Experiment(dark.root))
    assert levels(report)["image quality"] == "warning"
    assert "almost black" in next(c.detail for c in report.checks if c.name == "image quality")
    assert levels(report)["schedule"] == "warning"
    assert report.level == "warning" and not report.ok

    blurry = add_frames(Experiment.create(experiment.root.parent / "b", name="b"),
                        sharpness=50.0)
    blurred = health.check(blurry)
    assert "blurred" in next(c.detail for c in blurred.checks if c.name == "image quality")


def test_camera_failures_and_no_frames(experiment):
    report = health.check(Experiment(experiment.root))
    assert levels(report)["frames"] == "problem" and levels(report)["capture"] == "warning"

    exp = add_frames(experiment)
    for i in range(3):
        exp.append_frame(FrameRecord(timestamp_utc=iso_utc(utc_now()), camera="cam0",
                                     status="failed", notes=f"camera not found ({i})"))
    report = health.check(Experiment(exp.root))
    assert levels(report)["camera errors"] == "problem"
    assert "camera not found" in report.summary()


def test_disk_check(running, monkeypatch):
    monkeypatch.setattr(type(running), "free_disk_mb", lambda self: 50.0)
    report = health.check(running)
    assert levels(report)["disk"] == "problem"
    assert "below the" in next(c.detail for c in report.checks if c.name == "disk")


def test_alerts_go_to_a_webhook_and_a_command(running, tmp_path, monkeypatch):
    report = health.check(running)
    sent = {}

    def fake_urlopen(request, timeout=10):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return Response()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    marker = tmp_path / "alert.txt"
    errors = health.notify(report, "problem", webhook="https://example.invalid/hook",
                           command=f'printenv FUNGUS_SUMMARY > "{marker}"')
    assert errors == []
    assert sent["url"].endswith("/hook") and sent["body"]["event"] == "problem"
    assert sent["body"]["text"] == report.summary() and sent["body"]["checks"]
    assert marker.read_text().strip() == report.summary()


def test_a_failing_alert_is_reported_not_raised(running, monkeypatch):
    def boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(health, "send_webhook", boom)
    errors = health.notify(health.check(running), "problem", webhook="https://example.invalid")
    assert errors and "no network" in errors[0]


def test_capture_writes_a_heartbeat(experiment, monkeypatch):
    from fungus_cv.capture.scheduler import CaptureSession
    from tests.conftest import FakeClock

    exp = Experiment(experiment.root)
    text = exp.config_path.read_text().replace("interval: 30s", "interval: 1s")
    exp.config_path.write_text(text.replace("max_frames: null", "max_frames: 2"))
    exp = Experiment(exp.root)
    clock = FakeClock(start=T0)
    session = CaptureSession(exp, clock=clock, camera_factory=lambda cfg: _FakeCamera(cfg))
    session.run()

    beat = health.read_heartbeat(exp)
    assert beat.state == "finished" and beat.saved == 2 and beat.rounds == 2
    assert beat.experiment == exp.config.name and beat.interval_s == 1
    assert beat.free_disk_mb > 0


class _FakeCamera:
    def __init__(self, config):
        self.config = config
        self.warnings = []

    def open(self):
        return None

    def read(self):
        return syn.scene(120)

    def settings(self):
        return {}

    def close(self):
        return None


def test_health_cli(running):
    runner = CliRunner()
    result = runner.invoke(app, ["health", str(running.root)])
    assert result.exit_code == 0 and "[OK  ] capture" in result.output
    assert "healthy" in result.output

    path = health.status_path(running)
    beat = json.loads(path.read_text())
    beat["updated_utc"] = iso_utc(utc_now() - timedelta(days=1))
    path.write_text(json.dumps(beat))
    bad = runner.invoke(app, ["health", str(running.root)])
    assert bad.exit_code == 1 and "[FAIL] capture" in bad.output
    assert "-> check the computer is awake" in bad.output


# --- login services ----------------------------------------------------------------------


@pytest.mark.parametrize("platform_name, marker", [
    ("darwin", "LaunchAgents"), ("linux", "systemd"), ("win32", "task.xml")])
def test_service_files_per_platform(running, platform_name, marker):
    plan = service.plan(running.root, "capture", platform_name=platform_name)
    assert marker in str(plan.path)
    assert str(running.root.resolve()) in plan.contents
    assert plan.commands and plan.remove_commands
    if platform_name == "darwin":
        parsed = plistlib.loads(plan.contents.encode())
        assert parsed["RunAtLoad"] and parsed["KeepAlive"]
        assert parsed["ProgramArguments"][-2:] == ["capture", str(running.root.resolve())]
    if platform_name == "linux":
        assert "Restart=on-failure" in plan.contents and "capture" in plan.contents
    with pytest.raises(ValueError, match="no service support"):
        service.plan(running.root, platform_name="plan9")


def test_install_writes_the_file_without_registering(running):
    result = service.install(running.root, "health", ["--webhook", "https://example.invalid"],
                             register=False)
    assert result.wrote and not result.registered
    assert result.plan.path.exists() and service.installed(running.root, "health")
    assert "--webhook" in result.plan.contents
    removed = service.uninstall(running.root, "health")
    assert removed.wrote and not result.plan.path.exists()
    assert not service.installed(running.root, "health")


def test_service_cli_dry_run_and_confirmation(running, monkeypatch, tmp_path):
    runner = CliRunner()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: home)

    dry = runner.invoke(app, ["service", "install", str(running.root), "--dry-run"])
    assert dry.exit_code == 0 and "file contents" in dry.output
    assert "Runs:" in dry.output and not any(home.rglob("*fungus*"))

    declined = runner.invoke(app, ["service", "install", str(running.root)], input="n\n")
    assert declined.exit_code == 1 and "Nothing installed" in declined.output

    installed = runner.invoke(app, ["service", "install", str(running.root), "--yes",
                                    "--no-register"])
    assert installed.exit_code == 0 and "Wrote" in installed.output
    status = runner.invoke(app, ["service", "status", str(running.root)])
    assert "capture  installed" in status.output and "health   not installed" in status.output
    gone = runner.invoke(app, ["service", "uninstall", str(running.root)])
    assert gone.exit_code == 0 and "Removed" in gone.output
