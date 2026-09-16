"""Camera permission handling, camera names, display checks and the related CLI behaviour."""

import pytest
from typer.testing import CliRunner

from fungus_cv.capture import diagnostics, permissions
from fungus_cv.capture.permissions import (
    AUTHORIZED,
    DENIED,
    NOT_APPLICABLE,
    NOT_DETERMINED,
    RESTRICTED,
    CameraAccess,
    camera_access,
    camera_names,
    responsible_app,
)
from fungus_cv.cli import app

runner = CliRunner()


def chain(processes):
    """Fake `ps`: pid -> (parent pid, executable path)."""
    return lambda pid: processes.get(pid)


def test_responsible_app_is_outermost_app_bundle():
    processes = {
        300: (200, "/bin/zsh"),
        200: (100, "/Applications/Visual Studio Code.app/Contents/Frameworks/"
                   "Code Helper.app/Contents/MacOS/Code Helper"),
        100: (1, "/Applications/Visual Studio Code.app/Contents/MacOS/Electron"),
    }
    assert responsible_app(300, chain(processes)) == "Visual Studio Code"


def test_responsible_app_nested_tools_report_top_app():
    processes = {
        40: (30, "/bin/bash"),
        30: (20, "/Users/me/Library/Application Support/Claude/claude-code/claude.app/"
                 "Contents/MacOS/claude"),
        20: (1, "/Applications/Claude.app/Contents/Helpers/disclaimer"),
    }
    assert responsible_app(40, chain(processes)) == "Claude"


def test_responsible_app_falls_back_to_term_program(monkeypatch):
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    assert responsible_app(5, chain({5: (1, "/bin/zsh")})) == "Terminal"


@pytest.mark.parametrize("access, expected", [
    (CameraAccess(DENIED, "Terminal"), "Turn it on"),
    (CameraAccess(RESTRICTED, "Terminal"), "restricted"),
    (CameraAccess(NOT_DETERMINED, "Claude", prompt_refused=True, requested=True),
     "Terminal app"),
    (CameraAccess(NOT_DETERMINED, "iTerm", requested=True), "Click Allow"),
    (CameraAccess(NOT_DETERMINED, "iTerm"), "hasn't been decided"),
])
def test_advice_names_the_app_and_the_fix(access, expected):
    assert not access.ok
    assert expected in access.advice()


def test_ok_statuses():
    assert CameraAccess(AUTHORIZED).ok and CameraAccess(NOT_APPLICABLE).ok


def test_non_macos_needs_no_permission():
    assert camera_access(platform="linux").status == NOT_APPLICABLE
    assert camera_access(platform="win32").ok


def test_linux_camera_names_from_sysfs(tmp_path):
    for index, name in ((0, "Integrated Camera"), (2, "Logitech BRIO")):
        node = tmp_path / f"video{index}"
        node.mkdir()
        (node / "name").write_text(name + "\n")
    assert camera_names("linux", tmp_path) == {0: "Integrated Camera", 2: "Logitech BRIO"}


def test_gui_problem(monkeypatch):
    monkeypatch.setattr(diagnostics, "opencv_gui_backend", lambda: "NONE")
    assert "headless" in diagnostics.gui_problem("darwin", {})
    monkeypatch.setattr(diagnostics, "opencv_gui_backend", lambda: "GTK3")
    assert "No display" in diagnostics.gui_problem("linux", {})
    assert diagnostics.gui_problem("linux", {"DISPLAY": ":0"}) == ""


@pytest.mark.parametrize("platform_name, text", [
    ("win32", "Let desktop apps access your camera"),
    ("linux", "video' group"),
    ("darwin", "fungus doctor"),
])
def test_no_camera_hint(platform_name, text):
    assert text in diagnostics.no_camera_hint(platform_name)


def deny(**_):
    return CameraAccess(NOT_DETERMINED, "Claude", prompt_refused=True, requested=True)


def test_cameras_stops_with_advice_when_macos_refuses(monkeypatch):
    monkeypatch.setattr(permissions, "camera_access", deny)
    result = runner.invoke(app, ["cameras"])
    assert result.exit_code == 2
    assert "Terminal app" in result.output


def test_preview_checks_display_before_camera(monkeypatch):
    monkeypatch.setattr(diagnostics, "gui_problem", lambda: "No display available")
    result = runner.invoke(app, ["preview"])
    assert result.exit_code == 2 and "No display" in result.output


def test_capture_preflight_reports_bad_camera(experiment, monkeypatch):
    from fungus_cv.capture import camera as camera_module

    monkeypatch.setattr(permissions, "camera_access",
                        lambda **_: CameraAccess(NOT_APPLICABLE))

    class Broken(camera_module.Camera):
        def open(self):
            raise camera_module.CameraError("could not open camera 'cam0' (index 0)")

    monkeypatch.setattr(camera_module, "Camera", Broken)
    result = runner.invoke(app, ["capture", str(experiment.root), "--max-frames", "1"])
    assert result.exit_code == 1
    assert "could not open camera" in result.output and "--no-check-cameras" in result.output
    assert experiment.read_frames() == []  # nothing started


def test_doctor_reports_checks(monkeypatch):
    monkeypatch.setattr(permissions, "camera_access",
                        lambda **_: CameraAccess(DENIED, "Terminal"))
    monkeypatch.setattr(permissions, "camera_names", lambda: [])
    result = runner.invoke(app, ["doctor", "--no-probe"])
    assert "Camera permission: denied for Terminal" in result.output
    assert "Turn it on" in result.output
    assert result.exit_code == 1
