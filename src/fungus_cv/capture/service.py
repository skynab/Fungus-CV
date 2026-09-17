"""Start capture (or health checks) automatically, and keep it running.

Writes the service file your operating system expects, registers it, and can show or remove
it again. Nothing is installed unless you ask for it: `fungus service install <folder>`.

- macOS: a launchd agent in ``~/Library/LaunchAgents`` (runs at login, restarted if it exits)
- Linux: a systemd **user** unit in ``~/.config/systemd/user`` (``Restart=on-failure``)
- Windows: a Task Scheduler task that runs at logon

A user agent runs when you are logged in. For a machine that must capture while logged out,
a system-wide service is needed; the file written here is a good starting point for that.
"""

from __future__ import annotations

import logging
import plistlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

MACOS, LINUX, WINDOWS = "darwin", "linux", "win32"


@dataclass
class ServicePlan:
    """What would be (or was) installed."""

    name: str
    path: Path
    contents: str
    commands: list[list[str]] = field(default_factory=list)  # to register it
    remove_commands: list[list[str]] = field(default_factory=list)
    platform: str = sys.platform
    note: str = ""


def safe_name(experiment: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(experiment).resolve().name).strip("-") or "run"


def fungus_command(experiment: Path, action: str = "capture",
                   extra: list[str] | None = None) -> list[str]:
    """The command the service runs: this interpreter's `fungus`, with absolute paths."""
    executable = shutil.which("fungus")
    base = [executable] if executable else [sys.executable, "-m", "fungus_cv.cli"]
    return base + [action, str(Path(experiment).resolve())] + list(extra or [])


def plan(experiment: Path, action: str = "capture", extra: list[str] | None = None,
         platform_name: str | None = None) -> ServicePlan:
    """Work out the file and commands for this computer, without touching anything."""
    platform_name = platform_name or sys.platform
    name = safe_name(experiment)
    command = fungus_command(experiment, action, extra)
    root = Path(experiment).resolve()
    log_file = root / f"{action}-service.log"

    if platform_name == MACOS:
        label = f"com.fungus-cv.{name}.{action}"
        path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        contents = plistlib.dumps({
            "Label": label, "ProgramArguments": command, "RunAtLoad": True, "KeepAlive": True,
            "WorkingDirectory": str(root), "StandardOutPath": str(log_file),
            "StandardErrorPath": str(log_file),
        }).decode()
        return ServicePlan(
            label, path, contents,
            commands=[["launchctl", "unload", str(path)], ["launchctl", "load", "-w", str(path)]],
            remove_commands=[["launchctl", "unload", "-w", str(path)]], platform=platform_name,
            note="macOS also needs camera permission for the program that runs it; the packaged "
                 "app has its own entry under Privacy & Security -> Camera.")
    if platform_name.startswith(LINUX):
        label = f"fungus-{name}-{action}"
        path = Path.home() / ".config" / "systemd" / "user" / f"{label}.service"
        contents = (
            "[Unit]\n"
            f"Description=Fungus-CV {action} for {root.name}\n\n"
            "[Service]\n"
            f"ExecStart={' '.join(command)}\n"
            f"WorkingDirectory={root}\n"
            "Restart=on-failure\nRestartSec=30\n\n"
            "[Install]\nWantedBy=default.target\n")
        return ServicePlan(
            label, path, contents,
            commands=[["systemctl", "--user", "daemon-reload"],
                      ["systemctl", "--user", "enable", "--now", label]],
            remove_commands=[["systemctl", "--user", "disable", "--now", label],
                             ["systemctl", "--user", "daemon-reload"]], platform=platform_name,
            note="`loginctl enable-linger $USER` keeps a user service running after logout.")
    if platform_name == WINDOWS:
        label = f"Fungus-CV {name} {action}"
        path = Path(experiment).resolve() / f"{action}-task.xml"
        quoted = subprocess.list2cmdline(command)
        contents = quoted + "\n"
        return ServicePlan(
            label, path, contents,
            commands=[["schtasks", "/Create", "/F", "/SC", "ONLOGON", "/TN", label, "/TR",
                       quoted]],
            remove_commands=[["schtasks", "/Delete", "/F", "/TN", label]], platform=platform_name,
            note="Task Scheduler runs this at logon; tick 'Run whether user is logged on or "
                 "not' in Task Scheduler for an unattended machine.")
    raise ValueError(f"no service support for {platform_name}")


@dataclass
class ServiceResult:
    plan: ServicePlan
    wrote: bool = False
    registered: bool = False
    output: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _run(commands: list[list[str]], result: ServiceResult, ignore_errors: bool = False) -> None:
    for command in commands:
        if shutil.which(command[0]) is None:
            result.errors.append(f"{command[0]} not found; run this yourself: "
                                 f"{subprocess.list2cmdline(command)}")
            continue
        finished = subprocess.run(command, capture_output=True, text=True, check=False)
        text = (finished.stdout + finished.stderr).strip()
        if text:
            result.output.append(text)
        if finished.returncode and not ignore_errors:
            result.errors.append(f"{subprocess.list2cmdline(command)} failed: {text}")


def install(experiment: Path, action: str = "capture", extra: list[str] | None = None,
            register: bool = True) -> ServiceResult:
    service = plan(experiment, action, extra)
    result = ServiceResult(service)
    service.path.parent.mkdir(parents=True, exist_ok=True)
    service.path.write_text(service.contents, encoding="utf-8")
    result.wrote = True
    if register:
        # The first command (e.g. unload) may fail simply because nothing is installed yet.
        _run(service.commands[:-1], result, ignore_errors=True)
        _run(service.commands[-1:], result)
        result.registered = not result.errors
    return result


def uninstall(experiment: Path, action: str = "capture") -> ServiceResult:
    service = plan(experiment, action)
    result = ServiceResult(service)
    _run(service.remove_commands, result, ignore_errors=True)
    if service.path.exists():
        service.path.unlink()
        result.wrote = True
    return result


def installed(experiment: Path, action: str = "capture") -> bool:
    return plan(experiment, action).path.exists()
