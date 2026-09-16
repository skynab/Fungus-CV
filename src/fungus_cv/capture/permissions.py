"""Camera permission checks and camera names, per operating system.

macOS asks the user once per *app* (Terminal, iTerm, VS Code, ...), not per program: a Python
script inherits the permission of the app it was started from. OpenCV requests permission
and gives up immediately, so a first run fails even if the user then clicks Allow. Checking
and waiting here avoids that, and explains what to do when macOS won't ask at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

AUTHORIZED = "authorized"
NOT_DETERMINED = "not_determined"
DENIED = "denied"
RESTRICTED = "restricted"
UNKNOWN = "unknown"  # can't tell (e.g. PyObjC missing)
NOT_APPLICABLE = "not_applicable"  # this OS has no per-app camera permission we can check

_AV_STATUS = {0: NOT_DETERMINED, 1: RESTRICTED, 2: DENIED, 3: AUTHORIZED}


@dataclass
class CameraAccess:
    status: str
    app: str | None = None  # app whose permission applies (macOS)
    prompt_refused: bool = False  # macOS answered "no" without asking the user
    requested: bool = False  # whether permission was asked for (vs. only checked)

    @property
    def ok(self) -> bool:
        return self.status in (AUTHORIZED, UNKNOWN, NOT_APPLICABLE)

    def advice(self) -> str:
        app = self.app or "the app you are running this from (e.g. Terminal)"
        settings = "System Settings > Privacy & Security > Camera"
        if self.status == DENIED:
            return (f"Camera access was denied for {app}. Turn it on in {settings}, then quit "
                    f"and reopen {app} and run the command again.")
        if self.status == RESTRICTED:
            return ("Camera access is restricted on this Mac (e.g. by Screen Time or a device "
                    "management profile) and cannot be enabled from here.")
        if self.status == NOT_DETERMINED and self.prompt_refused:
            return (f"macOS will not show a camera permission prompt for {app}. Run this "
                    "command from the Terminal app (or iTerm) instead, where macOS asks once "
                    f"and you can click Allow; or add {app} in {settings}.")
        if self.status == NOT_DETERMINED and not self.requested:
            return (f"Camera permission for {app} hasn't been decided yet. Run `fungus cameras` "
                    "(or `fungus doctor --request-permission`) to ask for it.")
        if self.status == NOT_DETERMINED:
            return (f"No answer to the camera permission prompt for {app}. Click Allow when "
                    f"macOS asks, or enable {app} in {settings}, then run the command again.")
        return ""


# --- which app does macOS charge the permission to? -----------------------------------------


def _ps_parent(pid: int) -> tuple[int, str] | None:
    try:
        out = subprocess.run(["ps", "-o", "ppid=,comm=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    ppid, _, command = out.partition(" ")
    try:
        return int(ppid), command.strip()
    except ValueError:
        return None


def _app_name(path: str) -> str | None:
    """``/Applications/Foo Bar.app/Contents/...`` -> ``Foo Bar`` (outermost app bundle)."""
    if ".app/" not in path + "/":
        return None
    bundle = (path + "/").split(".app/")[0]
    return Path(bundle).name


def responsible_app(pid: int | None = None, parent_of=_ps_parent) -> str | None:
    """The highest app bundle among this process's ancestors (macOS asks per app)."""
    pid = pid or os.getpid()
    found = None
    seen = set()
    while pid and pid > 1 and pid not in seen:
        seen.add(pid)
        info = parent_of(pid)
        if info is None:
            break
        ppid, command = info
        name = _app_name(command)
        if name:
            found = name
        pid = ppid
    if found:
        return found
    term = os.environ.get("TERM_PROGRAM", "")
    return {"Apple_Terminal": "Terminal", "iTerm.app": "iTerm", "vscode": "Visual Studio Code"
            }.get(term, term or None)


# --- permission status -----------------------------------------------------------------------


def _avfoundation():
    try:
        import AVFoundation
        import Foundation
    except ImportError:
        return None
    return AVFoundation, Foundation


def camera_access(
    request: bool = True,
    timeout: float = 60.0,
    on_prompt: Callable[[str | None], None] | None = None,
    platform: str = sys.platform,
) -> CameraAccess:
    """Check camera permission; on macOS, ask for it if undecided and wait for the answer."""
    if platform != "darwin":
        return CameraAccess(NOT_APPLICABLE)
    app = responsible_app()
    frameworks = _avfoundation()
    if frameworks is None:
        return CameraAccess(UNKNOWN, app)
    av, foundation = frameworks
    status = _AV_STATUS.get(int(av.AVCaptureDevice.authorizationStatusForMediaType_(
        av.AVMediaTypeVideo)), UNKNOWN)
    if status != NOT_DETERMINED or not request:
        return CameraAccess(status, app)

    if on_prompt:
        on_prompt(app)
    done = threading.Event()
    started = time.monotonic()

    def handler(granted):
        done.set()

    av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(av.AVMediaTypeVideo, handler)
    while not done.is_set() and time.monotonic() - started < timeout:
        # Keep the run loop turning so the system can deliver the answer.
        foundation.NSRunLoop.currentRunLoop().runUntilDate_(
            foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1))
    status = _AV_STATUS.get(int(av.AVCaptureDevice.authorizationStatusForMediaType_(
        av.AVMediaTypeVideo)), UNKNOWN)
    # An instant "no" with the status still undecided means no prompt was ever shown.
    refused = done.is_set() and status == NOT_DETERMINED and \
        time.monotonic() - started < 1.0
    return CameraAccess(status, app, prompt_refused=refused, requested=True)


# --- camera names ----------------------------------------------------------------------------


def camera_names(platform: str = sys.platform, sysfs: Path = Path("/sys/class/video4linux")
                 ) -> dict[int, str] | list[str]:
    """Human-readable camera names.

    Linux: exact mapping from index (``/dev/videoN``) to name. macOS: the names macOS reports,
    as a list (OpenCV's index order usually, but not always, matches). Windows: empty.
    """
    if platform.startswith("linux"):
        names = {}
        for node in sorted(sysfs.glob("video*")):
            try:
                index = int(node.name.removeprefix("video"))
                names[index] = (node / "name").read_text(encoding="utf-8").strip()
            except (ValueError, OSError):
                continue
        return names
    if platform == "darwin":
        frameworks = _avfoundation()
        if frameworks is None:
            return []
        av, _ = frameworks
        types = [av.AVCaptureDeviceTypeBuiltInWideAngleCamera]
        # External covers USB webcams and iPhones used as webcams. (Asking for the
        # Continuity Camera type directly makes macOS print an Info.plist warning.)
        for name in ("AVCaptureDeviceTypeExternal", "AVCaptureDeviceTypeExternalUnknown"):
            if hasattr(av, name):
                types.append(getattr(av, name))
                break
        session = av.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(  # noqa: E501
            types, av.AVMediaTypeVideo, 0)
        return [str(d.localizedName()) for d in session.devices()]
    return {}
