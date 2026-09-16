"""Environment checks behind `fungus doctor`: camera access, display, optional dependencies."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass

import cv2


@dataclass
class Check:
    name: str
    ok: bool | None  # None = informational / not applicable
    detail: str
    fix: str = ""


def opencv_gui_backend() -> str:
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", 1)[1].strip()
    return "UNKNOWN"


def gui_problem(platform_name: str = sys.platform, env=os.environ) -> str:
    """Why interactive windows can't open here, or "" if they should work."""
    backend = opencv_gui_backend()
    if backend.upper() in ("NONE", ""):
        return ("This OpenCV build has no window support (opencv-python-headless?). Install "
                "the regular build: pip uninstall opencv-python-headless && "
                "pip install opencv-python")
    if platform_name.startswith("linux") and not (env.get("DISPLAY") or
                                                  env.get("WAYLAND_DISPLAY")):
        return ("No display available (e.g. an SSH session). Run interactive commands on a "
                "desktop session, or connect with `ssh -X`.")
    return ""


def no_camera_hint(platform_name: str = sys.platform) -> str:
    """What usually hides cameras on this OS."""
    if platform_name.startswith("win"):
        return ("On Windows, check Settings > Privacy & security > Camera: 'Camera access' and "
                "'Let desktop apps access your camera' must be on. Also close apps that may "
                "hold the camera (Teams, Zoom, the Camera app), or try --backend msmf.")
    if platform_name.startswith("linux"):
        return ("On Linux, check that /dev/video* exists and your user is in the 'video' group "
                "(sudo usermod -aG video $USER, then log out and back in).")
    if platform_name == "darwin":
        return ("On macOS, close other apps using the camera, and check the cable/hub for USB "
                "webcams. Run `fungus doctor` for more checks.")
    return "Run `fungus doctor` for more checks."


def run_checks(probe_cameras: bool = True, request_permission: bool = False) -> list[Check]:
    from fungus_cv import __version__
    from fungus_cv.capture.camera import list_cameras, resolve_backend
    from fungus_cv.capture.permissions import (
        AUTHORIZED,
        NOT_APPLICABLE,
        UNKNOWN,
        camera_access,
        camera_names,
    )

    checks = [Check("fungus-cv", None, f"{__version__} on Python {platform.python_version()}, "
                                      f"{platform.platform()}")]

    backends = []
    try:
        backends = [cv2.videoio_registry.getBackendName(b)
                    for b in cv2.videoio_registry.getCameraBackends()]
    except AttributeError:
        pass
    wanted = resolve_backend("auto").upper()
    has_backend = wanted == "ANY" or wanted in backends or not backends
    checks.append(Check("OpenCV camera backends", has_backend,
                        f"OpenCV {cv2.__version__}: {', '.join(backends) or 'unknown'} "
                        f"(default here: {wanted.lower()})",
                        "" if has_backend else "reinstall opencv-python for this platform"))

    problem = gui_problem()
    checks.append(Check("Windows for preview/annotate", not problem,
                        f"OpenCV GUI: {opencv_gui_backend()}", problem))

    access = camera_access(request=request_permission)
    if access.status == NOT_APPLICABLE:
        detail = "no per-app camera permission to check on this OS"
        if sys.platform.startswith("linux"):
            devices = sorted(p for p in os.listdir("/dev") if p.startswith("video")) \
                if os.path.isdir("/dev") else []
            detail = f"video devices: {', '.join(devices) or 'none'}"
            if devices and not os.access(f"/dev/{devices[0]}", os.R_OK | os.W_OK):
                checks.append(Check("Camera permission", False, detail,
                                    "add your user to the 'video' group: sudo usermod -aG video "
                                    "$USER, then log out and back in"))
            else:
                checks.append(Check("Camera permission", None, detail))
        else:
            checks.append(Check("Camera permission", None, detail))
    elif access.status == UNKNOWN:
        checks.append(Check("Camera permission", None,
                            f"can't check (PyObjC not installed); app: {access.app}",
                            'pip install -e . (installs pyobjc-framework-AVFoundation)'))
    else:
        checks.append(Check("Camera permission", access.status == AUTHORIZED,
                            f"{access.status} for {access.app}", access.advice()))

    names = camera_names()
    listed = list(names.values()) if isinstance(names, dict) else names
    checks.append(Check("Cameras reported by the system", bool(listed) if listed != [] else None,
                        ", ".join(listed) if listed else "none reported (or not checkable here)"))

    if probe_cameras and access.ok:
        found = list_cameras(max_index=4)
        checks.append(Check("Cameras OpenCV can open", bool(found),
                            ", ".join(f"index {c['index']} {c['width']}x{c['height']}"
                                      for c in found) or "none",
                            "" if found else no_camera_hint()))

    for module, what in (("torch", "SAM 2 and trained models"), ("transformers", "SAM 2"),
                         ("torchvision", "SAM 2 and trained models")):
        try:
            mod = __import__(module)
            checks.append(Check(f"Optional: {module}", True,
                                f"{getattr(mod, '__version__', 'installed')} ({what})"))
        except ImportError:
            checks.append(Check(f"Optional: {module}", None, f"not installed ({what})",
                                'pip install -e ".[sam]"'))
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else \
            "mps" if torch.backends.mps.is_available() else "cpu"
        checks.append(Check("GPU for models", device != "cpu", device,
                            "" if device != "cpu" else "models will run on the CPU (slower)"))
    except ImportError:
        pass

    free_gb = shutil.disk_usage(os.getcwd()).free / 1e9
    checks.append(Check("Free disk space", free_gb > 5, f"{free_gb:.1f} GB here",
                        "" if free_gb > 5 else "long PNG time-lapses need several GB"))
    return checks
