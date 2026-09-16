"""Keep the computer from sleeping during long capture runs."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager

log = logging.getLogger(__name__)

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_awake(enabled: bool = True) -> Iterator[None]:
    """Prevent idle sleep while the block runs. Closing a laptop lid may still sleep it."""
    if not enabled:
        yield
        return

    proc: subprocess.Popen | None = None
    windows = False
    try:
        if sys.platform.startswith("win"):
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
            windows = True
        elif sys.platform == "darwin" and shutil.which("caffeinate"):
            proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        elif shutil.which("systemd-inhibit"):
            proc = subprocess.Popen(
                ["systemd-inhibit", "--what=idle:sleep", "--who=fungus-cv",
                 "--why=time-lapse capture", "sleep", "infinity"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            log.warning("don't know how to prevent sleep on this system; check power settings")
    except Exception as exc:  # never let this stop a capture run
        log.warning("could not prevent sleep: %s", exc)

    try:
        yield
    finally:
        if windows:
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        if proc is not None:
            proc.terminate()
