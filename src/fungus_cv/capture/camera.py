"""Cross-platform webcam access on top of OpenCV ``VideoCapture``."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from fungus_cv.config import CameraConfig

log = logging.getLogger(__name__)

BACKEND_IDS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "v4l2": cv2.CAP_V4L2,
    "gstreamer": cv2.CAP_GSTREAMER,
}

# Value written to CAP_PROP_AUTO_EXPOSURE for (auto, manual). These are driver
# conventions, not documented OpenCV constants, and differ between backends.
AUTO_EXPOSURE_VALUES = {
    "dshow": (0.75, 0.25),
    "v4l2": (3.0, 1.0),
}
DEFAULT_AUTO_EXPOSURE_VALUES = (1.0, 0.0)

# name -> (auto property, value property)
CONTROLS = {
    "exposure": (cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_EXPOSURE),
    "white_balance": (cv2.CAP_PROP_AUTO_WB, cv2.CAP_PROP_WB_TEMPERATURE),
    "focus": (cv2.CAP_PROP_AUTOFOCUS, cv2.CAP_PROP_FOCUS),
}

READBACK_PROPS = {
    "width": cv2.CAP_PROP_FRAME_WIDTH,
    "height": cv2.CAP_PROP_FRAME_HEIGHT,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "auto_white_balance": cv2.CAP_PROP_AUTO_WB,
    "white_balance": cv2.CAP_PROP_WB_TEMPERATURE,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
    "focus": cv2.CAP_PROP_FOCUS,
    "gain": cv2.CAP_PROP_GAIN,
}


class CameraError(RuntimeError):
    pass


def resolve_backend(name: str, platform: str = sys.platform) -> str:
    if name != "auto":
        return name
    if platform.startswith("win"):
        return "dshow"  # opens faster and exposes more controls than MSMF
    if platform == "darwin":
        return "avfoundation"
    if platform.startswith("linux"):
        return "v4l2"
    return "any"


def quiet_opencv_logs() -> None:
    """OpenCV prints noisy warnings when probing camera indices that don't exist."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass


Opener = Callable[[int, int], Any]


def _default_opener(index: int, backend_id: int) -> Any:
    return cv2.VideoCapture(index, backend_id)


# One lock per device index, held from open() to close(). Two threads touching the same
# device at once (e.g. a preview still closing while a capture opens it) crashes some
# drivers (DirectShow: access violation / heap corruption), so the second one waits.
_DEVICE_LOCKS: dict[int, threading.Lock] = {}
_DEVICE_LOCKS_GUARD = threading.Lock()
DEVICE_WAIT_SECONDS = 10.0


def _device_lock(index: int) -> threading.Lock:
    with _DEVICE_LOCKS_GUARD:
        return _DEVICE_LOCKS.setdefault(index, threading.Lock())


class Camera:
    """One physical camera. ``open()`` applies the configured format and controls."""

    def __init__(self, config: CameraConfig, opener: Opener = _default_opener, sleep=time.sleep):
        self.config = config
        self.backend = resolve_backend(config.backend)
        self._opener = opener
        self._sleep = sleep
        self._cap: Any = None
        self._lock_held: threading.Lock | None = None
        # Values captured for controls set to "lock"; reused on every reopen.
        self.locked: dict[str, float] = {}
        # Controls this camera/driver cannot lock; left on auto without retrying.
        self.unlockable: set[str] = set()
        self.warnings: list[str] = []

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> None:
        if self._cap is not None:
            return
        cfg = self.config
        lock = _device_lock(cfg.index)
        if not lock.acquire(timeout=DEVICE_WAIT_SECONDS):
            raise CameraError(
                f"camera {cfg.name!r} (index {cfg.index}) is in use by another part of the app "
                "(close the live preview or stop the capture first)"
            )
        try:
            cap = self._opener(cfg.index, BACKEND_IDS[self.backend])
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                raise CameraError(
                    f"could not open camera {cfg.name!r} (index {cfg.index}, "
                    f"backend {self.backend})"
                )
        except BaseException:
            lock.release()
            raise
        self._cap = cap
        self._lock_held = lock
        try:
            self._apply_format()
            self._apply_controls()
            if cfg.gain is not None:
                self._set(cv2.CAP_PROP_GAIN, cfg.gain, "gain")

            self._discard(cfg.warmup_frames)
            pending = [
                n
                for n in CONTROLS
                if getattr(cfg, n) == "lock" and n not in self.locked and n not in self.unlockable
            ]
            if pending:
                self._sleep(cfg.settle_seconds)
                self._discard(cfg.warmup_frames)
                self._lock(pending)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
                if self._lock_held is not None:
                    self._lock_held.release()
                    self._lock_held = None

    def __del__(self) -> None:
        # A camera dropped without close() must not keep the device locked for the process.
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read(self) -> np.ndarray:
        if self._cap is None:
            raise CameraError("camera is not open")
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            raise CameraError(f"camera {self.config.name!r} returned no frame")
        return frame

    def settings(self) -> dict[str, Any]:
        """What the driver reports right now (may not match what was requested)."""
        if self._cap is None:
            return {}
        out: dict[str, Any] = {"backend": self.backend}
        for name, prop in READBACK_PROPS.items():
            try:
                out[name] = float(self._cap.get(prop))
            except Exception:
                out[name] = None
        if self.locked:
            out["locked"] = dict(self.locked)
        return out

    def open_driver_settings(self) -> bool:
        """Show the driver's own settings dialog (DirectShow on Windows only)."""
        return self._cap is not None and bool(self._cap.set(cv2.CAP_PROP_SETTINGS, 1))

    # --- internals -----------------------------------------------------------------

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            log.warning("[%s] %s", self.config.name, message)

    def _set(self, prop: int, value: float, label: str) -> bool:
        ok = bool(self._cap.set(prop, float(value)))
        if not ok:
            self._warn(f"camera/driver did not accept {label}={value} (backend {self.backend})")
        return ok

    def _auto_values(self) -> tuple[float, float]:
        return AUTO_EXPOSURE_VALUES.get(self.backend, DEFAULT_AUTO_EXPOSURE_VALUES)

    def _set_auto(self, control: str, auto: bool) -> None:
        auto_prop, _ = CONTROLS[control]
        if control == "exposure":
            on, off = self._auto_values()
            value = on if auto else off
        else:
            value = 1.0 if auto else 0.0
        self._set(auto_prop, value, f"auto_{control}={'on' if auto else 'off'}")

    def _set_manual(self, control: str, value: float) -> None:
        _, value_prop = CONTROLS[control]
        self._set_auto(control, False)
        self._set(value_prop, value, control)

    def _apply_format(self) -> None:
        cfg = self.config
        if cfg.fourcc:
            self._set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc), "fourcc")
        if cfg.width:
            self._set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width, "width")
        if cfg.height:
            self._set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height, "height")

    def _apply_controls(self) -> None:
        for control in CONTROLS:
            setting = getattr(self.config, control)
            if setting == "auto":
                self._set_auto(control, True)
            elif setting == "lock":
                if control in self.locked:
                    self._set_manual(control, self.locked[control])
                else:
                    self._set_auto(control, True)
            else:
                self._set_manual(control, float(setting))

    def _lock(self, controls: list[str]) -> None:
        before = self._brightness()
        newly_locked = []
        for control in controls:
            auto_prop, value_prop = CONTROLS[control]
            value = float(self._cap.get(value_prop))
            off = self._auto_values()[1] if control == "exposure" else 0.0
            if (control == "white_balance" and value <= 0) or not self._cap.set(auto_prop, off):
                self.unlockable.add(control)
                self._warn(f"cannot lock {control} on this camera/driver; it stays on auto")
                self._set_auto(control, True)
                continue
            self._set(value_prop, value, control)
            self.locked[control] = value
            newly_locked.append(control)
            log.info("[%s] locked %s at %s", self.config.name, control, value)

        self._discard(self.config.warmup_frames)
        after = self._brightness()
        # Some drivers report a meaningless value (often 0) for controls they don't really
        # support. If locking made the image much darker, undo it rather than save black frames.
        if newly_locked and before is not None and after is not None and before > 10:
            if after < 0.25 * before:
                for control in newly_locked:
                    self.locked.pop(control, None)
                    self.unlockable.add(control)
                    self._set_auto(control, True)
                self._warn(
                    f"locking {newly_locked} darkened the image ({before:.0f} -> {after:.0f}); "
                    "reverted to auto. Set explicit values in config.yaml instead."
                )
                self._discard(self.config.warmup_frames)

    def _brightness(self) -> float | None:
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            return None
        return float(frame.mean())

    def _discard(self, n: int) -> None:
        for _ in range(n):
            self._cap.read()


def list_cameras(
    max_index: int = 8, backend: str = "auto", opener: Opener = _default_opener
) -> list[dict]:
    """Probe camera indices 0..max_index-1 and report the ones that deliver a frame."""
    quiet_opencv_logs()
    resolved = resolve_backend(backend)
    found = []
    for index in range(max_index):
        lock = _device_lock(index)
        if not lock.acquire(timeout=DEVICE_WAIT_SECONDS):
            continue  # held open by a preview or capture in this process
        cap = None
        try:
            cap = opener(index, BACKEND_IDS[resolved])
            if cap is None or not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            found.append(
                {
                    "index": index,
                    "backend": resolved,
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                }
            )
        finally:
            if cap is not None:
                cap.release()
            lock.release()
    return found
