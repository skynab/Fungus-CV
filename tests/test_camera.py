import cv2
import pytest

from fungus_cv.capture.camera import Camera, CameraError, list_cameras, resolve_backend
from fungus_cv.config import CameraConfig

from .conftest import FakeCapture


def make_camera(cfg: CameraConfig, **fake_kwargs):
    caps = []

    def opener(index, backend):
        cap = FakeCapture(index, backend, **fake_kwargs)
        caps.append(cap)
        return cap

    return Camera(cfg, opener=opener, sleep=lambda s: None), caps


@pytest.mark.parametrize(
    "platform, expected",
    [("win32", "dshow"), ("darwin", "avfoundation"), ("linux", "v4l2"), ("sunos5", "any")],
)
def test_resolve_backend(platform, expected):
    assert resolve_backend("auto", platform) == expected
    assert resolve_backend("msmf", platform) == "msmf"


def test_lock_captures_values_and_reapplies_on_reopen():
    cfg = CameraConfig(backend="v4l2", width=64, height=48, warmup_frames=1)
    cam, caps = make_camera(cfg)
    cam.open()
    assert cam.locked == {"exposure": -6.0, "white_balance": 4600.0, "focus": 30.0}
    assert caps[0].props[cv2.CAP_PROP_AUTO_EXPOSURE] == 1.0  # v4l2 manual mode
    assert caps[0].props[cv2.CAP_PROP_AUTO_WB] == 0.0
    cam.close()

    cam.open()  # second open reuses locked values instead of re-settling
    second = caps[1]
    assert (cv2.CAP_PROP_EXPOSURE, -6.0) in second.set_calls
    assert (cv2.CAP_PROP_AUTO_EXPOSURE, 3.0) not in second.set_calls
    cam.close()


def test_fixed_and_auto_controls():
    cfg = CameraConfig(backend="dshow", exposure=-4, white_balance="auto", focus="auto")
    cam, caps = make_camera(cfg)
    cam.open()
    props = caps[0].props
    assert props[cv2.CAP_PROP_AUTO_EXPOSURE] == 0.25
    assert props[cv2.CAP_PROP_EXPOSURE] == -4
    assert props[cv2.CAP_PROP_AUTO_WB] == 1.0
    assert cam.locked == {}


def test_unsupported_controls_stay_auto_with_warning():
    cfg = CameraConfig(backend="avfoundation", warmup_frames=0)
    cam, _ = make_camera(cfg, supported=False)
    cam.open()
    assert cam.locked == {}
    assert cam.unlockable == {"exposure", "white_balance", "focus"}
    assert any("cannot lock" in w for w in cam.warnings)
    assert cam.read() is not None


def test_lock_reverted_if_image_goes_dark():
    cfg = CameraConfig(backend="v4l2", white_balance="auto", focus="auto", warmup_frames=0)
    cam, caps = make_camera(cfg)

    original_set = FakeCapture.set

    def darkening_set(self, prop, value):
        if prop == cv2.CAP_PROP_EXPOSURE:
            self.frame_value = 2
        return original_set(self, prop, value)

    FakeCapture.set = darkening_set
    try:
        cam.open()
    finally:
        FakeCapture.set = original_set
    assert "exposure" not in cam.locked
    assert "exposure" in cam.unlockable
    assert caps[0].props[cv2.CAP_PROP_AUTO_EXPOSURE] == 3.0  # back to auto


def test_open_failure_raises():
    def opener(index, backend):
        cap = FakeCapture()
        cap.opened = False
        return cap

    with pytest.raises(CameraError):
        Camera(CameraConfig(), opener=opener).open()


def test_list_cameras_reports_working_indices():
    def opener(index, backend):
        cap = FakeCapture(index, backend)
        cap.opened = index in (0, 2)
        return cap

    found = list_cameras(max_index=4, backend="v4l2", opener=opener)
    assert [c["index"] for c in found] == [0, 2]
    assert found[0]["width"] == 64
