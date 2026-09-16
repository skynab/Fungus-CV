from datetime import timedelta

import cv2

from fungus_cv.capture.camera import Camera
from fungus_cv.capture.scheduler import CaptureSession
from fungus_cv.storage import parse_iso_utc

from .conftest import FakeCapture


def camera_factory(fake_kwargs=None, caps=None):
    def factory(cfg):
        def opener(index, backend):
            cap = FakeCapture(index, backend, **(fake_kwargs or {}))
            if caps is not None:
                caps.append(cap)
            return cap

        return Camera(cfg, opener=opener, sleep=lambda s: None)

    return factory


def configure(exp, **capture):
    for key, value in capture.items():
        setattr(exp.config.capture, key, value)
    exp.config.cameras[0].warmup_frames = 0
    exp.config.cameras[0].settle_seconds = 0


def test_captures_on_fixed_schedule(experiment, clock):
    configure(experiment, interval=10.0, max_frames=4)
    session = CaptureSession(experiment, clock=clock, camera_factory=camera_factory())
    summary = session.run()

    assert summary.saved == 4 and summary.failed == 0
    rows = experiment.read_frames()
    times = [parse_iso_utc(r["timestamp_utc"]) for r in rows]
    assert [(b - a) for a, b in zip(times, times[1:])] == [timedelta(seconds=10)] * 3
    for row in rows:
        assert (experiment.root / row["file"]).exists()
        assert row["status"] == "ok"
        assert row["camera_settings"]["locked"]["exposure"] == -6.0
        assert float(row["mean_brightness"]) == 128.0


def test_duration_limit(experiment, clock):
    configure(experiment, interval=60.0, duration=300.0)
    summary = CaptureSession(experiment, clock=clock, camera_factory=camera_factory()).run()
    assert summary.saved == 6  # t = 0, 60, ..., 300
    assert summary.stopped_reason == "duration reached"


def test_retry_reopens_camera_after_failed_read(experiment, clock):
    configure(experiment, interval=10.0, max_frames=1, retries=2, retry_delay=1.0)
    cam_cfg = experiment.config.cameras[0]
    cam_cfg.exposure = cam_cfg.white_balance = cam_cfg.focus = "auto"
    failures_left = [2]  # like a USB hiccup: the next two reads fail, whichever handle
    caps = []

    def factory(cfg):
        def opener(index, backend):
            cap = FakeCapture(index, backend)
            real_read = cap.read

            def read():
                if failures_left[0] > 0:
                    failures_left[0] -= 1
                    return False, None
                return real_read()

            cap.read = read
            caps.append(cap)
            return cap

        return Camera(cfg, opener=opener, sleep=lambda s: None)

    summary = CaptureSession(experiment, clock=clock, camera_factory=factory).run()
    assert summary.saved == 1 and summary.failed == 0
    assert len(caps) == 3  # closed and reopened after each failed attempt


def test_failure_is_recorded(experiment, clock):
    configure(experiment, interval=10.0, max_frames=2, retries=1, retry_delay=0.0)
    session = CaptureSession(
        experiment, clock=clock, camera_factory=camera_factory({"fail_reads": 10**6})
    )
    summary = session.run()
    assert summary.saved == 0 and summary.failed == 2
    rows = experiment.read_frames()
    assert [r["status"] for r in rows] == ["failed", "failed"]
    assert "no frame" in rows[0]["notes"]


def test_missed_slots_are_skipped_not_bunched(experiment, clock):
    configure(experiment, interval=10.0, max_frames=3)
    session = CaptureSession(experiment, clock=clock, camera_factory=camera_factory())

    original = session.capture_round

    def slow_round(scheduled=None):
        records = original(scheduled)
        if session.summary.rounds == 1:
            clock.t += timedelta(seconds=35)  # e.g. the laptop slept
        return records

    session.capture_round = slow_round
    summary = session.run()
    times = [parse_iso_utc(r["timestamp_utc"]) for r in experiment.read_frames()]
    offsets = [(t - times[0]).total_seconds() for t in times]
    # Slots at 10s and 20s were missed; the 30s slot is taken as soon as possible (35s).
    assert offsets == [0.0, 35.0, 40.0]
    assert summary.skipped_slots == 2


def test_start_at_keeps_schedule_grid(experiment, clock):
    start = clock.now() - timedelta(seconds=90)
    configure(experiment, interval=60.0, max_frames=2, start_at=start)
    CaptureSession(experiment, clock=clock, camera_factory=camera_factory()).run()
    times = [parse_iso_utc(r["timestamp_utc"]) for r in experiment.read_frames()]
    # Restarting mid-run waits for the next slot on the original grid (start + 120s).
    assert times == [start + timedelta(seconds=120), start + timedelta(seconds=180)]


def test_camera_closed_between_long_intervals(experiment, clock):
    configure(experiment, interval=3600.0, max_frames=2)
    caps = []
    CaptureSession(experiment, clock=clock, camera_factory=camera_factory(caps=caps)).run()
    assert len(caps) == 2
    assert all(not c.opened for c in caps)


def test_jpg_output(experiment, clock):
    configure(experiment, interval=1.0, max_frames=1, image_format="jpg")
    CaptureSession(experiment, clock=clock, camera_factory=camera_factory()).run()
    row = experiment.read_frames()[0]
    assert row["file"].endswith(".jpg")
    assert cv2.imread(str(experiment.root / row["file"])) is not None
