from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from fungus_cv.storage import Experiment


class FakeCapture:
    """Stands in for ``cv2.VideoCapture``; controls behave like a cooperative driver."""

    def __init__(self, index=0, backend=0, fail_reads=0, supported=True, frame_value=128):
        self.index = index
        self.backend = backend
        self.opened = True
        self.fail_reads = fail_reads
        self.supported = supported
        self.frame_value = frame_value
        self.props: dict[int, float] = {
            cv2.CAP_PROP_EXPOSURE: -6.0,
            cv2.CAP_PROP_WB_TEMPERATURE: 4600.0,
            cv2.CAP_PROP_FOCUS: 30.0,
        }
        self.set_calls: list[tuple[int, float]] = []
        self.reads = 0

    def isOpened(self):  # noqa: N802 - mirrors OpenCV
        return self.opened

    def read(self):
        self.reads += 1
        if self.fail_reads > 0:
            self.fail_reads -= 1
            return False, None
        w = int(self.props.get(cv2.CAP_PROP_FRAME_WIDTH, 64))
        h = int(self.props.get(cv2.CAP_PROP_FRAME_HEIGHT, 48))
        return True, np.full((h, w, 3), self.frame_value, np.uint8)

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        if not self.supported:
            return False
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def release(self):
        self.opened = False


class FakeClock:
    def __init__(self, start: datetime | None = None):
        self.t = start or datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
        self.slept = 0.0

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.t += timedelta(seconds=seconds)


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    exp = Experiment.create(tmp_path / "exp", name="test")
    return exp


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
