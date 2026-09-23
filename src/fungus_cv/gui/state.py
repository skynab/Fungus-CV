"""The currently open experiment, shared by all pages."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Signal

from fungus_cv.gui.preview import CameraPreview
from fungus_cv.storage import Experiment

MAX_RECENT = 10


class AppState(QObject):
    experiment_changed = Signal(object)  # Experiment or None
    config_changed = Signal()
    camera_changed = Signal(object)  # which camera the pages work on (None = the only one)
    busy_changed = Signal(bool)  # a capture is running (pages should not change settings)

    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        self.settings = settings or QSettings()
        self.experiment: Experiment | None = None
        self.camera: str | None = None  # with several cameras, the one being worked on
        self.capturing = False
        # One live preview for the whole window (a camera serves one reader at a time).
        self.preview = CameraPreview(self)

    # --- experiments -------------------------------------------------------------------

    def open(self, path: Path) -> Experiment:
        experiment = Experiment(Path(path))  # raises with a readable message if invalid
        self.experiment = experiment
        self.camera = experiment.config.cameras[0].name if len(
            experiment.config.cameras) > 1 else None
        self._remember(experiment.root)
        self.experiment_changed.emit(experiment)
        return experiment

    def create(self, path: Path, name: str | None = None) -> Experiment:
        Experiment.create(Path(path), name)
        return self.open(path)

    def reload(self) -> None:
        """Re-read config.yaml after it was edited."""
        if self.experiment is not None:
            self.experiment = Experiment(self.experiment.root)
            self.config_changed.emit()

    def close(self) -> None:
        self.experiment = None
        self.experiment_changed.emit(None)

    def set_camera(self, camera: str | None) -> None:
        if camera != self.camera:
            self.camera = camera
            self.camera_changed.emit(camera)

    def results_dir(self):
        """Where the current camera's results live."""
        from fungus_cv.analyze.pipeline import results_dir_for

        return None if self.experiment is None else results_dir_for(self.experiment, self.camera)

    def set_capturing(self, value: bool) -> None:
        self.capturing = value  # set first: stopping the preview below tells the pages
        if value:
            self.preview.stop()  # the capture needs the camera the preview holds open
        self.busy_changed.emit(value)

    # --- recent folders ------------------------------------------------------------------

    def recent(self) -> list[Path]:
        raw = self.settings.value("recent_experiments", [], list) or []
        return [Path(p) for p in raw if (Path(p) / "config.yaml").exists()]

    def _remember(self, root: Path) -> None:
        items = [str(root.resolve())] + [str(p) for p in self.recent()
                                         if p.resolve() != root.resolve()]
        self.settings.setValue("recent_experiments", items[:MAX_RECENT])
