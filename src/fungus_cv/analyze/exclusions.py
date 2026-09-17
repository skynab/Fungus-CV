"""Frames a person decided to leave out of fits (e.g. a hand in the picture), with reasons.

Stored in ``exclusions.json`` next to ``annotations.json``. They are a decision about the
data, not an analysis setting, so they survive re-analysis and apply to every run. Reports
and studies always honour them and list them in ``frame_flags.csv`` and the methods text.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fungus_cv.storage import Experiment, iso_utc, utc_now

EXCLUSIONS_NAME = "exclusions.json"
ALL_PLOTS = ""


@dataclass
class Exclusion:
    frame_file: str  # as in frames.csv, e.g. frames/2026-09-20T14-05-00.000Z_cam0.png
    reason: str
    plot: str = ALL_PLOTS  # "" = every plot
    created_utc: str = ""

    def applies_to(self, frame_file: str, plot: str) -> bool:
        return self.frame_file == frame_file and self.plot in (ALL_PLOTS, plot)


def path_for(experiment: Experiment) -> Path:
    return experiment.root / EXCLUSIONS_NAME


def load(experiment: Experiment) -> list[Exclusion]:
    path = path_for(experiment)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Exclusion(**e) for e in data.get("exclusions", [])]


def save(experiment: Experiment, exclusions: list[Exclusion]) -> None:
    path = path_for(experiment)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps({"exclusions": [asdict(e) for e in exclusions]}, indent=2),
                    encoding="utf-8")
    part.replace(path)


def resolve_frame(experiment: Experiment, frame: str) -> str:
    """A frame file from its path, file name or name without extension."""
    wanted = frame.strip()
    for row in experiment.read_frames():
        name = Path(row["file"]).name
        if wanted in (row["file"], name, Path(name).stem):
            return row["file"]
    raise ValueError(f"frame {frame!r} is not in frames.csv")


def exclude(experiment: Experiment, frame: str, reason: str, plot: str = ALL_PLOTS) -> Exclusion:
    if not reason.strip():
        raise ValueError("give a reason, so the exclusion can be explained later")
    frame_file = resolve_frame(experiment, frame)
    items = [e for e in load(experiment) if not (e.frame_file == frame_file and e.plot == plot)]
    item = Exclusion(frame_file, reason.strip(), plot, iso_utc(utc_now()))
    save(experiment, items + [item])
    return item


def include(experiment: Experiment, frame: str, plot: str | None = None) -> int:
    """Remove exclusions of a frame (for one plot, or all if ``plot`` is None)."""
    frame_file = resolve_frame(experiment, frame)
    items = load(experiment)
    kept = [e for e in items if not (e.frame_file == frame_file and
                                     (plot is None or e.plot == plot))]
    save(experiment, kept)
    return len(items) - len(kept)


def reason_for(exclusions: list[Exclusion], frame_file: str, plot: str) -> str:
    """The reason a frame is excluded for a plot ("" if it isn't)."""
    return next((e.reason for e in exclusions if e.applies_to(frame_file, plot)), "")
