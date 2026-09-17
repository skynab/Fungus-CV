"""Is an unattended run still healthy?

`fungus capture` writes a heartbeat (``status.json``) every round: when it last took a photo,
when the next one is due, how many were saved or failed, and how much disk is left. The checks
here read that and ``frames.csv`` and answer the questions that matter during a run of days or
weeks: is capture still alive, are photos arriving on schedule, are they usable, and is there
room left? Each check is ok, warning or problem, and `fungus health --watch` reports changes
through a webhook or a command of your choice.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from fungus_cv.storage import Experiment, iso_utc, parse_iso_utc, utc_now

log = logging.getLogger(__name__)

STATUS_NAME = "status.json"
OK, WARNING, PROBLEM = "ok", "warning", "problem"
LEVELS = (OK, WARNING, PROBLEM)
# A run is "late" once this many intervals have passed with nothing new.
LATE_INTERVALS = 2.0
VERY_LATE_INTERVALS = 5.0


@dataclass
class Heartbeat:
    updated_utc: str = ""
    experiment: str = ""
    state: str = ""  # running | finished
    pid: int = 0
    host: str = ""
    interval_s: float = 0.0
    last_frame_utc: str = ""
    next_due_utc: str = ""
    rounds: int = 0
    saved: int = 0
    failed: int = 0
    skipped_slots: int = 0
    free_disk_mb: float = 0.0
    stopped_reason: str = ""
    note: str = ""


def status_path(experiment: Experiment) -> Path:
    return experiment.root / STATUS_NAME


def write_heartbeat(experiment: Experiment, **fields) -> Heartbeat:
    """Record how the run is doing (atomically, so a reader never sees half a file)."""
    beat = Heartbeat(updated_utc=iso_utc(utc_now()), experiment=experiment.config.name,
                     pid=os.getpid(), host=socket.gethostname(),
                     free_disk_mb=round(experiment.free_disk_mb(), 1), **fields)
    path = status_path(experiment)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(asdict(beat), indent=2), encoding="utf-8")
    part.replace(path)
    return beat


def read_heartbeat(experiment: Experiment) -> Heartbeat | None:
    path = status_path(experiment)
    if not path.exists():
        return None
    try:
        return Heartbeat(**json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, TypeError) as exc:
        log.warning("cannot read %s (%s)", path, exc)
        return None


@dataclass
class Check:
    name: str
    level: str  # ok | warning | problem
    detail: str
    fix: str = ""

    @property
    def ok(self) -> bool:
        return self.level == OK


@dataclass
class HealthReport:
    experiment: str
    root: Path
    checked_utc: str
    checks: list[Check] = field(default_factory=list)
    heartbeat: Heartbeat | None = None

    @property
    def level(self) -> str:
        return max((c.level for c in self.checks), key=LEVELS.index, default=OK)

    @property
    def ok(self) -> bool:
        return self.level == OK

    @property
    def problems(self) -> list[Check]:
        return [c for c in self.checks if c.level != OK]

    def summary(self) -> str:
        if self.ok:
            return f"{self.experiment}: healthy"
        return f"{self.experiment}: " + "; ".join(f"{c.name} — {c.detail}" for c in self.problems)


def _age(iso: str, now: datetime) -> timedelta | None:
    try:
        return now - parse_iso_utc(iso)
    except (ValueError, AttributeError):
        return None


def _minutes(delta: timedelta) -> str:
    seconds = delta.total_seconds()
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def check(experiment: Experiment, quality_frames: int = 10,
          now: datetime | None = None) -> HealthReport:
    """Look at the heartbeat, the frames and the disk, and judge how the run is doing."""
    now = now or utc_now()
    report = HealthReport(experiment=experiment.config.name, root=experiment.root,
                          checked_utc=iso_utc(now))
    beat = read_heartbeat(experiment)
    report.heartbeat = beat
    interval = experiment.config.capture.interval
    rows = [r for r in experiment.read_frames() if r["status"] == "ok"]
    failures = [r for r in experiment.read_frames() if r["status"] == "failed"]

    # --- is capture running? ---------------------------------------------------------
    if beat is None:
        report.checks.append(Check(
            "capture", WARNING, "no status.json: capture has not run since this was added",
            "start capture with `fungus capture`, or ignore this for an imported experiment"))
    elif beat.state == "finished":
        report.checks.append(Check("capture", OK,
                                   f"finished ({beat.stopped_reason or 'stopped'})"))
    else:
        age = _age(beat.updated_utc, now)
        expected = max(interval, beat.interval_s or interval)
        if age is None:
            report.checks.append(Check("capture", WARNING, "status.json has no valid time"))
        elif age.total_seconds() > VERY_LATE_INTERVALS * expected:
            report.checks.append(Check(
                "capture", PROBLEM,
                f"no heartbeat for {_minutes(age)} (every {expected:.0f}s expected) — capture "
                f"looks dead (pid {beat.pid} on {beat.host})",
                "check the computer is awake and restart `fungus capture`"))
        elif age.total_seconds() > LATE_INTERVALS * expected:
            report.checks.append(Check("capture", WARNING,
                                       f"last heartbeat {_minutes(age)} ago"))
        else:
            report.checks.append(Check("capture", OK, f"running, last heartbeat "
                                                      f"{_minutes(age)} ago"))

    # --- are photos arriving? --------------------------------------------------------
    if not rows:
        report.checks.append(Check("frames", PROBLEM, "no photos yet",
                                   "check the camera with `fungus doctor`"))
    else:
        age = _age(rows[-1]["timestamp_utc"], now)
        if age is None:
            report.checks.append(Check("frames", WARNING, "last photo has no valid time"))
        elif age.total_seconds() > VERY_LATE_INTERVALS * interval:
            report.checks.append(Check(
                "frames", PROBLEM, f"last photo {_minutes(age)} ago, {len(rows)} in total",
                "check the camera and the capture log"))
        elif age.total_seconds() > LATE_INTERVALS * interval:
            report.checks.append(Check("frames", WARNING,
                                       f"last photo {_minutes(age)} ago ({len(rows)} in total)"))
        else:
            report.checks.append(Check("frames", OK,
                                       f"{len(rows)} photos, last {_minutes(age)} ago"))

    # --- failures and skipped slots --------------------------------------------------
    recent_failures = [r for r in failures if (_age(r["timestamp_utc"], now) or timedelta(0))
                       < timedelta(seconds=LATE_INTERVALS * max(interval, 1) * 10)]
    if recent_failures:
        last = recent_failures[-1].get("notes", "")
        report.checks.append(Check(
            "camera errors", WARNING if len(recent_failures) < 3 else PROBLEM,
            f"{len(recent_failures)} failed attempt(s) recently; last: {last[:120]}",
            "check the cable and that no other app holds the camera"))
    if beat is not None and beat.skipped_slots:
        report.checks.append(Check(
            "schedule", WARNING, f"{beat.skipped_slots} slot(s) skipped (computer asleep or "
            "busy?)", "stop the computer sleeping, or use a longer interval"))

    # --- are the photos usable? ------------------------------------------------------
    recent = rows[-quality_frames:]
    values = [(_float(r.get("mean_brightness")), _float(r.get("sharpness"))) for r in recent]
    brightness = [b for b, _ in values if b is not None]
    sharp = [s for _, s in values if s is not None]
    if brightness and sharp:
        reference = rows[0]
        ref_b, ref_s = _float(reference.get("mean_brightness")), _float(reference.get("sharpness"))
        latest_b, latest_s = brightness[-1], sharp[-1]
        problems = []
        if latest_b < 15:
            problems.append(f"almost black (brightness {latest_b:.0f})")
        elif ref_b and abs(latest_b / ref_b - 1) > 0.4:
            problems.append(f"brightness {latest_b:.0f} vs {ref_b:.0f} at the start")
        if ref_s and latest_s < 0.4 * ref_s:
            problems.append(f"blurred (sharpness {latest_s:.0f} vs {ref_s:.0f})")
        if problems:
            report.checks.append(Check("image quality", WARNING, "; ".join(problems),
                                       "check the light, the focus and whether anything moved"))
        else:
            report.checks.append(Check("image quality", OK,
                                       f"brightness {latest_b:.0f}, sharpness {latest_s:.0f}"))

    # --- disk ------------------------------------------------------------------------
    free_mb = experiment.free_disk_mb()
    minimum = experiment.config.capture.min_free_disk_mb
    per_frame_mb = _mean_frame_mb(experiment, rows)
    left = f", room for about {free_mb / per_frame_mb:,.0f} more photos" if per_frame_mb else ""
    if free_mb < minimum:
        report.checks.append(Check("disk", PROBLEM,
                                   f"{free_mb:,.0f} MB free, below the {minimum:,.0f} MB minimum",
                                   "free space; capture stops when it runs out"))
    elif per_frame_mb and free_mb / per_frame_mb < 50:
        report.checks.append(Check("disk", WARNING, f"{free_mb:,.0f} MB free{left}"))
    else:
        report.checks.append(Check("disk", OK, f"{free_mb / 1000:,.1f} GB free{left}"))
    return report


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_frame_mb(experiment: Experiment, rows: list[dict]) -> float:
    sizes = []
    for row in rows[-5:]:
        path = experiment.root / row["file"]
        if path.exists():
            sizes.append(path.stat().st_size / 1e6)
    return sum(sizes) / len(sizes) if sizes else 0.0


# --- alerts ----------------------------------------------------------------------------


def alert_payload(report: HealthReport, event: str) -> dict:
    return {
        "event": event,  # problem | recovered
        "experiment": report.experiment,
        "root": str(report.root),
        "level": report.level,
        "summary": report.summary(),
        "checked_utc": report.checked_utc,
        "host": platform.node(),
        "checks": [asdict(c) for c in report.checks],
    }


def send_webhook(url: str, payload: dict, timeout: float = 10.0) -> None:
    """POST the alert as JSON (Slack, Discord, ntfy, a home server, ...)."""
    import urllib.request

    data = json.dumps({**payload, "text": payload["summary"]}).encode()
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user URL
        log.info("alert sent to %s (%s)", url, response.status)


def run_command(command: str, payload: dict, timeout: float = 30.0) -> None:
    """Run a command with the alert in FUNGUS_* variables (e.g. to send mail)."""
    env = {**os.environ,
           "FUNGUS_EVENT": payload["event"], "FUNGUS_LEVEL": payload["level"],
           "FUNGUS_EXPERIMENT": payload["experiment"], "FUNGUS_SUMMARY": payload["summary"],
           "FUNGUS_ROOT": payload["root"], "FUNGUS_JSON": json.dumps(payload)}
    subprocess.run(command, shell=True, env=env, timeout=timeout, check=False)  # noqa: S602


def notify(report: HealthReport, event: str, webhook: str | None = None,
           command: str | None = None) -> list[str]:
    """Send one alert; returns what failed (sending must never stop the checks)."""
    payload = alert_payload(report, event)
    errors = []
    for name, action in (("webhook", lambda: send_webhook(webhook, payload) if webhook else None),
                         ("command", lambda: run_command(command, payload) if command else None)):
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - an alert that fails must not stop the watch
            log.error("%s alert failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
    return errors
