"""``fungus`` command-line interface."""

from __future__ import annotations

import logging
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import typer

from fungus_cv import __version__
from fungus_cv.config import parse_duration

app = typer.Typer(
    help="Time-lapse capture and measurement of spreading growth.",
    no_args_is_help=True,
    add_completion=False,
)


def _setup_logging(log_file: Path | None = None, verbose: bool = False) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _require_camera_access() -> None:
    """Stop with clear instructions if the OS won't let this process use a camera."""
    from fungus_cv.capture.permissions import camera_access

    def prompting(app):
        typer.echo(f"macOS is asking for camera permission for {app or 'this app'}; "
                   "click Allow (waiting up to 60 s)...")

    access = camera_access(request=True, on_prompt=prompting)
    if not access.ok:
        typer.secho(access.advice(), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


def _require_gui() -> None:
    """Interactive windows need an OpenCV build with a GUI and a display to show it on."""
    from fungus_cv.capture.diagnostics import gui_problem

    problem = gui_problem()
    if problem:
        typer.secho(problem, fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


def _load_experiment(path: Path):
    from fungus_cv.storage import Experiment

    try:
        return Experiment(path)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    except ValueError as exc:  # pydantic validation errors
        typer.secho(f"invalid config.yaml:\n{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Show the version and exit."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def gui(
    experiment: Path | None = typer.Argument(None, help="Experiment folder to open."),
) -> None:
    """Open the desktop application."""
    try:
        from fungus_cv.gui.app import main as gui_main
    except ImportError as exc:
        typer.secho(f"The desktop app needs PySide6: pip install -e \".[gui]\" ({exc})",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    raise typer.Exit(gui_main(["fungus-cv"] + ([str(experiment)] if experiment else [])))


@app.command()
def doctor(
    probe: bool = typer.Option(True, help="Try opening cameras (needs camera permission)."),
    request_permission: bool = typer.Option(
        False, help="On macOS, ask for camera permission if it hasn't been decided yet."),
) -> None:
    """Check the camera, permissions, display and optional dependencies on this computer."""
    from fungus_cv.capture.diagnostics import run_checks

    quiet = logging.getLogger()
    quiet.setLevel(logging.ERROR)
    failed = 0
    for check in run_checks(probe_cameras=probe, request_permission=request_permission):
        mark = {True: "OK  ", False: "FAIL", None: "info"}[check.ok]
        color = {True: typer.colors.GREEN, False: typer.colors.RED, None: None}[check.ok]
        typer.secho(f"[{mark}] {check.name}: {check.detail}", fg=color)
        if check.fix and check.ok is not True:
            typer.echo(f"       -> {check.fix}")
        failed += check.ok is False
    if failed:
        raise typer.Exit(1)


@app.command()
def init(
    path: Path = typer.Argument(..., help="Experiment folder to create."),
    name: str | None = typer.Option(None, help="Experiment name (default: folder name)."),
) -> None:
    """Create an experiment folder with a starter config.yaml."""
    from fungus_cv.storage import Experiment

    try:
        exp = Experiment.create(path, name)
    except FileExistsError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Created {exp.config_path}")
    typer.echo("Next: `fungus cameras`, adjust config.yaml, check framing with `fungus preview`.")


@app.command()
def cameras(
    max_index: int = typer.Option(8, help="Probe camera indices 0..N-1."),
    backend: str = typer.Option("auto", help="auto | dshow | msmf | avfoundation | v4l2"),
) -> None:
    """List connected cameras."""
    from fungus_cv.capture.camera import list_cameras
    from fungus_cv.capture.permissions import camera_names

    _require_camera_access()
    found = list_cameras(max_index=max_index, backend=backend)
    names = camera_names()
    if not found:
        from fungus_cv.capture.diagnostics import no_camera_hint

        typer.echo("No cameras found.")
        typer.echo(no_camera_hint())
        if names:
            listed = names.values() if isinstance(names, dict) else names
            typer.echo(f"The system reports: {', '.join(listed)}. It may be in use by another "
                       "app, or need a different --backend.")
        raise typer.Exit(1)
    for cam in found:
        label = ""
        if isinstance(names, dict) and cam["index"] in names:
            label = f"  {names[cam['index']]}"
        typer.echo(f"index {cam['index']}: {cam['width']}x{cam['height']} "
                   f"(backend {cam['backend']}){label}")
    if isinstance(names, list) and names:
        typer.echo(f"macOS reports: {', '.join(names)} (index order usually follows this list)")


@app.command()
def preview(
    experiment: Path | None = typer.Argument(
        None, help="Experiment folder; uses its camera settings. Omit to use --index."
    ),
    camera: str | None = typer.Option(None, help="Camera name from config.yaml."),
    index: int = typer.Option(0, help="Camera index when no experiment is given."),
) -> None:
    """Live view for framing and focus. Keys: q quit, s snapshot, g grid, d driver dialog."""
    import cv2

    from fungus_cv.capture.camera import Camera, CameraError
    from fungus_cv.config import CameraConfig
    from fungus_cv.quality import mean_brightness, sharpness

    _setup_logging()
    _require_gui()
    _require_camera_access()
    out_dir = Path.cwd()
    if experiment is not None:
        exp = _load_experiment(experiment)
        cam_cfg = exp.config.camera(camera) if camera else exp.config.cameras[0]
        out_dir = exp.root
    else:
        cam_cfg = CameraConfig(index=index)

    window = f"fungus preview - {cam_cfg.name}"
    show_grid = False
    cam = Camera(cam_cfg)
    try:
        cam.open()
    except CameraError as exc:
        typer.secho(f"{exc}. Check `fungus cameras` for available indices, and that no other "
                    "app is using the camera.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    with cam:
        typer.echo("q: quit   s: save snapshot   g: toggle grid   d: driver settings (Windows)")
        while True:
            frame = cam.read()
            view = frame.copy()
            h, w = view.shape[:2]
            if show_grid:
                for i in (1, 2):
                    cv2.line(view, (w * i // 3, 0), (w * i // 3, h), (0, 255, 255), 1)
                    cv2.line(view, (0, h * i // 3), (w, h * i // 3), (0, 255, 255), 1)
            s = cam.settings()
            lines = [
                f"{w}x{h}  brightness {mean_brightness(frame):.0f}  "
                f"sharpness {sharpness(frame):.0f}",
                f"exposure {s.get('exposure')}  wb {s.get('white_balance')}  "
                f"focus {s.get('focus')}  gain {s.get('gain')}",
            ]
            for i, text in enumerate(lines):
                y = 28 + 28 * i
                cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            scale = min(1.0, 1280 / w)
            if scale < 1:
                view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(window, view)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                show_grid = not show_grid
            elif key == ord("s"):
                path = out_dir / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(str(path), frame)
                typer.echo(f"saved {path}")
            elif key == ord("d") and not cam.open_driver_settings():
                typer.echo("driver settings dialog is only available with the dshow backend")
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    cv2.destroyAllWindows()


@app.command()
def capture(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    interval: str | None = typer.Option(None, help="Override interval, e.g. 10s, 2h."),
    duration: str | None = typer.Option(None, help="Override duration, e.g. 30m, 14d."),
    max_frames: int | None = typer.Option(None, help="Override number of capture rounds."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    check_cameras: bool = typer.Option(True, help="Open every camera once before starting, "
                                       "so a wrong index fails now rather than at each shot."),
) -> None:
    """Capture images on a schedule until stopped (Ctrl+C) or a limit is reached."""
    from fungus_cv.capture.power import keep_awake
    from fungus_cv.capture.scheduler import CaptureSession

    exp = _load_experiment(experiment)
    _setup_logging(exp.log_path, verbose)
    cap = exp.config.capture
    try:
        if interval is not None:
            cap.interval = parse_duration(interval)
        if duration is not None:
            cap.duration = parse_duration(duration)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    if max_frames is not None:
        cap.max_frames = max_frames

    _require_camera_access()
    if check_cameras:
        _check_cameras_open(exp)
    session = CaptureSession(exp)
    with keep_awake(cap.keep_awake):
        try:
            summary = session.run()
        except KeyboardInterrupt:
            session.close()
            summary = session.summary
            logging.getLogger(__name__).info(
                "stopped by user: %d saved, %d failed", summary.saved, summary.failed
            )
    if summary.saved == 0 and summary.failed > 0:
        raise typer.Exit(1)


def _check_cameras_open(exp) -> None:
    from fungus_cv.capture.camera import Camera, CameraError

    problems = []
    for cfg in exp.config.cameras:
        cam = Camera(cfg)
        try:
            cam.open()
            cam.read()
            typer.echo(f"camera {cfg.name}: ok")
        except CameraError as exc:
            problems.append(f"camera {cfg.name}: {exc}")
        finally:
            cam.close()
    if problems:
        typer.secho("\n".join(problems) + "\nFix config.yaml (see `fungus cameras`) or pass "
                    "--no-check-cameras to start anyway.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def snap(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
) -> None:
    """Take one picture from every camera right now."""
    from fungus_cv.capture.scheduler import CaptureSession

    exp = _load_experiment(experiment)
    _setup_logging(exp.log_path)
    _require_camera_access()
    session = CaptureSession(exp)
    try:
        records = session.capture_round()
    finally:
        session.close()
    for rec in records:
        typer.echo(f"{rec.camera}: {rec.status} {rec.file or rec.notes}")
    if any(r.status != "ok" for r in records):
        raise typer.Exit(1)


@app.command("import")
def import_(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    folder: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of images."),
    camera: str = typer.Option("import", help="Name to record for these images."),
    tz: str | None = typer.Option(
        None, help="Time zone of the photos' clock, e.g. America/Chicago (default: local)."
    ),
    recursive: bool = typer.Option(False, "--recursive", "-r"),
    watch: bool = typer.Option(False, help="Keep importing as photos appear (a synced folder "
                               "from a camera in the field)."),
    poll: str = typer.Option("1m", help="How often to look for new photos with --watch."),
    settle: str = typer.Option("10s", help="Ignore files changed this recently: a sync may "
                               "still be writing them (0 to import immediately)."),
) -> None:
    """Import existing photos, ordered by capture time (EXIF, file name, or file date)."""
    from fungus_cv.capture.importer import import_folder, watch_folder

    exp = _load_experiment(experiment)
    _setup_logging()
    settle_seconds = 0.0 if settle.strip().rstrip("s") in ("", "0") else parse_duration(settle)
    if watch:
        typer.echo(f"Watching {folder} every {poll}; Ctrl+C to stop.")

        def report(records) -> None:
            typer.echo(f"imported {len(records)} new photo(s); last "
                       f"{records[-1].file.split('/')[-1]}")

        try:
            total = watch_folder(exp.root, folder, camera=camera, timezone_name=tz,
                                 recursive=recursive, poll_seconds=parse_duration(poll),
                                 settle_seconds=settle_seconds, on_batch=report)
        except KeyboardInterrupt:
            total = 0
        typer.echo(f"Stopped after importing {total} photo(s) this session.")
        return
    records = import_folder(exp, folder, camera=camera, timezone_name=tz, recursive=recursive,
                            settle_seconds=settle_seconds)
    sources = Counter(r.source for r in records)
    typer.echo(f"Imported {len(records)} image(s): {dict(sources)}")


@app.command()
def status(experiment: Path = typer.Argument(..., help="Experiment folder.")) -> None:
    """Summarize what has been captured so far."""
    from fungus_cv.storage import parse_iso_utc

    exp = _load_experiment(experiment)
    rows = exp.read_frames()
    typer.echo(f"Experiment: {exp.config.name}  ({exp.root})")
    if not rows:
        typer.echo("No frames yet.")
        return
    by_camera: dict[str, list[dict]] = {}
    for row in rows:
        by_camera.setdefault(row["camera"], []).append(row)
    for name, cam_rows in sorted(by_camera.items()):
        ok = [r for r in cam_rows if r["status"] == "ok"]
        failed = len(cam_rows) - len(ok)
        typer.echo(f"\n{name}: {len(ok)} saved, {failed} failed")
        if ok:
            times = sorted(parse_iso_utc(r["timestamp_utc"]) for r in ok)
            typer.echo(f"  first: {times[0].astimezone():%Y-%m-%d %H:%M:%S %Z}")
            typer.echo(f"  last:  {times[-1].astimezone():%Y-%m-%d %H:%M:%S %Z}")
            if len(times) > 1:
                gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
                gaps.sort()
                typer.echo(
                    f"  spacing: median {gaps[len(gaps) // 2]:.1f}s, largest {gaps[-1]:.1f}s"
                )
    size_mb = sum(p.stat().st_size for p in exp.frames_dir.iterdir() if p.is_file()) / 1e6
    typer.echo(f"\nDisk: {size_mb:.1f} MB used, {exp.free_disk_mb() / 1e3:.1f} GB free")



# --- analysis -------------------------------------------------------------------------


@app.command()
def markers(
    output: Path = typer.Argument(Path("markers.png"), help="Image file to write (PNG or PDF)."),
    ids: str = typer.Option("0,1,2,3", help="Comma-separated marker ids."),
    size_mm: float = typer.Option(30.0, help="Marker edge length in mm."),
    dictionary: str = typer.Option("DICT_4X4_50"),
    page: str = typer.Option("letter", help="letter | a4"),
    dpi: int = typer.Option(300),
) -> None:
    """Make a printable sheet of ArUco markers for scale and alignment."""
    import cv2

    from fungus_cv.preprocess.markers import marker_sheet

    try:
        sheet = marker_sheet([int(i) for i in ids.split(",")], size_mm, dictionary, page, dpi)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    if output.suffix.lower() == ".pdf":
        from PIL import Image

        Image.fromarray(sheet).save(output, resolution=dpi)
    else:
        cv2.imwrite(str(output), sheet)
    typer.echo(f"Wrote {output}")
    typer.echo(
        "Print at 100% / actual size, check the 100 mm ruler, measure a marker's black square "
        "and put that value in analysis.markers.size_mm. Place markers in the same plane as "
        "the object, and keep them in view for the whole experiment."
    )


@app.command()
def annotate(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    field: bool = typer.Option(False, "--field", help="Outline field plots instead of a "
                               "base/tip/region (plots are named plot1, plot2, ...)."),
    camera: str | None = typer.Option(None, help="Which camera's view to annotate (each "
                                      "camera has its own annotations)."),
) -> None:
    """Click the base, tip and region to measure on the reference (first) frame."""
    _require_gui()
    from fungus_cv.analyze.pipeline import AnalysisError, Analyzer, annotations_path
    from fungus_cv.ui.interactive import Cancelled, annotate_field
    from fungus_cv.ui.interactive import annotate as run_annotate

    exp = _load_experiment(experiment)
    _setup_logging()
    try:  # show the reference as the pipeline prepares it (e.g. rectified)
        analyzer = Analyzer(exp, with_segmenter=False, require_annotations=False,
                            camera=camera)
    except AnalysisError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    row, image = analyzer.reference_row, analyzer.reference
    try:
        tool = annotate_field if field else run_annotate
        ann = tool(image, reference_file=row["file"])
    except Cancelled:
        typer.echo("Cancelled; nothing saved.")
        raise typer.Exit(1) from None
    path = annotations_path(exp, analyzer.camera)
    ann.save(path)
    if ann.plots:
        typer.echo(f"Saved {path} with plots {[p.name for p in ann.plots]}"
                   f"{', with neutral patch' if ann.reference_patch else ''}. Rename plots "
                   "by editing the names in the file.")
        return
    kind = f"path through {len(ann.path)} points" if ann.path else "straight axis"
    typer.echo(f"Saved {path} ({kind}, {ann.polyline().length:.1f} px"
               f"{', with neutral patch' if ann.reference_patch else ''})")


@app.command("pick-color")
def pick_color(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    frame: Path | None = typer.Option(
        None, help="Image to sample from (default: the last frame, where the target is largest)."
    ),
    write: bool = typer.Option(True, help="Write the ranges into config.yaml."),
) -> None:
    """Measure the target's color by dragging boxes over it, then save the HSV ranges."""
    _require_gui()
    import cv2

    from fungus_cv.config import replace_hsv_ranges_in_yaml
    from fungus_cv.ui.interactive import Cancelled
    from fungus_cv.ui.interactive import pick_color as run_pick

    exp = _load_experiment(experiment)
    if frame is not None:
        image = cv2.imread(str(frame))
        if image is None:
            typer.secho(f"cannot read {frame}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
    else:
        from fungus_cv.analyze.pipeline import AnalysisError, Analyzer

        try:  # sample colours as the segmenter will see them (rectified, lighting-corrected)
            analyzer = Analyzer(exp, with_segmenter=False, require_annotations=False)
            image = analyzer.aligned_frame(len(analyzer.frames) - 1)
        except AnalysisError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc
    try:
        ranges = run_pick(image)
    except Cancelled:
        typer.echo("Cancelled; nothing saved.")
        raise typer.Exit(1) from None

    snippet = "\n".join(f"  - lower: {list(lo)}\n    upper: {list(hi)}" for lo, hi in ranges)
    if write:
        text = exp.config_path.read_text(encoding="utf-8")
        exp.config_path.write_text(replace_hsv_ranges_in_yaml(text, ranges), encoding="utf-8")
        typer.echo(f"Updated hsv_ranges in {exp.config_path}:\n{snippet}")
    else:
        typer.echo(f"hsv_ranges:\n{snippet}")


@app.command()
def analyze(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    force: bool = typer.Option(False, help="Re-measure every frame."),
    watch: bool = typer.Option(False, help="Keep running and measure new frames as they arrive."),
    poll: str = typer.Option("30s", help="How often to check for new frames with --watch."),
    camera: str | None = typer.Option(None, help="Which camera to measure, or 'all'. With "
                                      "several cameras each one gets results/cameras/<name>/."),
) -> None:
    """Align, segment and measure every frame; results go to results/measurements.csv."""
    from fungus_cv.analyze.pipeline import AnalysisError
    from fungus_cv.analyze.pipeline import analyze as run_analyze
    from fungus_cv.analyze.pipeline import watch as run_watch

    exp = _load_experiment(experiment)
    _setup_logging()

    def show(summary) -> None:
        flagged = ", ".join(f"{k}: {v}" for k, v in summary.flagged.items()) or "none"
        typer.echo(
            f"measured {summary.processed}, already done {summary.skipped_existing}, "
            f"failed {summary.failed}; flags: {flagged} (settings {summary.settings_hash})"
        )

    if watch:
        if camera == "all":
            typer.secho("--watch measures one camera; give its name", fg=typer.colors.RED,
                        err=True)
            raise typer.Exit(1)
        typer.echo("Watching for new frames; Ctrl+C to stop.")
        try:
            run_watch(exp.root, parse_duration(poll), on_summary=show, camera=camera)
        except KeyboardInterrupt:
            pass
        return
    if camera == "all":
        from fungus_cv.analyze.pipeline import analyze_all_cameras

        results = analyze_all_cameras(
            exp, force=force, progress=lambda name: typer.echo(f"camera {name}:"))
        if not results:
            typer.secho("no frames yet", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        for name, outcome in results.items():
            if isinstance(outcome, str):
                typer.secho(f"  {name}: {outcome}", fg=typer.colors.RED)
            else:
                typer.echo(f"  {name}: ", nl=False)
                show(outcome)
        if all(isinstance(o, str) for o in results.values()):
            raise typer.Exit(1)
        return
    try:
        show(run_analyze(exp, force=force, camera=camera))
    except AnalysisError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


@app.command()
def report(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    metric: str | None = typer.Option(
        None, help="Column to plot: extent_mm (default), extent_px, coverage_pct, ..."
    ),
    t0: datetime | None = typer.Option(
        None, help="Start time for t = 0, e.g. when the towel touched the dye (local time)."
    ),
    time_unit: str = typer.Option("auto", help="auto | s | min | h | d"),
    include_flagged: bool = typer.Option(False, help="Fit flagged frames too."),
    video: bool = typer.Option(False, help="Also write an overlay time-lapse video."),
    exclude_jumps: bool = typer.Option(False, help="Leave frames that jump off the local "
                                       "trend out of the fits (they are always marked)."),
    plot: list[str] = typer.Option([], help="Plot(s) to report (repeatable). Default: all."),
    camera: str | None = typer.Option(None, help="Which camera's results to report."),
    model: list[str] = typer.Option([], help="Model(s) to fit (repeatable): linear, sqrt, "
                                    "sqrt_lag, power, logistic, gompertz, richards. Default: "
                                    "linear, sqrt, power, logistic."),
    bootstrap: int = typer.Option(1000, help="Block-bootstrap refits for 95% intervals "
                                  "(0 = off)."),
    errors: str = typer.Option("auto", help="Frame errors: auto (choose by AICc) | iid "
                               "(independent) | ar1 (correlated between neighbouring frames)."),
    time_to: list[float] = typer.Option([], help="Also report when the fitted curve reaches "
                                        "this value (repeatable), e.g. --time-to 20."),
    figure_format: list[str] = typer.Option([], "--format", help="Figure format(s): png "
                                            "(default), pdf, svg. Repeatable; pdf and svg are "
                                            "vector, for papers."),
    dpi: int = typer.Option(150, help="Resolution of the png figures."),
) -> None:
    """Fit growth models and write plots to results/report/ (one folder per field plot)."""
    from fungus_cv.analyze.fit import (
        DEFAULT_MODELS,
        MODELS,
        best_fit,
        format_derived,
        format_params,
    )
    from fungus_cv.analyze.report import (
        DEFAULT_EXCLUDE,
        DEFAULT_FORMATS,
        FIGURE_FORMATS,
        list_plots,
        make_report,
    )

    exp = _load_experiment(experiment)
    _setup_logging()
    if t0 is not None and t0.tzinfo is None:
        t0 = t0.astimezone()
    if errors not in ("auto", "iid", "ar1"):
        typer.secho("--errors must be auto, iid or ar1", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    bad_formats = [f for f in figure_format if f not in FIGURE_FORMATS]
    if bad_formats:
        typer.secho(f"unknown figure format(s) {bad_formats}; use {list(FIGURE_FORMATS)}",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    unknown = [m for m in model if m not in MODELS]
    if unknown:
        typer.secho(f"unknown model(s) {unknown}; choose from {list(MODELS)}",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    from fungus_cv.analyze.pipeline import results_dir_for

    results_dir = results_dir_for(exp, camera)
    try:
        plots = list(plot) or list_plots(exp, results_dir)
        results = [make_report(
            exp, metric=metric, t0=t0, time_unit=time_unit, results_dir=results_dir,
            exclude_flags=() if include_flagged else DEFAULT_EXCLUDE, video=video,
            exclude_jumps=exclude_jumps, plot=name, models=tuple(model) or DEFAULT_MODELS,
            formats=tuple(figure_format) or DEFAULT_FORMATS, dpi=dpi, time_to=tuple(time_to),
            bootstrap=bootstrap, errors=errors,
        ) for name in plots]
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    for result in results:
        where = "" if result.plot == "main" else f"[{result.plot}] "
        typer.echo(f"{where}{result.metric}: {result.n_used} frames used, "
                   f"{result.n_excluded} excluded, {result.retreats} retreat(s) and "
                   f"{result.jumps} jump(s) to check (see frame_flags.csv); "
                   f"time in {result.time_unit}")
        best_model = best_fit(result.fits)
        if result.fits and result.fits[0].bootstrap:
            b = result.fits[0].bootstrap
            typer.echo(f"  parameters: estimate ± SE [95% bootstrap interval, {b['n']} refits]")
        for fit in result.fits:
            if not fit.ok:
                typer.echo(f"  {fit.model:9s} failed: {fit.message}")
                continue
            best = "  <- best (lowest AICc)" if fit is best_model else ""
            chi2 = f"  χ²/dof={fit.reduced_chi2:.2f}" if fit.weighted else ""
            chi2 += f"  AR(1) φ={fit.ar1_phi:.2f}" if fit.error_model == "ar1" else ""
            typer.echo(f"  {fit.model:9s} {format_params(fit)}")
            derived = format_derived(fit)
            if derived:
                typer.echo(f"  {'':9s} {derived}")
            typer.echo(f"  {'':9s} R²={fit.r2:.4f}  AICc={fit.aicc:.1f}  "
                       f"weight={fit.akaike_weight:.2f}{chi2}  DW={fit.durbin_watson:.2f}{best}")
            for warning in fit.warnings:
                typer.echo(f"  {'':9s} warning: {warning}")
        for f in result.files:
            typer.echo(f"wrote {f}")


def _pick_frame_index(frames: list[dict], frame: str) -> int:
    if frame in ("last", "first"):
        return len(frames) - 1 if frame == "last" else 0
    if frame.lstrip("-").isdigit():
        return int(frame) % len(frames)
    for i, row in enumerate(frames):
        if Path(row["file"]).name == Path(frame).name:
            return i
    raise typer.BadParameter(f"frame {frame!r} not found (use first, last, an index or a file)")


@app.command()
def exclude(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    frame: str | None = typer.Argument(None, help="Frame file name (with or without extension)."),
    reason: str = typer.Option("", help="Why, e.g. 'hand in front of the camera'."),
    plot: str = typer.Option("", help="Only for this plot (default: all plots)."),
    remove: bool = typer.Option(False, "--remove", help="Include the frame again."),
) -> None:
    """Leave a frame out of fits by hand, with a reason. Without FRAME, list exclusions."""
    from fungus_cv.analyze import exclusions

    exp = _load_experiment(experiment)
    try:
        if frame is None:
            items = exclusions.load(exp)
            if not items:
                typer.echo("No frames excluded by hand.")
            for e in items:
                where = f" [{e.plot}]" if e.plot else ""
                typer.echo(f"{Path(e.frame_file).name}{where}: {e.reason}  ({e.created_utc})")
            return
        if remove:
            n = exclusions.include(exp, frame, plot or None)
            typer.echo(f"Removed {n} exclusion(s) for {frame}.")
            return
        item = exclusions.exclude(exp, frame, reason, plot)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Excluded {Path(item.frame_file).name}" + (f" [{plot}]" if plot else "")
               + f": {item.reason}. Reports and studies leave it out of fits.")


@app.command()
def prompt(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    frame: str = typer.Option(
        "last", help="Frame to click on: last (default), first, an index, or a file name. "
        "Pick one where the target is clearly visible."
    ),
    preview: bool = typer.Option(True, help="Show the SAM mask live while clicking."),
    reference: bool = typer.Option(False, "--reference",
                                   help="Prompt the reference object (e.g. stem) instead."),
    camera: str | None = typer.Option(None, help="Which camera's frames to prompt."),
) -> None:
    """Click on the target for SAM 2 (left = target, right = not target)."""
    _require_gui()
    from fungus_cv.analyze.pipeline import AnalysisError, Analyzer, prompts_path
    from fungus_cv.segment.prompts import Prompts
    from fungus_cv.segment.sam2 import Sam2VideoSegmenter, _RoiCrop
    from fungus_cv.ui.interactive import Cancelled, prompt_target

    exp = _load_experiment(experiment)
    _setup_logging()
    try:
        analyzer = Analyzer(exp, with_segmenter=False, camera=camera)
    except AnalysisError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    index = _pick_frame_index(analyzer.frames, frame)
    frame_file = analyzer.frames[index]["file"]
    image = analyzer.aligned_frame(index)

    block = exp.config.analysis.reference if reference else exp.config.analysis.target
    cfg = block.sam2
    path = prompts_path(exp, cfg.prompts_file, analyzer.camera)
    prompts = Prompts.load(path) if path.exists() else Prompts()
    preview_fn = None
    if preview:
        seg = Sam2VideoSegmenter(
            model_name=cfg.model, prompts=Prompts(), device=cfg.device,
            crop=_RoiCrop(analyzer.annotations.roi, cfg.crop_margin_px) if cfg.crop_to_roi
            else None,
            mask_threshold=cfg.mask_threshold, min_blob_area_px=cfg.min_blob_area_px,
        )
        preview_fn = seg.segment_single
    try:
        result = prompt_target(image, frame_file, preview_fn, prompts.for_file(frame_file))
    except Cancelled:
        typer.echo("Cancelled; nothing saved.")
        raise typer.Exit(1) from None
    prompts.set(result)
    prompts.save(path)
    typer.echo(f"Saved prompt for {frame_file} to {path} ({len(prompts.frames)} prompted frame(s))")
    if block.method != "sam2":
        which = "reference" if reference else "target"
        typer.echo(f"Note: set analysis.{which}.method: sam2 in config.yaml to use these prompts.")


@app.command()
def runs(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    camera: str | None = typer.Option(None, help="Which camera's runs to list."),
) -> None:
    """List analysis runs whose masks are on disk (for `fungus compare`)."""
    from fungus_cv.analyze.compare import list_runs

    exp = _load_experiment(experiment)
    found = list_runs(exp, camera)
    if not found:
        typer.echo("No runs yet; run `fungus analyze`.")
        return
    for r in found:
        typer.echo(f"{r.run_id}  {r.method:6s}  {r.n_masks:5d} masks  {r.updated_utc}")


@app.command()
def compare(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    run_a: str | None = typer.Argument(None, help="Run id/prefix or folder of masks. "
                                       "Default: second newest run."),
    run_b: str | None = typer.Argument(None, help="Default: newest run."),
    camera: str | None = typer.Option(None, help="Which camera's runs to compare."),
) -> None:
    """Compare two runs' masks per frame: IoU and extent difference."""
    from fungus_cv.analyze.compare import compare_runs, list_runs

    exp = _load_experiment(experiment)
    if run_a is None or run_b is None:
        found = list_runs(exp, camera)
        if len(found) < 2:
            typer.secho("need two runs; analyze with two different settings first",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        run_a = run_a or found[-2].run_id
        run_b = run_b or found[-1].run_id
    try:
        r = compare_runs(exp, run_a, run_b, camera)
    except (ValueError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{r.run_a} vs {r.run_b}: {r.n_frames} frames")
    typer.echo(f"  IoU  mean {r.iou_mean:.4f}  median {r.iou_median:.4f}  "
               f"min {r.iou_min:.4f} ({r.worst_frame})")
    typer.echo(f"  extent difference (b - a): mean {r.extent_diff_mean:+.3f} "
               f"sd {r.extent_diff_sd:.3f} {r.extent_unit}")
    typer.echo(f"wrote {r.csv_path}")


@app.command("validate")
def validate_cmd(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    hand_csv: Path = typer.Argument(..., help="CSV with columns frame and value (see "
                                    "--make-template)."),
    metric: str = typer.Option("extent_mm", help="Measurement column to compare."),
    plot: str | None = typer.Option(None, help="Only this plot (field experiments)."),
    make_template: int = typer.Option(0, help="Instead of validating, write HAND_CSV with this "
                                      "many evenly spaced frames to measure by hand."),
    camera: str | None = typer.Option(None, help="Which camera's results to check."),
) -> None:
    """Compare automatic measurements with hand measurements (Bland-Altman agreement)."""
    from fungus_cv.analyze.pipeline import results_dir_for
    from fungus_cv.analyze.validate import validate, write_template

    exp = _load_experiment(experiment)
    results_dir = results_dir_for(exp, camera)
    try:
        if make_template:
            if hand_csv.exists():
                raise ValueError(f"{hand_csv} already exists")
            n = write_template(exp, hand_csv, make_template, results_dir)
            typer.echo(f"Wrote {hand_csv} with {n} frames. Measure each frame by hand (same "
                       f"units as {metric}), fill in 'value', then run this command again "
                       "without --make-template.")
            return
        a = validate(exp, hand_csv, metric, plot, results_dir)
    except (ValueError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{a.metric}: n = {a.n} frames")
    typer.echo(f"  bias (automatic - hand) {a.bias:+.4f}  "
               f"(95% CI {a.bias_ci95[0]:+.4f} to {a.bias_ci95[1]:+.4f})")
    typer.echo(f"  95% limits of agreement {a.limits_of_agreement[0]:+.4f} to "
               f"{a.limits_of_agreement[1]:+.4f}  (sd of differences {a.sd_diff:.4f})")
    typer.echo(f"  MAE {a.mae:.4f}  RMSE {a.rmse:.4f}  r {a.pearson_r:.4f}  "
               f"automatic = {a.slope:.4f} x hand {a.intercept:+.4f}")
    typer.echo(f"  largest difference {a.worst_diff:+.4f} at {a.worst_frame}")
    if not math.isnan(a.within_2u):
        typer.echo(f"  uncertainty check: {100 * a.within_2u:.0f}% of differences within 2u "
                   f"(expect ~95%), RMS z {a.z_rms:.2f} (expect ~1)")
    if a.unmatched:
        typer.echo(f"  {len(a.unmatched)} hand row(s) not matched to analyzed frames: "
                   f"{', '.join(a.unmatched[:5])}")
    for f in a.files:
        typer.echo(f"wrote {f}")


@app.command()
def study(
    study_file: Path = typer.Argument(..., help="Study YAML: experiments with their condition "
                                      "and replicate, the metric and the model."),
    init: bool = typer.Option(False, "--init", help="Write an example study file and exit."),
    out: Path | None = typer.Option(None, help="Output folder. Default: <study>_results next "
                                    "to the study file."),
) -> None:
    """Fit every replicate, summarise conditions and compare them (Welch t, Holm-adjusted)."""
    from fungus_cv.analyze import study as study_mod
    from fungus_cv.analyze.fit import format_params

    if init:
        try:
            study_mod.write_template(study_file)
        except FileExistsError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"Wrote {study_file}. List your experiments and conditions, then run this "
                   "command without --init.")
        return
    _setup_logging()
    try:
        result = study_mod.run_study(study_file, out)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    s = result.study
    typer.echo(f"{s.name}: {result.metric} with {s.model}, time in {result.time_unit}")
    typer.echo("replicates:")
    for r in result.replicates:
        if r.fit is None:
            typer.echo(f"  {r.condition}/{r.replicate}: not used ({r.error or 'fit failed'})")
            continue
        typer.echo(f"  {r.condition}/{r.replicate}: {format_params(r.fit)}  "
                   f"R²={r.fit.r2:.3f}  n={r.fit.n}")
    for param in s.compared_params:
        typer.echo(f"{param}:")
        for row in (x for x in result.conditions if x["param"] == param):
            if row["n"] >= 2:
                typer.echo(f"  {row['condition']:14s} {row['mean']:.4g} ± {row['sd']:.2g} SD "
                           f"(95% CI {row['ci_low']:.4g} to {row['ci_high']:.4g}, "
                           f"n={row['n']})")
            else:
                typer.echo(f"  {row['condition']:14s} {row['mean']:.4g} (n={row['n']})")
        for row in (x for x in result.comparisons if x["param"] == param):
            if math.isnan(row["p"]):
                typer.echo(f"  {row['condition']} − {row['versus']}: {row['diff']:+.4g} "
                           "(need 2+ replicates in each for a test)")
                continue
            typer.echo(f"  {row['condition']} − {row['versus']}: {row['diff']:+.4g} "
                       f"(95% CI {row['ci_low']:+.4g} to {row['ci_high']:+.4g}), "
                       f"p={row['p']:.3g}, Holm p={row['p_holm']:.3g}, g={row['hedges_g']:.2f}")
    totals = [r for r in result.model_selection if r["condition"] == "(all)"]
    if len(totals) > 1:
        typer.echo("model selection over " + totals[0]["replicate"] + ": " + ", ".join(
            f"{r['model']} {r['akaike_weight']:.2f}" for r in totals))
    for w in result.warnings:
        typer.secho(f"warning: {w}", fg=typer.colors.YELLOW)
    typer.echo(f"wrote {result.out_dir} (methods.md has a draft methods paragraph)")


@app.command()
def combine(
    experiment: Path = typer.Argument(..., help="Experiment folder with several cameras."),
    metric: str = typer.Option("extent_mm", help="Measurement to combine."),
    camera: list[str] = typer.Option([], help="Cameras to use (repeatable). Default: all with "
                                     "frames."),
    method: str = typer.Option("max", help="max (each view shortens a length, so the largest "
                               "is closest to the truth) | mean | median."),
    plot: str | None = typer.Option(None, help="Plot to combine (field experiments)."),
    tolerance: str = typer.Option("60s", help="How far apart frames of different cameras may "
                                  "be and still count as the same moment."),
) -> None:
    """Combine several cameras' views of the same object into one measurement per moment."""
    from fungus_cv.analyze.combine import combine as run_combine

    exp = _load_experiment(experiment)
    _setup_logging()
    try:
        result = run_combine(exp, metric=metric, cameras=list(camera) or None, method=method,
                             plot=plot, tolerance_s=parse_duration(tolerance))
    except (ValueError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{metric} from {', '.join(result.cameras)} using '{method}': "
               f"{len(result.rows)} moment(s)"
               + (f", {result.unmatched} without every view" if result.unmatched else ""))
    if result.rows:
        spread = result.mean_spread
        if spread == spread:
            typer.echo(f"  views disagree by {spread:.4g} on average (largest minus smallest); "
                       "a big spread means the object is far from perpendicular to a camera")
        counts = Counter(r.best_camera for r in result.rows)
        typer.echo("  largest view came from: "
                   + ", ".join(f"{c} {n}x" for c, n in counts.most_common()))
    if result.note:
        typer.secho(f"note: {result.note}", fg=typer.colors.YELLOW)
    typer.echo(f"wrote {result.out_path}")


@app.command()
def sensitivity(
    experiment: Path = typer.Argument(..., help="Experiment folder (already analyzed)."),
    metric: str | None = typer.Option(None, help="Measurement to compare (default: as in "
                                      "`fungus report`)."),
    plot: str | None = typer.Option(None, help="Plot to use (field experiments)."),
    variant: list[str] = typer.Option([], help="Only these variants (repeatable); see --list."),
    list_variants: bool = typer.Option(False, "--list", help="Show the variants and exit."),
    keep_masks: bool = typer.Option(False, help="Keep each variant's masks and overlays."),
    out: Path | None = typer.Option(None, help="Output folder (default: "
                                    "results/sensitivity)."),
) -> None:
    """Re-run the analysis with key settings changed and report how much the results move."""
    from fungus_cv.analyze import sensitivity as sens

    exp = _load_experiment(experiment)
    _setup_logging()
    available = sens.default_variants(exp)
    if list_variants:
        for v in available:
            typer.echo(f"{v.name:26s} {v.description}")
        return
    chosen = available
    if variant:
        by_name = {v.name: v for v in available}
        unknown = [v for v in variant if v not in by_name]
        if unknown:
            typer.secho(f"unknown variant(s) {unknown}; see --list", fg=typer.colors.RED,
                        err=True)
            raise typer.Exit(1)
        chosen = [by_name[v] for v in variant]
    if not chosen:
        typer.secho("nothing to vary for these settings", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.echo(f"Re-analyzing {len(chosen)} variant(s); each one measures every frame again.")
    try:
        result = sens.run(exp, metric=metric, plot=plot, variants=chosen, out_dir=out,
                          keep_masks=keep_masks,
                          progress=lambda n, total, name: typer.echo(f"  [{n}/{total}] {name}"))
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    unc = result.baseline_uncertainty
    typer.echo(f"{result.metric} (plot {result.plot}): last value {result.baseline_final:.4g}"
               + (f", mean reported uncertainty {unc:.4g}" if unc == unc else ""))
    for v in sorted(result.variants, key=lambda v: -(v.mean_abs_change if v.ok and
                                                     v.mean_abs_change == v.mean_abs_change
                                                     else -1)):
        if not v.ok:
            typer.secho(f"  {v.variant.name:26s} failed: {v.error}", fg=typer.colors.YELLOW)
            continue
        in_u = (f"  = {v.change_in_uncertainties:.2f} u"
                if v.change_in_uncertainties == v.change_in_uncertainties else "")
        typer.echo(f"  {v.variant.name:26s} mean |change| {v.mean_abs_change:.4g}{in_u}"
                   f"  (max {v.max_abs_change:.4g}, at the last frame "
                   f"{v.final_change:+.4g})  — {v.variant.description}")
    worst = result.largest
    if worst is not None:
        typer.echo(f"Largest effect: {worst.variant.description}.")
        if worst.change_in_uncertainties == worst.change_in_uncertainties:
            verdict = ("smaller than the reported uncertainty, so the conclusions do not "
                       "depend on it" if worst.change_in_uncertainties < 1 else
                       "larger than the reported uncertainty: justify this setting in the "
                       "methods, or report the range")
            typer.echo(f"  That is {verdict}.")
    for f in result.files:
        typer.echo(f"wrote {f}")


service_app = typer.Typer(help="Run capture (or health checks) automatically at login.",
                          no_args_is_help=True)
app.add_typer(service_app, name="service")


@service_app.command("install")
def service_install(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    action: str = typer.Option("capture", help="capture | health (a health watch)."),
    extra: str = typer.Option("", help="Extra arguments for the command, e.g. "
                              "'--webhook https://...' for health."),
    register: bool = typer.Option(True, help="Also register it with the system."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be written."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
) -> None:
    """Install a login service so capture starts by itself (and restarts if it stops)."""
    import shlex

    from fungus_cv.capture import service as service_mod

    exp = _load_experiment(experiment)
    if action not in ("capture", "health"):
        typer.secho("action must be capture or health", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    arguments = shlex.split(extra)
    try:
        plan = service_mod.plan(exp.root, action, arguments)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Service: {plan.name}")
    typer.echo(f"File:    {plan.path}")
    typer.echo("Runs:    " + " ".join(service_mod.fungus_command(exp.root, action, arguments)))
    for command in plan.commands:
        typer.echo("Then:    " + " ".join(command))
    if plan.note:
        typer.echo(f"Note:    {plan.note}")
    if dry_run:
        typer.echo("\n--- file contents ---")
        typer.echo(plan.contents)
        return
    if not yes and not typer.confirm("Install this service on this computer?", default=False):
        typer.echo("Nothing installed.")
        raise typer.Exit(1)
    result = service_mod.install(exp.root, action, arguments, register=register)
    for line in result.output:
        typer.echo(line)
    for message in result.errors:
        typer.secho(message, fg=typer.colors.RED, err=True)
    typer.echo(f"Wrote {plan.path}" + (" and registered it." if result.registered
                                       else " (not registered)."))
    typer.echo(f"Remove it with `fungus service uninstall {exp.root}`"
               + (f" --action {action}" if action != "capture" else "") + ".")
    if result.errors:
        raise typer.Exit(1)


@service_app.command("uninstall")
def service_uninstall(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    action: str = typer.Option("capture", help="capture | health."),
) -> None:
    """Remove the login service for this experiment."""
    from fungus_cv.capture import service as service_mod

    exp = _load_experiment(experiment)
    result = service_mod.uninstall(exp.root, action)
    for line in result.output:
        typer.echo(line)
    typer.echo(f"Removed {result.plan.path}" if result.wrote
               else f"No service file at {result.plan.path}")


@service_app.command("status")
def service_status(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
) -> None:
    """Is a login service installed for this experiment?"""
    from fungus_cv.capture import service as service_mod

    exp = _load_experiment(experiment)
    for action in ("capture", "health"):
        try:
            plan = service_mod.plan(exp.root, action)
        except ValueError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc
        state = "installed" if plan.path.exists() else "not installed"
        typer.echo(f"{action:8s} {state}: {plan.path}")


@app.command()
def health(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    watch: bool = typer.Option(False, help="Keep checking and alert when something changes."),
    every: str = typer.Option("10m", help="How often to check with --watch."),
    webhook: str | None = typer.Option(None, help="POST alerts as JSON to this URL (Slack, "
                                       "ntfy, your own server)."),
    on_alert: str | None = typer.Option(None, help="Run this command on an alert; the details "
                                        "are in FUNGUS_SUMMARY, FUNGUS_LEVEL, FUNGUS_JSON..."),
    quality_frames: int = typer.Option(10, help="How many recent photos to judge quality on."),
) -> None:
    """Check an unattended run: is capture alive, are photos arriving and usable, is there room?"""
    import time as time_module

    from fungus_cv.capture import health as health_mod

    exp = _load_experiment(experiment)
    _setup_logging()
    colours = {"ok": typer.colors.GREEN, "warning": typer.colors.YELLOW,
               "problem": typer.colors.RED}

    def show(report) -> None:
        for c in report.checks:
            mark = {"ok": "OK  ", "warning": "WARN", "problem": "FAIL"}[c.level]
            typer.secho(f"[{mark}] {c.name}: {c.detail}", fg=colours[c.level])
            if c.fix and c.level != "ok":
                typer.echo(f"       -> {c.fix}")

    if not watch:
        report = health_mod.check(exp, quality_frames=quality_frames)
        show(report)
        typer.secho(report.summary(), fg=colours[report.level])
        if not report.ok:
            raise typer.Exit(1 if report.level == "problem" else 0)
        return

    seconds = parse_duration(every)
    typer.echo(f"Checking {exp.config.name} every {every}; Ctrl+C to stop."
               + (" Alerts go to the webhook." if webhook else "")
               + (" On alert: " + on_alert if on_alert else ""))
    previous = None
    try:
        while True:
            report = health_mod.check(_reload(exp), quality_frames=quality_frames)
            state = report.level
            if state != previous:  # only tell on a change, not every time
                show(report)
                if state != "ok":
                    errors = health_mod.notify(report, "problem", webhook, on_alert)
                elif previous is not None:
                    errors = health_mod.notify(report, "recovered", webhook, on_alert)
                else:
                    errors = []
                for message in errors:
                    typer.secho(f"alert failed: {message}", fg=typer.colors.RED, err=True)
                previous = state
            time_module.sleep(seconds)
    except KeyboardInterrupt:
        typer.echo("Stopped.")


def _reload(exp):
    """Re-read config.yaml and frames.csv, so a long watch sees changes."""
    from fungus_cv.storage import Experiment

    return Experiment(exp.root)


@app.command()
def archive(
    target: Path = typer.Argument(None, help="Experiment folder, or a study YAML file."),
    out: Path = typer.Argument(None, help="Bundle to write, e.g. dye-test-1.zip."),
    frames: bool = typer.Option(False, help="Include the photos (much larger; their hashes "
                                "are recorded either way)."),
    masks: bool = typer.Option(False, help="Include masks and overlays (analyze recreates "
                               "them)."),
    weights: bool = typer.Option(False, help="Include trained model weights."),
    verify: Path | None = typer.Option(None, help="Instead: check a bundle against its "
                                       "manifest."),
) -> None:
    """Pack everything needed to reproduce a result into one zip (for a data repository)."""
    from fungus_cv.analyze import archive as archive_mod

    _setup_logging()
    if verify is not None:
        try:
            check = archive_mod.verify(verify)
        except (OSError, ValueError, KeyError) as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc
        env = check.manifest.get("environment", {})
        typer.echo(f"{verify}: {check.manifest.get('kind', '?')} bundle, {check.files} file(s), "
                   f"made {check.manifest.get('created_utc', '?')} with fungus-cv "
                   f"{env.get('fungus_cv_version', '?')}"
                   + (f" (commit {env['git_commit'][:10]})" if env.get("git_commit") else ""))
        for name, items in (("changed", check.changed), ("missing", check.missing)):
            for item in items[:10]:
                typer.secho(f"  {name}: {item}", fg=typer.colors.RED)
        if check.extra:
            typer.echo(f"  {len(check.extra)} file(s) not in the manifest")
        typer.secho("Bundle is intact." if check.ok else "Bundle does NOT match its manifest.",
                    fg=typer.colors.GREEN if check.ok else typer.colors.RED)
        if not check.ok:
            raise typer.Exit(1)
        return
    if target is None or out is None:
        typer.secho("give an experiment folder (or study file) and the bundle to write, or "
                    "--verify BUNDLE", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    try:
        result = archive_mod.build(target, out, frames=frames, masks=masks, weights=weights)
    except (OSError, ValueError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    size = result.bytes_stored / 1e6
    typer.echo(f"Wrote {result.path} ({size:.1f} MB, {result.files} file(s); "
               f"{result.frames_included} of {result.frames_recorded} photo(s) included).")
    for note in result.notes:
        typer.secho(f"note: {note}", fg=typer.colors.YELLOW)
    typer.echo("Check it any time with `fungus archive --verify " + str(result.path) + "`.")


@app.command()
def power(
    effect: str = typer.Option(..., help="Difference to detect: a number in the parameter's "
                               "units, or a percentage of the reference mean with --study "
                               "(e.g. 20%)."),
    sd: float | None = typer.Option(None, help="Spread (SD) of the parameter between replicates "
                                    "of one condition."),
    sd_b: float | None = typer.Option(None, help="SD in the other condition, if different."),
    study_file: Path | None = typer.Option(None, "--study", help="Take the SDs (and the "
                                           "reference mean) from a pilot study already run."),
    param: str | None = typer.Option(None, help="Parameter of the pilot study, e.g. r or "
                                     "max_rate."),
    comparisons: int = typer.Option(1, help="How many comparisons the study makes (Holm)."),
    alpha: float = typer.Option(0.05),
    target: float = typer.Option(0.8, help="Power wanted."),
    n_max: int = typer.Option(20, help="Largest number of replicates to consider."),
    n: int | None = typer.Option(None, help="Also report the smallest difference this many "
                                 "replicates per condition can detect."),
) -> None:
    """How many replicates per condition a study needs to detect a difference."""
    from fungus_cv.analyze import power as power_mod

    reference_mean = math.nan
    source = "given"
    try:
        if study_file is not None:
            if not param:
                raise ValueError("--study needs --param")
            pilot = power_mod.spread_from_study(study_file, param)
            sd_a_value, sd_b_value = pilot.sd_a, pilot.sd_b
            reference_mean = pilot.mean_a
            source = (f"pilot {study_file.name}: {pilot.condition_a} SD {pilot.sd_a:.4g} "
                      f"(n={pilot.n_a}), {pilot.condition_b} SD {pilot.sd_b:.4g} "
                      f"(n={pilot.n_b}); observed difference "
                      f"{pilot.mean_b - pilot.mean_a:+.4g}")
        elif sd is not None:
            sd_a_value, sd_b_value = sd, sd_b if sd_b is not None else sd
        else:
            raise ValueError("give --sd, or a pilot study with --study and --param")
        text = effect.strip()
        if text.endswith("%"):
            if math.isnan(reference_mean):
                raise ValueError("a percentage needs --study (to know the reference mean)")
            difference = abs(float(text[:-1]) / 100 * reference_mean)
        else:
            difference = abs(float(text))
        result = power_mod.power_table(difference, sd_a_value, sd_b_value, alpha, comparisons,
                                       target, n_max)
    except (ValueError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"Detecting a difference of {difference:.4g} (SDs {sd_a_value:.4g} and "
               f"{sd_b_value:.4g}; {source}), Welch test, alpha {alpha}"
               + (f" over {comparisons} comparisons" if comparisons > 1 else "") + ":")
    for row in result.rows:
        bar = "#" * round(row.power * 30)
        mark = "  <- enough" if row.n == result.needed else ""
        typer.echo(f"  {row.n:3d} per condition: power {row.power:5.2f}  {bar}{mark}")
    if result.needed:
        typer.secho(f"{result.needed} replicates per condition give {target:.0%} power.",
                    fg=typer.colors.GREEN)
    else:
        typer.secho(f"Even {n_max} replicates per condition don't reach {target:.0%} power: "
                    "reduce the spread between replicates, or aim for a larger difference.",
                    fg=typer.colors.YELLOW)
    if n is not None:
        smallest = power_mod.detectable_effect(n, sd_a_value, sd_b_value, alpha, comparisons,
                                               target)
        typer.echo(f"With {n} per condition, the smallest difference found {target:.0%} of the "
                   f"time is {smallest:.4g}"
                   + (f" ({100 * smallest / abs(reference_mean):.0f}% of the reference mean)"
                      if reference_mean == reference_mean and reference_mean else "") + ".")


@app.command("validate-suite")
def validate_suite_cmd(
    suite: Path = typer.Argument(..., help="Suite YAML listing experiments, hand labels, models "
                                 "and the accuracy each must reach."),
    init: bool = typer.Option(False, "--init", help="Write an example suite file and exit."),
    save_baseline: bool = typer.Option(False, help="Store this run's numbers as the baseline "
                                       "that later runs are compared with (max_drift)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show every metric, not just "
                                 "the checked ones."),
) -> None:
    """Re-run every accuracy check on hand-labeled real data; exit code 1 if any fails."""
    from fungus_cv.analyze import suite as suite_mod

    if init:
        try:
            suite_mod.write_template(suite)
        except FileExistsError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"Wrote {suite}. Point it at your experiments, hand labels and models, then "
                   "run this command without --init.")
        return
    _setup_logging()
    try:
        result = suite_mod.run_suite(
            suite, progress=lambda case, check: typer.echo(f"running {case} / {check} ..."),
            require_baseline=not save_baseline)
    except (FileNotFoundError, ValueError) as exc:  # missing or invalid suite file
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    for c in result.checks:
        colour = typer.colors.GREEN if c.passed else typer.colors.RED
        typer.secho(f"{'PASS' if c.passed else 'FAIL'}  {c.case} / {c.check}", fg=colour)
        if c.error:
            typer.echo(f"    error: {c.error}")
        for m in c.metrics:
            if m.status == "info" and not verbose:
                continue
            drift = "" if m.drift is None else f"  (baseline {m.baseline:.4g}, {m.drift:+.4g})"
            detail = f"  <- {'; '.join(m.failures)}" if m.failures else ""
            typer.echo(f"    {m.metric:28s} {m.value:10.4g}{drift}{detail}")
        for note in c.notes:
            typer.echo(f"    note: {note}")
    if not result.baseline_found:
        typer.echo("No baseline.json yet: run with --save-baseline once the results look right.")
    typer.echo(f"wrote {result.out_dir}")
    if save_baseline:
        typer.echo(f"saved baseline {suite_mod.save_baseline(result, suite)}")
    n_fail = sum(not c.passed for c in result.checks)
    typer.secho(f"{len(result.checks) - n_fail}/{len(result.checks)} checks passed",
                fg=typer.colors.GREEN if not n_fail else typer.colors.RED)
    if n_fail:
        raise typer.Exit(1)


# --- training your own models --------------------------------------------------------------

dataset_app = typer.Typer(help="Build labeled datasets for training models.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("export")
def dataset_export(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    dataset: Path = typer.Argument(..., help="Dataset folder (created if missing)."),
    run: str | None = typer.Option(None, help="Run id/prefix whose masks to start from "
                                   "(default: newest run; see `fungus runs`)."),
    count: int = typer.Option(20, help="How many frames, spread evenly over time."),
    full_frame: bool = typer.Option(False, help="Export whole frames instead of the region."),
    group: str | None = typer.Option(None, help="Group name (default: experiment name). "
                                     "Validation never mixes groups with training."),
    camera: str | None = typer.Option(None, help="Which camera's run to export from."),
) -> None:
    """Copy frames and their masks into a dataset, to be corrected with `fungus label`."""
    from fungus_cv.analyze.compare import list_runs
    from fungus_cv.analyze.pipeline import AnalysisError
    from fungus_cv.learn.dataset import Dataset
    from fungus_cv.learn.export import export_from_run

    exp = _load_experiment(experiment)
    _setup_logging()
    if run is None:
        found = list_runs(exp, camera)
        if not found:
            typer.secho("no analysis runs yet; run `fungus analyze` first", fg=typer.colors.RED,
                        err=True)
            raise typer.Exit(1)
        run = found[-1].run_id
    ds = Dataset.open_or_create(dataset)
    try:
        added = export_from_run(exp, ds, run, count=count, crop_to_roi=not full_frame,
                                group=group, camera=camera)
    except (AnalysisError, ValueError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Added {len(added)} item(s) from run {run} to {ds.root} "
               f"({len(ds.items)} total). Next: `fungus label {ds.root}`")


@dataset_app.command("add-pairs")
def dataset_add_pairs(
    dataset: Path = typer.Argument(..., help="Dataset folder (created if missing)."),
    images: Path = typer.Argument(..., exists=True, file_okay=False),
    masks: Path = typer.Argument(..., exists=True, file_okay=False,
                                 help="Masks with the same file names; non-zero = target."),
    group: str = typer.Option(..., help="Group name, e.g. where the images came from."),
    reviewed: bool = typer.Option(False, help="Mark as already checked by a person."),
) -> None:
    """Add images with masks made elsewhere (e.g. CVAT, Label Studio, GIMP)."""
    from fungus_cv.learn.dataset import Dataset
    from fungus_cv.learn.export import add_pairs

    _setup_logging()
    ds = Dataset.open_or_create(dataset)
    added = add_pairs(ds, images, masks, group, reviewed)
    typer.echo(f"Added {len(added)} item(s) ({len(ds.items)} total)")


@dataset_app.command("info")
def dataset_info(dataset: Path = typer.Argument(..., help="Dataset folder.")) -> None:
    """Show items per group and how many are reviewed."""
    from fungus_cv.learn.dataset import Dataset
    from fungus_cv.learn.export import dataset_stats

    try:
        ds = Dataset.open(dataset)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{ds.name}: {len(ds.items)} item(s), "
               f"{sum(i.reviewed for i in ds.items)} reviewed")
    for group, st in sorted(dataset_stats(ds).items()):
        typer.echo(f"  {group or '(no group)'}: {st['items']} items, {st['reviewed']} reviewed, "
                   f"target covers {st['mean_target_pct']}% on average")


@dataset_app.command("suggest")
def dataset_suggest(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    dataset: Path = typer.Argument(..., help="Dataset folder (created if missing)."),
    model: Path | None = typer.Option(None, help="Trained model: pick frames it is least sure "
                                      "about."),
    run: str | None = typer.Option(None, help="Run id/prefix whose masks start the labels (and "
                                   "that the model or --against is compared with)."),
    against: str | None = typer.Option(None, help="A second run: pick frames where the two "
                                       "runs disagree most (no model needed)."),
    count: int = typer.Option(20, help="How many frames to add."),
    max_candidates: int = typer.Option(200, help="Frames scored, spread evenly over time."),
    full_frame: bool = typer.Option(False, help="Export whole frames instead of the region."),
    group: str | None = typer.Option(None, help="Group name (default: experiment name)."),
    device: str = typer.Option("auto"),
) -> None:
    """Add the frames most worth labeling: where a model is unsure or methods disagree."""
    from fungus_cv.analyze.pipeline import AnalysisError
    from fungus_cv.learn.active import suggest_frames
    from fungus_cv.learn.dataset import Dataset

    exp = _load_experiment(experiment)
    _setup_logging()
    ds = Dataset.open_or_create(dataset)
    try:
        result = suggest_frames(exp, ds, count=count, model_dir=model, run=run, against=against,
                                max_candidates=max_candidates, crop_to_roi=not full_frame,
                                group=group, device=device)
    except (AnalysisError, ValueError, FileNotFoundError, RuntimeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    scores = sorted((r["score"] for r in result.candidates), reverse=True)
    typer.echo(f"Scored {len(result.candidates)} frame(s); added {len(result.added)} "
               f"(score {min((i.priority for i in result.added), default=0):.2f}-"
               f"{max((i.priority for i in result.added), default=0):.2f}; median of all "
               f"{scores[len(scores) // 2]:.2f}).")
    for item in sorted(result.added, key=lambda i: -i.priority):
        detail = ", ".join(f"{k} {v:.2f}" for k, v in item.scores.items())
        typer.echo(f"  {item.priority:.2f}  {item.id}  ({detail})")
    typer.echo(f"wrote {result.csv_path}")
    typer.echo(f"Next: `fungus label {ds.root} --unreviewed --by-priority`")


@dataset_app.command("rank")
def dataset_rank(
    dataset: Path = typer.Argument(..., help="Dataset folder."),
    model: Path = typer.Option(..., help="Trained model to score the items with."),
    include_reviewed: bool = typer.Option(False, help="Score reviewed items too (e.g. to find "
                                          "label mistakes)."),
    device: str = typer.Option("auto"),
    top: int = typer.Option(10, help="How many to list."),
) -> None:
    """Score unreviewed items so `fungus label --by-priority` shows the most useful first."""
    from fungus_cv.learn.active import rank_items
    from fungus_cv.learn.dataset import Dataset

    _setup_logging()
    try:
        ds = Dataset.open(dataset)
        ranked = rank_items(ds, model, include_reviewed=include_reviewed, device=device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Scored {len(ranked)} item(s). Most worth labeling:")
    for item in ranked[:top]:
        detail = ", ".join(f"{k} {v:.2f}" for k, v in item.scores.items()
                           if isinstance(v, float))
        typer.echo(f"  {item.priority:.2f}  {item.id}  ({detail})")
    typer.echo(f"Next: `fungus label {ds.root} --unreviewed --by-priority`")


@app.command()
def label(
    dataset: Path = typer.Argument(..., help="Dataset folder."),
    unreviewed: bool = typer.Option(False, help="Only show items not yet reviewed."),
    start: int = typer.Option(0, help="Item number to start at."),
    by_priority: bool = typer.Option(False, help="Most useful items first (after `fungus "
                                     "dataset rank` or `suggest`)."),
    sam: bool = typer.Option(False, help="Press m to segment with SAM 2 clicks (needs the sam "
                             "extra)."),
    sam_model: str = typer.Option("facebook/sam2.1-hiera-small", help="SAM 2 model for --sam."),
    device: str = typer.Option("auto", help="Device for --sam."),
) -> None:
    """Correct masks with a brush (and SAM clicks); saving marks an item as reviewed."""
    _require_gui()
    from fungus_cv.learn.dataset import Dataset
    from fungus_cv.ui.interactive import edit_labels

    try:
        ds = Dataset.open(dataset)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    segment = None
    if sam:
        import importlib.util

        if importlib.util.find_spec("torch") is None or \
                importlib.util.find_spec("transformers") is None:
            typer.secho('--sam needs PyTorch and transformers: pip install -e ".[sam]"',
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        from fungus_cv.segment.prompts import Prompts
        from fungus_cv.segment.sam2 import Sam2VideoSegmenter

        segment = Sam2VideoSegmenter(model_name=sam_model, prompts=Prompts(),
                                     device=device).segment_single
    counts = edit_labels(ds, start=start, only_unreviewed=unreviewed, by_priority=by_priority,
                         sam=segment)
    typer.echo(f"Saved {counts['saved']} item(s); "
               f"{sum(i.reviewed for i in ds.items)}/{len(ds.items)} reviewed")


@app.command("train")
def train_cmd(
    dataset: Path = typer.Argument(..., help="Dataset folder."),
    output: Path = typer.Argument(..., help="New folder for the trained model."),
    encoder: str = typer.Option("resnet34", help="resnet18 (faster) | resnet34"),
    steps: int = typer.Option(3000, help="Training steps (batches)."),
    batch_size: int = typer.Option(8),
    patch_px: int = typer.Option(384, help="Training patch size in pixels."),
    learning_rate: float = typer.Option(3e-4),
    val_group: list[str] = typer.Option([], help="Group(s) to hold out for validation "
                                        "(repeatable). Default: chosen automatically."),
    include_unreviewed: bool = typer.Option(False, help="Also train on unreviewed masks."),
    flip_vertical: bool = typer.Option(False, help="Allow upside-down augmentation."),
    rotate90: bool = typer.Option(False, help="Allow 90-degree rotation augmentation."),
    color_jitter: float = typer.Option(1.0, help="Scale colour/brightness augmentation "
                                       "(0 = none, 2 = double)."),
    pretrained: bool = typer.Option(True, help="Start from ImageNet weights."),
    device: str = typer.Option("auto", help="auto | cuda | mps | cpu"),
    seed: int = typer.Option(0),
) -> None:
    """Train a segmentation model; writes model.pt and a model.json card."""
    from fungus_cv.learn.dataset import Dataset
    from fungus_cv.learn.train import TrainConfig, train

    _setup_logging()
    try:
        ds = Dataset.open(dataset)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    cfg = TrainConfig.with_color_jitter(
        color_jitter,
        encoder=encoder, pretrained=pretrained, patch_px=patch_px, batch_size=batch_size,
        steps=steps, learning_rate=learning_rate, val_groups=list(val_group),
        reviewed_only=not include_unreviewed, flip_vertical=flip_vertical, rotate90=rotate90,
        device=device, seed=seed,
    )
    try:
        result = train(ds, output, cfg)
    except (ValueError, FileExistsError, RuntimeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Trained on {result.n_train} item(s), validated on {result.n_val} "
               f"({result.split}) in {result.seconds / 60:.1f} min")
    if result.val_metrics:
        v = result.val_metrics
        typer.echo(f"  validation IoU mean {v['iou_mean']:.4f} (min {v['iou_min']:.4f}), "
                   f"boundary F1@2px {v['boundary_f1_2px_mean']:.4f}, "
                   f"threshold {result.threshold}")
    else:
        typer.echo("  no validation data: the model is unchecked; use `fungus evaluate`")
    typer.echo(f"Wrote {result.out_dir}. Use it with analysis.target.method: model and "
               f"analysis.target.model.path: {result.out_dir}")


@app.command()
def evaluate(
    model: Path = typer.Argument(..., help="Model folder from `fungus train`."),
    dataset: Path = typer.Argument(..., help="Labeled dataset to test on."),
    include_unreviewed: bool = typer.Option(False),
    threshold: float | None = typer.Option(None, help="Override the tuned threshold."),
    device: str = typer.Option("auto"),
) -> None:
    """Measure a model against labeled masks (IoU, Dice, precision, recall, boundary F1)."""
    import csv

    from fungus_cv.learn.dataset import Dataset
    from fungus_cv.learn.train import evaluate_model

    _setup_logging()
    try:
        ds = Dataset.open(dataset)
        rows, summary = evaluate_model(model, ds, not include_unreviewed, device, threshold)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    if not rows:
        typer.secho("no items to evaluate (none reviewed?)", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    out = model / f"evaluation_{ds.name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    typer.echo(f"threshold {summary['threshold']}")
    for label_, key in (("all items", "all"), ("not used for training", "not_in_training_set")):
        st = summary[key]
        if st["n"]:
            typer.echo(f"  {label_} (n={st['n']}): IoU {st['iou_mean']:.4f} "
                       f"(min {st['iou_min']:.4f})  Dice {st['dice_mean']:.4f}  "
                       f"precision {st['precision_mean']:.4f}  recall {st['recall_mean']:.4f}  "
                       f"boundary F1@2px {st['boundary_f1_2px_mean']:.4f}")
        else:
            typer.echo(f"  {label_}: none")
    if summary["all"]["n"] and not summary["not_in_training_set"]["n"]:
        typer.echo("  Warning: every item was used in training; these numbers are optimistic.")
    typer.echo(f"wrote {out}")

if __name__ == "__main__":
    app()
