"""``fungus`` command-line interface."""

from __future__ import annotations

import logging
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

    found = list_cameras(max_index=max_index, backend=backend)
    if not found:
        typer.echo("No cameras found.")
        if sys.platform == "darwin":
            typer.echo(
                "On macOS, allow camera access for your terminal app in "
                "System Settings > Privacy & Security > Camera."
            )
        raise typer.Exit(1)
    for cam in found:
        typer.echo(
            f"index {cam['index']}: {cam['width']}x{cam['height']} (backend {cam['backend']})"
        )


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

    from fungus_cv.capture.camera import Camera
    from fungus_cv.config import CameraConfig
    from fungus_cv.quality import mean_brightness, sharpness

    _setup_logging()
    out_dir = Path.cwd()
    if experiment is not None:
        exp = _load_experiment(experiment)
        cam_cfg = exp.config.camera(camera) if camera else exp.config.cameras[0]
        out_dir = exp.root
    else:
        cam_cfg = CameraConfig(index=index)

    window = f"fungus preview - {cam_cfg.name}"
    show_grid = False
    with Camera(cam_cfg) as cam:
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


@app.command()
def snap(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
) -> None:
    """Take one picture from every camera right now."""
    from fungus_cv.capture.scheduler import CaptureSession

    exp = _load_experiment(experiment)
    _setup_logging(exp.log_path)
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
) -> None:
    """Import existing photos, ordered by capture time (EXIF, file name, or file date)."""
    from fungus_cv.capture.importer import import_folder

    exp = _load_experiment(experiment)
    _setup_logging()
    records = import_folder(exp, folder, camera=camera, timezone_name=tz, recursive=recursive)
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


def _reference_image(exp):
    from fungus_cv.analyze.pipeline import AnalysisError, frames_for_analysis, read_image

    try:
        row = frames_for_analysis(exp)[0]
        return row, read_image(exp, row["file"])
    except AnalysisError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


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
def annotate(experiment: Path = typer.Argument(..., help="Experiment folder.")) -> None:
    """Click the base, tip and region to measure on the reference (first) frame."""
    from fungus_cv.analyze.pipeline import annotations_path
    from fungus_cv.ui.interactive import Cancelled
    from fungus_cv.ui.interactive import annotate as run_annotate

    exp = _load_experiment(experiment)
    row, image = _reference_image(exp)
    try:
        ann = run_annotate(image, reference_file=row["file"])
    except Cancelled:
        typer.echo("Cancelled; nothing saved.")
        raise typer.Exit(1) from None
    path = annotations_path(exp)
    ann.save(path)
    typer.echo(f"Saved {path} (axis length {ann.axis_length_px:.1f} px)")


@app.command("pick-color")
def pick_color(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    frame: Path | None = typer.Option(
        None, help="Image to sample from (default: the last frame, where the target is largest)."
    ),
    write: bool = typer.Option(True, help="Write the ranges into config.yaml."),
) -> None:
    """Measure the target's color by dragging boxes over it, then save the HSV ranges."""
    import cv2

    from fungus_cv.analyze.pipeline import frames_for_analysis, read_image
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
        _reference_image(exp)  # validates that frames exist
        image = read_image(exp, frames_for_analysis(exp)[-1]["file"])
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
        typer.echo("Watching for new frames; Ctrl+C to stop.")
        try:
            run_watch(exp.root, parse_duration(poll), on_summary=show)
        except KeyboardInterrupt:
            pass
        return
    try:
        show(run_analyze(exp, force=force))
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
) -> None:
    """Fit growth models and write plots to results/report/."""
    from fungus_cv.analyze.report import DEFAULT_EXCLUDE, make_report

    exp = _load_experiment(experiment)
    _setup_logging()
    if t0 is not None and t0.tzinfo is None:
        t0 = t0.astimezone()
    try:
        result = make_report(
            exp, metric=metric, t0=t0, time_unit=time_unit,
            exclude_flags=() if include_flagged else DEFAULT_EXCLUDE, video=video,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"{result.metric}: {result.n_used} frames used, {result.n_excluded} excluded, "
               f"{result.retreats} retreat(s) to check; time in {result.time_unit}")
    ok = [f for f in result.fits if f.ok]
    best_aic = min((f.aic for f in ok), default=None)
    for fit in result.fits:
        if not fit.ok:
            typer.echo(f"  {fit.model:9s} failed: {fit.message}")
            continue
        params = ", ".join(f"{k}={v:.4g}±{fit.stderr[k]:.2g}" for k, v in fit.params.items())
        best = "  <- lowest AIC" if fit.aic == best_aic else ""
        typer.echo(f"  {fit.model:9s} {params}  R²={fit.r2:.4f}  AIC={fit.aic:.1f}{best}")
    for f in result.files:
        typer.echo(f"wrote {f}")

if __name__ == "__main__":
    app()
