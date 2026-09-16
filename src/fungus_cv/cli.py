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
    from fungus_cv.analyze.pipeline import AnalysisError, Analyzer, annotations_path
    from fungus_cv.ui.interactive import Cancelled
    from fungus_cv.ui.interactive import annotate as run_annotate

    exp = _load_experiment(experiment)
    _setup_logging()
    try:  # show the reference as the pipeline prepares it (e.g. rectified)
        analyzer = Analyzer(exp, with_segmenter=False, require_annotations=False)
    except AnalysisError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    row, image = analyzer.reference_row, analyzer.reference
    try:
        ann = run_annotate(image, reference_file=row["file"])
    except Cancelled:
        typer.echo("Cancelled; nothing saved.")
        raise typer.Exit(1) from None
    path = annotations_path(exp)
    ann.save(path)
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
    exclude_jumps: bool = typer.Option(False, help="Leave frames that jump off the local "
                                       "trend out of the fits (they are always marked)."),
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
            exclude_jumps=exclude_jumps,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"{result.metric}: {result.n_used} frames used, {result.n_excluded} excluded, "
               f"{result.retreats} retreat(s) and {result.jumps} jump(s) to check "
               f"(see frame_flags.csv); time in {result.time_unit}")
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
def prompt(
    experiment: Path = typer.Argument(..., help="Experiment folder."),
    frame: str = typer.Option(
        "last", help="Frame to click on: last (default), first, an index, or a file name. "
        "Pick one where the target is clearly visible."
    ),
    preview: bool = typer.Option(True, help="Show the SAM mask live while clicking."),
    reference: bool = typer.Option(False, "--reference",
                                   help="Prompt the reference object (e.g. stem) instead."),
) -> None:
    """Click on the target for SAM 2 (left = target, right = not target)."""
    from fungus_cv.analyze.pipeline import AnalysisError, Analyzer
    from fungus_cv.segment.prompts import Prompts
    from fungus_cv.segment.sam2 import Sam2VideoSegmenter, _RoiCrop
    from fungus_cv.ui.interactive import Cancelled, prompt_target

    exp = _load_experiment(experiment)
    _setup_logging()
    try:
        analyzer = Analyzer(exp, with_segmenter=False)
    except AnalysisError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    index = _pick_frame_index(analyzer.frames, frame)
    frame_file = analyzer.frames[index]["file"]
    image = analyzer.aligned_frame(index)

    block = exp.config.analysis.reference if reference else exp.config.analysis.target
    cfg = block.sam2
    path = exp.root / cfg.prompts_file
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
def runs(experiment: Path = typer.Argument(..., help="Experiment folder.")) -> None:
    """List analysis runs whose masks are on disk (for `fungus compare`)."""
    from fungus_cv.analyze.compare import list_runs

    exp = _load_experiment(experiment)
    found = list_runs(exp)
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
) -> None:
    """Compare two runs' masks per frame: IoU and extent difference."""
    from fungus_cv.analyze.compare import compare_runs, list_runs

    exp = _load_experiment(experiment)
    if run_a is None or run_b is None:
        found = list_runs(exp)
        if len(found) < 2:
            typer.secho("need two runs; analyze with two different settings first",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        run_a = run_a or found[-2].run_id
        run_b = run_b or found[-1].run_id
    try:
        r = compare_runs(exp, run_a, run_b)
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
    make_template: int = typer.Option(0, help="Instead of validating, write HAND_CSV with this "
                                      "many evenly spaced frames to measure by hand."),
) -> None:
    """Compare automatic measurements with hand measurements (Bland-Altman agreement)."""
    from fungus_cv.analyze.validate import validate, write_template

    exp = _load_experiment(experiment)
    try:
        if make_template:
            if hand_csv.exists():
                raise ValueError(f"{hand_csv} already exists")
            n = write_template(exp, hand_csv, make_template)
            typer.echo(f"Wrote {hand_csv} with {n} frames. Measure each frame by hand (same "
                       f"units as {metric}), fill in 'value', then run this command again "
                       "without --make-template.")
            return
        a = validate(exp, hand_csv, metric)
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
    if a.unmatched:
        typer.echo(f"  {len(a.unmatched)} hand row(s) not matched to analyzed frames: "
                   f"{', '.join(a.unmatched[:5])}")
    for f in a.files:
        typer.echo(f"wrote {f}")


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
) -> None:
    """Copy frames and their masks into a dataset, to be corrected with `fungus label`."""
    from fungus_cv.analyze.compare import list_runs
    from fungus_cv.analyze.pipeline import AnalysisError
    from fungus_cv.learn.dataset import Dataset
    from fungus_cv.learn.export import export_from_run

    exp = _load_experiment(experiment)
    _setup_logging()
    if run is None:
        found = list_runs(exp)
        if not found:
            typer.secho("no analysis runs yet; run `fungus analyze` first", fg=typer.colors.RED,
                        err=True)
            raise typer.Exit(1)
        run = found[-1].run_id
    ds = Dataset.open_or_create(dataset)
    try:
        added = export_from_run(exp, ds, run, count=count, crop_to_roi=not full_frame,
                                group=group)
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


@app.command()
def label(
    dataset: Path = typer.Argument(..., help="Dataset folder."),
    unreviewed: bool = typer.Option(False, help="Only show items not yet reviewed."),
    start: int = typer.Option(0, help="Item number to start at."),
) -> None:
    """Correct masks with a brush; saving marks an item as reviewed."""
    from fungus_cv.learn.dataset import Dataset
    from fungus_cv.ui.interactive import edit_labels

    try:
        ds = Dataset.open(dataset)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    counts = edit_labels(ds, start=start, only_unreviewed=unreviewed)
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
    base = TrainConfig()
    cfg = TrainConfig(
        encoder=encoder, pretrained=pretrained, patch_px=patch_px, batch_size=batch_size,
        steps=steps, eval_every=max(1, min(base.eval_every, steps // 4 or 1)),
        learning_rate=learning_rate, val_groups=list(val_group),
        reviewed_only=not include_unreviewed, flip_vertical=flip_vertical, rotate90=rotate90,
        brightness_jitter=base.brightness_jitter * color_jitter,
        contrast_jitter=base.contrast_jitter * color_jitter,
        hue_jitter_deg=base.hue_jitter_deg * color_jitter,
        saturation_jitter=base.saturation_jitter * color_jitter,
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
