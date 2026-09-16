# Fungus-CV

Take time-lapse photos with a webcam, then measure how a spreading region (dye, moss, an infection) moves relative to a reference object over time. See [PLAN.md](PLAN.md) for the full roadmap.

**Status:** capture works (M1), and measuring with a color threshold works (M2). SAM and trainable models come next.

Works on Windows, macOS and Linux (Python 3.10+).

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

(With [uv](https://docs.astral.sh/uv/): `uv venv && uv pip install -e ".[dev]"`.)

## Quick start

```bash
fungus cameras                       # which cameras are connected?
fungus init experiments/dye-test-1   # creates the folder and config.yaml
fungus preview experiments/dye-test-1   # framing & focus: q quit, s snapshot, g grid
fungus snap experiments/dye-test-1      # one test picture
fungus capture experiments/dye-test-1 --interval 10s --duration 30m
fungus status experiments/dye-test-1
```

Press Ctrl+C to stop a capture. Each image is saved as soon as it's taken, so nothing is lost.

Import photos from a phone or another camera:

```bash
fungus import experiments/field-plot-3 path/to/photos --camera phone --tz America/Chicago
```

## Measuring (dye test workflow)

```bash
fungus markers markers.pdf --size-mm 30     # print at 100%, check the 100 mm ruler
# Tape one or more markers in the SAME PLANE as the towel, visible in every frame.
# Measure the printed black square with a ruler; set analysis.markers.size_mm in config.yaml.

fungus capture experiments/dye-test-1 --interval 10s --duration 30m
fungus annotate experiments/dye-test-1      # click base (waterline), tip (top of towel), region
fungus pick-color experiments/dye-test-1    # drag boxes over the dye; saves HSV ranges to config
fungus analyze experiments/dye-test-1       # or --watch to measure frames as they arrive
fungus report experiments/dye-test-1 --t0 "2026-09-20 14:05:00" --video
```

What `analyze` does for each frame:

1. **Align** it to the first (reference) frame using the markers. Without markers it uses image matching (ECC) on the area outside the region you annotated. This corrects a camera or towel that got bumped.
2. **Segment:** find the target pixels. Today that's a color threshold; SAM and trained models will plug into the same step.
3. **Measure** within the annotated region:
   - `extent_mm`: the front's median height across the towel's width, measured from the base along the base→tip axis.
   - `extent_max_mm`: the highest point the front reaches.
   - Area and coverage %.
4. **Uncertainty:** `extent_mm_unc` is a combined standard uncertainty from the marker scale, pixel size and alignment residual. It does not include how exactly the threshold places the dye edge; see *Accuracy notes*.
5. **Flag** frames that are doubtful: `align_failed`, `align_poor`, `no_target`, `brightness_changed`, `blurry`.

Results go to `results/measurements.csv`, with a mask and an overlay image per frame.

- **Provenance:** `results/run_info.json` records the scale, the software versions and every setting used. Each row has a `settings_hash`.
- **Settings changes:** if you change the settings, old results are moved to `results/archive/`, never mixed with new ones.
- **Re-runs:** only new frames are measured.

`report` fits these models to the data:

| Model | Formula | Use |
|---|---|---|
| linear | `a + b·t` | constant speed |
| sqrt | `k·√t` | capillary wicking (Lucas–Washburn) |
| power | `a·tⁿ` | **n ≈ 0.5 confirms wicking behavior**, so it's a good end-to-end check |
| logistic | `K / (1 + e^(−r(t − t_mid)))` | growth that levels off, e.g. infection |

It outputs parameters ± standard errors, R², and AIC (lower AIC = better model), weighted by the per-frame uncertainty. Output files:

- `results/report/`: the main plot, a quality-check plot, `fits.json` and an optional overlay video.
- Frames where the front moves back by more than 3σ are circled as worth checking.

### Accuracy notes
- Markers must lie in the same plane as the object, and the camera should face that plane square-on. The analysis warns if marker edges disagree by more than 2%, which suggests a tilted view.
- Set `--t0` to the moment the towel touched the dye. Otherwise t = 0 is the first photo.
- The dye-edge position depends on the color thresholds. To estimate that uncertainty, re-run with slightly wider and narrower `hsv_ranges` and compare. Archived results make this easy.

## What an experiment folder contains

```
experiments/dye-test-1/
  config.yaml     # settings for this experiment (edit this)
  frames/         # 2026-09-20T14-05-00.123Z_cam0.png ... (UTC time, sorts in time order)
  frames.csv      # one row per image or failed attempt (see below)
  capture.log
  annotations.json  # base, tip and region (from `fungus annotate`)
  results/        # measurements.csv, run_info.json, masks/, overlays/, report/, archive/
```

`frames.csv` columns: `timestamp_utc, camera, status (ok/failed), file, sha256, width, height, source, scheduled_utc, lag_s, mean_brightness, sharpness, camera_settings (JSON), notes`.

`mean_brightness` and `sharpness` are recorded so bad frames can be found later, for example when a light was switched on or the camera lost focus.

## Getting consistent images (important for measurements)

- **Camera controls** (`exposure`, `white_balance`, `focus` in `config.yaml`):
  - `lock` (default): the camera adjusts automatically once when it opens, then keeps that value for the whole run.
  - `auto`: the camera keeps adjusting.
  - A number: a fixed value you choose.

  Auto-adjusting cameras change color and brightness between frames, which looks like growth.
- Not every webcam or OS supports every control. OpenCV on **macOS** usually cannot set exposure, white balance or focus. You'll see a warning in the log, and the `notes` column records it. On **Windows**, press `d` in `fungus preview` to open the driver's settings window.
- Use steady, controlled lighting, avoid sunlight that changes during the day, mount the camera rigidly, and PNG output (the default).

## Long unattended runs

- `keep_awake: true` stops the computer from sleeping while it's idle. Closing a laptop lid can still put it to sleep, so change that in the power settings.
- If a run is interrupted, restart it with the same `start_at` in `config.yaml`. Capture continues on the original schedule.
- The camera is released between shots for intervals over 2 minutes, which helps it recover after being unplugged.
- To start capture automatically after a reboot:
  - **Windows:** Task Scheduler → *At startup* → run `.venv\Scripts\fungus.exe capture C:\path\to\experiment`
  - **macOS:** a `launchd` agent in `~/Library/LaunchAgents` with `RunAtLoad` and `KeepAlive`. The Python you run needs camera permission.
  - **Linux:** a `systemd --user` service with `Restart=on-failure`.

## Development

```bash
pytest
ruff check .
```

The tests use a fake camera, so they need no hardware. CI runs them on Windows, macOS and Linux.
