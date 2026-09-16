# Fungus-CV

Take time-lapse photos with a webcam, then measure how a spreading region (dye, moss, an infection) moves relative to a reference object over time. See [PLAN.md](PLAN.md) for the full roadmap.

**Status:** capture (M1), color-threshold measurement (M2), SAM 2 segmentation (M3), robustness to tilt/lighting/jumps (M4) and training your own models (M7) all work.

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

## Robustness: tilted cameras, changing light, bad frames

Each frame is prepared the same way before segmentation: **align** to the reference frame → **rectify** (optional) → **correct lighting** (optional). The interactive tools show frames prepared exactly this way, and so does anything that uses them later: annotations, prompts, color picking and dataset exports.

**Tilted camera** (`analysis.rectify.enabled: true`): frames are warped to a top-down view of the plane the markers lie on.
- All visible markers are fitted together so each becomes a square of the known size. Markers can be anywhere (on separate stakes in a field) and no printed layout is needed.
- Use 2–4 markers spread around the measured area. A single marker works, but its correction extrapolates across the image.
- Scenes showing the horizon (field shots) are handled by limiting the output to the area around the markers.
- **Re-run `fungus annotate` after turning rectification on**, because the prepared image changes shape.
- On a synthetic tilted scene, errors dropped from up to 9.6 mm to at most 0.26 mm.

**Changing light** (`analysis.lighting.method`):
- `patch` (most reliable): tape a white or grey card where it stays in view, and mark it in the optional last step of `fungus annotate`.
- `background`: uses everything outside the measured region; the background must not change.
- **How it works:** each color channel gets a gain so the reference region matches the first frame. This undoes overall brightness changes and color casts, but not shadows or glare on part of the scene.
- **Recorded per frame:** gains are saved (`light_gain_b/g/r`). Frames needing more than a 30% correction are flagged `lighting_changed`, and clipped reference pixels are flagged `saturated`.
- On a synthetic run where the lights dimmed, uncorrected frames lost the dye entirely (−80 mm of fake shrinkage). Corrected frames were within 0.1 mm.

**Bad frames:**
- `analyze` flags `align_failed`, `align_poor`, `blurry`, `brightness_changed`, `lighting_changed`, `saturated` and `no_target`.
- `report` also marks **jumps** (a frame far off the robust local trend of its neighbors) and **retreats** (the front moving back by more than 3σ).
- `results/report/frame_flags.csv` lists every frame's flags and whether it was used in the fits, so any exclusion can be audited.
- Frames flagged `align_failed` or `blurry` are left out of fits by default. Jumps are only marked, unless you pass `--exclude-jumps`.

## Segmenting with SAM 2 (for targets a color threshold can't separate)

Color thresholds work well for dye. Moss on bark or soil usually needs a model. SAM 2 finds the target from a few clicks, then **tracks it through the whole time-lapse**.

```bash
pip install -e ".[sam]"      # PyTorch + transformers. NVIDIA GPU: install the CUDA build of torch first
fungus prompt experiments/moss-1 --frame last   # click on the target; the mask previews live
# set analysis.target.method: sam2 in config.yaml
fungus analyze experiments/moss-1
fungus runs experiments/moss-1       # every run, color or sam2, with its masks
fungus compare experiments/moss-1    # IoU and extent difference between the two newest runs
```

- **Prompting:**
  - Left-click the target and right-click things that look similar but aren't.
  - `b` draws a box around the target instead of clicking.
  - Pick a frame where the target is clearly visible; the last frame is the default.
  - If tracking drifts, prompt more frames. Each prompted frame corrects the tracking from that point on.
- **Tracking:** SAM 2 tracks forward in time from the first prompted frame, and backward for earlier frames, including ones taken before the target appeared. Prompts are stored in `prompts.json`, and editing them starts a new, separately archived run.
- **Precision:** SAM works internally at about 1024 px, so each frame is first cropped to the annotated region (`crop_to_roi`). On synthetic tests this cut area error from about 190 px to about 3 px.
- **Models:** `sam2.1-hiera-tiny`, `-small` (default), `-base-plus` and `-large` trade speed for accuracy. Weights download on first use.
  - The device (CUDA, Apple GPU or CPU) is picked automatically.
  - On an M1 Pro, `-small` takes about 0.8 s per frame.
  - New frames re-run tracking from the prompted frame, so each `analyze` gets slower as the run grows (minutes for a few hundred frames).
- **Validating:** `fungus compare` accepts any folder of mask PNGs, not just runs. Use it to check SAM against hand-labeled frames before trusting it on a new kind of scene. On synthetic dye, SAM 2-small matched the color threshold with mean IoU 0.991 and read the front **0.21 mm lower** on average; check for similar systematic offsets on real images.
- **Tests:** tests that load the real model run only with `FUNGUS_TEST_SAM=1`.

## Training your own model for a new use case

When a color threshold isn't reliable and SAM needs too much clicking, train a model on examples of your target: moss on bark, discolored leaves, a field's coloring. The loop:

```bash
# 1. Start labels from any analysis run (SAM or color): frames spread evenly over time
fungus dataset export experiments/moss-1 datasets/moss-bark --count 30
fungus dataset export experiments/moss-2 datasets/moss-bark --count 30   # more experiments = better
fungus dataset add-pairs datasets/moss-bark photos/ masks/ --group field-2026   # masks made elsewhere

# 2. Correct the masks: left drag paints, right drag erases; saving marks an item reviewed
fungus label datasets/moss-bark --unreviewed
fungus dataset info datasets/moss-bark

# 3. Train (GPU if available) and check on data the model has not seen
fungus train datasets/moss-bark models/moss-bark-v1
fungus evaluate models/moss-bark-v1 datasets/moss-bark-test

# 4. Use it: analysis.target.method: model, analysis.target.model.path: <model folder>
fungus analyze experiments/moss-3
```

- **Model:** a U-Net with an ImageNet-pretrained ResNet encoder (`resnet34` by default, `resnet18` is faster). It has a full-resolution path so mask edges aren't blurred. Large frames are processed in overlapping tiles.
- **Training data:** only **reviewed** labels are used by default, so model quality depends on labels a person checked. `--include-unreviewed` overrides this.
- **Validation without leakage:** consecutive time-lapse frames are near-duplicates, so validation never mixes them with training.
  - Whole groups (experiments) are held out.
  - With one group, the latest frames are held out.
  - `--val-group` chooses the groups explicitly.
- **Augmentation:** horizontal flips, scale, and small color and brightness changes. Upside-down flips and rotation are off by default, because growth direction often matters; `--flip-vertical` and `--rotate90` turn them on. Raise `--color-jitter` if lighting varies a lot.
- **Model card:** `model.json` records the dataset fingerprint (a hash of every image and mask), the exact train/validation items, the training settings and history, and validation metrics. The threshold is tuned on validation data, which also picks the best checkpoint, so **report accuracy from `fungus evaluate` on separate labeled data**. The analysis settings hash includes the model's weights hash, so results from different models are never mixed.
- **Metrics:** `fungus evaluate` reports IoU, Dice, precision, recall and **boundary F1 within 2 px** (how accurately edges are placed, which matters for front-position measurements) for items used in training and items not used in training, separately.
- **Time:** on an M1 Pro, 100 steps of `resnet18` at 256 px took about 25 s. Real datasets will want the default 3000 steps: tens of minutes on a GPU, much longer on a CPU.

## What an experiment folder contains

```
experiments/dye-test-1/
  config.yaml     # settings for this experiment (edit this)
  frames/         # 2026-09-20T14-05-00.123Z_cam0.png ... (UTC time, sorts in time order)
  frames.csv      # one row per image or failed attempt (see below)
  capture.log
  annotations.json  # base, tip, region and optional neutral patch (from `fungus annotate`)
  prompts.json    # SAM clicks (from `fungus prompt`)
  results/        # measurements.csv, run_info.json, report/, archive/,
                  # masks/<run>/, overlays/<run>/, runs/<run>.json, compare/
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
