# Fungus-CV — Plan

Time-lapse capture plus computer-vision measurement of how far a "spreading" region (dye, moss) has moved relative to a reference object (paper towel, plant stem, field plot).

---

## Status (2026-09-18)

**Every milestone below is written and tested, but nothing has been run on real experiment photos or in the field yet.** The accuracy figures quoted come from synthetic scenes with known answers (the real SAM 2 and U-Net models were run, on those synthetic scenes), and the statistics from simulation.

- **Works end to end** on synthetic data: capture → measure → uncertainty → fits → report, from the command line and from the desktop app.
- **Ready and waiting for real data:** `fungus validate-suite` re-runs every accuracy check against hand-labeled frames (Bland–Altman agreement, mask IoU, model metrics) and compares them with a saved baseline. It needs your dye and moss frames.
- **Try it without a camera:** `fungus demo` makes a realistic synthetic experiment with its truth, hand measurements and a validation suite.
- **The three things that still need you:** a real dye run (does the √t law come out?), ~20 hand-measured moss frames, and an 8-hour unattended capture on each OS. `fungus health --watch` is there to watch that last one.

See the README for how to use any of it.

---

## 1. Goals

| Stage | Scene | Spreading region | Reference object | Main metric |
|---|---|---|---|---|
| Test 1 | Paper towel dipped in blue dye | Blue wet area | Towel strip and waterline | Height of the wet front (mm) over time |
| Target A | Potted plant | Moss on the stem | Stem, from soil line to tip | Height reached on the stem (mm and % of stem) |
| Target B | Field or plot | Moss patch | Plot boundary or markers | Covered area (cm² and %) and how far the edge has advanced |
| Target C | Whole field | Color change across the field | Field boundary or markers | Color change per area over time (e.g. share of the field that is discolored) |

Other requirements:
- Runs on **Windows and macOS**, and Linux if possible.
- Capture can run unattended for days or weeks.
- Analysis works on **any folder of images**, so photos from a phone, trail camera or Raspberry Pi can be used as well as a webcam. This matters for field use.

### Decisions so far (2026-09-16)
- **Intervals:** seconds to minutes while testing with dye; hours between photos for real runs.
- **Hardware:** a webcam for now. Field hardware is still undecided, so importing existing photos stays supported.
- **Compute:** a modest GPU. Images are processed **as they arrive** (incremental), so slow models are fine.
- **Precision:** results are for a research paper. Calibration (markers, locked camera settings) is required, and every result needs an uncertainty estimate and a record of how it was produced.
- **Plant metrics:** both height reached and % of stem covered.
- **Adaptable detection:** being able to **train the program with labeled examples** for each new use case is a core feature, not a fallback. The custom-model milestone moves earlier, and the segmentation interface is designed around pluggable trained models.

---

## 2. Technology choices

**Language: Python 3.10+.** It is the only language where OpenCV, PyTorch, Segment Anything and the plotting and data tools all work well together. All of them run on Windows, macOS and Linux.

| Concern | Choice | Notes |
|---|---|---|
| Environment and packages | `uv` (or conda) + `pyproject.toml` | One command sets up the project on every OS |
| Camera capture | `opencv-python` | Backend per OS: `CAP_DSHOW`/`CAP_MSMF` (Windows), `CAP_AVFOUNDATION` (macOS), `CAP_V4L2` (Linux) |
| Segmentation (main) | **SAM 2** (Meta, `sam2`) | The video predictor lets you click the object **once** and carries the mask through the whole image sequence |
| Segmentation (text prompts, optional) | SAM 3 or Grounding-DINO + SAM | "moss" as a text prompt, with no clicks |
| Segmentation (simple baseline) | HSV color thresholding in OpenCV | Very good for blue dye; also a check on SAM's results |
| Later: custom model | Fine-tuned small model (e.g. YOLO-seg or U-Net) | Trained on masks produced by SAM and corrected by hand, if SAM struggles with moss |
| Hardware for inference | PyTorch on CUDA (Windows/Linux), MPS (Apple Silicon), CPU fallback | Picked automatically |
| Scale and alignment | OpenCV **ArUco markers** | Printed markers give pixel→mm scale and let the pipeline correct a camera that has been bumped |
| Data | Images on disk + `SQLite` or CSV for metadata and results | Simple and easy to move between machines |
| CLI | `typer` | `fungus capture`, `fungus annotate`, `fungus analyze`, `fungus report` |
| Config | YAML (`pydantic` validated) | One config file per experiment |
| Plots and fits | `pandas`, `matplotlib`, `scipy` | Curve fitting of growth rate |
| Desktop app | **PySide6 (Qt)** | Done: a page per step, packaged as `Fungus-CV.app` / `.exe`. Chosen over Streamlit so the camera, the click tools and long analyses run locally without a browser |
| Statistics | `scipy.stats` | Fit uncertainty (AR(1) errors, bootstrap), Welch tests with Holm correction, random-effects means |

---

## 3. Architecture

```
 ┌──────────┐   ┌───────────┐   ┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐
 │ Capture  │──▶│  Storage  │──▶│ Preprocess  │──▶│ Segmentation │──▶│ Measurement│──▶│ Analysis │
 │ (webcam /│   │ images +  │   │ align, scale│   │ target mask +│   │ extent,    │   │ curves,  │
 │  import) │   │ metadata  │   │ color/light │   │ reference    │   │ area, front│   │ rates,   │
 └──────────┘   └───────────┘   └─────────────┘   └──────────────┘   └────────────┘   │ overlays │
                                                                                      └──────────┘
```

Each stage reads the previous stage's output from disk. You can re-run analysis with a new model or settings without capturing again.

### Experiment folder layout (as built)
```
experiments/2026-09-20_dye_test/
  config.yaml            # cameras, interval, segmentation, measurement, uncertainty…
  frames/                # 2026-09-20T14-05-00.123Z_cam0.png  (UTC, sortable)
  frames.csv             # timestamp, camera, hash, size, brightness, sharpness, settings
  capture.log  status.json        # log and heartbeat (for `fungus health`)
  annotations.json       # base, path, region or field plots, neutral patch
  exclusions.json        # frames left out by hand, with reasons
  prompts.json           # SAM clicks (annotations_<cam>.json / prompts_<cam>.json with
                         # several cameras; their results go to results/cameras/<cam>/)
  results/
    measurements.csv  run_info.json  runs/<settings hash>.json   # what produced each row
    masks/<hash>/…png   overlays/<hash>/…jpg   masks/<hash>/sam2_state/  # resume tracking
    report/  validation/  compare/  sensitivity/  combined/  archive/
```

### Code layout (as built)
```
fungus_cv/
  capture/      camera.py, scheduler.py, importer.py (+ --watch), permissions.py,
                diagnostics.py, power.py, health.py (heartbeat, checks, alerts), service.py
  preprocess/   align.py (ArUco/ECC), markers.py (px→mm), rectify.py, lighting.py
  segment/      base.py (Segmenter interface), color.py, sam2.py, trained.py, prompts.py
  measure/      geometry.py, path.py (arc length), centerline.py, color_indices.py,
                color_classes.py (share of each named colour)
  learn/        dataset.py, export.py, active.py (what to label next), train.py, infer.py,
                coco.py, registry.py, profile.py, unet.py
  analyze/      pipeline.py, fit.py, report.py, study.py, suite.py, validate.py, compare.py,
                sensitivity.py, archive.py, combine.py, exclusions.py, overlay.py,
                spread.py, power.py, summary.py (shareable HTML page)
  gui/          main_window.py + pages/ (experiment, camera, capture, setup, prompt, analyze,
                report, labels, train, study, validation, doctor)
  ui/           interactive.py (OpenCV click tools for the command line)
  cli.py  config.py  storage.py  quality.py  provenance.py
tests/          synthetic scenes with known answers (dye, curved stems, tilted fields);
                real-image regression checks run through `fungus validate-suite`
```

Segmenters share one interface, `segment(frames, prompts) -> masks`. That makes it easy to swap the color-threshold version, SAM 2 and a later custom model, and to compare them.

---

## 4. Components in detail

### 4.1 Capture
- `fungus cameras`: list the cameras found, with their resolutions.
- `fungus capture --config config.yaml`: take a picture every *N* seconds or minutes.
  - **Lock exposure, white balance and focus** where the camera allows it. Auto-exposure changing between frames is the biggest source of false "movement". Store the actual settings with every frame.
  - Throw away a few warm-up frames each time before saving.
  - Reconnect automatically if the camera drops. Log any gaps. Check free disk space.
  - Take the picture from a clock-aligned schedule, so timing errors don't add up over time. Timestamps are in UTC.
  - Optional: keep the camera closed between shots (cooler, and it recovers better).
- Long unattended runs: `fungus service install` writes and registers the service (Task Scheduler on Windows, launchd on macOS, systemd user unit on Linux) so capture survives a reboot, after showing what it will do.
- Every round writes a heartbeat (`status.json`); `fungus health` says whether capture is alive, photos are arriving and usable, and disk is left, and `--watch` alerts through a webhook or any command when that changes.
- `fungus import <folder>`: bring in phone, trail-camera or Raspberry Pi images, using EXIF time or the filename as the timestamp. `--watch` keeps importing from a synced folder, skipping photos already imported and files still being copied.
- Note for macOS: the terminal or Python must be granted camera access in System Settings; `fungus doctor` explains the state and the fix.
- Several cameras are captured together and measured separately (`results/cameras/<name>/`), and `fungus combine` merges their views of one object.

### 4.2 Preprocessing
- **Registration:** line every frame up with frame 0. Use ArUco markers if present, otherwise ECC or feature matching (ORB). A camera or pot moved by a few pixels would otherwise look like growth.
- **Scale:** a printed ArUco marker (or ruler) of known size in the scene converts pixels to mm. For a field shot at an angle, 4 markers give a **top-down (homography) view**, so areas are measured correctly.
- **Lighting:** normalize to a gray/color card or a fixed patch of the scene. Flag frames that are too dark or blurry and skip them instead of letting them corrupt the data. Frames can also be excluded by hand with a reason (`exclusions.json`), which every report and study honours and lists.

### 4.3 Segmentation
1. **Color threshold (dye test):** blue in HSV space, cleaned up with morphology. Fast, easy to understand, and a check on SAM's results.
2. **SAM 2 video mode (main approach):**
   - `fungus annotate`: shows frame 0 (or the first frame where the target shows up) in an OpenCV window. The user clicks positive/negative points or draws a box for the **target** (dye or moss) and the **reference** (towel, stem, plot).
   - The video predictor carries both masks through the image sequence. Add corrections on later frames if a mask drifts.
   - Long runs: process in chunks with a little overlap. Use the smallest SAM 2 model that is accurate enough, so it runs on a laptop CPU or MPS.
   - Tracking memory is saved with the run and resumed, so a new frame costs one frame of tracking however long the time-lapse is.
3. **Text prompting (optional):** "moss", "plant stem". Not built; clicking has been enough so far.
4. **Trained model (built, M7/M10):** start from SAM or colour masks, correct them with a brush or SAM clicks in `fungus label`, and fine-tune a U-Net per use case. `fungus dataset suggest` picks the frames worth labeling next (where the model is unsure, or two methods disagree) instead of evenly spaced ones.
5. **Uncertainty (M8):** every frame is also segmented with narrower and wider settings; the spread becomes part of each measurement's uncertainty, so a soft edge is reported as less certain than a sharp one.

Moss and dye spread gradually, so the timeline is used as a check: flag any frame where the mask area jumps or shrinks suddenly.

### 4.4 Measurement (target relative to reference)
- **Axis:** from the reference mask, compute the main axis. For a towel or stem, take the skeleton or center line from base to tip (PCA for a straight object, skeleton for a curved stem). Base = waterline or soil line (clicked once, or taken from a marker).
- **Extent along the axis:** project the target pixels onto the axis and record the furthest point from the base, in px, mm and % of the reference length. To ignore stray pixels, use a high percentile (e.g. the 98th) rather than the absolute maximum.
- **Area and coverage:** target area inside the reference area (cm² and %). This is the main field metric.
- **Front speed:** change in extent per hour or day between frames.
- For the field: the distance the edge has moved from the starting patch (distance transform from the frame-0 mask), and the area reached over time.

### 4.5 Analysis and outputs
- `measurements.csv`: one row per frame and object.
- Fitted models and rates:
  - Dye wicking should follow **Lucas–Washburn, h ∝ √t**. This is a **good check that the whole pipeline works** before moving to moss.
  - Moss or infection: linear, power, **logistic**, Gompertz (lag and maximum rate), Richards and a √t with an estimated start time. AICc weights say which fits best.
  - Frames of one time-lapse are not independent; AR(1) errors are fitted where the data says so, and intervals come from a bootstrap.
- Plots: extent and coverage vs. time with the fits, a quality-check plot, and the best model with its residuals in units of the reported uncertainty. PNG for looking at, PDF/SVG for a paper.
- An overlay time-lapse video (masks, axis and front line drawn on each frame) for checking the results by eye. This is the fastest way to spot errors.
- Across replicates: `fungus study` fits each replicate, compares conditions with the replicate as the unit of inference, and writes a methods draft with the actual settings.
- `fungus archive` packs settings, results and hashes into one file for a data repository; `fungus sensitivity` says how much each debatable setting would move the result.

---

## 5. Milestones

✅ = done and checked. 🟡 = built and tested on synthetic data, waiting on real images, real hardware or another person to confirm it.

| # | Milestone | Done when |
|---|---|---|
| M0 ✅ | Project skeleton: packaging, CLI, config, CI running tests on Win/macOS/Linux (GitHub Actions) | `fungus --help` works on all 3 OSes |
| M1 🟡 | **Capture**: camera list, interval capture with locked settings, metadata, reconnect, import command | An 8-hour unattended run on Windows and macOS with no missed frames *(code done, plus permission handling, `fungus doctor`, a heartbeat, `fungus health` and `fungus service install`; the hardware run itself is still needed)* |
| M2 🟡 | **Dye test, simple version**: HSV threshold + ArUco scale + extent along the towel axis + CSV and plot | Measured height matches a ruler to within ~2 mm; the √t fit looks reasonable *(code done and verified on synthetic data to <0.5 mm, recovering n = 0.50; still needs a real dye run)* |
| M3 🟡 | **SAM 2 integration**: annotate tool, video propagation, device auto-detection, `Segmenter` interface | SAM 2 dye masks agree with the threshold masks (IoU > 0.9) *(done on synthetic data: mean IoU 0.991, min 0.913; still needs a real dye run. Built on HF `transformers`, streaming, cropped to the region)* |
| M4 🟡 | **Robustness**: registration, lighting check, bad-frame skipping, jump detection, overlay video | Bumping the camera or turning a lamp on mid-run does not create fake growth *(done on synthetic data: multi-marker perspective rectification (tilt error 9.6 → 0.26 mm), per-channel lighting correction from a neutral patch or the background (dimmed frames −80 mm → <0.1 mm), jump and retreat detection, frame_flags.csv; needs real-scene validation)* |
| M5 🟡 | **Moss on a plant**: stem skeleton axis, soil-line base, % of stem, logistic fit | Matches hand measurements on ~20 hand-labeled frames *(done on synthetic curved stems: centerline length within 0.5%, extent within 1.5 px; `fungus validate` gives Bland–Altman agreement with hand measurements. Needs real moss + ~20 hand-measured frames)* |
| M6 🟡 | **Field mode**: 4-marker top-down view, coverage and edge advance, multiple plots per image | Area error < ~10% against hand-drawn outlines *(done on a synthetic tilted field: area error +0.9 to +2.7% vs true outlines, 9% without rectification; named plots, per-plot reports/validation, edge advance, GCC/RCC/ExG colour indices. Needs a real field)* |
| M7 🟡 (core) | **Trainable detectors:** labeling workflow, fine-tuned model per use case, versioned models | Beats SAM alone on held-out moss frames; a new use case can be trained from labeled images *(workflow done: dataset export/add-pairs, brush editor, U-Net with group/time-aware validation, model cards, `evaluate`, `method: model`. Verified on synthetic dye only; needs real labeled moss)* |

Everything below was added after the first plan, to make the results publishable and the runs survivable.

| # | Milestone | Done when |
|---|---|---|
| M8 🟡 | **Honest uncertainty:** segmentation uncertainty per frame (each frame re-segmented narrower and wider), combined with scale, pixel and alignment terms; `fungus validate-suite` to re-run every accuracy check against hand labels with a baseline | The reported uncertainty covers the real error on hand-measured frames *(code done; `validate` reports the share of differences within 2u and the RMS z. Needs real hand measurements)* |
| M9 ✅ | **Statistics for a paper:** AICc and Akaike weights, AR(1) frame errors by generalized least squares, bootstrap intervals, reduced χ², Durbin–Watson, lag models (sqrt_lag, Gompertz, Richards); `fungus study` for replicates and conditions (Welch + Holm, random-effects means, methods draft) | Interval coverage checked by simulation: 0.94 / 0.90 / 0.71 at AR(1) 0 / 0.5 / 0.9, against 0.94 / 0.75 / 0.31 when frames are treated as independent |
| M10 🟡 | **Active learning and SAM-assisted labeling:** `dataset suggest` / `rank` (model uncertainty, or disagreement between two runs), brush + SAM clicks in the label editor | Labeling the suggested frames beats evenly spaced ones on real data *(picks the right frames on synthetic data; SAM click gave IoU 0.997 from one click. Needs real moss to show the gain)* |
| M11 ✅ | **Incremental SAM 2:** tracking memory saved with the run and resumed | A new frame costs one frame of tracking, and the masks are identical to re-tracking from the start (checked with the real model: frame 61 took 1.1 s instead of 56 s) |
| M12 ✅ | **Desktop app covers the whole workflow:** SAM prompts, labeling, training, studies, validation, live measuring during capture, excluding frames by hand | Every command-line feature has a page; GUI tests drive each page like a user (a crash in background tasks was found and fixed this way) |
| M13 🟡 | **Reproducibility and export:** `fungus archive` bundles (hash manifest + package versions) with `--verify`, `fungus sensitivity` (how much each debatable setting moves the result, in units of the reported uncertainty), vector figures and a residual panel | A colleague reproduces a result from the bundle alone *(bundles and checks work; not yet tried by another person on another machine)* |
| M15 ✅ | **Planning, model hygiene and a demo:** rates, lag and time-to-level with bootstrap intervals and 95% bands on the charts; `fungus power` (replicates needed, simulating the study's own Welch test); `fungus models` registry with `unfamiliar_input` drift warnings; resumable training with early stopping; `fungus demo` | Checked against theory and by simulation: band coverage 91–94% (independent errors), power matches the noncentral t, resumed training gives the same weights. The demo exposed and fixed a false "blurry" flag caused by growth itself |
| M16 🟡 | **Spread maps, label exchange and multi-class models:** `fungus spread` (arrival-time map, front speed by direction with standard errors); COCO export/import for CVAT and Label Studio; one model for several overlapping classes (stem and moss) with per-class thresholds and metrics | Speeds by direction within 10% on a synthetic anisotropic patch; COCO run-length code byte-identical to pycocotools; a two-class model learns stem and moss on synthetic plants. Needs real moss on real stems to confirm the gain over two separate passes |
| M17 🟡 | **Whole-field colour, sharing and app parity:** named colour classes (`pick-color --class`) give each plot's share of healthy / yellowing / brown per frame with an uncertainty from moving the class boundaries; `fungus summary` writes one self-contained HTML page; the app's Report page runs spread maps, sensitivity checks and the shareable page, and the Capture page watches run health | On a synthetic field browning at a steady rate through a lighting change, shares within 1.5 points of the truth and the fitted rate within 5%. Needs a real discolouring field (Target C) to confirm the classes hold across days and weather |
| M14 🟡 | **Long runs and the field:** several cameras measured separately and combined, capture heartbeat + `fungus health` with webhook/command alerts, `fungus service install`, `fungus import --watch` for synced photos | An unattended field run reports its own problems *(code done; the service registration and webhook alerts are untested against a real OS service and a real endpoint)* |

---

## 6. Testing strategy
- **Synthetic tests (done, ~280 of them):** dye on a towel, curved swaying stems, tilted fields and soft edges, all with known answers, run in CI on Windows, macOS and Linux. Desktop-app tests drive each page like a user on an offscreen display.
- **Real-model tests** (opt in with `FUNGUS_TEST_SAM=1`): SAM 2 against the colour threshold, and resumed tracking being identical to a full re-track.
- **Simulation** for the statistics: interval coverage measured over hundreds of simulated time-lapses rather than assumed.
- **Small set of real test images (still needed):** ~10 dye frames and a few moss frames with hand-drawn masks, plus ~20 hand measurements. `fungus validate-suite` runs them all and compares with a saved baseline, so a change to the code cannot quietly shift results. This is the main gap.
- **Hardware tests** (not in CI): a capture smoke test by hand on each OS, and an 8-hour unattended run watched by `fungus health`.

---

## 7. Risks and open questions

| Risk | Mitigation |
|---|---|
| Auto exposure or white balance makes masks flicker | Lock camera settings, use a gray card, check brightness, smooth over time |
| Camera or subject moves over days or weeks | Rigid mount, ArUco markers, registration step |
| Day/night and changing sunlight (field) | Capture at fixed times of day, a light source you control, skip bad frames |
| Moss looks too much like bark or soil for SAM without a trained model | Built (M7/M10): correct SAM masks with a brush or SAM clicks, fine-tune a U-Net, and let `dataset suggest` pick the frames worth labeling |
| SAM 2 is slow on CPU over thousands of frames | Solved for repeat runs: tracking resumes from saved memory, so each new frame costs one frame. The first pass is still slow on a CPU; use the smallest model that works |
| Settings chosen by hand (thresholds, alignment, lighting) quietly decide the result | `fungus sensitivity` re-runs with each one changed and reports the effect in units of the reported uncertainty |
| A long run dies unnoticed (camera unplugged, disk full, computer asleep) | Heartbeat + `fungus health --watch` with alerts; capture registered as a service that restarts |
| Webcams are low resolution, which limits mm accuracy | Choose the camera by mm/pixel needed; support importing images from better cameras |
| Webcam driver differences between OSes | Per-OS backends, `fungus cameras` to diagnose, import mode as a fallback |

**Questions asked at the start, and where they stand:**
1. ~~Image rate and duration~~ — seconds to minutes for dye, hours for real runs. Both work; capture is clock-aligned and analysis is incremental.
2. **Field capture hardware — still open.** Any of webcam + laptop, Raspberry Pi, trail camera or phone works: the Pi path and `import --watch` are built and documented, but nothing has been tried outdoors.
3. ~~What "infection" means on the plant~~ — both height reached and % of stem covered are measured (`extent_mm`, `covered_length_pct`, `reference_covered_pct`).
4. ~~GPU~~ — a modest GPU is available; CUDA, Apple GPU and CPU are picked automatically.
5. ~~How precise~~ — paper-level, so markers and calibration are required and every result carries an uncertainty and a record of how it was produced.

**Open now:**
- How many replicates per condition? `fungus power` now answers this from a pilot study's spread; it needs a pilot.
- Which metric is the headline for each use case, so the validation suite can hold it to a bound.
- Whether to keep the raw photos in the reproducibility bundle or deposit them separately (only their hashes are in the bundle by default).
