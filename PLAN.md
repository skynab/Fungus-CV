# Fungus-CV — Plan

Time-lapse capture plus computer-vision measurement of how far a "spreading" region (dye, moss) has moved relative to a reference object (paper towel, plant stem, field plot).

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
| Optional UI later | Streamlit or napari | Browse frames, fix masks, view charts |

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

### Experiment folder layout
```
experiments/2026-09-20_dye_test/
  config.yaml            # camera, interval, prompts, color ranges, marker size…
  frames/                # 2026-09-20T14-05-00Z_cam0.jpg  (UTC, sortable)
  frames.csv             # timestamp, camera, exposure, focus, file hash, flags
  prompts.json           # clicks/boxes for target + reference on frame 0
  masks/target/…png  masks/reference/…png
  results/measurements.csv
  results/plots/*.png  results/overlay_timelapse.mp4
```

### Code layout
```
fungus_cv/
  capture/      camera.py (per-OS backends, list cameras), scheduler.py, importer.py
  preprocess/   align.py (ArUco/ECC registration), calibrate.py (px→mm), color.py
  segment/      base.py (Segmenter interface), sam2_video.py, threshold.py, text_prompt.py
  measure/      axis.py, extent.py, area.py
  analyze/      fit.py (sqrt / linear / logistic), report.py, overlay.py
  cli.py  config.py  storage.py
tests/          synthetic images with known answers + a small real-image fixture set
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
- Long unattended runs: document how to run capture as a background service (Task Scheduler on Windows, launchd on macOS, systemd on Linux) so it survives a reboot.
- `fungus import <folder>`: bring in phone, trail-camera or Raspberry Pi images, using EXIF time or the filename as the timestamp.
- Note for macOS: the terminal or Python must be granted camera access in System Settings.

### 4.2 Preprocessing
- **Registration:** line every frame up with frame 0. Use ArUco markers if present, otherwise ECC or feature matching (ORB). A camera or pot moved by a few pixels would otherwise look like growth.
- **Scale:** a printed ArUco marker (or ruler) of known size in the scene converts pixels to mm. For a field shot at an angle, 4 markers give a **top-down (homography) view**, so areas are measured correctly.
- **Lighting:** normalize to a gray/color card or a fixed patch of the scene. Flag frames that are too dark or blurry and skip them instead of letting them corrupt the data.

### 4.3 Segmentation
1. **Color threshold (dye test):** blue in HSV space, cleaned up with morphology. Fast, easy to understand, and a check on SAM's results.
2. **SAM 2 video mode (main approach):**
   - `fungus annotate`: shows frame 0 (or the first frame where the target shows up) in an OpenCV window. The user clicks positive/negative points or draws a box for the **target** (dye or moss) and the **reference** (towel, stem, plot).
   - The video predictor carries both masks through the image sequence. Add corrections on later frames if a mask drifts.
   - Long runs: process in chunks with a little overlap. Use the smallest SAM 2 model that is accurate enough, so it runs on a laptop CPU or MPS.
3. **Text prompting (optional):** "moss", "plant stem". Useful when there are many plants or photos, so nobody has to click on each one.
4. **Fallback if moss is too subtle:** export SAM's masks, fix them in a labeling tool (e.g. CVAT or Label Studio), and fine-tune a small segmentation model. Moss on bark or soil has low contrast, so plan for this possibility.

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
  - Moss or infection: linear, exponential and **logistic** (sigmoid) fits, reporting growth rate, doubling time and lag.
- Plots: extent and coverage vs. time, with the fit and confidence band.
- An overlay time-lapse video (masks, axis and front line drawn on each frame) for checking the results by eye. This is the fastest way to spot errors.

---

## 5. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 ✅ | Project skeleton: packaging, CLI, config, CI running tests on Win/macOS/Linux (GitHub Actions) | `fungus --help` works on all 3 OSes |
| M1 🟡 | **Capture**: camera list, interval capture with locked settings, metadata, reconnect, import command | An 8-hour unattended run on Windows and macOS with no missed frames *(code done; hardware test still needed)* |
| M2 🟡 | **Dye test, simple version**: HSV threshold + ArUco scale + extent along the towel axis + CSV and plot | Measured height matches a ruler to within ~2 mm; the √t fit looks reasonable *(code done and verified on synthetic data to <0.5 mm, recovering n = 0.50; still needs a real dye run)* |
| M3 🟡 | **SAM 2 integration**: annotate tool, video propagation, device auto-detection, `Segmenter` interface | SAM 2 dye masks agree with the threshold masks (IoU > 0.9) *(done on synthetic data: mean IoU 0.991, min 0.913; still needs a real dye run. Built on HF `transformers`, streaming, cropped to the region)* |
| M4 🟡 | **Robustness**: registration, lighting check, bad-frame skipping, jump detection, overlay video | Bumping the camera or turning a lamp on mid-run does not create fake growth *(done on synthetic data: multi-marker perspective rectification (tilt error 9.6 → 0.26 mm), per-channel lighting correction from a neutral patch or the background (dimmed frames −80 mm → <0.1 mm), jump and retreat detection, frame_flags.csv; needs real-scene validation)* |
| M5 🟡 | **Moss on a plant**: stem skeleton axis, soil-line base, % of stem, logistic fit | Matches hand measurements on ~20 hand-labeled frames *(done on synthetic curved stems: centerline length within 0.5%, extent within 1.5 px; `fungus validate` gives Bland–Altman agreement with hand measurements. Needs real moss + ~20 hand-measured frames)* |
| M6 🟡 | **Field mode**: 4-marker top-down view, coverage and edge advance, multiple plots per image | Area error < ~10% against hand-drawn outlines *(done on a synthetic tilted field: area error +0.9 to +2.7% vs true outlines, 9% without rectification; named plots, per-plot reports/validation, edge advance, GCC/RCC/ExG colour indices. Needs a real field)* |
| M7 🟡 (core, can be started after M3) | **Trainable detectors:** a labeling workflow (start from SAM masks, correct them by hand), fine-tune a small segmentation model for each use case, keep versioned models; optional Streamlit dashboard | Beats SAM alone on held-out moss frames; a new use case can be trained from a set of labeled images *(workflow done: dataset export/add-pairs, brush label editor, U-Net training with group/time-aware validation, model cards, evaluate, `method: model`. Verified end to end on synthetic dye only; needs real labeled moss. Dashboard not started)* |

---

## 6. Testing strategy
- **Synthetic tests:** draw a "stem" and a colored region of known height or area in code, check the measurements match. Fast and run in CI on every OS.
- **Small set of real test images:** ~10 dye frames and a few moss frames with hand-drawn masks, used for regression checks of IoU and extent error.
- **Hardware tests** (not in CI): a capture smoke test run by hand on each OS.

---

## 7. Risks and open questions

| Risk | Mitigation |
|---|---|
| Auto exposure or white balance makes masks flicker | Lock camera settings, use a gray card, check brightness, smooth over time |
| Camera or subject moves over days or weeks | Rigid mount, ArUco markers, registration step |
| Day/night and changing sunlight (field) | Capture at fixed times of day, a light source you control, skip bad frames |
| Moss looks too much like bark or soil for SAM without a trained model | Plan for M7: fine-tune with SAM masks corrected by hand |
| SAM 2 is slow on CPU over thousands of frames | Smallest model that works, process in chunks, lower resolution for tracking and full resolution only for measurement |
| Webcams are low resolution, which limits mm accuracy | Choose the camera by mm/pixel needed; support importing images from better cameras |
| Webcam driver differences between OSes | Per-OS backends, `fungus cameras` to diagnose, import mode as a fallback |

**Questions to decide early:**
1. Expected image rate and duration (minutes for dye vs. hours or days for moss). This decides the default interval and storage needs.
2. Field capture hardware: webcam + laptop, Raspberry Pi, or trail camera or phone photos?
3. What does "infection" mean on the plant? Height reached, % of stem covered, or both?
4. Is a GPU available, or must everything run on a laptop CPU or Apple Silicon?
5. How precise must the results be (mm-level vs. rough trends)? This decides whether markers and calibration are required.
