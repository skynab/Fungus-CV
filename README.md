# Fungus-CV

Take time-lapse photos with a webcam, then measure how a spreading region (dye, moss, an infection) moves relative to a reference object over time. See [PLAN.md](PLAN.md) for the full roadmap.

**Status:** capture (M1), color-threshold measurement (M2), SAM 2 segmentation (M3), robustness to tilt/lighting/jumps (M4), measuring along curved stems (M5), field plots (M6) and training your own models (M7) all work. Field plots (M6) work too.

Works on Windows, macOS and Linux (Python 3.10+).

## Try it without a camera

```bash
fungus demo experiments/demo            # a synthetic dye time-lapse, annotated and scaled
fungus analyze experiments/demo
fungus report experiments/demo --model sqrt --model sqrt_lag --model power
fungus validate-suite experiments/demo/suite.yaml
fungus demo experiments/demo-study --study   # replicates in two conditions + study.yaml
```

The demo is a paper towel strip wicking blue dye, h = 12 mm·√(t − 0.5 min). It is built to be realistic: the wet front fades over a few millimetres and is uneven across the strip, there is camera shake, a slow change in room light, and sensor noise. The true height of every photo is in `demo_truth.csv`, and it comes with hand measurements and a validation suite, so every command has something to work on. In the app: File → Make Demo Experiment…

## Desktop app

Fungus-CV has a desktop app with a page for each step, grouped in the sidebar:

| Section | Page | What it does |
|---|---|---|
| Capture | **Experiment** | open or create an experiment, key settings |
| | **Camera** | live preview for framing and focus |
| | **Capture** | scheduled photos with progress; optionally **measures new frames as they arrive** and charts them live |
| Measure | **Set up measurement** | click the base, stem path, region or field plots; drag over colours |
| | **SAM prompts** | click on the target (or the stem) and see SAM 2's mask; save prompts per frame |
| | **Analyze** | run the analysis; per-frame results shown as the overlay, the aligned frame with its mask, or the photo as taken; **exclude a frame by hand with a reason** |
| | **Report** | fits with standard errors, bootstrap intervals, AICc weights and warnings; charts |
| Models | **Labels** | open or create a dataset; add frames evenly or where a model is least sure; brush and SAM clicks; mark reviewed; rank with a model |
| | **Train models** | train with a live loss / validation IoU chart, read the model card, use the model for the experiment, evaluate on other labeled data |
| Results | **Study** | edit a study (replicates and conditions), run it, read condition summaries, comparisons, figures and the methods draft |
| | **Validation** | Bland–Altman agreement with hand measurements; run the validation suite and save its baseline |
| This computer | **Diagnostics** | camera permission, cameras, GPU, disk |

```bash
pip install -e ".[gui]"
fungus gui                       # or: fungus gui experiments/dye-test-1
```

To build a standalone app (`Fungus-CV.app` on macOS, `Fungus-CV.exe` on Windows), see [packaging/README.md](packaging/README.md). **On macOS, the packaged app is the most reliable way to use the camera.** It has its own entry under Privacy & Security → Camera, and macOS asks for permission the first time. When `fungus` runs from a terminal, the permission belongs to that terminal app instead.

Everything in the app can also be done from the command line below, which is handy for scripts and remote machines.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

(With [uv](https://docs.astral.sh/uv/): `uv venv && uv pip install -e ".[dev]"`.)

## Quick start

Check the computer first. `fungus doctor` checks camera permission, the cameras the system reports, window support for `preview`/`annotate`, optional model dependencies, the GPU and disk space, and says how to fix each problem.

```bash
fungus doctor
```


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
4. **Uncertainty:** `extent_mm_unc` is a combined standard uncertainty from the marker scale, pixel size, alignment residual and segmentation (where the edge is placed); see *Segmentation uncertainty* below.
5. **Flag** frames that are doubtful: `align_failed`, `align_poor`, `no_target`, `brightness_changed`, `blurry`.

Results go to `results/measurements.csv`, with a mask and an overlay image per frame.

- **Provenance:** `results/run_info.json` records the scale, the software versions and every setting used. Each row has a `settings_hash`.
- **Settings changes:** if you change the settings, old results are moved to `results/archive/`, never mixed with new ones.
- **Re-runs:** only new frames are measured.

`report` fits these models to the data (the first four by default; pick others with `--model`, repeatable):

| Model | Formula | Use |
|---|---|---|
| linear | `a + b·t` | constant speed |
| sqrt | `k·√t` | capillary wicking (Lucas–Washburn) |
| power | `a·tⁿ` | **n ≈ 0.5 confirms wicking behavior**, so it's a good end-to-end check |
| logistic | `K / (1 + e^(−r(t − t_mid)))` | growth that levels off, e.g. infection |
| sqrt_lag | `k·√(t − t_lag)` | wicking when the exact start time isn't known |
| gompertz | `A·exp(−exp(μe/A·(λ − t) + 1))` | growth with a lag: μ = maximum rate, λ = lag time (Zwietering) |
| richards | `K / (1 + ν·e^(−r(t − t_mid)))^(1/ν)` | asymmetric S-curve; ν = 1 is the logistic |

For each model it reports:

- **Parameters:** estimate ± standard error [95% bootstrap interval]. The bootstrap uses 1000 refits by default; set `--bootstrap 0` to skip it. Frames are weighted by their uncertainty.
- **Model comparison:**
  - **AICc** (AIC corrected for small samples; lower is better).
  - **Akaike weight:** the share of evidence for each model among those fitted; the best model has the highest.
  - R² for reference.
- **Correlated errors (`--errors auto`):**
  - Consecutive frames often share errors (lighting drift, a slowly moving shadow). Fitting as if they were independent makes the intervals far too narrow.
  - Each model is therefore fitted both with independent errors and with first-order autoregressive (AR(1)) errors, using generalized least squares as in R's `gnls` with `corAR1`. AICc keeps the better of the two.
  - The correlation is defined per typical interval between photos, so uneven intervals are handled.
  - The bootstrap rebuilds correlated errors from the fitted AR(1) process.
- **Diagnostics:**
  - **χ²/dof:** about 1 means the per-frame uncertainties explain the scatter. Much more than 1 means something is missing.
  - **Durbin–Watson:** about 2 means residuals are independent. Well below 1 means streaks, usually a model that doesn't fit.
  - Plain-language warnings for either problem.

**Numbers read off the fitted curve**, each with a 95% interval from the same bootstrap refits:

- `max_rate`: the steepest rise (e.g. mm per hour), and `t_max_rate`, when it happens.
- `lag` (growth curves: logistic, Gompertz, Richards, sqrt_lag): where the tangent at the steepest point meets the level at the first frame, the usual definition of a lag phase.
- `time_to_<value>`: when the curve first reaches a value, with `--time-to 20` (repeatable), e.g. when dye reaches 20 mm. It is left empty if the curve doesn't get there within the data, rather than extrapolated.
- The main chart shades the best model's **95% band**. In simulation it held the true curve 91–94% of the time with independent frame errors, and 83–88% with strongly correlated ones (60 frames).
- Studies can compare these too: `params: [max_rate, lag, time_to_20]` in `study.yaml`.

**How well the intervals work.** On 150 simulated logistic time-lapses of 60 frames, 95% intervals for `t_mid` contained the true value this often:

| Frame errors | Treating frames as independent | `--errors auto` |
|---|---|---|
| independent | 94% | 94% |
| AR(1), correlation 0.5 | 75% | 90% |
| AR(1), correlation 0.9 | 31% | 71% |

With very strongly correlated errors there are few effectively independent frames, so intervals can still be too narrow. The fit warns when this happens; more replicates help more than more frames.

Output files:

- `results/report/`: the main plot, a quality-check plot, **`fit_and_residuals`** (the best model with its residuals in units of the reported uncertainty underneath — residuals should scatter within about ±2 with no pattern), `fits.json` (every statistic above) and an optional overlay video.
- Frames where the front moves back by more than 3σ are circled as worth checking.
- **Figures for a paper:** `--format pdf --format svg` (repeatable, `png` by default) writes vector figures that stay sharp at any size; `--dpi` sets the png resolution. In the app, tick "Also save PDF and SVG". Study figures are always written as png, pdf and svg.

## How much do the settings matter?

```bash
fungus sensitivity experiments/dye-test-1 --list     # what would be varied
fungus sensitivity experiments/dye-test-1            # re-run with each setting changed
```

Every variant changes **one** setting a reasonable person might have chosen differently, measures every frame again into `results/sensitivity/<variant>/`, and compares it with your current settings frame by frame. The experiment's own results are never touched.

| Setting varied | Variants |
|---|---|
| colour thresholds | wider and narrower by `analysis.uncertainty.hsv_delta`; no morphological clean-up |
| trained model | probability threshold 0.35 and 0.65 |
| SAM 2 | mask threshold ±1 |
| alignment | markers vs image matching (ECC) |
| lighting | correction on vs off |
| perspective | rectification on vs off (when markers are measured) |
| front position | 25th and 75th percentile across the width instead of the median |

- **The number that matters** is the mean change **in units of the reported uncertainty**. Well under 1 u means the choice doesn't change the conclusions; several u means it must be justified in the methods, or the range reported.
- **Outputs:** `results/sensitivity/sensitivity.csv` (mean, max and signed change, change at the last frame, change in u), `sensitivity.json` (with the full baseline settings) and a bar chart with the 1 u line.
- **Cost:** each variant measures every frame again. That is quick for colour thresholds and slow for SAM 2; `--variant` runs just one.
- Variant masks and overlays are deleted afterwards unless you pass `--keep-masks`.

## Archiving a result (for a paper or a data repository)

```bash
fungus archive experiments/dye-test-1 dye-test-1.zip            # settings + results
fungus archive experiments/dye-test-1 dye-test-1.zip --frames   # also the photos
fungus archive studies/moss.yaml moss-study.zip                 # a study and every replicate
fungus archive --verify dye-test-1.zip                          # check it later
```

The bundle holds `config.yaml`, `frames.csv`, annotations, hand exclusions, SAM prompts, measurements, reports, study results and the card of every trained model used.

- **Manifest:** `manifest.json` lists every file with its SHA-256 hash, **every photo's hash whether or not the photos are included**, the model weights' hash, and the versions of every installed package plus the fungus-cv version and git commit.
- **Left out by default:** masks and overlays (`analyze` recreates them), SAM tracking state, and the photos. Add them with `--masks`, `--weights` and `--frames`.
- **Checking:** `--verify` re-hashes everything and reports files that changed or went missing, so a download or a long-term archive can be trusted. Exit code 1 if anything fails.
- **In the app:** File → Archive Experiment…

## Studies: replicates and conditions

A study compares conditions (e.g. treated vs control) across replicate experiments:

```bash
fungus study studies/moss-treatment.yaml --init   # example file to edit
fungus study studies/moss-treatment.yaml
```

The file lists each experiment (or field plot) with its `condition`, an optional `replicate` name and an optional `t0`, such as the inoculation time. It also sets the `metric`, the `model` whose parameters are compared, and optionally which `params`, a `reference` condition and extra models for `also_fit`.

**How it's analyzed** (two stages, the standard approach for replicated growth curves):

1. Every replicate is fitted separately, with the same settings as `report`: weighting, AR(1) errors, bootstrap intervals.
2. The replicates' parameter estimates are then the data. **The replicate is the unit of inference**, so hundreds of frames from one time-lapse never count as hundreds of samples.

**Outputs** go to `<study>_results/`:

| File | Contents |
|---|---|
| `replicates.csv` | per replicate: every parameter with SE and 95% interval, R², AICc, χ²/dof, Durbin–Watson, warnings, settings hash |
| `conditions.csv` | per condition and parameter: n, mean, SD, SE, t-based 95% CI, median, range; random-effects (DerSimonian–Laird) mean using each replicate's own uncertainty, with between-replicate SD τ and I² |
| `comparisons.csv` | each condition vs the reference (or all pairs): difference with 95% CI, Welch's t, df, p, Holm-adjusted p (within each parameter), Hedges' g |
| `model_selection.csv` | AICc and Akaike weights per replicate, and summed over replicates |
| `data_long.csv` | every frame of every replicate in long format, for R or Python |
| `curves.png/.svg`, `parameters.png/.svg` | replicate data and fits with condition means; parameter estimates by condition |
| `study.json` | everything above plus the software version and git commit |
| `methods.md` | a draft methods paragraph with the actual settings, n per condition and tests, plus any warnings to resolve |

**Checks:**
- **Mismatched analysis settings:** if replicates were analyzed with different segmentation, alignment, lighting, measurement or uncertainty settings, the study warns, because a difference between conditions could come from the analysis.
- **Unusable replicates:** replicates whose fit fails are left out and listed.
- **Too few replicates:** a condition with fewer than 2 usable replicates gets no SD or test.

With 2–3 replicates per condition a t-test has little power. Report the estimates and confidence intervals, not only p-values.

### How many replicates?

```bash
fungus power --effect 0.05 --sd 0.02                           # SD between replicates, known
fungus power --study studies/pilot.yaml --param r --effect 20%  # SDs from a pilot study
fungus power --study studies/pilot.yaml --param max_rate --effect 20% --n 4
```

- **Simulates the study's own test:** Welch's t-test on per-replicate estimates, with the Holm correction for `--comparisons` treated conservatively.
- **Output:** the power for 2 to `--n-max` replicates per condition, the number needed for `--target` (80%) power, and with `--n` the smallest difference that many replicates can detect.
- **From a pilot study:** `--study` takes each condition's SD from `conditions.csv`, and `--effect 20%` means 20% of the reference condition's mean.
- **Checked against theory:** it matches the noncentral t result as replicates increase. At small counts it gives slightly lower power, because Welch's test doesn't assume equal spreads.
- **As a rule of thumb:** a difference of one SD between replicates needs about 17 replicates per condition. Reducing the spread between replicates (consistent inoculation, conditions and timing) buys more than extra replicates.

### Accuracy notes
- Markers must lie in the same plane as the object, and the camera should face that plane square-on. The analysis warns if marker edges disagree by more than 2%, which suggests a tilted view.
- Set `--t0` to the moment the towel touched the dye. Otherwise t = 0 is the first photo.
- The dye-edge position depends on the color thresholds. This is measured for every frame; see *Segmentation uncertainty*.

### Segmentation uncertainty

A sharp edge lands in the same place whatever the exact threshold, but a fading one doesn't. So each frame is also segmented once narrower and once wider, and the spread of the results goes into the uncertainty.

| Method | Narrower / wider | Setting |
|---|---|---|
| color | every HSV bound moved in / out (H, S, V) | `analysis.uncertainty.hsv_delta: [4, 20, 20]` |
| model | probability threshold + / − | `probability_delta: 0.1` |
| sam2 | `mask_threshold` + / − on SAM's logits | `logit_delta: 1.0` |

- **Cost:** the model's probabilities and SAM's logits are computed once and thresholded three times, so this is almost free. Colour thresholds run three times.
- **Statistics:** the nominal, narrow and wide values are taken as the bounds of a rectangular distribution: u = (max − min) / (2√3) (GUM type B).
- **Columns:**
  - `extent_mm_seg_unc` and `extent_px_seg_unc`: the segmentation term on its own. It is also added in quadrature to `extent_mm_unc`.
  - `target_area_mm2_unc`: scale (counted twice, since area scales with its square) and segmentation.
  - `coverage_pct_unc`: segmentation only.
- **Reports:** `report` weights fits by `<metric>_unc` for any metric that has one, so area fits are weighted too.
- **Limits:**
  - The deltas are a judgement call: choose them as the range of settings you'd consider equally right, and state them in your methods.
  - For `path_source: reference`, the stem centerline is kept fixed across the variants.
  - Turn it off with `analysis.uncertainty.segmentation: false`.
- **Existing results:** the new settings change the settings hash, so existing results are archived and measured again on the next `analyze`. That takes a while for SAM runs.

## Field plots: coverage, spread and colour

For a field (or a bench of pots) seen by one camera:

```bash
fungus markers markers.pdf --size-mm 100 --ids 0,1,2,3   # larger markers for a wide view
# Stake the 4 markers flat on the ground around the plots, visible in every photo.
# config.yaml: markers.size_mm, rectify.enabled: true, lighting.method: patch (or background)
fungus annotate experiments/field-1 --field     # outline each plot: plot1, plot2, ...
fungus pick-color experiments/field-1            # or SAM / a trained model for the patches
fungus analyze experiments/field-1
fungus report experiments/field-1 --metric target_area_mm2          # one folder per plot
fungus report experiments/field-1 --metric edge_advance_p95_mm --plot plot1
fungus report experiments/field-1 --metric gcc_p90                  # greenness, no segmentation
```

- **Top-down view:** with 4 markers and `rectify.enabled: true`, the field is measured as if seen from directly above, so areas are in true mm². Without rectification a single scale is wrong across a tilted view; the synthetic test was off by about 9%.
- **One row per plot per frame** in `measurements.csv` (`plot` column). Plots are named `plot1`, `plot2`, … by `annotate --field`; rename them in `annotations.json`. A plot can also carry its own `base`/`tip`/`path` to measure extent inside it.
- **Per-plot columns:** `target_area_mm2`, `coverage_pct`, `equivalent_radius_mm` (radius of a circle of the same area; useful for radial growth rates).
- **Colour indices,** computed for every plot on the lighting-corrected frame. These track whole-field greenness or discolouration without any segmentation:
  - `gcc_mean`, `gcc_p90` (green chromatic coordinate; its 90th percentile is the standard robust summary in phenology camera studies)
  - `rcc_mean`, the red chromatic coordinate
  - `exg_mean`, excess green
- **Edge advance** (`report --metric edge_advance_p95_mm` or `edge_advance_max_mm`): how far newly affected ground is from the patch's first outline, i.e. the first usable frame with any patch in the plot. It's computed from the saved masks, so it works for any segmentation method.
- **Reports and validation:** `report` writes `results/report/<plot>/` for each plot. `validate --make-template` writes one row per plot per frame, and `--plot` restricts the check to one plot, e.g. to compare areas with outlines traced by hand in ImageJ.

On a synthetic tilted field with two plots and irregular growing patches, rectified areas were within **+0.9 to +2.7%** of the true outlines (the plan's target was <10%). Errors were largest for the smallest patches, where edge pixels matter most.

## Several cameras

List more than one camera in `config.yaml` and each is captured, annotated, measured and reported **on its own**, because each sees a different scene:

```bash
fungus annotate experiments/plant-1 --camera cam1    # annotations_cam1.json
fungus prompt   experiments/plant-1 --camera cam1    # prompts_cam1.json
fungus analyze  experiments/plant-1 --camera all     # results/cameras/<name>/...
fungus report   experiments/plant-1 --camera cam1
fungus combine  experiments/plant-1 --metric extent_mm
```

- **Per-camera files:** `annotations_<camera>.json`, `prompts_<camera>.json` and `results/cameras/<camera>/`. With a single camera nothing changes (`annotations.json`, `results/`).
- `--camera` also works on `runs`, `compare`, `validate` and `dataset export`. In the app, a **Camera** selector appears in the status bar and every page follows it.
- **Combining views** (`fungus combine`): a camera shortens whatever leans toward or away from it, so **every view underestimates a length**. The default `max` takes the longest view, which is closest to the truth and still a lower bound; `--method mean|median` is available for metrics that aren't lengths. Frames within `--tolerance` (60 s) count as the same moment.
- The output CSV keeps each camera's value, which camera was largest, and the **spread** (largest minus smallest). A large spread means the object is far from perpendicular to at least one camera.
- This is a practical fix for foreshortening, not a 3D reconstruction: it does not need calibration between the cameras, and it cannot recover a length that no view sees.

## Moss on a plant stem (curved objects)

Set `analysis.measure.mode: path` to measure **along the stem** by arc length, instead of along a straight line.

- **Clicked path** (`path_source: annotation`): in `fungus annotate`, click the soil line, then several points up the stem to its tip. This suits a stem that doesn't move.
- **Stem segmented in every frame** (`path_source: reference`): set `analysis.reference.method` (color, sam2 or model; same settings blocks as `target`). This follows a stem that bends, sways or grows.
  - **Centerline:** the stem mask plus its moss (moss can cover the stem completely) is thinned to a skeleton. The longest branch from the soil line is kept, so side shoots and leaves are dropped. The far end is extended to the stem's edge, then smoothed over about twice the stem width; unsmoothed pixel tracing overstates length by up to 8%.
  - **SAM prompts:** for `reference.method: sam2`, prompt the stem with `fungus prompt --reference`.
  - **Saved masks:** stem masks are stored in `results/masks/<run>/reference/`, and overlays show the stem outline and the centerline.

Measurements (in addition to the usual columns):

| Column | Meaning |
|---|---|
| `extent_mm` / `extent_max_mm` | how far up the stem the moss reaches (median across the stem width / highest point), by arc length from the soil line |
| `axis_length_mm` | stem length along the centerline |
| `covered_length_mm`, `covered_length_pct` | length of stem with moss beside it, and as % of stem length (patchy infections count only the covered parts) |
| `reference_covered_pct` | % of the stem's area covered by moss |

`corridor_px` ignores moss further than that from the centerline, e.g. on the soil or on a neighboring plant. Use `fungus report --metric covered_length_pct` with the logistic model for infection curves.

On synthetic curved, swaying stems, centerline length was within 0.5% and moss extent within 1.5 px of the truth.

**The stem is measured as it appears in the image.** Parts bending toward or away from the camera look shorter. Use a camera view perpendicular to the plant, or two cameras.

### Checking against hand measurements (for publication)

```bash
fungus validate experiments/moss-1 hand.csv --make-template 20   # 20 evenly spaced frames
# measure those frames by hand (e.g. calipers or ImageJ), fill in the 'value' column
fungus validate experiments/moss-1 hand.csv --metric extent_mm
```

This reports Bland–Altman agreement between automatic and hand values:

- bias with its 95% CI
- 95% limits of agreement
- MAE, RMSE, Pearson r, and the regression line
- the frame with the largest difference

It also writes paired values and an agreement plot to `results/validation/`. If several people measure, add an `observer` column to track who measured each frame.

**Are the uncertainties honest?** When the metric has an uncertainty column (e.g. `extent_mm_unc`), `validate` also reports:
- the share of automatic − hand differences within 2 combined uncertainties (expect about 95%)
- the RMS of the differences divided by their uncertainties (expect about 1)

Put your own reading uncertainty in the optional `value_unc` column; it is combined in quadrature. Much less than 95% means the reported uncertainties are too small, or there is a bias.

### Validation suite: re-check accuracy after every change

Collect your hand-labeled real data once, then re-run every accuracy check with one command:

```bash
fungus validate-suite validation/suite.yaml --init    # example file to edit
fungus validate-suite validation/suite.yaml --save-baseline   # first time, once results look right
fungus validate-suite validation/suite.yaml           # after any code or settings change
```

A suite lists **cases**. Each case runs one or more **checks**, and each check sets bounds on its metrics (`min`, `max`, `abs_max` or `max_drift`):

| Check | Compares | Metrics |
|---|---|---|
| `masks` | the current run's masks with hand-drawn mask PNGs named like the frames | `iou_mean`, `iou_median`, `iou_min`, `extent_diff_mean`, `extent_diff_sd`, `n_frames` |
| `measurements` | measurements with a hand CSV (`validate`) | `bias`, `bias_ci_low/high`, `loa_low/high`, `sd_diff`, `mae`, `rmse`, `pearson_r`, `slope`, `intercept`, `within_2u`, `z_rms`, `n` |
| `model` | a trained model with a labeled test set (`evaluate`) | `iou_mean`, `dice_mean`, `precision_mean`, `recall_mean`, `boundary_f1_2px_mean`, `*_min`, and `unseen_*` for items not used in training |

- **Analysis first:** experiments are analyzed with the current code before checking (`analyze: false` skips that).
- **Baseline:** `max_drift` compares a metric with `baseline.json`, which `--save-baseline` writes. A change that moves IoU or bias more than you allow fails, even when the result is still within its absolute limits.
- **Outputs:** each run writes `suite_results/<time>/results.json` and `summary.csv`. They record every value, its baseline and drift, the pass/fail reasons, the settings hash of each experiment, each model's weights hash, and the fungus-cv version and git commit.
- **Exit code:** 1 if any check fails, so it can gate a merge.
- **From pytest:** `FUNGUS_SUITE=validation/suite.yaml pytest tests/test_suite.py`.
- **Starting set:** about 10 dye frames and 5 moss frames with hand masks, plus 20 hand-measured frames per experiment. Keep the data (or an archive of it) with the suite so results can be reproduced.

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
- **Incremental tracking:** after each `analyze`, SAM 2's tracking memory is saved with the run's masks (`results/masks/<run>/sam2_state/`, about 7–20 MB).
  - **Speed:** the next run resumes from that memory, so a new frame costs one frame of tracking however long the time-lapse is. With SAM 2-tiny on an M1 Pro, frame 61 took 1.1 s instead of 56 s for re-tracking from the prompted frame.
  - **Same results:** resumed masks are pixel-identical to re-tracking from the start, including across a later prompted frame.
  - **Interruptions:** the state is also saved every 25 frames, so a stopped run continues close to where it stopped.
  - **Starting over:** the saved state is only used when the model, crop, prompts, transformers/torch versions and every frame tracked so far still match. Otherwise tracking starts again from the prompted frame and the log says why, for example when older photos were imported in between or an earlier frame needs a mask again. Deleting `sam2_state/` also forces this.
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
fungus label datasets/moss-bark --unreviewed --sam     # m: click with SAM, then touch up
fungus dataset info datasets/moss-bark

# 3. Train (GPU if available) and check on data the model has not seen
fungus train datasets/moss-bark models/moss-bark-v1
fungus evaluate models/moss-bark-v1 datasets/moss-bark-test

# 4. Use it: analysis.target.method: model, analysis.target.model.path: <model folder>
fungus analyze experiments/moss-3

# 5. Improve it: label the frames the model is least sure about, retrain
fungus dataset suggest experiments/moss-3 datasets/moss-bark --model models/moss-bark-v1 --count 20
fungus label datasets/moss-bark --unreviewed --by-priority --sam
fungus train datasets/moss-bark models/moss-bark-v2
```

### Labeling the most useful frames first (active learning)

Evenly spaced frames mostly repeat what the model already gets right. `fungus dataset suggest` scores an experiment's frames and adds the most informative ones as unreviewed items:

| Source | Score | When |
|---|---|---|
| `--model M` | **TTA disagreement**: 1 − IoU between the model's mask and its masks for the image flipped left-right, 15% darker and 15% brighter; **ambiguous fraction**: pixels with probability 0.25–0.75, as a share of those plus the predicted target | after the first model |
| `--model M --run R` | the above plus **disagreement** between the model and run R's masks (e.g. SAM) | the model and SAM disagree |
| `--run A --against B` | **run disagreement**: 1 − IoU between two runs' masks, e.g. SAM vs colour, or two thresholds | first round, no model yet |

- **Score:** the mean of the available components, from 0 (confident) to 1. Scores are relative to the target's size, so compare them within a dataset.
- **Spread in time:** picks are kept apart so near-identical consecutive frames aren't chosen together. Frames already in the dataset are skipped.
- **Candidates:** up to `--max-candidates 200` evenly spaced frames are scored.
- **Starting mask:** run R's mask if given, otherwise the model's prediction.
- **Record:** every candidate's scores go to `<dataset>/suggestions/<experiment>_<time>.csv`. The score is stored on each item (`priority`, `scores` in `dataset.json`).
- **Items already in a dataset:** `fungus dataset rank DATASET --model M` scores unreviewed items the same way. It also scores disagreement between the model and each item's current mask, which is a good way to find a poor SAM pre-label. `--include-reviewed` looks for label mistakes too.
- **Order:** `fungus label --by-priority` shows the highest-scoring items first.

### Clicking with SAM while labeling

`fungus label DATASET --sam` (needs the `sam` extra; `--sam-model` picks the size):

1. Press **m** to switch to SAM clicks.
2. Left-click the target, right-click what isn't, or press **b** for a box. SAM's mask appears as a yellow outline.
3. Press **Enter** to replace the mask, **+** to add to it or **−** to subtract from it.
4. Press **m** again to fix details with the brush.

Undo (`z`) covers SAM edits too. A single click on synthetic dye matched the true area with IoU 0.997 (SAM 2-tiny).

- **Model:** a U-Net with an ImageNet-pretrained ResNet encoder (`resnet34` by default, `resnet18` is faster). It has a full-resolution path so mask edges aren't blurred. Large frames are processed in overlapping tiles.
- **Training data:** only **reviewed** labels are used by default, so model quality depends on labels a person checked. `--include-unreviewed` overrides this.
- **Validation without leakage:** consecutive time-lapse frames are near-duplicates, so validation never mixes them with training.
  - Whole groups (experiments) are held out.
  - With one group, the latest frames are held out.
  - `--val-group` chooses the groups explicitly.
- **Augmentation:** horizontal flips, scale, and small color and brightness changes. Upside-down flips and rotation are off by default, because growth direction often matters; `--flip-vertical` and `--rotate90` turn them on. Raise `--color-jitter` if lighting varies a lot.
- **Model card:** `model.json` records the dataset fingerprint (a hash of every image and mask), the exact train/validation items, the training settings and history, and validation metrics. The threshold is tuned on validation data, which also picks the best checkpoint, so **report accuracy from `fungus evaluate` on separate labeled data**. The analysis settings hash includes the model's weights hash, so results from different models are never mixed.
- **Metrics:** `fungus evaluate` reports IoU, Dice, precision, recall and **boundary F1 within 2 px** (how accurately edges are placed, which matters for front-position measurements) for items used in training and items not used in training, separately.
- **Interrupted training:** a checkpoint is written at every evaluation. `fungus train ... --resume` (or "Resume" in the app) continues from it with the same random state, so the model matches an uninterrupted run. `--patience N` stops early when validation IoU hasn't improved for N evaluations.
- **Which models you have:** `fungus models list [folder]` shows every trained model with its dataset, validation IoU and evaluations; `fungus models show MODEL` prints its card.
- **Noticing a model used on something new:** training records what the training images look like (colour, brightness, contrast, texture). During analysis each frame gets a `model_input_distance` from that profile, and frames far outside it are flagged **`unfamiliar_input`** — other light, another season, another camera. This doesn't say the masks are wrong, only that the model hasn't seen anything like this; check them, and label a few such frames. Give older models a profile with `fungus models profile MODEL DATASET`.
- **Time:** on an M1 Pro, 100 steps of `resnet18` at 256 px took about 25 s. Real datasets will want the default 3000 steps: tens of minutes on a GPU, much longer on a CPU.

## What an experiment folder contains

```
experiments/dye-test-1/
  config.yaml     # settings for this experiment (edit this)
  frames/         # 2026-09-20T14-05-00.123Z_cam0.png ... (UTC time, sorts in time order)
  frames.csv      # one row per image or failed attempt (see below)
  capture.log
  annotations.json  # base, path to tip, region or field plots, neutral patch (`fungus annotate`)
  prompts.json    # SAM clicks (from `fungus prompt`)
  results/        # measurements.csv, run_info.json, report/, archive/,
                  # masks/<run>/, overlays/<run>/, runs/<run>.json, compare/
```

`frames.csv` columns: `timestamp_utc, camera, status (ok/failed), file, sha256, width, height, source, scheduled_utc, lag_s, mean_brightness, sharpness, camera_settings (JSON), notes`.

`mean_brightness` and `sharpness` are recorded so bad frames can be found later, for example when a light was switched on or the camera lost focus.

## Troubleshooting cameras

- **macOS asks per app, not per program.** The permission belongs to the app you start `fungus` from (Terminal, iTerm, VS Code, …).
  - **Undecided:** the first camera command asks, waits up to 60 s for you to click Allow, then continues.
  - **Denied:** turn the app on in System Settings → Privacy & Security → Camera, then restart that app.
  - **"macOS will not show a camera permission prompt for <app>":** some apps (including the Claude desktop app) can't show the prompt. Run the command from **Terminal** instead, or add the app manually in Settings.
- **Windows:** Settings → Privacy & security → Camera → turn on *Camera access* and *Let desktop apps access your camera*. Close Teams, Zoom or the Camera app if they hold the camera, or try `--backend msmf`.
- **Linux:** your user needs to be in the `video` group (`sudo usermod -aG video $USER`, then log out and back in). Interactive windows need a desktop session; over SSH, use `ssh -X`.
- **Which index is which camera:**
  - **Linux:** `fungus cameras` prints the device names.
  - **macOS:** it lists the names macOS reports, which usually follow the same order.
  - **Everywhere:** `fungus preview --index N` shows the live image.
- **A wrong index in `config.yaml`:** `fungus capture` opens every camera before starting and stops with an error if one fails, instead of logging a failure at every scheduled shot. Use `--no-check-cameras` to skip that check.

## Getting consistent images (important for measurements)

- **Camera controls** (`exposure`, `white_balance`, `focus` in `config.yaml`):
  - `lock` (default): the camera adjusts automatically once when it opens, then keeps that value for the whole run.
  - `auto`: the camera keeps adjusting.
  - A number: a fixed value you choose.

  Auto-adjusting cameras change color and brightness between frames, which looks like growth.
- Not every webcam or OS supports every control. OpenCV on **macOS** usually cannot set exposure, white balance or focus. You'll see a warning in the log, and the `notes` column records it. On **Windows**, press `d` in `fungus preview` to open the driver's settings window.
- Use steady, controlled lighting, avoid sunlight that changes during the day, mount the camera rigidly, and PNG output (the default).

## Capturing in the field

A camera in the field usually only takes photos; the measuring happens on a laptop. Photos reach the laptop over rsync, Syncthing, a phone's photo folder or an SD card, and `--watch` picks them up as they land:

```bash
# on the laptop: import, measure and keep an eye on the run
fungus import experiments/field-1 ~/synced/field-1 --watch --camera pi --tz America/Chicago &
fungus analyze experiments/field-1 --watch &
fungus health experiments/field-1 --watch --every 30m --webhook https://hooks.example/...
```

- **Already imported photos are skipped** (they are matched by content, not by name), so `--watch` can run forever and a re-sync never duplicates anything.
- **Half-copied files are skipped:** anything changed in the last `--settle` (10 s) waits for the next pass, so a photo is never imported mid-copy. Use `--settle 0` to import immediately.
- **A folder that disappears** (an unmounted drive, a dropped network share) is waited for, not an error.
- Times come from EXIF, then the file name, then the file date; `--tz` says which zone a camera's clock is in.

### On a Raspberry Pi (or any headless computer)

```bash
sudo apt install python3-opencv python3-pip
pip install fungus-cv                      # or: pip install -e . from a copy of this repo
fungus cameras                             # a Pi camera appears through V4L2 as /dev/video0
fungus init experiments/field-1
fungus snap experiments/field-1            # one test photo; copy it over and check the framing
fungus service install experiments/field-1 # capture starts at every login/boot
```

- **Headless:** `capture`, `snap`, `import`, `analyze`, `health` and `status` need no screen. `preview`, `annotate`, `pick-color`, `prompt`, `label` and the app need one — do those on a laptop, using photos copied from the Pi.
- **Where to analyze:** either sync the photos to a laptop (above), or analyze on the Pi with a colour threshold or a small trained model. SAM 2 is too slow for a Pi.
- **Sending photos:** a cron job or systemd timer running `rsync -a --remove-source-files experiments/field-1/frames/ laptop:~/synced/field-1/` is enough. Keep `frames.csv` on the Pi as the capture log; the laptop rebuilds its own from the photos it imports.
- **Power and light:** capture at fixed times of day (sunlight changes), use a controlled light if you can, and mount the camera and markers rigidly.
- `fungus health --watch --on-alert ...` on the Pi tells you when the camera stops responding or the card fills up.

## Long unattended runs

- `keep_awake: true` stops the computer from sleeping while it's idle. Closing a laptop lid can still put it to sleep, so change that in the power settings.
- If a run is interrupted, restart it with the same `start_at` in `config.yaml`. Capture continues on the original schedule.
- The camera is released between shots for intervals over 2 minutes, which helps it recover after being unplugged.

### Is the run still healthy?

Capture writes a heartbeat (`status.json`) every round: when it last took a photo, when the next is due, how many were saved or failed, how many slots were skipped and how much disk is left.

```bash
fungus health experiments/moss-1                      # check once (exit 1 if something is wrong)
fungus health experiments/moss-1 --watch --every 30m --webhook https://hooks.example/...
fungus health experiments/moss-1 --watch --on-alert 'mail -s "$FUNGUS_SUMMARY" me@example.com'
```

| Check | Warning | Problem |
|---|---|---|
| capture | no heartbeat for 2 intervals | none for 5 intervals: the run looks dead |
| frames | last photo over 2 intervals ago | over 5 intervals ago, or none at all |
| camera errors | 1–2 failed attempts recently | 3 or more |
| schedule | slots skipped (computer asleep or busy) | — |
| image quality | brightness off by more than 40% from the first frame, nearly black, or sharpness below 40% of the first frame | — |
| disk | room for fewer than 50 more photos | below `min_free_disk_mb`: capture stops |

- `--watch` reports only when the state **changes**, and sends a "recovered" alert when it clears, so a long run doesn't spam you.
- `--webhook` POSTs JSON (with a `text` field, so Slack-style hooks show it directly). `--on-alert` runs any command with `FUNGUS_SUMMARY`, `FUNGUS_LEVEL`, `FUNGUS_EXPERIMENT` and `FUNGUS_JSON` set — that's how to send email.
- An alert that fails to send is logged; it never stops the checks.

### Starting capture automatically

```bash
fungus service install experiments/moss-1 --dry-run     # show what would be written
fungus service install experiments/moss-1               # asks before installing
fungus service install experiments/moss-1 --action health --extra '--webhook https://...'
fungus service status experiments/moss-1
fungus service uninstall experiments/moss-1
```

Writes the right file for your system and registers it: a **launchd agent** on macOS (runs at login, restarted if it exits), a **systemd user unit** on Linux (`Restart=on-failure`), or a **Task Scheduler** task on Windows (at logon). It always prints the file and the commands first and asks before installing anything.

- These run while you are logged in. For a machine that captures while logged out: `loginctl enable-linger $USER` on Linux, or tick "Run whether user is logged on or not" in Task Scheduler on Windows.
- On macOS the program that runs needs camera permission; see *Troubleshooting cameras*.

## Development

```bash
pytest
ruff check .
```

The tests use a fake camera, so they need no hardware. CI runs them on Windows, macOS and Linux.
