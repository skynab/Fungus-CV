---
name: run-on-mac
description: Set up, run and troubleshoot Fungus-CV on macOS — the desktop app, the demo, camera capture (and its per-app camera permission), long unattended runs, SAM 2 / trained models on the Apple GPU, tests and building Fungus-CV.app. Use when asked to run, start, launch, install, test or package the program on a Mac.
---

# Running Fungus-CV on macOS

Work from the repository root. Always call the virtualenv's executables directly
(`.venv/bin/fungus`, `.venv/bin/python`); shell activation does not persist between commands.

## 1. Make sure it is set up

Run the bundled check first; it creates `.venv` if missing, installs what is missing and prints
`fungus doctor`:

```bash
bash .claude/skills/run-on-mac/check.sh          # add --models for SAM 2 / trained models
```

What it does, if you need to do it by hand:

- Python 3.10+ (`python3 --version`); the project uses 3.12.
- `python3 -m venv .venv && .venv/bin/pip install -e ".[dev,gui]" pytest-qt` (pytest-qt is
  not in the `dev` extra; without it the GUI tests are skipped silently)
- Models (SAM 2, U-Net training): `.venv/bin/pip install -e ".[sam]"`. PyTorch uses the Apple
  GPU (`mps`) automatically when `device: auto`; nothing else to install. **Install it
  whenever the SAM Prompts page, or Analyze/Train with `method: sam2` or `method: model`,
  will be used** — without it they report the missing dependency; colour methods are fine.
- `.venv/bin/fungus doctor` — camera permission, cameras, window support, GPU, disk.

## 2. Pick what to run

| Goal | Command | Notes |
|---|---|---|
| Desktop app | `.venv/bin/fungus gui [EXPERIMENT]` | run it in the background (it blocks until the window closes) |
| Try it without a camera | `.venv/bin/fungus demo /tmp/fungus-demo && .venv/bin/fungus analyze /tmp/fungus-demo && .venv/bin/fungus report /tmp/fungus-demo` | the folder must not exist yet; also File → New demo in the app |
| Shareable results page | `.venv/bin/fungus summary EXPERIMENT` | writes `results/summary.html`; `open` it |
| Capture from a webcam | `.venv/bin/fungus capture EXPERIMENT --interval 10s --duration 30m` | see camera permission below |
| Everything else | `.venv/bin/fungus --help` | |

When launching the app for the user, start it in the background, wait a few seconds, then check
the log for errors before saying it is running. The window opens on the user's screen, not in
the terminal.

## 3. Camera permission (the usual problem)

macOS grants camera access **per app**, not per program: `fungus` inherits the permission of
whatever launched it.

- **From the Claude desktop app:** it cannot show the camera prompt, so `capture`, `snap`,
  `preview` and the app's Camera page fail with a permission message when started from here.
  Do not retry. Tell the user to run the command in **Terminal.app** (or iTerm), where macOS asks
  once, or to use the packaged app (section 6). Everything that reads existing photos
  (`demo`, `import`, `analyze`, `report`, `study`, `summary`, the app on an existing experiment)
  works from here.
- **Denied earlier:** System Settings → Privacy & Security → Camera → turn on the terminal app,
  then quit and reopen that app.
- `.venv/bin/fungus cameras` lists cameras by the names macOS reports; `fungus preview --index N`
  shows which is which.
- OpenCV on macOS usually cannot set exposure, white balance or focus; the log warns and the
  `notes` column records it. Lock them with the camera's own software if the run needs it.

## 4. Long unattended runs

- Capture keeps the Mac awake itself (`caffeinate -i` while it runs; `capture.keep_awake`).
  Still set System Settings → Battery/Energy → "Prevent automatic sleeping when the display is
  off" and keep it plugged in. A closed lid sleeps regardless unless an external display is
  attached.
- Start at login and restart if it dies: `.venv/bin/fungus service install EXPERIMENT`
  (a launchd agent; it prints the plist and asks before installing — confirm with the user).
  `fungus service status|uninstall EXPERIMENT`. The agent needs camera permission for the
  Python it runs; if capture fails there, grant it in Privacy & Security → Camera.
- Watch it: `.venv/bin/fungus health EXPERIMENT --watch --every 30m` (with `--webhook` or
  `--on-alert` for notifications), or the app's Capture page (Run health).

## 5. Tests

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q     # ~6 min; GUI tests run offscreen
.venv/bin/ruff check src tests
FUNGUS_TEST_SAM=1 .venv/bin/python -m pytest -q tests/test_sam.py tests/test_sam_incremental.py
```

The full run takes longer than a 2-minute tool timeout: run it in the background or with a
longer timeout, and check the exit code (not just the tail) before committing. The 4 skips are
opt-in SAM model tests and the real-image validation suite.

## 6. Building Fungus-CV.app

```bash
.venv/bin/pip install -e ".[app]"
.venv/bin/python packaging/build_app.py              # colour methods only, smaller
.venv/bin/pip install -e ".[sam]"                    # needed for the next line
.venv/bin/python packaging/build_app.py --with-models   # includes PyTorch (large)
open dist/Fungus-CV.app
```

`--with-models` bundles only what it can import, so it stops with an error unless `[sam]`
is installed here. An app built without it reports a missing dependency on the SAM Prompts
page and for `method: sam2` / `method: model`; rebuild with `--with-models` to add them.

The app has its own camera permission entry, so it is the reliable way to capture on a Mac.
It is signed ad hoc: fine on the Mac that built it; elsewhere, right-click → Open (or sign and
notarize with a Developer ID).

## 7. Known harmless noise

- After closing the app, a burst of `Exception ignored ... torch/library.py ... 'NoneType'
  object is not callable` is PyTorch's cleanup running during interpreter shutdown. Exit code 0
  means it closed normally.

## Report back

Say what was run and what the user should see (window open, files written, where), and quote
real errors from the log. Never claim the camera works unless a capture actually saved frames
(`fungus status EXPERIMENT`).
