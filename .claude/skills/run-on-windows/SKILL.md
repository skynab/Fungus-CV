---
name: run-on-windows
description: Set up, run and troubleshoot Fungus-CV on Windows — the desktop app, the demo, webcam capture (privacy settings, DirectShow/Media Foundation backends), long unattended runs with Task Scheduler, SAM 2 / trained models on an NVIDIA GPU, tests and building Fungus-CV.exe. Use when asked to run, start, launch, install, test or package the program on Windows.
---

# Running Fungus-CV on Windows

Work from the repository root in PowerShell. Call the virtualenv's executables directly
(`.venv\Scripts\fungus.exe`, `.venv\Scripts\python.exe`); activation does not persist between
commands, and `Activate.ps1` may be blocked by the execution policy anyway.

## 1. Make sure it is set up

Run the bundled check first; it creates `.venv` if missing, installs what is missing and prints
`fungus doctor`:

```powershell
powershell -ExecutionPolicy Bypass -File .claude\skills\run-on-windows\check.ps1            # add -Models for SAM 2 / trained models
```

What it does, if you need to do it by hand:

- Python 3.10+ from python.org or `winget install Python.Python.3.12`; use the `py` launcher
  (`py -3.12 --version`). The Microsoft Store "python" stub opens the Store instead of running.
- `py -3.12 -m venv .venv` then
  `.venv\Scripts\python.exe -m pip install -e ".[dev,gui]" pytest-qt` (pytest-qt is not in the
  `dev` extra; without it the GUI tests are skipped silently). `tzdata` comes with it on
  Windows, for time zones.
- **Models on an NVIDIA GPU:** install the CUDA build of PyTorch **first**, using the command
  from https://pytorch.org/get-started/locally/ (pip, your CUDA version; run it with
  `.venv\Scripts\python.exe -m pip ...`), then `.venv\Scripts\python.exe -m pip install -e ".[sam]"`. Installing `[sam]` alone gets the
  CPU-only build. Check with
  `.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`;
  `device: auto` then picks `cuda`.
- `.venv\Scripts\fungus.exe doctor` — cameras, window support, GPU, disk.

## 2. Pick what to run

| Goal | Command | Notes |
|---|---|---|
| Desktop app | `.venv\Scripts\fungus.exe gui [EXPERIMENT]` | blocks until the window closes: run it in the background |
| Try it without a camera | `.venv\Scripts\fungus.exe demo $env:TEMP\fungus-demo` then `analyze` and `report` on that folder | the folder must not exist yet; also File → New demo in the app |
| Shareable results page | `.venv\Scripts\fungus.exe summary EXPERIMENT` | writes `results\summary.html`; `start` it |
| Capture from a webcam | `.venv\Scripts\fungus.exe capture EXPERIMENT --interval 10s --duration 30m` | see cameras below |
| Everything else | `.venv\Scripts\fungus.exe --help` | |

When launching the app for the user, start it in the background, wait a few seconds, then check
its output for errors before saying it is running. The window opens on the user's desktop.

Paths: quote any path with spaces. Experiments under OneDrive-synced folders (Documents,
Desktop) get uploaded photo by photo and files may be locked while syncing; prefer a local
folder such as `C:\fungus\experiments` for long captures.

## 3. Cameras

- **Privacy:** Settings → Privacy & security → Camera → turn on *Camera access* **and** *Let
  desktop apps access your camera*. Windows has no per-app prompt for desktop programs; if these
  are off, every camera fails to open.
- **Busy camera:** only one program can use a webcam. Close Teams, Zoom, the Camera app and
  browser tabs using it.
- **Backend:** `backend: auto` uses DirectShow (`dshow`: opens fast, exposes most controls).
  If a camera fails to open or gives black frames, check with
  `.venv\Scripts\fungus.exe cameras --backend msmf` and, if Media Foundation works, set
  `backend: msmf` for that camera in `config.yaml`.
- **Which is which:** `.venv\Scripts\fungus.exe cameras`, then
  `.venv\Scripts\fungus.exe preview --index N` (or `preview EXPERIMENT` to use its camera
  settings). In `preview`, press `d` to open the driver's own settings window, where exposure,
  white balance and focus can be fixed by hand.
- `preview`, `annotate`, `pick-color`, `label` and the app need the user's desktop session; they
  do not work over a plain SSH session.

## 4. Long unattended runs

- Capture keeps Windows from idle sleep while it runs (`SetThreadExecutionState`;
  `capture.keep_awake`). Still set Power & battery → Screen and sleep → "Never" when plugged in,
  and "When I close the lid → Do nothing" for a laptop; Windows Update restarts can also end a
  run, so set active hours or pause updates for long runs.
- Start at logon: `.venv\Scripts\fungus.exe service install EXPERIMENT` registers a Task
  Scheduler task (`schtasks`); it prints the task and asks before installing — confirm with the
  user. `service status|uninstall EXPERIMENT`. To capture while nobody is logged on, open the
  task in Task Scheduler and tick "Run whether user is logged on or not".
- Watch it: `.venv\Scripts\fungus.exe health EXPERIMENT --watch --every 30m` (with `--webhook`
  or `--on-alert`), or the app's Capture page (Run health).

## 5. Tests

```powershell
$env:QT_QPA_PLATFORM = "offscreen"; .venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests
$env:FUNGUS_TEST_SAM = "1"; .venv\Scripts\python.exe -m pytest -q tests\test_sam.py tests\test_sam_incremental.py
```

The full run takes several minutes: run it in the background or with a long timeout and check
the exit code (`$LASTEXITCODE`) before committing. The 4 skips are the opt-in SAM model tests
and the real-image validation suite. CI runs the same tests on Windows (`.github/workflows/ci.yml`).

## 6. Building Fungus-CV.exe

```powershell
.venv\Scripts\python.exe -m pip install -e ".[app]"
.venv\Scripts\python.exe packaging\build_app.py                 # colour methods only, smaller
.venv\Scripts\python.exe packaging\build_app.py --with-models    # includes PyTorch (large)
.\dist\Fungus-CV\Fungus-CV.exe
```

Build on Windows (PyInstaller does not cross-compile) and ship the whole `dist\Fungus-CV`
folder. The exe is unsigned, so SmartScreen may say "Windows protected your PC": More info →
Run anyway, or sign it with a code-signing certificate before giving it to others.

## Report back

Say what was run and what the user should see (window open, files written, where), and quote
real errors. Never claim the camera works unless a capture actually saved frames
(`fungus status EXPERIMENT`). This skill's steps follow the code and CI; if something behaves
differently on the user's machine, say so rather than guessing.
