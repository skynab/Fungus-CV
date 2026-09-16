# Fungus-CV

Take time-lapse photos with a webcam, then measure how a spreading region (dye, moss, an infection) moves relative to a reference object over time. See [PLAN.md](PLAN.md) for the full roadmap.

**Status:** project setup and capture are done. Segmentation and measurement come next.

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

## What an experiment folder contains

```
experiments/dye-test-1/
  config.yaml     # settings for this experiment (edit this)
  frames/         # 2026-09-20T14-05-00.123Z_cam0.png ... (UTC time, sorts in time order)
  frames.csv      # one row per image or failed attempt (see below)
  capture.log
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
