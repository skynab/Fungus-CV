# Building the Fungus-CV desktop app

```bash
pip install -e ".[app]"                 # PySide6 + PyInstaller
python packaging/build_app.py           # color-based methods; smaller download

pip install -e ".[sam]"                 # PyTorch + transformers, needed for the next line
python packaging/build_app.py --with-models   # also SAM 2 / trained models (large)
```

PyInstaller bundles what it can import, so `--with-models` only works if `[sam]` is
installed in the environment you build from; the build stops and says so if it is not.
An app built **without** `--with-models` runs everything except the **SAM Prompts** page
and `method: sam2` / `method: model` in **Analyze** and **Train Models**, which report the
missing dependency. There is no way to add PyTorch to a built app afterwards -- build it
again with `--with-models`. With an NVIDIA GPU, install the CUDA build of PyTorch before
`[sam]` (see <https://pytorch.org/get-started/locally/>); `[sam]` alone bundles the
CPU-only build.

| OS | Output | Start it with |
|---|---|---|
| macOS | `dist/Fungus-CV.app` | double-click, or `open dist/Fungus-CV.app` |
| Windows | `dist\Fungus-CV\Fungus-CV.exe` | double-click |
| Linux | `dist/Fungus-CV/Fungus-CV` | run it |

Build on each operating system you want to support; PyInstaller does not cross-compile.

## Why an app helps with camera access on macOS

macOS grants camera access per *app*. A Python script inherits the permission of whatever
started it (Terminal, VS Code, …), and some of those apps cannot show the permission prompt.
`Fungus-CV.app` declares why it needs the camera (`NSCameraUsageDescription`) and gets its own
entry under System Settings → Privacy & Security → Camera. The first time you use the camera
macOS asks once; after that it remembers.

The build is signed ad hoc, which is enough on the computer that built it. To give the app to
other people, sign it with an Apple Developer ID and notarize it; otherwise macOS will warn that
the developer cannot be verified (right-click → Open works around this).

Running from source is still supported: `fungus gui` opens the same window (camera permission
then belongs to the terminal you started it from).
