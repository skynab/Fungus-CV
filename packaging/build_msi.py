"""Build the Windows installer (.msi) from dist/Fungus-CV with the WiX v5 `wix` tool.

Pinned to v5 (not v4): the Files element used for directory harvesting in Package.wxs was
added in WiX v5. v6+ requires an Open Source Maintenance Fee for revenue-generating users;
v5 does not, so it stays on v5 rather than tracking latest.

    dotnet tool install --global wix --version 5.0.2
    python packaging/build_app.py --with-models
    python packaging/build_msi.py

Output: dist/Fungus-CV-<version>-windows-x64.msi
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
sys.path.insert(0, str(ROOT / "src"))

from fungus_cv import __version__  # noqa: E402


def msi_version(version: str) -> str:
    """MSI versions are numeric a.b.c(.d); drop pre-release suffixes such as 'rc1'."""
    nums = re.findall(r"\d+", version)[:3]
    return ".".join(nums + ["0"] * (3 - len(nums)))


def main() -> None:
    app_dir = ROOT / "dist" / "Fungus-CV"
    if not (app_dir / "Fungus-CV.exe").exists():
        sys.exit(f"{app_dir}\\Fungus-CV.exe not found: run packaging/build_app.py first")
    icon = PACKAGING / "build" / "icon.ico"
    out = ROOT / "dist" / f"Fungus-CV-{__version__}-windows-x64.msi"
    subprocess.run(
        ["wix", "build", str(PACKAGING / "windows" / "Package.wxs"),
         "-arch", "x64",
         "-d", f"Version={msi_version(__version__)}",
         "-d", f"IconFile={icon}",
         "-bindpath", f"app={app_dir}",
         "-o", str(out)],
        check=True)
    print(f"\nBuilt {out}")


if __name__ == "__main__":
    main()
