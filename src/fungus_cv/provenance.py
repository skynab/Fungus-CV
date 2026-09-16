"""Which code produced a result: package version and git commit."""

from __future__ import annotations

import subprocess
from pathlib import Path

from fungus_cv import __version__


def software_provenance() -> dict:
    info = {"fungus_cv_version": __version__}
    here = Path(__file__).parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=here, capture_output=True, text=True, timeout=5)
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
            info["git_dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info
