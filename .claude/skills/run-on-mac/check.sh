#!/usr/bin/env bash
# Set up Fungus-CV on macOS if needed, then run `fungus doctor`.
# Usage: bash .claude/skills/run-on-mac/check.sh [--models]
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

if [[ "$(uname)" != "Darwin" ]]; then
  echo "This check is for macOS; see the run-on-windows skill on Windows." >&2
  exit 1
fi

PY=${PYTHON:-python3}
if [[ ! -x .venv/bin/python ]]; then
  "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || { echo "Python 3.10+ is needed (found $("$PY" --version 2>&1))" >&2; exit 1; }
  echo "Creating .venv with $("$PY" --version)"
  "$PY" -m venv .venv
fi

extras="dev,gui"
[[ "${1:-}" == "--models" ]] && extras="$extras,sam"
missing=0
.venv/bin/python -c 'import fungus_cv, PySide6, pytest, pytestqt' 2>/dev/null || missing=1
[[ "$extras" == *sam* ]] && { .venv/bin/python -c 'import torch, transformers' 2>/dev/null || missing=1; }
if [[ $missing == 1 ]]; then
  echo "Installing fungus-cv[$extras]"
  .venv/bin/pip install -q -e ".[$extras]" pytest-qt
fi

echo "Python: $(.venv/bin/python --version)"
if .venv/bin/python -c 'import torch' 2>/dev/null; then
  .venv/bin/python -c 'import torch; print("PyTorch", torch.__version__, "- Apple GPU (mps):", torch.backends.mps.is_available())'
else
  echo "PyTorch not installed (SAM 2 and trained models need: bash $0 --models)"
fi
.venv/bin/fungus doctor || true
