"""Pick the best available PyTorch device on Windows, macOS and Linux."""

from __future__ import annotations

import os
import sys


class MissingDependency(RuntimeError):
    pass


# Every feature that needs these says the same thing, because the fix is always the same.
MODEL_PACKAGES = ("torch", "torchvision", "transformers")
INSTALL_HINT = ('SAM 2 and trained models need PyTorch and transformers, which are not '
                'installed: pip install -e ".[sam]" (with an NVIDIA GPU, install the CUDA '
                'build of torch first -- see the README). A packaged app needs to be built '
                'with `build_app.py --with-models`. Colour methods work without them.')


def import_torch():
    if sys.platform == "darwin":
        # Let unsupported Apple GPU ops fall back to the CPU instead of failing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch
    except ImportError as exc:
        raise MissingDependency(INSTALL_HINT) from exc
    return torch


def import_transformers():
    try:
        import transformers
    except ImportError as exc:
        raise MissingDependency(INSTALL_HINT) from exc
    return transformers


def missing_dependency(exc: BaseException) -> str | None:
    """``INSTALL_HINT`` if ``exc`` is one of these packages missing, else None.

    Any import of them, wrapped or not, ends up in front of the user as a failed page; this
    turns "No module named 'torch'" into something they can act on.
    """
    if isinstance(exc, MissingDependency):
        return str(exc)
    if isinstance(exc, ModuleNotFoundError) and (exc.name or "").split(".")[0] in MODEL_PACKAGES:
        return INSTALL_HINT
    return None


def choose_device(preference: str = "auto"):
    """Return ``(device, dtype)``. CUDA uses bfloat16 where supported; MPS and CPU float32."""
    torch = import_torch()
    if preference == "auto":
        if torch.cuda.is_available():
            preference = "cuda"
        elif torch.backends.mps.is_available():
            preference = "mps"
        else:
            preference = "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but no CUDA GPU is available")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but Apple GPU (MPS) is not available")
    dtype = torch.float32
    if preference == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    return torch.device(preference), dtype
