"""Pick the best available PyTorch device on Windows, macOS and Linux."""

from __future__ import annotations

import os
import sys


class MissingDependency(RuntimeError):
    pass


def import_torch():
    if sys.platform == "darwin":
        # Let unsupported Apple GPU ops fall back to the CPU instead of failing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch
    except ImportError as exc:
        raise MissingDependency(
            'SAM needs PyTorch and transformers: pip install -e ".[sam]"'
        ) from exc
    return torch


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
