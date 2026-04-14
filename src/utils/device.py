"""
Resolve ``torch.device`` for the installed PyTorch build (CPU-only wheels vs CUDA).
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


def _cuda_tensor_works() -> bool:
    """True only if allocating on CUDA succeeds (catches CPU-only builds / broken drivers)."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except (AssertionError, RuntimeError):
        return False


def resolve_torch_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Default to the best available device, or validate an explicit request.

    Falls back to CPU when ``cuda`` is requested or auto-selected but not usable
    (e.g. PyTorch installed without CUDA support, or lazy CUDA init failure).
    """
    if device is None:
        if _cuda_tensor_works():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    d = torch.device(device)
    if d.type == "cuda":
        if not _cuda_tensor_works():
            logger.warning("CUDA requested or auto-selected but is not usable; using CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if d.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available; using CPU.")
            return torch.device("cpu")
    return d
