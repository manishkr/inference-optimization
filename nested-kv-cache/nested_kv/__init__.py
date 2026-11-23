"""Multiscale (global, block, local) KV-cache key scoring on CUDA.

Build the extension once with ``python build.py`` before importing.
"""
from functools import cache
import importlib.util
import os
from pathlib import Path

import torch

__all__ = ["nested_kv_scores", "best_device", "backend", "available_backends"]

_ROOT = Path(__file__).resolve().parents[1]
_ENTRY_POINTS = {None: "scores", "fast": "scores_fast", "portable": "scores_portable"}


@cache
def _extension():
    libraries = list((_ROOT / ".build/lib").glob("_nested_kv*.so"))
    if len(libraries) != 1:
        raise RuntimeError("CUDA extension not built; run `python build.py` first")
    spec = importlib.util.spec_from_file_location("_nested_kv", libraries[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device_rank(props):
    return (props.major, props.minor, props.multi_processor_count, props.total_memory)


def best_device():
    """The strongest visible GPU, or the one named by NESTED_KV_DEVICE.

    GPUs are ranked by compute capability, then SM count, then memory, so an
    H100 is chosen over a T4 when both are present.
    """
    override = os.environ.get("NESTED_KV_DEVICE")
    if override:
        return torch.device(override)
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")
    index = max(range(torch.cuda.device_count()),
                key=lambda i: _device_rank(torch.cuda.get_device_properties(i)))
    return torch.device("cuda", index)


def _device_index(device):
    device = best_device() if device is None else torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device}")
    return torch.cuda.current_device() if device.index is None else device.index


def backend(device=None):
    """Name of the kernel set ``nested_kv_scores`` uses on a device: 'fast' or 'portable'."""
    return _extension().backend(_device_index(device))


def available_backends(device=None):
    """Kernel sets that can run on a device."""
    return list(_extension().available_backends(_device_index(device)))


def nested_kv_scores(keys, backend=None):
    """Score keys ``[B, H, N, 128]`` (float16/bfloat16) against three anchors.

    Returns a new float32 tensor ``[3, B, H, N]`` ordered global, block, local.
    CUDA inputs are scored on their own device; CPU inputs are moved to
    ``best_device()`` first. ``backend`` forces 'fast' or 'portable'.
    """
    if backend not in _ENTRY_POINTS:
        raise ValueError(f"Unknown backend {backend!r}; expected 'fast' or 'portable'")
    if keys.device.type == "cpu":
        keys = keys.to(best_device())
    return getattr(_extension(), _ENTRY_POINTS[backend])(keys)
