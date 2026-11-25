"""Attention over ResidualQuant-compressed KV caches on CUDA.

Build the extension once with ``python build.py`` before importing.
"""
from functools import cache
import importlib.util
import os
from pathlib import Path

import torch

__all__ = ["residual_attention", "best_device", "backend", "available_backends", "partitions"]

_ROOT = Path(__file__).resolve().parents[1]
_BACKENDS = (None, "fast", "portable")


@cache
def _extension():
    libraries = list((_ROOT / ".build/lib").glob("_residual_kv*.so"))
    if len(libraries) != 1:
        raise RuntimeError("CUDA extension not built; run `python build.py` first")
    spec = importlib.util.spec_from_file_location("_residual_kv", libraries[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device_rank(props):
    return (props.major, props.minor, props.multi_processor_count, props.total_memory)


def best_device():
    """The strongest visible GPU, or the one named by RESIDUAL_KV_DEVICE.

    GPUs are ranked by compute capability, then SM count, then memory, so an
    H100 is chosen over a T4 when both are present.
    """
    override = os.environ.get("RESIDUAL_KV_DEVICE")
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
    """Kernel set ``residual_attention`` uses on a device: 'fast' or 'portable'."""
    return _extension().backend(_device_index(device))


def available_backends(device=None):
    """Kernel sets that can run on a device."""
    return list(_extension().available_backends(_device_index(device)))


def partitions(batch, length, device=None):
    """``(splits, tokens_per_partition)`` used for a batch size and cached length."""
    return tuple(_extension().partitions(batch, length, _device_index(device)))


def residual_attention(query, new_key, new_value, loop, key_cache, value_cache, key_mean,
                       backend=None):
    """Attention of Q query tokens over one loop of a ResidualQuant KV cache.

    ``query``, ``new_key`` and ``new_value`` are ``[B, 16, Q, 128]``
    float16/bfloat16: the queries, keys and values of Q tokens that are not in
    the cache yet. Query i attends to every cached token and to new tokens
    0..i. ``loop`` is 0, 1 or 2 for a residual loop and 3 for the anchor
    (last) loop. Each cache is the tuple ``(anchor_codes, anchor_scale,
    anchor_offset, residual_codes, residual_scale, residual_offset, alpha)``
    and ``key_mean`` is ``[16, 4, 128]`` float32. Returns new float32 tensors
    ``(out [B, 16, Q, 128], lse [B, 16, Q])``.

    CUDA inputs run on their own device; if every input is on the CPU they are
    moved to ``best_device()`` first. ``backend`` forces 'fast' or 'portable'.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected 'fast' or 'portable'")
    if not isinstance(loop, int) or isinstance(loop, bool):
        raise TypeError(f"loop must be an int, got {type(loop).__name__}")
    key_cache, value_cache = tuple(key_cache), tuple(value_cache)
    tensors = (query, new_key, new_value, *key_cache, *value_cache, key_mean)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("every input must be a torch.Tensor")
    if all(t.device.type == "cpu" for t in tensors):
        device = best_device()
        query, new_key, new_value, key_mean = (
            t.to(device) for t in (query, new_key, new_value, key_mean))
        key_cache = tuple(t.to(device) for t in key_cache)
        value_cache = tuple(t.to(device) for t in value_cache)
    out, lse = _extension().attention(query, new_key, new_value, loop, list(key_cache),
                                      list(value_cache), key_mean, backend or "")
    return out, lse
