"""Test cases, synthetic key fixtures and an FP64 oracle for multiscale scoring."""
from dataclasses import dataclass
from itertools import product

import torch

EPS = 1e-12
BLOCK = 256
WINDOW = 64
DIM = 128
ATOL = 1e-4
RTOL = 1e-3
MIN_ANCHOR_NORM = 1e-2
SEEDS = (1729, 8675309)
BOUNDARIES = (1, 63, 64, 65, 255, 256, 257, 511, 512, 1025, 8193, 16385, 32767, 32768)
DTYPES = ("float16", "bfloat16")
PATTERNS = ("random", "constant", "repeated", "zero_rows", "zeros", "partial_zero",
            "cancellation", "threshold_below", "threshold_above", "extreme", "tiny")


@dataclass(frozen=True)
class Case:
    dtype: str
    batch: int
    heads: int
    length: int
    pattern: str = "random"

    @property
    def shape(self):
        return (self.batch, self.heads, self.length, DIM)

    @property
    def name(self):
        return f"{self.dtype}-b{self.batch}-h{self.heads}-n{self.length}-{self.pattern}"


def performance_cases():
    return [Case(dtype, b, h, n) for dtype, b, h, n in
            product(DTYPES, (1, 2), (4, 8), (8192, 16384, 32768))]


def quick_cases():
    return [Case("float16", 1, 4, 63),
            Case("bfloat16", 2, 8, 257, "partial_zero"),
            Case("float16", 1, 8, 65, "zeros"),
            Case("bfloat16", 1, 4, 8192),
            Case("float16", 2, 8, 32768)]


def correctness_cases():
    """Random keys at every boundary length, plus every structured pattern at short lengths."""
    cases = [Case(dtype, b, h, n) for dtype, b, h, n in
             product(DTYPES, (1, 2), (4, 8), BOUNDARIES)]
    cases += [Case(dtype, b, h, n, pattern) for dtype, b, h, pattern, n in
              product(DTYPES, (1, 2), (4, 8), PATTERNS[1:], (65, 257, 1025))]
    return cases


def make_keys(case, seed, device="cpu"):
    """Deterministic keys for a case; generated on CPU so every device sees the same values."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(case.shape, generator=generator, dtype=torch.float32)
    direction = torch.randn((case.batch, case.heads, 1, DIM), generator=generator)
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(EPS)
    x += 3.0 * direction
    pattern = case.pattern
    if pattern == "constant":
        x[:] = direction
    elif pattern == "repeated":
        x[:] = x[:, :, :1].clone()
        x[:, :, 1::2] *= 0.5
    elif pattern == "zero_rows":
        x[:, :, ::7] = 0
    elif pattern == "zeros":
        x.zero_()
    elif pattern == "partial_zero":
        x[:, :, :min(BLOCK, case.length)] = 0
        if case.length > 768:
            x[:, :, 640:768] = 0
    elif pattern in ("cancellation", "threshold_below", "threshold_above"):
        x.zero_()
        x[..., 0] = 1
        x[:, :, 1::2, 0] = -1
        if pattern != "cancellation":
            x[..., 1] = 0.009 if pattern == "threshold_below" else 0.011
    elif pattern == "extreme":
        x = x.sign() * 10000.0
    elif pattern == "tiny":
        x *= 1e-8 if case.dtype == "float16" else 1e-15
    elif pattern != "random":
        raise ValueError(f"Unknown pattern: {pattern}")
    x = x.to(getattr(torch, case.dtype))
    assert torch.isfinite(x).all() and (x.float().abs() <= 10000).all()
    return x.to(device=device).contiguous()


@dataclass
class Oracle:
    scores: torch.Tensor
    anchor_norms: torch.Tensor
    zero_only: torch.Tensor


def oracle(keys):
    """Scores computed directly in FP64, with each window summed independently."""
    x = keys.to(dtype=torch.float64)
    u = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    b, h, n, _ = u.shape
    scores = torch.empty((3, b, h, n), dtype=torch.float64, device=x.device)
    norms = torch.empty_like(scores)
    zero_only = torch.empty_like(scores, dtype=torch.bool)
    nonzero = (x != 0).any(dim=-1)

    def write(scale, start, end, mean, empty):
        norm = torch.linalg.vector_norm(mean, dim=-1)
        unit = mean / norm.unsqueeze(-1).clamp_min(EPS)
        scores[scale, :, :, start:end] = -(u[:, :, start:end] * unit).sum(-1)
        norms[scale, :, :, start:end] = norm
        zero_only[scale, :, :, start:end] = empty

    write(0, 0, n, u.mean(2, keepdim=True), ~nonzero.any(2, keepdim=True))
    for start in range(0, n, BLOCK):
        end = min(n, start + BLOCK)
        write(1, start, end, u[:, :, start:end].mean(2, keepdim=True),
              ~nonzero[:, :, start:end].any(2, keepdim=True))
    windows = torch.nn.functional.pad(u, (0, 0, WINDOW - 1, 0)).unfold(2, WINDOW, 1)
    flags = torch.nn.functional.pad(nonzero, (WINDOW - 1, 0)).unfold(2, WINDOW, 1)
    for start in range(0, n, BLOCK):
        end = min(n, start + BLOCK)
        counts = torch.arange(start + 1, end + 1, device=x.device).clamp_max(WINDOW)
        means = windows[:, :, start:end].sum(-1) / counts[None, None, :, None]
        write(2, start, end, means, ~flags[:, :, start:end].any(-1))
    return Oracle(scores, norms, zero_only)


class ScoreMismatch(AssertionError):
    def __init__(self, message, rows=None):
        super().__init__(message)
        self.rows = rows


def compare(actual, expected):
    """Check scores against the oracle and return per-scale coverage rows.

    A score is skipped only when its anchor mean has norm below MIN_ANCHOR_NORM
    and at least one nonzero contributor; skipped scores must still be finite,
    and anchors with only zero contributors must give exactly zero.
    """
    if (not isinstance(actual, torch.Tensor) or actual.shape != expected.scores.shape
            or actual.dtype != torch.float32 or not actual.is_contiguous()):
        raise ScoreMismatch("Output must be a contiguous float32 [3, B, H, N] tensor")
    a = actual.to(device=expected.scores.device, dtype=torch.float64)
    if not torch.isfinite(a).all():
        raise ScoreMismatch("Non-finite score")
    excluded = (expected.anchor_norms < MIN_ANCHOR_NORM) & ~expected.zero_only
    checked = ~excluded
    errors = (a - expected.scores).abs()
    bad = checked & (errors > ATOL + RTOL * expected.scores.abs())
    bad |= expected.zero_only & (a != 0)
    rows = []
    for scale, name in enumerate(("global", "block", "local")):
        selected = errors[scale][checked[scale]]
        rows.append({"scale": name, "total": a[scale].numel(),
                     "checked": int(checked[scale].sum()),
                     "excluded": int(excluded[scale].sum()),
                     "zero_only": int(expected.zero_only[scale].sum()),
                     "max_checked_error": float(selected.max()) if selected.numel() else 0.0,
                     "failures": int(bad[scale].sum())})
    if bad.any():
        raise ScoreMismatch(f"Scores differ from the FP64 oracle: {rows}", rows)
    return rows
