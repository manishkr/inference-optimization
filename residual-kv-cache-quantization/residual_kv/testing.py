"""Test cases, synthetic ResidualQuant caches and an FP64 oracle with an error bound."""
from dataclasses import dataclass
from itertools import product
import math

import torch

HEADS = 16
DIM = 128
GROUP = 32
LOOPS = 4
ANCHOR_LOOP = 3
SQRT_DIM = math.sqrt(DIM)

# Error bound (see `oracle`).
UNIT = 2.0 ** -24          # FP32 unit roundoff
TOL_ABS = 2.0 ** -20
TOL_REL = 2.0 ** -16
ACCUMULATION = 4.0

SEEDS = (31337, 4242)
# Lengths around the kernels' boundaries: 8-token MMA tiles, 16-token warp
# tiles, 64-token block tiles and 256-token partitions.
LENGTHS = (1, 2, 3, 15, 16, 17, 63, 64, 65, 191, 192, 193, 255, 256, 257, 511, 513, 1000, 2049,
           4095, 8192, 12289, 16384, 24577, 32768)
QUERIES = (1, 2, 3, 5, 7, 8)
BATCHES = (1, 2, 5, 8)
DTYPES = ("float16", "bfloat16")
PATTERNS = ("quantized", "random", "zero_codes", "max_codes", "zero_meta", "zero_values",
            "alpha_signs", "zero_query", "peaked", "flat", "dominant_new", "fp8_extremes", "cancel",
            "tiny_keys", "huge_products", "key_rounding", "value_rounding", "weights_fp16",
            "weights_bf16", "query_code")
PRECISION_PATTERNS = ("key_rounding", "value_rounding", "weights_fp16", "weights_bf16")
QUANTIZED_TOKENS = 1 << 16   # largest B * N generated with the quantiser
WINDOW = 192                 # tokens carrying a precision pattern in long rows
FAR = 48.0                   # how much lower the other tokens of a long row score


@dataclass(frozen=True)
class Case:
    dtype: str
    batch: int
    queries: int
    length: int
    loop: int
    pattern: str = "quantized"

    @property
    def name(self):
        return (f"{self.dtype}-loop{self.loop}-b{self.batch}-q{self.queries}-n{self.length}"
                f"-{self.pattern}")

    @property
    def residual(self):
        return self.loop != ANCHOR_LOOP


@dataclass
class Inputs:
    query: torch.Tensor
    new_key: torch.Tensor
    new_value: torch.Tensor
    loop: int
    key_cache: tuple
    value_cache: tuple
    key_mean: torch.Tensor

    def args(self):
        return (self.query, self.new_key, self.new_value, self.loop, self.key_cache,
                self.value_cache, self.key_mean)

    def tensors(self):
        return [self.query, self.new_key, self.new_value, *self.key_cache, *self.value_cache,
                self.key_mean]


def performance_cases():
    """24 benchmark shapes; residual-loop cases are timed with loops 0, 1 and 2."""
    return [Case("bfloat16", b, q, n, loop, "random") for loop, q, b, n in
            product((1, ANCHOR_LOOP), (1, 4, 8), (1, 4), (8192, 32768))]


def quick_cases():
    return [Case("bfloat16", 1, 1, 17, 0, "random"),
            Case("float16", 2, 5, 513, 3, "cancel"),
            Case("bfloat16", 1, 3, 193, 1, "key_rounding"),
            Case("float16", 1, 8, 8192, 2, "quantized"),
            Case("bfloat16", 4, 4, 32768, 3, "random")]


def correctness_cases(full=False):
    """Every length with every loop, every (loop, Q, dtype) combination, every pattern."""
    cases = []
    for k in range(LOOPS * len(LENGTHS)):
        n, loop = LENGTHS[k // LOOPS], k % LOOPS
        q = QUERIES[(k // LOOPS + loop) % len(QUERIES)]
        dtype = DTYPES[(k // (LOOPS * len(QUERIES))) % 2]
        b = BATCHES[(k // 2) % len(BATCHES)]
        while b * n > 1 << 17:
            b = BATCHES[BATCHES.index(b) - 1]
        cases.append(Case(dtype, b, q, n, loop, "quantized" if b * n <= QUANTIZED_TOKENS else "random"))
    cases += [Case("float16", 16, 3, 255, 2, "quantized"), Case("bfloat16", 16, 8, 1000, 3, "random"),
              Case("float16", 16, 1, 8192, 1, "random")]
    lengths = (63, 256, 1000)
    for j, (pattern, loop, dtype) in enumerate(product(PATTERNS[2:], (2, ANCHOR_LOOP), DTYPES)):
        for n in lengths if full else lengths[j % 3:j % 3 + 1]:
            cases.append(Case(dtype, 1 + j % 2, QUERIES[(j + 2) % len(QUERIES)], n, loop, pattern))
    for pattern, n, loop in product(PRECISION_PATTERNS, (8192, 32768), (1, ANCHOR_LOOP)):
        cases.append(Case("float16" if loop == 1 else "bfloat16", 1, 3, n, loop, pattern))
    return cases


# ----------------------------------------------------------------------------- packing


def fp8(x):
    return x.clamp(-448, 448).to(torch.float8_e4m3fn)


def pack4(codes):
    c = codes.to(torch.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()


def pack2(codes):
    c = codes.to(torch.uint8)
    return (c[..., 0::4] | (c[..., 1::4] << 2) | (c[..., 2::4] << 4) | (c[..., 3::4] << 6)).contiguous()


def unpack4(packed):
    p = packed.to(torch.int32)
    return torch.stack((p & 0xF, p >> 4), dim=-1).flatten(-2)


def unpack2(packed):
    p = packed.to(torch.int32)
    return torch.stack([(p >> (2 * i)) & 3 for i in range(4)], dim=-1).flatten(-2)


def _expand(groups):
    return groups.repeat_interleave(GROUP, dim=-1)


def quantize(x, bits):
    """Group-wise affine quantiser (group size 32) with FP8 scale and offset."""
    g = x.unflatten(-1, (DIM // GROUP, GROUP))
    lo, hi = g.amin(-1), g.amax(-1)
    scale, offset = fp8((hi - lo) / (2 ** bits - 1)), fp8(lo)
    s, o = scale.float().unsqueeze(-1), offset.float().unsqueeze(-1)
    codes = torch.where(s > 0, ((g - o) / s.clamp_min(1e-30)).round(), torch.zeros_like(g))
    codes = codes.clamp(0, 2 ** bits - 1).flatten(-2)
    return codes, scale, offset


def dequantize(codes, scale, offset):
    return _expand(offset.float()) + _expand(scale.float()) * codes.float()


# ----------------------------------------------------------------------------- fixtures


class _Rng:
    def __init__(self, seed, device):
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

    def randn(self, *shape):
        return torch.randn(shape, generator=self.gen, device=self.device)

    def rand(self, *shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(shape, generator=self.gen, device=self.device)

    def randint(self, high, *shape, dtype=torch.int64):
        return torch.randint(0, high, shape, generator=self.gen, device=self.device, dtype=dtype)


def _random_cache(rng, b, n, packed=False):
    """Random metadata and codes; ``packed`` draws the code bytes directly."""
    cache = {"scale4": rng.rand(b, HEADS, n, 4, hi=0.6),
             "offset4": rng.rand(b, HEADS, n, 4, lo=-3, hi=3),
             "scale2": rng.rand(b, HEADS, 3, n, 4, hi=0.25),
             "offset2": rng.rand(b, HEADS, 3, n, 4, lo=-0.75, hi=0.75),
             "alpha": rng.rand(b, HEADS, 3, n, lo=-2.5, hi=2.5)}
    if packed:
        cache["packed4"] = rng.randint(256, b, HEADS, n, DIM // 2, dtype=torch.uint8)
        cache["packed2"] = rng.randint(256, b, HEADS, 3, n, DIM // 4, dtype=torch.uint8)
    else:
        cache["codes4"] = rng.randint(16, b, HEADS, n, DIM).float()
        cache["codes2"] = rng.randint(4, b, HEADS, 3, n, DIM).float()
    return cache


def _quantized_cache(rng, b, n, means):
    """Synthetic per-loop states quantised as in the paper: the last loop is the
    INT4 anchor, earlier loops are INT2 residuals after a least-squares alpha.

    Each loop's state is a damped copy of the previous loop's plus fresh noise,
    so neighbouring loops are strongly correlated; a few channels per head are
    amplified. ``means`` ([16, 4, 128]) are the key means, None for values."""
    gain = 1 + 5 * (rng.rand(1, HEADS, 1, DIM) < 0.03).float()
    h = rng.randn(b, HEADS, n, DIM) * gain
    states = []
    for _ in range(LOOPS):
        h = 0.8 * h + 0.6 * rng.randn(b, HEADS, n, DIM) * gain
        states.append(h)
    codes4, scale4, offset4 = quantize(states[ANCHOR_LOOP], 4)
    anchor = dequantize(codes4, scale4, offset4)
    if means is not None:
        anchor = anchor + means[None, :, ANCHOR_LOOP, None, :]
    out = {"codes4": codes4, "scale4": scale4.float(), "offset4": offset4.float(),
           "codes2": [], "scale2": [], "offset2": [], "alpha": []}
    norm = (anchor * anchor).sum(-1).clamp_min(1e-12)
    for r in range(3):
        alpha = ((states[r] * anchor).sum(-1) / norm).clamp(-4, 4).bfloat16().float()
        codes2, scale2, offset2 = quantize(states[r] - alpha.unsqueeze(-1) * anchor, 2)
        out["codes2"].append(codes2)
        out["scale2"].append(scale2.float())
        out["offset2"].append(offset2.float())
        out["alpha"].append(alpha)
    for key in ("codes2", "scale2", "offset2", "alpha"):
        out[key] = torch.stack(out[key], dim=2)
    return out


def _finish_cache(c):
    codes4 = c["packed4"] if "packed4" in c else pack4(c["codes4"])
    codes2 = c["packed2"] if "packed2" in c else pack2(c["codes2"])
    return (codes4.contiguous(), fp8(c["scale4"]), fp8(c["offset4"]), codes2.contiguous(),
            fp8(c["scale2"]), fp8(c["offset2"]), c["alpha"].clamp(-4, 4).bfloat16())


def _window(n, seed):
    """Tokens that carry a precision pattern: the whole row, or a window in long rows."""
    mask = torch.ones(n, dtype=torch.bool)
    if n > 4096:
        start = 777 + (seed % 5) * 1301
        mask[:] = False
        mask[start:start + WINDOW] = True
    return mask


def _components_with_sum(target, count, dtype):
    """``count`` values exact in ``dtype`` whose sum is ``target`` to about FP32 precision."""
    as_dtype = lambda v: torch.tensor(v, dtype=torch.float64).to(dtype).double().item()
    first = as_dtype(target / count)
    values = [first] * (count - 3)
    rest = target - first * (count - 3)
    for _ in range(3):
        values.append(as_dtype(rest))
        rest -= values[-1]
    return torch.tensor(values, dtype=torch.float32)


# Level gaps g for which exp(-g), exp(-2g) and exp(-3g) all lie just above a
# rounding midpoint of FP16 (0.6% to 0.9% of an ulp) or BF16 (0.5% to 5.5%).
WEIGHT_GAPS = {"weights_fp16": 0.616558, "weights_bf16": 0.942248}


def make_inputs(case, seed, device="cpu"):
    """Deterministic inputs for a case on one device."""
    device = torch.device(device)
    rng = _Rng(seed, device)
    b, n, q, pattern = case.batch, case.length, case.queries, case.pattern
    dtype = getattr(torch, case.dtype)
    key_mean = rng.rand(HEADS, LOOPS, DIM, lo=-1.5, hi=1.5)
    query, new_key, new_value = (rng.randn(b, HEADS, q, DIM) for _ in range(3))

    if pattern == "quantized":
        if b * n > QUANTIZED_TOKENS:
            raise ValueError(f"{case.name}: quantised fixtures are limited to B*N <= {QUANTIZED_TOKENS}")
        key_mean = 0.4 * rng.randn(HEADS, LOOPS, DIM)
        key, value = _quantized_cache(rng, b, n, key_mean), _quantized_cache(rng, b, n, None)
        new_key = new_key + key_mean[None, :, case.loop, None, :]
    else:
        packed = pattern == "random"
        key, value = _random_cache(rng, b, n, packed), _random_cache(rng, b, n, packed)

    if pattern == "zero_codes":
        for c in (key, value):
            c["codes4"].zero_()
            c["codes2"].zero_()
    elif pattern == "max_codes":
        for c in (key, value):
            c["codes4"].fill_(15)
            c["codes2"].fill_(3)
    elif pattern == "zero_meta":
        for c in (key, value):
            for name in ("scale4", "offset4", "scale2", "offset2"):
                c[name].zero_()
        key_mean.zero_()
    elif pattern == "zero_values":
        for name in ("scale4", "offset4", "scale2", "offset2"):
            value[name].zero_()
        new_value.zero_()
    elif pattern == "alpha_signs":
        phase = torch.arange(n, device=device) % 3
        for c in (key, value):
            c["alpha"] = torch.where(phase == 0, 0.0, torch.where(phase == 1, -c["alpha"].abs() - 0.25,
                                                                   c["alpha"].abs()))
    elif pattern == "zero_query":
        query.zero_()
    elif pattern == "peaked":
        query *= 6
    elif pattern == "flat":
        query *= 2e-3
    elif pattern == "dominant_new":
        new_key = 3 * query.clone()
    elif pattern == "fp8_extremes":
        choices = torch.tensor([2.0 ** -9, 5 * 2.0 ** -9, 6 * 2.0 ** -9, 448.0, -448.0, 0.0,
                                -3 * 2.0 ** -9], device=device)
        for c in (key, value):
            for name in ("scale4", "offset4", "scale2", "offset2"):
                pick = choices[rng.randint(len(choices), *c[name].shape)]
                c[name] = pick.abs() if name.startswith("scale") else pick
        key_mean.zero_()
        query *= 0.01
    elif pattern == "cancel":
        # In groups 0 and 2 the offset cancels the scaled codes exactly.
        for c in (key, value):
            for codes, scale, offset, bits in (("codes4", "scale4", "offset4", 4),
                                               ("codes2", "scale2", "offset2", 2)):
                shape = c[scale].shape
                level = 2 ** rng.randint(bits, *shape).float()            # powers of two
                step = 2.0 ** (rng.randint(6, *shape).float() - 5)       # 2^-5 .. 2^0
                cancel = (torch.arange(4, device=device) % 2 == 0).expand(shape)
                c[scale] = torch.where(cancel, step, c[scale])
                c[offset] = torch.where(cancel, -step * level, c[offset])
                c[codes] = torch.where(_expand(cancel), _expand(level), c[codes])
        key_mean.view(HEADS, LOOPS, 4, GROUP)[:, :, 0::2] = 0
        new_key = 3 * query.clone()
    elif pattern == "tiny_keys":
        # alpha 2^-14 times anchor keys of at most 2^-8: keys below FP16's normal range.
        for name in ("offset4", "scale2", "offset2"):
            key[name].zero_()
        key["scale4"].fill_(2.0 ** -9)
        key["codes4"] = rng.randint(3, *key["codes4"].shape).float()
        key["alpha"].fill_(2.0 ** -14)
        key_mean.zero_()
        query = 48 * torch.sign(query)
        new_key *= 1e-7
    elif pattern == "huge_products":
        for name in ("scale4", "offset4"):
            key[name] = rng.rand(*key[name].shape, lo=300, hi=448)
        key["alpha"] = rng.rand(*key["alpha"].shape, lo=3, hi=4)
        query = 60 * rng.rand(b, HEADS, q, DIM, lo=-1, hi=1)
        query[..., 1:] *= 1e-3
    elif pattern == "key_rounding":
        # Keys of 3 + 2^-10 + 2^-13 through the key means (alpha 1 + 2^-6 on
        # residual loops): FP16, TF32 and BF16 roundings of a key all change it.
        window = _window(n, seed).to(device)
        for name in ("scale4", "scale2", "offset2"):
            key[name].zero_()
        key["offset4"] = torch.where(window[None, None, :, None], 0.0, -5.5).expand_as(key["offset4"]).clone()
        key["alpha"].fill_(1 + 2.0 ** -6)
        key_mean.fill_(3 + 2.0 ** -10 + 2.0 ** -13)
        query = query.abs()
        new_key.zero_()
    elif pattern == "value_rounding":
        # Values 4 + 5 * 2^-9 (anchor loop) and 4 + 7 * 2^-9 (residual loops) are
        # FP16 rounding ties; the softmax concentrates on two tokens.
        value["codes4"].fill_(5)
        value["scale4"].fill_(2.0 ** -9)
        value["offset4"].fill_(4.0)
        value["scale2"].zero_()
        value["offset2"].fill_(2.0 ** -8)
        value["alpha"].fill_(1.0)
        hot = rng.randint(n, 2)
        for name in ("scale4", "scale2", "offset2"):
            key[name].zero_()
        key["offset4"].fill_(-1.0)
        key["offset4"][:, :, hot] = 4.0
        key["alpha"] = key["alpha"].abs() + 0.5
        key_mean.zero_()
        query = query.abs()
        new_key = -query.clone()
    elif pattern in WEIGHT_GAPS:
        # Four score levels, level(n) = (n ^ (n >> 2)) & 3, so every aligned
        # 4-token block holds all four. Keys of level / 8 on the first group and
        # query components summing to gap * sqrt(128) * 8 make the level gap
        # exact. Values are -1 on level-3 tokens and +1 elsewhere.
        target = WEIGHT_GAPS[pattern] * SQRT_DIM * 8
        idx = torch.arange(n, device=device)
        level = ((idx ^ (idx >> 2)) & 3).float()
        window = _window(n, seed).to(device)
        for c in (key, value):
            for name in ("scale4", "offset4", "scale2", "offset2"):
                c[name].zero_()
            c["alpha"].fill_(1.0)
        key["scale4"][..., 0] = 0.125
        key["codes4"].zero_()
        key["codes4"][..., :GROUP] = level[None, None, :, None]
        far = -float(round(FAR * SQRT_DIM / target))
        key["offset4"][..., 0] = torch.where(window, 0.0, far)[None, None, :]
        key_mean.zero_()
        value["offset4"][:] = torch.where(level == 3, -1.0, 1.0)[None, None, :, None]
        query.zero_()
        query[..., :GROUP] = _components_with_sum(target, GROUP, dtype).to(device)
        new_key.zero_()
        new_value.fill_(1.0)
    elif pattern == "query_code":
        # Query-code products with 13 significant bits.
        qv, code = (1 + 3 * 2.0 ** -10, 5) if case.dtype == "float16" else (1.8671875, 13)
        for name in ("offset4", "scale2", "offset2"):
            key[name].zero_()
        key["scale4"].fill_(1.0)
        key["codes4"] = code * rng.randint(2, *key["codes4"].shape).float()
        key["alpha"].fill_(1.0)
        key_mean.zero_()
        query = 0.25 * qv * (rng.randn(b, HEADS, q, DIM) > 0).float()
        new_key.zero_()
    elif pattern not in ("quantized", "random"):
        raise ValueError(f"Unknown pattern: {pattern}")

    cast = lambda t: t.to(dtype).contiguous()
    return Inputs(cast(query), cast(new_key), cast(new_value), case.loop,
                  _finish_cache(key), _finish_cache(value), key_mean.float().contiguous())


# ----------------------------------------------------------------------------- oracle


@dataclass
class Oracle:
    out: torch.Tensor         # [B, 16, Q, 128] float64
    lse: torch.Tensor         # [B, 16, Q]
    out_tol: torch.Tensor     # allowed |out - reference|
    lse_tol: torch.Tensor
    exact_zero: torch.Tensor  # channels that must come out as exact zeros


def _rows(t, start, stop):
    """Rows [start, stop) of a [B, 16, ...] tensor flattened to [B * 16, ...]."""
    return t.flatten(0, 1)[start:stop]


def _rebuild(cache, loop, start, stop, mean_anchor=None, mean_loop=None):
    """FP64 elements of one cache loop, and the sums of the magnitudes of their terms."""
    c4 = unpack4(_rows(cache[0], start, stop)).double()
    s4 = _expand(_rows(cache[1], start, stop).double())
    o4 = _expand(_rows(cache[2], start, stop).double())
    m3 = [mean_anchor[:, None, :].expand_as(c4)] if mean_anchor is not None else []
    if loop == ANCHOR_LOOP:
        terms = [o4, s4 * c4, *m3]
    else:
        c2 = unpack2(_rows(cache[3], start, stop)[:, loop]).double()
        s2 = _expand(_rows(cache[4], start, stop)[:, loop].double())
        o2 = _expand(_rows(cache[5], start, stop)[:, loop].double())
        alpha = _rows(cache[6], start, stop)[:, loop].double().unsqueeze(-1)
        mr = [mean_loop[:, None, :].expand_as(c4)] if mean_loop is not None else []
        terms = [alpha * o4, alpha * s4 * c4, *(alpha * m for m in m3), o2, s2 * c2, *mr]
    return sum(terms), sum(t.abs() for t in terms)


def oracle(inputs, max_elements=1 << 26):
    """FP64 outputs and a per-element error bound for an FP32-accurate kernel.

    For query i, let p be the FP64 softmax over its visible tokens (the cache
    and new tokens 0..i). Let a[n] be the sum of |q_d * t| over every term t
    of key n (so a kernel's score error is about TOL_REL * c[n] with
    c[n] = 1 + a[n] / sqrt(128)), and b[n, d] the sum of the magnitudes of the
    terms of value element (n, d). To first order, score errors move the output
    by sum p[n] ds[n] (V[n] - ref) and value errors by sum p[n] dV[n]. Summing
    over many tokens in FP32 adds, per token, at most the smaller of a relative
    rounding of the whole sum and that token's own contribution:

      out: TOL_ABS + TOL_REL (sum p b + sum p c |V - ref|)
           + ACCUMULATION sum_n min(UNIT sum p b, p[n] (b[n] + |ref|))
      lse: TOL_ABS + TOL_REL sum p c + ACCUMULATION sum_n min(UNIT, p[n])
    """
    q = inputs.query
    batch, _, nq, _ = q.shape
    n = inputs.key_cache[0].shape[2]
    rows, total = batch * HEADS, n + nq
    device = q.device
    loop = inputs.loop
    heads = torch.arange(rows, device=device) % HEADS
    mean = inputs.key_mean.double()
    visible = torch.arange(total, device=device)[None, :] <= n + torch.arange(nq, device=device)[:, None]
    chunk = max(1, max_elements // (nq * total * DIM))
    results = {k: [] for k in ("out", "lse", "out_tol", "lse_tol", "exact_zero")}
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        h = heads[start:stop]
        k, k_mag = _rebuild(inputs.key_cache, loop, start, stop, mean[h, ANCHOR_LOOP],
                            mean[h, loop] if loop != ANCHOR_LOOP else None)
        v, v_mag = _rebuild(inputs.value_cache, loop, start, stop)
        nk = _rows(inputs.new_key, start, stop).double()
        nv = _rows(inputs.new_value, start, stop).double()
        k, k_mag = torch.cat((k, nk), 1), torch.cat((k_mag, nk.abs()), 1)
        v, v_mag = torch.cat((v, nv), 1), torch.cat((v_mag, nv.abs()), 1)
        qr = _rows(q, start, stop).double()
        scores = ((qr @ k.transpose(1, 2)) / SQRT_DIM).masked_fill(~visible, -math.inf)
        cond = 1 + (qr.abs() @ k_mag.transpose(1, 2)) / SQRT_DIM
        top = scores.amax(-1, keepdim=True)
        e = torch.exp(scores - top)
        z = e.sum(-1, keepdim=True)
        p = e / z
        ref = p @ v
        magnitude = p @ v_mag
        spread = torch.einsum("rqn,rqnd->rqd", p * cond, (v[:, None] - ref[:, :, None]).abs())
        accumulation = torch.minimum(UNIT * magnitude[:, :, None, :],
                                     p[..., None] * (v_mag[:, None] + ref.abs()[:, :, None])).sum(2)
        results["out"].append(ref)
        results["lse"].append((top + torch.log(z)).squeeze(-1))
        results["out_tol"].append(TOL_ABS + TOL_REL * (magnitude + spread) + ACCUMULATION * accumulation)
        results["lse_tol"].append(TOL_ABS + TOL_REL * (p * cond).sum(-1)
                                  + ACCUMULATION * torch.clamp(p, max=UNIT).sum(-1))
        results["exact_zero"].append(
            torch.where(visible[None, :, :, None], v_mag[:, None], 0.0).amax(2) == 0)
    merged = {key: torch.cat(parts) for key, parts in results.items()}
    return Oracle(*(merged[key].reshape(batch, HEADS, *merged[key].shape[1:])
                    for key in ("out", "lse", "out_tol", "lse_tol", "exact_zero")))


class Mismatch(AssertionError):
    def __init__(self, message, summary=None):
        super().__init__(message)
        self.summary = summary


def compare(out, lse, expected):
    """Check outputs against the oracle; returns the largest error-to-bound ratios."""
    b, h, nq = expected.lse.shape
    for t, shape, name in ((out, (b, h, nq, DIM), "out"), (lse, (b, h, nq), "lse")):
        if (not isinstance(t, torch.Tensor) or tuple(t.shape) != shape or t.dtype != torch.float32
                or not t.is_contiguous()):
            raise Mismatch(f"{name} must be a contiguous float32 tensor of shape {shape}")
    o = out.to(device=expected.out.device, dtype=torch.float64)
    s = lse.to(device=expected.lse.device, dtype=torch.float64)
    if not (torch.isfinite(o).all() and torch.isfinite(s).all()):
        raise Mismatch("non-finite output")
    out_ratio = (o - expected.out).abs() / expected.out_tol
    lse_ratio = (s - expected.lse).abs() / expected.lse_tol
    nonzero = int((expected.exact_zero & (o != 0)).sum())
    summary = {"out_ratio": float(out_ratio.max()), "lse_ratio": float(lse_ratio.max()),
               "out_failures": int((out_ratio > 1).sum()), "lse_failures": int((lse_ratio > 1).sum()),
               "exact_zero_channels": int(expected.exact_zero.sum()), "nonzero_failures": nonzero}
    if summary["out_failures"] or summary["lse_failures"] or nonzero:
        summary["worst_out_index"] = torch.nonzero(out_ratio == out_ratio.max())[0].tolist()
        raise Mismatch(f"outputs differ from the FP64 oracle: {summary}", summary)
    return summary
