"""CPU checks of the fixtures, the FP64 oracle, its error bound and device ranking."""
import math
from types import SimpleNamespace

import pytest
import torch

import residual_kv
from residual_kv.testing import (ANCHOR_LOOP, DIM, DTYPES, HEADS, LENGTHS, PATTERNS, QUERIES, Case,
                                 Mismatch, _rebuild, _rows, compare, correctness_cases, dequantize,
                                 make_inputs, oracle, pack2, pack4, performance_cases, quantize,
                                 quick_cases, unpack2, unpack4)


def test_case_grids():
    perf = performance_cases()
    assert len(perf) == len({c.name for c in perf}) == 24
    assert {(c.batch, c.length, c.queries) for c in perf} == {
        (b, n, q) for b in (1, 4) for n in (8192, 32768) for q in (1, 4, 8)}
    assert sum(c.residual for c in perf) == 12 and all(c.dtype == "bfloat16" for c in perf)
    assert len(quick_cases()) == 5
    cases = correctness_cases()
    assert len({c.name for c in cases}) == len(cases)
    for n in LENGTHS:
        assert {c.loop for c in cases if c.length == n} == {0, 1, 2, 3}
    triples = {(c.loop, c.queries, c.dtype) for c in cases}
    assert triples >= {(loop, q, d) for loop in range(4) for q in QUERIES for d in DTYPES}
    assert {c.pattern for c in cases} == set(PATTERNS)
    assert {c.batch for c in cases} >= {1, 2, 5, 8, 16}
    assert len(correctness_cases(full=True)) > len(cases)


def test_packing_round_trip():
    gen = torch.Generator().manual_seed(0)
    c4 = torch.randint(0, 16, (3, 7, DIM), generator=gen)
    c2 = torch.randint(0, 4, (3, 7, DIM), generator=gen)
    assert torch.equal(unpack4(pack4(c4)), c4) and torch.equal(unpack2(pack2(c2)), c2)
    assert pack4(c4).shape == (3, 7, 64) and pack2(c2).shape == (3, 7, 32)
    assert int(pack4(torch.tensor([[1, 2] + [0] * 126]))[0, 0]) == 0x21          # low nibble first
    assert int(pack2(torch.tensor([[1, 2, 3, 0] + [0] * 124]))[0, 0]) == 0b00111001  # lowest bits first


def test_quantizer_reconstructs_within_half_a_step():
    x = torch.randn(64, DIM, generator=torch.Generator().manual_seed(1)) * 3
    for bits in (4, 2):
        codes, scale, offset = quantize(x, bits)
        assert scale.dtype == offset.dtype == torch.float8_e4m3fn
        assert codes.min() >= 0 and codes.max() <= 2 ** bits - 1
        groups = (x - dequantize(codes, scale, offset)).abs().unflatten(-1, (4, 32)).amax(-1)
        lo = x.unflatten(-1, (4, 32)).amin(-1)
        slack = (lo - offset.float()).abs() + scale.float() * (2 ** bits)
        assert (groups <= slack).all()


def test_fixtures_are_deterministic():
    case = Case("bfloat16", 1, 3, 65, 1, "quantized")
    a, b, c = make_inputs(case, 7), make_inputs(case, 7), make_inputs(case, 8)
    for x, y, z in zip(a.tensors(), b.tensors(), c.tensors()):
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))
    assert not all(torch.equal(x.view(torch.uint8), z.view(torch.uint8))
                   for x, z in zip(a.tensors(), c.tensors()))


@pytest.mark.parametrize("pattern", PATTERNS)
def test_every_pattern_builds_valid_inputs(pattern):
    for loop in (0, ANCHOR_LOOP):
        x = make_inputs(Case("float16", 1, 2, 65, loop, pattern), 3)
        for t in x.tensors():
            assert t.is_contiguous() and torch.isfinite(t.float()).all()
        assert x.key_cache[1].float().min() >= 0 and x.key_cache[4].float().min() >= 0
        assert x.key_cache[6].float().abs().max() <= 4 and x.value_cache[6].float().abs().max() <= 4


def literal_reference(x):
    """Outputs computed from the index formulas of the operator definition."""
    b, _, nq, _ = x.query.shape
    n = x.key_cache[0].shape[2]
    r = x.loop
    d = torch.arange(DIM)
    g = d // 32

    def code4(cache, bi, hi):
        return ((cache[0][bi, hi][:, d // 2].long() >> (4 * (d % 2))) & 0xF).double()

    def code2(cache, bi, hi):
        return ((cache[3][bi, hi, r][:, d // 4].long() >> (2 * (d % 4))) & 0x3).double()

    def base(cache, bi, hi):
        return (cache[2][bi, hi].double()[:, g] + cache[1][bi, hi].double()[:, g] * code4(cache, bi, hi))

    def resid(cache, bi, hi):
        return cache[5][bi, hi, r].double()[:, g] + cache[4][bi, hi, r].double()[:, g] * code2(cache, bi, hi)

    out = torch.empty(b, HEADS, nq, DIM, dtype=torch.float64)
    lse = torch.empty(b, HEADS, nq, dtype=torch.float64)
    mean = x.key_mean.double()
    for bi in range(b):
        for hi in range(HEADS):
            a = base(x.key_cache, bi, hi) + mean[hi, 3]
            if r == ANCHOR_LOOP:
                k, v = a, base(x.value_cache, bi, hi)
            else:
                ka = x.key_cache[6][bi, hi, r].double()[:, None]
                va = x.value_cache[6][bi, hi, r].double()[:, None]
                k = ka * a + resid(x.key_cache, bi, hi) + mean[hi, r]
                v = va * base(x.value_cache, bi, hi) + resid(x.value_cache, bi, hi)
            k = torch.cat((k, x.new_key[bi, hi].double()))
            v = torch.cat((v, x.new_value[bi, hi].double()))
            for i in range(nq):
                s = (x.query[bi, hi, i].double() * k[:n + i + 1]).sum(-1) / math.sqrt(DIM)
                m = s.max()
                e = torch.exp(s - m)
                out[bi, hi, i] = (e[:, None] * v[:n + i + 1]).sum(0) / e.sum()
                lse[bi, hi, i] = m + torch.log(e.sum())
    return out, lse


@pytest.mark.parametrize("loop", [0, 2, 3])
@pytest.mark.parametrize("pattern", ["quantized", "random", "alpha_signs"])
def test_oracle_matches_literal_formulas(loop, pattern):
    x = make_inputs(Case("float16" if loop % 2 else "bfloat16", 1, 3, 9, loop, pattern), 11)
    expected = oracle(x)
    out, lse = literal_reference(x)
    torch.testing.assert_close(expected.out, out, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(expected.lse, lse, rtol=1e-12, atol=1e-12)


def test_oracle_chunking_does_not_change_results():
    x = make_inputs(Case("bfloat16", 2, 4, 33, 1, "random"), 5)
    whole, chunked = oracle(x), oracle(x, max_elements=1)
    for name in ("out", "lse", "out_tol", "lse_tol", "exact_zero"):
        assert torch.equal(getattr(whole, name), getattr(chunked, name))


def test_tolerance_accepts_oracle_and_rejects_errors():
    x = make_inputs(Case("float16", 1, 2, 65, 1, "quantized"), 2)
    expected = oracle(x)
    out, lse = expected.out.float(), expected.lse.float()
    summary = compare(out, lse, expected)
    assert summary["out_ratio"] < 0.1 and summary["lse_ratio"] < 0.1
    bad = out.clone()
    bad[0, 3, 1, 7] += 3 * float(expected.out_tol[0, 3, 1, 7])
    with pytest.raises(Mismatch):
        compare(bad, lse, expected)
    bad = lse.clone()
    bad[0, 0, 0] += 3 * float(expected.lse_tol[0, 0, 0])
    with pytest.raises(Mismatch):
        compare(out, bad, expected)
    bad = out.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(Mismatch):
        compare(bad, lse, expected)
    with pytest.raises(Mismatch):
        compare(out.double(), lse, expected)


def test_zero_value_channels_must_be_exactly_zero():
    x = make_inputs(Case("bfloat16", 1, 3, 65, 2, "zero_values"), 4)
    expected = oracle(x)
    assert expected.exact_zero.all() and (expected.out == 0).all()
    out, lse = expected.out.float(), expected.lse.float()
    compare(out, lse, expected)
    out[0, 1, 2, 3] = 1e-30
    with pytest.raises(Mismatch):
        compare(out, lse, expected)


def emulate(x, key=None, value=None, weight=None, product=None):
    """FP64 attention with one reduced-precision rounding injected."""
    b, _, nq, _ = x.query.shape
    n = x.key_cache[0].shape[2]
    heads = torch.arange(b * HEADS) % HEADS
    mean = x.key_mean.double()
    k, _ = _rebuild(x.key_cache, x.loop, 0, b * HEADS, mean[heads, 3],
                    mean[heads, x.loop] if x.loop != ANCHOR_LOOP else None)
    v, _ = _rebuild(x.value_cache, x.loop, 0, b * HEADS)
    k = torch.cat((k, _rows(x.new_key, 0, b * HEADS).double()), 1)
    v = torch.cat((v, _rows(x.new_value, 0, b * HEADS).double()), 1)
    if key is not None:
        k = k.to(key).double()
    if value is not None:
        v = v.to(value).double()
    q = _rows(x.query, 0, b * HEADS).double()
    if product is not None:
        s = (q[:, :, None, :] * k[:, None, :, :]).to(product).double().sum(-1)
    else:
        s = q @ k.transpose(1, 2)
    s = s / math.sqrt(DIM)
    visible = torch.arange(n + nq)[None, :] <= n + torch.arange(nq)[:, None]
    s = s.masked_fill(~visible, -math.inf)
    top = s.amax(-1, keepdim=True)
    e = torch.exp(s - top)
    num = e.to(weight).double() if weight is not None else e
    out = (num @ v) / e.sum(-1, keepdim=True)
    lse = top.squeeze(-1) + torch.log(e.sum(-1))
    return out.float().reshape(b, HEADS, nq, DIM), lse.float().reshape(b, HEADS, nq)


@pytest.mark.parametrize("pattern, dtype, n, fault", [
    ("key_rounding", "float16", 1000, dict(key=torch.float16)),
    ("key_rounding", "bfloat16", 1000, dict(key=torch.bfloat16)),
    ("key_rounding", "float16", 8192, dict(key=torch.float16)),
    ("value_rounding", "float16", 1000, dict(value=torch.float16)),
    ("value_rounding", "bfloat16", 1000, dict(value=torch.bfloat16)),
    ("value_rounding", "bfloat16", 8192, dict(value=torch.float16)),
    ("weights_fp16", "float16", 256, dict(weight=torch.float16)),
    ("weights_fp16", "bfloat16", 8192, dict(weight=torch.float16)),
    ("weights_bf16", "bfloat16", 256, dict(weight=torch.bfloat16)),
    ("weights_bf16", "float16", 8192, dict(weight=torch.bfloat16)),
    ("query_code", "float16", 1000, dict(product=torch.float16)),
    ("query_code", "bfloat16", 1000, dict(product=torch.float16)),
], ids=lambda v: v if isinstance(v, str) else str(v) if isinstance(v, int)
   else "-".join(f"{k}:{str(t)[6:]}" for k, t in v.items()))
def test_precision_fixtures_reject_reduced_precision(pattern, dtype, n, fault):
    """An FP64 kernel passes; the same kernel with one reduced-precision rounding fails."""
    for loop in (1, ANCHOR_LOOP):
        x = make_inputs(Case(dtype, 1, 2, n, loop, pattern), 9)
        expected = oracle(x)
        compare(*emulate(x), expected)
        with pytest.raises(Mismatch):
            compare(*emulate(x, **fault), expected)


def test_best_device_prefers_highest_capability(monkeypatch):
    gpus = [SimpleNamespace(name="Tesla T4", major=7, minor=5, multi_processor_count=40,
                            total_memory=15 * 2**30),
            SimpleNamespace(name="H100", major=9, minor=0, multi_processor_count=132,
                            total_memory=80 * 2**30),
            SimpleNamespace(name="A10", major=8, minor=6, multi_processor_count=72,
                            total_memory=24 * 2**30)]
    monkeypatch.delenv("RESIDUAL_KV_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(gpus))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: gpus[i])
    assert residual_kv.best_device() == torch.device("cuda", 1)
    gpus.pop(1)
    assert residual_kv.best_device() == torch.device("cuda", 1)
    monkeypatch.setenv("RESIDUAL_KV_DEVICE", "cuda:0")
    assert residual_kv.best_device() == torch.device("cuda", 0)
