"""CPU checks of the fixtures, the FP64 oracle, the tolerance check and device ranking."""
from types import SimpleNamespace

import pytest
import torch

import nested_kv
from nested_kv.testing import (BOUNDARIES, Case, ScoreMismatch, compare, correctness_cases,
                               make_keys, oracle, performance_cases, quick_cases)


def test_case_grids():
    cases = performance_cases()
    assert len(cases) == len({c.name for c in cases}) == 24
    assert {c.length for c in cases} == {8192, 16384, 32768}
    assert [c.length for c in quick_cases()] == [63, 257, 65, 8192, 32768]
    names = {c.name for c in correctness_cases()}
    for dtype in ("float16", "bfloat16"):
        for b in (1, 2):
            for h in (4, 8):
                for n in BOUNDARIES:
                    assert Case(dtype, b, h, n).name in names


def test_fixtures_are_deterministic():
    case = Case("bfloat16", 1, 4, 257, "zero_rows")
    assert torch.equal(make_keys(case, 7), make_keys(case, 7))
    assert not torch.equal(make_keys(case, 7), make_keys(case, 8))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_constant_zero_and_single_token(dtype):
    for pattern, target in (("constant", -1.0), ("zeros", 0.0)):
        expected = oracle(make_keys(Case(dtype, 1, 4, 257, pattern), 123))
        assert torch.allclose(expected.scores, torch.full_like(expected.scores, target), atol=1e-12)
    expected = oracle(make_keys(Case(dtype, 2, 8, 1), 456))
    assert torch.allclose(expected.scores, -torch.ones(3, 2, 8, 1, dtype=torch.float64))


def test_exclusion_does_not_hide_nonfinite_or_nonzero():
    keys = make_keys(Case("float16", 1, 4, 64, "cancellation"), 1)
    expected = oracle(keys)
    actual = expected.scores.float()
    assert compare(actual, expected)[0]["excluded"] == 256
    actual[0].fill_(float("nan"))
    with pytest.raises(ScoreMismatch):
        compare(actual, expected)
    expected = oracle(torch.zeros_like(keys))
    actual = expected.scores.float()
    actual[0, 0, 0, 0] = 1e-8
    with pytest.raises(ScoreMismatch):
        compare(actual, expected)


def test_threshold_sides_and_error_detection():
    for pattern, excluded in (("threshold_below", 256), ("threshold_above", 0)):
        expected = oracle(make_keys(Case("bfloat16", 1, 4, 64, pattern), 1))
        assert compare(expected.scores.float(), expected)[0]["excluded"] == excluded
    expected = oracle(make_keys(Case("float16", 1, 4, 1025), 2))
    assert all(row["excluded"] == 0 for row in compare(expected.scores.float(), expected))
    bad = expected.scores.float()
    bad[1, 0, 0, 255] += 0.01
    with pytest.raises(ScoreMismatch):
        compare(bad, expected)


def test_partial_zero_regions():
    expected = oracle(make_keys(Case("bfloat16", 2, 8, 1025, "partial_zero"), 5))
    assert expected.zero_only[1, :, :, :256].all()
    assert expected.zero_only[2, :, :, 703:768].all()
    assert (expected.scores[expected.zero_only] == 0).all()


def test_best_device_prefers_highest_capability(monkeypatch):
    gpus = [SimpleNamespace(name="Tesla T4", major=7, minor=5, multi_processor_count=40,
                            total_memory=15 * 2**30),
            SimpleNamespace(name="H100", major=9, minor=0, multi_processor_count=132,
                            total_memory=80 * 2**30),
            SimpleNamespace(name="A10", major=8, minor=6, multi_processor_count=72,
                            total_memory=24 * 2**30)]
    monkeypatch.delenv("NESTED_KV_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(gpus))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: gpus[i])
    assert nested_kv.best_device() == torch.device("cuda", 1)
    gpus.pop(1)
    assert nested_kv.best_device() == torch.device("cuda", 1)
    monkeypatch.setenv("NESTED_KV_DEVICE", "cuda:0")
    assert nested_kv.best_device() == torch.device("cuda", 0)
