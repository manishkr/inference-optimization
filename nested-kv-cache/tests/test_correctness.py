"""Every correctness case against the FP64 oracle, for each backend the GPU supports."""
import pytest
import torch

import nested_kv
from nested_kv.testing import SEEDS, Case, compare, correctness_cases, make_keys, oracle

CASES = correctness_cases()


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_matches_oracle(case, backend, device, full):
    for seed in SEEDS if full else SEEDS[:1]:
        keys = make_keys(case, seed, device=device)
        actual = nested_kv.nested_kv_scores(keys, backend=backend)
        compare(actual, oracle(keys))


@pytest.mark.parametrize("case", [Case("float16", 1, 4, 63), Case("bfloat16", 2, 8, 1025)],
                         ids=lambda c: c.name)
def test_automatic_backend_matches_oracle(case, device):
    keys = make_keys(case, SEEDS[0], device=device)
    compare(nested_kv.nested_kv_scores(keys), oracle(keys))


@pytest.mark.parametrize("case", [Case("float16", 2, 8, 65), Case("bfloat16", 1, 4, 1025),
                                  Case("float16", 2, 8, 8193)], ids=lambda c: c.name)
def test_unaligned_input_matches_oracle(case, device):
    keys = make_keys(case, SEEDS[0], device=device)
    buffer = torch.empty(keys.numel() + 1, dtype=keys.dtype, device=device)
    shifted = buffer[1:].view_as(keys).copy_(keys)
    assert shifted.is_contiguous() and shifted.data_ptr() % 16 != 0
    compare(nested_kv.nested_kv_scores(shifted), oracle(keys))


def test_cpu_input_runs_on_best_device(device):
    keys = make_keys(Case("float16", 1, 4, 300), SEEDS[0])
    actual = nested_kv.nested_kv_scores(keys)
    assert actual.device == device
    compare(actual, oracle(keys))
