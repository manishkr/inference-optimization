"""Every correctness case against the FP64 oracle, for each backend the GPU supports."""
import pytest
import torch

import residual_kv
from residual_kv.testing import SEEDS, Case, compare, correctness_cases, make_inputs, oracle

CASES = correctness_cases(full=False)
FULL_ONLY = [c for c in correctness_cases(full=True) if c not in set(CASES)]
_ORACLES = {}


def _expected(case, seed, inputs):
    key = (case, seed)
    if key not in _ORACLES:
        _ORACLES[key] = oracle(inputs)
    return _ORACLES[key]


def _check(case, backend, device, seeds):
    for seed in seeds:
        inputs = make_inputs(case, seed, device=device)
        out, lse = residual_kv.residual_attention(*inputs.args(), backend=backend)
        compare(out, lse, _expected(case, seed, inputs))
        del inputs, out, lse


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_matches_oracle(case, backend, device, full):
    _check(case, backend, device, SEEDS if full else SEEDS[:1])


@pytest.mark.parametrize("case", FULL_ONLY, ids=[c.name for c in FULL_ONLY])
def test_matches_oracle_extra_lengths(case, backend, device, full):
    if not full:
        pytest.skip("pattern cases at every length run with --full")
    _check(case, backend, device, SEEDS)


@pytest.mark.parametrize("case", [Case("float16", 1, 1, 63, 0, "random"),
                                  Case("bfloat16", 2, 5, 1000, 3, "quantized")],
                         ids=lambda c: c.name)
def test_automatic_backend_matches_oracle(case, device):
    inputs = make_inputs(case, SEEDS[0], device=device)
    compare(*residual_kv.residual_attention(*inputs.args()), oracle(inputs))


def test_cpu_inputs_run_on_best_device(device):
    inputs = make_inputs(Case("bfloat16", 1, 2, 300, 1, "quantized"), SEEDS[0])
    out, lse = residual_kv.residual_attention(*inputs.args())
    assert out.device == device and lse.device == device
    compare(out, lse, oracle(inputs))


def test_bf16_queries_needing_a_second_part(backend, device):
    """BF16 query components far below the query's largest one."""
    case = Case("bfloat16", 1, 3, 257, 1, "random")
    inputs = make_inputs(case, SEEDS[0], device=device)
    q = inputs.query.float()
    q[..., 0] = 60.0
    q[..., 1::4] *= 2.0 ** -40
    q[..., 2::8] = 2.0 ** -100
    inputs.query = q.bfloat16().contiguous()
    compare(*residual_kv.residual_attention(*inputs.args(), backend=backend), oracle(inputs))
