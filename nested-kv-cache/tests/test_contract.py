"""Input validation and runtime behaviour of the CUDA operator."""
import pytest
import torch

import nested_kv
from nested_kv.testing import Case, compare, make_keys, oracle


def keys_for(device, n=257, dtype="float16", b=1, h=4, seed=0):
    return make_keys(Case(dtype, b, h, n), seed, device=device)


@pytest.mark.parametrize("make_bad, reason", [
    (lambda k: k.float(), "float32"),
    (lambda k: k.double(), "float64"),
    (lambda k: k[0], "rank 3"),
    (lambda k: k.unsqueeze(0), "rank 5"),
    (lambda k: k.repeat(3, 1, 1, 1), "batch 3"),
    (lambda k: k[:, :2], "2 heads"),
    (lambda k: k.repeat(1, 3, 1, 1), "12 heads"),
    (lambda k: k[..., :64].contiguous(), "head dim 64"),
    (lambda k: k[:, :, :0], "empty sequence"),
    (lambda k: k.repeat(1, 1, 128, 1)[:, :, :32769].contiguous(), "too long"),
    (lambda k: k.transpose(2, 3).contiguous().transpose(2, 3), "non-contiguous"),
    (lambda k: k[:, :, ::2], "strided"),
], ids=lambda x: x if isinstance(x, str) else None)
def test_rejects_invalid_inputs(device, backend, make_bad, reason):
    bad = make_bad(keys_for(device))
    with pytest.raises(RuntimeError):
        nested_kv.nested_kv_scores(bad, backend=backend)


def test_rejects_unknown_backend(device):
    with pytest.raises(ValueError):
        nested_kv.nested_kv_scores(keys_for(device), backend="other")


def test_output_is_fresh_contiguous_float32(device, backend):
    keys = keys_for(device, n=1025, b=2, h=8)
    out = nested_kv.nested_kv_scores(keys, backend=backend)
    assert out.shape == (3, 2, 8, 1025)
    assert out.dtype == torch.float32 and out.is_contiguous() and out.device == keys.device
    assert out.untyped_storage().nbytes() == out.numel() * out.element_size()
    assert out.untyped_storage().data_ptr() != keys.untyped_storage().data_ptr()


def test_input_is_preserved(device, backend):
    keys = keys_for(device, n=8193, dtype="bfloat16")
    before = keys.clone()
    nested_kv.nested_kv_scores(keys, backend=backend)
    torch.cuda.synchronize(device)
    assert torch.equal(keys.view(torch.int16), before.view(torch.int16))


def test_non_default_stream(device, backend):
    keys = keys_for(device, n=4097)
    expected = oracle(keys)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        out = nested_kv.nested_kv_scores(keys, backend=backend)
    stream.synchronize()
    compare(out, expected)


def test_retained_outputs_survive_later_calls(device, backend):
    inputs = [keys_for(device, n=n, seed=s) for s, n in enumerate((257, 1025, 8192, 300))]
    outputs = [nested_kv.nested_kv_scores(k, backend=backend) for k in inputs]
    for _ in range(3):
        for k in inputs:
            nested_kv.nested_kv_scores(k, backend=backend)
    for keys, out in zip(inputs, outputs):
        compare(out, oracle(keys))


def test_repeated_calls_are_deterministic(device, backend):
    keys = keys_for(device, n=16384, dtype="bfloat16", b=2, h=8)
    first = nested_kv.nested_kv_scores(keys, backend=backend)
    second = nested_kv.nested_kv_scores(keys, backend=backend)
    assert torch.equal(first, second)


def test_backends_agree(device):
    names = nested_kv.available_backends(device)
    if len(names) < 2:
        pytest.skip("only one backend supported on this device")
    keys = keys_for(device, n=32768, b=2, h=8)
    results = [nested_kv.nested_kv_scores(keys, backend=name) for name in names]
    torch.testing.assert_close(results[0], results[1], atol=1e-5, rtol=1e-4)


def test_automatic_selection_reports_backend(device):
    assert nested_kv.backend(device) in nested_kv.available_backends(device)
    if nested_kv.available_backends(device) == ["portable"]:
        with pytest.raises(RuntimeError, match="shared memory"):
            nested_kv.nested_kv_scores(keys_for(device), backend="fast")
