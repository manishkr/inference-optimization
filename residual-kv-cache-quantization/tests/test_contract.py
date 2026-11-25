"""Input validation and runtime behaviour of the CUDA operator."""
import pytest
import torch

import residual_kv
from residual_kv.testing import Case, compare, make_inputs, oracle

REJECTED = (RuntimeError, TypeError, ValueError)


def inputs_for(device, n=257, dtype="bfloat16", b=1, q=3, loop=1, pattern="random", seed=0):
    return make_inputs(Case(dtype, b, q, n, loop, pattern), seed, device=device)


def call(x, backend=None, **overrides):
    args = dict(query=x.query, new_key=x.new_key, new_value=x.new_value, loop=x.loop,
                key_cache=x.key_cache, value_cache=x.value_cache, key_mean=x.key_mean)
    args.update(overrides)
    return residual_kv.residual_attention(**args, backend=backend)


def replace(cache, index, tensor):
    cache = list(cache)
    cache[index] = tensor
    return tuple(cache)


def misaligned(t):
    """Contiguous copy of ``t`` whose data pointer is one element past a 16-byte boundary."""
    flat = torch.empty(t.numel() + 16, dtype=t.dtype, device=t.device)
    start = 1 if flat.data_ptr() % 16 == 0 else 0
    view = flat[start:start + t.numel()].view(t.shape)
    view.copy_(t)
    assert view.is_contiguous() and view.data_ptr() % 16 != 0
    return view


INVALID = {
    "query float32": lambda x: dict(query=x.query.float()),
    "new_key dtype differs": lambda x: dict(new_key=x.new_key.half()),
    "query rank 3": lambda x: dict(query=x.query[0]),
    "query 8 heads": lambda x: dict(query=x.query[:, :8].contiguous()),
    "query dim 64": lambda x: dict(query=x.query[..., :64].contiguous()),
    "new_value count differs": lambda x: dict(new_value=x.new_value[:, :, :2].contiguous()),
    "zero queries": lambda x: dict(query=x.query[:, :, :0], new_key=x.new_key[:, :, :0],
                                  new_value=x.new_value[:, :, :0]),
    "nine queries": lambda x: dict(query=x.query.repeat(1, 1, 3, 1), new_key=x.new_key.repeat(1, 1, 3, 1),
                                  new_value=x.new_value.repeat(1, 1, 3, 1)),
    "batch mismatch": lambda x: dict(query=x.query.repeat(2, 1, 1, 1)),
    "loop -1": lambda x: dict(loop=-1),
    "loop 4": lambda x: dict(loop=4),
    "loop float": lambda x: dict(loop=1.0),
    "loop bool": lambda x: dict(loop=True),
    "loop tensor": lambda x: dict(loop=torch.tensor(1)),
    "six cache tensors": lambda x: dict(key_cache=x.key_cache[:6]),
    "codes as int8": lambda x: dict(key_cache=replace(x.key_cache, 0, x.key_cache[0].view(torch.int8))),
    "scale as uint8": lambda x: dict(key_cache=replace(x.key_cache, 1, x.key_cache[1].view(torch.uint8))),
    "scale as e5m2": lambda x: dict(value_cache=replace(x.value_cache, 4,
                                                        x.value_cache[4].view(torch.float8_e5m2))),
    "alpha float16": lambda x: dict(key_cache=replace(x.key_cache, 6, x.key_cache[6].half())),
    "value length differs": lambda x: dict(value_cache=tuple(t[..., :100, :] if t.dim() == 4 and i != 6
                                                             else t for i, t in enumerate(x.value_cache))),
    "residual loops 2": lambda x: dict(key_cache=replace(x.key_cache, 3, x.key_cache[3][:, :, :2].contiguous())),
    "non-contiguous codes": lambda x: dict(
        key_cache=replace(x.key_cache, 0, x.key_cache[0].transpose(2, 3).contiguous().transpose(2, 3))),
    "misaligned codes": lambda x: dict(value_cache=replace(x.value_cache, 0, misaligned(x.value_cache[0]))),
    "misaligned scale": lambda x: dict(key_cache=replace(x.key_cache, 4, misaligned(x.key_cache[4]))),
    "key_mean float16": lambda x: dict(key_mean=x.key_mean.half()),
    "key_mean 3 loops": lambda x: dict(key_mean=x.key_mean[:, :3].contiguous()),
    "key_mean on cpu": lambda x: dict(key_mean=x.key_mean.cpu()),
    "query on cpu": lambda x: dict(query=x.query.cpu()),
}


@pytest.mark.parametrize("reason", list(INVALID))
def test_rejects_invalid_inputs(device, backend, reason):
    x = inputs_for(device)
    with pytest.raises(REJECTED):
        call(x, backend, **INVALID[reason](x))


@pytest.mark.parametrize("n", [0, 32769])
def test_rejects_unsupported_length(device, n):
    x = inputs_for(device, n=1)
    def resize(cache):
        return tuple(t.new_zeros(t.shape[:-2] + (n,) + t.shape[-1:]) if t.dim() >= 4 and i != 6
                     else t.new_zeros(t.shape[:-1] + (n,)) for i, t in enumerate(cache))
    with pytest.raises(REJECTED):
        call(x, key_cache=resize(x.key_cache), value_cache=resize(x.value_cache))


def test_element_aligned_inputs_are_accepted(device, backend):
    x = inputs_for(device, n=300, q=4)
    expected = oracle(x)
    shifted = dict(query=misaligned(x.query), new_key=misaligned(x.new_key),
                   new_value=misaligned(x.new_value),
                   key_cache=replace(x.key_cache, 6, misaligned(x.key_cache[6])),
                   key_mean=misaligned(x.key_mean))
    compare(*call(x, backend, **shifted), expected)


def test_rejects_unknown_backend(device):
    with pytest.raises(ValueError):
        call(inputs_for(device), backend="other")


def test_outputs_are_fresh_contiguous_float32(device, backend):
    x = inputs_for(device, n=1000, b=2, q=5)
    out, lse = call(x, backend)
    assert out.shape == (2, 16, 5, 128) and lse.shape == (2, 16, 5)
    for t in (out, lse):
        assert t.dtype == torch.float32 and t.is_contiguous() and t.device == x.query.device
        assert t.untyped_storage().nbytes() == t.numel() * t.element_size()
    pointers = {t.untyped_storage().data_ptr() for t in x.tensors()}
    assert out.untyped_storage().data_ptr() not in pointers
    assert lse.untyped_storage().data_ptr() not in pointers


@pytest.mark.parametrize("loop", [0, 3])
def test_inputs_are_preserved(device, backend, loop):
    x = inputs_for(device, n=12289, loop=loop, pattern="quantized")
    before = [t.clone() for t in x.tensors()]
    call(x, backend)
    torch.cuda.synchronize(device)
    for old, new in zip(before, x.tensors()):
        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))


def test_non_default_stream(device, backend):
    x = inputs_for(device, n=4097, q=4)
    expected = oracle(x)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        out, lse = call(x, backend)
    stream.synchronize()
    compare(out, lse, expected)


def test_retained_outputs_survive_later_calls(device, backend):
    inputs = [inputs_for(device, n=n, loop=n % 4, seed=s) for s, n in enumerate((193, 1000, 8192, 300))]
    results = [call(x, backend) for x in inputs]
    for _ in range(3):
        for x in inputs:
            call(x, backend)
    for x, (out, lse) in zip(inputs, results):
        compare(out, lse, oracle(x))


def test_reused_input_storage_is_reread(device, backend):
    x = inputs_for(device, n=2049, q=4, seed=1)
    call(x, backend)
    fresh = inputs_for(device, n=2049, q=4, seed=2, pattern="quantized")
    for old, new in zip(x.tensors(), fresh.tensors()):
        old.copy_(new)
    compare(*call(x, backend), oracle(fresh))


def test_repeated_calls_are_deterministic(device, backend):
    x = inputs_for(device, n=16384, b=2, q=8, loop=2)
    first, second = call(x, backend), call(x, backend)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


@pytest.mark.parametrize("loop", [1, 3])
def test_backends_agree(device, loop):
    names = residual_kv.available_backends(device)
    if len(names) < 2:
        pytest.skip("only one backend supported on this device")
    x = inputs_for(device, n=32768, b=1, q=8, loop=loop, pattern="quantized")
    (out_a, lse_a), (out_b, lse_b) = (call(x, name) for name in names)
    torch.testing.assert_close(out_a, out_b, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(lse_a, lse_b, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("b, n", [(1, 1), (1, 256), (1, 257), (5, 1000), (2, 12289), (8, 32768)])
def test_temporary_memory_per_token(device, b, n):
    splits, part_len = residual_kv.partitions(b, n, device)
    assert splits * part_len >= n > (splits - 1) * part_len
    if n <= 256:
        assert splits == 1
    scratch = splits * 8 * (128 + 2) * 4 if splits > 1 else 0
    assert scratch <= 64 * n


@pytest.mark.parametrize("n", [257, 12289])
def test_peak_allocation_within_budget(device, backend, n):
    x = inputs_for(device, n=n, b=2, q=8)
    call(x, backend)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    out, lse = call(x, backend)
    torch.cuda.synchronize(device)
    outputs = out.untyped_storage().nbytes() + lse.untyped_storage().nbytes()
    assert torch.cuda.max_memory_allocated(device) - base - outputs <= 64 * 2 * 16 * n


def test_automatic_selection_reports_backend(device):
    assert residual_kv.backend(device) in residual_kv.available_backends(device)
