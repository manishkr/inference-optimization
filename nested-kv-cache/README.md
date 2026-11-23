# Nested KV-cache scoring

CUDA implementation of multiscale key scoring for KV-cache compression, following
the NestedKV method described in [`docs/nestedkv.pdf`](docs/nestedkv.pdf). Each
key is compared against three anchors: the mean of the whole sequence, the mean of
its 256-token block and the mean of its trailing 64-token window.

## Operator

```python
import nested_kv

scores = nested_kv.nested_kv_scores(keys)   # keys: [B, H, N, 128] float16/bfloat16
# scores: new float32 tensor [3, B, H, N], ordered global, block, local
```

Supported shapes: `B ∈ {1, 2}`, `H ∈ {4, 8}`, `1 ≤ N ≤ 32768`, contiguous input.
Other devices, ranks, shapes, dtypes or layouts raise an error. For each batch and
head, with zero-based positions and end-exclusive slices:

```text
eps = 1e-12
normalize(v)    = v / max(||v||, eps)
u[i]            = normalize(float32(keys[i]))
global_anchor   = mean(u[0:N])
block_anchor[i] = mean(u[256*floor(i/256) : min(256*floor(i/256) + 256, N)])
local_anchor[i] = mean(u[max(0, i-63) : i+1])
scores[0, i]    = -dot(u[i], normalize(global_anchor))
scores[1, i]    = -dot(u[i], normalize(block_anchor[i]))
scores[2, i]    = -dot(u[i], normalize(local_anchor[i]))
```

Means divide by the actual number of contributors, so partial blocks and the
first 63 windows are handled. Zero vectors normalize to zero, and normalization
and accumulation are FP32.

## Hardware selection

Selection happens at two levels, both automatic:

1. **GPU:** `nested_kv.best_device()` picks the strongest visible GPU, ranked by
   compute capability, then SM count, then memory. With both a T4 and an H100
   present it picks the H100.
   - CPU tensors passed to `nested_kv_scores` are moved there.
   - CUDA tensors run on their own device.
   - `NESTED_KV_DEVICE=cuda:N` overrides the choice.
2. **Kernels:** chosen per device from its opt-in shared-memory limit.

| Backend    | Kernels | Shared memory per block | Used on |
|------------|---------|-------------------------|---------|
| `fast`     | 2: fused block/local pass, then global pass | ~93 KiB | H100 and other GPUs with ≥ 93 KiB opt-in |
| `portable` | 2: row norms and block sums, then streaming scores | ~63 KiB | T4 and other GPUs from sm_75 up |

The portable scoring kernel runs one CTA per SM over a contiguous run of 128-row tiles. It
keeps the rows in a shared-memory ring, so the 64 rows before each tile are still on chip, and
it prefetches the next tile while scoring the current one. Each key is read once by each kernel.

```python
nested_kv.best_device()                          # e.g. device(type='cuda', index=0)
nested_kv.backend()                              # 'fast' or 'portable'
nested_kv.available_backends()                   # backends the device can run
nested_kv.nested_kv_scores(keys, backend="portable")   # force one
```

## Setup

Requires an NVIDIA driver with CUDA 13 support (≥ 580). Everything else is installed
into the project: uv (if it isn't on `PATH`), Python 3.12, torch 2.9.1 (CUDA 13), and
nvcc 13.2 with its headers from pip, so no system CUDA toolkit is needed.

```bash
cd nested-kv-cache
./bolt.sh setup              # install uv if needed, sync the locked environment, build
./bolt.sh info               # toolchain, GPUs, selected GPU and backend
```

`bolt.sh` is the single entry point:

| Command | What it does |
|---|---|
| `./bolt.sh setup` | Install uv into `.tools/` if missing, `uv sync --locked`, build |
| `./bolt.sh build [--force]` | Build `.build/lib/_nested_kv*.so`; skipped when sources are unchanged |
| `./bolt.sh test [--cpu\|--gpu\|--all] [--full] [pytest args]` | Run the tests |
| `./bolt.sh bench [--quick] [--backend NAME] [--output PATH]` | Latency benchmark |
| `./bolt.sh cases [--quick\|--perf]` | List test or benchmark cases |
| `./bolt.sh info` | Show toolchain, GPUs, selected GPU and backend |
| `./bolt.sh clean [--all]` | Remove build output; `--all` also removes `.venv` and `.tools` |

Run `./bolt.sh help` for every option. The same steps also work with plain uv:
`uv sync`, `uv run python build.py`, `uv run pytest`, `uv run python benchmarks/bench.py`.

The build targets sm_75 and sm_90 SASS plus sm_75 PTX. Set
`NESTED_KV_CUDA_ARCHS` (for example `"80;86;90"`) to compile for other GPUs.

## Tests

```bash
./bolt.sh test               # oracle checks, every correctness case, runtime checks
./bolt.sh test --cpu         # CPU-only checks, no GPU needed
./bolt.sh test --gpu -k n32768
./bolt.sh test --full        # correctness cases with both seeds
```

- `tests/test_oracle.py` (CPU): fixtures, the FP64 oracle, the tolerance check and GPU ranking.
- `tests/test_correctness.py`: compares each backend with an FP64 oracle at `atol=1e-4`,
  `rtol=1e-3`. It covers both dtypes, every `B`/`H` pair, boundary lengths
  `1, 63, 64, 65, 255, 256, 257, 511, 512, 1025, 8193, 16385, 32767, 32768`, and random,
  constant, repeated, zero-row, all-zero, partial-zero, cancelling, near-threshold,
  extreme and tiny inputs.
  - A score is skipped only when its anchor mean has norm below `1e-2` and a nonzero
    contributor. Skipped scores must still be finite.
  - Anchors with only zero contributors must give exactly zero.
- `tests/test_contract.py`:
  - rejection of invalid inputs;
  - fresh contiguous output;
  - input preservation;
  - non-default streams;
  - outputs that are kept while later calls run;
  - determinism;
  - agreement between backends.

Backends the GPU can't run are skipped with the reason stated.

## Benchmark

```bash
./bolt.sh bench                  # 24 cases: N ∈ {8192, 16384, 32768} × dtype × B × H
./bolt.sh bench --quick          # 5 cases
./bolt.sh bench --backend portable --output results.json
```

Every call gets a fresh input, and the L2 cache is flushed before each call.
Calls are queued behind a GPU sleep so host launch time is excluded, then timed
with CUDA events. The reported latency is the median.

Tesla T4 (`portable` backend), median latency in µs:

| N | B×H = 1×4 | 1×8 | 2×4 | 2×8 |
|---:|---:|---:|---:|---:|
| 8192  | 94 | 164 | 166 | 303 |
| 16384 | 167 | 303 | 303 | 576 |
| 32768 | 303 | 578 | 580 | 1139 |

(float16; bfloat16 is within 2%.)

## Layout

```
bolt.sh                 setup, build, test, benchmark entry point
csrc/api.cpp            input validation, backend selection, Python bindings
csrc/scores_fast.cu     fast backend kernels and launch
csrc/common.cuh         device helpers for the fast backend
csrc/scores_portable.cu portable backend kernels and launch
csrc/backends.h         backend interface
nested_kv/__init__.py   Python API and device selection
nested_kv/testing.py    test cases, fixtures, FP64 oracle, tolerance check
tests/                  pytest suite
benchmarks/bench.py     latency benchmark
docs/nestedkv.pdf       paper
```
