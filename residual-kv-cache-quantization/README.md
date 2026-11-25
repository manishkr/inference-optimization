# ResidualQuant KV-cache attention

CUDA attention kernels that read a KV cache compressed with ResidualQuant
([`docs/residualquant.pdf`](docs/residualquant.pdf)). The paper targets looped
Transformers; here the layout is that of the four-loop Ouro models (16 KV heads of
dimension 128). Following the paper's Last-loop configuration:

- the last loop's keys and values are INT4 anchors;
- the three earlier loops are INT2 residuals relative to `alpha × anchor`, where alpha is
  fitted per token by least squares;
- every 32-channel group carries an FP8 scale and offset;
- keys get fixed per-head means added back.

The kernels work straight from the packed codes, split long rows into partitions and
merge them (paper, Section 4.2 and Appendix G.2).

A call handles one loop of one layer. As in the paper, the tokens being decoded are
not in the cache yet: their keys and values are passed uncompressed. `Q` such tokens
(1 to 8) can be processed at once. Query `i` attends to every cached token and to new
tokens `0..i`.

## Operator

```python
import residual_kv

out, lse = residual_kv.residual_attention(
    query, new_key, new_value,   # [B, 16, Q, 128], all float16 or all bfloat16
    loop,                        # 0, 1, 2: residual loops; 3: anchor (last) loop
    key_cache, value_cache,      # 7-tuples (see below), each holding every loop
    key_mean)                    # [16, 4, 128] float32, per head and loop
# out: new float32 [B, 16, Q, 128]; lse: new float32 [B, 16, Q] (log of the softmax normaliser)
```

Each cache is `(anchor_codes, anchor_scale, anchor_offset, residual_codes,
residual_scale, residual_offset, alpha)`:

| Tensor | Shape | dtype | Contents |
|---|---|---|---|
| `anchor_codes` | `[B, 16, N, 64]` | uint8 | INT4 codes, channel `2k` in the low nibble of byte `k` |
| `anchor_scale`, `anchor_offset` | `[B, 16, N, 4]` | float8_e4m3fn | one per 32-channel group |
| `residual_codes` | `[B, 16, 3, N, 32]` | uint8 | INT2 codes, channel `4k + j` in bits `2j..2j+1` of byte `k` |
| `residual_scale`, `residual_offset` | `[B, 16, 3, N, 4]` | float8_e4m3fn | one per group and residual loop |
| `alpha` | `[B, 16, 3, N]` | bfloat16 | per token and residual loop |

Batch sizes up to 16, up to 8 queries and up to 32768 cached tokens are supported. Every tensor must be contiguous and on
the same CUDA device. Code tensors must be 16-byte aligned and scale/offset tensors
4-byte aligned. Anything else raises an error. Inputs must be finite and `|alpha| ≤ 4`.

With `D4`/`D2` the group-wise dequantisers (`offset + scale * code`), `μ_r` the key
mean of loop `r` and `d` the head dimension (128), the paper's reconstruction is

```text
A        = D4(key anchor) + μ_3
K_r      = A                                       (r = 3)
         = alpha_r * A + D2(key residual r) + μ_r  (r = 0, 1, 2)
V_r      = D4(value anchor)                        (r = 3)
         = alpha_r * D4(value anchor) + D2(value residual r)
```

The new tokens' keys and values are appended after the `N` cached ones, and for query `i`

```text
out[i] = softmax_n(query[i] . K[n] / sqrt(d)) @ V      over n = 0 .. N + i
lse[i] = log sum_n exp(query[i] . K[n] / sqrt(d))
```

## Numerics

The kernels aim for FP32 accuracy throughout: every operation that is not exact is
either an FP32 operation or one with comparable relative error. The tensor-core path
gets there as follows.

- **Exact tensor-core operands.** An E4M3 number has 4 significant bits and a code has
  at most 4, so `scale * code` has at most 8 significant bits and lies in `[2^-9, 6720]`.
  It is therefore exact in FP16. The query-key product multiplies these values with the
  query on `mma.m16n8k8` with FP32 accumulation. Group offsets, alpha and key means are
  applied afterwards in FP32, using per-group sums of the query and its dot products
  with the means.
- **Queries.** Each query is multiplied by `2^k`, with `k` chosen so that its largest
  component lies in `[2^14, 2^15)`, which makes FP16 queries exact. BF16 components
  that FP16 cannot hold go into a second part, scaled by another `2^32`, in the
  otherwise unused rows 8-15 of the MMA. The query is therefore exact unless its
  components span more than a factor of 2^63.
- **Softmax weights.** In the value product `scale * code` is one operand. The other is
  the softmax weight (times alpha for the anchor part), scaled by `2^13` or `2^15` and
  split into two FP16 parts. Weights down to 2^-16 of the running maximum are carried
  to within 2^-22 of their value. Smaller weights have an absolute error below 2^-38 of
  the maximum. Value offsets are accumulated in FP32 per group.
- **Softmax.** Online softmax in base 2, using the `ex2.approx` and `lg2.approx`
  instructions.

The tests compare every output element with an FP64 reference. The allowed error comes
from a first-order analysis of an FP32-accurate kernel. For query `i`, let `p` be the
FP64 softmax and `ref` the FP64 output. Let `a[n]` be the sum of `|q_d · t|` over every
term `t` of key `n`, with `c[n] = 1 + a[n] / sqrt(d)`. Let `b[n, d]` be the sum of the
magnitudes of the terms of value element `(n, d)`.

```text
out: 2^-20 + 2^-16 (Σ p b + Σ p c |V - ref|) + 4 Σ_n min(2^-24 Σ p b, p[n] (b[n] + |ref|))
lse: 2^-20 + 2^-16 Σ p c                     + 4 Σ_n min(2^-24, p[n])
```

The last term of each bound covers the FP32 sums over tokens. Each token adds at most the smaller of a relative rounding of the whole sum and its own
contribution. A channel in which every visible value term is zero has to be exactly
zero in the output.

## Hardware selection

Selection happens at two levels, both automatic:

1. **GPU:** `residual_kv.best_device()` picks the strongest visible GPU, ranked by compute
   capability, then SM count, then memory.
   - If every input is on the CPU, the inputs are moved there.
   - CUDA tensors run on their own device.
   - `RESIDUAL_KV_DEVICE=cuda:N` overrides the choice.
2. **Kernels:** `fast` on every GPU from sm_75 (Turing) up; `portable` otherwise.

| Backend | Kernels | Products | Used on |
|---|---|---|---|
| `fast` | partition kernel + merge | tensor cores (`mma.m16n8k8`, FP16 operands, FP32 accumulation) | sm_75 and newer |
| `portable` | partition kernel + merge | CUDA cores, every element rebuilt in FP32 | any GPU, and as a cross-check |

Each `(batch, head)` row is split into at most `ceil(N / 256)` partitions, sized from the
GPU's SM count, with one block of four warps per partition. In the `fast` kernel each
warp stages 16-token tiles of codes and metadata in shared memory, loading the next tile
while it computes the current one. It then runs the query-key product, the online
softmax and the value product for all `Q` queries together. A second kernel merges the
partitions and adds the new tokens. A row that fits in one partition is finished by the
partition kernel itself, without temporary memory. Otherwise the partial results take
`Q × 520` bytes per partition.

```python
residual_kv.best_device()               # e.g. device(type='cuda', index=0)
residual_kv.backend()                   # 'fast' or 'portable'
residual_kv.available_backends()        # backends the device can run
residual_kv.partitions(4, 32768)        # (partitions, tokens per partition) for B=4, N=32768
residual_kv.residual_attention(..., backend="portable")   # force one
```

## Setup

Requires an NVIDIA driver with CUDA 13 support (≥ 580). Everything else is installed
into the project: uv (if it isn't on `PATH`), Python 3.12, torch 2.9.1 (CUDA 13), and
nvcc 13.2 with its headers from pip, so no system CUDA toolkit is needed.

```bash
cd residual-kv-cache-quantization
./bolt.sh setup              # install uv if needed, sync the locked environment, build
./bolt.sh info               # toolchain, GPUs, selected GPU and backend
```

`bolt.sh` is the single entry point:

| Command | What it does |
|---|---|
| `./bolt.sh setup` | Install uv into `.tools/` if missing, `uv sync --locked`, build |
| `./bolt.sh build [--force]` | Build `.build/lib/_residual_kv*.so`; skipped when sources are unchanged |
| `./bolt.sh test [--cpu\|--gpu\|--all] [--full] [pytest args]` | Run the tests |
| `./bolt.sh bench [--quick] [--backend NAME] [--compare] [--output PATH]` | Latency benchmark |
| `./bolt.sh cases [--quick\|--perf]` | List test or benchmark cases |
| `./bolt.sh info` | Show toolchain, GPUs, selected GPU and backend |
| `./bolt.sh clean [--all]` | Remove build output; `--all` also removes `.venv` and `.tools` |

Run `./bolt.sh help` for every option. The same steps also work with plain uv:
`uv sync`, `uv run python build.py`, `uv run pytest`, `uv run python benchmarks/bench.py`.

The build targets sm_75 and sm_90 SASS plus sm_75 PTX. Set `RESIDUAL_KV_CUDA_ARCHS`
(for example `"80;86;90"`) to compile for other GPUs.

## Tests

```bash
./bolt.sh test               # oracle checks, every correctness case, runtime checks
./bolt.sh test --cpu         # CPU-only checks, no GPU needed
./bolt.sh test --gpu -k "n32768 and loop3"
./bolt.sh test --full        # every seed, structured patterns at every short length
```

- `tests/test_oracle.py` (CPU):
  - case grids, code packing and the quantiser;
  - the FP64 oracle against a literal transcription of the formulas;
  - the error bound;
  - kernels emulated with one reduced-precision step must fail: FP16 or BF16 keys,
    values or softmax weights, and FP16 query-code products;
  - GPU ranking.
- `tests/test_correctness.py`: compares each backend with the FP64 oracle.
  - Lengths sit around the kernels' boundaries (8-, 16- and 64-token tiles, 256-token
    partitions), from 1 to 32768, each with every loop.
  - Every `(loop, Q, dtype)` combination is covered, with `B` up to 16.
  - Fixtures:
    - caches built by quantising correlated synthetic per-loop states as the paper does
      (affine group quantisation, least-squares alpha);
    - random caches;
    - codes that are all 0 or all at their maximum, zero metadata, zero values, zero and
      negative alpha;
    - zero, peaked and flat queries, and dominant new tokens;
    - FP8 subnormal and largest metadata, and groups whose offset cancels their codes;
    - keys below FP16's normal range, and products beyond it;
    - precision fixtures:
      - keys and values that FP16, TF32 or BF16 cannot hold;
      - softmax-weight ratios just above FP16 or BF16 rounding midpoints;
      - query-code products with 13 significant bits;
      - also confined to a short window inside rows of 8192 and 32768 tokens.
- `tests/test_contract.py`:
  - rejection of invalid inputs;
  - acceptance of element-aligned queries;
  - fresh contiguous outputs;
  - input preservation;
  - non-default streams;
  - retained outputs and reused input storage;
  - determinism;
  - temporary-memory limits;
  - agreement between backends.

Backends the GPU can't run are skipped with the reason stated.

## Benchmark

```bash
./bolt.sh bench                  # 24 cases: loop type x Q in {1, 4, 8} x B in {1, 4} x N in {8192, 32768}
./bolt.sh bench --quick          # 5 cases
./bolt.sh bench --compare --output results.json
```

Every call gets fresh BF16 inputs, and the L2 cache is flushed before each call.
Residual-loop cases cycle through loops 0, 1 and 2. Calls are queued behind a GPU sleep
so host launch time is excluded, then timed with CUDA events. The reported latency is
the median. A residual-loop call reads 228 bytes of cache per token; an anchor-loop
call reads 144.

Tesla T4 (application clock 585 MHz), median latency in µs and speedup over the
`portable` backend:

| Loop | Q | B=1, N=8192 | B=1, N=32768 | B=4, N=8192 | B=4, N=32768 |
|---|---:|---:|---:|---:|---:|
| residual | 1 | 155 (1.9x) | 508 (2.2x) | 524 (2.5x) | 2024 (2.6x) |
| residual | 4 | 160 (2.1x) | 514 (2.4x) | 530 (2.7x) | 2031 (2.9x) |
| residual | 8 | 166 (2.5x) | 520 (2.9x) | 540 (3.1x) | 2044 (3.3x) |
| anchor | 1 | 104 (2.2x) | 314 (2.7x) | 340 (3.0x) | 1229 (3.3x) |
| anchor | 4 | 108 (2.5x) | 319 (3.1x) | 346 (3.4x) | 1283 (3.6x) |
| anchor | 8 | 113 (3.0x) | 324 (3.9x) | 354 (4.0x) | 1245 (4.5x) |

With residual loops weighted 3:1, `fast` is 2.70x faster than
`portable`. The 32768-token rows read the cache at 230 to 245 GB/s.

## Layout

```
bolt.sh                    setup, build, test, benchmark entry point
csrc/api.cpp               input validation, partition planning, backend selection, bindings
csrc/attention_fast.cu     tensor-core partition kernel
csrc/attention_portable.cu CUDA-core partition kernel
csrc/merge.cu              partition merge with the new tokens
csrc/common.cuh            device helpers shared by the kernels (FP8 decoding, finalisation)
csrc/params.h              kernel parameters
csrc/backends.h            backend interface
residual_kv/__init__.py    Python API and device selection
residual_kv/testing.py     test cases, fixtures, FP64 oracle and error bound
tests/                     pytest suite
benchmarks/bench.py        latency benchmark
docs/residualquant.pdf     paper
```
