"""GPU latency of residual_attention on fresh inputs with a cold L2 cache.

Each sample queues a batch of calls behind a short GPU sleep so host launch
work is hidden. Before every call the L2 cache is flushed by writing a buffer
larger than it, and the call is timed between CUDA events on the current
stream. Every call in a batch has its own inputs; residual-loop cases cycle
through loops 0, 1 and 2. A case's latency is the median over all timed calls.
Summaries weight residual-loop cases 3:1 against anchor-loop cases: a model
with four loops attends through loops 0-2 three times for every anchor-loop pass.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build  # noqa: E402
import residual_kv  # noqa: E402
from residual_kv.testing import make_inputs, performance_cases, quick_cases  # noqa: E402

BATCH_BYTES = 1 << 30
MAX_BATCH = 32
# Cache bytes one call reads per token: codes and metadata of keys and values.
READ_BYTES_PER_TOKEN = {True: 2 * (64 + 8 + 32 + 8 + 2), False: 2 * (64 + 8)}


def measure(case, device, backend, samples, warmup):
    inputs = [make_inputs(case, 1000, device=device)]
    per_call = sum(t.numel() * t.element_size() for t in inputs[0].tensors())
    calls = max(2, min(MAX_BATCH, BATCH_BYTES // per_call))
    inputs += [make_inputs(case, 1000 + i, device=device) for i in range(1, calls)]
    if case.residual:
        for i, x in enumerate(inputs):
            x.loop = i % 3
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    flush = torch.empty(max(2 * l2_bytes, 1 << 24), dtype=torch.uint8, device=device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(calls)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(calls)]
    latencies = []
    for sample in range(warmup + samples):
        order = torch.randperm(calls).tolist()
        torch.cuda._sleep(50_000_000)
        outputs = []
        for slot, index in enumerate(order):
            flush.zero_()
            starts[slot].record()
            outputs.append(residual_kv.residual_attention(*inputs[index].args(), backend=backend))
            ends[slot].record()
        torch.cuda.synchronize(device)
        if sample >= warmup:
            latencies.extend(1000 * s.elapsed_time(e) for s, e in zip(starts, ends))
        del outputs
    return statistics.median(latencies), min(latencies)


def weighted_geomean(values, weights):
    return math.exp(sum(w * math.log(v) for v, w in zip(values, weights)) / sum(weights))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="time the five quick cases only")
    parser.add_argument("--backend", choices=["fast", "portable"], default=None,
                        help="force a backend (default: automatic)")
    parser.add_argument("--compare", action="store_true",
                        help="also time the portable backend and report the speedup")
    parser.add_argument("--samples", type=int, default=10, help="timed batches per case")
    parser.add_argument("--warmup", type=int, default=2, help="untimed batches per case")
    parser.add_argument("--output", type=Path, help="write results as JSON")
    args = parser.parse_args()

    build.build()
    device = residual_kv.best_device()
    torch.cuda.set_device(device)
    chosen = args.backend or residual_kv.backend(device)
    print(f"device: {torch.cuda.get_device_name(device)} ({device}), backend: {chosen}")

    cases = quick_cases() if args.quick else performance_cases()
    rows = []
    header = f"{'case':<46} {'median us':>10} {'min us':>10} {'GB/s':>8}"
    if args.compare:
        header += f" {'portable us':>12} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for case in cases:
        median, best = measure(case, device, chosen, args.samples, args.warmup)
        moved = case.batch * 16 * case.length * READ_BYTES_PER_TOKEN[case.residual]
        row = {"case": case.name, "median_us": median, "min_us": best, "gbps": moved / median / 1e3,
               "weight": 3 if case.residual else 1}
        line = f"{case.name:<46} {median:>10.2f} {best:>10.2f} {row['gbps']:>8.1f}"
        if args.compare:
            reference, _ = measure(case, device, "portable", args.samples, args.warmup)
            row["portable_us"] = reference
            line += f" {reference:>12.2f} {reference / median:>8.2f}"
        rows.append(row)
        print(line, flush=True)

    weights = [r["weight"] for r in rows]
    summary = {"geomean_us": statistics.geometric_mean(r["median_us"] for r in rows),
               "weighted_geomean_us": weighted_geomean([r["median_us"] for r in rows], weights)}
    print(f"\ngeometric mean latency: {summary['geomean_us']:.2f} us "
          f"(residual loops weighted 3:1: {summary['weighted_geomean_us']:.2f} us)")
    if args.compare:
        summary["weighted_speedup"] = weighted_geomean(
            [r["portable_us"] / r["median_us"] for r in rows], weights)
        print(f"speedup over portable, residual loops weighted 3:1: {summary['weighted_speedup']:.2f}x")
    if args.output:
        args.output.write_text(json.dumps({
            "device": torch.cuda.get_device_name(device), "backend": chosen, **summary,
            "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
