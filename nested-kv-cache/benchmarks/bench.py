"""GPU latency of nested_kv_scores on fresh inputs with a cold L2 cache.

Each sample queues a batch of calls behind a short GPU sleep so host launch
work is hidden. Before every call the L2 cache is flushed by writing a buffer
larger than it, and the call is timed between CUDA events on the current
stream. A case's latency is the median over all timed calls.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build  # noqa: E402
import nested_kv  # noqa: E402
from nested_kv.testing import make_keys, performance_cases, quick_cases  # noqa: E402

BATCH_BYTES = 1 << 30
MAX_BATCH = 32


def batch_size(case):
    nbytes = case.batch * case.heads * case.length * 128 * 2
    return max(2, min(MAX_BATCH, BATCH_BYTES // nbytes))


def measure(case, device, backend, samples, warmup):
    calls = batch_size(case)
    inputs = [make_keys(case, 1000 + i, device=device) for i in range(calls)]
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
            outputs.append(nested_kv.nested_kv_scores(inputs[index], backend=backend))
            ends[slot].record()
        torch.cuda.synchronize(device)
        if sample >= warmup:
            latencies.extend(1000 * s.elapsed_time(e) for s, e in zip(starts, ends))
        del outputs
    return statistics.median(latencies), min(latencies)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="time the five quick cases only")
    parser.add_argument("--backend", choices=["fast", "portable"], default=None,
                        help="force a backend (default: automatic)")
    parser.add_argument("--samples", type=int, default=10, help="timed batches per case")
    parser.add_argument("--warmup", type=int, default=2, help="untimed batches per case")
    parser.add_argument("--output", type=Path, help="write results as JSON")
    args = parser.parse_args()

    build.build()
    device = nested_kv.best_device()
    torch.cuda.set_device(device)
    chosen = args.backend or nested_kv.backend(device)
    print(f"device: {torch.cuda.get_device_name(device)} ({device}), backend: {chosen}")

    cases = quick_cases() if args.quick else performance_cases()
    rows = []
    header = f"{'case':<38} {'median us':>10} {'min us':>10} {'GB/s':>8}"
    print(header)
    print("-" * len(header))
    for case in cases:
        median, best = measure(case, device, args.backend, args.samples, args.warmup)
        moved = case.batch * case.heads * case.length * (128 * 2 + 3 * 4)
        rows.append({"case": case.name, "median_us": median, "min_us": best,
                     "gbps": moved / median / 1e3})
        print(f"{case.name:<38} {median:>10.2f} {best:>10.2f} {moved / median / 1e3:>8.1f}")

    geomean = statistics.geometric_mean(r["median_us"] for r in rows)
    print(f"\ngeometric mean latency: {geomean:.2f} us")
    if args.output:
        args.output.write_text(json.dumps({
            "device": torch.cuda.get_device_name(device), "backend": chosen,
            "geomean_us": geomean, "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
