"""Benchmark for Artemis: writes artemis_results.json (numeric only). Diagnostics go to stderr."""
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import dsa_cpu  # noqa: E402
from dsa_cpu.attention import dsa_attention, sparse_available  # noqa: E402
from dsa_cpu.config import GLM52_SMALL  # noqa: E402
from dsa_cpu.indexer import Indexer  # noqa: E402
from dsa_cpu.synthetic import make_batches, make_weights  # noqa: E402
from tests import goldens  # noqa: E402

assert Path(dsa_cpu.__file__).resolve().is_relative_to(ROOT), f"dsa_cpu imported from {dsa_cpu.__file__}, not this checkout"

WARMUP, ITERS = 2, 5
CFG = GLM52_SMALL


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def timed_point(name, weights):
    times = []
    for i in range(WARMUP + ITERS):
        batches = make_batches(name, CFG, weights, seed=1000 + i)
        t0 = time.perf_counter()
        for b in batches:
            dsa_attention(CFG, weights, b)
        times.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times[WARMUP:])
    log(f"{name}: {med:.2f} ms (median of {ITERS}: {[round(t, 1) for t in times[WARMUP:]]})")
    return med


def timed_indexer_8k(weights):
    times = []
    for i in range(WARMUP + ITERS):
        b = make_batches("prefill_8k", CFG, weights, seed=2000 + i)[0]
        indexer = Indexer(CFG, weights)
        t0 = time.perf_counter()
        indexer.forward(b.x, b.q_lora, b.positions, b.index_k_cache)
        times.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times[WARMUP:])
    log(f"indexer at 8k: {med:.2f} ms")
    return med


def sparse_correct(weights) -> int:
    if not sparse_available(CFG):
        log("sparse path not implemented: dsa_sparse_correct = 0")
        return 0
    problems = []
    for name in goldens.POINTS:
        outs, topks = [], []
        for b in make_batches(name, CFG, weights):
            out, topk = dsa_attention(CFG, weights, b)
            outs.append(out)
            topks.append(topk)
        problems += goldens.check_sparse_point(name, torch.cat(outs), None if topks[0] is None else torch.cat(topks))
    for p in problems:
        log(f"golden check: {p}")
    log(f"dsa_sparse_correct = {0 if problems else 1}")
    return 0 if problems else 1


def main():
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    (ROOT / "artemis_results.json").unlink(missing_ok=True)
    weights = make_weights(CFG)
    results = {"dsa_sparse_correct": sparse_correct(weights)}
    results["prefill_ms_4k"] = timed_point("prefill_4k", weights)
    results["prefill_ms_8k"] = timed_point("prefill_8k", weights)
    results["decode_ms_8k"] = timed_point("decode_8k", weights)
    results["indexer_ms_8k"] = timed_indexer_8k(weights) if results["dsa_sparse_correct"] else 0.0
    # Intel's target configuration: 8k tokens in, 1k tokens out. Time to first token plus 1,000 decode steps.
    results["gen_ms_8k_in_1k_out"] = results["prefill_ms_8k"] + 1000 * results["decode_ms_8k"]
    with open(ROOT / "artemis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(json.dumps(results))


if __name__ == "__main__":
    main()
