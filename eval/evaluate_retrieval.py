from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from src.retriever import query_bugs


def reciprocal_rank(retrieved_ids: list[str], relevant: list[str]) -> float:
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved_ids: list[str], relevant: list[str], k: int) -> int:
    return 1 if any(rid in relevant for rid in retrieved_ids[:k]) else 0


def evaluate(benchmark_path: str, output_path: str | None = None, use_subsystem_filter: bool = True):
    with open(benchmark_path) as f:
        dataset = json.load(f)

    # warmup 3 runs excluded from metrics
    if dataset:
        for _ in range(3):
            try:
                query_bugs(dataset[0]["query_text"], subsystem=None if not use_subsystem_filter else dataset[0].get("query_subsystem"))
            except Exception:
                pass

    hit1 = hit3 = hit5 = 0
    mrr = 0.0
    latencies: list[float] = []
    # also track without filter for ablation
    hit3_nofilter = 0
    mrr_nofilter = 0.0

    results = []

    for entry in dataset:
        qid = entry["query_id"]
        qtext = entry["query_text"]
        relevant = entry.get("relevant_ids", [])
        subsystem = entry.get("query_subsystem") if use_subsystem_filter else None

        # timed query with filter
        t0 = time.perf_counter()
        try:
            retrieved = query_bugs(qtext, subsystem=subsystem)
            retrieved_ids = [r.bug_id for r in retrieved]
        except Exception as exc:
            print(f"query failed {qid}: {exc}")
            retrieved_ids = []
            retrieved = []
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        hr1 = hit_at_k(retrieved_ids, relevant, 1)
        hr3 = hit_at_k(retrieved_ids, relevant, 3)
        hr5 = hit_at_k(retrieved_ids, relevant, 5)
        rr = reciprocal_rank(retrieved_ids, relevant)

        hit1 += hr1
        hit3 += hr3
        hit5 += hr5
        mrr += rr

        # ablation no filter
        try:
            retrieved_nf = query_bugs(qtext, subsystem=None)
            ids_nf = [r.bug_id for r in retrieved_nf]
            hit3_nofilter += hit_at_k(ids_nf, relevant, 3)
            mrr_nofilter += reciprocal_rank(ids_nf, relevant)
        except Exception:
            pass

        results.append({
            "query_id": qid,
            "relevant_ids": relevant,
            "retrieved_ids": retrieved_ids[:5],
            "hit@1": hr1,
            "hit@3": hr3,
            "hit@5": hr5,
            "rr": rr,
            "latency_ms": round(latency, 2),
        })

    n = len(dataset) or 1
    metrics = {
        "total": n,
        "hit_rate@1": round(hit1 / n, 4),
        "hit_rate@3": round(hit3 / n, 4),
        "hit_rate@5": round(hit5 / n, 4),
        "mrr": round(mrr / n, 4),
        "hit_rate@3_nofilter": round(hit3_nofilter / n, 4),
        "mrr_nofilter": round(mrr_nofilter / n, 4),
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0, 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "details": results,
    }

    print(f"Evaluated {n} queries (filter={use_subsystem_filter})")
    print(f"  HitRate@1: {metrics['hit_rate@1']:.3f}  @3: {metrics['hit_rate@3']:.3f}  @5: {metrics['hit_rate@5']:.3f}  MRR: {metrics['mrr']:.3f}")
    print(f"  HitRate@3 nofilter: {metrics['hit_rate@3_nofilter']:.3f}  MRR nofilter: {metrics['mrr_nofilter']:.3f}")
    print(f"  Latency p50: {metrics['latency_p50_ms']}ms  p95: {metrics['latency_p95_ms']}ms  mean: {metrics['latency_mean_ms']}ms")

    # threshold sweep hint
    if metrics["hit_rate@3"] < 0.75:
        print(f"  WARN: HitRate@3 {metrics['hit_rate@3']:.2f} < 0.75 target — tune DISTANCE_THRESHOLD or data")
    else:
        print("  PASS: HitRate@3 >= 0.75")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved results to {output_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval HitRate/MRR/latency")
    parser.add_argument("--benchmark", default="eval/benchmark_dataset.json")
    parser.add_argument("--output", default="eval/results.json")
    parser.add_argument("--no-filter", action="store_true", help="Disable subsystem filter")
    args = parser.parse_args()
    evaluate(args.benchmark, args.output, use_subsystem_filter=not args.no_filter)
