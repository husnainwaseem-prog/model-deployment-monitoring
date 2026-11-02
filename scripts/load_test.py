"""Load test: hammers /predict, then sends drifted inputs to trip the monitor.

Phase 1: N in-distribution requests (iris test-set rows)  -> expect no drift.
Phase 2: M shifted requests (sepal_length + 4 cm)         -> expect drift flag.

Writes results/load_test_results.json and results/monitoring_output.md.

Usage: python scripts/load_test.py [--url http://127.0.0.1:8000] [--n 500]
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def compute_stats(latencies):
    arr = np.array(latencies)
    return {"count": int(len(arr)),
            "mean_ms": round(float(arr.mean()), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "max_ms": round(float(arr.max()), 3)}


def fire(url, payloads, concurrency):
    latencies, errors = [], []
    thread_local = threading.local()

    def client():
        # one persistent client per thread -> connection reuse (keep-alive)
        if not hasattr(thread_local, "client"):
            thread_local.client = httpx.Client(timeout=30)
        return thread_local.client

    def one(payload):
        start = time.perf_counter()
        try:
            r = client().post(f"{url}/predict", json=payload)
            latencies.append((time.perf_counter() - start) * 1000)
            if r.status_code != 200:
                errors.append(r.status_code)
        except Exception as exc:  # noqa: BLE001 - counted as error
            errors.append(str(exc))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, payloads))
    wall = time.perf_counter() - t0
    return latencies, errors, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()

    iris = load_iris()
    _, X_test = train_test_split(iris.data, test_size=0.25, random_state=42,
                                 stratify=iris.target)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(X_test), args.n)
    phase1 = [dict(zip(FEATURES, [round(float(v), 2) for v in X_test[i]]))
              for i in idx]

    print(f"phase 1: {args.n} in-distribution requests ...", flush=True)
    lat1, err1, wall1 = fire(args.url, phase1, args.concurrency)

    # snapshot the monitor BEFORE the drifted phase, so "no drift on normal
    # traffic" is measured rather than assumed
    mon_phase1 = httpx.get(f"{args.url}/monitoring", timeout=30).json()

    phase2 = [dict(p, sepal_length=round(p["sepal_length"] + 4.0, 2))
              for p in phase1[:args.m]]
    print(f"phase 2: {args.m} drifted requests (sepal_length + 4cm) ...", flush=True)
    lat2, err2, wall2 = fire(args.url, phase2, args.concurrency)

    monitoring = httpx.get(f"{args.url}/monitoring", timeout=30).json()

    results = {
        "phase1_in_distribution": {
            "requests": args.n, "concurrency": args.concurrency,
            "wall_seconds": round(wall1, 2),
            "throughput_rps": round(args.n / wall1, 1),
            "errors": len(err1), **compute_stats(lat1),
        },
        "phase2_drifted": {
            "requests": args.m, "concurrency": args.concurrency,
            "wall_seconds": round(wall2, 2),
            "throughput_rps": round(args.m / wall2, 1),
            "errors": len(err2), **compute_stats(lat2),
        },
        "monitoring_after_phase1": mon_phase1,
        "monitoring_after": monitoring,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "load_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    p1, p2 = results["phase1_in_distribution"], results["phase2_drifted"]
    table = (
        "# Load test results\n\n"
        "| Phase | Requests | Concurrency | Throughput (req/s) | Mean latency | p95 latency | Errors |\n"
        "|---|---|---|---|---|---|---|\n"
        f"| 1 - in-distribution | {p1['requests']} | {p1['concurrency']} | {p1['throughput_rps']} | {p1['mean_ms']} ms | {p1['p95_ms']} ms | {p1['errors']} |\n"
        f"| 2 - drifted inputs | {p2['requests']} | {p2['concurrency']} | {p2['throughput_rps']} | {p2['mean_ms']} ms | {p2['p95_ms']} ms | {p2['errors']} |\n"
    )
    with open(os.path.join(RESULTS_DIR, "load_test_table.md"), "w") as f:
        f.write(table)

    mon = monitoring
    m1 = mon_phase1["drift"]
    drifted = ", ".join(mon["drift"].get("drifted_features", [])) or "none"
    mout = (
        "# Monitoring output (captured right after the load test)\n\n"
        "## Prediction latency\n\n"
        f"- predictions served: {mon['latency']['count']}\n"
        f"- mean: {mon['latency']['mean_ms']} ms, "
        f"p50: {mon['latency']['p50_ms']} ms, "
        f"p95: {mon['latency']['p95_ms']} ms, "
        f"max: {mon['latency']['max_ms']} ms\n\n"
        "## Input drift\n\n"
        f"- after phase 1 (in-distribution): drift detected **{m1['drift']}** "
        f"(window {m1['n_recent']} inputs)\n"
        f"- after phase 2 (shifted): drift detected **{mon['drift']['drift']}**\n"
        f"- drifted features: {drifted}\n"
        f"- recent window: {mon['drift']['n_recent']} inputs\n\n"
        "Phase 1 is in-distribution (expect no drift); phase 2 shifts "
        "sepal_length by +4 cm (expect the monitor to flag it). Both "
        "snapshots above are the live /monitoring responses.\n"
    )
    with open(os.path.join(RESULTS_DIR, "monitoring_output.md"), "w") as f:
        f.write(mout)

    print(table)
    print(f"drift detected: {mon['drift']['drift']} "
          f"(features: {drifted})")
    print(f"results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
