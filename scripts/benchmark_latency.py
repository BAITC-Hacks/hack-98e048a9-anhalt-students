"""Measure warm forecast-cycle latency and sequential throughput.

Run from the repository root: python scripts/benchmark_latency.py
Use --mock for deterministic offline weather (the real ML/physics pipeline runs).
Initialization, model training, warm-up and console output are not timed.
Weather retrieval, validation, ML, physics and the dispatcher audit are timed.
Results describe this machine and scenario, not verified KEGOC compliance.
"""

import argparse
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TARGET_DATE = "2026-02-14"
HORIZON_HOURS = 24
LATENCY_TARGET_MS = 50.0


def _validate_result(result: dict) -> None:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Forecast failed: {result.get('status')!r}")
    if len(result.get("timeline", [])) != HORIZON_HOURS:
        raise RuntimeError(f"Expected a complete {HORIZON_HOURS}-hour forecast")


def run_benchmark(iterations: int = 15) -> bool:
    """Return True only when every measured cycle is below the 50 ms target."""
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    from backend.agent.pipeline import SamrukWindAgentPipeline

    print("=" * 72)
    print(" Samruk WindAgent AI - Inference Latency Benchmark")
    print(f" Target: Shelek Wind Farm (Turbines 1 & 2); iterations: {iterations}")
    print(f" Scenario: {TARGET_DATE}, {HORIZON_HOURS} hours, farm")
    print(f" Runtime: Python {platform.python_version()} | {platform.platform()}")
    print(" Scope: warm forecast cycles, including weather retrieval/cache access.")
    print(" Excludes: startup, training, warm-up, console output and HTTP serving.")
    print("=" * 72)

    pipe = SamrukWindAgentPipeline()
    warmup = pipe.run_forecast_cycle(TARGET_DATE, HORIZON_HOURS, "farm")
    _validate_result(warmup)

    latencies = []
    model_types = set()
    weather_sources = set()
    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        result = pipe.run_forecast_cycle(TARGET_DATE, HORIZON_HOURS, "farm")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _validate_result(result)
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise RuntimeError("Timer returned an invalid elapsed time")
        latencies.append(elapsed_ms)
        benchmark = result.get("benchmark", {})
        model_types.add(benchmark.get("model_type", "UNKNOWN"))
        weather_sources.add(benchmark.get("data_provenance", "UNKNOWN"))
        print(
            f" Iteration {i:2d}: {elapsed_ms:8.2f} ms | "
            f"Status: {result['status']} | Hours: {len(result['timeline'])}"
        )

    mean_ms = statistics.mean(latencies)
    # Inclusive linear interpolation; a single sample has P95 equal to itself.
    p95_ms = (
        statistics.quantiles(latencies, n=100, method="inclusive")[94]
        if len(latencies) > 1 else latencies[0]
    )
    min_ms, max_ms = min(latencies), max(latencies)
    passed = max_ms < LATENCY_TARGET_MS
    print("-" * 72)
    print(" Summary Statistics:")
    print(f"   Mean Latency: {mean_ms:8.2f} ms")
    print(f"   Min Latency:  {min_ms:8.2f} ms")
    print(f"   Max Latency:  {max_ms:8.2f} ms")
    print(f"   P95 Latency:  {p95_ms:8.2f} ms (inclusive estimate; n={iterations})")
    print(f"   Throughput:   {1000.0 / mean_ms:8.1f} cycles/sec (sequential)")
    print(f"   Model paths: {', '.join(sorted(model_types))}")
    print(f"   Weather sources: {', '.join(sorted(weather_sources))}")
    print("=" * 72)
    print(
        f" [{'PASSED' if passed else 'FAILED'}] Local target: every measured "
        f"cycle < {LATENCY_TARGET_MS:g} ms; observed max = {max_ms:.2f} ms."
    )
    print(" This scenario benchmark does not establish KEGOC compliance.")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--mock", action="store_true", help="Use deterministic offline weather")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be a positive integer")
    if args.mock:
        os.environ["DEMO_MOCK_MODE"] = "true"
    try:
        return 0 if run_benchmark(args.iterations) else 1
    except Exception as exc:
        print(f"[ERROR] Benchmark aborted: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
