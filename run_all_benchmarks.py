"""Run question signal detection benchmarks on all available GGUF models.

Usage:
    python run_all_benchmarks.py [--models-dir DIR] [--output DIR] [--force]

Runs all models sequentially with memory guard between each.
Generates comparison table and per-model JSON reports.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_model import run_benchmark
from memory_guard import MemorySafetyGuard

DEFAULT_MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
DEFAULT_OUTPUT_DIR = os.environ.get("REPORT_DIR", "/tmp/ep_reports")


def find_models(models_dir: str) -> list[str]:
    """Find all GGUF models in directory, sorted by size (smallest first)."""
    models = []
    for f in os.listdir(models_dir):
        if f.endswith(".gguf"):
            path = os.path.join(models_dir, f)
            models.append((os.path.getsize(path), path, f))
    models.sort()
    return [path for _, path, _ in models]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Multi-model question signal benchmark runner")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Force recalibration (ignore cache)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    models = find_models(args.models_dir)
    if not models:
        print(f"No GGUF models found in {args.models_dir}")
        sys.exit(1)

    print(f"Found {len(models)} models:")
    for p in models:
        size_gb = os.path.getsize(p) / (1024**3)
        print(f"  {os.path.basename(p)} ({size_gb:.1f}GB)")
    print()

    reports = []
    for i, model_path in enumerate(models, 1):
        model_name = os.path.basename(model_path).replace(".gguf", "")
        report_path = os.path.join(args.output, f"{model_name}.json")

        print(f"\n{'='*70}")
        print(f"Model {i}/{len(models)}: {model_name}")
        print(f"{'='*70}")

        # Pre-flight memory check
        state = MemorySafetyGuard.audit()
        print(f"[MemGuard] Pre-model: swap={state['swap_gb']:.1f}GB load={state['load_1m']:.1f} free={state['free_pct']:.0f}%")

        can_load, reason = MemorySafetyGuard.can_load_model(model_path)
        if not can_load:
            print(f"[MemGuard] SKIPPED: {reason}")
            reports.append({"model": model_name, "skipped": True, "reason": reason})
            continue

        t0 = time.monotonic()
        try:
            result = run_benchmark(model_path, report_path)
            elapsed = time.monotonic() - t0
            result["elapsed_sec"] = elapsed
            reports.append(result)
        except (FileNotFoundError, RuntimeError, ValueError, OSError, KeyError) as e:
            print(f"[ERROR] Benchmark failed: {e}")
            reports.append({"model": model_name, "error": str(e)})

        # Post-model cleanup
        print(f"[MemGuard] Post-model cleanup...")
        MemorySafetyGuard.cleanup()

    # Generate comparison table
    print(f"\n\n{'='*80}")
    print("MULTI-MODEL COMPARISON")
    print(f"{'='*80}")

    header = f"{'Model':<35} {'Size':>5} {'Pass':>5} {'Counter':>8} {'Nonsense':>9} {'Ambig':>6} {'Meta':>6} {'Niche':>6} {'Time':>6}"
    print(header)
    print("-" * len(header))

    for r in reports:
        model = r.get("model", "?")
        if r.get("skipped") or r.get("error") or r.get("blocked"):
            reason = r.get("reason", r.get("error", "blocked"))
            print(f"{model:<35} {'---':>5} {'SKIP':>5} {'':>8} {'':>9} {'':>6} {'':>6} {'':>6} {'---':>6}  ({reason})")
            continue

        size = r.get("model_size_gb", 0)
        passed = r.get("passed", 0)
        total = r.get("total", 5)
        results = r.get("results", {})
        elapsed = r.get("elapsed_sec", 0)

        def fmt(test):
            res = results.get(test, {})
            if "error" in res:
                return "ERR"
            return f"{res.get('accuracy', 0):.2f}"

        print(f"{model:<35} {size:>4.1f}G {passed}/{total}   "
              f"{fmt('counterfactual'):>8} {fmt('nonsense'):>9} {fmt('ambiguous'):>6} "
              f"{fmt('meta'):>6} {fmt('niche'):>6} {elapsed:>5.0f}s")

    # Save summary
    summary_path = os.path.join(args.output, "comparison.json")
    with open(summary_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
