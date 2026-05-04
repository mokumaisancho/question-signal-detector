"""Benchmark question signal detector against any GGUF model.

Usage:
    python benchmark_model.py /path/to/model.gguf [report_name]

Runs the focused 5 edge case tests and reports calibration quality.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from two_pass_llama_detector import TwoPassLlamaDetector, ModelStabilityChecker
from ep_edge_cases import EdgeCaseTester
from memory_guard import MemorySafetyGuard


def run_benchmark(model_path: str, report_name: str | None = None) -> dict:
    """Run focused edge case benchmark on a model."""

    # === PHASE 0: Memory safety gate ===
    print("[MemGuard] Pre-flight audit...", flush=True)
    print(MemorySafetyGuard.report())

    can_load, reason = MemorySafetyGuard.can_load_model(model_path)
    print(f"[MemGuard] {reason}")

    if not can_load:
        print(f"[MemGuard] BLOCKED: {reason}", flush=True)
        return {"model": model_path.split("/")[-1], "blocked": True, "reason": reason}

    # === PHASE 1: Load model ===
    gc.collect()

    original_path = TwoPassLlamaDetector.MODEL_PATH
    TwoPassLlamaDetector.MODEL_PATH = model_path

    model_size_gb = os.path.getsize(model_path) / (1024**3)
    n_ctx = 512 if model_size_gb > 2.0 else 1024
    detector = TwoPassLlamaDetector(n_ctx=n_ctx)

    # Calibration
    known = [
        "What is gravity?",
        "What is the capital of France?",
        "What is DNA?",
        "What is the speed of light?",
    ]
    unknown = [
        "What is Mars Colony population in 2035?",
        "Can topological persistence detect phase transitions?",
    ]

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {model_path.split('/')[-1]}")
    print(f"{'='*70}")

    t0 = time.monotonic()
    detector.calibrate(known, unknown)
    cal_time = time.monotonic() - t0

    # === PHASE 2: Stability check ===
    stability = ModelStabilityChecker(detector)
    stability.calibrate()
    stab_result = stability.check()

    print(f"\n[Stability] score={stab_result['stability_score']:.3f} "
          f"stable={stab_result['is_stable']}")

    # === PHASE 3: Edge cases ===
    tester = EdgeCaseTester(detector)

    focused_cases = [
        ("counterfactual", tester.test_counterfactual),
        ("nonsense", tester.test_nonsense),
        ("ambiguous", tester.test_ambiguous),
        ("meta", tester.test_meta),
        ("niche", tester.test_niche),
    ]

    results = {}
    print(f"\n{'='*70}")
    print("FOCUSED EDGE CASE RESULTS")
    print(f"{'='*70}")

    for name, method in focused_cases:
        gc.collect()
        state = MemorySafetyGuard.audit()
        print(f"[MemGuard] Before {name}: swap={state['swap_gb']:.1f}GB load={state['load_1m']:.1f}", flush=True)

        if state["swap_gb"] > 5.0:
            print(f"[MemGuard] ABORT: swap too high mid-benchmark ({state['swap_gb']:.1f}GB)")
            results[name] = {"passed": False, "error": f"OOM risk: swap={state['swap_gb']:.1f}GB"}
            continue

        try:
            result = method()
            status = "PASS" if result.passed else "FAIL"
            meta = getattr(result, "metadata", {})

            if name == "counterfactual":
                key_metric = f"known_rate={meta.get('known_rate', 0):.2f}"
            elif name == "nonsense":
                key_metric = f"unknown_rate={meta.get('unknown_rate', 0):.2f}"
            elif name == "ambiguous":
                key_metric = f"unknown_rate={meta.get('unknown_rate', 0):.2f}"
            elif name == "meta":
                key_metric = f"known_rate={meta.get('known_rate', 0):.2f}"
            elif name == "niche":
                key_metric = f"niche_accuracy={meta.get('niche_accuracy', 0):.2f}"
            else:
                key_metric = ""

            print(f"\n[{name}]")
            print(f"  {status}: accuracy={result.accuracy:.3f}, mean_score={result.mean_score:.3f}")
            print(f"    {key_metric}")

            results[name] = {
                "passed": bool(result.passed),
                "accuracy": result.accuracy,
                "mean_score": result.mean_score,
                "metadata": meta,
            }
        except (RuntimeError, ValueError, KeyError, TypeError, AttributeError) as e:
            print(f"\n[{name}] ERROR: {e}")
            results[name] = {"passed": False, "error": str(e)}

    passed = sum(1 for r in results.values() if r.get("passed", False))
    total = len(focused_cases)

    print(f"\n{'='*70}")
    print(f"RESULT: {passed}/{total} passed")
    print(f"{'='*70}")

    # === PHASE 4: Cleanup ===
    detector._unload()
    TwoPassLlamaDetector.MODEL_PATH = original_path
    gc.collect()

    # Post-benchmark orphan check
    post = MemorySafetyGuard.audit()
    print(f"[MemGuard] Post-benchmark: swap={post['swap_gb']:.1f}GB orphans={len(post['orphans'])}")

    summary = {
        "model": model_path.split("/")[-1],
        "model_path": model_path,
        "model_size_gb": model_size_gb,
        "calibration_time_sec": cal_time,
        "stability": {
            "is_stable": stab_result["is_stable"],
            "score": stab_result["stability_score"],
        },
        "results": results,
        "passed": passed,
        "total": total,
        "post_audit": {
            "swap_gb": post["swap_gb"],
            "orphans": len(post["orphans"]),
        },
    }

    if report_name:
        with open(report_name, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nReport saved to {report_name}")

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_model.py <path/to/model.gguf> [report.json]")
        sys.exit(1)

    model_path = sys.argv[1]
    report_name = sys.argv[2] if len(sys.argv) > 2 else None

    run_benchmark(model_path, report_name)
