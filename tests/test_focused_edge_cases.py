"""Focused edge case test for hardened detector.

Runs only the 5 previously-failing edge cases to verify improvements.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ep_edge_cases import EdgeCaseTester
from two_pass_llama_detector import TwoPassLlamaDetector

def main() -> None:
    detector = TwoPassLlamaDetector()
    known = ["What is gravity?", "What is the capital of France?"]
    unknown = ["What is Mars Colony population in 2035?", "Can topological persistence detect phase transitions?"]
    detector.calibrate(known, unknown)

    tester = EdgeCaseTester(detector)

    # Run only the previously-failing edge cases
    tests_to_run = [
        ("counterfactual", tester.test_counterfactual),
        ("nonsense", tester.test_nonsense),
        ("ambiguous", tester.test_ambiguous),
        ("meta", tester.test_meta),
        ("niche", tester.test_niche),
    ]

    print("=" * 70)
    print("FOCUSED EDGE CASE TEST (Previously Failing)")
    print("=" * 70)

    passed = 0
    for name, method in tests_to_run:
        print(f"\n[{name}]")
        try:
            result = method()
            status = "PASS" if result.passed else "FAIL"
            print(f"  {status}: accuracy={result.accuracy:.3f}, mean_score={result.mean_score:.3f}")
            if hasattr(result, 'metadata') and result.metadata:
                for k, v in result.metadata.items():
                    print(f"    {k}={v}")
            if result.passed:
                passed += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'=' * 70}")
    print(f"RESULT: {passed}/{len(tests_to_run)} passed")
    print(f"{'=' * 70}")

    detector._unload()

if __name__ == "__main__":
    main()
