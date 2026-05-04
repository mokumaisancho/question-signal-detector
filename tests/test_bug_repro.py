"""Reproduction tests for identified bugs.

BUG-3-002: Entropy computed from only top-100 logprobs — truncation bias
BUG-3-003: stability_checker.check() called twice per detect()
BUG-3-001: Z-score denominator clamped at 0.1
"""

from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock, patch


# ── BUG-3-003: Double stability check ────────────────────────────

def test_detect_calls_check_once():
    """detect() must call stability_checker.check() exactly once."""
    from two_pass_llama_detector import TwoPassLlamaDetector, ModelStabilityChecker

    detector = TwoPassLlamaDetector()
    # Mock the LLM to avoid loading
    mock_llm = MagicMock()
    mock_llm.create_embedding.return_value = {"data": [{"embedding": [[1.0] * 4096]}]}
    mock_llm.return_value = {
        "choices": [{
            "text": "answer",
            "logprobs": {"top_logprobs": [{"a": -0.5, "b": -1.5}]}
        }]
    }
    detector._llm = mock_llm
    detector._loaded = True

    # Calibrate with dummy refs
    detector._known_refs = [np.array([1.0] * 4096)]
    detector._unknown_refs = [np.array([-1.0] * 4096)]
    detector._calibrated = True

    # Mock stability checker
    checker = MagicMock(spec=ModelStabilityChecker)
    checker.check.return_value = {"is_stable": True, "stability_score": 0.5}

    result = detector.detect("test question", stability_checker=checker)

    # BUG: check() is called twice — once at pre-flight, once at end
    assert checker.check.call_count == 1, (
        f"BUG-3-003: check() called {checker.check.call_count} times, expected 1"
    )


# ── BUG-3-001: Z-score denominator clamp ─────────────────────────

def test_zscore_no_clamp_for_small_std():
    """Z-score should not clamp denominator to 0.1 for small baseline std."""
    from two_pass_llama_detector import TwoPassLlamaDetector, ModelStabilityChecker

    detector = TwoPassLlamaDetector()
    mock_llm = MagicMock()
    mock_llm.create_embedding.return_value = {"data": [{"embedding": [[1.0] * 4096]}]}
    mock_llm.return_value = {
        "choices": [{
            "text": "",
            "logprobs": {"top_logprobs": [{"a": -0.5}]}
        }]
    }
    detector._llm = mock_llm
    detector._loaded = True

    checker = ModelStabilityChecker(detector)
    # Baseline matching what the mock produces:
    # - entropy=0 from single-token peaked dist (p=[1.0])
    # - norm=sqrt(4096)=64.0 from [1.0]*4096 embedding
    checker._baseline = {
        "entropy_mean": 0.0,
        "entropy_std": 0.01,  # Very stable model
        "entropy_min": 0.0,
        "entropy_max": 0.0,
        "norm_mean": 64.0,
        "norm_std": 0.5,
        "norm_min": 64.0,
        "norm_max": 64.0,
    }

    result = checker.check()

    # With baseline std = 0.01 and drift near 0, the z-score should be ~0
    # BUG: denominator was max(0.01, 0.1) = 0.1, causing 10x z-score inflation
    # After fix, denominator should be ~0.01, giving z-score ~0-1
    assert result["stability_score"] < 2.0, (
        f"BUG-3-001: stability_score={result['stability_score']:.3f} too high "
        f"due to denominator clamp. Expected < 2.0 for tiny drift with epsilon=1e-6."
    )


# ── BUG-3-002: Entropy truncation ────────────────────────────────

def test_entropy_reports_top100_mass():
    """_pass1_uncertainty must report top100_mass for truncation correction."""
    from two_pass_llama_detector import TwoPassLlamaDetector

    detector = TwoPassLlamaDetector()
    mock_llm = MagicMock()
    mock_llm.create_embedding.return_value = {"data": [{"embedding": [[1.0] * 4096]}]}

    # Simulate broad distribution: top-2 tokens with total mass 0.5
    # (tail has remaining 0.5)
    mock_llm.return_value = {
        "choices": [{
            "text": "x",
            "logprobs": {
                "top_logprobs": [{
                    "the": -0.693,   # ln(0.5) = -0.693
                    "a": -1.693,     # ln(0.25) = -1.386, but with normalization...
                }]
            }
        }]
    }
    detector._llm = mock_llm
    detector._loaded = True

    result = detector._pass1_uncertainty("test")

    # BUG: top100_mass is not computed or returned
    assert "top100_mass" in result, (
        "BUG-3-002: top100_mass missing from _pass1_uncertainty result. "
        "Cannot assess truncation bias without it."
    )


def test_entropy_truncation_detectable():
    """A peaked distribution (high top100_mass) should have different entropy
    characteristics than a broad distribution (low top100_mass)."""
    from two_pass_llama_detector import TwoPassLlamaDetector

    detector = TwoPassLlamaDetector()
    mock_llm = MagicMock()
    mock_llm.create_embedding.return_value = {"data": [{"embedding": [[1.0] * 4096]}]}

    # Peaked: one token dominates with p=0.99
    mock_llm.return_value = {
        "choices": [{
            "text": "the",
            "logprobs": {
                "top_logprobs": [{"the": -0.01005}]  # ln(0.99)
            }
        }]
    }
    detector._llm = mock_llm
    detector._loaded = True

    result_peaked = detector._pass1_uncertainty("test peaked")

    # Broad: 100 tokens each with p=0.005, tail has remaining 0.5 mass
    logprobs_broad = {f"tok{i}": -5.298 for i in range(100)}  # ln(0.005)
    mock_llm.return_value = {
        "choices": [{
            "text": "tok0",
            "logprobs": {"top_logprobs": [logprobs_broad]}
        }]
    }
    result_broad = detector._pass1_uncertainty("test broad")

    # Both should have top100_mass for comparison
    if "top100_mass" in result_peaked and "top100_mass" in result_broad:
        assert result_peaked["top100_mass"] > result_broad["top100_mass"], (
            f"Peaked (mass={result_peaked['top100_mass']:.3f}) should have "
            f"higher top100_mass than broad (mass={result_broad['top100_mass']:.3f})"
        )


if __name__ == "__main__":
    import sys
    failures = []

    tests = [
        ("BUG-3-003: double check", test_detect_calls_check_once),
        ("BUG-3-001: z-score clamp", test_zscore_no_clamp_for_small_std),
        ("BUG-3-002: top100_mass missing", test_entropy_reports_top100_mass),
        ("BUG-3-002: truncation detectable", test_entropy_truncation_detectable),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"PASS: {name}")
        except AssertionError as e:
            print(f"FAIL: {name} -> {e}")
            failures.append((name, str(e)))
        except Exception as e:
            print(f"ERROR: {name} -> {type(e).__name__}: {e}")
            failures.append((name, f"{type(e).__name__}: {e}"))

    print(f"\n{'='*60}")
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)}")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print(f"ALL PASS: {len(tests)}/{len(tests)}")
