"""Unit tests for edge-case routing logic (no model required).

Verifies that the question-type classifier routes questions correctly
and that the detector returns the expected routes.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from unittest.mock import MagicMock
import numpy as np


def test_question_type_routing() -> None:
    """Test that question types are classified correctly."""
    from ep_question_type import QuestionTypeClassifier

    classifier = QuestionTypeClassifier()

    test_cases = [
        ("What if gravity didn't exist?", "counterfactual"),
        ("What is the best programming language?", "subjective"),
        ("What is the color of Tuesday?", "nonsense"),
        ("What is your training cutoff?", "meta"),
        ("What is gravity?", "factual"),
    ]

    for question, expected in test_cases:
        result = classifier.classify(question)
        assert result == expected, f"Expected {expected} for '{question}', got {result}"

    print("  PASS: Question type routing")


def test_detector_routing() -> None:
    """Test detector routes by question type (mocked)."""
    from two_pass_llama_detector import TwoPassLlamaDetector

    detector = TwoPassLlamaDetector()
    detector._loaded = True
    detector._llm = MagicMock()

    # Mock _pass1_uncertainty
    mock_emb = np.ones(128) * 0.5
    detector._pass1_uncertainty = MagicMock(return_value={
        "next_token_entropy": 3.5,
        "hidden_norm": 70.0,
        "embedding": mock_emb,
        "top100_mass": 0.9,
        "n_tokens": 5,
    })

    # Mock _pass2_generate
    detector._pass2_generate = MagicMock(return_value="Test answer")

    # Test subjective routing
    result = detector.detect("What is the best programming language?")
    assert result["route"] == "subjective_abstain", f"Expected subjective_abstain, got {result['route']}"
    assert result["is_known"] is False
    print("  PASS: Subjective routing")

    # Test factual routing (standard path)
    detector._calibrated = True
    detector._known_refs = [mock_emb]
    detector._unknown_refs = [mock_emb * 2.0]

    result = detector.detect("What is gravity?")
    assert result["route"] == "standard"
    print("  PASS: Factual routing")


def test_consistency_checker() -> None:
    """Test self-consistency check logic."""
    from ep_consistency import SelfConsistencyChecker

    detector = MagicMock()
    detector._load = MagicMock()
    detector._llm = MagicMock()

    # Consistent answers
    detector._llm.side_effect = [
        {"choices": [{"text": " Objects would float."}]},
        {"choices": [{"text": " Everything would float away."}]},
        {"choices": [{"text": " No gravity means floating."}]},
    ]

    checker = SelfConsistencyChecker(detector, n_samples=3)
    result = checker.check("What if gravity didn't exist?")
    print(f"  Consistency score: {result['consistency_score']:.3f}, is_consistent={result['is_consistent']}")

    # Should be consistent (all about floating)
    assert result["is_consistent"] or result["consistency_score"] > 0.1
    print("  PASS: Consistency checker")


def test_coherence_probe() -> None:
    """Test semantic coherence probe logic."""
    from ep_coherence import SemanticCoherenceProbe

    probe = SemanticCoherenceProbe()
    probe._calibrated = True
    probe._natural_embeddings = [np.ones(128) * 0.5]

    detector = MagicMock()
    detector._pass1_uncertainty = MagicMock(return_value={
        "embedding": np.ones(128) * 0.5,
        "hidden_norm": 70.0,
        "next_token_entropy": 3.5,
        "top100_mass": 0.9,
    })

    result = probe.check("What is gravity?", detector)
    assert result["is_coherent"] is True
    print("  PASS: Coherence probe (coherent)")

    detector._pass1_uncertainty = MagicMock(return_value={
        "embedding": np.ones(128) * 3.0,  # Far from natural
        "hidden_norm": 40.0,
        "next_token_entropy": 4.5,
        "top100_mass": 0.6,
    })

    result = probe.check("What is the color of Tuesday?", detector)
    assert result["is_coherent"] is False
    print("  PASS: Coherence probe (incoherent)")


def main() -> None:
    print("=" * 70)
    print("ROUTING LOGIC UNIT TESTS (No Model Required)")
    print("=" * 70)

    test_question_type_routing()
    test_detector_routing()
    test_consistency_checker()
    test_coherence_probe()

    print("\n" + "=" * 70)
    print("ALL ROUTING TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
