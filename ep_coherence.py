"""Semantic coherence probe for nonsense detection.

Detects questions that are syntactically valid but semantically empty
by comparing the question embedding to a distribution of natural questions.

This closes the "syntactic camouflage" hallucination exploitation vector
where nonsense with familiar grammar tricks the detector into "known".
"""
from __future__ import annotations

import numpy as np


class SemanticCoherenceProbe:
    """Check if a question is semantically coherent."""

    # Distribution statistics for natural questions (calibrated empirically)
    # These are approximate; real calibration would compute from dataset
    NATURAL_QUESTION_STATS = {
        "mean_norm": 65.0,
        "std_norm": 8.0,
        "mean_entropy": 3.5,
        "std_entropy": 0.5,
    }

    def __init__(self) -> None:
        self._natural_embeddings: list[np.ndarray] = []
        self._calibrated = False

    def calibrate(self, natural_questions: list[str], detector) -> None:
        """Calibrate on a set of known-natural questions."""
        for q in natural_questions:
            result = detector._pass1_uncertainty(q)
            self._natural_embeddings.append(result["embedding"])
        self._calibrated = True

    def check(self, question: str, detector) -> dict:
        """Check semantic coherence of a question.

        Returns dict with:
        - coherence_score: 0.0-1.0 (higher = more coherent)
        - is_coherent: bool
        - distance_to_natural: float (embedding distance)
        """
        result = detector._pass1_uncertainty(question)
        emb = result["embedding"]

        if self._calibrated and self._natural_embeddings:
            # Distance to nearest natural question
            distances = [
                float(np.linalg.norm(emb - ref))
                for ref in self._natural_embeddings
            ]
            min_dist = min(distances)
            # Convert distance to coherence score (sigmoid-like)
            coherence_score = 1.0 / (1.0 + min_dist / 10.0)
        else:
            # Fallback: use norm and entropy heuristics
            norm = result["hidden_norm"]
            entropy = result["next_token_entropy"]

            stats = self.NATURAL_QUESTION_STATS
            norm_z = abs(norm - stats["mean_norm"]) / stats["std_norm"]
            ent_z = abs(entropy - stats["mean_entropy"]) / stats["std_entropy"]

            # High z-scores = unusual = potentially nonsense
            unusualness = (norm_z + ent_z) / 2.0
            coherence_score = 1.0 / (1.0 + unusualness)
            min_dist = unusualness * 10.0

        return {
            "coherence_score": float(coherence_score),
            "is_coherent": coherence_score >= 0.4,
            "distance_to_natural": float(min_dist),
            "entropy": result["next_token_entropy"],
            "norm": result["hidden_norm"],
        }


def _test_coherence() -> None:
    """Test coherence probe with mock detector."""
    from unittest.mock import MagicMock

    detector = MagicMock()

    # Mock: natural question has "normal" embedding, nonsense has unusual
    normal_emb = np.ones(128) * 0.5
    unusual_emb = np.ones(128) * 2.0  # Far from normal
    detector._pass1_uncertainty.side_effect = [
        # Calibration questions
        {"embedding": normal_emb, "hidden_norm": 68.0, "next_token_entropy": 3.4, "top100_mass": 0.9},
        {"embedding": normal_emb * 1.1, "hidden_norm": 70.0, "next_token_entropy": 3.5, "top100_mass": 0.9},
        # Check 1: natural question (close to calibration)
        {"embedding": normal_emb * 1.05, "hidden_norm": 69.0, "next_token_entropy": 3.45, "top100_mass": 0.9},
        # Check 2: nonsense question (far from calibration)
        {"embedding": unusual_emb, "hidden_norm": 45.0, "next_token_entropy": 4.5, "top100_mass": 0.7},
    ]

    probe = SemanticCoherenceProbe()
    probe.calibrate(["What is gravity?", "What is DNA?"], detector)

    result1 = probe.check("What is the capital of France?", detector)
    print(f"Natural question: coherence={result1['coherence_score']:.3f}, is_coherent={result1['is_coherent']}")
    assert result1["is_coherent"], "Expected coherent for natural question"

    result2 = probe.check("What is the color of Tuesday?", detector)
    print(f"Nonsense question: coherence={result2['coherence_score']:.3f}, is_coherent={result2['is_coherent']}")
    assert not result2["is_coherent"], "Expected incoherent for nonsense"

    print("All coherence tests passed.")


if __name__ == "__main__":
    _test_coherence()
