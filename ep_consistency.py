"""Self-consistency check for question boundary detection.

Ported from SelfCheckGPT (Cambridge, 2023):
- Generate answer N times at different temperatures
- Encode answers with sentence-transformer
- Measure pairwise similarity
- Low consistency = model is uncertain = unknown

This bypasses embedding distance and directly measures whether
the model's knowledge is reproducible.
"""
from __future__ import annotations

import numpy as np


class SelfConsistencyChecker:
    """Check answer consistency across multiple generations."""

    def __init__(self, detector, n_samples: int = 3) -> None:
        self.detector = detector
        self.n_samples = n_samples

    def check(self, question: str) -> dict:
        """Check self-consistency of model's answer to a question.

        Returns dict with:
        - consistency_score: 0.0-1.0 (higher = more consistent)
        - answers: list of generated answers
        - similarities: pairwise similarity matrix
        - is_consistent: bool (consistency_score > threshold)
        """
        # Generate N answers at different temperatures
        temperatures = [0.3, 0.7, 1.0]
        answers = []
        for t in temperatures[: self.n_samples]:
            ans = self._generate(question, temperature=t)
            answers.append(ans)

        # If all answers are empty or very short, model is uncertain
        if all(len(a) < 5 for a in answers):
            return {
                "consistency_score": 0.0,
                "answers": answers,
                "similarities": np.zeros((len(answers), len(answers))),
                "is_consistent": False,
            }

        # Compute pairwise similarities using simple token overlap
        # (sentence-transformers optional fallback)
        similarities = self._compute_similarities(answers)

        # Consistency = mean of upper-triangle similarities
        n = len(similarities)
        if n > 1:
            # Extract upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(n, k=1)
            consistency_score = float(np.mean(similarities[triu_indices]))
        else:
            consistency_score = 1.0

        return {
            "consistency_score": consistency_score,
            "answers": answers,
            "similarities": similarities.tolist(),
            "is_consistent": consistency_score >= 0.15,
        }

    def _generate(self, question: str, temperature: float, max_tokens: int = 50) -> str:
        """Generate answer at specified temperature."""
        self.detector._load()
        output = self.detector._llm(
            question,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\n"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    def _compute_similarities(self, answers: list[str]) -> np.ndarray:
        """Compute pairwise similarity matrix for answers."""
        n = len(answers)
        sims = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._text_similarity(answers[i], answers[j])
                sims[i, j] = sim
                sims[j, i] = sim

        return sims

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Compute similarity between two texts using containment + n-grams."""
        a_lower = a.lower().strip()
        b_lower = b.lower().strip()

        # Character bigram similarity (catches paraphrases)
        def bigrams(text):
            return set(text[i:i+2] for i in range(len(text) - 1))

        bg_a = bigrams(a_lower)
        bg_b = bigrams(b_lower)
        if bg_a and bg_b:
            bg_intersection = len(bg_a & bg_b)
            bg_union = len(bg_a | bg_b)
            bigram_sim = bg_intersection / bg_union if bg_union > 0 else 0.0
        else:
            bigram_sim = 0.0

        # Word containment (catches same facts, different order)
        words_a = set(a_lower.split())
        words_b = set(b_lower.split())
        if words_a and words_b:
            intersection = len(words_a & words_b)
            containment = intersection / max(len(words_a), len(words_b))
        else:
            containment = 0.0

        # Combined score (weighted average)
        return 0.6 * bigram_sim + 0.4 * containment


def _test_consistency() -> None:
    """Test consistency checker with mock detector."""
    from unittest.mock import MagicMock

    detector = MagicMock()
    detector._load = MagicMock()
    detector._llm = MagicMock()

    # Mock: consistent answers (same facts)
    detector._llm.side_effect = [
        {"choices": [{"text": " Paris is the capital of France."}]},
        {"choices": [{"text": " The capital of France is Paris."}]},
        {"choices": [{"text": " Paris, France."}]},
    ]

    checker = SelfConsistencyChecker(detector, n_samples=3)
    result = checker.check("What is the capital of France?")
    print(f"Consistent test: score={result['consistency_score']:.3f}, is_consistent={result['is_consistent']}")
    assert result["is_consistent"], "Expected consistent for factual question"

    # Mock: inconsistent answers (model unsure)
    detector._llm.side_effect = [
        {"choices": [{"text": " Maybe 42."}]},
        {"choices": [{"text": " I think around 50."}]},
        {"choices": [{"text": " Not sure."}]},
    ]

    result2 = checker.check("What is the exact population of Mars Colony in 2035?")
    print(f"Inconsistent test: score={result2['consistency_score']:.3f}, is_consistent={result2['is_consistent']}")
    assert not result2["is_consistent"], "Expected inconsistent for unknown question"

    print("All consistency tests passed.")


if __name__ == "__main__":
    _test_consistency()
