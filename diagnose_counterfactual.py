"""Debug counterfactual self-consistency check."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from two_pass_llama_detector import TwoPassLlamaDetector
from ep_consistency import SelfConsistencyChecker

def main() -> None:
    detector = TwoPassLlamaDetector()
    detector._load()

    q = "What if gravity didn't exist?"
    print(f"Question: {q}")
    print("=" * 70)

    # Generate answers at different temperatures
    for t in [0.3, 0.7, 1.0]:
        output = detector._llm(
            q,
            max_tokens=50,
            temperature=t,
            stop=["\n", "Question:"],
            echo=False,
        )
        ans = output["choices"][0]["text"].strip()
        print(f"\nT={t}: {ans}")

    print("\n" + "=" * 70)
    print("CONSISTENCY CHECK")
    print("=" * 70)

    checker = SelfConsistencyChecker(detector, n_samples=3)
    result = checker.check(q)
    print(f"Consistency score: {result['consistency_score']:.3f}")
    print(f"Is consistent: {result['is_consistent']}")
    print("\nAnswers:")
    for i, ans in enumerate(result['answers']):
        print(f"  {i+1}. {ans}")

    print("\nSimilarity matrix:")
    for row in result['similarities']:
        print(f"  {[f'{x:.3f}' for x in row]}")

    detector._unload()

if __name__ == "__main__":
    main()
