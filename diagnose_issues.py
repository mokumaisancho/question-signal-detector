"""Diagnose why counterfactual and meta tests still fail."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from two_pass_llama_detector import TwoPassLlamaDetector

def main() -> None:
    detector = TwoPassLlamaDetector()
    known = ["What is gravity?", "What is the capital of France?"]
    unknown = ["What is Mars Colony population in 2035?", "Can topological persistence detect phase transitions?"]
    detector.calibrate(known, unknown)

    print("=" * 70)
    print("COUNTERFACTUAL DIAGNOSIS")
    print("=" * 70)

    cf_questions = [
        "What if gravity didn't exist?",
        "If the Earth had two moons, what would tides be like?",
        "What would happen if water froze at 50 degrees?",
    ]

    for q in cf_questions:
        result = detector.detect(q)
        print(f"\nQ: {q}")
        print(f"  route={result.get('route', 'N/A')}")
        print(f"  is_known={result['is_known']}")
        print(f"  score={result['uncertainty_score']:.3f}")
        if 'consistency_score' in result:
            print(f"  consistency={result['consistency_score']:.3f}")
        if result.get('generated_answer'):
            print(f"  answer={result['generated_answer'][:80]}")

    print("\n" + "=" * 70)
    print("META DIAGNOSIS")
    print("=" * 70)

    meta_questions = [
        "What is your training cutoff date?",
        "Who created you?",
        "What model are you?",
        "How many layers do you have?",
        "Can you access the internet?",
    ]

    for q in meta_questions:
        result = detector.detect(q)
        print(f"\nQ: {q}")
        print(f"  route={result.get('route', 'N/A')}")
        print(f"  type={result.get('question_type', 'N/A')}")
        print(f"  is_known={result['is_known']}")
        print(f"  score={result['uncertainty_score']:.3f}")
        print(f"  entropy={result['next_token_entropy']:.2f}")
        print(f"  norm={result['hidden_norm']:.2f}")
        print(f"  embed_signal={result['embedding_signal']:.2f}")

    print("\n" + "=" * 70)
    print("NONSENSE DIAGNOSIS")
    print("=" * 70)

    nonsense_questions = [
        "What is the color of Tuesday?",
        "How fast does dark travel?",
        "Can you fold water?",
    ]

    for q in nonsense_questions:
        result = detector.detect(q)
        print(f"\nQ: {q}")
        print(f"  route={result.get('route', 'N/A')}")
        print(f"  type={result.get('question_type', 'N/A')}")
        print(f"  is_known={result['is_known']}")
        print(f"  score={result['uncertainty_score']:.3f}")

    detector._unload()


if __name__ == "__main__":
    main()
