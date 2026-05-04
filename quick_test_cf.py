"""Quick counterfactual test (3 questions only)."""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from two_pass_llama_detector import TwoPassLlamaDetector

def main() -> None:
    detector = TwoPassLlamaDetector()
    detector.calibrate(["What is gravity?", "What is the capital of France?"],
                       ["What is Mars Colony population in 2035?", "Can topological persistence detect phase transitions?"])

    questions = [
        ("What if gravity didn't exist?", "known"),
        ("If the Earth had two moons, what would tides be like?", "known"),
        ("What would happen if water froze at 50 degrees?", "known"),
    ]

    correct = 0
    for q, expected in questions:
        result = detector.detect(q)
        pred = "known" if result["is_known"] else "unknown"
        ok = pred == expected
        correct += ok
        print(f"{q[:50]:50} -> {pred:7} (expected {expected}) {'OK' if ok else 'FAIL'}")
        if "consistency_score" in result:
            print(f"  consistency={result['consistency_score']:.3f}, route={result.get('route','N/A')}")

    print(f"\nResult: {correct}/{len(questions)} correct")
    detector._unload()

if __name__ == "__main__":
    main()
