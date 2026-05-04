"""Debug generation output."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from two_pass_llama_detector import TwoPassLlamaDetector

def main() -> None:
    detector = TwoPassLlamaDetector()
    detector._load()

    questions = [
        "What is gravity?",
        "What if gravity didn't exist?",
        "What is the capital of France?",
    ]

    for q in questions:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print('=' * 70)

        # Generate with different settings
        for temp in [0.3, 0.7]:
            output = detector._llm(
                q,
                max_tokens=50,
                temperature=temp,
                stop=["\n\n", "Question:"],
                echo=False,
            )
            ans = output["choices"][0]["text"].strip()
            print(f"  T={temp}: '{ans}'")

    detector._unload()

if __name__ == "__main__":
    main()
