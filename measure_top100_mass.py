"""Empirical measurement of top100_mass on real Llama-2-7B."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from two_pass_llama_detector import TwoPassLlamaDetector

detector = TwoPassLlamaDetector()

test_questions = [
    ("What is gravity?", "known"),
    ("What is the capital of France?", "known"),
    ("What is DNA?", "known"),
    ("What is machine learning?", "known"),
    ("What is the speed of light?", "known"),
    ("What is CRISPR?", "known"),
    ("What is Python used for?", "known"),
    ("Who wrote Hamlet?", "known"),
    ("What is photosynthesis?", "known"),
    ("What is the largest planet?", "known"),
    ("Can topological persistence detect phase transitions?", "unknown"),
    ("Can sheaf cohomology detect misinformation cascades?", "unknown"),
    ("Who won the 2032 presidential election?", "unknown"),
    ("Does the Wasserstein distance predict discovery novelty?", "unknown"),
    ("Can persistent homology detect mode collapse?", "unknown"),
    ("What is Mars Colony population in 2035?", "unknown"),
    ("Does quantum error correction work on topological qubits?", "unknown"),
    ("What is the GDP of Mars in 2040?", "unknown"),
    ("Can hyperbolic geometry improve LLM reasoning?", "unknown"),
    ("What is the cure for Alzheimer's in 2028?", "unknown"),
]

print("=" * 80)
print("EMPIRICAL TOP100_MASS MEASUREMENT")
print("=" * 80)
print(f"{'Question':<55} {'Label':>8} {'P_S':>8} {'H_K':>8}")
print("-" * 80)

results = {"known": [], "unknown": []}

for q, label in test_questions:
    try:
        p1 = detector._pass1_uncertainty(q)
        p_s = p1["top100_mass"]
        h_k = p1["next_token_entropy"]
        results[label].append((p_s, h_k))
        q_short = q[:52] + "..." if len(q) > 55 else q
        print(f"{q_short:<55} {label:>8} {p_s:>8.4f} {h_k:>8.4f}")
    except Exception as e:
        print(f"{q[:50]:<55} {label:>8} ERROR: {e}")

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

import numpy as np
for label in ["known", "unknown"]:
    data = results[label]
    if not data:
        continue
    p_vals = [d[0] for d in data]
    h_vals = [d[1] for d in data]
    print(f"\n{label.upper()}:")
    print(f"  top100_mass:  mean={np.mean(p_vals):.4f}, std={np.std(p_vals):.4f}, min={np.min(p_vals):.4f}, max={np.max(p_vals):.4f}")
    print(f"  entropy:      mean={np.mean(h_vals):.4f}, std={np.std(h_vals):.4f}, min={np.min(h_vals):.4f}, max={np.max(h_vals):.4f}")

    # Compute theoretical bounds using empirical means
    from top100_soundness_analysis import entropy_bounds
    h_min, h_max = entropy_bounds(np.mean(p_vals), np.mean(h_vals))
    print(f"  H_true bounds: [{h_min:.3f}, {h_max:.3f}]")

detector._unload()
