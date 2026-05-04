"""Extract FULL vocabulary logits and compare with top-100 approximation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from two_pass_llama_detector import TwoPassLlamaDetector


def full_distribution_entropy(detector, question: str):
    """
    Get true entropy from FULL vocabulary distribution.

    Uses llama-cpp-python's eval() + logits property to access raw logits
    for all 32K vocabulary tokens, then computes exact softmax and entropy.
    """
    detector._load()
    llm = detector._llm

    # Tokenize the question
    tokens = llm.tokenize(question.encode())

    # Evaluate to get logits for last position
    llm.eval(tokens)

    # Get raw logits for entire vocabulary
    # _scores shape: (n_tokens, n_vocab), last position = [-1]
    logits = llm._scores[-1].copy()

    # Compute full softmax
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)

    # Full vocabulary entropy
    log_probs = np.log(probs + 1e-10)
    h_full = float(-np.sum(probs * log_probs))

    # Top-100 entropy (same as detector method)
    top_k = 100
    top_indices = np.argpartition(logits, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(logits[top_indices])[::-1]]
    top_probs = probs[top_indices]
    top_probs_norm = top_probs / np.sum(top_probs)
    h_top100 = float(-np.sum(top_probs_norm * np.log(top_probs_norm + 1e-10)))
    top100_mass = float(np.sum(top_probs))

    # Also compute top-10 and top-1000 for sensitivity analysis
    extras = {}
    for k, name in [(10, "top10"), (1000, "top1000")]:
        k_indices = np.argpartition(logits, -k)[-k:]
        k_probs = probs[k_indices]
        k_mass = float(np.sum(k_probs))
        k_probs_norm = k_probs / k_mass
        h_k = float(-np.sum(k_probs_norm * np.log(k_probs_norm + 1e-10)))
        extras[f"h_{name}"] = h_k
        extras[f"{name}_mass"] = k_mass

    return {
        "h_full": h_full,
        "h_top100": h_top100,
        "top100_mass": top100_mass,
        "n_vocab": len(probs),
        **extras,
    }


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

print("=" * 100)
print("FULL VOCABULARY vs TOP-K ENTROPY COMPARISON")
print("=" * 100)
print(f"{'Question':<45} {'Label':>8} {'H_full':>8} {'H_top10':>8} {'H_top100':>8} {'H_top1k':>8} {'bias%':>8}")
print("-" * 100)

results = {"known": [], "unknown": []}

for q, label in test_questions:
    try:
        r = full_distribution_entropy(detector, q)

        # Bias: how much does top-100 underestimate true entropy?
        bias_pct = 100.0 * (r["h_full"] - r["h_top100"]) / r["h_full"] if r["h_full"] > 0 else 0

        results[label].append(r)
        q_short = q[:42] + "..." if len(q) > 45 else q
        print(f"{q_short:<45} {label:>8} {r['h_full']:>8.4f} {r['h_top10']:>8.4f} {r['h_top100']:>8.4f} {r['h_top1000']:>8.4f} {bias_pct:>7.1f}%")
    except Exception as e:
        print(f"{q[:42]:<45} {label:>8} ERROR: {e}")

print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

for label in ["known", "unknown"]:
    data = results[label]
    if not data:
        continue

    print(f"\n{label.upper()} (n={len(data)}):")

    for metric in ["h_full", "h_top10", "h_top100", "h_top1000"]:
        vals = [d[metric] for d in data]
        print(f"  {metric:>10}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    for metric in ["top10_mass", "top100_mass", "top1000_mass"]:
        vals = [d[metric] for d in data]
        print(f"  {metric:>10}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

    # Bias analysis
    biases = []
    for d in data:
        if d["h_full"] > 0:
            biases.append(100.0 * (d["h_full"] - d["h_top100"]) / d["h_full"])
    print(f"  {'top100_bias':>10}: mean={np.mean(biases):.2f}%, std={np.std(biases):.2f}%")

print("\n" + "=" * 100)
print("TREND ANALYSIS")
print("=" * 100)

# Is there a systematic trend in bias?
all_data = results["known"] + results["unknown"]
if all_data:
    h_full_vals = [d["h_full"] for d in all_data]
    h_top100_vals = [d["h_top100"] for d in all_data]
    top100_mass_vals = [d["top100_mass"] for d in all_data]

    # Correlation: does bias increase with full entropy?
    biases = [(d["h_full"] - d["h_top100"]) / d["h_full"] * 100 for d in all_data if d["h_full"] > 0]

    print(f"\nCorrelation between top100_mass and full entropy: {np.corrcoef(top100_mass_vals, h_full_vals)[0,1]:.3f}")
    print(f"Mean top-100 underestimation bias: {np.mean(biases):.2f}%")
    print(f"Std of bias: {np.std(biases):.2f}%")

    # Check if K=100 is sufficient for classification
    print("\n--- Classification sufficiency ---")
    for k_name, k_key in [("top10", "h_top10"), ("top100", "h_top100"), ("top1000", "h_top1000")]:
        known_vals = [d[k_key] for d in results["known"]]
        unknown_vals = [d[k_key] for d in results["unknown"]]
        known_mean = np.mean(known_vals)
        unknown_mean = np.mean(unknown_vals)
        separation = abs(known_mean - unknown_mean) / np.sqrt(
            np.var(known_vals) + np.var(unknown_vals)
        ) if (np.var(known_vals) + np.var(unknown_vals)) > 0 else 0
        print(f"  {k_name:>8}: known_mean={known_mean:.3f}, unknown_mean={unknown_mean:.3f}, separation={separation:.2f} SD")

detector._unload()
