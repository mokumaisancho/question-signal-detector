"""Test detector robustness to word order and format variations."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from two_pass_llama_detector import TwoPassLlamaDetector


def full_distribution_entropy(detector, question: str):
    """Get full + top-K entropy from detector."""
    detector._load()
    llm = detector._llm

    tokens = llm.tokenize(question.encode())
    llm.eval(tokens)
    logits = llm._scores[-1].copy()

    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)

    log_probs = np.log(probs + 1e-10)
    h_full = float(-np.sum(probs * log_probs))

    # top-100
    top_k = 100
    top_indices = np.argpartition(logits, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(logits[top_indices])[::-1]]
    top_probs = probs[top_indices]
    top100_mass = float(np.sum(top_probs))
    top_probs_norm = top_probs / top100_mass
    h_top100 = float(-np.sum(top_probs_norm * np.log(top_probs_norm + 1e-10)))

    return {
        "h_full": h_full,
        "h_top100": h_top100,
        "top100_mass": top100_mass,
    }


detector = TwoPassLlamaDetector()

# Base questions with paraphrases
test_sets = [
    {
        "name": "gravity",
        "label": "known",
        "variants": [
            "What is gravity?",
            "Explain gravity.",
            "How does gravity work?",
            "Define gravity.",
            "What do we mean by gravity?",
            "Gravity: what is it?",
        ],
    },
    {
        "name": "france_capital",
        "label": "known",
        "variants": [
            "What is the capital of France?",
            "Which city is the capital of France?",
            "France's capital is what?",
            "Name the capital of France.",
            "The capital of France is?",
            "Do you know the capital of France?",
        ],
    },
    {
        "name": "speed_of_light",
        "label": "known",
        "variants": [
            "What is the speed of light?",
            "How fast does light travel?",
            "State the speed of light.",
            "The speed of light is?",
            "At what velocity does light propagate?",
            "Tell me the speed of light.",
        ],
    },
    {
        "name": "crispr",
        "label": "known",
        "variants": [
            "What is CRISPR?",
            "Explain CRISPR technology.",
            "What does CRISPR stand for?",
            "Describe the CRISPR system.",
            "CRISPR: explain it.",
            "How does CRISPR work?",
        ],
    },
    {
        "name": "topological_persistence",
        "label": "unknown",
        "variants": [
            "Can topological persistence detect phase transitions?",
            "Is it possible to use topological persistence for phase transition detection?",
            "Do phase transitions show up in topological persistence?",
            "Topological persistence and phase transitions: connected?",
            "Would topological persistence reveal a phase transition?",
            "Can we detect phase transitions via topological persistence?",
        ],
    },
    {
        "name": "wasserstein_novelty",
        "label": "unknown",
        "variants": [
            "Does the Wasserstein distance predict discovery novelty?",
            "Can Wasserstein distance measure how novel a discovery is?",
            "Is there a link between Wasserstein distance and scientific novelty?",
            "Would Wasserstein distance indicate novel discoveries?",
            "Do novel discoveries have distinctive Wasserstein distances?",
            "Predicting novelty with Wasserstein distance: does it work?",
        ],
    },
    {
        "name": "mars_colony",
        "label": "unknown",
        "variants": [
            "What is Mars Colony population in 2035?",
            "How many people live on Mars in 2035?",
            "The population of Mars Colony in 2035 is?",
            "In 2035, what is the Mars Colony population?",
            "Mars Colony: population count in 2035?",
            "Do we know the Mars Colony population for 2035?",
        ],
    },
    {
        "name": "hyperbolic_geometry_llm",
        "label": "unknown",
        "variants": [
            "Can hyperbolic geometry improve LLM reasoning?",
            "Would hyperbolic geometry make LLMs reason better?",
            "Is hyperbolic geometry beneficial for language model reasoning?",
            "Do LLMs benefit from hyperbolic geometric representations?",
            "Hyperbolic geometry and LLM reasoning: any connection?",
            "Can we enhance LLM reasoning with hyperbolic geometry?",
        ],
    },
]

print("=" * 120)
print("ROBUSTNESS TO WORD ORDER AND FORMAT VARIATIONS")
print("=" * 120)

all_results = []

for test_set in test_sets:
    name = test_set["name"]
    label = test_set["label"]
    variants = test_set["variants"]

    print(f"\n{'─' * 120}")
    print(f"SET: {name} ({label})")
    print(f"{'─' * 120}")
    print(f"{'Variant':<65} {'H_full':>8} {'H_top100':>9} {'P_S':>8} {'σ_score':>9}")
    print(f"{'-' * 120}")

    set_results = []
    for variant in variants:
        try:
            # Get uncertainty metrics
            p1 = detector._pass1_uncertainty(variant)
            emb_norm = p1["hidden_norm"]

            # Get full entropy
            fe = full_distribution_entropy(detector, variant)

            # Combined score (simplified, without embedding distance)
            entropy_norm = fe["h_top100"] / 5.0
            norm_signal = (emb_norm - 20.0) / 10.0
            truncation_signal = 1.0 - fe["top100_mass"]
            combined = 0.4 * entropy_norm + 0.3 * norm_signal + 0.1 * truncation_signal

            result = {
                "variant": variant,
                "label": label,
                "set": name,
                "h_full": fe["h_full"],
                "h_top100": fe["h_top100"],
                "top100_mass": fe["top100_mass"],
                "hidden_norm": emb_norm,
                "combined": combined,
            }
            set_results.append(result)
            all_results.append(result)

            v_short = variant[:60] + "..." if len(variant) > 65 else variant
            print(f"{v_short:<65} {fe['h_full']:>8.4f} {fe['h_top100']:>9.4f} {fe['top100_mass']:>8.4f} {combined:>9.4f}")
        except Exception as e:
            print(f"{variant[:60]:<65} ERROR: {e}")

    if set_results:
        h_full_vals = [r["h_full"] for r in set_results]
        h_top100_vals = [r["h_top100"] for r in set_results]
        top100_mass_vals = [r["top100_mass"] for r in set_results]
        combined_vals = [r["combined"] for r in set_results]

        print(f"\n  STATS: H_full = {np.mean(h_full_vals):.3f} ± {np.std(h_full_vals):.3f}  "
              f"H_top100 = {np.mean(h_top100_vals):.3f} ± {np.std(h_top100_vals):.3f}  "
              f"P_S = {np.mean(top100_mass_vals):.3f} ± {np.std(top100_mass_vals):.3f}  "
              f"σ_score = {np.mean(combined_vals):.3f} ± {np.std(combined_vals):.3f}")

        cv_hfull = np.std(h_full_vals) / np.mean(h_full_vals) * 100 if np.mean(h_full_vals) > 0 else 0
        cv_htop100 = np.std(h_top100_vals) / np.mean(h_top100_vals) * 100 if np.mean(h_top100_vals) > 0 else 0
        cv_ps = np.std(top100_mass_vals) / np.mean(top100_mass_vals) * 100 if np.mean(top100_mass_vals) > 0 else 0
        cv_combined = np.std(combined_vals) / np.mean(combined_vals) * 100 if np.mean(combined_vals) > 0 else 0

        print(f"  CV%:   H_full = {cv_hfull:.1f}%  H_top100 = {cv_htop100:.1f}%  "
              f"P_S = {cv_ps:.1f}%  σ_score = {cv_combined:.1f}%")

# Overall summary
print("\n" + "=" * 120)
print("OVERALL ROBUSTNESS SUMMARY")
print("=" * 120)

for label in ["known", "unknown"]:
    label_results = [r for r in all_results if r["label"] == label]
    if not label_results:
        continue

    print(f"\n{label.upper()} (n={len(label_results)} variants across {len(test_sets)//2} concepts):")

    # Group by concept
    concepts = {}
    for r in label_results:
        concepts.setdefault(r["set"], []).append(r)

    print(f"\n{'Concept':<30} {'n':>3} {'H_full_cv%':>10} {'H_top100_cv%':>12} {'P_S_cv%':>10} {'σ_cv%':>10} {'Range':>10}")
    print("-" * 120)

    for concept, results in sorted(concepts.items()):
        n = len(results)
        h_full_vals = [r["h_full"] for r in results]
        h_top100_vals = [r["h_top100"] for r in results]
        ps_vals = [r["top100_mass"] for r in results]
        combined_vals = [r["combined"] for r in results]

        cv_hfull = np.std(h_full_vals) / np.mean(h_full_vals) * 100 if np.mean(h_full_vals) > 0 else 0
        cv_htop100 = np.std(h_top100_vals) / np.mean(h_top100_vals) * 100 if np.mean(h_top100_vals) > 0 else 0
        cv_ps = np.std(ps_vals) / np.mean(ps_vals) * 100 if np.mean(ps_vals) > 0 else 0
        cv_combined = np.std(combined_vals) / np.mean(combined_vals) * 100 if np.mean(combined_vals) > 0 else 0

        h_range = np.max(h_full_vals) - np.min(h_full_vals)

        print(f"{concept:<30} {n:>3} {cv_hfull:>9.1f}% {cv_htop100:>11.1f}% {cv_ps:>9.1f}% {cv_combined:>9.1f}% {h_range:>9.3f}")

    # Aggregate stats
    all_cv_hfull = []
    all_cv_htop100 = []
    all_cv_ps = []
    all_cv_combined = []
    all_ranges = []

    for concept, results in concepts.items():
        h_full_vals = [r["h_full"] for r in results]
        h_top100_vals = [r["h_top100"] for r in results]
        ps_vals = [r["top100_mass"] for r in results]
        combined_vals = [r["combined"] for r in results]

        if np.mean(h_full_vals) > 0:
            all_cv_hfull.append(np.std(h_full_vals) / np.mean(h_full_vals) * 100)
        if np.mean(h_top100_vals) > 0:
            all_cv_htop100.append(np.std(h_top100_vals) / np.mean(h_top100_vals) * 100)
        if np.mean(ps_vals) > 0:
            all_cv_ps.append(np.std(ps_vals) / np.mean(ps_vals) * 100)
        if np.mean(combined_vals) > 0:
            all_cv_combined.append(np.std(combined_vals) / np.mean(combined_vals) * 100)
        all_ranges.append(np.max(h_full_vals) - np.min(h_full_vals))

    print(f"\n{'AGGREGATE':<30} {'':>3} {np.mean(all_cv_hfull):>9.1f}% {np.mean(all_cv_htop100):>11.1f}% "
          f"{np.mean(all_cv_ps):>9.1f}% {np.mean(all_cv_combined):>9.1f}% {np.mean(all_ranges):>9.3f}")

# Cross-format stability: same concept, different formats
print("\n" + "=" * 120)
print("FORMAT-SPECIFIC ANALYSIS")
print("=" * 120)

formats = {
    "question": ["What is", "Which", "How does", "Do you know", "Would"],
    "imperative": ["Explain", "Define", "Describe", "State", "Name", "Tell me"],
    "statement": ["The", "is?", ": what", "and"],
}

for label in ["known", "unknown"]:
    print(f"\n{label.upper()}:")
    for fmt_name, prefixes in formats.items():
        fmt_results = [r for r in all_results
                       if r["label"] == label
                       and any(r["variant"].startswith(p) or p in r["variant"][:20] for p in prefixes)]
        if fmt_results:
            h_vals = [r["h_full"] for r in fmt_results]
            print(f"  {fmt_name:<12}: n={len(fmt_results):>2}, H_full={np.mean(h_vals):.3f} ± {np.std(h_vals):.3f}")

# Decision stability: do paraphrases flip the is_known decision?
print("\n" + "=" * 120)
print("DECISION STABILITY")
print("=" * 120)

THRESHOLD = 0.5
for test_set in test_sets:
    name = test_set["name"]
    label = test_set["label"]
    variants = test_set["variants"]

    set_results = [r for r in all_results if r["set"] == name]
    if not set_results:
        continue

    decisions = ["known" if r["combined"] < THRESHOLD else "unknown" for r in set_results]
    unique_decisions = set(decisions)

    if len(unique_decisions) == 1:
        stability = "STABLE"
        decision_str = list(unique_decisions)[0]
    else:
        stability = "UNSTABLE"
        decision_str = f"mixed ({decisions.count('known')}/{len(decisions)} known)"

    correct = (list(unique_decisions)[0] == label) if len(unique_decisions) == 1 else "mixed"
    print(f"{name:<35} {label:<7} → {decision_str:<25} [{stability}]")

detector._unload()
