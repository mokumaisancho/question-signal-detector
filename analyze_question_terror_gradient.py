"""Analyze how question format impacts the probability distribution gradient.

Compares full 32K distributions for different phrasings of the same question
to identify WHERE changes occur (head vs tail) and whether certain formats
produce systematically different impact patterns.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from two_pass_llama_detector import TwoPassLlamaDetector


def get_full_probs(detector, text: str):
    """Return full probability distribution over vocabulary."""
    detector._load()
    llm = detector._llm
    tokens = llm.tokenize(text.encode())
    llm.eval(tokens)
    logits = llm._scores[-1].copy()
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    return probs


def kl_divergence(p, q, eps=1e-10):
    """KL(p || q) = sum p * log(p/q)."""
    return np.sum(p * np.log((p + eps) / (q + eps)))


def js_divergence(p, q, eps=1e-10):
    """Jensen-Shannon divergence."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def rank_by_probs(probs):
    """Return indices sorted by probability descending."""
    return np.argsort(probs)[::-1]


def compare_distributions(detector, base_text, variant_text):
    """Compare two distributions in detail."""
    p_base = get_full_probs(detector, base_text)
    p_var = get_full_probs(detector, variant_text)

    # Full metrics
    kl = kl_divergence(p_base, p_var)
    js = js_divergence(p_base, p_var)

    # Entropies
    h_base = -np.sum(p_base * np.log(p_base + 1e-10))
    h_var = -np.sum(p_var * np.log(p_var + 1e-10))
    delta_h = h_var - h_base

    # Top-K overlap
    ranks_base = rank_by_probs(p_base)
    ranks_var = rank_by_probs(p_var)

    overlaps = {}
    for k in [1, 5, 10, 20, 50, 100]:
        set_base = set(ranks_base[:k])
        set_var = set(ranks_var[:k])
        overlap = len(set_base & set_var) / k
        overlaps[f"top{k}"] = overlap

    # Probability mass shift by rank bands
    # Head: top 10, Mid: 11-100, Tail: 101-1000, Deep tail: 1001+
    bands = {
        "head_top10": (0, 10),
        "mid_11_100": (10, 100),
        "tail_101_1k": (100, 1000),
        "deep_1k+": (1000, len(p_base)),
    }

    band_shifts = {}
    for band_name, (start, end) in bands.items():
        # Mass in this band for each distribution
        mass_base = np.sum(p_base[ranks_base[start:end]])
        mass_var = np.sum(p_var[ranks_var[start:end]])
        band_shifts[band_name] = {
            "mass_base": mass_base,
            "mass_var": mass_var,
            "delta": mass_var - mass_base,
        }

    # L1 distance by rank (how much does each rank's probability change?)
    # Align by rank position (compare P_base[rank_k] vs P_var[rank_k])
    l1_by_rank = np.abs(p_base[ranks_base] - p_var[ranks_base])

    # Total variation distance
    tvd = 0.5 * np.sum(np.abs(p_base - p_var))

    return {
        "kl": kl,
        "js": js,
        "delta_h": delta_h,
        "overlaps": overlaps,
        "band_shifts": band_shifts,
        "l1_by_rank": l1_by_rank,
        "tvd": tvd,
        "h_base": h_base,
        "h_var": h_var,
    }


detector = TwoPassLlamaDetector()

# Test sets with base + variants organized by format category
test_sets = [
    {
        "name": "gravity",
        "base": "What is gravity?",
        "variants": [
            ("interrogative_what", "What is gravity?"),
            ("interrogative_how", "How does gravity work?"),
            ("interrogative_why", "Why do objects fall due to gravity?"),
            ("imperative_explain", "Explain gravity."),
            ("imperative_define", "Define gravity."),
            ("imperative_describe", "Describe how gravity works."),
            ("statement_question", "Gravity is a force that?"),
            ("statement_fill", "The nature of gravity is?"),
            ("indirect", "We need to understand gravity."),
        ],
    },
    {
        "name": "france_capital",
        "base": "What is the capital of France?",
        "variants": [
            ("interrogative_what", "What is the capital of France?"),
            ("interrogative_which", "Which city is the capital of France?"),
            ("interrogative_do", "Do you know the capital of France?"),
            ("imperative_name", "Name the capital of France."),
            ("imperative_state", "State the capital of France."),
            ("imperative_tell", "Tell me the capital of France."),
            ("statement_question", "The capital of France is?"),
            ("statement_fill", "France's capital city is?"),
            ("indirect", "I need to know France's capital."),
        ],
    },
    {
        "name": "topological_persistence",
        "base": "Can topological persistence detect phase transitions?",
        "variants": [
            ("interrogative_can", "Can topological persistence detect phase transitions?"),
            ("interrogative_is", "Is it possible to use topological persistence for phase transition detection?"),
            ("interrogative_do", "Do phase transitions show up in topological persistence?"),
            ("imperative_explain", "Explain whether topological persistence detects phase transitions."),
            ("imperative_analyze", "Analyze if topological persistence reveals phase transitions."),
            ("statement_question", "Topological persistence and phase transitions: connected?"),
            ("statement_hypothesis", "Topological persistence may detect phase transitions."),
            ("indirect", "Researchers study topological persistence and phase transitions."),
        ],
    },
    {
        "name": "mars_colony",
        "base": "What is Mars Colony population in 2035?",
        "variants": [
            ("interrogative_what", "What is Mars Colony population in 2035?"),
            ("interrogative_how", "How many people live on Mars in 2035?"),
            ("interrogative_do", "Do we know the Mars Colony population for 2035?"),
            ("imperative_state", "State the Mars Colony population in 2035."),
            ("imperative_tell", "Tell me the Mars Colony population in 2035."),
            ("statement_question", "The Mars Colony population in 2035 is?"),
            ("statement_hypothesis", "Mars Colony in 2035 has a population of."),
            ("indirect", "People wonder about Mars Colony population in 2035."),
        ],
    },
]

print("=" * 130)
print("QUESTION TERROR GRADIENT ANALYSIS")
print("=" * 130)
print("\nMeasuring how different question formats impact the next-token distribution.")
print("KL divergence and mass shifts reveal where phrasing changes redirect probability mass.\n")

# Collect all results for aggregate analysis
all_comparisons = []
format_effects = {
    "interrogative": [],
    "imperative": [],
    "statement": [],
    "indirect": [],
}

for test_set in test_sets:
    name = test_set["name"]
    base = test_set["base"]
    variants = test_set["variants"]

    print(f"\n{'━' * 130}")
    print(f"CONCEPT: {name}")
    print(f"BASE: \"{base}\"")
    print(f"{'━' * 130}")

    # Compare each variant to base
    print(f"\n{'Variant':<45} {'KL':>8} {'JS':>8} {'ΔH':>8} {'TVD':>8} {'Top1':>6} {'Top5':>6} {'Top10':>7} {'Top100':>8}")
    print("-" * 130)

    for fmt_name, variant_text in variants:
        cmp = compare_distributions(detector, base, variant_text)
        all_comparisons.append({
            "set": name,
            "format": fmt_name.split("_")[0],
            "variant": fmt_name,
            **cmp,
        })

        fmt_prefix = fmt_name.split("_")[0]
        format_effects[fmt_prefix].append(cmp)

        v_short = variant_text[:40] + "..." if len(variant_text) > 43 else variant_text
        print(f"{v_short:<45} {cmp['kl']:>8.4f} {cmp['js']:>8.4f} {cmp['delta_h']:>8.4f} {cmp['tvd']:>8.4f} "
              f"{cmp['overlaps']['top1']:>6.2f} {cmp['overlaps']['top5']:>6.2f} {cmp['overlaps']['top10']:>7.2f} {cmp['overlaps']['top100']:>8.2f}")

    # Band analysis for this set
    print(f"\n{'Band':<20} {'Mass(Base)':>12} {'Mass(Var)':>12} {'ΔMass':>10} {'%Change':>10}")
    print("-" * 80)
    for band_name in ["head_top10", "mid_11_100", "tail_101_1k", "deep_1k+"]:
        deltas = []
        for cmp_item in all_comparisons[-len(variants):]:
            band = cmp_item["band_shifts"][band_name]
            deltas.append(band["delta"])
        mean_delta = np.mean(deltas)
        mean_base = np.mean([cmp_item["band_shifts"][band_name]["mass_base"] for cmp_item in all_comparisons[-len(variants):]])
        pct_change = 100 * mean_delta / mean_base if mean_base > 0 else 0
        print(f"{band_name:<20} {mean_base:>12.4f} {mean_base + mean_delta:>12.4f} {mean_delta:>10.4f} {pct_change:>9.1f}%")

# Aggregate format analysis
print("\n" + "=" * 130)
print("FORMAT IMPACT ANALYSIS")
print("=" * 130)

print(f"\n{'Format':<15} {'N':>4} {'Mean_KL':>10} {'Mean_JS':>10} {'Mean_ΔH':>10} {'Mean_TVD':>10} {'Top1_Ovlp':>10} {'Top10_Ovlp':>11}")
print("-" * 100)
for fmt_name, comparisons in format_effects.items():
    if not comparisons:
        continue
    n = len(comparisons)
    mean_kl = np.mean([c["kl"] for c in comparisons])
    mean_js = np.mean([c["js"] for c in comparisons])
    mean_dh = np.mean([c["delta_h"] for c in comparisons])
    mean_tvd = np.mean([c["tvd"] for c in comparisons])
    mean_top1 = np.mean([c["overlaps"]["top1"] for c in comparisons])
    mean_top10 = np.mean([c["overlaps"]["top10"] for c in comparisons])
    print(f"{fmt_name:<15} {n:>4} {mean_kl:>10.4f} {mean_js:>10.4f} {mean_dh:>10.4f} {mean_tvd:>10.4f} {mean_top1:>10.2f} {mean_top10:>11.2f}")

# Band shift analysis by format
print("\n" + "=" * 130)
print("MASS SHIFT GRADIENT BY FORMAT")
print("=" * 130)
print("\nAverage probability mass shift vs base, by distribution band:")

bands = ["head_top10", "mid_11_100", "tail_101_1k", "deep_1k+"]
print(f"\n{'Format':<15} {'N':>4} ", end="")
for band in bands:
    print(f"{band:>14}", end="")
print()
print("-" * 90)

for fmt_name, comparisons in format_effects.items():
    if not comparisons:
        continue
    n = len(comparisons)
    print(f"{fmt_name:<15} {n:>4} ", end="")
    for band in bands:
        deltas = [c["band_shifts"][band]["delta"] for c in comparisons]
        mean_delta = np.mean(deltas)
        print(f"{mean_delta:>+14.4f}", end="")
    print()

# Concept-specific terror analysis
print("\n" + "=" * 130)
print("TERROR GRADIENT BY CONCEPT")
print("=" * 130)
print("\nWhich concepts are most 'terrified' by format changes? (higher KL = more impact)")

print(f"\n{'Concept':<30} {'Mean_KL':>10} {'Max_KL':>10} {'Format_worst':>15} {'Format_best':>15}")
print("-" * 90)

for test_set in test_sets:
    name = test_set["name"]
    set_comps = [c for c in all_comparisons if c["set"] == name]
    if not set_comps:
        continue

    kls = [c["kl"] for c in set_comps]
    mean_kl = np.mean(kls)
    max_kl = np.max(kls)
    worst = max(set_comps, key=lambda c: c["kl"])
    best = min(set_comps, key=lambda c: c["kl"])

    print(f"{name:<30} {mean_kl:>10.4f} {max_kl:>10.4f} {worst['variant']:<15} {best['variant']:<15}")

# Rank-level gradient visualization
print("\n" + "=" * 130)
print("RANK-LEVEL PROBABILITY CHANGE GRADIENT")
print("=" * 130)
print("\nAverage L1 distance per rank position (how much probability shifts at each rank):")

for test_set in test_sets:
    name = test_set["name"]
    set_comps = [c for c in all_comparisons if c["set"] == name]
    if not set_comps:
        continue

    # Stack L1 by rank
    l1_matrix = np.array([c["l1_by_rank"][:100] for c in set_comps])
    mean_l1 = l1_matrix.mean(axis=0)

    print(f"\n{name}:")
    print(f"  Ranks 1-5:   {mean_l1[0]:.6f}, {mean_l1[1]:.6f}, {mean_l1[2]:.6f}, {mean_l1[3]:.6f}, {mean_l1[4]:.6f}")
    print(f"  Ranks 6-10:  {mean_l1[5]:.6f}, {mean_l1[6]:.6f}, {mean_l1[7]:.6f}, {mean_l1[8]:.6f}, {mean_l1[9]:.6f}")
    print(f"  Ranks 11-20: mean={mean_l1[10:20].mean():.6f}, max={mean_l1[10:20].max():.6f}")
    print(f"  Ranks 21-50: mean={mean_l1[20:50].mean():.6f}, max={mean_l1[20:50].max():.6f}")
    print(f"  Ranks 51-100: mean={mean_l1[50:100].mean():.6f}, max={mean_l1[50:100].max():.6f}")

    # Find where most change happens
    head_mass = mean_l1[:10].sum()
    mid_mass = mean_l1[10:100].sum()
    tail_mass = mean_l1[100:].sum() if len(mean_l1) > 100 else 0
    total = head_mass + mid_mass + tail_mass
    print(f"  Change distribution: head(1-10)={100*head_mass/total:.1f}%, mid(11-100)={100*mid_mass/total:.1f}%, tail(100+)={100*tail_mass/total:.1f}%")

detector._unload()
