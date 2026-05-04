"""
Test whether GPT-2-medium attention patterns discriminate known vs unknown questions.

Hypothesis: when the model "knows" the answer, attention heads focus on specific tokens.
When it doesn't know, attention is diffuse.

Metrics:
  - Attention entropy: Shannon entropy of attention distribution (low = focused)
  - Attention sparsity: fraction of attention on top-k positions (high = focused)
  - Attention variance: variance across heads (low = consensus)
  - Max attention concentration: fraction of total attention on top-1 position
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KNOWN_QUESTIONS = [
    "What is gravity?",
    "What is the capital of France?",
    "What is machine learning?",
    "What is DNA?",
    "What is the speed of light?",
]

UNKNOWN_QUESTIONS = [
    "Can topological persistence detect phase transitions in LLM training?",
    "Can sheaf cohomology detect misinformation cascades?",
    "Does the Wasserstein distance between question signal distributions predict discovery novelty?",
    "Who won the 2032 presidential election?",
    "What is the population of Mars Colony in 2035?",
]

MODEL_NAME = "gpt2-medium"
TOP_K_SPARSITY = 5  # top-k positions for sparsity metric


@dataclass
class AttentionMetrics:
    """Container for attention-focus metrics for a single question."""

    question: str
    group: str  # "known" or "unknown"
    entropy_mean: float
    entropy_std: float
    sparsity_mean: float
    sparsity_std: float
    variance_mean: float
    variance_std: float
    concentration_mean: float
    concentration_std: float
    n_layers: int
    n_heads: int
    n_tokens: int


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def compute_attention_entropy(attn_weights: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy per head.

    Parameters
    ----------
    attn_weights : ndarray, shape (n_heads, seq_len, seq_len)
        Attention probability distributions (rows sum to 1).

    Returns
    -------
    ndarray, shape (n_heads,)
        Mean entropy per head across query positions.
    """
    eps = 1e-12
    p = attn_weights + eps
    entropy = -np.sum(p * np.log(p), axis=-1)  # (n_heads, seq_len)
    return entropy.mean(axis=-1)  # (n_heads,)


def compute_attention_sparsity(attn_weights: np.ndarray, k: int) -> np.ndarray:
    """Fraction of attention mass on top-k positions per head.

    Parameters
    ----------
    attn_weights : ndarray, shape (n_heads, seq_len, seq_len)
    k : int
        Number of top positions to consider.

    Returns
    -------
    ndarray, shape (n_heads,)
        Mean sparsity per head across query positions.
    """
    # Sort descending along key dimension
    sorted_weights = np.sort(attn_weights, axis=-1)[:, :, ::-1]
    top_k_mass = sorted_weights[:, :, :k].sum(axis=-1)  # (n_heads, seq_len)
    return top_k_mass.mean(axis=-1)  # (n_heads,)


def compute_max_concentration(attn_weights: np.ndarray) -> np.ndarray:
    """Fraction of total attention captured by the single top position per head.

    Parameters
    ----------
    attn_weights : ndarray, shape (n_heads, seq_len, seq_len)

    Returns
    -------
    ndarray, shape (n_heads,)
        Mean max-concentration per head across query positions.
    """
    max_attn = np.max(attn_weights, axis=-1)  # (n_heads, seq_len)
    return max_attn.mean(axis=-1)  # (n_heads,)


def compute_cross_head_variance(metrics_per_head: np.ndarray) -> float:
    """Variance of a per-head metric across all heads.

    Parameters
    ----------
    metrics_per_head : ndarray, shape (n_heads,)

    Returns
    -------
    float
    """
    return float(np.var(metrics_per_head))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_question(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    question: str,
    group: str,
    device: torch.device,
) -> AttentionMetrics:
    """Run forward pass on one question, return attention metrics."""
    inputs = tokenizer(question, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    n_tokens = input_ids.shape[-1]

    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # tuple of (1, n_heads, seq, seq)

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    # Collect per-head metrics across all layers
    all_entropy: list[np.ndarray] = []
    all_sparsity: list[np.ndarray] = []
    all_concentration: list[np.ndarray] = []

    for layer_attn in attentions:
        attn = layer_attn[0].cpu().numpy()  # (n_heads, seq, seq)
        all_entropy.append(compute_attention_entropy(attn))
        all_sparsity.append(compute_attention_sparsity(attn, TOP_K_SPARSITY))
        all_concentration.append(compute_max_concentration(attn))

    # Stack: (n_layers, n_heads)
    entropy_matrix = np.stack(all_entropy)
    sparsity_matrix = np.stack(all_sparsity)
    concentration_matrix = np.stack(all_concentration)

    # Flatten to (n_layers * n_heads,) for aggregate stats
    entropy_flat = entropy_matrix.flatten()
    sparsity_flat = sparsity_matrix.flatten()
    concentration_flat = concentration_matrix.flatten()

    # Cross-head variance: for each layer, compute variance across heads
    variance_per_layer = [
        compute_cross_head_variance(entropy_matrix[l])
        for l in range(n_layers)
    ]

    return AttentionMetrics(
        question=question,
        group=group,
        entropy_mean=float(entropy_flat.mean()),
        entropy_std=float(entropy_flat.std()),
        sparsity_mean=float(sparsity_flat.mean()),
        sparsity_std=float(sparsity_flat.std()),
        variance_mean=float(np.mean(variance_per_layer)),
        variance_std=float(np.std(variance_per_layer)),
        concentration_mean=float(concentration_flat.mean()),
        concentration_std=float(concentration_flat.std()),
        n_layers=n_layers,
        n_heads=n_heads,
        n_tokens=n_tokens,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: list[AttentionMetrics]) -> None:
    """Print a formatted comparison table."""
    header = (
        f"{'Group':<8} {'Question':<72} "
        f"{'Entropy':>10} {'Sparsity':>10} "
        f"{'Variance':>10} {'Concen.':>10} {'Tokens':>7}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("ATTENTION FOCUS METRICS — GPT-2-medium")
    print("=" * len(header))
    print(header)
    print(sep)

    current_group = None
    for r in results:
        if r.group != current_group:
            if current_group is not None:
                print(sep)
            current_group = r.group
            label = "KNOWN (model should know)" if r.group == "known" else "UNKNOWN (model should NOT know)"
            print(f"--- {label} ---")

        q_short = r.question[:70]
        print(
            f"{r.group:<8} {q_short:<72} "
            f"{r.entropy_mean:>8.4f}{r.entropy_std:>+5.3f} "
            f"{r.sparsity_mean:>8.4f}{r.sparsity_std:>+5.3f} "
            f"{r.variance_mean:>8.5f}{r.variance_std:>+6.5f} "
            f"{r.concentration_mean:>8.4f}{r.concentration_std:>+5.3f} "
            f"{r.n_tokens:>7}"
        )

    print(sep)


def print_discrimination_analysis(
    known_results: list[AttentionMetrics],
    unknown_results: list[AttentionMetrics],
) -> None:
    """Compute whether any metric discriminates known vs unknown."""
    print("\n" + "=" * 80)
    print("DISCRIMINATION ANALYSIS: Known vs Unknown")
    print("=" * 80)

    metrics_spec = [
        ("Entropy (mean)", "entropy_mean"),
        ("Sparsity (mean)", "sparsity_mean"),
        ("Variance (mean)", "variance_mean"),
        ("Concentration (mean)", "concentration_mean"),
    ]

    print(
        f"{'Metric':<25} {'Known Mean':>12} {'Unknown Mean':>13} "
        f"{'Diff':>10} {'Cohen d':>10} {'Signal?':>10}"
    )
    print("-" * 80)

    for label, attr in metrics_spec:
        known_vals = np.array([getattr(r, attr) for r in known_results])
        unknown_vals = np.array([getattr(r, attr) for r in unknown_results])

        k_mean = known_vals.mean()
        u_mean = unknown_vals.mean()
        diff = u_mean - k_mean

        # Pooled std for Cohen's d
        n_k, n_u = len(known_vals), len(unknown_vals)
        pooled_std = np.sqrt(
            ((n_k - 1) * known_vals.std() ** 2 + (n_u - 1) * unknown_vals.std() ** 2)
            / (n_k + n_u - 2)
        ) if (n_k + n_u) > 2 else 1e-12

        cohens_d = diff / pooled_std if pooled_std > 1e-12 else 0.0

        # "Signal" if |d| > 0.5 (medium effect size)
        signal = "YES" if abs(cohens_d) > 0.5 else "no"

        print(
            f"{label:<25} {k_mean:>12.5f} {u_mean:>13.5f} "
            f"{diff:>+10.5f} {cohens_d:>+10.3f} {signal:>10}"
        )

    print("-" * 80)
    print("\nInterpretation:")
    print("  Cohen's d:  |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large")
    print("  'Signal?' = YES when |d| > 0.5 (medium effect size threshold)")
    print()

    # Per-head breakdown: check if certain layers/heads discriminate better
    print("=" * 80)
    print("LAYER-BY-LAYER ANALYSIS (entropy per layer, averaged across heads)")
    print("=" * 80)
    print(
        f"{'Layer':>6} {'Known Ent.':>12} {'Unknown Ent.':>13} {'Diff':>10} {'Cohen d':>10}"
    )
    print("-" * 55)

    # Re-extract per-layer entropy from stored results
    # Since we stored only aggregated stats, recompute would require re-running.
    # Instead, report per-question layer-level info from the raw data.
    # For now, we report the aggregate difference per metric.
    print("  (Per-layer analysis requires re-running with per-layer storage.")
    print("   The above aggregate metrics are sufficient for initial hypothesis test.)")
    print()


def print_per_head_detail(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    known_q: str,
    unknown_q: str,
    device: torch.device,
) -> None:
    """Print detailed per-layer entropy for one known and one unknown question."""
    print("=" * 80)
    print("DETAILED PER-LAYER ENTROPY: Known vs Unknown (example pair)")
    print("=" * 80)

    for label, question in [("KNOWN", known_q), ("UNKNOWN", unknown_q)]:
        inputs = tokenizer(question, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions

        print(f"\n{label}: \"{question}\"")
        print(f"  {'Layer':>6} {'Mean Entropy':>14} {'Min Entropy':>12} {'Max Entropy':>12} {'Std':>10}")
        for i, layer_attn in enumerate(attentions):
            attn = layer_attn[0].cpu().numpy()  # (n_heads, seq, seq)
            ent = compute_attention_entropy(attn)
            print(
                f"  {i:>6} {ent.mean():>14.4f} {ent.min():>12.4f} "
                f"{ent.max():>12.4f} {ent.std():>10.4f}"
            )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    # Load model and tokenizer
    print(f"Loading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME, attn_implementation="eager"
    ).to(device)
    model.eval()
    print(
        f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
        f"{model.config.n_layer} layers, {model.config.n_head} heads"
    )

    # Evaluate all questions
    all_results: list[AttentionMetrics] = []

    print("\nEvaluating KNOWN questions...")
    for q in KNOWN_QUESTIONS:
        r = evaluate_question(model, tokenizer, q, "known", device)
        all_results.append(r)
        print(f"  [{r.n_tokens} tokens] {q}")

    print("Evaluating UNKNOWN questions...")
    for q in UNKNOWN_QUESTIONS:
        r = evaluate_question(model, tokenizer, q, "unknown", device)
        all_results.append(r)
        print(f"  [{r.n_tokens} tokens] {q}")

    # Report
    known_results = [r for r in all_results if r.group == "known"]
    unknown_results = [r for r in all_results if r.group == "unknown"]

    print_table(all_results)
    print_discrimination_analysis(known_results, unknown_results)

    # Detailed per-layer breakdown for first known vs first unknown
    print_per_head_detail(
        model, tokenizer,
        KNOWN_QUESTIONS[0], UNKNOWN_QUESTIONS[0],
        device,
    )

    # Final verdict
    print("=" * 80)
    print("HYPOTHESIS TEST RESULT")
    print("=" * 80)
    k_ent = np.array([r.entropy_mean for r in known_results])
    u_ent = np.array([r.entropy_mean for r in unknown_results])
    k_sp = np.array([r.sparsity_mean for r in known_results])
    u_sp = np.array([r.sparsity_mean for r in unknown_results])
    k_con = np.array([r.concentration_mean for r in known_results])
    u_con = np.array([r.concentration_mean for r in unknown_results])

    print(f"  Known  entropy:    {k_ent.mean():.5f} +/- {k_ent.std():.5f}")
    print(f"  Unknown entropy:   {u_ent.mean():.5f} +/- {u_ent.std():.5f}")
    print(f"  Known  sparsity:   {k_sp.mean():.5f} +/- {k_sp.std():.5f}")
    print(f"  Unknown sparsity:  {u_sp.mean():.5f} +/- {u_sp.std():.5f}")
    print(f"  Known  concentration: {k_con.mean():.5f} +/- {k_con.std():.5f}")
    print(f"  Unknown concentration:{u_con.mean():.5f} +/- {u_con.std():.5f}")

    ent_diff = u_ent.mean() - k_ent.mean()
    sp_diff = u_sp.mean() - k_sp.mean()
    con_diff = u_con.mean() - k_con.mean()

    print(f"\n  Entropy diff (unknown - known):        {ent_diff:+.5f}")
    print(f"  Sparsity diff (unknown - known):       {sp_diff:+.5f}")
    print(f"  Concentration diff (unknown - known):  {con_diff:+.5f}")

    if ent_diff > 0.01:
        print("\n  >> Unknown questions show HIGHER entropy (more diffuse attention)")
        print("     This SUPPORTS the hypothesis.")
    elif ent_diff < -0.01:
        print("\n  >> Unknown questions show LOWER entropy (more focused attention)")
        print("     This CONTRADICTS the hypothesis.")
    else:
        print("\n  >> No meaningful entropy difference detected.")
        print("     Hypothesis NOT supported by this metric.")

    if sp_diff < -0.01:
        print("  >> Unknown questions show LOWER sparsity (less focused top-k)")
        print("     This SUPPORTS the hypothesis.")
    elif sp_diff > 0.01:
        print("  >> Unknown questions show HIGHER sparsity (more focused top-k)")
        print("     This CONTRADICTS the hypothesis.")

    if con_diff < -0.01:
        print("  >> Unknown questions show LOWER max concentration")
        print("     This SUPPORTS the hypothesis.")
    elif con_diff > 0.01:
        print("  >> Unknown questions show HIGHER max concentration")
        print("     This CONTRADICTS the hypothesis.")


if __name__ == "__main__":
    main()
