# CoT Path Selection Benchmark Design

**Date:** 2026-05-05
**Status:** Experiment ready, awaiting execution

## Issue Statement

When generating multiple Chain-of-Thought (CoT) reasoning chains for the same problem, the model produces chains of varying quality. **Can we use entropy signals — measured before and during generation — to select the best chain without knowing the answer?**

This matters because:

1. **Inference cost reduction**: If entropy selection matches majority vote accuracy but only uses 1 chain (instead of N), inference cost drops by N×.
2. **Signal value**: If entropy selection beats majority vote, the entropy signal contains information that voting on answers doesn't capture — a non-trivial finding.
3. **Scalability**: Entropy measurement costs 1 forward pass (~0.1s). Running N full chains costs N × generation time. Selection trades N generations for 1 generation + N entropy measurements.

## What We're Trying to Prove

**Primary hypothesis**: Pre-generation entropy and entropy trajectory during generation correlate with CoT chain correctness. Selecting chains by lowest entropy / most converging trajectory improves accuracy over random selection.

**Success criteria**:
- Entropy selection closes >30% of the gap between random baseline and oracle
- OR entropy selection matches majority vote accuracy at 1/N the inference cost

**Failure criteria**:
- Entropy selection performs no better than random → entropy is not a useful signal for reasoning quality
- Entropy selection is strictly worse than majority vote and does not save cost → the approach has no advantage

## Why GSM8K

| Property | Why It Matters |
|----------|---------------|
| Ground truth = integer | Exact match grading, zero ambiguity, no subjective judgment |
| Qwen3.5-4B ≈ 55-65% accuracy | Sweet spot: some chains succeed, some fail — selection has room to matter |
| CoT proven to help on math | Reasoning chains are meaningful, not decorative |
| 18 TPS × 3 chains × ~200 tokens ≈ 33s/question | 50 questions ≈ 28 minutes — feasible on M1 16GB |
| Entropy trajectory should correlate | Converging on a number → decreasing entropy; flailing → increasing |

**Why not the existing 5-category question set**: Those categories (counterfactual, nonsense, ambiguous, meta, known) test *knowledge boundary detection* — "is this question answerable?" CoT path selection tests *reasoning quality* — "did this chain reach the right answer?" These are different axes. Math problems stress the reasoning chain itself.

## Experiment Design

```
For each GSM8K question Q (N=50):
  1. Create CoT prompt
  2. For each chain i in 1..3:
     a. Generate chain at temperature=0.7 (diversity)
     b. Forward pass on full sequence → initial entropy + trajectory
     c. Extract final numerical answer
     d. Grade: exact match on integer answer
     e. Clear compute graph cache (mx.synchronize + mx.clear_cache)
  3. Record: (question, true_answer, chain_results[])
```

### Selection Strategies

| Strategy | Selection Criterion | Purpose |
|----------|--------------------|---------|
| **A) Random** | Take chain 1 | Baseline — what you get without selection |
| **B) Lowest entropy** | argmin(initial_entropy) | Does pre-generation uncertainty predict quality? |
| **C) Best trajectory** | argmin(trajectory_trend) | Does convergence predict quality? |
| **D) Majority vote** | Mode of rounded answers | Standard ensemble — strong baseline |
| **E) Oracle** | Pick any correct chain | Upper bound — best possible selection |

### Key Comparison: Entropy vs Majority Vote

Majority vote is already a strong baseline (uses all N chains). The critical comparison:

| Outcome | Interpretation |
|---------|---------------|
| Entropy > Majority vote | **Strong result** — entropy signal captures information voting misses |
| Entropy ≈ Majority vote | **Useful** — same accuracy at 1/N inference cost |
| Entropy < Majority vote but > Random | **Partial** — entropy helps but not as much as voting |
| Entropy ≈ Random | **Null result** — entropy is not useful for reasoning quality |

### Metric

**accuracy@1**: Fraction of questions where the selected chain's extracted answer matches the true answer (within ±0.5 for integer comparison).

## Expected Results (Qwen3.5-4B MLX)

| Strategy | Expected Accuracy | Inference Cost |
|----------|------------------|----------------|
| Random (A) | 0.55-0.65 | 1× generation |
| Lowest entropy (B) | 0.60-0.70 | 1× generation + 3× forward pass |
| Best trajectory (C) | 0.62-0.72 | 1× generation + 3× (generation + forward pass) |
| Majority vote (D) | 0.68-0.78 | 3× generation |
| Oracle (E) | 0.85-0.95 | — (unreachable) |

The trajectory strategy (C) requires generating all chains first to compute trajectories, then selecting one. This costs the same as majority vote. If it matches majority vote, it's not a cost win but validates the trajectory signal.

The entropy strategy (B) is the cost play: only forward passes needed for selection, then generate 1 chain. Total cost: 3 forward passes + 1 generation ≈ 1.3× single generation.

## Model Reset Protocol

MLX generate() is stateless — no persistent KV cache between calls. Each call creates and discards its own cache internally. However, to prevent compute graph memory accumulation:

```python
# After each chain generation + forward pass
mx.synchronize()   # Complete all pending GPU/ANE operations
mx.clear_cache()   # Free compute graph memory
```

This is not equivalent to llama-cpp's `llm.reset()` (which clears a persistent KV cache). MLX has no such cache. The clear_cache() call prevents gradual memory growth from MLX's automatic differentiation graph.

## Implications

### If entropy selection works
- **Local inference optimization**: On resource-constrained hardware (M1 16GB), generate N chains cheaply (forward pass only) and only fully generate the best one.
- **Inference scaling law**: Quality improves not by scaling model size, but by scaling inference-time compute (more chains) and selecting intelligently.
- **Transfer to detection pipeline**: If entropy predicts reasoning quality, it likely also predicts detection confidence.

### If it doesn't work
- Entropy is useful for knowledge boundary detection (pass 1 of the existing pipeline) but NOT for reasoning quality assessment.
- The two signals (knowledge uncertainty vs reasoning uncertainty) are fundamentally different.
- Majority vote remains the best cheap strategy for multi-chain selection.

## Script

`cot_benchmark.py` — run with:
```bash
python3 cot_benchmark.py --dry-run          # 3 questions, verify setup
python3 cot_benchmark.py --n-questions 50   # Full benchmark (~28 min)
python3 cot_benchmark.py --n-questions 100  # Extended run (~55 min)
```

Results saved to `/tmp/cot_benchmark_<timestamp>.json`.
