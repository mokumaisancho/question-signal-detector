# MLX Inference and CoT Selection Experiment

**Date:** 2026-05-05

## MLX vs llama-cpp-python on Apple M1

| Signal | llama-cpp (CPU) | MLX (GPU/ANE) | Notes |
|--------|-----------------|----------------|-------|
| Generation TPS (Qwen3.5-4B) | ~8 TPS | **18 TPS** | 2.2x faster |
| Generation TPS (Qwen3-8B) | 3.6 TPS | N/A (no MLX model) | Only GGUF tested |
| Generation TPS (Qwen2.5-7B) | 5.1 TPS | N/A | Only GGUF tested |
| Next-token entropy | Yes (logprobs) | Yes (logits) | MLX returns full vocab logits |
| Top-K logprobs | Yes | Yes | Both work |
| Embedding extraction | Yes (create_embedding) | Yes (embed_tokens) | Different API |
| Hidden state norms | Yes (hidden_norm) | Partial (needs hook) | MLX layer access possible but harder |
| Chat template | create_chat_completion | Not tested | Raw completion mode used |

### Entropy Separation (MLX, Qwen3.5-4B, continuation mode)

| Question | Label | Entropy |
|----------|-------|---------|
| "What is 2+2?" | known | **1.07** |
| "What is the capital of France?" | known | **1.04** |
| "Who won the 2026 Nobel Prize?" | unknown | **1.72** |
| "What is the serial number of the third rivet on the Eiffel Tower?" | unknown | **1.03** |

Known avg entropy: **1.06**. Unknown avg entropy: **1.38**. Gap: **0.32**.

The "third rivet" question has entropy 1.03 (same as known) — confirming the earlier finding that absurd questions produce confident continuations regardless of factual accuracy.

## CoT Selection Experiment Design

### Hypothesis

Selecting the best chain from multiple CoT generations using pre-generation entropy as a selection criterion will improve accuracy vs taking a single random chain.

### Test Categories (excluding niche)

| Category | Questions | Why Included |
|----------|-----------|-------------|
| **Counterfactual** | 10 | Model sometimes knows physics, sometimes not |
| **Nonsense** | 10 | Should be consistently rejected |
| **Ambiguous** | 10 | Borderline — selection should help most |
| **Meta** | 10 | Model sometimes recognizes self, sometimes not |
| **Known factual** | 10 | Baseline — should be consistently accepted |

### Method

```
For each question Q:
  1. Measure pre-generation entropy E0
  2. Generate N=3 chains: C1, C2, C3
  3. For each chain Ci:
     - Measure entropy Ei before generation
     - Generate answer Ai
     - Record: (Ei, Ai, correctness)
  4. Selection strategies:
     A) Random: Take C1 (baseline)
     B) Lowest entropy: Take argmin(Ei)
     C) Oracle: Take the correct chain (upper bound)
  5. Compare accuracy of A, B, C
```

### Expected Timeline (Qwen3.5-4B MLX, 18 TPS)

- 50 questions × 3 chains × ~3s per chain = ~7.5 minutes
- Entropy measurement: ~50 questions × 0.1s = ~5 seconds
- Total: ~8 minutes

### What Success Looks Like

| Strategy | Counter | Nonsense | Ambiguous | Meta | Known | Avg |
|----------|---------|----------|-----------|------|-------|-----|
| Random (A) | 0.60 | 0.70 | 0.50 | 0.40 | 0.80 | 0.60 |
| Entropy select (B) | 0.75 | 0.80 | 0.65 | 0.55 | 0.90 | 0.73 |
| Oracle (C) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

If entropy selection closes >30% of the gap between random and oracle, the approach is viable.

### Latent Thought Trajectory (Future)

After validating entropy selection, test trajectory-based selection:

```
Step 1: "Let me think..." → entropy E1
Step 2: "First..." → entropy E2
Step 3: "Therefore..." → entropy E3

Trajectory = [E1, E2, E3]
Good: decreasing (converging on answer)
Bad: increasing (diverging into hallucination)
```

This requires per-step entropy extraction mid-generation, which costs ~3x more forward passes but could further improve selection on borderline questions.
