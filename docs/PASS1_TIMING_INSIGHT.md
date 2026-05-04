# Pass 1 Timing Insight: The First Token IS the Plan

## The Measurement Moment

In `_pass1_uncertainty()`, the timing is precise:

1. Question tokens are processed through the model
2. At the last question token position, the model has built a hidden state representation
3. `max_tokens=1` generates exactly one token — this first generated token **IS** the "planning" output

The measurement captures the model's confidence at the exact moment it commits to an initial answer direction. The entropy of that first token's distribution **IS** the model's planning confidence.

## What Is Being Measured

The signals can be partially separated:

| Signal | What It Measures | Independence |
|--------|-----------------|-------------|
| **Embedding + hidden norm** | How the question impacts the model's internal state | Independent of any answer generation |
| **Entropy + top100_mass** | How confident the model is in its initial answer plan | The first generated token distribution |

## Why There Is No Separate Planning Phase

Autoregressive transformers do not deliberate before generating. The first token **IS** the plan. The hidden state at the last question position encodes everything the model "knows" about how to respond, and the first generated token samples from that state.

This is why Pass 1 works: by stopping at exactly 1 token, we capture the model's initial commitment **before** autoregressive generation forces it down a completion path that suppresses uncertainty.

## The Contamination Point

Any measurement after token 2, 3, etc. would already be contaminated by the generation trajectory. The autoregressive nature means:

- Token 1: Pure planning signal (model's honest assessment)
- Token 2+: Contaminated by commitment to token 1's direction
- Full generation: Hijacked by attention focusing on completion, not uncertainty

## Why This Validates the Two-Pass Architecture

The two-pass design is not arbitrary — it mirrors the cognitive structure of autoregressive generation:

```
Pass 1 (1 token):  Capture the "pre-commitment" uncertainty
Pass 2 (N tokens): Allow the model to execute its plan IF uncertainty is low
```

If Pass 1 says "unknown," we abort before the model is forced to generate a coherent (but potentially hallucinated) answer. This prevents the "terror of generation" from drowning out the uncertainty signal.

## The Embedding vs Generation Distinction

The embedding (`create_embedding`) is computed on the **question tokens only** — no generation has occurred. The entropy is computed from the **first generated token's distribution** — the model's first commitment. These are two different moments:

1. **Embedding moment**: "How does the model represent this question internally?"
2. **Entropy moment**: "How confidently can the model choose its first answer token?"

Both are pre-generation measurements, but they capture different aspects of the model's knowledge state.

## Implication for Detector Design

This means the detector's four signals are not redundant:

- **Hidden norm**: Strength of internal representation (high = familiar concept)
- **Entropy**: Confidence in initial plan (low = confident first token)
- **top100_mass**: Concentration of probability mass (high = peaked distribution)
- **Embedding distance**: Similarity to calibrated references (close = known concept)

Each signal captures a different facet of the model's knowledge at the critical pre-commitment moment.

## Further Assessment Required

| Assessment | Purpose |
|-----------|---------|
| **Multi-token trajectory analysis** | Does uncertainty increase or decrease after token 1? |
| **Attention weight analysis** | How do attention patterns differ at token 1 vs token 5? |
| **Hidden state evolution** | How does the hidden representation change across generation steps? |
| **Comparison with beam search** | Does beam search reveal alternative plans that greedy sampling misses? |
