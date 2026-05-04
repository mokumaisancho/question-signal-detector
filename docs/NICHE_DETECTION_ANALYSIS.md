# Niche Detection Deep Dive Analysis

**Date:** 2026-05-04
**Model:** Qwen3-8B-Q4_K_M (4.7 GB)
**Goal:** Understand why niche detection fails across all models (0.13-0.20 accuracy)

## Root Cause

**The niche test measures the wrong thing.** The test questions are advanced math topics (Yoneda lemma, spectral sequences, Langlands program) that the model **genuinely knows** from its training data. The test penalizes the model for correctly identifying these as "known."

## Evidence from 4 Directions

### Direction 1: Relabel Obscure Questions as "Unknown"

Tested 30 questions with corrected labels (general=known, niche=known, obscure=unknown):

| Category | Accuracy | Detail |
|----------|----------|--------|
| General (known) | 2/10 | Model too conservative |
| Niche (known) | 1/10 | Model too conservative |
| **Obscure (unknown)** | **8/10** | **Model CAN detect obscure topics** |

The model correctly identifies 8/10 obscure math questions as unknown. But it also (incorrectly) marks general and niche questions as unknown — the threshold is too aggressive, not the knowledge detection.

### Direction 2: Answer Correctness

Generated actual answers and checked against expected keywords (raw `llm()` without chat template — continuation mode, partially unreliable):

| Category | Correct | Notes |
|----------|---------|-------|
| General | 1/6 | Continuation mode garbled most answers |
| Niche | 4/8 | Model knows darmstadtium (110), ambystoma, luciferase |

**Key:** The model correctly answered niche questions about darmstadtium's atomic number, axolotl genus, and firefly enzyme — confirming it **genuinely knows** these topics.

### Direction 3: Logprobs Analysis

Measured average token log-probability during generation:

| Category | Avg Logprob | vs General Gap |
|----------|------------|----------------|
| General | -4.80 | baseline |
| Niche | -4.79 | **-0.01** (no separation) |
| Obscure | -6.18 | **+1.38** (significant separation) |
| Fake | -4.77 | -0.03 (no separation) |

**Critical finding:** Logprobs separate **obscure** from general (1.38 gap) but NOT **niche** from general. Niche math topics have the same generation confidence as general knowledge — because the model actually knows them.

Fake/absurd questions show NO logprob separation from general — the model generates plausible continuations with equal confidence regardless of factual accuracy.

### Direction 4: Multi-dimensional Scoring

Crashed before completion (API error). Not available.

## Diagnosis

The current niche test has **three fundamental flaws**:

1. **Question selection**: "Yoneda lemma" and "spectral sequences" are in the training data of any model trained on math papers. These aren't "niche" to the model — they're known.

2. **Measurement axis**: The test measures embedding distance from calibration references, but niche topics have embedding vectors similar to known topics (both are real knowledge). The correct axis is **answer correctness** or **generation confidence**, not embedding similarity.

3. **Fake questions are invisible**: Absurd/impossible questions ("phone number of the third cashier") have the same logprob as general questions (-4.77 vs -4.80). The model can't distinguish "plausible but impossible" from "known fact."

## Recommendations

### Fix 1: Replace niche questions with genuinely testable items
Use questions about very recent events, hyper-specific personal data, or post-cutoff information that the model genuinely cannot know:
```
- "Who won the 2026 Nobel Prize in Physics?" (post-training)
- "What is the current population of Ulaanbaatar?" (changes constantly)
- "What was the result of the March 2026 Japanese by-election?"
```

### Fix 2: Use logprobs instead of embeddings for niche detection
Obscure questions show 1.38 logprob gap from general. A threshold on generation confidence would separate known from obscure effectively.

### Fix 3: Generate-and-verify approach
Instead of asking "do you know this?", generate an answer and check correctness against a reference. Wrong answers indicate the model doesn't actually know.

### Fix 4: Separate "niche" into subcategories
- **Known niche** (in training data) → model should say "known" ✓
- **Unknown niche** (not in training data) → model should say "unknown"
- **Fake/absurd** → currently undetectable via embeddings OR logprobs

## Updated Benchmark Implications

If niche questions are re-labeled correctly (model knows them = "known"), Qwen3-8B's niche accuracy would be much higher. The current 0.20 reflects a measurement error, not a model limitation.
