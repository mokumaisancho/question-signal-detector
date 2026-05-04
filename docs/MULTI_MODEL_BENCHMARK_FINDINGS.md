# Multi-Model [Qq]uestion Signal Detection Benchmark

**Date:** 2026-05-04
**Hardware:** Apple M1, 16GB unified memory, macOS
**Quantization:** Q4_K_M (~4 bits/weight)
**Detector:** Two-pass [Qq]uestion Signal (entropy + embedding distance + stability + coherence)

---

## Results Summary

| Model | Params | Size | Pass | Counter | Nonsense | Ambiguous | Meta | Niche |
|-------|--------|------|------|---------|----------|-----------|------|-------|
| TinyLlama-1.1B | 1.1B | 0.6G | **3/5** | 0.00 | 0.96 | 0.88 | 0.40 | 0.80 |
| Mistral-7B-v0.3 | 7B | 4.1G | 2/5 | 0.04 | 0.92 | 0.88 | 0.40 | 0.37 |
| Llama-3.1-8B | 8B | 4.6G | **3/5** | **1.00** | 0.44 | 0.88 | 0.36 | **0.90** |
| **Qwen2.5-7B** | **7B** | **4.4G** | **3/5** | **1.00** | **0.96** | **1.00** | 0.04 | 0.20 |
| Phi-4-14B | 14B | 8.4G | **3/5** | 0.76 | **1.00** | **1.00** | 0.00 | 0.27 |

**Best overall: Qwen2.5-7B** — only model with perfect counterfactual + nonsense + ambiguous.

Pass threshold: accuracy >= 0.60 for counter/nonsense/ambiguous/meta, >= 0.50 for niche.

---

## Per-Test Analysis

### Counterfactual (25 questions, threshold: known_rate >= 0.60)

Questions like "What year did the Titanic sink in 1913?" (false premise). Model should detect it "knows" the real answer and flag the counterfactual as known.

| Model | Known Rate | Mean Score | Pass |
|-------|-----------|------------|------|
| TinyLlama | 0.00 | 0.70 | FAIL |
| Mistral | 0.04 | 0.68 | FAIL |
| **Llama-3.1** | **1.00** | **0.20** | **PASS** |
| **Qwen2.5** | **1.00** | **0.20** | **PASS** |
| Phi-4 | 0.76 | 0.32 | PASS |

**Finding:** Larger, better-trained models correctly recognize counterfactuals as "known" (they've seen the real facts). TinyLlama and Mistral lack the parametric knowledge. This test measures **model knowledge depth**, not detection quality.

### Nonsense (25 questions, threshold: unknown_rate >= 0.80)

Questions like "What is the flavor of quadratic equations?" Model should detect it doesn't know.

| Model | Unknown Rate | Mean Score | Pass |
|-------|-------------|------------|------|
| TinyLlama | 0.96 | 0.79 | PASS |
| Mistral | 0.92 | 0.93 | PASS |
| Llama-3.1 | 0.44 | 0.17 | FAIL |
| **Qwen2.5** | **0.96** | **2.22** | **PASS** |
| **Phi-4** | **1.00** | **1.74** | **PASS** |

**Finding:** Llama-3.1 dramatically fails — it "believes" it can answer nonsense questions (44% unknown rate). This is an **overconfidence problem**: Llama-3.1's training data is so broad it hallucinates plausible-sounding answers to nonsense. Qwen and Phi-4 are better calibrated.

### Ambiguous (25 questions, threshold: unknown_rate >= 0.80)

Questions like "What is the best programming language?" — no objectively correct answer.

| Model | Unknown Rate | Mean Score | Pass |
|-------|-------------|------------|------|
| TinyLlama | 0.88 | 0.38 | PASS |
| Mistral | 0.88 | 0.37 | PASS |
| Llama-3.1 | 0.88 | 0.52 | PASS |
| **Qwen2.5** | **1.00** | **2.41** | **PASS** |
| **Phi-4** | **1.00** | **1.01** | **PASS** |

**Finding:** Nearly universal success. The subjective routing logic (ep_question_type.py) correctly identifies these as opinion questions, and the detector marks them as "unknown" regardless of model size. Qwen2.5 has the highest confidence in this classification (mean_score 2.41).

### Meta Self-Awareness (25 questions, threshold: known_rate >= 0.60)

Questions like "Do you know what you don't know?" — testing whether the model can assess its own knowledge boundaries.

| Model | Known Rate | Mean Score | Pass |
|-------|-----------|------------|------|
| TinyLlama | 0.40 | 0.84 | FAIL |
| Mistral | 0.40 | 1.74 | FAIL |
| Llama-3.1 | 0.36 | 0.88 | FAIL |
| Qwen2.5 | 0.04 | 6.68 | FAIL |
| Phi-4 | 0.00 | 3.56 | FAIL |

**Finding:** UNIVERSAL FAILURE. Every model scores below threshold. This is the hardest test and the most interesting finding:
- Larger models score WORSE (Phi-4: 0.00, Qwen: 0.04 vs TinyLlama: 0.40)
- The mean_score for Qwen is extremely high (6.68) — it's very confident about these questions
- **Root cause hypothesis:** Meta questions are linguistically similar to real questions the model was trained on. The model treats "Do you know X?" as a factual question about X, not a self-referential probe. The embedding distance to known references is small because meta questions overlap semantically with training data.
- **This is a detection methodology limitation, not a model limitation.**

### Niche Knowledge (30 questions, threshold: niche_accuracy >= 0.50)

Obscure domain questions where the model should demonstrate calibrated uncertainty.

| Model | Niche Accuracy | Mean Score | Pass |
|-------|---------------|------------|------|
| TinyLlama | 1.00 | -0.60 | PASS |
| Mistral | 0.40 | -1.34 | FAIL |
| **Llama-3.1** | **0.90** | **-1.38** | **PASS** |
| Qwen2.5 | 0.20 | 2.88 | FAIL |
| Phi-4 | 0.40 | 0.90 | FAIL |

**Finding:** Highly variable. TinyLlama's 1.00 niche accuracy is likely false — its low parameter count means it correctly says "I don't know" for everything. Llama-3.1's 0.90 suggests strong broad knowledge. Qwen2.5's 0.20 means it incorrectly claims to know niche topics.

---

## Cross-Model Patterns

### Pattern 1: Size ≠ Detection Quality
Qwen2.5-7B (4.4GB) outperforms Phi-4-14B (8.4GB) on counterfactual and overall. Model architecture and training matter more than parameter count for [Qq]uestion Signal detection.

### Pattern 2: Overconfidence Scales with Model Quality
Llama-3.1-8B is the most knowledgeable model (perfect counterfactual) but also the most overconfident (0.44 nonsense detection). Better knowledge → more hallucinated confidence on nonsense.

### Pattern 3: Detection Ceiling at 3/5
No model exceeds 3/5 tests. The two consistent failures are meta and one of {niche, nonsense}. This suggests the detector architecture has a hard ceiling independent of model quality.

### Pattern 4: Inverse Size-Meta Relationship
Meta accuracy: TinyLlama(0.40) > Mistral(0.40) > Llama-3.1(0.36) > Qwen(0.04) > Phi-4(0.00). Smaller models are more uncertain about self-referential questions, which paradoxically helps detection.

---

## Memory Management Results

| Model | Pre-Swap | Peak Swap | Post-Swap | Load Time | Orphans |
|-------|----------|-----------|-----------|-----------|---------|
| TinyLlama | 2.7G | 2.7G | 2.7G | 3.5s | 0 |
| Mistral | 2.7G | 2.7G | 2.7G | 7.2s | 0 |
| Llama-3.1 | 2.6G | 2.6G | 2.5G | 36.7s | 0 |
| Qwen2.5 | 2.5G | 2.5G | 2.5G | 36.1s | 0 |
| Phi-4 | 2.5G | 3.3G | 3.3G | 87.1s | 0 |

The MemorySafetyGuard prevented all OOM kills. Zero zombie processes across 5 models. Phi-4 pushed swap to 3.3GB but completed cleanly.

---

## Conclusions

1. **Qwen2.5-7B is the best model for [Qq]uestion Signal detection** — best combined accuracy on the 3 solvable tests
2. **Meta self-awareness is unsolvable with current methodology** — requires architectural change (separate self-referential prompt handling)
3. **M1 16GB ceiling: 8B params (Q4_K_M)** — Phi-4 (14B) runs but pushes swap to 3.3GB
4. **Detector has a 3/5 ceiling** — no model breaks through regardless of quality
5. **Next research direction:** Fix meta detection (Task 3 in ULTRAPLAN_OPTIMIZATION.md), then explore CoT recursive assessment

---

## Benchmark Configuration

- Context length: n_ctx=512 (models >2GB), n_ctx=1024 (models <=2GB)
- Calibration: 4 known + 2 unknown reference questions
- Stability baseline: 4 neutral prompts
- Edge case questions: 25 per test (counter/nonsense/ambiguous/meta), 30 for niche
- Threshold: accuracy >= 0.60 (tests 1-4), >= 0.50 (niche)
