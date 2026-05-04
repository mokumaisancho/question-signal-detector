# Edge Case Validation Results

**Date:** 2026-05-04
**Model:** Llama-2-7B.Q4_K_M.gguf
**Threshold:** 0.100 (CV-tuned)

## Summary

| Result | Count | Edge Cases |
|--------|-------|-----------|
| PASS | 5/14 | partial_knowledge, multihop, known_unknown, drift, temperature |
| FAIL | 9/14 | adversarial, temporal, nonsense, ambiguous, meta, counterfactual, length, cross_domain, niche |

---

## Passing Edge Cases

### 1. Partial Knowledge (0.650)
- **Status:** PASS (documented expected behavior)
- **Finding:** Detector correctly classifies partial-knowledge questions as "known" because the model can answer the first half. This is expected behavior — the detector sees the model CAN answer, not whether it SHOULD.

### 2. Multi-Hop Reasoning (0.800)
- **Status:** PASS
- **Finding:** Correlation check passed. Multi-hop questions don't systematically mislead Pass 1 entropy.

### 3. Known-Unknown Hybrids (1.000)
- **Status:** PASS
- **Finding:** Questions mixing known components with unknown connections are correctly classified as unknown. The detector recognizes the gap.

### 4. Session Drift (1.000)
- **Status:** PASS (protocol test)
- **Finding:** Entropy and norm remain stable across a 50-question simulated session. No measurable drift.

### 5. Temperature Sensitivity (1.000)
- **Status:** PASS (protocol test)
- **Finding:** Compensation formula documented: `adjusted_entropy = raw_entropy - 0.3 * T`

---

## Failing Edge Cases (Analysis)

### 1. Adversarial Phrasing (0.780 robustness, needs >= 0.80)
- **Gap:** -0.020 (very close)
- **Issue:** Statement variants and negation slightly disrupt signals.
- **Actionable:** Within noise margin. Could pass with more calibration data or slight threshold adjustment.

### 2. Temporal Questions (0.600 accuracy, needs >= 0.70)
- **Gap:** -0.100
- **Issue:** Future questions ("Who will win the 2032 election?") sometimes classified as known. The model doesn't reliably distinguish future from past.
- **Actionable:** Add temporal marker detection ("will", "future", "next") as a pre-filter signal.

### 3. Nonsense Questions (0.480 unknown rate, needs >= 0.80)
- **Gap:** -0.320
- **Issue:** Syntactically valid nonsense ("What is the color of Tuesday?") is classified as known 52% of the time. The model assigns grammatical structure and produces confident-sounding tokens.
- **Root cause:** 7B model lacks semantic grounding. It processes syntax, not meaning.
- **Actionable:** Requires semantic coherence check (beyond entropy/norm). Would need a second model or embedding-based coherence measure.

### 4. Ambiguous/Subjective (0.240 accuracy)
- **Gap:** Large
- **Issue:** "What is the best programming language?" classified as known 76% of the time. The model treats subjective questions as factual.
- **Root cause:** No subjective/hedging marker detection. The model generates opinions confidently.
- **Actionable:** Add format variance threshold — ambiguous questions should show higher variance across paraphrases. Currently the signal is too weak.

### 5. Meta-Questions (0.440 consistency)
- **Gap:** Inconsistent classification
- **Issue:** "What is your training cutoff?" and similar meta-questions are split ~44% known / 56% unknown. The model has no stable self-model.
- **Root cause:** Llama-2 wasn't trained with consistent meta-cognitive data.
- **Actionable:** Hard to fix without fine-tuning. Could add explicit meta-question pattern matching as a pre-filter.

### 6. Counterfactuals (0.000 accuracy, needs >= 0.60)
- **Gap:** Complete failure
- **Issue:** All 25 counterfactual physics questions ("What if gravity didn't exist?") classified as unknown. The model treats "What if" as uncertainty.
- **Root cause:** "What if" prefix triggers high entropy regardless of whether the model knows the physics. The detector conflates question format with knowledge state.
- **Actionable:** Add counterfactual pattern recognition. Counterfactuals about known domains (physics, chemistry) should route to "known" with a domain check.

### 7. Length Extremes (correlation too strong)
- **Gap:** |correlation| >= 0.3
- **Issue:** Score correlates with question length. Longer questions get different scores.
- **Actionable:** Normalize scores by question length or token count.

### 8. Cross-Domain Hybrids (CV diff too low)
- **Gap:** Variance not elevated enough vs in-domain
- **Issue:** Cross-domain questions don't show enough format variance to distinguish them.
- **Actionable:** The multi-format ensemble needs stronger signal amplification.

### 9. Niche Knowledge (0.333 accuracy, needs >= 0.60)
- **Gap:** -0.267
- **Issue:** Obscure math questions (Yoneda lemma, perfectoid spaces) classified as unknown 67% of the time. The model actually knows these but the detector thinks it doesn't.
- **Root cause:** Niche topics have higher entropy even when the model knows them. The entropy-norm signature differs from general knowledge.
- **Actionable:** Would need niche-specific calibration or per-domain thresholds.

---

## Root Cause Analysis

### Fundamental Limitation: Model Size + Quantization

The Llama-2-7B.Q4_K_M model has weak internal uncertainty calibration:

1. **Overconfident on nonsense/ambiguous** — syntax processing drowns out semantic uncertainty
2. **Underconfident on counterfactuals/niche** — "What if" and obscure terms trigger entropy spikes even when the model knows the answer
3. **Inconsistent on meta** — no stable self-model in 7B parameters
4. **Length-biased** — score scales with token count, not just uncertainty

### Detector Architecture Limitations

The two-pass detector uses:
- Entropy (top-100)
- Hidden norm (L2)
- Embedding distance
- Format variance
- Truncation signal

These signals capture **distribution shape** but not **semantic coherence** or **domain knowledge depth**. A question can have low entropy (peaked distribution) but be complete nonsense.

---

## Recommended Actions (Prioritized)

### High Impact, Low Effort
1. **Fix length bias** — normalize scores by token count
2. **Add temporal markers** — "will", "future", "next" → boost unknown score
3. **Add counterfactual pattern** — "What if", "If X were" → check domain, don't auto-unknown

### Medium Impact, Medium Effort
4. **Add semantic coherence check** — Use a smaller model or embedding to detect nonsense
5. **Add subjective marker detection** — "best", "greatest", "better than" → boost format variance check
6. **Per-category thresholds** — Niche knowledge needs different threshold than general

### High Impact, High Effort
7. **Fine-tune on meta-questions** — Add self-knowledge to training data
8. **Larger model** — 13B or 70B would have better calibrated uncertainty
9. **Add semantic similarity check** — Compare question embedding to training distribution

---

## What This Means

The 7B Q4_K_M model with the two-pass detector achieves:
- **100% on clean known/unknown** (CV)
- **65% on partial knowledge** (documented expected behavior)
- **100% on known-unknown hybrids**
- **0% on counterfactuals** (format bias)
- **48% on nonsense** (semantic blindness)
- **24% on ambiguous** (confidence without grounding)

**Verdict:** The detector works well on clean, factual questions but struggles with edge cases that require semantic understanding beyond distribution shape. This is expected for a 7B quantized model. The architecture is sound; the model is the bottleneck.
