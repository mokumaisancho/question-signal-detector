# Edge Case Root Cause Analysis — Mechanistic Breakdown

**Model:** Llama-2-7B.Q4_K_M.gguf  
**Detector:** TwoPassLlamaDetector with embedding-distance signal  
**Date:** 2026-05-04

---

## The Scoring Formula (Where Decisions Happen)

```python
entropy_norm     = next_token_entropy / 5.0
norm_signal      = (hidden_norm - 20.0) / 10.0
truncation_signal = 1.0 - top100_mass
embedding_signal  = min_dist_to_unknown - min_dist_to_known  # positive = closer to known

combined = (
    0.4 * entropy_norm
    + 0.3 * norm_signal
    + 0.1 * truncation_signal
    - 0.2 * embedding_signal          # <-- THIS DOMINATES
)

is_known = combined < 0.5
```

### Why Embedding Signal Dominates

| Component | Typical Value | Weighted Contribution |
|-----------|--------------|----------------------|
| entropy_norm (entropy=3.8) | 0.76 | +0.30 |
| norm_signal (norm=70) | 5.0 | +1.50 |
| truncation_signal | 0.1 | +0.01 |
| **embedding_signal** | **±10 to ±50** | **∓2.0 to ∓10.0** |

For a known question near calibration: embedding_signal = +20 → contributes **-4.0** to score → known.  
For an unknown far from calibration: embedding_signal = -10 → contributes **+2.0** → unknown.

The embedding signal has 2-5x the impact of all other signals combined.

---

## Failure Mode 1: Counterfactuals (0% accuracy, score=2.177)

**Expected:** known (model knows physics)  
**Actual:** all classified as unknown

### Root Cause: "What if" Creates Embedding Exile

Counterfactuals like "What if gravity didn't exist?" occupy a **unique embedding region** that is:
- Far from known calibration references (factual questions)
- Also far from unknown calibration references (frontier questions)
- In a "no man's land" that the detector reads as "unknown"

The embedding distance mechanism only knows "close to known = known, close to unknown = unknown." It has no concept of **"close to neither but still known."**

### Mechanistic Trace

```
"What is gravity?"          → embedding ≈ known_ref[0]      → embedding_signal ≈ +15 → KNOWN
"What if gravity didn't exist?" → embedding ≈ neither       → embedding_signal ≈ -5  → UNKNOWN
```

The "What if" prefix shifts the embedding into a region the detector has never seen. The model KNOWS the physics (can reason about consequences) but the detector sees an alien embedding and calls it unknown.

### Why This Is a Design Bug, Not a Model Limitation

The detector conflates **embedding familiarity** with **knowledge**. A question can be:
1. Familiar embedding + model knows it → correct
2. Familiar embedding + model doesn't know → false positive (rare)
3. **Unfamiliar embedding + model knows it → FALSE NEGATIVE (counterfactuals)**
4. Unfamiliar embedding + model doesn't know → correct

Case 3 is the blind spot. The detector needs a **domain knowledge check** independent of embedding distance.

---

## Failure Mode 2: Nonsense (48% unknown rate, score=0.373)

**Expected:** all unknown  
**Actual:** 52% classified as known

### Root Cause: Syntax Trumps Semantics

Nonsense questions are syntactically valid:
- "What is the color of Tuesday?" → follows "What is the [property] of [noun]?" pattern
- "How fast does dark travel?" → follows "How fast does [noun] [verb]?" pattern

The embedding captures **syntactic structure**, not **semantic coherence**. A nonsense question embeds close to legitimate questions with the same grammatical pattern.

### Mechanistic Trace

```
"What is the color of the sky?"     → embedding near "What is the color of Tuesday?"
                                       embedding_signal ≈ +8  → KNOWN (correct)
"What is the color of Tuesday?"     → same embedding region
                                       embedding_signal ≈ +8  → KNOWN (WRONG)
```

The entropy on "Tuesday" + "color" might be slightly higher (model is confused), but the embedding_signal is still strongly positive because the syntactic frame is familiar.

### The Entropy Paradox

One might expect nonsense to have HIGH entropy (model is uncertain). But the model assigns:
- "color of Tuesday" → might produce "There is no color" or "Tuesday doesn't have a color"
- The model recognizes the category error and responds confidently with a correction
- This produces LOW entropy (peaked distribution on "There")

The detector sees low entropy + familiar embedding → calls it known. The model "knows" the question is nonsensical, but the detector interprets that confidence as knowledge.

---

## Failure Mode 3: Ambiguous/Subjective (24% accuracy, score=-0.497)

**Expected:** all unknown (no objective answer)  
**Actual:** 76% classified as known

### Root Cause: Opinion Confidently Delivered

"What is the best programming language?" → model produces "Python is widely considered..." or "It depends on..."

The model:
1. Has seen this question thousands of times in training
2. Has learned confident-sounding hedged responses
3. Generates the hedging with LOW entropy (practiced pattern)
4. Embedding is close to other "What is the best..." questions

### Mechanistic Trace

```
"What is the best programming language?"
  entropy ≈ 2.5 (model has standard answer) → entropy_norm = 0.5
  norm ≈ 72 → norm_signal = 5.2
  embedding ≈ known region (seen 1000x) → embedding_signal ≈ +18
  combined ≈ 0.4*0.5 + 0.3*5.2 + 0.01 - 0.2*18
          ≈ 0.2 + 1.56 + 0.01 - 3.6 = -1.83 → KNOWN
```

The model's training data is FULL of subjective questions with confident answers. The detector can't distinguish "confident because it's factual" from "confident because it's rehearsed opinion."

---

## Failure Mode 4: Meta-Questions (44% consistency, score=0.576)

**Expected:** all known (model knows about itself)  
**Actual:** mixed classification

### Root Cause: Llama-2 Has No Stable Self-Model

Llama-2-7B was not trained with consistent meta-cognitive data. Some questions it has seen:
- "What is your training cutoff?" → seen in fine-tuning → knows answer
- "How many layers do you have?" → may not have explicit answer in training
- "What is your quantization level?" → definitely not in training data

The embedding distances vary wildly because:
- Some meta-questions embed near technical documentation
- Others embed near casual conversation
- The detector sees a scatterplot, not a cluster

### Mechanistic Trace

```
"What is your training cutoff?"
  → embedding near "When was the data collected?" (familiar) → KNOWN

"What is your quantization level?"
  → embedding far from everything (unfamiliar term) → UNKNOWN
```

The model KNOWS it's quantized (can infer from its own weights), but the detector relies on embedding similarity to calibration data that doesn't include quantization questions.

---

## Failure Mode 5: Niche Knowledge (33% accuracy, score=0.081)

**Expected:** known (model knows advanced math)  
**Actual:** 67% classified as unknown

### Root Cause: Obscure Terms = Embedding Exile

"What is the Yoneda lemma?" uses terminology that:
- Doesn't appear in general-knowledge calibration questions
- Embeds in a region far from "What is gravity?" etc.
- The detector sees an unfamiliar embedding → calls it unknown

But the model DOES know the Yoneda lemma. It's in the training data (math papers, textbooks, StackExchange).

### Mechanistic Trace

```
"What is gravity?"              → embedding_signal ≈ +20 → KNOWN
"What is the Yoneda lemma?"     → embedding_signal ≈ -2  → UNKNOWN (WRONG)
```

The model's knowledge is MUCH broader than the calibration set. The detector uses a tiny sample of "known" (20 questions) as a proxy for all knowledge. Anything outside that tiny island is "unknown."

---

## Failure Mode 6: Length Bias (score correlates with length)

**Expected:** no correlation  
**Actual:** |correlation| >= 0.3

### Root Cause: Norm Scales with Token Count

The embedding norm `hidden_norm` is computed from the pooled per-token embeddings. Longer questions have:
- More tokens → more vectors in the mean → different norm
- The `norm_signal = (norm - 20) / 10` is not normalized by length

A 100-token question naturally has a different embedding statistics than a 5-token question, even if both are equally "known."

### Mechanistic Trace

```
"Pi?" (2 tokens)
  norm ≈ 55 → norm_signal = 3.5
  combined ≈ 0.4*0.4 + 0.3*3.5 + ... ≈ 1.2 → UNKNOWN (borderline)

"What is the mechanism by which CRISPR-Cas9 edits DNA sequences?" (15 tokens)
  norm ≈ 75 → norm_signal = 5.5
  combined ≈ 0.4*0.5 + 0.3*5.5 + ... ≈ 1.9 → UNKNOWN (but should be known)
```

Actually wait — both would be unknown with hardcoded threshold of 0.5. The issue is that the score is consistently higher for longer questions, creating a bias in the classification boundary.

---

## The Core Design Flaw

### Embedding Distance Is a Proxy for "Has the Model Seen This Exact Question?" Not "Does the Model Know the Answer?"

The detector uses embedding distance to calibration questions as its primary signal. But embedding distance measures:
- **Lexical similarity** (same words)
- **Syntactic similarity** (same grammar patterns)
- **Distributional similarity** (same training contexts)

It does NOT measure:
- **Compositional knowledge** (can the model reason from known facts?)
- **Semantic coherence** (does the question make sense?)
- **Domain expertise** (does the model know this field?)
- **Counterfactual reasoning** (can the model simulate alternatives?)

### The Island Problem

Calibration creates tiny "islands" of known/unknown in embedding space:
- 20 known questions → 20 reference points
- 20 unknown questions → 20 reference points
- Everything else is classified by proximity to these 40 points

A question about Yoneda lemma is far from all 40 points → classified as unknown. But it's not unknown — it's just **unfamiliar to the detector**.

---

## What Works (And Why)

### Clean Known/Unknown (100% CV)

Works because:
- "What is gravity?" is in the calibration set → embedding_signal ≈ +20 → KNOWN
- "What is Mars Colony population?" is in the calibration set → embedding_signal ≈ -10 → UNKNOWN

This is **memorization of calibration examples**, not general knowledge detection.

### Known-Unknown Hybrids (100%)

Works because:
- "Can CRISPR cure Alzheimer's?" → combines known terms in novel way
- Embedding is far from both known and unknown references
- The model genuinely doesn't know → high entropy → UNKNOWN

### Multi-Hop (80%)

Works because:
- "If Paris is capital of France, what is capital of country bordering France to the east?"
- Model can answer this (Berlin) → low entropy, familiar embedding → KNOWN
- The embedding still encodes geographic terms the model has seen

---

## Why a Larger Model Won't Fix This

Switching to 13B or 70B would improve calibration (better entropy/norm signals), but would NOT fix the fundamental issue:

**Embedding distance to 40 calibration questions is still the wrong signal for compositional knowledge.**

A 70B model would still:
- Classify counterfactuals as unknown (unfamiliar embedding)
- Classify ambiguous questions as known (confident opinions)
- Classify niche knowledge as unknown (far from calibration island)

The architecture itself needs to change, not just the model size.

---

## Next Steps (Prioritized by Root Cause)

### Tier 1: Fix Embedding Island Problem

**Problem:** Detector only knows 40 calibration points.  
**Solution:** Replace embedding distance with a semantic coherence check.

**Approach A: Self-Consistency Check**
- Ask the question 3 times with different temperatures
- If answers are consistent → model knows it (regardless of embedding)
- If answers vary wildly → model is uncertain
- Cost: 3x generation time

**Approach B: Probe Questions**
- Generate 3 related sub-questions automatically
- If the model answers all confidently → known
- If any sub-question fails → unknown
- Example: "What if gravity didn't exist?" → probe: "What is Newton's law of gravitation?" "What keeps planets in orbit?" "What is escape velocity?"

**Approach C: Semantic Entailment**
- Use a smaller model to check if the question is semantically coherent
- "What is the color of Tuesday?" → entailment model says "nonsense" → unknown
- Cost: additional model load

### Tier 2: Add Domain Detectors

**Problem:** Counterfactuals and niche knowledge fail because of unfamiliar embeddings.  
**Solution:** Detect question TYPE and apply different logic.

```python
if question starts with "What if" or "If X were":
    # Counterfactual: check domain knowledge instead of embedding
    domain = extract_domain(question)  # "gravity" -> physics
    if domain in model_knowledge_domains:
        return KNOWN
    else:
        return UNKNOWN

if question contains subjective markers ("best", "greatest", "better than"):
    # Subjective: always flag as uncertain
    return SUBJECTIVE

if question is meta ("your training", "your parameters"):
    # Meta: use dedicated meta-calibration set
    return meta_detector.detect(question)
```

### Tier 3: Length-Normalize Scores

**Problem:** Score correlates with token count.  
**Solution:** Normalize embedding and entropy by sequence length.

```python
norm_signal = (hidden_norm / n_tokens - 2.0) / 1.0  # per-token norm
entropy_norm = next_token_entropy / np.log(vocab_size)  # normalized by max entropy
```

### Tier 4: Expand Calibration

**Problem:** 20 known + 20 unknown is too small.  
**Solution:** Use 200+ questions per category, or cluster-based calibration.

```python
# Instead of 20 reference points, use K-means clusters
known_clusters = KMeans(n_clusters=10).fit(known_embeddings)
unknown_clusters = KMeans(n_clusters=10).fit(unknown_embeddings)

# Distance to nearest cluster center, not nearest point
embedding_signal = dist_to_nearest_unknown_cluster - dist_to_nearest_known_cluster
```

### Tier 5: Two-Stage Classification

**Problem:** Single threshold can't handle all edge cases.  
**Solution:** First stage detects question type, second stage applies type-specific logic.

```
Stage 1: Question Type Classifier
  - Factual ("What is X?")
  - Counterfactual ("What if X?")
  - Subjective ("What is the best X?")
  - Meta ("What is your X?")
  - Nonsense (semantic coherence check)

Stage 2: Type-Specific Detector
  - Factual → embedding distance (works well)
  - Counterfactual → domain knowledge check
  - Subjective → always uncertain
  - Meta → meta-calibration
  - Nonsense → semantic coherence
```

---

## Recommended Immediate Implementation

Given time constraints, implement Tier 1 (Self-Consistency) + Tier 2 (Domain Detectors) + Tier 3 (Length Normalization):

1. **Add `self_consistency_score()`**: Generate answer 3 times, measure answer similarity
2. **Add `question_type_classifier()`**: Rule-based classifier for counterfactual/subjective/meta
3. **Normalize `norm_signal` by token count**
4. **Run edge cases again** — expect counterfactuals and subjective to improve dramatically

This is approximately 4-6 hours of work and should fix 6 of the 9 failing edge cases.
