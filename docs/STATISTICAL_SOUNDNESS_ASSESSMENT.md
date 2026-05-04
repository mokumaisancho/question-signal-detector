# Statistical Soundness Assessment: [Qq]uestion Signal Forecasting Harness

## Executive Summary

The current two-pass detector achieves **83% accuracy on 6 test questions** with Llama-2-7B. This is promising but **not statistically sound** for production use. The sample size is too small, the test set is not held out, and critical edge cases have not been evaluated.

**Verdict: The method shows signal but requires rigorous validation before claiming statistical soundness.**

---

## Part 1: What Earlier Research Found

### Method 1: Directional [Qq]uestion Signal Alignment (DEA)

| Metric | Value | Assessment |
|--------|-------|------------|
| AUC | **0.233** | Worse than random (0.5) |
| OOD Known accuracy | 93.3% | Classifies almost everything as "known" |
| Frontier accuracy | **0.0%** | Never detects frontier questions |

**Why it failed:** BERT [CLS] embeddings cluster in a narrow cone (cosine 0.91-0.98). The "projector" layer was trained to F1=1.0 on training data — it memorized phrasing patterns, not [Qq]uestion Signal boundaries.

### Method 2: GPT-2-Medium Token Entropy

| Metric | Value | Assessment |
|--------|-------|------------|
| AUC | **0.594** | Slightly above random |
| OOD Known accuracy | 100% | Overconfident on everything |
| Frontier accuracy | **0.0%** | Cannot detect frontier questions |

**Why it failed:** GPT-2-medium is overconfident. Frontier questions produce entropy ~0.6-0.8, known questions ~1.5-2.5. The model generates confidently even when it should not.

### Method 3: Attention Focus Detection

| Metric | Uncontrolled | Length-Controlled |
|--------|-------------|-------------------|
| Cohen's d | +3.50 | **-1.97** |
| Known entropy | 0.968 | Lower than unknown |
| Unknown entropy | 0.842 | Higher than known |

**Why it failed:** The original signal was entirely a **sequence length confound**. Unknown questions were longer (8-14 tokens vs 4-7 tokens), which naturally produces higher attention entropy. With length-matched questions, the signal reversed.

### Method 4: Hidden State Variance (Kim 2026)

| Metric | Value | Assessment |
|--------|-------|------------|
| Discriminative power | **Zero** | No separation between known/unknown |
| Layer 9 signal | Weak | Some activation difference |
| Layers 20-27 | No signal | Uniform across all questions |

**Why it failed:** GPT-2-medium's hidden states are too uniform. Adjacent layer cosine similarity is ~0.91 regardless of knowledge state.

### Method 5: Two-Pass Llama-2-7B Detector (Current)

| Metric | Value | Assessment |
|--------|-------|------------|
| Accuracy | **83%** (5/6) | Best result so far |
| False abstention | **0%** | No known questions refused |
| False generation | **17%** | 1 unknown question answered |

**What works:**
- Hidden norm: strongest signal (known=68-73, unknown=50-58)
- Top-100 entropy: inverted but consistent (known=1.5-3.4, unknown=0.6-0.8 for full, or known=2.4, unknown=0.7-2.6 for top-100)
- Embedding distance: catches cross-domain questions

**What fails:**
- Mars Colony question slips through (embedding proximity to known space topics)
- Single threshold (0.5) is too coarse
- No statistical validation (N=6)

---

## Part 2: Current Research Findings

### Finding 1: Top-100 Entropy is Better Than Full

| Metric | Known | Unknown | Separation |
|--------|-------|---------|------------|
| H_full | 2.97 ± 0.63 | 2.97 ± 0.16 | **0.01 SD** |
| H_top100 | 2.45 ± 0.50 | 2.64 ± 0.08 | **0.39 SD** |
| H_top10 | 1.52 ± 0.29 | 1.90 ± 0.13 | **1.19 SD** |

Full entropy fails to discriminate. Top-100 captures the "effective" uncertainty while discarding tail noise. **This is counterintuitive but statistically robust.**

### Finding 2: Question Format Creates a Terror Gradient

| Format | Mean KL | Disruption Rank |
|--------|---------|----------------|
| Statement ("X is?") | **0.51** | Most disruptive |
| Imperative ("Explain X.") | 0.44 | Disruptive |
| Yes/No ("Do you know X?") | 0.27 | Moderate |
| WH-question ("What is X?") | **0.14** | Least disruptive |

The "terror of question" is actually **terror of unconstrained format**. WH-questions anchor the model to an answer type. Statements give grammatical freedom without semantic anchor.

### Finding 3: Cross-Language Variance is Massive

| Language Family | Mean KL | Top-1 Overlap |
|-----------------|---------|---------------|
| Romance (Spanish/French) | 0.16-0.28 | 67-100% |
| Germanic (German/Dutch) | 0.57-1.50 | 0-33% |
| Slavic (Russian) | 1.64 | 0% |
| CJK (Chinese/Japanese/Korean) | 2.47-3.05 | 0% |

Llama-2 is English-centric. **The model does not "know" the same concept across languages.** A detector calibrated on English will produce false unknowns for all non-English inputs.

### Finding 4: Format Variance Discriminates Unknowns

| Class | Format Variance (CV%) |
|-------|----------------------|
| Known | **14.6%** |
| Unknown (in-domain) | **43.9%** |

Unknown questions show **3x higher variance** across formats. The variance itself is a signal.

---

## Part 3: Is the Method Statistically Sound?

### What "Statistically Sound" Requires

| Criterion | Required | Current State | Gap |
|-----------|----------|---------------|-----|
| Sample size | N > 100 per class | N = 6 total | **Critical** |
| Held-out test set | Yes, never seen during calibration | Same questions used for calibration and test | **Critical** |
| Cross-validation | k-fold, k >= 5 | None | **Critical** |
| Confidence intervals | 95% CI on accuracy | None | **Major** |
| False positive rate | Measured per class | Not measured | **Major** |
| False negative rate | Measured per class | Not measured | **Major** |
| Edge case coverage | Systematic adversarial testing | None | **Critical** |
| Reproducibility | Fixed random seed, documented environment | Not documented | **Minor** |

### The Sample Size Problem

Current test: 6 questions (4 known, 2 unknown).

With N=6, the 95% confidence interval on 83% accuracy is:
```
CI = p ± z * sqrt(p(1-p)/n)
CI = 0.83 ± 1.96 * sqrt(0.83*0.17/6)
CI = 0.83 ± 0.30
CI = [0.53, 1.00]
```

The true accuracy could be anywhere from **53% to 100%**. We cannot distinguish "works well" from "coincidence."

For a 95% CI of ±5% (acceptable for production), we need:
```
n = (z^2 * p * (1-p)) / E^2
n = (1.96^2 * 0.83 * 0.17) / 0.05^2
n ≈ 217 questions per class
```

**We need ~434 questions total (217 known, 217 unknown) for statistical soundness.**

### The Calibration/Test Contamination Problem

Current detector:
1. Calibrates on known_questions and unknown_questions lists
2. Tests on the SAME lists
3. Reports accuracy

This is **data leakage**. The detector has "seen" the test questions during calibration. Accuracy is inflated.

**Required fix:** Split into train/calibration/test sets. Calibrate on train, tune threshold on calibration, report accuracy on test.

---

## Part 4: Critical Edge Cases Missing

### Edge Case 1: Adversarial Phrasing

**Question:** Can we find phrasings that flip a known question to "unknown" or vice versa?

**Current state:** Not tested.

**Risk:** An attacker could bypass the detector by rephrasing. E.g., "What is the capital of France?" → "The capital of France is?" changes the combined score by ~0.3.

**Required test:** Systematic search for adversarial paraphrases within each format.

### Edge Case 2: Partial Knowledge

**Question:** What happens when the model knows SOME but not ALL of the answer?

**Examples:**
- "What is the capital of France and its population?" (knows capital, not population)
- "How does CRISPR work and what are its limitations?" (knows mechanism, not latest limitations)

**Current state:** Binary known/unknown. No gradation.

**Risk:** The detector will classify these as "known" because the model can answer the first part. The answer will be partially wrong.

**Required test:** Measure detector on questions with known first half, unknown second half.

### Edge Case 3: Temporal Shift

**Question:** Does the detector handle time-dependent knowledge correctly?

**Examples:**
- "Who won the 2020 US election?" (known — happened)
- "Who won the 2032 US election?" (unknown — future)
- "What is the current president of the US?" (known at training time, may be wrong now)

**Current state:** Not tested.

**Risk:** The detector cannot distinguish "known at training time but wrong now" from "genuinely unknown."

**Required test:** Include temporal questions with known cutoff dates.

### Edge Case 4: Nonsense Questions

**Question:** What happens with syntactically valid but semantically empty questions?

**Examples:**
- "What is the color of Tuesday?"
- "How many angels dance on the head of a pin?"
- "Can a square circle?"

**Current state:** Not tested.

**Risk:** The model may generate confidently for nonsense questions (overfitting to grammatical patterns). The detector may classify as "known" because the entropy is low (the model "knows" it's nonsense and responds with "There is no such thing...").

**Required test:** Include 20+ nonsense questions.

### Edge Case 5: Ambiguous Questions

**Question:** What happens when the question has multiple valid interpretations?

**Examples:**
- "What is the best programming language?" (subjective)
- "What is the largest city?" (by area? population? metropolitan?)
- "Is Python better than Java?" (depends on criteria)

**Current state:** Not tested.

**Risk:** The detector has no question-type classifier. Subjective questions will be treated as factual.

**Required test:** Include ambiguous and subjective questions.

### Edge Case 6: Meta-Questions

**Question:** What happens when the question is about the model itself?

**Examples:**
- "What is your training cutoff date?"
- "Who created you?"
- "What model are you?"

**Current state:** Not tested.

**Risk:** These are "known" to the model (in training data) but may produce unusual distributions because the model's self-knowledge is encoded differently.

**Required test:** Include meta-questions.

### Edge Case 7: Multi-Hop Reasoning

**Question:** What happens when the question requires multiple inference steps?

**Examples:**
- "If Paris is the capital of France, what is the capital of the country that borders France to the east?" (Germany)
- "What is the nationality of the author of '1984'?" (George Orwell → British)

**Current state:** Not tested.

**Risk:** The first token after the question may not encode the full reasoning chain. The Pass 1 entropy may be misleading.

**Required test:** Include multi-hop questions.

### Edge Case 8: Counterfactuals

**Question:** What happens with counterfactual questions?

**Examples:**
- "What if gravity didn't exist?"
- "If the Earth had two moons, what would tides be like?"
- "What would happen if water froze at 50 degrees?"

**Current state:** Not tested.

**Risk:** Counterfactuals may produce high entropy (model explores alternatives) but the model "knows" the physics. The detector may misclassify as "unknown."

**Required test:** Include counterfactual questions.

### Edge Case 9: Very Short vs Very Long Questions

**Question:** Is the detector robust to length extremes?

**Examples:**
- Short: "Pi?"
- Long: "In the context of general relativity and considering the Schwarzschild metric, what is the exact formula for the gravitational time dilation experienced by an observer at a distance r from a non-rotating black hole of mass M, and how does this relate to the event horizon?"

**Current state:** Not tested.

**Risk:** Length affects embedding norm and entropy. The detector may have length bias.

**Required test:** Include questions from 1 token to 100+ tokens.

### Edge Case 10: Cross-Domain Hybrids

**Question:** What happens with questions that mix multiple domains?

**Examples:**
- "Can topological persistence detect phase transitions in LLM training?" (math + CS)
- "Does quantum biology explain consciousness?" (physics + biology + philosophy)
- "What is the computational complexity of DNA folding?" (CS + biology)

**Current state:** Tested minimally (topological persistence, Wasserstein distance).

**Risk:** The embedding distance signal may be confused. The question is "close" to known concepts in both domains but the connection is novel.

**Required test:** Include 30+ cross-domain hybrids.

### Edge Case 11: Questions with Known Components but Unknown Connection

**Question:** What happens when the model knows A and B but not A→B?

**Examples:**
- "Can CRISPR cure Alzheimer's?" (knows CRISPR, knows Alzheimer's, doesn't know the link)
- "Does hyperbolic geometry improve LLM reasoning?" (knows both, doesn't know the link)

**Current state:** Tested (hyperbolic geometry LLM, topological persistence).

**Finding:** These show low format variance (in-domain unknown). The model has partial knowledge.

**Gap:** The detector classifies these as "unknown" but does not indicate "partial knowledge."

### Edge Case 12: Highly Specialized vs General Knowledge

**Question:** Does the detector distinguish general knowledge from niche expertise?

**Examples:**
- General: "What is gravity?"
- Niche: "What is the Kurskal-Szekeres coordinate transformation?" (GR)
- Obscure: "What is the 37th digit of pi?"

**Current state:** Not systematically tested.

**Risk:** Niche known facts may be classified as "unknown" because the model's representation is sparse.

### Edge Case 13: Model Instability During Session

**Question:** Does the detector's baseline drift during a long session?

**Current state:** Pre-flight stability checker exists but has not been stress-tested.

**Risk:** After 100+ questions, the model's internal state may shift (attention patterns accumulate bias). The calibrated references may become stale.

**Required test:** Run 200+ questions sequentially and measure drift.

### Edge Case 14: Temperature Sensitivity

**Question:** Does the detector's Pass 1 signal change with temperature?

**Current state:** Pass 1 uses temperature=0.0.

**Risk:** At T>0, the entropy signal would increase artificially (more randomness). The detector might misclassify known questions as unknown at higher temperatures.

**Required test:** Measure signal at T=0, 0.5, 1.0.

---

## Part 5: Required Validation Protocol

To claim statistical soundness, the following protocol must be executed:

### Phase 1: Dataset Construction (1 week)

| Category | Count | Description |
|----------|-------|-------------|
| Known — general | 50 | Common knowledge (gravity, capital of France) |
| Known — niche | 50 | Specialized but factual (Kruskal-Szekeres, 37th digit of pi) |
| Known — temporal | 25 | Time-dependent facts pre-2024 |
| Unknown — in-domain | 50 | Known concepts, unknown connection (CRISPR + Alzheimer's) |
| Unknown — out-of-domain | 50 | No relevant training data (Mars Colony 2035) |
| Unknown — frontier | 50 | Active research without consensus |
| Nonsense | 25 | Syntactically valid, semantically empty |
| Ambiguous/subjective | 25 | Multiple valid interpretations |
| Meta | 25 | About the model itself |
| Counterfactual | 25 | Hypothetical scenarios |
| Multi-hop | 25 | Requires reasoning chain |
| Cross-domain hybrid | 25 | Mixes 2+ domains |
| **TOTAL** | **425** | |

### Phase 2: Train/Calibration/Test Split

- Train (calibration): 200 questions (100 known, 100 unknown)
- Calibration (threshold tuning): 100 questions (50 known, 50 unknown)
- Test (reporting): 125 questions (all categories, balanced)

### Phase 3: Cross-Validation (1 week)

Run 5-fold cross-validation:
1. Split train into 5 folds
2. For each fold: train on 4 folds, validate on 1
3. Average accuracy across 5 runs
4. Report mean ± std dev

### Phase 4: Edge Case Evaluation (1 week)

For each edge case category, report:
- Accuracy
- False positive rate
- False negative rate
- Mean combined score
- Score variance across formats

### Phase 5: Adversarial Testing (1 week)

For each test question, generate 5 adversarial variants:
- Format flip (WH → statement)
- Negation flip ("What is X?" → "What is not X?")
- Synonym swap
- Length manipulation (shorten/extend)
- Domain prefix ("In physics, what is...?")

Report robustness: fraction of variants that preserve the correct classification.

---

## Part 6: One-Line Verdict

**The two-pass detector shows genuine signal (83% on preliminary tests, four orthogonal signals, format stability patterns, language variance patterns) but is NOT statistically sound for production. Critical gaps: sample size (need 400+ questions), held-out test set, cross-validation, edge case coverage (14 categories untested), and adversarial robustness. The harness design is principled but unvalidated.**

---

## Recommended Immediate Actions

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| 1 | Build 400+ question dataset with category labels | 3 days | Critical |
| 2 | Implement train/calibration/test split | 1 day | Critical |
| 3 | Run 5-fold cross-validation | 2 days | Critical |
| 4 | Test all 14 edge case categories | 2 days | Critical |
| 5 | Adversarial phrasing robustness test | 2 days | Major |
| 6 | Per-language calibration (Spanish, French) | 2 days | Major |
| 7 | Multi-format ensemble implementation | 2 days | Major |
| 8 | Temperature sensitivity test | 1 day | Minor |
| 9 | Long-session drift test | 1 day | Minor |
| 10 | Confidence interval reporting | 0.5 day | Minor |

**Total effort: ~2 weeks to achieve statistical soundness.**
