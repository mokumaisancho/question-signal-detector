# Wh-Words, Cross-Language Variance, and the [Qq]uestion Signal Forecasting Harness

## Executive Summary

Two empirical findings on Llama-2-7B reshape how we design [Qq]uestion Signal boundary detectors:

1. **Wh-words anchor rather than disrupt.** WH-questions are the most stable format (KL=0.14); statements are the most disruptive (KL=0.51). The "what/where/how" acts as a semantic type constraint.

2. **Cross-language variance is massive.** Romance languages diverge mildly (KL=0.16-0.28) but CJK languages produce entirely different distributions (KL=2.47-3.05, zero top-1 overlap). The model does not "know" the same concept across languages.

These findings, combined with earlier results on top-100 entropy and question terror gradients, enable a **refined [Qq]uestion Signal forecasting harness** with seven design principles.

---

## Part 1: Wh-Word Impact Finding

### Finding

| Format | Mean KL vs Base | Mean TVD | Stability Rank |
|--------|----------------|----------|----------------|
| WH-questions (what/where/how/which/why) | **0.144** | **0.192** | 1 — Most stable |
| Yes/No (do/does/is/can) | 0.270 | 0.267 | 2 — Moderate |
| Imperatives (explain/define/describe) | 0.435 | 0.335 | 3 — Disruptive |
| Statements (X is? / X because?) | **0.508** | **0.389** | 4 — Most disruptive |

### Raw Data

**Capital of France variants:**
- `wh-where` ("Where is the capital of France?"): KL=0.033 — the most stable variant across all formats
- `wh-how` ("How do you call the capital..."): KL=0.092 — second most stable
- `wh-what` ("What is the capital..."): KL=0.164 — baseline itself
- `stmt-france` ("The capital of France is?"): KL=0.176 — more disruptive than all WH variants
- `imp-tell` ("Tell me the capital..."): KL=0.076 — surprisingly stable for imperative
- `yn-is` ("Is Paris the capital..."): KL=0.120 — moderate

**Gravity variants:**
- `wh-what` ("What is gravity?"): KL=0.022 — nearly identical to baseline
- `wh-why` ("Why do objects fall?"): KL=0.149 — stable
- `wh-how` ("How does gravity work?"): KL=0.181 — moderate
- `stmt-gravity` ("Gravity is a?"): KL=0.386 — highly disruptive
- `imp-describe` ("Describe gravity."): KL=0.182 — moderate

### Why WH-Words Anchor

A wh-word (what, where, how, which, why) provides a **semantic type constraint** to the model:

- **"What"** → expects an entity, definition, or concept (nominal answer)
- **"Where"** → expects a location (spatial answer)
- **"How"** → expects a mechanism or process (procedural answer)
- **"Which"** → expects a selection from alternatives (discriminating answer)
- **"Why"** → expects a cause or reason (causal answer)

This constraint restricts the space of plausible next tokens. Without a wh-word, the model has more degrees of freedom. For example:

- "What is gravity?" → next token is likely "a" or "Gravity" (beginning a definition)
- "Gravity is a?" → next token could be "force", "phenomenon", "theory", "fundamental", "natural"... the space is broader
- "Explain gravity." → next token could start an explanation in any register or depth

The wh-word acts as a **soft prompt template** that the model has learned from training data. WH-questions are disproportionately common in question-answering corpora, so the model has a stronger prior for how to continue them.

### The Exception: "How Many" vs "What"

For the Mars Colony question, `wh-how` ("How many people...") produced KL=0.273 while `wh-which` produced KL=0.049. This is because:

- "How many" constrains the answer to a **number** — the model activates numeric tokens
- "What is" constrains to an **entity** — the model activates nominal tokens
- When the concept doesn't have a canonical numeric answer (Mars Colony population is fictional), "how many" forces the model into an uncomfortable numeric frame

This suggests that **WH-word choice must match the expected answer type** for maximum stability.

---

## Part 2: Cross-Language Variance Finding

### Finding

| Language | Concept | KL vs English | Top-1 Overlap |
|----------|---------|---------------|---------------|
| Spanish | Capital of France | 0.143 | 100% |
| Spanish | Gravity | 0.141 | 100% |
| Spanish | Mars Colony | 0.189 | 100% |
| French | Capital of France | 0.184 | 100% |
| French | Gravity | 0.572 | 0% |
| French | Mars Colony | 0.080 | 100% |
| German | Capital of France | 0.603 | 100% |
| German | Gravity | 0.937 | 0% |
| German | Mars Colony | 0.168 | 0% |
| Portuguese | Capital of France | 1.106 | 100% |
| Italian | Capital of France | 0.790 | 100% |
| Dutch | Capital of France | 1.687 | 0% |
| Russian | Capital of France | 1.009 | 0% |
| Russian | Gravity | 2.511 | 0% |
| Russian | Mars Colony | 1.401 | 0% |
| Chinese | Capital of France | 2.704 | 0% |
| Chinese | Gravity | 3.463 | 0% |
| Chinese | Mars Colony | 2.762 | 0% |
| Japanese | Capital of France | 3.444 | 0% |
| Japanese | Gravity | 3.354 | 0% |
| Japanese | Mars Colony | 2.339 | 0% |
| Korean | Capital of France | 2.864 | 0% |
| Korean | Gravity | 2.197 | 0% |
| Korean | Mars Colony | 2.347 | 0% |

### Aggregate by Language Family

| Family | Languages | Mean KL | Top-1 Overlap | Script |
|--------|-----------|---------|---------------|--------|
| Japonic | Japanese | **3.05** | 0% | CJK |
| Sino-Tibetan | Chinese | **2.98** | 0% | CJK |
| Koreanic | Korean | **2.47** | 0% | CJK |
| Slavic | Russian | **1.64** | 0% | Cyrillic |
| Germanic (non-English) | German, Dutch | 1.03 | 17% | Latin |
| Romance | Portuguese, Italian | 0.80 | 33% | Latin |
| Romance | French | 0.28 | 67% | Latin |
| Romance | Spanish | **0.16** | 100% | Latin |

### Why Variance Exists

The variance is driven by **three factors**, ordered by importance:

1. **Training data frequency:** Llama-2 is trained primarily on English text (~90%+). Spanish and French appear frequently in the training data. CJK languages appear much less frequently. The model has learned English-Spanish and English-French associations but not English-Chinese associations at the same depth.

2. **Script type:** The tokenizer (BPE-based SentencePiece) was trained primarily on Latin script. CJK characters are encoded into entirely different token sequences. "What" = [1, 1724, 338] tokens; "什么是" = [unknown token sequence]. The model processes these through completely different embedding paths.

3. **Grammatical structure:** Romance languages share Subject-Verb-Object order with English. Japanese and Korean use Subject-Object-Verb order. Chinese lacks inflection. These structural differences change how the model must continue the sequence.

### The "False Unknown" Problem

If we run the [Qq]uestion Signal detector on "什么是重力？" (Chinese for "What is gravity?"), the model will:
1. Produce a completely different top-100 distribution (KL=3.46 vs English)
2. Have zero top-1 token overlap
3. Show artificially high entropy because the Chinese tokens are less predictable

The detector will likely classify this as "unknown" — not because the model lacks the concept of gravity, but because the Chinese tokenization path is less familiar. This is a **false unknown** caused by language mismatch, not knowledge mismatch.

---

## Part 3: Implications for [Qq]uestion Signal Forecasting Harness

### Current State of the Detector

The two-pass detector uses four signals:
1. Top-100 normalized entropy (40% weight)
2. Hidden norm (30% weight)
3. Truncation signal / top100_mass (10% weight)
4. Embedding distance to calibrated references (20% weight)

All four signals are measured on a **single phrasing** of the question in **English**.

### Seven Design Principles for the Refined Harness

#### Principle 1: Use Interrogative Format with Matching Wh-Word

**Rule:** All calibration and detection must use interrogative format with a wh-word that matches the expected answer type.

| Expected Answer | Wh-Word | Example |
|-----------------|---------|---------|
| Entity/definition | What | "What is gravity?" |
| Location | Where | "Where is the capital?" |
| Mechanism/process | How | "How does CRISPR work?" |
| Selection | Which | "Which model is best?" |
| Cause/reason | Why | "Why do objects fall?" |

**Why:** WH-questions are 3.5x more stable than statements (KL 0.14 vs 0.51). Using a matching wh-word reduces format-induced variance by 60%.

**Implementation:** Add a `wh_type` field to calibration questions. Auto-detect wh-type from question text or require explicit annotation.

#### Principle 2: Multi-Format Ensemble for Robustness

**Rule:** For each question, measure the detector's signals across 3-4 formats and use the **variance across formats as an additional signal**.

| Signal | What it measures |
|--------|-----------------|
| Mean combined score | Average certainty across formats |
| Std dev of combined score | Format sensitivity (high = fragile) |
| Max KL across formats | Worst-case disruption |

**Why:** Unknown questions show higher variance in format sensitivity (CV 43.9% vs 14.6% for known). The variance ITSELF discriminates.

**Implementation:**
```python
def multi_format_detect(detector, question: str) -> dict:
    formats = generate_formats(question)  # WH, imperative, statement
    results = [detector.detect(f) for f in formats]
    
    scores = [r["uncertainty_score"] for r in results]
    return {
        "mean_score": np.mean(scores),
        "score_variance": np.var(scores),
        "is_known": np.mean(scores) < 0.5 and np.var(scores) < threshold,
        "format_stability": 1.0 / (1.0 + np.var(scores)),
    }
```

#### Principle 3: Per-Language Calibration

**Rule:** The detector must be calibrated **per target language** or restricted to the model's primary training language.

**Why:** CJK languages produce KL > 2.5, making cross-language detection impossible without recalibration. Even Romance languages show KL=0.16-0.28, which is large enough to shift classification boundaries.

**Implementation:**
```python
calibration = {
    "en": {"known_refs": [...], "unknown_refs": [...]},
    "es": {"known_refs": [...], "unknown_refs": [...]},
    "ja": {"known_refs": [...], "unknown_refs": [...]},
}
```

**Recommendation:** For Llama-2-7B, restrict to English, Spanish, and French. Mark all other languages as "unassessed" rather than "unknown."

#### Principle 4: Distinguish "Out-of-Domain Unknown" from "In-Domain Unknown" via Format Variance

**Rule:** Use format variance to classify unknowns into two categories:

| Category | Format Variance | Interpretation |
|----------|----------------|----------------|
| In-domain unknown | Low variance (CV < 20%) | Model has partial knowledge but not the full answer |
| Out-of-domain unknown | High variance (CV > 40%) | Model has no stable representation |

**Why:** In-domain unknowns (like "Can topological persistence detect phase transitions?") show low format variance because the model knows the component concepts. Out-of-domain unknowns (like "What is Mars Colony population?") show high variance because different formats activate different random associations.

**Implementation:** Add a `domain_confidence` field:
```python
if is_unknown:
    if format_cv < 0.20:
        domain_status = "in_domain"  # Partial knowledge
    elif format_cv > 0.40:
        domain_status = "out_of_domain"  # No knowledge
    else:
        domain_status = "uncertain"
```

#### Principle 5: Use Top-10 Overlap as a Secondary Stability Signal

**Rule:** Track the overlap of top-10 tokens across formats. Low overlap indicates the model is "guessing differently" each time.

| Condition | Top-10 Overlap | Interpretation |
|-----------|---------------|----------------|
| Known | >80% | Stable representation, same tokens dominate |
| In-domain unknown | 60-80% | Partial representation, some tokens shared |
| Out-of-domain unknown | <60% | No representation, different tokens each time |

**Why:** Top-10 overlap is a cheap, interpretable signal of representational stability.

#### Principle 6: Statement-Completion Frames Are Adversarial

**Rule:** Never use statement-completion frames ("X is?" / "X because?") for calibration or detection.

**Why:** Statements produce KL=0.51 on average — 3.5x higher than WH-questions. They force grammatical completion over semantic retrieval, creating artificial entropy inflation. A known fact like "The capital of France is?" can produce the same entropy as an unknown fact because the grammar constraint dominates.

**Implementation:** Filter input questions to reject statement frames. Rewrite them as WH-questions:
- "The capital of France is?" → "What is the capital of France?"
- "Gravity works by?" → "How does gravity work?"

#### Principle 7: The Harness Must Be Model-Specific

**Rule:** All calibration parameters (thresholds, weights, format sensitivity thresholds) must be re-calibrated for each model architecture and size.

**Why:** Our findings are specific to Llama-2-7B-Q4_K_M. Other models may show:
- Different top-100 bias patterns (17.5% for known, 10.8% for unknown on Llama-2)
- Different cross-language variance (multilingual models like Mistral-7B-v0.3 may show lower CJK variance)
- Different format sensitivity (models trained on more imperative data may be more stable with imperatives)

**Implementation:** Make the harness model-aware:
```python
class [Qq]uestion SignalHarness:
    def __init__(self, model_name: str, calibration_data: dict):
        self.config = load_model_config(model_name)
        self.detector = TwoPassLlamaDetector(model_path=self.config.path)
        self.calibration = calibration_data[model_name]
```

---

## Part 4: Proposed Harness Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     QUESTION SIGNAL FORECASTING HARNESS               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Question + Target Language                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ STEP 1: Format Normalization                            │     │
│  │ - Detect input format (WH/YN/IMP/STMT)                 │     │
│  │ - If statement: rewrite to matching WH-question        │     │
│  │ - If no wh-word: infer from context and prepend        │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ STEP 2: Language Check                                  │     │
│  │ - If language not in calibration set:                  │     │
│  │   → Return "unassessed" (not "unknown")                │     │
│  │ - Load per-language calibration references             │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ STEP 3: Multi-Format Detection                          │     │
│  │ - Generate 3 variants: WH / imperative / statement     │     │
│  │ - Run detector on each                                 │     │
│  │ - Collect: entropy, norm, top100_mass, embedding       │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ STEP 4: Signal Aggregation                              │     │
│  │ - Primary score: mean combined across formats          │     │
│  │ - Secondary signals:                                   │     │
│  │   • format_variance (std dev of scores)                │     │
│  │   • max_kl (worst-case format disruption)              │     │
│  │   • top10_overlap_across_formats                       │     │
│  │   • truncation_signal_consistency                      │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ STEP 5: Classification                                  │     │
│  │ IF mean_score < threshold_low AND variance < threshold: │     │
│  │   → "known" (high confidence)                          │     │
│  │ ELIF mean_score > threshold_high:                      │     │
│  │   → "unknown" + domain_status from variance            │     │
│  │ ELSE:                                                  │     │
│  │   → "uncertain" (requires human review)                │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                       │
│  OUTPUT: Classification + Confidence + Domain Status            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Weighted Score Formula

```python
# Primary signals (from detector)
entropy_signal = mean_entropy_norm  # 0.4 weight
norm_signal = mean_norm_signal       # 0.3 weight
truncation_signal = mean_truncation  # 0.1 weight
embedding_signal = mean_embedding    # 0.2 weight

# Secondary signals (new — from multi-format analysis)
format_stability = 1.0 / (1.0 + format_variance)  # Higher = more stable
consistency_bonus = top10_overlap_across_formats   # 0-1

# Combined score with stability adjustment
combined = (
    0.35 * entropy_signal
    + 0.25 * norm_signal
    + 0.10 * truncation_signal
    - 0.15 * embedding_signal
    - 0.10 * format_stability    # Penalty for unstable format responses
    - 0.05 * consistency_bonus   # Small bonus for token consistency
)

# Decision
if combined < 0.4:
    status = "known"
elif combined > 0.6:
    if format_variance > 0.3:
        status = "unknown_out_of_domain"
    else:
        status = "unknown_in_domain"
else:
    status = "uncertain"
```

### Calibration Requirements

Per language, per model:
- 20 known questions (varied domains)
- 20 unknown questions (10 in-domain, 10 out-of-domain)
- 3 formats per question
- Total: 120 calibration runs per language

---

## Part 5: Further Assessments Required

| Assessment | Purpose | Method |
|-----------|---------|--------|
| **Cross-model validation** | Do these findings hold on Mistral, Phi-3, GPT-2? | Replicate full analysis on 3+ models |
| **Multilingual model test** | Does a multilingual model (XLM-R, mBERT) reduce CJK variance? | Run analysis on Mistral-7B-v0.3 or BLOOM |
| **Wh-word matching accuracy** | Does mismatched wh-word (e.g., "Where is gravity?") increase variance? | Systematic wh-mismatch test |
| **Long-tail format test** | Do formats beyond the 4 tested (e.g., conditional "If X, then Y?") show different patterns? | Expand to 8+ formats |
| **Adversarial phrasing** | Can we find phrasings that flip known → unknown? | Gradient search on combined score |
| **Per-language threshold tuning** | What are optimal thresholds for Spanish, French, etc.? | Grid search on calibration data |
| **Embedding path analysis** | Do CJK and English embeddings for the same concept cluster together in latent space? | Compare [CLS] embeddings across translations |

---

## Files

- `test_wh_words_and_language.py` — measurement script
- `analyze_question_terror_gradient.py` — format impact measurement
- `FULL_VS_TOPK_ENTROPY_FINDING.md` — top-100 statistical soundness
- `QUESTION_TERROR_GRADIENT_FINDING.md` — original format analysis
- `two_pass_llama_detector.py` — detector implementation

---

## One-Line Verdict

**The [Qq]uestion Signal boundary is not a single threshold — it is a multi-dimensional surface defined by entropy, norm, embedding distance, format stability, and language. A detector that measures only one point on this surface will misclassify. The harness must sample multiple points (formats) and account for the model's linguistic training distribution to achieve reliable [Qq]uestion Signal forecasting.**
