# Question Terror Gradient: How Format Impacts LLM Distributions

## Executive Summary

**The "terror of question" insight requires refinement.** Direct interrogatives ("What is X?") are the LEAST disruptive format. Statement-completion frames ("X is?") are the MOST disruptive. Imperatives and indirect statements fall in between.

The impact is concentrated in the **head of the distribution** (ranks 1-10), which sees 73-81% of all probability mass shifts. The tail (ranks 100+) is essentially unaffected by phrasing changes.

Unknown questions show **higher variance** in format sensitivity — some are extremely fragile (Mars Colony), others are robust (Topological Persistence). Known questions show **consistent moderate sensitivity**.

---

## Key Findings

### 1. Format Impact Ranking (Most to Least Disruptive)

| Format | Mean KL | Mean TVD | Mean ΔH | Pattern |
|--------|---------|----------|---------|---------|
| **Statement** | 0.346 | 0.300 | -0.98 | Most terrifying — forces specific completion frame |
| **Indirect** | 0.261 | 0.256 | -0.49 | Moderate — contextual embedding |
| **Imperative** | 0.157 | 0.217 | -0.46 | Moderate — explicit demand |
| **Interrogative** | **0.147** | **0.189** | **-0.20** | **Least terrifying — open-ended inquiry** |

Direct questions are **2.3x less disruptive** than statement frames. This contradicts the naive intuition that "asking questions terrorizes the model."

### 2. Distribution of Change — The Terror Gradient

| Rank Band | % of Total Probability Change |
|-----------|-------------------------------|
| Ranks 1-10 (head) | **73-81%** |
| Ranks 11-100 (mid) | **19-27%** |
| Ranks 100+ (tail) | **~0%** |

Format changes do not touch the tail. All reshuffling happens in the head. This validates the top-100 capture approach — if the tail doesn't change, full distribution analysis adds no robustness to phrasing variation.

### 3. Mass Shift Pattern

All formats shift probability mass **into the head** (top 10) and **out of the mid/tail**:

| Format | Head Δ (top 10) | Mid Δ (11-100) | Tail Δ (100+) |
|--------|----------------|----------------|---------------|
| Statement | **+12.1%** | **-9.8%** | -2.3% |
| Imperative | +4.8% | -3.7% | -1.1% |
| Interrogative | +3.3% | -1.9% | -1.4% |

Changing format makes the model **more peaked**, not less. The model narrows its focus when the grammatical frame constrains it.

### 4. Known vs Unknown Sensitivity Variance

| Concept | Label | Mean KL | Max KL | Sensitivity |
|---------|-------|---------|--------|-------------|
| Mars Colony | unknown | **0.420** | **1.031** | **Extremely fragile** |
| Gravity | known | 0.173 | 0.316 | Moderate |
| France Capital | known | 0.152 | 0.330 | Moderate |
| Topological Persistence | unknown | **0.107** | **0.144** | **Robust** |

**Unknown questions show higher variance in sensitivity.** Some unknown questions (Mars Colony) are extremely fragile to phrasing — the model has no stable representation and different frames send it in completely different directions. Other unknown questions (Topological Persistence) are robust — the model has partial knowledge of the domain and maintains a consistent response pattern regardless of phrasing.

**Known questions are consistently moderate.** The model has stable representations and format changes produce predictable, bounded effects.

---

## Why Unknown Questions Show Higher Variance

### The "Partial Knowledge" Hypothesis

Unknown questions fall into two categories that explain the variance:

**Category A: Out-of-domain unknown (high sensitivity)**
- Example: "What is Mars Colony population in 2035?"
- The model has NO knowledge of the specific concept
- Different phrasings activate different associative patterns:
  - "What is Mars Colony..." → activates Mars facts, gets confused
  - "How many people live on Mars..." → activates population statistics, different confusion
  - "The Mars Colony population is?" → forces completion frame, highest terror
- Result: Each phrasing produces a completely different next-token distribution

**Category B: In-domain unknown (low sensitivity)**
- Example: "Can topological persistence detect phase transitions?"
- The model knows the individual concepts (topological persistence, phase transitions)
- It lacks the specific connection but has domain structure
- Different phrasings all activate the same domain association
- Result: Distribution shifts are small and consistent regardless of format

**Known questions are always in-domain** — the model has stable representations, so format changes produce bounded, predictable shifts.

### The "Associative Activation" Mechanism

When an unknown question is asked:
1. **Interrogative format** ("What is X?") → activates "X" as a concept → model searches knowledge graph → doesn't find → moderate confusion
2. **Imperative format** ("State X") → activates "state facts about X" → model tries to recall facts → same confusion, different frame
3. **Statement format** ("X is?") → forces grammatical completion → model generates whatever fits syntactically → highest variance because grammar constraints override semantic search

For **out-of-domain unknowns**, each format activates a different associative path, leading to divergent distributions.

For **in-domain unknowns**, all formats activate the same domain neighborhood, leading to convergent distributions.

For **known questions**, all formats converge on the same stable representation.

---

## Implications for Detector Design

### 1. Calibration Format Matters

The detector's calibration should use **interrogative format** exclusively:
- Most stable across variants
- Smallest KL divergence between paraphrases
- Produces the most consistent entropy and norm signals

### 2. Avoid Statement-Completion Frames

Statement frames like "X is?" or "The nature of X is?" should be avoided:
- Produce the largest distribution shifts
- Force grammatical completion over semantic retrieval
- Inflate entropy artificially through syntactic constraint rather than genuine uncertainty

### 3. Top-10 Is Where the Action Is

Since 73-81% of format-induced changes happen in ranks 1-10:
- The top-100 capture is validated — it contains all the relevant signal
- Full distribution analysis would not improve robustness to phrasing
- The tail (ranks 100+) is phrasing-invariant noise

### 4. Detecting "Partial Knowledge" via Format Variance

The variance in format sensitivity ITSELF is a signal:
- High variance across formats → out-of-domain unknown (model has no anchor)
- Low variance across formats → in-domain unknown (model has partial anchor) or known
- This could be a **fifth signal** for the detector: measure KL divergence across a small set of paraphrases

### 5. Further Assessments Required

| Assessment | Purpose | Method |
|-----------|---------|--------|
| **Domain distance correlation** | Does sensitivity correlate with embedding distance to known concepts? | Measure format-KL vs min_known_dist |
| **Token-level analysis** | Which specific tokens change across formats? | Track top-10 token identity shifts |
| **Multi-format ensemble** | Can combining signals from multiple formats improve discrimination? | Average combined scores across formats |
| **Adversarial phrasing** | Can we find phrasings that flip the decision? | Gradient-based or heuristic search for borderline cases |
| **Cross-model validation** | Does this pattern hold across models (GPT-2, Mistral, etc.)? | Replicate on other architectures |

---

## Methodology

- Model: Llama-2-7B-Q4_K_M (3.9GB GGUF)
- Vocabulary: 32,000 tokens
- Base questions: 4 concepts (2 known, 2 unknown)
- Variants per concept: 6-9 phrasings across 4 formats
- Full logits extracted via `llama_cpp.Llama.eval()` + `_scores[-1]`
- Metrics: KL divergence, Jensen-Shannon divergence, total variation distance, top-K overlap, per-band mass shift

---

## Files

- `analyze_question_terror_gradient.py` — measurement script
- `two_pass_llama_detector.py` — detector implementation
