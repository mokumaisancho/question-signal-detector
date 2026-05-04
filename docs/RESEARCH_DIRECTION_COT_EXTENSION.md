# Research Direction: [Qq]uestion Signal Filter → Chain-of-Thought Recursive Assessment

**Date:** 2026-05-04  
**Status:** Conceptual — requires PoC validation  
**Priority:** High (extends current framework to reasoning traces)

---

## The Core Idea

The current two-pass architecture detects uncertainty **before** generation starts. The natural extension is to run Pass 1 **at every step** of a chain-of-thought (CoT) reasoning trace, creating a recursive uncertainty monitor that can:

1. Flag uncertain inference steps in real-time
2. Detect divergence across multiple reasoning trajectories
3. Backtrack to the last "cold" (confident) step when uncertainty spikes
4. Abort with "uncertain" rather than hallucinate forward

---

## What Holds Now (No Additional Work Needed)

### 1. Per-Step Signal Extraction

Your Pass 1 signals are extractable at any token position:

```python
# After generating step k of a CoT:
reasoning_so_far = "Step 1: ... Step 2: ... Step k:"
p1 = detector._pass1_uncertainty(reasoning_so_far)
# Gives: entropy, norm, embedding_distance, top100_mass for position k
```

**Why it holds:** The `_pass1_uncertainty` method takes any text string and returns the next-token distribution. Concatenating reasoning steps is just a longer string.

### 2. Temperature as Diagnostic (Not Hyperparameter)

Your signals already distinguish "cold" vs "hot" thoughts:

| State | Entropy | Norm | top100_mass | Interpretation |
|-------|---------|------|-------------|----------------|
| Cold (confident) | Low | High | High | Likely sound inference |
| Hot (uncertain) | High | Low | Low | Likely hallucinatory |

**Why it holds:** These are properties of the next-token distribution, independent of whether the context is a single question or a multi-step reasoning trace.

### 3. Question-Type Classifier at Step Boundaries

Your classifier can flag when reasoning drifts into risky territory:

```python
# After step k, check if the next step is counterfactual:
next_step_prompt = reasoning_so_far + " Step k+1:"
qtype = QuestionTypeClassifier().classify(next_step_prompt)
if qtype == "counterfactual":
    # Flag before generating the commitment
    flag_uncertain()
```

**Why it holds:** The classifier operates on text strings. A reasoning step is just a text string.

### 4. Self-Consistency for Trajectory Validation

Your consistency check scales to reasoning paths:

```python
# Sample 3 complete CoT trajectories for the same problem
trajectories = [generate_cot(problem, seed=i) for i in range(3)]
# Compare step-by-step where they diverge
divergence_points = find_divergence(trajectories)
# Divergence = uncertainty
```

**Why it holds:** The `_text_similarity` method compares any two text strings. CoT steps are text strings.

---

## What Does NOT Hold Yet (Requires PoC)

### 1. Accumulated Context Bias

**Problem:** In CoT, step 5's input is steps 1-4 concatenated. The embedding at step 5 encodes the entire reasoning trajectory, not just the current sub-question. Norm and entropy may drift due to sequence length, not due to uncertainty about step 5 specifically.

**Evidence:** Your length bias test showed |correlation| > 0.3 between question length and score. In a 10-step CoT, the accumulated context is 10x longer.

**What PoC needs to show:**
- Generate a 5-step CoT on a simple math problem
- Measure Pass 1 signals at each step
- Show that signals from steps 1-3 (where the model is confident) are distinguishable from steps 4-5 (where uncertainty might spike)
- Control for length: compare signals at the same token count but with different reasoning content

### 2. Self-Fulfilling Trajectory

**Problem:** Once the model generates "Let's assume X," step 2's uncertainty is measured AFTER that commitment is baked into the context. The model may show low entropy at step 2 not because it confidently knows step 2, but because step 1 locked it into a path.

**Evidence:** Your counterfactual test shows the model generates garbage like "gravity doesn't exist" when asked "What if gravity didn't exist?" — the premise corrupts the reasoning.

**What PoC needs to show:**
- Generate two trajectories: one with correct premise, one with incorrect premise injected at step 2
- Show that steps 3-5 have measurably different (higher) uncertainty in the incorrect-premise trajectory
- Show that the difference is not just due to length

### 3. Consistency Check Scaling

**Problem:** For a 10-step chain, sampling 3 trajectories means 30 generations. Your current consistency metric (bigram + containment similarity) may break down on long reasoning traces — two correct paths can use completely different vocabulary.

**Evidence:** Your consistency test on counterfactuals gave score=0.277 for answers that were semantically related but lexically different. On 10-step traces, the divergence would be amplified.

**What PoC needs to show:**
- Generate 3 trajectories for a 5-step math problem
- Show that the consensus steps (all 3 agree) have lower uncertainty signals than divergence steps
- Show that the metric works even when trajectories use different vocabulary

### 4. Model Quality Floor

**Problem:** Your focused test shows 2/5 edge cases fixed with the 7B Q4_K_M model. If the model can't handle a single counterfactual question, per-step counterfactual detection within a chain is speculative.

**Evidence:** The model generates empty answers, Hebrew characters, and non-sequiturs for counterfactuals.

**What PoC needs to show:**
- The model must first handle basic counterfactuals before CoT extension is viable
- OR: show that the per-step filter catches the model's confusion before it compounds

---

## PoC Specification

### Goal
Demonstrate that Pass 1 signals can distinguish confident inference steps from uncertain ones within a CoT trace.

### Test Problem
Use a 5-step arithmetic or logic problem (e.g., "If a train travels 60 mph for 2 hours, then 40 mph for 3 hours, what is the average speed?")

### Procedure

```python
# Step 1: Generate baseline CoT
baseline_cot = generate_cot(problem, correct_premise=True)
# Measures: [entropy_1, entropy_2, entropy_3, entropy_4, entropy_5]

# Step 2: Inject incorrect premise at step 2
corrupted_cot = generate_cot(problem, inject_error_at_step=2)
# Measures: [entropy_1, entropy_2', entropy_3', entropy_4', entropy_5']

# Step 3: Compare
for step in [3, 4, 5]:
    assert entropy_corrupted[step] > entropy_baseline[step]
    assert norm_corrupted[step] < norm_baseline[step]
```

### Success Criteria
1. Corrupted trajectory shows higher entropy in steps 3-5 (p < 0.05, paired t-test)
2. Corrupted trajectory shows lower norm in steps 3-5
3. Backtracking: re-sampling from step 2 produces lower uncertainty than continuing from step 3
4. Question-type classifier flags the incorrect premise as "counterfactual" or "uncertain"

### Estimated Effort
- 8-12 hours
- Requires: working 7B+ model, 5 test problems, statistical analysis

---

## Architecture Sketch

```
Input: Problem P
  |
  v
[Generate Step 1]
  |
  v
[Pass 1 on "P + Step 1"] -> Signals (entropy, norm, mass)
  |
  v
[Signals < threshold?] ----NO----> [Flag: Step 1 uncertain]
  |                                 [Options: retry, abort, verify]
  | YES
  v
[Generate Step 2]
  |
  v
[Pass 1 on "P + Step 1 + Step 2"] -> Signals
  |
  v
[Classify Step 2 type] ----counterfactual----> [Flag: risky path]
  |                                 |
  | other                           |
  v                                 v
[Signals < threshold?]             [Inject verification sub-question]
  |                                 |
  | NO                              |
  v                                 |
[Flag: Step 2 uncertain] <---------
  |
  | YES
  v
[Generate Step 3]
  ...
  |
  v
[After N steps: sample 3 trajectories]
  |
  v
[Compare for divergence]
  |
  v
[Divergence?] ----YES---> [Backtrack to last consensus step]
  |                        [Re-sample from there]
  | NO
  v
[Output final answer]
```

---

## Immediate Next Steps

### Before CoT PoC:
1. **Fix counterfactual edge cases** — model quality must improve or detector must compensate
2. **Validate length normalization** — current per-token norm approach may be wrong
3. **Run full edge case validation** — confirm ≥ 10/14 pass

### CoT PoC:
1. Select 5 test problems (math, logic, factual reasoning)
2. Implement per-step signal extraction
3. Implement incorrect-premise injection
4. Run statistical comparison (baseline vs corrupted)
5. Document results

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Context length bias dominates signals | High | Normalize by step position, not token count |
| Model too small for CoT (7B) | High | Use 13B+ or accept higher noise floor |
| Consistency check too slow (3x per step) | Medium | Cache trajectories, use cheaper similarity |
| Signals not statistically significant | Medium | Increase N (10+ problems), use paired tests |

---

## Open Questions

1. Should the uncertainty threshold be per-step or cumulative?
2. How far back should backtracking go? (1 step? To last consensus?)
3. Can the question-type classifier generalize to reasoning-step types (not just question types)?
4. Does the embedding distance signal work when the "known" reference is a reasoning step, not a factual question?
