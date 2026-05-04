# ULTRAPLAN2: Harden [Qq]uestion Signal Detector Against 14 Critical Edge Cases

Statistically sound [Qq]uestion Signal boundary detection for LLM parametric knowledge. Covering adversarial phrasing, partial knowledge, temporal shift, nonsense, ambiguity, meta-questions, multi-hop reasoning, counterfactuals, length extremes, cross-domain hybrids, known-unknown connections, niche vs general knowledge, session drift, and temperature sensitivity. Must investigate existing GitHub datasets and benchmarks before building custom test suites.

**Total estimated hours:** 104.0
**Tasks:** 52
**Papers consulted:** 8
**Knowledge gaps:** 0

## Critical Path

- **dataset-research-step-1** (2.0h) — Investigate existing GitHub datasets for [Qq]uestion Signal uncertainty, abstention, and adversarial robustness (step 1/2)
- **dataset-research-step-2** (2.0h) — Investigate existing GitHub datasets for [Qq]uestion Signal uncertainty, abstention, and adversarial robustness (step 2/2)
- **dataset-construct-step-1** (2.0h) — Build 400+ question dataset with 13 category labels, leveraging existing datasets where possible (step 1/4)
- **dataset-construct-step-2** (2.0h) — Build 400+ question dataset with 13 category labels, leveraging existing datasets where possible (step 2/4)
- **dataset-construct-step-3** (2.0h) — Build 400+ question dataset with 13 category labels, leveraging existing datasets where possible (step 3/4)
- **dataset-construct-step-4** (2.0h) — Build 400+ question dataset with 13 category labels, leveraging existing datasets where possible (step 4/4)
- **split-framework-step-1** (2.0h) — Implement train/calibration/test split with stratification by category (step 1/2)
- **split-framework-step-2** (2.0h) — Implement train/calibration/test split with stratification by category (step 2/2)
- **cross-validation-step-1** (2.0h) — Implement 5-fold cross-validation with per-fold threshold tuning (step 1/3)
- **cross-validation-step-2** (2.0h) — Implement 5-fold cross-validation with per-fold threshold tuning (step 2/3)
- **cross-validation-step-3** (2.0h) — Implement 5-fold cross-validation with per-fold threshold tuning (step 3/3)
- **edge-adversarial-step-1** (2.0h) — Edge Case 1: Adversarial phrasing — systematic paraphrase generation and robustness measurement (step 1/2)
- **edge-adversarial-step-2** (2.0h) — Edge Case 1: Adversarial phrasing — systematic paraphrase generation and robustness measurement (step 2/2)
- **edge-partial-step-1** (2.0h) — Edge Case 2: Partial knowledge — measure detector on questions with known first half, unknown second half (step 1/2)
- **edge-partial-step-2** (2.0h) — Edge Case 2: Partial knowledge — measure detector on questions with known first half, unknown second half (step 2/2)
- **edge-temporal-step-1** (2.0h) — Edge Case 3: Temporal shift — include questions with known cutoff dates, future events, stale facts (step 1/2)
- **edge-temporal-step-2** (2.0h) — Edge Case 3: Temporal shift — include questions with known cutoff dates, future events, stale facts (step 2/2)
- **edge-nonsense-step-1** (2.0h) — Edge Case 4: Nonsense questions — syntactically valid but semantically empty (step 1/2)
- **edge-nonsense-step-2** (2.0h) — Edge Case 4: Nonsense questions — syntactically valid but semantically empty (step 2/2)
- **edge-ambiguous-step-1** (2.0h) — Edge Case 5: Ambiguous/subjective questions — multiple valid interpretations (step 1/2)
- **edge-ambiguous-step-2** (2.0h) — Edge Case 5: Ambiguous/subjective questions — multiple valid interpretations (step 2/2)
- **edge-meta-step-1** (2.0h) — Edge Case 6: Meta-questions — questions about the model itself (step 1/2)
- **edge-meta-step-2** (2.0h) — Edge Case 6: Meta-questions — questions about the model itself (step 2/2)
- **edge-multihop-step-1** (2.0h) — Edge Case 7: Multi-hop reasoning — questions requiring multiple inference steps (step 1/2)
- **edge-multihop-step-2** (2.0h) — Edge Case 7: Multi-hop reasoning — questions requiring multiple inference steps (step 2/2)
- **edge-counterfactual-step-1** (2.0h) — Edge Case 8: Counterfactuals — hypothetical scenarios (step 1/2)
- **edge-counterfactual-step-2** (2.0h) — Edge Case 8: Counterfactuals — hypothetical scenarios (step 2/2)
- **edge-length-step-1** (2.0h) — Edge Case 9: Length extremes — questions from 1 token to 100+ tokens (step 1/2)
- **edge-length-step-2** (2.0h) — Edge Case 9: Length extremes — questions from 1 token to 100+ tokens (step 2/2)
- **edge-crossdomain-step-1** (2.0h) — Edge Case 10: Cross-domain hybrids — questions mixing 2+ domains (step 1/2)
- **edge-crossdomain-step-2** (2.0h) — Edge Case 10: Cross-domain hybrids — questions mixing 2+ domains (step 2/2)
- **edge-knownunknown-step-1** (2.0h) — Edge Case 11: Known components with unknown connection — model knows A and B but not A->B (step 1/2)
- **edge-knownunknown-step-2** (2.0h) — Edge Case 11: Known components with unknown connection — model knows A and B but not A->B (step 2/2)
- **edge-niche-step-1** (2.0h) — Edge Case 12: Niche vs general knowledge — general vs specialized vs obscure facts (step 1/2)
- **edge-niche-step-2** (2.0h) — Edge Case 12: Niche vs general knowledge — general vs specialized vs obscure facts (step 2/2)
- **edge-drift-step-1** (2.0h) — Edge Case 13: Model instability during session — run 200+ questions sequentially and measure drift (step 1/2)
- **edge-drift-step-2** (2.0h) — Edge Case 13: Model instability during session — run 200+ questions sequentially and measure drift (step 2/2)
- **edge-temperature-step-1** (2.0h) — Edge Case 14: Temperature sensitivity — measure signal at T=0, 0.5, 1.0 (step 1/2)
- **edge-temperature-step-2** (2.0h) — Edge Case 14: Temperature sensitivity — measure signal at T=0, 0.5, 1.0 (step 2/2)
- **multi-format-ensemble-step-1** (2.0h) — Implement multi-format ensemble: WH / imperative / statement variants with variance as signal (step 1/2)
- **multi-format-ensemble-step-2** (2.0h) — Implement multi-format ensemble: WH / imperative / statement variants with variance as signal (step 2/2)
- **per-language-calibration-step-1** (2.0h) — Per-language calibration for Spanish, French, German, Chinese, Japanese (step 1/2)
- **per-language-calibration-step-2** (2.0h) — Per-language calibration for Spanish, French, German, Chinese, Japanese (step 2/2)
- **harness-integration-step-1** (2.0h) — Wire all edge case handlers, multi-format ensemble, per-language calibration into [Qq]uestion SignalForecastingHarness (step 1/2)
- **harness-integration-step-2** (2.0h) — Wire all edge case handlers, multi-format ensemble, per-language calibration into [Qq]uestion SignalForecastingHarness (step 2/2)
- **statistical-reporting-step-1** (2.0h) — Confidence intervals, per-class FPR/FNR, calibration curves, ECE (step 1/2)
- **statistical-reporting-step-2** (2.0h) — Confidence intervals, per-class FPR/FNR, calibration curves, ECE (step 2/2)
- **final-validation-step-1** (2.0h) — End-to-end validation: full pipeline on held-out test set with all edge cases (step 1/2)
- **final-validation-step-2** (2.0h) — End-to-end validation: full pipeline on held-out test set with all edge cases (step 2/2)

## Parallel Groups

### Wave 1
- dataset-research-step-1 (2.0h, MVP-0) — Investigate existing GitHub datasets for [Qq]uestion Signal uncertainty, abstention, and adversarial robustness (step 1/2)
  - **Agent:** Agent-1 (Dataset/Research)
  - **Acceptance:** Document at least 5 usable datasets with license, size, category coverage, and integration effort
### Wave 2
- dataset-research-step-2 (2.0h, MVP-0) — Investigate existing GitHub datasets for [Qq]uestion Signal uncertainty, abstention, and adversarial robustness (step 2/2)
  - **Agent:** Agent-1 (Dataset/Research)
  - **Acceptance:** Final dataset selection report with recommended adoption vs build-new decisions per category
### Wave 3
- dataset-construct-step-1 (2.0h, MVP-0) — Build 400+ question dataset with 13 category labels, leveraging existing datasets where possible (step 1/4)
  - **Agent:** Agent-1 (Dataset/Research)
  - **Acceptance:** 100+ questions created/imported with category labels, JSON schema validated
- per-language-calibration-step-1 (2.0h, MVP-1) — Per-language calibration for Spanish, French, German, Chinese, Japanese (step 1/2)
  - **Agent:** Agent-2 (Core Detector)
  - **Acceptance:** Translation pipeline working, 20 known + 20 unknown questions per language, KL divergence measured vs English
### Wave 4
- dataset-construct-step-2 (2.0h, MVP-0) — Build 400+ question dataset with 13 category labels, leveraging existing datasets where possible (step 2/4)
  - **Agent:** Agent-1 (Dataset/Research)
  - **Acceptance:** 200+ questions created/imported, category balance verified (min 20 per edge case category)
- per-language-calibration-step-2 (2.0h, MVP-1) — Per-language calibration for Spanish, French, German, Chinese, Japanese (step 2/2)
  - **Agent:** Agent-2 (Core Detector)
  - **Acceptance:** Per-language threshold files generated, cross-language variance report documented
### Wave 5
- dataset-construct-step-3 (2.0h, MVP-0) — Build 400+ question dataset with 13 category labels, leveraging existing datasets where possible (step 3/4)
  - **Agent:** Agent-1 (Dataset/Research)
  - **Acceptance:** 300+ questions created/imported, adversarial variants generated (5 per test question)
- split-framework-step-1 (2.0h, MVP-0) — Implement train/calibration/test split with stratification by category (step 1/2)
  - **Agent:** Agent-2 (Core Detector)
  - **Acceptance:** Split function produces stratified splits, category distribution preserved within 5% tolerance
### Wave 6
- dataset-construct-step-4 (2.0h, MVP-0) — Build 400+ question dataset with 13 category labels, leveraging existing datasets where possible (step 4/4)
  - **Agent:** Agent-1 (Dataset/Research)
  - **Acceptance:** 425+ questions finalized, all categories have >=20 questions, dataset manifest complete
- split-framework-step-2 (2.0h, MVP-0) — Implement train/calibration/test split with stratification by category (step 2/2)
  - **Agent:** Agent-2 (Core Detector)
  - **Acceptance:** Data leakage audit passed — zero overlap between splits, question hashes verified unique
### Wave 7
- cross-validation-step-1 (2.0h, MVP-1) — Implement 5-fold cross-validation with per-fold threshold tuning (step 1/3)
  - **Agent:** Agent-2 (Core Detector)
  - **Acceptance:** CV framework runs without error, produces accuracy per fold, threshold per fold
- multi-format-ensemble-step-1 (2.0h, MVP-1) — Implement multi-format ensemble: WH / imperative / statement variants with variance as signal (step 1/2)
  - **Agent:** Agent-2 (Core Detector)
  - **Acceptance:** Format generator produces 3 variants per question, detector runs all variants, variance computed
### Wave 8
- cross-validation-step-2 (2.0h, MVP-1) — Implement 5-fold cross-validation with per-fold threshold tuning (step 2/3)
  - **Agent:** Agent-2 (Core Detector)
  - **Acceptance:** Mean accuracy and std dev across 5 folds reported, threshold stability verified (std < 0.1)
- multi-format-ensemble-step-2 (2.0h, MVP-1) — Implement multi-format ensemble: WH / imperative / statement variants with variance as signal (step 2/2)
  - **Agent:** Agent-2 (Core Detector)
  - **Acceptance:** Format variance discriminates known (CV < 20%) from unknown (CV > 40%) on calibration set
- edge-adversarial-step-1 (2.0h, MVP-2) — Edge Case 1: Adversarial phrasing — systematic paraphrase generation and robustness measurement (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** 5 adversarial variants per test question generated (format flip, negation, synonym, length, domain prefix)
### Wave 9
- cross-validation-step-3 (2.0h, MVP-1) — Implement 5-fold cross-validation with per-fold threshold tuning (step 3/3)
  - **Agent:** Agent-2 (Core Detector)
  - **Acceptance:** 95% CI on accuracy computed and reported, CI width < 0.10
- edge-adversarial-step-2 (2.0h, MVP-2) — Edge Case 1: Adversarial phrasing — systematic paraphrase generation and robustness measurement (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Robustness score >= 80% (fraction of variants preserving correct classification)
- edge-partial-step-1 (2.0h, MVP-2) — Edge Case 2: Partial knowledge — measure detector on questions with known first half, unknown second half (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** 20+ partial-knowledge questions created, detector classifications recorded
### Wave 10
- edge-partial-step-2 (2.0h, MVP-2) — Edge Case 2: Partial knowledge — measure detector on questions with known first half, unknown second half (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Partial knowledge classified as "known" in > 50% of cases (documented as expected failure)
- edge-temporal-step-1 (2.0h, MVP-2) — Edge Case 3: Temporal shift — include questions with known cutoff dates, future events, stale facts (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** 20+ temporal questions created (pre-cutoff, post-cutoff, future, "current" at training time)
- edge-nonsense-step-1 (2.0h, MVP-2) — Edge Case 4: Nonsense questions — syntactically valid but semantically empty (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** 20+ nonsense questions created, detector classifications recorded
### Wave 11
- edge-temporal-step-2 (2.0h, MVP-2) — Edge Case 3: Temporal shift — include questions with known cutoff dates, future events, stale facts (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Temporal classification accuracy >= 70% on held-out temporal questions
- edge-nonsense-step-2 (2.0h, MVP-2) — Edge Case 4: Nonsense questions — syntactically valid but semantically empty (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Nonsense questions classified as "unknown" in >= 80% of cases
- edge-ambiguous-step-1 (2.0h, MVP-2) — Edge Case 5: Ambiguous/subjective questions — multiple valid interpretations (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** 20+ ambiguous/subjective questions created
- edge-meta-step-1 (2.0h, MVP-2) — Edge Case 6: Meta-questions — questions about the model itself (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** 20+ meta-questions created, detector classifications recorded
### Wave 12
- edge-ambiguous-step-2 (2.0h, MVP-2) — Edge Case 5: Ambiguous/subjective questions — multiple valid interpretations (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Ambiguous questions show higher format variance than factual questions (CV difference >= 10%)
- edge-meta-step-2 (2.0h, MVP-2) — Edge Case 6: Meta-questions — questions about the model itself (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Meta-questions classified consistently (all "known" or all "uncertain") with explanation
- edge-multihop-step-1 (2.0h, MVP-2) — Edge Case 7: Multi-hop reasoning — questions requiring multiple inference steps (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** 20+ multi-hop questions created with verified answer chains
- edge-counterfactual-step-1 (2.0h, MVP-2) — Edge Case 8: Counterfactuals — hypothetical scenarios (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** 20+ counterfactual questions created
### Wave 13
- edge-multihop-step-2 (2.0h, MVP-2) — Edge Case 7: Multi-hop reasoning — questions requiring multiple inference steps (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Multi-hop Pass 1 entropy does not mislead (correlation with correctness >= 0.5)
- edge-counterfactual-step-2 (2.0h, MVP-2) — Edge Case 8: Counterfactuals — hypothetical scenarios (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Counterfactuals classified as "known" (model knows physics) in >= 60% of cases
- edge-length-step-1 (2.0h, MVP-2) — Edge Case 9: Length extremes — questions from 1 token to 100+ tokens (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Length sweep from 1 to 100+ tokens executed, embedding norm and entropy recorded
- edge-crossdomain-step-1 (2.0h, MVP-2) — Edge Case 10: Cross-domain hybrids — questions mixing 2+ domains (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** 20+ cross-domain hybrid questions created
### Wave 14
- edge-length-step-2 (2.0h, MVP-2) — Edge Case 9: Length extremes — questions from 1 token to 100+ tokens (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** No length bias detected (accuracy correlation with length |r| < 0.3)
- edge-crossdomain-step-2 (2.0h, MVP-2) — Edge Case 10: Cross-domain hybrids — questions mixing 2+ domains (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Cross-domain hybrids show higher format variance than in-domain questions (CV difference >= 15%)
- edge-knownunknown-step-1 (2.0h, MVP-2) — Edge Case 11: Known components with unknown connection — model knows A and B but not A->B (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** 20+ known-unknown-connection questions created
- edge-niche-step-1 (2.0h, MVP-2) — Edge Case 12: Niche vs general knowledge — general vs specialized vs obscure facts (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** 20+ questions across general/niche/obscure spectrum created
### Wave 15
- edge-knownunknown-step-2 (2.0h, MVP-2) — Edge Case 11: Known components with unknown connection — model knows A and B but not A->B (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Known-unknown-connection questions classified as "unknown" with "in_domain" status (CV < 20%)
- edge-niche-step-2 (2.0h, MVP-2) — Edge Case 12: Niche vs general knowledge — general vs specialized vs obscure facts (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Niche known facts not misclassified as unknown (accuracy on niche >= 60%)
- edge-drift-step-1 (2.0h, MVP-2) — Edge Case 13: Model instability during session — run 200+ questions sequentially and measure drift (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** 200+ question sequential run executed, baseline entropy and norm tracked per 50-question block
- edge-temperature-step-1 (2.0h, MVP-2) — Edge Case 14: Temperature sensitivity — measure signal at T=0, 0.5, 1.0 (step 1/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Temperature sweep executed at T=0, 0.5, 1.0, entropy and norm signals recorded
### Wave 16
- edge-drift-step-2 (2.0h, MVP-2) — Edge Case 13: Model instability during session — run 200+ questions sequentially and measure drift (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Drift within acceptable bounds (entropy shift < 0.5 SD, norm shift < 0.3 SD over full session)
- edge-temperature-step-2 (2.0h, MVP-2) — Edge Case 14: Temperature sensitivity — measure signal at T=0, 0.5, 1.0 (step 2/2)
  - **Agent:** Agent-3 (Edge Case Testing)
  - **Acceptance:** Temperature compensation formula derived: adjusted_entropy = raw_entropy - k * T, where k calibrated from data
- harness-integration-step-1 (2.0h, MVP-1) — Wire all edge case handlers, multi-format ensemble, per-language calibration into [Qq]uestion SignalForecastingHarness (step 1/2)
  - **Agent:** Agent-4 (Integration/Validation)
  - **Acceptance:** Harness class compiles, all 14 edge case handlers registered, multi-format ensemble wired, per-language calibration loaded
### Wave 17
- harness-integration-step-2 (2.0h, MVP-1) — Wire all edge case handlers, multi-format ensemble, per-language calibration into [Qq]uestion SignalForecastingHarness (step 2/2)
  - **Agent:** Agent-4 (Integration/Validation)
  - **Acceptance:** End-to-end harness test passes on 10 sample questions (2 per category), output schema validated
- statistical-reporting-step-1 (2.0h, MVP-1) — Confidence intervals, per-class FPR/FNR, calibration curves, ECE (step 1/2)
  - **Agent:** Agent-4 (Integration/Validation)
  - **Acceptance:** Statistical report module produces CI, FPR/FNR per class, calibration curve PNG, ECE score
### Wave 18
- statistical-reporting-step-2 (2.0h, MVP-1) — Confidence intervals, per-class FPR/FNR, calibration curves, ECE (step 2/2)
  - **Agent:** Agent-4 (Integration/Validation)
  - **Acceptance:** All reports generated from test set output, 95% CI width < 0.10, ECE < 0.15
- final-validation-step-1 (2.0h, MVP-2) — End-to-end validation: full pipeline on held-out test set with all edge cases (step 1/2)
  - **Agent:** Agent-4 (Integration/Validation)
  - **Acceptance:** Full pipeline runs on 125 held-out test questions without error, all categories represented
### Wave 19
- final-validation-step-2 (2.0h, MVP-2) — End-to-end validation: full pipeline on held-out test set with all edge cases (step 2/2)
  - **Agent:** Agent-4 (Integration/Validation)
  - **Acceptance:** Overall accuracy >= 80%, no category has accuracy < 60%, all acceptance criteria from edge cases satisfied

## Research Annotations

### dataset-research-step-1 [A] adopt — yinzhangyue/SelfAware
Dataset of 1,032 unanswerable questions + 2,337 answerable questions. Directly usable for Edge Case 4 (nonsense) and Edge Case 2 (partial knowledge). MIT license. Integration effort: low — JSON format, question/answer pairs.

### dataset-research-step-1 [A] adopt — sylinrl/TruthfulQA
817 questions across 38 categories testing hallucination and factuality. Usable for Edge Case 5 (ambiguous/subjective) and Edge Case 12 (niche vs general). Apache-2.0 license. Integration effort: medium — needs category mapping to our schema.

### dataset-research-step-1 [A] adopt — facebook/AbstentionBench
20 datasets for abstention evaluation with known/unknown labels. Directly usable for train/calibration/test splits and baseline comparison. Unknown license (Meta). Integration effort: low — structured format with abstention labels.

### dataset-research-step-1 [A] adopt — Microsoft/promptbench
Unified evaluation framework with adversarial prompt generation. Usable for Edge Case 1 (adversarial phrasing). MIT license. Integration effort: medium — needs adapter to our detector API.

### dataset-research-step-1 [A] adopt — caisa-lab/llm-QA-robustness
Linguistic style adversarial paraphrase robustness dataset. Directly usable for Edge Case 1 (adversarial phrasing). MIT license. Integration effort: low — paraphrase pairs in JSON.

### dataset-research-step-1 [R] reference — jxzhangyhu/Awesome-LLM-Uncertainty-Reliability-Robustness
Paper collection, not a dataset. Reference for methodology and metric selection. No integration needed.

### dataset-research-step-1 [R] reference — MiaoXiong2320/llm-uncertainty
ICLR 2024 paper code. Reference for entropy-based uncertainty quantification. No direct integration.

### dataset-research-step-1 [A] adopt — tanyang3/llm-benchmark
Prompt decomposition and variance analysis. Usable for Edge Case 1 (adversarial) and multi-format ensemble validation. MIT license. Integration effort: medium.

### dataset-research-step-2 [N] novel
No existing dataset covers all 14 edge cases simultaneously. Custom dataset construction required for: Edge Case 3 (temporal shift), Edge Case 6 (meta-questions), Edge Case 7 (multi-hop), Edge Case 8 (counterfactuals), Edge Case 9 (length extremes), Edge Case 10 (cross-domain hybrids), Edge Case 11 (known-unknown connection), Edge Case 13 (session drift), Edge Case 14 (temperature sensitivity).

### edge-adversarial-step-1 [A] adopt — Microsoft/promptbench
Use promptbench's adversarial prompt generation (character-level, word-level, sentence-level) as baseline. Extend with our format-specific variants (WH->statement, negation flip, domain prefix).

### edge-adversarial-step-1 [A] adopt — caisa-lab/llm-QA-robustness
Use their linguistic paraphrase pairs for synonym swap and length manipulation baselines.

### multi-format-ensemble-step-1 [N] novel
No existing implementation combines format variance as a detector signal. Our approach is novel: use variance across WH/imperative/statement as a fifth signal (after entropy, norm, truncation, embedding distance).

### per-language-calibration-step-1 [N] novel
No existing [Qq]uestion Signal detector calibrates per-language. Cross-language variance findings (KL 2.47-3.05 for CJK) make this a novel requirement.

## All Tasks

| Task | Hours | Tier | Novelty | Dependencies | Agent |
|------|-------|------|---------|--------------|-------|
| dataset-research-step-1 | 2.0 | MVP-0 | adopt | - | Agent-1 |
| dataset-research-step-2 | 2.0 | MVP-0 | adopt | dataset-research-step-1 | Agent-1 |
| dataset-construct-step-1 | 2.0 | MVP-0 | novel | dataset-research-step-2 | Agent-1 |
| dataset-construct-step-2 | 2.0 | MVP-0 | novel | dataset-construct-step-1 | Agent-1 |
| dataset-construct-step-3 | 2.0 | MVP-0 | novel | dataset-construct-step-2 | Agent-1 |
| dataset-construct-step-4 | 2.0 | MVP-0 | novel | dataset-construct-step-3 | Agent-1 |
| split-framework-step-1 | 2.0 | MVP-0 | novel | dataset-construct-step-4 | Agent-2 |
| split-framework-step-2 | 2.0 | MVP-0 | novel | split-framework-step-1 | Agent-2 |
| cross-validation-step-1 | 2.0 | MVP-1 | novel | split-framework-step-2 | Agent-2 |
| cross-validation-step-2 | 2.0 | MVP-1 | novel | cross-validation-step-1 | Agent-2 |
| cross-validation-step-3 | 2.0 | MVP-1 | novel | cross-validation-step-2 | Agent-2 |
| edge-adversarial-step-1 | 2.0 | MVP-2 | adopt | cross-validation-step-3 | Agent-3 |
| edge-adversarial-step-2 | 2.0 | MVP-2 | novel | edge-adversarial-step-1 | Agent-3 |
| edge-partial-step-1 | 2.0 | MVP-2 | adopt | cross-validation-step-3 | Agent-3 |
| edge-partial-step-2 | 2.0 | MVP-2 | novel | edge-partial-step-1 | Agent-3 |
| edge-temporal-step-1 | 2.0 | MVP-2 | novel | cross-validation-step-3 | Agent-3 |
| edge-temporal-step-2 | 2.0 | MVP-2 | novel | edge-temporal-step-1 | Agent-3 |
| edge-nonsense-step-1 | 2.0 | MVP-2 | adopt | cross-validation-step-3 | Agent-3 |
| edge-nonsense-step-2 | 2.0 | MVP-2 | novel | edge-nonsense-step-1 | Agent-3 |
| edge-ambiguous-step-1 | 2.0 | MVP-2 | adopt | cross-validation-step-3 | Agent-3 |
| edge-ambiguous-step-2 | 2.0 | MVP-2 | novel | edge-ambiguous-step-1 | Agent-3 |
| edge-meta-step-1 | 2.0 | MVP-2 | novel | cross-validation-step-3 | Agent-3 |
| edge-meta-step-2 | 2.0 | MVP-2 | novel | edge-meta-step-1 | Agent-3 |
| edge-multihop-step-1 | 2.0 | MVP-2 | novel | cross-validation-step-3 | Agent-3 |
| edge-multihop-step-2 | 2.0 | MVP-2 | novel | edge-multihop-step-1 | Agent-3 |
| edge-counterfactual-step-1 | 2.0 | MVP-2 | novel | cross-validation-step-3 | Agent-3 |
| edge-counterfactual-step-2 | 2.0 | MVP-2 | novel | edge-counterfactual-step-1 | Agent-3 |
| edge-length-step-1 | 2.0 | MVP-2 | novel | cross-validation-step-3 | Agent-3 |
| edge-length-step-2 | 2.0 | MVP-2 | novel | edge-length-step-1 | Agent-3 |
| edge-crossdomain-step-1 | 2.0 | MVP-2 | novel | cross-validation-step-3 | Agent-3 |
| edge-crossdomain-step-2 | 2.0 | MVP-2 | novel | edge-crossdomain-step-1 | Agent-3 |
| edge-knownunknown-step-1 | 2.0 | MVP-2 | novel | cross-validation-step-3 | Agent-3 |
| edge-knownunknown-step-2 | 2.0 | MVP-2 | novel | edge-knownunknown-step-1 | Agent-3 |
| edge-niche-step-1 | 2.0 | MVP-2 | adopt | cross-validation-step-3 | Agent-3 |
| edge-niche-step-2 | 2.0 | MVP-2 | novel | edge-niche-step-1 | Agent-3 |
| edge-drift-step-1 | 2.0 | MVP-2 | novel | cross-validation-step-3 | Agent-3 |
| edge-drift-step-2 | 2.0 | MVP-2 | novel | edge-drift-step-1 | Agent-3 |
| edge-temperature-step-1 | 2.0 | MVP-2 | novel | cross-validation-step-3 | Agent-3 |
| edge-temperature-step-2 | 2.0 | MVP-2 | novel | edge-temperature-step-1 | Agent-3 |
| multi-format-ensemble-step-1 | 2.0 | MVP-1 | novel | cross-validation-step-3 | Agent-2 |
| multi-format-ensemble-step-2 | 2.0 | MVP-1 | novel | multi-format-ensemble-step-1 | Agent-2 |
| per-language-calibration-step-1 | 2.0 | MVP-1 | novel | dataset-construct-step-4 | Agent-2 |
| per-language-calibration-step-2 | 2.0 | MVP-1 | novel | per-language-calibration-step-1 | Agent-2 |
| harness-integration-step-1 | 2.0 | MVP-1 | novel | multi-format-ensemble-step-2, per-language-calibration-step-2 | Agent-4 |
| harness-integration-step-2 | 2.0 | MVP-1 | novel | harness-integration-step-1 | Agent-4 |
| statistical-reporting-step-1 | 2.0 | MVP-1 | novel | harness-integration-step-2 | Agent-4 |
| statistical-reporting-step-2 | 2.0 | MVP-1 | novel | statistical-reporting-step-1 | Agent-4 |
| final-validation-step-1 | 2.0 | MVP-2 | novel | statistical-reporting-step-2, all edge-*-step-2 | Agent-4 |
| final-validation-step-2 | 2.0 | MVP-2 | novel | final-validation-step-1 | Agent-4 |

## Dependency Graph (No Contradictions)

```
MVP-0 Infrastructure Layer:
  dataset-research-step-1
    -> dataset-research-step-2
      -> dataset-construct-step-1 -> dataset-construct-step-2 -> dataset-construct-step-3 -> dataset-construct-step-4
        -> split-framework-step-1 -> split-framework-step-2
          -> cross-validation-step-1 -> cross-validation-step-2 -> cross-validation-step-3
        -> per-language-calibration-step-1 -> per-language-calibration-step-2

MVP-1 Core Hardening Layer (branches from cross-validation-step-3):
  cross-validation-step-3
    -> multi-format-ensemble-step-1 -> multi-format-ensemble-step-2
    -> harness-integration-step-1 -> harness-integration-step-2
      -> statistical-reporting-step-1 -> statistical-reporting-step-2
        -> final-validation-step-1 -> final-validation-step-2

MVP-2 Edge Case Layer (all branch from cross-validation-step-3, converge at final-validation):
  cross-validation-step-3
    -> edge-adversarial-step-1 -> edge-adversarial-step-2
    -> edge-partial-step-1 -> edge-partial-step-2
    -> edge-temporal-step-1 -> edge-temporal-step-2
    -> edge-nonsense-step-1 -> edge-nonsense-step-2
    -> edge-ambiguous-step-1 -> edge-ambiguous-step-2
    -> edge-meta-step-1 -> edge-meta-step-2
    -> edge-multihop-step-1 -> edge-multihop-step-2
    -> edge-counterfactual-step-1 -> edge-counterfactual-step-2
    -> edge-length-step-1 -> edge-length-step-2
    -> edge-crossdomain-step-1 -> edge-crossdomain-step-2
    -> edge-knownunknown-step-1 -> edge-knownunknown-step-2
    -> edge-niche-step-1 -> edge-niche-step-2
    -> edge-drift-step-1 -> edge-drift-step-2
    -> edge-temperature-step-1 -> edge-temperature-step-2

  All edge-*-step-2 + statistical-reporting-step-2 -> final-validation-step-1
```

**Dependency assessment:** No contradictions. MVP-0 layer (dataset, split, CV) is strictly sequential and must complete before any MVP-1 or MVP-2 work. MVP-1 and MVP-2 are parallel branches from cross-validation-step-3. Edge cases 1-14 are all independent after CV framework is ready — they can run in any order or in parallel. Final validation is a single convergence point requiring ALL prior work.

## Agent Assignments

| Agent | Role | Skills | Tasks |
|-------|------|--------|-------|
| **Agent-1** | Dataset/Research | Web search, data cleaning, JSON schema design, dataset integration | dataset-research-step-1/2, dataset-construct-step-1/2/3/4 |
| **Agent-2** | Core Detector | Python, numpy, statistical inference, cross-validation, llama-cpp | split-framework-step-1/2, cross-validation-step-1/2/3, multi-format-ensemble-step-1/2, per-language-calibration-step-1/2 |
| **Agent-3** | Edge Case Testing | Python, pytest, experimental design, metric computation, llama-cpp | edge-adversarial-step-1/2 through edge-temperature-step-1/2 (all 14 edge cases) |
| **Agent-4** | Integration/Validation | Python, system integration, CI/CD, reporting, end-to-end testing | harness-integration-step-1/2, statistical-reporting-step-1/2, final-validation-step-1/2 |

## MVP Tier Definitions

| Tier | Definition | Deliverables | Exit Criteria |
|------|-----------|--------------|---------------|
| **MVP-0** | Infrastructure | 425-question dataset, train/calibration/test split, 5-fold CV framework | Dataset manifest complete, split preserves category balance, CV runs without error |
| **MVP-1** | Core detector hardening | Multi-format ensemble, per-language calibration, statistical reporting, harness integration | Combined score formula includes format stability penalty, per-language thresholds active, 95% CI < 0.10 |
| **MVP-2** | Full edge case coverage | All 14 edge cases tested with acceptance criteria, final validation on held-out set | Each edge case has >=20 questions, all acceptance criteria satisfied, overall accuracy >= 80% |

## GitHub Repositories (Adopt vs Reference vs Novel)

### Adopt (direct integration)

1. **yinzhangyue/SelfAware** — 1,032 unanswerable + 2,337 answerable questions. Use for Edge Case 4 (nonsense) and Edge Case 2 (partial knowledge). MIT license.
2. **sylinrl/TruthfulQA** — 817 questions across 38 categories. Use for Edge Case 5 (ambiguous) and Edge Case 12 (niche vs general). Apache-2.0.
3. **facebook/AbstentionBench** — 20 datasets with abstention labels. Use for train/calibration/test baseline and known/unknown labeling. Meta license.
4. **Microsoft/promptbench** — Adversarial prompt generation. Use for Edge Case 1 (adversarial phrasing). MIT license.
5. **caisa-lab/llm-QA-robustness** — Linguistic paraphrase pairs. Use for Edge Case 1 (adversarial). MIT license.
6. **tanyang3/llm-benchmark** — Prompt decomposition and variance. Use for multi-format ensemble validation. MIT license.

### Reference (methodology only)

7. **jxzhangyhu/Awesome-LLM-Uncertainty-Reliability-Robustness** — Paper collection. Reference for metric selection and methodology.
8. **MiaoXiong2320/llm-uncertainty** — ICLR 2024 code. Reference for entropy-based uncertainty quantification approach.

### Novel (no existing repo covers this)

- Multi-format ensemble with variance-as-signal
- Per-language calibration for [Qq]uestion Signal boundary detection
- Temperature sensitivity compensation formula
- Session drift measurement protocol
- Cross-domain hybrid question generation
- Known-components-unknown-connection question generation
