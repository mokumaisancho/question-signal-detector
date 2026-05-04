# Next POC Paths: From Two-Pass to Production

## What the Two-Pass Detector Brings

**Core innovation**: Uncertainty detection happens BEFORE autoregressive generation starts.
This is not incremental improvement — it is a structural fix for the "terror of question"
problem where forced answering drowns out the model's own uncertainty signal.

**Current results (Llama-2-7B Q4_K_M)**:
- 83% accuracy on known/unknown discrimination
- 0% false abstention on known topics (all known questions answered)
- Pre-flight stability check prevents querying an unstable model

**What works**:
- Hidden norm is the strongest signal (known=68-73, unknown=50-58)
- Next-token entropy is inverted but consistent (known=1.8-3.4, unknown=0.6-0.8)
- Embedding distance to calibrated references catches most cross-domain questions

**What fails**:
- Mars Colony question slips through (embedding proximity to known space topics)
- Single threshold (0.5) is too coarse for edge cases

---

## Path A: Harden the Two-Pass Detector (1-2 weeks)

### A1. Multi-threshold Decision Boundary
Replace single threshold with domain-aware thresholds:
- Physics/chemistry: lower threshold (model is strong here)
- Frontier math/cs: higher threshold (model is weak here)
- Use the gridded knowledge map (8x4 domain/depth grid) to select threshold

### A2. Ensemble Pass 1 Signals
Current: 0.5*entropy + 0.3*norm - 0.2*embedding
Better: add a third signal — **token rank variance** across multiple temperature samples.
Run Pass 1 at T=0, T=0.5, T=1.0. If top token changes across temperatures,
uncertainty is high. This costs 3x minimal generation but catches edge cases.

### A3. Dynamic Calibration
Current calibration is static. Instead:
- After each answered question, update known_refs if answer was verified correct
- After each abstention, add to unknown_refs
- Track embedding drift over time; recalibrate when drift > threshold

### A4. Production Memory Guard
From topologicalinferencing patterns:
- Context manager for guaranteed unload: `with detector.session(): ...`
- Automatic unload after N seconds of inactivity
- Memory usage telemetry (RSS before/after load)
- Batch calibration with explicit `gc.collect()` between samples

**Effort**: 1-2 weeks. **Risk**: Low. **ROI**: High for immediate deployment.

---

## Path B: Gridded Knowledge Map Integration (2-3 weeks)

The detector currently uses a flat embedding space. The gridded map adds structure.

### B1. Domain/Depth Classifier
Before running the detector, classify the question into the 8x4 grid:
- Domain: embed question, cosine similarity to domain centroids
- Depth: rule-based from linguistic markers ("what is"=L0, "can X detect Y"=L3)

### B2. Per-Cell Coverage Scores
Each cell has a coverage score (0-1) from behavioral probing:
- High coverage (0.7+) → use detector with relaxed threshold
- Low coverage (0.3-) → abstain immediately, skip detector
- Medium coverage → run detector with strict threshold

### B3. Hole Detection
From topologicalinferencing: find cells where embedding density is high but
accuracy is low. These are "false confidence zones" — the model thinks it knows
but doesn't. Flag questions mapping to these cells as high-risk.

### B4. Cross-Model Consensus
Run Pass 1 on 2-3 models (Llama-2-7B, Gemma-2B, GPT-2-medium).
If all agree "known" → answer. If all agree "unknown" → abstain.
If split → run Pass 2 only on models that said known, return consensus answer.

**Effort**: 2-3 weeks. **Risk**: Medium (grid calibration requires ground truth).
**ROI**: Highest for explainability — the grid is inspectable.

---

## Path C: Active Parameter Detection (Research, 4-6 weeks)

The most principled long-term approach. Requires SAE (Sparse Autoencoder)
training infrastructure.

### C1. SAE Training on Llama-2-7B Activations
- Extract activations at layers 8-16 (mid-to-late) for 10k prompts
- Train SAE with 8k-32k features, L1 sparsity penalty
- This is compute-heavy: ~6-12 hours on M1 Max with 32GB

### C2. Feature Interpretation
- For each SAE feature, find top-activating examples
- Manually label features as "known-topic-X" or "uncertainty-marker"
- Hallucination phenomenology (Ruscio & Thompson 2026) suggests:
  uncertain inputs activate 2-3x more features = higher intrinsic dimensionality

### C3. Real-Time Feature Monitoring
During Pass 1, extract SAE features for the question.
- Count active features (L0 norm)
- Measure feature activation entropy
- High feature count + low activation entropy = uncertain

### C4. Causal Intervention
Patch specific uncertainty features and observe if output changes.
If disabling a feature makes the model abstain, that feature encodes
"knowledge confidence."

**Effort**: 4-6 weeks. **Risk**: High (SAE training may not converge;
feature interpretation is labor-intensive). **ROI**: Highest if it works —
this is the only path that truly escapes the black box.

---

## Path D: Rule-Based Replacement for GPT-2 (1 week)

User asked: "what features does gpt-2 handle instead of multiple focused features?"

GPT-2-medium handles these in one black box:
- Token prediction (next-token probability)
- Syntactic parsing (implicit in attention patterns)
- Semantic similarity (implicit in embedding space)
- Coherence maintenance (implicit in layer-to-layer transformations)

**Rule-based replacements**:
| GPT-2 Feature | Rule-Based Alternative | Accuracy Tradeoff |
|---|---|---|
| Token prediction | N-gram frequency table + backoff | -30% but deterministic |
| Syntactic parsing | SpaCy dependency parser | Equivalent, faster |
| Semantic similarity | TF-IDF + cosine similarity | -15% but explainable |
| Coherence | Template filling + slot constraints | -40% but no hallucination |

**Verdict**: Rule-based replacements lose significant accuracy but gain
explainability. For the [Qq]uestion Signal boundary task specifically, a hybrid is
best: use TF-IDF for domain classification (Path B) + GPT-2 only for
generating the actual answer when the boundary check passes.

---

## Recommended Sequencing

1. **Week 1**: Path A1-A2 (harden two-pass detector)
   - Multi-threshold + ensemble Pass 1
   - Target: 90%+ accuracy, eliminate Mars-type edge cases

2. **Week 2-3**: Path B1-B2 (gridded map integration)
   - Build domain/depth classifier
   - Connect coverage scores to detector thresholds
   - This gives you the "reference frame" you wanted

3. **Week 4+**: Path C (active parameters) as research
   - Validate on GPT-2-small first (faster iteration)
   - Only scale to Llama-2-7B if GPT-2 results are promising

4. **Continuous**: Path A4 (memory guard)
   - Port all topologicalinferencing patterns
   - Add telemetry and automatic resource management
