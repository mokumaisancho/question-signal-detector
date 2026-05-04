# [Qq]uestion Signal Boundary Detection: Three Approaches

## What We're Trying to Detect

**Does the model know the answer to this question internally?**

Not "is it in Wikipedia." Not "is it at the frontier of human knowledge."
Internal parametric knowledge — what the model's weights encode.

---

## Approach 1: Active Parameter Detection

### Concept

Identify which neurons/parameters activate for a given input. If well-trodden
pathways fire (many training examples hit these weights), the model "knows."
If sparse/weak pathways fire, the model doesn't know.

### Research Base

- **Gemma Scope** (Google DeepMind, 2024): Sparse Autoencoders (SAEs) on all
  layers of Gemma 2 (2B-27B). Decomposes polysemantic neurons into monosemantic
  features. Open-sourced. ArXiv: 2408.05147 (293 citations).

- **Circuit Tracing** (Anthropic, 2025): Cross-Layer Transcoders as "replacement
  models" to trace attribution graphs through the model. Open-sourced tools.
  transformer-circuits.pub

- **Hallucination Phenomenology** (Ruscio & Thompson, 2026): Models DO internally
  detect uncertainty — uncertain inputs have 2-3x higher intrinsic dimensionality.
  But this signal is weakly coupled to output. Cross-entropy training provides no
  attractor for abstention. ArXiv: 2603.13911

- **SAEs on GPT-2 Small** (Kissane et al., 2024): Validated that SAE methodology
  works at GPT-2 scale (124M params). Found causally meaningful features.
  ArXiv: 2406.17759

- **Safety Neurons** (Chen et al., 2024): Only ~5% of neurons control safety
  behavior. Patching just these restores 90%+ safety. Shows behavioral patterns
  ARE localized to small neuron subsets. ArXiv: 2406.14144

### How It Would Work

1. Train SAE on GPT-2-medium activations (or use pre-trained if available)
2. For each question, extract sparse feature activations at mid-to-late layers
3. Measure: are the activated features "known" (high density in training data)
   or "novel" (rarely activated, uncertain)?
4. The hallucination phenomenology paper suggests: measure intrinsic
   dimensionality of the activation neighborhood. Known = low-dim, unknown = high-dim.

### Feasibility

- **Validated at GPT-2 scale**: Yes. Kissane et al. demonstrated SAEs on GPT-2 Small.
- **Open tooling**: Gemma Scope provides training code. Anthropic provides circuit tracing.
- **Effort**: 2-3 weeks to train SAE on GPT-2-medium, build feature analysis pipeline.

### Risks

- SAE training requires significant compute (hours on GPU)
- Feature interpretation still requires human judgment
- The "active parameters" for knowledge might be distributed, not localized
- Doesn't fully escape the black-box problem — you're just making the box
  more transparent, not eliminating it

### Status: Most promising long-term direction, but heavy engineering.

---

## Approach 2: Dynamic Focus Detection (Attention Patterns)

### Concept

Monitor attention patterns during inference. Focused attention = model recognizes
the content = known. Diffuse attention = model is uncertain = unknown.

### Experimental Results

**First test (uncontrolled)**: Cohen's d > 3.5 on all metrics. Looked extremely
promising.

**Second test (length-controlled)**: Signal REVERSED. Cohen's d = -1.97.

| Metric | Known Mean | Unknown Mean | Gap | Direction |
|---|---|---|---|---|
| Entropy | 0.968 | 0.842 | -0.126 | WRONG |
| Sparsity | 0.868 | 0.899 | +0.032 | WRONG |
| Concentration | 0.731 | 0.769 | +0.038 | WRONG |

The original signal was entirely a sequence length confound. Unknown questions
were longer (8-14 tokens vs 4-7 tokens), which naturally produces higher attention
entropy. With length-matched questions, known questions actually have slightly
higher entropy.

### Why It Failed

1. **Sequence length dominates**: More tokens → more attention positions → higher
   entropy. This overwhelms any [Qq]uestion Signal signal.

2. **GPT-2-medium's attention is too uniform**: Adjacent layer cosine similarity
   is ~0.91 for ALL questions (we saw this in the hidden state test). The model
   doesn't differentially focus based on knowledge state.

3. **Causal attention structure**: In GPT-2's causal attention, each position
   attends to all previous positions. The attention distribution is dominated
   by token proximity, not content relevance.

### Could It Work With Larger Models?

Maybe. Larger models (7B+) have more structured attention patterns with clearer
specialization. But:
- The user said "LLMs do not provide detail on their logic" — skeptical of ML signals
- We'd need a GPU with 16GB+ VRAM
- The fundamental length confound persists regardless of model size

### Status: Dead end for GPT-2-medium. Unlikely to work without major model upgrade.

---

## Approach 3: Layered/Gridded Knowledge Map

### Concept

Map the model's knowledge space into a structured topology. Each cell has a
coverage score = how well the model knows this area. New questions are mapped
to cells, and coverage score determines known/unknown.

### Grid Design

**8 x 4 grid** (32 cells total):

| Axis 1: Domain (8) | Axis 2: Depth (4) |
|---|---|
| Physics | L0: Established facts |
| Computer Science | L1: Intermediate concepts |
| Mathematics | L2: Active research |
| Biology | L3: Frontier speculation |
| Chemistry | |
| Philosophy | |
| History/Social | |
| Other | |

Example: "What is CRISPR?" → (Biology, L0) → coverage 0.85 → KNOWN
"Can topological persistence detect phase transitions?" → (Mathematics, L3) → coverage 0.15 → UNKNOWN

### How Coverage Is Measured

**Method A — Behavioral probing (primary)**: For each cell, test the model with
10-15 questions that have known ground-truth answers. Coverage = accuracy on probes.

**Method B — Training data statistics (secondary)**: Estimate fraction of training
data in each cell from corpus statistics. Gives a prior.

**Method C — Embedding density (tertiary)**: Embed representative sentences per cell.
High intra-cluster similarity = coherent representation = strong knowledge.

### Mapping Questions to Grid Cells

1. **Domain**: Embed question, compare to reference embeddings per domain.
   Highest similarity wins. (Works because domain vocabulary is strongly differentiated.)

2. **Depth**: Extract linguistic markers:
   - "what is", "define" → L0-L1
   - "how does", "explain" → L1-L2
   - "recently", "latest" → L2-L3
   - "can X detect Y", "does X predict Y" → L3 (cross-domain causal)

### Bootstrapping

**Phase 1**: Initialize from training data priors (Week 1)
**Phase 2**: Automated behavioral probing — test model on probe questions,
  measure accuracy. Cross-validate by asking same question 5 different ways;
  high variance = low coverage (Week 2-3)
**Phase 3**: Human validation at ambiguous cells only (Week 4)

### Reuse from topologicalinferencing

| Component | Transferable? |
|---|---|
| Grid extraction pipeline | Yes — same embedding + clustering approach |
| Multi-model consensus | Yes — 3-5 models vote on cell membership |
| Transition matrix (Markov) | Yes — same math, different axis |
| Hole detection algorithm | Yes — cells where density is high but accuracy is low |
| Hole patching (two-hop bridge) | Yes — find intermediate cells for knowledge gaps |
| Navigator/classifier | Yes — adapt classify() for domain+depth |

### Key Insight

The grid is EXPLICIT and INSPECTABLE. You can see:
- Which cell a question maps to
- What the coverage score is for that cell
- Why the score is what it is (which probe questions failed)
- Where the holes are (high density but low accuracy = false confidence)

This addresses the user's concern about explainability — the knowledge boundary
is a visible structure, not a scalar from a black box.

### Risks

1. **Grid resolution**: 32 cells might be too coarse. Cross-domain questions
   ("Can physics method solve biology problem?") don't map cleanly to one cell.
   Mitigation: multi-cell mapping with confidence weighting.

2. **Coverage testing requires ground truth**: For L0-L1, easy (Wikipedia).
   For L2-L3, hard — who decides what's "active research" vs "frontier"?
   Mitigation: use publication date as proxy; papers from 2024-2025 = L2,
   preprints with no peer review = L3.

3. **Model dependency**: Coverage scores are model-specific. A new model
   requires re-probing all cells. Mitigation: this is a feature, not a bug —
   each model gets its own map.

4. **Static vs dynamic**: The grid is static once built. If the model is
   updated, the grid needs re-calibration. Mitigation: version the grid
   alongside the model.

### Status: Most tractable for POC. Builds on existing work. Explainable by design.

---

## Comparison

| Criterion | Active Parameters | Attention Focus | Gridded Map |
|---|---|---|---|
| Experimental validation | Literature yes, GPT-2 TBD | **Failed on GPT-2-medium** | Literature similar concepts |
| Explainability | Partial (SAE features) | None (black box) | **High (explicit grid)** |
| Implementation effort | 2-3 weeks (SAE training) | Done (failed) | 1-2 weeks |
| Model dependency | High (per-model SAE) | High | Medium (per-model coverage) |
| Escapes black box? | Partially | No | **Yes (behavioral testing)** |
| Handles cross-domain? | Unknown | N/A | Multi-cell mapping |
| Key risk | SAE training compute | Dead end | Grid resolution |

---

## Recommended Path Forward

1. **Gridded map** as the primary POC — explainable, tractable, builds on
   topologicalinferencing. 1-2 week implementation.

2. **Active parameters** as the research direction — most principled long-term
   approach, but requires SAE training infrastructure. Validate on GPT-2-small
   first (faster iteration).

3. **Attention focus** — abandoned. Signal was a length confound.
