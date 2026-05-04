# Question Signal Detector — Pre-Generation Uncertainty Detection for Local LLMs

A two-pass architecture that detects when a local LLM **does not know** an answer *before* generating one. By measuring entropy, embedding distance, and hidden-state norms on a minimal forward pass, the system decides whether to generate, abstain, or route to a different strategy — eliminating hallucinations at the source rather than filtering them post-hoc.

## Why This Matters

Current LLM pipelines generate first, then try to detect hallucinations after the fact. This is backwards. A model's internal state *before* it starts generating contains a more honest uncertainty signal than anything it produces afterward. This project exploits that insight:

- **"Think step by step"** is a manual attempt to route reasoning — this system **automatically routes** based on detected question type
- **Chain-of-thought** prompts try to improve self-assessment — this system measures entropy/embeddings **before generation starts**
- **Few-shot examples** are hand-picked references — this system **calibrates automatically** against known/unknown embedding prototypes
- **Confidence ratings** ("rate 1-10") are post-hoc rationalization — this system reads the model's **internal state** pre-generation

## Architecture

```
                         ┌─────────────────────┐
                         │   Input Question     │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Question-Type       │
                         │  Classifier          │
                         │  (factual/subjective │
                         │   nonsense/meta/     │
                         │   counterfactual/    │
                         │   ambiguous)         │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            ┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
            │  Subjective  │ │  Nonsense  │ │  Factual/  │
            │  → Abstain   │ │  → Coherence│ │  Meta/     │
            │  (always     │ │     Check   │ │  Counter-  │
            │  uncertain)  │ │  → Reject   │ │  factual   │
            └──────────────┘ └────────────┘ │            │
                                           └─────┬──────┘
                                                 │
                                    ┌────────────▼────────────┐
                                    │   PASS 1: Uncertainty   │
                                    │   Detection (minimal)   │
                                    │                         │
                                    │   • Embedding vector    │
                                    │   • Next-token entropy  │
                                    │   • Top-100 logprob mass│
                                    │   • Hidden-state norm   │
                                    │                         │
                                    │   KV cache reset before │
                                    │   each evaluation       │
                                    └────────────┬────────────┘
                                                 │
                                    ┌────────────▼────────────┐
                                    │   Calibrated Threshold  │
                                    │                         │
                                    │   Distance to known    │
                                    │   reference embeddings  │
                                    │   vs unknown references │
                                    └────────────┬────────────┘
                                                 │
                                      ┌──────────┼──────────┐
                                      │                     │
                               ┌──────▼──────┐      ┌──────▼──────┐
                               │   KNOWN     │      │  UNKNOWN    │
                               │             │      │             │
                               │  PASS 2:    │      │  Abstain    │
                               │  Generate   │      │  immediately│
                               │  Answer     │      │             │
                               └─────────────┘      └─────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Two-Pass Detector | `two_pass_llama_detector.py` | Core: embedding + entropy detection, KV cache reset, calibrated thresholds |
| Question-Type Classifier | `ep_question_type.py` | Routes questions by type (factual, subjective, nonsense, meta, counterfactual, ambiguous) |
| Self-Consistency Checker | `ep_consistency.py` | Multi-sample consistency for counterfactual/meta questions |
| Semantic Coherence Probe | `ep_coherence.py` | Detects nonsense via embedding norm anomalies |
| Edge Case Tester | `ep_edge_cases.py` | 5-category benchmark suite |
| Memory Safety Guard | `memory_guard.py` | Prevents OOM kills on constrained hardware |
| Benchmark Runner | `benchmark_model.py` | Per-model evaluation with memory monitoring |

### Detection Signals (Pass 1)

1. **Embedding distance** — Distance from question embedding to calibrated known/unknown reference clusters. Questions near known references are likely answerable; those near unknown references or far from both are likely unanswerable.

2. **Next-token entropy** — Shannon entropy of the top-100 logprobs at the question's last token. High entropy = model is uncertain what comes next. Low entropy = model has a confident continuation.

3. **Top-100 probability mass** — What fraction of the total probability is captured by the top 100 tokens. Low mass = probability is spread across many tokens = uncertainty.

4. **Hidden-state norm** — L2 norm of the model's hidden state at the last position. Deviations from baseline indicate abnormal model states ("terror of question" effect).

### Question-Type Routing

| Type | Strategy | Rationale |
|------|----------|-----------|
| Factual | Embedding + entropy | Standard known/unknown detection |
| Subjective | Always abstain | No objectively correct answer |
| Nonsense | Coherence check → reject | Semantically incoherent questions |
| Meta | Self-consistency (2 samples) | "What are you?" — diverse but correct self-descriptions |
| Counterfactual | Self-consistency (3 samples) | Physics violations need consistent domain knowledge |
| Ambiguous | Embedding + entropy | Could be known or unknown depending on interpretation |

### KV Cache Reset

Each `detect()` call resets the model's KV cache to prevent context leakage between questions. Without this reset, residual key-value pairs from previous evaluations contaminate subsequent detections, causing accuracy to degrade as the benchmark progresses.

## Results

### Benchmark Suite (5 Tests)

Each test evaluates a different edge case the detector should handle:

| Test | Questions | Expected | Threshold |
|------|-----------|----------|-----------|
| **Counterfactual** | Physics violations ("What if gravity pushed up?") | Detect as known (model knows physics) | accuracy >= 0.60 |
| **Nonsense** | Gibberish ("What is the color of the concept of sadness?") | Detect as unknown | accuracy >= 0.60 |
| **Ambiguous** | Vague questions ("What is the best programming language?") | Detect as unknown | accuracy >= 0.60 |
| **Meta** | Self-knowledge ("What are you?") | Detect as known | accuracy >= 0.60 |
| **Niche** | Obscure topics | Detect known vs unknown | accuracy >= 0.60 |

### Multi-Model Benchmark

**Hardware:** Apple M1 (8-core), 16 GB unified memory, macOS Sonoma 14.4

| Model | Parameters | Quantization | Size | Counter | Nonsense | Ambiguous | Meta | Niche | Total |
|-------|-----------|-------------|------|---------|----------|-----------|------|-------|-------|
| **Qwen3-8B** | 8B | Q4_K_M | 4.7 GB | **1.00** | **1.00** | **1.00** | **0.68** | 0.20 | **4/5** |
| **Qwen2.5-7B** | 7B | Q4_K_M | 4.4 GB | **1.00** | **0.96** | **1.00** | **0.72** | 0.20 | **4/5** |
| **Qwen3.5-4B** | 4B | Q4_K_M | 2.6 GB | 0.20 | **1.00** | **1.00** | 0.00 | 0.13 | **2/5** |

Models tested but excluded from final table:
- **Llama-3.1-8B-Instruct** (4.7 GB) — 3/5 (previous session)
- **TinyLlama-1.1B** (638 MB) — detector too small for reliable signals
- **Qwen3-30B-A3B** (9-12 GB) — OOM killed on 16 GB unified memory
- **Phi-4** (8.6 GB) — blocked by memory guard

### KV Cache Fix Impact

The KV cache reset between detections produced significant accuracy improvements:

| Test | Qwen3-8B Before | Qwen3-8B After | Qwen2.5-7B Before | Qwen2.5-7B After |
|------|-----------------|-----------------|-------------------|-------------------|
| Counterfactual | 0.20 | **1.00** | 0.20* | **1.00** |
| Meta | 0.00 | **0.68** | 0.00* | **0.72** |
| Nonsense | 1.00 | **1.00** | 1.00* | **0.96** |
| Ambiguous | 1.00 | **1.00** | 1.00* | **1.00** |

*Estimated from same-era run without KV reset.

### Niche Detection Limitation

Niche accuracy remains low (0.13-0.20) across all models. Root cause analysis revealed the test measures the wrong thing — "niche" questions (Yoneda lemma, spectral sequences, Langlands program) are genuinely in the model's training data. The model correctly identifies them as "known," but the test labels them as expecting different treatment. See `NICHE_DETECTION_ANALYSIS.md` for the full investigation.

Four candidate signals were tested for better niche separation:

| Signal | Best Accuracy | Separation |
|--------|--------------|------------|
| First-token entropy | 0.62 | No gap between known/post-cutoff |
| Answer consistency (3 samples) | 0.62 | Broken — raw llm() continuation mode |
| Token logprob variance | 0.62 | Marginal for absurd (0.65 vs 0.40) |
| Generate-then-verify | 0.62 | Broken — logprobs not extracted |

All signals max out at 0.62 because raw `llm()` without chat template produces continuations, not answers. The signal is dominated by continuation behavior, not knowledge state. A chat-template-based approach would likely yield stronger separation.

## Related Work

This is an engineering prototype, not a research contribution. Each technique used here is well-established individually. The table below maps what exists and where this project stands relative to published work.

| Paper / System | Year | What It Does | Overlap With This Project |
|---------------|------|-------------|--------------------------|
| **Confidence-Aware Routing** (Nandakishor, arXiv:2510.01237) | 2025 | Pre-generation hallucination mitigation using reference embeddings. Routes to 4 pathways based on confidence. | Very close. Uses reference embeddings + pre-generation routing. We add question-type classification but the core idea of embedding-based pre-generation gating is the same. |
| **Semantic Uncertainty** (Kuhn, Gal, Farquhar, ICLR 2023) | 2023 | Measures uncertainty over meanings via semantic entropy after generating multiple outputs. | Different direction — post-generation vs pre-generation. But the goal (detecting model uncertainty) is the same. |
| **Know Your Limits: Abstention in LLMs** (Wen et al., TACL 2025) | 2025 | Survey of when and how LLMs should refuse to answer. Covers query-level, model-level, and value-based abstention. | Our question-type routing is a specific instance of the query-level abstention framework they describe. |
| **HaluNet** (Tong et al., arXiv:2512.24562) | 2025 | Multi-granular uncertainty: token-level + semantic + distributional. Operates during/after generation. | Uses similar signals (logprobs, embeddings) but applies them post-generation rather than pre-generation. |
| **Knowledge Boundary of LLMs** (Li et al., ACL 2025) | 2025 | Taxonomizes LLM knowledge into 4 types. Addresses knowability as a conceptual framework. | Our question-type classification is a simpler version of their knowledge taxonomy. |
| **AbstentionBench** (NeurIPS 2025) | 2025 | Benchmark for LLM abstention on unanswerable questions. | Our 5-category benchmark could be evaluated against their framework. |

### What This Project Contributes

We do not claim a novel technique. The contribution is an **engineered integration** of existing ideas that works on consumer hardware:

1. Pre-generation entropy/embedding detection — well-established (see Nandakishor 2025)
2. Question-type classification — simple rule-based, not novel
3. Two-pass detect-then-generate — the architecture pattern is straightforward
4. Calibrated reference embeddings — standard prototype-based classification

The integration is useful because it runs on a 16 GB MacBook with local GGUF models, not because it introduces new methodology.

### Limitations

- **Small-scale evaluation.** 3 models, 5 test categories, 10 questions each. No statistical significance testing. Results may not generalize to other models or question distributions.
- **GGUF-specific.** The detector uses llama-cpp-python's embedding and logprob APIs. It does not work with API-based models (OpenAI, Anthropic) or other inference engines (vLLM, TGI) without modification.
- **Threshold-dependent.** Accuracy depends on calibration questions. Different calibration sets produce different thresholds. There is no principled way to choose optimal calibration data.
- **Niche detection is broken.** The niche test conflates "topics the model knows" with "topics humans consider obscure." The Yoneda lemma is niche to most humans but well-represented in training data. This is a test design flaw, not a detector flaw, but it means 1/5 categories is unreliable.
- **No comparison with baselines.** We did not compare against semantic uncertainty, calibrated confidence, or retrieval-augmented approaches on the same test set. The 4/5 result cannot be contextualized without baseline numbers.
- **Question-type classifier is rule-based.** It uses keyword matching and heuristics, not a learned classifier. It will misclassify edge cases (e.g., "What is the meaning of life?" could be philosophical or factual).
- **Single-machine results.** All benchmarks run on one M1 MacBook. Memory constraints prevented testing larger models (30B+). Results on different hardware may differ.
- **Chat template not used.** All model calls use raw `llm()` completion, not `create_chat_completion()`. This means the model produces continuations rather than conversational answers, which weakens consistency-based signals.

## Installation

```bash
# Clone
git clone https://github.com/mokumaisancho/question-signal-detector.git
cd question-signal-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install llama-cpp-python numpy torch

# Set model path (required)
export MODEL_PATH=/path/to/your/model.gguf
# Or for multi-model benchmarks:
export MODELS_DIR=/path/to/models/directory
```

### Supported Models

Any GGUF model compatible with llama-cpp-python. Tested with:

| Model | File | Source |
|-------|------|--------|
| Qwen3-8B-Q4_K_M | `Qwen3-8B-Q4_K_M.gguf` | [HuggingFace](https://huggingface.co/Qwen) |
| Qwen2.5-7B-Instruct-Q4_K_M | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | [HuggingFace](https://huggingface.co/Qwen) |
| Qwen3.5-4B-Q4_K_M | `Qwen3.5-4B-Q4_K_M.gguf` | [HuggingFace](https://huggingface.co/Qwen) |
| Llama-3.1-8B-Instruct-Q4_K_M | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | [HuggingFace](https://huggingface.co/meta-llama) |

## Usage

### Single Model Benchmark

```bash
python benchmark_model.py /path/to/model.gguf report.json
```

### Multi-Model Benchmark

```bash
python run_all_benchmarks.py --models-dir /path/to/models/ --output /tmp/reports/
```

### Programmatic API

```python
from two_pass_llama_detector import TwoPassLlamaDetector

detector = TwoPassLlamaDetector(MODEL_PATH="models/Qwen3-8B-Q4_K_M.gguf")

# Calibrate with known/unknown references
detector.calibrate(
    known_questions=["What is gravity?", "Who wrote Hamlet?"],
    unknown_questions=["What is Mars Colony population in 2035?"],
)

# Detect — no generation if uncertain
result = detector.detect("What is the capital of France?")
print(result["is_known"])          # True
print(result["uncertainty_score"]) # 0.15 (low = confident)

result = detector.detect("Who won the 2026 Nobel Prize in Physics?")
print(result["is_known"])          # False
print(result["route"])             # "standard_unknown"
```

## Project Structure

```
ep_2/
├── README.md
├── .gitignore
├── two_pass_llama_detector.py   # Core detector (two-pass architecture)
├── ep_question_type.py   # Question-type classifier
├── ep_consistency.py     # Self-consistency checker
├── ep_coherence.py       # Semantic coherence probe
├── ep_edge_cases.py      # 5-category benchmark suite
├── ep_harness.py         # Detection harness
├── ep_validation.py      # Cross-validation and validation
├── ep_dataset.py         # Dataset utilities
├── ep_cv.py              # Cross-validation implementation
├── ep_split.py           # Data splitting
├── ep_reporting.py       # Report generation
├── ep_multi_format.py    # Multi-format support
├── ep_per_language.py    # Per-language analysis
├── memory_guard.py              # OOM prevention and orphan cleanup
├── benchmark_model.py           # Single-model benchmark runner
├── run_all_benchmarks.py        # Multi-model benchmark runner
├── diagnose_*.py                # Diagnostic scripts
├── measure_*.py                 # Measurement/analysis scripts
├── docs/                        # Research findings and analysis (21 files)
│   ├── NICHE_DETECTION_ANALYSIS.md
│   ├── QUESTION_TERROR_GRADIENT_FINDING.md
│   ├── MULTI_MODEL_BENCHMARK_FINAL.md
│   ├── STATISTICAL_SOUNDNESS_ASSESSMENT.md
│   └── ...                      # See docs/ for full list
└── tests/                       # Unit tests and benchmark scripts
    ├── test_routing_logic.py    # Routing tests (4 tests, mock-based)
    ├── test_bug_repro.py        # Bug reproduction tests (4 tests)
    └── ...                      # Analysis scripts named test_*.py
```

## Key Findings

### The "Terror of Question" Effect
Wh-questions ("Who is...", "What is...") trigger different internal model states than declarative prompts. The model's hidden-state norm shifts measurably depending on question phrasing, even for questions about the same topic. See `QUESTION_TERROR_GRADIENT_FINDING.md`.

### Pre-Generation Signal Is More Honest
Confidence ratings produced by the model ("I am 90% confident") are post-hoc rationalization. The entropy and embedding signals extracted before generation starts provide a more accurate picture of what the model actually knows. See `META_SELF_AWARENESS_FINDING.md`.

### Top-100 vs Full Entropy
Using only the top-100 logprobs (instead of the full vocabulary distribution) introduces a systematic truncation bias. For high-entropy cases, top-100 mass drops below 60%, meaning 40%+ of probability mass is in the long tail. See `FULL_VS_TOPK_ENTROPY_FINDING.md`.

### Memory Constraints Shape Results
On 16 GB unified memory, models above ~5 GB encounter OOM during sustained benchmarks. The Memory Safety Guard prevents system crashes by monitoring swap, load, and free memory, and by killing orphan processes from crashed runs. See `RCA_MEMORY_MANAGEMENT.md`.

## License

MIT
