# Ultraplan: [Qq]uestion Signal Detection Benchmark Optimization

**Date:** 2026-05-04
**Goal:** Optimize benchmark speed 3-5x, fix universal meta-failure, enable reliable multi-model comparison
**Constraint:** M1 16GB, sequential model loading, no OOM

---

## Current State

5 models benchmarked, all complete. Results:
- 3/5 models pass 3/5 tests (TinyLlama, Llama-3.1, Qwen, Phi-4)
- **Meta self-awareness: 0% across ALL models** (hard failure, not a detection issue)
- ~390-420 llama_cpp calls per benchmark, ~60-90 min total for 5 models
- Memory guard works: zero OOM kills, zero orphans

## Bottleneck Analysis

| Component | Calls | % of Total | Optimization Potential |
|-----------|-------|-----------|----------------------|
| Counterfactual (SelfConsistency x3) | ~150 | 36% | Cache samples, reduce n_samples |
| Nonsense (coherence probe) | ~100 | 24% | Batch embedding+generation |
| Stability check (redundant) | ~16 | 4% | Cache after calibrate |
| Niche | ~60 | 14% | Pre-compute in-domain baseline |
| Meta | ~50 | 12% | Standard detect, hard to optimize |
| Calibration | ~12 | 3% | Cache to disk per model |
| Model loading | 5x | N/A | Can't parallelize (memory) |

**Top wins:** Cache calibration + stability to disk (eliminate 28 calls), batch embedding+generation in _pass1_uncertainty (-40% calls), reduce SelfConsistency n_samples from 3→2 or cache.

---

## Plan — 4 Tasks, 2 Agents

### Task 1: Cache Calibration & Stability to Disk
**Agent:** senior-python-engineer
**File:** `two_pass_llama_detector.py` + `memory_guard.py`
**Dependencies:** None (start first)
**Acceptance criteria:**
- Calibration embeddings cached to `~/.cache/[Qq]uestion Signal/{model_hash}_cal.json`
- Stability baseline cached to `~/.cache/[Qq]uestion Signal/{model_hash}_stab.json`
- Re-running same model skips calibration if cache exists
- `--force` flag bypasses cache
- Benchmark runtime drops ~15% on repeated runs
**Estimated savings:** ~28 llama calls, ~2-3 min per model

### Task 2: Batch Embedding + Generation in _pass1_uncertainty
**Agent:** senior-python-engineer
**File:** `two_pass_llama_detector.py`
**Dependencies:** None (parallel with Task 1)
**Acceptance criteria:**
- Single `llama()` call produces both embedding and logprobs where possible
- Fallback to two calls only if batched call fails
- All existing tests pass
- No regression in detection accuracy (±0.05 tolerance)
**Estimated savings:** ~40% reduction in llama calls
**GitHub reference:** ThunderLLAMA (https://github.com/lisihao/ThunderLLAMA) — fused QKV Metal kernels, KV cache quantization

### Task 3: Fix Meta Self-Awareness Detection
**Agent:** senior-python-engineer (research + implement)
**File:** `ep_edge_cases.py` + `two_pass_llama_detector.py`
**Dependencies:** Task 1 (cached calibration needed for quick iteration)
**Acceptance criteria:**
- Root cause analysis: why do ALL models score 0-4% on meta?
- Hypothesis: meta questions trigger "I should know this" → high confidence → false known
- Fix may require: separate calibration for self-referential prompts, or modified scoring
- At least 1 model achieves >50% meta accuracy after fix
- No regression on other 4 tests
**GitHub reference:** Kateryna (https://github.com/Zaneham/Kateryna) — ternary logic for overconfidence detection, may help separate "confident wrong" from "confident correct"

### Task 4: Automated Multi-Model Benchmark Runner
**Agent:** senior-python-engineer
**File:** New `run_all_benchmarks.py`
**Dependencies:** Tasks 1, 2 (must have optimizations first)
**Acceptance criteria:**
- Single command: `python run_all_benchmarks.py` runs all models in sequence
- Automatic memory guard between models (already in benchmark_model.py)
- Generates comparison table + JSON reports
- Skips models that can't fit (Phi-4 on loaded system)
- Progress tracking: "Model 3/5: Qwen2.5-7B..."
- Total runtime <30 min for 5 models (from ~60-90 min)
**GitHub reference:** LLM-Benchmarking-Harness-4bit (https://github.com/Lalu2002/LLM-Benchmarking-Harness-4bit) — sequential-load-with-cleanup pattern

---

## Development Order

```
Task 1 (cache) ─────────────────────┐
                                     ├──→ Task 4 (runner)
Task 2 (batch) ─────────────────────┤
                                     │
Task 3 (meta fix) ── depends on T1 ─┘
```

Tasks 1 and 2 run in parallel. Task 3 starts after Task 1. Task 4 after all others.

## Contradiction Check

- No contradictions in dev order: T1/T2 are independent, T3 only needs T1's cache, T4 needs all
- Memory: caching is disk-based, no memory concern. Batching reduces peak memory (one call vs two)
- Accuracy: Task 2 has ±0.05 tolerance gate. Task 3 must not regress other tests.

## MVP (Minimum Viable Product)

Tasks 1 + 2 alone give ~50% speed improvement. That's the MVP.
Task 3 is the research value (fixing universal failure mode).
Task 4 is convenience.

---

## GitHub Repos (don't reinvent)

| Repo | URL | What to use |
|------|-----|-------------|
| ThunderLLAMA | https://github.com/lisihao/ThunderLLAMA | KV cache quantization, fused Metal kernels for M1 |
| llama-bench | https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench | Raw speed comparison patterns |
| Kateryna | https://github.com/Zaneham/Kateryna | Ternary overconfidence detection (meta fix reference) |
| LLM-Benchmarking-Harness-4bit | https://github.com/Lalu2002/LLM-Benchmarking-Harness-4bit | Sequential load + cleanup pattern |
| mac-llm-bench | https://github.com/enescingoz/mac-llm-bench | Apple Silicon baseline comparison data |
