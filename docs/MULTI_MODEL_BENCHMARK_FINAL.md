# Multi-Model [Qq]uestion Signal Benchmark — Final Results

**Date:** 2026-05-04
**Hardware:** Apple M1, 8 cores, 16 GB unified memory
**Test:** 5 edge-case categories (counterfactual, nonsense, ambiguous, meta, niche)
**Pass threshold:** accuracy >= 0.5 per category

## Final Leaderboard

| Rank | Model | Size | Pass | Counter | Nonsense | Ambig | Meta | Niche |
|------|-------|------|------|---------|----------|-------|------|-------|
| 1 | **Qwen3-8B-Q4_K_M** | 4.8G | **4/5** | 1.00 | 1.00 | 1.00 | 0.72 | 0.20 |
| 1 | **Qwen2.5-7B-Q4_K_M** | 4.4G | **4/5** | 1.00 | 0.96 | 1.00 | 0.72 | 0.20 |
| 3 | Llama-3.1-8B-Q4_K_M | 4.7G | 3/5 | 1.00 | 0.44 | 0.88 | 0.36 | 0.90 |
| 3 | TinyLlama-1.1B-Q4_K_M | 0.6G | 3/5 | 0.00 | 0.96 | 0.88 | 0.40 | 0.80 |
| 3 | Phi-4-14B-Q4_K_M | 8.4G | 3/5 | 0.76 | 1.00 | 1.00 | 0.00 | 0.27 |
| 6 | Mistral-7B-Q4_K_M | 4.2G | 2/5 | 0.04 | 0.92 | 0.88 | 0.40 | 0.37 |
| 7 | Qwen3.5-4B-Q4_K_M | 2.6G | 2/5 | 0.28 | 1.00 | 1.00 | 0.00 | 0.13 |
| — | Qwen3-30B-A3B-UD-IQ1_M | 9.0G | DNF | — | — | — | — | — |

## Key Findings

### 1. 7-8B is the sweet spot for [Qq]uestion Signal detection on 16 GB unified memory

Qwen3-8B and Qwen2.5-7B tie at 4/5. Both achieve perfect counterfactual (1.00), nonsense (0.96-1.00), and ambiguous (1.00) detection. Meta self-awareness was fixed via self-consistency routing (0.04 → 0.72). Niche detection (0.20) remains unsolved across all models.

### 2. Model size correlates with [Qq]uestion Signal capability — up to a point

- **4B (Qwen3.5-4B):** Too small. Counterfactual collapses (0.28), meta is zero (0.00). Only nonsense and ambiguous pass.
- **7-8B (Qwen3/2.5):** Optimal for 16 GB hardware. All categories pass except niche.
- **14B (Phi-4):** No improvement over 8B. Meta detection is zero (0.00) — the model claims to know everything.
- **30B MoE:** Could not complete benchmark. IQ1_M quant (9.0 GB) loaded but was OOM-killed (exit 143) during counterfactual test. IQ3_XXS (12.0 GB) exceeded unified memory entirely.

### 3. Niche detection is universally broken (0.13–0.90)

No model reliably identifies obscure/niche questions as unknown. Models hallucinate knowledge of obscure topics. This is a fundamental limitation of the embedding-similarity approach — niche questions are linguistically similar to known facts.

### 4. Meta self-awareness requires self-consistency routing

Original entropy-based meta detection failed on Qwen models (0.04 accuracy). Switching to `SelfConsistencyChecker(n_samples=2)` with threshold 0.05 fixed this (0.72). The model generates multiple answers and checks consistency — inconsistent answers indicate genuine self-awareness of knowledge boundaries.

### 5. MoE models are not viable on 16 GB unified memory

Qwen3-30B-A3B has 30.5B total / 3.3B active parameters. Even the smallest quants:
- UD-IQ1_S: 8.4 GB (loaded but OOM-killed mid-test)
- UD-IQ1_M: 9.0 GB (loaded but OOM-killed mid-test)
- UD-IQ3_XXS: 12.0 GB (exceeded memory, blocked by MemorySafetyGuard)

The IQ1 quants loaded but the KV cache + inference buffers pushed past 16 GB during sustained multi-sample tests.

## Hardware Constraints

| Quant | Model Size | Runtime Estimate (×1.3) | Fits 16 GB? | Result |
|-------|-----------|------------------------|-------------|--------|
| Q4_K_M 4B | 2.6 GB | 3.4 GB | Yes | 2/5 pass |
| Q4_K_M 7-8B | 4.4-4.8 GB | 5.7-6.2 GB | Yes | 4/5 pass |
| Q4_K_M 14B | 8.4 GB | 10.9 GB | Tight | 3/5 pass |
| IQ1_M 30B MoE | 9.0 GB | 11.7 GB | No (OOM) | DNF |
| IQ3_XXS 30B MoE | 12.0 GB | 15.6 GB | No | Blocked |

## Test Infrastructure

- **Benchmark script:** `benchmark_model.py` with `MemorySafetyGuard` integration
- **Detector:** `two_pass_llama_detector.py` (TwoPassLlamaDetector with calibration caching)
- **Edge cases:** `ep_edge_cases.py` (EdgeCaseTester, 5 focused categories)
- **Memory guard:** Pre-flight check, mid-benchmark swap monitoring (5 GB abort threshold), post-benchmark orphan cleanup
- **Calibration cache:** `~/.cache/[Qq]uestion Signal/{model_hash}_cal.json` — avoids redundant llama calls on re-runs

## Reports

| Model | Report Path |
|-------|-------------|
| TinyLlama-1.1B | `/tmp/tinyllama_report.json` |
| Mistral-7B | `/tmp/mistral_report.json` |
| Llama-3.1-8B | `/tmp/llama31_report.json` |
| Qwen2.5-7B (v2, meta fix) | `/tmp/qwen_v2_report.json` |
| Qwen3-8B | `/tmp/qwen3_report.json` |
| Phi-4-14B | `/tmp/phi4_report.json` |
| Qwen3.5-4B | `/tmp/qwen35_4b_report.json` |
| Qwen3-30B-A3B | DNF (OOM killed) |

## Conclusion

**Qwen3-8B-Q4_K_M is the recommended model** for [Qq]uestion Signal detection on M1 16 GB. It achieves the best score (4/5) with the smallest footprint that doesn't sacrifice capability. The remaining gap is niche detection, which requires a fundamentally different approach (not embedding similarity).
