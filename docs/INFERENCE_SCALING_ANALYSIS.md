# Inference Scaling Analysis: Local vs Hardware Upgrade vs Cloud

**Date:** 2026-05-05
**Context:** M1 16GB running Qwen3-8B at 3.6 TPS (llama-cpp CPU), Qwen2.5-7B at 5.1 TPS

## Current Baseline

| Model | Backend | TPS | Detection Accuracy | Fit in 16GB |
|-------|---------|-----|--------------------|-------------|
| Qwen3-8B Q4 | llama-cpp CPU | 3.6 | 4/5 categories | Yes (~5GB) |
| Qwen2.5-7B Q4 | llama-cpp CPU | 5.1 | 4/5 categories | Yes (~4.4GB) |
| Qwen3.5-4B Q4 | llama-cpp CPU | ~8 | 2/5 categories | Yes (~2.5GB) |
| Qwen3.5-4B MLX | MLX GPU/ANE | 18 | Not benchmarked | Yes (~2.5GB) |

## Hardware Upgrade Path

### Apple Silicon (same codebase, zero migration)

| Hardware | Memory BW | 8B TPS | 4B TPS | 32B TPS | Cost |
|----------|-----------|--------|--------|---------|------|
| M1 16GB (current, CPU) | 68 GB/s | 3.6 | 8 | — | $0 |
| M1 16GB (MLX/Metal) | 68 GB/s | 15-25 | 25-40 | — | $0 |
| M4 Base 16GB | 120 GB/s | 25-30 | 40-50 | — | ~$1,300 |
| M4 Pro 48GB | 273 GB/s | 55-68 | 90-110 | 12-15 | ~$3,500 |
| M4 Max 128GB | 546 GB/s | 93+ | 159 | 25-30 | ~$5,500 |

**Key insight:** Switching from CPU llama-cpp to MLX on the same M1 gives 4-7x TPS improvement with no hardware change. The two-pass pipeline runs at effective TPS × 2 because pass 1 (entropy only) is faster than pass 2 (generation).

**Memory bandwidth formula:** `Max tok/s ≈ Memory Bandwidth / Model Size in Memory`. M1 = 68 GB/s, 8B Q4 = ~5GB → theoretical max ~13.6 tok/s. CPU achieves ~3.6 because GPU/Metal shaders have higher throughput than CPU cores.

### NVIDIA GPUs

| GPU | VRAM | 8B TPS | 32B TPS | 72B TPS | Cost |
|-----|------|--------|---------|---------|------|
| RTX 3090 | 24GB | 48-55 | 35-40 | — | $700-999 used |
| RTX 4090 | 24GB | 62 | 55-60 | — | $1,599+ |
| RTX 5090 | 32GB | 95 | 80+ | 35-45 | $1,999+ |

### Official Qwen3 Benchmarks (SGLang, H20 96GB, single input)

| Model | BF16 tok/s | FP8 tok/s | INT4 tok/s |
|-------|-----------|----------|-----------|
| Qwen3-4B | 133 | 201 | 200 |
| Qwen3-8B | 82 | 150 | 144 |
| Qwen3-14B | 47 | 97 | 96 |
| Qwen3-32B | 21 | 46 | 48 |
| Qwen3-30B-A3B (MoE) | 137 | 156 | 31 (GPTQ) |

## Model Size vs Detection Quality

| Aspect | Scaling Behavior |
|--------|-----------------|
| Counterfactual/meta detection | **Strong scaling.** 32B significantly higher on reasoning (MMLU-Pro: 72.6 vs 55.1 for 8B) |
| Niche/unusual topic detection | **Moderate scaling.** 4B→8B jump gives 10-15% on knowledge tasks. Beyond 14B, diminishing returns |
| Two-pass architecture | **Compensates for model size.** Two-pass 8B likely captures 80-90% of single-pass 32B |
| Entropy signal quality | **Unclear.** No published data on how entropy separation scales with model size |

**Practical recommendation:** The two-pass architecture with 8B likely matches or exceeds single-pass 14B at detection tasks. Upgrading to 14B/32B improves hardest-case counterfactual detection by 10-20% but costs 4-8x compute.

## Cost per 1000 Questions (Two-Pass Pipeline)

| Platform | Model | TPS | Time/1K | Cost/1K |
|----------|-------|-----|---------|---------|
| M1 16GB CPU | Qwen3-8B Q4 | 3.6 | ~54 min | $0 |
| M1 16GB MLX | Qwen3-8B Q4 | 18 | ~11 min | $0 |
| M4 Pro 48GB | Qwen3-8B Q4 | 60 | ~3 min | $0 |
| RTX 4090 | Qwen3-32B Q4 | 55 | ~3.5 min | $0 |
| Cloud GPT-4.1 Mini | API | ~100+ | <1 min | ~$1.12 |
| Cloud Haiku 4.5 | API | ~100+ | <1 min | ~$2.40 |
| Cloud Sonnet 4.6 | API | ~100+ | <1 min | ~$9.00 |

## Recommendations

### Immediate (zero cost)
- Switch Qwen3.5-4B to MLX backend → 8 TPS → 18 TPS (2.2x)
- This makes CoT selection experiments viable (50 questions × 3 chains = ~8 min)

### Short-term (zero cost)
- Switch Qwen3-8B to MLX/Metal → 3.6 TPS → 15-25 TPS (4-7x)
- Enables real-time detection (<5s per question including both passes)

### Medium-term ($1,600-2,000)
- RTX 4090 or 5090 → run Qwen3-32B at 55-80 TPS
- Significant improvement on counterfactual/meta categories

### What NOT to do
- Don't upgrade to 14B on M1 16GB — only marginal accuracy gain, won't fit with overhead
- Don't use cloud APIs for bulk processing — $1-10/1K adds up, and local is free
- Don't pursue 72B without 32+ GB VRAM — won't fit in memory

## Sources

- [Qwen3 Official Speed Benchmark](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)
- [Qwen 3 Hardware Guide 2026](https://www.compute-market.com/blog/qwen-3-local-hardware-guide-2026)
- [Apple Silicon LLM Inference Optimization](https://blog.starmorph.com/blog/apple-silicon-llm-inference-optimization-guide)
- [LLM API Pricing Comparison April 2026](https://pecollective.com/blog/llm-api-pricing-comparison/)
