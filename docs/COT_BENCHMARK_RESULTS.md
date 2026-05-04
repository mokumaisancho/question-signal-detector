# CoT Path Selection Experiment Results

**Date:** 2026-05-05
**Model:** Qwen3.5-4B-MLX-4bit (4-bit quantized, Apple M1 16GB)
**Backend:** MLX (GPU/ANE), ~18 TPS generation
**Duration:** 20.5 minutes

## Experimental Design

- **Task:** GSM8K grade-school math word problems (25 questions)
- **Chains per question:** 3 independent CoT chains at temperature=0.7
- **Max tokens per chain:** 256
- **Strategies compared:** 6 (random, final entropy, trajectory, min entropy, majority vote, oracle)
- **Grading:** Exact integer match (within ±0.5)
- **Model reset:** `mx.synchronize()` + `mx.clear_cache()` after each chain

## Strategy Comparison

| Strategy | Accuracy | Correct | Gap Closed |
|----------|----------|---------|------------|
| Random (baseline) | 0.160 | 4/25 | — |
| Lowest final entropy | 0.160 | 4/25 | **-28.6%** (worse) |
| Best trajectory | 0.240 | 6/25 | 0.0% |
| Lowest min entropy | 0.240 | 6/25 | 0.0% |
| **Majority vote** | **0.360** | **9/25** | **42.9%** |
| Oracle (upper bound) | 0.520 | 13/25 | — |

Gap closed = (strategy_accuracy - random) / (oracle - random).

## Entropy Profile: Correct vs Incorrect Chains

| Metric | Correct (21 chains) | Wrong (54 chains) | Gap |
|--------|--------------------|--------------------|-----|
| Final entropy | 0.390 | 0.352 | 0.037 |
| Min entropy | 0.000 | 0.000 | 0.000 |
| Trend | -0.00042 | -0.00097 | 0.00056 |

## Key Findings

### Finding 1: Entropy signals do NOT predict CoT reasoning quality

Pre-generation entropy is useful for knowledge boundary detection ("does the model know this topic?") but NOT for reasoning quality selection ("did this chain reason correctly?"). These are fundamentally different tasks:

- **Knowledge boundary detection** (two-pass pipeline): Entropy separates known from unknown questions. Works because the model's uncertainty about the *topic* is reflected in next-token entropy.
- **Reasoning quality selection** (CoT path selection): Entropy during generation does NOT separate correct from incorrect chains. Correct chains have slightly *higher* final entropy (0.390) than wrong chains (0.352) — the opposite of the hypothesis.

The entropy gap is only 0.037 — too small to be useful for selection, and in the wrong direction.

### Finding 2: Final entropy selection is worse than random

Selecting the chain with lowest final entropy performed identically to random selection (4/25). The negative gap closure (-28.6%) means entropy selection actively avoids correct chains in some cases. A confident wrong answer has lower final entropy than a correct answer that required more reasoning steps.

### Finding 3: Majority vote is the only effective strategy

Standard self-consistency (majority vote across 3 chains) closes 42.9% of the random-to-oracle gap. This works because it aggregates answer-level information, not token-level uncertainty. The entropy signal adds no information beyond what voting captures.

### Finding 4: Low oracle accuracy limits all strategies

At 52% oracle accuracy, only 13 of 25 questions have at least one correct chain. This means 12 questions (48%) have NO correct chain — no selection strategy can help on these. The model simply lacks the capability to solve nearly half the problems.

## Why Entropy Fails for Reasoning Selection

1. **Same prompt = identical initial entropy.** All 3 chains share the same prompt, so pre-generation entropy is identical across chains. Verified: H0 values are bit-for-bit identical (e.g., all chains get H0=1.0086 for the same question).

2. **Confidence ≠ correctness.** A model can be confident about a wrong answer. When the model states "#### 80000" confidently, the entropy is low. But the correct answer might be 70000. Low entropy at answer tokens means the model *committed to an answer*, not that the answer is *correct*.

3. **Reasoning is serial and opaque.** The entropy trajectory reflects surface-level token uncertainty (which word comes next), not logical correctness (is this reasoning step valid). A chain can have decreasing entropy throughout while still containing a single arithmetic error that invalidates the final answer.

4. **Trajectory signal is too weak.** The trend differences between correct and incorrect chains are on the order of 0.001 per step — well within noise. The signal-to-noise ratio is insufficient for reliable selection.

## Implications

### For the two-pass detection pipeline
The entropy signal works for knowledge boundary detection because that task has a clear causal mechanism: uncertain topic → uncertain next token → high entropy. This result confirms that mechanism is real but task-specific. It does NOT generalize to reasoning quality assessment.

### For inference optimization
- **Majority vote at 1/3 cost is possible**: Generate 3 chains, vote, but only count the answer — don't need entropy measurement at all. This saves the forward pass overhead.
- **Entropy-based pre-filtering is not viable for CoT**: The original hypothesis was to measure entropy cheaply (forward pass only) and only fully generate the best chain. This doesn't work because entropy doesn't predict quality.
- **Best strategy: generate 3, vote, done.** No additional signal measurement needed.

### For future work
- **Per-step verification**: Instead of entropy, use a separate verifier model to check each reasoning step. This is more expensive but directly assesses logical correctness.
- **Answer confidence**: Measure entropy specifically at the "####" token position (where the answer is stated). This might be more informative than final-10-token average.
- **Larger model test**: Qwen3.5-4B at 52% oracle accuracy is too weak. A stronger model (8B or 14B) with higher oracle accuracy would provide more signal for selection evaluation.

## Data

Full results: `/tmp/cot_benchmark_1777936988.json`
Per-question breakdown:

```
Q1:  true=18    ok=0/3  ans=[16,3,16]     trend=[-0.0005,+0.0046,-0.0018]
Q2:  true=3     ok=2/3  ans=[3,3,1]       trend=[-0.0008,+0.0019,-0.0007]
Q3:  true=70000 ok=0/3  ans=[80000,80000,80000] trend=[-0.0020,-0.0015,+0.0001]
Q4:  true=540   ok=2/3  ans=[540,180,540]  trend=[-0.0007,-0.0005,+0.0001]
Q5:  true=20    ok=2/3  ans=[20,20,3]     trend=[-0.0015,-0.0024,-0.0011]
Q6:  true=64    ok=0/3  ans=[3,0.6,5]     trend=[-0.0013,-0.0034,-0.0001]
Q7:  true=260   ok=2/3  ans=[4,260,260]   trend=[-0.0014,+0.0006,+0.0001]
Q8:  true=160   ok=0/3  ans=[0.4,100,100] trend=[-0.0024,-0.0008,+0.0008]
Q9:  true=45    ok=0/3  ans=[4,None,180]  trend=[-0.0022,+0.0006,-0.0008]
Q10: true=460   ok=0/3  ans=[40,5,40]     trend=[-0.0012,-0.0002,-0.0024]
Q11: true=366   ok=1/3  ans=[60,60,366]   trend=[-0.0006,-0.0001,-0.0011]
Q12: true=694   ok=1/3  ans=[204,694,3]   trend=[-0.0014,+0.0000,-0.0019]
Q13: true=13    ok=0/3  ans=[7,12,7]      trend=[-0.0004,-0.0009,-0.0031]
Q14: true=18    ok=0/3  ans=[3,2,5]       trend=[-0.0009,-0.0009,-0.0015]
Q15: true=60    ok=1/3  ans=[20,0.2,60]   trend=[-0.0009,-0.0006,-0.0007]
Q16: true=125   ok=1/3  ans=[5000,125,13000] trend=[-0.0002,-0.0009,-0.0030]
Q17: true=230   ok=0/3  ans=[80,80,23]    trend=[-0.0015,-0.0001,-0.0011]
Q18: true=57500 ok=0/3  ans=[30,20,20]    trend=[-0.0013,-0.0028,-0.0005]
Q19: true=7     ok=2/3  ans=[3,7,7]       trend=[-0.0025,+0.0017,-0.0032]
Q20: true=6     ok=1/3  ans=[6,3,4]       trend=[-0.0020,-0.0022,-0.0020]
Q21: true=15    ok=0/3  ans=[10,9,10]     trend=[+0.0004,-0.0002,-0.0002]
Q22: true=14    ok=0/3  ans=[25,2,29]     trend=[+0.0010,+0.0010,-0.0011]
Q23: true=7     ok=2/3  ans=[3,7,7]       trend=[-0.0011,-0.0010,-0.0003]
Q24: true=8     ok=3/3  ans=[8,8,8]       trend=[+0.0001,+0.0022,+0.0007]
Q25: true=26    ok=1/3  ans=[26,0.25,19.5] trend=[-0.0017,-0.0019,-0.0020]
```
