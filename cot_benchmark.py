#!/usr/bin/env python3
"""
GSM8K CoT Path Selection Benchmark

Evaluates whether entropy signals can identify the best Chain-of-Thought chain
from multiple independent generations.

KEY FINDING from dry-run: Initial entropy (H0) is IDENTICAL across chains because
all chains share the same prompt. H0-based selection degenerates to random.
Differentiation must come from entropy trajectory DURING generation.

Strategies compared:
  A) Random (baseline)
  B) Lowest final entropy       — low entropy at answer tokens = high confidence
  C) Most converging trajectory — decreasing entropy trend during reasoning
  D) Lowest min entropy         — chain that reaches peak confidence
  E) Majority vote              — standard self-consistency baseline
  F) Oracle (upper bound)

Model reset: mx.synchronize() + mx.clear_cache() after each chain.
MLX generate() is stateless — no persistent KV cache.

Usage:
  python3 cot_benchmark.py --dry-run          # 3 questions
  python3 cot_benchmark.py --n-questions 50   # Full benchmark (~30 min)
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ---- Configuration ----

DEFAULT_MODEL = os.environ.get(
    "COT_MODEL",
    "/Volumes/BUF_2T_02/QwenMLB/models/Qwen3.5-4B-MLX-4bit",
)

COT_PROMPT = """Solve this math problem step by step. Show your work.
Write the final answer after ####.

Problem: {question}

Solution:"""

FINAL_ENTROPY_WINDOW = 10  # last N tokens for "final entropy"


# ---- Data Classes ----


@dataclass
class ChainResult:
    text: str
    answer: Optional[float]
    initial_entropy: float
    trajectory: list          # per-step entropies during generation
    trajectory_trend: float   # linear slope (neg = converging)
    final_entropy: float      # avg entropy of last N tokens
    min_entropy: float        # minimum entropy reached during generation
    correct: bool
    generation_time: float


@dataclass
class QuestionResult:
    question: str
    true_answer: float
    chains: list


# ---- Core Functions ----


def compute_entropy_from_logits(logits_1d: np.ndarray) -> float:
    logits_1d = logits_1d - np.max(logits_1d)
    exp_l = np.exp(logits_1d)
    sum_exp = np.sum(exp_l)
    log_probs = logits_1d - np.log(sum_exp)
    probs = exp_l / sum_exp
    return float(-np.sum(probs * log_probs))


def extract_answer(text: str) -> Optional[float]:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    match = re.search(r"(?:answer is|equals|=)\s*\$?\s*(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    numbers = re.findall(r"(?<!\d)-?[\d,]+\.?\d*(?!\d)", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def load_gsm8k(n_questions: int = 50) -> list:
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        results = []
        for item in ds:
            if len(results) >= n_questions:
                break
            ans_match = re.search(r"####\s*(-?[\d,]+\.?\d*)", item["answer"])
            if ans_match:
                ans = float(ans_match.group(1).replace(",", ""))
                results.append((item["question"], ans))
        return results
    except Exception:
        pass

    import tempfile
    import urllib.request
    url = (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        "master/grade_school_math/data/test.jsonl"
    )
    cache = os.path.join(tempfile.gettempdir(), "gsm8k_test.jsonl")
    if not os.path.exists(cache):
        print("Downloading GSM8K test set...")
        urllib.request.urlretrieve(url, cache)

    results = []
    with open(cache) as f:
        for line in f:
            if len(results) >= n_questions:
                break
            item = json.loads(line)
            ans_match = re.search(r"####\s*(-?[\d,]+\.?\d*)", item["answer"])
            if ans_match:
                ans = float(ans_match.group(1).replace(",", ""))
                results.append((item["question"], ans))
    return results


def generate_chain(model, m, tokenizer, prompt: str,
                   max_tokens: int = 256, temperature: float = 0.7) -> ChainResult:
    """Generate one CoT chain with entropy collection. Clears cache after."""
    prompt_ids = tokenizer.encode(prompt)

    sampler = make_sampler(temp=temperature)
    t0 = time.monotonic()
    response = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, sampler=sampler, verbose=False,
    )
    gen_time = time.monotonic() - t0

    # Determine full token sequence
    response_ids = tokenizer.encode(response)
    if (len(response_ids) > len(prompt_ids)
            and response_ids[:len(prompt_ids)] == prompt_ids):
        generated_text = tokenizer.decode(response_ids[len(prompt_ids):])
        full_ids = response_ids
    else:
        generated_text = response
        full_ids = prompt_ids + response_ids

    # Forward pass for entropy signals
    full_tokens = mx.array(full_ids)[None, :]
    logits = m(full_tokens)

    prompt_len = len(prompt_ids)
    total_len = len(full_ids)

    # Initial entropy
    init_logits = np.array(logits[0, prompt_len - 1, :].astype(mx.float32))
    initial_entropy = compute_entropy_from_logits(init_logits)

    # Trajectory entropy (each generation step)
    trajectory = []
    for i in range(prompt_len, total_len - 1):
        step_logits = np.array(logits[0, i, :].astype(mx.float32))
        trajectory.append(compute_entropy_from_logits(step_logits))

    # Derived metrics
    if len(trajectory) >= 3:
        trend = float(np.polyfit(range(len(trajectory)), trajectory, 1)[0])
    else:
        trend = 0.0

    final_entropy = float(np.mean(trajectory[-FINAL_ENTROPY_WINDOW:])) if trajectory else initial_entropy
    min_entropy = float(min(trajectory)) if trajectory else initial_entropy

    # Model reset
    mx.synchronize()
    mx.clear_cache()

    answer = extract_answer(generated_text)

    return ChainResult(
        text=generated_text,
        answer=answer,
        initial_entropy=initial_entropy,
        trajectory=trajectory,
        trajectory_trend=trend,
        final_entropy=final_entropy,
        min_entropy=min_entropy,
        correct=False,
        generation_time=gen_time,
    )


# ---- Strategy Evaluation ----

STRATEGIES = [
    "random",
    "lowest_final_entropy",
    "best_trajectory",
    "lowest_min_entropy",
    "majority_vote",
    "oracle",
]


def select_chain(chains: list, strategy: str, n_chains: int) -> int:
    """Return index of selected chain."""
    if strategy == "random":
        return 0
    elif strategy == "lowest_final_entropy":
        return min(range(n_chains), key=lambda i: chains[i].final_entropy)
    elif strategy == "best_trajectory":
        return min(range(n_chains), key=lambda i: chains[i].trajectory_trend)
    elif strategy == "lowest_min_entropy":
        return min(range(n_chains), key=lambda i: chains[i].min_entropy)
    elif strategy == "oracle":
        # Return first correct, or 0 if none correct
        for i in range(n_chains):
            if chains[i].correct:
                return i
        return 0
    return 0


def compute_strategies(results: list, n_chains: int) -> dict:
    scores = {}
    for name in STRATEGIES:
        correct = 0
        for r in results:
            if name == "majority_vote":
                answers = [c.answer for c in r.chains if c.answer is not None]
                if answers:
                    rounded = [round(a) for a in answers]
                    vote = Counter(rounded).most_common(1)[0][0]
                    correct += abs(vote - r.true_answer) < 0.5
            else:
                idx = select_chain(r.chains, name, n_chains)
                correct += r.chains[idx].correct
        total = len(results)
        scores[name] = {"accuracy": correct / total if total else 0,
                        "correct": correct, "total": total}
    return scores


def print_evaluation(results: list, n_chains: int):
    scores = compute_strategies(results, n_chains)

    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<25s} {'Accuracy':>10s} {'Count':>10s}")
    print("-" * 45)
    for name in STRATEGIES:
        data = scores[name]
        print(f"  {name:<23s} {data['accuracy']:>8.3f}   "
              f"{data['correct']:>4d}/{data['total']}")

    # Entropy analysis: correct vs incorrect chains
    print("\n--- Entropy Profile: Correct vs Incorrect Chains ---")
    c_final, w_final = [], []
    c_trend, w_trend = [], []
    c_min, w_min = [], []
    for r in results:
        for c in r.chains:
            if c.correct:
                c_final.append(c.final_entropy)
                c_trend.append(c.trajectory_trend)
                c_min.append(c.min_entropy)
            else:
                w_final.append(c.final_entropy)
                w_trend.append(c.trajectory_trend)
                w_min.append(c.min_entropy)

    if c_final and w_final:
        print(f"  Correct ({len(c_final):3d} chains): "
              f"final_H={np.mean(c_final):.3f} min_H={np.mean(c_min):.3f} "
              f"trend={np.mean(c_trend):+.5f}")
        print(f"  Wrong   ({len(w_final):3d} chains): "
              f"final_H={np.mean(w_final):.3f} min_H={np.mean(w_min):.3f} "
              f"trend={np.mean(w_trend):+.5f}")
        print(f"  Final H gap: {abs(np.mean(c_final) - np.mean(w_final)):.3f}")
        print(f"  Min H gap:   {abs(np.mean(c_min) - np.mean(w_min)):.3f}")
        print(f"  Trend gap:   {abs(np.mean(c_trend) - np.mean(w_trend)):.5f}")

    # Gap analysis
    rand = scores["random"]["accuracy"]
    oracle = scores["oracle"]["accuracy"]
    gap = oracle - rand
    print(f"\n  Random→Oracle gap: {gap:.3f}")
    if gap > 0:
        for key in ["lowest_final_entropy", "best_trajectory",
                     "lowest_min_entropy", "majority_vote"]:
            closed = (scores[key]["accuracy"] - rand) / gap
            print(f"  {key} closes {closed:.1%} of gap")


# ---- Benchmark Runner ----


def run_benchmark(model_path: str, n_questions: int = 50,
                  n_chains: int = 3, max_tokens: int = 256,
                  temperature: float = 0.7):
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    m = model.model if hasattr(model, "model") else model
    print("Model loaded.\n")

    questions = load_gsm8k(n_questions)
    print(f"Loaded {len(questions)} GSM8K questions")
    print(f"Config: {n_chains} chains x {max_tokens} max tokens x T={temperature}")
    est = len(questions) * n_chains * max_tokens / 18 / 60
    print(f"Estimated: ~{est:.0f} min at 18 TPS\n")

    results = []
    t_start = time.monotonic()

    for i, (question, true_answer) in enumerate(questions):
        prompt = COT_PROMPT.format(question=question)
        chains = []

        for c in range(n_chains):
            chain = generate_chain(model, m, tokenizer, prompt,
                                   max_tokens=max_tokens, temperature=temperature)
            chain.correct = (chain.answer is not None
                             and abs(chain.answer - true_answer) < 0.5)
            chains.append(chain)

        qr = QuestionResult(question=question, true_answer=true_answer, chains=chains)
        results.append(qr)

        n_correct = sum(1 for ch in chains if ch.correct)
        elapsed = time.monotonic() - t_start
        ans_str = ",".join(str(c.answer) if c.answer else "None" for c in chains)
        trends = ",".join(f"{c.trajectory_trend:+.4f}" for c in chains)
        print(f"[{i+1:3d}/{len(questions)}] true={true_answer:>8.1f} "
              f"ok={n_correct}/{n_chains} trend=[{trends}] ans=[{ans_str}] "
              f"({elapsed:.0f}s)")

    total_time = time.monotonic() - t_start
    print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f} min)")

    print_evaluation(results, n_chains)

    # Save JSON
    output = {
        "metadata": {
            "model": model_path,
            "n_questions": len(questions),
            "n_chains": n_chains,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "total_seconds": round(total_time, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "initial_entropy identical across chains (same prompt) - "
                    "selection uses trajectory/final/min entropy only",
        },
        "strategies": compute_strategies(results, n_chains),
        "results": [
            {
                "question": r.question[:120],
                "true_answer": r.true_answer,
                "chains": [
                    {
                        "answer": c.answer,
                        "final_entropy": round(c.final_entropy, 4),
                        "min_entropy": round(c.min_entropy, 4),
                        "trajectory_trend": round(c.trajectory_trend, 5),
                        "trajectory_len": len(c.trajectory),
                        "correct": c.correct,
                        "gen_time": round(c.generation_time, 2),
                    }
                    for c in r.chains
                ],
            }
            for r in results
        ],
    }

    out_path = f"/tmp/cot_benchmark_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GSM8K CoT Path Selection Benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-chains", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 3 questions for verification")
    args = parser.parse_args()

    if args.dry_run:
        args.n_questions = 3

    run_benchmark(args.model, args.n_questions, args.n_chains,
                  args.max_tokens, args.temperature)
