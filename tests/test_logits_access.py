"""Test how to access full logits from llama-cpp-python."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from two_pass_llama_detector import TwoPassLlamaDetector

detector = TwoPassLlamaDetector()
detector._load()
llm = detector._llm

question = "What is gravity?"
tokens = llm.tokenize(question.encode())
print(f"Tokens: {tokens}")
print(f"n_tokens: {len(tokens)}")

# Evaluate
llm.eval(tokens)

# Check _scores
print(f"\n_scores type: {type(llm._scores)}")
if hasattr(llm._scores, 'shape'):
    print(f"_scores shape: {llm._scores.shape}")
if hasattr(llm._scores, '__len__'):
    print(f"_scores len: {len(llm._scores)}")

# Try to access logits for last token
import numpy as np
if isinstance(llm._scores, np.ndarray):
    print(f"_scores ndim: {llm._scores.ndim}")
    if llm._scores.ndim == 2:
        # Shape might be (n_tokens, n_vocab)
        print(f"Last token logits shape: {llm._scores[-1].shape}")
        print(f"Sample values: {llm._scores[-1][:5]}")
    elif llm._scores.ndim == 1:
        print(f"_scores[:5]: {llm._scores[:5]}")

# Try eval_logits
print("\n--- eval_logits ---")
print(f"eval_logits type: {type(llm.eval_logits)}")
if hasattr(llm.eval_logits, 'shape'):
    print(f"eval_logits shape: {llm.eval_logits.shape}")
if callable(llm.eval_logits):
    result = llm.eval_logits()
    print(f"eval_logits() result type: {type(result)}")
    if hasattr(result, 'shape'):
        print(f"eval_logits() shape: {result.shape}")

detector._unload()
