import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from two_pass_llama_detector import TwoPassLlamaDetector

MODELS = {
    "Qwen3-8B": "/Volumes/BUF_2T_02/models/Qwen3-8B-Q4_K_M.gguf",
    "Qwen2.5-7B": "/Volumes/BUF_2T_02/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
}

QUESTION = "What is the capital of France?"

for name, path in MODELS.items():
    gc.collect()
    TwoPassLlamaDetector.MODEL_PATH = path
    d = TwoPassLlamaDetector(n_ctx=512)
    d._load()
    llm = d._llm

    # Warmup
    llm("Hello", max_tokens=5, temperature=0.1)

    # Measure TPS (3 runs)
    speeds = []
    for i in range(3):
        t0 = time.monotonic()
        resp = llm(QUESTION, max_tokens=50, temperature=0.7)
        elapsed = time.monotonic() - t0
        # Count tokens from logprobs
        td = resp["choices"][0].get("logprobs", {})
        if td and "tokens" in td:
            n_tokens = len(td["tokens"])
        else:
            n_tokens = max(1, len(resp["choices"][0]["text"].split()))
        tps = n_tokens / elapsed if elapsed > 0 else 0
        speeds.append((tps, n_tokens, elapsed))
        print(f"  {name} run {i+1}: {n_tokens} tokens in {elapsed:.2f}s = {tps:.1f} TPS", flush=True)

    avg_tps = sum(s[0] for s in speeds) / len(speeds)
    print(f"{name}: avg {avg_tps:.1f} TPS", flush=True)

    del d
    gc.collect()

print("DONE", flush=True)
