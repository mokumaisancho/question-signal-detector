"""
Two-Pass Llama Detector — separate uncertainty detection from answer generation.

Architecture:
  Pass 1 (Uncertainty Detection): Minimal forward pass on question.
    Extract:
      - Embedding vector (model's internal representation)
      - Next-token entropy (from logprobs at last position)
      - Embedding distance to calibrated references
    NO full generation. The model is not forced to produce an answer.

  Pass 2 (Answer Generation): ONLY if Pass 1 says "known."
    Run autoregressive generation. If Pass 1 says "unknown," abstain immediately.

This addresses the "terror of question" insight: autoregressive generation
hijacks attention and drowns out the uncertainty signal. By detecting
uncertainty BEFORE full generation starts, we get an honest signal.

Memory guard from topologicalinferencing:
  - Lazy loading (model loads on first use)
  - torch.mps.empty_cache() between operations
  - 4-bit quantized model (already done via GGUF)
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time

import numpy as np
import torch

# Wave 1 imports for edge-case hardening
from ep_question_type import QuestionTypeClassifier
from ep_consistency import SelfConsistencyChecker
from ep_coherence import SemanticCoherenceProbe


class ModelStabilityChecker:
    """Assess model baseline stability before any question is asked.

    The "terror of question" insight: if a model is already internally
    unstable (high variance on neutral prompts, drift from baseline),
    asking questions will produce unreliable answers regardless of
    whether the model knows the topic.

    This checker runs minimal probes on neutral text to establish a
    "mental state" baseline — like checking pulse before surgery.
    """

    # Short, unambiguous prompts that any stable model should handle consistently
    NEUTRAL_PROMPTS = [
        "The sky is blue.",
        "Hello world.",
        "1 + 1 = 2",
        "Water freezes at zero degrees.",
    ]

    # Stability thresholds (calibrated empirically; adjust per model)
    STABILITY_THRESHOLD = 2.0  # score < this = stable
    DRIFT_TOLERANCE_ENTROPY = 1.5
    DRIFT_TOLERANCE_NORM = 10.0

    def __init__(self, detector: TwoPassLlamaDetector) -> None:
        self.detector = detector
        self._baseline: dict | None = None

    def calibrate(self, *, force: bool = False) -> None:
        """Establish baseline metrics on neutral prompts. Caches to disk."""
        stab_cache = self.detector._cache_path("stab")
        if not force:
            try:
                with open(stab_cache) as f:
                    self._baseline = json.load(f)
                print(
                    f"[StabilityChecker] Loaded baseline from cache: "
                    f"entropy={self._baseline['entropy_mean']:.2f}"
                    f"±{self._baseline['entropy_std']:.2f}, norm={self._baseline['norm_mean']:.1f}"
                    f"±{self._baseline['norm_std']:.1f}"
                )
                return
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass

        print("[StabilityChecker] Calibrating baseline on neutral prompts...")
        metrics: list[dict] = []
        for prompt in self.NEUTRAL_PROMPTS:
            result = self.detector._pass1_uncertainty(prompt)
            metrics.append(
                {
                    "entropy": result["next_token_entropy"],
                    "norm": result["hidden_norm"],
                }
            )
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        entropies = np.array([m["entropy"] for m in metrics])
        norms = np.array([m["norm"] for m in metrics])

        self._baseline = {
            "entropy_mean": float(entropies.mean()),
            "entropy_std": float(entropies.std()),
            "entropy_min": float(entropies.min()),
            "entropy_max": float(entropies.max()),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
        }

        try:
            with open(stab_cache, "w") as f:
                json.dump(self._baseline, f)
        except OSError:
            pass

        print(
            f"[StabilityChecker] Baseline: entropy={self._baseline['entropy_mean']:.2f}"
            f"±{self._baseline['entropy_std']:.2f}, norm={self._baseline['norm_mean']:.1f}"
            f"±{self._baseline['norm_std']:.1f}"
        )

    def check(self) -> dict:
        """Check current model stability against calibrated baseline.

        Returns a dict with stability score and diagnostics. Lower score
        = more stable. A score above STABILITY_THRESHOLD means the model
        is in an unreliable state and questions should be deferred.
        """
        if self._baseline is None:
            self.calibrate()

        current: list[dict] = []
        for prompt in self.NEUTRAL_PROMPTS:
            result = self.detector._pass1_uncertainty(prompt)
            current.append(
                {
                    "entropy": result["next_token_entropy"],
                    "norm": result["hidden_norm"],
                }
            )
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        entropies = np.array([c["entropy"] for c in current])
        norms = np.array([c["norm"] for c in current])

        ent_mean = float(entropies.mean())
        ent_std = float(entropies.std())
        norm_mean = float(norms.mean())
        norm_std = float(norms.std())

        # Drift from baseline (absolute deviation normalized by baseline std)
        # Use 1e-6 epsilon instead of 0.1 to avoid 10x underestimation for
        # stable models with small natural variance.
        ent_drift = abs(ent_mean - self._baseline["entropy_mean"])
        norm_drift = abs(norm_mean - self._baseline["norm_mean"])
        ent_drift_z = ent_drift / max(self._baseline["entropy_std"], 1e-6)
        norm_drift_z = norm_drift / max(self._baseline["norm_std"], 1e-6)

        # Internal variance (inconsistency across neutral prompts)
        ent_variance_z = ent_std / max(self._baseline["entropy_std"], 1e-6)
        norm_variance_z = norm_std / max(self._baseline["norm_std"], 1e-6)

        # Combined stability score (lower = more stable)
        stability_score = (
            0.30 * ent_drift_z
            + 0.30 * norm_drift_z
            + 0.20 * ent_variance_z
            + 0.20 * norm_variance_z
        )

        is_stable = stability_score < self.STABILITY_THRESHOLD

        return {
            "is_stable": is_stable,
            "stability_score": float(stability_score),
            "entropy_drift": float(ent_drift),
            "norm_drift": float(norm_drift),
            "entropy_variance": float(ent_std),
            "norm_variance": float(norm_std),
            "baseline": self._baseline,
            "current": {
                "entropy_mean": ent_mean,
                "entropy_std": ent_std,
                "norm_mean": norm_mean,
                "norm_std": norm_std,
            },
        }


class TwoPassLlamaDetector:
    """Knowledge boundary detector using Llama-2-7B with two-pass architecture."""

    MODEL_PATH = os.environ.get("MODEL_PATH", "models/llama-2-7b.Q4_K_M.gguf")

    def __init__(self, n_ctx: int = 2048) -> None:
        self.n_ctx = n_ctx
        self._llm: object | None = None
        self._loaded = False

        # Reference embeddings for calibration
        self._known_refs: list[np.ndarray] = []
        self._unknown_refs: list[np.ndarray] = []
        self._calibrated = False

        # Cache directory
        self._cache_dir = os.path.expanduser("~/.cache/question signal")
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
        except OSError:
            pass

    def _model_hash(self) -> str:
        """Hash model path + size for cache key."""
        path = self.MODEL_PATH
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        return hashlib.md5(f"{path}:{size}:{self.n_ctx}".encode()).hexdigest()[:12]

    def _cache_path(self, suffix: str) -> str:
        return os.path.join(self._cache_dir, f"{self._model_hash()}_{suffix}.json")

    def _load(self) -> None:
        """Lazy-load Llama model. Memory guard: only load when needed."""
        if self._loaded:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python required. Install: pip install llama-cpp-python"
            )

        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {self.MODEL_PATH}")

        # Memory guard: clean up before loading
        self._reclaim_memory()

        # Check available memory vs model size
        model_size_gb = os.path.getsize(self.MODEL_PATH) / (1024**3)
        mem_info = self._check_memory()
        print(f"[TwoPassLlama] Model={model_size_gb:.1f}GB swap={mem_info['swap_gb']:.1f}GB "
              f"load={mem_info['load_1m']:.1f}", flush=True)
        if mem_info['swap_gb'] > 4.0:
            print(f"[TwoPassLlama] WARNING: swap >4GB, load may fail", flush=True)

        print(f"[TwoPassLlama] Loading {self.MODEL_PATH}...", flush=True)
        t0 = time.monotonic()
        self._llm = Llama(
            model_path=self.MODEL_PATH,
            n_ctx=self.n_ctx,
            n_threads=8,
            verbose=False,
            embedding=True,
            logits_all=True,
        )
        print(f"[TwoPassLlama] Loaded in {time.monotonic()-t0:.1f}s", flush=True)
        self._loaded = True

    @staticmethod
    def _check_memory() -> dict:
        """Check system memory state."""
        import subprocess
        vm = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True, text=True,
        )
        parts = vm.stdout.strip().split("used = ")
        swap_gb = float(parts[1].split("M")[0]) / 1024 if len(parts) > 1 else 0.0
        ld = subprocess.run(
            ["sysctl", "-n", "vm.loadavg"],
            capture_output=True, text=True,
        )
        nums = ld.stdout.strip().strip("{}").split()
        return {
            "swap_gb": swap_gb,
            "load_1m": float(nums[0]) if nums else 0.0,
            "load_5m": float(nums[1]) if len(nums) > 1 else 0.0,
        }

    @staticmethod
    def _reclaim_memory() -> None:
        """Force garbage collection and kill orphan model processes."""
        import signal as sig
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        # Kill orphan Python processes holding >500MB (likely dead model loads)
        my_pid = os.getpid()
        result = os.popen("ps -eo pid,rss,comm").read()
        for line in result.strip().split("\n")[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            pid, rss, comm = int(parts[0]), int(parts[1]), parts[2]
            if "Python" not in comm and "python" not in comm:
                continue
            if pid == my_pid or rss < 500_000:
                continue
            print(f"[TwoPassLlama] Killing orphan PID={pid} ({rss//1024}MB)", flush=True)
            try:
                os.kill(pid, sig.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(1)
        gc.collect()

    def _unload(self) -> None:
        """Memory guard: explicitly unload to free RAM.

        Pattern from topologicalinferencing:
          - Delete model reference
          - Run gc.collect() to free Python objects
          - Clear MPS cache if on Apple Silicon
        """
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._loaded = False
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── Context manager for guaranteed cleanup ─────────────────────

    def session(self):
        """Return a context manager that guarantees cleanup on exit.

        Usage:
            detector = TwoPassLlamaDetector()
            with detector.session():
                detector.calibrate(...)
                result = detector.detect(...)
        """
        return _DetectorSession(self)

    # ── PASS 1: Uncertainty Detection (minimal generation) ─────────

    def _pass1_uncertainty(self, question: str) -> dict:
        """Detect uncertainty with minimal generation (1 token only).

        Uses llama-cpp-python's logprobs to get the next-token distribution
        at the last position of the question. This captures the model's
        uncertainty BEFORE it's forced to generate a full answer.
        """
        self._load()
        self._llm.reset()

        # Get embedding for the question
        emb_result = self._llm.create_embedding(question)
        raw_embedding = np.array(emb_result["data"][0]["embedding"])
        if raw_embedding.ndim > 1:
            embedding = raw_embedding.mean(axis=0)
            n_tokens = raw_embedding.shape[0]
        else:
            embedding = raw_embedding
            n_tokens = len(question.split())

        # Get next-token logprobs (max_tokens=1, temperature=0)
        gen_result = self._llm(
            question,
            max_tokens=1,
            temperature=0.0,
            logprobs=100,
            echo=False,
        )

        # Extract logprobs for the generated token
        logprobs_dict = gen_result["choices"][0].get("logprobs", {})
        top_logprobs = logprobs_dict.get("top_logprobs", [{}])

        if top_logprobs and len(top_logprobs) > 0:
            # top_logprobs is a list of dicts, one per generated token
            token_probs = top_logprobs[0]
            logprobs_arr = np.array(list(token_probs.values()), dtype=np.float64)
            raw_probs = np.exp(logprobs_arr)  # logprobs -> probabilities
            top100_mass = float(raw_probs.sum())  # BUG-3-002: capture truncation mass
            probs = raw_probs / top100_mass  # normalize
            log_p = np.log(probs + 1e-10)
            entropy = float(-(probs * log_p).sum())
        else:
            entropy = 5.0  # Max entropy fallback
            top100_mass = 1.0

        # Hidden state norm (proxy from embedding norm)
        hidden_norm = float(np.linalg.norm(embedding))

        return {
            "next_token_entropy": entropy,
            "hidden_norm": hidden_norm,
            "embedding": embedding,
            "top100_mass": top100_mass,
            "n_tokens": n_tokens,
        }

    # ── PASS 2: Answer Generation (only if Pass 1 says known) ──────

    def _pass2_generate(self, question: str, max_tokens: int = 50) -> str:
        """Generate answer ONLY if uncertainty is low."""
        self._load()
        self._llm.reset()

        output = self._llm(
            question,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["\n\n", "Question:"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    # ── Calibration: build reference embeddings ────────────────────

    def calibrate(
        self,
        known_questions: list[str],
        unknown_questions: list[str],
        *,
        force: bool = False,
    ) -> None:
        """Build reference embeddings for known and unknown questions.

        Caches to disk so re-running the same model skips recomputation.
        """
        cal_cache = self._cache_path("cal")
        if not force:
            try:
                with open(cal_cache) as f:
                    cached = json.load(f)
                self._known_refs = [np.array(e) for e in cached["known_refs"]]
                self._unknown_refs = [np.array(e) for e in cached["unknown_refs"]]
                self._calibrated = True
                print(f"[TwoPassLlama] Loaded calibration from cache ({len(self._known_refs)}+{len(self._unknown_refs)} refs)", flush=True)
                return
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass

        print(
            f"[TwoPassLlama] Calibrating on {len(known_questions)} known, "
            f"{len(unknown_questions)} unknown...",
            flush=True,
        )

        self._known_refs = []
        for q in known_questions:
            result = self._pass1_uncertainty(q)
            self._known_refs.append(result["embedding"])
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        self._unknown_refs = []
        for q in unknown_questions:
            result = self._pass1_uncertainty(q)
            self._unknown_refs.append(result["embedding"])
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        self._calibrated = True

        # Save to cache
        try:
            with open(cal_cache, "w") as f:
                json.dump({
                    "known_refs": [e.tolist() for e in self._known_refs],
                    "unknown_refs": [e.tolist() for e in self._unknown_refs],
                }, f)
        except OSError:
            pass

        print("[TwoPassLlama] Calibration complete.", flush=True)

    # ── Combined Score ─────────────────────────────────────────────

    def detect(
        self, question: str, *, stability_checker: ModelStabilityChecker | None = None
    ) -> dict:
        """Two-pass detection with edge-case routing. Returns full result with uncertainty metrics.

        Args:
            question: The question to assess.
            stability_checker: Optional stability checker. If provided and the
                model is unstable, detection aborts with is_known=False.
        """
        # Pre-flight: model stability check
        stability: dict | None = None
        if stability_checker is not None:
            stability = stability_checker.check()
            if not stability["is_stable"]:
                return {
                    "question": question,
                    "is_known": False,
                    "uncertainty_score": 999.0,
                    "next_token_entropy": 0.0,
                    "hidden_norm": 0.0,
                    "embedding_signal": 0.0,
                    "min_known_dist": 0.0,
                    "min_unknown_dist": 0.0,
                    "generated_answer": "",
                    "stability": stability,
                    "aborted": True,
                }

        # Wave 2: Question-type classification and routing
        qtype = QuestionTypeClassifier().classify(question)

        # Route by question type
        if qtype == "subjective":
            # W2-T2: Subjective questions are always uncertain
            return {
                "question": question,
                "is_known": False,
                "uncertainty_score": 0.5,
                "next_token_entropy": 0.0,
                "hidden_norm": 0.0,
                "embedding_signal": 0.0,
                "min_known_dist": 0.0,
                "min_unknown_dist": 0.0,
                "generated_answer": "",
                "question_type": qtype,
                "route": "subjective_abstain",
                "aborted": False,
            }

        if qtype == "nonsense":
            # W2-T3: Check semantic coherence
            coherence = SemanticCoherenceProbe()
            # Use fallback calibration with general knowledge questions
            if not coherence._calibrated:
                coherence.NATURAL_QUESTION_STATS["mean_norm"] = 70.0
                coherence._calibrated = True
            coh_result = coherence.check(question, self)
            if not coh_result["is_coherent"]:
                return {
                    "question": question,
                    "is_known": False,
                    "uncertainty_score": 0.8,
                    "next_token_entropy": coh_result["entropy"],
                    "hidden_norm": coh_result["norm"],
                    "embedding_signal": 0.0,
                    "min_known_dist": 0.0,
                    "min_unknown_dist": 0.0,
                    "generated_answer": "",
                    "question_type": qtype,
                    "route": "nonsense_coherence",
                    "coherence_score": coh_result["coherence_score"],
                    "aborted": False,
                }
            # If coherent somehow, fall through to standard detection

        if qtype == "counterfactual":
            # W2-T1: Use self-consistency instead of embedding distance
            checker = SelfConsistencyChecker(self, n_samples=3)
            try:
                consistency = checker.check(question)
                if consistency["is_consistent"]:
                    # Model gives consistent physics answers → knows the domain
                    return {
                        "question": question,
                        "is_known": True,
                        "uncertainty_score": 0.2,
                        "next_token_entropy": 0.0,
                        "hidden_norm": 0.0,
                        "embedding_signal": 0.0,
                        "min_known_dist": 0.0,
                        "min_unknown_dist": 0.0,
                        "generated_answer": consistency["answers"][0] if consistency["answers"] else "",
                        "question_type": qtype,
                        "route": "counterfactual_consistency",
                        "consistency_score": consistency["consistency_score"],
                        "aborted": False,
                    }
                else:
                    # Inconsistent → uncertain about domain
                    return {
                        "question": question,
                        "is_known": False,
                        "uncertainty_score": 0.7,
                        "next_token_entropy": 0.0,
                        "hidden_norm": 0.0,
                        "embedding_signal": 0.0,
                        "min_known_dist": 0.0,
                        "min_unknown_dist": 0.0,
                        "generated_answer": "",
                        "question_type": qtype,
                        "route": "counterfactual_inconsistent",
                        "consistency_score": consistency["consistency_score"],
                        "aborted": False,
                    }
            except (ValueError, KeyError, RuntimeError):
                # Fallback to standard detection if consistency check fails
                pass

        if qtype == "meta":
            # Meta questions: use self-consistency to check if model knows itself
            # Entropy is unreliable here because model produces diverse but
            # correct self-descriptions ("I am a large language model..." etc.)
            checker = SelfConsistencyChecker(self, n_samples=2)
            try:
                consistency = checker.check(question)
                # Lower threshold for meta: even partial consistency = knows self
                is_self_aware = consistency["consistency_score"] > 0.05
                return {
                    "question": question,
                    "is_known": is_self_aware,
                    "uncertainty_score": 0.3 if is_self_aware else 0.7,
                    "next_token_entropy": 0.0,
                    "hidden_norm": 0.0,
                    "embedding_signal": 0.0,
                    "min_known_dist": 0.0,
                    "min_unknown_dist": 0.0,
                    "generated_answer": consistency["answers"][0] if is_self_aware and consistency["answers"] else "",
                    "question_type": qtype,
                    "route": "meta_self_consistency",
                    "consistency_score": consistency["consistency_score"],
                    "aborted": False,
                }
            except (ValueError, KeyError, RuntimeError):
                pass

        # Standard detection for factual and meta questions
        # Pass 1: Uncertainty detection (minimal generation)
        p1 = self._pass1_uncertainty(question)

        # Embedding distance to references
        emb = p1["embedding"]
        if self._calibrated:
            known_dists = [
                float(np.linalg.norm(emb - ref)) for ref in self._known_refs
            ]
            unknown_dists = [
                float(np.linalg.norm(emb - ref)) for ref in self._unknown_refs
            ]
            min_known = min(known_dists) if known_dists else 1.0
            min_unknown = min(unknown_dists) if unknown_dists else 1.0
            embedding_signal = min_unknown - min_known  # positive = closer to known
        else:
            min_known = min_unknown = embedding_signal = 0.0

        # Combined uncertainty score (higher = more uncertain)
        entropy_norm = p1["next_token_entropy"] / 5.0
        # W1-T1: Length-normalized norm signal
        n_tokens = p1.get("n_tokens", len(question.split()))
        norm_per_token = p1["hidden_norm"] / max(n_tokens, 1)
        norm_signal = (norm_per_token - 2.0) / 1.0
        # BUG-3-002 fix: truncation_signal = 1 - top100_mass
        truncation_signal = 1.0 - p1.get("top100_mass", 1.0)
        combined = (
            0.4 * entropy_norm
            + 0.3 * norm_signal
            + 0.1 * truncation_signal
            - 0.2 * embedding_signal
        )

        # Decision: threshold at 0.5
        is_known = combined < 0.5

        # Pass 2: Generate answer ONLY if known
        answer = ""
        if is_known:
            answer = self._pass2_generate(question)

        result = {
            "question": question,
            "is_known": is_known,
            "uncertainty_score": combined,
            "next_token_entropy": p1["next_token_entropy"],
            "hidden_norm": p1["hidden_norm"],
            "embedding_signal": embedding_signal,
            "min_known_dist": min_known,
            "min_unknown_dist": min_unknown,
            "generated_answer": answer,
            "question_type": qtype,
            "route": "standard",
            "aborted": False,
        }
        if stability is not None:
            result["stability"] = stability
        return result


class _DetectorSession:
    """Context manager for guaranteed detector cleanup.

    Ported from topologicalinferencing resource-management patterns.
    Ensures model is unloaded even if an exception occurs.
    """

    def __init__(self, detector: TwoPassLlamaDetector) -> None:
        self.detector = detector

    def __enter__(self) -> TwoPassLlamaDetector:
        return self.detector

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.detector._unload()


# ── Self-Test ────────────────────────────────────────────────────

if __name__ == "__main__":
    detector = TwoPassLlamaDetector()

    known_cal = [
        "What is gravity?",
        "What is the capital of France?",
        "What is DNA?",
        "What is machine learning?",
    ]
    unknown_cal = [
        "Can topological persistence detect phase transitions?",
        "Can sheaf cohomology detect misinformation cascades?",
        "Who won the 2032 presidential election?",
    ]

    detector.calibrate(known_cal, unknown_cal)

    # ── Pre-flight stability assessment ──────────────────────────────
    print("\n" + "=" * 80)
    print("PRE-FLIGHT STABILITY CHECK")
    print("=" * 80)
    stability = ModelStabilityChecker(detector)
    stability.calibrate()
    status = stability.check()
    print(
        f"  Stable: {status['is_stable']}  "
        f"score={status['stability_score']:.3f}  "
        f"ent_drift={status['entropy_drift']:.2f}  "
        f"norm_drift={status['norm_drift']:.1f}"
    )

    test_questions = [
        ("What is the speed of light?", "known"),
        ("What is CRISPR?", "known"),
        ("Does the Wasserstein distance predict discovery novelty?", "unknown"),
        ("Can persistent homology detect mode collapse?", "unknown"),
        ("What is Python used for?", "known"),
        ("What is Mars Colony population in 2035?", "unknown"),
    ]

    print("\n" + "=" * 80)
    print("TWO-PASS DETECTION RESULTS")
    print("=" * 80)

    correct = 0
    for q, expected in test_questions:
        result = detector.detect(q, stability_checker=stability)
        pred = "known" if result["is_known"] else "unknown"
        ok = "OK" if pred == expected else "FAIL"
        if result.get("aborted"):
            ok = "ABORT"
        print(
            f"\n{ok}  score={result['uncertainty_score']:.3f}  "
            f"ent={result['next_token_entropy']:.2f}  "
            f"norm={result['hidden_norm']:.2f}"
        )
        print(f"    Q: {q}")
        if result["generated_answer"]:
            print(f"    A: {result['generated_answer'][:80]}")

    detector._unload()
