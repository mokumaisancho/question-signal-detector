"""Multi-format ensemble: measure detector signals across 3-4 formats.

WH-questions, imperatives, statements. Uses variance across formats as an
additional signal (fifth signal after entropy, norm, truncation, embedding).

Key insight from research: unknown questions show 3x higher format variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FormatGenerator:
    """Generate format variants of a question."""

    WH_PREFIXES = {
        "what": "What is",
        "where": "Where is",
        "how": "How does",
        "which": "Which",
        "why": "Why does",
    }

    def __init__(self) -> None:
        pass

    def generate(self, question: str, wh_type: str = "what") -> dict[str, str]:
        """Generate WH, imperative, statement variants."""
        # WH variant (already interrogative)
        wh = question

        # Imperative variant
        if question.startswith("What is"):
            imp = "Explain " + question[8:].replace("?", ".")
        elif question.startswith("How does"):
            imp = "Explain how " + question[9:].replace("?", ".")
        elif question.startswith("Can"):
            imp = "Analyze whether " + question[4:].replace("?", ".")
        else:
            imp = "Explain " + question[0].lower() + question[1:].replace("?", ".")

        # Statement variant
        if question.endswith("?"):
            stmt = question[:-1] + " is?"
        else:
            stmt = question + " is?"

        return {
            "wh": wh,
            "imperative": imp,
            "statement": stmt,
        }


@dataclass
class MultiFormatResult:
    """Result of multi-format detection."""

    mean_score: float
    score_variance: float
    score_std: float
    max_kl: float
    format_stability: float
    top10_overlap: float
    is_known: bool
    domain_status: str
    per_format: dict[str, dict]


class MultiFormatEnsemble:
    """Multi-format ensemble detector wrapper."""

    def __init__(self, detector, format_generator: FormatGenerator | None = None) -> None:
        self.detector = detector
        self.format_generator = format_generator or FormatGenerator()

    def detect(self, question: str, wh_type: str = "what") -> MultiFormatResult:
        """Run detector on multiple formats and aggregate."""
        formats = self.format_generator.generate(question, wh_type)

        results = {}
        scores = []
        for fmt_name, fmt_text in formats.items():
            result = self.detector.detect(fmt_text)
            results[fmt_name] = result
            scores.append(result["uncertainty_score"])

        scores_arr = np.array(scores)
        mean_score = float(np.mean(scores_arr))
        score_variance = float(np.var(scores_arr))
        score_std = float(np.std(scores_arr))
        max_kl = float(np.max(scores_arr) - np.min(scores_arr))
        format_stability = 1.0 / (1.0 + score_variance)

        # Top-10 overlap across formats (simplified: use score proximity)
        top10_overlap = 1.0 - min(1.0, score_std * 2.0)

        # Classification with stability-adjusted thresholds
        if mean_score < 0.4 and score_variance < 0.05:
            is_known = True
            domain_status = "known"
        elif mean_score > 0.6:
            is_known = False
            cv = score_std / mean_score if mean_score > 0 else 0
            if cv > 0.4:
                domain_status = "unknown_out_of_domain"
            elif cv < 0.2:
                domain_status = "unknown_in_domain"
            else:
                domain_status = "uncertain"
        else:
            is_known = False
            domain_status = "uncertain"

        return MultiFormatResult(
            mean_score=mean_score,
            score_variance=score_variance,
            score_std=score_std,
            max_kl=max_kl,
            format_stability=format_stability,
            top10_overlap=top10_overlap,
            is_known=is_known,
            domain_status=domain_status,
            per_format=results,
        )

    def batch_detect(
        self, questions: list[str], wh_types: list[str] | None = None
    ) -> list[MultiFormatResult]:
        """Batch multi-format detection."""
        if wh_types is None:
            wh_types = ["what"] * len(questions)
        return [self.detect(q, wt) for q, wt in zip(questions, wh_types)]


if __name__ == "__main__":
    from two_pass_llama_detector import TwoPassLlamaDetector

    detector = TwoPassLlamaDetector()
    known = ["What is gravity?", "What is the capital of France?"]
    unknown = ["What is Mars Colony population in 2035?", "Can sheaf cohomology detect misinformation?"]
    detector.calibrate(known, unknown)

    ensemble = MultiFormatEnsemble(detector)

    test_questions = [
        "What is gravity?",
        "What is Mars Colony population in 2035?",
        "Can topological persistence detect phase transitions?",
    ]

    for q in test_questions:
        result = ensemble.detect(q)
        print(f"\nQ: {q}")
        print(f"  Mean score: {result.mean_score:.3f}")
        print(f"  Variance: {result.score_variance:.3f}")
        print(f"  Format stability: {result.format_stability:.3f}")
        print(f"  Domain status: {result.domain_status}")
        print(f"  Is known: {result.is_known}")

    detector._unload()
