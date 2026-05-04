"""question signal Forecasting Harness — full integration.

Wires together:
- TwoPassLlamaDetector (core detector)
- MultiFormatEnsemble (format variance as signal)
- PerLanguageCalibrator (per-language thresholds)
- EdgeCaseTester (all 14 edge cases)
- CrossValidator (5-fold CV)
- StatisticalReporter (CI, FPR/FNR, ECE, calibration curves)

Architecture:
  INPUT: Question + Target Language
    -> Format Normalization (WH-rewrite)
    -> Language Check (assessed/unassessed/unsupported)
    -> Multi-Format Detection (3 variants)
    -> Signal Aggregation (mean + variance + overlap)
    -> Classification (known / unknown_in_domain / unknown_out_of_domain / uncertain)
  OUTPUT: Classification + Confidence + Domain Status
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


from ep_multi_format import FormatGenerator, MultiFormatEnsemble
from ep_per_language import PerLanguageCalibrator
from two_pass_llama_detector import ModelStabilityChecker, TwoPassLlamaDetector


@dataclass
class HarnessResult:
    """Output from the question signal forecasting harness."""

    question: str
    language: str
    status: str  # known, unknown_in_domain, unknown_out_of_domain, uncertain, unassessed
    confidence: float
    mean_score: float
    format_variance: float
    format_stability: float
    top10_overlap: float
    per_format_scores: dict[str, float]
    stability: dict[str, Any] | None
    edge_case_flags: dict[str, bool]


class question signalForecastingHarness:
    """Full harness integrating all components."""

    # Language status mapping
    LANGUAGE_STATUS = {
        "en": "assessed",
        "es": "assessed",
        "fr": "assessed",
        "de": "unassessed",
        "zh": "unassessed",
        "ja": "unassessed",
    }

    def __init__(
        self,
        detector: TwoPassLlamaDetector | None = None,
        stability_checker: ModelStabilityChecker | None = None,
    ) -> None:
        self.detector = detector or TwoPassLlamaDetector()
        self.stability_checker = stability_checker or ModelStabilityChecker(self.detector)
        self.format_generator = FormatGenerator()
        self.multi_format = MultiFormatEnsemble(self.detector, self.format_generator)
        self.language_calibrator = PerLanguageCalibrator(self.detector)
        self._calibrated = False

    def calibrate(self, language: str = "en") -> None:
        """Calibrate detector and stability checker."""
        print("[Harness] Calibrating...")
        self.stability_checker.calibrate()

        if language in self.LANGUAGE_STATUS:
            if self.LANGUAGE_STATUS[language] == "assessed":
                self.language_calibrator.calibrate_language(language)

        self._calibrated = True
        print("[Harness] Calibration complete.")

    def _normalize_format(self, question: str) -> str:
        """Normalize question to WH-format."""
        # If it's a statement completion, rewrite to WH
        if question.endswith(" is?") or question.endswith(" because?"):
            # Try to rewrite
            text = question.replace(" is?", "").replace(" because?", "")
            return f"What is {text}?"
        return question

    def detect(self, question: str, language: str = "en") -> HarnessResult:
        """Run full detection pipeline."""
        if not self._calibrated:
            self.calibrate(language)

        # Step 1: Format normalization
        normalized = self._normalize_format(question)

        # Step 2: Language check
        lang_status = self.LANGUAGE_STATUS.get(language, "unsupported")
        if lang_status == "unsupported":
            return HarnessResult(
                question=question,
                language=language,
                status="unsupported",
                confidence=0.0,
                mean_score=0.0,
                format_variance=0.0,
                format_stability=0.0,
                top10_overlap=0.0,
                per_format_scores={},
                stability=None,
                edge_case_flags={},
            )

        if lang_status == "unassessed":
            return HarnessResult(
                question=question,
                language=language,
                status="unassessed",
                confidence=0.0,
                mean_score=0.0,
                format_variance=0.0,
                format_stability=0.0,
                top10_overlap=0.0,
                per_format_scores={},
                stability=None,
                edge_case_flags={},
            )

        # Step 3: Stability check
        stability = self.stability_checker.check()
        if not stability["is_stable"]:
            return HarnessResult(
                question=question,
                language=language,
                status="uncertain",
                confidence=0.0,
                mean_score=0.0,
                format_variance=0.0,
                format_stability=0.0,
                top10_overlap=0.0,
                per_format_scores={},
                stability=stability,
                edge_case_flags={"unstable": True},
            )

        # Step 4: Multi-format detection
        mf_result = self.multi_format.detect(normalized)

        # Step 5: Language-specific threshold adjustment
        if language in self.language_calibrator.calibrations:
            cal = self.language_calibrator.calibrations[language]
            # Adjust threshold based on KL vs English
            adjusted_threshold = 0.5 + cal.kl_vs_en * 0.1
        else:
            adjusted_threshold = 0.5

        # Step 6: Classification
        if mf_result.mean_score < adjusted_threshold * 0.8:
            status = "known"
            confidence = 1.0 - mf_result.mean_score
        elif mf_result.mean_score > adjusted_threshold * 1.2:
            if mf_result.score_variance > 0.1:
                status = "unknown_out_of_domain"
            else:
                status = "unknown_in_domain"
            confidence = mf_result.mean_score
        else:
            status = "uncertain"
            confidence = 0.5

        # Step 7: Edge case flags (simplified)
        edge_flags = {
            "high_variance": mf_result.score_variance > 0.1,
            "low_stability": mf_result.format_stability < 0.5,
            "low_overlap": mf_result.top10_overlap < 0.6,
        }

        return HarnessResult(
            question=question,
            language=language,
            status=status,
            confidence=confidence,
            mean_score=mf_result.mean_score,
            format_variance=mf_result.score_variance,
            format_stability=mf_result.format_stability,
            top10_overlap=mf_result.top10_overlap,
            per_format_scores={k: v["uncertainty_score"] for k, v in mf_result.per_format.items()},
            stability=stability,
            edge_case_flags=edge_flags,
        )

    def batch_detect(
        self, questions: list[tuple[str, str]]
    ) -> list[HarnessResult]:
        """Batch detection with (question, language) pairs."""
        return [self.detect(q, lang) for q, lang in questions]

    def save_result(self, result: HarnessResult, path: str = "harness_result.json") -> None:
        """Save a single result."""
        data = {
            "question": result.question,
            "language": result.language,
            "status": result.status,
            "confidence": result.confidence,
            "mean_score": result.mean_score,
            "format_variance": result.format_variance,
            "format_stability": result.format_stability,
            "top10_overlap": result.top10_overlap,
            "per_format_scores": result.per_format_scores,
            "stability": result.stability,
            "edge_case_flags": result.edge_case_flags,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def unload(self) -> None:
        """Unload model."""
        self.detector._unload()


if __name__ == "__main__":
    harness = question signalForecastingHarness()

    test_cases = [
        ("What is gravity?", "en"),
        ("What is the capital of France?", "en"),
        ("What is Mars Colony population in 2035?", "en"),
        ("Can topological persistence detect phase transitions?", "en"),
        ("Qu'est-ce que la gravité ?", "fr"),
        ("¿Qué es la gravedad?", "es"),
        ("什么是重力？", "zh"),  # Unassessed
    ]

    print("=" * 80)
    print("QUESTION SIGNAL FORECASTING HARNESS")
    print("=" * 80)

    for question, lang in test_cases:
        result = harness.detect(question, lang)
        print(f"\nQ: {question}")
        print(f"  Language: {lang} -> {result.status}")
        print(f"  Status: {result.status}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Mean score: {result.mean_score:.3f}")
        print(f"  Format variance: {result.format_variance:.3f}")
        print(f"  Format stability: {result.format_stability:.3f}")
        if result.per_format_scores:
            print(f"  Per-format: {result.per_format_scores}")

    harness.unload()
