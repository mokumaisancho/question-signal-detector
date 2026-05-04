"""5-fold cross-validation with per-fold threshold tuning.

Computes:
- Mean accuracy and std dev across folds
- 95% confidence interval
- Per-fold optimal threshold
- Threshold stability (std < 0.1)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np

from ep_dataset import question signalDataset


@dataclass
class CVResult:
    """Result of cross-validation."""

    fold_accuracies: list[float]
    fold_thresholds: list[float]
    mean_accuracy: float
    std_accuracy: float
    mean_threshold: float
    std_threshold: float
    ci_95_low: float
    ci_95_high: float
    per_fold_details: list[dict]


class CrossValidator:
    """5-fold cross-validation for question signal detector."""

    def __init__(self, detector_class, seed: int = 42) -> None:
        self.detector_class = detector_class
        self.seed = seed
        np.random.seed(seed)

    def evaluate_fold(
        self,
        train_questions: list,
        val_questions: list,
        threshold: float = 0.5,
    ) -> dict:
        """Evaluate a single fold."""
        # Instantiate detector
        detector = self.detector_class()

        # Calibrate on train
        known = [q.text for q in train_questions if q.label == "known"]
        unknown = [q.text for q in train_questions if q.label == "unknown"]
        detector.calibrate(known, unknown)

        # Evaluate on validation
        correct = 0
        predictions = []
        for q in val_questions:
            result = detector.detect(q.text)
            pred_known = result["is_known"]
            expected_known = q.label == "known"
            is_correct = pred_known == expected_known
            if is_correct:
                correct += 1
            predictions.append(
                {
                    "question": q.text,
                    "predicted": "known" if pred_known else "unknown",
                    "expected": q.label,
                    "correct": is_correct,
                    "score": result["uncertainty_score"],
                }
            )

        accuracy = correct / len(val_questions) if val_questions else 0.0
        detector._unload()

        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "n_correct": correct,
            "n_total": len(val_questions),
        }

    def tune_threshold(
        self,
        train_questions: list,
        calibration_questions: list,
    ) -> float:
        """Tune threshold on calibration set."""
        detector = self.detector_class()
        known = [q.text for q in train_questions if q.label == "known"]
        unknown = [q.text for q in train_questions if q.label == "unknown"]
        detector.calibrate(known, unknown)

        # Collect scores
        scores = []
        labels = []
        for q in calibration_questions:
            result = detector.detect(q.text)
            scores.append(result["uncertainty_score"])
            labels.append(1 if q.label == "known" else 0)

        detector._unload()

        # Grid search threshold
        best_thresh = 0.5
        best_acc = 0.0
        for thresh in np.linspace(0.1, 1.0, 50):
            preds = [1 if s < thresh else 0 for s in scores]
            acc = sum(1 for p, lbl in zip(preds, labels) if p == lbl) / len(labels)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        return float(best_thresh)

    def run(
        self,
        dataset: question signalDataset,
        n_folds: int = 5,
    ) -> CVResult:
        """Run n-fold cross-validation."""
        questions = dataset.questions.copy()
        random.shuffle(questions)

        fold_size = len(questions) // n_folds
        fold_accuracies = []
        fold_thresholds = []
        per_fold_details = []

        for fold in range(n_folds):
            print(f"\n[CV] Fold {fold + 1}/{n_folds}")

            # Split into train and validation
            val_start = fold * fold_size
            val_end = val_start + fold_size
            val = questions[val_start:val_end]
            train = questions[:val_start] + questions[val_end:]

            # Further split train into train/calibration (80/20)
            cal_split = int(len(train) * 0.2)
            random.shuffle(train)
            train_cal = train[cal_split:]
            calibration = train[:cal_split]

            # Tune threshold
            threshold = self.tune_threshold(train_cal, calibration)
            print(f"  Threshold: {threshold:.3f}")

            # Evaluate
            result = self.evaluate_fold(train_cal, val, threshold)
            acc = result["accuracy"]
            print(f"  Accuracy: {acc:.3f} ({result['n_correct']}/{result['n_total']})")

            fold_accuracies.append(acc)
            fold_thresholds.append(threshold)
            per_fold_details.append(
                {
                    "fold": fold + 1,
                    "threshold": threshold,
                    "accuracy": acc,
                    "n_correct": result["n_correct"],
                    "n_total": result["n_total"],
                }
            )

        mean_acc = float(np.mean(fold_accuracies))
        std_acc = float(np.std(fold_accuracies))
        mean_thresh = float(np.mean(fold_thresholds))
        std_thresh = float(np.std(fold_thresholds))

        # 95% CI
        z = 1.96
        ci_low = mean_acc - z * std_acc / np.sqrt(n_folds)
        ci_high = mean_acc + z * std_acc / np.sqrt(n_folds)

        print("\n[CV] Summary:")
        print(f"  Mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"  Threshold: {mean_thresh:.3f} ± {std_thresh:.3f}")
        print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

        return CVResult(
            fold_accuracies=fold_accuracies,
            fold_thresholds=fold_thresholds,
            mean_accuracy=mean_acc,
            std_accuracy=std_acc,
            mean_threshold=mean_thresh,
            std_threshold=std_thresh,
            ci_95_low=ci_low,
            ci_95_high=ci_high,
            per_fold_details=per_fold_details,
        )

    def save(self, result: CVResult, path: str = "cv_result.json") -> None:
        """Save CV results."""
        data = {
            "fold_accuracies": result.fold_accuracies,
            "fold_thresholds": result.fold_thresholds,
            "mean_accuracy": result.mean_accuracy,
            "std_accuracy": result.std_accuracy,
            "mean_threshold": result.mean_threshold,
            "std_threshold": result.std_threshold,
            "ci_95_low": result.ci_95_low,
            "ci_95_high": result.ci_95_high,
            "per_fold_details": result.per_fold_details,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[CV] Results saved to {path}")


if __name__ == "__main__":
    from two_pass_llama_detector import TwoPassLlamaDetector

    ds = question signalDataset(seed=42)
    ds.build()

    cv = CrossValidator(TwoPassLlamaDetector, seed=42)
    result = cv.run(ds, n_folds=5)
    cv.save(result)
