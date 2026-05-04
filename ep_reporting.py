"""Statistical reporting for question boundary detection.

Produces:
- 95% confidence intervals on accuracy
- Per-class FPR/FNR
- Calibration curves
- Expected Calibration Error (ECE)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class StatisticalReport:
    """Full statistical report."""

    n_total: int
    accuracy: float
    ci_95_low: float
    ci_95_high: float
    fpr: float
    fnr: float
    precision: float
    recall: float
    f1: float
    ece: float
    per_category: dict[str, dict]
    calibration_bins: list[dict]


class StatisticalReporter:
    """Generate statistical reports from detection results."""

    def __init__(self) -> None:
        pass

    def compute_metrics(
        self,
        predictions: list[dict],
    ) -> dict[str, float]:
        """Compute basic metrics from predictions."""
        n = len(predictions)
        correct = sum(1 for p in predictions if p["correct"])
        accuracy = correct / n if n else 0.0

        # Known = positive class
        tp = sum(1 for p in predictions if p["expected"] == "known" and p["predicted"] == "known")
        fp = sum(1 for p in predictions if p["expected"] == "unknown" and p["predicted"] == "known")
        tn = sum(1 for p in predictions if p["expected"] == "unknown" and p["predicted"] == "unknown")
        fn = sum(1 for p in predictions if p["expected"] == "known" and p["predicted"] == "unknown")

        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            "accuracy": accuracy,
            "fpr": fpr,
            "fnr": fnr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    def compute_ci(
        self,
        accuracy: float,
        n: int,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Compute confidence interval using Wilson score interval."""
        if n == 0:
            return 0.0, 0.0

        z = 1.96 if confidence == 0.95 else 2.576
        p = accuracy

        # Wilson score interval
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom

        return max(0.0, center - margin), min(1.0, center + margin)

    def compute_ece(
        self,
        predictions: list[dict],
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error.

        ECE = sum(bin_weight * |accuracy - confidence|)
        """
        # Convert to binary: known=1, unknown=0
        confidences = []
        accuracies = []
        for p in predictions:
            # Use inverse score as confidence (lower score = higher confidence in "known")
            score = p.get("score", 0.5)
            conf = max(0.0, min(1.0, 1.0 - score))
            confidences.append(conf)
            accuracies.append(1.0 if p["correct"] else 0.0)

        conf_arr = np.array(confidences)
        acc_arr = np.array(accuracies)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            in_bin = (conf_arr > low) & (conf_arr <= high)
            if i == 0:
                in_bin = (conf_arr >= low) & (conf_arr <= high)

            bin_size = np.sum(in_bin)
            if bin_size > 0:
                bin_acc = np.mean(acc_arr[in_bin])
                bin_conf = np.mean(conf_arr[in_bin])
                ece += (bin_size / len(predictions)) * abs(bin_acc - bin_conf)

        return float(ece)

    def compute_calibration_bins(
        self,
        predictions: list[dict],
        n_bins: int = 10,
    ) -> list[dict]:
        """Compute calibration bins for plotting."""
        confidences = []
        accuracies = []
        for p in predictions:
            score = p.get("score", 0.5)
            conf = max(0.0, min(1.0, 1.0 - score))
            confidences.append(conf)
            accuracies.append(1.0 if p["correct"] else 0.0)

        conf_arr = np.array(confidences)
        acc_arr = np.array(accuracies)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = []

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            in_bin = (conf_arr > low) & (conf_arr <= high)
            if i == 0:
                in_bin = (conf_arr >= low) & (conf_arr <= high)

            bin_size = int(np.sum(in_bin))
            if bin_size > 0:
                bin_acc = float(np.mean(acc_arr[in_bin]))
                bin_conf = float(np.mean(conf_arr[in_bin]))
                bins.append({
                    "bin": i,
                    "range": [float(low), float(high)],
                    "n": bin_size,
                    "accuracy": bin_acc,
                    "confidence": bin_conf,
                    "gap": abs(bin_acc - bin_conf),
                })

        return bins

    def per_category_metrics(
        self,
        predictions: list[dict],
    ) -> dict[str, dict]:
        """Compute metrics per category."""
        from collections import defaultdict

        by_category = defaultdict(list)
        for p in predictions:
            cat = p.get("category", "unknown")
            by_category[cat].append(p)

        result = {}
        for cat, cat_preds in by_category.items():
            metrics = self.compute_metrics(cat_preds)
            n = len(cat_preds)
            ci_low, ci_high = self.compute_ci(metrics["accuracy"], n)
            result[cat] = {
                **metrics,
                "n": n,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
            }

        return result

    def generate_report(
        self,
        predictions: list[dict],
    ) -> StatisticalReport:
        """Generate full statistical report."""
        metrics = self.compute_metrics(predictions)
        n = len(predictions)
        ci_low, ci_high = self.compute_ci(metrics["accuracy"], n)
        ece = self.compute_ece(predictions)
        bins = self.compute_calibration_bins(predictions)
        per_cat = self.per_category_metrics(predictions)

        return StatisticalReport(
            n_total=n,
            accuracy=metrics["accuracy"],
            ci_95_low=ci_low,
            ci_95_high=ci_high,
            fpr=metrics["fpr"],
            fnr=metrics["fnr"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            ece=ece,
            per_category=per_cat,
            calibration_bins=bins,
        )

    def save_report(
        self,
        report: StatisticalReport,
        path: str = "statistical_report.json",
    ) -> None:
        """Save report to JSON."""
        data = {
            "n_total": report.n_total,
            "accuracy": report.accuracy,
            "ci_95_low": report.ci_95_low,
            "ci_95_high": report.ci_95_high,
            "fpr": report.fpr,
            "fnr": report.fnr,
            "precision": report.precision,
            "recall": report.recall,
            "f1": report.f1,
            "ece": report.ece,
            "per_category": report.per_category,
            "calibration_bins": report.calibration_bins,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Reporting] Saved report to {path}")

    def print_summary(self, report: StatisticalReport) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("STATISTICAL REPORT")
        print("=" * 70)
        print(f"Total questions: {report.n_total}")
        print(f"Accuracy: {report.accuracy:.3f}")
        print(f"95% CI: [{report.ci_95_low:.3f}, {report.ci_95_high:.3f}]")
        print(f"FPR: {report.fpr:.3f} | FNR: {report.fnr:.3f}")
        print(f"Precision: {report.precision:.3f} | Recall: {report.recall:.3f} | F1: {report.f1:.3f}")
        print(f"ECE: {report.ece:.3f}")
        print("\nPer-category:")
        for cat, m in report.per_category.items():
            print(f"  {cat:25} n={m['n']:3} acc={m['accuracy']:.3f} fpr={m['fpr']:.3f} fnr={m['fnr']:.3f}")


if __name__ == "__main__":
    # Demo with synthetic data
    predictions = []
    for i in range(100):
        is_known = i < 50
        score = 0.2 + np.random.random() * 0.3 if is_known else 0.6 + np.random.random() * 0.3
        pred_known = score < 0.5
        predictions.append({
            "question": f"Q{i}",
            "expected": "known" if is_known else "unknown",
            "predicted": "known" if pred_known else "unknown",
            "correct": pred_known == is_known,
            "score": score,
            "category": "test",
        })

    reporter = StatisticalReporter()
    report = reporter.generate_report(predictions)
    reporter.print_summary(report)
    reporter.save_report(report)
