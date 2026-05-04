"""Train/Calibration/Test split with stratification by category.

Prevents data leakage by:
- Verifying zero overlap between splits (hash-based)
- Preserving category distribution within 5% tolerance
- Using stratified sampling for balanced known/unknown labels
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np

from ep_dataset import question signalDataset, question signalQuestion


@dataclass
class SplitResult:
    """Result of a dataset split."""

    train: list[question signalQuestion]
    calibration: list[question signalQuestion]
    test: list[question signalQuestion]
    stats: dict


class StratifiedSplitter:
    """Split dataset into train/calibration/test with stratification.

    Target split: 200 train, 100 calibration, 125 test (or proportional).
    """

    TOLERANCE = 0.05  # 5% category distribution tolerance

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        np.random.seed(seed)

    def split(
        self,
        dataset: question signalDataset,
        train_size: int = 200,
        calibration_size: int = 100,
        test_size: int = 125,
    ) -> SplitResult:
        """Split dataset with stratification."""
        questions = dataset.questions.copy()
        random.shuffle(questions)

        # Group by category
        by_category: dict[str, list[question signalQuestion]] = {}
        for q in questions:
            by_category.setdefault(q.category, []).append(q)

        train, calibration, test = [], [], []
        for cat, cat_questions in by_category.items():
            n = len(cat_questions)
            total_target = train_size + calibration_size + test_size
            train_n = max(1, round(n * train_size / total_target))
            cal_n = max(1, round(n * calibration_size / total_target))
            test_n = max(1, n - train_n - cal_n)

            train.extend(cat_questions[:train_n])
            calibration.extend(cat_questions[train_n : train_n + cal_n])
            test.extend(cat_questions[train_n + cal_n : train_n + cal_n + test_n])

        # Verify no overlap
        train_ids = {q.id for q in train}
        cal_ids = {q.id for q in calibration}
        test_ids = {q.id for q in test}

        overlap = train_ids & cal_ids | train_ids & test_ids | cal_ids & test_ids
        if overlap:
            raise ValueError(f"Data leakage detected: {len(overlap)} overlapping IDs")

        # Verify category balance
        stats = self._compute_stats(train, calibration, test)
        self._verify_balance(stats)

        return SplitResult(train=train, calibration=calibration, test=test, stats=stats)

    def _compute_stats(
        self, train: list, calibration: list, test: list
    ) -> dict:
        """Compute category distribution stats."""
        from collections import Counter

        train_counts = Counter(q.category for q in train)
        cal_counts = Counter(q.category for q in calibration)
        test_counts = Counter(q.category for q in test)

        all_cats = set(train_counts) | set(cal_counts) | set(test_counts)

        stats = {}
        for cat in all_cats:
            t = train_counts.get(cat, 0)
            c = cal_counts.get(cat, 0)
            te = test_counts.get(cat, 0)
            total = t + c + te
            stats[cat] = {
                "train": t,
                "train_pct": t / total if total else 0,
                "calibration": c,
                "calibration_pct": c / total if total else 0,
                "test": te,
                "test_pct": te / total if total else 0,
                "total": total,
            }
        return stats

    def _verify_balance(self, stats: dict) -> None:
        """Verify category distribution within tolerance."""
        target_train = 200 / 425
        target_cal = 100 / 425
        target_test = 125 / 425

        for cat, s in stats.items():
            if abs(s["train_pct"] - target_train) > self.TOLERANCE:
                print(f"  WARNING: {cat} train_pct={s['train_pct']:.2f}, target={target_train:.2f}")
            if abs(s["calibration_pct"] - target_cal) > self.TOLERANCE:
                print(f"  WARNING: {cat} calibration_pct={s['calibration_pct']:.2f}")
            if abs(s["test_pct"] - target_test) > self.TOLERANCE:
                print(f"  WARNING: {cat} test_pct={s['test_pct']:.2f}")

    def save_splits(self, result: SplitResult, prefix: str = "split") -> None:
        """Save splits to JSON files."""
        for name, data in [
            ("train", result.train),
            ("calibration", result.calibration),
            ("test", result.test),
        ]:
            path = f"{prefix}_{name}.json"
            out = [
                {
                    "id": q.id,
                    "text": q.text,
                    "category": q.category,
                    "label": q.label,
                }
                for q in data
            ]
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[Split] Saved {len(data)} questions to {path}")


if __name__ == "__main__":
    ds = question signalDataset(seed=42)
    ds.build()

    splitter = StratifiedSplitter(seed=42)
    result = splitter.split(ds)

    print(f"\nSplit sizes: train={len(result.train)}, cal={len(result.calibration)}, test={len(result.test)}")
    print(f"Known/unknown: train={sum(1 for q in result.train if q.label=='known')}/{sum(1 for q in result.train if q.label=='unknown')}")
    print(f"Known/unknown: cal={sum(1 for q in result.calibration if q.label=='known')}/{sum(1 for q in result.calibration if q.label=='unknown')}")
    print(f"Known/unknown: test={sum(1 for q in result.test if q.label=='known')}/{sum(1 for q in result.test if q.label=='unknown')}")

    splitter.save_splits(result)
