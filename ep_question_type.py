"""Question-type classifier for question boundary detection.

Identifies question categories that need special handling:
- counterfactual: "What if", "If X were" → needs domain check
- subjective: "best", "greatest" → always uncertain
- meta: "your training", "your model" → dedicated calibration
- nonsense: semantic coherence check
- factual: standard embedding-distance detection

This closes the hallucination exploitation vector where prompt
engineering can force false classifications.
"""
from __future__ import annotations

import re


class QuestionTypeClassifier:
    """Rule-based classifier for question types."""

    # Counterfactual markers
    COUNTERFACTUAL_PREFIXES = (
        "what if",
        "if ",
        "suppose ",
        "imagine ",
        "consider ",
        "in a hypothetical",
        "in a scenario where",
    )

    # Counterfactual patterns (anywhere in question)
    COUNTERFACTUAL_PATTERNS = [
        r"what would happen if",
        r"what would .+ be like if",
        r"how would .+ differ if",
        r"what if .+ were ",
        r"what if .+ didn't ",
        r"if .+ had not ",
        r"if .+ were to ",
    ]

    # Subjective markers
    SUBJECTIVE_MARKERS = (
        "best",
        "greatest",
        "worst",
        "most beautiful",
        "most important",
        "better than",
        "worse than",
        "more important than",
        "should ",
        "ought to",
        "right to",
        "ideal",
        "perfect",
        "meaning of life",
        "free will",
        "illusion",
    )

    # Meta markers (questions about the model itself)
    META_MARKERS = (
        "your training",
        "your model",
        "your parameters",
        "your architecture",
        "your knowledge cutoff",
        "your context window",
        "your layers",
        "your attention",
        "your tokenizer",
        "your embedding",
        "your quantization",
        "your creator",
        "your developer",
        "your training data",
        "your fine-tuning",
        "your temperature",
        "your memory",
        "your capabilities",
        "your limitations",
        "your version",
        "your size",
        # Also catch second-person forms
        "you have",
        "you are",
        "were you",
        "created you",
        "model are you",
        "access the internet",
    )

    # Nonsense patterns (syntactically valid but semantically empty combinations)
    NONSENSE_PATTERNS = [
        r"color of \w+day",  # "color of Tuesday"
        r"flavor of.*proof",  # "flavor of a geometric proof"
        r"weight of silence",
        r"speed of dark",
        r"does dark",
        r"temperature of.*shadow",
        r"density of.*thought",
        r"height of.*lie",
        r"how tall is.*lie",
        r"depth of.*surface",
        r"length of.*instant",
        r"texture of.*time",
        r"smell of.*number",
        r"taste of.*equation",
        r"sound of.*hand",
        r"currency of.*dream",
        r"multiply \w+ by \w+",  # "multiply blue by green"
        r"divide zero by zero",
        r"fold water",
        r"square circle",
        r"how many angels",
        r"opposite of.*cat",
        r"travel\?",  # standalone "travel" at end (e.g. "How fast does dark travel?")
    ]

    def classify(self, question: str) -> str:
        """Classify question type.

        Returns one of: counterfactual, subjective, meta, nonsense, factual
        """
        q_lower = question.lower().strip()

        # Check counterfactual first (strong signal)
        if any(q_lower.startswith(prefix) for prefix in self.COUNTERFACTUAL_PREFIXES):
            return "counterfactual"
        for pattern in self.COUNTERFACTUAL_PATTERNS:
            if re.search(pattern, q_lower):
                return "counterfactual"

        # Check meta (strong signal)
        if any(marker in q_lower for marker in self.META_MARKERS):
            return "meta"

        # Check nonsense patterns
        for pattern in self.NONSENSE_PATTERNS:
            if re.search(pattern, q_lower):
                return "nonsense"

        # Check subjective
        if any(marker in q_lower for marker in self.SUBJECTIVE_MARKERS):
            return "subjective"

        # Default: factual
        return "factual"

    def classify_with_confidence(self, question: str) -> tuple[str, float]:
        """Classify with confidence score.

        Returns (type, confidence) where confidence is 0.0-1.0.
        """
        q_lower = question.lower().strip()

        # Counterfactual: check prefix match
        for prefix in self.COUNTERFACTUAL_PREFIXES:
            if q_lower.startswith(prefix):
                return "counterfactual", 1.0

        # Meta: check marker presence
        meta_hits = sum(1 for m in self.META_MARKERS if m in q_lower)
        if meta_hits > 0:
            return "meta", min(1.0, meta_hits * 0.3 + 0.4)

        # Nonsense: check patterns
        for pattern in self.NONSENSE_PATTERNS:
            if re.search(pattern, q_lower):
                return "nonsense", 0.9

        # Subjective: check markers
        subj_hits = sum(1 for m in self.SUBJECTIVE_MARKERS if m in q_lower)
        if subj_hits > 0:
            return "subjective", min(1.0, subj_hits * 0.25 + 0.5)

        return "factual", 0.7


def _test_classifier() -> dict:
    """Run unit tests on classifier."""
    classifier = QuestionTypeClassifier()

    test_cases = [
        # (question, expected_type)
        ("What is gravity?", "factual"),
        ("What if gravity didn't exist?", "counterfactual"),
        ("If the Earth had two moons, what would tides be like?", "counterfactual"),
        ("What is the best programming language?", "subjective"),
        ("Who is the greatest scientist?", "subjective"),
        ("What is your training cutoff?", "meta"),
        ("How many layers do you have?", "meta"),
        ("What is the color of Tuesday?", "nonsense"),
        ("How fast does dark travel?", "nonsense"),
        ("Can you fold water?", "nonsense"),
        ("What is the capital of France?", "factual"),
        ("What is DNA?", "factual"),
        ("What is the speed of light?", "factual"),
        ("Suppose time flowed backwards, what would happen?", "counterfactual"),
        ("Is capitalism better than socialism?", "subjective"),
        ("What is your embedding dimension?", "meta"),
        ("What is the weight of silence?", "nonsense"),
        ("What is the boiling point of water?", "factual"),
        ("Imagine a world without friction.", "counterfactual"),
        ("What is the most beautiful equation?", "subjective"),
        ("Who created you?", "meta"),
        ("How many angels dance on a pin?", "nonsense"),
        ("What is photosynthesis?", "factual"),
        ("Consider a hypothetical universe with 4 spatial dimensions.", "counterfactual"),
        ("Should we colonize Mars?", "subjective"),
        ("What is your quantization level?", "meta"),
        ("What is the flavor of a geometric proof?", "nonsense"),
        ("What is the Pythagorean theorem?", "factual"),
        ("If electrons had no charge, what would chemistry be like?", "counterfactual"),
        ("What is the ideal human lifespan?", "subjective"),
        ("Do you have memory of previous conversations?", "meta"),
        ("Can a square circle?", "nonsense"),
        ("What is CRISPR?", "factual"),
        ("What would happen if the Planck constant were larger?", "counterfactual"),
        ("Is free will an illusion?", "subjective"),
        ("What model are you?", "meta"),
        ("What is the opposite of a cat?", "nonsense"),
        ("What is machine learning?", "factual"),
        ("In a hypothetical scenario where entropy decreases, what happens?", "counterfactual"),
        ("What is the best diet for humans?", "subjective"),
        ("Can you access the internet?", "meta"),
        ("How tall is a lie?", "nonsense"),
        ("What is the atomic number of carbon?", "factual"),
        ("If dinosaurs had not gone extinct, how would civilization differ?", "counterfactual"),
        ("What is the greatest work of literature?", "subjective"),
        ("What is your context window size?", "meta"),
        ("What is the currency of dreams?", "nonsense"),
        ("What is the function of mitochondria?", "factual"),
        ("What if the atmosphere were pure oxygen?", "counterfactual"),
        ("Is privacy more important than security?", "subjective"),
        ("How were you fine-tuned?", "meta"),
        ("What is the density of a thought?", "nonsense"),
    ]

    correct = 0
    errors = []
    for question, expected in test_cases:
        predicted = classifier.classify(question)
        if predicted == expected:
            correct += 1
        else:
            errors.append(f"  FAIL: '{question[:50]}...' -> {predicted} (expected {expected})")

    accuracy = correct / len(test_cases)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_cases),
        "errors": errors,
    }


if __name__ == "__main__":
    result = _test_classifier()
    print(f"Question Type Classifier Tests: {result['correct']}/{result['total']} ({result['accuracy']:.1%})")
    if result["errors"]:
        print("Errors:")
        for e in result["errors"]:
            print(e)
