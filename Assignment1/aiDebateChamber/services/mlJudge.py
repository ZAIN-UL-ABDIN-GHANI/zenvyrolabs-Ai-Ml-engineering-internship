"""Regression judge for the Autonomous AI Debate Chamber.

This module provides two pieces of functionality:
- feature extraction for debate text
- a scikit-learn regression judge that scores each argument and compares two
  arguments to determine a winner

The training schema matches the bundled CSV at data/historical_debates.csv.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)

FEATURE_COLUMNS = [
    "word_count",
    "sentence_count",
    "average_sentence_length",
    "unique_words",
    "lexical_diversity",
    "question_count",
    "average_word_length",
    "punctuation_density",
    "readability_estimate",
    "argument_length",
    "complexity_score",
    "persuasiveness",
]
TARGET_COLUMN = "human_score"

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_PUNCTUATION_RE = re.compile(r"[.,!?;:()\-\"'\[\]{}]")
_VOWELS = set("aeiouy")

_POSITIVE_MARKERS = {
    "evidence",
    "clearly",
    "undoubtedly",
    "therefore",
    "because",
    "consistently",
    "advantage",
    "rigor",
    "strong",
    "must",
    "should",
    "important",
    "benefit",
    "benefits",
    "improve",
    "effective",
}
_NEGATIVE_MARKERS = {
    "maybe",
    "perhaps",
    "uncertain",
    "weak",
    "could",
    "might",
    "guess",
    "unsure",
    "doubt",
    "unclear",
}
_COMPLEXITY_MARKERS = {
    "however",
    "although",
    "moreover",
    "furthermore",
    "nevertheless",
    "consequently",
    "subsequently",
    "therefore",
    "whereas",
    "nonetheless",
    "undoubtedly",
    "ultimately",
}


def _safe_divide(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")


def _count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    groups = 0
    previous_was_vowel = False
    for character in word:
        is_vowel = character in _VOWELS
        if is_vowel and not previous_was_vowel:
            groups += 1
        previous_was_vowel = is_vowel
    if word.endswith("e") and groups > 1:
        groups -= 1
    return max(groups, 1)


def _flesch_reading_ease(words: List[str], sentence_count: int) -> float:
    if not words:
        return 0.0
    syllables = sum(_count_syllables(word) for word in words)
    words_per_sentence = _safe_divide(len(words), max(sentence_count, 1))
    syllables_per_word = _safe_divide(syllables, len(words))
    return round(206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word, 2)


def _score_complexity(
    *,
    words: List[str],
    sentence_count: int,
    average_sentence_length: float,
    average_word_length: float,
    punctuation_density: float,
) -> float:
    complex_markers = sum(1 for word in words if word.lower() in _COMPLEXITY_MARKERS)
    long_words = sum(1 for word in words if len(word) >= 8)
    nested_clause_bonus = 1.2 if sentence_count >= 4 else 0.0
    score = (
        1.2
        + 0.25 * average_sentence_length
        + 0.35 * average_word_length
        + 0.6 * complex_markers
        + 0.15 * long_words
        + 14.0 * punctuation_density
        + nested_clause_bonus
    )
    return round(max(0.5, min(score, 24.0)), 3)


def _score_persuasiveness(
    words: List[str],
    text: str,
    average_sentence_length: float,
    question_count: int,
) -> float:
    lower_words = [word.lower() for word in words]
    positive_hits = sum(1 for word in lower_words if word in _POSITIVE_MARKERS)
    negative_hits = sum(1 for word in lower_words if word in _NEGATIVE_MARKERS)
    assertive_language = sum(1 for word in lower_words if word in {"must", "will", "should", "clearly", "evidence"})
    emphasis_bonus = text.count("!") * 0.5 + question_count * 0.2

    score = (
        0.8
        + 0.85 * positive_hits
        + 0.35 * assertive_language
        + 0.25 * max(average_sentence_length / 5.0, 0.0)
        + emphasis_bonus
        - 0.45 * negative_hits
    )
    return round(max(0.0, min(score, 10.0)), 3)


def extract_features(text: str) -> Dict[str, float]:
    """Extract the training features used by the regression judge."""
    cleaned_text = text or ""
    words = _tokenize(cleaned_text)
    word_count = len(words)
    sentence_parts = [part.strip() for part in _SENTENCE_RE.findall(cleaned_text) if part.strip()]
    sentence_count = max(len(sentence_parts), 1 if cleaned_text.strip() else 0)
    unique_words = len({word.lower() for word in words})
    lexical_diversity = round(_safe_divide(unique_words, word_count), 3)
    average_sentence_length = round(_safe_divide(word_count, sentence_count), 3)
    average_word_length = round(_safe_divide(sum(len(word) for word in words), word_count), 3)
    question_count = cleaned_text.count("?")
    punctuation_count = len(_PUNCTUATION_RE.findall(cleaned_text))
    punctuation_density = round(_safe_divide(punctuation_count, max(len(cleaned_text), 1)), 4)
    readability_estimate = _flesch_reading_ease(words, sentence_count or 1)
    argument_length = len(cleaned_text)

    complexity_score = _score_complexity(
        words=words,
        sentence_count=sentence_count,
        average_sentence_length=average_sentence_length,
        average_word_length=average_word_length,
        punctuation_density=punctuation_density,
    )
    persuasiveness = _score_persuasiveness(words, cleaned_text, average_sentence_length, question_count)

    return {
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "average_sentence_length": float(average_sentence_length),
        "unique_words": float(unique_words),
        "lexical_diversity": float(lexical_diversity),
        "question_count": float(question_count),
        "average_word_length": float(average_word_length),
        "punctuation_density": float(punctuation_density),
        "readability_estimate": float(readability_estimate),
        "argument_length": float(argument_length),
        "complexity_score": float(complexity_score),
        "persuasiveness": float(persuasiveness),
    }


@dataclass
class DebatePrediction:
    score: float
    features: Dict[str, float]


class DebateRegressionJudge:
    """Train and apply a regression model over debate text features."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        self.data_path = data_path or os.path.join(_PROJECT_ROOT, "data", "historical_debates.csv")
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=250,
            random_state=self.random_state,
            min_samples_leaf=2,
        )
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}

        if os.path.exists(self.data_path):
            try:
                self.train_model(self.data_path)
            except Exception:
                self.is_trained = False

    def train_model(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """Train the regressor using the bundled historical debate CSV."""
        path = dataset_path or self.data_path
        if not os.path.isabs(path):
            path = os.path.join(_PROJECT_ROOT, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training dataset not found: {path}")

        frame = pd.read_csv(path)
        missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in frame.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
        if len(frame) < 5:
            raise ValueError("Training dataset must contain at least 5 rows.")

        features = frame[FEATURE_COLUMNS]
        target = frame[TARGET_COLUMN]
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=self.random_state,
        )

        self.model.fit(x_train, y_train)
        predictions = self.model.predict(x_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = math.sqrt(mse)
        metrics = {
            "rows": int(len(frame)),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2_score(y_test, predictions)), 4),
        }
        self.is_trained = True
        self.metrics = metrics
        return metrics

    def predict_score(self, text: str) -> DebatePrediction:
        """Predict the human score for a single debate argument."""
        if not self.is_trained:
            self.train_model(self.data_path)

        features = extract_features(text)
        frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        raw_score = float(self.model.predict(frame)[0])
        clamped_score = round(max(1.0, min(10.0, raw_score)), 2)
        return DebatePrediction(score=clamped_score, features=features)

    def evaluate_debate(self, advocate_text: str, challenger_text: str) -> Dict[str, Any]:
        """Score both sides and return a verdict payload."""
        advocate = self.predict_score(advocate_text)
        challenger = self.predict_score(challenger_text)

        if advocate.score > challenger.score:
            winner = "Agent A"
        elif challenger.score > advocate.score:
            winner = "Agent B"
        else:
            winner = "Tie"

        return {
            "status": "evaluated",
            "winner": winner,
            "advocate_score": advocate.score,
            "challenger_score": challenger.score,
            "margin": round(abs(advocate.score - challenger.score), 2),
            "advocate_features": advocate.features,
            "challenger_features": challenger.features,
            "training_metrics": self.metrics,
        }


__all__ = [
    "DebatePrediction",
    "DebateRegressionJudge",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "extract_features",
]