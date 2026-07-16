"""
services/mlJudge.py

Machine-learning scoring service for the Autonomous AI Debate Chamber.

Responsibilities
-----------------
1. Convert raw debate-argument text into a fixed set of numeric linguistic
   features (word count, lexical diversity, readability, etc.).
2. Train a RandomForestRegressor on `data/historical_debates.csv` to predict
   a human-style quality score (1-10) from those features.
3. Score live debate turns and decide a winner between the advocate (Agent A)
   and challenger (Agent B) sides.

Design note on feature consistency
-----------------------------------
`extract_features()` is the single source of truth for turning text into a
feature vector. It is used both by `data/generate_historical_debates.py`
(indirectly, via the same formulas) and by `predict_score()` at inference
time, so the regressor always sees the same schema, in the same order, that
it was trained on. Mismatched train/predict feature sets is the most common
way this kind of pipeline silently breaks, so `FEATURE_COLUMNS` is the only
place that ordering is defined.
"""
from __future__ import annotations

import logging
import os
import pickle
import re
from typing import Dict, List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Canonical feature order. Training (train_model) and inference (predict_score)
# both select columns using this exact list, so column order can never drift
# between the two.
FEATURE_COLUMNS: List[str] = [
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


class DatasetError(Exception):
    """Raised when the historical dataset is missing, unreadable, or malformed."""


class ModelNotTrainedError(Exception):
    """Raised when a prediction is requested before any model is trained or loaded."""


# ---------------------------------------------------------------------------
# Task 7 - Feature extraction functions.
# Each function is intentionally small, pure, and independently testable.
# ---------------------------------------------------------------------------

def extract_word_count(text: str) -> int:
    """Number of whitespace-delimited tokens in the text."""
    return len(text.split())


def _split_sentences(text: str) -> List[str]:
    """Split on ., !, ? (one or more) and drop empty fragments."""
    fragments = re.split(r"[.!?]+", text)
    return [s.strip() for s in fragments if s.strip()]


def extract_sentence_count(text: str) -> int:
    """
    Count sentences via punctuation splitting. Text with no terminal
    punctuation still counts as one sentence; empty text counts as zero.
    """
    if not text.strip():
        return 0
    return max(len(_split_sentences(text)), 1)


def extract_average_sentence_length(text: str) -> float:
    """Mean words per sentence: word_count / sentence_count."""
    sentence_count = extract_sentence_count(text)
    if sentence_count == 0:
        return 0.0
    return round(extract_word_count(text) / sentence_count, 3)


def _tokenize_words(text: str) -> List[str]:
    """Lowercased alphanumeric word tokens with surrounding punctuation stripped."""
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def extract_unique_words(text: str) -> int:
    """Count of distinct case-insensitive word tokens (vocabulary size)."""
    return len(set(_tokenize_words(text)))


def extract_lexical_diversity(text: str) -> float:
    """
    Type-token ratio: unique_words / word_count.
    A higher ratio (closer to 1.0) means less word repetition / richer
    vocabulary; a lower ratio means more repetitive phrasing.
    """
    words = _tokenize_words(text)
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 3)


def extract_question_count(text: str) -> int:
    """Number of '?' characters, used as a proxy for rhetorical questioning."""
    return text.count("?")


def extract_average_word_length(text: str) -> float:
    """Mean character length of word tokens (punctuation excluded)."""
    words = _tokenize_words(text)
    if not words:
        return 0.0
    return round(sum(len(w) for w in words) / len(words), 3)


def extract_punctuation_density(text: str) -> float:
    """
    Ratio of punctuation characters to total characters. Higher density can
    indicate more structured, emphatic, or clause-heavy phrasing.
    """
    if not text:
        return 0.0
    punctuation_chars = re.findall(r'[.,;:!?\-\'"()]', text)
    return round(len(punctuation_chars) / len(text), 4)


def _count_syllables(word: str) -> int:
    """
    Heuristic syllable counter (no phonetic dictionary available offline):
    counts vowel-group transitions, drops a trailing silent 'e', and floors
    at one syllable per word. This is the standard approximation used when
    computing Flesch-style readability without a CMUdict-style lookup.
    """
    word = word.lower()
    vowels = "aeiouy"
    syllables = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            syllables += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(syllables, 1)


def extract_readability(text: str) -> float:
    """
    Approximate Flesch Reading Ease:
        206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    Higher = easier to read. Clipped to [0, 100] since the raw formula can
    exceed that range on very short or very dense inputs.
    """
    words = _tokenize_words(text)
    sentence_count = extract_sentence_count(text)
    if not words or sentence_count == 0:
        return 0.0
    syllable_count = sum(_count_syllables(w) for w in words)
    score = 206.835 - 1.015 * (len(words) / sentence_count) - 84.6 * (syllable_count / len(words))
    return round(max(0.0, min(100.0, score)), 2)


def extract_argument_length(text: str) -> int:
    """Total character length of the raw argument text."""
    return len(text)


def extract_complexity(text: str) -> float:
    """
    Composite linguistic-complexity heuristic:
        0.4 * average_sentence_length + 1.5 * average_word_length + 5 * lexical_diversity
    This is a deliberately simple, bounded, monotonic proxy for "how
    sophisticated the argument reads" (longer sentences, longer words, and
    richer vocabulary all push it up). It is NOT a validated readability
    metric on its own -- extract_readability() covers that -- it exists
    purely to give the regressor an aggregate complexity signal.
    NOTE: data/generate_historical_debates.py mirrors this exact formula so
    the synthetic training rows are internally consistent with live text.
    """
    avg_sentence_length = extract_average_sentence_length(text)
    avg_word_length = extract_average_word_length(text)
    diversity = extract_lexical_diversity(text)
    score = 0.4 * avg_sentence_length + 1.5 * avg_word_length + 5 * diversity
    return round(score, 3)


# Heuristic markers associated with persuasive / rhetorical debate language.
_PERSUASIVE_MARKERS = (
    "must", "should", "clearly", "undoubtedly", "obviously", "in fact",
    "the truth is", "everyone knows", "proven", "evidence shows", "imagine",
    "we cannot ignore", "it is essential", "without question", "critically",
    "simply put", "make no mistake",
)


def extract_persuasiveness(text: str) -> float:
    """
    Heuristic persuasiveness score (roughly 0-10): density of persuasive /
    certainty language, exclamation marks, and rhetorical questions,
    normalized by argument length so short and long arguments are
    comparable. This is a lexical heuristic, not a semantic or fact-checked
    measure of how convincing the argument actually is.
    """
    lowered = text.lower()
    marker_hits = sum(lowered.count(marker) for marker in _PERSUASIVE_MARKERS)
    exclamations = text.count("!")
    questions = extract_question_count(text)
    word_count = max(extract_word_count(text), 1)
    raw_score = (marker_hits * 2 + exclamations + questions) / word_count * 100
    return round(min(raw_score, 10.0), 3)


def extract_features(text: str) -> Dict[str, float]:
    """
    Run the full feature battery over one piece of debate text and return a
    flat dict keyed exactly by FEATURE_COLUMNS. This is the single function
    both training-data generation and live prediction should reason about.
    """
    return {
        "word_count": extract_word_count(text),
        "sentence_count": extract_sentence_count(text),
        "average_sentence_length": extract_average_sentence_length(text),
        "unique_words": extract_unique_words(text),
        "lexical_diversity": extract_lexical_diversity(text),
        "question_count": extract_question_count(text),
        "average_word_length": extract_average_word_length(text),
        "punctuation_density": extract_punctuation_density(text),
        "readability_estimate": extract_readability(text),
        "argument_length": extract_argument_length(text),
        "complexity_score": extract_complexity(text),
        "persuasiveness": extract_persuasiveness(text),
    }


# ---------------------------------------------------------------------------
# Task 6 - Regression judge.
# ---------------------------------------------------------------------------

class DebateRegressionJudge:
    """
    Scores debate arguments on a 1-10 scale using a RandomForestRegressor
    trained on historical (synthetic) debate statistics, and turns a pair of
    argument texts into a scored winner verdict for the Flask API layer.
    """

    def __init__(self, model_path: str = "models/debate_judge_model.pkl") -> None:
        """
        Args:
            model_path: where the trained model is persisted with pickle.
                If a model already exists at this path it is loaded eagerly
                so the API can serve predictions without requiring a
                training call on every process restart.
        """
        self.model_path = model_path
        self.model: Optional[RandomForestRegressor] = None
        self._try_load_existing_model()

    def _try_load_existing_model(self) -> None:
        """Best-effort load of a previously-trained model from disk."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("Loaded existing regression model from %s", self.model_path)
            except Exception as exc:  # noqa: BLE001 - defensive, we degrade gracefully
                logger.warning("Could not load persisted model at %s (%s); will require retraining.",
                                self.model_path, exc)
                self.model = None

    @staticmethod
    def _resolve_dataset_path(dataset_path: str) -> str:
        """
        Accepts either 'data/historical_debates.csv' or a bare
        'historical_debates.csv' and resolves whichever actually exists on
        disk, since different callers may pass either convention.
        """
        candidates = [dataset_path]
        if dataset_path == "data/historical_debates.csv":
            candidates.append("historical_debates.csv")
        elif dataset_path == "historical_debates.csv":
            candidates.append("data/historical_debates.csv")
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise DatasetError(
            f"Historical dataset not found. Tried: {candidates}. "
            f"Run `python data/generate_historical_debates.py` first."
        )

    def train_model(self, dataset_path: str = "data/historical_debates.csv") -> Dict[str, object]:
        """
        Load the historical dataset, split 80/20, train a
        RandomForestRegressor to predict human_score from FEATURE_COLUMNS,
        persist the model to disk, and return training metrics.

        Raises:
            DatasetError: dataset missing, unreadable, or missing required columns.
        """
        resolved_path = self._resolve_dataset_path(dataset_path)

        try:
            df = pd.read_csv(resolved_path)
        except Exception as exc:
            raise DatasetError(f"Failed to read dataset at '{resolved_path}': {exc}") from exc

        required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        missing_columns = [c for c in required_columns if c not in df.columns]
        if missing_columns:
            raise DatasetError(f"Dataset at '{resolved_path}' is missing required columns: {missing_columns}")

        if len(df) < 10:
            raise DatasetError(f"Dataset at '{resolved_path}' has only {len(df)} rows; need at least 10 to train.")

        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics: Dict[str, object] = {
            "status": "success",
            "mse": round(float(mean_squared_error(y_test, predictions)), 4),
            "r2_score": round(float(r2_score(y_test, predictions)), 4),
            "n_train_samples": int(len(X_train)),
            "n_test_samples": int(len(X_test)),
            "features_used": FEATURE_COLUMNS,
            "dataset_path": resolved_path,
        }

        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

        self.model = model
        logger.info("Model trained on %s: mse=%s r2=%s", resolved_path, metrics["mse"], metrics["r2_score"])
        return metrics

    def predict_score(self, text: str) -> float:
        """
        Extract features from raw argument text and predict a score in
        [1, 10] using the trained RandomForestRegressor.

        Raises:
            ModelNotTrainedError: no trained/loaded model is available.
            ValueError: text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot score empty argument text.")
        if self.model is None:
            self._try_load_existing_model()
        if self.model is None:
            raise ModelNotTrainedError(
                "No trained regression model is available yet. "
                "Call POST /api/machine-learning/train first."
            )

        features = extract_features(text)
        feature_frame = pd.DataFrame([features])[FEATURE_COLUMNS]
        raw_score = float(self.model.predict(feature_frame)[0])
        clipped_score = max(1.0, min(10.0, raw_score))
        return round(clipped_score, 2)

    def evaluate_debate(self, advocate_text: str, challenger_text: str) -> Dict[str, object]:
        """
        Score both sides of a debate turn/transcript and pick a winner.
        `advocate_text` corresponds to Agent A's argument(s); `challenger_text`
        corresponds to Agent B's argument(s).
        """
        advocate_score = self.predict_score(advocate_text)
        challenger_score = self.predict_score(challenger_text)
        if advocate_score > challenger_score:
            winner = "Advocate"
        elif challenger_score > advocate_score:
            winner = "Challenger"
        else:
            winner = "Tie"
        return {
            "advocate_score": advocate_score,
            "challenger_score": challenger_score,
            "winner": winner,
        }
