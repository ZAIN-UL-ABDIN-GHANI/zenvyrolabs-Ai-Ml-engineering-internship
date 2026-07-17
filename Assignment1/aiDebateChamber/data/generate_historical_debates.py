"""
data/generate_historical_debates.py

Generates data/historical_debates.csv: a synthetic dataset of historical
debate-argument statistics used to train the RandomForestRegressor in
services/mlJudge.py.

Why synthetic data
-------------------
No real corpus of debate transcripts with human-assigned quality scores was
available for this project. Rather than leave the dataset empty (which
would make Task 6's regression model untrainable), this script generates
statistically realistic rows so the full ML pipeline -- load, split, train,
predict -- can be exercised end-to-end and demonstrated.

Methodology
-----------
1. Each row starts from a hidden "rhetorical quality" latent value drawn
   from a Beta(5, 3) distribution scaled to 0-10 (skewed toward the
   middle/upper range, similar to how most real debate turns are
   "reasonably competent" rather than uniformly random).
2. Surface features (word_count, sentence structure, vocabulary richness,
   punctuation, etc.) are sampled from ranges typical of a single debate
   turn (roughly 30-380 words, 8-32 words/sentence).
3. `complexity_score` is computed with the *exact same formula* as
   `extract_complexity()` in services/mlJudge.py (0.4 * avg_sentence_length
   + 1.5 * avg_word_length + 5 * lexical_diversity), so synthetic rows are
   internally consistent with what live text would produce for that
   feature.
4. `human_score` (the regression target) is a noisy weighted combination of
   the latent quality, complexity, persuasiveness, and vocabulary richness,
   with a small penalty for sentences that are too short or too long. This
   gives the RandomForestRegressor real, learnable signal rather than pure
   noise, while still leaving meaningful variance (measurement noise is
   added explicitly).
5. `winner` is a synthetic per-row descriptive label (Agent A / Agent B),
   derived from human_score plus noise. It is NOT used as a model feature
   or target -- it exists only to satisfy the dataset schema and for
   human-readable inspection of the CSV.

Run directly to (re)generate the CSV:
    python data/generate_historical_debates.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)  # fixed seed -> reproducible dataset
N_ROWS = 600
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_debates.csv")


def _generate_row() -> dict:
    """Synthesize one historical-debate-turn row of engineered features + label."""
    # Hidden ground-truth quality for this historical turn.
    latent_quality = RNG.beta(5, 3) * 10  # mean ~6.25, bounded [0, 10]

    # Wide range so the model sees everything from one-line non-answers to
    # long, developed arguments -- without short inputs, the regressor never
    # learns that brevity should hurt the score, and extrapolates poorly.
    word_count = int(np.clip(RNG.normal(140, 65), 3, 380))
    words_per_sentence = float(np.clip(RNG.normal(17, 4), 6, 32))
    sentence_count = max(1, round(word_count / words_per_sentence))
    average_sentence_length = round(word_count / sentence_count, 3)

    lexical_diversity = float(np.clip(RNG.normal(0.62, 0.12), 0.25, 0.95))
    unique_words = int(round(lexical_diversity * word_count))

    question_count = int(RNG.poisson(1.1))
    average_word_length = float(np.clip(RNG.normal(5.1, 0.7), 3.2, 8.0))
    punctuation_density = float(np.clip(RNG.normal(0.055, 0.02), 0.01, 0.15))

    # Mirrors extract_readability()'s Flesch approximation, using a sampled
    # syllables-per-word figure in place of exact per-word syllable counts.
    approx_syllables_per_word = float(np.clip(RNG.normal(1.55, 0.25), 1.1, 2.3))
    readability_estimate = round(
        float(np.clip(206.835 - 1.015 * words_per_sentence - 84.6 * approx_syllables_per_word, 0, 100)), 2
    )

    argument_length = int(round(word_count * (average_word_length + 1) + RNG.normal(0, 15)))
    argument_length = max(argument_length, word_count)

    # Must mirror extract_complexity() in services/mlJudge.py exactly.
    complexity_score = round(
        0.4 * average_sentence_length + 1.5 * average_word_length + 5 * lexical_diversity, 3
    )

    persuasiveness = float(np.clip(RNG.normal(latent_quality * 0.7, 1.5), 0, 10))

    # Mild penalty for sentences that are unusually short (choppy) or long (rambling).
    length_penalty = -0.03 * abs(average_sentence_length - 18)

    # A one-word non-answer shouldn't be able to out-score a developed
    # paragraph just because a trivial vocabulary ratio inflates
    # lexical_diversity/complexity_score. `substance` ramps 0 -> 1 as
    # word_count grows from ~0 to ~65 words and gates how much of the
    # "quality" terms apply, so brevity is penalized directly and explicitly
    # rather than left for the model to guess at from out-of-range inputs.
    substance = float(np.clip(np.log1p(word_count) / np.log1p(65), 0.0, 1.0))

    quality_terms = (
        0.45 * latent_quality
        + 0.15 * complexity_score
        + 0.25 * persuasiveness
        + 0.10 * (lexical_diversity * 10)
    )
    score_raw = substance * quality_terms + length_penalty + RNG.normal(0, 0.6)
    human_score = round(float(np.clip(score_raw, 1.0, 10.0)), 2)

    # Threshold ~ empirical mean of human_score so the label lands close to
    # a 50/50 split rather than skewing toward one side.
    winner_roll = human_score + RNG.normal(0, 0.8)
    winner = "Agent A" if winner_roll >= 6.98 else "Agent B"

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_sentence_length": average_sentence_length,
        "unique_words": unique_words,
        "lexical_diversity": round(lexical_diversity, 3),
        "question_count": question_count,
        "average_word_length": round(average_word_length, 3),
        "punctuation_density": round(punctuation_density, 4),
        "readability_estimate": readability_estimate,
        "argument_length": argument_length,
        "complexity_score": complexity_score,
        "persuasiveness": round(persuasiveness, 3),
        "human_score": human_score,
        "winner": winner,
    }


def main() -> None:
    """Generate N_ROWS synthetic rows and write them to historical_debates.csv."""
    rows = [_generate_row() for _ in range(N_ROWS)]
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    print(df.describe(include="all").to_string())


if __name__ == "__main__":
    main()
