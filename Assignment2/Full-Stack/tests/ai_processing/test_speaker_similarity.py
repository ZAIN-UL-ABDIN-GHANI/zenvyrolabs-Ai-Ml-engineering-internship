from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_studio.ai_processing import SpeakerSimilarityAnalyzer, SpeakerSimilarityError


def _resemblyzer_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("resemblyzer") is not None


class _SequentialFakeEncoder:
    """Models the real usage pattern: one encoder instance loaded once,
    then .encode() called twice (once per clip) — NOT a fresh instance
    per clip. Returns vectors in call order."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = iter(vectors)
        self.released = False

    def encode(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.array(next(self._vectors), dtype=np.float32)

    def release(self) -> None:
        self.released = True


class TestSpeakerSimilarityAnalyzer:
    def test_identical_vectors_score_100_percent(self, tone_wav_factory):
        clip_a, clip_b = tone_wav_factory(440, 0.5), tone_wav_factory(300, 0.5)
        analyzer = SpeakerSimilarityAnalyzer(
            encoder_factory=lambda: _SequentialFakeEncoder([[3.0, 4.0], [3.0, 4.0]])
        )
        result = analyzer.compare(clip_a, clip_b)
        assert result.similarity_percent == pytest.approx(100.0)

    def test_orthogonal_vectors_score_0_percent(self, tone_wav_factory):
        clip_a, clip_b = tone_wav_factory(440, 0.5), tone_wav_factory(300, 0.5)
        analyzer = SpeakerSimilarityAnalyzer(
            encoder_factory=lambda: _SequentialFakeEncoder([[1.0, 0.0], [0.0, 1.0]])
        )
        result = analyzer.compare(clip_a, clip_b)
        assert result.similarity_percent == pytest.approx(0.0)

    def test_opposite_vectors_are_clamped_to_0_not_negative(self, tone_wav_factory):
        clip_a, clip_b = tone_wav_factory(440, 0.5), tone_wav_factory(300, 0.5)
        analyzer = SpeakerSimilarityAnalyzer(
            encoder_factory=lambda: _SequentialFakeEncoder([[1.0, 0.0], [-1.0, 0.0]])
        )
        result = analyzer.compare(clip_a, clip_b)
        assert result.similarity_percent == pytest.approx(0.0)  # clamped, not -100

    def test_encoder_is_released_after_comparison(self, tone_wav_factory):
        clip_a, clip_b = tone_wav_factory(440, 0.5), tone_wav_factory(300, 0.5)
        encoder = _SequentialFakeEncoder([[1.0, 0.0], [1.0, 0.0]])
        SpeakerSimilarityAnalyzer(encoder_factory=lambda: encoder).compare(clip_a, clip_b)
        assert encoder.released is True

    def test_missing_file_raises_clearly(self, tmp_path: Path, tone_wav_factory):
        analyzer = SpeakerSimilarityAnalyzer(
            encoder_factory=lambda: _SequentialFakeEncoder([[1.0, 0.0], [1.0, 0.0]])
        )
        with pytest.raises(SpeakerSimilarityError, match="must exist"):
            analyzer.compare(tmp_path / "missing.wav", tone_wav_factory(440, 0.5))

    def test_encoder_failure_is_wrapped_and_still_releases(self, tone_wav_factory):
        class _BrokenEncoder:
            def __init__(self):
                self.released = False

            def encode(self, samples, sample_rate):
                raise RuntimeError("GPU out of memory")

            def release(self):
                self.released = True

        broken = _BrokenEncoder()
        analyzer = SpeakerSimilarityAnalyzer(encoder_factory=lambda: broken)
        with pytest.raises(SpeakerSimilarityError, match="GPU out of memory"):
            analyzer.compare(tone_wav_factory(440, 0.5), tone_wav_factory(300, 0.5))
        assert broken.released is True

    def test_missing_dependency_at_construction_is_wrapped_not_raw(self, tone_wav_factory):
        # Regression test: encoder construction used to happen outside the
        # try block, so a missing backend package (e.g. resemblyzer not
        # installed) would propagate as a raw ImportError instead of a
        # clean SpeakerSimilarityError.
        def _factory():
            raise ImportError("No module named 'resemblyzer'")

        analyzer = SpeakerSimilarityAnalyzer(encoder_factory=_factory)
        with pytest.raises(SpeakerSimilarityError, match="resemblyzer"):
            analyzer.compare(tone_wav_factory(440, 0.5), tone_wav_factory(300, 0.5))

    def test_invalid_audio_is_wrapped_not_raw(self, tmp_path: Path, tone_wav_factory):
        invalid = tmp_path / "invalid.wav"
        invalid.write_bytes(b"not actually audio data")
        analyzer = SpeakerSimilarityAnalyzer(
            encoder_factory=lambda: _SequentialFakeEncoder([[1.0, 0.0], [1.0, 0.0]])
        )
        with pytest.raises(SpeakerSimilarityError, match="Similarity analysis failed"):
            analyzer.compare(invalid, tone_wav_factory(440, 0.5))


class TestCosineSimilarityMath:
    def test_zero_vector_returns_zero_not_a_division_error(self):
        result = SpeakerSimilarityAnalyzer._cosine_similarity(
            np.array([0.0, 0.0]), np.array([1.0, 0.0])
        )
        assert result == 0.0


class TestResemblyzerRealBackend:
    """Exercises the real, production default backend end-to-end.

    Skips gracefully (not fail) if resemblyzer isn't installed in a given
    environment, since it's a heavy optional dependency — but runs real
    inference wherever it is available, verifying F-19's actual fix rather
    than only its dependency-injection seam.
    """

    pytestmark = pytest.mark.skipif(
        not _resemblyzer_available(), reason="resemblyzer/torch not installed in this environment"
    )

    def test_default_encoder_factory_is_resemblyzer(self):
        import inspect

        from voice_studio.ai_processing.speaker_similarity import _ResemblyzerEncoderAdapter

        default = inspect.signature(SpeakerSimilarityAnalyzer.__init__).parameters["encoder_factory"].default
        assert default is _ResemblyzerEncoderAdapter

    def test_real_embedding_generation_succeeds(self, voice_like_wav_factory):
        analyzer = SpeakerSimilarityAnalyzer()
        clip_a = voice_like_wav_factory(f0=150, seconds=2.0)
        clip_b = voice_like_wav_factory(f0=150, seconds=2.0)
        result = analyzer.compare(clip_a, clip_b)
        assert 0.0 <= result.similarity_percent <= 100.0

    def test_same_speaker_scores_higher_than_different_speaker(self, voice_like_wav_factory):
        analyzer = SpeakerSimilarityAnalyzer()

        same_speaker_a = voice_like_wav_factory(f0=150, seconds=3.0)
        same_speaker_b = voice_like_wav_factory(f0=150, seconds=3.0)
        same_result = analyzer.compare(same_speaker_a, same_speaker_b)

        speaker_a = voice_like_wav_factory(f0=150, seconds=3.0)
        speaker_c = voice_like_wav_factory(f0=280, seconds=3.0)
        diff_result = analyzer.compare(speaker_a, speaker_c)

        assert same_result.similarity_percent > diff_result.similarity_percent

    def test_no_network_access_required(self, voice_like_wav_factory, monkeypatch):
        # Guards against a regression back to a HF-Hub-downloading backend:
        # block socket connections and confirm inference still succeeds.
        import socket

        def _blocked(*args, **kwargs):
            raise AssertionError("Network access attempted during inference")

        monkeypatch.setattr(socket.socket, "connect", _blocked)
        analyzer = SpeakerSimilarityAnalyzer()
        clip_a = voice_like_wav_factory(f0=150, seconds=1.5)
        clip_b = voice_like_wav_factory(f0=150, seconds=1.5)
        result = analyzer.compare(clip_a, clip_b)
        assert result.similarity_percent >= 0.0
