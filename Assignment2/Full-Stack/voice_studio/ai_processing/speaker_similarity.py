from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from pydub import AudioSegment


class SpeakerSimilarityError(RuntimeError):
    """Raised when two audio clips cannot be compared."""


class AudioEncoder(Protocol):
    """Structural interface: embeds audio samples into a fixed-size vector,
    so the production speaker-embedding backend can be swapped for a test
    double."""

    def encode(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...


class _ResemblyzerEncoderAdapter:
    """Wraps Resemblyzer's VoiceEncoder behind AudioEncoder.

    Resemblyzer's d-vector embeddings are trained via a speaker-verification
    objective (GE2E loss: explicitly clustering same-speaker utterances and
    separating different speakers, independent of spoken content) — the
    correct class of model for this task. Pretrained weights ship inside the
    package itself, so no network access is required at runtime.
    """

    def __init__(self) -> None:
        from resemblyzer import VoiceEncoder

        self._encoder = VoiceEncoder()

    def encode(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        from resemblyzer import preprocess_wav

        processed = preprocess_wav(samples, source_sr=sample_rate)
        return self._encoder.embed_utterance(processed)

    def release(self) -> None:
        import torch

        del self._encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    similarity_percent: float


class SpeakerSimilarityAnalyzer:
    """AI Processing Layer component (SDS §5.4) scoring how similar two clips
    sound, using speaker-embedding cosine similarity.

    Backed by Resemblyzer's VoiceEncoder by default — a model trained
    specifically for speaker verification, closing the Stage 1 Analysis's
    F-19 finding that the previous Whisper-ASR-encoder-based approach
    conflated spoken content with speaker identity.
    """

    def __init__(
        self,
        encoder_factory: Callable[[], AudioEncoder] = _ResemblyzerEncoderAdapter,
        max_duration_ms: int = 15000,
    ) -> None:
        self._encoder_factory = encoder_factory
        self._max_duration_ms = max_duration_ms

    def compare(self, audio_a: Path, audio_b: Path) -> SimilarityResult:
        if not Path(audio_a).exists() or not Path(audio_b).exists():
            raise SpeakerSimilarityError("Both audio files must exist to compare.")

        encoder = None
        try:
            encoder = self._encoder_factory()
            emb_a = encoder.encode(*self._load_samples(audio_a))
            emb_b = encoder.encode(*self._load_samples(audio_b))
        except Exception as error:
            raise SpeakerSimilarityError(f"Similarity analysis failed: {error}") from error
        finally:
            release = getattr(encoder, "release", None)
            if callable(release):
                release()

        similarity = self._cosine_similarity(emb_a, emb_b)
        return SimilarityResult(similarity_percent=max(0.0, min(100.0, similarity * 100.0)))

    def _load_samples(self, path: Path) -> tuple[np.ndarray, int]:
        audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000)
        if len(audio) > self._max_duration_ms:
            audio = audio[: self._max_duration_ms]
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        return samples, 16000

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
