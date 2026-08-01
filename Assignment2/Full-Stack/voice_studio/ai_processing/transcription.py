from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from pydub import AudioSegment


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be transcribed."""


class SpeechRecognizer(Protocol):
    """Structural interface matching transformers' ASR pipeline call shape,
    so the real Whisper pipeline can be swapped for a test double."""

    def __call__(self, audio_path: str, chunk_length_s: int, generate_kwargs: dict) -> dict: ...


def _load_default_whisper_pipeline():
    import whisper

    model = whisper.load_model("base")

    class WhisperRecognizer:
        def __call__(self, audio_path: str, chunk_length_s: int = 30, generate_kwargs: dict | None = None):
            result = model.transcribe(audio_path)
            return {"text": result["text"]}

    return WhisperRecognizer()

@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    audio_path: Path
    max_duration_ms: int = 8000


class TranscriptionEngine:
    """AI Processing Layer component wrapping Whisper ASR (SDS §5.4).

    The Whisper pipeline is loaded lazily via an injectable factory, so this
    class's audio-preparation and error-handling logic is testable without
    downloading real model weights; production callers use the default.
    """

    def __init__(
        self,
        recognizer_factory: Callable[[], SpeechRecognizer] = _load_default_whisper_pipeline,
        temp_dir: Path | None = None,
    ) -> None:
        self._recognizer_factory = recognizer_factory
        self._temp_dir = temp_dir

    def transcribe(self, request: TranscriptionRequest) -> str:
        if not request.audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {request.audio_path}")

        trimmed_path = self._prepare_audio(request)
        recognizer = self._recognizer_factory()
        try:
            result = recognizer(str(trimmed_path), chunk_length_s=30, generate_kwargs={"task": "transcribe"})
        except Exception as error:
            raise TranscriptionError(f"Whisper transcription failed: {error}") from error
        finally:
            del recognizer
            gc.collect()
            self._release_gpu_memory()

        text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
        if not text:
            raise TranscriptionError("Whisper returned an empty transcription.")
        return text

    def _prepare_audio(self, request: TranscriptionRequest) -> Path:
        audio = AudioSegment.from_file(request.audio_path)
        if len(audio) > request.max_duration_ms:
            audio = audio[: request.max_duration_ms]
        temp_dir = self._temp_dir or request.audio_path.parent
        temp_dir.mkdir(parents=True, exist_ok=True)
        trimmed_path = temp_dir / "extract_temp.wav"
        audio.export(trimmed_path, format="wav")
        return trimmed_path

    @staticmethod
    def _release_gpu_memory() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
