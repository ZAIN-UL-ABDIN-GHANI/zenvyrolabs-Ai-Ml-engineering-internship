from __future__ import annotations

from pathlib import Path

import pytest

from voice_studio.ai_processing import TranscriptionEngine, TranscriptionError, TranscriptionRequest


class _FakeRecognizer:
    def __init__(self, text: str = "Hello world.") -> None:
        self._text = text
        self.received_kwargs: dict | None = None

    def __call__(self, audio_path: str, chunk_length_s: int, generate_kwargs: dict) -> dict:
        self.received_kwargs = {"chunk_length_s": chunk_length_s, "generate_kwargs": generate_kwargs}
        return {"text": self._text}


class TestTranscriptionEngine:
    def test_transcribe_returns_stripped_text(self, real_wav_file: Path):
        fake = _FakeRecognizer(text="  Hello world.  ")
        engine = TranscriptionEngine(recognizer_factory=lambda: fake)
        assert engine.transcribe(TranscriptionRequest(audio_path=real_wav_file)) == "Hello world."

    def test_transcribe_passes_correct_task_kwargs(self, real_wav_file: Path):
        fake = _FakeRecognizer()
        engine = TranscriptionEngine(recognizer_factory=lambda: fake)
        engine.transcribe(TranscriptionRequest(audio_path=real_wav_file))
        assert fake.received_kwargs == {"chunk_length_s": 30, "generate_kwargs": {"task": "transcribe"}}

    def test_missing_audio_file_raises_clearly(self, tmp_path: Path):
        engine = TranscriptionEngine(recognizer_factory=lambda: _FakeRecognizer())
        with pytest.raises(TranscriptionError, match="not found"):
            engine.transcribe(TranscriptionRequest(audio_path=tmp_path / "missing.wav"))

    def test_empty_transcription_raises(self, real_wav_file: Path):
        engine = TranscriptionEngine(recognizer_factory=lambda: _FakeRecognizer(text="   "))
        with pytest.raises(TranscriptionError, match="empty"):
            engine.transcribe(TranscriptionRequest(audio_path=real_wav_file))

    def test_recognizer_failure_is_wrapped(self, real_wav_file: Path):
        def _broken_recognizer(*args, **kwargs):
            raise RuntimeError("model exploded")

        engine = TranscriptionEngine(recognizer_factory=lambda: _broken_recognizer)
        with pytest.raises(TranscriptionError, match="model exploded"):
            engine.transcribe(TranscriptionRequest(audio_path=real_wav_file))

    def test_long_audio_is_trimmed_before_transcription(self, tone_wav_factory):
        clip = tone_wav_factory(440, 2.0)  # 2000ms, longer than the 500ms max below
        fake = _FakeRecognizer()
        engine = TranscriptionEngine(recognizer_factory=lambda: fake)
        engine.transcribe(TranscriptionRequest(audio_path=clip, max_duration_ms=500))

        from pydub import AudioSegment

        trimmed = engine._prepare_audio(TranscriptionRequest(audio_path=clip, max_duration_ms=500))
        assert len(AudioSegment.from_file(trimmed)) <= 500
