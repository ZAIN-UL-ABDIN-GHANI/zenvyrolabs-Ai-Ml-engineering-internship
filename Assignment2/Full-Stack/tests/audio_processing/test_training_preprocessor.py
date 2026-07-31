from __future__ import annotations

from pathlib import Path

import pytest

from voice_studio.audio_processing import TrainingAudioPreprocessor, TrainingPreprocessError
from voice_studio.domain import TrainingSession, TrainingStage


class TestTrainingAudioPreprocessor:
    def test_rejects_a_session_with_no_raw_audio(self, tmp_path: Path):
        session = TrainingSession(session_id="empty")
        with pytest.raises(TrainingPreprocessError):
            TrainingAudioPreprocessor().process(session, tmp_path / "out")

    def test_removes_real_silence_gaps_and_produces_chunks(
        self, voice_like_wav_factory, tmp_path: Path
    ):
        clip = voice_like_wav_factory(f0=150, seconds=4.0, silence_before=0.0, silence_after=2.0)
        session = TrainingSession(session_id="sess-1")
        session.add_raw_audio(clip)

        result = TrainingAudioPreprocessor().process(session, tmp_path / "out")

        assert result.stage is TrainingStage.SEGMENTED
        assert len(result.segment_paths) >= 1
        from pydub import AudioSegment

        total_ms = sum(len(AudioSegment.from_file(p)) for p in result.segment_paths)
        assert total_ms < 4500  # the ~2s trailing silence must have been removed

    def test_fully_silent_clip_does_not_crash(self, fully_silent_wav_factory, tmp_path: Path):
        # Direct regression test for Stage 1 Analysis F-10 (-inf dBFS on apply_gain).
        clip = fully_silent_wav_factory(seconds=2.0)
        session = TrainingSession(session_id="sess-silent")
        session.add_raw_audio(clip)

        result = TrainingAudioPreprocessor().process(session, tmp_path / "out")

        assert result.stage is TrainingStage.SEGMENTED

    def test_fully_silent_clip_raises_no_runtime_warning(self, fully_silent_wav_factory, tmp_path: Path):
        import warnings

        clip = fully_silent_wav_factory(seconds=2.0)
        session = TrainingSession(session_id="sess-silent-2")
        session.add_raw_audio(clip)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            TrainingAudioPreprocessor().process(session, tmp_path / "out")
