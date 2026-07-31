from __future__ import annotations

from pathlib import Path

import pytest

from voice_studio.domain import TrainingSession, TrainingSessionError, TrainingStage


class TestTrainingSession:
    def test_rejects_empty_session_id(self):
        with pytest.raises(TrainingSessionError):
            TrainingSession(session_id="  ")

    def test_starts_in_raw_stage(self):
        session = TrainingSession(session_id="sess-1")
        assert session.stage is TrainingStage.RAW

    def test_advance_to_requires_raw_audio(self):
        session = TrainingSession(session_id="sess-1")
        with pytest.raises(TrainingSessionError):
            session.advance_to(TrainingStage.NORMALIZED)

    def test_advance_to_moves_forward(self):
        session = TrainingSession(session_id="sess-1")
        session.add_raw_audio(Path("raw.wav"))
        session.advance_to(TrainingStage.NORMALIZED)
        assert session.stage is TrainingStage.NORMALIZED

    def test_advance_to_rejects_moving_backward(self):
        session = TrainingSession(session_id="sess-1")
        session.add_raw_audio(Path("raw.wav"))
        session.advance_to(TrainingStage.NORMALIZED)
        with pytest.raises(TrainingSessionError):
            session.advance_to(TrainingStage.RAW)

    def test_cannot_add_raw_audio_after_advancing(self):
        session = TrainingSession(session_id="sess-1")
        session.add_raw_audio(Path("raw.wav"))
        session.advance_to(TrainingStage.NORMALIZED)
        with pytest.raises(TrainingSessionError):
            session.add_raw_audio(Path("more.wav"))

    def test_is_ready_for_dataset_only_true_at_final_stage(self):
        session = TrainingSession(session_id="sess-1")
        session.add_raw_audio(Path("raw.wav"))
        assert session.is_ready_for_dataset is False
        for stage in (
            TrainingStage.NORMALIZED,
            TrainingStage.FILTERED,
            TrainingStage.SEGMENTED,
            TrainingStage.DATASET_READY,
        ):
            session.advance_to(stage)
        assert session.is_ready_for_dataset is True
