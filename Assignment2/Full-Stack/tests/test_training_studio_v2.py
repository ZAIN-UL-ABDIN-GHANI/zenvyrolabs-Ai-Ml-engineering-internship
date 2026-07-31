from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment


class TestTrainingStudioWiredToRealPipeline:
    """Exercises app.preprocess_training_audio() — the real Gradio-facing
    handler — end-to-end, confirming it now delegates to
    voice_studio.audio_processing.TrainingAudioPreprocessor instead of the
    old duplicated inline implementation, while keeping the exact same
    (session_dir, log) contract the UI already depends on.
    """

    def test_clean_audio_produces_a_session_with_chunks(self, app_module, voice_like_wav_factory):
        clip = voice_like_wav_factory(f0=150, seconds=3.0, noise_level=50)
        session_dir, log = app_module.preprocess_training_audio(str(clip), chunk_seconds=10, normalize_db=-20.0)

        assert session_dir is not None
        assert Path(session_dir).is_dir()
        assert "✅" in log
        assert "Noise filtering applied" in log
        chunks = list(Path(session_dir).glob("chunk_*.wav"))
        assert len(chunks) >= 1
        assert AudioSegment.from_file(chunks[0])  # genuinely decodable output

    def test_noisy_audio_is_still_processed_successfully(self, app_module, voice_like_wav_factory):
        clip = voice_like_wav_factory(f0=150, seconds=3.0, noise_level=2500)
        session_dir, log = app_module.preprocess_training_audio(str(clip), chunk_seconds=10, normalize_db=-20.0)

        assert session_dir is not None
        assert "✅" in log
        assert len(list(Path(session_dir).glob("chunk_*.wav"))) >= 1

    def test_silent_segments_do_not_crash_the_pipeline(self, app_module, voice_like_wav_factory):
        # Regression guard for Stage 1 Analysis F-10 (-inf dBFS on apply_gain),
        # now exercised through the real wired-up UI handler, not just the
        # underlying preprocessor in isolation.
        clip = voice_like_wav_factory(f0=150, seconds=2.0, silence_after=2.0)
        session_dir, log = app_module.preprocess_training_audio(str(clip), chunk_seconds=10, normalize_db=-20.0)

        assert session_dir is not None
        assert "❌" not in log

    def test_fully_silent_file_does_not_crash(self, app_module, fully_silent_wav_factory):
        clip = fully_silent_wav_factory(seconds=2.0)
        session_dir, log = app_module.preprocess_training_audio(str(clip), chunk_seconds=10, normalize_db=-20.0)

        assert session_dir is not None
        assert "❌" not in log

    def test_invalid_audio_returns_a_clean_readable_error_not_a_crash(self, app_module, invalid_audio_file):
        session_dir, log = app_module.preprocess_training_audio(
            str(invalid_audio_file), chunk_seconds=10, normalize_db=-20.0
        )

        assert session_dir is None
        assert log.startswith("❌")
        assert "not be read as audio" in log
        # the raw ffmpeg build banner must not leak into the user-facing message
        assert "configuration:" not in log
        assert len(log) < 300

    def test_no_file_uploaded_returns_a_clear_message_without_touching_the_pipeline(self, app_module):
        session_dir, log = app_module.preprocess_training_audio(None, chunk_seconds=10, normalize_db=-20.0)
        assert session_dir is None
        assert "Upload an audio file" in log

    def test_sequential_sessions_get_distinct_directories(self, app_module, voice_like_wav_factory):
        clip_a = voice_like_wav_factory(f0=150, seconds=2.0, noise_level=50)
        clip_b = voice_like_wav_factory(f0=180, seconds=2.0, noise_level=50)

        dir_a, _ = app_module.preprocess_training_audio(str(clip_a), chunk_seconds=10, normalize_db=-20.0)
        dir_b, _ = app_module.preprocess_training_audio(str(clip_b), chunk_seconds=10, normalize_db=-20.0)

        assert dir_a != dir_b
