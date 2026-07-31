from __future__ import annotations

from pathlib import Path

import pytest

from voice_studio.ai_processing import VoiceGenerationEngine, VoiceGenerationRequest
from voice_studio.config import EngineSettings


class TestVoiceGenerationEngine:
    def test_generate_writes_config_and_locates_output(
        self, test_path_settings, real_wav_file: Path, tmp_path: Path
    ):
        engine = VoiceGenerationEngine(paths=test_path_settings, engine=EngineSettings())
        output_dir = tmp_path / "out"

        result = engine.generate(
            VoiceGenerationRequest(
                reference_audio_path=real_wav_file,
                reference_text="Reference text.",
                target_text="Hello, world.",
                output_dir=output_dir,
            )
        )

        assert result.exists()
        config_text = (output_dir / "generation_config.toml").read_text()
        assert 'model = "F5TTS_Base"' in config_text
        assert "Hello, world." in config_text

    def test_missing_f5_tts_executable_fails_fast(self, real_wav_file: Path, tmp_path: Path):
        from voice_studio.config import PathSettings

        settings = PathSettings.from_root(root=tmp_path)  # no fake executables installed here
        engine = VoiceGenerationEngine(paths=settings, engine=EngineSettings())

        with pytest.raises(FileNotFoundError):
            engine.generate(
                VoiceGenerationRequest(
                    reference_audio_path=real_wav_file,
                    reference_text="ref",
                    target_text="hello",
                    output_dir=tmp_path / "out",
                )
            )
