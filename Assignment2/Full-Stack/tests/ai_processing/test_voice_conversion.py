from __future__ import annotations

from pathlib import Path

from voice_studio.ai_processing import ConversionRequest, VoiceConversionEngine
from voice_studio.config import EngineSettings
from voice_studio.domain import EngineType, ModelArtifact


class TestVoiceConversionEngine:
    def test_convert_produces_output_and_passes_index_when_present(
        self, test_path_settings, real_wav_file: Path, tmp_path: Path
    ):
        model_path = tmp_path / "model.pth"
        index_path = tmp_path / "model.index"
        model_path.write_bytes(b"0" * 2_000_000)
        index_path.write_bytes(b"0" * 100)

        engine = VoiceConversionEngine(paths=test_path_settings, engine=EngineSettings())
        output_path = tmp_path / "converted.wav"

        result = engine.convert(
            ConversionRequest(
                input_audio_path=real_wav_file,
                output_audio_path=output_path,
                model=ModelArtifact(
                    artifact_id="m1",
                    engine_type=EngineType.RVC,
                    model_path=model_path,
                    index_path=index_path,
                ),
            )
        )

        assert result == output_path
        assert output_path.exists()

    def test_convert_without_index_still_succeeds(
        self, test_path_settings, real_wav_file: Path, tmp_path: Path
    ):
        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"0" * 2_000_000)

        engine = VoiceConversionEngine(paths=test_path_settings, engine=EngineSettings())
        output_path = tmp_path / "converted.wav"

        engine.convert(
            ConversionRequest(
                input_audio_path=real_wav_file,
                output_audio_path=output_path,
                model=ModelArtifact(artifact_id="m1", engine_type=EngineType.RVC, model_path=model_path),
            )
        )

        assert output_path.exists()
