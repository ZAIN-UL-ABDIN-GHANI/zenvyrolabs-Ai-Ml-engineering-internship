from __future__ import annotations

from pathlib import Path

import pytest

from voice_studio.ai_processing import NeuralNarrationEngine, VoiceConversionEngine, VoiceGenerationEngine
from voice_studio.application import LineVoicingRequest, PronunciationRoutingService
from voice_studio.config import EngineSettings
from voice_studio.domain import EngineType, ModelArtifact, PronunciationPolicy, VoiceProfile


@pytest.fixture
def routing_service(test_path_settings) -> PronunciationRoutingService:
    engine_settings = EngineSettings()
    return PronunciationRoutingService(
        policy=PronunciationPolicy(),
        voice_generation=VoiceGenerationEngine(test_path_settings, engine_settings),
        neural_narration=NeuralNarrationEngine(test_path_settings),
        voice_conversion=VoiceConversionEngine(test_path_settings, engine_settings),
        narration_voice=engine_settings.default_narration_voice,
    )


@pytest.fixture
def voice_profile(real_wav_file: Path) -> VoiceProfile:
    return VoiceProfile(name="ARIA", reference_audio_path=real_wav_file, reference_text="ref text")


@pytest.fixture
def conversion_model(tmp_path: Path) -> ModelArtifact:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"0" * 2_000_000)
    return ModelArtifact(artifact_id="m1", engine_type=EngineType.RVC, model_path=model_path)


class TestPronunciationRoutingService:
    def test_english_text_takes_the_direct_clone_path(
        self, routing_service, voice_profile, conversion_model, tmp_path: Path
    ):
        result = routing_service.voice_line(
            LineVoicingRequest(
                text="Hey everyone, welcome!",
                voice_profile=voice_profile,
                conversion_model=conversion_model,
                output_dir=tmp_path / "out",
                line_index=1,
            )
        )
        assert result.exists()
        assert "infer_cli" in result.name  # produced by the fake F5-TTS stub, not edge-tts/rvc

    def test_romanized_hindi_takes_the_hybrid_path_and_transliterates(
        self, routing_service, voice_profile, conversion_model, tmp_path: Path
    ):
        output_dir = tmp_path / "out"
        result = routing_service.voice_line(
            LineVoicingRequest(
                text="Subscribe karo aur like karo",
                voice_profile=voice_profile,
                conversion_model=conversion_model,
                output_dir=output_dir,
                line_index=2,
            )
        )
        assert result.name == "line_0002_converted.wav"
        assert (output_dir / "line_0002_narrated.wav").exists()

    def test_hybrid_path_without_a_conversion_model_raises_clearly(
        self, routing_service, voice_profile, tmp_path: Path
    ):
        with pytest.raises(ValueError, match="requires a voice conversion model"):
            routing_service.voice_line(
                LineVoicingRequest(
                    text="Subscribe karo aur like karo",
                    voice_profile=voice_profile,
                    conversion_model=None,
                    output_dir=tmp_path / "out",
                    line_index=3,
                )
            )
