from __future__ import annotations

from pathlib import Path

import pytest

from voice_studio.ai_processing import NeuralNarrationEngine, VoiceConversionEngine, VoiceGenerationEngine
from voice_studio.application import PodcastEngine, PodcastEngineError, PodcastScriptParser, PronunciationRoutingService
from voice_studio.audio_processing import AudioStitcher
from voice_studio.config import EngineSettings
from voice_studio.domain import PronunciationPolicy, VoiceProfile
from voice_studio.storage import ModelStoreRepository, VoiceStoreRepository


@pytest.fixture
def podcast_engine(test_path_settings, tmp_path: Path) -> PodcastEngine:
    engine_settings = EngineSettings()
    voice_store = VoiceStoreRepository(root=tmp_path / "saved_voices")
    model_store = ModelStoreRepository(root=tmp_path / "rvc_models")

    routing_service = PronunciationRoutingService(
        policy=PronunciationPolicy(),
        voice_generation=VoiceGenerationEngine(test_path_settings, engine_settings),
        neural_narration=NeuralNarrationEngine(test_path_settings),
        voice_conversion=VoiceConversionEngine(test_path_settings, engine_settings),
        narration_voice=engine_settings.default_narration_voice,
    )
    return PodcastEngine(
        parser=PodcastScriptParser(),
        voice_store=voice_store,
        model_store=model_store,
        routing_service=routing_service,
        stitcher=AudioStitcher(),
    ), voice_store, model_store


class TestPodcastEngine:
    def test_full_pipeline_produces_a_playable_file_and_reports_warnings(
        self, podcast_engine, real_wav_file: Path, tmp_path: Path
    ):
        engine, voice_store, _model_store = podcast_engine
        voice_store.save(VoiceProfile(name="ARIA", reference_audio_path=real_wav_file, reference_text="Hi."))
        voice_store.save(
            VoiceProfile(
                name="GOJO", reference_audio_path=real_wav_file, reference_text="Hi.", model_artifact_id="GOJO"
            )
        )
        rvc_models_dir = tmp_path / "rvc_models"
        rvc_models_dir.mkdir(parents=True, exist_ok=True)
        (rvc_models_dir / "GOJO.pth").write_bytes(b"0" * 2_000_000)

        script = (
            "ARIA: Hey everyone, welcome to the podcast.\n"
            "GOJO: Subscribe karo aur like karo\n"
            "UNKNOWN_SPEAKER: nobody saved this voice\n"
            "this line has no colon at all\n"
        )
        result = engine.generate(script, title="Episode 1: Test!", output_dir=tmp_path / "out")

        assert result.output_path.exists()
        assert len(result.warnings) == 2
        assert any("UNKNOWN_SPEAKER" in w for w in result.warnings)
        assert any("no ':' separator" in w for w in result.warnings)

    def test_empty_script_raises_podcast_engine_error(self, podcast_engine, tmp_path: Path):
        engine, _voice_store, _model_store = podcast_engine
        with pytest.raises(PodcastEngineError):
            engine.generate("no colon anywhere", title="Empty", output_dir=tmp_path / "out")

    def test_no_matching_voices_raises_podcast_engine_error(self, podcast_engine, tmp_path: Path):
        engine, _voice_store, _model_store = podcast_engine
        with pytest.raises(PodcastEngineError, match="no saved voices matched"):
            engine.generate("GHOST: hello", title="Nobody Home", output_dir=tmp_path / "out")
