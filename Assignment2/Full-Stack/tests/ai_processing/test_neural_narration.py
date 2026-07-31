from __future__ import annotations

from pathlib import Path

from voice_studio.ai_processing import NarrationRequest, NeuralNarrationEngine


class TestNeuralNarrationEngine:
    def test_narrate_produces_a_playable_file(self, test_path_settings, tmp_path: Path):
        engine = NeuralNarrationEngine(paths=test_path_settings)
        output_path = tmp_path / "narrated.wav"

        result = engine.narrate(
            NarrationRequest(text="Hello there", output_path=output_path, voice="en-US-GuyNeural")
        )

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_rate_and_pitch_are_formatted_as_signed_values(self, test_path_settings):
        assert NeuralNarrationEngine._signed_percent(10) == "+10%"
        assert NeuralNarrationEngine._signed_percent(-5) == "-5%"
        assert NeuralNarrationEngine._signed_hertz(0) == "+0Hz"
