from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voice_studio.ai_processing import (
    ConversionRequest,
    NarrationRequest,
    NeuralNarrationEngine,
    VoiceConversionEngine,
    VoiceGenerationEngine,
    VoiceGenerationRequest,
)
from voice_studio.domain import (
    Language,
    ModelArtifact,
    PronunciationPolicy,
    RoutingDecision,
    VoiceProfile,
)


@dataclass(frozen=True, slots=True)
class LineVoicingRequest:
    text: str
    voice_profile: VoiceProfile
    conversion_model: ModelArtifact | None
    output_dir: Path
    line_index: int


class PronunciationRoutingService:
    """
    Routes every line either through direct voice cloning or
    native-language narration + RVC conversion.
    """

    def __init__(
        self,
        policy: PronunciationPolicy,
        voice_generation: VoiceGenerationEngine,
        neural_narration: NeuralNarrationEngine,
        voice_conversion: VoiceConversionEngine,
        narration_voice: str,
    ) -> None:
        self._policy = policy
        self._voice_generation = voice_generation
        self._neural_narration = neural_narration
        self._voice_conversion = voice_conversion
        self._narration_voice = narration_voice

    def voice_line(self, request: LineVoicingRequest) -> Path:
        evaluation = self._policy.evaluate(request.text)

        if evaluation.decision is RoutingDecision.DIRECT_CLONE:
            return self._direct_clone(request)

        return self._hybrid_native_then_convert(request, evaluation.language)

    def _direct_clone(self, request: LineVoicingRequest) -> Path:
        return self._voice_generation.generate(
            VoiceGenerationRequest(
                reference_audio_path=request.voice_profile.reference_audio_path,
                reference_text=request.voice_profile.reference_text,
                target_text=request.text,
                output_dir=request.output_dir,
            )
        )

    def _hybrid_native_then_convert(
        self,
        request: LineVoicingRequest,
        language: Language,
    ) -> Path:
        """
        If no RVC model exists, simply use F5-TTS voice cloning.
        Otherwise use:
            EdgeTTS -> RVC -> Final Voice
        """

        # Fallback when no RVC model is available
        if request.conversion_model is None:
            return self._direct_clone(request)

        narration_text = self._localize_text(request.text, language)

        narrated_path = (
            request.output_dir
            / f"line_{request.line_index:04d}_narrated.wav"
        )

        self._neural_narration.narrate(
            NarrationRequest(
                text=narration_text,
                output_path=narrated_path,
                voice=self._narration_voice,
            )
        )

        converted_path = (
            request.output_dir
            / f"line_{request.line_index:04d}_converted.wav"
        )

        return self._voice_conversion.convert(
            ConversionRequest(
                input_audio_path=narrated_path,
                output_audio_path=converted_path,
                model=request.conversion_model,
            )
        )

    @staticmethod
    def _localize_text(text: str, language: Language) -> str:
        if language is not Language.ROMANIZED_HINDI_URDU:
            return text

        from transliterate import roman_to_devanagari

        return roman_to_devanagari(text)