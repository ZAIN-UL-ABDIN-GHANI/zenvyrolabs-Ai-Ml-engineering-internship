from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique

from voice_studio.domain.language import Language, LanguageDetector


@unique
class RoutingDecision(Enum):
    DIRECT_CLONE = "direct_clone"
    HYBRID_NATIVE_THEN_CONVERT = "hybrid_native_then_convert"


@dataclass(frozen=True, slots=True)
class RoutingEvaluation:
    language: Language
    decision: RoutingDecision


class PronunciationPolicy:
    """Domain policy deciding how a line of text must be voiced.

    Implements the SDS §6.4 / §12 requirement: any text identified as
    Hindi/Urdu (native script or romanized) is always routed through
    native narration before voice conversion, never cloned directly.
    """

    _NATIVE_ROUTING_LANGUAGES = frozenset(
        {Language.ROMANIZED_HINDI_URDU, Language.DEVANAGARI, Language.URDU_SCRIPT}
    )

    def __init__(self, detector: LanguageDetector | None = None) -> None:
        self._detector = detector or LanguageDetector()

    def evaluate(self, text: str) -> RoutingEvaluation:
        language = self._detector.detect(text)
        decision = (
            RoutingDecision.HYBRID_NATIVE_THEN_CONVERT
            if language in self._NATIVE_ROUTING_LANGUAGES
            else RoutingDecision.DIRECT_CLONE
        )
        return RoutingEvaluation(language=language, decision=decision)

    def decide(self, text: str) -> RoutingDecision:
        return self.evaluate(text).decision
