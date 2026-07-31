from __future__ import annotations

import re
from enum import Enum, unique

_DEVANAGARI_RANGE: tuple[int, int] = (0x0900, 0x097F)
_URDU_ARABIC_RANGE: tuple[int, int] = (0x0600, 0x06FF)
_WORD_PATTERN = re.compile(r"[A-Za-z]+")
_ROMANIZED_MATCH_THRESHOLD = 0.15
_ROMANIZED_MIN_ABSOLUTE_MATCHES = 1

# English words that appear verbatim in COMMON_WORDS (correct for
# transliteration — "subscribe karo" -> "सब्सक्राइब करो", "log kya kahenge"
# -> "लोग क्या कहेंगे") but are poor *detection* signals: each is also a
# common, unambiguous English word, so counting it toward the
# romanized-Hindi/Urdu ratio produces false positives on plain English
# sentences (e.g. "Do you want me to log the main results" or "Subscribe
# to my channel"). Excluded only from detection; transliterate.COMMON_WORDS
# itself — the transliteration vocabulary — is completely untouched.
_DETECTION_VOCABULARY_EXCLUSIONS = frozenset(
    {
        "subscribe", "like", "comment", "video", "channel", "share", "hello", "the",
        "do", "me", "main", "log", "par", "ya", "char",
    }
)


@unique
class Language(Enum):
    ENGLISH = "english"
    DEVANAGARI = "devanagari"
    URDU_SCRIPT = "urdu_script"
    ROMANIZED_HINDI_URDU = "romanized_hindi_urdu"


def _load_default_detection_vocabulary() -> frozenset[str]:
    """The transliteration vocabulary (transliterate.COMMON_WORDS), filtered
    down to words that are meaningful Hindi/Urdu detection signals — i.e.
    with common English-word entries excluded. transliterate.COMMON_WORDS
    itself is never modified; this is a detection-only derived view."""
    from transliterate import COMMON_WORDS

    return frozenset(COMMON_WORDS.keys()) - _DETECTION_VOCABULARY_EXCLUSIONS


class LanguageDetector:
    """Classifies text so Pronunciation Routing can decide how it must be voiced."""

    def __init__(self, romanized_vocabulary: frozenset[str] | None = None) -> None:
        self._vocabulary = romanized_vocabulary or _load_default_detection_vocabulary()

    def detect(self, text: str) -> Language:
        stripped = text.strip()
        if not stripped:
            return Language.ENGLISH
        if self._contains_codepoint_in(stripped, _DEVANAGARI_RANGE):
            return Language.DEVANAGARI
        if self._contains_codepoint_in(stripped, _URDU_ARABIC_RANGE):
            return Language.URDU_SCRIPT
        if self._is_romanized_hindi_urdu(stripped):
            return Language.ROMANIZED_HINDI_URDU
        return Language.ENGLISH

    @staticmethod
    def _contains_codepoint_in(text: str, code_range: tuple[int, int]) -> bool:
        low, high = code_range
        return any(low <= ord(char) <= high for char in text)

    def _is_romanized_hindi_urdu(self, text: str) -> bool:
        words = [word.lower() for word in _WORD_PATTERN.findall(text)]
        if not words:
            return False
        matches = sum(1 for word in words if word in self._vocabulary)
        if matches < _ROMANIZED_MIN_ABSOLUTE_MATCHES:
            return False
        return (matches / len(words)) >= _ROMANIZED_MATCH_THRESHOLD
