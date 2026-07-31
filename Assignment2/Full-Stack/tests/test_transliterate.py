from __future__ import annotations

import pytest

from transliterate import roman_to_devanagari, transliterate_word


class TestBundledSelfTestSentences:
    """Locks in the exact sentences shipped in transliterate.py's own __main__."""

    @pytest.mark.parametrize(
        "roman,expected",
        [
            ("Hello kya haal hai bhai", "हेलो क्या हाल है भाई"),
            ("Subscribe karo aur like karo", "सब्सक्राइब करो और लाइक करो"),
        ],
    )
    def test_known_good_sentences_unchanged(self, roman, expected):
        assert roman_to_devanagari(roman) == expected


class TestF12WordFinalVowelLength:
    """Regression tests: word-final short 'i' now uses the conventional long matra."""

    def test_hi_intensifier(self):
        assert transliterate_word("hi") == "ही"

    def test_buri(self):
        assert transliterate_word("buri") == "बुरी"

    def test_mid_word_short_i_is_unaffected(self):
        # Only the word-final vowel should be promoted to long; "dil" keeps its
        # genuinely short i since it is not word-final within "dilchasp".
        assert transliterate_word("dilchasp") == "दिलचस्प"


class TestF12ConsonantClusters:
    """Regression tests: genuine consonant clusters now get a halant."""

    def test_dilchasp_via_dictionary(self):
        assert transliterate_word("dilchasp") == "दिलचस्प"

    def test_masti_via_phonetic_fallback(self):
        # Not in COMMON_WORDS — exercises the general heuristic, not the
        # dictionary shortcut.
        assert transliterate_word("masti") == "मस्ती"


class TestRomanToDevanagariTextHandling:
    def test_already_devanagari_text_passes_through_unchanged(self):
        text = "यह पहले से देवनागरी में है"
        assert roman_to_devanagari(text) == text

    def test_punctuation_is_preserved(self):
        result = roman_to_devanagari("kya haal hai?")
        assert result.endswith("?")

    def test_empty_string_returns_empty_string(self):
        assert roman_to_devanagari("") == ""
