from __future__ import annotations

import pytest

from voice_studio.domain import Language, LanguageDetector


@pytest.fixture
def detector() -> LanguageDetector:
    return LanguageDetector()


class TestPureHindi:
    @pytest.mark.parametrize(
        "text",
        [
            "हेलो क्या हाल है भाई",
            "आज हम एक बहुत ही दिलचस्प कहानी सुनेंगे",
            "यह एक टेस्ट है",
        ],
    )
    def test_devanagari_script_detected(self, detector, text):
        assert detector.detect(text) == Language.DEVANAGARI


class TestPureUrdu:
    @pytest.mark.parametrize(
        "text",
        [
            "یہ ایک ٹیسٹ ہے",
            "آپ کیسے ہیں",
        ],
    )
    def test_urdu_script_detected(self, detector, text):
        assert detector.detect(text) == Language.URDU_SCRIPT


class TestRomanHindi:
    @pytest.mark.parametrize(
        "text",
        [
            "Aaj hum ek bahut hi dilchasp kahani sunenge",
            "Ye duniya bahut buri hai lekin hum theek hain",
            "Kya haal hai?",
            "karo",
        ],
    )
    def test_romanized_hindi_detected(self, detector, text):
        assert detector.detect(text) == Language.ROMANIZED_HINDI_URDU


class TestRomanUrdu:
    @pytest.mark.parametrize(
        "text",
        [
            "Aap kaisay hain, main theek hoon",
            "Aap kaisay hain, how are you today",
        ],
    )
    def test_romanized_urdu_detected(self, detector, text):
        assert detector.detect(text) == Language.ROMANIZED_HINDI_URDU


class TestMixedHindiEnglish:
    @pytest.mark.parametrize(
        "text",
        [
            "This is mostly English lekin thoda Hindi bhi hai",
            "Hey karo the video and subscribe",
            "Ye video bahut acha hai, please like karo",
            "Subscribe karo aur like karo",
        ],
    )
    def test_mixed_hindi_english_routes_as_hindi(self, detector, text):
        assert detector.detect(text) == Language.ROMANIZED_HINDI_URDU


class TestMixedUrduEnglish:
    @pytest.mark.parametrize(
        "text",
        [
            "یہ ایک اچھا دن ہے and today is great",
        ],
    )
    def test_mixed_urdu_native_script_routes_as_urdu(self, detector, text):
        assert detector.detect(text) == Language.URDU_SCRIPT


class TestPureEnglishFalsePositiveGuards:
    """Confirms plain English never falsely triggers native-language routing,
    including sentences containing words that are also valid Roman-Hindi
    transliteration entries (subscribe/like/the/do/me/main/log/etc.)."""

    @pytest.mark.parametrize(
        "text",
        [
            "Hey everyone! Welcome to the podcast.",
            "The quick brown fox jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
            "Subscribe to my channel and share this video, please comment below",
            "Do you want me to log the main results and check the char buffer",
            "Hello there!",
            "Like and subscribe for more videos",
            "In a world full of noise, silence becomes the loudest statement anyone can make today",
            (
                "Yesterday I went to the store to buy some groceries. The weather was "
                "nice, so I decided to walk instead of driving."
            ),
        ],
    )
    def test_plain_english_never_misdetected(self, detector, text):
        assert detector.detect(text) == Language.ENGLISH


class TestEdgeAndInvalidInput:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "\n\t  \n",
            "12345",
            "!!!???...",
            "😀🎉🔥",
            "a",
            "अ",
        ],
    )
    def test_degenerate_input_never_raises(self, detector, text):
        detector.detect(text)

    def test_empty_string_defaults_to_english(self, detector):
        assert detector.detect("") == Language.ENGLISH

    def test_whitespace_only_defaults_to_english(self, detector):
        assert detector.detect("   \n\t  ") == Language.ENGLISH

    def test_single_devanagari_character_detected_as_devanagari(self, detector):
        assert detector.detect("अ") == Language.DEVANAGARI

    def test_very_long_english_text_does_not_misclassify(self, detector):
        long_text = "This is a perfectly normal English sentence. " * 200
        assert detector.detect(long_text) == Language.ENGLISH

    def test_a_single_coincidental_loanword_does_not_misclassify_short_english(self, detector):
        assert detector.detect("Hello there!") == Language.ENGLISH

    def test_custom_vocabulary_can_be_injected(self):
        custom_detector = LanguageDetector(romanized_vocabulary=frozenset({"foo", "bar"}))
        assert custom_detector.detect("foo bar baz qux") == Language.ROMANIZED_HINDI_URDU
        assert custom_detector.detect("Subscribe karo aur like karo") == Language.ENGLISH


class TestDetectionVocabularyExcludesAmbiguousEnglishWords:
    """Regression tests for the specific ambiguous COMMON_WORDS entries found
    during production review: each is correct for transliteration but must
    not, by itself or in combination, trigger native-language routing."""

    @pytest.mark.parametrize("word", ["the", "do", "me", "main", "log", "hello", "like", "subscribe"])
    def test_ambiguous_word_alone_is_english(self, detector, word):
        assert detector.detect(word) == Language.ENGLISH

    def test_transliteration_vocabulary_itself_is_unaffected(self):
        from transliterate import COMMON_WORDS

        for word in ("the", "do", "me", "main", "log", "hello", "subscribe", "like", "channel", "video"):
            assert word in COMMON_WORDS
