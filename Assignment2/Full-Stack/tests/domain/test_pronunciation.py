from __future__ import annotations

from voice_studio.domain import Language, PronunciationPolicy, RoutingDecision


class TestPronunciationPolicy:
    def test_english_text_routes_to_direct_clone(self):
        policy = PronunciationPolicy()
        assert policy.decide("Hey everyone, welcome!") is RoutingDecision.DIRECT_CLONE

    def test_romanized_hindi_routes_to_hybrid(self):
        policy = PronunciationPolicy()
        assert policy.decide("Subscribe karo aur like karo") is RoutingDecision.HYBRID_NATIVE_THEN_CONVERT

    def test_devanagari_routes_to_hybrid(self):
        policy = PronunciationPolicy()
        assert policy.decide("हेलो क्या हाल है भाई") is RoutingDecision.HYBRID_NATIVE_THEN_CONVERT

    def test_evaluate_exposes_both_language_and_decision(self):
        policy = PronunciationPolicy()
        evaluation = policy.evaluate("Subscribe karo aur like karo")
        assert evaluation.language is Language.ROMANIZED_HINDI_URDU
        assert evaluation.decision is RoutingDecision.HYBRID_NATIVE_THEN_CONVERT

    def test_mixed_hindi_english_routes_to_hybrid(self):
        policy = PronunciationPolicy()
        assert policy.decide("Hey karo the video and subscribe") is RoutingDecision.HYBRID_NATIVE_THEN_CONVERT

    def test_mixed_urdu_english_native_script_routes_to_hybrid(self):
        policy = PronunciationPolicy()
        assert policy.decide("یہ ایک اچھا دن ہے and today is great") is RoutingDecision.HYBRID_NATIVE_THEN_CONVERT

    def test_youtube_style_english_never_falsely_routes_to_hybrid(self):
        policy = PronunciationPolicy()
        text = "Subscribe to my channel and share this video, please comment below"
        assert policy.decide(text) is RoutingDecision.DIRECT_CLONE
