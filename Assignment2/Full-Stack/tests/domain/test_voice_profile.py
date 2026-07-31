from __future__ import annotations

import pytest

from voice_studio.domain import VoiceProfile, VoiceProfileError


class TestVoiceProfileValidation:
    def test_accepts_a_well_formed_name(self, real_wav_file):
        profile = VoiceProfile(name="ARIA", reference_audio_path=real_wav_file, reference_text="Hello.")
        assert profile.name == "ARIA"

    @pytest.mark.parametrize(
        "bad_name",
        [
            "bad name!",
            "../../etc/passwd",
            "/tmp/evil_abs",
            "",
            "x" * 65,
        ],
    )
    def test_rejects_unsafe_or_invalid_names(self, real_wav_file, bad_name):
        # Regression test for Stage 1 Analysis F-06 (path-traversal via voice name).
        with pytest.raises(VoiceProfileError):
            VoiceProfile(name=bad_name, reference_audio_path=real_wav_file, reference_text="Hello.")

    def test_rejects_empty_reference_text(self, real_wav_file):
        with pytest.raises(VoiceProfileError):
            VoiceProfile(name="ARIA", reference_audio_path=real_wav_file, reference_text="   ")

    def test_coerces_string_path_to_path_object(self, real_wav_file):
        profile = VoiceProfile(name="ARIA", reference_audio_path=str(real_wav_file), reference_text="Hi.")
        assert profile.reference_audio_path == real_wav_file


class TestVoiceProfileModelLinking:
    def test_has_trained_model_is_false_by_default(self, real_wav_file):
        profile = VoiceProfile(name="ARIA", reference_audio_path=real_wav_file, reference_text="Hi.")
        assert profile.has_trained_model is False

    def test_with_model_artifact_returns_a_new_linked_profile(self, real_wav_file):
        profile = VoiceProfile(name="ARIA", reference_audio_path=real_wav_file, reference_text="Hi.")
        linked = profile.with_model_artifact("model-001")

        assert linked.has_trained_model is True
        assert linked.model_artifact_id == "model-001"
        assert profile.has_trained_model is False  # original is unchanged (frozen/immutable)
