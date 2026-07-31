from __future__ import annotations

from pathlib import Path

import pytest

from voice_studio.domain import VoiceProfile, VoiceProfileError
from voice_studio.storage import VoiceNotFoundError, VoiceStoreRepository


@pytest.fixture
def store(tmp_path: Path) -> VoiceStoreRepository:
    return VoiceStoreRepository(root=tmp_path / "saved_voices")


class TestVoiceStoreRepository:
    def test_save_then_load_round_trips(self, store, real_wav_file):
        store.save(VoiceProfile(name="ARIA", reference_audio_path=real_wav_file, reference_text="Hello."))
        loaded = store.load("ARIA")
        assert loaded.reference_text == "Hello."
        assert loaded.reference_audio_path.exists()

    def test_model_artifact_id_survives_save_and_load(self, store, real_wav_file):
        store.save(
            VoiceProfile(
                name="GOJO",
                reference_audio_path=real_wav_file,
                reference_text="Hi.",
                model_artifact_id="model-001",
            )
        )
        assert store.load("GOJO").model_artifact_id == "model-001"

    def test_list_names_reflects_saved_voices(self, store, real_wav_file):
        store.save(VoiceProfile(name="ARIA", reference_audio_path=real_wav_file, reference_text="Hi."))
        store.save(VoiceProfile(name="GOJO", reference_audio_path=real_wav_file, reference_text="Hi."))
        assert store.list_names() == ["ARIA", "GOJO"]

    def test_delete_removes_the_voice(self, store, real_wav_file):
        store.save(VoiceProfile(name="ARIA", reference_audio_path=real_wav_file, reference_text="Hi."))
        store.delete("ARIA")
        assert store.list_names() == []

    def test_load_missing_voice_raises_not_found(self, store):
        with pytest.raises(VoiceNotFoundError):
            store.load("does_not_exist")

    def test_delete_missing_voice_raises_not_found(self, store):
        with pytest.raises(VoiceNotFoundError):
            store.delete("does_not_exist")

    @pytest.mark.parametrize("malicious_name", ["../../etc/passwd", "/tmp/evil_abs", "..\\..\\evil"])
    def test_path_traversal_is_blocked(self, store, malicious_name):
        # Reproduction of Stage 1 Analysis F-06.
        with pytest.raises(VoiceProfileError):
            store.load(malicious_name)
