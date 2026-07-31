from __future__ import annotations

from pathlib import Path

import pytest

from voice_studio.storage import ModelIntegrityError, ModelNotFoundError, ModelStoreRepository

_REAL_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REAL_GOJO_MODEL = _REAL_PROJECT_ROOT / "rvc_models" / "Gojo_Satoru.pth"


@pytest.fixture
def model_store_root(tmp_path: Path) -> Path:
    return tmp_path / "rvc_models"


@pytest.fixture
def store(model_store_root: Path) -> ModelStoreRepository:
    return ModelStoreRepository(root=model_store_root)


class TestModelStoreRepository:
    def test_list_artifact_ids_finds_pth_files(self, store, model_store_root):
        (model_store_root / "voice_a.pth").write_bytes(b"0" * 2_000_000)
        (model_store_root / "voice_b.pth").write_bytes(b"0" * 2_000_000)
        assert store.list_artifact_ids() == ["voice_a", "voice_b"]

    def test_load_returns_artifact_with_matching_index(self, store, model_store_root):
        (model_store_root / "voice_a.pth").write_bytes(b"0" * 2_000_000)
        (model_store_root / "voice_a.index").write_bytes(b"0" * 100)
        artifact = store.load("voice_a")
        assert artifact.has_retrieval_index is True

    def test_load_missing_artifact_raises_not_found(self, store):
        with pytest.raises(ModelNotFoundError):
            store.load("does_not_exist")

    def test_undersized_file_raises_integrity_error(self, store, model_store_root):
        # A corrupted/placeholder file well under any real checkpoint's size.
        (model_store_root / "broken.pth").write_bytes(b"0" * 332)
        with pytest.raises(ModelIntegrityError):
            store.load("broken")

    @pytest.mark.parametrize("malicious_id", ["../../etc/passwd", "a/b", "a\\b", "..", ""])
    def test_path_traversal_is_blocked(self, store, malicious_id):
        with pytest.raises(ModelNotFoundError):
            store.load(malicious_id)

    @pytest.mark.skipif(
        not _REAL_GOJO_MODEL.exists(), reason="real Gojo_Satoru.pth not present in this checkout"
    )
    def test_the_actual_corrupted_gojo_model_is_flagged(self):
        # Direct regression test for Stage 1 Analysis F-01, against the real
        # 332-byte AppleDouble stub shipped in the original archive.
        real_store = ModelStoreRepository(root=_REAL_GOJO_MODEL.parent)
        with pytest.raises(ModelIntegrityError):
            real_store.load("Gojo_Satoru")
