from __future__ import annotations

import pytest

from voice_studio.domain import EngineType, ModelArtifact, ModelArtifactError


class TestModelArtifact:
    def test_rejects_empty_artifact_id(self):
        with pytest.raises(ModelArtifactError):
            ModelArtifact(artifact_id="  ", engine_type=EngineType.RVC, model_path="model.pth")

    def test_has_retrieval_index_false_when_absent(self):
        artifact = ModelArtifact(artifact_id="m1", engine_type=EngineType.RVC, model_path="model.pth")
        assert artifact.has_retrieval_index is False

    def test_has_retrieval_index_true_when_present(self):
        artifact = ModelArtifact(
            artifact_id="m1",
            engine_type=EngineType.RVC,
            model_path="model.pth",
            index_path="model.index",
        )
        assert artifact.has_retrieval_index is True

    def test_coerces_string_paths(self):
        artifact = ModelArtifact(artifact_id="m1", engine_type=EngineType.RVC, model_path="model.pth")
        assert artifact.model_path.name == "model.pth"
