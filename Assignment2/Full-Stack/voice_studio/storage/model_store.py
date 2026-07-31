from __future__ import annotations

from pathlib import Path

from voice_studio.domain import EngineType, ModelArtifact

_MIN_VALID_MODEL_BYTES = 1_000_000  # real RVC checkpoints run tens of MB+; anything
# under this is almost certainly corrupted or placeholder data (Stage 1 Analysis, F-01).


class ModelNotFoundError(LookupError):
    """Raised when a requested model artifact does not exist in the store."""


class ModelIntegrityError(ValueError):
    """Raised when a model file exists but is too small to be a real checkpoint."""


class ModelStoreRepository:
    """Filesystem-backed catalog of ModelArtifact aggregates (SDS §14).

    Realizes the Model Store for file-pair-based engines (currently RVC),
    reusing the existing flat rvc_models/ directory layout unchanged.
    """

    _MODEL_SUFFIX = ".pth"
    _INDEX_SUFFIX = ".index"

    def __init__(self, root: Path, engine_type: EngineType = EngineType.RVC) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._engine_type = engine_type

    def list_artifact_ids(self) -> list[str]:
        return sorted(p.stem for p in self._root.glob(f"*{self._MODEL_SUFFIX}"))

    def load(self, artifact_id: str) -> ModelArtifact:
        model_path = self._resolve(artifact_id, self._MODEL_SUFFIX)
        if not model_path.is_file():
            raise ModelNotFoundError(f"No model artifact named {artifact_id!r}.")

        size = model_path.stat().st_size
        if size < _MIN_VALID_MODEL_BYTES:
            raise ModelIntegrityError(
                f"Model artifact {artifact_id!r} is only {size} bytes — under the "
                f"{_MIN_VALID_MODEL_BYTES}-byte sanity floor for a real checkpoint, "
                "likely corrupted or a placeholder file."
            )

        index_path = self._resolve(artifact_id, self._INDEX_SUFFIX)
        return ModelArtifact(
            artifact_id=artifact_id,
            engine_type=self._engine_type,
            model_path=model_path,
            index_path=index_path if index_path.is_file() else None,
        )

    def exists(self, artifact_id: str) -> bool:
        return self._resolve(artifact_id, self._MODEL_SUFFIX).is_file()

    def _resolve(self, artifact_id: str, suffix: str) -> Path:
        if not artifact_id or any(sep in artifact_id for sep in ("/", "\\", "..")):
            raise ModelNotFoundError(f"Invalid model artifact id: {artifact_id!r}.")
        resolved = (self._root / f"{artifact_id}{suffix}").resolve()
        if resolved.parent != self._root:
            raise ModelNotFoundError(
                f"Resolved path for {artifact_id!r} escapes the model store root."
            )
        return resolved
