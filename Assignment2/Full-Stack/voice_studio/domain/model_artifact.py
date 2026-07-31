from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path


class ModelArtifactError(ValueError):
    """Raised when a ModelArtifact fails domain validation."""


@unique
class EngineType(Enum):
    F5_TTS = "f5_tts"
    RVC = "rvc"


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    artifact_id: str
    engine_type: EngineType
    model_path: Path
    index_path: Path | None = None
    source_session_id: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ModelArtifactError("Artifact id must not be empty.")
        if not isinstance(self.model_path, Path):
            object.__setattr__(self, "model_path", Path(self.model_path))
        if self.index_path is not None and not isinstance(self.index_path, Path):
            object.__setattr__(self, "index_path", Path(self.index_path))

    @property
    def has_retrieval_index(self) -> bool:
        return self.index_path is not None
