from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from pathlib import Path


class TrainingSessionError(ValueError):
    """Raised when a TrainingSession fails domain validation."""


@unique
class TrainingStage(Enum):
    RAW = "raw"
    NORMALIZED = "normalized"
    FILTERED = "filtered"
    SEGMENTED = "segmented"
    DATASET_READY = "dataset_ready"


@dataclass(slots=True)
class TrainingSession:
    session_id: str
    raw_audio_paths: list[Path] = field(default_factory=list)
    stage: TrainingStage = TrainingStage.RAW
    segment_paths: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise TrainingSessionError("Session id must not be empty.")

    def add_raw_audio(self, path: Path) -> None:
        if self.stage is not TrainingStage.RAW:
            raise TrainingSessionError(
                f"Cannot add raw audio once session has advanced to {self.stage.value!r}."
            )
        self.raw_audio_paths.append(Path(path))

    def advance_to(self, stage: TrainingStage) -> None:
        if not self.raw_audio_paths:
            raise TrainingSessionError("Cannot advance a session with no raw audio.")
        ordering = list(TrainingStage)
        if ordering.index(stage) <= ordering.index(self.stage):
            raise TrainingSessionError(
                f"Cannot move from {self.stage.value!r} back to {stage.value!r}."
            )
        self.stage = stage

    @property
    def is_ready_for_dataset(self) -> bool:
        return self.stage is TrainingStage.DATASET_READY
