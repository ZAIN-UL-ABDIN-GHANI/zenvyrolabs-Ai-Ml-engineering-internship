from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_VALID_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class VoiceProfileError(ValueError):
    """Raised when a VoiceProfile fails domain validation."""


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    name: str
    reference_audio_path: Path
    reference_text: str
    model_artifact_id: str | None = None

    def __post_init__(self) -> None:
        self.validate_name(self.name)
        if not self.reference_text.strip():
            raise VoiceProfileError("Reference text must not be empty.")
        if not isinstance(self.reference_audio_path, Path):
            object.__setattr__(self, "reference_audio_path", Path(self.reference_audio_path))

    @staticmethod
    def validate_name(name: str) -> None:
        if not _VALID_NAME_PATTERN.match(name):
            raise VoiceProfileError(
                f"Voice profile name {name!r} must be 1-64 characters "
                "of letters, digits, underscore, or hyphen only."
            )

    @property
    def has_trained_model(self) -> bool:
        return self.model_artifact_id is not None

    def with_model_artifact(self, model_artifact_id: str) -> VoiceProfile:
        return VoiceProfile(
            name=self.name,
            reference_audio_path=self.reference_audio_path,
            reference_text=self.reference_text,
            model_artifact_id=model_artifact_id,
        )
