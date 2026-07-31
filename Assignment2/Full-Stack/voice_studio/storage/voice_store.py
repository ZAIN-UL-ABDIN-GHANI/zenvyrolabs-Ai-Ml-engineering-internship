from __future__ import annotations

import shutil
from pathlib import Path

from voice_studio.domain import VoiceProfile, VoiceProfileError


class VoiceNotFoundError(LookupError):
    """Raised when a requested voice profile does not exist in the store."""


class VoiceStoreRepository:
    """Filesystem-backed persistence for VoiceProfile aggregates (SDS §14).

    Every path is derived exclusively from a name validated by
    VoiceProfile.validate_name, then re-checked for containment within
    the store root — closing the path-traversal defect from the Stage 1
    Analysis (F-06) with defense-in-depth per SDS §18.2.
    """

    _AUDIO_FILENAME = "audio.wav"
    _TEXT_FILENAME = "text.txt"
    _MODEL_LINK_FILENAME = "model_artifact_id.txt"

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, profile: VoiceProfile) -> VoiceProfile:
        voice_dir = self._voice_dir(profile.name)
        voice_dir.mkdir(parents=True, exist_ok=True)

        destination_audio = voice_dir / self._AUDIO_FILENAME
        source_audio = Path(profile.reference_audio_path).resolve()
        if source_audio != destination_audio.resolve():
            shutil.copyfile(source_audio, destination_audio)

        (voice_dir / self._TEXT_FILENAME).write_text(profile.reference_text, encoding="utf-8")

        model_link = voice_dir / self._MODEL_LINK_FILENAME
        if profile.model_artifact_id is not None:
            model_link.write_text(profile.model_artifact_id, encoding="utf-8")
        elif model_link.exists():
            model_link.unlink()

        return VoiceProfile(
            name=profile.name,
            reference_audio_path=destination_audio,
            reference_text=profile.reference_text,
            model_artifact_id=profile.model_artifact_id,
        )

    def load(self, name: str) -> VoiceProfile:
        voice_dir = self._voice_dir(name)
        if not voice_dir.is_dir():
            raise VoiceNotFoundError(f"No saved voice profile named {name!r}.")

        text_path = voice_dir / self._TEXT_FILENAME
        reference_text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else ""

        model_link = voice_dir / self._MODEL_LINK_FILENAME
        model_artifact_id = model_link.read_text(encoding="utf-8").strip() if model_link.exists() else None

        return VoiceProfile(
            name=name,
            reference_audio_path=voice_dir / self._AUDIO_FILENAME,
            reference_text=reference_text or "(no reference text saved)",
            model_artifact_id=model_artifact_id or None,
        )

    def delete(self, name: str) -> None:
        voice_dir = self._voice_dir(name)
        if not voice_dir.is_dir():
            raise VoiceNotFoundError(f"No saved voice profile named {name!r}.")
        shutil.rmtree(voice_dir)

    def list_names(self) -> list[str]:
        return sorted(entry.name for entry in self._root.iterdir() if entry.is_dir())

    def _voice_dir(self, name: str) -> Path:
        VoiceProfile.validate_name(name)
        resolved = (self._root / name).resolve()
        if resolved.parent != self._root:
            raise VoiceProfileError(f"Resolved path for {name!r} escapes the voice store root.")
        return resolved
