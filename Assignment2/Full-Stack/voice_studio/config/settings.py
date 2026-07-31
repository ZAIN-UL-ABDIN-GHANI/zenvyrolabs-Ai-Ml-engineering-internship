from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _project_root() -> Path:
    override = os.environ.get("VOICE_STUDIO_ROOT")
    return Path(override).resolve() if override else Path(__file__).resolve().parents[2]


def _venv_executable(venv_dir: Path, name: str) -> Path:
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class PathSettings:
    root: Path
    temp_dir: Path
    saved_voices_dir: Path
    rvc_models_dir: Path
    training_data_dir: Path
    hf_cache_dir: Path
    edge_tts_executable: Path
    f5_tts_executable: Path
    rvc_python_executable: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> PathSettings:
        resolved_root = (root or _project_root()).resolve()
        primary_venv = resolved_root / "venv"
        rvc_venv = resolved_root / "rvc_venv"
        return cls(
            root=resolved_root,
            temp_dir=resolved_root / "temp",
            saved_voices_dir=resolved_root / "saved_voices",
            rvc_models_dir=resolved_root / "rvc_models",
            training_data_dir=resolved_root / "training_data",
            hf_cache_dir=resolved_root / "hf_cache",
            edge_tts_executable=_venv_executable(primary_venv, "edge-tts"),
            f5_tts_executable=_venv_executable(primary_venv, "f5-tts_infer-cli"),
            rvc_python_executable=_venv_executable(rvc_venv, "python"),
        )


@dataclass(frozen=True, slots=True)
class EngineSettings:
    f5_tts_model_name: str = "F5TTS_Base"
    f5_tts_nfe_step: int = 16
    default_narration_voice: str = "en-US-GuyNeural"
    rvc_default_pitch: int = 0
    rvc_default_f0_method: str = "rmvpe"


@dataclass(frozen=True, slots=True)
class FeatureFlags:
    enable_dramatic_story_mode: bool = False
    enable_voice_to_voice_rvc: bool = False
    enable_perfect_pronunciation_clone: bool = False
    auto_apply_pronunciation_routing: bool = True


@dataclass(frozen=True, slots=True)
class VoiceStudioSettings:
    paths: PathSettings
    engine: EngineSettings
    features: FeatureFlags

    @classmethod
    def load(cls, root: Path | None = None) -> VoiceStudioSettings:
        return cls(
            paths=PathSettings.from_root(root),
            engine=EngineSettings(
                f5_tts_model_name=os.environ.get("VOICE_STUDIO_F5TTS_MODEL", "F5TTS_Base"),
                default_narration_voice=os.environ.get(
                    "VOICE_STUDIO_NARRATION_VOICE", "en-US-GuyNeural"
                ),
            ),
            features=FeatureFlags(
                enable_dramatic_story_mode=_env_flag("VOICE_STUDIO_ENABLE_DRAMATIC_MODE"),
                enable_voice_to_voice_rvc=_env_flag("VOICE_STUDIO_ENABLE_RVC_TAB"),
                enable_perfect_pronunciation_clone=_env_flag("VOICE_STUDIO_ENABLE_PERFECT_CLONE"),
                auto_apply_pronunciation_routing=_env_flag(
                    "VOICE_STUDIO_AUTO_ROUTE_PRONUNCIATION", default=True
                ),
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> VoiceStudioSettings:
    return VoiceStudioSettings.load()
