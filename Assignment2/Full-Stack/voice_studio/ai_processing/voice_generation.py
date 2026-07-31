from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from voice_studio.config import EngineSettings, PathSettings
from voice_studio.infrastructure import ProcessRunner

_OUTPUT_FILENAME_PATTERN = re.compile(r"([\w\-]+\.wav)")


class VoiceGenerationError(RuntimeError):
    """Raised when the Voice Generation engine fails to produce audio."""


@dataclass(frozen=True, slots=True)
class VoiceGenerationRequest:
    reference_audio_path: Path
    reference_text: str
    target_text: str
    output_dir: Path
    speed: float = 1.0


class VoiceGenerationEngine:
    """AI Processing Layer component wrapping the F5-TTS zero-shot engine (SDS §6.2)."""

    def __init__(
        self,
        paths: PathSettings,
        engine: EngineSettings,
        runner: ProcessRunner | None = None,
    ) -> None:
        self._paths = paths
        self._engine = engine
        self._runner = runner or ProcessRunner(timeout_seconds=300)

    def generate(self, request: VoiceGenerationRequest) -> Path:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        config_path = self._write_config(request)
        result = self._runner.run(self._paths.f5_tts_executable, ["-c", str(config_path)])
        return self._extract_output_path(result.stdout, request.output_dir)

    def _write_config(self, request: VoiceGenerationRequest) -> Path:
        config = {
            "model": self._engine.f5_tts_model_name,
            "ref_audio": str(request.reference_audio_path),
            "ref_text": request.reference_text,
            "gen_text": request.target_text,
            "speed": request.speed,
            "nfe_step": self._engine.f5_tts_nfe_step,
            "output_dir": str(request.output_dir),
        }
        config_path = request.output_dir / "generation_config.toml"
        config_path.write_bytes(tomli_w.dumps(config).encode("utf-8"))
        return config_path

    def _extract_output_path(self, stdout: str, output_dir: Path) -> Path:
        match = _OUTPUT_FILENAME_PATTERN.search(stdout)
        if match:
            candidate = output_dir / match.group(1)
            if candidate.exists():
                return candidate
        generated = sorted(output_dir.glob("infer_cli_*.wav"), key=lambda p: p.stat().st_mtime)
        if generated:
            return generated[-1]
        raise VoiceGenerationError(
            f"F5-TTS reported success but no output .wav file could be located in {output_dir}."
        )
