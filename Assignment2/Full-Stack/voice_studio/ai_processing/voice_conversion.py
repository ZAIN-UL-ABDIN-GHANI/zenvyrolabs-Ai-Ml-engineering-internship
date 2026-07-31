from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voice_studio.config import EngineSettings, PathSettings
from voice_studio.domain import ModelArtifact
from voice_studio.infrastructure import ProcessRunner


class VoiceConversionError(RuntimeError):
    """Raised when the Voice Conversion engine fails to produce audio."""


@dataclass(frozen=True, slots=True)
class ConversionRequest:
    input_audio_path: Path
    output_audio_path: Path
    model: ModelArtifact
    pitch_shift: int = 0
    f0_method: str = "rmvpe"


class VoiceConversionEngine:
    """AI Processing Layer component wrapping RVC, run in its isolated runtime (SDS D-3)."""

    _INFER_SCRIPT = "rvc_infer.py"

    def __init__(
        self,
        paths: PathSettings,
        engine: EngineSettings,
        runner: ProcessRunner | None = None,
    ) -> None:
        self._paths = paths
        self._engine = engine
        self._runner = runner or ProcessRunner(timeout_seconds=180)

    def convert(self, request: ConversionRequest) -> Path:
        request.output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        script_path = self._paths.root / self._INFER_SCRIPT

        arguments = [
            str(script_path),
            "--input", str(request.input_audio_path),
            "--output", str(request.output_audio_path),
            "--model", str(request.model.model_path),
            "--pitch", str(request.pitch_shift),
            "--method", request.f0_method,
        ]
        if request.model.has_retrieval_index:
            arguments += ["--index", str(request.model.index_path)]

        self._runner.run(self._paths.rvc_python_executable, arguments)

        if not request.output_audio_path.exists():
            raise VoiceConversionError(
                f"RVC reported success but produced no audio at {request.output_audio_path}."
            )
        return request.output_audio_path
