from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voice_studio.config import PathSettings
from voice_studio.infrastructure import ProcessRunner


class NeuralNarrationError(RuntimeError):
    """Raised when the Neural Narration engine fails to produce audio."""


@dataclass(frozen=True, slots=True)
class NarrationRequest:
    text: str
    output_path: Path
    voice: str
    rate_percent: int = 0
    pitch_hz: int = 0


class NeuralNarrationEngine:
    """AI Processing Layer component wrapping Microsoft Edge-TTS (SDS §5.4)."""

    def __init__(self, paths: PathSettings, runner: ProcessRunner | None = None) -> None:
        self._paths = paths
        self._runner = runner or ProcessRunner(timeout_seconds=120)

    def narrate(self, request: NarrationRequest) -> Path:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        arguments = [
            "--voice", request.voice,
            "--text", request.text,
            "--write-media", str(request.output_path),
            "--rate", self._signed_percent(request.rate_percent),
            "--pitch", self._signed_hertz(request.pitch_hz),
        ]
        self._runner.run(self._paths.edge_tts_executable, arguments)

        if not request.output_path.exists() or request.output_path.stat().st_size == 0:
            raise NeuralNarrationError(
                f"Edge-TTS reported success but produced no audio at {request.output_path}."
            )
        return request.output_path

    @staticmethod
    def _signed_percent(value: int) -> str:
        return f"{value:+d}%"

    @staticmethod
    def _signed_hertz(value: int) -> str:
        return f"{value:+d}Hz"
