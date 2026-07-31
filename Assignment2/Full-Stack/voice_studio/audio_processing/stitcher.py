from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment


class AudioStitchError(RuntimeError):
    """Raised when audio segments cannot be stitched together."""


@dataclass(frozen=True, slots=True)
class StitchSettings:
    crossfade_ms: int = 120
    silence_padding_ms: int = 150


class AudioStitcher:
    """Audio Processing Layer component joining generated line clips (SDS §6.6).

    Each transition pads with a short silence then crossfades into it,
    smoothing the waveform's tail instead of truncating it abruptly —
    closing the "robotic cuts" defect from the Stage 1 Analysis (F-15).
    """

    def __init__(self, settings: StitchSettings | None = None) -> None:
        self._settings = settings or StitchSettings()

    def stitch(self, segment_paths: list[Path], output_path: Path) -> Path:
        if not segment_paths:
            raise AudioStitchError("Cannot stitch an empty list of audio segments.")

        output_path = Path(output_path)
        combined = AudioSegment.from_file(segment_paths[0])
        for segment_path in segment_paths[1:]:
            combined = self._append_with_crossfade(combined, segment_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(output_path, format=output_path.suffix.lstrip(".") or "wav")
        return output_path

    def _append_with_crossfade(self, combined: AudioSegment, segment_path: Path) -> AudioSegment:
        next_segment = AudioSegment.from_file(segment_path)
        padded = AudioSegment.silent(duration=self._settings.silence_padding_ms) + next_segment
        crossfade = min(self._settings.crossfade_ms, len(combined), len(padded))
        return combined.append(padded, crossfade=crossfade)
