from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from voice_studio.application.podcast_script_parser import PodcastScriptParser
from voice_studio.application.pronunciation_routing import (
    LineVoicingRequest,
    PronunciationRoutingService,
)
from voice_studio.audio_processing import AudioStitcher
from voice_studio.domain import ModelArtifact, VoiceProfile
from voice_studio.infrastructure import get_logger
from voice_studio.storage import (
    ModelIntegrityError,
    ModelNotFoundError,
    ModelStoreRepository,
    VoiceNotFoundError,
    VoiceStoreRepository,
)

_logger = get_logger("podcast_engine")


class PodcastEngineError(RuntimeError):
    """Raised when a podcast cannot be generated."""


@dataclass(slots=True)
class PodcastGenerationResult:
    output_path: Path
    warnings: list[str] = field(default_factory=list)


class PodcastEngine:
    """Application Layer orchestrator for the Multi-Voice Podcast capability (SDS §6.3).

    Parses the script, resolves each speaker to a saved voice, routes
    every line through Pronunciation Routing, then stitches the results
    with a crossfade — the full pipeline the Stage 1 Analysis found
    missing (F-09) or unrealistic (F-15) in the original tab.
    """

    def __init__(
        self,
        parser: PodcastScriptParser,
        voice_store: VoiceStoreRepository,
        model_store: ModelStoreRepository,
        routing_service: PronunciationRoutingService,
        stitcher: AudioStitcher,
    ) -> None:
        self._parser = parser
        self._voice_store = voice_store
        self._model_store = model_store
        self._routing_service = routing_service
        self._stitcher = stitcher

    def generate(self, raw_script: str, title: str, output_dir: Path) -> PodcastGenerationResult:
        _logger.info("Podcast generation started: title=%r", title)
        parse_result = self._parser.parse(raw_script, title=title)
        if parse_result.script.is_empty:
            _logger.error("Podcast generation aborted: no valid dialogue lines in script.")
            raise PodcastEngineError("Script contained no valid dialogue lines.")

        warnings = list(parse_result.warnings)
        for warning in warnings:
            _logger.warning(warning)

        segment_paths: list[Path] = []

        for line in parse_result.script.lines:
            try:
                voice_profile = self._voice_store.load(line.speaker)
            except VoiceNotFoundError:
                message = (
                    f"Line {line.line_number}: no saved voice for speaker "
                    f"{line.speaker!r} — line skipped."
                )
                warnings.append(message)
                _logger.warning(message)
                continue

            conversion_model = self._resolve_conversion_model(voice_profile, warnings, line.line_number)
            _logger.debug(
                "Voicing line %s for speaker %r (native routing model=%s)",
                line.line_number, line.speaker, conversion_model.artifact_id if conversion_model else None,
            )
            segment_path = self._routing_service.voice_line(
                LineVoicingRequest(
                    text=line.text,
                    voice_profile=voice_profile,
                    conversion_model=conversion_model,
                    output_dir=output_dir,
                    line_index=line.line_number,
                )
            )
            segment_paths.append(segment_path)

        if not segment_paths:
            _logger.error("Podcast generation failed: no lines could be voiced.")
            raise PodcastEngineError("No lines could be voiced; no saved voices matched the script.")

        output_path = output_dir / f"{self._safe_filename(title)}.wav"
        self._stitcher.stitch(segment_paths, output_path)
        _logger.info(
            "Podcast generation complete: title=%r lines_voiced=%d warnings=%d output=%s",
            title, len(segment_paths), len(warnings), output_path,
        )
        return PodcastGenerationResult(output_path=output_path, warnings=warnings)

    def _resolve_conversion_model(
        self, voice_profile: VoiceProfile, warnings: list[str], line_number: int
    ) -> ModelArtifact | None:
        if voice_profile.model_artifact_id is None:
            return None
        try:
            return self._model_store.load(voice_profile.model_artifact_id)
        except (ModelNotFoundError, ModelIntegrityError) as error:
            warnings.append(f"Line {line_number}: {error} — native routing will fail if needed.")
            return None

    @staticmethod
    def _safe_filename(title: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in title.strip()) or "podcast"
