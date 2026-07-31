from __future__ import annotations

from dataclasses import dataclass, field

from voice_studio.domain import DialogueLineError, PodcastScript

_MAX_SPEAKER_LENGTH = 50


@dataclass(slots=True)
class ScriptParseResult:
    script: PodcastScript
    warnings: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class PodcastScriptParser:
    """Parses raw multi-speaker script text into a validated PodcastScript.

    Malformed lines are skipped individually and reported as warnings
    rather than silently dropped or failing the whole script.
    """

    def parse(self, raw_text: str, title: str = "Untitled Episode") -> ScriptParseResult:
        script = PodcastScript(title=title)
        warnings: list[str] = []

        for line_number, raw_line in enumerate(raw_text.strip().splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            speaker, separator, dialogue = line.partition(":")
            speaker = speaker.strip()
            dialogue = dialogue.strip()

            if not separator:
                warnings.append(f"Line {line_number}: no ':' separator found — skipped.")
                continue
            if not speaker:
                warnings.append(f"Line {line_number}: missing speaker name — skipped.")
                continue
            if len(speaker) > _MAX_SPEAKER_LENGTH:
                warnings.append(
                    f"Line {line_number}: speaker name exceeds "
                    f"{_MAX_SPEAKER_LENGTH} characters — skipped."
                )
                continue
            if not dialogue:
                warnings.append(f"Line {line_number}: empty dialogue for {speaker!r} — skipped.")
                continue

            try:
                script.add_line(speaker, dialogue)
            except DialogueLineError as error:
                warnings.append(f"Line {line_number}: {error}")

        return ScriptParseResult(script=script, warnings=warnings)
