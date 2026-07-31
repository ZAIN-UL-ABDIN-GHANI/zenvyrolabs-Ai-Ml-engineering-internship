from __future__ import annotations

from dataclasses import dataclass, field


class DialogueLineError(ValueError):
    """Raised when a DialogueLine fails domain validation."""


class PodcastScriptError(ValueError):
    """Raised when a PodcastScript fails domain validation."""


@dataclass(frozen=True, slots=True)
class DialogueLine:
    speaker: str
    text: str
    line_number: int

    def __post_init__(self) -> None:
        if not self.speaker.strip():
            raise DialogueLineError("Speaker must not be empty.")
        if not self.text.strip():
            raise DialogueLineError("Dialogue text must not be empty.")
        if self.line_number < 1:
            raise DialogueLineError("Line number must be a positive integer.")


@dataclass(slots=True)
class PodcastScript:
    title: str
    lines: list[DialogueLine] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise PodcastScriptError("Script title must not be empty.")

    def add_line(self, speaker: str, text: str) -> DialogueLine:
        line = DialogueLine(speaker=speaker, text=text, line_number=len(self.lines) + 1)
        self.lines.append(line)
        return line

    @property
    def speakers(self) -> set[str]:
        return {line.speaker for line in self.lines}

    @property
    def is_empty(self) -> bool:
        return len(self.lines) == 0
