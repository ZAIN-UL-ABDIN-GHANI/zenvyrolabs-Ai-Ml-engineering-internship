from __future__ import annotations

import pytest

from voice_studio.domain import DialogueLineError, PodcastScript, PodcastScriptError


class TestDialogueLine:
    def test_add_line_assigns_sequential_line_numbers(self):
        script = PodcastScript(title="Episode 1")
        first = script.add_line("NARUTO", "Hey!")
        second = script.add_line("LUFFY", "Yo!")

        assert first.line_number == 1
        assert second.line_number == 2

    def test_rejects_empty_speaker(self):
        script = PodcastScript(title="Episode 1")
        with pytest.raises(DialogueLineError):
            script.add_line("", "Hey!")

    def test_rejects_empty_text(self):
        script = PodcastScript(title="Episode 1")
        with pytest.raises(DialogueLineError):
            script.add_line("NARUTO", "   ")


class TestPodcastScript:
    def test_rejects_empty_title(self):
        with pytest.raises(PodcastScriptError):
            PodcastScript(title="  ")

    def test_is_empty_reflects_line_count(self):
        script = PodcastScript(title="Episode 1")
        assert script.is_empty is True
        script.add_line("NARUTO", "Hey!")
        assert script.is_empty is False

    def test_speakers_deduplicates(self):
        script = PodcastScript(title="Episode 1")
        script.add_line("NARUTO", "Hey!")
        script.add_line("NARUTO", "Again!")
        script.add_line("LUFFY", "Yo!")
        assert script.speakers == {"NARUTO", "LUFFY"}
