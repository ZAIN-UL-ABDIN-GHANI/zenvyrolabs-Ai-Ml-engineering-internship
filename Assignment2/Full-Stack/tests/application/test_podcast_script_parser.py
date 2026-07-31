from __future__ import annotations

from voice_studio.application import PodcastScriptParser


class TestPodcastScriptParser:
    def test_parses_a_well_formed_line(self):
        result = PodcastScriptParser().parse("NARUTO: Hey guys!")
        assert len(result.script.lines) == 1
        assert result.script.lines[0].speaker == "NARUTO"
        assert result.script.lines[0].text == "Hey guys!"

    def test_extra_space_before_colon_still_parses(self):
        # The literal example the original README cited as crash-inducing.
        result = PodcastScriptParser().parse("NARUTO : Hey guys!")
        assert result.script.lines[0].speaker == "NARUTO"
        assert not result.warnings

    def test_multi_word_speaker_name_now_parses(self):
        # F-14: previously silently dropped by the alnum/underscore-only regex.
        result = PodcastScriptParser().parse("IRON MAN: Hello there!")
        assert result.script.lines[0].speaker == "IRON MAN"

    def test_punctuated_speaker_name_now_parses(self):
        # F-14: previously silently dropped (apostrophe rejected by old regex).
        result = PodcastScriptParser().parse("O'Brien: Hi there")
        assert result.script.lines[0].speaker == "O'Brien"

    def test_colon_inside_dialogue_does_not_truncate_the_line(self):
        result = PodcastScriptParser().parse("NARUTO: Meet me at 3:00, ok?")
        assert result.script.lines[0].text == "Meet me at 3:00, ok?"

    def test_line_with_no_colon_produces_a_warning_not_silent_loss(self):
        # F-14: previously vanished with no trace.
        result = PodcastScriptParser().parse("Some random line with no colon")
        assert result.script.is_empty
        assert len(result.warnings) == 1
        assert "no ':' separator" in result.warnings[0]

    def test_blank_lines_are_skipped_without_warning(self):
        result = PodcastScriptParser().parse("NARUTO: Hey!\n\n\nLUFFY: Yo!")
        assert len(result.script.lines) == 2
        assert not result.warnings

    def test_mixed_valid_and_invalid_lines_keeps_valid_ones_and_warns_on_invalid(self):
        script_text = "NARUTO: Hey!\nno colon here\nLUFFY: Yo!"
        result = PodcastScriptParser().parse(script_text)
        assert [line.speaker for line in result.script.lines] == ["NARUTO", "LUFFY"]
        assert len(result.warnings) == 1

    def test_overlong_speaker_name_is_rejected_with_a_warning(self):
        result = PodcastScriptParser().parse(f"{'X' * 51}: Hello")
        assert result.script.is_empty
        assert result.has_warnings
