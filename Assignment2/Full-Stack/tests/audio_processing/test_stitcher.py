from __future__ import annotations

from pathlib import Path

import pytest
from pydub import AudioSegment

from voice_studio.audio_processing import AudioStitcher, AudioStitchError, StitchSettings


class TestAudioStitcher:
    def test_stitch_rejects_empty_segment_list(self, tmp_path: Path):
        with pytest.raises(AudioStitchError):
            AudioStitcher().stitch([], tmp_path / "out.wav")

    def test_stitch_joins_segments_into_one_playable_file(self, tone_wav_factory, tmp_path: Path):
        segments = [tone_wav_factory(440, 0.5), tone_wav_factory(300, 0.5)]
        output_path = tmp_path / "combined.wav"

        result = AudioStitcher().stitch(segments, output_path)

        assert result == output_path
        combined = AudioSegment.from_file(output_path)
        assert len(combined) > 0

    def test_crossfade_shortens_total_duration_versus_plain_concatenation(
        self, tone_wav_factory, tmp_path: Path
    ):
        segments = [tone_wav_factory(440, 1.0), tone_wav_factory(300, 1.0)]
        naive_total_ms = sum(len(AudioSegment.from_file(p)) for p in segments)

        stitched = AudioStitcher(StitchSettings(crossfade_ms=200, silence_padding_ms=100)).stitch(
            segments, tmp_path / "out.wav"
        )
        stitched_ms = len(AudioSegment.from_file(stitched))

        # padding is added but the crossfade overlap means it's not simply
        # naive_total + padding — the hard-cut (F-15) baseline is strictly larger
        assert stitched_ms < naive_total_ms + 100 * len(segments)

    def test_single_segment_passes_through_unchanged_in_length(self, tone_wav_factory, tmp_path: Path):
        segment = tone_wav_factory(440, 0.5)
        original_len = len(AudioSegment.from_file(segment))
        result = AudioStitcher().stitch([segment], tmp_path / "out.wav")
        assert len(AudioSegment.from_file(result)) == original_len

    def test_accepts_plain_string_output_path(self, tone_wav_factory, tmp_path: Path):
        segments = [tone_wav_factory(440, 0.5), tone_wav_factory(300, 0.5)]
        result = AudioStitcher().stitch(segments, str(tmp_path / "out_str.wav"))
        assert isinstance(result, Path)
        assert result.exists()

    def test_near_max_amplitude_segments_do_not_clip(self, tmp_path: Path):
        import wave
        import numpy as np

        sr = 8000

        def loud_tone(path, freq, seconds, amplitude=31000):
            n = int(sr * seconds)
            t = np.arange(n) / sr
            data = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)
            with wave.open(str(path), "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(data.tobytes())

        clip1, clip2 = tmp_path / "loud1.wav", tmp_path / "loud2.wav"
        loud_tone(clip1, 440, 1.0)
        loud_tone(clip2, 300, 1.0)

        result = AudioStitcher().stitch([clip1, clip2], tmp_path / "stitched.wav")
        seg = AudioSegment.from_file(result)
        samples = np.array(seg.get_array_of_samples())
        assert np.abs(samples).max() < 32767  # never hits the digital ceiling
