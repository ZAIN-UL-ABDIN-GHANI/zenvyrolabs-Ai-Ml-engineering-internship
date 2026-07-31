from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import noisereduce as nr
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from voice_studio.domain import TrainingSession, TrainingStage


class TrainingPreprocessError(RuntimeError):
    """Raised when training audio cannot be preprocessed."""


@dataclass(frozen=True, slots=True)
class PreprocessSettings:
    target_dbfs: float = -20.0
    silence_threshold_db: float = -40.0
    min_silence_len_ms: int = 400
    keep_silence_ms: int = 150
    chunk_seconds: float = 10.0
    min_final_chunk_seconds: float = 2.0
    sample_rate: int = 16000
    noise_reduction_strength: float = 0.75  # 0-1; conservative to avoid over-suppressing speech


class TrainingAudioPreprocessor:
    """Audio Processing Layer component completing the Task 3 training pipeline (SDS §9).

    Runs normalization, then noise reduction, then silence removal,
    then segmentation — the two middle stages did not exist at all in
    the original implementation (Stage 1 Analysis, F-10).
    """

    def __init__(self, settings: PreprocessSettings | None = None) -> None:
        self._settings = settings or PreprocessSettings()

    def process(self, session: TrainingSession, output_dir: Path) -> TrainingSession:
        if not session.raw_audio_paths:
            raise TrainingPreprocessError(f"Session {session.session_id!r} has no raw audio.")

        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_index = 0

        for raw_path in session.raw_audio_paths:
            audio = self._load_and_normalize(raw_path)
            audio = self._reduce_noise(audio)
            audio = self._remove_silence(audio)

            for chunk in self._segment(audio):
                chunk_path = output_dir / f"chunk_{chunk_index:04d}.wav"
                chunk.export(chunk_path, format="wav")
                session.segment_paths.append(chunk_path)
                chunk_index += 1

        session.advance_to(TrainingStage.NORMALIZED)
        session.advance_to(TrainingStage.FILTERED)
        session.advance_to(TrainingStage.SEGMENTED)
        return session

    def _load_and_normalize(self, path: Path) -> AudioSegment:
        audio = (
            AudioSegment.from_file(path)
            .set_channels(1)
            .set_frame_rate(self._settings.sample_rate)
        )
        if audio.dBFS == float("-inf"):
            return audio  # a fully silent clip cannot be gain-normalized; pass through untouched
        return audio.apply_gain(self._settings.target_dbfs - audio.dBFS)

    def _reduce_noise(self, audio: AudioSegment) -> AudioSegment:
        if audio.dBFS == float("-inf"):
            return audio  # nothing to denoise in silence; avoids a divide-by-zero in the spectral gate
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        reduced = nr.reduce_noise(
            y=samples,
            sr=audio.frame_rate,
            prop_decrease=self._settings.noise_reduction_strength,
        )
        reduced_int16 = np.clip(reduced, -32768, 32767).astype(np.int16)
        return AudioSegment(
            data=reduced_int16.tobytes(),
            sample_width=audio.sample_width,
            frame_rate=audio.frame_rate,
            channels=audio.channels,
        )

    def _remove_silence(self, audio: AudioSegment) -> AudioSegment:
        ranges = detect_nonsilent(
            audio,
            min_silence_len=self._settings.min_silence_len_ms,
            silence_thresh=self._settings.silence_threshold_db,
        )
        if not ranges:
            return audio

        padding = self._settings.keep_silence_ms
        trimmed = AudioSegment.empty()
        for start, end in ranges:
            trimmed += audio[max(start - padding, 0) : min(end + padding, len(audio))]
        return trimmed

    def _segment(self, audio: AudioSegment) -> Iterator[AudioSegment]:
        chunk_ms = int(self._settings.chunk_seconds * 1000)
        min_final_ms = int(self._settings.min_final_chunk_seconds * 1000)
        for start in range(0, len(audio), chunk_ms):
            chunk = audio[start : start + chunk_ms]
            if len(chunk) >= min_final_ms:
                yield chunk
