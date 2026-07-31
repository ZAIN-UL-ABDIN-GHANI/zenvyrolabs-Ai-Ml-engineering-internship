from __future__ import annotations

import stat
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def real_wav_file(tmp_path: Path) -> Path:
    """A genuine minimal PCM WAV file (not fake bytes) for pydub-based tests."""
    import struct
    import wave

    path = tmp_path / "reference.wav"
    with wave.open(str(path), "w") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        frames = b"".join(struct.pack("<h", 0) for _ in range(8000))
        writer.writeframes(frames)
    return path


def _make_tone_wav(path: Path, freq: int, seconds: float, frame_rate: int = 8000) -> Path:
    import math
    import struct
    import wave

    n = int(frame_rate * seconds)
    with wave.open(str(path), "w") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(frame_rate)
        frames = b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * freq * i / frame_rate)))
            for i in range(n)
        )
        writer.writeframes(frames)
    return path


@pytest.fixture
def tone_wav_factory(tmp_path: Path):
    """Factory fixture producing real, decodable tone WAV files on demand."""
    counter = {"n": 0}

    def _factory(freq: int = 440, seconds: float = 0.5) -> Path:
        counter["n"] += 1
        return _make_tone_wav(tmp_path / f"tone_{counter['n']}.wav", freq, seconds)

    return _factory


@pytest.fixture
def voice_like_wav_factory(tmp_path: Path):
    """Factory producing synthetic but spectrally voice-like clips (harmonic
    series + amplitude envelope), since a plain sine tone is indistinguishable
    from stationary noise to a real noise-reduction algorithm."""
    import numpy as np
    import wave

    counter = {"n": 0}
    frame_rate = 16000

    def _factory(
        f0: int = 150,
        seconds: float = 4.0,
        silence_before: float = 0.0,
        silence_after: float = 0.0,
        noise_level: float = 400,
    ) -> Path:
        counter["n"] += 1
        n = int(frame_rate * seconds)
        t = np.arange(n) / frame_rate
        envelope = np.sin(np.pi * np.arange(n) / max(n, 1)) ** 0.5
        signal = sum(6000 / h * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 6)) * envelope
        noise = np.random.default_rng(42).normal(0, noise_level, n)
        voiced = (signal + noise).astype(np.int16)
        silence = np.zeros(int(frame_rate * silence_before), dtype=np.int16)
        trailing = np.zeros(int(frame_rate * silence_after), dtype=np.int16)
        samples = np.concatenate([silence, voiced, trailing])

        path = tmp_path / f"voice_{counter['n']}.wav"
        with wave.open(str(path), "w") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(frame_rate)
            writer.writeframes(samples.tobytes())
        return path

    return _factory


@pytest.fixture
def invalid_audio_file(tmp_path: Path) -> Path:
    """A file with an audio-like extension that is not actually decodable audio."""
    path = tmp_path / "invalid.wav"
    path.write_bytes(b"this is not audio data, just plain bytes")
    return path


@pytest.fixture
def app_module(tmp_path: Path, monkeypatch):
    """Loads the real app.py fresh, fully isolated to a temporary project root."""
    import importlib.util

    monkeypatch.setenv("VOICE_STUDIO_ROOT", str(tmp_path))
    from voice_studio.config.settings import get_settings

    get_settings.cache_clear()

    project_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(f"app_under_test_{id(tmp_path)}", project_root / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fully_silent_wav_factory(tmp_path: Path):
    """Factory producing the exact -inf dBFS edge case from Stage 1 Analysis F-10."""
    import wave

    counter = {"n": 0}

    def _factory(seconds: float = 2.0) -> Path:
        counter["n"] += 1
        path = tmp_path / f"silent_{counter['n']}.wav"
        with wave.open(str(path), "w") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b"\x00\x00" * int(16000 * seconds))
        return path

    return _factory


def _write_executable(path: Path, script: str) -> Path:
    path.write_text(textwrap.dedent(script))
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def fake_f5_tts_executable(tmp_path: Path) -> Path:
    """A stand-in f5-tts_infer-cli matching its real -c <config>/stdout contract."""
    return _write_executable(
        tmp_path / "fake_f5_tts_infer_cli",
        """\
        #!/usr/bin/env python3
        import sys, tomllib, wave, struct
        config_path = sys.argv[sys.argv.index("-c") + 1]
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        out_dir = config["output_dir"]
        out_file = f"{out_dir}/infer_cli_basic.wav"
        with wave.open(out_file, "w") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
            w.writeframes(struct.pack("<h", 0) * 800)
        print(f"Generation complete. Saved to infer_cli_basic.wav in {out_dir}")
        """,
    )


@pytest.fixture
def fake_edge_tts_executable(tmp_path: Path) -> Path:
    """A stand-in edge-tts matching its real --write-media contract."""
    return _write_executable(
        tmp_path / "fake_edge_tts",
        """\
        #!/usr/bin/env python3
        import sys, wave, struct
        out = sys.argv[sys.argv.index("--write-media") + 1]
        with wave.open(out, "w") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
            w.writeframes(struct.pack("<h", 0) * 800)
        print("Edge-TTS narration complete.")
        """,
    )


@pytest.fixture
def fake_rvc_python_executable(tmp_path: Path) -> Path:
    """A stand-in for the isolated rvc_venv's python running rvc_infer.py."""
    return _write_executable(
        tmp_path / "fake_rvc_python",
        """\
        #!/usr/bin/env python3
        import sys, wave, struct
        args = sys.argv[2:]  # skip fake-python + rvc_infer.py script path
        out = args[args.index("--output") + 1]
        with wave.open(out, "w") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
            w.writeframes(struct.pack("<h", 0) * 800)
        print("RVC conversion complete.")
        """,
    )


@pytest.fixture
def fake_failing_executable(tmp_path: Path) -> Path:
    """A stand-in that always exits non-zero, for failure-path tests."""
    return _write_executable(
        tmp_path / "fake_failing_executable",
        """\
        #!/usr/bin/env python3
        import sys
        print("simulated engine failure", file=sys.stderr)
        sys.exit(1)
        """,
    )


@pytest.fixture
def test_path_settings(tmp_path, fake_f5_tts_executable, fake_edge_tts_executable, fake_rvc_python_executable):
    from voice_studio.config import PathSettings

    base = PathSettings.from_root(root=tmp_path)
    return PathSettings(
        root=base.root,
        temp_dir=base.temp_dir,
        saved_voices_dir=base.saved_voices_dir,
        rvc_models_dir=base.rvc_models_dir,
        training_data_dir=base.training_data_dir,
        hf_cache_dir=base.hf_cache_dir,
        edge_tts_executable=fake_edge_tts_executable,
        f5_tts_executable=fake_f5_tts_executable,
        rvc_python_executable=fake_rvc_python_executable,
    )
