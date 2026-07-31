from __future__ import annotations

from pathlib import Path

import pytest

from voice_studio.infrastructure import ProcessExecutionError, ProcessRunner


class TestProcessRunner:
    def test_missing_executable_fails_fast_with_a_clear_message(self, tmp_path: Path):
        runner = ProcessRunner()
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Required executable not found"):
            runner.run(missing, [])

    def test_non_zero_exit_raises_process_execution_error(self, fake_failing_executable: Path):
        runner = ProcessRunner()
        with pytest.raises(ProcessExecutionError) as excinfo:
            runner.run(fake_failing_executable, [])
        assert excinfo.value.returncode == 1
        assert "simulated engine failure" in excinfo.value.stderr

    def test_successful_run_returns_captured_output(self, fake_edge_tts_executable: Path, tmp_path: Path):
        out = tmp_path / "out.wav"
        runner = ProcessRunner()
        result = runner.run(fake_edge_tts_executable, ["--write-media", str(out)])
        assert result.returncode == 0
        assert "complete" in result.stdout.lower()
        assert out.exists()
