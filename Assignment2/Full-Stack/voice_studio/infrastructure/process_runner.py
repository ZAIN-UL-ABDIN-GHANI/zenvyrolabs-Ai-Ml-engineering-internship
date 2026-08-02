from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class ProcessExecutionError(RuntimeError):
    """Raised when an external process exits with a non-zero status."""

    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"Command {command!r} exited with code {returncode}: {stderr.strip()[:500]}"
        )


@dataclass(frozen=True, slots=True)
class ProcessResult:
    stdout: str
    stderr: str
    returncode: int


class ProcessRunner:
    """Runs external executables safely."""

    def __init__(self, timeout_seconds: float | None = 9600) -> None:
        # Default timeout: 1 hour
        self._timeout_seconds = timeout_seconds

    def run(
        self,
        executable: Path,
        arguments: list[str],
        cwd: Path | None = None,
    ) -> ProcessResult:

        if not executable.exists():
            raise FileNotFoundError(
                f"Required executable not found: {executable}"
            )

        command = [str(executable), *arguments]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=str(cwd) if cwd else None,
                timeout=self._timeout_seconds,
                shell=False,
            )

        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Process exceeded timeout ({self._timeout_seconds} seconds).\n"
                f"Command: {' '.join(command)}"
            ) from exc

        if completed.returncode != 0:
            raise ProcessExecutionError(
                command,
                completed.returncode,
                completed.stderr,
            )

        return ProcessResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )