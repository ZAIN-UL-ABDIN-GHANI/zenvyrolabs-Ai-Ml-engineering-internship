from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voice_studio.config import PathSettings
from voice_studio.infrastructure.logging_setup import get_logger

_logger = get_logger("monitoring")
_MIN_FREE_DISK_MB = 500


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    name: str
    healthy: bool
    detail: str
    critical: bool = True  # non-critical checks are reported but don't fail the overall report


@dataclass(frozen=True, slots=True)
class HealthReport:
    checks: tuple[HealthCheckResult, ...]
    checked_at: datetime

    @property
    def healthy(self) -> bool:
        return all(check.healthy for check in self.checks if check.critical)

    @property
    def unhealthy_checks(self) -> tuple[HealthCheckResult, ...]:
        return tuple(c for c in self.checks if not c.healthy)


class DependencyHealthChecker:
    """Infrastructure Layer component realizing SDS §17.2's Dependency Health
    (and a light Resource Utilization) category: verifies required engine
    executables, runtimes, model-store accessibility, and disk headroom
    *before* a user's click fails deep inside a tab — directly closing the
    "first the user hears about it is a cryptic per-click failure" gap the
    Stage 1 Analysis identified across F-01/F-02/F-03/F-23.
    """

    def __init__(self, paths: PathSettings, min_free_disk_mb: int = _MIN_FREE_DISK_MB) -> None:
        self._paths = paths
        self._min_free_disk_mb = min_free_disk_mb

    def check_all(self) -> HealthReport:
        checks = (
            self._check_exists("f5_tts_executable", self._paths.f5_tts_executable),
            self._check_exists("edge_tts_executable", self._paths.edge_tts_executable),
            self._check_exists("rvc_python_executable", self._paths.rvc_python_executable),
            self._check_exists("rvc_infer_script", self._paths.root / "rvc_infer.py"),
            self._check_directory("saved_voices_dir", self._paths.saved_voices_dir),
            self._check_directory("rvc_models_dir", self._paths.rvc_models_dir),
            self._check_directory("training_data_dir", self._paths.training_data_dir),
            self._check_disk_space("saved_voices_disk", self._paths.saved_voices_dir),
            self._check_disk_space("training_data_disk", self._paths.training_data_dir),
            self._check_gpu_available(),
        )
        report = HealthReport(checks=checks, checked_at=datetime.now(timezone.utc))
        self._log_report(report)
        return report

    def _check_exists(self, name: str, path: Path) -> HealthCheckResult:
        if path.exists():
            return HealthCheckResult(name, True, f"Found at {path}")
        return HealthCheckResult(name, False, f"Missing: {path}")

    def _check_directory(self, name: str, path: Path) -> HealthCheckResult:
        path.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            return HealthCheckResult(name, True, f"Accessible at {path}")
        return HealthCheckResult(name, False, f"Not accessible: {path}")

    def _check_disk_space(self, name: str, path: Path) -> HealthCheckResult:
        try:
            path.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(path)
        except OSError as error:
            return HealthCheckResult(name, False, f"Could not read disk usage: {error}")
        free_mb = usage.free / (1024 * 1024)
        if free_mb < self._min_free_disk_mb:
            return HealthCheckResult(
                name, False, f"Only {free_mb:.0f} MB free (below {self._min_free_disk_mb} MB floor)"
            )
        return HealthCheckResult(name, True, f"{free_mb:.0f} MB free")

    def _check_gpu_available(self) -> HealthCheckResult:
        try:
            import torch
        except ImportError:
            return HealthCheckResult("gpu", False, "torch not installed — cannot check GPU", critical=False)
        if torch.cuda.is_available():
            return HealthCheckResult("gpu", True, f"CUDA available: {torch.cuda.get_device_name(0)}", critical=False)
        return HealthCheckResult("gpu", False, "No CUDA-capable GPU detected (CPU fallback)", critical=False)

    def _log_report(self, report: HealthReport) -> None:
        if report.healthy:
            _logger.info("Dependency health check: all critical checks passed (%d total)", len(report.checks))
        for check in report.unhealthy_checks:
            level = _logger.warning if check.critical else _logger.info
            level("Health check %s: %s (critical=%s)", check.name, check.detail, check.critical)
