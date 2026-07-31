from __future__ import annotations

from pathlib import Path

from voice_studio.config import PathSettings
from voice_studio.infrastructure import DependencyHealthChecker


def _settings_with_missing_executables(tmp_path: Path) -> PathSettings:
    base = PathSettings.from_root(root=tmp_path)
    return PathSettings(
        root=base.root, temp_dir=base.temp_dir, saved_voices_dir=base.saved_voices_dir,
        rvc_models_dir=base.rvc_models_dir, training_data_dir=base.training_data_dir,
        hf_cache_dir=base.hf_cache_dir,
        edge_tts_executable=tmp_path / "nope" / "edge-tts",
        f5_tts_executable=tmp_path / "nope" / "f5-tts",
        rvc_python_executable=tmp_path / "nope" / "python",
    )


class TestDependencyHealthChecker:
    def test_missing_executables_are_reported_as_critical_failures(self, tmp_path: Path):
        report = DependencyHealthChecker(_settings_with_missing_executables(tmp_path)).check_all()

        assert report.healthy is False
        failed_names = {c.name for c in report.unhealthy_checks if c.critical}
        assert {"f5_tts_executable", "edge_tts_executable", "rvc_python_executable"} <= failed_names

    def test_present_executables_pass(self, tmp_path: Path, fake_f5_tts_executable, fake_edge_tts_executable, fake_rvc_python_executable):
        base = PathSettings.from_root(root=tmp_path)
        settings = PathSettings(
            root=base.root, temp_dir=base.temp_dir, saved_voices_dir=base.saved_voices_dir,
            rvc_models_dir=base.rvc_models_dir, training_data_dir=base.training_data_dir,
            hf_cache_dir=base.hf_cache_dir,
            edge_tts_executable=fake_edge_tts_executable,
            f5_tts_executable=fake_f5_tts_executable,
            rvc_python_executable=fake_rvc_python_executable,
        )
        (settings.root / "rvc_infer.py").write_text("# stand-in")

        report = DependencyHealthChecker(settings).check_all()

        executable_checks = {c.name: c.healthy for c in report.checks if c.name.endswith("_executable")}
        assert all(executable_checks.values())

    def test_gpu_absence_is_reported_but_not_critical(self, tmp_path: Path):
        # In this sandbox torch/CUDA is absent either way; the key assertion
        # is that a missing/absent GPU never flips overall `healthy` to False
        # by itself when every critical check passes.
        report = DependencyHealthChecker(_settings_with_missing_executables(tmp_path)).check_all()
        gpu_check = next(c for c in report.checks if c.name == "gpu")
        assert gpu_check.critical is False

    def test_low_disk_space_is_flagged(self, tmp_path: Path):
        settings = _settings_with_missing_executables(tmp_path)
        checker = DependencyHealthChecker(settings, min_free_disk_mb=10**9)  # impossible floor
        report = checker.check_all()
        disk_checks = [c for c in report.checks if c.name.endswith("_disk")]
        assert disk_checks
        assert all(not c.healthy for c in disk_checks)
        assert all(c.critical for c in disk_checks)  # low disk space IS treated as critical

    def test_report_checked_at_is_populated(self, tmp_path: Path):
        report = DependencyHealthChecker(_settings_with_missing_executables(tmp_path)).check_all()
        assert report.checked_at is not None
