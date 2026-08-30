"""Actionable local installation diagnostics."""

import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from kwola.storage import LmdbRunStore


class DiagnosticCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    passed: bool
    detail: str


class DoctorReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    checks: tuple[DiagnosticCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def run_doctor(require_gpus: int = 0) -> DoctorReport:
    checks = [
        _check_python(),
        _check_platform(),
        _check_ffmpeg(),
        _check_lmdb(),
        *_check_browsers(),
        _check_torch(require_gpus),
    ]
    return DoctorReport(checks=tuple(checks))


def _check_python() -> DiagnosticCheck:
    current = sys.version_info[:2]
    return DiagnosticCheck(
        name="python",
        passed=current == (3, 12),
        detail=f"{platform.python_version()} (required: 3.12.x)",
    )


def _check_platform() -> DiagnosticCheck:
    production = platform.system() == "Linux"
    detail = platform.platform()
    if not production:
        detail += " (development only; Linux is the acceptance authority)"
    return DiagnosticCheck(name="platform", passed=True, detail=detail)


def _check_ffmpeg() -> DiagnosticCheck:
    executable = shutil.which("ffmpeg")
    return DiagnosticCheck(
        name="ffmpeg",
        passed=executable is not None,
        detail=executable or "ffmpeg was not found on PATH",
    )


def _check_lmdb() -> DiagnosticCheck:
    try:
        with tempfile.TemporaryDirectory(prefix="kwola-doctor-") as directory:
            with LmdbRunStore(Path(directory) / "test.lmdb", map_size=1024**2) as store:
                store.put("doctor", "probe", {"ok": True})
                passed = store.get("doctor", "probe") == {"ok": True}
        return DiagnosticCheck(name="lmdb", passed=passed, detail="atomic write/read probe")
    except Exception as error:
        return DiagnosticCheck(name="lmdb", passed=False, detail=f"{type(error).__name__}: {error}")


def _check_browsers() -> tuple[DiagnosticCheck, ...]:
    executable = shutil.which("playwright")
    if executable is None:
        failed = DiagnosticCheck(
            name="playwright", passed=False, detail="playwright was not found on PATH"
        )
        return (failed,)
    result = subprocess.run(
        [executable, "install", "--list"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    output = result.stdout + result.stderr
    return tuple(
        DiagnosticCheck(
            name=f"browser:{name}",
            passed=f"/{name}-" in output,
            detail="installed" if f"/{name}-" in output else "Playwright browser is missing",
        )
        for name in ("chromium", "firefox")
    )


def _check_torch(require_gpus: int) -> DiagnosticCheck:
    count = torch.cuda.device_count()
    nccl = torch.distributed.is_nccl_available()
    passed = count >= require_gpus and (require_gpus < 2 or nccl)
    detail = f"torch={torch.__version__}, cuda_devices={count}, nccl={nccl}"
    if count < require_gpus:
        detail += f"; requires at least {require_gpus} GPU(s)"
    elif require_gpus >= 2 and not nccl:
        detail += "; two-rank acceptance requires NCCL"
    return DiagnosticCheck(name="torch", passed=passed, detail=detail)
