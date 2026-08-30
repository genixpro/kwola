#!/usr/bin/env python3
"""Run the fresh-schema Kwola 1.1 rig release gate and capture evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from kwola.config import load_config
from kwola.storage import LmdbRunStore, load_manifest, verify_checkpoint


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    name: str
    command: tuple[str, ...]
    return_code: int
    duration_seconds: float
    log: str
    log_sha256: str


class AcceptanceFailure(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        _require_linux()
        evidence_dir = arguments.evidence_dir.resolve()
        _prepare_evidence_dir(evidence_dir)
        runner = AcceptanceRunner(
            evidence_dir=evidence_dir,
            kros1_url=arguments.kros1_url,
            kros3_url=arguments.kros3_url,
        )
        runner.run()
    except (AcceptanceFailure, OSError, ValueError) as error:
        print(f"rig acceptance failed: {error}", file=sys.stderr)
        return 1
    return 0


class AcceptanceRunner:
    def __init__(
        self,
        *,
        evidence_dir: Path,
        kros1_url: str,
        kros3_url: str,
    ) -> None:
        self.evidence_dir = evidence_dir
        self.kros1_url = kros1_url
        self.kros3_url = kros3_url
        self.logs = evidence_dir / "logs"
        self.logs.mkdir()
        self.commands: list[CommandEvidence] = []
        self.artifact_hashes: dict[str, str] = {}
        self.metrics: dict[str, Any] = {}
        self.kwola = _required_command("kwola")
        self.pytest = _required_command("pytest")
        self.ruff = _required_command("ruff")
        self.mypy = _required_command("mypy")

    def run(self) -> None:
        fresh_run = self.evidence_dir / "fresh-rig-run"
        self._run(
            "ruff-format",
            (self.ruff, "format", "--check", "kwola", "tests", __file__),
        )
        self._run("ruff-check", (self.ruff, "check", "kwola", "tests", __file__))
        self._run("mypy", (self.mypy, "--strict", "kwola", __file__))
        self._run("pytest", (self.pytest, "-q"))
        self._run("doctor", (self.kwola, "doctor", "--require-gpus", "2"))
        self._run(
            "browser-contract",
            (self.pytest, "-q", "tests/rig/test_browser_action_contract.py", "--no-cov"),
            environment={"KWOLA_RIG_ACCEPTANCE": "1", "KWOLA_KROS1_URL": self.kros1_url},
        )
        self._run(
            "initialize-fresh-rig",
            (
                self.kwola,
                "init",
                self.kros3_url,
                "--profile",
                "rig",
                "--run-dir",
                str(fresh_run),
                "--seed",
                "1100",
            ),
        )
        for browser in ("chromium", "firefox"):
            self._run(
                f"fresh-{browser}",
                (
                    self.kwola,
                    "test-step",
                    str(fresh_run),
                    "--browser",
                    browser,
                    "--random",
                ),
            )
        _verify_instrumented_run(fresh_run)
        self._run(
            "fresh-single-gpu",
            (self.kwola, "train-step", str(fresh_run), "--gpu", "0"),
        )
        self.artifact_hashes["fresh_single_gpu_checkpoint"] = _checkpoint_hash(fresh_run)
        self._run_concurrent_training(fresh_run)
        self.artifact_hashes["fresh_distributed_checkpoint"] = _checkpoint_hash(fresh_run)
        benchmark = self._run("benchmark", (self.kwola, "benchmark", str(fresh_run)))
        self.metrics = _verify_benchmark(self.evidence_dir / benchmark.log)
        self._run_dependency_gate()
        process_scan = self._run(
            "final-process-scan",
            ("ps", "-ax", "-o", "pid=,command="),
        )
        _verify_clean_process_scan(self.evidence_dir / process_scan.log)
        self._write_manifest()

    def _run_concurrent_training(self, fresh_run: Path) -> None:
        commands = {
            "concurrent-ddp": (self.kwola, "train-step", str(fresh_run)),
            "concurrent-browser": (
                self.kwola,
                "test-step",
                str(fresh_run),
                "--browser",
                "firefox",
                "--random",
            ),
        }
        started = time.monotonic()
        processes: dict[str, tuple[subprocess.Popen[str], Path, Any]] = {}
        try:
            for name, command in commands.items():
                log_path = self.logs / f"{name}.log"
                stream = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                processes[name] = (process, log_path, stream)
            failures = []
            for name, (process, log_path, stream) in processes.items():
                return_code = process.wait()
                stream.close()
                evidence = _command_evidence(
                    name, commands[name], return_code, time.monotonic() - started, log_path
                )
                self.commands.append(evidence)
                if return_code:
                    failures.append(name)
            if failures:
                raise AcceptanceFailure(f"concurrent commands failed: {', '.join(failures)}")
        finally:
            for process, _log_path, stream in processes.values():
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if not stream.closed:
                    stream.close()

    def _run_dependency_gate(self) -> None:
        python_json = self.evidence_dir / "pip-audit.json"
        npm_json = self.evidence_dir / "npm-audit.json"
        self._run(
            "pip-audit",
            (_required_command("pip-audit"), "--format", "json", "--output", str(python_json)),
            accepted_return_codes=(0, 1),
        )
        npm = self._run(
            "npm-audit",
            (_required_command("npm"), "audit", "--json"),
            accepted_return_codes=(0, 1),
        )
        shutil.copyfile(self.evidence_dir / npm.log, npm_json)
        self._run(
            "dependency-release-gate",
            (
                sys.executable,
                "scripts/audit_dependencies.py",
                "--python-json",
                str(python_json),
                "--npm-json",
                str(npm_json),
                "--exceptions",
                "security/advisory-exceptions.json",
                "--require-no-exceptions",
            ),
        )

    def _run(
        self,
        name: str,
        command: tuple[str, ...],
        *,
        environment: dict[str, str] | None = None,
        accepted_return_codes: tuple[int, ...] = (0,),
    ) -> CommandEvidence:
        log_path = self.logs / f"{name}.log"
        merged_environment = os.environ.copy()
        merged_environment.update(environment or {})
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                env=merged_environment,
                check=False,
            )
        evidence = _command_evidence(
            name, command, result.returncode, time.monotonic() - started, log_path
        )
        self.commands.append(evidence)
        if result.returncode not in accepted_return_codes:
            raise AcceptanceFailure(f"{name} exited {result.returncode}; inspect {evidence.log}")
        return evidence

    def _write_manifest(self) -> None:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=True
        ).stdout.strip()
        payload = {
            "schema_version": 1,
            "created_at": time.time(),
            "git_revision": revision,
            "python": sys.version,
            "platform": sys.platform,
            "environment_versions": _environment_versions(),
            "kros1_url": self.kros1_url,
            "kros3_url": self.kros3_url,
            "metrics": self.metrics,
            "artifact_hashes": self.artifact_hashes,
            "commands": [asdict(item) for item in self.commands],
        }
        target = self.evidence_dir / "manifest.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--kros1-url", required=True)
    parser.add_argument("--kros3-url", required=True)
    return parser


def _prepare_evidence_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise AcceptanceFailure(f"evidence directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _require_linux() -> None:
    if sys.platform != "linux":
        raise AcceptanceFailure("rig acceptance is authoritative only on Linux")


def _required_command(name: str) -> str:
    command = shutil.which(name)
    if command is None:
        raise AcceptanceFailure(f"required command is unavailable: {name}")
    return command


def _command_evidence(
    name: str,
    command: tuple[str, ...],
    return_code: int,
    duration: float,
    log_path: Path,
) -> CommandEvidence:
    return CommandEvidence(
        name=name,
        command=command,
        return_code=return_code,
        duration_seconds=duration,
        log=str(log_path.relative_to(log_path.parents[1])),
        log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
    )


def _verify_instrumented_run(run_dir: Path) -> None:
    config = load_config(run_dir)
    with LmdbRunStore(
        run_dir / config.storage.database_directory,
        map_size=config.storage.database_map_size_bytes,
        readonly=True,
    ) as store:
        steps = [record for _key, record in store.scan("testing_steps")]
        traces = [record for _key, record in store.scan("traces")]
        resources = [record for _key, record in store.scan("resources")]
    browsers = {str(record.get("browser")) for record in steps}
    if browsers != {"chromium", "firefox"}:
        raise AcceptanceFailure(f"fresh run did not record both browsers: {sorted(browsers)}")
    if not traces:
        raise AcceptanceFailure("fresh run did not record browser traces")
    if not any(record.get("rewrite_kind") == "javascript" for record in resources):
        raise AcceptanceFailure("fresh run did not record rewritten JavaScript")


def _checkpoint_hash(run_dir: Path) -> str:
    manifest = load_manifest(run_dir)
    if manifest.checkpoint is None:
        raise AcceptanceFailure(f"run did not publish a checkpoint: {run_dir}")
    return hashlib.sha256(verify_checkpoint(run_dir, manifest.checkpoint).read_bytes()).hexdigest()


def _verify_benchmark(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AcceptanceFailure("benchmark output is not a JSON object")
    if not payload.get("passed"):
        raise AcceptanceFailure("benchmark did not pass")
    if float(payload["samples_per_second"]) < 145:
        raise AcceptanceFailure("benchmark throughput is below 145 samples/second")
    if float(payload["median_optimizer_seconds"]) > 1.35:
        raise AcceptanceFailure("benchmark optimizer median exceeds 1.35 seconds")
    if float(payload["peak_vram_gib"]) > 5.0:
        raise AcceptanceFailure("benchmark peak VRAM exceeds 5 GiB")
    return cast(dict[str, Any], payload)


def _environment_versions() -> dict[str, str]:
    versions = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    for package in ("kwola", "playwright", "torch", "mitmproxy", "numpy"):
        versions[package] = importlib.metadata.version(package)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        result = subprocess.run(
            (nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"),
            capture_output=True,
            text=True,
            check=False,
        )
        versions["nvidia_gpus"] = result.stdout.strip()
    return versions


def _verify_clean_process_scan(path: Path) -> None:
    patterns = (
        "kwola-testing-",
        "kwola-training-",
        "playwright/driver",
        "instrument_javascript.cjs",
        "mitmdump",
        "chromium",
        "chrome --",
        "firefox",
    )
    leaked = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if any(pattern in line for pattern in patterns)
    ]
    if leaked:
        raise AcceptanceFailure(f"runtime processes remain after acceptance: {leaked[:5]}")


if __name__ == "__main__":
    raise SystemExit(main())
