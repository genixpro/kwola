"""Supervised concurrent testing and adaptive training loop."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .messages import ArtifactReference, WorkerCommand, WorkerResult
from .supervisor import WorkerHandler, WorkerSupervisor
from .testing import TestingRunner
from .training import TrainingRunner

Log = Callable[[str], None]


class ExperimentRunner:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir

    def run(self) -> int:
        iteration = 0
        try:
            while True:
                commands = [self._command("testing", iteration)]
                if iteration > 0:
                    commands.append(self._command("training", iteration))
                results = self._run_concurrently(commands)
                failed = [result for result in results if result.status != "completed"]
                if failed:
                    detail = "; ".join(result.error_message or result.status for result in failed)
                    raise RuntimeError(f"experiment worker failed: {detail}")
                iteration += 1
        except KeyboardInterrupt:
            return 130

    def _command(self, name: str, iteration: int) -> WorkerCommand:
        return WorkerCommand(
            command_id=f"{name}-{iteration:08d}",
            name=name,
            parameters={"run_dir": str(self._run_dir)},
        )

    @staticmethod
    def _run_concurrently(commands: list[WorkerCommand]) -> tuple[WorkerResult, ...]:
        handlers: dict[str, WorkerHandler] = {
            "testing": _testing_worker,
            "training": _training_worker,
        }
        with ThreadPoolExecutor(max_workers=len(commands)) as executor:
            futures = [
                executor.submit(_supervised, handlers[command.name], command)
                for command in commands
            ]
            return tuple(future.result() for future in futures)


def _supervised(handler: WorkerHandler, command: WorkerCommand) -> WorkerResult:
    with WorkerSupervisor(handler) as supervisor:
        return supervisor.run(command, timeout_seconds=3600)


def _testing_worker(command: WorkerCommand, log: Log) -> WorkerResult:
    run_dir = Path(str(command.parameters["run_dir"]))
    log(f"starting browser testing for {run_dir}")
    result = TestingRunner(run_dir).run()
    return _worker_result(command, result.artifacts, result.metrics)


def _training_worker(command: WorkerCommand, log: Log) -> WorkerResult:
    run_dir = Path(str(command.parameters["run_dir"]))
    log(f"starting model training for {run_dir}")
    result = TrainingRunner(run_dir).run()
    return _worker_result(command, result.artifacts, result.metrics)


def _worker_result(
    command: WorkerCommand,
    artifacts: tuple[str, ...],
    metrics: dict[str, float],
) -> WorkerResult:
    references = tuple(ArtifactReference(kind="blob", location=path) for path in artifacts)
    return WorkerResult(
        command_id=command.command_id,
        status="completed",
        values={"metrics": metrics},
        artifacts=references,
    )
