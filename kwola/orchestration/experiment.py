"""Continuously pipelined browser producers and one supervised trainer."""

import logging
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from kwola.config import load_config
from kwola.storage import LmdbRunStore

from .messages import ArtifactReference, WorkerCommand, WorkerResult
from .results import RunnerResult
from .supervisor import WorkerHandler, WorkerSupervisor
from .telemetry import TelemetryWriter
from .testing import TestingRunner
from .training import TrainingRunner

Log = Callable[[str], None]


class ExperimentRunner:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._config = load_config(run_dir)
        self._sequence = 0

    def run(self) -> int:
        workers = self._config.orchestration.browser_workers + 1
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kwola-pipeline")
        try:
            with TelemetryWriter(
                self._run_dir, self._config.orchestration.telemetry_interval_seconds
            ) as telemetry:
                return self._pipeline(executor, telemetry)
        except KeyboardInterrupt:
            return 130
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _pipeline(self, executor: ThreadPoolExecutor, telemetry: TelemetryWriter) -> int:
        active: dict[Future[WorkerResult], tuple[str, int, WorkerCommand]] = {}
        for slot in range(self._config.orchestration.browser_workers):
            self._submit(executor, telemetry, active, "testing", slot)
        training_active = False
        if self._trace_count() >= self._config.orchestration.minimum_traces_before_training:
            self._submit(executor, telemetry, active, "training", 0)
            training_active = True
        while True:
            completed, _pending = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                worker, slot, command = active.pop(future)
                result = future.result()
                self._record_completion(telemetry, worker, slot, command, result)
                if worker == "testing":
                    self._submit(executor, telemetry, active, "testing", slot)
                else:
                    self._require_success(result)
                    training_active = False
            if not training_active and self._training_ready():
                self._submit(executor, telemetry, active, "training", 0)
                training_active = True

    def _submit(
        self,
        executor: ThreadPoolExecutor,
        telemetry: TelemetryWriter,
        active: dict[Future[WorkerResult], tuple[str, int, WorkerCommand]],
        worker: str,
        slot: int,
    ) -> None:
        command = self._command(worker, slot)
        handler = _testing_worker if worker == "testing" else _training_worker
        future = executor.submit(
            _supervised,
            handler,
            command,
            self._config.orchestration.worker_timeout_seconds,
        )
        active[future] = (worker, slot, command)
        telemetry.record(
            "worker_submitted", worker=worker, slot=slot, command_id=command.command_id
        )

    def _command(self, name: str, slot: int) -> WorkerCommand:
        sequence = self._sequence
        self._sequence += 1
        parameters: dict[str, object] = {"run_dir": str(self._run_dir), "slot": slot}
        if name == "testing":
            browsers = self._config.browser.enabled
            parameters["browser"] = browsers[slot % len(browsers)].value
        return WorkerCommand(
            command_id=f"{name}-{sequence:08d}-slot-{slot:02d}",
            name=name,
            parameters=parameters,
        )

    def _training_ready(self) -> bool:
        return self._trace_count() >= self._config.orchestration.minimum_traces_before_training

    def _trace_count(self) -> int:
        database = self._run_dir / self._config.storage.database_directory
        with LmdbRunStore(
            database,
            map_size=self._config.storage.database_map_size_bytes,
            readonly=True,
        ) as store:
            return sum(1 for _ in store.scan("traces"))

    @staticmethod
    def _record_completion(
        telemetry: TelemetryWriter,
        worker: str,
        slot: int,
        command: WorkerCommand,
        result: WorkerResult,
    ) -> None:
        telemetry.record(
            "worker_completed",
            worker=worker,
            slot=slot,
            command_id=command.command_id,
            status=result.status,
            error_type=result.error_type,
            error_message=result.error_message,
            duration_seconds=float(result.values.get("duration_seconds", 0)),
            metrics=result.values.get("metrics", {}),
        )

    @staticmethod
    def _require_success(result: WorkerResult) -> None:
        if result.status != "completed":
            raise RuntimeError(f"experiment worker failed: {result.error_message or result.status}")

    def _adapt_training(self, results: tuple[WorkerResult, ...]) -> None:
        durations = {
            result.command_id.split("-", maxsplit=1)[0]: float(
                result.values.get("duration_seconds", 0)
            )
            for result in results
        }
        if "training" not in durations or "testing" not in durations:
            return
        config = load_config(self._run_dir)
        database = self._run_dir / config.storage.database_directory
        with LmdbRunStore(
            database,
            map_size=config.storage.database_map_size_bytes,
            compression_level=config.storage.codec_compression_level,
        ) as store:
            state = store.get("run", "state") or {}
            current = int(
                state.get(
                    "scheduled_training_iterations",
                    config.training.batches_per_iteration,
                )
            )
            delta = config.training.batch_iteration_adjustment
            candidate = (
                current + delta if durations["training"] < durations["testing"] else current - delta
            )
            state["scheduled_training_iterations"] = max(
                config.training.min_batches_per_iteration,
                min(config.training.max_batches_per_iteration, candidate),
            )
            store.put("run", "state", state)


def _supervised(
    handler: WorkerHandler, command: WorkerCommand, timeout_seconds: float
) -> WorkerResult:
    with WorkerSupervisor(handler) as supervisor:
        try:
            return supervisor.run(command, timeout_seconds=timeout_seconds)
        finally:
            logger = logging.getLogger(__name__)
            for message in supervisor.logs():
                logger.warning("worker %s: %s", command.command_id, message)


def _testing_worker(command: WorkerCommand, log: Log) -> WorkerResult:
    import cv2
    import torch

    from kwola.domain.actions import BrowserKind

    run_dir = Path(str(command.parameters["run_dir"]))
    config = load_config(run_dir)
    torch.set_num_threads(config.orchestration.browser_cpu_threads)
    cv2.setNumThreads(config.orchestration.browser_cpu_threads)
    log(f"starting browser testing for {run_dir}")
    result = TestingRunner(run_dir).run(
        browser=BrowserKind(str(command.parameters["browser"])),
        environment_index=int(command.parameters["slot"]),
    )
    for warning in result.warnings:
        log(warning.model_dump_json())
    return _worker_result(command, result)


def _training_worker(command: WorkerCommand, log: Log) -> WorkerResult:
    run_dir = Path(str(command.parameters["run_dir"]))
    log(f"starting model training for {run_dir}")
    result = TrainingRunner(run_dir).run()
    for warning in result.warnings:
        log(warning.model_dump_json())
    return _worker_result(command, result)


def _worker_result(
    command: WorkerCommand,
    result: RunnerResult,
) -> WorkerResult:
    references = tuple(ArtifactReference(kind="blob", location=path) for path in result.artifacts)
    return WorkerResult(
        command_id=command.command_id,
        status="completed",
        values={
            "metrics": result.metrics,
            "duration_seconds": result.duration_seconds,
            "warnings": [warning.model_dump(mode="json") for warning in result.warnings],
        },
        artifacts=references,
    )
