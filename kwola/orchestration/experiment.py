"""Continuously pipelined browser producers and one supervised trainer."""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from kwola.config import load_config
from kwola.storage import LmdbRunStore

from .messages import ArtifactReference, WorkerCommand, WorkerResult
from .results import RunnerResult
from .supervisor import WorkerHandler, WorkerSupervisor
from .telemetry import TelemetryWriter
from .testing import TestingRunner
from .training import TrainingRunner

Log = Callable[[str], None]
ActiveWorkers = dict[Future[WorkerResult], tuple[str, int, WorkerCommand]]


@dataclass(slots=True)
class _PipelineState:
    failures: dict[int, int]
    training_active: bool = False
    browser_durations: list[float] = field(default_factory=list)


class _TerminalWorkerFailure(RuntimeError):
    pass


class ExperimentRunner:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._config = load_config(run_dir)
        self._sequence = 0
        self._cancel = threading.Event()

    def run(self) -> int:
        self._cancel.clear()
        workers = self._config.orchestration.browser_workers + 1
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kwola-pipeline")
        with TelemetryWriter(
            self._run_dir, self._config.orchestration.telemetry_interval_seconds
        ) as telemetry:
            try:
                return self._pipeline(executor, telemetry)
            except KeyboardInterrupt as error:
                telemetry.record(
                    "pipeline_failed",
                    worker="orchestration",
                    error_type=type(error).__name__,
                    error_message="pipeline interrupted",
                )
                return 130
            except _TerminalWorkerFailure:
                raise
            except Exception as error:
                telemetry.record(
                    "pipeline_failed",
                    worker="orchestration",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                raise
            finally:
                self._cancel.set()
                executor.shutdown(wait=True, cancel_futures=True)

    def _pipeline(self, executor: ThreadPoolExecutor, telemetry: TelemetryWriter) -> int:
        active: ActiveWorkers = {}
        state = _PipelineState(
            failures={slot: 0 for slot in range(self._config.orchestration.browser_workers)}
        )
        startup_stagger = min(0.5, self._config.browser.action_settle_seconds)
        for slot in range(self._config.orchestration.browser_workers):
            self._submit(
                executor,
                telemetry,
                active,
                "testing",
                slot,
                retry_delay=slot * startup_stagger,
            )
        self._start_training_if_ready(executor, telemetry, active, state)
        while True:
            completed, _pending = wait(active, return_when=FIRST_COMPLETED)
            batch = []
            for future in completed:
                worker, slot, command = active.pop(future)
                batch.append((worker, slot, command, future.result()))
            for worker, slot, command, result in sorted(batch, key=lambda item: item[0]):
                self._record_completion(telemetry, worker, slot, command, result)
                if worker == "testing":
                    self._handle_testing(executor, telemetry, active, state, slot, result)
                else:
                    self._handle_training(telemetry, state, result)
            self._start_training_if_ready(executor, telemetry, active, state)

    def _start_training_if_ready(
        self,
        executor: ThreadPoolExecutor,
        telemetry: TelemetryWriter,
        active: ActiveWorkers,
        state: _PipelineState,
    ) -> None:
        if state.training_active or not self._training_ready():
            return
        state.browser_durations.clear()
        self._submit(executor, telemetry, active, "training", 0)
        state.training_active = True

    def _handle_testing(
        self,
        executor: ThreadPoolExecutor,
        telemetry: TelemetryWriter,
        active: ActiveWorkers,
        state: _PipelineState,
        slot: int,
        result: WorkerResult,
    ) -> None:
        if result.status == "completed":
            previous = state.failures[slot]
            state.failures[slot] = 0
            if previous:
                telemetry.record("worker_recovered", worker="testing", slot=slot, failures=previous)
            if state.training_active:
                state.browser_durations.append(float(result.values.get("duration_seconds", 0)))
            self._submit(executor, telemetry, active, "testing", slot)
            return
        failures = state.failures[slot] + 1
        state.failures[slot] = failures
        if failures >= self._config.orchestration.browser_max_consecutive_failures:
            telemetry.record(
                "pipeline_failed",
                worker="testing",
                slot=slot,
                consecutive_failures=failures,
                error_type=result.error_type,
                error_message=result.error_message,
            )
            self._cancel.set()
            self._require_success(result)
        delay = self._retry_delay(failures)
        telemetry.record(
            "worker_retry_scheduled",
            worker="testing",
            slot=slot,
            consecutive_failures=failures,
            delay_seconds=delay,
            error_type=result.error_type,
            error_message=result.error_message,
        )
        self._submit(executor, telemetry, active, "testing", slot, retry_delay=delay)

    def _handle_training(
        self, telemetry: TelemetryWriter, state: _PipelineState, result: WorkerResult
    ) -> None:
        if result.status != "completed":
            telemetry.record(
                "pipeline_failed",
                worker="training",
                slot=0,
                error_type=result.error_type,
                error_message=result.error_message,
            )
            self._cancel.set()
            self._require_success(result)
        self._adapt_training(
            float(result.values.get("duration_seconds", 0)),
            tuple(state.browser_durations),
            telemetry,
        )
        state.training_active = False
        state.browser_durations.clear()

    def _retry_delay(self, failures: int) -> float:
        base = self._config.orchestration.browser_retry_base_seconds
        maximum = self._config.orchestration.browser_retry_max_seconds
        return float(min(base * 2 ** (failures - 1), maximum))

    def _submit(
        self,
        executor: ThreadPoolExecutor,
        telemetry: TelemetryWriter,
        active: ActiveWorkers,
        worker: str,
        slot: int,
        retry_delay: float = 0.0,
    ) -> None:
        command = self._command(worker, slot)
        handler = _testing_worker if worker == "testing" else _training_worker
        future = executor.submit(
            _supervised,
            handler,
            command,
            self._config.orchestration.worker_timeout_seconds,
            self._cancel,
            retry_delay,
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
            raise _TerminalWorkerFailure(
                f"experiment worker failed: {result.error_message or result.status}"
            )

    def _adapt_training(
        self,
        training_duration: float,
        browser_durations: tuple[float, ...],
        telemetry: TelemetryWriter,
    ) -> None:
        if not browser_durations:
            return
        config = load_config(self._run_dir)
        database = self._run_dir / config.storage.database_directory
        browser_median = float(median(browser_durations))
        adjustment: tuple[int, int] = (0, 0)
        with LmdbRunStore(
            database,
            map_size=config.storage.database_map_size_bytes,
            compression_level=config.storage.codec_compression_level,
        ) as store:

            def adapt(current: dict[str, Any] | None) -> dict[str, Any]:
                nonlocal adjustment
                state = dict(current or {})
                previous = int(
                    state.get(
                        "scheduled_training_iterations", config.training.batches_per_iteration
                    )
                )
                delta = config.training.batch_iteration_adjustment
                candidate = (
                    previous + delta if training_duration < browser_median else previous - delta
                )
                updated = max(
                    config.training.min_batches_per_iteration,
                    min(config.training.max_batches_per_iteration, candidate),
                )
                state["scheduled_training_iterations"] = updated
                adjustment = (previous, updated)
                return state

            store.update("run", "state", adapt)
        telemetry.record(
            "training_schedule_adjusted",
            previous_iterations=adjustment[0],
            scheduled_iterations=adjustment[1],
            training_duration_seconds=training_duration,
            browser_median_duration_seconds=browser_median,
            browser_sample_count=len(browser_durations),
        )


def _supervised(
    handler: WorkerHandler,
    command: WorkerCommand,
    timeout_seconds: float,
    cancel_event: threading.Event,
    retry_delay: float,
) -> WorkerResult:
    if retry_delay and cancel_event.wait(retry_delay):
        return WorkerResult(
            command_id=command.command_id,
            status="cancelled",
            error_type="WorkerCancelled",
            error_message="retry cancelled by experiment shutdown",
        )
    with WorkerSupervisor(handler) as supervisor:
        try:
            return supervisor.run(
                command,
                timeout_seconds=timeout_seconds,
                cancel_event=cancel_event,
            )
        except Exception as error:
            return WorkerResult(
                command_id=command.command_id,
                status="failed",
                error_type=type(error).__name__,
                error_message=str(error),
            )
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
