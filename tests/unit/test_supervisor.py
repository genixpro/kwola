import multiprocessing
import os
import threading
import time

import pytest

from kwola.orchestration import (
    WorkerCommand,
    WorkerCrashed,
    WorkerResult,
    WorkerSupervisor,
    WorkerTimeout,
)


def successful_worker(command: WorkerCommand, log: object) -> WorkerResult:
    assert callable(log)
    log("started")
    return WorkerResult(
        command_id=command.command_id,
        status="completed",
        values={"answer": command.parameters["value"]},
    )


def slow_worker(command: WorkerCommand, log: object) -> WorkerResult:
    del log
    time.sleep(float(command.parameters["delay"]))
    return WorkerResult(command_id=command.command_id, status="completed")


def crashing_worker(command: WorkerCommand, log: object) -> WorkerResult:
    del command, log
    raise KeyboardInterrupt("worker crash")


def abrupt_worker(command: WorkerCommand, log: object) -> WorkerResult:
    del command, log
    os._exit(4)


def test_worker_result_and_logs_are_collected() -> None:
    context = multiprocessing.get_context("spawn")
    supervisor = WorkerSupervisor(successful_worker, context=context)

    result = supervisor.run(
        WorkerCommand(command_id="1", name="test", parameters={"value": 42}),
        timeout_seconds=5,
    )

    assert result.values == {"answer": 42}
    assert "started" in supervisor.logs()


def test_worker_timeout_forces_shutdown() -> None:
    with WorkerSupervisor(slow_worker, graceful_shutdown_seconds=0.01) as supervisor:
        with pytest.raises(WorkerTimeout, match="exceeded"):
            supervisor.run(
                WorkerCommand(command_id="2", name="slow", parameters={"delay": 2}),
                timeout_seconds=0.05,
            )


def test_worker_base_exception_is_a_typed_failed_result() -> None:
    supervisor = WorkerSupervisor(crashing_worker)

    result = supervisor.run(
        WorkerCommand(command_id="3", name="crash"),
        timeout_seconds=5,
    )

    assert result.status == "failed"
    assert result.error_type == "KeyboardInterrupt"


def test_abrupt_process_exit_is_reported() -> None:
    supervisor = WorkerSupervisor(abrupt_worker)

    with pytest.raises(WorkerCrashed, match="code 4"):
        supervisor.run(
            WorkerCommand(command_id="4", name="exit"),
            timeout_seconds=5,
        )


def test_supervisor_rejects_reuse_and_oversized_commands() -> None:
    supervisor = WorkerSupervisor(successful_worker)
    supervisor._process = object()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="active worker"):
        supervisor.run(WorkerCommand(command_id="5", name="reuse"), 1)
    supervisor._process = None
    command = WorkerCommand(
        command_id="6",
        name="large",
        parameters={"payload": "x" * (1024 * 1024)},
    )
    with pytest.raises(ValueError, match="1 MiB"):
        supervisor.run(command, 1)


class ForcedProcess:
    def __init__(self) -> None:
        self.checks = 0
        self.killed = False

    def is_alive(self) -> bool:
        self.checks += 1
        return self.checks <= 2

    def terminate(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def kill(self) -> None:
        self.killed = True


def test_supervisor_forces_termination_after_grace_period() -> None:
    supervisor = WorkerSupervisor(successful_worker)
    process = ForcedProcess()
    supervisor._process = process  # type: ignore[assignment]
    supervisor.cancel()
    assert process.killed


def test_supervisor_cancellation_returns_typed_result_and_stops_worker() -> None:
    cancelled = threading.Event()
    cancelled.set()
    with WorkerSupervisor(slow_worker, graceful_shutdown_seconds=0.01) as supervisor:
        result = supervisor.run(
            WorkerCommand(command_id="7", name="slow", parameters={"delay": 2}),
            timeout_seconds=5,
            cancel_event=cancelled,
        )

    assert result.status == "cancelled"
    assert result.error_type == "WorkerCancelled"
