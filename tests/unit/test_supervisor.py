import multiprocessing
import os
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
