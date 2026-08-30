"""Context-managed worker lifetime, timeout, cancellation, and logs."""

import multiprocessing
import queue
import time
import traceback
from collections.abc import Callable
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import Any, Self

from .messages import WorkerCommand, WorkerResult

WorkerHandler = Callable[[WorkerCommand, Callable[[str], None]], WorkerResult]


class WorkerTimeout(TimeoutError):
    pass


class WorkerCrashed(RuntimeError):
    pass


def _worker_entry(
    handler: WorkerHandler,
    command_json: str,
    result_queue: Queue[Any],
    log_queue: Queue[Any],
) -> None:
    command = WorkerCommand.model_validate_json(command_json)

    def log(message: str) -> None:
        log_queue.put(message)

    try:
        result = handler(command, log)
    except BaseException as error:
        log(traceback.format_exc())
        result = WorkerResult(
            command_id=command.command_id,
            status="failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )
    result_queue.put(result.model_dump_json())


class WorkerSupervisor:
    def __init__(
        self,
        handler: WorkerHandler,
        *,
        context: BaseContext | None = None,
        graceful_shutdown_seconds: float = 3.0,
    ) -> None:
        self._handler = handler
        self._context = context or multiprocessing.get_context("spawn")
        self._graceful_shutdown_seconds = graceful_shutdown_seconds
        self._process: BaseProcess | None = None
        self._results: Queue[Any] | None = None
        self._logs: Queue[Any] | None = None
        self._collected_logs: list[str] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.cancel()

    def run(self, command: WorkerCommand, timeout_seconds: float) -> WorkerResult:
        if self._process is not None:
            raise RuntimeError("supervisor already has an active worker")
        if len(command.model_dump_json()) > 1024 * 1024:
            raise ValueError("control messages cannot exceed 1 MiB; pass an artifact reference")
        self._results = self._context.Queue()
        self._logs = self._context.Queue()
        self._process = self._context.Process(
            target=_worker_entry,
            args=(self._handler, command.model_dump_json(), self._results, self._logs),
            name=f"kwola-{command.name}-{command.command_id}",
        )
        self._process.start()
        try:
            return self._wait(command, timeout_seconds)
        finally:
            self._drain_logs()
            self._cleanup_queues()
            self._process = None

    def cancel(self) -> None:
        process = self._process
        if process is None or not process.is_alive():
            return
        process.terminate()
        process.join(self._graceful_shutdown_seconds)
        if process.is_alive():
            process.kill()
            process.join()

    def logs(self) -> tuple[str, ...]:
        self._drain_logs()
        return tuple(self._collected_logs)

    def _drain_logs(self) -> None:
        if self._logs is None:
            return
        while True:
            try:
                self._collected_logs.append(str(self._logs.get_nowait()))
            except queue.Empty:
                return

    def _wait(self, command: WorkerCommand, timeout_seconds: float) -> WorkerResult:
        assert self._process is not None
        assert self._results is not None
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                return WorkerResult.model_validate_json(self._results.get(timeout=0.05))
            except queue.Empty:
                if not self._process.is_alive():
                    self._process.join()
                    raise WorkerCrashed(
                        f"worker {command.name!r} exited with code {self._process.exitcode}"
                    ) from None
        self.cancel()
        raise WorkerTimeout(f"worker {command.name!r} exceeded {timeout_seconds:.3f}s")

    def _cleanup_queues(self) -> None:
        for process_queue in (self._results, self._logs):
            if process_queue is not None:
                process_queue.close()
                process_queue.join_thread()
        self._results = None
        self._logs = None
