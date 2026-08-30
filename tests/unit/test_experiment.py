from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from kwola.config import load_config
from kwola.hooks import LifecycleEventName
from kwola.orchestration import experiment as experiment_module
from kwola.orchestration.experiment import ExperimentRunner, _PipelineState, _worker_result
from kwola.orchestration.initialize import initialize_run
from kwola.orchestration.messages import WorkerCommand, WorkerResult
from kwola.orchestration.results import RunnerResult, RunnerWarning
from kwola.storage import LmdbRunStore


def test_pipeline_adapts_from_simultaneous_training_and_browser_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 4)
    with LmdbRunStore(tmp_path / "run.lmdb") as store:
        for index in range(5):
            store.put("traces", f"trace-{index}", {"index": index})
    runner = ExperimentRunner(tmp_path)
    executor = ScriptedExecutor((_result("testing-1", 5.0), _result("training-1", 2.0)))
    telemetry = RecordedTelemetry()
    waits = 0

    def complete_once(
        active: dict[Future[WorkerResult], object], return_when: object
    ) -> tuple[set[Future[WorkerResult]], set[Future[WorkerResult]]]:
        nonlocal waits
        del return_when
        waits += 1
        if waits > 1:
            raise PipelineObserved
        return set(active), set()

    monkeypatch.setattr(experiment_module, "wait", complete_once)

    with pytest.raises(PipelineObserved):
        runner._pipeline(executor, telemetry)  # type: ignore[arg-type]

    assert _scheduled_iterations(tmp_path) == 11
    adjustment = next(row for row in telemetry.rows if row[0] == "training_schedule_adjusted")
    assert adjustment[1]["browser_median_duration_seconds"] == 5.0
    assert adjustment[1]["browser_sample_count"] == 1


def test_adaptation_clamps_bounds_and_skips_empty_browser_window(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 4)
    runner = ExperimentRunner(tmp_path)
    telemetry = RecordedTelemetry()
    config = load_config(tmp_path)

    _set_scheduled_iterations(tmp_path, config.training.max_batches_per_iteration)
    runner._adapt_training(1.0, (10.0, 12.0), telemetry)  # type: ignore[arg-type]
    assert _scheduled_iterations(tmp_path) == config.training.max_batches_per_iteration

    _set_scheduled_iterations(tmp_path, config.training.min_batches_per_iteration)
    runner._adapt_training(10.0, (1.0, 2.0), telemetry)  # type: ignore[arg-type]
    assert _scheduled_iterations(tmp_path) == config.training.min_batches_per_iteration

    before = len(telemetry.rows)
    runner._handle_training(  # type: ignore[arg-type]
        telemetry,
        _PipelineState({0: 0}, training_active=True),
        _result("training-empty", 2.0),
    )
    assert len(telemetry.rows) == before


def test_browser_retries_back_off_recover_and_fail_on_fifth_attempt(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 4)
    runner = ExperimentRunner(tmp_path)
    executor = ScriptedExecutor(())
    telemetry = RecordedTelemetry()
    active: dict[Future[WorkerResult], tuple[str, int, WorkerCommand]] = {}
    state = _PipelineState({0: 0, 1: 0}, training_active=True)
    failed = WorkerResult(
        command_id="testing-failed",
        status="failed",
        error_type="TimeoutError",
        error_message="page timeout",
    )

    for _ in range(4):
        runner._handle_testing(  # type: ignore[arg-type]
            executor, telemetry, active, state, 0, failed
        )
    runner._handle_testing(executor, telemetry, active, state, 1, failed)  # type: ignore[arg-type]

    retries = [
        row[1]["delay_seconds"] for row in telemetry.rows if row[0] == "worker_retry_scheduled"
    ]
    assert retries[:4] == [1.0, 2.0, 4.0, 8.0]
    assert state.failures == {0: 4, 1: 1}

    recovered = _result("testing-recovered", 3.0)
    runner._handle_testing(  # type: ignore[arg-type]
        executor, telemetry, active, state, 1, recovered
    )
    assert state.failures[1] == 0
    assert state.browser_durations == [3.0]
    assert any(row[0] == "worker_recovered" for row in telemetry.rows)

    with pytest.raises(RuntimeError, match="page timeout"):
        runner._handle_testing(  # type: ignore[arg-type]
            executor, telemetry, active, state, 0, failed
        )
    assert runner._cancel.is_set()
    assert telemetry.rows[-1][0] == "pipeline_failed"


def test_worker_result_contains_serialized_runner_warnings() -> None:
    runner = RunnerResult(
        status="completed",
        step_id="testing-1",
        duration_seconds=2.0,
        warnings=(
            RunnerWarning(
                hook="report",
                event=LifecycleEventName.RUN_FINISHED,
                fatal=False,
                error_type="OSError",
                message="disk busy",
            ),
        ),
    )

    result = _worker_result(WorkerCommand(command_id="test-1", name="testing"), runner)

    assert result.values["warnings"][0]["hook"] == "report"


def test_rig_commands_round_robin_browsers_and_keep_training_independent(tmp_path: Path) -> None:
    initialize_run("https://example.com", "rig", tmp_path, 5)
    runner = ExperimentRunner(tmp_path)

    first = runner._command("testing", 0)
    second = runner._command("testing", 1)
    training = runner._command("training", 0)

    assert first.parameters["browser"] == "chromium"
    assert second.parameters["browser"] == "firefox"
    assert "browser" not in training.parameters
    assert len({first.command_id, second.command_id, training.command_id}) == 3


def _result(command_id: str, duration: float) -> WorkerResult:
    return WorkerResult(
        command_id=command_id,
        status="completed",
        values={"duration_seconds": duration},
    )


def _scheduled_iterations(run_dir: Path) -> int:
    config = load_config(run_dir)
    with LmdbRunStore(
        run_dir / config.storage.database_directory,
        map_size=config.storage.database_map_size_bytes,
        readonly=True,
    ) as store:
        state = store.get("run", "state") or {}
        return int(state["scheduled_training_iterations"])


def _set_scheduled_iterations(run_dir: Path, iterations: int) -> None:
    with LmdbRunStore(run_dir / "run.lmdb") as store:
        store.update(
            "run",
            "state",
            lambda current: {**(current or {}), "scheduled_training_iterations": iterations},
        )


class PipelineObserved(RuntimeError):
    pass


class RecordedTelemetry:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, Any]]] = []

    def record(self, event: str, **values: Any) -> None:
        self.rows.append((event, values))


class ScriptedExecutor:
    def __init__(self, results: tuple[WorkerResult, ...]) -> None:
        self.results = list(results)

    def submit(self, *_: object) -> Future[WorkerResult]:
        future: Future[WorkerResult] = Future()
        if self.results:
            future.set_result(self.results.pop(0))
        return future
