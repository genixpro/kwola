from pathlib import Path

from kwola.config import load_config
from kwola.hooks import LifecycleEventName
from kwola.orchestration.experiment import ExperimentRunner, _worker_result
from kwola.orchestration.initialize import initialize_run
from kwola.orchestration.messages import WorkerCommand, WorkerResult
from kwola.orchestration.results import RunnerResult, RunnerWarning
from kwola.storage import LmdbRunStore


def test_experiment_adapts_training_iterations_to_worker_durations(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 4)
    runner = ExperimentRunner(tmp_path)

    runner._adapt_training((_result("testing-1", 5.0), _result("training-1", 2.0)))
    assert _scheduled_iterations(tmp_path) == 11

    runner._adapt_training((_result("testing-2", 1.0), _result("training-2", 4.0)))
    assert _scheduled_iterations(tmp_path) == 10


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
