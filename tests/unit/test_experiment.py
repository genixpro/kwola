from pathlib import Path

from kwola.config import load_config
from kwola.orchestration.experiment import ExperimentRunner
from kwola.orchestration.initialize import initialize_run
from kwola.orchestration.messages import WorkerResult
from kwola.storage import LmdbRunStore


def test_experiment_adapts_training_iterations_to_worker_durations(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 4)
    runner = ExperimentRunner(tmp_path)

    runner._adapt_training((_result("testing-1", 5.0), _result("training-1", 2.0)))
    assert _scheduled_iterations(tmp_path) == 11

    runner._adapt_training((_result("testing-2", 1.0), _result("training-2", 4.0)))
    assert _scheduled_iterations(tmp_path) == 10


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
