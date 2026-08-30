import json
from pathlib import Path

from kwola.bin.cli import main
from kwola.orchestration.initialize import initialize_run
from kwola.orchestration.status import pipeline_status
from kwola.orchestration.telemetry import TelemetryWriter, read_telemetry
from kwola.storage import LmdbRunStore
from kwola.training.telemetry import record_training_progress


def test_telemetry_is_durable_and_status_aggregates_pipeline_rates(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 4)
    with TelemetryWriter(tmp_path, interval_seconds=60) as telemetry:
        telemetry.record("worker_submitted", worker="testing", command_id="testing-1")
        telemetry.record(
            "resources",
            cpu_percent=50.0,
            gpus=[{"index": 0, "gpu_percent": 80.0}],
        )
    record_training_progress(
        tmp_path,
        event="training_progress",
        step_iterations_completed=8,
        end_to_end_samples_per_second=120.0,
    )
    with LmdbRunStore(tmp_path / "run.lmdb") as store:
        store.put("testing_steps", "testing-00000000", {"reward": 2.5})
        store.put("traces", "testing-00000000-trace-0000", {"reward": 2.5})
        store.put(
            "training_steps",
            "training-00000000",
            {"iterations": 10, "optimizer_seconds": 2.0},
        )
        store.update(
            "run",
            "state",
            lambda current: {
                **(current or {}),
                "scheduled_training_iterations": 8,
            },
        )

    status = pipeline_status(tmp_path)

    assert status["in_flight"] == {"testing": 1}
    assert status["traces"] == 1
    assert status["training_iterations"] == 10
    assert status["optimizer_sample_rate_per_second"] == 20.0
    assert status["scheduled_training_iterations"] == 8
    assert status["resources"]["cpu_percent"] == 50.0
    assert status["recent_resource_averages"] == {
        "sample_count": 1,
        "cpu_percent": 50.0,
        "gpu_percent": {"0": 80.0},
    }
    assert status["latest_training_progress"]["step_iterations_completed"] == 8


def test_telemetry_reader_skips_partial_or_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.jsonl"
    path.write_text('{"event":"ok"}\ninvalid\n[]\n', encoding="utf-8")

    assert read_telemetry(path) == [{"event": "ok"}]
    assert read_telemetry(tmp_path / "missing") == []


def test_status_cli_payload_remains_json_serializable(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 5)

    assert json.loads(json.dumps(pipeline_status(tmp_path)))["configured_browser_workers"] == 1


def test_status_cli_prints_the_live_payload(tmp_path: Path, capsys: object) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 6)

    assert main(["status", str(tmp_path)]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert json.loads(output)["training_steps"] == 0
