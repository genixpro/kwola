"""Aggregate durable run counters into an operator-facing pipeline status."""

import time
from pathlib import Path
from typing import Any

from kwola.config import load_config
from kwola.storage import LmdbRunStore

from .telemetry import read_telemetry


def pipeline_status(run_dir: Path) -> dict[str, Any]:
    config = load_config(run_dir)
    with LmdbRunStore(
        run_dir / config.storage.database_directory,
        map_size=config.storage.database_map_size_bytes,
        readonly=True,
    ) as store:
        state = store.get("run", "state") or {}
        training = [record for _key, record in store.scan("training_steps")]
        testing = [record for _key, record in store.scan("testing_steps")]
        traces = sum(1 for _ in store.scan("traces"))
        bugs = sum(1 for _ in store.scan("bugs"))
    events = read_telemetry(run_dir / "telemetry" / "pipeline.jsonl")
    progress = read_telemetry(run_dir / "telemetry" / "training-progress.jsonl")
    start = next(
        (float(row["timestamp"]) for row in reversed(events) if row["event"] == "pipeline_started"),
        time.time(),
    )
    elapsed = max(time.time() - start, 1e-9)
    iterations = sum(int(row.get("iterations", 0)) for row in training)
    optimizer_seconds = sum(float(row.get("optimizer_seconds", 0)) for row in training)
    global_samples = iterations * config.training.batch_size * config.training.world_size
    latest_resources = next((row for row in reversed(events) if row["event"] == "resources"), None)
    in_flight = _in_flight(events)
    return {
        "elapsed_seconds": elapsed,
        "configured_browser_workers": config.orchestration.browser_workers,
        "in_flight": in_flight,
        "browser_worker_health": _browser_worker_health(
            events, config.orchestration.browser_workers
        ),
        "testing_steps": len(testing),
        "traces": traces,
        "trace_rate_per_second": traces / elapsed,
        "reward": sum(float(row.get("reward", 0)) for row in testing),
        "bugs": bugs,
        "training_steps": len(training),
        "training_iterations": iterations,
        "training_step_rate_per_second": len(training) / elapsed,
        "iteration_rate_per_second": iterations / elapsed,
        "global_sample_rate_per_second": global_samples / elapsed,
        "optimizer_sample_rate_per_second": (
            global_samples / optimizer_seconds if optimizer_seconds > 0 else 0.0
        ),
        "scheduled_training_iterations": int(
            state.get("scheduled_training_iterations", config.training.batches_per_iteration)
        ),
        "replay_sample_credit": int(state.get("replay_sample_credit", 0)),
        "replay_samples_per_new_trace": config.training.replay_samples_per_new_trace,
        "resources": latest_resources,
        "recent_resource_averages": _resource_averages(events),
        "latest_training_progress": progress[-1] if progress else None,
    }


def _browser_worker_health(
    events: list[dict[str, Any]], worker_count: int
) -> dict[str, dict[str, Any]]:
    health: dict[str, dict[str, Any]] = {
        str(slot): {
            "consecutive_failures": 0,
            "retry_delay_seconds": 0.0,
            "last_error": None,
        }
        for slot in range(worker_count)
    }
    for row in events:
        if row.get("event") == "pipeline_started":
            for state in health.values():
                state.update(
                    consecutive_failures=0,
                    retry_delay_seconds=0.0,
                    last_error=None,
                )
            continue
        slot = row.get("slot")
        if not isinstance(slot, int) or str(slot) not in health:
            continue
        state = health[str(slot)]
        if row.get("event") == "worker_retry_scheduled":
            state["consecutive_failures"] = int(row.get("consecutive_failures", 0))
            state["retry_delay_seconds"] = float(row.get("delay_seconds", 0))
            error_type = row.get("error_type") or "WorkerFailure"
            error_message = row.get("error_message") or "failed"
            state["last_error"] = f"{error_type}: {error_message}"
        elif row.get("event") == "worker_recovered":
            state.update(
                consecutive_failures=0,
                retry_delay_seconds=0.0,
                last_error=None,
            )
    return health


def _in_flight(events: list[dict[str, Any]]) -> dict[str, int]:
    active: dict[str, str] = {}
    for row in events:
        if row.get("event") == "pipeline_started":
            active.clear()
            continue
        command_id = row.get("command_id")
        if not isinstance(command_id, str):
            continue
        if row.get("event") == "worker_submitted":
            active[command_id] = str(row.get("worker", "unknown"))
        elif row.get("event") == "worker_completed":
            active.pop(command_id, None)
    counts: dict[str, int] = {}
    for worker in active.values():
        counts[worker] = counts.get(worker, 0) + 1
    return counts


def _resource_averages(events: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [row for row in events if row.get("event") == "resources"][-12:]
    if not samples:
        return {}
    cpu_values = [float(row.get("cpu_percent", 0)) for row in samples]
    gpu_values: dict[int, list[float]] = {}
    for row in samples:
        gpus = row.get("gpus", [])
        if not isinstance(gpus, list):
            continue
        for gpu in gpus:
            if isinstance(gpu, dict) and "index" in gpu:
                gpu_values.setdefault(int(gpu["index"]), []).append(
                    float(gpu.get("gpu_percent", 0))
                )
    return {
        "sample_count": len(samples),
        "cpu_percent": sum(cpu_values) / len(cpu_values),
        "gpu_percent": {
            str(index): sum(values) / len(values) for index, values in gpu_values.items()
        },
    }
