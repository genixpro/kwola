from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from kwola.orchestration.initialize import initialize_run
from kwola.orchestration.results import RunnerResult
from kwola.storage import LmdbRunStore
from kwola.training import distributed as distributed_training
from kwola.training.optimizer import OptimizerMetrics


class Queue:
    def __init__(self) -> None:
        self.values: list[str] = []

    def put(self, value: str) -> None:
        self.values.append(value)


class Coordinator:
    def __init__(self, settings: object) -> None:
        self.settings = settings
        self.device = torch.device("cpu")
        self.is_publisher = True
        self.barriers = 0

    def __enter__(self) -> "Coordinator":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def barrier(self) -> None:
        self.barriers += 1


def test_training_rank_reduces_and_publishes_rank_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 1)
    queue = Queue()
    coordinator: Coordinator | None = None

    def coordinator_factory(settings: object) -> Coordinator:
        nonlocal coordinator
        coordinator = Coordinator(settings)
        return coordinator

    monkeypatch.setattr(
        distributed_training,
        "DistributedSettings",
        lambda rank, world, device, method: SimpleNamespace(
            rank=rank, world_size=world, local_device=device, init_method=method
        ),
    )
    monkeypatch.setattr(distributed_training, "DistributedCoordinator", coordinator_factory)
    monkeypatch.setattr(
        distributed_training,
        "_rank_step",
        lambda *_args: (
            OptimizerMetrics(3.0, 2.0, 4.0),
            object(),
            object(),
            object(),
            5,
            100,
            7,
            0.25,
            0.5,
        ),
    )
    monkeypatch.setattr(distributed_training.distributed, "all_reduce", lambda *_a, **_k: None)
    published: list[tuple[int, float]] = []

    def publish(*_args: object) -> RunnerResult:
        published.append((5, 3.0))
        return RunnerResult(status="completed", step_id="training-00000005", duration_seconds=1)

    monkeypatch.setattr(distributed_training, "_publish_result", publish)

    distributed_training._training_rank(0, 1, (0,), "unused", tmp_path, queue, ())

    assert coordinator is not None and coordinator.barriers == 1
    assert published == [(5, 3.0)]
    assert RunnerResult.model_validate_json(queue.values[0]).status == "completed"


class RankModel:
    def to(self, _device: torch.device) -> "RankModel":
        return self

    def eval(self) -> "RankModel":
        return self

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, _state: object, strict: bool = True) -> None:
        return None


def test_rank_step_builds_model_optimizer_and_uses_scheduled_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 2)
    with LmdbRunStore(tmp_path / "run.lmdb") as store:
        store.put(
            "run",
            "state",
            {
                "training_steps": 4,
                "training_iterations": 20,
                "scheduled_training_iterations": 3,
            },
        )
        for index in range(12):
            store.put("traces", f"trace-{index}", {"index": index})
    model = RankModel()
    monkeypatch.setattr(distributed_training, "TraceNet", lambda *_args, **_values: model)
    monkeypatch.setattr(distributed_training, "_load_models", lambda *_args: None)
    monkeypatch.setattr(distributed_training, "DistributedDataParallel", lambda value, **_v: value)
    optimizer = SimpleNamespace(optimizer=SimpleNamespace())
    monkeypatch.setattr(distributed_training, "ModelOptimizer", lambda *_args: optimizer)
    received: list[tuple[int, int, int, object]] = []

    def iterations(
        _run: Path,
        _coordinator: object,
        _model: object,
        _target: object,
        _optimizer: object,
        training_step: int,
        training_index: int,
        count: int,
        initial: object,
    ) -> tuple[OptimizerMetrics, float, float]:
        received.append((training_step, training_index, count, initial))
        return OptimizerMetrics(1.0, 2.0, 3.0), 0.4, 0.2

    monkeypatch.setattr(distributed_training, "_rank_iterations", iterations)
    coordinator = SimpleNamespace(
        device=torch.device("cpu"), settings=SimpleNamespace(local_device=0)
    )

    result = distributed_training._rank_step(
        tmp_path,
        coordinator,
        "initial",  # type: ignore[arg-type]
    )

    assert result[4:] == (4, 12, 3, 0.4, 0.2)
    assert received == [(4, 20, 3, "initial")]


class BatchAssembler:
    def __init__(self) -> None:
        self.indexes: list[tuple[int, ...]] = []

    def trace_count(self) -> int:
        return 1

    def assemble(self, **values: Any) -> object:
        sample_indexes = tuple(values["sample_indexes"])
        self.indexes.append(sample_indexes)
        return SimpleNamespace(sample_indexes=sample_indexes)


def test_shared_batches_progress_and_checkpoint_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 3)
    assembler = BatchAssembler()
    monkeypatch.setattr(distributed_training, "_assembler", lambda *_args, **_values: assembler)
    monkeypatch.setattr(
        distributed_training,
        "share_batch",
        lambda batch: SimpleNamespace(shared=batch.sample_indexes),
    )
    shared = distributed_training._shared_initial_batches(tmp_path)
    assert [batch.shared for batch in shared] == [(0, 0, 0, 0)]

    progress: list[dict[str, object]] = []
    monkeypatch.setattr(distributed_training.time, "perf_counter", lambda: 12.0)
    monkeypatch.setattr(distributed_training.torch.cuda, "memory_allocated", lambda _d: 10)
    monkeypatch.setattr(distributed_training.torch.cuda, "memory_reserved", lambda _d: 20)
    monkeypatch.setattr(
        distributed_training,
        "record_training_progress",
        lambda _run, **values: progress.append(values),
    )
    coordinator = SimpleNamespace(device=torch.device("cpu"), settings=SimpleNamespace(rank=0))
    distributed_training._record_progress(
        tmp_path,
        coordinator,  # type: ignore[arg-type]
        10,
        1,
        4,
        [OptimizerMetrics(1.0, 0.5, 8.0)],
        0.2,
        0.1,
        10.0,
    )
    assert progress[0]["training_iteration"] == 12
    assert progress[0]["gpu_memory_reserved_bytes"] == 20

    model = SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0])})
    target = SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([2.0])})
    optimizer = SimpleNamespace(optimizer=SimpleNamespace(state_dict=lambda: {"state": {}}))
    result = distributed_training._publish_result(
        tmp_path,
        model,  # type: ignore[arg-type]
        target,  # type: ignore[arg-type]
        optimizer,  # type: ignore[arg-type]
        0,
        4,
        OptimizerMetrics(1.5, 0.5, 8.0),
        1.5,
        2.0,
        2,
        0.2,
        0.1,
    )

    assert result.status == "completed"
    assert result.artifacts == ("checkpoints/checkpoint-00000001.pt",)
    payload = torch.load(tmp_path / result.artifacts[0], weights_only=True)
    assert payload["learning_schema_version"] == 2
    assert payload["model"]["weight"] == 1
    assert payload["target_model"]["weight"] == 2
    assert payload["optimizer"] == {"state": {}}
    with LmdbRunStore(tmp_path / "run.lmdb", readonly=True) as store:
        state = store.get("run", "state")
    assert state is not None
    assert state["training_steps"] == 1
    assert state["training_iterations"] == 2
    assert state["training_trace_count"] == 4
