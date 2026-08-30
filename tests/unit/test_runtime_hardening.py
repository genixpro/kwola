from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from kwola.browser.session import BrowserSessionCoordinator
from kwola.domain.actions import Action, ActionKind, ActionMap, BrowserKind
from kwola.hooks import HookRegistry
from kwola.instrumentation.telemetry import ConsoleEntry, NetworkEntry, TelemetryBuffer
from kwola.orchestration.initialize import initialize_run
from kwola.orchestration.results import RunnerResult
from kwola.orchestration.testing import TestingRunner as BrowserTestingRunner
from kwola.orchestration.training import TrainingRunner
from kwola.storage import LmdbRunStore
from kwola.training import batch_stream
from kwola.training import distributed as distributed_training
from kwola.training.benchmark import run_benchmark
from kwola.training.ddp import DistributedCoordinator, DistributedSettings
from kwola.training.optimizer import OptimizerMetrics
from kwola.training.replay import ReplaySampler


class FakeAssembler:
    def __init__(self, failure_offset: int | None = None) -> None:
        self.offsets: list[int] = []
        self.failure_offset = failure_offset

    def assemble(self, **values: Any) -> int:
        offset = int(values["sample_indexes"][0])
        self.offsets.append(offset)
        if offset == self.failure_offset:
            raise ValueError("assembly failed")
        return offset


def test_batch_stream_offsets_initial_batch_and_prefetch_failures() -> None:
    coordinator = SimpleNamespace(settings=SimpleNamespace(rank=1))
    direct = FakeAssembler()
    sampler = ReplaySampler(100, 4, 2, 1, seed=1, training_step=0)
    expected = [sampler.batch_indexes(index)[0] for index in range(3)]
    direct_batches = list(
        batch_stream.batches(direct, coordinator, 3, None, 4, sampler, -10.0, False)
    )
    assert [batch for batch, _duration in direct_batches] == expected

    prefetched = FakeAssembler()
    sampler = ReplaySampler(100, 4, 2, 1, seed=1, training_step=0)
    prefetched_batches = list(
        batch_stream.batches(prefetched, coordinator, 3, 99, 4, sampler, -10.0, True)
    )
    assert [batch for batch, _duration in prefetched_batches] == [99, *expected[1:]]

    sampler = ReplaySampler(100, 4, 2, 1, seed=1, training_step=0)
    failing = FakeAssembler(failure_offset=sampler.batch_indexes(0)[0])
    with pytest.raises(ValueError, match="assembly failed"):
        list(batch_stream.batches(failing, coordinator, 1, None, 4, sampler, -10.0, True))


def test_cpu_distributed_coordinator_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        distributed_training.distributed,
        "init_process_group",
        lambda **_values: events.append("start"),
    )
    monkeypatch.setattr(
        distributed_training.distributed, "barrier", lambda: events.append("barrier")
    )
    monkeypatch.setattr(
        distributed_training.distributed,
        "destroy_process_group",
        lambda: events.append("close"),
    )
    monkeypatch.setattr(distributed_training.distributed, "all_reduce", lambda *_a, **_k: None)
    coordinator = DistributedCoordinator(
        DistributedSettings(0, 1, 0, "tcp://127.0.0.1:1", backend="gloo")
    )

    with pytest.raises(RuntimeError, match="not started"):
        coordinator.barrier()
    coordinator.start()
    assert coordinator.is_publisher
    assert coordinator.device == torch.device("cpu")
    assert not coordinator.propagate_failure(False)
    coordinator.barrier()
    with pytest.raises(RuntimeError, match="already started"):
        coordinator.start()
    coordinator.close()
    coordinator.close()
    assert events == ["start", "barrier", "close"]

    with pytest.raises(ValueError, match="rank/world size"):
        DistributedSettings(2, 1, 0, "unused", backend="gloo")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        DistributedSettings(0, 1, 0, "unused")


def test_nccl_barrier_names_the_local_device(monkeypatch: pytest.MonkeyPatch) -> None:
    barrier_calls: list[dict[str, object]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(distributed_training.distributed, "init_process_group", lambda **_v: None)
    monkeypatch.setattr(
        distributed_training.distributed,
        "barrier",
        lambda **values: barrier_calls.append(values),
    )
    monkeypatch.setattr(distributed_training.distributed, "destroy_process_group", lambda: None)
    coordinator = DistributedCoordinator(DistributedSettings(0, 1, 1, "tcp://127.0.0.1:1"))

    with coordinator:
        coordinator.barrier()

    assert barrier_calls == [{"device_ids": [1]}]


class FakePage:
    url = "https://example.com/app"

    def __init__(self) -> None:
        self.waits: list[float] = []

    def wait_for_timeout(self, milliseconds: float) -> None:
        self.waits.append(milliseconds)

    def evaluate(self, _script: str, _coordinates: list[int]) -> str:
        return "pointer"

    def content(self) -> str:
        return "<html>ok</html>"


class FakeAdapter:
    def __init__(self, fail_navigation: bool = False) -> None:
        self.page = FakePage()
        self.fail_navigation = fail_navigation
        self.started = False
        self.closed = False
        self.allowed_checks = 0

    def start(self) -> None:
        self.started = True

    def navigate(self, _target: str) -> None:
        if self.fail_navigation:
            raise RuntimeError("navigation failed")

    def ensure_allowed(self) -> None:
        self.allowed_checks += 1

    def close(self) -> None:
        self.closed = True


class FakeProxy:
    def __init__(self) -> None:
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True


def _browser_session(adapter: FakeAdapter, proxy: FakeProxy) -> BrowserSessionCoordinator:
    telemetry = TelemetryBuffer()
    telemetry.record_console(ConsoleEntry("error", "broken", "https://example.com"))
    telemetry.record_network(NetworkEntry("GET", "https://example.com/api", 500))
    action_map = ActionMap((), 800, 600, "test")
    extractor = SimpleNamespace(extract=lambda _page: action_map)
    executor = SimpleNamespace(execute=lambda _page, _action: None)
    waiter = SimpleNamespace(wait=lambda _page: True)
    screenshots = SimpleNamespace(capture=lambda _page: b"png")
    autologin = SimpleNamespace(run=lambda _page: None)
    branches = SimpleNamespace(
        collect=lambda _page: SimpleNamespace(available=True, symbols=(7, 9))
    )
    return BrowserSessionCoordinator(
        adapter,  # type: ignore[arg-type]
        extractor,  # type: ignore[arg-type]
        executor,  # type: ignore[arg-type]
        waiter,  # type: ignore[arg-type]
        screenshots,  # type: ignore[arg-type]
        autologin,  # type: ignore[arg-type]
        telemetry,
        branches,  # type: ignore[arg-type]
        proxy,  # type: ignore[arg-type]
        clock=lambda: 12.0,
        action_settle_seconds=0.5,
    )


def test_browser_session_observes_executes_and_cleans_up() -> None:
    adapter = FakeAdapter()
    proxy = FakeProxy()
    session = _browser_session(adapter, proxy)

    observation = session.start("https://example.com")
    following = session.execute(Action(ActionKind.CLICK, 10, 20))

    assert observation.branch_symbols == (7, 9)
    assert observation.branch_trace_available
    assert following.errors == (
        "console:broken",
        "network:500:https://example.com/api:",
    )
    assert session.cursor_at(10, 20) == "pointer"
    assert session.page_html() == "<html>ok</html>"
    assert adapter.page.waits == [500.0]
    session.close()
    assert adapter.closed and proxy.closed


def test_browser_session_start_failure_closes_adapter_and_proxy() -> None:
    adapter = FakeAdapter(fail_navigation=True)
    proxy = FakeProxy()
    session = _browser_session(adapter, proxy)

    with pytest.raises(RuntimeError, match="navigation failed"):
        session.start("https://example.com")

    assert adapter.closed and proxy.closed


def test_testing_runner_records_completed_step_with_fake_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 10)
    runner = BrowserTestingRunner(tmp_path, clock=lambda: 20.0, hooks=HookRegistry(()))
    adapter = FakeAdapter()
    proxy = FakeProxy()
    session = _browser_session(adapter, proxy)
    monkeypatch.setattr(runner, "_session", lambda *_args: session)
    monkeypatch.setattr(runner, "_actions", lambda *_args: [1.0, -0.25])

    result = runner.run(browser=BrowserKind.CHROMIUM, random_policy=True, viewport=(800, 600))

    assert result.metrics == {"traces": 2, "reward": 0.75}
    with LmdbRunStore(tmp_path / "run.lmdb", readonly=True) as store:
        assert store.get("testing_steps", "testing-00000000") == {
            "browser": "chromium",
            "random": True,
            "trace_count": 2,
            "reward": 0.75,
        }


class FakeOptimizer:
    def __init__(self) -> None:
        self.calls = 0

    def step_training(self, _batch: object, _target: object, *_args: object) -> OptimizerMetrics:
        self.calls += 1
        return OptimizerMetrics(float(self.calls), 0.5, 8.0)


class FakeTarget:
    def __init__(self) -> None:
        self.loads = 0

    def load_state_dict(self, _state: object) -> None:
        self.loads += 1


def test_training_runner_iterations_state_record_and_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 11)
    runner = TrainingRunner(tmp_path, hooks=HookRegistry(()))
    optimizer = FakeOptimizer()
    target = FakeTarget()
    model = SimpleNamespace(state_dict=lambda: {"weight": 1})

    class Assembler:
        def trace_count(self) -> int:
            return 1

        def assemble(self, **_values: object) -> object:
            return object()

    monkeypatch.setattr(runner, "_assembler", lambda *_args, **_values: Assembler())

    metrics = runner._iterations(  # type: ignore[arg-type]
        optimizer,
        model,
        target,
        torch.device("cpu"),
        training_step=0,
        training_index=249,
        iteration_count=2,
    )
    runner._record("training-00000000", metrics, 2)
    monkeypatch.setattr(
        runner,
        "_run_single",
        lambda _gpu: RunnerResult(
            status="completed", step_id="training-00000001", duration_seconds=1.0
        ),
    )
    result = runner.run()

    assert optimizer.calls == 2
    assert target.loads == 1
    assert metrics.duration_seconds == 1.0
    assert result.status == "completed"
    with LmdbRunStore(tmp_path / "run.lmdb", readonly=True) as store:
        assert store.get("run", "state")["training_iterations"] == 2  # type: ignore[index]


def test_cpu_benchmark_executes_complete_optimizer_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 12)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = run_benchmark(tmp_path, iterations=1)

    assert result.device == "cpu"
    assert result.passed
    assert result.samples_per_second > 0


class FakeQueue:
    def get(self) -> str:
        return RunnerResult(
            status="completed", step_id="training-00000000", duration_seconds=1.0
        ).model_dump_json()


class FakeSpawnContext:
    def SimpleQueue(self) -> FakeQueue:
        return FakeQueue()


def test_distributed_entry_validation_simulation_and_rank_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    testing_run = tmp_path / "testing"
    rig_run = tmp_path / "rig"
    initialize_run("https://example.com", "testing", testing_run, 13)
    initialize_run("https://example.com", "rig", rig_run, 14)

    with pytest.raises(RuntimeError, match="one device index"):
        distributed_training.run_distributed_training(testing_run)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="insufficient CUDA"):
        distributed_training.run_distributed_training(rig_run)

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(distributed_training, "_prepare_cache", lambda _path: None)
    monkeypatch.setattr(distributed_training, "_shared_initial_batches", lambda _path: ())
    monkeypatch.setattr(distributed_training, "_free_port", lambda: 12345)
    monkeypatch.setattr(
        distributed_training.multiprocessing, "get_context", lambda _kind: FakeSpawnContext()
    )
    monkeypatch.setattr(distributed_training, "spawn", lambda *_args, **_values: None)
    assert distributed_training.run_distributed_training(rig_run).status == "completed"

    metrics = OptimizerMetrics(2.0, 4.0, 5.0)
    distributed_training._record_step(rig_run, 0, 2.0, metrics, 3, 10.0, 1.0, 2.0, 0.5)
    with LmdbRunStore(rig_run / "run.lmdb", readonly=True) as store:
        record = store.get("training_steps", "training-00000000")
    assert record is not None and record["ranks"] == 2


def test_distributed_cache_and_iteration_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 15)
    with pytest.raises(RuntimeError, match="at least one"):
        distributed_training._prepare_cache(tmp_path)
    with LmdbRunStore(tmp_path / "run.lmdb") as store:
        store.put("traces", "trace-1", {"screenshot": "unused"})
    distributed_training._prepare_cache(tmp_path)
    assert (
        distributed_training._load_models(
            tmp_path,
            None,
            SimpleNamespace(),
            SimpleNamespace(),
            torch.device("cpu"),  # type: ignore[arg-type]
        )
        is None
    )
    monkeypatch.setattr(distributed_training, "_free_port", lambda: 12345)
    assert distributed_training._free_port() == 12345

    batches = [(object(), 0.1), (object(), 0.2)]
    monkeypatch.setattr(distributed_training, "batches", lambda *_args, **_values: iter(batches))
    monkeypatch.setattr(distributed_training, "batch_to_device", lambda batch, _device: batch)
    monkeypatch.setattr(
        distributed_training,
        "_assembler",
        lambda *_args, **_values: SimpleNamespace(trace_count=lambda: 1),
    )
    progress: list[int] = []
    monkeypatch.setattr(
        distributed_training,
        "_record_progress",
        lambda _run, _coord, _index, iteration, *_args: progress.append(iteration),
    )
    optimizer = FakeOptimizer()
    target = FakeTarget()
    model = SimpleNamespace(state_dict=lambda: {})
    coordinator = SimpleNamespace(
        is_publisher=True,
        device=torch.device("cpu"),
        settings=SimpleNamespace(rank=0),
    )

    metrics, assembly, transfer = distributed_training._rank_iterations(
        tmp_path,
        coordinator,  # type: ignore[arg-type]
        model,  # type: ignore[arg-type]
        target,  # type: ignore[arg-type]
        optimizer,  # type: ignore[arg-type]
        0,
        249,
        2,
        None,
    )

    assert metrics.loss == 1.5
    assert assembly == pytest.approx(0.3)
    assert transfer >= 0
    assert progress == [1]
    assert target.loads == 1
