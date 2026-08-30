import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from kwola.domain.actions import Action, ActionKind, ActionMap, BrowserKind
from kwola.domain.observations import Observation, Viewport
from kwola.hooks import HookRegistry
from kwola.instrumentation import addon as addon_module
from kwola.instrumentation.addon import InstrumentationAddon
from kwola.instrumentation.telemetry import TelemetryBuffer
from kwola.orchestration import experiment as experiment_module
from kwola.orchestration import telemetry as pipeline_telemetry
from kwola.orchestration.experiment import ExperimentRunner
from kwola.orchestration.initialize import initialize_run
from kwola.orchestration.messages import WorkerCommand, WorkerResult
from kwola.orchestration.results import RunnerResult
from kwola.orchestration.testing import TestingRunner as BrowserTestingRunner
from kwola.orchestration.training import TrainingRunner
from kwola.storage import LmdbRunStore, load_manifest
from kwola.training import distributed_diagnostic
from kwola.training.distributed_diagnostic import DistributedDiagnosticResult
from kwola.training.optimizer import OptimizerMetrics


class FakeHeaders(dict[str, str]):
    pass


class FakeRequest:
    def __init__(self, url: str, method: str = "GET") -> None:
        self.url = url
        self.method = method
        self.headers = FakeHeaders({"User-Agent": "Browser"})


class FakeResponse:
    def __init__(self, content: bytes, content_type: str, status: int = 200) -> None:
        self.content = content
        self.status_code = status
        self.headers = FakeHeaders(
            {
                "Content-Type": content_type,
                "Set-Cookie": "secret",
                "X-Public": "yes",
            }
        )


class FakeFlow:
    def __init__(self, url: str, response: FakeResponse | None) -> None:
        self.request = FakeRequest(url)
        self.response = response
        self.error: object | None = None


class FakeResources:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def capture(self, **values: Any) -> None:
        self.records.append(values)


def test_instrumentation_addon_rewrites_captures_and_sanitizes() -> None:
    telemetry = TelemetryBuffer()
    resources = FakeResources()
    addon = InstrumentationAddon(
        telemetry,
        resources,  # type: ignore[arg-type]
        rewrite_html=True,
        rewrite_javascript=True,
        capture_resources=True,
    )
    addon._javascript.rewrite = lambda _url, source: source + b"-instrumented"  # type: ignore[method-assign]
    addon._html.rewrite = lambda source: source + b"-html"  # type: ignore[method-assign]

    javascript = FakeFlow(
        "https://example.com/app.js?version=1",
        FakeResponse(b"source", "application/octet-stream"),
    )
    addon.requestheaders(javascript)  # type: ignore[arg-type]
    addon.response(javascript)  # type: ignore[arg-type]
    html = FakeFlow("https://example.com/", FakeResponse(b"page", "text/html; charset=utf-8"))
    addon.response(html)  # type: ignore[arg-type]
    addon.response(FakeFlow("https://example.com/none", None))  # type: ignore[arg-type]
    failed = FakeFlow("https://example.com/fail", None)
    failed.error = RuntimeError("connection reset")
    addon.error(failed)  # type: ignore[arg-type]

    assert javascript.request.headers["Accept-Encoding"] == "identity"
    assert javascript.request.headers["X-Kwola"] == "true"
    assert javascript.response is not None
    assert javascript.response.content.endswith(b"instrumented")
    assert resources.records[0]["rewrite_kind"] == "javascript"
    assert resources.records[0]["headers"] == {
        "Content-Type": "application/octet-stream",
        "X-Public": "yes",
    }
    _console, network = telemetry.snapshot()
    assert [entry.status for entry in network] == [200, 200, 0]
    addon.done()


def test_instrumentation_rewrite_failure_and_passthrough() -> None:
    addon = InstrumentationAddon(
        TelemetryBuffer(),
        FakeResources(),  # type: ignore[arg-type]
        rewrite_html=True,
        rewrite_javascript=False,
        capture_resources=False,
    )
    addon._html.rewrite = lambda _source: (_ for _ in ()).throw(UnicodeError("bad"))  # type: ignore[method-assign]
    source = b"unchanged"
    assert addon._rewrite("https://example.com", "text/html", source) == (source, None)
    assert addon._rewrite("https://example.com/image", "image/png", source) == (source, None)
    assert addon_module._is_javascript("https://example.com/code.mjs?x=1", "text/plain")


def test_pipeline_resource_sampling_and_gpu_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    assert pipeline_telemetry._cpu_percent((100, 40), (200, 70)) == 70.0
    assert pipeline_telemetry._cpu_percent(None, (1, 1)) == 0.0
    assert pipeline_telemetry._cpu_percent((1, 1), (1, 1)) == 0.0
    monkeypatch.setattr(pipeline_telemetry, "_memory_sample", lambda: {"memory_total_kib": 10})
    original_process_tree_sample = pipeline_telemetry._process_tree_sample
    monkeypatch.setattr(
        pipeline_telemetry, "_process_tree_sample", lambda _pid: {"process_count": 2}
    )
    original_gpu_sample = pipeline_telemetry._gpu_sample
    monkeypatch.setattr(
        pipeline_telemetry,
        "_gpu_sample",
        lambda: [{"index": 0, "gpu_percent": 50.0}],
    )
    sample = pipeline_telemetry._resource_sample((100, 40), (200, 70))
    assert sample["memory_total_kib"] == 10
    assert sample["process_count"] == 2
    assert sample["gpus"][0]["gpu_percent"] == 50.0  # type: ignore[index]

    completed = SimpleNamespace(
        stdout="0, 80, 20, 1000, 120.5, 65\ninvalid\n",
    )
    monkeypatch.setattr(pipeline_telemetry, "_gpu_sample", original_gpu_sample)
    monkeypatch.setattr(pipeline_telemetry.subprocess, "run", lambda *_a, **_k: completed)
    assert pipeline_telemetry._gpu_sample() == [
        {
            "index": 0,
            "gpu_percent": 80.0,
            "memory_percent": 20.0,
            "memory_used_mib": 1000.0,
            "power_watts": 120.5,
            "temperature_c": 65.0,
        }
    ]

    def unavailable(*_args: object, **_values: object) -> None:
        raise OSError("missing")

    monkeypatch.setattr(pipeline_telemetry.subprocess, "run", unavailable)
    assert pipeline_telemetry._gpu_sample() == []
    monkeypatch.setattr(pipeline_telemetry, "_process_tree_sample", original_process_tree_sample)
    assert pipeline_telemetry._process_tree_sample(os.getpid())["process_count"] >= 1


class DiagnosticQueue:
    def __init__(self) -> None:
        self.value = DistributedDiagnosticResult(
            passed=True,
            world_size=2,
            losses=(1.0, 2.0),
            devices=("cuda:0", "cuda:1"),
        ).model_dump_json()

    def get(self) -> str:
        return self.value

    def put(self, value: str) -> None:
        self.value = value


class DiagnosticContext:
    def SimpleQueue(self) -> DiagnosticQueue:
        return DiagnosticQueue()


class FakeDiagnosticCoordinator:
    def __init__(self, settings: object) -> None:
        self.settings = settings
        self.device = torch.device("cpu")
        self.is_publisher = True
        self.barriers = 0

    def __enter__(self) -> "FakeDiagnosticCoordinator":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def barrier(self) -> None:
        self.barriers += 1


class FakeDiagnosticModel:
    def to(self, _device: torch.device) -> "FakeDiagnosticModel":
        return self


def test_distributed_diagnostic_validation_and_simulated_rank(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="two CUDA"):
        distributed_diagnostic.run_two_rank_diagnostic()

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(distributed_diagnostic.distributed, "is_nccl_available", lambda: True)
    monkeypatch.setattr(
        distributed_diagnostic.multiprocessing, "get_context", lambda _kind: DiagnosticContext()
    )
    monkeypatch.setattr(distributed_diagnostic, "_free_port", lambda: 12345)
    monkeypatch.setattr(distributed_diagnostic, "spawn", lambda *_args, **_values: None)
    assert distributed_diagnostic.run_two_rank_diagnostic().passed

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed_diagnostic, "DistributedCoordinator", FakeDiagnosticCoordinator)
    monkeypatch.setattr(distributed_diagnostic, "TraceNet", lambda *_a, **_k: FakeDiagnosticModel())
    monkeypatch.setattr(
        distributed_diagnostic, "DistributedDataParallel", lambda model, **_k: model
    )
    monkeypatch.setattr(
        distributed_diagnostic,
        "ModelOptimizer",
        lambda *_args: SimpleNamespace(step=lambda _request: OptimizerMetrics(1.5, 0.1, 20.0)),
    )
    monkeypatch.setattr(distributed_diagnostic, "diagnostic_batch", lambda **_values: object())

    def gather(gathered: list[torch.Tensor], _loss: torch.Tensor) -> None:
        gathered[0].fill_(1.5)
        gathered[1].fill_(2.5)

    monkeypatch.setattr(distributed_diagnostic.distributed, "all_gather", gather)
    queue = DiagnosticQueue()
    distributed_diagnostic._diagnostic_rank(0, 2, "unused", queue)
    assert json.loads(queue.value)["losses"] == [1.5, 2.5]

    monkeypatch.setattr(
        distributed_diagnostic,
        "run_two_rank_diagnostic",
        lambda: DistributedDiagnosticResult(
            passed=True,
            world_size=2,
            losses=(1.0, 2.0),
            devices=("cuda:0", "cuda:1"),
        ),
    )
    distributed_diagnostic.main()
    assert '"passed": true' in capsys.readouterr().out


class FakeSupervisor:
    result: WorkerResult | Exception

    def __init__(self, _handler: object) -> None:
        self.result = type(self).result

    def __enter__(self) -> "FakeSupervisor":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def run(self, *_args: object, **_values: object) -> WorkerResult:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def logs(self) -> tuple[str, ...]:
        return ("worker log",)


def test_supervised_wrapper_success_failure_and_cancelled_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = WorkerCommand(command_id="testing-1", name="testing")
    completed = WorkerResult(command_id=command.command_id, status="completed")
    monkeypatch.setattr(experiment_module, "WorkerSupervisor", FakeSupervisor)
    FakeSupervisor.result = completed
    assert (
        experiment_module._supervised(lambda *_a: completed, command, 1, threading.Event(), 0)
        == completed
    )

    FakeSupervisor.result = RuntimeError("supervisor failed")
    failed = experiment_module._supervised(lambda *_a: completed, command, 1, threading.Event(), 0)
    assert failed.status == "failed" and failed.error_type == "RuntimeError"

    cancelled = threading.Event()
    cancelled.set()
    result = experiment_module._supervised(lambda *_a: completed, command, 1, cancelled, 1)
    assert result.status == "cancelled"


class FakePolicy:
    def __init__(self, *_args: object) -> None:
        pass

    def select(self, *_args: object, **_values: object) -> Action:
        return Action(ActionKind.CLICK, 10, 20)


class FakeRecorder:
    def __init__(self, *_args: object) -> None:
        self.calls = 0

    def record(self, *_args: object) -> float:
        self.calls += 1
        return 0.5

    def claim_initial(self, *_args: object) -> None:
        return None


class ActionSession:
    def __init__(self, observation: Observation) -> None:
        self.observation = observation

    def cursor_at(self, _x: int, _y: int) -> str:
        return "pointer"

    def page_html(self) -> str:
        return "<html></html>"

    def execute(self, _action: Action) -> Observation:
        return self.observation


def _observation() -> Observation:
    return Observation(
        url="https://example.com",
        screenshot=b"png",
        viewport=Viewport(800, 600),
        action_map=ActionMap((), 800, 600, "test"),
        timestamp=1.0,
    )


def test_testing_actions_session_construction_and_sample_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 20)
    runner = BrowserTestingRunner(tmp_path, hooks=HookRegistry(()))
    monkeypatch.setattr(experiment_module, "TestingRunner", BrowserTestingRunner)
    from kwola.orchestration import testing as testing_module

    monkeypatch.setattr(testing_module, "InferencePolicy", FakePolicy)
    monkeypatch.setattr(testing_module, "TraceRecorder", FakeRecorder)
    monkeypatch.setattr(
        testing_module,
        "ProxyService",
        lambda *_args: SimpleNamespace(
            start=lambda: None,
            close=lambda: None,
            port=12345,
            server="http://127.0.0.1:12345",
        ),
    )
    with LmdbRunStore(tmp_path / "run.lmdb") as store:
        rewards = runner._actions(
            ActionSession(_observation()),  # type: ignore[arg-type]
            store,
            "testing-00000000",
            0,
            _observation(),
            [],
            True,
            0,
        )
        session = runner._session(BrowserKind.CHROMIUM, runner._viewport(None), store)
        session.close()
        assert runner._proxy(store, TelemetryBuffer()) is not None

        prepared: list[tuple[str, int]] = []

        class FakeSampleAssembler:
            def __init__(self, *_args: object, **_values: object) -> None:
                pass

            def prepare_step(self, step_id: str, workers: int) -> None:
                prepared.append((step_id, workers))

        monkeypatch.setattr(testing_module, "RecordedSampleAssembler", FakeSampleAssembler)
        runner._prepare_samples(store, "testing-00000000")

    assert len(rewards) == runner._config.policy.testing_sequence_length
    assert prepared == [("testing-00000000", runner._config.training.sample_cache_workers)]


class FakeTrainingModel:
    def __init__(self, *_args: object, **_values: object) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def to(self, _device: torch.device) -> "FakeTrainingModel":
        return self

    def eval(self) -> "FakeTrainingModel":
        return self

    def parameters(self) -> Any:
        return iter((self.weight,))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": self.weight.detach()}

    def load_state_dict(self, _payload: object, strict: bool = True) -> None:
        return None

    def load_checkpoint_state_dict(self, payload: object) -> None:
        self.load_state_dict(payload)


class FakeTrainingOptimizer:
    def __init__(self, *_args: object) -> None:
        self.optimizer = SimpleNamespace(
            state_dict=lambda: {"state": {}},
            load_state_dict=lambda _payload: None,
        )


def test_single_training_path_checkpoint_load_and_distributed_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kwola.orchestration import training as training_module

    run_dir = tmp_path / "single"
    initialize_run("https://example.com", "testing", run_dir, 21)
    with LmdbRunStore(run_dir / "run.lmdb") as store:
        store.put("traces", "trace-1", {"screenshot": "unused"})
    runner = TrainingRunner(run_dir, hooks=HookRegistry(()))
    monkeypatch.setattr(training_module, "TraceNet", FakeTrainingModel)
    monkeypatch.setattr(training_module, "ModelOptimizer", FakeTrainingOptimizer)
    monkeypatch.setattr(
        runner,
        "_assembler",
        lambda _store, **_values: SimpleNamespace(prepare_cache=lambda _workers: 1),
    )
    monkeypatch.setattr(
        runner,
        "_iterations",
        lambda *_args: OptimizerMetrics(1.0, 0.5, 8.0),
    )
    monkeypatch.setattr(runner, "_load_checkpoint", lambda *_args: None)
    monkeypatch.setattr(runner, "_maybe_publish", lambda *_args: None)
    result = runner._run_single(None)
    assert result.metrics["loss"] == 1.0

    model = FakeTrainingModel()
    target = FakeTrainingModel()
    optimizer = FakeTrainingOptimizer()
    TrainingRunner._load_checkpoint(runner, model, target, optimizer, None)  # type: ignore[arg-type]
    metadata = SimpleNamespace(file="checkpoint.pt")
    monkeypatch.setattr(training_module, "verify_checkpoint", lambda *_args: Path("checkpoint.pt"))
    monkeypatch.setattr(
        training_module.torch,
        "load",
        lambda *_args, **_values: {
            "learning_schema_version": 2,
            "model": {},
            "target_model": {},
            "optimizer": {},
        },
    )
    TrainingRunner._load_checkpoint(  # type: ignore[arg-type]
        runner, model, target, optimizer, metadata
    )

    class FakePublisher:
        def __init__(self, *_args: object) -> None:
            pass

        def publish(self, **values: Any) -> tuple[Path, object]:
            assert values["rank"] == 0
            return run_dir / "checkpoints" / "checkpoint.pt", load_manifest(run_dir)

    monkeypatch.setattr(training_module, "CheckpointPublisher", FakePublisher)
    assert (
        TrainingRunner._maybe_publish(  # type: ignore[arg-type]
            runner, model, target, optimizer, load_manifest(run_dir), 1
        )
        is not None
    )

    rig_dir = tmp_path / "rig"
    initialize_run("https://example.com", "rig", rig_dir, 22)
    rig_runner = TrainingRunner(rig_dir, hooks=HookRegistry(()))
    monkeypatch.setattr(
        distributed_diagnostic,
        "run_two_rank_diagnostic",
        lambda: DistributedDiagnosticResult(
            passed=True, world_size=2, losses=(1.0, 1.0), devices=("cuda:0", "cuda:1")
        ),
    )
    from kwola.training import distributed as distributed_training

    monkeypatch.setattr(
        distributed_training,
        "run_distributed_training",
        lambda _path: RunnerResult(
            status="completed", step_id="training-00000000", duration_seconds=1.0
        ),
    )
    assert rig_runner.run().status == "completed"


def test_experiment_run_success_and_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 23)
    runner = ExperimentRunner(tmp_path)
    monkeypatch.setattr(runner, "_pipeline", lambda *_args: 7)
    assert runner.run() == 7
    monkeypatch.setattr(
        runner,
        "_pipeline",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert runner.run() == 130
