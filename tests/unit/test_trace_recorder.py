from pathlib import Path

from kwola.config import load_config
from kwola.domain.actions import Action, ActionKind, ActionMap, ActionTarget
from kwola.domain.observations import Observation, Viewport
from kwola.orchestration.initialize import initialize_run
from kwola.orchestration.trace_recorder import NoveltyState, TraceRecorder
from kwola.storage import LmdbRunStore


def test_trace_recorder_persists_features_blobs_and_bugs(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 1)
    config = load_config(tmp_path)
    before = _observation(b"before", "https://example.com", (), ())
    after = _observation(b"after", "https://example.com/next", (4,), (8,), ("console:broken",))
    artifacts: list[str] = []
    with LmdbRunStore(tmp_path / "run.lmdb", map_size=1024**2) as store:
        recorder = TraceRecorder(tmp_path, config, store, artifacts)
        reward = recorder.record(
            "testing-00000000",
            0,
            Action(ActionKind.CLICK, 2, 2),
            before,
            after,
            NoveltyState.initial(before),
            "pointer",
            ("<html>before</html>", "<html>after</html>"),
        )
        trace = store.get("traces", "testing-00000000-trace-0000")
        assert trace["reward"] == reward
        assert trace["new_branch_symbols"] == [4]
        assert trace["cursor"] == "pointer"
        assert store.scan("bugs")
    assert len(artifacts) == 3
    assert all((tmp_path / path).is_file() for path in artifacts)


def _observation(
    screenshot: bytes,
    url: str,
    branches: tuple[int, ...],
    network: tuple[int, ...],
    errors: tuple[str, ...] = (),
) -> Observation:
    target = ActionTarget(0, 0, 10, 10, "button", can_click=True)
    return Observation(
        url,
        screenshot,
        Viewport(10, 10),
        ActionMap((target,), 10, 10, "1"),
        1.0,
        branches,
        network,
        ("message",),
        errors,
    )
