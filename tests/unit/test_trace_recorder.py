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
    final = _observation(b"final", "https://example.com/final", (4, 5), (8,))
    artifacts: list[str] = []
    with LmdbRunStore(tmp_path / "run.lmdb", map_size=1024**2) as store:
        recorder = TraceRecorder(tmp_path, config, store, artifacts)
        novelty = NoveltyState.initial(before)
        reward = recorder.record(
            "testing-00000000",
            0,
            Action(ActionKind.CLICK, 2, 2),
            before,
            after,
            novelty,
            "pointer",
            ("<html>before</html>", "<html>after</html>"),
        )
        trace = store.get("traces", "testing-00000000-trace-0000")
        assert trace is not None
        assert trace["reward"] == reward
        assert trace["new_branch_symbols"] == [4]
        assert trace["cursor"] == "pointer"
        assert store.scan("bugs")
        recorder.record(
            "testing-00000000",
            1,
            Action(ActionKind.CLICK, 3, 3),
            after,
            final,
            novelty,
            "pointer",
            ("<html>after</html>", "<html>final</html>"),
        )
        following = store.get("traces", "testing-00000000-trace-0001")
        assert following is not None
        assert following["screenshot_before"] == trace["screenshot"]
    assert len(artifacts) == 6
    assert all((tmp_path / path).is_file() for path in artifacts)
    assert len(tuple((tmp_path / "blobs" / "screenshots").iterdir())) == 3


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
