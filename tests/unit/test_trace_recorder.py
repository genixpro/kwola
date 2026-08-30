from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

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


def test_campaign_coverage_claim_rewards_exactly_one_concurrent_trace(
    tmp_path: Path,
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 2)
    config = load_config(tmp_path)
    before = _observation(b"same", "https://example.com", (), ())
    after = _observation(b"same", "https://example.com", (0x1234_5678_9ABC_DEF0,), ())
    artifacts: list[str] = []
    with LmdbRunStore(tmp_path / "run.lmdb", map_size=1024**2) as store:

        def record(index: int) -> float:
            recorder = TraceRecorder(tmp_path, config, store, artifacts)
            return recorder.record(
                f"testing-{index:08d}",
                0,
                Action(ActionKind.CLICK, 2, 2),
                before,
                after,
                NoveltyState.initial(before),
                "pointer",
                (None, None),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            rewards = tuple(executor.map(record, (0, 1)))

        traces = tuple(store.scan("traces"))
        coverage = tuple(store.scan("coverage_symbols"))

    assert len(coverage) == 1
    assert coverage[0][0] == "123456789abcdef0"
    assert sorted(bool(trace["new_branch_symbols"]) for _key, trace in traces) == [False, True]
    assert max(rewards) - min(rewards) == pytest.approx(config.policy.rewards.new_code_executed)


def test_initial_page_coverage_is_claimed_without_action_reward(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 3)
    config = load_config(tmp_path)
    initial = _observation(b"same", "https://example.com", (77,), ())
    artifacts: list[str] = []
    with LmdbRunStore(tmp_path / "run.lmdb", map_size=1024**2) as store:
        recorder = TraceRecorder(tmp_path, config, store, artifacts)
        assert recorder.claim_initial(initial) == (77,)
        reward = recorder.record(
            "testing-00000000",
            0,
            Action(ActionKind.CLICK, 2, 2),
            initial,
            initial,
            NoveltyState.initial(initial),
            "pointer",
            (None, None),
        )
        trace = store.get("traces", "testing-00000000-trace-0000")

    assert trace is not None
    assert trace["new_branch_symbols"] == []
    assert reward < config.policy.rewards.new_code_executed


def _observation(
    screenshot: bytes,
    url: str,
    branches: tuple[int, ...],
    network: tuple[int, ...],
    errors: tuple[str, ...] = (),
) -> Observation:
    target = ActionTarget(0, 0, 10, 10, "button", can_click=True)
    encoded_ok, encoded = cv2.imencode(".png", np.full((10, 10), screenshot[0], dtype=np.uint8))
    assert encoded_ok
    return Observation(
        url,
        encoded.tobytes(),
        Viewport(10, 10),
        ActionMap((target,), 10, 10, "1"),
        1.0,
        branches,
        network,
        ("message",),
        errors,
        True,
    )
