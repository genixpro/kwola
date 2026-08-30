import random
from typing import Any

from kwola.agent import RandomActionPolicy
from kwola.domain.actions import ActionKind, ActionMap, ActionTarget
from kwola.instrumentation.branches import BranchTraceCollector
from kwola.instrumentation.telemetry import ConsoleEntry, NetworkEntry, TelemetryBuffer


def test_random_policy_covers_capabilities_and_is_seeded() -> None:
    target = ActionTarget(
        1,
        2,
        20,
        30,
        "input",
        can_click=True,
        can_right_click=True,
        can_type=True,
        can_scroll=True,
        can_scroll_up=True,
        can_scroll_down=True,
    )
    action_map = ActionMap((target,), 100, 80, "1")
    first = RandomActionPolicy(random.Random(7), ("hello",)).select(action_map)
    second = RandomActionPolicy(random.Random(7), ("hello",)).select(action_map)
    assert first == second
    assert first.source == "weighted_random"
    assert 1 <= first.x < 20 and 2 <= first.y < 30


def test_random_policy_types_generated_text_and_falls_back() -> None:
    typing = ActionTarget(0, 0, 1, 1, "input", can_type=True)
    action_map = ActionMap((typing,), 10, 10, "1")
    actions = [RandomActionPolicy(random.Random(seed), ()).select(action_map) for seed in range(20)]
    typed = next(action for action in actions if action.kind is ActionKind.TYPE)
    assert typed.text is not None and 4 <= len(typed.text) <= 20
    empty = ActionMap((), 10, 20, "1")
    fallback = RandomActionPolicy(random.Random(1), ()).select(empty)
    assert fallback.kind is ActionKind.CLICK
    assert fallback.source == "random_fallback"


def test_random_policy_selects_each_single_capability() -> None:
    cases = (
        (ActionTarget(0, 0, 2, 2, "div", can_click=True), ActionKind.CLICK),
        (ActionTarget(0, 0, 2, 2, "div", can_right_click=True), ActionKind.RIGHT_CLICK),
        (ActionTarget(0, 0, 2, 2, "div", can_scroll_up=True), ActionKind.SCROLL),
        (ActionTarget(0, 0, 2, 2, "div", can_scroll_down=True), ActionKind.SCROLL),
    )
    for target, expected in cases:
        action = RandomActionPolicy(random.Random(2), ()).select(ActionMap((target,), 2, 2, "1"))
        assert action.kind is expected


def test_telemetry_snapshot_symbols_and_drain() -> None:
    telemetry = TelemetryBuffer()
    telemetry.record_console(ConsoleEntry("error", "broken", "https://example.test"))
    telemetry.record_network(NetworkEntry("GET", "https://example.test/a", 200))
    telemetry.record_network(NetworkEntry("GET", "https://example.test/a", 200))
    telemetry.record_network(NetworkEntry("GET", "https://example.test/b", 0, "failed"))
    console, network = telemetry.snapshot()
    assert len(console) == 1 and len(network) == 3
    assert len(telemetry.network_symbols()) == 1
    assert telemetry.drain() == (console, network)
    assert telemetry.snapshot() == ((), ())


class CounterPage:
    def evaluate(self, script: str) -> dict[str, Any]:
        assert "values.fill(0)" in script
        return {"app": [0, 2, 0, 1], "empty": [0], "invalid": "bad"}


def test_branch_trace_collector_returns_stable_executed_symbols() -> None:
    collector = BranchTraceCollector()
    first = collector.collect(CounterPage())  # type: ignore[arg-type]
    second = collector.collect(CounterPage())  # type: ignore[arg-type]
    assert first == second
    assert len(first) == 2
