from dataclasses import dataclass, field

import pytest

from kwola.hooks import (
    HookExecutionError,
    HookRegistry,
    LifecycleEvent,
    LifecycleEventName,
)


@dataclass
class RecordingHook:
    name: str
    order: int
    fatal: bool
    output: list[str]
    fails: bool = False
    events: frozenset[LifecycleEventName] = field(
        default_factory=lambda: frozenset({LifecycleEventName.RUN_STARTED})
    )

    def handle(self, event: LifecycleEvent) -> None:
        self.output.append(f"{self.name}:{event.name}")
        if self.fails:
            raise ValueError("broken")

    def close(self) -> None:
        self.output.append(f"close:{self.name}")


def event() -> LifecycleEvent:
    return LifecycleEvent(LifecycleEventName.RUN_STARTED, 1.0, "run-1")


def test_hooks_run_in_order_and_close_in_reverse() -> None:
    output: list[str] = []
    registry = HookRegistry(
        (
            RecordingHook("later", 20, True, output),
            RecordingHook("first", 10, True, output),
        )
    )

    assert registry.dispatch(event()) == ()
    assert registry.close() == ()
    assert output == [
        "first:run_started",
        "later:run_started",
        "close:later",
        "close:first",
    ]


def test_best_effort_failure_is_reported_and_dispatch_continues() -> None:
    output: list[str] = []
    registry = HookRegistry(
        (
            RecordingHook("best-effort", 1, False, output, fails=True),
            RecordingHook("next", 2, True, output),
        )
    )

    failures = registry.dispatch(event())

    assert failures[0].hook == "best-effort"
    assert failures[0].event is LifecycleEventName.RUN_STARTED
    assert output[-1] == "next:run_started"


def test_fatal_failure_identifies_hook_and_event() -> None:
    registry = HookRegistry((RecordingHook("fatal-hook", 1, True, [], fails=True),))

    with pytest.raises(HookExecutionError, match=r"fatal-hook.*run_started"):
        registry.dispatch(event())


def test_duplicate_hook_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        HookRegistry(
            (
                RecordingHook("same", 1, True, []),
                RecordingHook("same", 2, True, []),
            )
        )
