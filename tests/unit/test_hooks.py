from dataclasses import dataclass, field

import pytest

from kwola.hooks import (
    HookExecutionError,
    HookRegistry,
    LifecycleEvent,
    LifecycleEventName,
)
from kwola.orchestration.lifecycle import RunnerLifecycle
from kwola.orchestration.results import RunnerResult


@dataclass
class RecordingHook:
    name: str
    order: int
    fatal: bool
    output: list[str]
    fails: bool = False
    close_fails: bool = False
    events: frozenset[LifecycleEventName] = field(
        default_factory=lambda: frozenset({LifecycleEventName.RUN_STARTED})
    )

    def handle(self, event: LifecycleEvent) -> None:
        self.output.append(f"{self.name}:{event.name}")
        if self.fails:
            raise ValueError("broken")

    def close(self) -> None:
        self.output.append(f"close:{self.name}")
        if self.close_fails:
            raise RuntimeError("close broken")


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


def test_runner_warnings_are_serialized_logged_and_closed_in_reverse(
    caplog: pytest.LogCaptureFixture,
) -> None:
    output: list[str] = []
    registry = HookRegistry(
        (
            RecordingHook("first", 1, False, output, fails=True),
            RecordingHook("second", 2, False, output, close_fails=True),
        )
    )
    lifecycle = RunnerLifecycle(registry, "run-1", lambda: 1.0)

    lifecycle.dispatch(LifecycleEventName.RUN_STARTED)
    lifecycle.finish(None)
    result = RunnerResult(
        status="completed",
        step_id="step-1",
        duration_seconds=1,
        warnings=lifecycle.warnings,
    )

    payload = result.model_dump(mode="json")
    assert [warning["hook"] for warning in payload["warnings"]] == ["first", "second"]
    assert payload["warnings"][0]["event"] == "run_started"
    assert payload["warnings"][1]["event"] == "shutdown"
    assert output[-2:] == ["close:second", "close:first"]
    assert "runner hook first failed" in caplog.text


def test_cleanup_hook_failure_does_not_hide_primary_exception() -> None:
    registry = HookRegistry(
        (
            RecordingHook(
                "finish",
                1,
                True,
                [],
                fails=True,
                events=frozenset({LifecycleEventName.RUN_FINISHED}),
            ),
        )
    )
    lifecycle = RunnerLifecycle(registry, "run-1", lambda: 1.0)
    primary = ValueError("primary")

    lifecycle.finish(primary)

    assert lifecycle.warnings[0].hook == "finish"
