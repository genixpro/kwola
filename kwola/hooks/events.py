"""Typed lifecycle event envelope."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class LifecycleEventName(StrEnum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    SESSION_STARTED = "session_started"
    SESSION_FINISHED = "session_finished"
    BEFORE_ACTION = "before_action"
    AFTER_ACTION = "after_action"
    TRACE_RECORDED = "trace_recorded"
    SAMPLE_PREPARED = "sample_prepared"
    TRAINING_ITERATION_FINISHED = "training_iteration_finished"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    name: LifecycleEventName
    occurred_at: float
    run_id: str
    subject_id: str | None = None
    payload: tuple[tuple[str, Any], ...] = ()

    def values(self) -> dict[str, Any]:
        return dict(self.payload)
