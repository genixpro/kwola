"""Execution-session aggregate."""

from dataclasses import dataclass
from enum import StrEnum

from .actions import BrowserKind
from .traces import Trace


class SessionStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    browser: BrowserKind
    target: str
    seed: int
    status: SessionStatus
    traces: tuple[Trace, ...] = ()
    started_at: float | None = None
    finished_at: float | None = None
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.status is SessionStatus.FAILED and not self.failure:
            raise ValueError("failed sessions require a failure message")
