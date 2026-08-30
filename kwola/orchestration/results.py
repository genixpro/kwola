"""Validated public runner results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kwola.hooks import LifecycleEventName


class RunnerWarning(BaseModel):
    """A serialized hook failure that did not replace runner work."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    hook: str
    event: LifecycleEventName
    fatal: bool
    error_type: str
    message: str


class RunnerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["completed", "failed", "cancelled"]
    step_id: str
    duration_seconds: float = Field(ge=0)
    artifacts: tuple[str, ...] = ()
    metrics: dict[str, float | int] = Field(default_factory=dict)
    error: str | None = None
    warnings: tuple[RunnerWarning, ...] = ()
