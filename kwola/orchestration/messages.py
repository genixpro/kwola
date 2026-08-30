"""Validated process control messages."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactReference(BoundaryModel):
    kind: Literal["blob", "record", "checkpoint"]
    location: str = Field(min_length=1)
    sha256: str | None = None


class WorkerCommand(BoundaryModel):
    command_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    artifacts: tuple[ArtifactReference, ...] = ()


class WorkerResult(BoundaryModel):
    command_id: str
    status: Literal["completed", "failed", "cancelled"]
    values: dict[str, Any] = Field(default_factory=dict)
    artifacts: tuple[ArtifactReference, ...] = ()
    error_type: str | None = None
    error_message: str | None = None
