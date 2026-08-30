"""Training sample and batch descriptions without framework coupling."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Sample:
    id: str
    session_id: str
    trace_ids: tuple[str, ...]
    tensors: tuple[tuple[str, Any], ...]
    cache_version: int


@dataclass(frozen=True, slots=True)
class Batch:
    sample_ids: tuple[str, ...]
    tensors: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.sample_ids:
            raise ValueError("batches cannot be empty")
