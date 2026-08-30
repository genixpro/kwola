"""Narrow dependency-inversion interfaces."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, TypeVar

from .actions import Action, ActionMap
from .observations import Observation

T = TypeVar("T")


class BrowserController(Protocol):
    def observe(self) -> Observation: ...

    def discover_actions(self) -> ActionMap: ...

    def execute(self, action: Action) -> None: ...


class Instrumentation(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class RecordStore(Protocol):
    def put(self, collection: str, key: str, value: Mapping[str, Any]) -> None: ...

    def get(self, collection: str, key: str) -> Mapping[str, Any] | None: ...


class BlobStore(Protocol):
    def write(self, namespace: str, name: str, data: bytes) -> Path: ...

    def read(self, namespace: str, name: str) -> bytes: ...


class Clock(Protocol):
    def now(self) -> float: ...


class RandomSource(Protocol):
    def random(self) -> float: ...

    def choice(self, values: tuple[T, ...]) -> T: ...
