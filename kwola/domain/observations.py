"""Observed browser state."""

from dataclasses import dataclass

from .actions import ActionMap


@dataclass(frozen=True, slots=True)
class Viewport:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("viewport dimensions must be positive")


@dataclass(frozen=True, slots=True)
class Observation:
    url: str
    screenshot: bytes
    viewport: Viewport
    action_map: ActionMap
    timestamp: float
    branch_symbols: tuple[int, ...] = ()
    network_symbols: tuple[int, ...] = ()
    console_messages: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
