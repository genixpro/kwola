"""One browser action and its measured outcome."""

from dataclasses import dataclass

from .actions import Action


@dataclass(frozen=True, slots=True)
class Trace:
    id: str
    action: Action
    started_at: float
    finished_at: float
    reward: float
    url_before: str
    url_after: str
    new_branch_symbols: tuple[int, ...] = ()
    new_network_symbols: tuple[int, ...] = ()
    errors: tuple[str, ...] = ()
    screenshot_blob: str | None = None

    def __post_init__(self) -> None:
        if self.finished_at < self.started_at:
            raise ValueError("trace finished before it started")
