"""Thread-safe, drainable browser telemetry collector."""

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, slots=True)
class ConsoleEntry:
    level: str
    message: str
    url: str


@dataclass(frozen=True, slots=True)
class NetworkEntry:
    method: str
    url: str
    status: int
    failure: str | None = None


class TelemetryBuffer:
    def __init__(self) -> None:
        self._lock = Lock()
        self._console: list[ConsoleEntry] = []
        self._network: list[NetworkEntry] = []

    def record_console(self, entry: ConsoleEntry) -> None:
        with self._lock:
            self._console.append(entry)

    def record_network(self, entry: NetworkEntry) -> None:
        with self._lock:
            self._network.append(entry)

    def snapshot(self) -> tuple[tuple[ConsoleEntry, ...], tuple[NetworkEntry, ...]]:
        with self._lock:
            return tuple(self._console), tuple(self._network)

    def drain(self) -> tuple[tuple[ConsoleEntry, ...], tuple[NetworkEntry, ...]]:
        with self._lock:
            console = tuple(self._console)
            network = tuple(self._network)
            self._console.clear()
            self._network.clear()
            return console, network
