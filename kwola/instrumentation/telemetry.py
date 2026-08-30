"""Thread-safe, drainable browser telemetry collector."""

import hashlib
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

    def network_symbols(self) -> tuple[int, ...]:
        with self._lock:
            urls = {entry.url for entry in self._network if entry.status > 0}
        return tuple(sorted(_network_symbol(url) for url in urls))


def _network_symbol(url: str) -> int:
    digest = hashlib.blake2b(url.encode(), digest_size=8)
    return int.from_bytes(digest.digest())
