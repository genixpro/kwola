"""Collection of Babel-installed branch counters from a browser page."""

import hashlib
from dataclasses import dataclass
from typing import Any

from playwright.sync_api import Page

_SNAPSHOT_SCRIPT = """
() => {
  if (!window.kwolaCounters) return null;
  const result = {};
  for (const [name, values] of Object.entries(window.kwolaCounters)) {
    result[name] = Array.from(values);
    values.fill(0);
  }
  return result;
}
"""


@dataclass(frozen=True, slots=True)
class BranchTraceSnapshot:
    available: bool
    symbols: tuple[int, ...]


class BranchTraceCollector:
    def __init__(self, timeout_seconds: float = 30.0, restore_timeout_seconds: float = 15.0):
        self._timeout_ms = timeout_seconds * 1000
        self._restore_timeout_ms = restore_timeout_seconds * 1000

    def collect(self, page: Page) -> BranchTraceSnapshot:
        page.set_default_timeout(self._timeout_ms)
        try:
            raw = page.evaluate(_SNAPSHOT_SCRIPT)
        finally:
            page.set_default_timeout(self._restore_timeout_ms)
        if not isinstance(raw, dict):
            return BranchTraceSnapshot(False, ())
        return BranchTraceSnapshot(True, tuple(sorted(_executed_symbols(raw))))


def _executed_symbols(counters: dict[str, Any]) -> set[int]:
    symbols: set[int] = set()
    for resource, values in counters.items():
        if not isinstance(resource, str) or not isinstance(values, list):
            continue
        for index, count in enumerate(values):
            if isinstance(count, int) and count > 0:
                digest = hashlib.blake2b(f"{resource}:{index}".encode(), digest_size=8)
                symbols.add(int.from_bytes(digest.digest()))
    return symbols
