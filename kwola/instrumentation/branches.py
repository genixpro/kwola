"""Collection of Babel-installed branch counters from a browser page."""

import hashlib
from typing import Any

from playwright.sync_api import Page

_SNAPSHOT_SCRIPT = """
() => {
  if (!window.kwolaCounters) return {};
  const result = {};
  for (const [name, values] of Object.entries(window.kwolaCounters)) {
    result[name] = Array.from(values);
    values.fill(0);
  }
  return result;
}
"""


class BranchTraceCollector:
    def collect(self, page: Page) -> tuple[int, ...]:
        raw = page.evaluate(_SNAPSHOT_SCRIPT)
        if not isinstance(raw, dict):
            return ()
        return tuple(sorted(_executed_symbols(raw)))


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
