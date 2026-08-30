"""Bounded page and network-idle waits."""

from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeout


class NetworkWaiter:
    def __init__(self, timeout_seconds: float, idle_seconds: float = 0.0) -> None:
        self._timeout_ms = timeout_seconds * 1000
        self._idle_ms = idle_seconds * 1000

    def wait(self, page: Page) -> bool:
        try:
            page.wait_for_load_state("domcontentloaded", timeout=self._timeout_ms)
            page.wait_for_load_state("networkidle", timeout=self._timeout_ms)
            if self._idle_ms:
                page.wait_for_timeout(self._idle_ms)
            return True
        except PlaywrightTimeout:
            return False
