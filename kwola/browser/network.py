"""Bounded page and network-idle waits."""

from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeout


class NetworkWaiter:
    def __init__(self, timeout_seconds: float) -> None:
        self._timeout_ms = timeout_seconds * 1000

    def wait(self, page: Page) -> bool:
        try:
            page.wait_for_load_state("domcontentloaded", timeout=self._timeout_ms)
            page.wait_for_load_state("networkidle", timeout=self._timeout_ms)
            return True
        except PlaywrightTimeout:
            return False
