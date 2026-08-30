"""Screenshot capture service."""

from playwright.sync_api import Page


class ScreenshotService:
    def capture(self, page: Page) -> bytes:
        return page.screenshot(type="png", full_page=False)
