"""Thin Playwright lifecycle adapter."""

from typing import Self

from playwright.sync_api import (
    Browser,
    BrowserContext,
    ConsoleMessage,
    Page,
    Playwright,
    Response,
    sync_playwright,
)

from kwola.config.models import BrowserConfig, ViewportConfig
from kwola.domain.actions import BrowserKind
from kwola.instrumentation.telemetry import ConsoleEntry, NetworkEntry, TelemetryBuffer


class PlaywrightBrowserAdapter:
    def __init__(
        self,
        config: BrowserConfig,
        browser_kind: BrowserKind,
        viewport: ViewportConfig,
        telemetry: TelemetryBuffer,
        proxy_server: str | None = None,
    ) -> None:
        self._config = config
        self._browser_kind = browser_kind
        self._viewport = viewport
        self._telemetry = telemetry
        self._proxy_server = proxy_server
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("browser adapter has not been started")
        return self._page

    def start(self) -> None:
        if self._playwright is not None:
            raise RuntimeError("browser adapter is already started")
        self._playwright = sync_playwright().start()
        browser_type = getattr(self._playwright, self._browser_kind.value)
        proxy = {"server": self._proxy_server} if self._proxy_server else None
        self._browser = browser_type.launch(headless=self._config.headless, proxy=proxy)
        self._context = self._browser.new_context(
            viewport={"width": self._viewport.width, "height": self._viewport.height},
            ignore_https_errors=self._proxy_server is not None,
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(self._config.action_timeout_seconds * 1000)
        self._page.on("console", self._record_console)
        self._page.on("response", self._record_response)

    def navigate(self, target: str) -> None:
        self.page.goto(
            target,
            wait_until="domcontentloaded",
            timeout=self._config.page_load_timeout_seconds * 1000,
        )

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    def _record_console(self, message: ConsoleMessage) -> None:
        self._telemetry.record_console(ConsoleEntry(message.type, message.text, self.page.url))

    def _record_response(self, response: Response) -> None:
        request = response.request
        self._telemetry.record_network(
            NetworkEntry(str(request.method), str(response.url), int(response.status))
        )
