"""Thin Playwright lifecycle adapter."""

from typing import Self
from urllib.parse import urljoin

from playwright.sync_api import (
    Browser,
    BrowserContext,
    ConsoleMessage,
    Page,
    Playwright,
    Response,
    Route,
    sync_playwright,
)
from playwright.sync_api import Error as PlaywrightError

from kwola.config.models import BrowserConfig, ViewportConfig
from kwola.domain.actions import BrowserKind
from kwola.instrumentation.telemetry import ConsoleEntry, NetworkEntry, TelemetryBuffer

from .navigation import NavigationPolicy


class PlaywrightBrowserAdapter:
    def __init__(
        self,
        config: BrowserConfig,
        browser_kind: BrowserKind,
        viewport: ViewportConfig,
        telemetry: TelemetryBuffer,
        navigation: NavigationPolicy,
        proxy_server: str | None = None,
        *,
        capture_console: bool = True,
        capture_network: bool = True,
    ) -> None:
        self._config = config
        self._browser_kind = browser_kind
        self._viewport = viewport
        self._telemetry = telemetry
        self._navigation = navigation
        self._proxy_server = proxy_server
        self._capture_console = capture_console
        self._capture_network = capture_network
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._blocked_popups = 0

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
        self._context.route("**/*", self._route_request)
        self._page = self._context.new_page()
        self._context.on("page", self._handle_new_page)
        self._page.set_default_timeout(self._config.action_timeout_seconds * 1000)
        if self._capture_console:
            self._page.on("console", self._record_console)
        if self._capture_network:
            self._page.on("response", self._record_response)

    def navigate(self, target: str) -> None:
        self._navigation.require_allowed(target)
        self.page.goto(
            target,
            wait_until="domcontentloaded",
            timeout=self._config.page_load_timeout_seconds * 1000,
        )
        self.ensure_allowed()

    def ensure_allowed(self) -> None:
        self._close_blocked_pages()
        self._navigation.require_allowed(self.page.url)

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
        self._blocked_popups = 0

    def _record_console(self, message: ConsoleMessage) -> None:
        self._telemetry.record_console(ConsoleEntry(message.type, message.text, self.page.url))

    def _record_response(self, response: Response) -> None:
        request = response.request
        self._telemetry.record_network(
            NetworkEntry(str(request.method), str(response.url), int(response.status))
        )

    def _route_request(self, route: Route) -> None:
        request = route.request
        if not self._navigation.prevent_offsite:
            route.continue_()
            return
        if request.is_navigation_request():
            if not self._navigation.allows(request.url):
                self._abort_navigation(route, request.url)
                return
            response = route.fetch(max_redirects=0)
            location = response.headers.get("location")
            if 300 <= response.status < 400 and location:
                destination = urljoin(request.url, location)
                if not self._navigation.allows(destination):
                    self._abort_navigation(route, destination)
                    return
            route.fulfill(response=response)
            return
        route.continue_()

    def _abort_navigation(self, route: Route, destination: str) -> None:
        request = route.request
        failure = "blocked off-origin document navigation"
        self._telemetry.record_network(NetworkEntry(request.method, destination, 0, failure))
        route.abort(error_code="blockedbyclient")
        try:
            frame = request.frame
            page = frame.page
            if page is not self._page and frame is page.main_frame:
                page.close()
        except PlaywrightError:
            if not self._close_uncommitted_popups():
                self._blocked_popups += 1

    def _close_uncommitted_popups(self) -> bool:
        if self._context is None:
            return False
        closed = False
        for page in self._context.pages:
            if page is not self._page and page.url in {"", "about:blank"}:
                page.close()
                closed = True
        return closed

    def _handle_new_page(self, page: Page) -> None:
        if self._blocked_popups or not self._navigation.allows(page.url):
            self._blocked_popups = max(0, self._blocked_popups - 1)
            page.close()

    def _close_blocked_pages(self) -> None:
        if self._context is None:
            return
        for page in self._context.pages:
            if page is not self._page and not self._navigation.allows(page.url):
                page.close()
