from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from playwright.sync_api import Error as PlaywrightError

from kwola.browser import NavigationPolicy, OffsiteNavigationError, PlaywrightBrowserAdapter
from kwola.config import profile_config
from kwola.domain.actions import BrowserKind
from kwola.instrumentation import TelemetryBuffer

Response = tuple[int, str, bytes]


@contextmanager
def server(render: Callable[[str], Response]) -> Iterator[tuple[str, list[str]]]:
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            status, content_type, body = render(self.path)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if status == 302:
                self.send_header("Location", body.decode())
                body = b""
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_POST = do_GET

        def log_message(self, *_: object) -> None:
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", requests
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def foreign_response(path: str) -> Response:
    if path == "/asset.js":
        return 200, "text/javascript", b"window.crossOriginAssetLoaded = true;"
    return 200, "text/html", b"<title>foreign document</title>"


@pytest.mark.parametrize("browser_kind", [BrowserKind.CHROMIUM, BrowserKind.FIREFOX])
def test_every_cross_origin_document_navigation_is_blocked_but_subresources_load(
    browser_kind: BrowserKind,
) -> None:
    with server(foreign_response) as (foreign, foreign_requests):

        def primary_response(path: str) -> Response:
            if path == "/redirect":
                return 302, "text/plain", f"{foreign}/document".encode()
            body = f"""
                <a id="anchor" href="{foreign}/anchor">anchor</a>
                <form action="{foreign}/form"><button id="form">form</button></form>
                <button id="script" onclick="location.href='{foreign}/script'">script</button>
                <button id="popup" onclick="window.open('{foreign}/popup')">popup</button>
                <iframe src="{foreign}/iframe"></iframe>
            """.encode()
            return 200, "text/html", body

        with server(primary_response) as (primary, _primary_requests):
            telemetry = TelemetryBuffer()
            config = profile_config("testing", primary, 1).browser
            adapter = PlaywrightBrowserAdapter(
                config,
                browser_kind,
                config.viewports[0],
                telemetry,
                NavigationPolicy(primary),
            )
            with adapter:
                adapter.navigate(primary)
                if browser_kind is BrowserKind.FIREFOX:
                    adapter.page.evaluate("url => fetch(url, {mode: 'no-cors'})", f"{foreign}/api")
                    assert "/api" in foreign_requests
                assert "/iframe" not in foreign_requests

                for selector, destination in (
                    ("#anchor", "anchor"),
                    ("#form", "form"),
                    ("#script", "script"),
                    ("#popup", "popup"),
                ):
                    adapter.navigate(primary)
                    adapter.page.locator(selector).click(no_wait_after=True)
                    for _ in range(20):
                        _console, network = telemetry.snapshot()
                        if any(
                            entry.failure is not None
                            and entry.url.rsplit("/", maxsplit=1)[-1].startswith(destination)
                            for entry in network
                        ):
                            break
                        adapter.page.wait_for_timeout(50)
                    adapter.page.wait_for_timeout(100)
                    try:
                        adapter.ensure_allowed()
                    except OffsiteNavigationError:
                        assert adapter.page.url.startswith(("chrome-error:", "about:"))
                assert len(adapter.page.context.pages) == 1

                with pytest.raises((PlaywrightError, OffsiteNavigationError)):
                    adapter.navigate(f"{primary}/redirect")
                assert "/document" not in foreign_requests

            _console, network = telemetry.snapshot()
            blocked = [entry for entry in network if entry.failure is not None]
            assert {
                entry.url.rsplit("/", maxsplit=1)[-1].split("?", maxsplit=1)[0] for entry in blocked
            } >= {
                "anchor",
                "form",
                "iframe",
                "popup",
                "script",
                "document",
            }


@pytest.mark.parametrize("browser_kind", [BrowserKind.CHROMIUM, BrowserKind.FIREFOX])
def test_explicitly_allowed_redirect_origin_can_commit(browser_kind: BrowserKind) -> None:
    with server(foreign_response) as (foreign, _foreign_requests):
        with server(lambda _path: (302, "text/plain", f"{foreign}/document".encode())) as (
            primary,
            _primary_requests,
        ):
            config = profile_config("testing", primary, 1).browser
            with PlaywrightBrowserAdapter(
                config,
                browser_kind,
                config.viewports[0],
                TelemetryBuffer(),
                NavigationPolicy(primary, allowed_origins=(foreign,)),
            ) as adapter:
                adapter.navigate(primary)
                assert adapter.page.url == f"{foreign}/document"
