"""Mitmproxy addon for rewriting, resource capture, and network telemetry."""

from collections.abc import Mapping

from mitmproxy import http

from .resources import ResourceRegistry
from .rewriting import HtmlRewriter, JavaScriptRewriter, RewriteError
from .telemetry import NetworkEntry, TelemetryBuffer


class InstrumentationAddon:
    def __init__(
        self,
        telemetry: TelemetryBuffer,
        resources: ResourceRegistry,
        *,
        rewrite_html: bool,
        rewrite_javascript: bool,
        capture_resources: bool,
    ) -> None:
        self._telemetry = telemetry
        self._resources = resources
        self._rewrite_html = rewrite_html
        self._rewrite_javascript = rewrite_javascript
        self._capture_resources = capture_resources
        self._html = HtmlRewriter()
        self._javascript = JavaScriptRewriter()

    def requestheaders(self, flow: http.HTTPFlow) -> None:
        flow.request.headers["Accept-Encoding"] = "identity"
        flow.request.headers["X-Kwola"] = "true"
        user_agent = flow.request.headers.get("User-Agent", "")
        if "Kwola" not in user_agent:
            flow.request.headers["User-Agent"] = f"{user_agent} Kwola".strip()

    def response(self, flow: http.HTTPFlow) -> None:
        response = flow.response
        if response is None:
            return
        content_type = response.headers.get("Content-Type", "")
        original = bytes(response.content or b"")
        delivered, rewrite_kind = self._rewrite(flow.request.url, content_type, original)
        if delivered != original:
            response.content = delivered
        self._record_network(flow, response.status_code)
        if self._capture_resources and original:
            self._resources.capture(
                url=flow.request.url,
                status=response.status_code,
                content_type=content_type,
                headers=_headers(response.headers),
                original=original,
                delivered=delivered,
                rewrite_kind=rewrite_kind,
            )

    def error(self, flow: http.HTTPFlow) -> None:
        failure = str(flow.error) if flow.error is not None else "unknown proxy error"
        self._telemetry.record_network(
            NetworkEntry(flow.request.method, flow.request.url, 0, failure)
        )

    def done(self) -> None:
        self._javascript.close()

    def _rewrite(self, url: str, content_type: str, source: bytes) -> tuple[bytes, str | None]:
        media_type = content_type.partition(";")[0].strip().lower()
        try:
            if self._rewrite_javascript and _is_javascript(url, media_type):
                rewritten = self._javascript.rewrite(url, source)
                return rewritten, "javascript" if rewritten != source else None
            if self._rewrite_html and media_type in {"text/html", "application/xhtml+xml"}:
                rewritten = self._html.rewrite(source)
                return rewritten, "html" if rewritten != source else None
        except (RewriteError, UnicodeError):
            return source, None
        return source, None

    def _record_network(self, flow: http.HTTPFlow, status: int) -> None:
        self._telemetry.record_network(NetworkEntry(flow.request.method, flow.request.url, status))


def _is_javascript(url: str, media_type: str) -> bool:
    javascript_types = {
        "application/ecmascript",
        "application/javascript",
        "application/x-javascript",
        "text/ecmascript",
        "text/javascript",
    }
    return media_type in javascript_types or url.partition("?")[0].endswith((".js", ".mjs"))


def _headers(headers: Mapping[str, str]) -> dict[str, str]:
    excluded = {"set-cookie", "authorization", "proxy-authorization"}
    return {key: value for key, value in headers.items() if key.lower() not in excluded}
