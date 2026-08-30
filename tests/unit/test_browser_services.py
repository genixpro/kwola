from typing import Any

import pytest

from kwola.browser import ActionMapExtractor, NavigationPolicy, PlaywrightBrowserAdapter
from kwola.config import profile_config
from kwola.domain.actions import BrowserKind
from kwola.instrumentation import TelemetryBuffer


class FakePage:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.script = ""

    def evaluate(self, script: str) -> dict[str, Any]:
        self.script = script
        return self.result


def target(**overrides: Any) -> dict[str, Any]:
    values = {
        "left": 1,
        "top": 2,
        "right": 10,
        "bottom": 20,
        "elementType": "button",
        "keywords": "submit login",
        "canClick": True,
        "canRightClick": False,
        "canType": False,
        "canScroll": False,
        "canScrollUp": False,
        "canScrollDown": False,
        "attributes": {"href": "", "id": "submit"},
    }
    values.update(overrides)
    return values


def test_navigation_policy_enforces_exact_normalized_origins() -> None:
    policy = NavigationPolicy("https://app.example.com/start")

    assert policy.allows("/same-site")
    assert policy.allows("https://APP.example.com:443/resource")
    assert not policy.allows("https://cdn.example.com/resource")
    assert not policy.allows("https://example.net/")
    assert not NavigationPolicy("http://127.0.0.1:3001/").allows("http://127.0.0.1:3003/")
    assert not NavigationPolicy("http://127.0.0.1:3001/").allows("http://127.0.0.2/")
    assert not NavigationPolicy("https://app.example.co.uk").allows("https://attacker.co.uk")
    assert not policy.allows("data:text/html,blocked")


def test_navigation_policy_supports_allowlists_and_disabled_containment() -> None:
    allowed = NavigationPolicy(
        "https://example.com", allowed_origins=("https://login.example.net",)
    )
    assert allowed.allows("https://login.example.net/start")
    assert not allowed.allows("https://login.example.net:444/start")
    assert NavigationPolicy("https://example.com", prevent_offsite=False).allows(
        "data:text/html,allowed"
    )


def test_navigation_policy_normalizes_idn_ipv6_and_default_ports() -> None:
    assert NavigationPolicy("https://BÜCHER.example").allows("https://xn--bcher-kva.example:443/")
    assert NavigationPolicy("http://[::1]").allows("http://[::1]:80/path")
    assert not NavigationPolicy("http://[::1]").allows("http://[::1]:81/path")


@pytest.mark.parametrize(
    "origin",
    (
        "ftp://example.com",
        "https://user@example.com",
        "https://example.com/path",
        "https://example.com?query=yes",
        "https://example.com#fragment",
        "data:text/html,invalid",
        "blob:https://example.com/id",
        "about:blank",
    ),
)
def test_navigation_policy_rejects_invalid_configured_origins(origin: str) -> None:
    with pytest.raises(ValueError):
        NavigationPolicy("https://example.com", allowed_origins=(origin,))


def test_action_extractor_filters_offsite_links_and_versions_asset() -> None:
    page = FakePage(
        {
            "width": 800,
            "height": 600,
            "error": None,
            "targets": [
                target(),
                target(elementType="a", attributes={"href": "https://outside.invalid"}),
            ],
        }
    )
    extractor = ActionMapExtractor(NavigationPolicy("https://example.com"))

    action_map = extractor.extract(page)  # type: ignore[arg-type]

    assert action_map.asset_version == "1"
    assert action_map.viewport_width == 800
    assert len(action_map.targets) == 1
    assert action_map.targets[0].attribute("id") == "submit"
    assert "window.kwolaEvents" in page.script


def test_browser_route_allows_cross_origin_subresources() -> None:
    class Request:
        method = "GET"
        url = "https://cdn.example.net/asset.js"

        @staticmethod
        def is_navigation_request() -> bool:
            return False

    class Route:
        request = Request()
        continued = False

        def continue_(self) -> None:
            self.continued = True

    config = profile_config("testing", "https://example.com", 1).browser
    adapter = PlaywrightBrowserAdapter(
        config,
        BrowserKind.CHROMIUM,
        config.viewports[0],
        TelemetryBuffer(),
        NavigationPolicy("https://example.com"),
    )
    route = Route()

    adapter._route_request(route)  # type: ignore[arg-type]

    assert route.continued

    unrestricted = PlaywrightBrowserAdapter(
        config,
        BrowserKind.CHROMIUM,
        config.viewports[0],
        TelemetryBuffer(),
        NavigationPolicy("https://example.com", prevent_offsite=False),
    )
    unrestricted_route = Route()
    unrestricted._route_request(unrestricted_route)  # type: ignore[arg-type]
    assert unrestricted_route.continued
