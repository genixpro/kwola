from typing import Any

from kwola.browser import ActionMapExtractor, NavigationPolicy


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


def test_navigation_policy_handles_relative_urls_subdomains_and_ips() -> None:
    policy = NavigationPolicy("https://app.example.com/start")

    assert policy.allows("/same-site")
    assert policy.allows("https://cdn.example.com/resource")
    assert not policy.allows("https://example.net/")
    assert NavigationPolicy("http://127.0.0.1:3001/").allows("http://127.0.0.1:3003/")
    assert not NavigationPolicy("http://127.0.0.1:3001/").allows("http://127.0.0.2/")


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
