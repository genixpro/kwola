"""Versioned DOM action-map extraction."""

from importlib import resources
from typing import Any

from playwright.sync_api import Page

from kwola.domain.actions import ActionMap, ActionTarget

from .navigation import NavigationPolicy

ACTION_MAP_ASSET_VERSION = "1"


class ActionMapExtractor:
    def __init__(self, navigation: NavigationPolicy) -> None:
        self._navigation = navigation
        self._script = (
            resources.files("kwola.browser")
            .joinpath("assets", "action_map_v1.js")
            .read_text(encoding="utf-8")
        )

    def extract(self, page: Page) -> ActionMap:
        raw: dict[str, Any] = page.evaluate(self._script)
        if raw.get("error"):
            raise RuntimeError(f"action-map JavaScript failed: {raw['error']}")
        targets = tuple(
            target for item in raw["targets"] if (target := self._target(item)) is not None
        )
        return ActionMap(
            targets=targets,
            viewport_width=int(raw["width"]),
            viewport_height=int(raw["height"]),
            asset_version=ACTION_MAP_ASSET_VERSION,
        )

    def _target(self, item: dict[str, Any]) -> ActionTarget | None:
        attributes = tuple(
            sorted((str(key), str(value)) for key, value in item["attributes"].items())
        )
        href = dict(attributes).get("href")
        if item["elementType"] == "a" and href and not self._navigation.allows(href):
            return None
        return ActionTarget(
            left=int(item["left"]),
            top=int(item["top"]),
            right=int(item["right"]),
            bottom=int(item["bottom"]),
            element_type=str(item["elementType"]),
            keywords=str(item["keywords"]),
            can_click=bool(item["canClick"]),
            can_right_click=bool(item["canRightClick"]),
            can_type=bool(item["canType"]),
            can_scroll=bool(item["canScroll"]),
            can_scroll_up=bool(item["canScrollUp"]),
            can_scroll_down=bool(item["canScrollDown"]),
            attributes=attributes,
        )
