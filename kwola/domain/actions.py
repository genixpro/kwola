"""Browser-action value objects."""

from dataclasses import dataclass
from enum import StrEnum


class BrowserKind(StrEnum):
    CHROMIUM = "chromium"
    FIREFOX = "firefox"


class ActionKind(StrEnum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    CLEAR = "clear"
    TYPE = "type"
    SCROLL = "scroll"


@dataclass(frozen=True, slots=True)
class ActionTarget:
    left: int
    top: int
    right: int
    bottom: int
    element_type: str
    keywords: str = ""
    can_click: bool = False
    can_right_click: bool = False
    can_type: bool = False
    can_scroll: bool = False
    can_scroll_up: bool = False
    can_scroll_down: bool = False
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.left > self.right or self.top > self.bottom:
            raise ValueError("action target bounds are inverted")

    @property
    def center(self) -> tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    def attribute(self, name: str) -> str | None:
        return dict(self.attributes).get(name)


@dataclass(frozen=True, slots=True)
class ActionMap:
    targets: tuple[ActionTarget, ...]
    viewport_width: int
    viewport_height: int
    asset_version: str

    def __post_init__(self) -> None:
        if self.viewport_width < 1 or self.viewport_height < 1:
            raise ValueError("action-map viewport must be positive")


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    x: int
    y: int
    text: str | None = None
    direction: str | None = None
    source: str = "policy"

    def __post_init__(self) -> None:
        if self.kind is ActionKind.TYPE and self.text is None:
            raise ValueError("type actions require text")
        if self.kind is ActionKind.SCROLL and self.direction not in {"up", "down"}:
            raise ValueError("scroll actions require an up/down direction")
