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
class ActionChannel:
    """One stable TraceNet output channel and its browser behavior."""

    name: str
    kind: ActionKind
    weight: float
    keywords: tuple[str, ...] = ()
    text_strategy: str | None = None
    fixed_text: str | None = None
    direction: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("action channel names cannot be empty")
        if self.weight <= 0:
            raise ValueError("action channel weights must be positive")
        if self.kind is ActionKind.TYPE and not (self.text_strategy or self.fixed_text):
            raise ValueError("typing channels require a text strategy or fixed text")
        if self.kind is ActionKind.SCROLL and self.direction not in {"up", "down"}:
            raise ValueError("scroll channels require an up/down direction")


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
    channel: str | None = None

    def __post_init__(self) -> None:
        if self.kind is ActionKind.TYPE and self.text is None:
            raise ValueError("type actions require text")
        if self.kind is ActionKind.SCROLL and self.direction not in {"up", "down"}:
            raise ValueError("scroll actions require an up/down direction")

    @property
    def channel_name(self) -> str:
        return (
            self.channel
            or {
                ActionKind.CLICK: "click",
                ActionKind.DOUBLE_CLICK: "doubleClick",
                ActionKind.RIGHT_CLICK: "rightClick",
                ActionKind.CLEAR: "clear",
                ActionKind.TYPE: "typeRandomLetters",
                ActionKind.SCROLL: "scrollUp" if self.direction == "up" else "scrollDown",
            }[self.kind]
        )
