"""Seeded weighted-random browser action selection."""

import random
import string

from kwola.domain.actions import Action, ActionKind, ActionMap, ActionTarget

ELEMENT_WEIGHTS = {
    "a": 0.2,
    "input": 2.0,
    "button": 0.7,
    "p": 0.7,
    "span": 0.7,
    "div": 0.7,
    "canvas": 1.0,
    "html": 2.5,
    "body": 2.5,
}


class RandomActionPolicy:
    def __init__(self, source: random.Random, typing_strings: tuple[str, ...]) -> None:
        self._random = source
        self._typing_strings = typing_strings

    def select(self, action_map: ActionMap) -> Action:
        candidates = tuple(target for target in action_map.targets if self._kinds(target))
        if not candidates:
            return self._fallback(action_map)
        weights = tuple(ELEMENT_WEIGHTS.get(target.element_type, 0.5) for target in candidates)
        target = self._random.choices(candidates, weights=weights, k=1)[0]
        kind, direction = self._random.choice(self._kinds(target))
        x = self._random.randint(max(0, target.left), max(0, target.right - 1))
        y = self._random.randint(max(0, target.top), max(0, target.bottom - 1))
        text = self._typing_text() if kind is ActionKind.TYPE else None
        return Action(kind, x, y, text=text, direction=direction, source="weighted_random")

    def _kinds(self, target: ActionTarget) -> tuple[tuple[ActionKind, str | None], ...]:
        kinds: list[tuple[ActionKind, str | None]] = []
        if target.can_click:
            kinds.append((ActionKind.CLICK, None))
        if target.can_right_click:
            kinds.append((ActionKind.RIGHT_CLICK, None))
        if target.can_type:
            kinds.extend(((ActionKind.TYPE, None), (ActionKind.CLEAR, None)))
        if target.can_scroll_up:
            kinds.append((ActionKind.SCROLL, "up"))
        if target.can_scroll_down:
            kinds.append((ActionKind.SCROLL, "down"))
        return tuple(kinds)

    def _typing_text(self) -> str:
        if self._typing_strings:
            return self._random.choice(self._typing_strings)
        length = self._random.randint(4, 20)
        return "".join(self._random.choice(string.ascii_lowercase) for _ in range(length))

    def _fallback(self, action_map: ActionMap) -> Action:
        return Action(
            ActionKind.CLICK,
            self._random.randrange(action_map.viewport_width),
            self._random.randrange(action_map.viewport_height),
            source="random_fallback",
        )
