"""Seeded legacy-equivalent weighted-random browser action selection."""

import random
import string
from datetime import date

from kwola.domain.actions import Action, ActionChannel, ActionKind, ActionMap, ActionTarget

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
    def __init__(
        self,
        source: random.Random,
        channels: tuple[ActionChannel, ...],
        *,
        weighted: bool = True,
    ) -> None:
        self._random = source
        self._channels = channels
        self._weighted = weighted

    def select(self, action_map: ActionMap) -> Action:
        candidates = tuple(target for target in action_map.targets if self.allowed(target))
        if not candidates:
            return self._fallback(action_map)
        weights = tuple(
            ELEMENT_WEIGHTS.get(target.element_type, 0.5) if self._weighted else 1.0
            for target in candidates
        )
        target = self._random.choices(candidates, weights=weights, k=1)[0]
        allowed = self.allowed(target)
        channel_weights = tuple(
            self._channel_weight(channel, target) if self._weighted else 1.0 for channel in allowed
        )
        channel = self._random.choices(allowed, weights=channel_weights, k=1)[0]
        x = self._random.randint(max(0, target.left), max(0, target.right - 1))
        y = self._random.randint(max(0, target.top), max(0, target.bottom - 1))
        text = self.typing_text(channel) if channel.kind is ActionKind.TYPE else None
        return Action(
            channel.kind,
            x,
            y,
            text=text,
            direction=channel.direction,
            source="weighted_random" if self._weighted else "random",
            channel=channel.name,
        )

    def allowed(self, target: ActionTarget) -> tuple[ActionChannel, ...]:
        return tuple(channel for channel in self._channels if _allowed(channel, target))

    @staticmethod
    def _channel_weight(channel: ActionChannel, target: ActionTarget) -> float:
        keywords = target.keywords.lower()
        boost = 1.5 if any(word.lower() in keywords for word in channel.keywords) else 1.0
        return channel.weight * boost

    def typing_text(self, channel: ActionChannel) -> str:
        if channel.fixed_text is not None:
            return channel.fixed_text
        strategy = channel.text_strategy
        choices = {
            "letters": string.ascii_lowercase,
            "number": string.digits,
            "brackets": "{}[]()",
            "math": "*=+<>-",
            "symbol": "\"';:/?,!^&#@",
        }
        if strategy in choices:
            maximum = 20 if strategy == "letters" else 5 if strategy == "number" else 3
            minimum = 4 if strategy == "letters" else 1
            return self._random_string(choices[strategy], self._random.randint(minimum, maximum))
        if strategy == "email":
            return f"testing_{self._random_string(string.ascii_lowercase, 20)}@kwola.io"
        if strategy == "phone":
            return self._random_string(string.digits, 10)
        if strategy == "address":
            return f"{self._random.randint(1, 9999)} Test Street"
        if strategy == "paragraph":
            return " ".join(self._random_string(string.ascii_lowercase, 8) for _ in range(12))
        if strategy == "date":
            return date.fromordinal(self._random.randint(730120, 739617)).isoformat()
        if strategy == "credit_card":
            return self._random_string(string.digits, 16)
        if strategy == "url":
            return f"https://{self._random_string(string.ascii_lowercase, 12)}.example/"
        raise ValueError(f"unknown typing strategy: {strategy}")

    def _random_string(self, characters: str, length: int) -> str:
        return "".join(self._random.choice(characters) for _ in range(length))

    def _fallback(self, action_map: ActionMap) -> Action:
        channel = self._random.choice(self._channels)
        text = self.typing_text(channel) if channel.kind is ActionKind.TYPE else None
        return Action(
            channel.kind,
            self._random.randrange(action_map.viewport_width),
            self._random.randrange(action_map.viewport_height),
            text,
            channel.direction,
            "random_fallback",
            channel.name,
        )


def _allowed(channel: ActionChannel, target: ActionTarget) -> bool:
    if channel.kind in {ActionKind.CLICK, ActionKind.DOUBLE_CLICK}:
        return target.can_click
    if channel.kind is ActionKind.RIGHT_CLICK:
        return target.can_right_click
    if channel.kind is ActionKind.CLEAR:
        return target.can_type and bool(target.attribute("value"))
    if channel.kind is ActionKind.TYPE:
        if not target.can_type or bool(target.attribute("value")):
            return False
        password = target.attribute("type") == "password"
        return channel.name == "typePassword" if password else channel.name != "typePassword"
    if channel.direction == "up":
        return target.can_scroll_up
    if channel.direction == "down":
        return target.can_scroll_down
    return False
