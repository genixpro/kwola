"""Stable legacy action-channel catalogue without legacy APIs."""

from kwola.config.models import PolicyConfig
from kwola.domain.actions import ActionChannel, ActionKind


def action_catalog(config: PolicyConfig) -> tuple[ActionChannel, ...]:
    actions = config.actions
    weights = actions.weights
    channels = [ActionChannel("click", ActionKind.CLICK, weights.click)]
    channels.extend(_fixed_channels(config))
    channels.extend(_generated_channels(config))
    if actions.double_click:
        channels.append(ActionChannel("doubleClick", ActionKind.DOUBLE_CLICK, weights.double_click))
    if actions.right_click:
        channels.append(ActionChannel("rightClick", ActionKind.RIGHT_CLICK, weights.right_click))
    if any(channel.kind is ActionKind.TYPE for channel in channels):
        channels.append(ActionChannel("clear", ActionKind.CLEAR, weights.clear))
    if actions.scrolling:
        channels.append(
            ActionChannel("scrollUp", ActionKind.SCROLL, weights.scrolling, direction="up")
        )
        channels.append(
            ActionChannel("scrollDown", ActionKind.SCROLL, weights.scrolling, direction="down")
        )
    return tuple(sorted(channels, key=lambda channel: channel.name))


def _fixed_channels(config: PolicyConfig) -> list[ActionChannel]:
    actions = config.actions
    weights = actions.weights
    fixed = (
        ("typeEmail", actions.email, weights.type_email, ("email", "user")),
        ("typePassword", actions.password, weights.type_password, ("pass",)),
        ("typeName", actions.name, weights.type_name, ("name",)),
        ("typeParagraph", actions.paragraph, weights.type_paragraph, ()),
    )
    channels = []
    for name, value, weight, keywords in fixed:
        if value:
            channels.append(
                ActionChannel(name, ActionKind.TYPE, weight, keywords, fixed_text=value)
            )
    for index, value in enumerate(config.custom_typing_strings):
        channels.append(
            ActionChannel(
                f"typeCustom{index}", ActionKind.TYPE, weights.custom_type, fixed_text=value
            )
        )
    return channels


def _generated_channels(config: PolicyConfig) -> list[ActionChannel]:
    actions = config.actions
    weights = actions.weights
    generated: tuple[tuple[str, bool, float, str, tuple[str, ...]], ...] = (
        ("typeRandomLetters", actions.random_letters, weights.random_letters, "letters", ()),
        (
            "typeRandomAddress",
            actions.random_address,
            weights.random_generated,
            "address",
            ("address", "street"),
        ),
        (
            "typeRandomEmail",
            actions.random_email,
            weights.random_generated,
            "email",
            ("email", "user"),
        ),
        (
            "typeRandomPhoneNumber",
            actions.random_phone_number,
            weights.random_generated,
            "phone",
            ("phone", "cell", "mobile"),
        ),
        (
            "typeRandomParagraph",
            actions.random_paragraph,
            weights.random_generated,
            "paragraph",
            (),
        ),
        (
            "typeRandomDateTime",
            actions.random_date_time,
            weights.random_generated,
            "date",
            ("date", "time"),
        ),
        (
            "typeRandomCreditCard",
            actions.random_credit_card,
            weights.random_generated,
            "credit_card",
            ("card", "credit"),
        ),
        ("typeRandomURL", actions.random_url, weights.random_generated, "url", ("url", "uri")),
        (
            "typeNumber",
            actions.random_number,
            weights.type_number,
            "number",
            ("num", "count", "int", "float", "amount"),
        ),
        ("typeBrackets", actions.random_brackets, weights.type_brackets, "brackets", ()),
        ("typeMath", actions.random_math, weights.type_math, "math", ()),
        ("typeOtherSymbol", actions.random_other_symbol, weights.type_other_symbol, "symbol", ()),
    )
    return [
        ActionChannel(name, ActionKind.TYPE, weight, keywords, strategy)
        for name, enabled, weight, strategy, keywords in generated
        if enabled
    ]
