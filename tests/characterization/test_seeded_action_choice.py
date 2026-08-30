import random

from kwola.agent import action_catalog
from kwola.agent.random_policy import RandomActionPolicy
from kwola.config import profile_config
from kwola.domain.actions import ActionMap, ActionTarget


def test_seeded_weighted_action_sequence_matches_characterization() -> None:
    config = profile_config("testing", "https://example.com", 7)
    policy = RandomActionPolicy(random.Random(1234), action_catalog(config.policy))
    action_map = ActionMap(
        (
            ActionTarget(0, 0, 100, 50, "button", can_click=True),
            ActionTarget(
                100,
                0,
                200,
                50,
                "input",
                can_click=True,
                can_type=True,
                attributes=(("value", ""),),
            ),
        ),
        200,
        50,
        "1",
    )

    actions = tuple(policy.select(action_map) for _ in range(4))

    assert tuple(action.channel for action in actions) == (
        "typeCustom0",
        "typeRandomEmail",
        "typeNumber",
        "click",
    )
    assert tuple((action.x, action.y) for action in actions) == (
        (100, 5),
        (174, 2),
        (164, 31),
        (186, 5),
    )
    assert actions[1].text == "testing_vwcdylhaazalutptoecf@kwola.io"
