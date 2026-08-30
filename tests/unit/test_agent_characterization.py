import math

import pytest

from kwola.agent import ExplorationSchedule, RewardCalculator, RewardSignals
from kwola.config import profile_config


def signals(**overrides: bool | float | None) -> RewardSignals:
    values: dict[str, bool | float | None] = {
        "action_succeeded": True,
        "code_executed": True,
        "new_branches_executed": True,
        "network_traffic": True,
        "new_network_traffic": True,
        "screenshot_changed": True,
        "screenshot_new": True,
        "url_changed": True,
        "url_new": True,
        "log_output": True,
        "code_prevalence_log_normalized_z_score": None,
    }
    values.update(overrides)
    return RewardSignals(**values)  # type: ignore[arg-type]


def test_present_rewards_match_legacy_characterization() -> None:
    config = profile_config("testing", "https://example.com", 1).policy.rewards
    calculator = RewardCalculator(config)

    assert calculator.present(signals()) == pytest.approx(0.459)
    assert calculator.present(
        signals(
            action_succeeded=False,
            code_executed=False,
            new_branches_executed=False,
            network_traffic=False,
            new_network_traffic=False,
            screenshot_changed=False,
            screenshot_new=False,
            url_changed=False,
            url_new=False,
            log_output=False,
        )
    ) == pytest.approx(-0.04)
    expected_prevalence_reward = ((2.718**-1.5) + 1) * 0.3 * 0.5
    assert calculator.present(
        signals(code_prevalence_log_normalized_z_score=1.5)
    ) == pytest.approx(0.159 + expected_prevalence_reward)


def test_discounted_future_rewards_exclude_the_current_frame() -> None:
    config = profile_config("testing", "https://example.com", 1).policy.rewards
    calculator = RewardCalculator(config)
    frames = (
        signals(),
        signals(action_succeeded=False),
        signals(code_executed=False),
    )
    present = calculator.present_many(frames)

    discounted = calculator.discounted_future(frames)

    assert discounted[2] == 0
    assert discounted[1] == pytest.approx(present[2] * config.discount_rate)
    assert discounted[0] == pytest.approx(
        present[1] * config.discount_rate + present[2] * config.discount_rate**2
    )


def test_exploration_probability_matches_legacy_equation() -> None:
    policy = profile_config("testing", "https://example.com", 1).policy
    schedule = ExplorationSchedule(policy.exploration, policy.testing_sequence_length)

    probability = schedule.probability(
        action_index=2,
        session_index=1,
        session_count=3,
        test_step_index=100,
    )
    action_rate = 0.4 + (1.0 - 0.4) * 0.5
    test_rate = 0.01 + (0.8 - 0.01) * (100 / 999)
    expected = math.sqrt(action_rate) * math.sqrt(1.0) * (1 - math.sqrt(test_rate))

    assert probability.random == pytest.approx(expected)
    assert probability.weighted_random == pytest.approx(expected)


def test_single_session_uses_midpoint_axis() -> None:
    policy = profile_config("testing", "https://example.com", 1).policy
    schedule = ExplorationSchedule(policy.exploration, policy.testing_sequence_length)

    assert schedule.probability(
        action_index=0,
        session_index=0,
        session_count=1,
        test_step_index=0,
    ).random > 0
