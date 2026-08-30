import math

import pytest

from kwola.agent import (
    ExplorationProbabilities,
    ExplorationSchedule,
    RewardCalculator,
    RewardSignals,
)
from kwola.config import profile_config


def signals(**overrides: bool | float | None) -> RewardSignals:
    values: dict[str, bool | float | None] = {
        "branch_trace_available": True,
        "code_executed": True,
        "new_branches_executed": True,
        "network_traffic": True,
        "new_network_traffic": True,
        "screenshot_changed": True,
        "screenshot_new": True,
        "url_changed": True,
        "url_new": True,
        "log_output": True,
    }
    values.update(overrides)
    return RewardSignals(**values)  # type: ignore[arg-type]


def test_present_rewards_match_legacy_characterization() -> None:
    config = profile_config("testing", "https://example.com", 1).policy.rewards
    calculator = RewardCalculator(config)

    assert calculator.present(signals()) == pytest.approx(0.459)
    assert calculator.present(
        signals(
            branch_trace_available=True,
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
    ) == pytest.approx(-0.02)
    assert calculator.present(signals(branch_trace_available=False)) == pytest.approx(0.158)


def test_missing_branch_instrumentation_has_neither_code_bonus_nor_penalty() -> None:
    config = profile_config("testing", "https://example.com", 1).policy.rewards
    calculator = RewardCalculator(config)
    unavailable = signals(
        branch_trace_available=False,
        code_executed=False,
        new_branches_executed=False,
    )
    productive = signals(
        branch_trace_available=True,
        code_executed=True,
        new_branches_executed=False,
    )
    unproductive = signals(
        branch_trace_available=True,
        code_executed=False,
        new_branches_executed=False,
    )

    assert calculator.present(productive) - calculator.present(unavailable) == pytest.approx(
        config.code_executed
    )
    assert calculator.present(unproductive) - calculator.present(unavailable) == pytest.approx(
        config.no_code_executed
    )


def test_discounted_future_rewards_exclude_the_current_frame() -> None:
    config = profile_config("testing", "https://example.com", 1).policy.rewards
    calculator = RewardCalculator(config)
    frames = (
        signals(),
        signals(branch_trace_available=False),
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

    assert (
        schedule.probability(
            action_index=0,
            session_index=0,
            session_count=1,
            test_step_index=0,
        ).random
        > 0
    )


def test_exploration_thresholds_reject_weighted_probability_above_total() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ExplorationProbabilities(random=0.2, weighted_random=0.3)
