"""Mathematically unchanged present and discounted reward calculation."""

from dataclasses import dataclass

from kwola.config.models import RewardConfig


@dataclass(frozen=True, slots=True)
class RewardSignals:
    action_succeeded: bool
    code_executed: bool
    new_branches_executed: bool
    network_traffic: bool
    new_network_traffic: bool
    screenshot_changed: bool
    screenshot_new: bool
    url_changed: bool
    url_new: bool
    log_output: bool
    code_prevalence_log_normalized_z_score: float | None = None


class RewardCalculator:
    def __init__(self, config: RewardConfig) -> None:
        self._config = config

    def present(self, signals: RewardSignals) -> float:
        config = self._config
        reward = config.action_success if signals.action_succeeded else config.action_failure
        reward += config.code_executed if signals.code_executed else config.no_code_executed
        reward += self._new_code_reward(signals)
        reward += config.network_traffic if signals.network_traffic else config.no_network_traffic
        reward += (
            config.new_network_traffic
            if signals.new_network_traffic
            else config.no_new_network_traffic
        )
        reward += (
            config.screenshot_changed if signals.screenshot_changed else config.no_screenshot_change
        )
        reward += config.new_screenshot if signals.screenshot_new else config.no_new_screenshot
        reward += config.url_changed if signals.url_changed else config.no_url_change
        reward += config.new_url if signals.url_new else config.no_new_url
        reward += config.log_output if signals.log_output else config.no_log_output
        return reward

    def present_many(self, signals: tuple[RewardSignals, ...]) -> tuple[float, ...]:
        return tuple(self.present(item) for item in signals)

    def discounted_future(self, signals: tuple[RewardSignals, ...]) -> tuple[float, ...]:
        reversed_results: list[float] = []
        current = 0.0
        for reward in reversed(self.present_many(signals)):
            current *= self._config.discount_rate
            reversed_results.append(current)
            current += reward
        return tuple(reversed(reversed_results))

    def _new_code_reward(self, signals: RewardSignals) -> float:
        config = self._config
        if not signals.new_branches_executed:
            return config.no_new_code_executed
        prevalence = signals.code_prevalence_log_normalized_z_score
        if prevalence is None:
            return config.new_code_executed
        adjusted = float(config.code_prevalence_exponential_base**-prevalence)
        return (adjusted + 1) * config.new_code_executed * 0.5
