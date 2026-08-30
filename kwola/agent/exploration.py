"""The historical three-axis exploration schedule as a pure service."""

import math
from dataclasses import dataclass

from kwola.config.models import ExplorationAxisConfig, ExplorationConfig


@dataclass(frozen=True, slots=True)
class ExplorationProbabilities:
    random: float
    weighted_random: float

    def __post_init__(self) -> None:
        if self.weighted_random > self.random:
            raise ValueError("weighted-random probability cannot exceed total random probability")


class ExplorationSchedule:
    def __init__(self, config: ExplorationConfig, sequence_length: int) -> None:
        if sequence_length < 2:
            raise ValueError("sequence length must be at least two")
        self._config = config
        self._sequence_length = sequence_length

    def probability(
        self,
        *,
        action_index: int,
        session_index: int,
        session_count: int,
        test_step_index: int,
    ) -> ExplorationProbabilities:
        action_portion = action_index / (self._sequence_length - 1)
        session_portion = session_index / (session_count - 1) if session_count > 1 else 0.5
        capped_test_step = min(self._config.max_test_step_index, test_step_index)
        test_step_portion = capped_test_step / (self._config.max_test_step_index - 1)
        action = self._interpolate(self._config.action, action_portion)
        session = self._interpolate(self._config.session, session_portion)
        test_step = self._interpolate(self._config.test_step, test_step_portion)
        return ExplorationProbabilities(
            random=math.sqrt(action[0]) * math.sqrt(session[0]) * (1 - math.sqrt(test_step[0])),
            weighted_random=(
                math.sqrt(action[1]) * math.sqrt(session[1]) * (1 - math.sqrt(test_step[1]))
            ),
        )

    @staticmethod
    def _interpolate(axis: ExplorationAxisConfig, portion: float) -> tuple[float, float]:
        random = axis.start_random_rate + (axis.end_random_rate - axis.start_random_rate) * portion
        weighted = (
            axis.start_weighted_random_rate
            + (axis.end_weighted_random_rate - axis.start_weighted_random_rate) * portion
        )
        return random, weighted
