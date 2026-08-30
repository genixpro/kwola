"""Aggregate diagnostic loss over every primary TraceNet head."""

import torch
from torch import Tensor

from kwola.config.models import LossConfig


def aggregate_loss(outputs: dict[str, Tensor], config: LossConfig) -> Tensor:
    action_probabilities = outputs["actionProbabilities"].clamp_min(1e-12)
    action_loss = -torch.log(action_probabilities.flatten(1).max(dim=1).values).mean()
    present = outputs["presentRewards"].square().mean()
    discounted = outputs["discountFutureRewards"].square().mean()
    state = outputs["stateValues"].square().mean()
    advantage = outputs["advantage"].square().mean()
    return (
        action_loss * config.action_probability
        + present * config.present_reward
        + discounted * config.discounted_future_reward
        + state * config.state_value
        + advantage * config.advantage
    )
