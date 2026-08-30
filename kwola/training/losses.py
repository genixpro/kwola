"""Aggregate diagnostic loss over every primary TraceNet head."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kwola.config.models import LossConfig, TrainingConfig

from .samples import TrainingBatch


@dataclass(frozen=True, slots=True)
class BehaviorLoss:
    total: Tensor
    present_reward: Tensor
    discounted_reward: Tensor
    state_value: Tensor
    advantage: Tensor
    action_probability: Tensor


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


def behavior_loss(
    model: nn.Module,
    target_model: nn.Module,
    batch: TrainingBatch,
    config: TrainingConfig,
    training_index: int,
    world_size: int,
    discount_rate: float,
    max_discounted_reward: float,
) -> BehaviorLoss:
    outputs = model(batch.request)
    discounted_target = _discounted_target(
        target_model,
        batch,
        training_index,
        world_size,
        discount_rate,
        max_discounted_reward,
    )
    indexes = torch.arange(len(batch.sample_ids), device=batch.present_rewards.device)
    selected = (indexes, batch.action_indexes, batch.action_y, batch.action_x)
    present_prediction = outputs["presentRewards"][selected]
    discounted_prediction = outputs["discountFutureRewards"][selected]
    advantage_prediction = outputs["advantage"][selected]
    state_prediction = outputs["stateValues"][:, 0]
    expected_value = batch.present_rewards + discounted_target
    present = (present_prediction - batch.present_rewards).square().mean()
    discounted = (discounted_prediction - discounted_target).square().mean()
    state = (state_prediction - expected_value.detach()).square().mean()
    advantage_target = expected_value.detach() - state_prediction.detach()
    advantage = (advantage_prediction - advantage_target).square().mean()
    action = _action_probability_loss(outputs, batch)
    weighted = _phase_weighted(
        present, discounted, state, advantage, action, config, training_index, world_size
    )
    return BehaviorLoss(weighted, present, discounted, state, advantage, action)


def _discounted_target(
    target_model: nn.Module,
    batch: TrainingBatch,
    training_index: int,
    world_size: int,
    discount_rate: float,
    max_discounted_reward: float,
) -> Tensor:
    target_model.eval()
    with torch.no_grad():
        outputs = target_model(batch.next_request)
        total = outputs["presentRewards"]
        if training_index >= 2 * world_size:
            total = total + outputs["discountFutureRewards"]
        best = total.flatten(1).max(dim=1).values
        valid = batch.next_state_valid & (best >= batch.next_request.impossible_action_reward)
        discounted = best * valid * discount_rate
        return torch.clamp(discounted, max=max_discounted_reward)


def _action_probability_loss(outputs: dict[str, Tensor], batch: TrainingBatch) -> Tensor:
    probabilities = outputs["actionProbabilities"].clamp_min(1e-12)
    best = outputs["advantage"].detach().flatten(1).argmax(dim=1)
    selected = probabilities.flatten(1).gather(1, best[:, None])[:, 0]
    return -torch.log(selected).mean()


def _phase_weighted(
    present: Tensor,
    discounted: Tensor,
    state: Tensor,
    advantage: Tensor,
    action: Tensor,
    config: TrainingConfig,
    training_index: int,
    world_size: int,
) -> Tensor:
    weights = config.losses
    result = present * weights.present_reward
    if training_index >= world_size:
        result = result + discounted * weights.discounted_future_reward
    if training_index >= 3 * world_size:
        result = result + state * weights.state_value
    if training_index >= 4 * world_size:
        result = result + advantage * weights.advantage
    if training_index >= 5 * world_size:
        result = result + action * weights.action_probability
    return result
