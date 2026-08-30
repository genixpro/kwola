"""Legacy-equivalent masked TraceNet loss calculation."""

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
    trace_prediction: Tensor
    execution_feature: Tensor
    cursor_prediction: Tensor


def aggregate_loss(outputs: dict[str, Tensor], config: LossConfig) -> Tensor:
    """Finite smoke-test loss used only by diagnostics and benchmarks."""
    probabilities = outputs["actionProbabilities"].clamp_min(1e-12)
    action = -torch.log(probabilities.flatten(1).max(dim=1).values).mean()
    return (
        outputs["presentRewards"].square().mean() * config.present_reward
        + outputs["discountFutureRewards"].square().mean() * config.discounted_future_reward
        + outputs["stateValues"].square().mean() * config.state_value
        + outputs["advantage"].square().mean() * config.advantage
        + action * config.action_probability
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
    discounted, valid = _discounted_target(
        target_model,
        batch,
        training_index,
        world_size,
        discount_rate,
        max_discounted_reward,
    )
    primary = _primary_losses(outputs, batch, discounted, valid, config)
    auxiliary = _auxiliary_losses(outputs, batch, config.losses)
    reward_total = _phase_total(primary, config, training_index, world_size)
    total = reward_total + sum(auxiliary)
    return BehaviorLoss(
        total,
        primary[0],
        primary[1],
        primary[2],
        primary[3],
        primary[4],
        auxiliary[0],
        auxiliary[1],
        auxiliary[2],
    )


def _primary_losses(
    outputs: dict[str, Tensor],
    batch: TrainingBatch,
    discounted: Tensor,
    valid: Tensor,
    config: TrainingConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    present_maps = outputs["presentRewards"]
    probability_maps = outputs["actionProbabilities"]
    indexes = torch.arange(len(batch.sample_ids), device=present_maps.device)
    selected = (indexes, batch.action_indexes)
    reward_masks = _reward_masks(batch, present_maps)
    combo = reward_masks * batch.request.pixel_action_maps[selected]
    present_prediction = present_maps[selected] * combo
    discounted_prediction = outputs["discountFutureRewards"][selected] * combo
    advantage_prediction = outputs["advantage"][selected] * combo
    present_target = combo * batch.present_rewards[:, None, None]
    discounted_target = combo * discounted[:, None, None]
    state = outputs["stateValues"]
    expected = batch.present_rewards + discounted
    advantage_target = combo * (expected[:, None, None] - state.detach()[:, :, None])
    counts = combo.sum(dim=(1, 2)).clamp_min(1)
    weights = config.losses
    present = ((present_target - present_prediction) * combo).square().sum((1, 2)) / counts
    future = ((discounted_target - discounted_prediction) * combo).square().sum((1, 2)) / counts
    advantage = ((advantage_target - advantage_prediction) * combo).square().sum((1, 2)) / counts
    present = (present * valid * weights.present_reward).mean()
    future = (future * valid * weights.discounted_future_reward).mean()
    advantage = (advantage * weights.advantage).mean()
    state_loss = ((state[:, 0] - expected.detach()).square() * weights.state_value).mean()
    action = _action_probability_loss(outputs, probability_maps, batch, config)
    return present, future, state_loss, advantage, action


def _reward_masks(batch: TrainingBatch, reference: Tensor) -> Tensor:
    if batch.reward_pixel_masks is not None:
        return batch.reward_pixel_masks.type_as(reference)
    masks = torch.zeros_like(reference[:, 0])
    indexes = torch.arange(len(batch.sample_ids), device=reference.device)
    masks[indexes, batch.action_y, batch.action_x] = 1
    return masks


def _action_probability_loss(
    outputs: dict[str, Tensor],
    probabilities: Tensor,
    batch: TrainingBatch,
    config: TrainingConfig,
) -> Tensor:
    advantage = outputs["advantage"].detach()
    _batch_size, action_count, height, width = advantage.shape
    best = advantage.flatten(1).argmax(dim=1)
    best_action = torch.div(best, height * width, rounding_mode="floor")
    pixel = torch.remainder(best, height * width)
    best_y = torch.div(pixel, width, rounding_mode="floor")
    best_x = torch.remainder(pixel, width)
    half = config.action_probability_square_size // 2
    x_values = torch.arange(width, device=advantage.device)[None, None, :]
    y_values = torch.arange(height, device=advantage.device)[None, :, None]
    square = (
        (x_values >= (best_x - half).clamp_min(0)[:, None, None])
        & (x_values < (best_x + half).clamp_max(width - 1)[:, None, None])
        & (y_values >= (best_y - half).clamp_min(0)[:, None, None])
        & (y_values < (best_y + half).clamp_max(height - 1)[:, None, None])
    )
    target = torch.zeros_like(probabilities)
    action_mask = torch.nn.functional.one_hot(best_action, num_classes=action_count)
    target = (
        square[:, None].type_as(target)
        * action_mask[:, :, None, None]
        * batch.request.pixel_action_maps
    )
    counts = target.sum((2, 3)).clamp_min(1)[:, :, None, None]
    target = target / counts
    difference = (target - probabilities) * batch.request.pixel_action_maps
    return difference.abs().sum((1, 2, 3)).mean() * config.losses.action_probability


def _auxiliary_losses(
    outputs: dict[str, Tensor], batch: TrainingBatch, weights: LossConfig
) -> tuple[Tensor, Tensor, Tensor]:
    zero = outputs["presentRewards"].sum() * 0
    trace = zero
    execution = zero
    cursor = zero
    if "predictedTraces" in outputs and "decayingFutureSymbolEmbedding" in outputs:
        trace = (
            outputs["predictedTraces"] - outputs["decayingFutureSymbolEmbedding"]
        ).square().mean() * weights.execution_trace
    if "predictedExecutionFeatures" in outputs and batch.execution_features is not None:
        execution = (
            outputs["predictedExecutionFeatures"] - batch.execution_features
        ).abs().mean() * weights.execution_feature
    if "predictedCursor" in outputs and batch.cursors is not None:
        cursor = (
            outputs["predictedCursor"] - batch.cursors
        ).abs().mean() * weights.cursor_prediction
    return trace, execution, cursor


def _phase_total(
    losses: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    config: TrainingConfig,
    training_index: int,
    world_size: int,
) -> Tensor:
    present, discounted, state, advantage, action = losses
    result = present
    result = result + discounted if training_index >= world_size else result + discounted * 0
    result = result + state if training_index >= 3 * world_size else result + state * 0
    result = result + advantage if training_index >= 4 * world_size else result + advantage * 0
    return result + action if training_index >= 5 * world_size else result + action * 0


def _discounted_target(
    target_model: nn.Module,
    batch: TrainingBatch,
    training_index: int,
    world_size: int,
    discount_rate: float,
    max_discounted_reward: float,
) -> tuple[Tensor, Tensor]:
    target_model.eval()
    with torch.no_grad():
        outputs = target_model(batch.next_request)
        total = outputs["presentRewards"]
        if training_index >= 2 * world_size:
            total = total + outputs["discountFutureRewards"]
        best = total.flatten(1).max(dim=1).values
        valid = batch.next_state_valid & (best >= batch.next_request.impossible_action_reward)
        return torch.clamp(best * valid * discount_rate, max=max_discounted_reward), valid
