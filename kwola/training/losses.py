"""Masked Double-DQN losses for TraceNet's decomposed value maps."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kwola.config.models import LossConfig, TrainingConfig

from .samples import TrainingBatch


@dataclass(frozen=True, slots=True)
class QLoss:
    total: Tensor
    present_reward: Tensor
    discounted_reward: Tensor
    trace_prediction: Tensor
    execution_feature: Tensor
    cursor_prediction: Tensor
    mean_selected_q: Tensor
    mean_bootstrap_target: Tensor
    mean_absolute_td_error: Tensor


def aggregate_loss(outputs: dict[str, Tensor], config: LossConfig) -> Tensor:
    """Finite smoke-test loss used only by diagnostics and benchmarks."""
    return (
        outputs["presentRewards"].square().mean() * config.present_reward
        + outputs["discountFutureRewards"].square().mean() * config.discounted_future_reward
    )


def q_learning_loss(
    model: nn.Module,
    target_model: nn.Module,
    batch: TrainingBatch,
    config: TrainingConfig,
    discount_rate: float,
    max_discounted_reward: float,
) -> QLoss:
    outputs = model(batch.request)
    future_targets = _double_dqn_targets(
        model,
        target_model,
        batch,
        discount_rate,
        max_discounted_reward,
    )
    present, future, selected_q, td_error = _value_losses(
        outputs, batch, future_targets, config.losses
    )
    auxiliary = _auxiliary_losses(outputs, batch, config.losses)
    return QLoss(
        total=present + future + sum(auxiliary),
        present_reward=present,
        discounted_reward=future,
        trace_prediction=auxiliary[0],
        execution_feature=auxiliary[1],
        cursor_prediction=auxiliary[2],
        mean_selected_q=selected_q.mean(),
        mean_bootstrap_target=future_targets.mean(),
        mean_absolute_td_error=td_error.abs().mean(),
    )


def _value_losses(
    outputs: dict[str, Tensor],
    batch: TrainingBatch,
    future_targets: Tensor,
    weights: LossConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    present_maps = outputs["presentRewards"]
    future_maps = outputs["discountFutureRewards"]
    indexes = torch.arange(len(batch.sample_ids), device=present_maps.device)
    selected = (indexes, batch.action_indexes)
    combo = _reward_masks(batch, present_maps) * batch.request.pixel_action_maps[selected]
    counts = combo.sum(dim=(1, 2)).clamp_min(1)

    present_prediction = present_maps[selected]
    future_prediction = future_maps[selected]
    present_target = batch.present_rewards[:, None, None].expand_as(present_prediction)
    future_target = future_targets[:, None, None].expand_as(future_prediction)
    present_per_pixel = nn.functional.smooth_l1_loss(
        present_prediction, present_target, reduction="none"
    )
    future_per_pixel = nn.functional.smooth_l1_loss(
        future_prediction, future_target, reduction="none"
    )
    present = ((present_per_pixel * combo).sum((1, 2)) / counts).mean()
    future = ((future_per_pixel * combo).sum((1, 2)) / counts).mean()

    selected_q = (present_prediction + future_prediction) * combo
    selected_q = selected_q.sum((1, 2)) / counts
    return_target = batch.present_rewards + future_targets
    td_error = selected_q - return_target
    return (
        present * weights.present_reward,
        future * weights.discounted_future_reward,
        selected_q,
        td_error,
    )


def _double_dqn_targets(
    model: nn.Module,
    target_model: nn.Module,
    batch: TrainingBatch,
    discount_rate: float,
    maximum: float,
) -> Tensor:
    model_was_training = model.training
    model.eval()
    target_model.eval()
    try:
        with torch.no_grad():
            online_values = model(batch.next_request)["actionValues"]
            target_values = target_model(batch.next_request)["actionValues"]
            best_actions = online_values.flatten(1).argmax(dim=1)
            selected = target_values.flatten(1).gather(1, best_actions[:, None])[:, 0]
            discounted = torch.clamp(selected * discount_rate, min=-maximum, max=maximum)
            return discounted * batch.next_state_valid.type_as(discounted)
    finally:
        if model_was_training:
            model.train()


def _reward_masks(batch: TrainingBatch, reference: Tensor) -> Tensor:
    if batch.reward_pixel_masks is not None:
        return batch.reward_pixel_masks.type_as(reference)
    masks = torch.zeros_like(reference[:, 0])
    indexes = torch.arange(len(batch.sample_ids), device=reference.device)
    masks[indexes, batch.action_y, batch.action_x] = 1
    return masks


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
