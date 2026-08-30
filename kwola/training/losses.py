"""Masked Double-DQN and conservative offline-Q losses for TraceNet value maps."""

import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kwola.agent.tiled_inference import evaluate_tiled_batch, tile_regions
from kwola.config.models import LossConfig, TrainingConfig

from .samples import TrainingBatch


@dataclass(frozen=True, slots=True)
class QLoss:
    total: Tensor
    present_reward: Tensor
    discounted_reward: Tensor
    conservative_q: Tensor
    trace_prediction: Tensor
    execution_feature: Tensor
    cursor_prediction: Tensor
    mean_selected_q: Tensor
    mean_bootstrap_target: Tensor
    mean_absolute_td_error: Tensor
    raw_trace_prediction: Tensor
    raw_execution_feature: Tensor
    raw_cursor_prediction: Tensor
    execution_feature_accuracy: Tensor
    execution_feature_f1: Tensor
    cursor_accuracy: Tensor
    target_evaluation_seconds: float
    mean_next_tiles: float


@dataclass(frozen=True, slots=True)
class AuxiliaryLosses:
    trace: Tensor
    execution: Tensor
    cursor: Tensor
    raw_trace: Tensor
    raw_execution: Tensor
    raw_cursor: Tensor
    execution_accuracy: Tensor
    execution_f1: Tensor
    cursor_accuracy: Tensor


def aggregate_loss(outputs: dict[str, Tensor], config: LossConfig) -> Tensor:
    """Finite smoke-test loss used only by diagnostics and benchmarks."""
    reward = (
        outputs["presentRewards"].square().mean() * config.present_reward
        + outputs["discountFutureRewards"].square().mean() * config.discounted_future_reward
    )
    auxiliary = sum(
        outputs[name].square().mean()
        for name in ("predictedTraces", "predictedExecutionFeatures", "predictedCursor")
        if name in outputs
    )
    return reward + auxiliary


def q_learning_loss(
    model: nn.Module,
    target_model: nn.Module,
    batch: TrainingBatch,
    config: TrainingConfig,
    discount_rate: float,
    max_discounted_reward: float,
) -> QLoss:
    outputs = model(batch.request)
    target_started = time.perf_counter()
    future_targets = _double_dqn_targets(
        model,
        target_model,
        batch,
        discount_rate,
        max_discounted_reward,
        (config.crop_width, config.crop_height),
    )
    target_evaluation_seconds = time.perf_counter() - target_started
    mean_next_tiles = 1.0
    if batch.next_requests:
        tile_counts = [
            len(
                tile_regions(
                    request.backbone.image.shape[-1],
                    request.backbone.image.shape[-2],
                    (config.crop_width, config.crop_height),
                )
            )
            for request in batch.next_requests
        ]
        mean_next_tiles = sum(tile_counts) / len(tile_counts)
    present, future, selected_q, td_error = _value_losses(
        outputs, batch, future_targets, config.losses
    )
    conservative = _conservative_q_loss(outputs, batch, selected_q, config.losses)
    future_symbol_target = _future_symbol_target(target_model, batch)
    auxiliary = _auxiliary_losses(outputs, batch, config.losses, future_symbol_target)
    return QLoss(
        total=(
            present
            + future
            + conservative
            + auxiliary.trace
            + auxiliary.execution
            + auxiliary.cursor
        ),
        present_reward=present,
        discounted_reward=future,
        conservative_q=conservative,
        trace_prediction=auxiliary.trace,
        execution_feature=auxiliary.execution,
        cursor_prediction=auxiliary.cursor,
        mean_selected_q=selected_q.mean(),
        mean_bootstrap_target=future_targets.mean(),
        mean_absolute_td_error=td_error.abs().mean(),
        raw_trace_prediction=auxiliary.raw_trace,
        raw_execution_feature=auxiliary.raw_execution,
        raw_cursor_prediction=auxiliary.raw_cursor,
        execution_feature_accuracy=auxiliary.execution_accuracy,
        execution_feature_f1=auxiliary.execution_f1,
        cursor_accuracy=auxiliary.cursor_accuracy,
        target_evaluation_seconds=target_evaluation_seconds,
        mean_next_tiles=mean_next_tiles,
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


def _conservative_q_loss(
    outputs: dict[str, Tensor],
    batch: TrainingBatch,
    selected_q: Tensor,
    weights: LossConfig,
) -> Tensor:
    """Keep unsupported valid actions below the demonstrated action value."""
    values = outputs["actionValues"]
    valid = batch.request.pixel_action_maps > 0
    zero = values.masked_select(valid).sum() * 0
    if weights.conservative_q == 0:
        return zero

    indexes = torch.arange(len(batch.sample_ids), device=values.device)
    demonstrated = torch.zeros_like(valid)
    demonstrated_region = (
        _reward_masks(batch, values)
        * batch.request.pixel_action_maps[indexes, batch.action_indexes]
    ) > 0
    demonstrated[indexes, batch.action_indexes] = demonstrated_region
    alternatives = valid & ~demonstrated
    has_alternative = alternatives.flatten(1).any(dim=1)
    alternative_max = values.masked_fill(~alternatives, torch.finfo(values.dtype).min)
    alternative_max = alternative_max.flatten(1).amax(dim=1)
    penalties = nn.functional.relu(
        alternative_max - selected_q.detach() + weights.conservative_q_margin
    )
    penalties = penalties * has_alternative.type_as(penalties)
    count = has_alternative.sum().clamp_min(1)
    return penalties.sum() / count * weights.conservative_q + zero


def _double_dqn_targets(
    model: nn.Module,
    target_model: nn.Module,
    batch: TrainingBatch,
    discount_rate: float,
    maximum: float,
    tile_size: tuple[int, int] | None = None,
) -> Tensor:
    model_was_training = model.training
    model.eval()
    target_model.eval()
    try:
        with torch.no_grad():
            if batch.next_requests:
                if tile_size is None:
                    raise ValueError("full next-state targets require an inference tile size")
                online_outputs = evaluate_tiled_batch(model, batch.next_requests, tile_size)
                target_outputs = evaluate_tiled_batch(target_model, batch.next_requests, tile_size)
                selected_values = []
                valid_actions = []
                for request, online, target in zip(
                    batch.next_requests, online_outputs, target_outputs, strict=True
                ):
                    online_values = online["actionValues"].flatten()
                    best_action = online_values.argmax()
                    selected_values.append(target["actionValues"].flatten()[best_action])
                    valid_actions.append(bool(request.pixel_action_maps.gt(0).any()))
                selected = torch.stack(selected_values)
                has_valid_action = torch.tensor(
                    valid_actions, dtype=torch.bool, device=selected.device
                )
            else:
                if batch.next_request is None:
                    raise ValueError("training batch has no next-state request")
                online_values = model(batch.next_request)["actionValues"]
                target_values = target_model(batch.next_request)["actionValues"]
                best_actions = online_values.flatten(1).argmax(dim=1)
                selected = target_values.flatten(1).gather(1, best_actions[:, None])[:, 0]
                has_valid_action = batch.next_request.pixel_action_maps.flatten(1).gt(0).any(dim=1)
            discounted = torch.clamp(selected * discount_rate, min=-maximum, max=maximum)
            valid_bootstrap = batch.next_state_valid & has_valid_action
            return discounted * valid_bootstrap.type_as(discounted)
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
    outputs: dict[str, Tensor],
    batch: TrainingBatch,
    weights: LossConfig,
    future_symbol_target: Tensor | None = None,
) -> AuxiliaryLosses:
    zero = outputs["presentRewards"].sum() * 0
    raw_trace = zero
    raw_execution = zero
    raw_cursor = zero
    execution_accuracy = zero.detach()
    execution_f1 = zero.detach()
    cursor_accuracy = zero.detach()
    if "predictedTraces" in outputs and future_symbol_target is not None:
        predictions = nn.functional.normalize(outputs["predictedTraces"], dim=1)
        raw_trace = (1 - nn.functional.cosine_similarity(predictions, future_symbol_target)).mean()
    if "predictedExecutionFeatures" in outputs and batch.execution_features is not None:
        logits = outputs["predictedExecutionFeatures"]
        targets = batch.execution_features.type_as(logits)
        raw_execution = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        predicted = logits >= 0
        expected = targets >= 0.5
        execution_accuracy = (predicted == expected).float().mean()
        true_positive = (predicted & expected).sum().float()
        false_positive = (predicted & ~expected).sum().float()
        false_negative = (~predicted & expected).sum().float()
        execution_f1 = (2 * true_positive) / (
            2 * true_positive + false_positive + false_negative
        ).clamp_min(1)
    if "predictedCursor" in outputs and batch.cursors is not None:
        logits = outputs["predictedCursor"]
        raw_cursor = nn.functional.cross_entropy(logits, batch.cursors.long())
        cursor_accuracy = (logits.argmax(dim=1) == batch.cursors).float().mean()
    return AuxiliaryLosses(
        trace=raw_trace * weights.execution_trace,
        execution=raw_execution * weights.execution_feature,
        cursor=raw_cursor * weights.cursor_prediction,
        raw_trace=raw_trace,
        raw_execution=raw_execution,
        raw_cursor=raw_cursor,
        execution_accuracy=execution_accuracy,
        execution_f1=execution_f1,
        cursor_accuracy=cursor_accuracy,
    )


def _future_symbol_target(target_model: nn.Module, batch: TrainingBatch) -> Tensor | None:
    if getattr(batch.request, "future_symbol_indexes", None) is None:
        return None
    target_model.eval()
    method = getattr(target_model, "future_symbol_embedding", None)
    if method is None:
        return None
    with torch.no_grad():
        target = method(batch.request)
        if not isinstance(target, Tensor):
            raise TypeError("future symbol embedding must be a tensor")
        return nn.functional.normalize(target, dim=1).detach()
