from dataclasses import replace

import torch
from torch import Tensor, nn

from kwola.agent.model_heads import TraceNetHeads
from kwola.config import profile_config
from kwola.training.batches import diagnostic_batch
from kwola.training.losses import (
    _auxiliary_losses,
    _conservative_q_loss,
    _double_dqn_targets,
    _future_symbol_target,
    _value_losses,
)
from kwola.training.samples import TrainingBatch


class FixedModel(nn.Module):
    def __init__(self, outputs: dict[str, Tensor]) -> None:
        super().__init__()
        self.outputs = outputs

    def forward(self, _request: object) -> dict[str, Tensor]:
        return self.outputs


class CoordinateQ(nn.Module):
    def forward(self, request: object) -> dict[str, Tensor]:
        coordinates = request.backbone.coordinate_image[:, :1]  # type: ignore[attr-defined]
        zero = torch.zeros_like(coordinates)
        return {
            "presentRewards": coordinates,
            "discountFutureRewards": zero,
            "actionValues": TraceNetHeads.action_values(
                coordinates,
                zero,
                request.pixel_action_maps,  # type: ignore[attr-defined]
                -10.0,
            ),
        }


class FutureTarget(nn.Module):
    def future_symbol_embedding(self, _request: object) -> Tensor:
        return torch.tensor([[3.0, 4.0], [3.0, 4.0]], requires_grad=True)


def _batch(next_valid: Tensor) -> TrainingBatch:
    request = diagnostic_batch(
        batch_size=2,
        num_actions=2,
        edge=8,
        seed=3,
        device=torch.device("cpu"),
        impossible_reward=-10.0,
    )
    masks = torch.zeros(2, 8, 8)
    masks[:, 0, 0] = 1
    return TrainingBatch(
        request=request,
        next_request=request,
        present_rewards=torch.tensor([2.0, -1.0]),
        next_state_valid=next_valid,
        action_indexes=torch.tensor([0, 1]),
        action_x=torch.tensor([0, 0]),
        action_y=torch.tensor([0, 0]),
        sample_ids=("a", "b"),
        reward_pixel_masks=masks,
    )


def test_double_dqn_selects_online_action_and_evaluates_target() -> None:
    online = torch.tensor([[[[1.0, 5.0]], [[-10.0, -10.0]]], [[[100.0, 0.0]], [[0.0, 0.0]]]])
    target = torch.tensor([[[[9.0, 4.0]], [[8.0, 7.0]]], [[[100.0, 0.0]], [[0.0, 0.0]]]])
    targets = _double_dqn_targets(
        FixedModel({"actionValues": online}),
        FixedModel({"actionValues": target}),
        _batch(torch.tensor([True, False])),
        discount_rate=0.85,
        maximum=10.0,
    )

    torch.testing.assert_close(targets, torch.tensor([3.4, 0.0]))


def test_double_dqn_clips_positive_and_negative_bootstraps_symmetrically() -> None:
    online = torch.tensor([[[[2.0]], [[1.0]]], [[[1.0]], [[2.0]]]])
    target = torch.tensor([[[[100.0]], [[0.0]]], [[[0.0]], [[-100.0]]]])

    targets = _double_dqn_targets(
        FixedModel({"actionValues": online}),
        FixedModel({"actionValues": target}),
        _batch(torch.tensor([True, True])),
        discount_rate=0.85,
        maximum=10.0,
    )

    torch.testing.assert_close(targets, torch.tensor([10.0, -10.0]))


def test_double_dqn_does_not_bootstrap_an_empty_nonterminal_action_mask() -> None:
    batch = _batch(torch.tensor([True, True]))
    empty_request = replace(
        batch.next_request,
        pixel_action_maps=torch.zeros_like(batch.next_request.pixel_action_maps),
    )
    batch = replace(batch, next_request=empty_request)
    values = torch.full((2, 2, 8, 8), 100.0)

    targets = _double_dqn_targets(
        FixedModel({"actionValues": values}),
        FixedModel({"actionValues": values}),
        batch,
        discount_rate=0.85,
        maximum=10.0,
    )

    torch.testing.assert_close(targets, torch.zeros(2))


def test_double_dqn_full_viewport_finds_action_outside_first_tile() -> None:
    request = diagnostic_batch(
        batch_size=1,
        num_actions=1,
        edge=64,
        seed=4,
        device=torch.device("cpu"),
        impossible_reward=-10.0,
    )
    batch = TrainingBatch(
        request=request,
        next_request=None,
        next_requests=(request,),
        present_rewards=torch.zeros(1),
        next_state_valid=torch.ones(1, dtype=torch.bool),
        action_indexes=torch.zeros(1, dtype=torch.long),
        action_x=torch.zeros(1, dtype=torch.long),
        action_y=torch.zeros(1, dtype=torch.long),
        sample_ids=("full",),
    )

    targets = _double_dqn_targets(
        CoordinateQ(), CoordinateQ(), batch, discount_rate=0.85, maximum=10.0, tile_size=(32, 32)
    )

    torch.testing.assert_close(targets, torch.tensor([0.85]))


def test_auxiliary_objectives_use_cosine_multilabel_and_categorical_losses() -> None:
    batch = _batch(torch.tensor([False, False]))
    batch = replace(
        batch,
        request=replace(batch.request, future_symbol_indexes=torch.tensor([1])),
        execution_features=torch.zeros(2, 12),
        cursors=torch.tensor([3, 5]),
    )
    trace = torch.tensor([[3.0, 4.0], [3.0, 4.0]], requires_grad=True)
    execution = torch.zeros(2, 12, requires_grad=True)
    cursor = torch.zeros(2, 37, requires_grad=True)
    outputs = {
        "presentRewards": torch.zeros(2, 2, 8, 8, requires_grad=True),
        "predictedTraces": trace,
        "predictedExecutionFeatures": execution,
        "predictedCursor": cursor,
    }
    target = _future_symbol_target(FutureTarget(), batch)
    assert target is not None
    assert not target.requires_grad
    torch.testing.assert_close(target.norm(dim=1), torch.ones(2))

    losses = _auxiliary_losses(
        outputs, batch, profile_config("testing", "https://example.com", 1).training.losses, target
    )

    torch.testing.assert_close(losses.raw_trace, torch.tensor(0.0))
    torch.testing.assert_close(losses.raw_execution, torch.tensor(2.0).log())
    torch.testing.assert_close(losses.raw_cursor, torch.tensor(37.0).log())
    (losses.trace + losses.execution + losses.cursor).backward()
    assert trace.grad is not None and execution.grad is not None and cursor.grad is not None


def test_conservative_q_margin_lowers_the_best_unsupported_action() -> None:
    config = profile_config("testing", "https://example.com", 1)
    batch = _batch(torch.tensor([False, False]))
    values = torch.zeros(2, 2, 8, 8, requires_grad=True)
    with torch.no_grad():
        values[0, 1, 0, 1] = 5.0
        values[1, 0, 0, 1] = 4.0

    loss = _conservative_q_loss(
        {"actionValues": values}, batch, torch.zeros(2), config.training.losses
    )
    loss.backward()

    assert loss > 0
    assert values.grad is not None
    assert values.grad[0, 1, 0, 1] > 0
    assert values.grad[1, 0, 0, 1] > 0
    assert values.grad[0, 0, 0, 0] == 0
    assert values.grad[1, 1, 0, 0] == 0


def test_conservative_q_zero_term_ignores_masked_value_sentinels() -> None:
    config = profile_config("testing", "https://example.com", 1)
    batch = _batch(torch.tensor([False, False]))
    valid = torch.zeros_like(batch.request.pixel_action_maps)
    valid[0, 0, 0, 0] = 1
    valid[0, 1, 0, 1] = 1
    valid[1, 1, 0, 0] = 1
    valid[1, 0, 0, 1] = 1
    batch = replace(batch, request=replace(batch.request, pixel_action_maps=valid))
    values = torch.full(
        valid.shape,
        torch.finfo(torch.float32).min,
        requires_grad=True,
    )
    with torch.no_grad():
        values[valid.bool()] = 0

    loss = _conservative_q_loss(
        {"actionValues": values}, batch, torch.zeros(2), config.training.losses
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_present_reward_trains_on_terminal_and_only_recorded_region() -> None:
    config = profile_config("testing", "https://example.com", 1)
    present = torch.zeros(2, 2, 8, 8, requires_grad=True)
    future = torch.zeros(2, 2, 8, 8, requires_grad=True)
    batch = _batch(torch.tensor([False, False]))

    present_loss, future_loss, _selected_q, _td_error = _value_losses(
        {"presentRewards": present, "discountFutureRewards": future},
        batch,
        torch.zeros(2),
        config.training.losses,
    )
    (present_loss + future_loss).backward()

    assert present_loss > 0
    assert future_loss == 0
    assert present.grad is not None
    assert torch.count_nonzero(present.grad) == 2
    assert present.grad[0, 0, 0, 0] != 0
    assert present.grad[1, 1, 0, 0] != 0


def test_action_values_mask_impossible_actions_once() -> None:
    present = torch.tensor([[[[2.0, 100.0]]]])
    future = torch.tensor([[[[3.0, 100.0]]]])
    masks = torch.tensor([[[[1.0, 0.0]]]])

    values = TraceNetHeads.action_values(present, future, masks, -10.0)

    assert values[0, 0, 0, 0] == 5
    assert values[0, 0, 0, 1] == torch.finfo(values.dtype).min
