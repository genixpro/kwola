from dataclasses import replace

import torch
from torch import Tensor, nn

from kwola.agent.model_heads import TraceNetHeads
from kwola.config import profile_config
from kwola.training.batches import diagnostic_batch
from kwola.training.losses import _conservative_q_loss, _double_dqn_targets, _value_losses
from kwola.training.samples import TrainingBatch


class FixedModel(nn.Module):
    def __init__(self, outputs: dict[str, Tensor]) -> None:
        super().__init__()
        self.outputs = outputs

    def forward(self, _request: object) -> dict[str, Tensor]:
        return self.outputs


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
