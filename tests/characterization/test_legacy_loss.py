import torch
from torch import Tensor, nn

from kwola.config import profile_config
from kwola.config.models import LossConfig
from kwola.training.batches import diagnostic_batch
from kwola.training.losses import behavior_loss
from kwola.training.samples import TrainingBatch


class FixedModel(nn.Module):
    def __init__(self, outputs: dict[str, Tensor]) -> None:
        super().__init__()
        self.outputs = outputs

    def forward(self, _request: object) -> dict[str, Tensor]:
        return self.outputs


def test_masked_multi_head_loss_matches_legacy_equations() -> None:
    config = profile_config("testing", "https://example.com", 9)
    request = diagnostic_batch(
        batch_size=2,
        num_actions=2,
        edge=8,
        seed=3,
        device=torch.device("cpu"),
        impossible_reward=-10.0,
    )
    masks = torch.zeros(2, 8, 8)
    masks[0, 1:4, 2:5] = 1
    masks[1, 3:7, 1:6] = 1
    batch = TrainingBatch(
        request,
        request,
        torch.tensor([0.25, -0.1]),
        torch.tensor([True, True]),
        torch.tensor([0, 1]),
        torch.tensor([3, 2]),
        torch.tensor([2, 4]),
        ("a", "b"),
        masks,
    )
    current = _outputs(2, 2, 8, offset=0.01)
    target = _outputs(2, 2, 8, offset=0.02)
    result = behavior_loss(
        FixedModel(current), FixedModel(target), batch, config.training, 6, 1, 0.85, 10.0
    )
    expected = _legacy_primary(current, target, batch, config.training.losses)
    for actual, reference in zip(
        (
            result.present_reward,
            result.discounted_reward,
            result.state_value,
            result.advantage,
            result.action_probability,
        ),
        expected,
        strict=True,
    ):
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(result.total, sum(expected), rtol=1e-5, atol=1e-6)


def _outputs(batch: int, actions: int, edge: int, offset: float) -> dict[str, Tensor]:
    values = torch.arange(batch * actions * edge * edge, dtype=torch.float32)
    maps = values.reshape(batch, actions, edge, edge) / 1000 + offset
    probabilities = torch.softmax(maps.flatten(1), dim=1).reshape_as(maps)
    return {
        "presentRewards": maps,
        "discountFutureRewards": maps * 0.5,
        "stateValues": torch.tensor([[0.1], [0.2]]),
        "advantage": maps * 0.25,
        "actionProbabilities": probabilities,
    }


def _legacy_primary(
    current: dict[str, Tensor],
    target: dict[str, Tensor],
    batch: TrainingBatch,
    weights: LossConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    indexes = torch.arange(2)
    selected = (indexes, batch.action_indexes)
    assert batch.reward_pixel_masks is not None
    combo = batch.reward_pixel_masks * batch.request.pixel_action_maps[selected]
    best = (target["presentRewards"] + target["discountFutureRewards"]).flatten(1).max(1).values
    future_target = torch.clamp(best * 0.85, max=10)
    expected = batch.present_rewards + future_target
    count = combo.sum((1, 2)).clamp_min(1)
    present = (
        (combo * batch.present_rewards[:, None, None] - current["presentRewards"][selected] * combo)
        * combo
    ).square().sum((1, 2)) / count
    future = (
        (combo * future_target[:, None, None] - current["discountFutureRewards"][selected] * combo)
        * combo
    ).square().sum((1, 2)) / count
    advantage_target = combo * (
        expected[:, None, None] - current["stateValues"].detach()[:, :, None]
    )
    advantage = ((advantage_target - current["advantage"][selected] * combo) * combo).square().sum(
        (1, 2)
    ) / count
    state = (current["stateValues"][:, 0] - expected).square().mean() * weights.state_value
    action = _legacy_action_loss(current, batch, weights.action_probability)
    return (
        present.mean() * weights.present_reward,
        future.mean() * weights.discounted_future_reward,
        state,
        advantage.mean() * weights.advantage,
        action,
    )


def _legacy_action_loss(outputs: dict[str, Tensor], batch: TrainingBatch, weight: float) -> Tensor:
    advantage = outputs["advantage"]
    probabilities = outputs["actionProbabilities"]
    _, _actions, height, width = advantage.shape
    best = advantage.flatten(1).argmax(1)
    action = torch.div(best, height * width, rounding_mode="floor")
    pixel = best % (height * width)
    y, x = torch.div(pixel, width, rounding_mode="floor"), pixel % width
    target = torch.zeros_like(probabilities)
    for sample in range(2):
        target[
            sample,
            int(action[sample]),
            max(0, int(y[sample]) - 3) : min(height - 1, int(y[sample]) + 3),
            max(0, int(x[sample]) - 3) : min(width - 1, int(x[sample]) + 3),
        ] = 1
    target *= batch.request.pixel_action_maps
    target /= target.sum((2, 3)).clamp_min(1)[:, :, None, None]
    return ((target - probabilities) * batch.request.pixel_action_maps).abs().sum(
        (1, 2, 3)
    ).mean() * weight
