from types import SimpleNamespace

import torch
from torch import Tensor, nn

from kwola.agent.model_heads import TraceNetHeads
from kwola.config import profile_config
from kwola.training.losses import q_learning_loss
from kwola.training.samples import TrainingBatch


class TabularSpatialQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.present = nn.Parameter(torch.zeros(2, 2))
        self.future = nn.Parameter(torch.zeros(2, 2))

    def forward(self, request: object) -> dict[str, Tensor]:
        state = request.backbone.step_number.long()  # type: ignore[attr-defined]
        present = self.present[state, :, None, None]
        future = self.future[state, :, None, None]
        values = TraceNetHeads.action_values(
            present,
            future,
            request.pixel_action_maps,  # type: ignore[attr-defined]
            -10.0,
        )
        return {
            "presentRewards": present,
            "discountFutureRewards": future,
            "actionValues": values,
        }


def _request(states: tuple[int, ...]) -> object:
    return SimpleNamespace(
        backbone=SimpleNamespace(step_number=torch.tensor(states)),
        pixel_action_maps=torch.ones(len(states), 2, 1, 1),
    )


def _batch(
    states: tuple[int, ...],
    actions: tuple[int, ...],
    rewards: tuple[float, ...],
    next_states: tuple[int, ...],
    next_valid: tuple[bool, ...],
) -> TrainingBatch:
    count = len(states)
    return TrainingBatch(
        request=_request(states),  # type: ignore[arg-type]
        next_request=_request(next_states),  # type: ignore[arg-type]
        present_rewards=torch.tensor(rewards),
        next_state_valid=torch.tensor(next_valid),
        action_indexes=torch.tensor(actions),
        action_x=torch.zeros(count, dtype=torch.long),
        action_y=torch.zeros(count, dtype=torch.long),
        sample_ids=tuple(str(index) for index in range(count)),
    )


def _step(model: TabularSpatialQ, target: TabularSpatialQ, batch: TrainingBatch) -> None:
    config = profile_config("testing", "https://example.com", 1)
    optimizer = getattr(model, "_test_optimizer", None)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        model._test_optimizer = optimizer  # type: ignore[attr-defined]
    optimizer.zero_grad(set_to_none=True)
    loss = q_learning_loss(model, target, batch, config.training, 0.85, 10.0).total
    loss.backward()
    optimizer.step()


def test_terminal_bandit_converges_to_the_higher_reward_action() -> None:
    model = TabularSpatialQ()
    target = TabularSpatialQ()
    batch = _batch((0, 0), (0, 1), (0.1, 1.0), (0, 0), (False, False))

    for _ in range(150):
        _step(model, target, batch)

    values = model(_request((0,)))["actionValues"].flatten()
    assert int(values.argmax()) == 1
    assert values[1] > 0.95


def test_delayed_reward_propagates_into_the_first_actions_future_map() -> None:
    model = TabularSpatialQ()
    target = TabularSpatialQ()
    first = _batch((0,), (0,), (0.0,), (1,), (True,))
    terminal = _batch((1,), (1,), (1.0,), (1,), (False,))

    for _ in range(200):
        _step(model, target, terminal)
        target.load_state_dict(model.state_dict())
        _step(model, target, first)
        target.load_state_dict(model.state_dict())

    assert model.present[0, 0].abs() < 0.01
    assert model.future[0, 0] > 0.8
    assert int(model(_request((0,)))["actionValues"].flatten().argmax()) == 0
