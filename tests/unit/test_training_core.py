import copy

import torch

from kwola.agent import TraceNet
from kwola.config import profile_config
from kwola.training.batches import diagnostic_batch
from kwola.training.losses import aggregate_loss, q_learning_loss
from kwola.training.optimizer import ModelOptimizer
from kwola.training.samples import TrainingBatch


def test_diagnostic_batch_and_aggregate_loss_are_finite() -> None:
    config = profile_config("testing", "https://example.com", 4)
    request = diagnostic_batch(
        batch_size=2,
        num_actions=6,
        edge=64,
        seed=8,
        device=torch.device("cpu"),
        impossible_reward=config.policy.rewards.impossible_action,
    )
    model = TraceNet(config.model, 6)
    outputs = model(request)
    loss = aggregate_loss(outputs, config.training.losses)
    assert request.backbone.image.shape == (2, 1, 64, 64)
    assert torch.isfinite(loss)


def test_q_loss_and_optimizer_are_finite() -> None:
    config = profile_config("testing", "https://example.com", 5)
    request = diagnostic_batch(
        batch_size=2,
        num_actions=6,
        edge=64,
        seed=9,
        device=torch.device("cpu"),
        impossible_reward=config.policy.rewards.impossible_action,
    )
    batch = TrainingBatch(
        request=request,
        next_request=request,
        present_rewards=torch.tensor([0.1, 0.2]),
        next_state_valid=torch.tensor([True, False]),
        action_indexes=torch.tensor([0, 1]),
        action_x=torch.tensor([1, 2]),
        action_y=torch.tensor([2, 1]),
        sample_ids=("a", "b"),
    )
    model = TraceNet(config.model, 6)
    target = copy.deepcopy(model)
    losses = q_learning_loss(model, target, batch, config.training, 0.85, 10.0)
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.mean_absolute_td_error)
    optimizer = ModelOptimizer(model, config.training)
    diagnostic_metrics = optimizer.step(request)
    behavior_metrics = optimizer.step_training(batch, target, 0.85, 10.0)
    assert diagnostic_metrics.samples_per_second > 0
    assert behavior_metrics.samples_per_second > 0
