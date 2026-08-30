"""Focused optimizer execution."""

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from kwola.agent.tracenet import TraceNetRequest
from kwola.config.models import TrainingConfig

from .losses import aggregate_loss, behavior_loss
from .samples import TrainingBatch


@dataclass(frozen=True, slots=True)
class OptimizerMetrics:
    loss: float
    duration_seconds: float
    samples_per_second: float


class ModelOptimizer:
    def __init__(self, model: nn.Module, config: TrainingConfig) -> None:
        self.model = model
        self.config = config
        optimizer_type = torch.optim.Adamax if config.optimizer == "adamax" else torch.optim.Adam
        self.optimizer = optimizer_type(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.gradient_beta, config.squared_gradient_beta),
            weight_decay=config.weight_decay,
        )

    def step(self, request: TraceNetRequest) -> OptimizerMetrics:
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(request)
        loss = aggregate_loss(outputs, self.config.losses)
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
        if request.backbone.image.is_cuda:
            torch.cuda.synchronize(request.backbone.image.device)
        duration = time.perf_counter() - started
        batch_size = int(request.backbone.image.shape[0])
        return OptimizerMetrics(float(loss.detach()), duration, batch_size / duration)

    def step_training(
        self,
        batch: TrainingBatch,
        target_model: nn.Module,
        training_index: int,
        discount_rate: float,
        max_discounted_reward: float,
    ) -> OptimizerMetrics:
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = behavior_loss(
            self.model,
            target_model,
            batch,
            self.config,
            training_index,
            self.config.world_size,
            discount_rate,
            max_discounted_reward,
        )
        losses.total.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
        if batch.request.backbone.image.is_cuda:
            torch.cuda.synchronize(batch.request.backbone.image.device)
        duration = time.perf_counter() - started
        throughput = len(batch.sample_ids) / duration
        return OptimizerMetrics(float(losses.total.detach()), duration, throughput)


def load_optimizer_checkpoint(
    optimizer: torch.optim.Optimizer,
    state_dict: dict[str, Any],
    appended_parameter_count: int,
) -> None:
    """Load optimizer state, appending freshly introduced model parameters when needed."""
    current = optimizer.state_dict()
    saved_groups = state_dict.get("param_groups")
    current_groups = current.get("param_groups")
    if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
        optimizer.load_state_dict(state_dict)
        return
    if len(saved_groups) != len(current_groups):
        optimizer.load_state_dict(state_dict)
        return
    saved_counts = [len(group["params"]) for group in saved_groups]
    current_counts = [len(group["params"]) for group in current_groups]
    if saved_counts == current_counts:
        optimizer.load_state_dict(state_dict)
        return
    if any(saved > current for saved, current in zip(saved_counts, current_counts, strict=True)):
        optimizer.load_state_dict(state_dict)
        return
    if sum(current_counts) - sum(saved_counts) != appended_parameter_count:
        optimizer.load_state_dict(state_dict)
        return
    migrated = deepcopy(state_dict)
    for saved, current_group in zip(migrated["param_groups"], current_groups, strict=True):
        saved["params"] = list(current_group["params"])
    optimizer.load_state_dict(migrated)
