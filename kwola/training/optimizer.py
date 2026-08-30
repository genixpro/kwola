"""Focused optimizer execution."""

import time
from dataclasses import dataclass

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
