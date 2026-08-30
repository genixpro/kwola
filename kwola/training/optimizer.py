"""Focused optimizer execution."""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from kwola.agent.tracenet import TraceNetRequest
from kwola.config.models import TrainingConfig

from .losses import aggregate_loss, q_learning_loss
from .samples import TrainingBatch


@dataclass(frozen=True, slots=True)
class OptimizerMetrics:
    loss: float
    duration_seconds: float
    samples_per_second: float
    present_loss: float = 0.0
    future_loss: float = 0.0
    mean_selected_q: float = 0.0
    mean_bootstrap_target: float = 0.0
    mean_absolute_td_error: float = 0.0
    gradient_norm: float = 0.0


def summarize_optimizer_metrics(
    results: Sequence[OptimizerMetrics], samples: int
) -> OptimizerMetrics:
    count = len(results)
    duration = sum(result.duration_seconds for result in results)
    return OptimizerMetrics(
        loss=sum(result.loss for result in results) / count,
        duration_seconds=duration,
        samples_per_second=samples / duration,
        present_loss=sum(result.present_loss for result in results) / count,
        future_loss=sum(result.future_loss for result in results) / count,
        mean_selected_q=sum(result.mean_selected_q for result in results) / count,
        mean_bootstrap_target=sum(result.mean_bootstrap_target for result in results) / count,
        mean_absolute_td_error=sum(result.mean_absolute_td_error for result in results) / count,
        gradient_norm=sum(result.gradient_norm for result in results) / count,
    )


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
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        if request.backbone.image.is_cuda:
            torch.cuda.synchronize(request.backbone.image.device)
        duration = time.perf_counter() - started
        batch_size = int(request.backbone.image.shape[0])
        return OptimizerMetrics(
            float(loss.detach()),
            duration,
            batch_size / duration,
            gradient_norm=float(gradient_norm.detach()),
        )

    def step_training(
        self,
        batch: TrainingBatch,
        target_model: nn.Module,
        discount_rate: float,
        max_discounted_reward: float,
    ) -> OptimizerMetrics:
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = q_learning_loss(
            self.model,
            target_model,
            batch,
            self.config,
            discount_rate,
            max_discounted_reward,
        )
        losses.total.backward()  # type: ignore[no-untyped-call]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        if batch.request.backbone.image.is_cuda:
            torch.cuda.synchronize(batch.request.backbone.image.device)
        duration = time.perf_counter() - started
        throughput = len(batch.sample_ids) / duration
        return OptimizerMetrics(
            float(losses.total.detach()),
            duration,
            throughput,
            present_loss=float(losses.present_reward.detach()),
            future_loss=float(losses.discounted_reward.detach()),
            mean_selected_q=float(losses.mean_selected_q.detach()),
            mean_bootstrap_target=float(losses.mean_bootstrap_target.detach()),
            mean_absolute_td_error=float(losses.mean_absolute_td_error.detach()),
            gradient_norm=float(gradient_norm.detach()),
        )
