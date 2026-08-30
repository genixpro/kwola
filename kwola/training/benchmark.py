"""Standard forward/backward/optimizer benchmark."""

import copy
import math
import statistics
from dataclasses import replace
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from kwola.agent import TraceNet, action_catalog
from kwola.config import RunConfig, load_config

from .batches import diagnostic_batch
from .optimizer import ModelOptimizer
from .samples import TrainingBatch


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    passed: bool
    device: str
    median_optimizer_seconds: float = Field(gt=0)
    samples_per_second: float = Field(gt=0)
    peak_vram_gib: float = Field(ge=0)
    median_target_evaluation_seconds: float = Field(ge=0)
    mean_next_tiles: float = Field(ge=1)


def _benchmark_batch(config: RunConfig, action_count: int, device: torch.device) -> TrainingBatch:
    request = diagnostic_batch(
        batch_size=config.training.batch_size,
        num_actions=action_count,
        edge=64 if config.profile == "testing" else 320,
        seed=config.seed,
        device=device,
        impossible_reward=config.policy.rewards.impossible_action,
    )
    request = replace(
        request,
        compute_auxiliary=True,
        auxiliary_action_type=torch.zeros(
            config.training.batch_size, dtype=torch.long, device=device
        ),
        auxiliary_action_x=torch.zeros(config.training.batch_size, dtype=torch.long, device=device),
        auxiliary_action_y=torch.zeros(config.training.batch_size, dtype=torch.long, device=device),
        auxiliary_pixel_masks=torch.ones(
            config.training.batch_size,
            request.backbone.image.shape[-2],
            request.backbone.image.shape[-1],
            device=device,
        ),
        future_symbol_indexes=request.backbone.recent_symbol_indexes,
        future_symbol_offsets=request.backbone.recent_symbol_offsets,
        future_symbol_weights=request.backbone.recent_symbol_weights,
    )
    next_edge = 96 if config.profile == "testing" else 384
    next_requests = tuple(
        diagnostic_batch(
            batch_size=1,
            num_actions=action_count,
            edge=next_edge,
            seed=config.seed + index + 1,
            device=device,
            impossible_reward=config.policy.rewards.impossible_action,
        )
        for index in range(config.training.batch_size)
    )
    return TrainingBatch(
        request=request,
        next_request=None,
        next_requests=next_requests,
        present_rewards=torch.zeros(config.training.batch_size, device=device),
        next_state_valid=torch.ones(config.training.batch_size, dtype=torch.bool, device=device),
        action_indexes=torch.zeros(config.training.batch_size, dtype=torch.long, device=device),
        action_x=torch.zeros(config.training.batch_size, dtype=torch.long, device=device),
        action_y=torch.zeros(config.training.batch_size, dtype=torch.long, device=device),
        sample_ids=tuple(str(index) for index in range(config.training.batch_size)),
        reward_pixel_masks=request.auxiliary_pixel_masks,
        execution_features=torch.zeros(config.training.batch_size, 12, device=device),
        cursors=torch.zeros(config.training.batch_size, dtype=torch.long, device=device),
    )


def run_benchmark(run_dir: Path, iterations: int = 5) -> BenchmarkResult:
    config = load_config(run_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = True
    action_count = len(action_catalog(config.policy))
    model = TraceNet(config.model, num_actions=action_count).to(device)
    optimizer = ModelOptimizer(model, config.training)
    target = copy.deepcopy(model).eval()
    batch = _benchmark_batch(config, action_count, device)
    optimizer.step_training(
        batch,
        target,
        config.policy.rewards.discount_rate,
        config.policy.rewards.max_discounted_reward,
    )
    measurements = [
        optimizer.step_training(
            batch,
            target,
            config.policy.rewards.discount_rate,
            config.policy.rewards.max_discounted_reward,
        )
        for _ in range(iterations)
    ]
    median = statistics.median(item.duration_seconds for item in measurements)
    throughput = config.training.batch_size / median
    vram = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    passed = all(
        math.isfinite(value)
        for item in measurements
        for value in (item.loss, item.duration_seconds, item.gradient_norm)
    )
    return BenchmarkResult(
        passed=passed,
        device=str(device),
        median_optimizer_seconds=median,
        samples_per_second=throughput,
        peak_vram_gib=vram,
        median_target_evaluation_seconds=statistics.median(
            item.target_evaluation_seconds for item in measurements
        ),
        mean_next_tiles=sum(item.mean_next_tiles for item in measurements) / len(measurements),
    )
