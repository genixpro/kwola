"""Standard forward/backward/optimizer benchmark."""

import statistics
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from kwola.agent import TraceNet
from kwola.config import load_config

from .batches import diagnostic_batch
from .optimizer import ModelOptimizer


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    passed: bool
    device: str
    median_optimizer_seconds: float = Field(gt=0)
    samples_per_second: float = Field(gt=0)
    peak_vram_gib: float = Field(ge=0)


def run_benchmark(run_dir: Path, iterations: int = 5) -> BenchmarkResult:
    config = load_config(run_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = True
    model = TraceNet(config.model, num_actions=6).to(device)
    optimizer = ModelOptimizer(model, config.training)
    request = diagnostic_batch(
        batch_size=config.training.batch_size,
        num_actions=6,
        edge=64 if config.profile == "testing" else 320,
        seed=config.seed,
        device=device,
        impossible_reward=config.policy.rewards.impossible_action,
    )
    optimizer.step(request)
    measurements = [optimizer.step(request) for _ in range(iterations)]
    median = statistics.median(item.duration_seconds for item in measurements)
    throughput = config.training.batch_size / median
    vram = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    passed = device.type != "cuda" or (median <= 1.35 and throughput >= 145 and vram <= 5.0)
    return BenchmarkResult(
        passed=passed,
        device=str(device),
        median_optimizer_seconds=median,
        samples_per_second=throughput,
        peak_vram_gib=vram,
    )
