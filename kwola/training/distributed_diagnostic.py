"""Two-rank NCCL forward/backward/optimizer acceptance diagnostic."""

import multiprocessing
import socket
from contextlib import closing
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict
from torch import distributed
from torch.multiprocessing import spawn  # type: ignore[attr-defined]
from torch.nn.parallel import DistributedDataParallel

from kwola.agent import TraceNet, action_catalog
from kwola.config import profile_config

from .batches import diagnostic_batch
from .ddp import DistributedCoordinator, DistributedSettings
from .optimizer import ModelOptimizer


class DistributedDiagnosticResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    passed: bool
    world_size: int
    losses: tuple[float, ...]
    devices: tuple[str, ...]


def run_two_rank_diagnostic() -> DistributedDiagnosticResult:
    if torch.cuda.device_count() < 2 or not distributed.is_nccl_available():
        raise RuntimeError("two-rank diagnostic requires two CUDA devices and NCCL")
    context = multiprocessing.get_context("spawn")
    results: Any = context.SimpleQueue()
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    spawn(  # type: ignore[no-untyped-call]
        _diagnostic_rank, args=(2, init_method, results), nprocs=2, join=True
    )
    return DistributedDiagnosticResult.model_validate_json(results.get())


def _diagnostic_rank(rank: int, world_size: int, init_method: str, results: Any) -> None:
    # Match production's reward-only heads. Two steps are intentional: DDP reports
    # an unfinished reduction from unused parameters at the next forward pass.
    config = profile_config("rig", "https://example.com", 991)
    settings = DistributedSettings(rank, world_size, rank, init_method)
    with DistributedCoordinator(settings) as coordinator:
        action_count = len(action_catalog(config.policy))
        model = TraceNet(config.model, num_actions=action_count).to(coordinator.device)
        parallel = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        optimizer = ModelOptimizer(parallel, config.training)
        request = diagnostic_batch(
            batch_size=2,
            num_actions=action_count,
            edge=64,
            seed=config.seed + rank,
            device=coordinator.device,
            impossible_reward=config.policy.rewards.impossible_action,
        )
        metrics = optimizer.step(request)
        metrics = optimizer.step(request)
        loss = torch.tensor([metrics.loss], device=coordinator.device)
        gathered = [torch.zeros_like(loss) for _ in range(world_size)]
        distributed.all_gather(gathered, loss)
        coordinator.barrier()
        if coordinator.is_publisher:
            result = DistributedDiagnosticResult(
                passed=all(torch.isfinite(item).item() for item in gathered),
                world_size=world_size,
                losses=tuple(float(item.item()) for item in gathered),
                devices=tuple(f"cuda:{index}" for index in range(world_size)),
            )
            results.put(result.model_dump_json())


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as connection:
        connection.bind(("127.0.0.1", 0))
        return int(connection.getsockname()[1])


def main() -> None:
    print(run_two_rank_diagnostic().model_dump_json(indent=2))


if __name__ == "__main__":
    main()
