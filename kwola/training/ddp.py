"""Explicit distributed process-group ownership."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Self

import torch
from torch import distributed


@dataclass(frozen=True, slots=True)
class DistributedSettings:
    rank: int
    world_size: int
    local_device: int
    init_method: str
    backend: str = "nccl"
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid distributed rank/world size")
        if self.backend == "nccl" and not torch.cuda.is_available():
            raise ValueError("NCCL coordination requires CUDA")


class DistributedCoordinator:
    def __init__(self, settings: DistributedSettings) -> None:
        self.settings = settings
        self._started = False

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def is_publisher(self) -> bool:
        return self.settings.rank == 0

    @property
    def device(self) -> torch.device:
        if self.settings.backend == "nccl":
            return torch.device("cuda", self.settings.local_device)
        return torch.device("cpu")

    def start(self) -> None:
        if self._started:
            raise RuntimeError("distributed coordinator is already started")
        if self.settings.backend == "nccl":
            torch.cuda.set_device(self.settings.local_device)
        distributed.init_process_group(
            backend=self.settings.backend,
            init_method=self.settings.init_method,
            rank=self.settings.rank,
            world_size=self.settings.world_size,
            timeout=timedelta(seconds=self.settings.timeout_seconds),
        )
        self._started = True

    def barrier(self) -> None:
        self._require_started()
        distributed.barrier()

    def propagate_failure(self, failed: bool) -> bool:
        self._require_started()
        value = torch.tensor([int(failed)], device=self.device)
        distributed.all_reduce(value, op=distributed.ReduceOp.MAX)
        return bool(value.item())

    def close(self) -> None:
        if not self._started:
            return
        distributed.destroy_process_group()
        self._started = False

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("distributed coordinator is not started")
