"""PyTorch shared-memory batch spooling for spawned training ranks."""

import platform
from dataclasses import fields, is_dataclass, replace
from typing import Any, cast

import torch
from torch import Tensor

from .samples import TrainingBatch


def share_batch(batch: TrainingBatch) -> TrainingBatch:
    """Move CPU tensor storage into /dev/shm without serializing image payloads."""

    if platform.system() != "Linux":
        return batch

    def share(tensor: Tensor) -> Tensor:
        if tensor.is_cuda:
            raise ValueError("only CPU batches can be shared")
        return cast(Tensor, tensor.share_memory_())  # type: ignore[no-untyped-call]

    return _map_tensors(batch, share)


def batch_to_device(batch: TrainingBatch, device: torch.device) -> TrainingBatch:
    def transfer(tensor: Tensor) -> Tensor:
        dtype = torch.float32 if tensor.dtype is torch.uint8 else tensor.dtype
        return tensor.to(device, dtype=dtype, non_blocking=True)

    return _map_tensors(batch, transfer)


def pin_batch(batch: TrainingBatch) -> TrainingBatch:
    """Copy CPU tensor storage into page-locked memory for asynchronous CUDA transfer."""

    def pin(tensor: Tensor) -> Tensor:
        if tensor.is_cuda or tensor.is_pinned():
            return tensor
        return tensor.contiguous().pin_memory()

    return _map_tensors(batch, pin)


def _map_tensors[T](value: T, transform: Any) -> T:
    if isinstance(value, Tensor):
        return cast(T, transform(value))
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _map_tensors(getattr(value, field.name), transform)
            for field in fields(value)
        }
        return cast(T, replace(value, **updates))
    if isinstance(value, tuple):
        return cast(T, tuple(_map_tensors(item, transform) for item in value))
    return value
