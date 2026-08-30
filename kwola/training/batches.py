"""Deterministic model-input assembly for diagnostics and cache fallbacks."""

import torch
from torch import Tensor

from kwola.agent.model_backbone import BackboneInput
from kwola.agent.tracenet import TraceNetRequest


def diagnostic_batch(
    *,
    batch_size: int,
    num_actions: int,
    edge: int,
    seed: int,
    device: torch.device,
    impossible_reward: float,
) -> TraceNetRequest:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = torch.rand(batch_size, 1, edge, edge, generator=generator).to(device)
    recent_images = torch.rand(
        batch_size, num_actions, edge, edge, generator=generator
    ).to(device)
    recent_vector = torch.rand(batch_size, 5 * num_actions, generator=generator).to(device)
    indexes, offsets, weights = _symbols(batch_size, device)
    symbol_set = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
    symbol_mask = torch.zeros(batch_size, 4, dtype=torch.bool, device=device)
    steps = torch.arange(1, batch_size + 1, dtype=torch.float32, device=device)
    masks = torch.ones(batch_size, num_actions, edge, edge, device=device)
    backbone = BackboneInput(
        image,
        recent_images,
        recent_vector,
        indexes,
        offsets,
        weights,
        indexes,
        offsets,
        weights,
        symbol_set,
        symbol_mask,
        steps,
    )
    return TraceNetRequest(backbone, masks, impossible_reward)


def _symbols(batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    indexes = torch.arange(1, batch_size * 2 + 1, dtype=torch.long, device=device)
    offsets = torch.arange(0, batch_size * 2, 2, dtype=torch.long, device=device)
    weights = torch.ones(batch_size * 2, dtype=torch.float32, device=device)
    return indexes, offsets, weights
