import platform
from dataclasses import replace

import pytest
import torch

from kwola.training.batches import diagnostic_batch
from kwola.training.samples import TrainingBatch
from kwola.training.spool import batch_to_device, pin_batch, share_batch


def test_batch_spool_uses_shared_and_contiguous_pinned_tensor_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = diagnostic_batch(
        batch_size=1,
        num_actions=2,
        edge=8,
        seed=1,
        device=torch.device("cpu"),
        impossible_reward=-10.0,
    )
    batch = TrainingBatch(
        request,
        request,
        torch.ones(1),
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        ("sample",),
    )
    shared = share_batch(batch)
    assert shared.request.backbone.image.is_shared() or platform.system() != "Linux"
    moved = batch_to_device(shared, torch.device("cpu"))
    assert torch.equal(moved.present_rewards, batch.present_rewards)
    compact = replace(
        batch,
        request=replace(batch.request, pixel_action_maps=batch.request.pixel_action_maps.byte()),
    )
    expanded = batch_to_device(compact, torch.device("cpu"))
    assert expanded.request.pixel_action_maps.dtype is torch.float32
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    overlapping = replace(batch, present_rewards=torch.ones(1).expand(2))
    pinned = pin_batch(overlapping)
    assert pinned.present_rewards.is_contiguous()
