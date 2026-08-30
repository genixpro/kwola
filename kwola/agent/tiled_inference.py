"""Blend training-sized TraceNet tiles into full-viewport inference maps."""

from dataclasses import replace

import torch
from torch import Tensor, nn

from .model_heads import TraceNetHeads
from .tracenet import TraceNetRequest


def evaluate_tiled(
    model: nn.Module,
    request: TraceNetRequest,
    tile_size: tuple[int, int],
) -> dict[str, Tensor]:
    """Evaluate a full viewport using overlapping crops matching training geometry."""
    if request.compute_auxiliary:
        raise ValueError("tiled inference does not support auxiliary predictions")
    tile_width, tile_height = tile_size
    if tile_width < 8 or tile_height < 8 or tile_width % 8 or tile_height % 8:
        raise ValueError("inference tile dimensions must be positive multiples of eight")
    image = request.backbone.image
    height, width = image.shape[-2:]
    if width <= tile_width and height <= tile_height:
        return model(request)  # type: ignore[no-any-return]

    effective_width = width if width <= tile_width + tile_width // 8 else tile_width
    effective_height = height if height <= tile_height + tile_height // 8 else tile_height
    x_starts = _tile_starts(width, effective_width)
    y_starts = _tile_starts(height, effective_height)
    present_sum: Tensor | None = None
    future_sum: Tensor | None = None
    weights = torch.zeros(image.shape[0], 1, height, width, dtype=image.dtype, device=image.device)
    stamp: Tensor | None = None
    for top in y_starts:
        for left in x_starts:
            bottom = min(height, top + effective_height)
            right = min(width, left + effective_width)
            tile_request = _tile_request(request, left, top, right, bottom, stamp is None)
            tile_output = model(tile_request)
            present = tile_output["presentRewards"]
            future = tile_output["discountFutureRewards"]
            if present_sum is None:
                shape = (present.shape[0], present.shape[1], height, width)
                present_sum = torch.zeros(shape, dtype=present.dtype, device=present.device)
                future_sum = torch.zeros_like(present_sum)
            blend = _blend_weights(bottom - top, right - left, present)
            present_sum[..., top:bottom, left:right] += present * blend
            assert future_sum is not None
            future_sum[..., top:bottom, left:right] += future * blend
            weights[..., top:bottom, left:right] += blend
            if "stamp" in tile_output:
                stamp = tile_output["stamp"]

    assert present_sum is not None and future_sum is not None
    present_map = present_sum / weights.clamp_min(torch.finfo(weights.dtype).eps)
    future_map = future_sum / weights.clamp_min(torch.finfo(weights.dtype).eps)
    output = {
        "presentRewards": present_map,
        "discountFutureRewards": future_map,
        "actionValues": TraceNetHeads.action_values(
            present_map,
            future_map,
            request.pixel_action_maps,
            request.impossible_action_reward,
        ),
    }
    if stamp is not None:
        output["stamp"] = stamp
    return output


def _tile_request(
    request: TraceNetRequest,
    left: int,
    top: int,
    right: int,
    bottom: int,
    output_stamp: bool,
) -> TraceNetRequest:
    backbone = replace(
        request.backbone,
        image=request.backbone.image[..., top:bottom, left:right],
        recent_actions_image=request.backbone.recent_actions_image[..., top:bottom, left:right],
    )
    return replace(
        request,
        backbone=backbone,
        pixel_action_maps=request.pixel_action_maps[..., top:bottom, left:right],
        output_stamp=request.output_stamp and output_stamp,
    )


def _tile_starts(length: int, tile: int) -> tuple[int, ...]:
    if length <= tile:
        return (0,)
    stride = max(8, tile // 2)
    difference = length - tile
    segments = (difference + stride - 1) // stride
    return tuple(round(index * difference / segments) for index in range(segments + 1))


def _blend_weights(height: int, width: int, reference: Tensor) -> Tensor:
    def axis(size: int) -> Tensor:
        if size == 1:
            return torch.ones(1, dtype=reference.dtype, device=reference.device)
        positions = torch.linspace(-1.0, 1.0, size, dtype=reference.dtype, device=reference.device)
        return (1.0 - positions.abs()).clamp_min(0.05)

    return axis(height)[:, None] * axis(width)[None, :]
