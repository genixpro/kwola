"""Blend training-sized TraceNet tiles into full-viewport inference maps."""

from collections import defaultdict
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from .model_backbone import BackboneInput
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

    regions = tile_regions(width, height, tile_size)
    present_sum: Tensor | None = None
    future_sum: Tensor | None = None
    weights = torch.zeros(image.shape[0], 1, height, width, dtype=image.dtype, device=image.device)
    stamp: Tensor | None = None
    for left, top, right, bottom in regions:
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


@dataclass(frozen=True, slots=True)
class _Tile:
    sample: int
    left: int
    top: int
    right: int
    bottom: int
    request: TraceNetRequest


def evaluate_tiled_batch(
    model: nn.Module,
    requests: tuple[TraceNetRequest, ...],
    tile_size: tuple[int, int],
) -> tuple[dict[str, Tensor], ...]:
    """Evaluate differently sized batch-one viewports using shape-grouped tiles."""
    if not requests:
        return ()
    groups: dict[tuple[int, int], list[_Tile]] = defaultdict(list)
    shapes: list[tuple[int, int]] = []
    for sample, request in enumerate(requests):
        if request.backbone.image.shape[0] != 1:
            raise ValueError("tiled batch requests must contain exactly one sample")
        if request.compute_auxiliary:
            raise ValueError("tiled target evaluation does not support auxiliary predictions")
        height, width = request.backbone.image.shape[-2:]
        shapes.append((height, width))
        for left, top, right, bottom in tile_regions(width, height, tile_size):
            tile = _Tile(
                sample,
                left,
                top,
                right,
                bottom,
                _tile_request(request, left, top, right, bottom, False),
            )
            groups[(bottom - top, right - left)].append(tile)

    present_sums: list[Tensor | None] = [None] * len(requests)
    future_sums: list[Tensor | None] = [None] * len(requests)
    weight_sums = [
        torch.zeros(
            1,
            1,
            height,
            width,
            dtype=requests[index].backbone.image.dtype,
            device=requests[index].backbone.image.device,
        )
        for index, (height, width) in enumerate(shapes)
    ]
    maximum_group_size = len(requests)
    for grouped_tiles in groups.values():
        for start in range(0, len(grouped_tiles), maximum_group_size):
            tiles = grouped_tiles[start : start + maximum_group_size]
            joined = _join_single_requests(tuple(tile.request for tile in tiles))
            output = model(joined)
            _accumulate_tiles(
                tiles,
                output,
                shapes,
                present_sums,
                future_sums,
                weight_sums,
            )

    results = []
    for sample, request in enumerate(requests):
        present_sum = present_sums[sample]
        future_sum = future_sums[sample]
        assert present_sum is not None and future_sum is not None
        weights = weight_sums[sample].clamp_min(torch.finfo(weight_sums[sample].dtype).eps)
        present = present_sum / weights
        future = future_sum / weights
        results.append(
            {
                "presentRewards": present,
                "discountFutureRewards": future,
                "actionValues": TraceNetHeads.action_values(
                    present,
                    future,
                    request.pixel_action_maps,
                    request.impossible_action_reward,
                ),
            }
        )
    return tuple(results)


def _accumulate_tiles(
    tiles: list[_Tile],
    output: dict[str, Tensor],
    shapes: list[tuple[int, int]],
    present_sums: list[Tensor | None],
    future_sums: list[Tensor | None],
    weight_sums: list[Tensor],
) -> None:
    for row, tile in enumerate(tiles):
        present = output["presentRewards"][row : row + 1]
        future = output["discountFutureRewards"][row : row + 1]
        height, width = shapes[tile.sample]
        present_sum = present_sums[tile.sample]
        future_sum = future_sums[tile.sample]
        if present_sum is None:
            shape = (1, present.shape[1], height, width)
            present_sum = torch.zeros(shape, dtype=present.dtype, device=present.device)
            future_sum = torch.zeros_like(present_sum)
            present_sums[tile.sample] = present_sum
            future_sums[tile.sample] = future_sum
        assert future_sum is not None
        blend = _blend_weights(tile.bottom - tile.top, tile.right - tile.left, present)
        present_sum[..., tile.top : tile.bottom, tile.left : tile.right] += present * blend
        future_sum[..., tile.top : tile.bottom, tile.left : tile.right] += future * blend
        weight_sums[tile.sample][..., tile.top : tile.bottom, tile.left : tile.right] += blend


def tile_regions(
    width: int, height: int, tile_size: tuple[int, int]
) -> tuple[tuple[int, int, int, int], ...]:
    """Return the canonical inference regions for a viewport."""
    tile_width, tile_height = tile_size
    if tile_width < 8 or tile_height < 8 or tile_width % 8 or tile_height % 8:
        raise ValueError("inference tile dimensions must be positive multiples of eight")
    effective_width = width if width <= tile_width + tile_width // 8 else tile_width
    effective_height = height if height <= tile_height + tile_height // 8 else tile_height
    return tuple(
        (left, top, min(width, left + effective_width), min(height, top + effective_height))
        for top in _tile_starts(height, effective_height)
        for left in _tile_starts(width, effective_width)
    )


def _join_single_requests(requests: tuple[TraceNetRequest, ...]) -> TraceNetRequest:
    first = requests[0]
    backbones = tuple(request.backbone for request in requests)
    recent = _join_bags(
        tuple(
            (
                item.recent_symbol_indexes,
                item.recent_symbol_weights,
            )
            for item in backbones
        )
    )
    coverage = _join_bags(
        tuple(
            (
                item.coverage_symbol_indexes,
                item.coverage_symbol_weights,
            )
            for item in backbones
        )
    )
    per_sample_symbols = [
        item.coverage_symbols_set[~item.coverage_symbols_key_mask[0]] for item in backbones
    ]
    symbol_set = torch.unique(
        torch.cat(per_sample_symbols),
        sorted=True,
    )
    symbol_mask = torch.stack([~torch.isin(symbol_set, values) for values in per_sample_symbols])
    backbone = BackboneInput(
        image=torch.cat([item.image for item in backbones]),
        recent_actions_image=torch.cat([item.recent_actions_image for item in backbones]),
        action_mask_image=torch.cat([item.action_mask_image for item in backbones]),
        coordinate_image=torch.cat([item.coordinate_image for item in backbones]),
        action_map_available_image=torch.cat(
            [item.action_map_available_image for item in backbones]
        ),
        recent_actions_vector=torch.cat([item.recent_actions_vector for item in backbones]),
        recent_symbol_indexes=recent[0],
        recent_symbol_offsets=recent[1],
        recent_symbol_weights=recent[2],
        coverage_symbol_indexes=coverage[0],
        coverage_symbol_offsets=coverage[1],
        coverage_symbol_weights=coverage[2],
        coverage_symbols_set=symbol_set,
        coverage_symbols_key_mask=symbol_mask,
        step_number=torch.cat([item.step_number for item in backbones]),
    )
    return TraceNetRequest(
        backbone,
        torch.cat([request.pixel_action_maps for request in requests]),
        first.impossible_action_reward,
    )


def _join_bags(bags: tuple[tuple[Tensor, Tensor], ...]) -> tuple[Tensor, Tensor, Tensor]:
    offsets = []
    offset = 0
    for indexes, _weights in bags:
        offsets.append(offset)
        offset += indexes.numel()
    return (
        torch.cat([indexes for indexes, _weights in bags]),
        torch.tensor(offsets, dtype=torch.long, device=bags[0][0].device),
        torch.cat([weights for _indexes, weights in bags]),
    )


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
        action_mask_image=request.backbone.action_mask_image[..., top:bottom, left:right],
        coordinate_image=request.backbone.coordinate_image[..., top:bottom, left:right],
        action_map_available_image=request.backbone.action_map_available_image[
            ..., top:bottom, left:right
        ],
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
