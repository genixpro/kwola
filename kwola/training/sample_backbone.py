"""Spatial and symbolic backbone inputs for recorded samples."""

from collections.abc import Sequence

import torch
from torch import Tensor

from kwola.agent.model_backbone import BackboneInput
from kwola.agent.spatial import coordinate_image
from kwola.domain.actions import ActionChannel, ActionKind

from .action_masks import action_map_is_available
from .sample_features import recent_actions, step_symbol_features, weighted_symbol_bags
from .sample_preparation import PreparedSample
from .trace_index import TraceIndex

SymbolBag = tuple[tuple[int, float], ...]
SymbolFeatureCache = dict[str, tuple[SymbolBag, SymbolBag]]


def assemble_backbone(
    selected: Sequence[PreparedSample],
    traces: TraceIndex,
    images: Tensor,
    masks: Tensor,
    device: torch.device,
    symbol_cache: SymbolFeatureCache,
    channels: tuple[ActionChannel, ...] | None,
    dictionary_size: int,
    history: int,
    radius: int,
    decay: float,
) -> BackboneInput:
    batch_size, _, height, width = images.shape
    action_count = len(channels or tuple(ActionKind))
    action_images = torch.zeros(batch_size, action_count, height, width, device=device)
    action_vectors = torch.zeros(batch_size, history * action_count, device=device)
    for index, item in enumerate(selected):
        recent_actions(
            action_images[index],
            action_vectors[index],
            item.trace,
            traces.step(item.trace),
            (width, height),
            channels,
            history,
            radius,
            decay,
            item.crop,
        )
    coverage, recent = _cached_symbol_features(selected, traces, dictionary_size, symbol_cache)
    symbol_set, symbol_mask = _attention_symbols(coverage, device)
    steps = torch.tensor(
        [int(item.trace["index"]) for item in selected], dtype=torch.float32, device=device
    )
    coordinates = torch.stack(
        [
            coordinate_image(
                item.crop.width,
                item.crop.height,
                full_width=item.crop.image_width,
                full_height=item.crop.image_height,
                left=item.crop.left,
                top=item.crop.top,
                device=device,
            )
            for item in selected
        ]
    )
    availability = (
        torch.tensor(
            [float(action_map_is_available(item.trace, channels)) for item in selected],
            dtype=images.dtype,
            device=device,
        )
        .reshape(batch_size, 1, 1, 1)
        .expand(batch_size, 1, height, width)
    )
    return BackboneInput(
        images,
        action_images,
        masks,
        coordinates,
        availability,
        action_vectors,
        *weighted_symbol_bags(recent, device),
        *weighted_symbol_bags(coverage, device),
        symbol_set,
        symbol_mask,
        steps,
    )


def _cached_symbol_features(
    selected: Sequence[PreparedSample],
    traces: TraceIndex,
    dictionary_size: int,
    cache: SymbolFeatureCache,
) -> tuple[list[SymbolBag], list[SymbolBag]]:
    features = []
    for item in selected:
        cached = cache.get(item.key)
        if cached is None:
            cache.update(step_symbol_features(traces.step(item.trace), dictionary_size))
            cached = cache[item.key]
        features.append(cached)
    return [item[0] for item in features], [item[1] for item in features]


def _attention_symbols(
    bags: Sequence[tuple[tuple[int, float], ...]], device: torch.device
) -> tuple[Tensor, Tensor]:
    values = [frozenset(value for value, _weight in bag) for bag in bags]
    symbols = tuple(sorted({value for bag in values for value in bag}))
    symbol_set = torch.tensor(symbols, dtype=torch.long, device=device)
    mask = torch.tensor(
        [[symbol not in bag for symbol in symbols] for bag in values],
        dtype=torch.bool,
        device=device,
    )
    return symbol_set, mask
