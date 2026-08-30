import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from kwola.agent.tracenet import TraceNetRequest
from kwola.domain.actions import ActionChannel, ActionKind
from kwola.storage import LmdbRunStore

from .action_masks import cached_cropped_action_masks
from .cache import SampleCache
from .image_cache import DecodedImageCache
from .sample_backbone import SymbolFeatureCache, assemble_backbone
from .sample_features import (
    action_index_for_trace,
    coordinate_tensors,
    cursor_index,
    execution_features,
    future_symbols,
    reward_mask,
    scaled_action,
    weighted_symbol_bags,
)
from .sample_preparation import PreparedSample, prepare_sample, validate_screenshots
from .trace_index import TraceIndex, cache_payload, trace_order

ACTION_KINDS = tuple(ActionKind)


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    request: TraceNetRequest
    next_request: TraceNetRequest | None
    present_rewards: Tensor
    next_state_valid: Tensor
    action_indexes: Tensor
    action_x: Tensor
    action_y: Tensor
    sample_ids: tuple[str, ...]
    reward_pixel_masks: Tensor | None = None
    execution_features: Tensor | None = None
    cursors: Tensor | None = None
    next_requests: tuple[TraceNetRequest, ...] = ()


@dataclass(frozen=True, slots=True)
class _AuxiliaryTensors:
    enabled: bool
    actions: Tensor | None
    x: Tensor | None
    y: Tensor | None
    indexes: Tensor | None
    offsets: Tensor | None
    weights: Tensor | None


class RecordedSampleAssembler:
    def __init__(
        self,
        run_dir: Path,
        store: LmdbRunStore,
        *,
        symbol_dictionary_size: int,
        discount_rate: float,
        max_discounted_reward: float,
        cache_version: int,
        channels: tuple[ActionChannel, ...] | None = None,
        recent_action_history: int = 5,
        recent_action_radius: int = 40,
        recent_action_decay: float = 0.8,
        image_downscale_ratio: float = 1.0,
        crop_size: tuple[int, int] | None = None,
        next_crop_size: tuple[int, int] | None = None,
        crop_random: tuple[int, int] = (0, 0),
        decoded_image_cache_size: int = 0,
        decoded_image_cache_directory: Path | None = None,
        compact_cpu_tensors: bool = False,
        freeze_records: bool = False,
        trace_ids: Sequence[str] | None = None,
        enable_trace_prediction: bool = True,
        enable_execution_feature_prediction: bool = True,
        enable_cursor_prediction: bool = True,
        source: random.Random | None = None,
    ) -> None:
        self._run_dir = run_dir
        self._store = store
        self._symbol_dictionary_size = symbol_dictionary_size
        self._discount_rate = discount_rate
        self._max_discounted_reward = max_discounted_reward
        self._cache = SampleCache(store, cache_version)
        self._channels = channels
        self._recent_action_history = recent_action_history
        self._recent_action_radius = recent_action_radius
        self._recent_action_decay = recent_action_decay
        self._crop_size = crop_size
        self._next_crop_size = next_crop_size
        self._crop_random = crop_random
        self._images = DecodedImageCache(
            decoded_image_cache_size, image_downscale_ratio, decoded_image_cache_directory
        )
        self._compact_cpu_tensors = compact_cpu_tensors
        self._freeze_records = freeze_records
        self._trace_ids = frozenset(trace_ids) if trace_ids is not None else None
        self._trace_snapshot: list[tuple[str, dict[str, Any]]] | None = None
        self._trace_index: TraceIndex | None = None
        self._symbol_features: SymbolFeatureCache = {}
        self._trace_prediction = enable_trace_prediction
        self._execution_prediction = enable_execution_feature_prediction
        self._cursor_prediction = enable_cursor_prediction
        self._random = source or random.Random(0)

    def assemble(
        self,
        *,
        batch_size: int,
        edge: int | None = None,
        device: torch.device,
        impossible_reward: float,
        offset: int = 0,
        sample_indexes: Sequence[int] | None = None,
        next_edge: int | None = None,
    ) -> TrainingBatch:
        traces = self._recorded_traces()
        if not traces:
            raise RuntimeError("training requires at least one recorded browser trace")
        if sample_indexes is not None and len(sample_indexes) != batch_size:
            raise ValueError("sample index count must match batch size")
        indexes = sample_indexes or tuple(offset + index for index in range(batch_size))
        selected = [traces[index % len(traces)] for index in indexes]
        index = self._trace_index or TraceIndex.build(traces)
        return self._batch(selected, index, edge, next_edge, device, impossible_reward)

    def trace_count(self) -> int:
        return len(self._recorded_traces())

    def prepare_cache(self, workers: int = 0) -> int:
        traces = self._recorded_traces()
        validate_screenshots(self._run_dir, traces, workers)
        return len(traces)

    def prepare_step(self, step_id: str, workers: int = 0) -> int:
        traces = list(self._store.scan_prefix("traces", f"{step_id}-trace-"))
        trace_ids = tuple(key for key, _trace in traces)
        self._cache.get_or_rebuild(step_id, partial(cache_payload, trace_ids))
        validate_screenshots(self._run_dir, traces, workers)
        return len(traces)

    def _recorded_traces(self) -> list[tuple[str, dict[str, Any]]]:
        if self._trace_snapshot is not None:
            return self._trace_snapshot
        traces = sorted(self._store.scan("traces"), key=trace_order)
        if self._trace_ids is not None:
            traces = [record for record in traces if record[0] in self._trace_ids]
        groups: dict[str, list[str]] = {}
        by_id = dict(traces)
        for key, trace in traces:
            groups.setdefault(str(trace["step_id"]), []).append(key)
        prepared: list[tuple[str, dict[str, Any]]] = []
        for step_id, trace_ids in groups.items():
            payload, _rebuilt = self._cache.get_or_rebuild(
                step_id, partial(cache_payload, tuple(trace_ids))
            )
            cached_ids = payload.get("trace_ids", [])
            if not isinstance(cached_ids, list) or cached_ids != trace_ids:
                self._cache.invalidate(step_id)
                cached_ids = trace_ids
            prepared.extend((key, by_id[key]) for key in cached_ids)
        if self._freeze_records:
            self._trace_snapshot = prepared
            self._trace_index = TraceIndex.build(prepared)
        return prepared

    def _batch(
        self,
        selected: Sequence[tuple[str, dict[str, Any]]],
        traces: TraceIndex,
        edge: int | None,
        next_edge: int | None,
        device: torch.device,
        impossible_reward: float,
    ) -> TrainingBatch:
        symbol_cache = self._symbol_features if self._freeze_records else {}
        action_mask_cache: dict[str, Tensor] = {}
        current = [self._prepare(item, edge, current=True) for item in selected]
        reward_masks = _reward_masks(current, device)
        request = self._request(
            current,
            traces,
            device,
            impossible_reward,
            True,
            symbol_cache,
            action_mask_cache,
            reward_masks,
        )
        next_samples, next_valid = traces.next_samples(selected)
        following = [
            self._prepare(item, next_edge or edge, current=False, full=edge is None)
            for item in next_samples
        ]
        next_requests = tuple(
            self._request(
                [item],
                traces,
                device,
                impossible_reward,
                False,
                symbol_cache,
                action_mask_cache,
            )
            for item in following
        )
        next_request = None
        if edge is not None:
            next_request = self._request(
                following,
                traces,
                device,
                impossible_reward,
                False,
                symbol_cache,
                action_mask_cache,
            )
        coordinates = [
            scaled_action(item.trace, (item.crop.width, item.crop.height), item.crop)
            for item in current
        ]
        x, y = coordinate_tensors(coordinates, device)
        return TrainingBatch(
            request=request,
            next_request=next_request,
            present_rewards=torch.tensor(
                [trace["reward"] for _key, trace in selected], device=device
            ),
            next_state_valid=torch.tensor(next_valid, dtype=torch.bool, device=device),
            action_indexes=torch.tensor(
                [action_index_for_trace(trace, self._channels) for _key, trace in selected],
                device=device,
            ),
            action_x=x,
            action_y=y,
            sample_ids=tuple(key for key, _trace in selected),
            reward_pixel_masks=reward_masks,
            execution_features=_execution_targets(selected, device, self._execution_prediction),
            cursors=_cursor_targets(selected, device, self._cursor_prediction),
            next_requests=next_requests,
        )

    def _request(
        self,
        selected: Sequence[PreparedSample],
        traces: TraceIndex,
        device: torch.device,
        impossible_reward: float,
        auxiliary: bool,
        symbol_cache: SymbolFeatureCache,
        action_mask_cache: dict[str, Tensor],
        auxiliary_pixel_masks: Tensor | None = None,
    ) -> TraceNetRequest:
        images = torch.stack([_crop_image(item) for item in selected]).to(device)
        masks = torch.stack(
            [
                cached_cropped_action_masks(
                    item.key,
                    item.trace,
                    item.crop,
                    self._channels,
                    self._compact_cpu_tensors,
                    action_mask_cache,
                )
                for item in selected
            ]
        ).to(device)
        if masks.dtype is torch.uint8 and device.type != "cpu":
            masks = masks.float()
        aux = _auxiliary_tensors(
            selected,
            traces,
            self._channels,
            self._symbol_dictionary_size,
            device,
            auxiliary,
            self._trace_prediction,
            self._execution_prediction,
            self._cursor_prediction,
        )
        return TraceNetRequest(
            assemble_backbone(
                selected,
                traces,
                images,
                masks,
                device,
                symbol_cache,
                self._channels,
                self._symbol_dictionary_size,
                self._recent_action_history,
                self._recent_action_radius,
                self._recent_action_decay,
            ),
            masks,
            impossible_reward,
            compute_auxiliary=aux.enabled,
            auxiliary_action_type=aux.actions,
            auxiliary_action_x=aux.x,
            auxiliary_action_y=aux.y,
            auxiliary_pixel_masks=auxiliary_pixel_masks,
            future_symbol_indexes=aux.indexes,
            future_symbol_offsets=aux.offsets,
            future_symbol_weights=aux.weights,
        )

    def _prepare(
        self,
        item: tuple[str, dict[str, Any]],
        edge: int | None,
        *,
        current: bool,
        full: bool = False,
    ) -> PreparedSample:
        return prepare_sample(
            self._run_dir,
            self._images,
            self._channels,
            self._crop_size,
            self._next_crop_size,
            self._crop_random,
            self._random,
            item,
            edge,
            current=current,
            full=full,
        )


def _crop_image(item: PreparedSample) -> Tensor:
    crop = item.crop
    image = item.image[crop.top : crop.bottom, crop.left : crop.right]
    return torch.from_numpy(image).unsqueeze(0)


def _reward_masks(selected: Sequence[PreparedSample], device: torch.device) -> Tensor:
    return torch.stack(
        [
            reward_mask(
                item.trace,
                (item.crop.width, item.crop.height),
                item.crop,
                item.image[item.crop.top : item.crop.bottom, item.crop.left : item.crop.right],
            )
            for item in selected
        ]
    ).to(device)


def _execution_targets(
    selected: Sequence[tuple[str, dict[str, Any]]], device: torch.device, enabled: bool
) -> Tensor | None:
    if not enabled:
        return None
    return torch.tensor(
        [execution_features(trace) for _key, trace in selected],
        dtype=torch.float32,
        device=device,
    )


def _cursor_targets(
    selected: Sequence[tuple[str, dict[str, Any]]], device: torch.device, enabled: bool
) -> Tensor | None:
    if not enabled:
        return None
    return torch.tensor(
        [cursor_index(trace) for _key, trace in selected], dtype=torch.long, device=device
    )


def _auxiliary_tensors(
    selected: Sequence[PreparedSample],
    traces: TraceIndex,
    channels: tuple[ActionChannel, ...] | None,
    symbol_dictionary_size: int,
    device: torch.device,
    auxiliary: bool,
    trace_prediction: bool,
    execution_prediction: bool,
    cursor_prediction: bool,
) -> _AuxiliaryTensors:
    enabled = auxiliary and any((trace_prediction, execution_prediction, cursor_prediction))
    actions = None
    x = None
    y = None
    if enabled:
        actions = torch.tensor(
            [action_index_for_trace(item.trace, channels) for item in selected], device=device
        )
        coordinates = [
            scaled_action(item.trace, (item.crop.width, item.crop.height), item.crop)
            for item in selected
        ]
        x, y = coordinate_tensors(coordinates, device)
    indexes = None
    offsets = None
    weights = None
    if auxiliary and trace_prediction:
        future = [
            future_symbols(item.trace, traces.step(item.trace), symbol_dictionary_size)
            for item in selected
        ]
        indexes, offsets, weights = weighted_symbol_bags(future, device)
    return _AuxiliaryTensors(enabled, actions, x, y, indexes, offsets, weights)
