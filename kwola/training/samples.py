"""Deterministic reconstruction of training tensors from recorded traces."""

import random
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from kwola.agent.model_backbone import BackboneInput
from kwola.agent.tracenet import TraceNetRequest
from kwola.domain.actions import ActionChannel, ActionKind
from kwola.storage import LmdbRunStore

from .action_masks import action_masks
from .cache import SampleCache
from .geometry import Crop, action_crop, centered_crop, random_crop
from .image_cache import DecodedImageCache
from .sample_features import (
    action_index_for_trace,
    coverage_symbols,
    cursor_vector,
    execution_features,
    future_symbols,
    recent_actions,
    recent_symbols,
    reward_mask,
    scaled_action,
    weighted_symbol_bags,
)
from .trace_index import TraceIndex, cache_payload, trace_order

ACTION_KINDS = tuple(ActionKind)


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    request: TraceNetRequest
    next_request: TraceNetRequest
    present_rewards: Tensor
    next_state_valid: Tensor
    action_indexes: Tensor
    action_x: Tensor
    action_y: Tensor
    sample_ids: tuple[str, ...]
    reward_pixel_masks: Tensor | None = None
    execution_features: Tensor | None = None
    cursors: Tensor | None = None


@dataclass(frozen=True, slots=True)
class _PreparedSample:
    key: str
    trace: dict[str, Any]
    image: NDArray[np.float32]
    crop: Crop


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
        freeze_records: bool = False,
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
        self._images = DecodedImageCache(decoded_image_cache_size, image_downscale_ratio)
        self._freeze_records = freeze_records
        self._trace_snapshot: list[tuple[str, dict[str, Any]]] | None = None
        self._trace_index: TraceIndex | None = None
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
        next_edge: int | None = None,
    ) -> TrainingBatch:
        traces = self._recorded_traces()
        if not traces:
            raise RuntimeError("training requires at least one recorded browser trace")
        selected = [traces[(offset + index) % len(traces)] for index in range(batch_size)]
        index = self._trace_index or TraceIndex.build(traces)
        return self._batch(selected, index, edge, next_edge, device, impossible_reward)

    def prepare_cache(self, workers: int = 0) -> int:
        traces = self._recorded_traces()
        _validate_screenshots(self._run_dir, traces, workers)
        return len(traces)

    def prepare_step(self, step_id: str, workers: int = 0) -> int:
        traces = list(self._store.scan_prefix("traces", f"{step_id}-trace-"))
        trace_ids = tuple(key for key, _trace in traces)
        self._cache.get_or_rebuild(step_id, partial(cache_payload, trace_ids))
        _validate_screenshots(self._run_dir, traces, workers)
        return len(traces)

    def _recorded_traces(self) -> list[tuple[str, dict[str, Any]]]:
        if self._trace_snapshot is not None:
            return self._trace_snapshot
        traces = sorted(self._store.scan("traces"), key=trace_order)
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
            if not isinstance(cached_ids, list) or any(key not in by_id for key in cached_ids):
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
        current = [self._prepare(item, edge, current=True) for item in selected]
        request = self._request(current, traces, device, impossible_reward, True)
        next_samples, next_valid = traces.next_samples(selected)
        following = [self._prepare(item, next_edge or edge, current=False) for item in next_samples]
        next_request = self._request(following, traces, device, impossible_reward, False)
        coordinates = [
            scaled_action(item.trace, (item.crop.width, item.crop.height), item.crop)
            for item in current
        ]
        x, y = _coordinate_tensors(coordinates, device)
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
            reward_pixel_masks=torch.stack(
                [
                    reward_mask(
                        item.trace,
                        (item.crop.width, item.crop.height),
                        item.crop,
                        item.image[
                            item.crop.top : item.crop.bottom,
                            item.crop.left : item.crop.right,
                        ],
                    )
                    for item in current
                ]
            ).to(device),
            execution_features=(
                torch.tensor(
                    [execution_features(trace) for _key, trace in selected],
                    dtype=torch.float32,
                    device=device,
                )
                if self._execution_prediction
                else None
            ),
            cursors=(
                torch.stack([cursor_vector(trace) for _key, trace in selected]).to(device)
                if self._cursor_prediction
                else None
            ),
        )

    def _request(
        self,
        selected: Sequence[_PreparedSample],
        traces: TraceIndex,
        device: torch.device,
        impossible_reward: float,
        auxiliary: bool,
    ) -> TraceNetRequest:
        images = torch.stack([self._crop_image(item) for item in selected]).to(device)
        masks = torch.stack(
            [
                action_masks(
                    item.trace,
                    (item.crop.image_width, item.crop.image_height),
                    self._channels,
                )[:, item.crop.top : item.crop.bottom, item.crop.left : item.crop.right]
                for item in selected
            ]
        ).to(device)
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
            self._backbone(selected, traces, images, device),
            masks,
            impossible_reward,
            compute_auxiliary=aux.enabled,
            auxiliary_action_type=aux.actions,
            auxiliary_action_x=aux.x,
            auxiliary_action_y=aux.y,
            future_symbol_indexes=aux.indexes,
            future_symbol_offsets=aux.offsets,
            future_symbol_weights=aux.weights,
        )

    def _prepare(
        self, item: tuple[str, dict[str, Any]], edge: int | None, *, current: bool
    ) -> _PreparedSample:
        key, trace = item
        path = self._run_dir / str(trace.get("screenshot_before", trace["screenshot"]))
        image = self._images.decode(path, edge)
        desired = self._crop_size if current else self._next_crop_size
        if edge is not None:
            desired = (edge, edge)
        desired = desired or (image.shape[1], image.shape[0])
        if current:
            x, y = scaled_action(trace, (image.shape[1], image.shape[0]))
            crop = action_crop(
                x,
                y,
                image.shape[1],
                image.shape[0],
                *desired,
                *self._crop_random,
                self._random,
            )
        elif edge is not None:
            crop = centered_crop(
                image.shape[1] / 2,
                image.shape[0] / 2,
                image.shape[1],
                image.shape[0],
                *desired,
            )
        else:
            crop = random_crop(image.shape[1], image.shape[0], *desired, self._random)
        return _PreparedSample(key, trace, image, _effective_crop(crop))

    @staticmethod
    def _crop_image(item: _PreparedSample) -> Tensor:
        crop = item.crop
        image = item.image[crop.top : crop.bottom, crop.left : crop.right]
        return torch.from_numpy(image).unsqueeze(0)

    def _backbone(
        self,
        selected: Sequence[_PreparedSample],
        traces: TraceIndex,
        images: Tensor,
        device: torch.device,
    ) -> BackboneInput:
        batch_size, _, height, width = images.shape
        action_count = len(self._channels or ACTION_KINDS)
        action_images = torch.zeros(batch_size, action_count, height, width, device=device)
        action_vectors = torch.zeros(
            batch_size, self._recent_action_history * action_count, device=device
        )
        for index, item in enumerate(selected):
            recent_actions(
                action_images[index],
                action_vectors[index],
                item.trace,
                traces.step(item.trace),
                (width, height),
                self._channels,
                self._recent_action_history,
                self._recent_action_radius,
                self._recent_action_decay,
                item.crop,
            )
        coverage = [
            coverage_symbols(item.trace, traces.step(item.trace), self._symbol_dictionary_size)
            for item in selected
        ]
        recent = [
            recent_symbols(item.trace, traces.step(item.trace), self._symbol_dictionary_size)
            for item in selected
        ]
        recent_tensors = weighted_symbol_bags(recent, device)
        coverage_tensors = weighted_symbol_bags(coverage, device)
        symbol_set, symbol_mask = _attention_symbols(coverage, device)
        steps = torch.tensor(
            [int(item.trace["index"]) for item in selected],
            dtype=torch.float32,
            device=device,
        )
        return BackboneInput(
            images,
            action_images,
            action_vectors,
            *recent_tensors,
            *coverage_tensors,
            symbol_set,
            symbol_mask,
            steps,
        )


def _validate_screenshots(
    run_dir: Path, traces: Sequence[tuple[str, dict[str, Any]]], workers: int
) -> None:
    validate = partial(_validate_screenshot, run_dir)
    if workers > 0:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tuple(executor.map(validate, (trace for _key, trace in traces)))
        return
    for _key, trace in traces:
        validate(trace)


def _validate_screenshot(run_dir: Path, trace: Mapping[str, Any]) -> None:
    path = run_dir / str(trace.get("screenshot_before", trace["screenshot"]))
    if not path.is_file() or not path.read_bytes():
        raise ValueError(f"missing or empty screenshot blob: {path}")


def _auxiliary_tensors(
    selected: Sequence[_PreparedSample],
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
        x, y = _coordinate_tensors(coordinates, device)
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


def _coordinate_tensors(
    coordinates: Sequence[tuple[int, int]], device: torch.device
) -> tuple[Tensor, Tensor]:
    return (
        torch.tensor([item[0] for item in coordinates], device=device),
        torch.tensor([item[1] for item in coordinates], device=device),
    )


def _attention_symbols(
    bags: Sequence[tuple[tuple[int, float], ...]], device: torch.device
) -> tuple[Tensor, Tensor]:
    values = [tuple(value for value, _weight in bag) for bag in bags]
    symbols = tuple(sorted({value for bag in values for value in bag}))
    symbol_set = torch.tensor(symbols, dtype=torch.long, device=device)
    mask = torch.tensor(
        [[symbol not in bag for symbol in symbols] for bag in values],
        dtype=torch.bool,
        device=device,
    )
    return symbol_set, mask


def _effective_crop(crop: Crop) -> Crop:
    return Crop(
        crop.left,
        crop.top,
        min(crop.right, crop.image_width),
        min(crop.bottom, crop.image_height),
        crop.image_width,
        crop.image_height,
    )
