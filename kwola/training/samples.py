"""Deterministic reconstruction of training tensors from recorded traces."""

import random
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from kwola.agent.model_backbone import BackboneInput
from kwola.agent.tracenet import TraceNetRequest
from kwola.domain.actions import ActionChannel, ActionKind
from kwola.storage import LmdbRunStore

from .cache import SampleCache
from .geometry import Crop, action_crop, centered_crop, process_screenshot, random_crop
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


class _DecodedImageCache:
    def __init__(self, capacity: int, downscale_ratio: float) -> None:
        self._capacity = capacity
        self._downscale_ratio = downscale_ratio
        self._values: OrderedDict[tuple[str, int | None], NDArray[np.float32]] = OrderedDict()

    def decode(self, path: Path, edge: int | None) -> NDArray[np.float32]:
        key = (str(path), edge)
        cached = self._values.get(key)
        if cached is not None:
            self._values.move_to_end(key)
            return cached
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"invalid screenshot blob: {path}")
        decoded_uint8 = cast(NDArray[np.uint8], decoded)
        if edge is None:
            image = process_screenshot(decoded_uint8, self._downscale_ratio)
        else:
            resized = cv2.resize(decoded_uint8, (edge, edge), interpolation=cv2.INTER_AREA)
            image = cast(NDArray[np.float32], resized[:, :, 0].astype(np.float32) / 255.0)
        if self._capacity:
            self._values[key] = image
            self._values.move_to_end(key)
            while len(self._values) > self._capacity:
                self._values.popitem(last=False)
        return image


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
        self._images = _DecodedImageCache(decoded_image_cache_size, image_downscale_ratio)
        self._freeze_records = freeze_records
        self._trace_snapshot: list[tuple[str, dict[str, Any]]] | None = None
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
        return self._batch(selected, traces, edge, next_edge, device, impossible_reward)

    def prepare_cache(self, workers: int = 0) -> int:
        traces = self._recorded_traces()
        self._validate_screenshots(traces, workers)
        return len(traces)

    def prepare_step(self, step_id: str, workers: int = 0) -> int:
        traces = list(self._store.scan_prefix("traces", f"{step_id}-trace-"))
        trace_ids = tuple(key for key, _trace in traces)
        self._cache.get_or_rebuild(step_id, partial(_cache_payload, trace_ids))
        self._validate_screenshots(traces, workers)
        return len(traces)

    def _validate_screenshots(
        self, traces: Sequence[tuple[str, dict[str, Any]]], workers: int
    ) -> None:
        if workers > 0:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                tuple(executor.map(self._validate_screenshot, (trace for _key, trace in traces)))
            return
        for _key, trace in traces:
            self._validate_screenshot(trace)

    def _validate_screenshot(self, trace: Mapping[str, Any]) -> None:
        path = self._run_dir / str(trace.get("screenshot_before", trace["screenshot"]))
        if not path.is_file() or not path.read_bytes():
            raise ValueError(f"missing or empty screenshot blob: {path}")

    def _recorded_traces(self) -> list[tuple[str, dict[str, Any]]]:
        if self._trace_snapshot is not None:
            return self._trace_snapshot
        traces = sorted(self._store.scan("traces"), key=_trace_order)
        groups: dict[str, list[str]] = {}
        by_id = dict(traces)
        for key, trace in traces:
            groups.setdefault(str(trace["step_id"]), []).append(key)
        prepared: list[tuple[str, dict[str, Any]]] = []
        for step_id, trace_ids in groups.items():
            payload, _rebuilt = self._cache.get_or_rebuild(
                step_id, partial(_cache_payload, tuple(trace_ids))
            )
            cached_ids = payload.get("trace_ids", [])
            if not isinstance(cached_ids, list) or any(key not in by_id for key in cached_ids):
                self._cache.invalidate(step_id)
                cached_ids = trace_ids
            prepared.extend((key, by_id[key]) for key in cached_ids)
        if self._freeze_records:
            self._trace_snapshot = prepared
        return prepared

    def _batch(
        self,
        selected: Sequence[tuple[str, dict[str, Any]]],
        traces: Sequence[tuple[str, dict[str, Any]]],
        edge: int | None,
        next_edge: int | None,
        device: torch.device,
        impossible_reward: float,
    ) -> TrainingBatch:
        current = [self._prepare(item, edge, current=True) for item in selected]
        request = self._request(current, traces, device, impossible_reward, True)
        next_samples, next_valid = _next_samples(selected, traces)
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
            execution_features=torch.tensor(
                [execution_features(trace) for _key, trace in selected],
                dtype=torch.float32,
                device=device,
            ),
            cursors=torch.stack([cursor_vector(trace) for _key, trace in selected]).to(device),
        )

    def _request(
        self,
        selected: Sequence[_PreparedSample],
        traces: Sequence[tuple[str, dict[str, Any]]],
        device: torch.device,
        impossible_reward: float,
        auxiliary: bool,
    ) -> TraceNetRequest:
        images = torch.stack([self._crop_image(item) for item in selected]).to(device)
        masks = torch.stack(
            [
                _action_masks(
                    item.trace,
                    (item.crop.image_width, item.crop.image_height),
                    self._channels,
                )[:, item.crop.top : item.crop.bottom, item.crop.left : item.crop.right]
                for item in selected
            ]
        ).to(device)
        actions = torch.tensor(
            [action_index_for_trace(item.trace, self._channels) for item in selected],
            device=device,
        )
        coordinates = [
            scaled_action(item.trace, (item.crop.width, item.crop.height), item.crop)
            for item in selected
        ]
        x, y = _coordinate_tensors(coordinates, device)
        future = [
            future_symbols(item.trace, traces, self._symbol_dictionary_size) for item in selected
        ]
        indexes, offsets, weights = weighted_symbol_bags(future, device)
        return TraceNetRequest(
            self._backbone(selected, traces, images, device),
            masks,
            impossible_reward,
            compute_auxiliary=auxiliary,
            auxiliary_action_type=actions if auxiliary else None,
            auxiliary_action_x=x if auxiliary else None,
            auxiliary_action_y=y if auxiliary else None,
            future_symbol_indexes=indexes if auxiliary else None,
            future_symbol_offsets=offsets if auxiliary else None,
            future_symbol_weights=weights if auxiliary else None,
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
        traces: Sequence[tuple[str, dict[str, Any]]],
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
                traces,
                (width, height),
                self._channels,
                self._recent_action_history,
                self._recent_action_radius,
                self._recent_action_decay,
                item.crop,
            )
        coverage = [
            coverage_symbols(item.trace, traces, self._symbol_dictionary_size) for item in selected
        ]
        recent = [
            recent_symbols(item.trace, traces, self._symbol_dictionary_size) for item in selected
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


def _trace_order(item: tuple[str, Mapping[str, Any]]) -> tuple[str, int]:
    return str(item[1]["step_id"]), int(item[1]["index"])


def _cache_payload(trace_ids: Sequence[str]) -> dict[str, Any]:
    return {"trace_ids": list(trace_ids)}


def _next_samples(
    selected: Sequence[tuple[str, dict[str, Any]]],
    traces: Sequence[tuple[str, dict[str, Any]]],
) -> tuple[list[tuple[str, dict[str, Any]]], list[bool]]:
    lookup = {(str(trace["step_id"]), int(trace["index"])): (key, trace) for key, trace in traces}
    samples = []
    validity = []
    for current in selected:
        trace = current[1]
        next_trace = lookup.get((str(trace["step_id"]), int(trace["index"]) + 1))
        samples.append(next_trace or current)
        validity.append(next_trace is not None)
    return samples, validity


def _coordinate_tensors(
    coordinates: Sequence[tuple[int, int]], device: torch.device
) -> tuple[Tensor, Tensor]:
    return (
        torch.tensor([item[0] for item in coordinates], device=device),
        torch.tensor([item[1] for item in coordinates], device=device),
    )


def _action_masks(
    trace: Mapping[str, Any], size: tuple[int, int], channels: tuple[ActionChannel, ...] | None
) -> Tensor:
    width, height = size
    targets = trace.get("action_targets")
    if not isinstance(targets, list) or not targets:
        return torch.ones(len(channels or ACTION_KINDS), height, width)
    masks = torch.zeros(len(channels or ACTION_KINDS), height, width)
    for target in targets:
        _paint_target(masks, target, trace.get("viewport", [width, height]), channels)
    return masks if bool(masks.any()) else torch.ones_like(masks)


def _paint_target(
    masks: Tensor,
    target: Mapping[str, Any],
    viewport: Sequence[int],
    channels: tuple[ActionChannel, ...] | None,
) -> None:
    height, width = masks.shape[-2:]
    left, top, right, bottom = (int(value) for value in target["bounds"])
    x1 = min(width - 1, max(0, left * width // int(viewport[0])))
    y1 = min(height - 1, max(0, top * height // int(viewport[1])))
    x2 = min(width, max(x1 + 1, right * width // int(viewport[0])))
    y2 = min(height, max(y1 + 1, bottom * height // int(viewport[1])))
    for index, channel in enumerate(channels or _generic_channels()):
        if _target_supports(target, channel):
            masks[index, y1:y2, x1:x2] = 1


def _generic_channels() -> tuple[ActionChannel, ...]:
    return tuple(
        ActionChannel(
            kind.value,
            kind,
            1.0,
            text_strategy="letters" if kind is ActionKind.TYPE else None,
            direction="up" if kind is ActionKind.SCROLL else None,
        )
        for kind in ACTION_KINDS
    )


def _target_supports(target: Mapping[str, Any], channel: ActionChannel) -> bool:
    if channel.kind in {ActionKind.CLICK, ActionKind.DOUBLE_CLICK}:
        return bool(target.get("click"))
    if channel.kind is ActionKind.RIGHT_CLICK:
        return bool(target.get("right_click"))
    if channel.kind in {ActionKind.CLEAR, ActionKind.TYPE}:
        return bool(target.get("type"))
    return bool(target.get("scroll_up" if channel.direction == "up" else "scroll_down"))


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
