"""Deterministic reconstruction of training tensors from recorded traces."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor

from kwola.agent.model_backbone import BackboneInput
from kwola.agent.tracenet import TraceNetRequest
from kwola.domain.actions import ActionKind
from kwola.storage import LmdbRunStore

from .cache import SampleCache

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
    ) -> None:
        self._run_dir = run_dir
        self._store = store
        self._symbol_dictionary_size = symbol_dictionary_size
        self._discount_rate = discount_rate
        self._max_discounted_reward = max_discounted_reward
        self._cache = SampleCache(store, cache_version)

    def assemble(
        self,
        *,
        batch_size: int,
        edge: int,
        device: torch.device,
        impossible_reward: float,
        offset: int = 0,
    ) -> TrainingBatch:
        traces = self._recorded_traces()
        if not traces:
            raise RuntimeError("training requires at least one recorded browser trace")
        selected = [traces[(offset + index) % len(traces)] for index in range(batch_size)]
        return self._batch(selected, traces, edge, device, impossible_reward)

    def prepare_cache(self) -> int:
        return len(self._recorded_traces())

    def _recorded_traces(self) -> list[tuple[str, dict[str, Any]]]:
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
        return prepared

    def _batch(
        self,
        selected: Sequence[tuple[str, dict[str, Any]]],
        all_traces: Sequence[tuple[str, dict[str, Any]]],
        edge: int,
        device: torch.device,
        impossible_reward: float,
    ) -> TrainingBatch:
        request = self._request(selected, edge, device, impossible_reward)
        next_samples, next_valid = _next_samples(selected, all_traces)
        next_request = self._request(next_samples, edge, device, impossible_reward)
        x, y = _action_coordinates(selected, edge, device)
        present = torch.tensor([trace["reward"] for _key, trace in selected], device=device)
        return TrainingBatch(
            request,
            next_request,
            present,
            torch.tensor(next_valid, dtype=torch.bool, device=device),
            torch.tensor([_action_index(trace) for _key, trace in selected], device=device),
            x,
            y,
            tuple(key for key, _trace in selected),
        )

    def _request(
        self,
        selected: Sequence[tuple[str, dict[str, Any]]],
        edge: int,
        device: torch.device,
        impossible_reward: float,
    ) -> TraceNetRequest:
        images = torch.stack([self._image(trace, edge) for _key, trace in selected]).to(device)
        masks = torch.stack([_action_masks(trace, edge) for _key, trace in selected]).to(device)
        actions = torch.tensor([_action_index(trace) for _key, trace in selected], device=device)
        x, y = _action_coordinates(selected, edge, device)
        backbone = self._backbone(selected, images, actions, x, y, device)
        return TraceNetRequest(backbone, masks, impossible_reward)

    def _image(self, trace: Mapping[str, Any], edge: int) -> Tensor:
        path = self._run_dir / str(trace["screenshot"])
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError(f"invalid screenshot blob: {path}")
        resized = cv2.resize(decoded, (edge, edge), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0)

    def _backbone(
        self,
        selected: Sequence[tuple[str, dict[str, Any]]],
        images: Tensor,
        actions: Tensor,
        x: Tensor,
        y: Tensor,
        device: torch.device,
    ) -> BackboneInput:
        batch_size, _, edge, _ = images.shape
        recent_images = torch.zeros(batch_size, len(ACTION_KINDS), edge, edge, device=device)
        recent_vectors = torch.zeros(batch_size, 5 * len(ACTION_KINDS), device=device)
        for index in range(batch_size):
            recent_images[index, actions[index], y[index], x[index]] = 1
            recent_vectors[index, actions[index]] = 1
        bags = [_symbols(trace, self._symbol_dictionary_size) for _key, trace in selected]
        indexes, offsets, weights = _symbol_bags(bags, device)
        symbol_set, symbol_mask = _attention_symbols(bags, batch_size, device)
        steps = torch.tensor(
            [int(trace.get("index", 0)) + 1 for _key, trace in selected],
            dtype=torch.float32,
            device=device,
        )
        return BackboneInput(
            images,
            recent_images,
            recent_vectors,
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


def _action_index(trace: Mapping[str, Any]) -> int:
    kind = ActionKind(str(trace["action"]["kind"]))
    return ACTION_KINDS.index(kind)


def _action_coordinates(
    samples: Sequence[tuple[str, dict[str, Any]]], edge: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    coordinates = []
    for _key, trace in samples:
        viewport = trace.get("viewport", [edge, edge])
        action = trace["action"]
        x = min(edge - 1, max(0, int(action["x"]) * edge // int(viewport[0])))
        y = min(edge - 1, max(0, int(action["y"]) * edge // int(viewport[1])))
        coordinates.append((x, y))
    return (
        torch.tensor([item[0] for item in coordinates], device=device),
        torch.tensor([item[1] for item in coordinates], device=device),
    )


def _action_masks(trace: Mapping[str, Any], edge: int) -> Tensor:
    targets = trace.get("action_targets")
    if not isinstance(targets, list) or not targets:
        return torch.ones(len(ACTION_KINDS), edge, edge)
    viewport = trace.get("viewport", [edge, edge])
    masks = torch.zeros(len(ACTION_KINDS), edge, edge)
    for target in targets:
        _paint_target(masks, target, viewport, edge)
    return masks if bool(masks.any()) else torch.ones_like(masks)


def _paint_target(
    masks: Tensor,
    target: Mapping[str, Any],
    viewport: Sequence[int],
    edge: int,
) -> None:
    left, top, right, bottom = (int(value) for value in target["bounds"])
    x1 = min(edge - 1, max(0, left * edge // int(viewport[0])))
    y1 = min(edge - 1, max(0, top * edge // int(viewport[1])))
    x2 = min(edge, max(x1 + 1, right * edge // int(viewport[0])))
    y2 = min(edge, max(y1 + 1, bottom * edge // int(viewport[1])))
    capabilities = {
        ActionKind.CLICK: bool(target.get("click")),
        ActionKind.DOUBLE_CLICK: bool(target.get("click")),
        ActionKind.RIGHT_CLICK: bool(target.get("right_click")),
        ActionKind.CLEAR: bool(target.get("type")),
        ActionKind.TYPE: bool(target.get("type")),
        ActionKind.SCROLL: bool(target.get("scroll")),
    }
    for kind, enabled in capabilities.items():
        if enabled:
            masks[ACTION_KINDS.index(kind), y1:y2, x1:x2] = 1


def _symbols(trace: Mapping[str, Any], dictionary_size: int) -> tuple[int, ...]:
    raw = list(trace.get("branch_symbols", [])) + list(trace.get("network_symbols", []))
    symbols = {int(value) % dictionary_size for value in raw}
    return tuple(sorted(symbols or {0}))


def _symbol_bags(
    bags: Sequence[tuple[int, ...]], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    indexes = torch.tensor([value for bag in bags for value in bag], device=device)
    offsets = torch.tensor(np.cumsum([0, *[len(bag) for bag in bags[:-1]]]), device=device)
    weights = torch.ones(len(indexes), dtype=torch.float32, device=device)
    return indexes.long(), offsets.long(), weights


def _attention_symbols(
    bags: Sequence[tuple[int, ...]], batch_size: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    symbols = tuple(sorted({value for bag in bags for value in bag}))
    symbol_set = torch.tensor(symbols, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, len(symbols), dtype=torch.bool, device=device)
    return symbol_set, mask
