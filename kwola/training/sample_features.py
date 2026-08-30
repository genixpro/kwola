"""Temporal symbol, action-history, and supervised sample features."""

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from kwola.domain.actions import ActionChannel, ActionKind

from .geometry import Crop, scale_coordinate

ACTION_KINDS = tuple(ActionKind)


def weighted_symbol_bags(
    bags: Sequence[tuple[tuple[int, float], ...]], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    indexes = torch.tensor([value for bag in bags for value, _weight in bag], device=device)
    offsets = torch.tensor(np.cumsum([0, *[len(bag) for bag in bags[:-1]]]), device=device)
    weights = torch.tensor(
        [weight for bag in bags for _value, weight in bag],
        dtype=torch.float32,
        device=device,
    )
    return indexes.long(), offsets.long(), weights


def coverage_symbols(
    trace: Mapping[str, Any], traces: Sequence[tuple[str, dict[str, Any]]], size: int
) -> tuple[tuple[int, float], ...]:
    index = int(trace["index"])
    values = {
        int(symbol) % size
        for item in _step_traces(trace, traces)
        if int(item["index"]) < index
        for symbol in item.get("branch_symbols", [])
    }
    return tuple((value, 1.0) for value in sorted(values or {0}))


def recent_symbols(
    trace: Mapping[str, Any], traces: Sequence[tuple[str, dict[str, Any]]], size: int
) -> tuple[tuple[int, float], ...]:
    current = int(trace["index"])
    weights: dict[int, float] = {}
    for item in _step_traces(trace, traces):
        age = current - int(item["index"])
        if age <= 0:
            continue
        weight = 0.9 ** (age - 1)
        for symbol in item.get("branch_symbols", []):
            mapped = int(symbol) % size
            weights[mapped] = min(1.0, weights.get(mapped, 0.0) + weight)
    return tuple(sorted(weights.items())) or ((0, 1.0),)


def future_symbols(
    trace: Mapping[str, Any], traces: Sequence[tuple[str, dict[str, Any]]], size: int
) -> tuple[tuple[int, float], ...]:
    current = int(trace["index"])
    weights: dict[int, float] = {}
    for item in _step_traces(trace, traces):
        distance = int(item["index"]) - current
        if distance < 0:
            continue
        weight = 0.7**distance
        for symbol in item.get("branch_symbols", []):
            mapped = int(symbol) % size
            weights[mapped] = weights.get(mapped, 0.0) + weight
    return tuple(sorted(weights.items())) or ((0, 1.0),)


def recent_actions(
    image: Tensor,
    vector: Tensor,
    trace: Mapping[str, Any],
    traces: Sequence[tuple[str, dict[str, Any]]],
    size: int | tuple[int, int],
    channels: tuple[ActionChannel, ...] | None,
    history: int,
    radius: int,
    decay: float,
    crop: Crop | None = None,
) -> None:
    prior = [
        item for item in _step_traces(trace, traces) if int(item["index"]) < int(trace["index"])
    ][-history:]
    action_count = image.shape[0]
    for position, item in enumerate(reversed(prior)):
        action_index = action_index_for_trace(item, channels)
        vector[position * action_count + action_index] = 1
        x, y = scaled_action(item, size, crop)
        limit = min(image.shape[-2:])
        _paint_action_circle(image[action_index], x, y, min(radius, limit), decay**position)


def action_index_for_trace(
    trace: Mapping[str, Any], channels: tuple[ActionChannel, ...] | None
) -> int:
    action = trace["action"]
    channel_name = action.get("channel")
    if channels is not None and channel_name:
        names = tuple(channel.name for channel in channels)
        if channel_name in names:
            return names.index(str(channel_name))
    kind = ActionKind(str(action["kind"]))
    if channels is None:
        return ACTION_KINDS.index(kind)
    for index, channel in enumerate(channels):
        if channel.kind is kind:
            return index
    raise ValueError(f"recorded action {kind.value} is not configured")


def scaled_action(
    trace: Mapping[str, Any],
    size: int | tuple[int, int],
    crop: Crop | None = None,
) -> tuple[int, int]:
    width, height = _size(size)
    viewport = trace.get("viewport", [width, height])
    action = trace["action"]
    full_width = width if crop is None else crop.image_width
    full_height = height if crop is None else crop.image_height
    x = scale_coordinate(int(action["x"]), int(viewport[0]), full_width)
    y = scale_coordinate(int(action["y"]), int(viewport[1]), full_height)
    return (
        min(width - 1, max(0, x - (crop.left if crop else 0))),
        min(height - 1, max(0, y - (crop.top if crop else 0))),
    )


def reward_mask(
    trace: Mapping[str, Any],
    size: int | tuple[int, int],
    crop: Crop | None = None,
    processed_image: NDArray[np.float32] | None = None,
) -> Tensor:
    width, height = _size(size)
    x, y = scaled_action(trace, size, crop)
    mask: NDArray[np.uint8] = np.zeros((height, width), dtype=np.uint8)
    viewport = trace.get("viewport", [width, height])
    for target in trace.get("action_targets", []):
        left, top, right, bottom = (int(value) for value in target["bounds"])
        action = trace["action"]
        if left <= int(action["x"]) <= right and top <= int(action["y"]) <= bottom:
            full_width = crop.image_width if crop else width
            full_height = crop.image_height if crop else height
            ratio_x = full_width / int(viewport[0])
            ratio_y = full_height / int(viewport[1])
            x1 = max(0, int(left * ratio_x) - (crop.left if crop else 0))
            y1 = max(0, int(top * ratio_y) - (crop.top if crop else 0))
            x2 = min(width, max(x1 + 1, int(right * ratio_x) - (crop.left if crop else 0)))
            y2 = min(height, max(y1 + 1, int(bottom * ratio_y) - (crop.top if crop else 0)))
            mask[y1:y2, x1:x2] = 1
    mask[max(0, y - 2) : min(height, y + 3), max(0, x - 2) : min(width, x + 3)] = 1
    if processed_image is not None:
        mask = _flood_reward(processed_image, mask, x, y)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=3)
    return torch.from_numpy((blurred > 0.1).astype(np.float32))


def execution_features(trace: Mapping[str, Any]) -> tuple[float, ...]:
    raw = trace.get("execution_features")
    if isinstance(raw, list) and len(raw) == 12:
        return tuple(float(bool(value)) for value in raw)
    return (
        1.0,
        float(bool(trace.get("errors"))),
        float(bool(trace.get("new_errors"))),
        float(bool(trace.get("branch_symbols"))),
        float(bool(trace.get("new_branch_symbols"))),
        float(bool(trace.get("network_symbols"))),
        float(bool(trace.get("new_network_symbols"))),
        float(bool(trace.get("screenshot_changed"))),
        float(bool(trace.get("screenshot_new"))),
        float(trace.get("url_before") != trace.get("url_after")),
        float(bool(trace.get("url_new"))),
        float(bool(trace.get("log_output"))),
    )


CURSORS = (
    "alias",
    "all-scroll",
    "auto",
    "cell",
    "context-menu",
    "col-resize",
    "copy",
    "crosshair",
    "default",
    "e-resize",
    "ew-resize",
    "grab",
    "grabbing",
    "help",
    "move",
    "n-resize",
    "ne-resize",
    "nesw-resize",
    "ns-resize",
    "nw-resize",
    "nwse-resize",
    "no-drop",
    "none",
    "not-allowed",
    "pointer",
    "progress",
    "row-resize",
    "s-resize",
    "se-resize",
    "sw-resize",
    "text",
    "url",
    "w-resize",
    "wait",
    "zoom-in",
    "zoom-out",
    "none",
)


def cursor_vector(trace: Mapping[str, Any]) -> Tensor:
    cursor = str(trace.get("cursor", "none"))
    index = CURSORS.index(cursor) if cursor in CURSORS else CURSORS.index("none")
    result = torch.zeros(len(CURSORS), dtype=torch.float32)
    result[index] = 1
    return result


def _step_traces(
    trace: Mapping[str, Any], traces: Sequence[tuple[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    step_id = str(trace["step_id"])
    return [item for _key, item in traces if str(item["step_id"]) == step_id]


def _paint_action_circle(image: Tensor, x: int, y: int, radius: int, gain: float) -> None:
    yy, xx = torch.meshgrid(
        torch.arange(image.shape[0], device=image.device),
        torch.arange(image.shape[1], device=image.device),
        indexing="ij",
    )
    distance = torch.sqrt((xx - x).square() + (yy - y).square())
    circle = torch.where(
        distance < radius,
        ((radius - distance) / radius * 0.7 + 0.3) * gain,
        torch.zeros_like(distance),
    )
    image.copy_(torch.minimum(torch.ones_like(image), image + circle))


def _size(size: int | tuple[int, int]) -> tuple[int, int]:
    return (size, size) if isinstance(size, int) else size


def _flood_reward(
    image: NDArray[np.float32], allowed: NDArray[np.uint8], x: int, y: int
) -> NDArray[np.uint8]:
    height, width = image.shape
    flooded = np.zeros((height, width), dtype=np.uint8)
    seed_left, seed_right = max(0, x - 2), min(width - 1, x + 2)
    seed_top, seed_bottom = max(0, y - 2), min(height - 1, y + 2)
    quantized = np.rint(image * 100).astype(np.uint8)
    for seed_y in range(seed_top, seed_bottom + 1):
        for seed_x in range(seed_left, seed_right + 1):
            flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
            cv2.floodFill(quantized.copy(), flood_mask, (seed_x, seed_y), (255,), (0,), (0,))
            flooded |= flood_mask[1:-1, 1:-1]
    return np.bitwise_and(flooded, allowed)
