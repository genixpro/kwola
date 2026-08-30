"""Observation encoding and action-space masks for TraceNet inference."""

import cv2
import numpy as np
import torch
from torch import Tensor

from kwola.config.models import ModelConfig
from kwola.domain.actions import ActionKind, ActionMap, ActionTarget
from kwola.domain.observations import Observation

from .model_backbone import BackboneInput
from .tracenet import TraceNetRequest

ACTION_KINDS = tuple(ActionKind)


class ObservationEncoder:
    def __init__(self, config: ModelConfig, edge: int, impossible_reward: float) -> None:
        self._config = config
        self._edge = edge
        self._impossible_reward = impossible_reward

    def encode(self, observation: Observation, device: torch.device) -> TraceNetRequest:
        image = self._image(observation).to(device)
        masks = action_masks(observation.action_map, self._edge).unsqueeze(0).to(device)
        symbols = _symbols(observation, self._config.symbol_dictionary_size, device)
        offsets = torch.tensor([0], dtype=torch.long, device=device)
        weights = torch.ones(len(symbols), dtype=torch.float32, device=device)
        recent_images = torch.zeros(1, len(ACTION_KINDS), self._edge, self._edge, device=device)
        recent_vector = torch.zeros(
            1, self._config.recent_action_history * len(ACTION_KINDS), device=device
        )
        backbone = BackboneInput(
            image,
            recent_images,
            recent_vector,
            symbols,
            offsets,
            weights,
            symbols,
            offsets,
            weights,
            symbols,
            torch.zeros(1, len(symbols), dtype=torch.bool, device=device),
            torch.tensor([1.0], device=device),
        )
        return TraceNetRequest(backbone, masks, self._impossible_reward)

    def _image(self, observation: Observation) -> Tensor:
        encoded = np.frombuffer(observation.screenshot, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError("observation contains an invalid screenshot")
        resized = cv2.resize(decoded, (self._edge, self._edge), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized.astype(np.float32) / 255.0).reshape(
            1, 1, self._edge, self._edge
        )


def action_masks(action_map: ActionMap, edge: int) -> Tensor:
    masks = torch.zeros(len(ACTION_KINDS), edge, edge)
    for target in action_map.targets:
        _paint(masks, target, action_map, edge)
    return masks if bool(masks.any()) else torch.ones_like(masks)


def _paint(masks: Tensor, target: ActionTarget, action_map: ActionMap, edge: int) -> None:
    x1 = min(edge - 1, max(0, target.left * edge // action_map.viewport_width))
    y1 = min(edge - 1, max(0, target.top * edge // action_map.viewport_height))
    x2 = min(edge, max(x1 + 1, target.right * edge // action_map.viewport_width))
    y2 = min(edge, max(y1 + 1, target.bottom * edge // action_map.viewport_height))
    capabilities = {
        ActionKind.CLICK: target.can_click,
        ActionKind.DOUBLE_CLICK: target.can_click,
        ActionKind.RIGHT_CLICK: target.can_right_click,
        ActionKind.CLEAR: target.can_type,
        ActionKind.TYPE: target.can_type,
        ActionKind.SCROLL: target.can_scroll,
    }
    for kind, enabled in capabilities.items():
        if enabled:
            masks[ACTION_KINDS.index(kind), y1:y2, x1:x2] = 1


def _symbols(observation: Observation, size: int, device: torch.device) -> Tensor:
    raw = (*observation.branch_symbols, *observation.network_symbols)
    values = sorted({int(value) % size for value in raw} or {0})
    return torch.tensor(values, dtype=torch.long, device=device)
