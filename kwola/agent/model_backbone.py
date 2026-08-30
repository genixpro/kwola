"""TraceNet visual, symbol, attention, and stamp feature backbone."""

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from kwola.config.models import ModelConfig

from .normalization import SpatialGroupNorm


@dataclass(frozen=True, slots=True)
class BackboneInput:
    image: Tensor
    recent_actions_image: Tensor
    recent_actions_vector: Tensor
    recent_symbol_indexes: Tensor
    recent_symbol_offsets: Tensor
    recent_symbol_weights: Tensor
    coverage_symbol_indexes: Tensor
    coverage_symbol_offsets: Tensor
    coverage_symbol_weights: Tensor
    coverage_symbols_set: Tensor
    coverage_symbols_key_mask: Tensor
    step_number: Tensor


@dataclass(frozen=True, slots=True)
class BackboneOutput:
    merged: Tensor
    stamp: Tensor
    height: int
    width: int


class TraceNetBackbone(nn.Module):
    def __init__(self, config: ModelConfig, num_actions: int) -> None:
        super().__init__()
        self.config = config
        self.stamp_size = config.additional_stamp_edge**2 * config.additional_stamp_depth
        self.recent_symbols = nn.EmbeddingBag(
            config.symbol_dictionary_size, config.symbol_embedding_size, mode="sum"
        )
        self.attention_keys = nn.Embedding(
            config.symbol_dictionary_size, config.symbol_embedding_size
        )
        self.attention_values = nn.Embedding(
            config.symbol_dictionary_size, config.symbol_embedding_size
        )
        self.attention = nn.MultiheadAttention(config.symbol_embedding_size, 2)
        self.query_projection = nn.Linear(config.pixel_features, config.symbol_embedding_size)
        self.recent_symbol_projection = nn.Sequential(
            nn.Linear(config.symbol_embedding_size, self.stamp_size - 1), nn.ELU()
        )
        self.recent_action_projection = nn.Sequential(
            nn.Linear(config.recent_action_history * num_actions, config.recent_action_features),
            nn.ELU(),
        )
        self.visual = _visual_layers(config, num_actions)

    @property
    def merged_features(self) -> int:
        return (
            self.config.pixel_features
            + self.config.symbol_embedding_size
            + self.config.additional_stamp_depth
            + self.config.recent_action_features
        )

    def forward(self, data: BackboneInput) -> BackboneOutput:
        visual_input = torch.cat([data.image, data.recent_actions_image], dim=1)
        pixels = self.visual(visual_input)
        batch, _, height, width = pixels.shape
        recent = self._recent_symbol_bag(data)
        attended = self._attention(data, pixels, batch, height, width)
        actions = self.recent_action_projection(data.recent_actions_vector)
        additional = torch.cat(
            [
                torch.log10(data.step_number + torch.ones_like(data.step_number)).reshape(-1, 1),
                self.recent_symbol_projection(recent),
            ],
            dim=1,
        )
        stamp, stamp_layer = self._stamp(additional, height, width)
        action_image = actions.reshape(batch, self.config.recent_action_features, 1, 1)
        action_image = action_image.repeat(1, 1, height, width)
        merged = torch.cat([attended, action_image, stamp_layer, pixels], dim=1)
        return BackboneOutput(merged, stamp, height, width)

    def _recent_symbol_bag(self, data: BackboneInput) -> Tensor:
        recent = self.recent_symbols(
            data.recent_symbol_indexes,
            data.recent_symbol_offsets,
            per_sample_weights=data.recent_symbol_weights,
        )
        return nn.functional.normalize(recent, dim=1)

    def _attention(
        self, data: BackboneInput, pixels: Tensor, batch: int, height: int, width: int
    ) -> Tensor:
        transposed = pixels.transpose(1, 3).transpose(0, 2)
        queries = self.query_projection(
            transposed.reshape(batch * height * width, self.config.pixel_features)
        ).reshape(-1, batch, self.config.symbol_embedding_size)
        keys = self.attention_keys(data.coverage_symbols_set)
        values = self.attention_values(data.coverage_symbols_set)
        keys = keys.reshape(-1, 1, self.config.symbol_embedding_size).repeat(1, batch, 1)
        values = values.reshape(-1, 1, self.config.symbol_embedding_size).repeat(1, batch, 1)
        result, _weights = self.attention(
            queries, keys, values, key_padding_mask=data.coverage_symbols_key_mask
        )
        features = result.reshape(height, width, batch, self.config.symbol_embedding_size)
        return cast(Tensor, features.transpose(0, 2).transpose(1, 3))

    def _stamp(self, additional: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
        config = self.config
        stamp = additional.reshape(
            -1,
            config.additional_stamp_depth,
            config.additional_stamp_edge,
            config.additional_stamp_edge,
        )
        tiled = stamp.repeat(
            1,
            1,
            height // config.additional_stamp_edge + 1,
            width // config.additional_stamp_edge + 1,
        )
        return stamp, tiled[:, :, :height, :width]


def _visual_layers(config: ModelConfig, num_actions: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    inputs = 1 + num_actions
    for layer in config.layers[:4]:
        modules.extend(
            [
                nn.Conv2d(
                    inputs,
                    layer.kernels,
                    layer.kernel_size,
                    layer.stride,
                    layer.padding,
                    layer.dilation,
                ),
                nn.ELU(),
                SpatialGroupNorm(layer.kernels),
            ]
        )
        inputs = layer.kernels
    return nn.Sequential(*modules)
