"""TraceNet topology with reward-map and auxiliary prediction ownership."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kwola.config.models import ModelConfig

from .model_backbone import BackboneInput, BackboneOutput, TraceNetBackbone
from .model_heads import TraceNetHeads


@dataclass(frozen=True, slots=True)
class TraceNetRequest:
    backbone: BackboneInput
    pixel_action_maps: Tensor
    impossible_action_reward: float
    compute_rewards: bool = True
    compute_auxiliary: bool = False
    output_stamp: bool = False
    auxiliary_action_type: Tensor | None = None
    auxiliary_action_x: Tensor | None = None
    auxiliary_action_y: Tensor | None = None
    auxiliary_pixel_masks: Tensor | None = None
    future_symbol_indexes: Tensor | None = None
    future_symbol_offsets: Tensor | None = None
    future_symbol_weights: Tensor | None = None


class TraceNet(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        num_actions: int,
        execution_feature_count: int = 12,
        cursor_count: int = 37,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.backbone = TraceNetBackbone(config, num_actions)
        self.heads = TraceNetHeads(
            config,
            self.backbone.merged_features,
            num_actions,
            execution_feature_count,
            cursor_count,
        )

    def forward(self, request: TraceNetRequest) -> dict[str, Tensor]:
        features = self.backbone(request.backbone)
        output: dict[str, Tensor] = {}
        action_values: Tensor | None = None
        if request.compute_rewards:
            present, discounted = self.heads.reward_maps(features.merged)
            action_values = self.heads.action_values(
                present,
                discounted,
                request.pixel_action_maps,
                request.impossible_action_reward,
            )
            output["presentRewards"] = present
            output["discountFutureRewards"] = discounted
            output["actionValues"] = action_values
        if request.output_stamp:
            output["stamp"] = features.stamp.detach()
        self._auxiliary(output, request, features, action_values)
        self._future_embedding(output, request)
        return output

    def _future_embedding(self, output: dict[str, Tensor], request: TraceNetRequest) -> None:
        if request.future_symbol_indexes is None:
            return
        assert request.future_symbol_offsets is not None
        assert request.future_symbol_weights is not None
        output["decayingFutureSymbolEmbedding"] = self.future_symbol_embedding(request).detach()

    def future_symbol_embedding(self, request: TraceNetRequest) -> Tensor:
        if request.future_symbol_indexes is None:
            raise ValueError("future symbol indexes are required")
        assert request.future_symbol_offsets is not None
        assert request.future_symbol_weights is not None
        embedding = self.backbone.recent_symbols(
            request.future_symbol_indexes,
            request.future_symbol_offsets,
            per_sample_weights=request.future_symbol_weights,
        )
        return nn.functional.normalize(embedding, dim=1)

    def _auxiliary(
        self,
        output: dict[str, Tensor],
        request: TraceNetRequest,
        features: BackboneOutput,
        total_reward: Tensor | None,
    ) -> None:
        auxiliary_heads = (
            self.heads.predicted_trace,
            self.heads.predicted_execution,
            self.heads.predicted_cursor,
        )
        if not request.compute_auxiliary or not any(auxiliary_heads):
            return
        indexes = self._action_indexes(request, total_reward)
        actions = torch.tensor(
            [action for action, _x, _y in indexes],
            dtype=torch.long,
            device=features.merged.device,
        )
        if request.auxiliary_pixel_masks is not None:
            masks = nn.functional.interpolate(
                request.auxiliary_pixel_masks[:, None].type_as(features.merged),
                size=features.merged.shape[-2:],
                mode="area",
            )
            counts = masks.sum((2, 3)).clamp_min(torch.finfo(features.merged.dtype).eps)
            joined = (features.merged * masks).sum((2, 3)) / counts
        else:
            selected = [
                features.merged[index, :, y // 8, x // 8].unsqueeze(0)
                for index, (_action, x, y) in enumerate(indexes)
            ]
            joined = torch.cat(selected, dim=0)
        if self.heads.action_embedding is None:
            effects = joined
        else:
            effects = torch.cat([joined, self.heads.action_embedding(actions)], dim=1)
        if self.heads.predicted_trace is not None:
            output["predictedTraces"] = self.heads.predicted_trace(effects)
        if self.heads.predicted_execution is not None:
            output["predictedExecutionFeatures"] = self.heads.predicted_execution(effects)
        if self.heads.predicted_cursor is not None:
            output["predictedCursor"] = self.heads.predicted_cursor(joined)

    def _action_indexes(
        self, request: TraceNetRequest, total_reward: Tensor | None
    ) -> tuple[tuple[int, int, int], ...]:
        if request.auxiliary_action_type is not None:
            assert request.auxiliary_action_x is not None and request.auxiliary_action_y is not None
            return tuple(
                (int(action), int(x), int(y))
                for action, x, y in zip(
                    request.auxiliary_action_type,
                    request.auxiliary_action_x,
                    request.auxiliary_action_y,
                    strict=True,
                )
            )
        if total_reward is None:
            raise ValueError("auxiliary inference requires reward maps or explicit action indexes")
        indexes = []
        for sample in total_reward:
            action = int(sample.reshape(self.num_actions, -1).max(dim=1)[0].argmax())
            y = int(sample[action].max(dim=1)[0].argmax())
            x = int(sample[action, y].argmax())
            indexes.append((action, x, y))
        return tuple(indexes)
