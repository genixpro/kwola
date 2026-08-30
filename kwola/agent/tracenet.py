"""TraceNet topology with focused backbone and prediction-head ownership."""

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
    compute_action_probabilities: bool = True
    compute_state_values: bool = True
    compute_advantage_values: bool = True
    compute_auxiliary: bool = False
    output_stamp: bool = False
    auxiliary_action_type: Tensor | None = None
    auxiliary_action_x: Tensor | None = None
    auxiliary_action_y: Tensor | None = None


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
        total_reward: Tensor | None = None
        if request.compute_rewards:
            present, discounted, total_reward = self.heads.reward_maps(
                features.merged,
                request.pixel_action_maps,
                request.impossible_action_reward,
            )
            output["presentRewards"] = present
            output["discountFutureRewards"] = discounted
        if request.output_stamp:
            output["stamp"] = features.stamp.detach()
        if request.compute_action_probabilities:
            output["actionProbabilities"] = self.heads.action_probabilities(
                features.merged, request.pixel_action_maps
            )
        if request.compute_state_values:
            output["stateValues"] = self.heads.state_value(features.state_features)
        if request.compute_advantage_values:
            values = self.heads.advantage(features.merged)
            masks = request.pixel_action_maps
            output["advantage"] = values * masks + (1 - masks) * request.impossible_action_reward
        self._auxiliary(output, request, features, total_reward)
        return output

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
        selected = [
            features.merged[index, :, y // 8, x // 8].unsqueeze(0)
            for index, (_action, x, y) in enumerate(indexes)
        ]
        joined = torch.cat(selected, dim=0)
        for name, head in (
            ("predictedTraces", self.heads.predicted_trace),
            ("predictedExecutionFeatures", self.heads.predicted_execution),
            ("predictedCursor", self.heads.predicted_cursor),
        ):
            if head is not None:
                output[name] = head(joined)

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
