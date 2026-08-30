from typing import Any

import torch

from kwola.agent.model_backbone import BackboneInput
from kwola.agent.tracenet import TraceNet, TraceNetRequest
from kwola.components.agents.TraceNet import TraceNet as LegacyTraceNet
from kwola.config import profile_config


def legacy_config() -> dict[str, Any]:
    config = profile_config("testing", "https://example.com", 1)
    model = config.model
    values: dict[str, Any] = {
        "neural_network_additional_features_stamp_edge_size": model.additional_stamp_edge,
        "neural_network_additional_features_stamp_depth_size": model.additional_stamp_depth,
        "symbol_embedding_size": model.symbol_embedding_size,
        "neural_network_recent_actions_feature_size": model.recent_action_features,
        "symbol_dictionary_size": model.symbol_dictionary_size,
        "neural_network_pixel_features": model.pixel_features,
        "testing_recent_actions_vector_number_of_recent_traces": model.recent_action_history,
        "enable_trace_prediction_loss": model.enable_trace_prediction,
        "enable_execution_feature_prediction_loss": model.enable_execution_feature_prediction,
        "enable_cursor_prediction_loss": model.enable_cursor_prediction,
        "reward_impossible_action": config.policy.rewards.impossible_action,
    }
    for index, layer in enumerate(model.layers, start=1):
        values.update(
            {
                f"neural_network_layer_{index}_num_kernels": layer.kernels,
                f"neural_network_layer_{index}_kernel_size": layer.kernel_size,
                f"neural_network_layer_{index}_stride": layer.stride,
                f"neural_network_layer_{index}_padding": layer.padding,
                f"neural_network_layer_{index}_dilation": layer.dilation,
            }
        )
    for name in ("present_reward", "discounted_future_reward", "actor", "advantage"):
        values[f"neural_network_{name}_convolution_kernel_size"] = (
            model.prediction_head_kernel_size
        )
        values[f"neural_network_{name}_convolution_stride"] = model.prediction_head_stride
        values[f"neural_network_{name}_convolution_padding"] = model.prediction_head_padding
    return values


def inputs(batch: int, actions: int) -> tuple[dict[str, Any], TraceNetRequest]:
    torch.manual_seed(19)
    image = torch.rand(batch, 1, 64, 64)
    recent_actions_image = torch.rand(batch, actions, 64, 64)
    recent_actions_vector = torch.rand(batch, 5 * actions)
    symbol_indexes = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    weights = torch.ones(4)
    symbols_set = torch.tensor([1, 2, 3], dtype=torch.long)
    symbol_mask = torch.zeros(batch, 3, dtype=torch.bool)
    step = torch.tensor([1.0, 2.0])
    masks = torch.ones(batch, actions, 64, 64)
    old = {
        "image": image,
        "recentActionsImage": recent_actions_image,
        "recentActionsVector": recent_actions_vector,
        "recentSymbolIndexes": symbol_indexes,
        "recentSymbolOffsets": offsets,
        "recentSymbolWeights": weights,
        "coverageSymbolIndexes": symbol_indexes,
        "coverageSymbolOffsets": offsets,
        "coverageSymbolWeights": weights,
        "coverageSymbolsSet": symbols_set,
        "coverageSymbolsKeyMask": symbol_mask,
        "stepNumber": step,
        "pixelActionMaps": masks,
        "computeRewards": True,
        "outputStamp": True,
        "outputFutureSymbolEmbedding": False,
        "computeActionProbabilities": True,
        "computeStateValues": True,
        "computeAdvantageValues": True,
        "computeExtras": False,
    }
    backbone = BackboneInput(
        image=image,
        recent_actions_image=recent_actions_image,
        recent_actions_vector=recent_actions_vector,
        recent_symbol_indexes=symbol_indexes,
        recent_symbol_offsets=offsets,
        recent_symbol_weights=weights,
        coverage_symbol_indexes=symbol_indexes,
        coverage_symbol_offsets=offsets,
        coverage_symbol_weights=weights,
        coverage_symbols_set=symbols_set,
        coverage_symbols_key_mask=symbol_mask,
        step_number=step,
    )
    request = TraceNetRequest(backbone, masks, -10.0, output_stamp=True)
    return old, request


def test_refactored_tracenet_matches_legacy_cpu_outputs() -> None:
    actions = 4
    config = profile_config("testing", "https://example.com", 1)
    torch.manual_seed(7)
    legacy = LegacyTraceNet(legacy_config(), actions, 12, 37)
    torch.manual_seed(8)
    current = TraceNet(config.model, actions, 12, 37)
    legacy_state = legacy.state_dict()
    current_state = current.state_dict()
    assert [value.shape for value in legacy_state.values()] == [
        value.shape for value in current_state.values()
    ]
    current.load_state_dict(
        {
            key: legacy_value
            for key, legacy_value in zip(current_state, legacy_state.values(), strict=True)
        }
    )
    legacy.eval()
    current.eval()
    old_input, request = inputs(2, actions)

    with torch.no_grad():
        expected = legacy(old_input)
        actual = current(request)

    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
