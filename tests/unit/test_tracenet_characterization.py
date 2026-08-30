from dataclasses import replace

import pytest
import torch

from kwola.agent.model_backbone import BackboneInput
from kwola.agent.normalization import SpatialGroupNorm
from kwola.agent.tiled_inference import evaluate_tiled
from kwola.agent.tracenet import TraceNet, TraceNetRequest
from kwola.config import profile_config


def inputs(batch: int, actions: int) -> TraceNetRequest:
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
    return TraceNetRequest(backbone, masks, -10.0, output_stamp=True)


def test_tracenet_emits_decomposed_and_combined_action_values() -> None:
    config = profile_config("testing", "https://example.com", 1)
    torch.manual_seed(7)
    model = TraceNet(config.model, 4, 12, 37).eval()

    with torch.no_grad():
        outputs = model(inputs(2, 4))

    assert outputs.keys() == {
        "presentRewards",
        "discountFutureRewards",
        "actionValues",
        "stamp",
    }
    assert outputs["presentRewards"].shape == (2, 4, 64, 64)
    assert outputs["discountFutureRewards"].shape == (2, 4, 64, 64)
    assert outputs["stamp"].shape == (2, 5, 2, 2)
    torch.testing.assert_close(
        outputs["actionValues"],
        outputs["presentRewards"] + outputs["discountFutureRewards"],
    )


def test_tracenet_masks_impossible_action_values() -> None:
    config = profile_config("testing", "https://example.com", 1)
    model = TraceNet(config.model, 4).eval()
    request = inputs(2, 4)
    masks = request.pixel_action_maps.clone()
    masks[:, 0, 0, 0] = 0

    with torch.no_grad():
        outputs = model(replace(request, pixel_action_maps=masks))

    assert torch.all(
        outputs["actionValues"][:, 0, 0, 0] == torch.finfo(outputs["actionValues"].dtype).min
    )


def test_schema_v2_model_state_loading_is_strict() -> None:
    config = profile_config("testing", "https://example.com", 1)
    model = TraceNet(config.model, 4)
    incomplete = dict(model.state_dict())
    incomplete.pop(next(iter(incomplete)))

    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(incomplete, strict=True)


def test_future_trace_embedding_is_a_detached_auxiliary_target() -> None:
    config = profile_config("testing", "https://example.com", 1)
    model = TraceNet(config.model, 4)
    request = inputs(2, 4)
    backbone = request.backbone
    request = replace(
        request,
        future_symbol_indexes=backbone.recent_symbol_indexes,
        future_symbol_offsets=backbone.recent_symbol_offsets,
        future_symbol_weights=backbone.recent_symbol_weights,
    )

    output = model(request)["decayingFutureSymbolEmbedding"]

    assert not output.requires_grad


def test_reward_only_profile_has_no_trainable_parameters_detached_from_loss() -> None:
    config = profile_config("rig", "https://example.com", 1)
    model = TraceNet(config.model, 4)

    outputs = model(inputs(2, 4))
    (outputs["presentRewards"].mean() + outputs["discountFutureRewards"].mean()).backward()

    assert [name for name, parameter in model.named_parameters() if parameter.grad is None] == []


def test_spatial_normalization_is_mode_independent_and_checkpoint_compatible() -> None:
    layer = SpatialGroupNorm(16)
    before = dict(layer.state_dict())
    layer.load_state_dict(torch.nn.BatchNorm2d(16).state_dict(), strict=True)
    values = torch.rand(2, 16, 8, 8)

    training = layer.train()(values)
    evaluation = layer.eval()(values)

    torch.testing.assert_close(training, evaluation)
    assert layer.state_dict().keys() == before.keys()
    torch.testing.assert_close(layer.running_mean, before["running_mean"])
    torch.testing.assert_close(layer.running_var, before["running_var"])


def test_tiled_inference_reconstructs_full_sized_masked_maps() -> None:
    config = profile_config("testing", "https://example.com", 1)
    model = TraceNet(config.model, 4).eval()
    request = inputs(2, 4)
    masks = request.pixel_action_maps.clone()
    masks[:, 0, 0, 0] = 0
    request = replace(request, pixel_action_maps=masks)

    with torch.no_grad():
        output = evaluate_tiled(model, request, (32, 32))

    assert output["presentRewards"].shape == (2, 4, 64, 64)
    assert output["discountFutureRewards"].shape == (2, 4, 64, 64)
    assert output["stamp"].shape == (2, 5, 2, 2)
    assert torch.all(
        output["actionValues"][:, 0, 0, 0] == torch.finfo(output["actionValues"].dtype).min
    )
