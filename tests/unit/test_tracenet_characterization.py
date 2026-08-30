from dataclasses import replace

import pytest
import torch

from kwola.agent.model_backbone import BackboneInput
from kwola.agent.normalization import SpatialGroupNorm
from kwola.agent.spatial import coordinate_image
from kwola.agent.tiled_inference import evaluate_tiled, evaluate_tiled_batch
from kwola.agent.tracenet import TraceNet, TraceNetRequest
from kwola.config import profile_config
from kwola.training.batches import diagnostic_batch


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
        action_mask_image=masks,
        coordinate_image=coordinate_image(64, 64).unsqueeze(0).repeat(batch, 1, 1, 1),
        action_map_available_image=torch.ones(batch, 1, 64, 64),
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
    torch.testing.assert_close(output.norm(dim=1), torch.ones(2))


def test_auxiliary_profile_has_no_trainable_parameters_detached_from_complete_loss() -> None:
    config = profile_config("rig", "https://example.com", 1)
    model = TraceNet(config.model, 4)

    request = inputs(2, 4)
    request = replace(
        request,
        compute_auxiliary=True,
        auxiliary_action_type=torch.tensor([0, 1]),
        auxiliary_action_x=torch.tensor([16, 32]),
        auxiliary_action_y=torch.tensor([16, 32]),
        auxiliary_pixel_masks=torch.ones(2, 64, 64),
        future_symbol_indexes=request.backbone.recent_symbol_indexes,
        future_symbol_offsets=request.backbone.recent_symbol_offsets,
        future_symbol_weights=request.backbone.recent_symbol_weights,
    )
    outputs = model(request)
    sum(value.mean() for name, value in outputs.items() if name != "stamp").backward()

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


def test_shape_grouped_tiled_batch_matches_individual_inference() -> None:
    config = profile_config("testing", "https://example.com", 1)
    model = TraceNet(config.model, 4).eval()
    requests = tuple(
        diagnostic_batch(
            batch_size=1,
            num_actions=4,
            edge=64,
            seed=seed,
            device=torch.device("cpu"),
            impossible_reward=-10.0,
        )
        for seed in (3, 7)
    )

    with torch.no_grad():
        grouped = evaluate_tiled_batch(model, requests, (32, 32))
        individual = tuple(evaluate_tiled(model, request, (32, 32)) for request in requests)

    for grouped_output, individual_output in zip(grouped, individual, strict=True):
        torch.testing.assert_close(
            grouped_output["presentRewards"], individual_output["presentRewards"]
        )
        torch.testing.assert_close(
            grouped_output["discountFutureRewards"], individual_output["discountFutureRewards"]
        )


def test_effect_auxiliaries_are_conditioned_on_action_channel() -> None:
    config = profile_config("testing", "https://example.com", 1)
    model = TraceNet(config.model, 4)
    embedding = model.heads.action_embedding
    assert embedding is not None
    spatial = torch.zeros(2, model.backbone.merged_features)
    effects = torch.cat([spatial, embedding(torch.tensor([0, 1]))], dim=1)

    assert model.heads.predicted_execution is not None
    predictions = model.heads.predicted_execution(effects)

    assert not torch.equal(predictions[0], predictions[1])
