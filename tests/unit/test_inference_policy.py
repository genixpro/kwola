import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from kwola.agent import (
    ExplorationProbabilities,
    InferencePolicy,
    PolicyMode,
    TraceNet,
    action_catalog,
)
from kwola.agent.encoding import ACTION_KINDS, ObservationEncoder, action_masks
from kwola.config import load_config
from kwola.domain.actions import Action, ActionKind, ActionMap, ActionTarget
from kwola.domain.observations import Observation, Viewport
from kwola.orchestration.initialize import initialize_run
from kwola.storage import (
    LEARNING_SCHEMA_VERSION,
    CheckpointIntegrityError,
    CheckpointPublisher,
    load_manifest,
)


class NoExploreRandom(random.Random):
    def random(self) -> float:
        return 0.999999


class FixedDrawRandom(random.Random):
    def __init__(self, draw: float) -> None:
        super().__init__(0)
        self._draw = draw

    def random(self) -> float:
        return self._draw


class SequenceDrawRandom(random.Random):
    def __init__(self, *draws: float) -> None:
        super().__init__(0)
        self._draws = iter(draws)

    def random(self) -> float:
        return next(self._draws)


class FixedActionPolicy:
    def __init__(self, source: str) -> None:
        self._source = source

    def select(self, _action_map: ActionMap) -> Action:
        return Action(ActionKind.CLICK, 1, 1, source=self._source, channel="click")


def test_action_masks_map_target_capabilities() -> None:
    target = ActionTarget(
        10,
        10,
        50,
        50,
        "input",
        can_click=True,
        can_right_click=True,
        can_type=True,
        can_scroll=True,
    )
    masks = action_masks(ActionMap((target,), 100, 100, "1"), 20)
    assert masks.shape == (len(ACTION_KINDS), 20, 20)
    assert masks[:, 2:10, 2:10].sum() > 0
    assert bool(action_masks(ActionMap((), 10, 10, "1"), 8).all())


def test_observation_encoder_restores_temporal_inference_state(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 12)
    config = load_config(tmp_path)
    channels = action_catalog(config.policy)
    encoder = ObservationEncoder(
        config.model,
        None,
        config.policy.rewards.impossible_action,
        channels,
        config.training.recent_action_image_radius,
        config.training.recent_action_image_decay,
    )
    click = Action(ActionKind.CLICK, 16, 24, source="model", channel="click")

    request = encoder.encode(
        _observation(),
        torch.device("cpu"),
        recent_actions=(click,),
        coverage_symbols=(11, 22),
        recent_symbol_history=((11,), (22,)),
        step_number=7,
    )
    backbone = request.backbone
    click_index = tuple(channel.name for channel in channels).index("click")

    assert backbone.step_number.tolist() == [7.0]
    assert backbone.coverage_symbol_indexes.tolist() == [11, 22]
    assert backbone.recent_symbol_indexes.tolist() == [11, 22]
    assert backbone.recent_symbol_weights.tolist() == pytest.approx([0.9, 1.0])
    assert backbone.recent_actions_vector[0, click_index] == 1
    assert backbone.recent_actions_image[0, click_index].sum() > 0
    assert backbone.coordinate_image[0, 0, 0, 0] == -1
    assert backbone.coordinate_image[0, 1, -1, -1] == 1
    assert bool(backbone.action_map_available_image.all())


def test_encoder_distinguishes_unavailable_action_map_from_all_valid_fallback(
    tmp_path: Path,
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 12)
    config = load_config(tmp_path)
    observation = _observation()
    observation = replace(
        observation,
        action_map=ActionMap((), observation.viewport.width, observation.viewport.height, "1"),
    )
    request = ObservationEncoder(
        config.model, None, config.policy.rewards.impossible_action
    ).encode(observation, torch.device("cpu"))

    assert bool(request.pixel_action_maps.all())
    assert not bool(request.backbone.action_map_available_image.any())


def test_default_inference_symbols_do_not_mix_network_and_branch_namespaces(
    tmp_path: Path,
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 12)
    config = load_config(tmp_path)
    request = ObservationEncoder(
        config.model, None, config.policy.rewards.impossible_action
    ).encode(_observation(), torch.device("cpu"))

    assert request.backbone.coverage_symbol_indexes.tolist() == [1, 2]


def test_policy_uses_checkpoint_unless_random_is_forced(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 12)
    config = load_config(tmp_path)
    model = TraceNet(config.model, len(action_catalog(config.policy)))
    manifest = load_manifest(tmp_path)
    published = CheckpointPublisher(tmp_path).publish(
        rank=0,
        generation=1,
        writer=lambda stream: torch.save(
            {
                "learning_schema_version": LEARNING_SCHEMA_VERSION,
                "model": model.state_dict(),
                "target_model": model.state_dict(),
                "optimizer": {},
            },
            stream,
        ),
        manifest=manifest,
        now=1.0,
    )
    assert published is not None
    with patch("kwola.agent.policy.torch.load", wraps=torch.load) as load:
        policy = InferencePolicy(tmp_path, config, NoExploreRandom(3))
    assert load.call_args.kwargs["weights_only"] is True
    observation = _observation()
    with patch.object(policy._encoder, "encode", wraps=policy._encoder.encode) as encode:
        modeled = policy.select(
            observation,
            action_index=4,
            test_step_index=999,
            force_random=False,
            capture_diagnostics=True,
        )
        diagnostics = policy.take_diagnostics()
        policy.select(observation, action_index=5, test_step_index=999, force_random=False)
    forced = policy.select(observation, action_index=4, test_step_index=999, force_random=True)
    assert modeled.source == "model"
    assert forced.source == "weighted_random"
    assert diagnostics is not None
    assert diagnostics.checkpoint_generation == 1
    assert diagnostics.present_rewards is not None
    assert diagnostics.future_rewards is not None
    assert diagnostics.stamp is not None
    assert diagnostics.predicted_channel is not None
    assert diagnostics.action_masks.shape[0] == len(action_catalog(config.policy))
    second_context = encode.call_args_list[1].kwargs
    assert len(second_context["recent_actions"]) == 1
    assert second_context["coverage_symbols"] == (1, 2)
    assert second_context["recent_symbol_history"] == ((1, 2),)
    assert second_context["step_number"] == 5


@pytest.mark.parametrize(
    ("draws", "expected"),
    (((0.1,), "weighted"), ((0.8, 0.1), "q_weighted"), ((0.8, 0.3), "greedy")),
)
def test_exploration_uses_random_then_model_weighting_as_independent_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    draws: tuple[float, ...],
    expected: str,
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 13)
    config = load_config(tmp_path)
    policy = InferencePolicy(tmp_path, config, SequenceDrawRandom(*draws))
    policy._schedule = SimpleNamespace(  # type: ignore[assignment]
        probability=lambda **_values: ExplorationProbabilities(0.5, 0.2)
    )
    policy._weighted_random_policy = FixedActionPolicy("weighted")  # type: ignore[assignment]
    policy._model = object()  # type: ignore[assignment]
    monkeypatch.setattr(
        policy,
        "_evaluate_model",
        lambda *_args: (object(), {}),
    )
    monkeypatch.setattr(
        policy,
        "_action_from_output",
        lambda *_args, weighted=False: Action(
            ActionKind.CLICK,
            1,
            1,
            source="q_weighted" if weighted else "greedy",
            channel="click",
        ),
    )

    selected = policy.select(_observation(), action_index=0, test_step_index=0, force_random=False)

    assert selected.source == expected


def test_forced_random_always_uses_weighted_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 14)
    config = load_config(tmp_path)
    policy = InferencePolicy(tmp_path, config, FixedDrawRandom(0.99))
    policy._weighted_random_policy = FixedActionPolicy("weighted")  # type: ignore[assignment]
    policy._model = object()  # type: ignore[assignment]

    selected = policy.select(_observation(), action_index=0, test_step_index=0, force_random=True)

    assert selected.source == "weighted"


def test_greedy_mode_bypasses_both_exploration_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 14)
    config = load_config(tmp_path)
    policy = InferencePolicy(tmp_path, config, FixedDrawRandom(0.0))
    policy._model = object()  # type: ignore[assignment]
    monkeypatch.setattr(policy, "_evaluate_model", lambda *_args: (object(), {}))
    monkeypatch.setattr(
        policy,
        "_action_from_output",
        lambda *_args, weighted=False: Action(
            ActionKind.CLICK, 1, 1, source="weighted" if weighted else "model", channel="click"
        ),
    )

    selected = policy.select(
        _observation(), action_index=0, test_step_index=0, mode=PolicyMode.GREEDY
    )

    assert selected.source == "model"


def test_model_weighted_exploration_samples_from_q_values(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 15)
    config = load_config(tmp_path)
    policy = InferencePolicy(tmp_path, config, FixedDrawRandom(0.0))
    values = torch.full(
        (1, len(action_catalog(config.policy)), 1, 2),
        torch.finfo(torch.float32).min,
    )
    values[0, 0, 0] = torch.tensor([1.0, 2.0])

    selected = policy._action_from_output(_observation(), {"actionValues": values}, weighted=True)

    assert selected.source == "weighted_random"
    assert selected.x == 32


def test_policy_rejects_a_corrupted_checkpoint_before_loading(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 12)
    config = load_config(tmp_path)
    model = TraceNet(config.model, len(action_catalog(config.policy)))
    published = CheckpointPublisher(tmp_path).publish(
        rank=0,
        generation=1,
        writer=lambda stream: torch.save(
            {
                "learning_schema_version": LEARNING_SCHEMA_VERSION,
                "model": model.state_dict(),
                "target_model": model.state_dict(),
                "optimizer": {},
            },
            stream,
        ),
        manifest=load_manifest(tmp_path),
        now=1.0,
    )
    assert published is not None
    published[0].write_bytes(b"corrupted")

    with pytest.raises(CheckpointIntegrityError, match="digest mismatch"):
        InferencePolicy(tmp_path, config, NoExploreRandom(3))


def test_policy_suppresses_repeated_target_until_new_branch(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 12)
    config = load_config(tmp_path)
    policy_config = config.policy.model_copy(update={"max_repeat_maps_without_new_branches": 1})
    policy = InferencePolicy(
        tmp_path,
        config.model_copy(update={"policy": policy_config}),
        random.Random(4),
    )
    image = np.full((64, 128), 128, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    targets = (
        ActionTarget(0, 0, 63, 64, "button", can_click=True),
        ActionTarget(64, 0, 127, 64, "button", can_click=True),
    )
    observation = Observation(
        url="https://example.com",
        screenshot=encoded.tobytes(),
        viewport=Viewport(128, 64),
        action_map=ActionMap(targets, 128, 64, "1"),
        timestamp=1.0,
        branch_symbols=(1,),
    )

    first = policy.select(observation, action_index=0, test_step_index=1, force_random=True)
    second = policy.select(observation, action_index=1, test_step_index=1, force_random=True)

    assert (first.x < 64) != (second.x < 64)


def _observation() -> Observation:
    image = np.full((64, 64), 128, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    target = ActionTarget(
        0,
        0,
        64,
        64,
        "input",
        can_click=True,
        can_right_click=True,
        can_type=True,
        can_scroll=True,
    )
    return Observation(
        url="https://example.com",
        screenshot=encoded.tobytes(),
        viewport=Viewport(64, 64),
        action_map=ActionMap((target,), 64, 64, "1"),
        timestamp=1.0,
        branch_symbols=(1, 2),
        network_symbols=(3,),
    )
