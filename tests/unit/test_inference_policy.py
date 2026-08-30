import random
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from kwola.agent import InferencePolicy, TraceNet, action_catalog
from kwola.agent.encoding import ACTION_KINDS, action_masks
from kwola.config import load_config
from kwola.domain.actions import ActionMap, ActionTarget
from kwola.domain.observations import Observation, Viewport
from kwola.orchestration.initialize import initialize_run
from kwola.storage import CheckpointIntegrityError, CheckpointPublisher, load_manifest


class NoExploreRandom(random.Random):
    def random(self) -> float:
        return 0.999999


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


def test_policy_uses_checkpoint_unless_random_is_forced(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 12)
    config = load_config(tmp_path)
    model = TraceNet(config.model, len(action_catalog(config.policy)))
    manifest = load_manifest(tmp_path)
    published = CheckpointPublisher(tmp_path).publish(
        rank=0,
        generation=1,
        writer=lambda stream: torch.save({"model": model.state_dict(), "optimizer": {}}, stream),
        manifest=manifest,
        now=1.0,
    )
    assert published is not None
    with patch("kwola.agent.policy.torch.load", wraps=torch.load) as load:
        policy = InferencePolicy(tmp_path, config, NoExploreRandom(3))
    assert load.call_args.kwargs["weights_only"] is True
    observation = _observation()
    modeled = policy.select(observation, action_index=4, test_step_index=999, force_random=False)
    forced = policy.select(observation, action_index=4, test_step_index=999, force_random=True)
    assert modeled.source == "model"
    assert forced.source == "weighted_random"


def test_policy_rejects_a_corrupted_checkpoint_before_loading(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 12)
    config = load_config(tmp_path)
    model = TraceNet(config.model, len(action_catalog(config.policy)))
    published = CheckpointPublisher(tmp_path).publish(
        rank=0,
        generation=1,
        writer=lambda stream: torch.save({"model": model.state_dict()}, stream),
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
