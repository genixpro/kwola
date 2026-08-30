"""Checkpoint-backed inference with scheduled seeded exploration."""

import random
from dataclasses import replace
from pathlib import Path

import torch

from kwola.config.models import RunConfig
from kwola.domain.actions import Action, ActionKind, ActionTarget
from kwola.domain.observations import Observation
from kwola.storage import load_manifest, verify_checkpoint

from .actions import action_catalog
from .encoding import ObservationEncoder
from .exploration import ExplorationSchedule
from .random_policy import RandomActionPolicy
from .tracenet import TraceNet


class InferencePolicy:
    def __init__(self, run_dir: Path, config: RunConfig, source: random.Random) -> None:
        self._run_dir = run_dir
        self._config = config
        self._random = source
        self._channels = action_catalog(config.policy)
        self._random_policy = RandomActionPolicy(
            source, self._channels, weighted=config.policy.weighted_random_actions
        )
        self._schedule = ExplorationSchedule(
            config.policy.exploration, config.policy.testing_sequence_length
        )
        self._encoder = ObservationEncoder(
            config.model, None, config.policy.rewards.impossible_action, self._channels
        )
        self._model = self._load_model()
        self._recent_actions: list[Action] = []
        self._seen_branches: set[int] = set()

    def select(
        self,
        observation: Observation,
        *,
        action_index: int,
        test_step_index: int,
        force_random: bool,
    ) -> Action:
        observation = self._filtered_observation(observation)
        exploration = self._schedule.probability(
            action_index=action_index,
            session_index=0,
            session_count=1,
            test_step_index=test_step_index,
        )
        if force_random or self._model is None or self._random.random() < exploration.random:
            action = self._random_policy.select(observation.action_map)
        else:
            action = self._model_action(observation)
        self._recent_actions.append(action)
        return action

    def _filtered_observation(self, observation: Observation) -> Observation:
        new_branches = set(observation.branch_symbols) - self._seen_branches
        self._seen_branches.update(observation.branch_symbols)
        if new_branches:
            self._recent_actions.clear()
        policy = self._config.policy
        if not policy.repeat_action_override or policy.max_repeat_maps_without_new_branches == 0:
            return observation
        targets = tuple(
            target
            for target in observation.action_map.targets
            if self._repeat_count(target) < policy.max_repeat_maps_without_new_branches
        )
        if not targets:
            return observation
        action_map = replace(observation.action_map, targets=targets)
        return replace(observation, action_map=action_map)

    def _repeat_count(self, target: ActionTarget) -> int:
        return sum(
            target.left <= action.x <= target.right
            and target.top <= action.y <= target.bottom
            and any(channel.kind is action.kind for channel in self._random_policy.allowed(target))
            for action in self._recent_actions
        )

    def _load_model(self) -> TraceNet | None:
        manifest = load_manifest(self._run_dir)
        if manifest.checkpoint is None:
            return None
        model = TraceNet(self._config.model, len(self._channels))
        checkpoint = verify_checkpoint(self._run_dir, manifest.checkpoint)
        payload = torch.load(checkpoint, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(payload["model"])
        return model.eval()

    def _model_action(self, observation: Observation) -> Action:
        assert self._model is not None
        request = self._encoder.encode(observation, torch.device("cpu"))
        with torch.no_grad():
            probabilities = self._model(request)["actionProbabilities"][0]
        flat = int(probabilities.flatten().argmax())
        height, width = probabilities.shape[-2:]
        pixels = height * width
        channel = self._channels[flat // pixels]
        pixel = flat % pixels
        y, x = divmod(pixel, width)
        viewport = observation.viewport
        action_x = min(viewport.width - 1, x * viewport.width // width)
        action_y = min(viewport.height - 1, y * viewport.height // height)
        generated = (
            self._random_policy.typing_text(channel) if channel.kind is ActionKind.TYPE else None
        )
        return Action(
            channel.kind,
            action_x,
            action_y,
            generated,
            channel.direction,
            "model",
            channel.name,
        )
