"""Checkpoint-backed inference with scheduled seeded exploration."""

import random
import string
from pathlib import Path

import torch

from kwola.config.models import RunConfig
from kwola.domain.actions import Action, ActionKind
from kwola.domain.observations import Observation
from kwola.storage import load_manifest

from .encoding import ACTION_KINDS, ObservationEncoder
from .exploration import ExplorationSchedule
from .random_policy import RandomActionPolicy
from .tracenet import TraceNet


class InferencePolicy:
    def __init__(self, run_dir: Path, config: RunConfig, source: random.Random) -> None:
        self._run_dir = run_dir
        self._config = config
        self._random = source
        self._random_policy = RandomActionPolicy(source, config.policy.custom_typing_strings)
        self._schedule = ExplorationSchedule(
            config.policy.exploration, config.policy.testing_sequence_length
        )
        edge = 64 if config.profile == "testing" else 320
        self._encoder = ObservationEncoder(
            config.model, edge, config.policy.rewards.impossible_action
        )
        self._edge = edge
        self._model = self._load_model()

    def select(
        self,
        observation: Observation,
        *,
        action_index: int,
        test_step_index: int,
        force_random: bool,
    ) -> Action:
        exploration = self._schedule.probability(
            action_index=action_index,
            session_index=0,
            session_count=1,
            test_step_index=test_step_index,
        )
        if force_random or self._model is None or self._random.random() < exploration.random:
            return self._random_policy.select(observation.action_map)
        return self._model_action(observation)

    def _load_model(self) -> TraceNet | None:
        manifest = load_manifest(self._run_dir)
        if manifest.checkpoint is None:
            return None
        model = TraceNet(self._config.model, len(ACTION_KINDS))
        payload = torch.load(
            self._run_dir / manifest.checkpoint.file,
            map_location=torch.device("cpu"),
        )
        model.load_state_dict(payload["model"])
        return model.eval()

    def _model_action(self, observation: Observation) -> Action:
        assert self._model is not None
        request = self._encoder.encode(observation, torch.device("cpu"))
        with torch.no_grad():
            probabilities = self._model(request)["actionProbabilities"][0]
        flat = int(probabilities.flatten().argmax())
        pixels = self._edge * self._edge
        kind = ACTION_KINDS[flat // pixels]
        pixel = flat % pixels
        y, x = divmod(pixel, self._edge)
        viewport = observation.viewport
        action_x = min(viewport.width - 1, x * viewport.width // self._edge)
        action_y = min(viewport.height - 1, y * viewport.height // self._edge)
        text = self._typing_text() if kind is ActionKind.TYPE else None
        direction = self._random.choice(("up", "down")) if kind is ActionKind.SCROLL else None
        return Action(kind, action_x, action_y, text, direction, "model")

    def _typing_text(self) -> str:
        if self._config.policy.custom_typing_strings:
            return self._random.choice(self._config.policy.custom_typing_strings)
        length = self._random.randint(4, 20)
        return "".join(self._random.choice(string.ascii_lowercase) for _ in range(length))
