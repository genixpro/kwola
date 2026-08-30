"""Checkpoint-backed inference with scheduled seeded exploration."""

import random
from dataclasses import replace
from pathlib import Path

import torch

from kwola.config.models import RunConfig
from kwola.domain.actions import Action, ActionKind, ActionTarget
from kwola.domain.observations import Observation
from kwola.storage import load_manifest, require_learning_schema, verify_checkpoint

from .actions import action_catalog
from .diagnostics import InferenceDiagnostics, capture_inference_diagnostics
from .encoding import ObservationEncoder
from .exploration import ExplorationSchedule
from .random_policy import RandomActionPolicy
from .tracenet import TraceNet, TraceNetRequest


class InferencePolicy:
    def __init__(self, run_dir: Path, config: RunConfig, source: random.Random) -> None:
        self._run_dir = run_dir
        self._config = config
        self._random = source
        self._channels = action_catalog(config.policy)
        self._weighted_random_policy = RandomActionPolicy(source, self._channels, weighted=True)
        self._uniform_random_policy = RandomActionPolicy(source, self._channels, weighted=False)
        self._schedule = ExplorationSchedule(
            config.policy.exploration, config.policy.testing_sequence_length
        )
        self._encoder = ObservationEncoder(
            config.model,
            None,
            config.policy.rewards.impossible_action,
            self._channels,
            config.training.recent_action_image_radius,
            config.training.recent_action_image_decay,
        )
        self._checkpoint_generation: int | None = None
        self._model = self._load_model()
        self._last_diagnostics: InferenceDiagnostics | None = None
        self._model_actions: list[Action] = []
        self._repeat_actions: list[Action] = []
        self._coverage_symbols: set[int] = set()
        self._recent_symbol_history: list[tuple[int, ...]] = []
        self._seen_branches: set[int] = set()

    def select(
        self,
        observation: Observation,
        *,
        action_index: int,
        test_step_index: int,
        force_random: bool,
        capture_diagnostics: bool = False,
    ) -> Action:
        self._last_diagnostics = None
        self._record_observation(observation)
        observation = self._filtered_observation(observation)
        exploration = self._schedule.probability(
            action_index=action_index,
            session_index=0,
            session_count=1,
            test_step_index=test_step_index,
        )
        draw = self._random.random()
        evaluation: tuple[TraceNetRequest, dict[str, torch.Tensor]] | None = None
        if force_random or self._model is None or draw < exploration.weighted_random:
            action = self._weighted_random_policy.select(observation.action_map)
        elif draw < exploration.random:
            action = self._uniform_random_policy.select(observation.action_map)
        else:
            if capture_diagnostics:
                evaluation = self._evaluate_model(observation, action_index, True)
                action = self._action_from_output(observation, evaluation[1])
            else:
                action = self._model_action(observation, action_index)
        if capture_diagnostics:
            if evaluation is None:
                evaluation = self._diagnostic_evaluation(observation, action_index)
            request, output = evaluation
            predicted = self._predicted(output, observation) if output else None
            self._last_diagnostics = capture_inference_diagnostics(
                request,
                output or None,
                self._channels,
                checkpoint_generation=self._checkpoint_generation,
                predicted=predicted,
                map_downscale=self._config.reporting.debug_video_map_downscale,
            )
        self._model_actions.append(action)
        self._model_actions = self._model_actions[-self._config.model.recent_action_history :]
        self._repeat_actions.append(action)
        return action

    def take_diagnostics(self) -> InferenceDiagnostics | None:
        result = self._last_diagnostics
        self._last_diagnostics = None
        return result

    def _record_observation(self, observation: Observation) -> None:
        if not self._model_actions:
            return
        self._coverage_symbols.update(observation.branch_symbols)
        self._recent_symbol_history.append(observation.branch_symbols)

    def _filtered_observation(self, observation: Observation) -> Observation:
        new_branches = set(observation.branch_symbols) - self._seen_branches
        self._seen_branches.update(observation.branch_symbols)
        if new_branches:
            self._repeat_actions.clear()
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
            and any(
                channel.kind is action.kind
                for channel in self._weighted_random_policy.allowed(target)
            )
            for action in self._repeat_actions
        )

    def _load_model(self) -> TraceNet | None:
        manifest = load_manifest(self._run_dir)
        if manifest.checkpoint is None:
            return None
        self._checkpoint_generation = manifest.checkpoint.generation
        model = TraceNet(self._config.model, len(self._channels))
        checkpoint = verify_checkpoint(self._run_dir, manifest.checkpoint)
        payload = require_learning_schema(
            torch.load(checkpoint, map_location=torch.device("cpu"), weights_only=True)
        )
        model.load_state_dict(payload["model"], strict=True)  # type: ignore[arg-type]
        return model.eval()

    def _model_action(self, observation: Observation, action_index: int) -> Action:
        _request, output = self._evaluate_model(observation, action_index, False)
        return self._action_from_output(observation, output)

    def _evaluate_model(
        self, observation: Observation, action_index: int, output_stamp: bool
    ) -> tuple[TraceNetRequest, dict[str, torch.Tensor]]:
        assert self._model is not None
        request = self._encoder.encode(
            observation,
            torch.device("cpu"),
            recent_actions=tuple(self._model_actions),
            coverage_symbols=tuple(sorted(self._coverage_symbols)),
            recent_symbol_history=tuple(self._recent_symbol_history),
            step_number=action_index,
        )
        if output_stamp:
            request = replace(request, output_stamp=True)
        with torch.no_grad():
            output = self._model(request)
        return request, output

    def _diagnostic_evaluation(
        self, observation: Observation, action_index: int
    ) -> tuple[TraceNetRequest, dict[str, torch.Tensor]]:
        if self._model is not None:
            return self._evaluate_model(observation, action_index, True)
        request = self._encoder.encode(
            observation,
            torch.device("cpu"),
            recent_actions=tuple(self._model_actions),
            coverage_symbols=tuple(sorted(self._coverage_symbols)),
            recent_symbol_history=tuple(self._recent_symbol_history),
            step_number=action_index,
        )
        return request, {}

    def _action_from_output(
        self,
        observation: Observation,
        output: dict[str, torch.Tensor],
    ) -> Action:
        action_values = output["actionValues"][0]
        flat = int(action_values.flatten().argmax())
        height, width = action_values.shape[-2:]
        pixels = height * width
        channel = self._channels[flat // pixels]
        pixel = flat % pixels
        y, x = divmod(pixel, width)
        viewport = observation.viewport
        action_x = min(viewport.width - 1, x * viewport.width // width)
        action_y = min(viewport.height - 1, y * viewport.height // height)
        generated = (
            self._weighted_random_policy.typing_text(channel)
            if channel.kind is ActionKind.TYPE
            else None
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

    def _predicted(
        self, output: dict[str, torch.Tensor], observation: Observation
    ) -> tuple[int, int, int, float]:
        values = output["actionValues"][0]
        flat = int(values.flatten().argmax())
        height, width = values.shape[-2:]
        pixels = height * width
        channel = flat // pixels
        pixel = flat % pixels
        y, x = divmod(pixel, width)
        viewport = observation.viewport
        return (
            channel,
            min(viewport.width - 1, x * viewport.width // width),
            min(viewport.height - 1, y * viewport.height // height),
            float(values.flatten()[flat]),
        )
