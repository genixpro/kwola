"""Recorded-trace persistence and reward feature ownership."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kwola.agent import RewardCalculator, RewardSignals
from kwola.config.models import RunConfig
from kwola.domain.actions import Action
from kwola.domain.observations import Observation
from kwola.storage import AtomicBlobStore, LmdbRunStore
from kwola.training.image_cache import DecodedImageCache


@dataclass(slots=True)
class NoveltyState:
    branches: set[int]
    screenshots: set[str]
    urls: set[str]
    errors: set[str]

    @classmethod
    def initial(cls, observation: Observation) -> "NoveltyState":
        digest = hashlib.sha256(observation.screenshot).hexdigest()
        return cls(
            set(observation.branch_symbols),
            {digest},
            {observation.url},
            set(observation.errors),
        )


@dataclass(frozen=True, slots=True)
class TraceFeatures:
    new_branches: tuple[int, ...]
    new_network: tuple[int, ...]
    new_errors: tuple[str, ...]
    screenshot_changed: bool
    screenshot_new: bool
    url_changed: bool
    url_new: bool
    log_output: bool


class TraceRecorder:
    def __init__(
        self,
        run_dir: Path,
        config: RunConfig,
        store: LmdbRunStore,
        artifacts: list[str],
    ) -> None:
        self._run_dir = run_dir
        self._config = config
        self._store = store
        self._artifacts = artifacts
        self._blobs = AtomicBlobStore(run_dir / config.storage.blobs_directory)
        self._training_images = DecodedImageCache(
            0,
            config.model.image_downscale_ratio,
            run_dir / config.storage.cache_directory / "decoded-images",
        )
        self._previous_after: tuple[str, str] | None = None

    def record(
        self,
        step_id: str,
        index: int,
        action: Action,
        before: Observation,
        after: Observation,
        novelty: NoveltyState,
        cursor: str,
        html: tuple[str | None, str | None],
    ) -> float:
        trace_id = f"{step_id}-trace-{index:04d}"
        features = _features(before, after, novelty)
        reward = _reward(self._config, after, features)
        screenshots = self._screenshots(trace_id, before, after)
        html_paths = self._html(trace_id, html)
        self._store.put(
            "traces",
            trace_id,
            self._payload(
                step_id,
                index,
                action,
                before,
                after,
                features,
                reward,
                cursor,
                screenshots,
                html_paths,
            ),
        )
        _record_bugs(self._store, trace_id, after.errors)
        return reward

    def _screenshots(
        self, trace_id: str, before: Observation, after: Observation
    ) -> tuple[str, str]:
        before_digest = hashlib.sha256(before.screenshot).hexdigest()
        if self._previous_after is not None and self._previous_after[0] == before_digest:
            before_relative = self._previous_after[1]
        else:
            before_path = self._blobs.write(
                "screenshots", f"{trace_id}-before.png", before.screenshot
            )
            self._training_images.store_encoded(before_path, before.screenshot)
            before_relative = str(before_path.relative_to(self._run_dir))
        after_path = self._blobs.write("screenshots", f"{trace_id}.png", after.screenshot)
        self._training_images.store_encoded(after_path, after.screenshot)
        after_relative = str(after_path.relative_to(self._run_dir))
        self._previous_after = (hashlib.sha256(after.screenshot).hexdigest(), after_relative)
        self._artifacts.append(after_relative)
        return before_relative, after_relative

    def _html(
        self, trace_id: str, values: tuple[str | None, str | None]
    ) -> tuple[str | None, str | None]:
        paths: list[str | None] = []
        for suffix, value in zip(("before", "after"), values, strict=True):
            if value is None:
                paths.append(None)
                continue
            path = self._blobs.write("html", f"{trace_id}-{suffix}.html", value.encode())
            relative = str(path.relative_to(self._run_dir))
            self._artifacts.append(relative)
            paths.append(relative)
        return paths[0], paths[1]

    @staticmethod
    def _payload(
        step_id: str,
        index: int,
        action: Action,
        before: Observation,
        after: Observation,
        features: TraceFeatures,
        reward: float,
        cursor: str,
        screenshots: tuple[str, str],
        html: tuple[str | None, str | None],
    ) -> dict[str, Any]:
        return {
            "step_id": step_id,
            "index": index,
            "action": _action_payload(action),
            "url_before": before.url,
            "url_after": after.url,
            "reward": reward,
            "branch_symbols": list(after.branch_symbols),
            "network_symbols": list(after.network_symbols),
            "errors": list(after.errors),
            "new_errors": list(features.new_errors),
            "new_branch_symbols": list(features.new_branches),
            "new_network_symbols": list(features.new_network),
            "screenshot_changed": features.screenshot_changed,
            "screenshot_new": features.screenshot_new,
            "url_new": features.url_new,
            "log_output": features.log_output,
            "cursor": cursor,
            "viewport": [before.viewport.width, before.viewport.height],
            "action_targets": [_target_payload(target) for target in before.action_map.targets],
            "screenshot_before": screenshots[0],
            "screenshot": screenshots[1],
            "html_before": html[0],
            "html_after": html[1],
        }


def _features(before: Observation, after: Observation, state: NoveltyState) -> TraceFeatures:
    new_branches = tuple(sorted(set(after.branch_symbols) - state.branches))
    new_network = tuple(sorted(set(after.network_symbols) - set(before.network_symbols)))
    new_errors = tuple(sorted(set(after.errors) - state.errors))
    digest = hashlib.sha256(after.screenshot).hexdigest()
    result = TraceFeatures(
        new_branches,
        new_network,
        new_errors,
        before.screenshot != after.screenshot,
        digest not in state.screenshots,
        before.url != after.url,
        after.url not in state.urls,
        len(after.console_messages) > len(before.console_messages),
    )
    state.branches.update(after.branch_symbols)
    state.screenshots.add(digest)
    state.urls.add(after.url)
    state.errors.update(after.errors)
    return result


def _reward(config: RunConfig, after: Observation, features: TraceFeatures) -> float:
    return RewardCalculator(config.policy.rewards).present(
        RewardSignals(
            action_succeeded=True,
            code_executed=bool(after.branch_symbols),
            new_branches_executed=bool(features.new_branches),
            network_traffic=bool(after.network_symbols),
            new_network_traffic=bool(features.new_network),
            screenshot_changed=features.screenshot_changed,
            screenshot_new=features.screenshot_new,
            url_changed=features.url_changed,
            url_new=features.url_new,
            log_output=features.log_output,
        )
    )


def _action_payload(action: Action) -> dict[str, Any]:
    return {
        "kind": action.kind.value,
        "x": action.x,
        "y": action.y,
        "text": action.text,
        "direction": action.direction,
        "source": action.source,
        "channel": action.channel_name,
    }


def _target_payload(target: Any) -> dict[str, Any]:
    return {
        "bounds": [target.left, target.top, target.right, target.bottom],
        "click": target.can_click,
        "right_click": target.can_right_click,
        "type": target.can_type,
        "scroll": target.can_scroll,
        "scroll_up": target.can_scroll_up,
        "scroll_down": target.can_scroll_down,
    }


def _record_bugs(store: LmdbRunStore, trace_id: str, errors: tuple[str, ...]) -> None:
    for message in errors:
        fingerprint = hashlib.sha256(message.encode()).hexdigest()
        existing = store.get("bugs", fingerprint)
        traces = list(existing.get("trace_ids", [])) if existing else []
        if trace_id not in traces:
            traces.append(trace_id)
        store.put(
            "bugs",
            fingerprint,
            {"message": message, "fingerprint": fingerprint, "trace_ids": traces},
        )
