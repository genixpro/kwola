"""One explicit browser-testing runner."""

import random
import time
from collections.abc import Callable
from pathlib import Path

from kwola.agent import RandomActionPolicy, RewardCalculator, RewardSignals
from kwola.browser import (
    ActionExecutor,
    ActionMapExtractor,
    AutologinService,
    BrowserSessionCoordinator,
    NavigationPolicy,
    PlaywrightBrowserAdapter,
)
from kwola.browser.network import NetworkWaiter
from kwola.browser.screenshots import ScreenshotService
from kwola.config import load_config
from kwola.config.models import ViewportConfig
from kwola.domain.actions import Action, BrowserKind
from kwola.domain.observations import Observation
from kwola.instrumentation import TelemetryBuffer
from kwola.storage import AtomicBlobStore, LmdbRunStore

from .results import RunnerResult


class TestingRunner:
    def __init__(self, run_dir: Path, clock: Callable[[], float] = time.time) -> None:
        self._run_dir = run_dir
        self._clock = clock
        self._config = load_config(run_dir)

    def run(
        self,
        *,
        random_policy: bool = False,
        browser: BrowserKind | None = None,
        viewport: tuple[int, int] | None = None,
    ) -> RunnerResult:
        started = self._clock()
        browser_kind = browser or self._config.browser.enabled[0]
        if browser_kind not in self._config.browser.enabled:
            raise ValueError(f"browser {browser_kind} is not enabled for this run")
        selected_viewport = self._viewport(viewport)
        step_index = self._step_index()
        step_id = f"testing-{step_index:08d}"
        session = self._session(browser_kind, selected_viewport)
        artifacts: list[str] = []
        try:
            observation = session.start(str(self._config.target))
            policy = RandomActionPolicy(
                random.Random(self._config.seed + step_index),
                self._config.policy.custom_typing_strings,
            )
            rewards: list[float] = []
            for trace_index in range(self._config.policy.testing_sequence_length):
                action = policy.select(observation.action_map)
                before = observation
                observation = session.execute(action)
                reward = self._record_trace(
                    step_id,
                    trace_index,
                    action,
                    before,
                    observation,
                    artifacts,
                )
                rewards.append(reward)
            self._complete_step(step_id, browser_kind, rewards, random_policy)
            return RunnerResult(
                status="completed",
                step_id=step_id,
                duration_seconds=self._clock() - started,
                artifacts=tuple(artifacts),
                metrics={"traces": len(rewards), "reward": sum(rewards)},
            )
        finally:
            session.close()

    def _session(
        self, browser: BrowserKind, viewport: ViewportConfig
    ) -> BrowserSessionCoordinator:
        telemetry = TelemetryBuffer()
        navigation = NavigationPolicy(
            str(self._config.target), self._config.browser.prevent_offsite_navigation
        )
        extractor = ActionMapExtractor(navigation)
        executor = ActionExecutor()
        waiter = NetworkWaiter(self._config.browser.page_load_timeout_seconds)
        adapter = PlaywrightBrowserAdapter(
            self._config.browser, browser, viewport, telemetry
        )
        autologin = AutologinService(
            self._config.browser.autologin, extractor, executor, waiter
        )
        return BrowserSessionCoordinator(
            adapter,
            extractor,
            executor,
            waiter,
            ScreenshotService(),
            autologin,
            telemetry,
            clock=self._clock,
            action_settle_seconds=self._config.browser.action_settle_seconds,
        )

    def _record_trace(
        self,
        step_id: str,
        trace_index: int,
        action: Action,
        before: Observation,
        after: Observation,
        artifacts: list[str],
    ) -> float:
        reward = RewardCalculator(self._config.policy.rewards).present(
            RewardSignals(
                action_succeeded=True,
                code_executed=False,
                new_branches_executed=False,
                network_traffic=False,
                new_network_traffic=False,
                screenshot_changed=before.screenshot != after.screenshot,
                screenshot_new=False,
                url_changed=before.url != after.url,
                url_new=False,
                log_output=len(after.console_messages) > len(before.console_messages),
            )
        )
        trace_id = f"{step_id}-trace-{trace_index:04d}"
        blob = AtomicBlobStore(self._run_dir / self._config.storage.blobs_directory)
        screenshot = blob.write("screenshots", f"{trace_id}.png", after.screenshot)
        artifacts.append(str(screenshot.relative_to(self._run_dir)))
        with self._store() as store:
            store.put(
                "traces",
                trace_id,
                {
                    "step_id": step_id,
                    "index": trace_index,
                    "action": {
                        "kind": action.kind.value,
                        "x": action.x,
                        "y": action.y,
                        "text": action.text,
                        "direction": action.direction,
                        "source": action.source,
                    },
                    "url_before": before.url,
                    "url_after": after.url,
                    "reward": reward,
                    "screenshot": str(screenshot.relative_to(self._run_dir)),
                },
            )
        return reward

    def _step_index(self) -> int:
        with self._store() as store:
            state = store.get("run", "state") or {}
            return int(state.get("testing_steps", 0))

    def _complete_step(
        self, step_id: str, browser: BrowserKind, rewards: list[float], random_policy: bool
    ) -> None:
        with self._store() as store:
            state = store.get("run", "state") or {}
            state["testing_steps"] = int(state.get("testing_steps", 0)) + 1
            store.put("run", "state", state)
            store.put(
                "testing_steps",
                step_id,
                {
                    "browser": browser.value,
                    "random": random_policy,
                    "trace_count": len(rewards),
                    "reward": sum(rewards),
                },
            )

    def _store(self) -> LmdbRunStore:
        return LmdbRunStore(
            self._run_dir / self._config.storage.database_directory,
            map_size=self._config.storage.database_map_size_bytes,
            compression_level=self._config.storage.codec_compression_level,
        )

    def _viewport(self, override: tuple[int, int] | None) -> ViewportConfig:
        if override is None:
            return self._config.browser.viewports[0]
        return ViewportConfig(width=override[0], height=override[1])
