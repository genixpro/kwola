"""One explicit browser-testing runner."""

import hashlib
import random
import time
from collections.abc import Callable
from pathlib import Path

from kwola.agent import InferencePolicy, RewardCalculator, RewardSignals
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
from kwola.hooks import HookRegistry, LifecycleEvent, LifecycleEventName
from kwola.instrumentation import (
    BranchTraceCollector,
    InstrumentationAddon,
    ProxyService,
    ResourceRegistry,
    TelemetryBuffer,
)
from kwola.storage import AtomicBlobStore, LmdbRunStore

from .results import RunnerResult


class TestingRunner:
    def __init__(
        self,
        run_dir: Path,
        clock: Callable[[], float] = time.time,
        hooks: HookRegistry | None = None,
    ) -> None:
        self._run_dir = run_dir
        self._clock = clock
        self._config = load_config(run_dir)
        self._hooks = hooks or HookRegistry()

    def run(
        self,
        *,
        random_policy: bool = False,
        browser: BrowserKind | None = None,
        viewport: tuple[int, int] | None = None,
    ) -> RunnerResult:
        started = self._clock()
        self._dispatch(LifecycleEventName.RUN_STARTED)
        try:
            browser_kind = browser or self._config.browser.enabled[0]
            if browser_kind not in self._config.browser.enabled:
                raise ValueError(f"browser {browser_kind} is not enabled for this run")
            selected_viewport = self._viewport(viewport)
            with self._store() as store:
                return self._run_step(
                    store, started, browser_kind, selected_viewport, random_policy
                )
        finally:
            self._dispatch(LifecycleEventName.RUN_FINISHED)
            self._hooks.close()

    def _run_step(
        self,
        store: LmdbRunStore,
        started: float,
        browser_kind: BrowserKind,
        viewport: ViewportConfig,
        random_policy: bool,
    ) -> RunnerResult:
        step_index = self._step_index(store)
        step_id = f"testing-{step_index:08d}"
        session = self._session(browser_kind, viewport, store)
        artifacts: list[str] = []
        try:
            observation = session.start(str(self._config.target))
            self._dispatch(LifecycleEventName.SESSION_STARTED, step_id)
            policy = InferencePolicy(
                self._run_dir,
                self._config,
                random.Random(self._config.seed + step_index),
            )
            rewards: list[float] = []
            seen_branches = set(observation.branch_symbols)
            for trace_index in range(self._config.policy.testing_sequence_length):
                action = policy.select(
                    observation,
                    action_index=trace_index,
                    test_step_index=step_index,
                    force_random=random_policy,
                )
                trace_id = f"{step_id}-trace-{trace_index:04d}"
                self._dispatch(LifecycleEventName.BEFORE_ACTION, trace_id)
                before = observation
                observation = session.execute(action)
                self._dispatch(LifecycleEventName.AFTER_ACTION, trace_id)
                reward = self._record_trace(
                    step_id,
                    trace_index,
                    action,
                    before,
                    observation,
                    artifacts,
                    store,
                    seen_branches,
                )
                self._dispatch(
                    LifecycleEventName.TRACE_RECORDED,
                    trace_id,
                    (("reward", reward),),
                )
                rewards.append(reward)
            self._complete_step(store, step_id, browser_kind, rewards, random_policy)
            return RunnerResult(
                status="completed",
                step_id=step_id,
                duration_seconds=self._clock() - started,
                artifacts=tuple(artifacts),
                metrics={"traces": len(rewards), "reward": sum(rewards)},
            )
        finally:
            session.close()
            self._dispatch(LifecycleEventName.SESSION_FINISHED, step_id)

    def _session(
        self,
        browser: BrowserKind,
        viewport: ViewportConfig,
        store: LmdbRunStore,
    ) -> BrowserSessionCoordinator:
        telemetry = TelemetryBuffer()
        navigation = NavigationPolicy(
            str(self._config.target), self._config.browser.prevent_offsite_navigation
        )
        extractor = ActionMapExtractor(navigation)
        executor = ActionExecutor()
        waiter = NetworkWaiter(self._config.browser.page_load_timeout_seconds)
        proxy = self._proxy(store, telemetry)
        proxy_server = proxy.server if proxy is not None else None
        adapter = PlaywrightBrowserAdapter(
            self._config.browser, browser, viewport, telemetry, proxy_server
        )
        autologin = AutologinService(self._config.browser.autologin, extractor, executor, waiter)
        return BrowserSessionCoordinator(
            adapter,
            extractor,
            executor,
            waiter,
            ScreenshotService(),
            autologin,
            telemetry,
            BranchTraceCollector() if proxy is not None else None,
            proxy,
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
        store: LmdbRunStore,
        seen_branches: set[int],
    ) -> float:
        new_branches = set(after.branch_symbols) - seen_branches
        seen_branches.update(after.branch_symbols)
        new_network = set(after.network_symbols) - set(before.network_symbols)
        reward = RewardCalculator(self._config.policy.rewards).present(
            RewardSignals(
                action_succeeded=True,
                code_executed=bool(after.branch_symbols),
                new_branches_executed=bool(new_branches),
                network_traffic=bool(after.network_symbols),
                new_network_traffic=bool(new_network),
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
                "branch_symbols": list(after.branch_symbols),
                "network_symbols": list(after.network_symbols),
                "errors": list(after.errors),
                "viewport": [before.viewport.width, before.viewport.height],
                "action_targets": [
                    {
                        "bounds": [target.left, target.top, target.right, target.bottom],
                        "click": target.can_click,
                        "right_click": target.can_right_click,
                        "type": target.can_type,
                        "scroll": target.can_scroll,
                        "scroll_up": target.can_scroll_up,
                        "scroll_down": target.can_scroll_down,
                    }
                    for target in before.action_map.targets
                ],
                "screenshot": str(screenshot.relative_to(self._run_dir)),
            },
        )
        self._record_bugs(store, trace_id, after.errors)
        return reward

    @staticmethod
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

    @staticmethod
    def _step_index(store: LmdbRunStore) -> int:
        state = store.get("run", "state") or {}
        return int(state.get("testing_steps", 0))

    @staticmethod
    def _complete_step(
        store: LmdbRunStore,
        step_id: str,
        browser: BrowserKind,
        rewards: list[float],
        random_policy: bool,
    ) -> None:
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

    def _proxy(self, store: LmdbRunStore, telemetry: TelemetryBuffer) -> ProxyService | None:
        config = self._config.instrumentation
        if not config.enabled:
            return None
        blobs = AtomicBlobStore(self._run_dir / self._config.storage.blobs_directory)
        resources = ResourceRegistry(store, blobs, self._run_dir)
        addon = InstrumentationAddon(
            telemetry,
            resources,
            rewrite_html=config.rewrite_html,
            rewrite_javascript=config.rewrite_javascript,
            capture_resources=config.capture_resources,
        )
        return ProxyService(addon, config.proxy_port)

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

    def _dispatch(
        self,
        name: LifecycleEventName,
        subject_id: str | None = None,
        payload: tuple[tuple[str, object], ...] = (),
    ) -> None:
        self._hooks.dispatch(
            LifecycleEvent(
                name=name,
                occurred_at=self._clock(),
                run_id=self._run_dir.name,
                subject_id=subject_id,
                payload=payload,
            )
        )
