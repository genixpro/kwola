"""One explicit browser-testing runner."""

import random
import time
from collections.abc import Callable
from pathlib import Path

from kwola.agent import InferencePolicy, action_catalog
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
from kwola.domain.actions import BrowserKind
from kwola.hooks import (
    HookRegistry,
    LifecycleEvent,
    LifecycleEventName,
    testing_core_hooks,
)
from kwola.instrumentation import (
    BranchTraceCollector,
    InstrumentationAddon,
    ProxyService,
    ResourceRegistry,
    TelemetryBuffer,
)
from kwola.storage import AtomicBlobStore, LmdbRunStore
from kwola.training.samples import RecordedSampleAssembler

from .results import RunnerResult
from .trace_recorder import NoveltyState, TraceRecorder


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
        self._hooks = hooks or HookRegistry(testing_core_hooks(run_dir, self._config))

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
            with self._store() as store:
                return self._run_step(
                    store,
                    started,
                    browser_kind,
                    self._viewport(viewport),
                    random_policy,
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
            recorder = TraceRecorder(self._run_dir, self._config, store, artifacts)
            novelty = NoveltyState.initial(observation)
            rewards = []
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
                cursor = session.cursor_at(action.x, action.y)
                before_html = self._html(session)
                observation = session.execute(action)
                after_html = self._html(session)
                self._dispatch(
                    LifecycleEventName.AFTER_ACTION,
                    trace_id,
                    (
                        ("store", store),
                        ("console_messages", len(observation.console_messages)),
                        ("network_symbols", len(observation.network_symbols)),
                    ),
                )
                reward = recorder.record(
                    step_id,
                    trace_index,
                    action,
                    before,
                    observation,
                    novelty,
                    cursor,
                    (before_html, after_html),
                )
                self._dispatch(
                    LifecycleEventName.TRACE_RECORDED,
                    trace_id,
                    (("store", store), ("reward", reward)),
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
            self._dispatch(
                LifecycleEventName.SESSION_FINISHED,
                step_id,
                (
                    ("store", store),
                    ("prepare_samples", lambda: self._prepare_samples(store)),
                ),
            )

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
        waiter = NetworkWaiter(
            self._config.browser.page_load_timeout_seconds,
            self._config.browser.network_idle_seconds,
        )
        proxy = self._proxy(store, telemetry)
        adapter = PlaywrightBrowserAdapter(
            self._config.browser,
            browser,
            viewport,
            telemetry,
            proxy.server if proxy else None,
            capture_console=self._config.instrumentation.capture_console,
            capture_network=self._config.instrumentation.capture_network,
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
            BranchTraceCollector(
                self._config.instrumentation.branch_trace_timeout_seconds,
                self._config.browser.action_timeout_seconds,
            )
            if proxy
            else None,
            proxy,
            clock=self._clock,
            action_settle_seconds=self._config.browser.action_settle_seconds,
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

    def _html(self, session: BrowserSessionCoordinator) -> str | None:
        return session.page_html() if self._config.instrumentation.capture_html else None

    def _store(self) -> LmdbRunStore:
        return LmdbRunStore(
            self._run_dir / self._config.storage.database_directory,
            map_size=self._config.storage.database_map_size_bytes,
            compression_level=self._config.storage.codec_compression_level,
        )

    def _prepare_samples(self, store: LmdbRunStore) -> None:
        config = self._config
        assembler = RecordedSampleAssembler(
            self._run_dir,
            store,
            symbol_dictionary_size=config.model.symbol_dictionary_size,
            discount_rate=config.policy.rewards.discount_rate,
            max_discounted_reward=config.policy.rewards.max_discounted_reward,
            cache_version=config.training.sample_cache_version,
            channels=action_catalog(config.policy),
            recent_action_history=config.model.recent_action_history,
            recent_action_radius=config.training.recent_action_image_radius,
            recent_action_decay=config.training.recent_action_image_decay,
            image_downscale_ratio=config.model.image_downscale_ratio,
            crop_size=(config.training.crop_width, config.training.crop_height),
            next_crop_size=(config.training.next_crop_width, config.training.next_crop_height),
            crop_random=(config.training.crop_random_x, config.training.crop_random_y),
        )
        assembler.prepare_cache(config.training.sample_cache_workers)

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
