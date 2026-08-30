"""One explicit browser-testing runner."""

import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from kwola.agent import InferenceDiagnostics, InferencePolicy, PolicyMode, action_catalog
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
from kwola.config.models import RunConfig, ViewportConfig
from kwola.domain.actions import BrowserKind
from kwola.domain.observations import Observation
from kwola.hooks import HookRegistry, LifecycleEventName, testing_core_hooks
from kwola.instrumentation import (
    BranchTraceCollector,
    InstrumentationAddon,
    ProxyService,
    ResourceRegistry,
    TelemetryBuffer,
)
from kwola.storage import AtomicBlobStore, LmdbRunStore
from kwola.training.samples import RecordedSampleAssembler

from .lifecycle import RunnerLifecycle
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
        self._lifecycle = RunnerLifecycle(self._hooks, run_dir.name, clock)

    def run(
        self,
        *,
        random_policy: bool = False,
        policy_mode: PolicyMode = PolicyMode.SCHEDULED,
        browser: BrowserKind | None = None,
        viewport: tuple[int, int] | None = None,
        environment_index: int = 0,
        policy_seed: int | None = None,
    ) -> RunnerResult:
        started = self._clock()
        result: RunnerResult
        primary_error: BaseException | None = None
        try:
            self._dispatch(LifecycleEventName.RUN_STARTED)
            if random_policy:
                if policy_mode is not PolicyMode.SCHEDULED:
                    raise ValueError(
                        "random_policy cannot be combined with an explicit policy mode"
                    )
                policy_mode = PolicyMode.RANDOM
            browser_kind = browser or self._config.browser.enabled[0]
            if browser_kind not in self._config.browser.enabled:
                raise ValueError(f"browser {browser_kind} is not enabled for this run")
            with self._store() as store:
                result = self._run_step(
                    store,
                    started,
                    browser_kind,
                    self._viewport(viewport),
                    policy_mode,
                    environment_index,
                    policy_seed,
                )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            self._lifecycle.finish(primary_error)
        return result.model_copy(update={"warnings": self._lifecycle.warnings})

    def _run_step(
        self,
        store: LmdbRunStore,
        started: float,
        browser_kind: BrowserKind,
        viewport: ViewportConfig,
        policy_mode: PolicyMode,
        environment_index: int,
        policy_seed: int | None,
    ) -> RunnerResult:
        step_index = _reserve_step_index(store)
        step_id = f"testing-{step_index:08d}"
        session = self._session(browser_kind, viewport, store)
        artifacts: list[str] = []
        diagnostics: list[InferenceDiagnostics | None] = []
        primary_error: BaseException | None = None
        try:
            observation = session.start(str(self._config.target))
            self._dispatch(LifecycleEventName.SESSION_STARTED, step_id)
            rewards, fitness = self._actions(
                session,
                store,
                step_id,
                step_index,
                observation,
                artifacts,
                policy_mode,
                environment_index,
                policy_seed,
                diagnostics,
                _debug_video_due(self._config, step_index),
            )
            best_fitness = max(fitness) if fitness else None
            _complete_step(store, step_id, browser_kind, rewards, policy_mode, best_fitness)
            metrics: dict[str, int | float] = {
                "traces": len(rewards),
                "reward": sum(rewards),
            }
            if best_fitness is not None:
                metrics["application_fitness"] = best_fitness
            return RunnerResult(
                status="completed",
                step_id=step_id,
                duration_seconds=self._clock() - started,
                artifacts=tuple(artifacts),
                metrics=metrics,
            )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                session.close()
            except Exception:
                if primary_error is None:
                    raise
            self._lifecycle.dispatch_preserving(
                LifecycleEventName.SESSION_FINISHED,
                primary_error,
                step_id,
                (
                    ("store", store),
                    ("prepare_samples", lambda: self._prepare_samples(store, step_id)),
                    ("diagnostics", tuple(diagnostics)),
                ),
            )

    def _actions(
        self,
        session: BrowserSessionCoordinator,
        store: LmdbRunStore,
        step_id: str,
        step_index: int,
        observation: Observation,
        artifacts: list[str],
        policy_mode: PolicyMode,
        environment_index: int,
        policy_seed: int | None,
        diagnostics: list[InferenceDiagnostics | None],
        capture_diagnostics: bool,
    ) -> tuple[list[float], list[float]]:
        seed = (
            policy_seed
            if policy_seed is not None
            else self._config.seed + step_index + environment_index * 1_000_003
        )
        policy = InferencePolicy(self._run_dir, self._config, random.Random(seed))
        recorder = TraceRecorder(self._run_dir, self._config, store, artifacts)
        novelty = NoveltyState.initial(observation)
        recorder.claim_initial(observation)
        rewards: list[float] = []
        fitness = _fitness_values(observation)
        for trace_index in range(self._config.policy.testing_sequence_length):
            action = policy.select(
                observation,
                action_index=trace_index,
                test_step_index=step_index,
                mode=policy_mode,
                capture_diagnostics=capture_diagnostics,
            )
            if capture_diagnostics:
                diagnostics.append(policy.take_diagnostics())
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
            fitness.extend(_fitness_values(observation))
        return rewards, fitness

    def _session(
        self,
        browser: BrowserKind,
        viewport: ViewportConfig,
        store: LmdbRunStore,
    ) -> BrowserSessionCoordinator:
        telemetry = TelemetryBuffer()
        navigation = NavigationPolicy(
            str(self._config.target),
            self._config.browser.prevent_offsite_navigation,
            tuple(str(origin) for origin in self._config.browser.allowed_navigation_origins),
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
            navigation,
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

    def _prepare_samples(self, store: LmdbRunStore, step_id: str) -> None:
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
            decoded_image_cache_size=config.training.decoded_image_cache_size,
        )
        assembler.prepare_step(step_id, config.training.sample_cache_workers)

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
        self._lifecycle.dispatch(name, subject_id, payload)


def _reserve_step_index(store: LmdbRunStore) -> int:
    def reserve(current: dict[str, Any] | None) -> dict[str, Any]:
        state = dict(current or {})
        index = int(cast(int | str, state.get("next_testing_step", state.get("testing_steps", 0))))
        state["next_testing_step"] = index + 1
        return state

    state = store.update("run", "state", reserve)
    return int(state["next_testing_step"]) - 1


def _fitness_values(observation: Observation) -> list[float]:
    return [observation.application_fitness] if observation.application_fitness is not None else []


def _debug_video_due(config: RunConfig, step_index: int) -> bool:
    reporting = config.reporting
    return reporting.debug_videos and step_index % reporting.debug_video_every_testing_steps == 0


def _complete_step(
    store: LmdbRunStore,
    step_id: str,
    browser: BrowserKind,
    rewards: list[float],
    policy_mode: PolicyMode,
    application_fitness: float | None,
) -> None:
    def complete(current: dict[str, Any] | None) -> dict[str, Any]:
        state = dict(current or {})
        completed = int(step_id.rsplit("-", maxsplit=1)[1]) + 1
        state["testing_steps"] = max(int(cast(int | str, state.get("testing_steps", 0))), completed)
        return state

    store.update("run", "state", complete)
    store.put(
        "testing_steps",
        step_id,
        {
            "browser": browser.value,
            "random": policy_mode is PolicyMode.RANDOM,
            "policy_mode": policy_mode.value,
            "trace_count": len(rewards),
            "reward": sum(rewards),
            "application_fitness": application_fitness,
        },
    )
