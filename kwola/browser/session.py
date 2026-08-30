"""Small browser-session coordinator composed from focused services."""

import time
from collections.abc import Callable

from kwola.domain.actions import Action, ActionMap
from kwola.domain.observations import Observation, Viewport
from kwola.instrumentation import BranchTraceCollector, BranchTraceSnapshot, ProxyService
from kwola.instrumentation.telemetry import TelemetryBuffer

from .adapter import PlaywrightBrowserAdapter
from .autologin import AutologinService
from .discovery import ActionMapExtractor
from .executor import ActionExecutor
from .network import NetworkWaiter
from .screenshots import ScreenshotService


class BrowserSessionCoordinator:
    def __init__(
        self,
        adapter: PlaywrightBrowserAdapter,
        extractor: ActionMapExtractor,
        executor: ActionExecutor,
        waiter: NetworkWaiter,
        screenshots: ScreenshotService,
        autologin: AutologinService,
        telemetry: TelemetryBuffer,
        branch_traces: BranchTraceCollector | None = None,
        proxy: ProxyService | None = None,
        clock: Callable[[], float] = time.time,
        action_settle_seconds: float = 0.25,
    ) -> None:
        self._adapter = adapter
        self._extractor = extractor
        self._executor = executor
        self._waiter = waiter
        self._screenshots = screenshots
        self._autologin = autologin
        self._telemetry = telemetry
        self._branch_traces = branch_traces
        self._proxy = proxy
        self._clock = clock
        self._action_settle_ms = action_settle_seconds * 1000

    def start(self, target: str) -> Observation:
        if self._proxy is not None:
            self._proxy.start()
        try:
            self._adapter.start()
            self._adapter.navigate(target)
            self._waiter.wait(self._adapter.page)
            self._autologin.run(self._adapter.page)
            return self.observe()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"browser session cleanup also failed: {cleanup_error}")
            raise

    def close(self) -> None:
        failure: BaseException | None = None
        try:
            self._adapter.close()
        except BaseException as error:
            failure = error
        if self._proxy is not None:
            try:
                self._proxy.close()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(f"proxy cleanup also failed: {error}")
        if failure is not None:
            raise failure

    def discover_actions(self) -> ActionMap:
        return self._extractor.extract(self._adapter.page)

    def execute(self, action: Action) -> Observation:
        self._executor.execute(self._adapter.page, action)
        self._adapter.page.wait_for_timeout(self._action_settle_ms)
        self._waiter.wait(self._adapter.page)
        self._adapter.ensure_allowed()
        return self.observe()

    def cursor_at(self, x: int, y: int) -> str:
        value = self._adapter.page.evaluate(
            "([x,y]) => { const e=document.elementFromPoint(x,y); "
            "return e ? getComputedStyle(e).cursor : 'none'; }",
            [x, y],
        )
        return str(value or "none")

    def page_html(self) -> str:
        return self._adapter.page.content()

    def observe(self) -> Observation:
        self._adapter.ensure_allowed()
        action_map = self.discover_actions()
        console, network = self._telemetry.snapshot()
        branches = self._collect_branches()
        return Observation(
            url=self._adapter.page.url,
            screenshot=self._screenshots.capture(self._adapter.page),
            viewport=Viewport(action_map.viewport_width, action_map.viewport_height),
            action_map=action_map,
            timestamp=self._clock(),
            branch_symbols=branches.symbols,
            network_symbols=self._telemetry.network_symbols(),
            console_messages=tuple(entry.message for entry in console),
            errors=_errors(console, network),
            branch_trace_available=branches.available,
            application_fitness=self._application_fitness(),
        )

    def _collect_branches(self) -> BranchTraceSnapshot:
        if self._branch_traces is None:
            return BranchTraceSnapshot(False, ())
        return self._branch_traces.collect(self._adapter.page)

    def _application_fitness(self) -> float | None:
        value = self._adapter.page.evaluate(
            "() => { const value = window.kwolaCumulativeFitness; "
            "return typeof value === 'number' && Number.isFinite(value) ? value : null; }",
            None,
        )
        return float(value) if isinstance(value, int | float) else None


def _errors(console: tuple[object, ...], network: tuple[object, ...]) -> tuple[str, ...]:
    messages = []
    for entry in console:
        level = getattr(entry, "level", "")
        if level in {"error", "assert"}:
            messages.append(f"console:{getattr(entry, 'message', '')}")
    for entry in network:
        status = int(getattr(entry, "status", 0))
        failure = getattr(entry, "failure", None)
        if failure or status >= 400:
            messages.append(f"network:{status}:{getattr(entry, 'url', '')}:{failure or ''}")
    return tuple(messages)
