"""Small browser-session coordinator composed from focused services."""

import time
from collections.abc import Callable

from kwola.domain.actions import Action, ActionMap
from kwola.domain.observations import Observation, Viewport
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
        self._clock = clock
        self._action_settle_ms = action_settle_seconds * 1000

    def start(self, target: str) -> Observation:
        self._adapter.start()
        try:
            self._adapter.navigate(target)
            self._waiter.wait(self._adapter.page)
            self._autologin.run(self._adapter.page)
            return self.observe()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._adapter.close()

    def discover_actions(self) -> ActionMap:
        return self._extractor.extract(self._adapter.page)

    def execute(self, action: Action) -> Observation:
        self._executor.execute(self._adapter.page, action)
        self._adapter.page.wait_for_timeout(self._action_settle_ms)
        self._waiter.wait(self._adapter.page)
        return self.observe()

    def observe(self) -> Observation:
        action_map = self.discover_actions()
        console, _network = self._telemetry.snapshot()
        return Observation(
            url=self._adapter.page.url,
            screenshot=self._screenshots.capture(self._adapter.page),
            viewport=Viewport(action_map.viewport_width, action_map.viewport_height),
            action_map=action_map,
            timestamp=self._clock(),
            console_messages=tuple(entry.message for entry in console),
        )
