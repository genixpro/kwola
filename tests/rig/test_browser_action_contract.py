import os

import pytest

from kwola.browser.adapter import PlaywrightBrowserAdapter
from kwola.browser.discovery import ActionMapExtractor
from kwola.browser.executor import ActionExecutor
from kwola.browser.navigation import NavigationPolicy
from kwola.browser.network import NetworkWaiter
from kwola.browser.screenshots import ScreenshotService
from kwola.config import profile_config
from kwola.config.models import ViewportConfig
from kwola.domain.actions import Action, ActionKind, BrowserKind
from kwola.instrumentation.telemetry import TelemetryBuffer

pytestmark = pytest.mark.skipif(
    os.environ.get("KWOLA_RIG_ACCEPTANCE") != "1",
    reason="requires the Kros services and rig browsers",
)


@pytest.mark.parametrize("browser", (BrowserKind.CHROMIUM, BrowserKind.FIREFOX))
def test_kros_action_executor_contract(browser: BrowserKind) -> None:
    target = "http://127.0.0.1:3001/"
    browser_config = profile_config("testing", target, 71).browser
    adapter = PlaywrightBrowserAdapter(
        browser_config,
        browser,
        ViewportConfig(width=1024, height=768),
        TelemetryBuffer(),
        NavigationPolicy(target, prevent_offsite=True),
    )
    extractor = ActionMapExtractor(NavigationPolicy(target, prevent_offsite=True))
    executor = ActionExecutor()
    waiter = NetworkWaiter(browser_config.page_load_timeout_seconds, 0.1)
    try:
        adapter.start()
        adapter.navigate(target)
        assert waiter.wait(adapter.page)
        initial = extractor.extract(adapter.page)
        login = next(
            item
            for item in initial.targets
            if item.can_click and any(word in item.keywords for word in ("login", "log in"))
        )
        executor.execute(adapter.page, Action(ActionKind.CLICK, *login.center))
        assert waiter.wait(adapter.page)
        login_map = extractor.extract(adapter.page)
        text_target = next(item for item in login_map.targets if item.can_type)
        executor.execute(
            adapter.page,
            Action(ActionKind.TYPE, *text_target.center, text="contract@example.test"),
        )
        executor.execute(adapter.page, Action(ActionKind.CLEAR, *text_target.center))
        executor.execute(adapter.page, Action(ActionKind.RIGHT_CLICK, *text_target.center))
        executor.execute(adapter.page, Action(ActionKind.DOUBLE_CLICK, *text_target.center))
        executor.execute(
            adapter.page,
            Action(ActionKind.SCROLL, 512, 384, direction="down"),
        )
        executor.execute(adapter.page, Action(ActionKind.SCROLL, 512, 384, direction="up"))
        assert ScreenshotService().capture(adapter.page).startswith(b"\x89PNG")
        assert adapter.page.url.startswith(target)
    finally:
        adapter.close()
