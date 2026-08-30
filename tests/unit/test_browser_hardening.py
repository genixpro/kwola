import builtins
from types import SimpleNamespace

import pytest

from kwola.browser.adapter import PlaywrightBrowserAdapter
from kwola.browser.autologin import AutologinFailure, AutologinService
from kwola.browser.executor import ActionExecutor
from kwola.browser.navigation import NavigationPolicy
from kwola.config import profile_config
from kwola.config.models import LoginConfig
from kwola.domain.actions import Action, ActionKind, ActionMap, ActionTarget, BrowserKind
from kwola.instrumentation import certificates
from kwola.instrumentation import proxy as proxy_module
from kwola.instrumentation.proxy import ProxyService, ProxyStartupError
from kwola.instrumentation.telemetry import TelemetryBuffer


class FakeMouse:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def click(self, *args: object, **values: object) -> None:
        self.calls.append(("click", args, values))

    def dblclick(self, *args: object, **values: object) -> None:
        self.calls.append(("dblclick", args, values))

    def move(self, *args: object, **values: object) -> None:
        self.calls.append(("move", args, values))

    def wheel(self, *args: object, **values: object) -> None:
        self.calls.append(("wheel", args, values))


class FakeKeyboard:
    def __init__(self) -> None:
        self.pressed: list[str] = []
        self.typed: list[str] = []

    def press(self, value: str) -> None:
        self.pressed.append(value)

    def type(self, value: str) -> None:
        self.typed.append(value)


def test_action_executor_covers_every_browser_action(monkeypatch: pytest.MonkeyPatch) -> None:
    page = SimpleNamespace(mouse=FakeMouse(), keyboard=FakeKeyboard())
    executor = ActionExecutor()
    actions = (
        Action(ActionKind.CLICK, 1, 2),
        Action(ActionKind.DOUBLE_CLICK, 3, 4),
        Action(ActionKind.RIGHT_CLICK, 5, 6),
        Action(ActionKind.CLEAR, 7, 8),
        Action(ActionKind.TYPE, 9, 10, text="hello"),
        Action(ActionKind.SCROLL, 11, 12, direction="up"),
        Action(ActionKind.SCROLL, 13, 14, direction="down"),
    )
    monkeypatch.setattr("kwola.browser.executor.sys.platform", "linux")

    for action in actions:
        executor.execute(page, action)  # type: ignore[arg-type]

    assert page.keyboard.pressed == ["Control+A", "Backspace"]
    assert page.keyboard.typed == ["hello"]
    assert ("click", (5, 6), {"button": "right"}) in page.mouse.calls
    assert ("wheel", (0, -600), {}) in page.mouse.calls
    assert ("wheel", (0, 600), {}) in page.mouse.calls
    with pytest.raises(ValueError, match="unsupported"):
        executor.execute(page, SimpleNamespace(kind="unsupported", x=0, y=0))  # type: ignore[arg-type]


def _login_target(kind: str) -> ActionTarget:
    if kind == "email":
        return ActionTarget(0, 0, 100, 20, "input", "email user", can_type=True)
    if kind == "password":
        return ActionTarget(0, 30, 100, 50, "input", "password", can_type=True)
    return ActionTarget(0, 60, 100, 80, "button", "login submit", can_click=True)


class LoginExtractor:
    def __init__(self, maps: list[ActionMap]) -> None:
        self.maps = maps

    def extract(self, _page: object) -> ActionMap:
        return self.maps.pop(0) if len(self.maps) > 1 else self.maps[0]


class LoginExecutor:
    def __init__(self) -> None:
        self.actions: list[Action] = []

    def execute(self, page: object, action: Action) -> None:
        self.actions.append(action)
        if action.kind is ActionKind.CLICK:
            page.url = f"https://example.test/page-{len(self.actions)}"  # type: ignore[attr-defined]


def test_autologin_resolves_environment_credentials_and_submits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOGIN_EMAIL", "person@example.test")
    monkeypatch.setenv("LOGIN_PASSWORD", "correct-horse")
    config = LoginConfig(
        enabled=True,
        email_environment="LOGIN_EMAIL",
        password_environment="LOGIN_PASSWORD",
    )
    action_map = ActionMap(
        (_login_target("email"), _login_target("password"), _login_target("submit")),
        800,
        600,
        "test",
    )
    executor = LoginExecutor()
    waiter = SimpleNamespace(wait=lambda _page: True)
    service = AutologinService(
        config,
        LoginExtractor([action_map]),  # type: ignore[arg-type]
        executor,  # type: ignore[arg-type]
        waiter,  # type: ignore[arg-type]
    )
    page = SimpleNamespace(url="https://example.test/login")

    assert service.run(page)  # type: ignore[arg-type]
    assert [action.text for action in executor.actions[:2]] == [
        "person@example.test",
        "correct-horse",
    ]
    assert executor.actions[-1].kind is ActionKind.CLICK


def test_autologin_rejects_incomplete_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGIN_EMAIL", "person@example.test")
    monkeypatch.setenv("LOGIN_PASSWORD", "correct-horse")
    config = LoginConfig(
        enabled=True,
        email_environment="LOGIN_EMAIL",
        password_environment="LOGIN_PASSWORD",
    )
    action_map = ActionMap((_login_target("email"),), 800, 600, "test")
    service = AutologinService(
        config,
        LoginExtractor([action_map]),  # type: ignore[arg-type]
        LoginExecutor(),  # type: ignore[arg-type]
        SimpleNamespace(wait=lambda _page: True),  # type: ignore[arg-type]
    )

    with pytest.raises(AutologinFailure, match="usable"):
        service.run(SimpleNamespace(url="https://example.test/login"))  # type: ignore[arg-type]


class ClosingResource:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.closed = False

    def close(self) -> None:
        self.closed = True
        if self.failure is not None:
            raise self.failure


class StoppingResource:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_browser_cleanup_attempts_every_resource_after_a_failure() -> None:
    config = profile_config("testing", "https://example.test", 1).browser
    adapter = PlaywrightBrowserAdapter(
        config,
        BrowserKind.CHROMIUM,
        config.viewports[0],
        TelemetryBuffer(),
        NavigationPolicy("https://example.test"),
    )
    context = ClosingResource(RuntimeError("context close failed"))
    browser = ClosingResource()
    playwright = StoppingResource()
    adapter._context = context  # type: ignore[assignment]
    adapter._browser = browser  # type: ignore[assignment]
    adapter._playwright = playwright  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="context close failed"):
        adapter.close()

    assert context.closed and browser.closed and playwright.stopped
    assert adapter._context is None and adapter._browser is None and adapter._playwright is None


def test_proxy_startup_surfaces_background_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_module, "_free_port", lambda: 12345)
    service = ProxyService(SimpleNamespace())  # type: ignore[arg-type]

    def fail() -> None:
        service._failure = RuntimeError("bind failed")
        service._ready.set()
        service._stopped.set()

    monkeypatch.setattr(service, "_run", fail)

    with pytest.raises(ProxyStartupError, match="bind failed"):
        service.start(timeout_seconds=1)
    assert service._thread is None


class CertificatePage:
    def __init__(self) -> None:
        self.visited = ""

    def goto(self, target: str, **_values: object) -> None:
        self.visited = target


class CertificateBrowser:
    def __init__(self) -> None:
        self.page = CertificatePage()
        self.closed = False

    def new_page(self) -> CertificatePage:
        return self.page

    def close(self) -> None:
        self.closed = True


class CertificateProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.waited = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, **_values: object) -> None:
        self.waited = True


def test_certificate_installation_cleans_up_proxy_and_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = CertificateProcess()
    browser = CertificateBrowser()
    chromium = SimpleNamespace(launch=lambda **_values: browser)
    playwright = SimpleNamespace(chromium=chromium)
    manager = SimpleNamespace(
        __enter__=lambda _self: playwright,
        __exit__=lambda _self, *_args: None,
    )
    manager_type = type("PlaywrightManager", (), dict(vars(manager)))
    monkeypatch.setattr(certificates.shutil, "which", lambda _name: "/bin/mitmdump")
    monkeypatch.setattr(certificates, "_free_port", lambda: 12345)
    monkeypatch.setattr(certificates.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(certificates, "sync_playwright", manager_type)
    monkeypatch.setattr(builtins, "input", lambda _prompt: "")

    certificates.install_certificate()

    assert browser.page.visited == "http://mitm.it/"
    assert browser.closed
    assert process.terminated and process.waited
