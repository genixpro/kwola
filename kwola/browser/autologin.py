"""Heuristic email/password login composed from browser services."""

from dataclasses import dataclass

from playwright.sync_api import Page

from kwola.config.models import LoginConfig
from kwola.domain.actions import Action, ActionKind, ActionTarget

from .discovery import ActionMapExtractor
from .executor import ActionExecutor
from .network import NetworkWaiter


class AutologinFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LoginElements:
    emails: tuple[ActionTarget, ...]
    passwords: tuple[ActionTarget, ...]
    submits: tuple[ActionTarget, ...]


class AutologinService:
    def __init__(
        self,
        config: LoginConfig,
        extractor: ActionMapExtractor,
        executor: ActionExecutor,
        waiter: NetworkWaiter,
    ) -> None:
        self._config = config
        self._extractor = extractor
        self._executor = executor
        self._waiter = waiter

    def run(self, page: Page) -> bool:
        if not self._config.enabled:
            return False
        elements = self._find(page)
        if not elements.emails and not elements.passwords and elements.submits:
            self._click(page, elements.submits[0])
            elements = self._find(page)
        typed_email = False
        if elements.emails and not elements.passwords and elements.submits:
            self._type(page, elements.emails[0], self._config.email or "")
            self._click(page, elements.submits[0])
            typed_email = True
            elements = self._find(page)
        incomplete = (
            (not elements.emails and not typed_email)
            or not elements.passwords
            or not elements.submits
        )
        if incomplete:
            raise AutologinFailure(
                "autologin could not find a usable email, password, and submit control"
            )
        before = page.url
        if not typed_email:
            self._type(page, elements.emails[0], self._config.email or "")
        self._type(page, elements.passwords[0], self._config.password or "")
        self._click(page, self._nearest_submit(elements))
        self._waiter.wait(page)
        return page.url != before

    def _find(self, page: Page) -> LoginElements:
        targets = self._extractor.extract(page).targets
        emails = tuple(
            target
            for target in targets
            if target.can_type
            and target.element_type == "input"
            and any(keyword in target.keywords for keyword in ("mail", "user", "name"))
        )
        passwords = tuple(
            target
            for target in targets
            if target.can_type and target.element_type == "input" and "pass" in target.keywords
        )
        submits = tuple(
            target
            for target in targets
            if target.can_click
            and any(keyword in target.keywords for keyword in ("log", "sub", "sign", "connexion"))
        )
        return LoginElements(emails, passwords, submits)

    def _nearest_submit(self, elements: LoginElements) -> ActionTarget:
        anchor = elements.passwords[0] if elements.passwords else elements.emails[0]
        below = tuple(target for target in elements.submits if target.top > anchor.bottom)
        choices = below or elements.submits
        return min(choices, key=lambda target: abs(target.top - anchor.bottom))

    def _type(self, page: Page, target: ActionTarget, text: str) -> None:
        x, y = target.center
        self._executor.execute(page, Action(ActionKind.TYPE, x, y, text=text, source="autologin"))

    def _click(self, page: Page, target: ActionTarget) -> None:
        x, y = target.center
        self._executor.execute(page, Action(ActionKind.CLICK, x, y, source="autologin"))
        self._waiter.wait(page)
