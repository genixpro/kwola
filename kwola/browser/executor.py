"""Playwright action execution."""

import sys

from playwright.sync_api import Page

from kwola.domain.actions import Action, ActionKind


class ActionExecutor:
    def execute(self, page: Page, action: Action) -> None:
        if action.kind is ActionKind.CLICK:
            page.mouse.click(action.x, action.y)
        elif action.kind is ActionKind.DOUBLE_CLICK:
            page.mouse.dblclick(action.x, action.y)
        elif action.kind is ActionKind.RIGHT_CLICK:
            page.mouse.click(action.x, action.y, button="right")
        elif action.kind is ActionKind.CLEAR:
            page.mouse.click(action.x, action.y)
            page.keyboard.press("Meta+A" if sys.platform == "darwin" else "Control+A")
            page.keyboard.press("Backspace")
        elif action.kind is ActionKind.TYPE:
            page.mouse.click(action.x, action.y)
            page.keyboard.type(action.text or "")
        elif action.kind is ActionKind.SCROLL:
            delta = -600 if action.direction == "up" else 600
            page.mouse.move(action.x, action.y)
            page.mouse.wheel(0, delta)
        else:
            raise ValueError(f"unsupported action kind: {action.kind}")
