"""Composable Playwright browser services."""

from .adapter import PlaywrightBrowserAdapter
from .autologin import AutologinFailure, AutologinService
from .discovery import ACTION_MAP_ASSET_VERSION, ActionMapExtractor
from .executor import ActionExecutor
from .navigation import NavigationPolicy, OffsiteNavigationError
from .session import BrowserSessionCoordinator

__all__ = [
    "ACTION_MAP_ASSET_VERSION",
    "ActionExecutor",
    "ActionMapExtractor",
    "AutologinFailure",
    "AutologinService",
    "BrowserSessionCoordinator",
    "NavigationPolicy",
    "OffsiteNavigationError",
    "PlaywrightBrowserAdapter",
]
