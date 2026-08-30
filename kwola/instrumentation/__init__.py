"""Proxy services, rewriting, resources, and browser telemetry."""

from .addon import InstrumentationAddon
from .branches import BranchTraceCollector
from .proxy import ProxyService
from .resources import ResourceRegistry
from .rewriting import HtmlRewriter, JavaScriptRewriter, RewriteError
from .telemetry import ConsoleEntry, NetworkEntry, TelemetryBuffer

__all__ = [
    "BranchTraceCollector",
    "ConsoleEntry",
    "HtmlRewriter",
    "InstrumentationAddon",
    "JavaScriptRewriter",
    "NetworkEntry",
    "ProxyService",
    "ResourceRegistry",
    "RewriteError",
    "TelemetryBuffer",
]
