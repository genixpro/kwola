"""Detected application failures."""

from dataclasses import dataclass
from enum import StrEnum


class BugKind(StrEnum):
    HTTP = "http"
    CONSOLE = "console"
    EXCEPTION = "exception"
    DOTNET_RPC = "dotnet_rpc"


@dataclass(frozen=True, slots=True)
class Bug:
    id: str
    kind: BugKind
    message: str
    session_id: str
    trace_id: str
    url: str
    first_seen_at: float
    details: tuple[tuple[str, str], ...] = ()
