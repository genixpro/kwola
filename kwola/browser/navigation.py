"""Exact-origin navigation policy and containment errors."""

from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit


class OffsiteNavigationError(RuntimeError):
    """Raised when a document commits outside the configured origin boundary."""


@dataclass(frozen=True, slots=True)
class NavigationPolicy:
    target: str
    prevent_offsite: bool = True
    allowed_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _origin(self.target)
        for value in self.allowed_origins:
            _configured_origin(value)

    @property
    def origins(self) -> frozenset[tuple[str, str, int]]:
        return frozenset(
            (_origin(self.target), *(_configured_origin(value) for value in self.allowed_origins))
        )

    def allows(self, candidate: str, current: str | None = None) -> bool:
        if not self.prevent_offsite:
            return True
        try:
            resolved = urljoin(current or self.target, candidate)
            return _origin(resolved) in self.origins
        except ValueError:
            return False

    def require_allowed(self, candidate: str, current: str | None = None) -> None:
        if not self.allows(candidate, current):
            raise OffsiteNavigationError(f"blocked off-origin navigation to {candidate}")


def _configured_origin(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value)
    if parsed.username or parsed.password:
        raise ValueError("navigation origins cannot contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("navigation origins cannot contain a path, query, or fragment")
    return _origin(value)


def _origin(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("navigation origins must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("navigation URLs cannot contain credentials")
    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower().strip(".")
        port = parsed.port or (443 if scheme == "https" else 80)
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"invalid navigation origin: {value}") from error
    return scheme, host, port
