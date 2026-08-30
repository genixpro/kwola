"""URL resolution and off-site navigation policy."""

from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urljoin, urlsplit


@dataclass(frozen=True, slots=True)
class NavigationPolicy:
    target: str
    prevent_offsite: bool = True

    def allows(self, candidate: str, current: str | None = None) -> bool:
        if not self.prevent_offsite:
            return True
        resolved = urlsplit(urljoin(current or self.target, candidate))
        target = urlsplit(self.target)
        if resolved.scheme not in {"http", "https"}:
            return resolved.scheme in {"", "about", "data", "blob"}
        return self._site(resolved.hostname) == self._site(target.hostname)

    @staticmethod
    def _site(hostname: str | None) -> str:
        host = (hostname or "").strip(".").lower()
        try:
            ip_address(host)
            return host
        except ValueError:
            parts = host.split(".")
            return ".".join(parts[-2:]) if len(parts) > 1 else host
