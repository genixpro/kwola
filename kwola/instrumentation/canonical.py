"""Canonical resource identities and stable JavaScript branch indexes."""

import hashlib
import re
from urllib.parse import urlparse, urlunparse

_UUID = re.compile(
    r"[a-f0-9]{8}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{4}(?:-?[a-f0-9]{12})?",
    re.IGNORECASE,
)
_DATE = re.compile(r"20\d{2}-\d\d-\d\d(?:T\d\d:\d\d:\d\d(?:\.\d{1,6})?)?")
_TIME = re.compile(r"\d\d:\d\d:\d\d(?:[.:]\d{1,6})?")
_LONG_NUMBER = re.compile(r"\d{8,}")
_PATH_ID = re.compile(r"/\d+")
_ALPHANUMERIC = re.compile(r"(?=[a-zA-Z]*\d)(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{16,}")


def canonicalize_url(url: str) -> str:
    parsed = list(urlparse(url))
    parsed[2] = _deunique(parsed[2])
    parsed[2] = _PATH_ID.sub("/__ID__", parsed[2])
    parsed[3] = ""
    parsed[4] = _deunique(parsed[4])
    parsed[5] = ""
    return urlunparse(parsed)


def resource_identity(url: str) -> str:
    return hashlib.sha256(canonicalize_url(url).encode()).hexdigest()[:10]


def _deunique(value: str) -> str:
    for pattern, label in (
        (_UUID, "HEXID"),
        (_DATE, "DATE"),
        (_TIME, "TIME"),
        (_LONG_NUMBER, "LONG"),
        (_ALPHANUMERIC, "ALPHANUMCODE"),
    ):
        value = pattern.sub(f"__{label}__", value)
    return value


class BranchIndexRealigner:
    _counter = re.compile(rb"(?<!window\.)globalKwolaCounter_\w+\[(\d+)\]")
    _any_counter = re.compile(rb"((?:window\.)?globalKwolaCounter_\w+\[)(\d+)(\])")
    _size = re.compile(rb"(globalKwolaCounter_\w+\s*=\s*new Uint32Array\()(\d+)(\))")

    def realign(self, prior: bytes, current: bytes) -> bytes:
        prior_signatures = _branch_signatures(prior, self._counter)
        current_signatures = _branch_signatures(current, self._counter)
        if not prior_signatures or not current_signatures:
            return current
        next_index = max(prior_signatures.values()) + 1
        mapping: dict[int, int] = {}
        for signature, index in current_signatures.items():
            mapped = prior_signatures.get(signature)
            if mapped is None:
                mapped = next_index
                next_index += 1
            mapping[index] = mapped

        def replace(match: re.Match[bytes]) -> bytes:
            old = int(match.group(2))
            return match.group(1) + str(mapping.get(old, old)).encode() + match.group(3)

        aligned = self._any_counter.sub(replace, current)
        required_size = max(mapping.values(), default=-1) + 1
        return self._size.sub(
            lambda match: (
                match.group(1)
                + str(max(required_size, int(match.group(2)))).encode()
                + match.group(3)
            ),
            aligned,
        )


def _branch_signatures(source: bytes, pattern: re.Pattern[bytes]) -> dict[bytes, int]:
    signatures = {}
    matches = list(pattern.finditer(source))
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(source)
        following = source[match.end() : min(end, match.end() + 240)]
        normalized = re.sub(rb"\s+|_[a-f0-9]{8,10}", b"", following)
        signatures[hashlib.sha256(normalized).digest()] = int(match.group(1))
    return signatures
