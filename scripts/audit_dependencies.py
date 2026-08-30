#!/usr/bin/env python3
"""Apply Kwola's severity and time-bounded exception policy to audit JSON."""

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Finding:
    advisory_id: str
    package: str
    severity: str
    fixable: bool
    source: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-json", type=Path, required=True)
    parser.add_argument("--npm-json", type=Path, required=True)
    parser.add_argument("--exceptions", type=Path, required=True)
    parser.add_argument(
        "--require-no-exceptions",
        action="store_true",
        help="fail when any active exception exists, even when no finding matches it",
    )
    arguments = parser.parse_args(argv)
    try:
        exceptions = _load_exceptions(arguments.exceptions)
        findings = [
            *_python_findings(_load_json(arguments.python_json)),
            *_npm_findings(_load_json(arguments.npm_json)),
        ]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"dependency audit policy error: {error}", file=sys.stderr)
        return 2

    failures = []
    exception_gate_failed = arguments.require_no_exceptions and bool(exceptions)
    if exception_gate_failed:
        print(f"FAIL release gate has {len(exceptions)} active dependency audit exception(s)")
    for finding in findings:
        if (finding.advisory_id, finding.package) in exceptions:
            print(
                f"EXCEPTED {finding.source} {finding.advisory_id} "
                f"{finding.package} severity={finding.severity}"
            )
        elif finding.fixable and finding.severity in {"high", "critical"}:
            failures.append(finding)
            print(
                f"FAIL {finding.source} {finding.advisory_id} "
                f"{finding.package} severity={finding.severity} fixable=true"
            )
        else:
            print(
                f"REPORT {finding.source} {finding.advisory_id} "
                f"{finding.package} severity={finding.severity} "
                f"fixable={str(finding.fixable).lower()}"
            )
    if not findings:
        print("PASS no dependency advisories reported")
    return 1 if failures or exception_gate_failed else 0


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_exceptions(path: Path) -> set[tuple[str, str]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError("audit exceptions must be a JSON list")
    active: set[tuple[str, str]] = set()
    required = {"advisory_id", "package", "rationale", "owner", "expires"}
    today = date.today()
    for entry in payload:
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise ValueError(f"each audit exception requires {sorted(required)}")
        expiry = date.fromisoformat(str(entry["expires"]))
        advisory_id = str(entry["advisory_id"])
        package = str(entry["package"])
        if expiry < today:
            raise ValueError(f"audit exception expired: {advisory_id} for {package}")
        active.add((advisory_id, package))
    return active


def _python_findings(payload: Any) -> list[Finding]:
    if not isinstance(payload, dict) or not isinstance(payload.get("dependencies"), list):
        raise ValueError("pip-audit JSON is missing its dependencies list")
    findings = []
    for dependency in payload["dependencies"]:
        package = str(dependency.get("name", "unknown"))
        for vulnerability in dependency.get("vulns", []):
            advisory_id = str(vulnerability.get("id", "unknown"))
            aliases = [str(value) for value in vulnerability.get("aliases", [])]
            severity = _lookup_python_severity(advisory_id, aliases)
            findings.append(
                Finding(
                    advisory_id=advisory_id,
                    package=package,
                    severity=severity,
                    fixable=bool(vulnerability.get("fix_versions")),
                    source="pip-audit",
                )
            )
    return findings


def _lookup_python_severity(advisory_id: str, aliases: list[str]) -> str:
    candidates = [value for value in (advisory_id, *aliases) if value]
    for candidate in candidates:
        url = f"https://api.osv.dev/v1/vulns/{candidate}"
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise ValueError(
                f"OSV audit service failed for {candidate}: HTTP {error.code}"
            ) from error
        except (OSError, urllib.error.URLError) as error:
            raise ValueError(f"OSV audit service unavailable for {candidate}: {error}") from error
        severity = _severity_from_payload(payload)
        if severity is not None:
            return severity
    raise ValueError(
        f"audit service did not provide a usable severity for {advisory_id}; refusing to pass"
    )


def _severity_from_payload(payload: dict[str, Any]) -> str | None:
    severity = str(payload.get("database_specific", {}).get("severity", "")).lower()
    if severity in {"low", "moderate", "medium", "high", "critical"}:
        return "moderate" if severity == "medium" else severity
    for score in payload.get("severity", []):
        try:
            raw_score = str(score.get("score", ""))
            numeric = (
                _cvss_v3_score(raw_score) if raw_score.startswith("CVSS:3.") else float(raw_score)
            )
        except (KeyError, ValueError):
            continue
        if numeric >= 9:
            return "critical"
        if numeric >= 7:
            return "high"
        if numeric >= 4:
            return "moderate"
        return "low"
    return None


def _cvss_v3_score(vector: str) -> float:
    metrics = dict(field.split(":", maxsplit=1) for field in vector.split("/")[1:])
    scope_changed = metrics["S"] == "C"
    attack_vector = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}[metrics["AV"]]
    attack_complexity = {"L": 0.77, "H": 0.44}[metrics["AC"]]
    privileges = (
        {"N": 0.85, "L": 0.68, "H": 0.5} if scope_changed else {"N": 0.85, "L": 0.62, "H": 0.27}
    )[metrics["PR"]]
    interaction = {"N": 0.85, "R": 0.62}[metrics["UI"]]
    impact_values = {"H": 0.56, "L": 0.22, "N": 0.0}
    impact_subscore = 1 - math.prod(1 - impact_values[metrics[name]] for name in ("C", "I", "A"))
    impact = (
        7.52 * (impact_subscore - 0.029) - 3.25 * (impact_subscore - 0.02) ** 15
        if scope_changed
        else 6.42 * impact_subscore
    )
    if impact <= 0:
        return 0.0
    exploitability = 8.22 * attack_vector * attack_complexity * privileges * interaction
    base = min(10.0, (1.08 if scope_changed else 1.0) * (impact + exploitability))
    return math.ceil(base * 10) / 10


def _npm_findings(payload: Any) -> list[Finding]:
    if not isinstance(payload, dict) or not isinstance(payload.get("vulnerabilities"), dict):
        raise ValueError("npm audit JSON is missing its vulnerabilities object")
    findings = []
    for package, vulnerability in payload["vulnerabilities"].items():
        advisory_ids = {
            str(via.get("source", via.get("name", "unknown")))
            for via in vulnerability.get("via", [])
            if isinstance(via, dict)
        } or {str(package)}
        for advisory_id in sorted(advisory_ids):
            findings.append(
                Finding(
                    advisory_id=advisory_id,
                    package=str(package),
                    severity=str(vulnerability.get("severity", "unknown")).lower(),
                    fixable=vulnerability.get("fixAvailable", False) is not False,
                    source="npm-audit",
                )
            )
    return findings


if __name__ == "__main__":
    raise SystemExit(main())
