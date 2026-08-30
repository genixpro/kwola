import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "audit_dependencies.py"
SPEC = importlib.util.spec_from_file_location("kwola_dependency_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_dependencies = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_dependencies
SPEC.loader.exec_module(audit_dependencies)


def test_expired_audit_exception_is_rejected(tmp_path: Path) -> None:
    exceptions = tmp_path / "exceptions.json"
    exceptions.write_text(
        json.dumps(
            [
                {
                    "advisory_id": "PYSEC-expired",
                    "package": "example",
                    "rationale": "fixture",
                    "owner": "tests",
                    "expires": "2020-01-01",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expired"):
        audit_dependencies._load_exceptions(exceptions)


def test_missing_osv_severity_refuses_to_classify_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit_dependencies, "_severity_from_payload", lambda payload: None)

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self, *_: object) -> bytes:
            return b"{}"

    monkeypatch.setattr(audit_dependencies.urllib.request, "urlopen", lambda *_a, **_k: Response())

    with pytest.raises(ValueError, match="usable severity"):
        audit_dependencies._lookup_python_severity("PYSEC-missing", [])


def test_osv_severity_parses_labels_numeric_scores_and_cvss_vectors() -> None:
    assert (
        audit_dependencies._severity_from_payload({"database_specific": {"severity": "medium"}})
        == "moderate"
    )
    assert audit_dependencies._severity_from_payload({"severity": [{"score": "9.1"}]}) == "critical"
    assert (
        audit_dependencies._severity_from_payload(
            {"severity": [{"score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H"}]}
        )
        == "high"
    )


@pytest.mark.parametrize(
    ("fix_available", "expected"),
    ((True, 1), (False, 1)),
)
def test_high_npm_finding_always_requires_an_exception(
    tmp_path: Path, fix_available: bool, expected: int
) -> None:
    python_json, npm_json, exceptions = _audit_inputs(tmp_path, fix_available)

    assert (
        audit_dependencies.main(
            [
                "--python-json",
                str(python_json),
                "--npm-json",
                str(npm_json),
                "--exceptions",
                str(exceptions),
            ]
        )
        == expected
    )


def test_matching_exception_allows_unfixable_high_finding(tmp_path: Path) -> None:
    python_json, npm_json, exceptions = _audit_inputs(tmp_path, False)
    exceptions.write_text(
        json.dumps(
            [
                {
                    "advisory_id": "GHSA-fixture",
                    "package": "example",
                    "rationale": "upstream has not published a compatible fix",
                    "owner": "tests",
                    "expires": "2099-01-01",
                }
            ]
        ),
        encoding="utf-8",
    )

    assert (
        audit_dependencies.main(
            [
                "--python-json",
                str(python_json),
                "--npm-json",
                str(npm_json),
                "--exceptions",
                str(exceptions),
            ]
        )
        == 0
    )


def test_release_gate_rejects_every_active_exception(tmp_path: Path) -> None:
    python_json, npm_json, exceptions = _audit_inputs(tmp_path, False)
    exceptions.write_text(
        json.dumps(
            [
                {
                    "advisory_id": "PYSEC-active",
                    "package": "example",
                    "rationale": "fixture",
                    "owner": "tests",
                    "expires": "2099-01-01",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = audit_dependencies.main(
        [
            "--python-json",
            str(python_json),
            "--npm-json",
            str(npm_json),
            "--exceptions",
            str(exceptions),
            "--require-no-exceptions",
        ]
    )

    assert result == 1


def _audit_inputs(tmp_path: Path, fix_available: bool) -> tuple[Path, Path, Path]:
    python_json = tmp_path / "python.json"
    npm_json = tmp_path / "npm.json"
    exceptions = tmp_path / "exceptions.json"
    python_json.write_text('{"dependencies": []}', encoding="utf-8")
    npm_json.write_text(
        json.dumps(
            {
                "vulnerabilities": {
                    "example": {
                        "severity": "high",
                        "fixAvailable": fix_available,
                        "via": [{"source": "GHSA-fixture"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    exceptions.write_text("[]", encoding="utf-8")
    return python_json, npm_json, exceptions
