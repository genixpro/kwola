from pathlib import Path

from kwola.instrumentation import HtmlRewriter, JavaScriptRewriter, ResourceRegistry
from kwola.storage import AtomicBlobStore, LmdbRunStore


def test_html_rewriter_removes_subresource_integrity() -> None:
    source = b'<script integrity="sha384-YWJjZA==" src="app.js"></script>'
    assert b"integrity" not in HtmlRewriter().rewrite(source)


def test_javascript_rewriter_installs_kwola_branch_counters() -> None:
    source = b"function choose(value) { if (value) return 1; return 2; } choose(true);"
    with JavaScriptRewriter() as rewriter:
        rewritten = rewriter.rewrite("https://example.test/app.js", source)
        assert rewriter.rewrite("https://example.test/app.js", source) == rewritten
        second = rewriter.rewrite("https://example.test/other.js", source)
    assert b"window.kwolaCounters" in rewritten
    assert b"globalKwolaCounter_" in rewritten
    assert b"globalKwolaCounter_" in second


def test_resource_registry_deduplicates_content_blobs(tmp_path: Path) -> None:
    blobs = AtomicBlobStore(tmp_path / "blobs")
    with LmdbRunStore(tmp_path / "run.lmdb", map_size=1024**2) as store:
        registry = ResourceRegistry(store, blobs, tmp_path)
        first = registry.capture(
            url="https://example.test/app.js",
            status=200,
            content_type="application/javascript",
            headers={"Content-Type": "application/javascript"},
            original=b"const value = 1;",
            delivered=b"const value = 1;",
            rewrite_kind=None,
        )
        second = registry.capture(
            url="https://example.test/app.js",
            status=200,
            content_type="application/javascript",
            headers={},
            original=b"const value = 1;",
            delivered=b"instrumented",
            rewrite_kind="javascript",
        )
        assert first == second
        assert store.get("resources", first)["rewrite_kind"] == "javascript"  # type: ignore[index]
    assert len(blobs.list("resources")) == 1
