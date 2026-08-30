from kwola.instrumentation.canonical import BranchIndexRealigner, canonicalize_url


def test_resource_urls_canonicalize_dynamic_identifiers() -> None:
    first = "https://example.com/api/12345678/app-a12b34c56d78e90f.js?build=2026-08-29#x"
    second = "https://example.com/api/87654321/app-f98e76d54c32b10a.js?build=2026-08-30#y"
    assert canonicalize_url(first) == canonicalize_url(second)
    assert canonicalize_url(first).endswith("?build=__DATE__")


def test_branch_indexes_survive_inserted_javascript_branches() -> None:
    prior = (
        b"var globalKwolaCounter_abcd1234 = new Uint32Array(2);"
        b"globalKwolaCounter_abcd1234[0] += 1;alpha();"
        b"globalKwolaCounter_abcd1234[1] += 1;beta();"
    )
    current = (
        b"var globalKwolaCounter_abcd1234 = new Uint32Array(3);"
        b"globalKwolaCounter_abcd1234[0] += 1;inserted();"
        b"globalKwolaCounter_abcd1234[1] += 1;alpha();"
        b"globalKwolaCounter_abcd1234[2] += 1;beta();"
    )
    aligned = BranchIndexRealigner().realign(prior, current)
    assert b"[0] += 1;alpha()" in aligned
    assert b"[1] += 1;beta()" in aligned
    assert b"[2] += 1;inserted()" in aligned
