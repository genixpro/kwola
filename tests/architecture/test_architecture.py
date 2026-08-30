import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[2] / "kwola"


def production_modules() -> tuple[Path, ...]:
    return tuple(PACKAGE.rglob("*.py"))


def test_production_size_limits() -> None:
    failures: list[str] = []
    for path in production_modules():
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        if len(lines) > 500:
            failures.append(f"module {path} has {len(lines)} lines")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.end_lineno:
                size = node.end_lineno - node.lineno + 1
                if size > 300:
                    failures.append(f"class {path}:{node.lineno} has {size} lines")
            if isinstance(node, ast.FunctionDef) and node.end_lineno:
                size = node.end_lineno - node.lineno + 1
                if size > 80:
                    failures.append(f"function {path}:{node.lineno} has {size} lines")
    assert not failures, "\n".join(failures)


def test_domain_has_no_infrastructure_imports() -> None:
    forbidden = {"playwright", "torch", "lmdb", "subprocess", "kwola.reporting"}
    failures: list[str] = []
    for path in (PACKAGE / "domain").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        for imported in imports:
            if any(imported == name or imported.startswith(f"{name}.") for name in forbidden):
                failures.append(f"{path} imports {imported}")
    assert not failures, "\n".join(failures)


def test_architecture_has_no_finalizers_or_atexit() -> None:
    failures: list[str] = []
    for path in production_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__del__":
                failures.append(f"{path}:{node.lineno} defines __del__")
            if isinstance(node, ast.Import) and any(alias.name == "atexit" for alias in node.names):
                failures.append(f"{path}:{node.lineno} imports atexit")
    assert not failures, "\n".join(failures)
