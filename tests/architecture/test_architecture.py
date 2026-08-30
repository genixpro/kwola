import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[2] / "kwola"
MIGRATION_EXCLUSIONS = {"components", "datamodels", "diagnostics", "tasks"}
LEGACY_BIN_FILES = {
    "benchmark_neural_network.py",
    "create_kros3_experiment.py",
    "full_internal_test_suite.py",
    "initialize.py",
    "install_proxy_cert.py",
    "rapid_local_test_suite.py",
    "regenerate_charts.py",
    "reset.py",
    "run_multiple.py",
    "run_test_step.py",
    "run_train_step.py",
    "test_chromedriver.py",
    "test_ffmpeg.py",
    "test_installation.py",
    "test_javascript_rewriting.py",
    "test_neural_network.py",
    "train_agent.py",
    "website_check.py",
}


def production_modules() -> tuple[Path, ...]:
    modules = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        if relative.parts[0] in MIGRATION_EXCLUSIONS:
            continue
        if relative.parts[0] == "bin" and path.name in LEGACY_BIN_FILES:
            continue
        if relative == Path("config/config.py") or relative == Path("config/logger.py"):
            continue
        modules.append(path)
    return tuple(modules)


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


def test_new_architecture_has_no_finalizers_or_atexit() -> None:
    failures: list[str] = []
    for path in production_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__del__":
                failures.append(f"{path}:{node.lineno} defines __del__")
            if isinstance(node, ast.Import) and any(alias.name == "atexit" for alias in node.names):
                failures.append(f"{path}:{node.lineno} imports atexit")
    assert not failures, "\n".join(failures)
