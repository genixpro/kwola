"""The single supported Kwola command-line interface."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kwola.domain.actions import BrowserKind
from kwola.orchestration.doctor import run_doctor
from kwola.orchestration.initialize import initialize_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kwola", description="AI-powered browser testing")
    subcommands = parser.add_subparsers(dest="command", required=True)
    _init_parser(subcommands)
    _run_parser(subcommands)
    _test_step_parser(subcommands)
    _train_step_parser(subcommands)
    _run_dir_parser(subcommands, "report", "Generate reports for a run")
    _doctor_parser(subcommands)
    _run_dir_parser(subcommands, "benchmark", "Benchmark model training for a run")
    _run_dir_parser(subcommands, "status", "Show live pipeline throughput for a run")
    _proxy_parser(subcommands)
    return parser


def _init_parser(subcommands: Any) -> None:
    command = subcommands.add_parser("init", help="Create a fresh run")
    command.add_argument("url")
    command.add_argument("--profile", choices=("testing", "standard", "rig"), default="rig")
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--seed", type=int, default=0)
    command.set_defaults(handler=_handle_init)


def _run_parser(subcommands: Any) -> None:
    command = subcommands.add_parser("run", help="Run testing and training")
    command.add_argument("run_dir", type=Path)
    command.set_defaults(handler=_handle_run)


def _test_step_parser(subcommands: Any) -> None:
    command = subcommands.add_parser("test-step", help="Run one browser testing step")
    command.add_argument("run_dir", type=Path)
    command.add_argument("--random", action="store_true", dest="random_policy")
    command.add_argument("--browser", choices=tuple(BrowserKind), type=BrowserKind)
    command.add_argument("--viewport", type=_viewport)
    command.set_defaults(handler=_handle_test_step)


def _train_step_parser(subcommands: Any) -> None:
    command = subcommands.add_parser("train-step", help="Run one optimizer step")
    command.add_argument("run_dir", type=Path)
    command.add_argument("--gpu", type=int)
    command.set_defaults(handler=_handle_train_step)


def _run_dir_parser(subcommands: Any, name: str, help_text: str) -> None:
    command = subcommands.add_parser(name, help=help_text)
    command.add_argument("run_dir", type=Path)
    handlers = {"report": _handle_report, "benchmark": _handle_benchmark, "status": _handle_status}
    command.set_defaults(handler=handlers[name])


def _doctor_parser(subcommands: Any) -> None:
    command = subcommands.add_parser("doctor", help="Diagnose this installation")
    command.add_argument("--require-gpus", type=int, default=0)
    command.set_defaults(handler=_handle_doctor)


def _proxy_parser(subcommands: Any) -> None:
    proxy = subcommands.add_parser("proxy", help="Manage the instrumentation proxy")
    proxy_commands = proxy.add_subparsers(dest="proxy_command", required=True)
    install = proxy_commands.add_parser("install-cert", help="Install the mitmproxy certificate")
    install.set_defaults(handler=_handle_proxy_install_cert)


def _viewport(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", maxsplit=1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("viewport must be WIDTHxHEIGHT") from error
    if width < 320 or height < 240:
        raise argparse.ArgumentTypeError("viewport must be at least 320x240")
    return width, height


def _handle_init(arguments: argparse.Namespace) -> int:
    manifest = initialize_run(arguments.url, arguments.profile, arguments.run_dir, arguments.seed)
    print(json.dumps(manifest.model_dump(mode="json"), indent=2))
    return 0


def _handle_run(arguments: argparse.Namespace) -> int:
    from kwola.orchestration.experiment import ExperimentRunner

    return ExperimentRunner(arguments.run_dir).run()


def _handle_test_step(arguments: argparse.Namespace) -> int:
    from kwola.orchestration.testing import TestingRunner

    result = TestingRunner(arguments.run_dir).run(
        random_policy=arguments.random_policy,
        browser=arguments.browser,
        viewport=arguments.viewport,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.status == "completed" else 1


def _handle_train_step(arguments: argparse.Namespace) -> int:
    from kwola.orchestration.training import TrainingRunner

    result = TrainingRunner(arguments.run_dir).run(gpu=arguments.gpu)
    print(result.model_dump_json(indent=2))
    return 0 if result.status == "completed" else 1


def _handle_report(arguments: argparse.Namespace) -> int:
    from kwola.reporting.service import ReportService

    artifacts = ReportService(arguments.run_dir).generate()
    print("\n".join(str(path) for path in artifacts))
    return 0


def _handle_doctor(arguments: argparse.Namespace) -> int:
    report = run_doctor(arguments.require_gpus)
    for check in report.checks:
        print(f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.detail}")
    return 0 if report.passed else 1


def _handle_benchmark(arguments: argparse.Namespace) -> int:
    from kwola.training.benchmark import run_benchmark

    result = run_benchmark(arguments.run_dir)
    print(result.model_dump_json(indent=2))
    return 0 if result.passed else 1


def _handle_status(arguments: argparse.Namespace) -> int:
    from kwola.orchestration.status import pipeline_status

    print(json.dumps(pipeline_status(arguments.run_dir), indent=2))
    return 0


def _handle_proxy_install_cert(arguments: argparse.Namespace) -> int:
    del arguments
    from kwola.instrumentation.certificates import install_certificate

    install_certificate()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
        return int(arguments.handler(arguments))
    except (ValidationError, OSError, RuntimeError, ValueError) as error:
        print(f"kwola: {error}", file=sys.stderr)
        return 2
