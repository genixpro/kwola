"""Console-script shim for the supported CLI."""

from .cli import main as _main


def main() -> None:
    raise SystemExit(_main())
