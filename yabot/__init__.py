__all__ = ["main", "run", "run_cli"]


def main() -> None:
    from .app import main as _main

    _main()


def run() -> None:
    from .app import run as _run

    _run()


def run_cli() -> None:
    from .cli import run as _run

    _run()
