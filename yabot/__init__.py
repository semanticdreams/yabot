__all__ = ["main", "run"]


def main() -> None:
    from .app import main as _main

    _main()


def run() -> None:
    from .app import run as _run

    _run()
