"""Allow ``python -m sbsi`` to invoke the installed ``sbsi`` CLI."""

from sbsi.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
