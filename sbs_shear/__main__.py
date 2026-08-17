"""Allow ``python -m sbs_shear`` to invoke the installed ``sbsi`` CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
