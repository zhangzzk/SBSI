"""Locations of resources shipped in an SBSI source checkout.

SBSI's public release includes example data and frozen model artifacts outside the
Python package directory.  Resolve those files from the imported package, never from
the process working directory.  This makes notebooks, command-line tools, and batch
jobs behave identically when SBSI is installed editable from a checkout.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"
RELEASE_MODELS_ROOT = REPOSITORY_ROOT / "models"


def repository_path(*parts: str) -> Path:
    """Return an existing path relative to the imported SBSI checkout.

    A regular wheel does not contain the repository's examples and large frozen
    artifacts.  The error therefore points users to the supported editable install
    instead of failing later with a path relative to an arbitrary working directory.
    """

    path = REPOSITORY_ROOT.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(
            f"SBSI checkout resource not found: {path}\n"
            "Install SBSI editable from a complete source checkout "
            "(`python -m pip install -e /path/to/SBSI`)."
        )
    return path


def example_path(*parts: str) -> Path:
    """Return an existing file or directory below the checkout's ``examples/``."""

    return repository_path("examples", *parts)


__all__ = [
    "EXAMPLES_ROOT",
    "PACKAGE_ROOT",
    "RELEASE_MODELS_ROOT",
    "REPOSITORY_ROOT",
    "example_path",
    "repository_path",
]
