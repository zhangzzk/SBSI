"""Import BlendEMU from a plain source checkout.

BlendEMU is distributed the way its own README describes: clone the repository and put
it on ``PYTHONPATH``.  It carries no packaging metadata, so ``pip install -e
/path/to/blendemu`` fails with "does not appear to be a Python project".  SBSI therefore
resolves the checkout itself instead of requiring an installed package.

``import_blendemu()`` returns the imported ``blendemu`` module, trying in order:

1. a ``blendemu`` already imported in this process, which cannot be swapped,
2. ``$BLENDEMU_ROOT``, the checkout root -- the same variable the example Slurm
   wrapper uses; an explicit setting wins over whatever the import path happens to
   offer, so a stale copy elsewhere in the environment cannot shadow it,
3. an importable ``blendemu`` package (installed, or on ``PYTHONPATH``),
4. a ``blendemu`` checkout sitting next to the SBSI checkout.

A namespace package -- a bare ``blendemu/`` directory with no ``__init__.py``, which is
what the import system finds when the process starts in the *parent* of a checkout --
does not count as a hit, and resolution continues past it.  Only BlendEMU's emulator
prediction needs any of this; the flow checkpoints need SBSI and PyTorch alone.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

from .paths import REPOSITORY_ROOT

ROOT_ENV_VAR = "BLENDEMU_ROOT"
SIBLING_ROOT = REPOSITORY_ROOT.parent / "blendemu"


def _is_importable_package(module: Optional[ModuleType]) -> bool:
    """True for a real package, False for ``None`` and for namespace packages."""

    return module is not None and getattr(module, "__file__", None) is not None


def _configured_root() -> Optional[Path]:
    configured = os.environ.get(ROOT_ENV_VAR)
    return Path(configured).expanduser() if configured else None


def _as_checkout(candidate: Optional[Path]) -> Optional[Path]:
    """Return the directory to add to ``sys.path`` for ``candidate``, if it is a checkout.

    A candidate may name either the repository root (which contains ``blendemu/``) or
    the inner package directory; both are accepted, since users guess either way.
    """

    if candidate is None:
        return None
    if (candidate / "blendemu" / "__init__.py").is_file():
        return candidate.resolve()
    if candidate.name == "blendemu" and (candidate / "__init__.py").is_file():
        return candidate.parent.resolve()
    return None


def find_checkout() -> Optional[Path]:
    """Return the checkout named by ``$BLENDEMU_ROOT``, else the sibling one, else None."""

    return _as_checkout(_configured_root()) or _as_checkout(SIBLING_ROOT)


def _forget_namespace_package() -> None:
    for name in [
        name for name in sys.modules if name == "blendemu" or name.startswith("blendemu.")
    ]:
        del sys.modules[name]


def _try_import() -> Optional[ModuleType]:
    """Import ``blendemu``, returning ``None`` when the package itself is absent.

    A ``ModuleNotFoundError`` for one of BlendEMU's own dependencies propagates: the
    checkout was found, and reporting it as a missing checkout would be misleading.
    """

    try:
        module = importlib.import_module("blendemu")
    except ModuleNotFoundError as error:
        if error.name not in {None, "blendemu"}:
            raise
        return None
    return module if _is_importable_package(module) else None


def _not_reachable_message() -> str:
    return (
        "BlendEMU is not importable in this Python environment. It is used as a source "
        "checkout on the import path and ships no pyproject.toml, so `pip install -e "
        "/path/to/blendemu` does not work. Point SBSI at your checkout:\n"
        "    export BLENDEMU_ROOT=/path/to/blendemu\n"
        f"or place the checkout at {SIBLING_ROOT}, or add it to PYTHONPATH. In a running "
        'notebook, set os.environ["BLENDEMU_ROOT"] and call again -- no kernel restart '
        "is needed."
    )


def _import_from(checkout: Path) -> Optional[ModuleType]:
    """Import ``blendemu`` from ``checkout``, in front of anything else on the path."""

    _forget_namespace_package()
    entry = str(checkout)
    if sys.path[:1] != [entry]:
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    importlib.invalidate_caches()
    return _try_import()


def import_blendemu() -> ModuleType:
    """Return the imported ``blendemu`` module, resolving a checkout if necessary.

    Raises ``ImportError`` when no checkout is reachable.  A resolved checkout is
    prepended to ``sys.path``, so BlendEMU's submodules import normally afterwards.
    """

    already_imported = sys.modules.get("blendemu")
    if _is_importable_package(already_imported):
        return already_imported

    configured_root = _configured_root()
    if configured_root is not None:
        checkout = _as_checkout(configured_root)
        if checkout is None:
            raise ImportError(
                f"{ROOT_ENV_VAR}={configured_root} is not a BlendEMU checkout: "
                f"{configured_root / 'blendemu' / '__init__.py'} does not exist. Set it to the "
                "root of a `git clone https://github.com/zhangzzk/blendemu.git`."
            )
        module = _import_from(checkout)
        if module is None:
            raise ImportError(
                f"{ROOT_ENV_VAR}={checkout} contains blendemu/__init__.py but importing it "
                "failed; check BlendEMU's own dependencies in this environment."
            )
        return module

    if already_imported is None:
        module = _try_import()
        if module is not None:
            return module

    sibling = _as_checkout(SIBLING_ROOT)
    if sibling is None:
        raise ImportError(_not_reachable_message())
    module = _import_from(sibling)
    if module is None:
        raise ImportError(f"{sibling} contains blendemu/__init__.py but importing it failed.")
    return module


__all__ = ["ROOT_ENV_VAR", "find_checkout", "import_blendemu"]
