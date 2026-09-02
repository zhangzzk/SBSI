"""Simulation-based shear inference.

The package root intentionally exposes only the small prediction API used by
the bundled example. Imports are resolved lazily so importing :mod:`sbsi`
does not load PyTorch, model checkpoints, or the catalogue-inference stack.
Specialized code should import its implementation module directly.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "EmulatorPairingConfig": (".forward_catalogue", "EmulatorPairingConfig"),
    "ModelPaths": (".models", "ModelPaths"),
    "ResponsePredictor": (".response", "ResponsePredictor"),
    "example_path": (".paths", "example_path"),
    "get_model": (".models", "get_model"),
    "load_catalogue": (".catalogue", "load_catalogue"),
    "load_emulator": (".models", "load_emulator"),
    "prepare_forward_catalogue": (
        ".forward_catalogue",
        "prepare_forward_catalogue",
    ),
    "predict_blend_response": (".response", "predict_blend_response"),
    "sample_measurement": (".response", "sample_measurement"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a supported root-level export on first access."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
