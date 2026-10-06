"""Small building blocks shared by the library's torch models.

Kept deliberately tiny: anything here is imported by modules that load the frozen V3
checkpoints, so it must stay free of side effects and free of anything version-dependent.
"""

from __future__ import annotations

from torch import nn

__all__ = ["activation_class"]

_ACTIVATIONS = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def activation_class(name):
    """Map a config activation name to its ``nn.Module`` class.

    Model configs are stored inside the checkpoints, so the accepted names are part of the
    on-disk format: adding one is fine, renaming or removing one silently breaks loading a
    saved bundle.
    """
    try:
        return _ACTIVATIONS[name.lower()]
    except KeyError:
        raise ValueError(f"Unsupported activation {name!r}") from None
