"""Filesystem roots for everything SBSI reads and writes.

SBSI does not simulate or measure anything itself.  It consumes catalogues that were
produced by a *separate* blendemu run and points at them by path.  Every such path is
resolved here, once, from an environment variable with a fallback default, so that
running the pipeline on another machine (or against another catalogue set) is a matter
of exporting a few variables rather than editing scripts.

Environment variables
---------------------
``SBSI_ROOT``            this repository (default: the parent of this package)
``SBSI_DATA_ROOT``       project-filesystem root for large data (default: ``$DATA_DIR``)
``SBSI_CATALOGUE_DIR``   finished blendemu detection/measurement catalogues
``SBSI_CACHE_DIR``       SBSI-derived caches (lookups, harvests, npz dumps)
``SBSI_DUMP_DIR``        per-object dumps used by the figure scripts
``SBSI_SIM_DIR``         half-shear simulation set root
``SBSI_CONST_SIM_DIR``   constant-shear ("gold") simulation set root
``BLENDEMU_ROOT``        the blendemu checkout (only needed for emulator use)
``BLENDEMU_MODELS``      trained emulator weights (default: ``$BLENDEMU_ROOT/models``)

Nothing here touches the filesystem at import time; call :func:`require` when a path
must exist and you want a readable error instead of a downstream ``FileNotFoundError``.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "SBSI_ROOT", "DATA_ROOT", "CATALOGUE_DIR", "CACHE_DIR", "DUMP_DIR",
    "SIM_DIR", "CONST_SIM_DIR", "RESULTS_DIR", "FIGURE_DIR", "MODEL_DIR",
    "BLENDEMU_ROOT", "BLENDEMU_MODELS",
    "catalogue", "cache", "result", "figure", "require", "describe",
]


def _env(name: str, default: Path | str) -> Path:
    """Read ``name`` from the environment, falling back to ``default``."""
    value = os.environ.get(name)
    return Path(value).expanduser() if value else Path(default).expanduser()


# --- repository ---------------------------------------------------------------------
SBSI_ROOT = _env("SBSI_ROOT", Path(__file__).resolve().parent.parent)

RESULTS_DIR = SBSI_ROOT / "results"      # small text/npz products, tracked-adjacent
FIGURE_DIR = SBSI_ROOT / "figures"
MODEL_DIR = SBSI_ROOT / "models"         # trained SBSI models (flows, selection MLPs)

# --- external data (project filesystem, not $HOME) ------------------------------------
DATA_ROOT = _env("SBSI_DATA_ROOT", os.environ.get("DATA_DIR") or "/project/ls-gruen/users/zekang.zhang")

CATALOGUE_DIR = _env("SBSI_CATALOGUE_DIR", DATA_ROOT / "sbsi_catalogues")
CACHE_DIR = _env("SBSI_CACHE_DIR", DATA_ROOT / "sbsi_caches")
DUMP_DIR = _env("SBSI_DUMP_DIR", DATA_ROOT / "sbsi_dumps")

# Simulation sets.  These are blendemu *outputs*: SBSI reads the input `gals_info`
# catalogues under them, it never writes here.
SIM_DIR = _env("SBSI_SIM_DIR", DATA_ROOT / "lsst_sims_fs2_25876")
CONST_SIM_DIR = _env("SBSI_CONST_SIM_DIR", DATA_ROOT / "lsst_sims_fs2_25876_constant")

# --- blendemu (optional; only the emulator path needs it) -----------------------------
BLENDEMU_ROOT = _env("BLENDEMU_ROOT", Path(os.environ.get("WORK_DIR") or "/home/z/Zekang.Zhang") / "blendemu")
BLENDEMU_MODELS = _env("BLENDEMU_MODELS", BLENDEMU_ROOT / "models")


# --- convenience accessors ------------------------------------------------------------
def catalogue(name: str) -> Path:
    """Path to a finished blendemu catalogue, e.g. ``det_meas_ngmix_g0.0_train.feather``."""
    return CATALOGUE_DIR / name


def cache(*parts: str) -> Path:
    """Path under the SBSI cache root, e.g. ``cache("derisk", "joint_triad.npz")``."""
    return CACHE_DIR.joinpath(*parts)


def result(*parts: str) -> Path:
    return RESULTS_DIR.joinpath(*parts)


def figure(*parts: str) -> Path:
    return FIGURE_DIR.joinpath(*parts)


def require(path: os.PathLike | str, what: str = "path") -> Path:
    """Return ``path`` if it exists, else raise with the variable to set to fix it."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{what} not found: {p}\n"
            "SBSI reads catalogues produced by a separate blendemu run. Point it at yours by\n"
            "exporting SBSI_CATALOGUE_DIR / SBSI_CACHE_DIR / SBSI_SIM_DIR (see sbs_shear/paths.py)."
        )
    return p


def describe() -> str:
    """One-line-per-root summary, handy at the top of a job log."""
    rows = [(n, globals()[n]) for n in (
        "SBSI_ROOT", "DATA_ROOT", "CATALOGUE_DIR", "CACHE_DIR", "DUMP_DIR",
        "SIM_DIR", "CONST_SIM_DIR", "BLENDEMU_ROOT", "BLENDEMU_MODELS")]
    width = max(len(n) for n, _ in rows)
    return "\n".join(f"{n:<{width}}  {p}{'' if Path(p).exists() else '   [missing]'}" for n, p in rows)


if __name__ == "__main__":
    print(describe())
