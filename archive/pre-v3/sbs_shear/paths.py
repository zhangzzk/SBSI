"""Canonical locations of the data this project reads.

Before 2026-08-04 these were re-declared in ~25 scripts: `CAT` had 12 definitions in two
spellings (7 with a trailing slash, 4 without, plus one pointing at a different tree
entirely), `CROWD` 7, and `BASE`/`CBASE` 16 across four spellings -- including one `BASE`
that actually held the CONSTANT tree. Copies that differ by a trailing slash are not
interchangeable, so "just import the other one" was never safe. Hence one definition each.

Trailing slashes
----------------
`CATALOGUES` and `CONST_SIM_BASE` keep their trailing slash because ~50 call sites build
paths by CONCATENATION (`CAT + "det_meas_....feather"`). Do not remove it, and do not write
``f"{CATALOGUES}/name"`` -- that yields a doubled slash. Concatenate, or use `catalogue()`.

Not an abstraction layer
------------------------
These are literal current locations, not a config system. Changing one changes which
catalogue every downstream number was computed from, so treat an edit here as a change to
the analysis, not to the code.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "CATALOGUES", "catalogue",
    "CROWD_LOOKUP", "NN_LOOKUP",
    "SIM_BASE", "CONST_SIM_BASE",
    "ABLATION_DIR", "CACHES",
]

# Finished blendemu detection/measurement catalogues (the half-shear legs live here).
CATALOGUES = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"

# Per-(case, input_index) neighbour-flux lookup: nbr_flux_near / _far / _max, cases 0-199.
#
# NOTE: this is an absolute path into the MAIN checkout. `results/` is gitignored, so a git
# WORKTREE has its own (usually empty) `results/`, and scripts run from a worktree read this
# table from the main tree while reading their other inputs locally. That inconsistency is
# recorded in WORKLOG 2026-08-04d and is deliberately NOT silently "fixed" here: repointing
# it changes which file the numbers came from.
CROWD_LOOKUP = "/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather"

# Per-(case, input_index) nearest-BRIGHTER-neighbour distance, cases 0-39.
NN_LOOKUP = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/nn_dist_c0-39.feather"

# Raw simulation trees. These two are DIFFERENT populations -- half-shear vs constant-shear
# (constgold); see CONVENTIONS.md. Do not substitute one for the other.
SIM_BASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
CONST_SIM_BASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"

# Same location, no trailing slash: the constant-tree callers are split between concatenation
# (`CBASE + "name"`) and f-strings (`f"{CBASE}/name"`), which need opposite spellings. DERIVED
# rather than written out again, so there is still exactly one literal to change.
CONST_SIM_DIR = CONST_SIM_BASE.rstrip("/")

# Model/scoring caches.
CACHES = Path("/project/ls-gruen/users/zekang.zhang/sbsi_caches")
ABLATION_DIR = CACHES / "ablation"


def catalogue(name):
    """Full path to a catalogue file, slash-safe.

    Equivalent to ``CATALOGUES + name``; prefer this over an f-string, which would double the
    separator.
    """
    return CATALOGUES + name
