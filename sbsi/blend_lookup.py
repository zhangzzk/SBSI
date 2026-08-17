"""Joining a per-object ``R_blend`` lookup onto a set of scored rows.

This is the single most dangerous join in the pipeline, so it lives in one place with
every guard the repo has learned it needs.

Why each guard exists
---------------------
* **Coverage.** A lookup built on a restricted domain returns nothing outside it. Filling
  those rows with ``R_blend = 0`` credits the model with a prediction it never made and is
  the recorded silent failure that produced a spurious **+28.9% m** (job 15366950, AGENTS.md
  "Two traps" #1). Unmatched rows are therefore always reported and always dropped by the
  caller -- never zero-filled.
* **Duplicate keys.** A lookup with a repeated ``(case, input_index)`` turns a left merge
  into a row *multiplication*: the frame silently grows and every downstream mean is taken
  over a reweighted population. Nothing about the resulting number looks wrong.
* **Row count.** The belt to the duplicate-key braces -- assert directly that the merge
  returned exactly the rows it was given.

Before 2026-08-04 these guards were spread across multiple scripts at different
strengths. They now live here so every supported response call uses the same join.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf

__all__ = ["DEFAULT_KEY", "join_blend"]

DEFAULT_KEY = ("case", "input_index")


def join_blend(ref, lookup, value_col, *, min_match=None, key=DEFAULT_KEY,
               extra_cols=(), label=None, verbose=True):
    """Attach ``value_col`` from ``lookup`` to the rows of ``ref``.

    Parameters
    ----------
    ref : DataFrame carrying at least ``key``. Row ORDER is preserved, so the returned
        arrays are positionally aligned with ``ref`` and with anything else indexed by it
        (e.g. a per-seed ``R_flow`` matrix).
    lookup : path to a feather lookup, or an already-loaded DataFrame.
    value_col : the blend column to pull across (e.g. ``"R_blend"``).
    min_match : refuse below this matched fraction. ``None`` reports coverage without
        enforcing a floor -- for callers that legitimately score a restricted subset and
        treat low coverage as a diagnostic rather than an error.
    extra_cols : further columns to carry across, returned in ``joined``.
    label : name used in messages; defaults to the lookup's basename.

    Returns
    -------
    values : float64 array, ``NaN`` where unmatched.
    keep : bool mask, ``True`` where matched. **Callers must drop ``~keep``, not fill it.**
    frac : matched fraction.
    joined : the merged frame (for ``extra_cols``).

    Raises
    ------
    SystemExit
        On a missing column, a duplicate key, a row-count change, or -- when ``min_match``
        is given -- coverage below it.
    """
    key = list(key)
    extra_cols = list(extra_cols)
    if label is None:
        label = os.path.basename(lookup) if isinstance(lookup, (str, os.PathLike)) else "lookup"

    wanted = key + [value_col] + [c for c in extra_cols if c not in key and c != value_col]

    if isinstance(lookup, (str, os.PathLike)):
        available = set(pf.read_table(lookup, memory_map=True).schema.names)
        missing = [c for c in wanted if c not in available]
        if missing:
            raise SystemExit(f"REFUSING: {label} is missing {missing}")
        # Read the whole table: column projection is ~4x SLOWER than a streaming read on
        # this filesystem (measured 9.1s vs 36.0s on a cold 8 GB file), because it turns one
        # sequential scan into thousands of strided per-column reads.
        lk = pf.read_table(lookup, memory_map=True).to_pandas()[wanted]
    else:
        missing = [c for c in wanted if c not in lookup.columns]
        if missing:
            raise SystemExit(f"REFUSING: {label} is missing {missing}")
        lk = lookup[wanted]

    n_lk = len(lk)
    lk = lk.drop_duplicates(subset=key)
    if len(lk) != n_lk:
        raise SystemExit(
            f"REFUSING: {label} has duplicate {tuple(key)} keys ({n_lk:,} rows -> "
            f"{len(lk):,} unique). A left merge would MULTIPLY rows and silently reweight "
            "every downstream mean.")

    n_before = len(ref)
    joined = ref[key].merge(lk, on=key, how="left")
    if len(joined) != n_before:
        raise SystemExit(
            f"REFUSING: merging {label} changed the row count ({n_before:,} -> {len(joined):,}).")

    values = joined[value_col].to_numpy(float)
    keep = np.isfinite(values)
    frac = float(keep.mean()) if n_before else 0.0

    if verbose:
        print(f"  {label:<48} matched {100 * frac:6.2f}% of {n_before:,} rows  "
              f"<{value_col}> = {np.nanmean(values) if keep.any() else float('nan'):.4f}")

    if min_match is not None and frac < min_match:
        raise SystemExit(
            f"REFUSING: {label} covers only {100 * frac:.1f}% of the {n_before:,} rows "
            f"(floor {100 * min_match:.1f}%). Zero-filling the rest is what produced a "
            "spurious +28.9% m (job 15366950); dropping this many rows would change the "
            "population the number describes.")

    return values, keep, frac, joined
