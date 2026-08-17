"""Intrinsic full-scene neighbour-flux summaries for constant-shear fields.

For every input galaxy, sum the flux of every other rendered input source in
three fixed angular shells: 0--1, 1--3, and 3--10 arcsec.  Fluxes are stored
relative to the primary flux as ``log10(1 + F_neighbour/F_primary)``.  The
lookup also records the flux in the third-brightest and later neighbours
inside 10 arcsec.  That last quantity directly tests whether a pair-only
BlendEMU prediction misses context supplied by additional galaxies.

No measured quantity, detection flag, emulator prediction, or constgold
response enters this lookup.  Output is keyed by (case, input_index).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

from sbs_shear.paths import CONST_SIM_DIR


TILE = "tile180.0_-0.5"
ZERO_POINT = 30.0
SHELL_EDGES_ARCSEC = np.asarray([0.0, 1.0, 3.0, 10.0])


def input_feather(case: int, sign: str, base: str) -> str:
    return f"{base}/case{case}_{sign}/real0/catalogues/input/gals_info_{TILE}.feather"


def scene_flux_summaries(pos: np.ndarray, flux: np.ndarray) -> dict[str, np.ndarray]:
    """Return symmetric all-neighbour flux/count summaries for one field."""
    n = len(pos)
    pairs = cKDTree(pos).query_pairs(
        r=SHELL_EDGES_ARCSEC[-1] / 3600.0, output_type="ndarray",
    )
    shell_flux = np.zeros((3, n), dtype=np.float64)
    shell_count = np.zeros((3, n), dtype=np.int32)
    top1 = np.zeros(n, dtype=np.float64)
    top2 = np.zeros(n, dtype=np.float64)
    if not len(pairs):
        return {
            "shell_flux": shell_flux,
            "shell_count": shell_count,
            "top1": top1,
            "top2": top2,
            "thirdplus": np.zeros(n, dtype=np.float64),
        }

    i, j = pairs[:, 0], pairs[:, 1]
    distance = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) * 3600.0
    inside = distance < SHELL_EDGES_ARCSEC[-1]
    i, j, distance = i[inside], j[inside], distance[inside]
    if not len(i):
        return {
            "shell_flux": shell_flux,
            "shell_count": shell_count,
            "top1": top1,
            "top2": top2,
            "thirdplus": np.zeros(n, dtype=np.float64),
        }
    shell = np.searchsorted(SHELL_EDGES_ARCSEC, distance, side="right") - 1
    if np.any((shell < 0) | (shell >= 3)):
        raise RuntimeError("KD-tree returned a pair outside the declared shells")

    # Each unordered pair contributes the secondary flux to both primaries.
    for s in range(3):
        use = shell == s
        np.add.at(shell_flux[s], i[use], flux[j[use]])
        np.add.at(shell_flux[s], j[use], flux[i[use]])
        np.add.at(shell_count[s], i[use], 1)
        np.add.at(shell_count[s], j[use], 1)

    primary = np.concatenate([i, j])
    secondary_flux = np.concatenate([flux[j], flux[i]])
    # Sort by primary, then decreasing secondary flux.  The first two entries
    # per primary are the two brightest neighbours, with ties retained.
    order = np.lexsort((-secondary_flux, primary))
    ps = primary[order]
    fs = secondary_flux[order]
    first = np.r_[True, ps[1:] != ps[:-1]]
    first_pos = np.flatnonzero(first)
    top1[ps[first_pos]] = fs[first_pos]
    candidate = first_pos + 1
    valid = candidate < len(ps)
    candidate = candidate[valid]
    owner = ps[first_pos[valid]]
    valid_owner = ps[candidate] == owner
    top2[owner[valid_owner]] = fs[candidate[valid_owner]]

    total = shell_flux.sum(axis=0)
    thirdplus = np.maximum(total - top1 - top2, 0.0)
    return {
        "shell_flux": shell_flux,
        "shell_count": shell_count,
        "top1": top1,
        "top2": top2,
        "thirdplus": thirdplus,
    }


def _log_ratio(neighbour_flux: np.ndarray, primary_flux: np.ndarray) -> np.ndarray:
    return np.log10(1.0 + neighbour_flux / primary_flux)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sign", default="0.02")
    ap.add_argument("--base", default=CONST_SIM_DIR)
    args = ap.parse_args()

    parts = []
    for case in args.cases:
        path = input_feather(case, args.sign, args.base)
        if not os.path.exists(path):
            print(f"case{case}: MISSING {path}", flush=True)
            continue
        table = pf.read_table(
            path, columns=["index_input", "RA_input", "DEC_input", "r_input"],
        ).to_pandas()
        pos = table[["RA_input", "DEC_input"]].to_numpy(float)
        primary_flux = 10.0 ** (-0.4 * (table["r_input"].to_numpy(float) - ZERO_POINT))
        summary = scene_flux_summaries(pos, primary_flux)
        sf = summary["shell_flux"]
        sc = summary["shell_count"]
        part = pd.DataFrame({
            "case": case,
            "input_index": table["index_input"].to_numpy(np.int64),
            "logflux_near_0_1": _log_ratio(sf[0], primary_flux),
            "logflux_mid_1_3": _log_ratio(sf[1], primary_flux),
            "logflux_far_3_10": _log_ratio(sf[2], primary_flux),
            "logflux_all_0_10": _log_ratio(sf.sum(axis=0), primary_flux),
            "logflux_thirdplus_0_10": _log_ratio(summary["thirdplus"], primary_flux),
            "n_near_0_1": sc[0],
            "n_mid_1_3": sc[1],
            "n_far_3_10": sc[2],
            "n_all_0_10": sc.sum(axis=0),
        })
        parts.append(part)
        print(
            f"case{case}: {len(part):,} galaxies; "
            f"<N10>={part['n_all_0_10'].mean():.2f}; "
            f"third+ fraction={(summary['thirdplus'] > 0).mean():.3%}",
            flush=True,
        )

    if not parts:
        raise SystemExit("no input cases found")
    output = pd.concat(parts, ignore_index=True)
    output.to_feather(args.output)
    print(
        f"wrote {args.output}: {len(output):,} rows over "
        f"{output['case'].nunique()} cases",
        flush=True,
    )


if __name__ == "__main__":
    main()
