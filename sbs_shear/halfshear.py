"""SNC forward SELF-response measured on the half-shear catalogue legs.

This is the sim-side counterpart to ``response.py`` (which measures the FLOW's response). It
reconstructs, from the raw legs, exactly the quantity ``compute_response_target_blend.py`` bins into
the flow's response target, so a target level can be re-derived, re-binned, or re-weighted
independently of the npz it was baked into.

WHY THE PROJECTION GIVES THE **SELF** RESPONSE. In the half-shear legs every galaxy carries a shear
of constant magnitude and INDEPENDENT random direction (``gamma1/2_input_p`` per object). Projecting
a primary's measured shape change onto its OWN shear direction therefore averages the neighbours'
contributions to zero -- their directions are uncorrelated with the primary's. That is the standard
``ghat_p`` projection, and it is why this number is comparable to a flow ``R_flow`` (self-only) and
NOT to a constgold ``R_sim`` (total). ``verify_independent_shear_directions`` checks the premise
rather than assuming it.

CONVENTION. Forward only (``0 -> +g``), matching the legs' construction; see CONVENTIONS.md 6c.
Never difference these against an antithetic constgold response without saying so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

CATALOGUE_DIR = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
LEG_PATHS = {
    0.02: CATALOGUE_DIR + "det_meas_crowd_g0.02_test_full.feather",
    0.05: CATALOGUE_DIR + "det_meas_crowd_g0.05_val_full.feather",
}
G0_LOOKUP = "/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather"

# The (case, input_index) packing compute_response_target_blend.py uses; kept identical so keys
# built here and there are interchangeable.
KEYMUL = 1_000_003

_BASE_COLS = ["case", "input_index", "gamma1_input_p", "gamma2_input_p",
              "measured_ngmix_g1", "measured_ngmix_g2", "r_input_p", "Re_input_p", "detected"]


def pack_key(case, input_index):
    return np.asarray(case, np.int64) * KEYMUL + np.asarray(input_index, np.int64)


def load_g0_lookup(path=G0_LOOKUP, cols=("ngmix0_g1", "ngmix0_g2")):
    """Shape-noise-cancellation reference: each object's measured shape in the unsheared leg."""
    t = pf.read_table(path, columns=["case", "input_index", *cols]).to_pandas()
    key = pack_key(t["case"], t["input_index"])
    order = np.argsort(key)
    return {"key": key[order],
            "e1": t[cols[0]].to_numpy(float)[order],
            "e2": t[cols[1]].to_numpy(float)[order]}


def verify_independent_shear_directions(df, min_cases=5):
    """Confirm shear DIRECTION varies within a case, which is what makes the projection self-only.

    Returns the mean within-case circular concentration of 2*theta: ~0 means directions are
    uniformly spread (neighbours cancel, projection is self-only); ~1 would mean a single shared
    direction per case, under which the projection would pick up the neighbours' shear too and the
    result would NOT be a self response.
    """
    theta2 = 2 * np.arctan2(df["gamma2_input_p"].to_numpy(float),
                            df["gamma1_input_p"].to_numpy(float))
    case = df["case"].to_numpy(np.int64)
    uc, inv = np.unique(case, return_inverse=True)
    if len(uc) < min_cases:
        return float("nan")
    n = np.bincount(inv)
    c = np.bincount(inv, weights=np.cos(theta2)) / n
    s = np.bincount(inv, weights=np.sin(theta2)) / n
    return float(np.hypot(c, s).mean())


def load_leg(path, mag_max, re_min, max_case, extra_cols=()):
    """Detected rows inside a true-property box, streamed batch by batch."""
    want = list(dict.fromkeys([*_BASE_COLS, *extra_cols]))
    parts = []
    with ipc.open_file(pa.memory_map(path)) as reader:
        use = [c for c in want if c in set(reader.schema.names)]
        for bi in range(reader.num_record_batches):
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(use).to_pandas()
            b = b[b["case"].to_numpy() <= max_case]
            if "detected" in b.columns:
                b = b[b["detected"].astype(bool)]
            b = b[(b["r_input_p"].to_numpy() < mag_max) & (b["Re_input_p"].to_numpy() > re_min)]
            if len(b):
                parts.append(b)
    return pd.concat(parts, ignore_index=True)


def self_response_per_object(df, g, g0, keep_cols=()):
    """One SNC forward self-response per (case, input_index).

    ``proj = (e - e_0) . ghat / g``, then all-pairs duplicates are collapsed to a single value per
    object -- the same thing the target's ``1/n_pairs`` weighting does, expressed as an average.

    Returns a DataFrame with ``key``, ``case``, ``R`` and any requested ``keep_cols`` (taken from
    the first row of each object, so only per-object quantities like true mag/size are meaningful).
    """
    g1 = df["gamma1_input_p"].to_numpy(float)
    g2 = df["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    df = df[gmag > 1e-6].reset_index(drop=True)
    g1, g2, gmag = g1[gmag > 1e-6], g2[gmag > 1e-6], gmag[gmag > 1e-6]

    key = pack_key(df["case"], df["input_index"])
    pos = np.clip(np.searchsorted(g0["key"], key), 0, len(g0["key"]) - 1)
    match = g0["key"][pos] == key
    e1 = df["measured_ngmix_g1"].to_numpy(float) - np.where(match, g0["e1"][pos], np.nan)
    e2 = df["measured_ngmix_g2"].to_numpy(float) - np.where(match, g0["e2"][pos], np.nan)
    proj = (e1 * (g1 / gmag) + e2 * (g2 / gmag)) / g

    fin = np.isfinite(proj)
    stats = {"n_rows": len(df), "snc_matched": float(match.mean()), "finite": float(fin.mean()),
             "gmag_median": float(np.median(gmag))}

    key, proj, df = key[fin], proj[fin], df[fin].reset_index(drop=True)
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    cnt = np.bincount(inv)
    out = {"key": uk,
           "case": (uk // KEYMUL).astype(np.int64),
           "R": np.bincount(inv, weights=proj) / cnt,
           "n_pairs": cnt}
    for c in keep_cols:
        out[c] = df[c].to_numpy()[first]      # per-object columns: any row of the group will do
    return pd.DataFrame(out), stats


def case_blocked(x, case):
    """Mean and CASE-blocked sem.

    Galaxies sharing a rendered field share a noise realisation, so rows are not independent and a
    naive ``std/sqrt(N)`` understates the error. Blocking on ``case`` restores it.
    """
    x = np.asarray(x, float)
    uc, inv = np.unique(np.asarray(case), return_inverse=True)
    mu = np.bincount(inv, weights=x) / np.bincount(inv)
    if len(uc) < 3:
        return float(x.mean()), float("nan"), len(uc)
    return float(x.mean()), float(mu.std(ddof=1) / np.sqrt(len(uc))), len(uc)
