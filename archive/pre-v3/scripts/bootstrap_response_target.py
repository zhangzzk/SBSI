"""How well is the response target's population-weighted mean actually DETERMINED?

The V2.1 flow trains against a target whose population-weighted self-response is 0.8509, while
constgold demands `R_sim - R_blend` = 0.8724 -- a 2.53% gap that WORKLOG 2026-08-04p pins the whole
+0.790% `m` on. That write-up quoted no error on 0.8509. If the target's own sampling error is of
order the gap, there is nothing to explain; if it is far below, the gap is a real systematic and the
snc reference / population candidates stay live. This script measures it.

WHY NOT A PLAIN sem. The rows are nested, and each level is a candidate correlation scale:

    row (one PAIR)  <  (case, input_index) = one noise realisation  <  input_index = one intrinsic
    galaxy seen across every case  <  case = one render (shared scene and noise field)

Treating rows as independent is the standard way to under-report an error here. The all-pairs
weighting `w = 1/n_pairs` already collapses the pair level BY CONSTRUCTION -- each (case,
input_index) sums to weight exactly 1 -- but nothing collapses the other two. `proj` differences the
SAME galaxy against its own g=0 shape, so the intrinsic ellipticity should largely cancel and the
input_index level should be quiet; that is a PREDICTION this script tests rather than assumes.

WHAT IT REPORTS. A cluster-robust (sandwich) sem at each level, which for a weighted mean
`R = sum_i w_i r_i / sum_i w_i` is

    Var(R) = [G/(G-1)] * sum_c u_c^2 / W^2 ,   u_c = sum_{i in c} w_i (r_i - R)

plus a case-level BOOTSTRAP, because the case level has only ~100 clusters and the sandwich formula
is asymptotic in cluster count. The two agreeing is the check that 100 is enough.

FIREWALL AND FIDELITY. Read-only: it fits nothing, tunes nothing, and writes no target. It
replicates the estimator of `compute_response_target_blend.py` rather than importing it -- that
builder produced the certified targets and is not worth refactoring for a diagnostic -- so the
replication is PROVEN rather than trusted: the script recomputes `global_R` and REFUSES unless it
reproduces the value stored in the npz to 1e-9. A drifted estimator cannot silently report an error
bar for a number it did not actually compute.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import (  # noqa: E402
    add_measurement_target_features, raw_columns_for_measurement_targets)
from sbs_shear import domain as sbs_domain  # noqa: E402
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402


def cluster_sem(r, w, labels, mean):
    """Cluster-robust sem of the weighted mean `sum(w*r)/sum(w)`, clustering on `labels`.

    Reduces to the ordinary weighted sem when every cluster holds one row, so the row-level call is
    the same code path as the rest -- no separate formula to keep in step.
    """
    codes = pd.factorize(labels, sort=False)[0]
    g = int(codes.max()) + 1
    u = np.bincount(codes, weights=w * (r - mean), minlength=g)
    var = (g / (g - 1.0)) * float((u ** 2).sum()) / float(w.sum()) ** 2
    return np.sqrt(max(var, 0.0)), g


def case_bootstrap(r, w, cases, n_boot, seed):
    """Resample CASES with replacement. The sandwich above is asymptotic in cluster count and there
    are only ~100 cases, so this is the check on whether that asymptotics has arrived."""
    codes = pd.factorize(cases, sort=False)[0]
    g = int(codes.max()) + 1
    num = np.bincount(codes, weights=w * r, minlength=g)   # per-case numerator and denominator:
    den = np.bincount(codes, weights=w, minlength=g)       # a resample is then a dot product
    rng = np.random.default_rng(seed)
    mult = rng.multinomial(g, np.full(g, 1.0 / g), size=n_boot).astype(float)
    means = (mult @ num) / (mult @ den)
    return float(means.std(ddof=1)), g


def load(args):
    """The loading and `proj` construction of compute_response_target_blend.py, replicated."""
    need = set(raw_columns_for_measurement_targets(args.target_cols))
    need |= {"gamma1_input_p", "gamma2_input_p", "detected"}
    need |= {"r_input_p", "Re_input_p", "distance", "neighbored", "input_index", "case"}
    if args.crowd_col:
        need.add(args.crowd_col)

    snc = pd.read_feather(args.snc_lookup, columns=["case", "input_index", *args.snc_cols])
    key = snc["case"].to_numpy(np.int64) * 1_000_003 + snc["input_index"].to_numpy(np.int64)
    order = np.argsort(key)
    snc_key = key[order]
    snc_e1 = snc[args.snc_cols[0]].to_numpy(float)[order]
    snc_e2 = snc[args.snc_cols[1]].to_numpy(float)[order]
    print(f"SNC lookup: {len(snc):,} rows")

    cuts = [list(c) for c in DEFAULT_SELECTION_CUTS]
    parts = []
    with ipc.open_file(args.catalogue) as r:
        avail = set(r.schema.names)
        cols = sorted(c for c in need if c in avail)
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            b = b[b["case"] <= args.max_case]
            if len(b) == 0:
                continue
            b = source_select_selection(b, cuts=cuts)
            b = sbs_domain.select_frame(b)
            if len(b) == 0:
                continue
            b = b[b["detected"].astype(bool)].reset_index(drop=True)
            if len(b):
                parts.append(b)
    df = pd.concat(parts, ignore_index=True)

    g1 = df["gamma1_input_p"].to_numpy(float)
    g2 = df["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    keep = gmag > 1e-6
    df = df[keep].reset_index(drop=True)
    g1, g2, gmag = g1[keep], g2[keep], gmag[keep]
    gh1, gh2 = g1 / gmag, g2 / gmag

    ii = df["input_index"].to_numpy(np.int64)
    ck = df["case"].to_numpy(np.int64) * 1_000_003 + ii
    _, inv, npairs = np.unique(ck, return_inverse=True, return_counts=True)
    w = (1.0 / npairs[inv]).astype(float)

    meas = add_measurement_target_features(df.copy())
    e1 = meas[args.target_cols[0]].to_numpy(float)
    e2 = meas[args.target_cols[1]].to_numpy(float)
    pos = np.clip(np.searchsorted(snc_key, ck), 0, len(snc_key) - 1)
    match = snc_key[pos] == ck
    print(f"SNC match after selection: {match.mean():.2%} ({int(match.sum()):,}/{len(match):,})")
    e1 = e1 - np.where(match, snc_e1[pos], np.nan)
    e2 = e2 - np.where(match, snc_e2[pos], np.nan)
    proj = e1 * gh1 + e2 * gh2

    # The builder's finite mask, which also drops the snc non-matches (they are NaN by construction
    # above). Dropping them is the builder's behaviour, not a choice made here.
    flux = df["r_input_p"].to_numpy(float)
    size = df["Re_input_p"].to_numpy(float)
    fin = np.isfinite(flux) & np.isfinite(size) & np.isfinite(proj)
    if args.crowd_col:
        fin &= np.isfinite(df[args.crowd_col].to_numpy(float))
    return proj[fin] / args.nominal_g, w[fin], df["case"].to_numpy(np.int64)[fin], ck[fin], ii[fin]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--npz", required=True, help="target npz whose global_R must be reproduced")
    ap.add_argument("--snc-lookup", required=True)
    ap.add_argument("--snc-cols", nargs=2, default=["ngmix0_g1", "ngmix0_g2"])
    ap.add_argument("--target-cols", nargs=2,
                    default=["measured_ngmix_g1", "measured_ngmix_g2"])
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--crowd-col", default="r_blend")
    ap.add_argument("--max-case", type=int, default=99)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demanded", type=float, default=0.8724,
                    help="what constgold needs from the flow (R_sim - R_blend), for the gap line")
    args = ap.parse_args()

    r, w, case, target, gal = load(args)
    W = float(w.sum())
    R = float((w * r).sum() / W)

    stored = float(np.load(args.npz, allow_pickle=True)["global_R"])
    print(f"\nglobal_R recomputed {R:.10f}  vs stored {stored:.10f}  (delta {R - stored:+.2e})")
    if abs(R - stored) > 1e-9:
        raise SystemExit(
            "REFUSING: the replicated estimator does not reproduce the stored global_R. An error bar "
            "for a number this script did not actually compute would be worthless.")

    print(f"\nrows {len(r):,}   effective N (sum of weights) {W:,.0f}")
    print(f"weighted mean response R = {R:.4f}")
    sd = np.sqrt(float((w * (r - R) ** 2).sum() / W))
    print(f"per-object response scatter (weighted sd) = {sd:.3f}  -> sd/R = {sd / abs(R):.1f}")

    print(f"\n{'='*78}\ncluster-robust sem, by the level treated as independent\n{'='*78}")
    print(f"  {'clustering level':>34}{'clusters':>12}{'sem':>11}{'% of R':>10}")
    for name, lab in (("row (pairs independent)", np.arange(len(r))),
                      ("(case, input_index) = noise realisation", target),
                      ("input_index = intrinsic galaxy", gal),
                      ("case = render", case)):
        sem, g = cluster_sem(r, w, lab, R)
        print(f"  {name:>34}{g:>12,}{sem:>11.5f}{100 * sem / abs(R):>9.3f}%")

    bs, g = case_bootstrap(r, w, case, args.n_boot, args.seed)
    print(f"\ncase-level bootstrap ({args.n_boot:,} resamples of {g} cases): sem = {bs:.5f} "
          f"({100 * bs / abs(R):.3f}% of R)")
    print("  agreement with the case-level sandwich above is the check that ~100 clusters is enough.")

    sem_case, _ = cluster_sem(r, w, case, R)
    worst = max(sem_case, bs)
    gap = args.demanded - R
    print(f"\n{'='*78}\nIS THE 2.53% GAP EXPLAINED BY SAMPLING NOISE?\n{'='*78}")
    print(f"  target  R          = {R:.4f} +- {worst:.4f}  (most conservative level)")
    print(f"  constgold demands  = {args.demanded:.4f}   (R_sim - R_blend; its OWN error not included)")
    print(f"  gap                = {gap:+.4f}  = {100 * gap / R:+.2f}%  = {gap / worst:.1f} sigma "
          f"on the target's error alone")
    print("\nBOOTSTRAP_RESP_TARGET_DONE", flush=True)


if __name__ == "__main__":
    main()
