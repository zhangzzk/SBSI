"""Build a V2.1 self-response target from random-direction antithetic legs.

The existing half-shear target is a forward difference, ``(e(+g)-e(0))/g``.
Constgold is a central difference, but its coherent fixed shear measures SELF +
neighbour response and therefore cannot be used directly as an R_flow target.
This diagnostic instead pairs random-direction half-shear renders at +/-g:

    R_self = <(e(+g) - e(-g)) dot ghat> / (2 g)

Random directions make neighbours average away, retaining the component split
``R_model = R_flow + R_blend``.  The +/- catalogues must have identical latent
objects and exactly opposite target shear vectors.  All joins and guards are
explicit; unmatched objects are dropped, never zero-filled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear import domain as sbs_domain  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    source_select_selection,
)

KEY_BITS = 40


def packed_key(case, input_index):
    """Collision-free packed (case, input_index) key with range guards."""
    case = np.asarray(case, dtype=np.int64)
    index = np.asarray(input_index, dtype=np.int64)
    if case.shape != index.shape:
        raise ValueError("case and input_index shapes differ")
    if case.size and case.min() < 0:
        raise ValueError("negative case is not supported")
    if index.size and (index.min() < 0 or index.max() >= (1 << KEY_BITS)):
        raise ValueError(f"input_index must fit in {KEY_BITS} bits")
    return (case << KEY_BITS) | index


def exact_match(reference_key, candidate_key):
    """Return reference mask and candidate positions for an exact unique-key join."""
    reference_key = np.asarray(reference_key, dtype=np.int64)
    candidate_key = np.asarray(candidate_key, dtype=np.int64)
    if candidate_key.size == 0:
        return np.zeros(reference_key.size, dtype=bool), np.zeros(reference_key.size, dtype=np.int64)
    order = np.argsort(candidate_key)
    sorted_key = candidate_key[order]
    if sorted_key.size > 1 and np.any(sorted_key[1:] == sorted_key[:-1]):
        raise RuntimeError("candidate catalogue has duplicate (case, input_index) keys")
    pos = np.searchsorted(sorted_key, reference_key)
    clip = np.clip(pos, 0, sorted_key.size - 1)
    hit = (pos < sorted_key.size) & (sorted_key[clip] == reference_key)
    return hit, order[clip]


def central_response(e1_plus, e2_plus, e1_minus, e2_minus,
                     g1_plus, g2_plus, g1_minus, g2_minus,
                     shear_atol=2.0e-7):
    """Per-object antithetic response, with an opposite-shear identity guard."""
    arrays = [np.asarray(x, dtype=float) for x in (
        e1_plus, e2_plus, e1_minus, e2_minus,
        g1_plus, g2_plus, g1_minus, g2_minus,
    )]
    if len({x.shape for x in arrays}) != 1:
        raise ValueError("antithetic response arrays have different shapes")
    e1p, e2p, e1m, e2m, g1p, g2p, g1m, g2m = arrays
    gmag = np.hypot(g1p, g2p)
    if np.any(~np.isfinite(gmag)) or np.any(gmag <= 0):
        raise RuntimeError("plus-leg shear has non-finite or zero magnitude")
    opposite = np.maximum(np.abs(g1p + g1m), np.abs(g2p + g2m))
    max_opposite = float(opposite.max(initial=0.0))
    if max_opposite > shear_atol:
        raise RuntimeError(
            f"+/- target shears are not antithetic: max component sum {max_opposite:.3e}"
        )
    gh1, gh2 = g1p / gmag, g2p / gmag
    response = ((e1p - e1m) * gh1 + (e2p - e2m) * gh2) / (2.0 * gmag)
    return response, gmag, max_opposite


def case_summary(values, cases):
    """Mean and case-to-case SEM, giving each case equal weight."""
    values = np.asarray(values, dtype=float)
    cases = np.asarray(cases, dtype=np.int64)
    labels = np.unique(cases)
    means = np.array([values[cases == c].mean() for c in labels], dtype=float)
    sem = means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else np.nan
    return float(means.mean()), float(sem), means


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plus-catalogue", required=True)
    ap.add_argument("--minus-catalogue", required=True)
    ap.add_argument("--snc-lookup", required=True,
                    help="g=0 lookup used for a same-row forward/backward diagnostic")
    ap.add_argument("--reference-target", required=True,
                    help="existing V2.1 target whose pre-registered grid edges are reused")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-case", type=int, default=0)
    ap.add_argument("--max-case", type=int, default=99)
    ap.add_argument("--min-match", type=float, default=0.98)
    ap.add_argument("--min-count", type=int, default=500)
    args = ap.parse_args()

    plus_cols = [
        "case", "input_index", "r_input_p", "Re_input_p", "distance", "neighbored",
        "gamma1_input_p", "gamma2_input_p", "measured_ngmix_g1", "measured_ngmix_g2",
        "r_blend",
    ]
    minus_cols = [
        "case", "input_index", "r_input_p", "Re_input_p",
        "gamma1_input_p", "gamma2_input_p", "measured_ngmix_g1", "measured_ngmix_g2",
    ]
    zero_cols = ["case", "input_index", "ngmix0_g1", "ngmix0_g2"]

    print(f"reading +g catalogue: {args.plus_catalogue}", flush=True)
    plus = pf.read_table(args.plus_catalogue, columns=plus_cols, memory_map=True).to_pandas()
    plus = plus[(plus["case"] >= args.min_case) & (plus["case"] <= args.max_case)]
    plus = source_select_selection(plus, cuts=DEFAULT_SELECTION_CUTS)
    plus = sbs_domain.select_frame(plus)
    plus = plus.reset_index(drop=True)
    # The half-shear catalogue contains both the sheared target and the
    # unsheared primary.  Match the canonical V2.1 response-target builder by
    # retaining only rows whose primary/target shear is nonzero before forming
    # the reference population and matching the antithetic legs.
    plus_gmag = np.hypot(
        plus["gamma1_input_p"].to_numpy(float),
        plus["gamma2_input_p"].to_numpy(float),
    )
    sheared = np.isfinite(plus_gmag) & (plus_gmag > 1.0e-6)
    print(
        f"+g nonzero-target-shear filter: {int(sheared.sum()):,}/{len(sheared):,}",
        flush=True,
    )
    plus = plus.loc[sheared].reset_index(drop=True)
    plus_key = packed_key(plus["case"].to_numpy(), plus["input_index"].to_numpy())
    if np.unique(plus_key).size != plus_key.size:
        raise SystemExit("REFUSING: +g selected population has duplicate keys")
    print(f"V2.1 +g population after canonical cuts: {len(plus):,}", flush=True)

    print(f"reading -g catalogue: {args.minus_catalogue}", flush=True)
    minus = pf.read_table(args.minus_catalogue, columns=minus_cols, memory_map=True).to_pandas()
    minus = minus[(minus["case"] >= args.min_case) & (minus["case"] <= args.max_case)].reset_index(drop=True)
    minus_key = packed_key(minus["case"].to_numpy(), minus["input_index"].to_numpy())
    hit_minus, pos_minus = exact_match(plus_key, minus_key)
    minus_frac = float(hit_minus.mean())
    print(f"-g exact-key coverage: {int(hit_minus.sum()):,}/{len(hit_minus):,} ({minus_frac:.4%})", flush=True)
    if minus_frac < args.min_match:
        raise SystemExit(f"REFUSING: -g match fraction {minus_frac:.3%} < {args.min_match:.3%}")

    print(f"reading g=0 SNC lookup: {args.snc_lookup}", flush=True)
    zero = pf.read_table(args.snc_lookup, columns=zero_cols, memory_map=True).to_pandas()
    zero = zero[(zero["case"] >= args.min_case) & (zero["case"] <= args.max_case)].reset_index(drop=True)
    zero_key = packed_key(zero["case"].to_numpy(), zero["input_index"].to_numpy())
    hit_zero, pos_zero = exact_match(plus_key, zero_key)
    zero_frac = float(hit_zero.mean())
    print(f"g=0 exact-key coverage: {int(hit_zero.sum()):,}/{len(hit_zero):,} ({zero_frac:.4%})", flush=True)
    if zero_frac < args.min_match:
        raise SystemExit(f"REFUSING: g=0 match fraction {zero_frac:.3%} < {args.min_match:.3%}")

    keep = hit_minus & hit_zero
    p = plus.loc[keep].reset_index(drop=True)
    m = minus.iloc[pos_minus[keep]].reset_index(drop=True)
    z = zero.iloc[pos_zero[keep]].reset_index(drop=True)
    for col in ("r_input_p", "Re_input_p"):
        delta = np.max(np.abs(p[col].to_numpy(float) - m[col].to_numpy(float)), initial=0.0)
        if delta > 2.0e-6:
            raise RuntimeError(f"latent property {col} differs across +/- legs: max {delta:.3e}")

    central, gmag, anti_guard = central_response(
        p["measured_ngmix_g1"], p["measured_ngmix_g2"],
        m["measured_ngmix_g1"], m["measured_ngmix_g2"],
        p["gamma1_input_p"], p["gamma2_input_p"],
        m["gamma1_input_p"], m["gamma2_input_p"],
    )
    gh1 = p["gamma1_input_p"].to_numpy(float) / gmag
    gh2 = p["gamma2_input_p"].to_numpy(float) / gmag
    ep1 = p["measured_ngmix_g1"].to_numpy(float)
    ep2 = p["measured_ngmix_g2"].to_numpy(float)
    em1 = m["measured_ngmix_g1"].to_numpy(float)
    em2 = m["measured_ngmix_g2"].to_numpy(float)
    e01 = z["ngmix0_g1"].to_numpy(float)
    e02 = z["ngmix0_g2"].to_numpy(float)
    forward = ((ep1 - e01) * gh1 + (ep2 - e02) * gh2) / gmag
    backward = ((e01 - em1) * gh1 + (e02 - em2) * gh2) / gmag
    identity = float(np.nanmax(np.abs(central - 0.5 * (forward + backward))))
    if identity > 2.0e-12:
        raise RuntimeError(f"central != (forward+backward)/2: max {identity:.3e}")

    finite = np.isfinite(central) & np.isfinite(forward) & np.isfinite(backward)
    p = p.loc[finite].reset_index(drop=True)
    central, forward, backward, gmag = (
        central[finite], forward[finite], backward[finite], gmag[finite]
    )
    cases = p["case"].to_numpy(np.int64)
    _, csem, ccase = case_summary(central, cases)
    _, fsem, _ = case_summary(forward, cases)
    _, bsem, _ = case_summary(backward, cases)
    _, dsem, dcase = case_summary(central - forward, cases)
    # Match the response-target builder: rows/objects carry equal weight. Case
    # means are used only for the uncertainty, not to redefine the estimand.
    cmean = float(central.mean())
    fmean = float(forward.mean())
    bmean = float(backward.mean())
    dmean = float((central - forward).mean())

    print("\nMATCHED-ROW ESTIMATOR COMPARISON (case-level SEM)")
    print(f"  forward  (0 -> +g): {fmean:+.6f} +- {fsem:.6f}")
    print(f"  backward (-g -> 0): {bmean:+.6f} +- {bsem:.6f}")
    print(f"  central  (-g -> +g): {cmean:+.6f} +- {csem:.6f}")
    print(f"  central - forward:   {dmean:+.6f} +- {dsem:.6f}")
    print(f"  median |g|={np.median(gmag):.6f}; antithetic shear guard={anti_guard:.3e}")

    ref = np.load(args.reference_target, allow_pickle=True)
    ef = np.asarray(ref["edges_flux"], dtype=float)
    es = np.asarray(ref["edges_size"], dtype=float)
    ec = np.asarray(ref["edges_crowd"], dtype=float)
    shape = np.asarray(ref["Rsim"]).shape
    expected = (len(ef) - 1, len(es) - 1, len(ec) - 1)
    if shape != expected:
        raise RuntimeError(f"reference target shape {shape} != edge shape {expected}")

    flux = p["r_input_p"].to_numpy(float)
    size = p["Re_input_p"].to_numpy(float)
    crowd = p["r_blend"].to_numpy(float)
    in_edges = (
        (flux >= ef[0]) & (flux <= ef[-1])
        & (size >= es[0]) & (size <= es[-1])
        & (crowd >= ec[0]) & (crowd <= ec[-1])
    )
    if in_edges.mean() < 0.999999:
        raise RuntimeError(f"reference grid covers only {in_edges.mean():.6%} of matched rows")
    fi = np.clip(np.digitize(flux, ef) - 1, 0, shape[0] - 1)
    si = np.clip(np.digitize(size, es) - 1, 0, shape[1] - 1)
    ci = np.clip(np.digitize(crowd, ec) - 1, 0, shape[2] - 1)
    target = np.full(shape, cmean, dtype=float)
    counts = np.zeros(shape, dtype=np.int64)
    for a in range(shape[0]):
        for b in range(shape[1]):
            for c in range(shape[2]):
                cell = (fi == a) & (si == b) & (ci == c)
                counts[a, b, c] = int(cell.sum())
                if counts[a, b, c] >= args.min_count:
                    target[a, b, c] = float(central[cell].mean())

    metadata = {
        "plus_catalogue": os.path.abspath(args.plus_catalogue),
        "minus_catalogue": os.path.abspath(args.minus_catalogue),
        "snc_lookup": os.path.abspath(args.snc_lookup),
        "reference_target": os.path.abspath(args.reference_target),
        "cases": [args.min_case, args.max_case],
        "minus_match_fraction": minus_frac,
        "zero_match_fraction": zero_frac,
        "antithetic_shear_guard": anti_guard,
        "domain": sbs_domain.metadata(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        edges_flux=ef,
        edges_size=es,
        edges_crowd=ec,
        edges_dist=ec,
        Rsim=target,
        counts=counts.astype(float),
        raw_counts=counts,
        nominal_g=float(np.median(gmag)),
        global_R=cmean,
        global_case_sem=csem,
        forward_matched_R=fmean,
        backward_matched_R=bmean,
        central_minus_forward=dmean,
        central_minus_forward_case_sem=dsem,
        response_estimator="antithetic_random_direction_self",
        crowd_col="r_blend",
        domain=sbs_domain.describe(),
        provenance_json=json.dumps(metadata, sort_keys=True),
    )

    print(f"\ngrid {shape}, N={len(central):,}, min cell={counts.min():,}, global={cmean:.6f}")
    print(f"target range {target.min():.6f}..{target.max():.6f}")
    print(f"wrote {args.output}")
    print("ANTITHETIC_SELF_TARGET_DONE", flush=True)


if __name__ == "__main__":
    main()
