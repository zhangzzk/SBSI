"""Does the ap7 catalogue annotate each pair ONCE (unordered) or TWICE (both directions)?

WHY IT MATTERS BEYOND FLOW #2. `scripts/eval_pair_population_match.py` found the constgold field
gives exactly 2.00x as many neighbours per primary within 7" as the ap7 training legs -- at EVERY
magnitude cut, with neighbour magnitude, size and separation distributions that are essentially
identical between the two. A clean factor of two with identical shapes is a counting convention, not
a depth or density difference.

The hypothesis: ap7 lists each unordered pair {A, B} once, so each galaxy is annotated as primary for
only about half of the neighbours that actually blend it, while a KD-tree over the field naturally
produces both ordered pairs.

If confirmed, it affects more than this model:
  * `eval_rblend_gap_summed.py` sums the per-pair ruler over a primary's ANNOTATED neighbours, so its
    `S_truth` would be a sum over half the blenders;
  * that would offer a mundane explanation for the recorded ~1.85x gap between the ruler's summed
    truth (0.071-0.073) and constgold's `<R_blend>` (0.1358), which `AGENTS.md` currently attributes
    to the two suites having different neighbour densities;
  * and any lookup built for flow #2 must match whichever convention the training labels used.

Note carefully what would NOT be affected: the PER-PAIR label and every per-pair number derived from
it. Annotating a pair once still gives an unbiased estimate of that pair's own response. This is a
question about SUMS, not about the ruler's per-pair validity.

THE TEST. For one case, take the annotated pair rows and rebuild the neighbour list geometrically
from the same catalogue's positions. Compare counts, and check directly whether a pair (A as primary,
B as neighbour) is accompanied by (B as primary, A as neighbour).

FIREWALL: reads half-shear input columns only; nothing is fitted.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy.spatial import cKDTree
from sbs_shear.paths import CATALOGUES as CAT

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

COLS = ["case", "input_index", "RA_input_p", "DEC_input_p", "RA_input_s", "DEC_input_s",
        "distance", "r_input_p", "r_input_s"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", default=CAT + "det_meas_ngmix_ap7_g0.0_pilot.feather")
    ap.add_argument("--case", type=int, default=None, help="default: the first case present")
    ap.add_argument("--aperture", type=float, default=7.0)
    args = ap.parse_args()

    parts = []
    with pa.memory_map(args.leg, "rb") as src:
        rd = ipc.open_file(src)
        for i in range(rd.num_record_batches):
            b = pa.Table.from_batches([rd.get_batch(i)]).select(COLS).to_pandas()
            parts.append(b)
    d = pd.concat(parts, ignore_index=True)
    case = args.case if args.case is not None else int(d["case"].iloc[0])
    d = d[d["case"] == case].reset_index(drop=True)
    # Rows with a non-finite secondary position carry no pair at all (an unpaired primary placeholder);
    # they are not annotations of anything and would crash the KD-tree.
    finite = np.isfinite(d[["RA_input_p", "DEC_input_p", "RA_input_s", "DEC_input_s"]]
                         .to_numpy(float)).all(1)
    print(f"case {case}: {len(d):,} rows, {int((~finite).sum()):,} with a non-finite position "
          f"(dropped as unpaired), {d['input_index'].nunique():,} distinct primaries")
    d = d[finite].reset_index(drop=True)

    # Position keys. An earlier version of this script rebuilt the neighbour list from the PRIMARY
    # positions alone and compared counts -- that was wrong, because neighbours are drawn from the
    # FULL field (many are too faint or undetected ever to be a primary), so the primary list is a
    # sparse subsample and the comparison was meaningless (it read 9.1x). The field has to be
    # reconstructed from BOTH members of every annotated pair.
    def key_of(ra_, de_):
        return np.char.add(np.char.add(np.round(ra_, 9).astype("U20"), "_"),
                           np.round(de_, 9).astype("U20"))

    kp = key_of(d["RA_input_p"].to_numpy(float), d["DEC_input_p"].to_numpy(float))
    ks = key_of(d["RA_input_s"].to_numpy(float), d["DEC_input_s"].to_numpy(float))

    field = pd.DataFrame({
        "k": np.concatenate([kp, ks]),
        "ra": np.concatenate([d["RA_input_p"].to_numpy(float), d["RA_input_s"].to_numpy(float)]),
        "de": np.concatenate([d["DEC_input_p"].to_numpy(float), d["DEC_input_s"].to_numpy(float)]),
    }).drop_duplicates("k")
    print(f"\nfield reconstructed from BOTH members of every annotated pair: "
          f"{len(field):,} distinct galaxies")
    x = field["ra"].to_numpy(float) * np.cos(np.deg2rad(field["de"].to_numpy(float))) * 3600.0
    y = field["de"].to_numpy(float) * 3600.0
    unordered = cKDTree(np.column_stack([x, y])).query_pairs(args.aperture, output_type="ndarray")
    print(f"  unordered pairs within {args.aperture} arcsec : {len(unordered):,}")
    print(f"  ordered pairs (both directions)              : {2*len(unordered):,}")
    print(f"  annotated rows in the catalogue              : {len(d):,}")
    print(f"  annotated / unordered = {len(d)/max(len(unordered),1):.3f}"
          f"   annotated / ordered = {len(d)/max(2*len(unordered),1):.3f}")
    print("  (a value near 1.000 in ONE of these two identifies the convention; note the")
    print("   reconstructed field is itself incomplete -- galaxies with no annotated pair at all")
    print("   are invisible to it -- so read this as corroboration, not as the primary test)")

    # THE PRIMARY TEST -- symmetry, restricted to pairs whose NEIGHBOUR is itself a primary
    # somewhere in this case. Unrestricted, the fraction would be low for an irrelevant reason:
    # most neighbours are too faint or undetected to appear as a primary at all, so their reverse
    # row cannot exist regardless of the convention.
    prim_keys = set(kp.tolist())
    elig = np.fromiter((k in prim_keys for k in ks), bool, len(ks))
    fwd = set(zip(kp.tolist(), ks.tolist()))
    rev_present = np.fromiter(((s, p) in fwd for p, s in zip(kp.tolist(), ks.tolist())), bool, len(kp))
    print(f"\nSYMMETRY (the primary test)")
    print(f"  annotated pairs whose NEIGHBOUR is also a primary in this case: "
          f"{int(elig.sum()):,} of {len(d):,} ({100*elig.mean():.2f}%)")
    frac = float(rev_present[elig].mean()) if elig.any() else np.nan
    print(f"  of those, fraction whose REVERSE row is also annotated: {100*frac:.2f}%")

    print(f"\n{'='*100}\nVERDICT\n{'='*100}")
    if frac < 0.2:
        print("  Each pair is annotated in ONE direction only. A primary's annotated neighbour list")
        print("  is therefore roughly HALF of the galaxies that actually blend it, which explains")
        print("  the exact 2.00x count ratio against a KD-tree over the field.")
        print("  CONSEQUENCE: any SUM over annotated neighbours -- including the ruler's `S_truth`")
        print("  -- is a sum over half the blenders. Per-pair numbers are unaffected.")
    elif frac > 0.8:
        print("  Pairs ARE annotated in both directions, so the 2.00x ratio is NOT a counting")
        print("  convention and must come from somewhere else (field density, or a selection the")
        print("  training cuts impose). Do not apply a factor of two anywhere.")
    else:
        print(f"  Partially symmetric ({100*frac:.1f}%) -- neither explanation is")
        print("  clean; investigate before relying on any summed quantity.")
    print("\nAP7_PAIR_SYMMETRY_DONE", flush=True)


if __name__ == "__main__":
    main()
