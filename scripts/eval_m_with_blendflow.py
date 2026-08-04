"""End-to-end: constgold `m` with FLOW #2's `R_blend` in place of BlendEMU's.

    m = R_sim / (R_flow + R_blend) - 1        (AGENTS.md sign convention, parameter-free)

Everything except `R_blend` is held fixed at the fiducial setup: the same per-object dumps, the same
16 dom6x6 flow-#1 seeds, the same `r_sim`. Only the blend term is swapped, so any change in `m` is
attributable to it. The fiducial emulator lookup is scored on the SAME rows in the same run, because
a number that cannot be compared to the thing it replaces is not a result.

SEEDS. `m` is a bias on a SHAPE response, so AGENTS.md's e-response standard binds and all 16 flow-#1
seeds are used, with the ratio formed INSIDE each seed and the spread taken across seeds (building it
from ensemble means and propagating the numerator alone is a recorded bug). Note what this error does
NOT contain: flow #2's OWN seed variance. One flow-#2 checkpoint is a single draw, so the quoted
error is the flow-#1 seed error only and is a LOWER BOUND until flow #2 is itself ensembled over 16
seeds. Stated on every line of output rather than in a footnote.

FIREWALL. constgold stays evaluation-only. Flow #2 was trained on half-shear pairs and never saw a
constgold measurement; this script reads constgold to SCORE it, exactly as the BlendEMU lookup is
scored. Nothing here fits, tunes or selects anything.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from plotting.plot_fid_flow_figures import BLEND_LOOKUP, load_dumps  # noqa: E402


def m_per_seed(rsim, rf, rb, mask):
    """(mean %, sem %, R_sim, R_flow, R_blend, N) with the ratio formed INSIDE each seed."""
    n = int(mask.sum())
    if n < 1000:
        return (np.nan,) * 5 + (n,)
    st = float(np.mean(rsim[mask]))
    bl = float(np.mean(rb[mask]))
    ms = np.array([st / (float(np.mean(rf[s, mask])) + bl) - 1.0 for s in range(rf.shape[0])])
    return (float(ms.mean()) * 100, float(ms.std(ddof=1) / np.sqrt(len(ms))) * 100,
            st, float(np.mean(rf[:, mask])), bl, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow-lookup", required=True,
                    help="feather from scripts/build_blend_lookup_flow.py")
    ap.add_argument("--emu-lookup", default=BLEND_LOOKUP)
    ap.add_argument("--flow-col", default="R_blend")
    ap.add_argument("--emu-col", default="R_blend",
                    help="pass --flow-lookup and --emu-lookup the SAME file from "
                         "build_blend_lookup_both.py with --flow-col R_blend_flow "
                         "--emu-col R_blend_emu to compare both models on one pair list")
    ap.add_argument("--in-domain-only", action="store_true",
                    help="keep only rows the flow's `in_domain` flag marks as inside its training "
                         "domain (recommended; outside it the network extrapolates)")
    args = ap.parse_args()

    seeds, ref, flows = load_dumps()
    if not seeds:
        raise SystemExit("REFUSING: no per-object dumps found")
    print(f"dumps: {len(ref):,} rows, {len(seeds)} flow-#1 seeds {seeds}")
    key = ["case", "input_index"]
    rsim = ref["r_sim"].to_numpy(float)
    rf = flows.astype(np.float64)

    def joined(path, col="R_blend", extra=None):
        lk = pf.read_table(path, memory_map=True).to_pandas()
        cols = key + [col] + (extra or [])
        j = ref[key].merge(lk[cols], on=key, how="left")
        v = j[col].to_numpy(float)
        ok = np.isfinite(v)
        print(f"  {os.path.basename(path):<48} matched {100*ok.mean():6.2f}%  "
              f"<{col}> = {np.nanmean(v):.4f}")
        return v, ok, j

    print("\nlookups:")
    matched_pairs = (os.path.abspath(args.emu_lookup) == os.path.abspath(args.flow_lookup))
    rb_emu, ok_emu, _ = joined(args.emu_lookup, col=args.emu_col)
    rb_flw, ok_flw, jf = joined(args.flow_lookup, col=args.flow_col,
                                extra=["in_domain", "n_neighbours"])
    if matched_pairs:
        print("  both columns come from ONE pair list (build_blend_lookup_both.py): the two rows")
        print("  below differ by the MODEL only -- no aperture, k-cap or selection confound.")

    # Rows either lookup does not cover are DROPPED, never zero-filled. Zero-filling is the recorded
    # silent failure that produced a spurious +28.9% m (AGENTS.md), and it credits a model with a
    # prediction it did not make.
    keep = ok_emu & ok_flw
    if args.in_domain_only:
        indom = jf["in_domain"].to_numpy()
        indom = np.where(np.isfinite(rb_flw), indom.astype(bool), False)
        keep &= indom
        print(f"\n--in-domain-only: restricting to the flow's training domain")
    print(f"rows scored by BOTH models: {int(keep.sum()):,} of {len(ref):,} "
          f"({100*keep.mean():.2f}%)")
    if keep.mean() < 0.5:
        print("  *** coverage below 50% -- the comparison population is no longer the deliverable "
              "population; read the numbers below as diagnostics only. ***")

    print(f"\n{'='*104}")
    print("constgold m -- ONLY R_blend differs between these two rows")
    print(f"{'='*104}")
    print(f"  {'R_blend source':>28}{'m %':>10}{'+- %':>9}{'R_sim':>10}{'R_flow':>10}"
          f"{'R_blend':>10}{'N':>14}")
    out = {}
    for name, rb in (("BlendEMU (fiducial)", rb_emu), ("FLOW #2", rb_flw)):
        r = m_per_seed(rsim, rf, np.nan_to_num(rb), keep)
        out[name] = r
        print(f"  {name:>28}{r[0]:>+10.3f}{r[1]:>9.3f}{r[2]:>10.4f}{r[3]:>10.4f}"
              f"{r[4]:>10.4f}{r[5]:>14,}")

    a, b = out["BlendEMU (fiducial)"], out["FLOW #2"]
    print(f"\n  difference (FLOW #2 - BlendEMU): {b[0]-a[0]:+.3f} pt of m, from a blend term "
          f"{b[4]-a[4]:+.4f} ({100*(b[4]/a[4]-1) if a[4] else np.nan:+.1f}%)")
    print("\n  THE QUOTED ERROR IS THE FLOW-#1 SEED SPREAD ONLY. Flow #2 contributes its own seed")
    print("  variance, which is NOT included -- one checkpoint is a single draw. Until flow #2 is")
    print("  ensembled over 16 seeds (AGENTS.md e-response standard), treat the error as a LOWER")
    print("  BOUND and do not quote this `m` as a certified number.")
    if matched_pairs:
        print("\n  Both R_blend columns are sums over the SAME pairs, so the difference above is")
        print("  attributable to the models. It is still not a licence to pick between them on `m`")
        print("  -- constgold is evaluation-only; the per-pair ruler decides.")
    else:
        print("\n  Also note the two lookups are built over different neighbour sets: flow #2 sums")
        print("  inside a 7\" aperture matching its ap7 training, while the BlendEMU lookup hands the")
        print("  whole field to `predict_response`. Part of any difference is that, not the model.")
        print("  Use build_blend_lookup_both.py to remove that confound.")
    print("\nEVAL_M_BLENDFLOW_DONE", flush=True)


if __name__ == "__main__":
    main()
