"""Does the FIDUCIAL V2 model also carry ~+0.8% m on the V2.1 population?

The V2.1 selection table (job 15527267) put the no-cut `m` at +0.841% +- 0.233%, against the
fiducial's -0.123%. Two readings, and they call for opposite responses:

  (a) the V2.1 retrain is worse, or
  (b) the fiducial -0.123% was a CANCELLATION between sub-populations, and the V2.1 domain simply
      keeps the positive half. AGENTS.md already records such a split on the V2 domain -- the
      emulator-kept 78% gives +0.830% and the dropped 22% gives -3.358%, averaging to ~0 -- but that
      split is by the EMULATOR's pair cuts, NOT by the V2.1 domain. It is a different partition of a
      different population and cannot settle this on its own.

This script runs the same decomposition with the V2.1 domain as the mask. It is legitimate because
the V2.1 domain NESTS STRICTLY INSIDE the V2 one: Re > 0.5" is inside Re > 0.3", and the S/N > 10
limiting magnitude peaks at 25.72 (at the Re = 0.5" edge), under V2's mag < 26. So the fiducial flow
is fully IN-domain on every V2.1 row and this is not an extrapolation.

READ IT AS: if V2.1-IN lands near +0.8% for the FIDUCIAL model too, the bias belongs to the
population and the V2.1 retrain is exonerated -- reading (b). If V2.1-IN lands near zero, the
fiducial handles this population fine and the retrain is what introduced the bias -- reading (a).

The emulator cannot drive the comparison either way: on 6,208,896 identical galaxies the V2.1 and
fiducial emulators agree to 1.8% (corr 0.989), with V2.1 slightly HIGHER, which pushes `m` down.

It changes nothing and fits nothing; it is a read-and-print over existing products.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from plotting.plot_fid_flow_figures import dump_paths, load_dumps  # noqa: E402
from sbs_shear import domain as sbs_domain  # noqa: E402

# The dumps carry true magnitude but NOT true size, and the V2.1 domain needs both. `Re_input_p`
# comes back off the constgold catalogue -- the SAME file the selection table reads, so the two
# paths cannot disagree about which galaxies the domain contains.
CG = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
      "constant_response_catalogue_train.feather")
EXTRA = ["r_input_p", "neighbored", "distance"]


def block(name, mask, ref, flows):
    """One row of the table. `m` is formed PER SEED then averaged (AGENTS.md: never from ensemble
    means), and the spread across seeds is reported as the error on that mean."""
    n = int(mask.sum())
    if n == 0:
        print(f"  {name:>26}{0:>12}")
        return
    rsim = float(ref["r_sim"].to_numpy(float)[mask].mean())
    rflow = float(flows[:, mask].mean())
    rbl = float(ref["R_blend"].to_numpy(float)[mask].mean())
    ms = np.array([rsim / (float(flows[s, mask].mean()) + rbl) - 1.0 for s in range(flows.shape[0])])
    err = float(np.std(ms, ddof=1) / np.sqrt(len(ms))) if len(ms) > 1 else np.nan
    mag = float(ref["r_input_p"].to_numpy(float)[mask].mean())
    re = float(ref["Re_input_p"].to_numpy(float)[mask].mean())
    nb = float(ref["neighbored"].to_numpy()[mask].mean())
    print(f"  {name:>26}{n:>12,}{100*n/len(ref):>8.2f}%{rsim:>10.4f}{rflow:>10.4f}{rbl:>10.4f}"
          f"{100*ms.mean():>+9.3f}{100*err:>8.3f}{mag:>9.3f}{re:>8.3f}{100*nb:>8.1f}%")


def main():
    seeds, ref, flows = load_dumps()
    if not seeds:
        raise SystemExit("REFUSING: no per-object dumps found")
    flows = flows.astype(np.float64)
    key = ["case", "input_index"]

    # `load_dumps` reads only case/input_index/r_sim/R_blend and has already masked `ref` down to
    # the tuned emulator's coverage, so the identifying properties are merged back on the key rather
    # than assumed positional. `flows` IS positional, hence the row-count assertions throughout.
    paths = dump_paths()
    props = pf.read_table(paths[sorted(paths)[0]], columns=key + EXTRA).to_pandas()
    n_before = len(ref)
    ref = ref.merge(props, on=key, how="left")
    if len(ref) != n_before:
        raise SystemExit(f"REFUSING: property merge changed the row count {n_before:,} -> "
                         f"{len(ref):,}; `flows` is positional and would misalign.")

    cat = pf.read_table(CG, columns=key + [sbs_domain.RE_COLUMN]).to_pandas()
    n_uniq = len(cat[key].drop_duplicates())
    if len(cat) != n_uniq:
        raise SystemExit(f"REFUSING: {len(cat):,} catalogue rows but only {n_uniq:,} unique keys; a "
                         "left merge would lengthen the frame and misalign the mask against flows.")
    ref = ref.merge(cat, on=key, how="left")
    if len(ref) != n_before:
        raise SystemExit(f"REFUSING: size merge changed the row count {n_before:,} -> {len(ref):,}.")

    mag = ref[sbs_domain.MAG_COLUMN].to_numpy(float)
    re = ref[sbs_domain.RE_COLUMN].to_numpy(float)
    miss = int((~np.isfinite(re)).sum())
    print(f"\ncatalogue: {len(cat):,} rows; joined true size onto {n_before:,} dump rows, "
          f"{miss:,} unmatched")
    if miss:
        # Not zero-filled and not silently dropped -- an unmatched row has no domain verdict, so it
        # would be counted as OUT by default and quietly inflate that group.
        raise SystemExit("REFUSING: some dump rows got no true size from the catalogue; the domain "
                         "mask would be wrong for them.")

    indom = sbs_domain.in_domain(mag, re)
    print(f"  {sbs_domain.describe()}")
    print(f"V2.1 domain covers {indom.sum():,} of {len(ref):,} dump rows ({100*indom.mean():.2f}%)")

    print(f"\n{'='*126}")
    print("The SAME fiducial dump rows and the SAME fiducial model, split by the V2.1 domain.")
    print("Every column is the FIDUCIAL V2 model (dom6x6 flow + tuned in-domain emulator).")
    print(f"{'='*126}")
    print(f"  {'group':>26}{'N':>12}{'share':>9}{'R_sim':>10}{'R_flow':>10}{'R_blend':>10}"
          f"{'m %':>9}{'+-':>8}{'mag_p':>9}{'Re_p':>8}{'nbrd':>9}")
    block("ALL (fiducial)", np.ones(len(ref), bool), ref, flows)
    block("V2.1 domain: IN", indom, ref, flows)
    block("V2.1 domain: OUT", ~indom, ref, flows)

    n_in = int(indom.sum())
    w = n_in / len(ref)
    print(f"\nweight check: {w:.3f} x m(IN) + {1-w:.3f} x m(OUT) should reproduce m(ALL).")
    print("\nCOMPARE m(IN) AGAINST THE V2.1 MODEL'S OWN +0.841% +- 0.233% (job 15527267, 4 seeds,")
    print("SWA-32). Same population, same catalogue, same estimator -- only the model differs.")
    print("  m(IN) ~ +0.8%  -> the bias is the POPULATION's; the V2.1 retrain did not cause it, and")
    print("                    the fiducial -0.123% was a cancellation that the V2.1 cut removes.")
    print("  m(IN) ~ 0      -> the fiducial handles this population fine; the V2.1 retrain owns the")
    print("                    bias, and the carried-over full-population coupling target is the")
    print("                    first suspect (jobs/job_flow_v21.sh flags it).")
    print("\nDIAG_V21_DOMAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
