"""CEILING TEST for a pin grid: score the TARGET ITSELF as if a flow bound to it perfectly.

WHAT A CEILING NUMBER MEANS. A trained flow is pulled toward its response target and can approach
it but never beat it, so substituting the target's own per-cell value for `R_flow` gives the BEST
result that grid could ever produce. Comparing two grids' ceilings therefore prices an edge
re-allocation BEFORE any GPU is spent on a retrain -- and comparing a ceiling against the flow that
was actually trained on it separates two very different failures:

    flow much worse than its ceiling  -> a BINDING problem (the flow is not reaching its target);
                                        refining the grid buys nothing until binding improves.
    flow at its ceiling               -> a TARGET problem; the grid is the thing to change.

SCOPE OF THESE NUMBERS -- READ BEFORE QUOTING. The oracle grids here are flux x size ONLY (one
crowd bin), because `harvest_grid_perobj.py` assigns the 3rd axis from `distance`/`neighbored` and
the fiducial grid's 3rd axis is `r_blend`, which does not exist on the constgold catalogue. So the
absolute ceiling reported here is NOT the fiducial 6x6x5 pin's ceiling; it is the ceiling of a
flux x size grid at the same cell budget. The DIFFERENCE between two such grids is the deliverable
and is a controlled comparison (identical axis structure, identical rows, identical R_blend); the
absolute values are a floor on what the fiducial grid can do, not an estimate of it.

THE ORACLE HAS NO SEED SPREAD. It is a deterministic lookup, so its `m` carries no seed error, while
the fiducial flow's is a 16-seed mean +- sem. The two are comparable as central values; the oracle
simply has no analogue of the flow's training noise. Do not read the absence of an error bar on the
oracle as precision -- it still inherits the target's own statistical and definitional error, which
is COMMON to every arm here and therefore cancels in the arm-to-arm comparison but not in the
absolute number.

FIREWALL. constgold supplies the per-object (mag, size) COORDINATES for the lookup, `r_sim` for
scoring, and nothing else; every grid is built from the half-shear ruler. No grid was selected on an
`m` -- the edges are fixed by `make_response_edges.py` on the ruler before this script runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plotting.plot_fid_flow_figures import CONST_CAT, _read_key_table, load_dumps  # noqa: E402

MIN_MATCH = 0.99


def load_oracle(path, ref):
    z = np.load(path)
    lk = pd.DataFrame({"case": z["case"].astype(np.int64),
                       "input_index": z["input_index"].astype(np.int64),
                       "R_oracle": z["value"].astype(float)})
    n_lk = len(lk)
    lk = lk.drop_duplicates(subset=["case", "input_index"])
    if len(lk) != n_lk:
        raise SystemExit(f"REFUSING: {path} has duplicate (case,input_index) keys "
                         f"({n_lk:,} -> {len(lk):,}); a merge would multiply rows.")
    j = ref[["case", "input_index"]].merge(lk, on=["case", "input_index"], how="left")
    if len(j) != len(ref):
        raise SystemExit(f"REFUSING: merge changed the row count for {path}.")
    v = j["R_oracle"].to_numpy(float)
    frac = float(np.isfinite(v).mean())
    print(f"  {Path(path).name}: matched {100*frac:.2f}% of dump rows, <R_oracle>={np.nanmean(v):.4f}")
    if frac < MIN_MATCH:
        raise SystemExit(f"REFUSING: {path} covers only {100*frac:.1f}% of the rows. Dropping that "
                         "many would change the population; zero-filling is the documented failure "
                         "mode that produced a spurious +28.9% m (job 15366950).")
    return v


def perbin(x, ysim, models, nb=12):
    """(centres, sim mean, sim sem, {label: residual %}) in nb equal-count bins of x."""
    ok = np.isfinite(x)
    q = np.quantile(x[ok], np.linspace(0, 1, nb + 1))
    q[0] -= 1e-9; q[-1] += 1e-9
    idx = np.clip(np.digitize(x, q) - 1, 0, nb - 1)
    idx = np.where(ok, idx, -1)
    cx = np.zeros(nb); sm = np.zeros(nb); se = np.zeros(nb)
    res = {k: np.zeros(nb) for k in models}
    for b in range(nb):
        m = idx == b
        n = int(m.sum())
        cx[b] = float(np.median(x[m]))
        sm[b] = float(np.mean(ysim[m]))
        se[b] = float(np.std(ysim[m], ddof=1) / np.sqrt(n))
        for k, y in models.items():
            res[k][b] = 100.0 * (float(np.mean(y[m])) / sm[b] - 1.0)
    return cx, sm, se, res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", action="append", required=True, metavar="LABEL=PATH",
                    help="repeatable; e.g. --oracle 'equal-count 6x6=/path/a.npz'")
    ap.add_argument("--nbins", type=int, default=12)
    args = ap.parse_args()

    seeds, ref, flows = load_dumps()
    if not seeds:
        raise SystemExit("no per-object dumps found")
    print(f"fiducial arm: {len(seeds)} seeds x {len(ref):,} rows")

    oracles = {}
    for spec in args.oracle:
        if "=" not in spec:
            raise SystemExit(f"--oracle needs LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        oracles[label] = load_oracle(path, ref)

    keep = np.ones(len(ref), dtype=bool)
    for v in oracles.values():
        keep &= np.isfinite(v)
    ref = ref.loc[keep].reset_index(drop=True)
    flows = flows[:, keep]
    oracles = {k: v[keep] for k, v in oracles.items()}
    print(f"common rows: {len(ref):,}")

    props = _read_key_table(CONST_CAT, ["case", "input_index", "r_input_p", "Re_input_p"],
                            set(ref["case"].unique().tolist()))
    ref = ref.merge(props, on=["case", "input_index"], how="left")
    del props

    ysim = ref["r_sim"].to_numpy(float)
    rb = ref["R_blend"].to_numpy(float)
    R_sim = float(np.mean(ysim))

    # ---- aggregate m -------------------------------------------------------------------------
    per = [100.0 * (R_sim / float(np.mean(flows[i].astype(float) + rb)) - 1.0)
           for i in range(len(seeds))]
    m_fid, s_fid = float(np.mean(per)), float(np.std(per, ddof=1) / np.sqrt(len(per)))
    print(f"\nAGGREGATE  (R_sim = {R_sim:.4f}, <R_blend> = {np.mean(rb):.4f})")
    print(f"  {'arm':<26} {'<R_flow>':>9}  {'m':>18}")
    print(f"  {'fiducial flow (16 seeds)':<26} {float(np.mean(flows)):9.4f}  "
          f"{m_fid:+8.3f} +- {s_fid:.3f} %")
    models = {}
    for label, v in oracles.items():
        mm = 100.0 * (R_sim / float(np.mean(v + rb)) - 1.0)
        print(f"  {'CEILING ' + label:<26} {float(np.mean(v)):9.4f}  {mm:+8.3f} %  (no seed error)")
        models["CEILING " + label] = v + rb
    models["fiducial flow"] = flows.mean(0).astype(float) + rb

    # ---- per-bin residuals -------------------------------------------------------------------
    order = ["fiducial flow"] + [f"CEILING {k}" for k in oracles]
    for col, name in (("r_input_p", "TRUE magnitude (bright -> faint)"),
                      ("Re_input_p", "TRUE size Re [arcsec] (small -> large)")):
        x = ref[col].to_numpy(float)
        cx, sm, se, res = perbin(x, ysim, models, nb=args.nbins)
        print(f"\nper-bin residual (model/truth - 1, %) vs {name}")
        print("    bin centre  : " + " ".join(f"{v:6.2f}" for v in cx))
        for k in order:
            print(f"    {k:<12}: " + " ".join(f"{v:6.2f}" for v in res[k]))
        rse = np.abs(se / sm) * 100.0
        print("    truth s.e.  : " + " ".join(f"{v:6.2f}" for v in rse))
        print("    " + "-" * 60)
        for k in order:
            r = res[k]
            print(f"    rms {k:<12} = {np.sqrt(np.mean(r**2)):6.3f}%   "
                  f"|worst| = {np.max(np.abs(r)):6.3f}%   "
                  f"bins resolved above truth s.e.: {int(np.sum(np.abs(r) > rse))}/{len(r)}")
        print("    NOTE: a bin whose |residual| is under the truth s.e. is NOT a resolved error;\n"
              "          rms must never be read without the s.e. row above it.")

    print("\nPIN_CEILING_DONE")


if __name__ == "__main__":
    main()
