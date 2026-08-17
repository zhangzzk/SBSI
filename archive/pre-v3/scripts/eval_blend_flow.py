"""Score FLOW #2 against BlendEMU on the SAME held-out pairs.

WHY THIS SCRIPT AND NOT THE TRAINER'S TABLE. The trainer prints the flow's held-out agreement next to
BlendEMU numbers quoted from an earlier run over the WHOLE ap7 sample. That is not a fair contest --
different rows, and one model is being read on data it never saw while the other is not. Here both
predictions are formed on the identical held-out rows (the cases named in the checkpoint), so the
comparison is like for like.

WHAT COUNTS AS A WIN. The emulator is already within a few percent from 1.5" outward; its failure is
-36% below 0.5" and -39% at 0.5-1". So the bar is: close the close-pair bins WITHOUT degrading the
wide ones. BlendEMU's own close-pair loss weighting recovered ~4 points below 1" while pushing 1-2"
from -5.4% to -10.1% (WORKLOG 2026-07-28l) -- that is the failure mode to check for, and it is why
every bin is printed rather than a single summary number.

THREE THINGS THAT ARE NOT CLAIMS OF THIS SCRIPT:
  * Per-pair accuracy is not `m`. `m` uses R_blend SUMMED over a primary's neighbours, so the summed
    table below is the closer proxy -- and even that is measured on half-shear, not constgold.
  * The label's own sem bounds every relative number here (~3% overall, ~6% below 1"). A model and
    the emulator agreeing to better than that are not distinguishable by this measurement.
  * Errors on the SUMMED quantity are formed over PRIMARIES, not pairs, so the within-primary
    correlation is handled by construction.

FIREWALL: half-shear only; constgold is never opened.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SBSI_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.blend_flow import (  # noqa: E402
    RAW, TARGETS, build_features, load_blend_flow, response_from_contexts, shifted_shape_columns)

BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
EMU_FEATURES = ["Re_input_p", "r_input_p", "sersic_n_input_p",
                "Re_input_s", "r_input_s", "sersic_n_input_s", "distance"]

SEP_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
SIZE_EDGES = [0.30, 0.38, 0.50, 0.75, 1.50]
MAG_EDGES = [18, 22, 23, 24, 25, 26]


def cluster_sem(vals, pid, mask):
    """Standard error of the mean, CLUSTERED BY PRIMARY.

    Every pair belonging to one primary is built from the SAME two-leg measured-shape difference
    `de`, so their labels share that primary's whole measurement noise and are strongly correlated.
    A plain `std/sqrt(n_pairs)` therefore understates the error by roughly `sqrt(<k>) ~ 2` in the
    wide-separation bins, where a primary contributes several neighbours. The cluster-robust form
    below sums residuals WITHIN each primary before squaring, which is exact under arbitrary
    within-primary correlation.
    """
    v = vals[mask]
    n = v.size
    if n < 2:
        return np.nan
    resid = v - v.mean()
    _, inv = np.unique(pid[mask], return_inverse=True)
    per_primary = np.bincount(inv, weights=resid)
    return float(np.sqrt((per_primary ** 2).sum()) / n)


def table(label, name, key, edges, truth, flow, emu, pid, fmt="{:.2f}"):
    print(f"\n[{label}]")
    print(f"  {name:>16}{'N':>12}{'truth':>10}{'sem':>9} | {'flow':>10}{'flow/tr-1%':>12}"
          f" | {'emu':>10}{'emu/tr-1%':>12}{'  winner':>10}")
    chi2 = {"flow": 0.0, "emu": 0.0}
    dof = 0

    def row(nm, m, count=True):
        nonlocal dof
        n = int(m.sum())
        if n < 200:
            return
        a = float(truth[m].mean())
        s = cluster_sem(truth, pid, m)
        f = float(flow[m].mean())
        e = float(emu[m].mean())
        rf = (f / a - 1) * 100 if a else np.nan
        re = (e / a - 1) * 100 if a else np.nan
        # "better" only if the gap in the label's own sem units differs by more than 1 -- otherwise
        # the measurement cannot tell them apart and saying one wins would be overclaiming
        df_, de_ = abs(f - a) / s, abs(e - a) / s
        w = "flow" if df_ < de_ - 1 else ("emu" if de_ < df_ - 1 else "tie")
        if count:
            chi2["flow"] += df_ ** 2
            chi2["emu"] += de_ ** 2
            dof += 1
        print(f"  {nm:>16}{n:>12,}{a:>10.5f}{s:>9.5f} | {f:>10.5f}{rf:>+12.2f}"
              f" | {e:>10.5f}{re:>+12.2f}{w:>10}")

    row("ALL", np.ones(len(truth), bool), count=False)
    for i in range(len(edges) - 1):
        row(f"[{fmt.format(edges[i])},{fmt.format(edges[i+1])})",
            (key >= edges[i]) & (key < edges[i + 1]))
    if dof:
        print(f"  -> over {dof} bins, chi2/dof against the label:  flow {chi2['flow']/dof:8.1f}"
              f"   |   emu {chi2['emu']/dof:8.1f}"
              f"   ({'FLOW' if chi2['flow'] < chi2['emu'] else 'EMU'} closer overall)")
    return chi2["flow"], chi2["emu"], dof


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="lsst_r_extnbr_indom_tuned")
    ap.add_argument("--batch-size", type=int, default=200000)
    ap.add_argument("--all-rows", action="store_true",
                    help="score on every row, not just the held-out cases (DIAGNOSTIC ONLY -- the "
                         "flow has seen the training rows and the number is not a fair comparison)")
    ap.add_argument("--save-npz", default=None)
    args = ap.parse_args()

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, std, yscale, meta = load_blend_flow(args.checkpoint, device=dev)
    delta = float(meta["delta"])
    derived = bool(meta.get("derived", True))
    val_cases = set(meta.get("val_cases", []))
    print(f"checkpoint : {args.checkpoint}")
    print(f"  trained to epoch {meta.get('epoch')}, held-out chi2/dof {meta.get('val_chi2_dof'):.2f}")
    print(f"  delta={delta}  derived={derived}  held-out cases={sorted(val_cases)}")
    print(f"device={dev}")

    need = list(dict.fromkeys(RAW + TARGETS + ["blend_truth", "blend_null", "case", "pid"]))
    if meta.get("crowding", False):
        # `k` is the stored per-primary neighbour count. The split is by CASE, so every row of a
        # held-out primary is kept and recomputing k would give the same answer -- but reading the
        # stored column removes the need to rely on that argument staying true.
        from pyarrow import ipc as _ipc
        if "k" in set(_ipc.open_file(args.pairset).schema.names):
            need.append("k")
    df = pd.read_feather(args.pairset, columns=need)
    if not args.all_rows:
        if not val_cases:
            raise SystemExit("REFUSING: the checkpoint names no held-out cases and --all-rows was "
                             "not given; there is no honest comparison to make")
        df = df[df["case"].isin(val_cases)].reset_index(drop=True)
        print(f"\nscoring the HELD-OUT cases only: {len(df):,} pairs")
    else:
        print(f"\n*** --all-rows: scoring ALL {len(df):,} pairs, INCLUDING the flow's training "
              f"rows. Diagnostic only. ***")

    truth = df["blend_truth"].to_numpy(np.float64)
    nullv = df["blend_null"].to_numpy(np.float64)
    g = np.isfinite(nullv)
    nm, nsem = nullv[g].mean(), nullv[g].std(ddof=1) / np.sqrt(g.sum())
    print(f"NULL TEST on these rows: {nm:+.5f} +- {nsem:.5f} ({abs(nm)/nsem:.1f} sigma)")
    if abs(nm) > 3 * nsem:
        print("  *** NULL FAILS -- the labels on this subset are not trustworthy. ***")

    # ---- flow #2 ----
    # `crowding` comes from the CHECKPOINT, not from a flag here: the feature block a model was
    # trained with is a property of that model, and building the wrong number of columns would either
    # crash on the standardizer or, worse, line the columns up against the wrong means.
    crowding = bool(meta.get("crowding", False))
    if crowding:
        print("  checkpoint uses the CROWDING block (nbr_flux_near/far/max, log_k)")
    X = std.transform(build_features(df, derived=derived, crowding=crowding))
    sh = shifted_shape_columns(df, delta)
    i0, i1 = meta["shape_indices"]
    m0, s0 = std.mean[i0], std.scale[i0]
    m1, s1 = std.mean[i1], std.scale[i1]
    flow_r = np.empty(len(df), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(df), args.batch_size):
            e = min(s + args.batch_size, len(df))
            c0 = torch.as_tensor(X[s:e]).to(dev)
            ctxs = {}
            for k in ("e1+", "e1-", "e2+", "e2-"):
                c = c0.clone()
                c[:, i0] = torch.as_tensor(((sh[k][0][s:e] - m0) / s0).astype(np.float32)).to(dev)
                c[:, i1] = torch.as_tensor(((sh[k][1][s:e] - m1) / s1).astype(np.float32)).to(dev)
                ctxs[k] = c
            flow_r[s:e] = response_from_contexts(
                model, c0, ctxs, (yscale[0], yscale[1]), delta).cpu().numpy()
    print(f"flow response computed ({time.time()-t0:.0f}s)", flush=True)

    # ---- BlendEMU, same rows ----
    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    pred_model = BlendingPredictor.load(BLEND_MODELS, tag=args.tag, conditions=COND, device="cpu")
    emu = pred_model.predict_on_pairs(
        df[EMU_FEATURES].copy(), task="response")["response"].to_numpy(float)
    cov = np.isfinite(emu).mean()
    print(f"BlendEMU tag={args.tag}: covers {100*cov:.2f}% of these rows "
          f"({int((~np.isfinite(emu)).sum()):,} outside its stored training cuts)")
    if cov < 0.999:
        print("  Rows the emulator does not cover are EXCLUDED from every comparison below rather")
        print("  than zero-filled -- zero-filling them would silently credit the emulator with a")
        print("  prediction it did not make. The flow has no such gap; it is defined everywhere.")
    ok = np.isfinite(truth) & np.isfinite(flow_r) & np.isfinite(emu)
    print(f"rows compared: {int(ok.sum()):,}")

    truth, flow_r, emu = truth[ok], flow_r[ok], emu[ok]
    dist = df["distance"].to_numpy(float)[ok]
    tre = df["Re_input_p"].to_numpy(float)[ok]
    tmag = df["r_input_p"].to_numpy(float)[ok]
    pid = df["pid"].to_numpy(np.int64)[ok]

    print(f"\n{'='*112}")
    print("PER-PAIR blending response -- flow #2 vs BlendEMU on identical held-out rows")
    print(f"{'='*112}")
    table("by PAIR SEPARATION (arcsec)", "distance", dist, SEP_EDGES, truth, flow_r, emu, pid,
          "{:.1f}")
    table("by PRIMARY TRUE SIZE", "Re_input_p", tre, SIZE_EDGES, truth, flow_r, emu, pid)
    table("by PRIMARY TRUE MAG", "r_input_p", tmag, MAG_EDGES, truth, flow_r, emu, pid, "{:.0f}")

    # ---- by CROWDING (added 2026-08-02k) ----
    # THE QUESTION THIS ANSWERS: does a pair's blending response depend on how many OTHER galaxies
    # share the aperture? Physically it should -- a third galaxy competes for the same light and
    # changes what the pair does to the primary's measured shape -- and NEITHER model can see it.
    # Flow #2 conditions on one neighbour at a time; BlendEMU's seven features are all pair-level.
    # If both are flat against the label here, a crowding input has nothing to fix and should not be
    # built. This table is the precondition for that work, run BEFORE spending GPU on it.
    _, kinv = np.unique(pid, return_inverse=True)
    kper = np.bincount(kinv)
    kcol = kper[kinv].astype(float)
    table("by NEIGHBOUR COUNT k (crowding)", "k", kcol, [1, 2, 3, 4, 6, 9, 99],
          truth, flow_r, emu, pid, "{:.0f}")

    # ---- SUMMED per primary: the quantity `m` actually uses ----
    _, inv = np.unique(pid, return_inverse=True)
    npid = inv.max() + 1
    St = np.bincount(inv, weights=truth, minlength=npid)
    Sf = np.bincount(inv, weights=flow_r, minlength=npid)
    Se = np.bincount(inv, weights=emu, minlength=npid)
    k = np.bincount(inv, minlength=npid)
    sem = St.std(ddof=1) / np.sqrt(npid)   # already per PRIMARY, so no clustering correction needed
    print(f"\n{'='*112}")
    print("SUMMED over each primary's neighbours -- the form `m` uses "
          "(build_blend_lookup.py sums per pair)")
    print(f"{'='*112}")
    print(f"  primaries {npid:,}   <k> {k.mean():.2f}")
    print(f"  {'':>16}{'S_truth':>11}{'sem':>10}{'S_flow':>11}{'flow/tr-1%':>12}"
          f"{'S_emu':>11}{'emu/tr-1%':>12}")
    print(f"  {'ALL':>16}{St.mean():>11.5f}{sem:>10.5f}{Sf.mean():>11.5f}"
          f"{(Sf.mean()/St.mean()-1)*100:>+12.2f}{Se.mean():>11.5f}"
          f"{(Se.mean()/St.mean()-1)*100:>+12.2f}")
    print(f"\n  The label's own precision here is {100*sem/abs(St.mean()):.2f}% of the mean. Neither")
    print("  model can be shown to be better than that, and a difference smaller than it is not a")
    print("  result. This is a HALF-SHEAR number; it does not by itself move constgold `m`.")

    if args.save_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_npz)), exist_ok=True)
        np.savez(args.save_npz, truth=truth, flow=flow_r, emu=emu, dist=dist, tre=tre,
                 tmag=tmag, pid=pid, tag=np.array(args.tag))
        print(f"\nsaved {args.save_npz}")
    print(f"\nEVAL_BLEND_FLOW_DONE  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
