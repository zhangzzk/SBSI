"""Per-pair R_blend ruler, BINNED BY THE PRIMARY'S MEASURED PROPERTIES.

WHY THIS EXISTS
---------------
On the constgold `mag>26` measured shell (N=286,822) the sim's total response is 0.34088 while the
fiducial model gives 0.44900; `R_flow` alone is 0.33624 (within 1.4% of the sim) and `R_blend`
supplies 0.11276 of the excess. Constgold measures only the SUM `R_flow + R_blend`, so it cannot say
whether the flow or the emulator is responsible. The per-pair half-shear ruler
(`scripts/eval_rblend_gap.py`) measures `R_blend` ALONE, per pair, with no constgold anywhere -- so it
can, provided we can bin it by the primary's MEASURED magnitude / S/N. We can: both legs carry the
full SExtractor block (`measured_mag_auto`, `measured_flux_radius`, `measured_flux_auto`,
`measured_fluxerr_auto`).

WHAT IS COMPARED (identical to eval_rblend_gap.py -- imported from it, not re-derived)
    truth  = ((e1_both - e1_0)*ghat1_s + (e2_both - e2_0)*ghat2_s) / |g_s|
    model  = BlendEMU response emulator on the pair's TRUE features
             (Re_input_p, r_input_p, sersic_n_input_p, Re_input_s, r_input_s, sersic_n_input_s,
              distance) -- i.e. the emulator cannot see the measurement realisation, which is exactly
             the hypothesis under test.
NULL: the same projection rotated 45 deg (spin-2 orthogonal) carries no blend signal. Reported
PER BIN here, not just globally -- a bin-selection artefact would show up there and nowhere else.

WHICH LEG DEFINES "MEASURED"
----------------------------
Default = the **g=0 leg**. That leg's neighbour is UNSHEARED, so every quantity measured on it is
statistically independent of `ghat_s`; conditioning on it therefore cannot induce a spurious
`<de . ghat_s>`. Binning on the SHEARED leg's measured mag is also computed (`by MEASURED MAG
(SHEARED leg)`) as a robustness check -- there the selection variable is weakly coupled to `ghat_s`
through the pair geometry, so its null is the thing to read first.

FIREWALL: half-shear legs only. constgold is never opened.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SBSI_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_rblend_gap import (BLEND_MODELS, CAT, COND, GAMMA, NGMIX,  # noqa: E402
                             PAIR_FEATURES, blend_truth)

PX = 0.2                       # arcsec per pixel (SExtractor FLUX_RADIUS is in pixels)
MEAS = ["measured_mag_auto", "measured_flux_radius",
        "measured_flux_auto", "measured_fluxerr_auto"]


def _stream(path, cols, max_case, keep_fn, label):
    """Read a leg batch-by-batch, applying `keep_fn` to each batch before concatenating."""
    t0, parts, nraw = time.time(), [], 0
    with ipc.open_file(path) as r:
        av = set(r.schema.names)
        missing = [c for c in cols if c not in av]
        if missing:
            raise KeyError(f"{path} lacks columns {missing}")
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            nraw += len(b)
            if max_case is not None:
                if int(b["case"].min()) > max_case:
                    break
                b = b[b["case"] <= max_case]
            b = keep_fn(b)
            if len(b):
                parts.append(b)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    print(f"  {label}: {nraw:,} raw -> {len(out):,} kept  ({time.time()-t0:.1f}s)", flush=True)
    return out


def load_legs_measured(gs_leg, g0_leg, max_case, re_min, mag_max, all_neighbours=False):
    """Both-sheared rows of the sheared leg matched to the g=0 leg, carrying BOTH legs' SExtractor
    measurements of the primary. Row selection is identical to eval_rblend_gap.load_legs."""
    gcols = (["case", "input_index", "detected", "neighbored"] + PAIR_FEATURES + NGMIX + GAMMA
             + MEAS)

    def keep_g(b):
        gp = np.hypot(b["gamma1_input_p"].to_numpy(float), b["gamma2_input_p"].to_numpy(float))
        gs = np.hypot(b["gamma1_input_s"].to_numpy(float), b["gamma2_input_s"].to_numpy(float))
        m = ((gp > 1e-6) & (gs > 1e-6)                       # the BOTH-sheared leg
             & b["detected"].astype(bool).to_numpy()
             & (b["Re_input_p"].to_numpy(float) > re_min)
             & (b["r_input_p"].to_numpy(float) < mag_max)
             # ngmix was only run where the PRIMARY is sheared; NaN rows are dropped by every
             # table anyway, so drop them here and save the memory
             & np.isfinite(b["measured_ngmix_g1"].to_numpy(float)))
        return b[m]

    zcols = ["case", "input_index", "detected", "Re_input_p", "r_input_p"] + NGMIX + MEAS

    def keep_0(b):
        # true properties are identical per (case,input_index) across legs, so the same domain cut
        # is safe here and keeps the reference leg small
        m = (b["detected"].astype(bool).to_numpy()
             & (b["Re_input_p"].to_numpy(float) > re_min)
             & (b["r_input_p"].to_numpy(float) < mag_max)
             & np.isfinite(b["measured_ngmix_g1"].to_numpy(float)))
        return b[m]

    nb = _stream(gs_leg, gcols, max_case, keep_g, "sheared leg (both-sheared, in-domain, detected)")
    if not all_neighbours:
        nb = nb.drop_duplicates(["case", "input_index"])
    else:
        k = len(nb) / max(nb[["case", "input_index"]].drop_duplicates().shape[0], 1)
        print(f"  keeping ALL neighbour rows: {k:.3f} rows per primary", flush=True)
    ref = _stream(g0_leg, zcols, max_case, keep_0, "g=0 leg (in-domain, detected)")
    ref = ref.drop_duplicates(["case", "input_index"])[["case", "input_index"] + NGMIX + MEAS]

    base = nb.merge(ref, on=["case", "input_index"], suffixes=("_g", "_0"))
    print(f"matched both-detected, in-domain: N={len(base):,}  cases={base['case'].nunique()}",
          flush=True)
    return base


_PID = None          # per-row primary id (0..NPID-1); set once in main()
_NPID = 0


def _sem(vals, pid):
    """Cluster-robust sem of the mean, clusters = the PRIMARY galaxy.

    With one neighbour row per primary (the 3" catalogue) every cluster has size 1 and this reduces
    to the ordinary sem. With the 7"-aperture catalogue a primary contributes ~4 neighbour rows whose
    measurement noise is the SAME `de` projected on different (independent) neighbour directions;
    the rows are therefore not independent and the ordinary sem would be optimistic.
    """
    n = vals.size
    if n == 0:
        return np.nan
    mean = vals.mean()
    if pid is None:
        return float(vals.std(ddof=1)) / np.sqrt(n)
    s = np.bincount(pid, weights=vals - mean, minlength=_NPID)
    nc = int(np.count_nonzero(np.bincount(pid, minlength=_NPID)))
    var = float((s * s).sum()) / n ** 2
    if nc > 1:
        var *= nc / (nc - 1.0)
    return np.sqrt(var)


def _row(name, truth, pred, null, m, wid=26):
    n = int(m.sum())
    if n < 200:
        return None
    pid = _PID[m] if _PID is not None else None
    t = float(truth[m].mean())
    sem = _sem(truth[m], pid)
    nl = float(null[m].mean())
    nsem = _sem(null[m], pid)
    line = f"  {name:<{wid}} {t:>8.4f} {sem:>8.4f}"
    cells = []
    for p in pred:
        b = float(p[m].mean())
        rel = (b / t - 1) * 100 if t else np.nan
        relsem = abs(b / t) * (sem / abs(t)) * 100 if t else np.nan
        cells.append(f" {b:>8.4f} {rel:>+8.2f} {relsem:>6.2f}")
    print(line + "".join(cells) + f" {nl:>+8.4f} {nsem:>7.4f} {n:>10,}")
    return dict(name=name, n=n, truth=t, sem=sem, null=nl, null_sem=nsem,
                pred=[float(p[m].mean()) for p in pred])


def table(label, name, key, edges, truth, pred, null, tags, sel=None, fmt="{:.2f}"):
    print(f"\n[{label}]")
    hdr = f"  {name:<26} {'truth':>8} {'sem':>8}"
    for t in tags:
        hdr += f" {t[-14:]:>8} {'rel%':>8} {'+-':>6}"
    hdr += f" {'null':>8} {'nsem':>7} {'N':>10}"
    print(hdr)
    good = np.isfinite(truth) & np.isfinite(null)
    for p in pred:
        good &= np.isfinite(p)
    if sel is not None:
        good &= sel
    rows = [_row("ALL", truth, pred, null, good)]
    kk = np.asarray(key, float)
    for i in range(len(edges) - 1):
        m = good & (kk >= edges[i]) & (kk < edges[i + 1])
        nm = f"[{fmt.format(edges[i])},{fmt.format(edges[i+1])})"
        rows.append(_row(nm, truth, pred, null, m))
    return [r for r in rows if r]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gs-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--max-case", type=int, default=None)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--tags", nargs="+",
                    default=["lsst_r_extnbr_indom_tuned", "lsst_r_extnbr_ho"],
                    help="BlendEMU tags to score (first = fiducial)")
    ap.add_argument("--shell-mag", type=float, default=26.0,
                    help="measured-mag boundary defining the constgold 'faint shell'")
    ap.add_argument("--all-neighbours", action="store_true",
                    help="keep every annotated neighbour row per primary instead of one "
                         "(use with the ap7 7\"-aperture legs; errors become cluster-robust)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    global _PID, _NPID
    t0 = time.time()
    base = load_legs_measured(args.gs_leg, args.g0_leg, args.max_case,
                              args.true_re_min, args.true_mag_max, args.all_neighbours)
    key = (base["case"].to_numpy(np.int64) << 32) + base["input_index"].to_numpy(np.int64)
    codes, _ = pd.factorize(key)
    _PID = codes.astype(np.int64)
    _NPID = int(_PID.max()) + 1
    print(f"clusters (distinct primaries) = {_NPID:,}  ->  {len(base)/_NPID:.3f} rows per primary")
    truth = blend_truth(base)
    null = blend_truth(base, rotate45=True)

    g = np.isfinite(null)
    nm, nsem = null[g].mean(), null[g].std(ddof=1) / np.sqrt(g.sum())
    print(f"\nGLOBAL NULL (45-deg rotated, must be ~0): {nm:+.5f} +- {nsem:.5f} "
          f"({abs(nm)/nsem:.1f} sigma)  N={g.sum():,}")
    if abs(nm) > 3 * nsem:
        print("  *** GLOBAL NULL FAILS -- do not trust anything below. ***")
    sd = float(np.nanstd(truth))
    iid = sd / np.sqrt(int(np.isfinite(truth).sum()))
    clu = _sem(truth[np.isfinite(truth)], _PID[np.isfinite(truth)])
    print(f"per-pair truth scatter: sd={sd:.3f}  ->  sem(iid)={iid:.5f}  "
          f"sem(cluster-robust)={clu:.5f}  inflation={clu/iid:.3f}x "
          f"(1.00x is expected when there is one row per primary)")

    # ---- emulator predictions (true features only; the emulator never sees a measurement) -------
    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    preds = []
    for tag in args.tags:
        pm = BlendingPredictor.load(BLEND_MODELS, tag=tag, conditions=COND, device="cpu")
        out = pm.predict_on_pairs(base[PAIR_FEATURES].copy(), task="response")
        preds.append(out["response"].to_numpy(float))
        print(f"emulator {tag}: <pred>={np.nanmean(preds[-1]):+.5f}   "
              f"(<truth>={np.nanmean(truth):+.5f})", flush=True)

    # ---- binning variables ----------------------------------------------------------------------
    mag0 = base["measured_mag_auto_0"].to_numpy(float)
    magg = base["measured_mag_auto_g"].to_numpy(float)
    sn0 = (base["measured_flux_auto_0"].to_numpy(float)
           / base["measured_fluxerr_auto_0"].to_numpy(float))
    sz0 = base["measured_flux_radius_0"].to_numpy(float) * PX
    tmag = base["r_input_p"].to_numpy(float)
    tre = base["Re_input_p"].to_numpy(float)
    dist = base["distance"].to_numpy(float)
    dmag = mag0 - tmag                      # measurement residual: >0 = scattered FAINT
    nbr = base["neighbored"].to_numpy(bool)

    print(f"\nbinning variables (g=0 leg): measured mag finite={np.isfinite(mag0).mean():.3%}  "
          f"median={np.nanmedian(mag0):.3f} | S/N median={np.nanmedian(sn0):.1f} | "
          f"measured size median={np.nanmedian(sz0):.4f}\" | "
          f"dmag = meas-true: median={np.nanmedian(dmag):+.4f} sd={np.nanstd(dmag):.4f}")
    print(f"neighboured fraction of the ruler sample: {nbr.mean():.3%}   "
          f"distance: median={np.nanmedian(dist):.3f}\" max={np.nanmax(dist):.4f}\"")
    shell = mag0 >= args.shell_mag
    print(f"faint shell (measured mag >= {args.shell_mag}): {shell.mean():.3%} of rows "
          f"({int(shell.sum()):,})")

    tags = list(args.tags)
    res = {}
    res["mag0"] = table("by PRIMARY MEASURED MAG (g=0 leg)  <-- THE TEST", "measured_mag_auto(0)",
                        mag0, [18, 22, 23, 24, 24.5, 25, 25.5, 26, 26.5, 27, 32],
                        truth, preds, null, tags)
    res["mag0_far"] = table("by PRIMARY MEASURED MAG (g=0 leg), separation >= 1\" only "
                            "[removes the known close-pair deficit]", "measured_mag_auto(0)",
                            mag0, [18, 24, 25, 25.5, 26, 26.5, 27, 32],
                            truth, preds, null, tags, sel=(dist >= 1.0))
    res["sn0"] = table("by PRIMARY MEASURED S/N (FLUX_AUTO/FLUXERR_AUTO, g=0 leg)", "S/N(0)",
                       sn0, [0, 5, 10, 15, 20, 30, 50, 100, 1e9],
                       truth, preds, null, tags, fmt="{:.0f}")
    res["size0"] = table("by PRIMARY MEASURED SIZE (FLUX_RADIUS*0.2\", g=0 leg)", "R_meas(0) [\"]",
                         sz0, [0.0, 0.55, 0.60, 0.65, 0.70, 0.80, 1.00, 10.0],
                         truth, preds, null, tags)
    res["dmag"] = table("by MEASUREMENT RESIDUAL dmag = measured(0) - TRUE mag "
                        "[>0 = scattered FAINT]", "dmag",
                        dmag, [-5, -0.5, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 1.0, 5.0],
                        truth, preds, null, tags)
    for lo, hi in ((24.0, 25.0), (25.0, 26.0)):
        res[f"mag0_true{lo}"] = table(
            f"by MEASURED MAG (g=0 leg) INSIDE the true-mag slice {lo}-{hi} "
            "[true props ~fixed, so the spread is pure measurement noise]",
            "measured_mag_auto(0)", mag0,
            [18, lo - 0.5, lo, lo + 0.5, hi, hi + 0.5, hi + 1.0, 32],
            truth, preds, null, tags, sel=(tmag >= lo) & (tmag < hi))
    res["magg"] = table("ROBUSTNESS: by PRIMARY MEASURED MAG (SHEARED leg) "
                        "[read its null first]", "measured_mag_auto(g)",
                        magg, [18, 24, 25, 25.5, 26, 26.5, 27, 32],
                        truth, preds, null, tags)
    res["tmag"] = table("CONTINUITY: by PRIMARY TRUE MAG", "r_input_p",
                        tmag, [18, 22, 23, 24, 25, 26], truth, preds, null, tags)
    res["tre"] = table("CONTINUITY: by PRIMARY TRUE SIZE", "Re_input_p",
                       tre, [0.30, 0.38, 0.50, 0.75, 1.50], truth, preds, null, tags)
    res["dist"] = table("CONTINUITY: by PAIR SEPARATION", "distance [\"]",
                        dist, [0, 1, 2, 3, 4, 5, 7, 10], truth, preds, null, tags)

    # 2x2: the shell question crossed with separation
    print(f"\n[SHELL SUMMARY: measured mag (g=0 leg) vs {args.shell_mag}, crossed with separation]")
    hdr = f"  {'cell':<26} {'truth':>8} {'sem':>8}"
    for t in tags:
        hdr += f" {t[-14:]:>8} {'rel%':>8} {'+-':>6}"
    hdr += f" {'null':>8} {'nsem':>7} {'N':>10}"
    print(hdr)
    good = np.isfinite(truth) & np.isfinite(null)
    for p in preds:
        good &= np.isfinite(p)
    cells = {}
    for lab, m in (("bright  (all sep)", good & ~shell),
                   ("SHELL   (all sep)", good & shell),
                   ("bright  sep<1\"", good & ~shell & (dist < 1)),
                   ("SHELL   sep<1\"", good & shell & (dist < 1)),
                   ("bright  sep>=1\"", good & ~shell & (dist >= 1)),
                   ("SHELL   sep>=1\"", good & shell & (dist >= 1))):
        cells[lab] = _row(lab, truth, preds, null, m)
    res["shell"] = [v for v in cells.values() if v]

    # how much of the sample the fiducial emulator's own stored cuts would actually score
    incut = (tmag > 18) & (tmag < 26) & (tre > 0.3) & (tre < 1.5)
    print(f"\nfiducial (_indom_tuned) stored primary cuts mag[18,26] Re[0.3,1.5] cover "
          f"{incut.mean():.3%} of the ruler sample; the faint shell specifically: "
          f"{incut[shell].mean():.3%}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, truth=truth, null=null,
                 **{f"pred_{t}": p for t, p in zip(tags, preds)},
                 mag0=mag0, magg=magg, sn0=sn0, sz0=sz0, dmag=dmag,
                 tmag=tmag, tre=tre, dist=dist, neighbored=nbr, pid=_PID,
                 case=base["case"].to_numpy(int), input_index=base["input_index"].to_numpy(int),
                 tags=np.array(tags))
        print(f"\nsaved {args.output}")
    print(f"RBLEND_GAP_MEASURED_DONE  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
