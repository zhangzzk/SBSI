"""Run the reframe joint-model ENSEMBLE on constgold and assemble the shape-response closure.

Truth (constgold, both-detected matched antithetic pairs, true-cut): R_sim = 0.5*(e_+ - e_-).ghat / g
= the COHERENT response R_self+R_blend (detection matched out -> a SEPARATE channel, not here).

Model per object: R_flow (mean-head self-response, primary-only shear, ngmix units) x BRIDGE (ngmix->
constgold responsivity, 1.0853 from eval_estimator_match) [+ R_blend for blended objects].

Decisive band = ISOLATED (`neighbored`=False, i.e. NO TRUE neighbour): there R_blend=0, so
m_iso = R_sim_iso / (R_flow_iso x bridge) - 1 is a PURE flow test -- exactly where the certified
(measured-conditioned) flow failed at -8..-18% (blend_moneyplot.txt). The reframe conditions on TRUE
properties (no errors-in-variables), so this is the make-or-break number.

ENSEMBLE over the 4 reframe seeds (owner: one realization is noisy) -> mean +- seed scatter.
FIREWALL: reads constgold for VALIDATION only (its measured shape = the truth); the flow trained on
det_meas half-shear, never constgold. Bridge measured firewall-cleanly on isolated objects.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from sbs_shear.forward_model import SetConditionedForwardModel  # noqa: E402
from sbs_shear.selection_model import TabularPreprocessor  # noqa: E402
from sbs_shear.scene_model import SetFeatureStandardizer  # noqa: E402
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402
from scripts.train_joint_forward import (  # noqa: E402
    shifted_feature_frame, intrinsic_shape, neighbor_padded,
    flow_response_perobj, det_response_perobj, SHIFTS)

CDIR = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
MAG_EDGES = np.array([18.0, 24.0, 25.0, 26.0])
SIZE_EDGES = np.array([0.30, 0.38, 0.50, 1.50])
# `GOALS.md:54` acceptance cut, held fixed no matter what --true-re-min/--true-mag-max load.
# Run with the load cut RELAXED and these still split the deliverable population out, so the
# ACCEPTANCE / rejected bands are measured on identical rows -- which is what shows whether a
# global number is a real closure or a cancellation between the two.
ACC_RE_MIN, ACC_MAG_MAX = 0.30, 26.0


def load_model(path, device):
    d = torch.load(path, map_location=device, weights_only=False)
    mc = d["model_config"]
    model = SetConditionedForwardModel(
        target_dim=mc["target_dim"], primary_dim=mc["primary_dim"], neighbor_dim=mc["neighbor_dim"],
        context_dim=mc["context_dim"], set_hidden_dim=mc["set_hidden_dim"],
        set_neighbor_layers=2, set_context_layers=2,
        flow_hidden_dim=mc["flow_hidden_dim"], flow_layers=mc["flow_layers"], n_flows=mc["n_flows"],
        mean_hidden=mc["mean_hidden"], det_hidden=mc["det_hidden"], det_layers=2,
        activation="silu", pooling="sum", base_flow="affine").to(device)
    model.load_state_dict(d["state_dict"])
    model.eval()
    pp = TabularPreprocessor.from_state(d["primary_preprocessor"])
    ns = SetFeatureStandardizer.from_state(d["neighbor_preprocessor"])
    sc = (float(d["target_transform"]["scales"][0]), float(d["target_transform"]["scales"][1]))
    return model, pp, ns, sc, float(d["delta"])


def build_contexts(base, pp, ns, delta, device):
    """5 shifted (primary, neighbour) context tensors + mask; primary-only shear (isolate R_self)."""
    intr = {}
    intr["e1p"], intr["e2p"] = intrinsic_shape(base, "p")
    intr["e1s"], intr["e2s"] = intrinsic_shape(base, "s")
    out = {}
    mask = None
    for name, (gdir, sgn) in SHIFTS.items():
        f, nbg = shifted_feature_frame(base, intr, gdir, sgn * delta, primary_only=True)
        out[f"p_{name}"] = torch.as_tensor(pp.transform_frame(f), dtype=torch.float32, device=device)
        npad, m = neighbor_padded(f, nbg, ns)
        out[f"n_{name}"] = torch.as_tensor(npad, dtype=torch.float32, device=device)
        if mask is None:
            mask = torch.as_tensor(m, dtype=torch.float32, device=device)
    out["mask"] = mask
    return out


@torch.no_grad()
def run_model_on(base, model, pp, ns, sc, delta, device, chunk=200_000):
    """Per-object R_flow (mean head, ngmix units) and R_detect, in chunks."""
    n = len(base)
    Rf = np.empty(n); Rd = np.empty(n)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        D = build_contexts(base.iloc[s:e].reset_index(drop=True), pp, ns, delta, device)
        idx = torch.arange(e - s, device=device)
        Rf[s:e] = flow_response_perobj(model, D, idx, sc, delta).cpu().numpy()
        Rd[s:e] = det_response_perobj(model, D, idx, delta).cpu().numpy()
        del D
    return Rf, Rd


def load_constgold(path, max_case, true_cut):
    cols = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
            "applied_g1", "applied_g2", "Re_input_p", "r_input_p", "sersic_n_input_p", "redshift_input_p",
            "axis_ratio_input_p", "position_angle_input_p",
            "Re_input_s", "r_input_s", "sersic_n_input_s", "redshift_input_s",
            "axis_ratio_input_s", "position_angle_input_s",
            "neighbored", "distance", "case", "input_index"]
    parts = []
    with ipc.open_file(path) as r:
        avail = set(r.schema.names); use = [c for c in cols if c in avail]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            if max_case is not None:
                if int(b["case"].min()) > max_case:
                    break
                b = b[b["case"] <= max_case]
            if len(b) == 0:
                continue
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            re_min, mag_max = true_cut
            b = b[(b["Re_input_p"].to_numpy(float) > re_min) & (b["r_input_p"].to_numpy(float) < mag_max)]
            if len(b):
                parts.append(b)
    base = pd.concat(parts, ignore_index=True)
    base["polarization_angle"] = 0.0        # constgold intrinsic (axis_ratio,pos_angle) is already sky-basis
    return base.reset_index(drop=True)


def _wmean_by(vals, sel, mag, size):
    """Return global + per-mag + per-size means of vals over rows in `sel`."""
    out = {"global": float(np.nanmean(vals[sel])) if sel.any() else np.nan}
    for tag, edges, v in [("mag", MAG_EDGES, mag), ("size", SIZE_EDGES, size)]:
        for i in range(len(edges) - 1):
            m = sel & (v >= edges[i]) & (v < edges[i + 1])
            out[f"{tag}[{edges[i]:.2f},{edges[i+1]:.2f})"] = float(np.nanmean(vals[m])) if m.any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-glob",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto/forward_joint_c0-99_truecut_seed*_joint.pt")
    ap.add_argument("--bridge", type=float, default=1.0853)
    ap.add_argument("--blend-lookup", default=None,
                    help="per-(case,input_index) R_blend feather (build_blend_lookup.py; BlendEMU emulator "
                         "summed over neighbours, ngmix units). Absent (case,input_index) => R_blend=0 "
                         "(isolated / r>26). Enables the FULL deliverable m = R_sim/((R_flow+R_blend)*bridge)-1.")
    ap.add_argument("--per-mag-bridge", action="store_true",
                    help="use the BIN-RESOLVED ngmix->constgold bridge (cg_iso/ng_iso per mag bin from "
                         "eval_estimator_match run 15201137) instead of the global scalar. This removes "
                         "the scalar-bridge artifact so m_iso reflects the FLOW's accuracy alone.")
    ap.add_argument("--max-case", type=int, default=None)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--nn-lookup", default=None,
                    help="per-(case,input_index) nn_dist_bright feather (build_nn_distance_lookup.py); "
                         "when given, isolation = no BRIGHTER true neighbour within --iso-radius (7\" convention).")
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--output", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/constgold_closure.npz")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    ckpts = sorted(glob.glob(args.ckpt_glob))
    print(f"device={device}  bridge={args.bridge}  ensemble={len(ckpts)} seeds:", flush=True)
    for c in ckpts:
        print("   ", os.path.basename(c))

    base = load_constgold(CDIR + "/constant_response_catalogue_train.feather",
                          args.max_case, (args.true_re_min, args.true_mag_max))
    if args.nn_lookup:
        import pyarrow.feather as pf
        nnl = pf.read_table(args.nn_lookup).to_pandas()[["case", "input_index", "nn_dist_bright"]]
        base = base.merge(nnl, on=["case", "input_index"], how="left").reset_index(drop=True)
        nnb = base["nn_dist_bright"].to_numpy(float)
        iso = (~np.isfinite(nnb)) | (nnb > args.iso_radius)   # no BRIGHTER true neighbour within iso_radius
        print(f"isolation: no brighter true neighbour within {args.iso_radius:g}\" (7\" convention)")
    else:
        iso = ~base["neighbored"].astype(bool).to_numpy()
    print(f"constgold true-cut: N={len(base):,}  cases={base['case'].nunique()}  "
          f"isolated={int(iso.sum()):,}  blended={int((~iso).sum()):,}  ({time.time()-t0:.1f}s)", flush=True)

    # ---- truth: coherent R_sim per object (both-detected matched pairs) ----
    g = np.hypot(base["applied_g1"].to_numpy(float), base["applied_g2"].to_numpy(float))
    gmed = float(np.median(g)); gh1 = base["applied_g1"].to_numpy(float) / g; gh2 = base["applied_g2"].to_numpy(float) / g
    e1p = base["measured_e1_plus"].to_numpy(float); e2p = base["measured_e2_plus"].to_numpy(float)
    e1m = base["measured_e1_minus"].to_numpy(float); e2m = base["measured_e2_minus"].to_numpy(float)
    Rsim = 0.5 * ((e1p - e1m) * gh1 + (e2p - e2m) * gh2) / gmed        # per-object coherent response
    mag = base["r_input_p"].to_numpy(float); size = base["Re_input_p"].to_numpy(float)
    bl = ~iso

    # ---- model: ensemble R_flow (mean head, ngmix units) ----
    Rf_seeds = []
    Rd_seeds = []
    for c in ckpts:
        model, pp, ns, sc, delta = load_model(c, device)
        Rf, Rd = run_model_on(base, model, pp, ns, sc, delta, device)
        Rf_seeds.append(Rf); Rd_seeds.append(Rd)
        print(f"   seed {os.path.basename(c).split('seed')[1][:3]}: <R_flow>={np.nanmean(Rf):+.4f} "
              f"<R_flow*bridge>={np.nanmean(Rf)*args.bridge:+.4f} <dP/dg>={np.nanmean(Rd):+.4f}  ({time.time()-t0:.1f}s)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    Rf_ens = np.mean(Rf_seeds, axis=0)                 # per-object ensemble mean
    # per-mag bridge (cg_iso/ng_iso per mag bin, estimator_match 15201137): [18,24) [24,25) [25,26)
    PER_MAG_BRIDGE = np.array([1.0546, 1.0847, 1.2364])
    if args.per_mag_bridge:
        bidx = np.clip(np.digitize(mag, MAG_EDGES) - 1, 0, len(PER_MAG_BRIDGE) - 1)
        br_obj = PER_MAG_BRIDGE[bidx]
        print(f"per-mag bridge ON: {PER_MAG_BRIDGE.tolist()} for mag bins {MAG_EDGES.tolist()}")
    else:
        br_obj = np.full(len(Rf_ens), args.bridge)
    Rf_cg = Rf_ens * br_obj                             # bridge -> constgold units
    Rd_ens = np.mean(Rd_seeds, axis=0)

    # ---- R_blend (firewall-clean BlendEMU emulator, summed over aperture, ngmix units) ----
    Rblend = np.zeros(len(base))
    if args.blend_lookup and os.path.exists(args.blend_lookup):
        import pyarrow.feather as pf
        rb = pf.read_table(args.blend_lookup).to_pandas()[["case", "input_index", "R_blend"]]
        merged = base[["case", "input_index"]].merge(rb, on=["case", "input_index"], how="left")
        Rblend = merged["R_blend"].fillna(0.0).to_numpy(float)
        print(f"R_blend lookup {os.path.basename(args.blend_lookup)}: matched "
              f"{merged['R_blend'].notna().mean():.1%}  mean(blended)={Rblend[bl].mean():.4f} (ngmix)  "
              f"x bridge={np.nanmean(Rblend[bl]*br_obj[bl]):.4f} (constgold)")
    Rmodel_full = Rf_cg + Rblend * br_obj               # (R_flow + R_blend) in constgold units

    # ---- closure ----
    def m_of(Rmodel, sel):
        num = _wmean_by(Rsim, sel, mag, size); den = _wmean_by(Rmodel, sel, mag, size)
        return {k: (num[k] / den[k] - 1.0) if np.isfinite(den[k]) and den[k] != 0 else np.nan for k in num}

    # ISOLATED: pure flow (no R_blend). model = R_flow*bridge
    m_iso = m_of(Rf_cg, iso)
    # BLENDED + GLOBAL: R_blend needed; report R_sim vs R_flow*bridge, residual = R_blend to supply
    def band_report(name, sel):
        rs = _wmean_by(Rsim, sel, mag, size); rf = _wmean_by(Rf_cg, sel, mag, size)
        print(f"\n[{name}]  N={int(sel.sum()):,}")
        print(f"  {'bin':22s} {'R_sim':>8} {'Rflow*br':>9} {'resid(=Rblend)':>15} {'m(no Rblend)%':>13}")
        for k in rs:
            resid = rs[k] - rf[k]
            mval = (rs[k]/rf[k]-1.0)*100 if np.isfinite(rf[k]) and rf[k] != 0 else np.nan
            print(f"  {k:22s} {rs[k]:+8.4f} {rf[k]:+9.4f} {resid:+15.4f} {mval:+13.2f}")

    # seed scatter on the two headline numbers
    iso_ms = [(np.nanmean(Rsim[iso]) / np.nanmean(Rf_seeds[s][iso] * br_obj[iso]) - 1) * 100 for s in range(len(ckpts))]
    glob_flowbr = [np.nanmean(Rf_seeds[s] * br_obj) for s in range(len(ckpts))]

    print("\n" + "=" * 78)
    print("REFRAME ENSEMBLE on CONSTGOLD (true-cut, coherent R_sim truth)")
    print("=" * 78)
    print(f"bridge (ngmix->constgold) = {args.bridge}")
    print(f"GLOBAL:  <R_sim>={np.nanmean(Rsim):+.4f}   <R_flow*bridge>={np.nanmean(Rf_cg):+.4f} "
          f"(seed scatter {np.std(glob_flowbr):.4f})   <dP/dg>={np.nanmean(Rd_ens):+.4f}")
    print(f"\n*** ISOLATED band (neighbored=False; PURE flow test, R_blend=0) ***")
    print(f"  <R_sim_iso>={np.nanmean(Rsim[iso]):+.4f}  <R_flow_iso*bridge>={np.nanmean(Rf_cg[iso]):+.4f}")
    print(f"  m_iso GLOBAL = {m_iso['global']*100:+.2f}%   (seed scatter {np.std(iso_ms):.2f}%%; per-seed {[round(x,2) for x in iso_ms]})")
    print(f"  --> certified (measured-cond) flow was -8..-18% here; reframe target ~0%")
    print(f"  per-mag m_iso: " + "  ".join(f"{k.split('[')[1][:-1]}:{m_iso[k]*100:+.1f}%" for k in m_iso if k.startswith('mag')))
    print(f"  per-size m_iso: " + "  ".join(f"{k.split('[')[1][:-1]}:{m_iso[k]*100:+.1f}%" for k in m_iso if k.startswith('size')))

    band_report("ISOLATED (no R_blend needed)", iso)
    band_report("BLENDED (resid col = R_blend the emulator must supply)", bl)
    band_report("ALL", np.ones(len(base), bool))

    print("\n" + "=" * 78)
    print("FULL DELIVERABLE  m = R_sim / ((R_flow + R_blend) * bridge) - 1   [per-mag bridge]")
    print("=" * 78)
    acc = (size > ACC_RE_MIN) & (mag < ACC_MAG_MAX)
    bands = [("ISOLATED", iso), ("BLENDED ", bl), ("ALL     ", np.ones(len(base), bool))]
    if not acc.all():                       # only informative when the load cut was relaxed
        bands += [("ACCEPTED", acc), ("REJECTED", ~acc)]
    for name, sel in bands:
        rs = np.nanmean(Rsim[sel]); rm = np.nanmean(Rmodel_full[sel])
        mm = (rs / rm - 1) * 100 if rm else np.nan
        # per-seed spread on the same band, so a band difference can be read against seed noise
        ms = [(rs / np.nanmean((Rf_seeds[s] + Rblend)[sel] * br_obj[sel]) - 1) * 100
              for s in range(len(ckpts))]
        print(f"  {name}: m = {mm:+6.2f}%   <R_sim>={rs:.4f}  <R_flow*br+R_blend*br>={rm:.4f}  "
              f"N={int(sel.sum()):,}  (seed sd {np.std(ms):.2f}%)")
    # per-mag ALL-population m (the deliverable marginal)
    allm = []
    for a, b in [(18, 24), (24, 25), (25, 26)]:
        s = (mag >= a) & (mag < b) & np.isfinite(Rmodel_full) & np.isfinite(Rsim)
        allm.append((f"{a}-{b}", (np.nanmean(Rsim[s]) / np.nanmean(Rmodel_full[s]) - 1) * 100))
    print("  per-mag ALL m: " + "  ".join(f"{k}:{v:+.1f}%" for k, v in allm))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, Rsim=Rsim, Rflow_ens=Rf_ens, Rflow_seeds=np.array(Rf_seeds), Rblend=Rblend,
             Rdet_ens=Rd_ens, bridge=args.bridge, mag=mag, size=size, iso=iso,
             m_iso_global=m_iso["global"], iso_seed_ms=np.array(iso_ms))
    print(f"\nsaved {args.output}   ({time.time()-t0:.1f}s)")
    print("CONSTGOLD_CLOSURE_DONE")


if __name__ == "__main__":
    main()
