#!/usr/bin/env python -B
"""Harvest the JOINT model's detection-head response dp = dP(s=1)/dgamma per object, then run the
BLEND-RESOLVED finite-difference gradient check against the ngmix b_true grid.

Same construction as scripts.train_forward_prototype.det_response_perobj (analytic +/-delta shear
shift of the scene context -> change in predicted P(detect), symmetric over e1/e2):
    dp = 0.5 * [ (sig(l_e1p)-sig(l_e1m)) + (sig(l_e2p)-sig(l_e2m)) ] / (2 delta)
harvested per object on the constant cert population, then binned by (a) true-mag decile and
(b) blend state (isolated + distance quantiles) using the SAME edges as derisk/btrue_detection_ngmix
so the head's dp can be compared bin-by-bin to the empirical selection bias b_true.

The head was supervised ONLY on b_true(MAG) -- reproducing the BLEND structure (isolated ~-7.5%
vs closest-blend ~-4.4%) is therefore an out-of-supervision generalization test.

DIAGNOSTIC / INFERENCE only; never wired into certified m = R_sim/(R_flow+R_blend)-1.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.ipc as ipc
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.forward_model import SetConditionedForwardModel
from sbs_shear.selection_model import TabularPreprocessor
from sbs_shear.scene_model import SetFeatureStandardizer
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection
from scripts.train_forward_prototype import (
    intrinsic_shape, shifted_feature_frame, neighbor_padded,
)

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

CD = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
BTRUE = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/btrue_detection_ngmix.npz"


def rebuild(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["model_config"]
    model = SetConditionedForwardModel(
        target_dim=cfg["target_dim"], primary_dim=cfg["primary_dim"], neighbor_dim=cfg["neighbor_dim"],
        context_dim=cfg["context_dim"], set_hidden_dim=cfg["set_hidden_dim"],
        set_neighbor_layers=2, set_context_layers=2,
        flow_hidden_dim=cfg["flow_hidden_dim"], flow_layers=cfg["flow_layers"],
        n_flows=cfg["n_flows"], mean_hidden=cfg["mean_hidden"], det_hidden=cfg["det_hidden"],
        det_layers=2, activation="silu", pooling="sum", base_flow="affine",
    ).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    primary_pre = TabularPreprocessor.from_state(ck["primary_preprocessor"])
    nbr_std = SetFeatureStandardizer.from_state(ck["neighbor_preprocessor"])
    delta = float(ck["delta"])
    return model, primary_pre, nbr_std, delta


@torch.no_grad()
def detresp_chunk(model, base, primary_pre, nbr_std, delta, device):
    """dp = dP(detect)/dgamma per object for one dataframe chunk (det_response_perobj construction)."""
    intr = {}
    intr["e1p"], intr["e2p"] = intrinsic_shape(base, "p")
    intr["e1s"], intr["e2s"] = intrinsic_shape(base, "s")
    sig = {}
    for name, gdir, sgn in [("e1p", (1.0, 0.0), +1.0), ("e1m", (1.0, 0.0), -1.0),
                            ("e2p", (0.0, 1.0), +1.0), ("e2m", (0.0, 1.0), -1.0)]:
        f, nbg = shifted_feature_frame(base, intr, gdir, sgn * delta)
        p = torch.as_tensor(primary_pre.transform_frame(f), dtype=torch.float32, device=device)
        npad, m = neighbor_padded(f, nbg, nbr_std)
        n = torch.as_tensor(npad, dtype=torch.float32, device=device)
        mask = torch.as_tensor(m, dtype=torch.float32, device=device)
        sig[name] = torch.sigmoid(model.detection_logit(model.context(p, n, mask)))
    dp = 0.5 * ((sig["e1p"] - sig["e1m"]) + (sig["e2p"] - sig["e2m"])) / (2.0 * delta)
    return dp.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, nargs="+")
    ap.add_argument("--catalogue", default=CD + "/constant_response_catalogue_train.feather")
    ap.add_argument("--btrue", default=BTRUE)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--out", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/detresp_head_blend.npz")
    ap.add_argument("--chunk-batches", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"device={device}")
    members = [rebuild(cp, device) for cp in args.checkpoint]
    delta = members[0][3]
    log(f"loaded {len(members)} member(s); delta={delta}")

    bt = np.load(args.btrue)
    mag_edges = bt["mag_edges"]; ed = bt["grid_edges_dist"]
    bt_mag = bt["mag_b"]; bt_grid = bt["grid_b"]; bt_gcnt = bt["grid_counts"]
    n_dist = int(bt["n_dist"])
    # empirical blend-bin b_true (N-weighted over mag x size), blend0 = isolated
    bt_blend = np.array([np.nansum(bt_grid[:, :, c] * bt_gcnt[:, :, c]) / max(bt_gcnt[:, :, c].sum(), 1)
                         for c in range(n_dist + 1)])
    log(f"b_true loaded: mag bins={len(bt_mag)}, blend bins={n_dist+1} (0=isolated), "
        f"blend b_true%={np.round(bt_blend*100,3).tolist()}")

    dps, mags, nbf, dist = [], [], [], []
    n_seen = 0
    want = ["case", "input_index", "input_index_sec", "neighbored", "distance", "polarization_angle",
            "shear_component_convention", "redshift_input_p", "redshift_input_s",
            "Re_input_p", "Re_input_s", "axis_ratio_input_p", "axis_ratio_input_s",
            "position_angle_input_p", "position_angle_input_s", "sersic_n_input_p", "sersic_n_input_s",
            "r_input_p", "r_input_s", "RA_input_p", "RA_input_s", "DEC_input_p", "DEC_input_s"]
    with ipc.open_file(args.catalogue) as reader:
        cols = [c for c in want if c in set(reader.schema.names)]
        buf = []
        def flush(buf):
            if not buf:
                return
            b = pd.concat(buf, ignore_index=True)
            b = b[b["case"].astype(int) >= args.min_case]
            if len(b) == 0:
                return
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                return
            dp = np.mean([detresp_chunk(mdl, b, pp, ns, delta, device)
                          for (mdl, pp, ns, _d) in members], axis=0)
            dps.append(dp.astype(np.float64))
            mags.append(b["r_input_p"].to_numpy(float))
            nbf.append(b["neighbored"].astype(bool).to_numpy())
            dist.append(b["distance"].to_numpy(float))
        for bi in range(reader.num_record_batches):
            t = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            n_seen += len(t)
            if int(t["case"].max()) < args.min_case:
                continue
            buf.append(t)
            if len(buf) >= args.chunk_batches:
                flush(buf); buf = []
        flush(buf)

    dp = np.concatenate(dps); mag = np.concatenate(mags)
    nbf = np.concatenate(nbf); dist = np.concatenate(dist)
    log(f"scanned {n_seen:,}; harvested dp for {len(dp):,} objects. <dp>={dp.mean()*100:+.3f}%")

    # --- PER-MAG gradient check ---
    mi = np.clip(np.digitize(mag, mag_edges) - 1, 0, len(mag_edges) - 2)
    print("\n" + "=" * 78)
    print("PER-MAG gradient check: head dp/dg vs empirical ngmix b_true")
    print(f"  {'bin':>3} {'N':>10} {'head%':>9} {'b_true%':>9} {'head/bt':>8}")
    head_mag = np.full(len(bt_mag), np.nan)
    for k in range(len(bt_mag)):
        s = mi == k
        if s.sum():
            head_mag[k] = dp[s].mean()
            print(f"  {k:>3} {int(s.sum()):>10,} {head_mag[k]*100:>+9.3f} {bt_mag[k]*100:>+9.3f} "
                  f"{head_mag[k]/bt_mag[k] if bt_mag[k] else float('nan'):>8.2f}")

    # --- BLEND-resolved gradient check (isolated + distance quantiles, b_true's absolute edges) ---
    di = np.where(nbf, 1 + np.clip(np.digitize(dist, ed) - 1, 0, n_dist - 1), 0)
    print("\n" + "=" * 78)
    print("BLEND-resolved gradient check (0=ISOLATED, 1..n=distance bins; b_true absolute edges)")
    print("  >>> the head was NOT supervised on blend -- this is out-of-supervision generalization")
    print(f"  {'blend':>10} {'N':>11} {'head%':>9} {'b_true%':>9} {'head/bt':>8}")
    head_blend = np.full(n_dist + 1, np.nan)
    for c in range(n_dist + 1):
        s = di == c
        tag = "ISOLATED" if c == 0 else f"blend d{c}"
        if s.sum():
            head_blend[c] = dp[s].mean()
            print(f"  {tag:>10} {int(s.sum()):>11,} {head_blend[c]*100:>+9.3f} {bt_blend[c]*100:>+9.3f} "
                  f"{head_blend[c]/bt_blend[c] if bt_blend[c] else float('nan'):>8.2f}")

    # --- MAG-CONTROLLED joint (mag-quartile x blend) gradient check (disambiguates the marginal
    #     blend discrepancy from a population mag-distribution confound) ---
    gef = bt["grid_edges_flux"]                       # b_true grid mag-quartile edges
    nfl = bt_grid.shape[0]
    gfi = np.clip(np.digitize(mag, gef) - 1, 0, nfl - 1)
    bt_ms = np.array([[np.nansum(bt_grid[a, :, c] * bt_gcnt[a, :, c]) / max(bt_gcnt[a, :, c].sum(), 1)
                       for c in range(n_dist + 1)] for a in range(nfl)])   # b_true (mag x blend), size-summed
    head_ms = np.full((nfl, n_dist + 1), np.nan)
    print("\n" + "=" * 78)
    print("MAG-CONTROLLED (mag-quartile x blend) gradient check  head% (b_true%)")
    print("  if head ISO stays ~0 across ALL mag rows -> real gap; if it tracks b_true -> marginal was a confound")
    hdr = "  ".join(["ISOLATED", "blend d1", "blend d2", "blend d3"][:n_dist + 1])
    print(f"  {'mag-q':>6}  {hdr}")
    for a in range(nfl):
        cells = []
        for c in range(n_dist + 1):
            s = (gfi == a) & (di == c)
            if s.sum() > 1000:
                head_ms[a, c] = dp[s].mean()
                cells.append(f"{head_ms[a,c]*100:+5.1f}({bt_ms[a,c]*100:+5.1f})")
            else:
                cells.append("   (thin)   ")
        print(f"  q{a} r<{gef[a+1]:.1f}: " + " ".join(cells))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, dp_mean=dp.mean(), head_mag=head_mag, bt_mag=bt_mag, mag_edges=mag_edges,
             head_blend=head_blend, bt_blend=bt_blend, dist_edges=ed, n_dist=n_dist,
             head_magblend=head_ms, bt_magblend=bt_ms, grid_edges_flux=gef)
    log(f"wrote {args.out}")
    print("HARVEST_DETRESP_BLEND_DONE", flush=True)


if __name__ == "__main__":
    main()
