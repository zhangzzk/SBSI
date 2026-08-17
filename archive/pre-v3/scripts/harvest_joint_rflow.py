#!/usr/bin/env python -B
"""Harvest the JOINT forward model's per-object R_flow onto the certification population.

Runs the trained SetConditionedForwardModel's mean-head central secant (the SAME shape-response
computation the trainer supervises, scripts.train_forward_prototype.flow_response_perobj) over
the constant-shear cert catalogue (cases 40-139), producing a per-(case,input_index) R_flow.

Output npz {case, input_index, value} plugs into eval_selection_robustness.py --rflow-override
for the head-to-head:  m_S = <r_sim>_S / (<R_flow_JOINT>_S + <R_blend>_S) - 1  vs the certified
tabular R_flow, R_blend kept SEPARATE (user's constraint) and unchanged.

This is INFERENCE with the experimental joint model; it does NOT modify the certified m or any
certified artifact.  R_blend is untouched.
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
from sbs_shear.measurement_model import TargetStandardizer
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection
from scripts.train_forward_prototype import (
    PRIMARY_FEATURES, NEIGHBOR_FEATURES, intrinsic_shape, shifted_feature_frame, neighbor_padded,
)

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

CD = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"


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
    target_std = TargetStandardizer.from_state(ck["target_transform"])
    sc = (float(target_std.scales[0]), float(target_std.scales[1]))
    delta = float(ck["delta"])
    return model, primary_pre, nbr_std, sc, delta


@torch.no_grad()
def rflow_chunk(model, base, primary_pre, nbr_std, sc, delta, device):
    """Central-secant mean-head shape response for one dataframe chunk (physical e-units)."""
    intr = {}
    intr["e1p"], intr["e2p"] = intrinsic_shape(base, "p")
    intr["e1s"], intr["e2s"] = intrinsic_shape(base, "s")
    mus = {}
    for name, gdir, sgn in [("e1p", (1.0, 0.0), +1.0), ("e1m", (1.0, 0.0), -1.0),
                            ("e2p", (0.0, 1.0), +1.0), ("e2m", (0.0, 1.0), -1.0)]:
        f, nbg = shifted_feature_frame(base, intr, gdir, sgn * delta)
        p = torch.as_tensor(primary_pre.transform_frame(f), dtype=torch.float32, device=device)
        npad, m = neighbor_padded(f, nbg, nbr_std)
        n = torch.as_tensor(npad, dtype=torch.float32, device=device)
        mask = torch.as_tensor(m, dtype=torch.float32, device=device)
        mus[name] = model.mu(model.context(p, n, mask))
    r = 0.5 * ((mus["e1p"][:, 0] - mus["e1m"][:, 0]) * sc[0]
               + (mus["e2p"][:, 1] - mus["e2m"][:, 1]) * sc[1]) / (2.0 * delta)
    return r.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, nargs="+",
                    help="one or more joint checkpoints; per-object R_flow is AVERAGED (seed ensemble)")
    ap.add_argument("--catalogue", default=CD + "/constant_response_catalogue_train.feather")
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-batches", type=int, default=8, help="record-batches per GPU chunk")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"device={device}")
    members = [rebuild(cp, device) for cp in args.checkpoint]
    sc, delta = members[0][3], members[0][4]
    log(f"loaded {len(members)} ensemble member(s); target_scales={sc} delta={delta}")

    want = ["case", "input_index", "input_index_sec", "neighbored", "distance", "polarization_angle",
            "shear_component_convention", "redshift_input_p", "redshift_input_s",
            "Re_input_p", "Re_input_s", "axis_ratio_input_p", "axis_ratio_input_s",
            "position_angle_input_p", "position_angle_input_s", "sersic_n_input_p", "sersic_n_input_s",
            "r_input_p", "r_input_s", "RA_input_p", "RA_input_s", "DEC_input_p", "DEC_input_s"]

    out_case, out_idx, out_r = [], [], []
    n_seen = n_kept = 0
    with ipc.open_file(args.catalogue) as reader:
        avail = set(reader.schema.names)
        cols = [c for c in want if c in avail]
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
            r = np.mean([rflow_chunk(mdl, b, pp, ns, sc, delta, device)
                         for (mdl, pp, ns, _sc, _d) in members], axis=0)
            out_case.append(b["case"].to_numpy(np.int64))
            out_idx.append(b["input_index"].to_numpy(np.int64))
            out_r.append(r.astype(np.float64))
        for bi in range(reader.num_record_batches):
            t = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            n_seen += len(t)
            # catalogue is case-ascending; skip early batches entirely below min_case
            if int(t["case"].max()) < args.min_case:
                continue
            buf.append(t)
            if len(buf) >= args.chunk_batches:
                flush(buf); buf = []
        flush(buf)
    if out_case:
        n_kept = sum(len(x) for x in out_case)
    case = np.concatenate(out_case); idx = np.concatenate(out_idx); val = np.concatenate(out_r)
    log(f"scanned {n_seen:,} rows; harvested R_flow for {n_kept:,} selected objects "
        f"(cases {int(case.min())}-{int(case.max())})")
    log(f"R_flow_joint: mean={val.mean():+.4f} std={val.std():.4f} "
        f"p[1,50,99]={np.percentile(val,[1,50,99]).round(3).tolist()}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, case=case, input_index=idx, value=val)
    log(f"wrote {args.out}")
    print("HARVEST_JOINT_RFLOW_DONE", flush=True)


if __name__ == "__main__":
    main()
