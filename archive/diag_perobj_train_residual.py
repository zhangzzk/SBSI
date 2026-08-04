"""IS THE SMALL-SIZE RESPONSE OFFSET ALREADY THERE ON THE TRAINING ROWS?

THE QUESTION THIS SETTLES. WORKLOG 2026-08-03s found a SHAPE/LEVEL dissociation: the per-object pin
flattens the within-cell ramp monotonically the harder it is driven (-32.3 -> -14.5 at rw450 -> -10.9
at rw5000), while the small-size region MEAN stays 2.7-3.1% low at every weight. Information,
capacity, weight, SWA and the train/score extraction mismatch are all ruled out. So the model matches
the SHAPE of its target and misses its LEVEL, and nothing in the pin fixes the level.

Two explanations remain, and they call for completely different work:

  OPTIMISATION   -- the offset is ALREADY PRESENT on the training rows, under the training objective,
                    in the training readout. The pin was applied and simply not minimised. Fix by
                    changing the schedule / parameterisation of the SAME objective.
  GENERALISATION -- the model matches t_i on the rows it was trained on and drifts only on the ruler.
                    Then the objective is fine and the gap is a population/domain effect, and no
                    amount of re-tuning the training loss will help.

Everything measured so far has been on the RULER (half-shear cases 0-39). This script measures on the
TRAINING catalogue itself, with the TRAINING readout, against the TRAINING target -- so a nonzero
residual here is non-compliance in the strictest possible sense, with no population change and no
extraction difference left to blame.

WHAT IS COMPARED. For each row of the g=0 training catalogue, under the same domain cuts the trainer
applies (true mag < 26, true Re > 0.3):

    r_i  the mean head's induced response, read out EXACTLY as the response loss reads it
         (`model_selfresp` from eval_selfresp_gap == build_shifted_context + model._mu, central
         difference at --response-delta 0.02)
    t_i  the per-object target the loss pulled r_i toward (the same joblib the trainer consumed)

and reports mean(r_i)/mean(t_i) - 1 by region. **No sim truth enters** -- this is model-vs-its-own-
target, which is the only quantity that can distinguish the two explanations above.

FIREWALL. constgold is never read and no `m` is computed. The half-shear ruler is not read either;
this is training rows only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_selfresp_gap import (  # noqa: E402
    CROWD, FLOW_COLS, GAMMA, NGMIX, NN, domain_cut, load_ruler, model_selfresp, read_leg)
from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    intrinsic_e_abs,
    source_select_selection,
)

CAT = ("/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
       "det_meas_crowd_conc_g0.0_train_full.feather")


def load_train_rows(path, mag_max, re_min, max_rows, need, rng):
    cuts = [list(c) for c in DEFAULT_SELECTION_CUTS]
    cuts[1][1] = float(mag_max)
    cuts[3][0] = float(re_min)
    parts, kept = [], 0
    with ipc.open_file(path) as r:
        cols = [c for c in need if c in set(r.schema.names)]
        miss = [c for c in need if c not in set(r.schema.names)]
        if miss:
            raise SystemExit(f"REFUSING: the training catalogue lacks {miss}")
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            b = source_select_selection(b, cuts=cuts)
            if "detected" in b.columns:
                b = b[b["detected"].astype(bool)]
            if len(b) == 0:
                continue
            parts.append(b.reset_index(drop=True))
            kept += len(b)
            if max_rows and kept >= max_rows:
                break
    import pandas as pd
    df = pd.concat(parts, ignore_index=True)
    if max_rows and len(df) > max_rows:
        df = df.iloc[rng.choice(len(df), size=max_rows, replace=False)].reset_index(drop=True)
    print(f"training rows after the trainer's own cuts: {len(df):,}")
    return df


def load_gs_rows(gS_leg, max_case, re_min, mag_max, crowd):
    """The g=0.05 leg WITHOUT the both-detected match -- the middle rung of the ladder.

    Differs from the ruler in exactly ONE respect: no requirement that the object also be detected
    in the g=0 leg. Comparing the two therefore prices the BOTH-DETECTED cut on its own, which is
    the prime suspect because it is shear-dependent and AGENTS.md already flags it as silently
    removing objects.
    """
    import pyarrow.feather as pf
    gS = domain_cut(read_leg(gS_leg, FLOW_COLS + NGMIX + GAMMA, max_case), re_min, mag_max)
    gp = np.hypot(gS["gamma1_input_p"].to_numpy(float), gS["gamma2_input_p"].to_numpy(float))
    gS = gS[gp > 1e-6].reset_index(drop=True).drop_duplicates(["case", "input_index"])
    nbf = pf.read_table(crowd).to_pandas()[["case", "input_index", "nbr_flux_near",
                                            "nbr_flux_far", "nbr_flux_max"]]
    n0 = len(gS)
    gS = gS.merge(nbf, on=["case", "input_index"], how="left")
    if len(gS) != n0:
        raise SystemExit(f"REFUSING: crowd merge changed the row count ({n0:,} -> {len(gS):,})")
    print(f"gS-leg rows (NO both-detected match): {len(gS):,}")
    return gS


def report(name, r, t, keep, nboot, rng):
    ok = keep & np.isfinite(r) & np.isfinite(t)
    if ok.sum() < 100:
        print(f"  {name:<26} too few rows ({int(ok.sum())})")
        return
    cen = 100.0 * (float(np.mean(r[ok])) / float(np.mean(t[ok])) - 1.0)
    idx = np.flatnonzero(ok)
    bs = []
    for _ in range(nboot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        bs.append(100.0 * (float(np.mean(r[pick])) / float(np.mean(t[pick])) - 1.0))
    print(f"  {name:<26} N={int(ok.sum()):>9,}   <r_i>={np.mean(r[ok]):+.4f}  "
          f"<t_i>={np.mean(t[ok]):+.4f}   r/t - 1 = {cen:+.2f} +- {np.std(bs):.2f} %")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--target", required=True, help="the per-object target joblib the trainer used")
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--response-delta", type=float, default=0.02)
    ap.add_argument("--response-difference", choices=["forward", "central"], default="central")
    ap.add_argument("--primary-mag-max", type=float, default=26.0)
    ap.add_argument("--primary-re-min", type=float, default=0.3)
    ap.add_argument("--max-rows", type=int, default=1_000_000)
    ap.add_argument("--small-re", type=float, default=0.386)
    ap.add_argument("--faint-mag", type=float, default=25.0)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--rows", choices=["train", "gs", "ruler"], default="train",
                    help="which population to measure model-vs-target on. train = the g=0 training "
                         "catalogue (where the pin was applied); gs = the g=0.05 leg with NO "
                         "both-detected match; ruler = the both-detected matched set every earlier "
                         "number was measured on. The ladder train -> gs -> ruler switches the "
                         "population differences on one at a time.")
    ap.add_argument("--gS-leg",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
                            "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--g0-leg",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
                            "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--max-case", type=int, default=39)
    args = ap.parse_args()

    pay = joblib.load(args.target)
    feats = pay["features"]
    rng = np.random.default_rng(0)

    # FLOW_COLS is the list eval_selfresp_gap maintains for exactly this readout (everything
    # build_shifted_context + rescale + the conditioning touch, primaries AND secondaries), so the
    # requirements are taken from one place rather than rediscovered one KeyError at a time.
    need = set(feats) | set(FLOW_COLS) | {"gamma1_input_p", "gamma2_input_p",
                                          "nbr_flux_near", "nbr_flux_far", "nbr_flux_max"}
    need.discard("e_abs")                                  # derived below, not a raw column
    if args.rows == "train":
        df = load_train_rows(args.catalogue, args.primary_mag_max, args.primary_re_min,
                             args.max_rows, sorted(need), rng)
    elif args.rows == "gs":
        df = load_gs_rows(args.gS_leg, args.max_case, args.primary_re_min,
                          args.primary_mag_max, CROWD)
    else:
        df = load_ruler(args.g0_leg, args.gS_leg, args.max_case, args.primary_re_min,
                        args.primary_mag_max, 7.0, CROWD, NN)["base"]
        print(f"ruler rows (both-detected matched): {len(df):,}")
    if args.rows != "train" and args.max_rows and len(df) > args.max_rows:
        df = df.iloc[rng.choice(len(df), size=args.max_rows, replace=False)].reset_index(drop=True)
        print(f"  subsampled to {len(df):,} rows")

    for nm, how in (pay.get("derived") or {}).items():
        if nm in df.columns:
            continue
        if how != "intrinsic_e_abs(axis_ratio_input_p)":
            raise SystemExit(f"REFUSING: unknown derivation `{how}` for `{nm}`")
        df[nm] = intrinsic_e_abs(df["axis_ratio_input_p"].to_numpy(float))
    t_i = pay["model"].predict(df[feats].to_numpy(float))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpts={len(args.ckpt)}  "
          f"readout={args.response_difference} delta={args.response_delta}")
    rs = []
    for ck in args.ckpt:
        bundle = load_measurement_model(ck, device=device)
        rs.append(model_selfresp(df, bundle, args.response_delta, args.response_difference, device))
        print(f"  {Path(ck).name}: <r_i> = {np.nanmean(rs[-1]):+.5f}")
    r_i = np.mean(rs, axis=0)

    re = df["Re_input_p"].to_numpy(float)
    mag = df["r_input_p"].to_numpy(float)
    print(f"\n=== MODEL vs ITS OWN TARGET on the `{args.rows}` rows (no sim truth involved) ===")
    print(f"    ensemble of {len(args.ckpt)} ckpt(s); errors bootstrapped over ROWS\n")
    report("ALL", r_i, t_i, np.ones(len(df), bool), args.nboot, rng)
    report(f"small size Re <= {args.small_re}", r_i, t_i, re <= args.small_re, args.nboot, rng)
    report(f"faint true mag > {args.faint_mag}", r_i, t_i, mag > args.faint_mag, args.nboot, rng)

    print("\nREAD IT AS: a nonzero small-size residual HERE means the pin was applied to these very\n"
          "rows and not minimised -- an OPTIMISATION/PARAMETERISATION failure. A residual near zero\n"
          "here, against the ~-2.8% measured on the ruler, would instead make it a GENERALISATION\n"
          "gap and would redirect the work away from the training objective entirely.")
    print("\nPEROBJ_TRAINRES_DONE")


if __name__ == "__main__":
    main()
