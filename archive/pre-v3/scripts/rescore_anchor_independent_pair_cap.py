"""Rescore measured independent-neighbour anchors with a wider pair search.

The certified V2.2 metadata uses k=20.  When primary and secondary catalogues
are the same, the zero-separation self match consumes one slot, so at most 19
actual neighbours can be scored.  This script changes only the catalogue-level
pair enumeration: it applies the frozen V2.2 model to every active neighbour
found by a wider KD-tree query and reuses already measured anchor shapes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu import data_utils, nz_utils  # noqa: E402
from blendemu.inference import BlendingPredictor  # noqa: E402
from build_anchorblend_independent_response import COND, input_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--measured-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-offset", type=int, default=200)
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--pair-k", type=int, default=64)
    parser.add_argument("--calibration")
    parser.add_argument("--active-only", action="store_true", default=True)
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    if rank >= args.n_cases:
        return
    case = args.case_offset + rank
    os.makedirs(args.output_dir, exist_ok=True)
    output = os.path.join(args.output_dir, f"case{case}.feather")
    if os.path.exists(output):
        raise FileExistsError(f"refusing to overwrite {output}")

    measured_path = os.path.join(args.measured_dir, f"case{case}.feather")
    measured = pd.read_feather(measured_path)
    if "input_index" not in measured and "index" in measured:
        measured = measured.rename(columns={"index": "input_index"})
    keep = [
        "input_index", "measured_e1_plus", "measured_e2_plus",
        "measured_e1_minus", "measured_e2_minus", "case",
    ]
    measured = measured[keep].copy()
    ids = measured.input_index.to_numpy(np.int64)

    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag="lsst_r_extnbr_v22",
        conditions=COND, device="cpu",
    )
    frame = input_frame(args.base, case, args.g)
    pairs = nz_utils.make_reg_features(
        frame, frame, r_max=10.0 / 3600.0, k=args.pair_k,
    )
    cuts, _, _ = predictor._select("regression")
    pairs = data_utils.source_select_reg(pairs, cuts=cuts).copy()
    pairs["distance"] *= 3600.0
    primary = next(
        column for column in pairs if column.startswith("index") and column.endswith("_input_p")
    )
    pairs = pairs.loc[pairs[primary].isin(ids)].copy()
    pairs["u1"] = pairs.gamma1_input_s.to_numpy(float) / args.g
    pairs["u2"] = pairs.gamma2_input_s.to_numpy(float) / args.g
    norm = np.hypot(pairs.u1, pairs.u2)
    if args.active_only:
        pairs = pairs.loc[norm > 0.5].copy()
        norm = np.hypot(pairs.u1, pairs.u2)
    if not np.allclose(norm, 1.0, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"case {case}: non-unit active directions")
    pairs = predictor.predict_on_pairs(
        pairs, task="response", rescaled=False, warn_extrapolation=False,
    )
    if args.calibration:
        calibration = json.load(open(args.calibration, encoding="utf-8"))
        boundary = np.asarray([
            -np.inf if value is None and index == 0 else
            np.inf if value is None else float(value)
            for index, value in enumerate(calibration["response_bin_edges"])
        ])
        coefficient = np.asarray(calibration["coefficients"], float)
        bin_index = np.clip(
            np.searchsorted(boundary, pairs.response.to_numpy(float), side="right") - 1,
            0, len(coefficient) - 1,
        )
        pairs["response"] = pairs.response.to_numpy(float) * coefficient[bin_index]
    pairs["response_u1"] = pairs.response * pairs.u1
    pairs["response_u2"] = pairs.response * pairs.u2
    aggregate = pairs.groupby(primary, sort=False).agg(
        prediction=("response", "sum"), n_pairs=("response", "size"),
        sum_u1=("u1", "sum"), sum_u2=("u2", "sum"),
        prediction_u1=("response_u1", "sum"),
        prediction_u2=("response_u2", "sum"),
    )
    aggregate.index.name = "input_index"
    joined = measured.set_index("input_index").join(aggregate, how="inner").reset_index()
    de1 = joined.measured_e1_plus - joined.measured_e1_minus
    de2 = joined.measured_e2_plus - joined.measured_e2_minus
    joined["label_sum"] = (de1 * joined.sum_u1 + de2 * joined.sum_u2) / (2 * args.g)
    joined["null_sum"] = (-de1 * joined.sum_u2 + de2 * joined.sum_u1) / (2 * args.g)
    joined["projected_prediction"] = (
        joined.prediction_u1 * joined.sum_u1 + joined.prediction_u2 * joined.sum_u2
    )
    joined["desired_gap"] = joined.prediction - joined.label_sum
    joined["projected_gap"] = joined.projected_prediction - joined.label_sum
    joined.to_feather(output)
    capped = float((joined.n_pairs >= args.pair_k - 1).mean())
    print(
        f"case {case}: anchors={len(joined):,} pairs={len(pairs):,} "
        f"mean_n={joined.n_pairs.mean():.3f} cap_fraction={capped:.4f}",
        flush=True,
    )
    print("ANCHOR_INDEPENDENT_PAIR_CAP_RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
