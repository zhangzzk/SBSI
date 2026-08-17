"""Apply frozen V2.2 to coherent anchors with k=20 and a wider k query."""
from __future__ import annotations

import argparse
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


def predict_sum(predictor, frame: pd.DataFrame, ids: np.ndarray, k: int) -> pd.DataFrame:
    pairs = nz_utils.make_reg_features(frame, frame, r_max=10.0 / 3600.0, k=k)
    cuts, _, _ = predictor._select("regression")
    pairs = data_utils.source_select_reg(pairs, cuts=cuts).copy()
    pairs["distance"] *= 3600.0
    primary = next(
        column for column in pairs if column.startswith("index") and column.endswith("_input_p")
    )
    pairs = pairs.loc[pairs[primary].isin(ids)].copy()
    pairs = predictor.predict_on_pairs(
        pairs, task="response", rescaled=False, warn_extrapolation=False,
    )
    return pairs.groupby(primary, sort=False).response.agg(["sum", "size"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-offset", type=int, default=200)
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--wide-k", type=int, default=64)
    args = parser.parse_args()
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    if rank >= args.n_cases:
        return
    case = args.case_offset + rank
    os.makedirs(args.output_dir, exist_ok=True)
    output = os.path.join(args.output_dir, f"case{case}.feather")
    if os.path.exists(output):
        raise FileExistsError(f"refusing to overwrite {output}")

    columns = [
        "case", "input_index", "R_blend_truth", "R_blend_lsst_r_extnbr_v22",
    ]
    reference = pd.read_feather(args.reference, columns=columns)
    reference = reference.loc[reference.case == case].copy()
    if reference.empty or reference.input_index.duplicated().any():
        raise RuntimeError(f"case {case}: invalid reference anchors")
    ids = reference.input_index.to_numpy(np.int64)
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag="lsst_r_extnbr_v22",
        conditions=COND, device="cpu",
    )
    frame = input_frame(args.base, case, 0.05)
    narrow = predict_sum(predictor, frame, ids, 20).rename(
        columns={"sum": "prediction_k20", "size": "n_pairs_k20"}
    )
    wide = predict_sum(predictor, frame, ids, args.wide_k).rename(
        columns={"sum": "prediction_wide", "size": "n_pairs_wide"}
    )
    output_frame = (
        reference.set_index("input_index").join(narrow, how="left").join(wide, how="left")
        .reset_index()
    )
    missing_narrow = output_frame.prediction_k20.isna()
    if np.any(missing_narrow & (np.abs(output_frame.R_blend_lsst_r_extnbr_v22) > 2e-7)):
        raise RuntimeError(f"case {case}: nonzero reference lacks k20 pairs")
    for prediction, count in (
        ("prediction_k20", "n_pairs_k20"), ("prediction_wide", "n_pairs_wide")
    ):
        missing = output_frame[prediction].isna()
        output_frame.loc[missing, prediction] = 0.0
        output_frame.loc[missing, count] = 0
    replay = np.max(np.abs(
        output_frame.prediction_k20 - output_frame.R_blend_lsst_r_extnbr_v22
    ))
    if replay > 2e-7:
        raise RuntimeError(f"case {case}: k20 replay mismatch {replay:.3e}")
    output_frame["gap_k20"] = output_frame.prediction_k20 - output_frame.R_blend_truth
    output_frame["gap_wide"] = output_frame.prediction_wide - output_frame.R_blend_truth
    output_frame["wide_increment"] = output_frame.prediction_wide - output_frame.prediction_k20
    output_frame.to_feather(output)
    print(
        f"case {case}: anchors={len(output_frame):,} replay={replay:.3e} "
        f"n20={output_frame.n_pairs_k20.mean():.3f} "
        f"nwide={output_frame.n_pairs_wide.mean():.3f} "
        f"increment={output_frame.wide_increment.mean():+.6f}",
        flush=True,
    )
    print("ANCHOR_COHERENT_PAIR_CAP_RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
