"""Score one fixed blending-response model on coherent anchor scenes."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd


BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu.inference import BlendingPredictor  # noqa: E402
from build_anchorblend_independent_response import COND, input_frame  # noqa: E402


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--reference-baseline-column",
        default="R_blend_lsst_r_extnbr_v22",
    )
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--reg-file", default=None)
    parser.add_argument("--expected-reg-sha256", default=None)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-offset", type=int, required=True)
    parser.add_argument("--n-cases", type=int, default=1)
    parser.add_argument("--sign", type=float, default=0.05)
    args = parser.parse_args()
    if args.reg_file is not None:
        if not os.path.isfile(args.reg_file):
            raise FileNotFoundError(args.reg_file)
        actual_hash = sha256(args.reg_file)
        if (
            args.expected_reg_sha256 is not None
            and actual_hash != args.expected_reg_sha256
        ):
            raise RuntimeError(
                f"regression model hash mismatch: {actual_hash} != "
                f"{args.expected_reg_sha256}"
            )
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    if rank >= args.n_cases:
        return
    case = args.case_offset + rank
    os.makedirs(args.output_dir, exist_ok=True)
    output = os.path.join(args.output_dir, f"case{case}.feather")
    if os.path.exists(output):
        raise FileExistsError(f"refusing to overwrite {output}")

    reference = pd.read_feather(args.reference)[
        ["case", "input_index", "R_blend_truth", args.reference_baseline_column]
    ].rename(columns={args.reference_baseline_column: "reference_baseline"})
    reference = reference.loc[reference.case == case].copy()
    if reference.empty or reference.input_index.duplicated().any():
        raise RuntimeError(f"case {case}: invalid reference")
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag=args.model_tag,
        conditions=COND, device="cpu", reg_file=args.reg_file,
        load_self=False,
    )
    frame = input_frame(args.base, case, args.sign)
    pairs = predictor.predict_response(frame, frame)
    primary = next(
        column for column in pairs if column.startswith("index") and column.endswith("_p")
    )
    pairs = pairs.loc[pairs[primary].isin(reference.input_index)].copy()
    aggregate = pairs.groupby(primary, sort=False).response.agg(["sum", "size"])
    aggregate.columns = ["prediction_model", "n_pairs"]
    aggregate.index.name = "input_index"
    scored = reference.set_index("input_index").join(aggregate, how="left").reset_index()
    missing = scored.prediction_model.isna()
    scored.loc[missing, ["prediction_model", "n_pairs"]] = 0.0
    scored["prediction_baseline"] = scored.reference_baseline
    scored["gap_baseline"] = scored.prediction_baseline - scored.R_blend_truth
    scored["gap_model"] = scored.prediction_model - scored.R_blend_truth
    scored["model_minus_baseline"] = scored.prediction_model - scored.prediction_baseline
    scored["model_tag"] = args.candidate_label or args.model_tag
    scored["model_source_tag"] = args.model_tag
    scored["model_sha256"] = (
        sha256(args.reg_file) if args.reg_file is not None else "metadata-default"
    )
    scored.to_feather(output)
    print(
        f"case {case}: anchors={len(scored):,} baseline={scored.prediction_baseline.mean():+.6f} "
        f"model={scored.prediction_model.mean():+.6f} truth={scored.R_blend_truth.mean():+.6f}",
        flush=True,
    )
    print("ANCHOR_RESPONSE_MODEL_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
