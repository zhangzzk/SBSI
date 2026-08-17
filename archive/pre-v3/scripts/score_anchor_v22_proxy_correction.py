#!/usr/bin/env python3
"""Score coherent anchors with a frozen V2.2 + physical-proxy correction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
BLENDEMU = Path("/home/z/Zekang.Zhang/blendemu")
for path in (ROOT, BLENDEMU):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from blendemu import data_utils  # noqa: E402
from scripts.prepare_v22_matched_decomposition import input_path  # noqa: E402
from scripts.v22_grouped_rscene_common import sha256, strict_json  # noqa: E402
from scripts.v22_proxy_correction_common import (  # noqa: E402
    CONDITIONAL_FEATURES,
    PROXY_DEFINITION,
    proxy_conditional_matrix,
)


RAW_FEATURES = [
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance",
]
MODEL_FEATURES = CONDITIONAL_FEATURES[:7]
RESCALE = {
    "pixel_rms": 0.312,
    "pixel_size": 0.2,
    "zero_mag": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
}


def block_base(case: int) -> Path:
    if case < 400 or case > 899:
        raise ValueError("coherent-anchor case must lie in 400--899")
    start = 100 * (case // 100)
    return Path(
        "/project/ls-gruen/users/zekang.zhang/"
        f"lsst_sims_fs2_25876_anchorblend_g002_c{start}-{start + 99}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-summary", required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--g", type=float, default=0.02)
    parser.add_argument("--pair-prefix", default="pairs_renderer_v22")
    args = parser.parse_args()

    summary_path = Path(args.model_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_feather = output_dir / f"case{args.case}.feather"
    output_json = output_dir / f"case{args.case}.json"
    if output_feather.exists() or output_json.exists():
        raise FileExistsError(f"refusing existing output for case {args.case}")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary["feature_names"] != CONDITIONAL_FEATURES:
        raise RuntimeError("proxy-correction feature definition drifted")
    if summary["training"]["case_window"] != [40, 199]:
        raise RuntimeError("anchor scoring requires the frozen c40--199 fit")
    model_path = Path(summary["model"])
    if sha256(model_path) != summary["model_sha256"]:
        raise RuntimeError("proxy-correction model hash differs from summary")
    booster = xgb.Booster({
        "device": "cpu",
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    })
    booster.load_model(model_path)

    base = block_base(args.case)
    pair_path = base / f"{args.pair_prefix}_case{args.case}.feather"
    anchor_path = base / f"anchors_case{args.case}.feather"
    catalogue_path = Path(input_path(str(base), args.case, +args.g))
    pairs = pd.read_feather(
        pair_path,
        columns=[
            "anchor_index", "secondary_index", "distance", "response", "n_pairs"
        ],
    ).sort_values(
        ["anchor_index", "secondary_index"], kind="stable"
    ).reset_index(drop=True)
    if pairs.empty or pairs.duplicated(["anchor_index", "secondary_index"]).any():
        raise RuntimeError(f"case {args.case}: empty or duplicate pair manifest")
    anchors = pd.read_feather(anchor_path, columns=["index"])
    if anchors["index"].duplicated().any():
        raise RuntimeError(f"case {args.case}: duplicate anchor index")
    latent = pd.read_feather(
        catalogue_path,
        columns=["index_input", "Re_input", "r_input", "sersic_n_input"],
    ).set_index("index_input", verify_integrity=True)
    primary = latent.loc[pairs.anchor_index.to_numpy(np.int64)]
    secondary = latent.loc[pairs.secondary_index.to_numpy(np.int64)]
    raw_frame = pd.DataFrame({
        "Re_input_p": primary.Re_input.to_numpy(np.float64),
        "Re_input_s": secondary.Re_input.to_numpy(np.float64),
        "r_input_p": primary.r_input.to_numpy(np.float64),
        "r_input_s": secondary.r_input.to_numpy(np.float64),
        "sersic_n_input_p": primary.sersic_n_input.to_numpy(np.float64),
        "sersic_n_input_s": secondary.sersic_n_input.to_numpy(np.float64),
        "distance": pairs.distance.to_numpy(np.float64),
    })
    raw = raw_frame[RAW_FEATURES].to_numpy(np.float32)
    scaled = data_utils.rescale(raw_frame.copy(), **RESCALE)[
        MODEL_FEATURES
    ].to_numpy(np.float32)

    anchor_index = pairs.anchor_index.to_numpy(np.int64)
    change = np.empty(len(pairs), dtype=bool)
    change[0] = True
    change[1:] = anchor_index[1:] != anchor_index[:-1]
    starts = np.flatnonzero(change)
    counts = np.diff(np.r_[starts, len(pairs)])
    if not np.array_equal(
        counts.astype(np.int64), pairs.n_pairs.to_numpy(np.int64)[starts]
    ):
        raise RuntimeError(f"case {args.case}: stored pair multiplicity differs")
    pair_prediction = pairs.response.to_numpy(np.float32)
    scene_prediction = np.add.reduceat(
        pair_prediction.astype(np.float64), starts
    )
    pair_proxy = np.power(
        10.0, -0.4 * (raw[:, 3].astype(float) - raw[:, 2].astype(float))
    ) / np.square(raw[:, 6].astype(float))
    scene_proxy = np.add.reduceat(pair_proxy, starts)
    if np.any(scene_proxy <= 0.0) or not np.isfinite(scene_proxy).all():
        raise RuntimeError(f"case {args.case}: invalid physical scene proxy")
    log_proxy = np.log10(scene_proxy)
    features = proxy_conditional_matrix(
        scaled, raw, pair_prediction,
        np.repeat(log_proxy, counts), np.repeat(counts, counts),
    )
    correction = booster.inplace_predict(features).astype(np.float64)
    correction *= float(summary["target_scale"])
    correction_sum = np.add.reduceat(correction, starts)
    corrected_sum = scene_prediction + correction_sum
    paired_anchor = anchor_index[starts]

    scored = pd.DataFrame({
        "input_index": paired_anchor,
        "n_pairs": counts.astype(np.int16),
        "scene_proxy_q_d2": scene_proxy,
        "log10_scene_proxy_q_d2": log_proxy,
        "R_blend_v22_replay": scene_prediction,
        "proxy_pair_correction_sum": correction_sum,
        "R_blend_proxy_corrected": corrected_sum,
    })
    output = anchors.rename(columns={"index": "input_index"}).merge(
        scored, on="input_index", how="left", validate="one_to_one"
    )
    no_pair = output.n_pairs.isna()
    output.loc[no_pair, [
        "n_pairs", "scene_proxy_q_d2", "R_blend_v22_replay",
        "proxy_pair_correction_sum", "R_blend_proxy_corrected",
    ]] = 0.0
    # One anchor over all 500 cases has no deployed pair.  Its correction is
    # identically zero; use -inf only as an explicit no-pair proxy sentinel.
    output.loc[no_pair, "log10_scene_proxy_q_d2"] = -np.inf
    output["n_pairs"] = output.n_pairs.astype(np.int16)
    output.insert(0, "case", np.int16(args.case))
    finite_columns = [
        "scene_proxy_q_d2", "R_blend_v22_replay",
        "proxy_pair_correction_sum", "R_blend_proxy_corrected",
    ]
    if not np.isfinite(output[finite_columns].to_numpy(float)).all():
        raise RuntimeError(f"case {args.case}: non-finite anchor score")

    temporary = output_feather.with_name(
        f".{output_feather.stem}.{os.getpid()}.tmp.feather"
    )
    output.to_feather(temporary)
    os.replace(temporary, output_feather)
    payload = {
        "case": args.case,
        "n_anchors": int(len(output)),
        "n_paired_anchors": int(len(scored)),
        "n_pairs": int(len(pairs)),
        "proxy_definition": PROXY_DEFINITION,
        "pair_correction_mean": float(correction.mean()),
        "scene_correction_mean": float(correction_sum.mean()),
        "model_summary": str(summary_path),
        "model_summary_sha256": sha256(summary_path),
        "model_sha256": summary["model_sha256"],
        "pair_manifest": str(pair_path),
        "renderer_catalogue": str(catalogue_path),
        "coherent_response_truth_opened": False,
        "constgold_opened": False,
    }
    strict_json(output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("ANCHOR_V22_PROXY_CORRECTION_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
