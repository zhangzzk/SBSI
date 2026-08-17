#!/usr/bin/env python3
"""Truth-blind coherent-anchor scoring for pooled-scene corrections."""

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
from scripts.score_anchor_crossfit_sequential import (  # noqa: E402
    BASE_FEATURES,
    CONDITIONAL_FEATURES,
    RAW_FEATURES,
    RESCALE,
    block_base,
    sha256,
    strict_json,
)
from scripts.train_case100_100_pair_pooled_scene_moment import (  # noqa: E402
    FEATURES,
    add_proxy_feature,
)


def load_model(path: Path, expected_hash: str, n_jobs: int) -> xgb.Booster:
    if sha256(path) != expected_hash:
        raise RuntimeError(f"model hash mismatch: {path}")
    model = xgb.Booster({"device": "cpu", "n_jobs": n_jobs})
    model.load_model(path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-summary", required=True)
    parser.add_argument("--ordinary-summary", required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--g", type=float, default=0.02)
    parser.add_argument("--pair-prefix", default="pairs_renderer_v22")
    args = parser.parse_args()
    pooled_path = Path(args.pooled_summary).resolve()
    ordinary_path = Path(args.ordinary_summary).resolve()
    pooled = json.loads(pooled_path.read_text(encoding="utf-8"))
    ordinary = json.loads(ordinary_path.read_text(encoding="utf-8"))
    if pooled["features"]["correction"] != FEATURES:
        raise RuntimeError("pooled correction feature drift")
    if ordinary["features"]["correction"] != CONDITIONAL_FEATURES:
        raise RuntimeError("ordinary correction feature drift")
    if pooled["deployment_stack"]["base_sha256"] != ordinary[
        "deployment_stack"
    ]["base_sha256"]:
        raise RuntimeError("ordinary and pooled models do not share a base")
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    base = load_model(
        Path(pooled["deployment_stack"]["base_model"]),
        pooled["deployment_stack"]["base_sha256"],
        n_jobs,
    )
    ordinary_model = load_model(
        Path(ordinary["deployment_stack"]["correction_model"]),
        ordinary["deployment_stack"]["correction_sha256"],
        n_jobs,
    )
    pair_only_info = pooled["final_refit"]["models"]["pair_only_q"]
    selected_info = pooled["final_refit"]["models"]["selected"]
    pair_only_model = load_model(
        Path(pair_only_info["model"]), pair_only_info["model_sha256"], n_jobs
    )
    selected_model = load_model(
        Path(selected_info["model"]), selected_info["model_sha256"], n_jobs
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_feather = output_dir / f"case{args.case}.feather"
    output_json = output_dir / f"case{args.case}.json"
    if output_feather.exists() or output_json.exists():
        raise FileExistsError(f"refusing existing output for case {args.case}")
    base_dir = block_base(args.case)
    pair_path = base_dir / f"{args.pair_prefix}_case{args.case}.feather"
    anchor_path = base_dir / f"anchors_case{args.case}.feather"
    catalogue_path = Path(input_path(str(base_dir), args.case, +args.g))
    pairs = pd.read_feather(
        pair_path,
        columns=[
            "anchor_index", "secondary_index", "distance", "response", "n_pairs"
        ],
    ).sort_values(["anchor_index", "secondary_index"], kind="stable").reset_index(drop=True)
    anchors = pd.read_feather(anchor_path, columns=["index"])
    latent = pd.read_feather(
        catalogue_path,
        columns=["index_input", "Re_input", "r_input", "sersic_n_input"],
    ).set_index("index_input", verify_integrity=True)
    primary = latent.loc[pairs.anchor_index.to_numpy(np.int64)]
    secondary = latent.loc[pairs.secondary_index.to_numpy(np.int64)]
    raw_frame = pd.DataFrame({
        "Re_input_p": primary.Re_input.to_numpy(float),
        "Re_input_s": secondary.Re_input.to_numpy(float),
        "r_input_p": primary.r_input.to_numpy(float),
        "r_input_s": secondary.r_input.to_numpy(float),
        "sersic_n_input_p": primary.sersic_n_input.to_numpy(float),
        "sersic_n_input_s": secondary.sersic_n_input.to_numpy(float),
        "distance": pairs.distance.to_numpy(float),
    })
    raw = raw_frame[RAW_FEATURES].to_numpy(np.float32)
    scaled = data_utils.rescale(raw_frame.copy(), **RESCALE)[
        BASE_FEATURES
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
        raise RuntimeError(f"case {args.case}: multiplicity mismatch")

    target_mean = float(pooled["source"]["target_mean"])
    target_scale = float(pooled["source"]["target_scale"])
    pair_base = (
        base.inplace_predict(scaled).astype(np.float64) * target_scale + target_mean
    )
    scene_base = np.add.reduceat(pair_base, starts)
    original_features = np.empty(
        (len(pair_base), len(CONDITIONAL_FEATURES)), dtype=np.float32
    )
    original_features[:, :7] = scaled
    original_features[:, 7] = pair_base
    original_features[:, 8] = -0.4 * (raw[:, 3] - raw[:, 2])
    original_features[:, 9] = np.log10(raw[:, 1] / raw[:, 0])
    original_features[:, 10] = np.repeat(scene_base, counts)
    original_features[:, 11] = np.repeat(np.log1p(counts), counts)
    pair_q = np.power(10.0, original_features[:, 8].astype(float)) / np.square(
        raw[:, 6].astype(float)
    )
    scene_q = np.add.reduceat(pair_q, starts)
    logq = np.log10(scene_q)
    pooled_features = add_proxy_feature(
        original_features, np.repeat(logq, counts)
    )
    ordinary_delta = (
        ordinary_model.inplace_predict(original_features).astype(float) * target_scale
    )
    pair_only_delta = (
        pair_only_model.inplace_predict(pooled_features).astype(float) * target_scale
    )
    selected_delta = (
        selected_model.inplace_predict(pooled_features).astype(float) * target_scale
    )
    paired_anchor = anchor_index[starts]
    scored = pd.DataFrame({
        "input_index": paired_anchor,
        "n_pairs": counts.astype(np.int16),
        "log10_scene_proxy_q_d2": logq,
        "R_blend_v22_replay": np.add.reduceat(
            pairs.response.to_numpy(float), starts
        ),
        "R_blend_raw_base": scene_base,
        "R_blend_ordinary": scene_base + np.add.reduceat(ordinary_delta, starts),
        "R_blend_pair_only_q": scene_base + np.add.reduceat(pair_only_delta, starts),
        "R_blend_pooled_scene": scene_base + np.add.reduceat(selected_delta, starts),
    })
    output = anchors.rename(columns={"index": "input_index"}).merge(
        scored, on="input_index", how="left", validate="one_to_one"
    )
    no_pair = output.n_pairs.isna()
    finite_columns = [
        "R_blend_v22_replay", "R_blend_raw_base", "R_blend_ordinary",
        "R_blend_pair_only_q", "R_blend_pooled_scene",
    ]
    output.loc[no_pair, ["n_pairs", *finite_columns]] = 0.0
    output.loc[no_pair, "log10_scene_proxy_q_d2"] = -np.inf
    output["n_pairs"] = output.n_pairs.astype(np.int16)
    output.insert(0, "case", np.int16(args.case))
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
        "selected_lambda": float(pooled["selection"]["selected_lambda"]),
        "pooled_summary": str(pooled_path),
        "pooled_summary_sha256": sha256(pooled_path),
        "ordinary_summary": str(ordinary_path),
        "ordinary_summary_sha256": sha256(ordinary_path),
        "pair_manifest": str(pair_path),
        "renderer_catalogue": str(catalogue_path),
        "coherent_response_truth_opened": False,
        "constgold_opened": False,
    }
    strict_json(output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("ANCHOR_PAIR_POOLED_SCENE_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
