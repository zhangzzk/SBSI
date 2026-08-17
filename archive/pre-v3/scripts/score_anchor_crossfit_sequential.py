#!/usr/bin/env python3
"""Score coherent-anchor pair manifests with two frozen sequential models.

Only model inputs and the previously deployed V2.2 pair predictions are read.
Coherent-anchor response truth is deliberately not opened by this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
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
from scripts.tune_sequential_rblend_optuna import (  # noqa: E402
    BASE_FEATURES,
    CONDITIONAL_FEATURES,
)


RAW_FEATURES = [
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance",
]
RESCALE = {
    "pixel_rms": 0.312,
    "pixel_size": 0.2,
    "zero_mag": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def block_base(case: int) -> Path:
    if case < 400 or case > 899:
        raise ValueError("coherent-anchor case must lie in 400--899")
    start = 100 * (case // 100)
    return Path(
        "/project/ls-gruen/users/zekang.zhang/"
        f"lsst_sims_fs2_25876_anchorblend_g002_c{start}-{start + 99}"
    )


def load_stack(summary_path: Path, n_jobs: int) -> tuple[dict, xgb.Booster, xgb.Booster]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["features"]["base"] != BASE_FEATURES:
        raise RuntimeError(f"base feature drift in {summary_path}")
    if summary["features"]["correction"] != CONDITIONAL_FEATURES:
        raise RuntimeError(f"correction feature drift in {summary_path}")
    if "crossfit" in summary and summary["crossfit"]["n_cases"] != 200:
        raise RuntimeError(f"expected all-200 crossfit in {summary_path}")
    deployment = summary.get("deployment_fit_all_200_cases_after_selection")
    if deployment is None:
        deployment = summary.get("deployment_fit_all_200_cases")
    if deployment is None:
        deployment = summary.get("deployment_stack")
    if deployment is None:
        raise RuntimeError(f"no deployment stack in {summary_path}")
    if "target_mean" not in summary["source"]:
        cache_value = summary["source"].get("label_cache")
        if cache_value is None:
            raise RuntimeError(f"no target standardization source in {summary_path}")
        cache_metadata = json.loads(
            (Path(cache_value) / "metadata.json").read_text(encoding="utf-8")
        )
        standardization = cache_metadata["source_standardization"]
        summary["source"]["target_mean"] = float(standardization["mean"])
        summary["source"]["target_scale"] = float(standardization["std"])
    base_path = Path(deployment["base_model"])
    correction_path = Path(deployment["correction_model"])
    if sha256(base_path) != deployment["base_sha256"]:
        raise RuntimeError(f"base-model hash mismatch in {summary_path}")
    if sha256(correction_path) != deployment["correction_sha256"]:
        raise RuntimeError(f"correction-model hash mismatch in {summary_path}")
    options = {"device": "cpu", "n_jobs": n_jobs}
    base = xgb.Booster(options)
    base.load_model(base_path)
    correction = xgb.Booster(options)
    correction.load_model(correction_path)
    return summary, base, correction


def stack_prediction(
    summary: dict,
    base: xgb.Booster,
    correction: xgb.Booster,
    scaled: np.ndarray,
    raw: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_mean = float(summary["source"]["target_mean"])
    target_scale = float(summary["source"]["target_scale"])
    dbase = xgb.DMatrix(scaled, feature_names=BASE_FEATURES)
    pair_base = (
        base.predict(dbase).astype(np.float64) * target_scale + target_mean
    )
    scene_base = np.add.reduceat(pair_base, starts)
    features = np.empty(
        (len(pair_base), len(CONDITIONAL_FEATURES)), dtype=np.float32
    )
    features[:, :7] = scaled
    features[:, 7] = pair_base.astype(np.float32)
    features[:, 8] = -0.4 * (raw[:, 3] - raw[:, 2])
    features[:, 9] = np.log10(raw[:, 1] / raw[:, 0])
    features[:, 10] = np.repeat(scene_base, counts).astype(np.float32)
    features[:, 11] = np.repeat(
        np.log1p(counts.astype(np.float32)), counts
    )
    dcorrection = xgb.DMatrix(features, feature_names=CONDITIONAL_FEATURES)
    # The correction target was standardized once as
    # (label - base_prediction) / target_scale.  Convert it back exactly once,
    # matching fit_crossfit and avoiding the double-scale typo in the original
    # deployment-only diagnostic.
    pair_correction = (
        correction.predict(dcorrection).astype(np.float64) * target_scale
    )
    scene_correction = np.add.reduceat(pair_correction, starts)
    return scene_base, scene_correction, scene_base + scene_correction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-summary")
    parser.add_argument("--pair-scene-summary")
    parser.add_argument(
        "--stack-summary",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "score an explicitly named sequential stack; repeat for multiple "
            "stacks. If omitted, --pair-summary and --pair-scene-summary are "
            "required for the historical two-stack behavior"
        ),
    )
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--g", type=float, default=0.02)
    parser.add_argument("--pair-prefix", default="pairs_renderer_v22")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_feather = output_dir / f"case{args.case}.feather"
    output_json = output_dir / f"case{args.case}.json"
    if output_feather.exists() or output_json.exists():
        raise FileExistsError(f"refusing existing output for case {args.case}")

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    if args.stack_summary:
        if args.pair_summary is not None or args.pair_scene_summary is not None:
            parser.error(
                "use either --stack-summary or the historical pair arguments"
            )
        summaries = {}
        for specification in args.stack_summary:
            if "=" not in specification:
                parser.error("--stack-summary must be NAME=PATH")
            name, path = specification.split("=", 1)
            if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                parser.error(f"invalid stack name {name!r}")
            if name in summaries:
                parser.error(f"duplicate stack name {name!r}")
            summaries[name] = Path(path).resolve()
    else:
        if args.pair_summary is None or args.pair_scene_summary is None:
            parser.error(
                "--pair-summary and --pair-scene-summary are required when "
                "--stack-summary is omitted"
            )
        summaries = {
            "pair": Path(args.pair_summary).resolve(),
            "pair_scene": Path(args.pair_scene_summary).resolve(),
        }
    stacks = {
        name: load_stack(path, n_jobs) for name, path in summaries.items()
    }

    base_dir = block_base(args.case)
    pair_path = base_dir / f"{args.pair_prefix}_case{args.case}.feather"
    anchor_path = base_dir / f"anchors_case{args.case}.feather"
    catalogue_path = Path(input_path(str(base_dir), args.case, +args.g))
    pairs = pd.read_feather(
        pair_path,
        columns=[
            "anchor_index", "secondary_index", "distance", "response", "n_pairs"
        ],
    ).sort_values(
        ["anchor_index", "secondary_index"], kind="stable"
    ).reset_index(drop=True)
    if pairs.empty or pairs.duplicated(
        ["anchor_index", "secondary_index"]
    ).any():
        raise RuntimeError(f"case {args.case}: empty or duplicate pair manifest")
    anchors = pd.read_feather(anchor_path, columns=["index"])
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
    if np.any(raw[:, :2] <= 0.0):
        raise RuntimeError(f"case {args.case}: non-positive size")
    scaled = data_utils.rescale(
        raw_frame.copy(), **RESCALE
    )[BASE_FEATURES].to_numpy(np.float32)

    anchor_index = pairs.anchor_index.to_numpy(np.int64)
    change = np.empty(len(pairs), dtype=bool)
    change[0] = True
    change[1:] = anchor_index[1:] != anchor_index[:-1]
    starts = np.flatnonzero(change)
    counts = np.diff(np.r_[starts, len(pairs)])
    if not np.array_equal(
        counts.astype(np.int64), pairs.n_pairs.to_numpy(np.int64)[starts]
    ):
        raise RuntimeError(f"case {args.case}: stored multiplicity differs")
    paired_anchor = anchor_index[starts]
    v22_scene = np.add.reduceat(
        pairs.response.to_numpy(np.float64), starts
    )
    scored = pd.DataFrame({
        "input_index": paired_anchor,
        "n_pairs": counts.astype(np.int16),
        "R_blend_v22_replay": v22_scene,
    })
    for name, (summary, base, correction) in stacks.items():
        scene_base, scene_correction, scene_combined = stack_prediction(
            summary, base, correction, scaled, raw, starts, counts
        )
        scored[f"R_blend_{name}_base"] = scene_base
        scored[f"R_blend_{name}_correction"] = scene_correction
        scored[f"R_blend_{name}"] = scene_combined

    output = anchors.rename(columns={"index": "input_index"}).merge(
        scored, on="input_index", how="left", validate="one_to_one"
    )
    no_pair = output.n_pairs.isna()
    score_columns = [
        column for column in output.columns
        if column not in ("input_index", "n_pairs")
    ]
    output.loc[no_pair, ["n_pairs", *score_columns]] = 0.0
    output["n_pairs"] = output.n_pairs.astype(np.int16)
    output.insert(0, "case", np.int16(args.case))
    if not np.isfinite(
        output.drop(columns=["case", "input_index"]).to_numpy(float)
    ).all():
        raise RuntimeError(f"case {args.case}: non-finite score")

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
        "mean_pairs_per_paired_anchor": float(counts.mean()),
        "summaries": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "objective_variant": stacks[name][0].get(
                    "objective_variant", "fixed_no_tuning"
                ),
            }
            for name, path in summaries.items()
        },
        "correction_physical_scaling": "one target_scale factor",
        "pair_manifest": str(pair_path),
        "renderer_catalogue": str(catalogue_path),
        "coherent_response_truth_opened": False,
        "constgold_opened": False,
    }
    strict_json(output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("ANCHOR_CROSSFIT_SEQUENTIAL_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
