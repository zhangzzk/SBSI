#!/usr/bin/env python3
"""Apply the frozen scene-informed pair correction to ConstGold V2.2 pairs.

For every supported primary-neighbour pair, this script replays the deployed
V2.2 response, builds the same pair-plus-scene coordinate used by the frozen
second-stage model, and adds its predicted residual.  Corrected pairs are then
summed per primary to form an evaluation-only ConstGold ``R_blend`` lookup.

ConstGold response truth is never read here.  The correction model, its feature
definition, and its strength are immutable inputs fixed from half-shear data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
BLENDEMU_ROOT = Path("/home/z/Zekang.Zhang/blendemu")
for path in (ROOT, BLENDEMU_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from blendemu.inference import BlendingPredictor  # noqa: E402
from sbs_shear.paths import CONST_SIM_DIR  # noqa: E402
from scripts.build_v22_constgold_pair_features import input_feather  # noqa: E402
from scripts.v22_grouped_rscene_common import (  # noqa: E402
    CONDITIONAL_FEATURES,
    SOURCE_TAG,
    conditional_matrix,
    physical_correction,
    sha256,
    strict_json,
)


MODEL_DIR = BLENDEMU_ROOT / "models"
KEY = ["case", "input_index"]
PRIMARY_COLUMN = "index_input_p"
SECONDARY_COLUMN = "index_input_s"
RAW_FEATURES = [
    "Re_input_p",
    "Re_input_s",
    "r_input_p",
    "r_input_s",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance",
]
SCALED_FEATURES = CONDITIONAL_FEATURES[:7]
CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}


def quantiles(values: np.ndarray) -> dict[str, float]:
    """Compact finite distribution summary for audit metadata."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("quantile input must be non-empty and finite")
    probabilities = (0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0)
    result = np.quantile(values, probabilities)
    return {
        f"q{int(round(probability * 1000)):04d}": float(value)
        for probability, value in zip(probabilities, result)
    }


def aggregate_pair_corrections(
    primary: np.ndarray,
    secondary: np.ndarray,
    pair_prediction: np.ndarray,
    pair_correction: np.ndarray,
) -> pd.DataFrame:
    """Sum baseline predictions and learned corrections per sorted primary."""
    primary = np.asarray(primary, dtype=np.int64)
    secondary = np.asarray(secondary, dtype=np.int64)
    pair_prediction = np.asarray(pair_prediction, dtype=np.float64)
    pair_correction = np.asarray(pair_correction, dtype=np.float64)
    if not (
        primary.shape == secondary.shape == pair_prediction.shape
        == pair_correction.shape
    ):
        raise ValueError("pair aggregation arrays have different shapes")
    if len(primary) == 0:
        raise ValueError("cannot aggregate an empty supported-pair table")
    if np.any(primary[1:] < primary[:-1]):
        raise ValueError("pairs must be sorted by primary")
    duplicate = (
        (primary[1:] == primary[:-1])
        & (secondary[1:] == secondary[:-1])
    )
    if np.any(duplicate):
        raise RuntimeError("duplicate primary-secondary pair")
    if not np.isfinite(pair_prediction).all() \
            or not np.isfinite(pair_correction).all():
        raise RuntimeError("non-finite pair prediction or correction")

    change = np.empty(len(primary), dtype=bool)
    change[0] = True
    change[1:] = primary[1:] != primary[:-1]
    starts = np.flatnonzero(change)
    counts = np.diff(np.r_[starts, len(primary)])
    if int(counts.max()) >= 256:
        raise RuntimeError("pair multiplicity exceeds uint8 scene feature")
    baseline = np.add.reduceat(pair_prediction, starts)
    correction = np.add.reduceat(pair_correction, starts)
    return pd.DataFrame({
        "input_index": primary[starts],
        "n_pairs": counts.astype(np.int16),
        "R_blend_v22": baseline,
        "grouped_pair_correction_sum": correction,
        "R_blend": baseline + correction,
    })


def scene_context(
    primary: np.ndarray, pair_prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return scene starts/counts and row-repeated frozen scene coordinates."""
    primary = np.asarray(primary, dtype=np.int64)
    pair_prediction = np.asarray(pair_prediction, dtype=np.float64)
    if len(primary) == 0 or pair_prediction.shape != primary.shape:
        raise ValueError("invalid primary/prediction arrays")
    if np.any(primary[1:] < primary[:-1]):
        raise ValueError("pairs must be sorted by primary")
    change = np.empty(len(primary), dtype=bool)
    change[0] = True
    change[1:] = primary[1:] != primary[:-1]
    starts = np.flatnonzero(change)
    counts = np.diff(np.r_[starts, len(primary)])
    if np.any(counts <= 0) or int(counts.max()) >= 256:
        raise RuntimeError("invalid scene multiplicity")
    # Match the training cache: sum in float64, then store the scene coordinate
    # in float32 before repeating it over pair rows.
    scene_sum = np.add.reduceat(pair_prediction, starts).astype(np.float32)
    scene_rows = np.repeat(scene_sum, counts)
    count_rows = np.repeat(counts.astype(np.uint8), counts)
    return starts, counts, scene_rows, count_rows


def selection_summary(frame: pd.DataFrame, selected: np.ndarray) -> dict[str, Any]:
    local = frame.loc[np.asarray(selected, dtype=bool)]
    if local.empty:
        raise RuntimeError("empty ConstGold scene summary selection")
    return {
        "n_scenes": int(len(local)),
        "mean_n_pairs": float(local.n_pairs.mean()),
        "mean_v22_scene_prediction": float(local.R_blend_v22.mean()),
        "mean_scene_correction": float(local.grouped_pair_correction_sum.mean()),
        "mean_corrected_scene_prediction": float(local.R_blend.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--model-summary", required=True)
    parser.add_argument("--reference-lookup", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--base", default=CONST_SIM_DIR)
    parser.add_argument("--sign", default="0.02")
    parser.add_argument("--tag", default=SOURCE_TAG)
    parser.add_argument("--replay-max-atol", type=float, default=1.0e-6)
    parser.add_argument("--replay-mean-atol", type=float, default=1.0e-8)
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    summary_output = Path(args.summary_json).resolve()
    for output in (output_path, summary_output):
        if output.exists():
            raise FileExistsError(f"refusing existing output {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    cases = sorted(set(int(case) for case in args.cases))
    if cases != list(range(cases[0], cases[-1] + 1)):
        raise ValueError("ConstGold cases must form one contiguous window")

    model_summary_path = Path(args.model_summary).resolve()
    with model_summary_path.open(encoding="utf-8") as handle:
        model_summary = json.load(handle)
    if model_summary["feature_names"] != CONDITIONAL_FEATURES:
        raise RuntimeError("frozen correction feature definition drifted")
    if model_summary["training"]["case_window"] != [40, 199]:
        raise RuntimeError("ConstGold transfer requires the final c40--199 fit")
    if model_summary["protocol"]["constgold_opened"] != 0:
        raise RuntimeError("frozen model provenance claims ConstGold was opened")
    model_path = Path(model_summary["model"]).resolve()
    if sha256(model_path) != model_summary["model_sha256"]:
        raise RuntimeError("frozen correction-model hash differs from summary")
    booster = xgb.Booster({
        "device": "cpu",
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    })
    booster.load_model(model_path)

    predictor = BlendingPredictor.load(
        str(MODEL_DIR), tag=args.tag, conditions=CONDITIONS, device="cpu"
    )
    cuts, r_max, k = predictor._select("regression")
    if args.tag != SOURCE_TAG:
        raise RuntimeError(f"expected frozen source tag {SOURCE_TAG}, got {args.tag}")
    if not np.isclose(float(r_max), 10.0) or int(k) != 20:
        raise RuntimeError(f"expected V2.2 r_max=10/k=20, got {r_max}/{k}")
    if predictor.bst_reg.feature_names != SCALED_FEATURES:
        raise RuntimeError(
            "V2.2 regression feature order differs from grouped-model source"
        )

    reference = pf.read_table(
        args.reference_lookup, columns=[*KEY, "R_blend"]
    ).to_pandas()
    reference = reference.loc[reference.case.isin(cases)].copy()
    if reference.empty or reference.duplicated(KEY).any():
        raise RuntimeError("empty or duplicate reference V2.2 lookup")

    parts: list[pd.DataFrame] = []
    case_audit: list[dict[str, Any]] = []
    pair_corrections: list[np.ndarray] = []
    for case in cases:
        path = input_feather(case, args.sign, args.base)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        field = pf.read_table(path).to_pandas().rename(
            columns=lambda name: name.replace("_input", "")
        )
        if field["index"].duplicated().any():
            raise RuntimeError(f"case {case}: duplicate latent input index")
        pairs = predictor.predict_response(field, field)
        required = {
            PRIMARY_COLUMN,
            SECONDARY_COLUMN,
            "response",
            *RAW_FEATURES,
            *SCALED_FEATURES,
        }
        if missing := required - set(pairs):
            raise KeyError(f"case {case}: replay lacks {sorted(missing)}")
        pairs = pairs.sort_values(
            [PRIMARY_COLUMN, SECONDARY_COLUMN], kind="stable"
        ).reset_index(drop=True)
        primary = pairs[PRIMARY_COLUMN].to_numpy(np.int64)
        secondary = pairs[SECONDARY_COLUMN].to_numpy(np.int64)
        pair_prediction = pairs.response.to_numpy(np.float64)
        _, counts, scene_rows, count_rows = scene_context(
            primary, pair_prediction
        )
        features = conditional_matrix(
            pairs[SCALED_FEATURES].to_numpy(np.float32),
            pairs[RAW_FEATURES].to_numpy(np.float32),
            pair_prediction.astype(np.float32),
            scene_rows,
            count_rows,
            slice(None),
        )
        correction = physical_correction(
            booster, features, float(model_summary["target_scale"])
        )
        scored = aggregate_pair_corrections(
            primary, secondary, pair_prediction, correction
        )
        scored.insert(0, "case", np.int16(case))

        expected = reference.loc[reference.case == case].sort_values(
            "input_index", kind="stable"
        )
        actual = scored.sort_values("input_index", kind="stable")
        if not np.array_equal(
            expected.input_index.to_numpy(np.int64),
            actual.input_index.to_numpy(np.int64),
        ):
            raise RuntimeError(f"case {case}: replay/reference lookup keys differ")
        difference = (
            actual.R_blend_v22.to_numpy(np.float64)
            - expected.R_blend.to_numpy(np.float64)
        )
        replay_max = float(np.max(np.abs(difference)))
        replay_mean = float(np.mean(np.abs(difference)))
        if replay_max > args.replay_max_atol or replay_mean > args.replay_mean_atol:
            raise RuntimeError(
                f"case {case}: V2.2 lookup replay differs, "
                f"max={replay_max:.3e} meanabs={replay_mean:.3e}"
            )
        parts.append(scored)
        pair_corrections.append(correction)
        case_audit.append({
            "case": int(case),
            "n_pairs": int(len(pairs)),
            "n_scenes": int(len(scored)),
            "mean_n_pairs": float(counts.mean()),
            "mean_pair_correction": float(correction.mean()),
            "mean_scene_correction": float(
                scored.grouped_pair_correction_sum.mean()
            ),
            "replay_max_abs": replay_max,
            "replay_mean_abs": replay_mean,
        })
        print(
            f"case {case}: {len(pairs):,} pairs, {len(scored):,} scenes, "
            f"<dR_pair>={correction.mean():+.6f}, "
            f"<dR_scene>={scored.grouped_pair_correction_sum.mean():+.6f}, "
            f"replay={replay_max:.2e}/{replay_mean:.2e}",
            flush=True,
        )
        del field, pairs, features, correction

    output = pd.concat(parts, ignore_index=True)
    if output.duplicated(KEY).any() or not np.isfinite(
        output.drop(columns=KEY).to_numpy(float)
    ).all():
        raise RuntimeError("final corrected lookup has duplicate keys or non-finite values")
    all_pair_correction = np.concatenate(pair_corrections)
    selections = {
        "all_supported": np.ones(len(output), dtype=bool),
        "outside_le_0p1": output.R_blend_v22.to_numpy(float) <= 0.1,
        "tail_gt_0p1": output.R_blend_v22.to_numpy(float) > 0.1,
        "tail_gt_0p2": output.R_blend_v22.to_numpy(float) > 0.2,
    }
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.feather"
    )
    output.to_feather(temporary)
    os.replace(temporary, output_path)

    payload = {
        "schema_version": 1,
        "design": (
            "evaluation-only application of frozen half-shear pair-residual "
            "model to V2.2 ConstGold pairs with full-scene context"
        ),
        "case_window": [cases[0], cases[-1]],
        "n_cases": len(cases),
        "n_supported_pairs": int(sum(item["n_pairs"] for item in case_audit)),
        "n_supported_scenes": int(len(output)),
        "source_tag": args.tag,
        "source_regression_cuts": cuts,
        "source_r_max_arcsec": float(r_max),
        "source_k": int(k),
        "model_summary": str(model_summary_path),
        "model_summary_sha256": sha256(model_summary_path),
        "model": str(model_path),
        "model_sha256": model_summary["model_sha256"],
        "feature_names": CONDITIONAL_FEATURES,
        "target_scale": float(model_summary["target_scale"]),
        "reference_lookup": str(Path(args.reference_lookup).resolve()),
        "output_lookup": str(output_path),
        "pair_correction": {
            "mean": float(all_pair_correction.mean()),
            "std": float(all_pair_correction.std(ddof=1)),
            "quantiles": quantiles(all_pair_correction),
        },
        "scene_correction": {
            "mean": float(output.grouped_pair_correction_sum.mean()),
            "std": float(output.grouped_pair_correction_sum.std(ddof=1)),
            "quantiles": quantiles(
                output.grouped_pair_correction_sum.to_numpy(float)
            ),
        },
        "scene_selections": {
            name: selection_summary(output, selected)
            for name, selected in selections.items()
        },
        "case_audit": case_audit,
        "firewall": {
            "constgold_response_truth_read": False,
            "constgold_used_for_training_tuning_or_selection": False,
            "correction_fixed_before_constgold_evaluation": True,
        },
    }
    strict_json(summary_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_GROUPED_RSCENE_CONSTGOLD_LOOKUP_DONE", flush=True)


if __name__ == "__main__":
    main()
