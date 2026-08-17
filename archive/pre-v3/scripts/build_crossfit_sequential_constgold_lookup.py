#!/usr/bin/env python3
"""Apply a frozen sequential base+correction R_blend model to ConstGold.

This is an evaluation-only pair replay.  ConstGold response truth is not read;
the output is a per-primary R_blend lookup for the existing swap evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as pf


ROOT = Path(__file__).resolve().parents[1]
BLENDEMU_ROOT = Path("/home/z/Zekang.Zhang/blendemu")
for path in (ROOT, BLENDEMU_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from blendemu.inference import BlendingPredictor  # noqa: E402
from sbs_shear.paths import CONST_SIM_DIR  # noqa: E402
from scripts.build_v22_constgold_pair_features import input_feather  # noqa: E402
from scripts.build_v22_grouped_rscene_constgold_lookup import (  # noqa: E402
    CONDITIONS,
    KEY,
    MODEL_DIR,
    PRIMARY_COLUMN,
    RAW_FEATURES,
    SCALED_FEATURES,
    SECONDARY_COLUMN,
    scene_context,
)
from scripts.score_anchor_crossfit_sequential import (  # noqa: E402
    load_stack,
    sha256,
    stack_prediction,
    strict_json,
)
from scripts.v22_grouped_rscene_common import SOURCE_TAG  # noqa: E402


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    probabilities = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
    quantiles = np.quantile(values, probabilities)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        **{
            f"q{int(round(probability * 100)):03d}": float(value)
            for probability, value in zip(probabilities, quantiles)
        },
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
        raise ValueError("ConstGold cases must form a contiguous window")

    model_summary_path = Path(args.model_summary).resolve()
    summary, base_booster, correction_booster = load_stack(
        model_summary_path,
        int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    )
    objective_variant = summary["objective_variant"]
    if objective_variant not in {"pair", "pair_scene"}:
        raise RuntimeError(
            f"unsupported sequential objective variant {objective_variant!r}"
        )
    if summary["protocol"]["constgold_opened"] not in (False, 0):
        raise RuntimeError("model provenance does not preserve ConstGold firewall")

    predictor = BlendingPredictor.load(
        str(MODEL_DIR), tag=args.tag, conditions=CONDITIONS, device="cpu"
    )
    cuts, r_max, k = predictor._select("regression")
    if args.tag != SOURCE_TAG or not np.isclose(float(r_max), 10.0) or int(k) != 20:
        raise RuntimeError("frozen V2.2 pair replay configuration drifted")
    if predictor.bst_reg.feature_names != SCALED_FEATURES:
        raise RuntimeError("V2.2 pair feature order drifted")

    reference = pf.read_table(
        args.reference_lookup, columns=[*KEY, "R_blend"]
    ).to_pandas()
    reference = reference.loc[reference.case.isin(cases)].copy()
    if reference.empty or reference.duplicated(KEY).any():
        raise RuntimeError("empty or duplicate reference V2.2 lookup")

    parts: list[pd.DataFrame] = []
    audits: list[dict] = []
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
            PRIMARY_COLUMN, SECONDARY_COLUMN, "response",
            *RAW_FEATURES, *SCALED_FEATURES,
        }
        if missing := required - set(pairs):
            raise KeyError(f"case {case}: pair replay lacks {sorted(missing)}")
        pairs = pairs.sort_values(
            [PRIMARY_COLUMN, SECONDARY_COLUMN], kind="stable"
        ).reset_index(drop=True)
        primary = pairs[PRIMARY_COLUMN].to_numpy(np.int64)
        secondary = pairs[SECONDARY_COLUMN].to_numpy(np.int64)
        duplicate = (
            (primary[1:] == primary[:-1])
            & (secondary[1:] == secondary[:-1])
        )
        if np.any(duplicate):
            raise RuntimeError(f"case {case}: duplicate pair")
        v22_pair = pairs.response.to_numpy(np.float64)
        starts, counts, _, _ = scene_context(primary, v22_pair)
        scene_base, scene_correction, scene_final = stack_prediction(
            summary,
            base_booster,
            correction_booster,
            pairs[SCALED_FEATURES].to_numpy(np.float32),
            pairs[RAW_FEATURES].to_numpy(np.float32),
            starts,
            counts,
        )
        scene_v22 = np.add.reduceat(v22_pair, starts)
        scored = pd.DataFrame({
            "case": np.int16(case),
            "input_index": primary[starts],
            "n_pairs": counts.astype(np.int16),
            "R_blend_v22": scene_v22,
            "R_blend_base": scene_base,
            "pair_correction_sum": scene_correction,
            "R_blend": scene_final,
        })
        expected = reference.loc[reference.case == case].sort_values(
            "input_index", kind="stable"
        )
        actual = scored.sort_values("input_index", kind="stable")
        if not np.array_equal(
            expected.input_index.to_numpy(np.int64),
            actual.input_index.to_numpy(np.int64),
        ):
            raise RuntimeError(f"case {case}: replay/reference keys differ")
        replay_delta = (
            actual.R_blend_v22.to_numpy(np.float64)
            - expected.R_blend.to_numpy(np.float64)
        )
        replay_max = float(np.max(np.abs(replay_delta)))
        replay_mean = float(np.mean(np.abs(replay_delta)))
        if replay_max > args.replay_max_atol or replay_mean > args.replay_mean_atol:
            raise RuntimeError(
                f"case {case}: V2.2 replay differs, "
                f"max={replay_max:.3e}, meanabs={replay_mean:.3e}"
            )
        parts.append(scored)
        audits.append({
            "case": int(case),
            "n_pairs": int(len(pairs)),
            "n_scenes": int(len(scored)),
            "mean_n_pairs": float(counts.mean()),
            "mean_v22_scene": float(scene_v22.mean()),
            "mean_base_scene": float(scene_base.mean()),
            "mean_correction_scene": float(scene_correction.mean()),
            "mean_final_scene": float(scene_final.mean()),
            "replay_max_abs": replay_max,
            "replay_mean_abs": replay_mean,
        })
        print(
            f"case {case}: pairs={len(pairs):,} scenes={len(scored):,} "
            f"v22={scene_v22.mean():+.6f} base={scene_base.mean():+.6f} "
            f"correction={scene_correction.mean():+.6f} "
            f"final={scene_final.mean():+.6f}",
            flush=True,
        )
        del field, pairs, scored

    output = pd.concat(parts, ignore_index=True)
    if output.duplicated(KEY).any() or not np.isfinite(
        output.drop(columns=KEY).to_numpy(float)
    ).all():
        raise RuntimeError("final lookup is duplicate or non-finite")
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.feather"
    )
    output.to_feather(temporary)
    os.replace(temporary, output_path)

    payload = {
        "schema_version": 1,
        "design": (
            f"frozen sequential {objective_variant} transfer to ConstGold"
        ),
        "objective_variant": objective_variant,
        "case_window": [cases[0], cases[-1]],
        "n_cases": len(cases),
        "n_supported_pairs": int(sum(item["n_pairs"] for item in audits)),
        "n_supported_scenes": int(len(output)),
        "source_tag": args.tag,
        "source_regression_cuts": cuts,
        "source_r_max_arcsec": float(r_max),
        "source_k": int(k),
        "model_summary": str(model_summary_path),
        "model_summary_sha256": sha256(model_summary_path),
        "base_model": summary[
            "deployment_fit_all_200_cases_after_selection"
        ]["base_model"],
        "correction_model": summary[
            "deployment_fit_all_200_cases_after_selection"
        ]["correction_model"],
        "correction_physical_scaling": "one target_scale factor",
        "reference_lookup": str(Path(args.reference_lookup).resolve()),
        "output_lookup": str(output_path),
        "distributions": {
            column: distribution(output[column].to_numpy(float))
            for column in (
                "R_blend_v22", "R_blend_base", "pair_correction_sum", "R_blend"
            )
        },
        "case_audit": audits,
        "firewall": {
            "constgold_response_truth_read": False,
            "constgold_used_for_training_tuning_or_selection": False,
            "models_fixed_before_constgold_evaluation": True,
        },
    }
    strict_json(summary_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("CROSSFIT_SEQUENTIAL_CONSTGOLD_LOOKUP_DONE", flush=True)


if __name__ == "__main__":
    main()
