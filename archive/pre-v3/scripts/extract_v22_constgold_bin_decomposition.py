"""Extract exact q3 self/deployed/other/additivity truth on common detections."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from scripts.extract_v22_matched_decomposition import (  # noqa: E402
    full_scene_features, read_anchor_leg, response_columns,
)


MODES = ("total", "self", "deployed", "other")


def parse_bases(items: list[str]) -> dict[str, str]:
    bases: dict[str, str] = {}
    for item in items:
        mode, sep, path = item.partition("=")
        if not sep or not mode or not path:
            raise ValueError(f"--base must be MODE=PATH, got {item!r}")
        if mode in bases:
            raise ValueError(f"duplicate mode {mode!r}")
        bases[mode] = path
    if set(bases) != set(MODES):
        raise ValueError(f"expected exactly {list(MODES)}, got {sorted(bases)}")
    return bases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", action="append", required=True, help="MODE=PATH")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--evaluation-dump", required=True,
                    help="one frozen V2.2 per-object constgold dump defining plotted keys")
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.02)
    ap.add_argument("--min-coverage", type=float, default=0.25,
                    help="minimum eight-leg common-anchor fraction per case")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    bases = parse_bases(args.base)
    manifest = Path(args.manifest_dir)

    evaluated = pf.read_table(
        args.evaluation_dump, columns=["case", "input_index", "R_blend"],
    ).to_pandas()
    evaluated = evaluated[evaluated["case"].isin(args.cases)].copy()
    consistency = evaluated.groupby(["case", "input_index"], sort=False)["R_blend"].agg(
        ["min", "max"]
    )
    if not np.allclose(consistency["min"], consistency["max"], rtol=0.0, atol=0.0):
        raise RuntimeError("evaluation dump has inconsistent repeated R_blend for one key")
    evaluated = evaluated.drop_duplicates(["case", "input_index"])

    parts: list[pd.DataFrame] = []
    case_summary = []
    for case in args.cases:
        manifest_anchors = pd.read_feather(manifest / f"anchors_case{case}.feather")
        eval_case = evaluated.loc[evaluated["case"] == case, ["input_index", "R_blend"]]
        anchors = manifest_anchors.merge(
            eval_case, left_on="index", right_on="input_index", how="inner",
            validate="one_to_one",
        ).drop(columns="input_index")
        if anchors.empty:
            raise RuntimeError(f"case {case}: no rendered anchors occur in evaluation dump")
        if not np.allclose(
            anchors["R_blend_lookup"], anchors["R_blend"], rtol=0.0, atol=0.0,
        ):
            raise RuntimeError(f"case {case}: manifest/evaluation R_blend differs")
        anchor_ids = anchors["index"].to_numpy(np.int64)
        joined = pd.DataFrame({"input_index": anchor_ids})
        for mode, base in bases.items():
            for sign_name, sign in (("plus", +args.g), ("minus", -args.g)):
                leg = read_anchor_leg(base, case, sign, anchor_ids)
                before = len(joined)
                rename = {
                    column: f"{column}_{mode}_{sign_name}"
                    for column in leg if column != "input_index"
                }
                joined = joined.merge(
                    leg.rename(columns=rename), on="input_index", how="inner",
                    validate="one_to_one",
                )
                print(
                    f"case {case} {mode}/{sign_name}: detected={len(leg):,}/"
                    f"{len(anchor_ids):,}; cumulative={len(joined):,}/{before:,}",
                    flush=True,
                )
        coverage = len(joined) / len(anchor_ids)
        if coverage < args.min_coverage:
            raise RuntimeError(
                f"case {case}: all-leg coverage {coverage:.2%} < "
                f"{args.min_coverage:.2%}"
            )

        scene = full_scene_features(bases["total"], case, +args.g, anchor_ids)
        joined = joined.merge(scene, on="input_index", how="left", validate="one_to_one")
        joined = joined.merge(
            anchors[["index", "R_blend_lookup", "R_blend_replayed"]].rename(
                columns={"index": "input_index"}
            ), on="input_index", how="left", validate="one_to_one",
        )
        if joined[["R_blend_lookup", "R_blend_replayed",
                   "logflux_abs_near_0_1", "logflux_abs_mid_1_3",
                   "logflux_abs_far_3_10"]].isna().any().any():
            raise RuntimeError(f"case {case}: manifest/scene feature join failed")

        # V2.2 flow conditions on the measured primary and intrinsic scene
        # crowding.  The shear-even self legs give the least treatment-dependent
        # primary measurement while retaining the identical rendered scene.
        joined["measured_mag_auto"] = 0.5 * (
            joined["measured_mag_auto_self_plus"]
            + joined["measured_mag_auto_self_minus"]
        )
        joined["measured_flux_radius"] = 0.5 * (
            joined["measured_flux_radius_self_plus"]
            + joined["measured_flux_radius_self_minus"]
        )

        for mode in MODES:
            r1, r2 = response_columns(joined, mode, args.g)
            joined[f"R_{mode}_truth"] = r1
            joined[f"R_{mode}_null"] = r2
        joined["R_blend_model"] = joined["R_blend_lookup"]
        joined["Delta_deployed"] = (
            joined["R_blend_model"] - joined["R_deployed_truth"]
        )
        joined["Contribution_other"] = -joined["R_other_truth"]
        joined["Delta_add_raw"] = (
            joined["R_total_truth"] - joined["R_self_truth"]
            - joined["R_deployed_truth"] - joined["R_other_truth"]
        )
        joined["Contribution_additivity"] = -joined["Delta_add_raw"]
        joined.insert(0, "case", case)
        parts.append(joined)
        case_summary.append({
            "case": case, "n_rendered_manifest_anchors": len(manifest_anchors),
            "n_evaluation_anchors": len(anchor_ids),
            "n_common_anchors": len(joined), "coverage": coverage,
            **{
                f"R_{mode}_truth": float(joined[f"R_{mode}_truth"].mean())
                for mode in MODES
            },
            "R_blend_model": float(joined["R_blend_model"].mean()),
            "Delta_deployed": float(joined["Delta_deployed"].mean()),
            "Contribution_other": float(joined["Contribution_other"].mean()),
            "Contribution_additivity": float(joined["Contribution_additivity"].mean()),
        })
        print(
            f"case {case}: common={len(joined):,}/{len(anchor_ids):,} ({coverage:.1%}) "
            f"Rtot={joined['R_total_truth'].mean():+.5f} "
            f"Rself={joined['R_self_truth'].mean():+.5f} "
            f"Rdep={joined['R_deployed_truth'].mean():+.5f} "
            f"Rother={joined['R_other_truth'].mean():+.5f} "
            f"Rblend={joined['R_blend_model'].mean():+.5f}", flush=True,
        )

    out = pd.concat(parts, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_feather(args.output)
    sidecar = os.path.splitext(args.output)[0] + ".json"
    with open(sidecar, "x", encoding="utf-8") as handle:
        json.dump({
            "g": args.g, "cases": args.cases, "modes": list(MODES),
            "n_rows": len(out), "n_cases": out["case"].nunique(),
            "evaluation_dump": args.evaluation_dump,
            "population_policy": (
                "unique (case,input_index) keys in the frozen V2.2 constgold evaluation dump; "
                "rendered q3 manifest is a disjoint superset"
            ),
            "common_detection_policy": "inner intersection across all four modes and both signs",
            "minimum_case_coverage": args.min_coverage,
            "component_identity": (
                "R_total = R_self + R_deployed + R_other + Delta_add_raw"
            ),
            "case_summary": case_summary,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}: {len(out):,} common anchors")
    print("V22_CONSTGOLD_Q3_DECOMP_EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
