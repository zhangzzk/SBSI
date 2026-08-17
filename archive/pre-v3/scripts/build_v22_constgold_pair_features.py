"""Replay frozen V2.2 constgold pairs and retain physical dominant-pair features.

This extends the response-dominance audit with the coordinates needed for a
like-for-like transfer from coherent anchors: the signed dominant response,
its true separation and latent secondary properties, plus response/flux/
distance ranks.  The V2.2 summed response is replayed against the frozen lookup
case by case.  No measured constgold response is read here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

from sbs_shear.paths import CONST_SIM_DIR


BLENDEMU_ROOT = "/home/z/Zekang.Zhang/blendemu"
MODEL_DIR = os.path.join(BLENDEMU_ROOT, "models")
TILE = "tile180.0_-0.5"
COND = dict(
    pixel_size=0.2,
    zero_point=30.0,
    psf_fwhm=0.73,
    moffat_beta=2.224,
    pixel_rms=0.312,
)
KEY = ["case", "input_index"]
SHELLS = [
    (0.0, 1.0, "d0_1"),
    (1.0, 2.0, "d1_2"),
    (2.0, 3.0, "d2_3"),
    (3.0, 5.0, "d3_5"),
    (5.0, 7.0, "d5_7"),
    (7.0, 10.000001, "d7_10"),
]


def input_feather(case: int, sign: str, base: str) -> str:
    return (
        f"{base}/case{case}_{sign}/real0/catalogues/input/"
        f"gals_info_{TILE}.feather"
    )


def _id_columns(pairs: pd.DataFrame) -> tuple[str, str]:
    columns = [name for name in pairs if name.startswith("index")]
    if len(columns) < 2:
        raise RuntimeError(f"expected at least two pair ID columns, found {columns}")
    return columns[0], columns[1]


def summarize_pair_features(pairs: pd.DataFrame, field: pd.DataFrame) -> pd.DataFrame:
    """Return one row per primary with stable dominant-pair coordinates."""
    primary_column, secondary_column = _id_columns(pairs)
    needed = {primary_column, secondary_column, "response", "distance"}
    if missing := needed - set(pairs):
        raise KeyError(f"pair table lacks {sorted(missing)}")
    latent_needed = {"index", "r", "Re", "sersic_n"}
    if missing := latent_needed - set(field):
        raise KeyError(f"latent field lacks {sorted(missing)}")
    if field["index"].duplicated().any():
        raise RuntimeError("latent field has duplicate indices")

    work = pairs[[primary_column, secondary_column, "response", "distance"]].rename(
        columns={primary_column: "input_index", secondary_column: "secondary_index"}
    ).copy()
    work[["input_index", "secondary_index"]] = work[
        ["input_index", "secondary_index"]
    ].astype(np.int64)
    if work.duplicated(["input_index", "secondary_index"]).any():
        raise RuntimeError("duplicate deployed V2.2 pair")

    primary = field[["index", "r", "Re", "sersic_n"]].rename(columns={
        "index": "input_index",
        "r": "primary_mag",
        "Re": "primary_size",
        "sersic_n": "primary_sersic_n",
    })
    secondary = field[["index", "r", "Re", "sersic_n"]].rename(columns={
        "index": "secondary_index",
        "r": "dominant_secondary_mag",
        "Re": "dominant_secondary_size",
        "sersic_n": "dominant_secondary_sersic_n",
    })
    work = work.merge(primary, on="input_index", how="left", validate="many_to_one")
    work = work.merge(
        secondary, on="secondary_index", how="left", validate="many_to_one"
    )
    latent_columns = [
        "primary_mag", "primary_size", "primary_sersic_n",
        "dominant_secondary_mag", "dominant_secondary_size",
        "dominant_secondary_sersic_n",
    ]
    if work[latent_columns].isna().any().any():
        raise RuntimeError("pair could not be joined to latent input properties")

    response = work["response"].to_numpy(float)
    distance = work["distance"].to_numpy(float)
    if not np.isfinite(response).all() or not np.isfinite(distance).all():
        raise RuntimeError("non-finite pair response or distance")
    work["abs_response"] = np.abs(response)
    work["positive_response"] = np.clip(response, 0.0, None)
    work["negative_response"] = np.clip(response, None, 0.0)
    work["abs_near_0_2"] = np.where(distance < 2.0, np.abs(response), 0.0)
    work["abs_mid_2_5"] = np.where(
        (distance >= 2.0) & (distance < 5.0), np.abs(response), 0.0
    )
    work["abs_far_5_10"] = np.where(distance >= 5.0, np.abs(response), 0.0)
    work["secondary_flux"] = np.power(
        10.0, -0.4 * work["dominant_secondary_mag"].to_numpy(float)
    )
    work["primary_flux"] = np.power(
        10.0, -0.4 * work["primary_mag"].to_numpy(float)
    )
    work["flux_ratio_primary"] = work.secondary_flux / work.primary_flux
    for lo, hi, name in SHELLS:
        selected = (distance >= lo) & (distance < hi)
        work[f"n_pair_{name}"] = selected.astype(np.int8)
        work[f"R_pair_{name}"] = np.where(selected, response, 0.0)
        work[f"R_abs_pair_{name}"] = np.where(
            selected, np.abs(response), 0.0
        )

    response_order = work.sort_values(
        ["input_index", "abs_response", "secondary_index"],
        ascending=[True, False, True], kind="mergesort",
    ).copy()
    response_order["response_rank"] = (
        response_order.groupby("input_index", sort=False).cumcount() + 1
    )
    flux_order = work.sort_values(
        ["input_index", "secondary_flux", "secondary_index"],
        ascending=[True, False, True], kind="mergesort",
    ).copy()
    flux_order["flux_rank"] = (
        flux_order.groupby("input_index", sort=False).cumcount() + 1
    )
    distance_order = work.sort_values(
        ["input_index", "distance", "secondary_index"],
        ascending=[True, True, True], kind="mergesort",
    ).copy()
    distance_order["distance_rank"] = (
        distance_order.groupby("input_index", sort=False).cumcount() + 1
    )
    ranks = flux_order[["input_index", "secondary_index", "flux_rank"]].merge(
        distance_order[["input_index", "secondary_index", "distance_rank"]],
        on=["input_index", "secondary_index"], validate="one_to_one",
    )
    response_order = response_order.merge(
        ranks, on=["input_index", "secondary_index"], validate="one_to_one"
    )

    grouped = work.groupby("input_index", sort=False)
    aggregate = dict(
        R_blend=("response", "sum"),
        R_positive_sum=("positive_response", "sum"),
        R_negative_sum=("negative_response", "sum"),
        R_abs_sum=("abs_response", "sum"),
        R_abs_near_0_2=("abs_near_0_2", "sum"),
        R_abs_mid_2_5=("abs_mid_2_5", "sum"),
        R_abs_far_5_10=("abs_far_5_10", "sum"),
        neighbour_flux_sum=("secondary_flux", "sum"),
        n_pairs=("response", "size"),
        closest_pair_distance=("distance", "min"),
        mean_pair_distance=("distance", "mean"),
        std_pair_distance=("distance", "std"),
    )
    for _, _, name in SHELLS:
        aggregate[f"n_pair_{name}"] = (f"n_pair_{name}", "sum")
        aggregate[f"R_pair_{name}"] = (f"R_pair_{name}", "sum")
        aggregate[f"R_abs_pair_{name}"] = (f"R_abs_pair_{name}", "sum")
    totals = grouped.agg(**aggregate)
    totals.loc[totals.n_pairs == 1, "std_pair_distance"] = 0.0
    maximum = grouped.abs_response.max().rename("maximum_abs_response")
    strong = work.assign(
        strong=work.abs_response >= 0.1 * work.input_index.map(maximum),
    ).groupby("input_index", sort=False).strong.sum().rename(
        "n_response_pairs_ge_10pct_max"
    )

    dominant = response_order.loc[response_order.response_rank == 1].copy()
    dominant = dominant.rename(columns={
        "response": "dominant_response",
        "distance": "dominant_distance",
        "abs_response": "dominant_abs_response",
        "flux_ratio_primary": "dominant_flux_ratio_primary",
        "flux_rank": "dominant_flux_rank",
        "distance_rank": "dominant_distance_rank",
    })
    runner = response_order.loc[response_order.response_rank == 2, [
        "input_index", "response", "distance", "abs_response",
    ]].rename(columns={
        "response": "runner_up_response",
        "distance": "runner_up_distance",
        "abs_response": "runner_up_abs_response",
    })
    keep = [
        "input_index", "secondary_index", "dominant_response",
        "dominant_distance", "dominant_abs_response", "primary_mag",
        "primary_size", "primary_sersic_n", "dominant_secondary_mag",
        "dominant_secondary_size", "dominant_secondary_sersic_n",
        "dominant_flux_ratio_primary", "dominant_flux_rank",
        "dominant_distance_rank", "secondary_flux",
    ]
    dominant = dominant[keep].merge(
        totals, on="input_index", validate="one_to_one"
    ).merge(runner, on="input_index", how="left", validate="one_to_one").merge(
        strong, on="input_index", validate="one_to_one"
    )
    dominant["has_runner_up"] = dominant.runner_up_abs_response.notna()
    dominant["runner_up_response"] = dominant.runner_up_response.fillna(0.0)
    dominant["runner_up_abs_response"] = dominant.runner_up_abs_response.fillna(0.0)
    dominant["top_abs_fraction"] = (
        dominant.dominant_abs_response / dominant.R_abs_sum
    )
    dominant["dominant_to_runner_up_abs_response"] = (
        dominant.dominant_abs_response
        / np.maximum(dominant.runner_up_abs_response, 1.0e-12)
    )
    dominant["dominant_flux_share_neighbours"] = (
        dominant.secondary_flux / dominant.neighbour_flux_sum
    )
    dominant["dominant_size_ratio_primary"] = (
        dominant.dominant_secondary_size / dominant.primary_size
    )
    dominant = dominant.drop(columns=["secondary_flux"])
    shell_count = sum(
        dominant[f"n_pair_{name}"] for _, _, name in SHELLS
    ).to_numpy(np.int64)
    if not np.array_equal(shell_count, dominant.n_pairs.to_numpy(np.int64)):
        raise RuntimeError("distance shells do not cover every deployed pair")
    shell_response = sum(
        dominant[f"R_pair_{name}"] for _, _, name in SHELLS
    ).to_numpy(float)
    signed_delta = shell_response - dominant.R_blend.to_numpy(float)
    if float(np.max(np.abs(signed_delta))) > 1.0e-6:
        raise RuntimeError(
            "signed distance-shell responses do not close: "
            f"max={np.max(np.abs(signed_delta)):.3e}"
        )
    shell_abs_response = sum(
        dominant[f"R_abs_pair_{name}"] for _, _, name in SHELLS
    ).to_numpy(float)
    absolute_delta = shell_abs_response - dominant.R_abs_sum.to_numpy(float)
    if float(np.max(np.abs(absolute_delta))) > 1.0e-6:
        raise RuntimeError(
            "absolute distance-shell responses do not close: "
            f"max={np.max(np.abs(absolute_delta)):.3e}"
        )
    if not np.isfinite(dominant.drop(columns=["runner_up_distance"]).select_dtypes(
        include=[np.number]
    ).to_numpy(float)).all():
        raise RuntimeError("non-finite final dominant-pair feature")
    return dominant


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--reference-lookup", required=True)
    ap.add_argument("--base", default=CONST_SIM_DIR)
    ap.add_argument("--sign", default="0.02")
    ap.add_argument("--tag", default="lsst_r_extnbr_v22")
    ap.add_argument("--replay-max-atol", type=float, default=1.0e-6)
    ap.add_argument("--replay-mean-atol", type=float, default=1.0e-8)
    args = ap.parse_args()
    for output in (args.output, args.summary_json):
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if BLENDEMU_ROOT not in sys.path:
        sys.path.insert(0, BLENDEMU_ROOT)
    from blendemu.inference import BlendingPredictor

    predictor = BlendingPredictor.load(
        MODEL_DIR, tag=args.tag, conditions=COND, device="cpu"
    )
    cuts, rmax, k = predictor._select("regression")
    if not np.isclose(float(rmax), 10.0) or int(k) != 20:
        raise RuntimeError(f"expected frozen V2.2 r_max=10/k=20, got {rmax}/{k}")

    reference = pf.read_table(
        args.reference_lookup, columns=["case", "input_index", "R_blend"]
    ).to_pandas()
    reference = reference.loc[reference.case.isin(args.cases)].copy()
    if reference.duplicated(KEY).any():
        raise RuntimeError("reference lookup has duplicate keys")

    parts: list[pd.DataFrame] = []
    audits = []
    for case in args.cases:
        path = input_feather(case, args.sign, args.base)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        field = pf.read_table(path).to_pandas().rename(
            columns=lambda name: name.replace("_input", "")
        )
        pairs = predictor.predict_response(field, field)
        summary = summarize_pair_features(pairs, field)
        summary.insert(0, "case", int(case))

        expected = reference.loc[reference.case == case].sort_values("input_index")
        actual = summary.sort_values("input_index")
        if not np.array_equal(
            expected.input_index.to_numpy(np.int64),
            actual.input_index.to_numpy(np.int64),
        ):
            raise RuntimeError(f"case {case}: replay/reference keys differ")
        difference = actual.R_blend.to_numpy(float) - expected.R_blend.to_numpy(float)
        maximum = float(np.max(np.abs(difference)))
        mean_absolute = float(np.mean(np.abs(difference)))
        if maximum > args.replay_max_atol or mean_absolute > args.replay_mean_atol:
            raise RuntimeError(
                f"case {case}: lookup replay mismatch max={maximum:.3e}, "
                f"meanabs={mean_absolute:.3e}"
            )
        parts.append(summary)
        audits.append({
            "case": int(case), "n_primaries": int(len(summary)),
            "n_pairs": int(len(pairs)), "replay_max_abs": maximum,
            "replay_mean_abs": mean_absolute,
        })
        print(
            f"case{case}: {len(summary):,} primaries, {len(pairs):,} pairs, "
            f"replay={maximum:.2e}/{mean_absolute:.2e}", flush=True,
        )

    output = pd.concat(parts, ignore_index=True)
    output.to_feather(args.output)
    payload = {
        "design": "frozen V2.2 pair replay; physical features of stable maximum-absolute-response pair",
        "cases": [int(x) for x in args.cases],
        "model_tag": args.tag,
        "regression_cuts": cuts,
        "n_rows": int(len(output)),
        "columns": output.columns.tolist(),
        "replay_audit": audits,
        "constgold_measured_response_opened": False,
    }
    with open(args.summary_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"wrote {args.output}: {len(output):,} rows")
    print("V22_CONSTGOLD_PAIR_FEATURES_DONE", flush=True)


if __name__ == "__main__":
    main()
