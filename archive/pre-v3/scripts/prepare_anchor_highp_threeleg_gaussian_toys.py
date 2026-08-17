"""Freeze catalogue-informed configurations for the controlled three-leg toy.

The anchor pool is selected only with model-side/latent quantities: held-out
coherent-anchor cases, faint primary magnitude, and raw V2.2 scene prediction.
For each requested multiplicity, the selected toy scene contains the K largest
absolute V2.2 pair predictions among neighbours within the close aperture.
Cases are unique across templates.  No measured per-anchor response or residual
is used to select a template.

The output source table embeds the exact flux ratios, circularized sizes, and
relative positions needed by the toy runner, so simulation jobs never have to
reconstruct selection or reopen the large feature catalogue.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["case", "input_index"]
FEATURE_COLUMNS = [
    *KEY,
    "scene_prediction",
    "primary_mag",
    "bias_truth_minus_model",  # loaded only for an explicit non-use audit
]
LATENT_COLUMNS = [
    "index", "RA", "DEC", "r", "Re", "axis_ratio", "sersic_n",
]


def design_for_case(designs: list[dict], case: int) -> dict:
    """Return the unique renderer-pair design containing ``case``."""
    matches = [
        design for design in designs
        if int(case) in {int(item["case"]) for item in design["per_case"]}
    ]
    if len(matches) != 1:
        raise RuntimeError(f"case {case} matched {len(matches)} pair designs")
    if matches[0].get("catalogue_source") != "rendered":
        raise RuntimeError("three-leg toy requires renderer-catalogue pairs")
    return matches[0]


def rank_close_pairs(pairs: pd.DataFrame, close_radius: float) -> pd.DataFrame:
    """Rank each anchor's close pairs by absolute model response."""
    required = {
        "case", "anchor_index", "secondary_index", "distance", "response",
        "n_pairs",
    }
    if missing := required - set(pairs):
        raise KeyError(f"pair table lacks {sorted(missing)}")
    local = pairs.loc[
        np.isfinite(pairs.distance)
        & np.isfinite(pairs.response)
        & (pairs.distance > 0.0)
        & (pairs.distance <= float(close_radius))
    ].copy()
    local["abs_response"] = np.abs(local.response.to_numpy(float))
    local = local.sort_values(
        ["case", "anchor_index", "abs_response", "distance", "secondary_index"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    )
    local["close_response_rank"] = (
        local.groupby(["case", "anchor_index"], sort=False).cumcount() + 1
    ).astype(np.int16)
    local["n_close_pairs"] = local.groupby(
        ["case", "anchor_index"], sort=False
    ).secondary_index.transform("size").astype(np.int16)
    return local


def choose_templates(
    candidates: pd.DataFrame,
    multiplicities: tuple[int, ...],
    n_per_multiplicity: int,
    seed: int,
) -> pd.DataFrame:
    """Choose one anchor per case, with cases unique across multiplicities."""
    required = {*KEY, "n_close_pairs"}
    if missing := required - set(candidates):
        raise KeyError(f"candidate table lacks {sorted(missing)}")
    rng = np.random.RandomState(int(seed))
    used_cases: set[int] = set()
    selected: list[pd.Series] = []
    # Hardest multiplicity first prevents lower-K strata consuming scarce cases.
    for multiplicity in sorted(multiplicities, reverse=True):
        eligible = candidates.loc[
            (candidates.n_close_pairs >= int(multiplicity))
            & ~candidates.case.astype(int).isin(used_cases)
        ].copy()
        cases = np.asarray(sorted(eligible.case.astype(int).unique()), dtype=int)
        if len(cases) < n_per_multiplicity:
            raise RuntimeError(
                f"K={multiplicity}: only {len(cases)} unused eligible cases for "
                f"{n_per_multiplicity} templates"
            )
        chosen_cases = rng.choice(cases, size=n_per_multiplicity, replace=False)
        for case in sorted(chosen_cases.tolist()):
            local = eligible.loc[eligible.case.astype(int) == int(case)].sort_values(
                ["case", "input_index"], kind="mergesort"
            )
            chosen = local.iloc[int(rng.randint(len(local)))].copy()
            chosen["multiplicity"] = int(multiplicity)
            selected.append(chosen)
            used_cases.add(int(case))
    output = pd.DataFrame(selected)
    output = output.sort_values(
        ["multiplicity", "case", "input_index"], kind="mergesort"
    ).reset_index(drop=True)
    output.insert(0, "template_id", np.arange(len(output), dtype=np.int16))
    if output.case.duplicated().any() or output.duplicated(KEY).any():
        raise RuntimeError("template selection failed the unique-case/key guard")
    expected = {
        int(value): int(n_per_multiplicity) for value in multiplicities
    }
    observed = {
        int(key): int(value)
        for key, value in output.multiplicity.value_counts().items()
    }
    if observed != expected:
        raise RuntimeError(f"multiplicity balance {observed} != {expected}")
    return output


def tangent_offsets(frame: pd.DataFrame, primary_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Small-angle source offsets in arcsec, centered on the primary."""
    by_id = frame.set_index("index", verify_integrity=True)
    primary = by_id.loc[int(primary_index)]
    cosdec = np.cos(np.deg2rad(float(primary.DEC)))
    x = (frame.RA.to_numpy(float) - float(primary.RA)) * cosdec * 3600.0
    y = (frame.DEC.to_numpy(float) - float(primary.DEC)) * 3600.0
    return x, y


def embed_template_sources(
    template: pd.Series,
    ranked_pairs: pd.DataFrame,
    latent: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    """Embed one primary and its response-leading close neighbours."""
    case = int(template.case)
    anchor = int(template.input_index)
    multiplicity = int(template.multiplicity)
    local_pairs = ranked_pairs.loc[
        (ranked_pairs.case.astype(int) == case)
        & (ranked_pairs.anchor_index.astype(int) == anchor)
        & (ranked_pairs.close_response_rank <= multiplicity)
    ].sort_values("close_response_rank", kind="mergesort")
    if len(local_pairs) != multiplicity:
        raise RuntimeError(
            f"template {template.template_id}: selected {len(local_pairs)} pairs "
            f"for K={multiplicity}"
        )
    if local_pairs.secondary_index.duplicated().any():
        raise RuntimeError("selected toy neighbours are not unique")
    ids = np.r_[anchor, local_pairs.secondary_index.to_numpy(np.int64)]
    by_id = latent.set_index("index", verify_integrity=True)
    missing = ids[~np.isin(ids, by_id.index.to_numpy(np.int64))]
    if len(missing):
        raise RuntimeError(f"latent catalogue lacks sources {missing[:5].tolist()}")
    scene = by_id.loc[ids].reset_index()
    x, y = tangent_offsets(scene, anchor)
    replay_distance = np.hypot(x[1:], y[1:])
    distance_error = float(np.max(np.abs(
        replay_distance - local_pairs.distance.to_numpy(float)
    )))
    if distance_error > 2.0e-3:
        raise RuntimeError(f"pair distance replay differs by {distance_error:.3g} arcsec")
    primary_mag_error = abs(float(scene.iloc[0].r) - float(template.primary_mag))
    if primary_mag_error > 2.0e-6:
        raise RuntimeError(
            f"primary magnitude replay differs by {primary_mag_error:.3g}"
        )

    rows: list[dict] = []
    for source_rank, source in scene.iterrows():
        q = float(np.clip(source.axis_ratio, 0.05, 1.0))
        common = {
            "template_id": int(template.template_id),
            "case": case,
            "input_index": anchor,
            "multiplicity": multiplicity,
            "source_rank": int(source_rank),
            "source_index": int(source["index"]),
            "source_role": "primary" if source_rank == 0 else "neighbour",
            "x_arcsec": float(x[source_rank]),
            "y_arcsec": float(y[source_rank]),
            "source_mag": float(source.r),
            "source_Re_semimajor_arcsec": float(source.Re),
            "source_axis_ratio": float(source.axis_ratio),
            "source_hlr_circularized_arcsec": float(source.Re) * np.sqrt(q),
            "source_sersic_n": float(source.sersic_n),
        }
        if source_rank == 0:
            common.update({
                "distance_arcsec": 0.0,
                "R_model_pair": 0.0,
                "abs_response_rank": 0,
            })
        else:
            pair = local_pairs.iloc[source_rank - 1]
            common.update({
                "distance_arcsec": float(pair.distance),
                "R_model_pair": float(pair.response),
                "abs_response_rank": int(pair.close_response_rank),
            })
        rows.append(common)

    template_row = {
        "template_id": int(template.template_id),
        "case": case,
        "input_index": anchor,
        "multiplicity": multiplicity,
        "scene_prediction_v22": float(template.scene_prediction),
        "primary_mag": float(template.primary_mag),
        "n_close_pairs_available": int(template.n_close_pairs),
        "selected_pair_prediction_sum": float(local_pairs.response.sum()),
        "selected_pair_abs_prediction_sum": float(local_pairs.abs_response.sum()),
        "max_selected_distance_arcsec": float(local_pairs.distance.max()),
        "max_pair_distance_replay_error_arcsec": distance_error,
        "primary_mag_replay_error": primary_mag_error,
    }
    return template_row, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--pair-design", nargs="+", required=True)
    parser.add_argument("--case-min", type=int, default=700)
    parser.add_argument("--case-max", type=int, default=899)
    parser.add_argument("--prediction-min", type=float, default=0.2)
    parser.add_argument("--primary-mag-min", type=float, default=24.5)
    parser.add_argument("--primary-mag-max", type=float, default=25.8)
    parser.add_argument("--close-radius", type=float, default=3.0)
    parser.add_argument("--multiplicities", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--n-per-multiplicity", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output-templates", required=True)
    parser.add_argument("--output-sources", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    outputs = (args.output_templates, args.output_sources, args.output_json)
    for output in outputs:
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")
    multiplicities = tuple(sorted(set(int(value) for value in args.multiplicities)))
    if not multiplicities or min(multiplicities) < 2:
        raise ValueError("multiplicities must contain integers >=2")
    if args.n_per_multiplicity < 2 or args.close_radius <= 0:
        raise ValueError("n-per-multiplicity >=2 and close-radius >0 are required")

    designs = []
    for path in args.pair_design:
        with open(path, encoding="utf-8") as handle:
            design = json.load(handle)
        design["_path"] = os.path.abspath(path)
        designs.append(design)

    features = pd.read_feather(args.features, columns=FEATURE_COLUMNS)
    pool = features.loc[
        features.case.between(args.case_min, args.case_max)
        & (features.scene_prediction > float(args.prediction_min))
        & (features.primary_mag >= float(args.primary_mag_min))
        & (features.primary_mag < float(args.primary_mag_max))
    ].copy()
    if pool.empty:
        raise RuntimeError("high-prediction faint-primary pool is empty")
    # The residual is deliberately discarded before any selection operation.
    pool = pool.drop(columns="bias_truth_minus_model")
    pool_keys_by_case = {
        int(case): set(local.input_index.astype(int).tolist())
        for case, local in pool.groupby("case", sort=True)
    }

    ranked_parts = []
    full_prediction_errors = []
    for case in sorted(pool_keys_by_case):
        design = design_for_case(designs, case)
        base = Path(design["manifest_dir"])
        pair_path = base / f"{design.get('pair_prefix', 'pairs')}_case{case}.feather"
        pairs = pd.read_feather(pair_path)
        pairs = pairs.loc[
            pairs.anchor_index.astype(int).isin(pool_keys_by_case[case])
        ].copy()
        model_sum = pairs.groupby("anchor_index", sort=False).response.sum()
        local_features = pool.loc[pool.case == case].set_index("input_index")
        shared = local_features.index.intersection(model_sum.index)
        if len(shared):
            full_prediction_errors.extend(
                (local_features.loc[shared].scene_prediction - model_sum.loc[shared])
                .abs().to_numpy(float).tolist()
            )
        ranked_parts.append(rank_close_pairs(pairs, args.close_radius))
    ranked = pd.concat(ranked_parts, ignore_index=True)
    close_counts = ranked.drop_duplicates(["case", "anchor_index"])[
        ["case", "anchor_index", "n_close_pairs"]
    ].rename(columns={"anchor_index": "input_index"})
    candidates = pool.merge(close_counts, on=KEY, how="inner", validate="one_to_one")
    templates = choose_templates(
        candidates,
        multiplicities,
        args.n_per_multiplicity,
        args.seed,
    )

    template_rows: list[dict] = []
    source_rows: list[dict] = []
    for template in templates.itertuples(index=False):
        row = pd.Series(template._asdict())
        design = design_for_case(designs, int(row.case))
        base = Path(design["manifest_dir"])
        latent = pd.read_feather(
            base / f"gals{int(row.case)}_{float(design['g'])}.feather",
            columns=LATENT_COLUMNS,
        )
        embedded, sources = embed_template_sources(row, ranked, latent)
        template_rows.append(embedded)
        source_rows.extend(sources)
    template_frame = pd.DataFrame(template_rows).sort_values("template_id")
    source_frame = pd.DataFrame(source_rows).sort_values(
        ["template_id", "source_rank"], kind="mergesort"
    )
    expected_sources = int((template_frame.multiplicity + 1).sum())
    if len(source_frame) != expected_sources:
        raise RuntimeError(f"embedded {len(source_frame)} sources != {expected_sources}")
    if source_frame.duplicated(["template_id", "source_rank"]).any():
        raise RuntimeError("duplicate embedded source rank")

    Path(args.output_templates).parent.mkdir(parents=True, exist_ok=True)
    template_frame.to_feather(args.output_templates)
    source_frame.to_feather(args.output_sources)
    payload = {
        "design": (
            "outcome-free held-out V2.2 high-scene-prediction faint-primary "
            "anchors; unique catalogue case per template; K largest-absolute-"
            "prediction neighbours inside the close aperture"
        ),
        "selection_uses_measured_response_or_residual": False,
        "source_features": os.path.abspath(args.features),
        "pair_designs": [design["_path"] for design in designs],
        "case_window": [int(args.case_min), int(args.case_max)],
        "prediction_rule": f"scene_prediction > {args.prediction_min}",
        "primary_magnitude_rule": (
            f"{args.primary_mag_min} <= primary_mag < {args.primary_mag_max}"
        ),
        "close_radius_arcsec": float(args.close_radius),
        "multiplicities": list(multiplicities),
        "n_per_multiplicity": int(args.n_per_multiplicity),
        "selection_seed": int(args.seed),
        "n_feature_pool": int(len(pool)),
        "n_candidates_with_close_pair": int(len(candidates)),
        "n_templates": int(len(template_frame)),
        "n_sources": int(len(source_frame)),
        "all_cases_distinct": bool(~template_frame.case.duplicated().any()),
        "max_full_scene_prediction_replay_error": float(
            np.max(full_prediction_errors) if full_prediction_errors else np.nan
        ),
        "max_selected_distance_replay_error_arcsec": float(
            template_frame.max_pair_distance_replay_error_arcsec.max()
        ),
        "outputs": {
            "templates": os.path.abspath(args.output_templates),
            "sources": os.path.abspath(args.output_sources),
        },
        "templates": template_frame.to_dict(orient="records"),
    }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(template_frame.to_string(index=False))
    print("ANCHOR_HIGHP_THREELEG_GAUSSIAN_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
