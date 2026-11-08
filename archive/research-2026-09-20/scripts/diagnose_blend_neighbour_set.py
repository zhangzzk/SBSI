#!/usr/bin/env python3
"""Compare the inference-time R_blend neighbour set with the labelled pair set.

The ConstGold `R_blend` lookup sums emulator predictions over the neighbours
found by the trained KD-tree aperture in the *complete* truth scene.  The only
`R_blend` quantity that has ever been compared against measured labels is the
sum over the pairs that exist in the response catalogue.  This diagnostic
evaluates both sums for the *same* primaries, the same emulator and the same
simulation, so that any difference is attributable to the neighbour set alone.

It reports nothing about `m`: it is a component audit of one additive term.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from sbsi.models import ModelPaths, load_emulator
from scripts.build_constgold_fixed_g0_blend_lookup import CONDITIONS, prepare_truth
from scripts.build_fixed_g0_response_profile_predictions import (
    BLEND_RESPONSE_COLUMNS,
    response_anchor_for_case,
    valid_blend_response_pairs,
)
from scripts.diagnose_constgold_fixed_g0_truth_support import PARTITION, truth_groups


REPORTED_GROUPS = (
    "all_fixed_g0",
    "inside_old_truth_support",
    "outside_old_truth_support",
    "outside_magnitude_only",
    "outside_size_only",
    "outside_both",
    "truth_faint",
    "truth_small",
    "truth_large",
    "truth_bright",
)
QUANTITIES = (
    "R_blend_full_scene",
    "R_blend_secondary_half",
    "R_blend_primary_half",
    "R_blend_labelled_model",
    "R_blend_labelled_measured",
    "pairs_full_scene",
    "pairs_secondary_half",
    "pairs_primary_half",
    "pairs_labelled",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument(
        "--input-pattern", required=True, help="truth format string containing {case}"
    )
    parser.add_argument("--response-catalogue", type=Path, required=True)
    parser.add_argument("--model-blend", type=Path, required=True)
    parser.add_argument(
        "--measurement-model",
        type=Path,
        required=True,
        help="flow checkpoint required by ModelPaths; unused by R_blend",
    )
    parser.add_argument("--emulator-model", type=Path, required=True)
    parser.add_argument("--emulator-metadata", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--response-shear", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def full_scene_response(predictor, primaries, truth, anchor_ids, primary_half_size):
    """Sum per-pair emulator predictions over the trained inference aperture.

    The sum is also split by the neighbour's structural target role.  The
    response catalogue can only label a pair whose neighbour is sheared, and the
    simulator shears one role at a time, so the secondary-half sum is the part
    of the inference-time total that measured labels can speak to.
    """
    predicted = predictor.predict_response(primaries, truth)
    for column in ("index_input_p", "index_input_s", "response", "distance"):
        if column not in predicted:
            raise RuntimeError(f"response prediction lacks {column!r}")
    response = predicted["response"].to_numpy(np.float64)
    if not np.isfinite(response).all():
        raise RuntimeError("non-finite full-scene blending response")
    # Work on an explicit float64 copy with a fresh index: the emulator returns
    # float32 responses, and summing a 16-term float32 group twice (whole, then
    # by role) does not reproduce the same value bit for bit.
    clean = pd.DataFrame(
        {
            "index_input_p": predicted["index_input_p"].to_numpy(np.int64),
            "index_input_s": predicted["index_input_s"].to_numpy(np.int64),
            "response": response,
            "distance": predicted["distance"].to_numpy(np.float64),
        }
    )
    secondary = clean["index_input_s"].to_numpy(np.int64) >= int(primary_half_size)
    keys = np.sort(np.unique(clean["index_input_p"].to_numpy(np.int64)))
    grouped = clean.groupby("index_input_p", sort=True)
    table = pd.DataFrame(
        {
            "input_index": keys,
            "R_blend_full_scene": grouped["response"].sum().reindex(keys).to_numpy(
                np.float64
            ),
            "pairs_full_scene": grouped["response"].size().reindex(keys).to_numpy(
                np.int64
            ),
            "max_distance_full_scene": grouped["distance"].max().reindex(
                keys
            ).to_numpy(np.float64),
        }
    )
    for label, mask in (("secondary_half", secondary), ("primary_half", ~secondary)):
        part = clean.loc[mask].groupby("index_input_p", sort=True)
        table[f"R_blend_{label}"] = (
            part["response"].sum().reindex(keys, fill_value=0.0).to_numpy(np.float64)
        )
        table[f"pairs_{label}"] = (
            part["response"].size().reindex(keys, fill_value=0).to_numpy(np.int64)
        )
    residual = np.abs(
        table["R_blend_secondary_half"].to_numpy(np.float64)
        + table["R_blend_primary_half"].to_numpy(np.float64)
        - table["R_blend_full_scene"].to_numpy(np.float64)
    )
    scale = np.abs(table["R_blend_full_scene"].to_numpy(np.float64)).max()
    if residual.max() > 1e-12 * max(scale, 1.0):
        raise RuntimeError(
            "role-split sums do not reconstruct the full-scene sum: "
            f"max residual {residual.max():.3e} on scale {scale:.3e}"
        )
    split_pairs = table["pairs_secondary_half"] + table["pairs_primary_half"]
    if not (split_pairs == table["pairs_full_scene"]).all():
        raise RuntimeError("role-split pair counts do not reconstruct the full count")
    unexpected = np.setdiff1d(keys, anchor_ids, assume_unique=False)
    if len(unexpected):
        raise RuntimeError(f"{len(unexpected)} non-anchor primaries in full-scene pairs")
    distances = clean["distance"].to_numpy(np.float64)
    report = {
        "pair_rows": int(len(clean)),
        "secondary_half_pair_rows": int(secondary.sum()),
        "primary_half_pair_rows": int((~secondary).sum()),
        "primary_half_size": int(primary_half_size),
        "primaries_with_pairs": int(len(table)),
        "anchor_without_full_scene_pairs": int(len(anchor_ids) - len(table)),
        "distance_arcsec_quantiles": {
            str(q): float(np.quantile(distances, q))
            for q in (0.5, 0.9, 0.99, 1.0)
        },
    }
    return table, report


def labelled_response(
    response_catalogue: Path,
    cases: tuple[int, ...],
    anchor_by_case: dict[int, np.ndarray],
    response_shear: float,
):
    """Sum the measured response over the pairs present in the label catalogue.

    The catalogue is scanned once for every requested case together, because it
    is ordered by case and a per-case rescan would re-read the same batches.
    """
    allowed = np.asarray(sorted(cases), dtype=np.int64)
    partial = {case: [] for case in cases}
    distances = {case: [] for case in cases}
    reports = {
        case: {
            "case_window_pair_rows": 0,
            "anchored_pair_rows": 0,
            "invalid_pair_rows_dropped": 0,
            "valid_pair_rows": 0,
        }
        for case in cases
    }
    previous_max = None
    with ipc.open_file(response_catalogue) as reader:
        missing = sorted(set(BLEND_RESPONSE_COLUMNS) - set(reader.schema.names))
        if missing:
            raise KeyError(f"response catalogue lacks {missing}")
        case_column = reader.schema.get_field_index("case")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            case_values = batch.column(case_column).to_numpy(zero_copy_only=False)
            batch_min, batch_max = int(case_values.min()), int(case_values.max())
            if previous_max is not None and batch_min < previous_max:
                raise RuntimeError("response catalogue is not ordered by case")
            previous_max = batch_max
            if batch_max < allowed.min():
                continue
            if batch_min > allowed.max():
                break
            frame = pa.Table.from_batches([batch]).select(
                BLEND_RESPONSE_COLUMNS
            ).to_pandas()
            frame = frame.loc[
                np.isin(frame["case"].to_numpy(np.int64), allowed)
            ].copy()
            if frame.empty:
                continue
            case_column_values = frame["case"].to_numpy(np.int64)
            anchored = np.zeros(len(frame), dtype=bool)
            for case in np.unique(case_column_values):
                local = case_column_values == case
                reports[int(case)]["case_window_pair_rows"] += int(local.sum())
                anchored[local] = np.isin(
                    frame.loc[local, "input_index"].to_numpy(np.int64),
                    anchor_by_case[int(case)],
                )
            frame = frame.loc[anchored].copy()
            if frame.empty:
                continue
            case_column_values = frame["case"].to_numpy(np.int64)
            valid = valid_blend_response_pairs(frame)
            for case in np.unique(case_column_values):
                local = case_column_values == case
                reports[int(case)]["anchored_pair_rows"] += int(local.sum())
                reports[int(case)]["invalid_pair_rows_dropped"] += int(
                    (local & ~valid).sum()
                )
                reports[int(case)]["valid_pair_rows"] += int((local & valid).sum())
            frame = frame.loc[valid]
            if frame.empty:
                continue
            frame = frame.assign(
                R_blend_labelled_measured=frame["delta_et1"].to_numpy(np.float64)
                / response_shear,
                pairs_labelled=1,
            )
            for case, block in frame.groupby("case", sort=False):
                distances[int(case)].append(block["distance"].to_numpy(np.float64))
                partial[int(case)].append(
                    block.loc[
                        :,
                        [
                            "input_index",
                            "R_blend_labelled_measured",
                            "pairs_labelled",
                        ],
                    ].groupby("input_index", sort=False, as_index=False).sum()
                )
            if (batch_index + 1) % 100 == 0:
                print(
                    f"BLEND_NEIGHBOUR_SET_LABEL_SCAN "
                    f"batch={batch_index + 1}/{reader.num_record_batches}",
                    flush=True,
                )
    tables = {}
    for case in cases:
        if not partial[case]:
            raise RuntimeError(f"case {case} has no valid anchored label pairs")
        table = pd.concat(partial[case], ignore_index=True).groupby(
            "input_index", sort=False, as_index=False
        ).sum()
        joined = np.concatenate(distances[case])
        reports[case]["primaries_with_pairs"] = int(len(table))
        reports[case]["anchor_without_labelled_pairs"] = int(
            len(anchor_by_case[case]) - len(table)
        )
        reports[case]["distance_arcsec_quantiles"] = {
            str(q): float(np.quantile(joined, q)) for q in (0.5, 0.9, 0.99, 1.0)
        }
        tables[case] = table
    return tables, reports


def case_group_means(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    groups = truth_groups(
        frame["true_r_magnitude"].to_numpy(np.float64),
        frame["true_Re_semimajor_arcsec"].to_numpy(np.float64),
    )
    means = {}
    for name in REPORTED_GROUPS:
        mask = groups[name]
        selected = frame.loc[mask]
        means[name] = {
            "n": int(mask.sum()),
            **{
                quantity: float(selected[quantity].mean()) if mask.any() else float("nan")
                for quantity in QUANTITIES
            },
        }
    return means


def bootstrap_group_summary(
    per_case: dict[int, dict[str, dict[str, float]]], n_boot: int, seed: int
) -> dict:
    cases = sorted(per_case)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(cases), size=(n_boot, len(cases)))
    summary = {}
    for name in REPORTED_GROUPS:
        counts = np.array([per_case[case][name]["n"] for case in cases], dtype=np.int64)
        values = {
            quantity: np.array(
                [per_case[case][name][quantity] for case in cases], dtype=np.float64
            )
            for quantity in QUANTITIES
        }
        if not np.isfinite(np.concatenate(list(values.values()))).all():
            # A rare tail group can be empty in some case; report that rather
            # than averaging over an undefined case mean.
            summary[name] = {
                "objects": int(counts.sum()),
                "cases": len(cases),
                "cases_without_members": [
                    int(case) for case in cases if per_case[case][name]["n"] == 0
                ],
                "status": "not_evaluable_on_every_case",
            }
            continue
        entry = {"objects": int(counts.sum()), "cases": len(cases), "status": "complete"}
        for quantity, series in values.items():
            replicates = series[draws].mean(axis=1)
            entry[quantity] = {
                "mean": float(series.mean()),
                "standard_error": float(replicates.std(ddof=1)),
            }
        for label, (top, bottom) in {
            "full_over_labelled_model": ("R_blend_full_scene", "R_blend_labelled_model"),
            "secondary_half_over_labelled_model": (
                "R_blend_secondary_half",
                "R_blend_labelled_model",
            ),
            "secondary_half_over_labelled_measured": (
                "R_blend_secondary_half",
                "R_blend_labelled_measured",
            ),
            "full_over_secondary_half": (
                "R_blend_full_scene",
                "R_blend_secondary_half",
            ),
            "labelled_model_over_measured": (
                "R_blend_labelled_model",
                "R_blend_labelled_measured",
            ),
            "full_over_labelled_measured": (
                "R_blend_full_scene",
                "R_blend_labelled_measured",
            ),
        }.items():
            replicates = values[top][draws].mean(axis=1) / values[bottom][draws].mean(
                axis=1
            )
            entry[label] = {
                "ratio": float(values[top].mean() / values[bottom].mean()),
                "standard_error": float(replicates.std(ddof=1)),
            }
        summary[name] = entry
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("blend neighbour-set diagnosis must run under Slurm")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cases = tuple(dict.fromkeys(int(case) for case in args.case))
    if len(cases) != len(args.case):
        raise ValueError("cases must be unique")
    if not np.isfinite(args.response_shear) or args.response_shear <= 0:
        raise ValueError("response shear must be positive and finite")
    for path in (
        args.response_catalogue,
        args.model_blend,
        args.measurement_model,
        args.emulator_model,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(args.emulator_metadata.read_text())
    regression = metadata["tasks"]["regression"]
    if regression.get("cuts") is not None:
        raise ValueError("this diagnosis requires emulator cuts=null")
    if regression.get("truth_analysis_cuts_applied") is not False:
        raise ValueError("this diagnosis requires no truth analysis cut")
    aperture = {"r_max_arcsec": regression["r_max"], "k": regression["k"]}

    paths = ModelPaths(
        flow_checkpoints=(args.measurement_model,),
        emulator_model=args.emulator_model,
        emulator_metadata=args.emulator_metadata,
    )
    predictor = load_emulator(paths, conditions=CONDITIONS, device=args.device)
    model_blend = pd.read_feather(args.model_blend)
    if model_blend.duplicated(["case", "input_index"]).any():
        raise ValueError("cached model blend predictions contain duplicate keys")

    anchor_path = args.domain_root / "response_anchor.feather"
    anchor_by_case = {
        case: response_anchor_for_case(anchor_path, case) for case in cases
    }
    labelled_tables, labelled_reports = labelled_response(
        args.response_catalogue, cases, anchor_by_case, args.response_shear
    )
    per_case_means = {}
    case_reports = {}
    for case in cases:
        print(f"BLEND_NEIGHBOUR_SET_START case={case}", flush=True)
        anchor_ids = anchor_by_case[case]
        truth = prepare_truth(Path(args.input_pattern.format(case=case)))
        aligned = truth.set_index("index", drop=False).reindex(anchor_ids)
        if aligned["index"].isna().any():
            raise RuntimeError(f"case {case} anchor contains identities absent from truth")
        primaries = aligned.reset_index(drop=True)

        full, full_report = full_scene_response(
            predictor, primaries, truth, anchor_ids, len(truth) // 2
        )
        labelled, labelled_report = labelled_tables[case], labelled_reports[case]
        cached = model_blend.loc[model_blend["case"] == case, ["input_index", "R_blend_model"]]
        if cached.empty:
            raise RuntimeError(f"cached model predictions lack case {case}")
        cached = cached.rename(columns={"R_blend_model": "R_blend_labelled_model"})

        # The labelled restriction is itself a selection: a primary earns a label
        # only if some pair of its survived the two-leg validity rule.  Record the
        # full-scene sum over every anchor primary and over each side of that
        # split, so the cost of the restriction is visible rather than implied.
        label_split = full.assign(
            has_label=full["input_index"].isin(labelled["input_index"])
        )
        anchor_split = {
            "all_anchor_primaries": label_split,
            "labelled_primaries": label_split.loc[label_split["has_label"]],
            "unlabelled_primaries": label_split.loc[~label_split["has_label"]],
        }
        full_scene_by_label = {
            name: {
                "n": int(len(part)),
                "mean_R_blend_full_scene": (
                    float(part["R_blend_full_scene"].mean()) if len(part) else None
                ),
                "mean_pairs_full_scene": (
                    float(part["pairs_full_scene"].mean()) if len(part) else None
                ),
            }
            for name, part in anchor_split.items()
        }

        frame = labelled.merge(cached, on="input_index", how="inner", validate="one_to_one")
        common_labelled = int(len(frame))
        frame = frame.merge(full, on="input_index", how="inner", validate="one_to_one")
        frame = frame.merge(
            primaries[["index", "r", "Re"]].rename(
                columns={
                    "index": "input_index",
                    "r": "true_r_magnitude",
                    "Re": "true_Re_semimajor_arcsec",
                }
            ),
            on="input_index",
            how="inner",
            validate="one_to_one",
        )
        if frame.empty:
            raise RuntimeError(f"case {case} has no common primaries")
        per_case_means[case] = case_group_means(frame)
        case_reports[str(case)] = {
            "anchor_rows": int(len(anchor_ids)),
            "full_scene": full_report,
            "labelled": labelled_report,
            "labelled_with_cached_model": common_labelled,
            "common_primaries": int(len(frame)),
            "full_scene_by_label": full_scene_by_label,
            "mean_R_blend_full_scene": float(frame["R_blend_full_scene"].mean()),
            "mean_R_blend_secondary_half": float(frame["R_blend_secondary_half"].mean()),
            "mean_R_blend_primary_half": float(frame["R_blend_primary_half"].mean()),
            "mean_R_blend_labelled_model": float(frame["R_blend_labelled_model"].mean()),
            "mean_R_blend_labelled_measured": float(
                frame["R_blend_labelled_measured"].mean()
            ),
        }
        print(
            f"BLEND_NEIGHBOUR_SET_DONE case={case} common={len(frame)} "
            f"full={frame['R_blend_full_scene'].mean():.6f} "
            f"secondary={frame['R_blend_secondary_half'].mean():.6f} "
            f"allanchor={full_scene_by_label['all_anchor_primaries']['mean_R_blend_full_scene']:.6f} "
            f"unlabelled_n={full_scene_by_label['unlabelled_primaries']['n']} "
            f"labelled_model={frame['R_blend_labelled_model'].mean():.6f} "
            f"labelled_measured={frame['R_blend_labelled_measured'].mean():.6f}",
            flush=True,
        )

    result = {
        "format_version": 1,
        "purpose": (
            "component audit: inference-time full-scene R_blend sum versus the "
            "labelled response-catalogue pair sum on identical primaries"
        ),
        "cases": list(cases),
        "aperture": aperture,
        "response_shear": args.response_shear,
        "partition": list(PARTITION),
        "domain": {
            "anchor": (
                "response anchor of the fixed-g0 v2 domain; no truth cut; "
                "no sheared-leg recut"
            ),
            "former_truth_support_used_only_as_diagnostic_partition": {
                "r_input": "18 < r_input < 25.8",
                "Re_input_semimajor_arcsec": "0.37 < Re_input < 1.5",
            },
        },
        "inputs": {
            "response_catalogue": str(args.response_catalogue),
            "response_catalogue_sha256": file_sha256(args.response_catalogue),
            "model_blend": str(args.model_blend),
            "model_blend_sha256": file_sha256(args.model_blend),
            "measurement_model_unused_for_R_blend": str(args.measurement_model),
            "measurement_model_sha256": file_sha256(args.measurement_model),
            "emulator_model": str(args.emulator_model),
            "emulator_model_sha256": file_sha256(args.emulator_model),
            "emulator_metadata": str(args.emulator_metadata),
            "emulator_metadata_sha256": file_sha256(args.emulator_metadata),
        },
        "limitations": [
            "This compares two model sums and one measured sum; it is not an m result.",
            "The measured label sum exists only for pairs the response catalogue contains.",
            "Case bootstrap over the supplied cases only; one emulator, one seed.",
        ],
        "case_reports": case_reports,
        "summary": bootstrap_group_summary(
            per_case_means, args.n_boot, args.bootstrap_seed
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, output)
    print(f"BLEND_NEIGHBOUR_SET_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
