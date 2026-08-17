"""One-shot 16-seed V2.2 constgold evaluation of a frozen R_blend emulator.

The per-object dumps already contain the same r_sim and R_flow for every
candidate.  This exact-key join replaces only R_blend, forms the response ratio
inside each flow seed, and reports the paired change.  Missing lookup rows are
never zero-filled.  This is evaluation-only; candidate selection must be fixed
before this script is run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf
import pyarrow.ipc as ipc

from scripts.eval_v2_indomain_m import catalogue_true_props, case_blocked_sim_sem


def key64(case: np.ndarray, primary: np.ndarray) -> np.ndarray:
    if len(primary) and int(np.max(primary)) >= (1 << 40):
        raise RuntimeError("input_index exceeds packed-key allocation")
    return (case.astype(np.int64) << 40) | primary.astype(np.int64)


def read_dump_case_range(
    path: str, columns: list[str], min_case: int, max_case: int | None
) -> pa.Table:
    """Read a dump, materializing only the requested case range when bounded."""
    if max_case is None:
        return pf.read_table(path, columns=columns)
    parts = []
    with ipc.open_file(pa.memory_map(path)) as reader:
        for batch_index in range(reader.num_record_batches):
            table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(columns)
            table = table.filter(pc.greater_equal(table["case"], pa.scalar(min_case)))
            table = table.filter(pc.less(table["case"], pa.scalar(max_case)))
            if table.num_rows:
                parts.append(table)
    if not parts:
        raise ValueError(f"no dump rows in requested case range [{min_case}, {max_case})")
    return pa.concat_tables(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-glob", required=True)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--lookup", required=True)
    parser.add_argument("--candidate-tag", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--min-case", type=int, default=40)
    parser.add_argument(
        "--max-case",
        type=int,
        default=None,
        help="optional exclusive upper case bound; full dumps are filtered before materialization",
    )
    parser.add_argument("--primary-mag-max", type=float, default=25.8)
    parser.add_argument("--primary-re-min", type=float, default=0.5)
    parser.add_argument("--domain-label", default="V2.2 rectangular primary domain")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    dumps = sorted(glob.glob(args.dump_glob))
    if len(dumps) != 16:
        raise SystemExit(f"reported m requires exactly 16 dumps; found {len(dumps)}")
    import torch
    checkpoint = torch.load(args.flow_checkpoint, map_location="cpu", weights_only=False)
    flow_features = list(checkpoint["condition_preprocessor"]["feature_names"])
    if "r_blend" in flow_features:
        raise SystemExit(
            "R_flow depends continuously on r_blend in this checkpoint; an additive-only swap "
            "would be incomplete and is therefore refused"
        )

    started = time.time()
    truth = catalogue_true_props(
        args.catalogue, args.min_case, started, max_case=args.max_case
    )
    reference = read_dump_case_range(
        dumps[0], ["case", "input_index"], args.min_case, args.max_case
    ).to_pandas()
    for column in ("case", "input_index"):
        if not np.array_equal(truth[column].to_numpy(), reference[column].to_numpy()):
            raise RuntimeError(f"catalogue/dump alignment differs in {column}")
    magnitude = truth.r_input_p.to_numpy(float)
    size = truth.Re_input_p.to_numpy(float)
    domain = (magnitude < args.primary_mag_max) & (size > args.primary_re_min)

    lookup = pf.read_table(args.lookup, columns=["case", "input_index", "R_blend"])
    lookup_key = key64(
        lookup["case"].to_numpy(zero_copy_only=False),
        lookup["input_index"].to_numpy(zero_copy_only=False),
    )
    lookup_value = lookup["R_blend"].to_numpy(zero_copy_only=False).astype(float)
    order = np.argsort(lookup_key)
    lookup_key, lookup_value = lookup_key[order], lookup_value[order]
    if len(lookup_key) != len(np.unique(lookup_key)):
        raise RuntimeError("candidate lookup has duplicate (case,input_index) keys")
    truth_key = key64(truth.case.to_numpy(np.int64), truth.input_index.to_numpy(np.int64))
    position = np.searchsorted(lookup_key, truth_key)
    clipped = np.clip(position, 0, len(lookup_key) - 1)
    matched = lookup_key[clipped] == truth_key
    match_fraction = float(np.mean(matched[domain]))
    if match_fraction < 0.999999:
        raise SystemExit(
            f"refusing candidate zero-fill: evaluation-domain match fraction={match_fraction:.8%}"
        )
    candidate_blend = lookup_value[clipped]
    mask = domain & matched
    if not np.isfinite(candidate_blend[mask]).all():
        raise SystemExit("candidate lookup contains non-finite R_blend inside the evaluation domain")

    baseline_m = []
    candidate_m = []
    components = []
    sim_reference = None
    for path in dumps:
        table = read_dump_case_range(
            path,
            ["case", "input_index", "r_sim", "R_flow", "R_blend"],
            args.min_case,
            args.max_case,
        )
        if not np.array_equal(table["case"].to_numpy(), reference.case.to_numpy()) \
                or not np.array_equal(
                    table["input_index"].to_numpy(), reference.input_index.to_numpy()
                ):
            raise RuntimeError(f"dump row alignment differs: {path}")
        r_sim = table["r_sim"].to_numpy(zero_copy_only=False).astype(float)
        r_flow = table["R_flow"].to_numpy(zero_copy_only=False).astype(float)
        r_blend = table["R_blend"].to_numpy(zero_copy_only=False).astype(float)
        if sim_reference is None:
            sim_reference = r_sim
        elif not np.array_equal(sim_reference, r_sim):
            raise RuntimeError("r_sim differs across flow-seed dumps")
        sim_mean = float(np.mean(r_sim[mask]))
        flow_mean = float(np.mean(r_flow[mask]))
        baseline_mean = float(np.mean(r_blend[mask]))
        candidate_mean = float(np.mean(candidate_blend[mask]))
        baseline_m.append(100.0 * (sim_mean / (flow_mean + baseline_mean) - 1.0))
        candidate_m.append(100.0 * (sim_mean / (flow_mean + candidate_mean) - 1.0))
        components.append([sim_mean, flow_mean, baseline_mean, candidate_mean])

    baseline_m = np.asarray(baseline_m)
    candidate_m = np.asarray(candidate_m)
    delta = candidate_m - baseline_m
    components = np.asarray(components)
    component_mean = components.mean(axis=0)
    case = reference.case.to_numpy()
    sim_sem_response = case_blocked_sim_sem(sim_reference, case, mask)
    absolute_sim_sem_baseline = 100.0 * sim_sem_response / sum(component_mean[1:3])
    absolute_sim_sem_candidate = 100.0 * sim_sem_response / (
        component_mean[1] + component_mean[3]
    )

    def seed_stat(values: np.ndarray) -> dict:
        return {
            "mean_percent": float(np.mean(values)),
            "seed_sem_percent": float(np.std(values, ddof=1) / np.sqrt(len(values))),
            "seed_sd_percent": float(np.std(values, ddof=1)),
            "n_seeds": int(len(values)),
            "per_seed_percent": values.tolist(),
        }

    payload = {
        "design": (
            f"frozen candidate one-shot swap into existing {args.domain_label} "
            "16-seed dumps; identical rows, r_sim and R_flow"
        ),
        "case_range": {
            "min_inclusive": args.min_case,
            "max_exclusive": args.max_case,
            "n_cases": int(len(np.unique(case[mask]))),
        },
        "evaluation_domain": {
            "label": args.domain_label,
            "primary_mag_max": args.primary_mag_max,
            "primary_re_min_arcsec": args.primary_re_min,
        },
        "candidate_tag": args.candidate_tag,
        "candidate_lookup": args.lookup,
        "n_rows_evaluation_domain": int(np.sum(mask)),
        "lookup_match_fraction_evaluation_domain": match_fraction,
        # Legacy aliases retained for readers of the original V2.2 result schema.
        "n_rows_v22_domain": int(np.sum(mask)),
        "lookup_match_fraction_v22_domain": match_fraction,
        "baseline_m": seed_stat(baseline_m),
        "candidate_m": seed_stat(candidate_m),
        "candidate_minus_baseline_m": seed_stat(delta),
        "absolute_sim_sem_percent": {
            "baseline": float(absolute_sim_sem_baseline),
            "candidate": float(absolute_sim_sem_candidate),
        },
        "mean_components": {
            "R_sim": float(component_mean[0]),
            "R_flow_same_both": float(component_mean[1]),
            "R_blend_baseline": float(component_mean[2]),
            "R_blend_candidate": float(component_mean[3]),
        },
        "flow_feature_names_verified": flow_features,
        "R_flow_invariant_under_swap": True,
        "R_blend_candidate_over_baseline_minus_one": float(
            component_mean[3] / component_mean[2] - 1.0
        ),
        "selection_firewall": {
            "candidate_fixed_before_constgold_evaluation": True,
            "candidate_selection_provenance": (
                "external to this evaluator; this script performs no selection or tuning"
            ),
            "constgold_used_for_training_tuning_or_selection": False,
            "constgold_role": "evaluation-only swap",
        },
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_SWAP_EMULATOR_M_DONE", flush=True)


if __name__ == "__main__":
    main()
