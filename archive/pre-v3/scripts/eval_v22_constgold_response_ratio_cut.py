"""Evaluate V2.2 constgold after removing response-ratio-dominated detections.

The mask is fixed from the anchored-scene diagnostic and applied identically to
the measured simulation response, flow response, and V2.2 blend response.  The
absolute calibration follows the 16-seed constgold convention.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pyarrow.feather as pf

from scripts.eval_v2_indomain_m import catalogue_true_props, case_blocked_sim_sem


def key64(case: np.ndarray, primary: np.ndarray) -> np.ndarray:
    case = np.asarray(case, dtype=np.int64)
    primary = np.asarray(primary, dtype=np.int64)
    if len(primary) and (np.min(primary) < 0 or np.max(primary) >= (1 << 40)):
        raise RuntimeError("input_index exceeds packed-key allocation")
    return (case << 40) | primary


def seed_stat(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean_percent": float(np.mean(values)),
        "seed_sem_percent": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        "seed_sd_percent": float(np.std(values, ddof=1)),
        "n_seeds": int(len(values)),
        "per_seed_percent": values.tolist(),
    }


def case_fraction_sem(case: np.ndarray, removed: np.ndarray, population: np.ndarray) -> float:
    values = []
    for current in np.unique(case[population]):
        local = population & (case == current)
        values.append(float(np.mean(removed[local])))
    values = np.asarray(values)
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-glob", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--dominance-lookup", required=True)
    ap.add_argument("--threshold", type=float, default=20.0)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing existing output {args.output}")
    dumps = sorted(glob.glob(args.dump_glob))
    if len(dumps) != 16:
        raise SystemExit(f"absolute constgold m requires exactly 16 dumps; found {len(dumps)}")

    started = time.time()
    truth = catalogue_true_props(args.catalogue, args.min_case, started)
    reference = pf.read_table(dumps[0], columns=["case", "input_index"]).to_pandas()
    for column in ("case", "input_index"):
        if not np.array_equal(truth[column].to_numpy(), reference[column].to_numpy()):
            raise RuntimeError(f"catalogue/dump alignment differs in {column}")
    magnitude = truth["r_input_p"].to_numpy(float)
    size = truth["Re_input_p"].to_numpy(float)
    domain = (magnitude < 25.8) & (size > 0.5)

    lookup = pf.read_table(
        args.dominance_lookup,
        columns=["case", "input_index", "dominant_to_runner_up_abs_response"],
    )
    lookup_key = key64(
        lookup["case"].to_numpy(zero_copy_only=False),
        lookup["input_index"].to_numpy(zero_copy_only=False),
    )
    ratio = lookup["dominant_to_runner_up_abs_response"].to_numpy(
        zero_copy_only=False,
    ).astype(float)
    order = np.argsort(lookup_key)
    lookup_key, ratio = lookup_key[order], ratio[order]
    if len(lookup_key) != len(np.unique(lookup_key)):
        raise RuntimeError("dominance lookup has duplicate keys")
    truth_key = key64(
        truth["case"].to_numpy(np.int64), truth["input_index"].to_numpy(np.int64),
    )
    position = np.searchsorted(lookup_key, truth_key)
    clipped = np.clip(position, 0, len(lookup_key) - 1)
    matched = lookup_key[clipped] == truth_key
    match_fraction = float(np.mean(matched[domain]))
    if match_fraction < 0.999999:
        raise SystemExit(
            f"refusing unmatched domain rows: lookup match fraction={match_fraction:.8%}"
        )
    coordinate = ratio[clipped]
    if not np.isfinite(coordinate[domain & matched]).all():
        raise RuntimeError("non-finite dominance coordinate in V2.2 domain")

    baseline = domain & matched
    removed = baseline & (coordinate > args.threshold)
    kept = baseline & ~removed
    if not removed.any() or not kept.any():
        raise RuntimeError("response-ratio cut produced an empty subset")
    masks = {"baseline": baseline, "kept": kept, "removed": removed}
    case = reference["case"].to_numpy(np.int64)

    per_seed_m = {name: [] for name in masks}
    per_seed_components = {name: [] for name in masks}
    sim_reference = None
    for path in dumps:
        table = pf.read_table(
            path, columns=["case", "input_index", "r_sim", "R_flow", "R_blend"],
        )
        if not np.array_equal(table["case"].to_numpy(), reference["case"].to_numpy()) \
                or not np.array_equal(
                    table["input_index"].to_numpy(), reference["input_index"].to_numpy()
                ):
            raise RuntimeError(f"dump row alignment differs: {path}")
        r_sim = table["r_sim"].to_numpy(zero_copy_only=False).astype(float)
        r_flow = table["R_flow"].to_numpy(zero_copy_only=False).astype(float)
        r_blend = table["R_blend"].to_numpy(zero_copy_only=False).astype(float)
        if sim_reference is None:
            sim_reference = r_sim
        elif not np.array_equal(sim_reference, r_sim):
            raise RuntimeError("r_sim differs across flow-seed dumps")
        for name, mask in masks.items():
            components = np.asarray([
                np.mean(r_sim[mask]), np.mean(r_flow[mask]), np.mean(r_blend[mask]),
            ])
            per_seed_components[name].append(components)
            per_seed_m[name].append(
                100.0 * (components[0] / (components[1] + components[2]) - 1.0)
            )

    per_seed_m = {name: np.asarray(value) for name, value in per_seed_m.items()}
    component_means = {
        name: np.mean(np.asarray(value), axis=0)
        for name, value in per_seed_components.items()
    }
    delta = per_seed_m["kept"] - per_seed_m["baseline"]
    absolute_errors = {}
    for name, mask in masks.items():
        seed_sem = float(np.std(per_seed_m[name], ddof=1) / 4.0)
        model_mean = float(np.sum(component_means[name][1:]))
        sim_sem = 100.0 * case_blocked_sim_sem(sim_reference, case, mask) / model_mean
        absolute_errors[name] = {
            "seed_sem_percent": seed_sem,
            "case_blocked_sim_sem_percent": float(sim_sem),
            "quadrature_total_sem_percent": float(np.hypot(seed_sem, sim_sem)),
        }

    def component_payload(name: str) -> dict:
        sim, flow, blend = component_means[name]
        return {
            "R_sim": float(sim), "R_flow": float(flow), "R_blend": float(blend),
            "R_model": float(flow + blend),
            "response_gap_sim_minus_model": float(sim - flow - blend),
        }

    removal_fraction = float(np.sum(removed) / np.sum(baseline))
    payload = {
        "design": (
            "frozen V2.2 constgold population; remove detections with "
            "max|R_pair|/runner-up|R_pair| above the anchor-fixed threshold; "
            "apply identical mask to R_sim, R_flow and R_blend"
        ),
        "threshold": args.threshold,
        "cut": f"dominant_to_runner_up_abs_response > {args.threshold:g}",
        "n_rows": {name: int(np.sum(mask)) for name, mask in masks.items()},
        "removed_fraction": removal_fraction,
        "removed_fraction_case_sem": case_fraction_sem(case, removed, baseline),
        "lookup_match_fraction_v22_domain": match_fraction,
        "m_percent": {name: seed_stat(value) for name, value in per_seed_m.items()},
        "kept_minus_baseline_m_percent": seed_stat(delta),
        "absolute_m_uncertainty": absolute_errors,
        "mean_components": {name: component_payload(name) for name in masks},
        "selection_provenance": {
            "coordinate_and_threshold_selected_on": "anchored coherent-scene simulations",
            "constgold_role": "evaluation of the fixed diagnostic cut",
            "model_retrained_or_corrected": False,
        },
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_CONSTGOLD_RESPONSE_RATIO_CUT_DONE", flush=True)


if __name__ == "__main__":
    main()
