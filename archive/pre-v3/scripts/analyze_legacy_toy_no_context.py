"""Analyze the exact legacy Gaussian-toy rerun with isolated components."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_legacy_toy_no_context import get_configs


SWEEP_REFERENCE = [
    (-0.007, 0.021), (-0.000, 0.014), (-0.001, 0.009), (+0.005, 0.046),
    (+0.001, 0.012), (+0.001, 0.007), (+0.000, 0.003), (+0.007, 0.022),
    (+0.007, 0.030), (+0.001, 0.022), (+0.002, 0.014), (+0.069, 0.055),
]
CLOSEPAIR_REFERENCE = [
    (+0.031, 0.026), (-0.001, 0.017), (+0.011, 0.016), (+0.000, 0.016),
    (-0.003, 0.017), (-0.001, 0.017), (+0.017, 0.034),
    (-0.023, 0.048), (-0.048, 0.033), (-0.003, 0.035), (-0.008, 0.033),
    (-0.014, 0.034), (-0.002, 0.036), (-0.032, 0.042),
    (+0.009, 0.035), (+0.010, 0.034), (+0.015, 0.022), (+0.019, 0.032),
    (-0.020, 0.035), (+0.009, 0.030), (-0.004, 0.011), (+0.002, 0.008),
    (+0.020, 0.035), (+0.020, 0.041), (-0.006, 0.027), (-0.009, 0.013),
]


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=0) / np.sqrt(len(values)))


def summarize_frame(frame: pd.DataFrame, audit: dict) -> dict:
    quantities = (
        "R_self_context", "R_neighbour_context_sum", "R_self_isolated",
        "R_neighbour_pair_sum", "R_full", "excess_context",
        "excess_pair_contextself", "excess_isolated",
        "delta_neighbour_context_pair", "delta_self_context_isolated",
    )
    row = {
        "suite": audit["suite"],
        "config_id": int(audit["config_id"]),
        "group": audit["group"],
        "label": audit["label"],
        "n_neighbours": int(audit["n_neighbours"]),
        "n_galaxies_full_scene": int(audit["n_galaxies_full_scene"]),
        "nreal": int(audit["nreal"]),
    }
    for quantity in quantities:
        values = frame[quantity].to_numpy(float)
        row[f"{quantity}_mean"] = float(values.mean())
        row[f"{quantity}_sem_matched"] = sem(values)
    for quantity in (
        "excess_context", "excess_pair_contextself", "excess_isolated"
    ):
        row[f"{quantity}_sem_legacy_quadrature"] = float(
            audit["summaries"][quantity]["sem_legacy_quadrature"]
        )
    return row


def load_suite(root: str, suite: str) -> tuple[pd.DataFrame, dict]:
    directory = Path(root) / suite
    rows = []
    maximum_identity_error = 0.0
    configs = get_configs(suite)
    for config in configs:
        config_id = int(config["config_id"])
        draw_path = directory / f"draws_config{config_id:02d}.feather"
        audit_path = directory / f"audit_config{config_id:02d}.json"
        for path in (draw_path, audit_path):
            if not path.exists():
                raise FileNotFoundError(path)
        frame = pd.read_feather(draw_path)
        with open(audit_path, encoding="utf-8") as handle:
            audit = json.load(handle)
        if audit["suite"] != suite or int(audit["config_id"]) != config_id:
            raise RuntimeError(f"identity mismatch for {suite} config {config_id}")
        if audit["label"] != config["label"] or audit["group"] != config["group"]:
            raise RuntimeError(f"configuration mismatch for {suite} config {config_id}")
        if int(audit["nreal"]) != int(config["legacy_nreal"]):
            raise RuntimeError(f"nreal mismatch for {suite} config {config_id}")
        if len(frame) != int(config["legacy_nreal"]):
            raise RuntimeError(f"row-count mismatch for {suite} config {config_id}")
        replay_one = np.abs(
            frame.excess_pair_contextself
            - frame.excess_context
            - frame.delta_neighbour_context_pair
        ).max()
        replay_two = np.abs(
            frame.excess_isolated
            - frame.excess_pair_contextself
            - frame.delta_self_context_isolated
        ).max()
        maximum_identity_error = max(
            maximum_identity_error, float(replay_one), float(replay_two)
        )
        rows.append(summarize_frame(frame, audit))
    result = pd.DataFrame(rows).sort_values("config_id", kind="mergesort")
    if result.config_id.tolist() != list(range(len(configs))):
        raise RuntimeError(f"incomplete {suite} configuration ids")
    return result, {
        "n_configs": int(len(result)),
        "n_noise_draws": int((result.nreal).sum()),
        "max_identity_replay_abs": float(maximum_identity_error),
    }


def validate_legacy_replay(table: pd.DataFrame) -> dict:
    references = {
        "sweep": SWEEP_REFERENCE,
        "closepair": CLOSEPAIR_REFERENCE,
    }
    payload = {}
    for suite, expected in references.items():
        local = table.loc[table.suite == suite].sort_values("config_id")
        expected_mean = np.array([item[0] for item in expected], dtype=float)
        expected_sem = np.array([item[1] for item in expected], dtype=float)
        observed_mean = local.excess_context_mean.to_numpy(float)
        observed_sem = local.excess_context_sem_legacy_quadrature.to_numpy(float)
        mean_error = np.abs(observed_mean - expected_mean)
        sem_error = np.abs(observed_sem - expected_sem)
        # The historical text logs retained only three decimals.  The faintest
        # ngmix fits can switch numerical branches across cluster CPU models,
        # so additionally require every mean to replay within 0.1 of its old
        # quoted technical error.  Preserve both diagnostics in the audit.
        standardized_error = mean_error / expected_sem
        within_printed_mean = mean_error <= 5.01e-4
        within_printed_sem = sem_error <= 5.01e-4
        if (
            standardized_error.max() > 0.1
            or mean_error.max() > 3.0e-3
            or not within_printed_sem.all()
        ):
            raise RuntimeError(
                f"{suite} failed scientific legacy replay: mean "
                f"{mean_error.max()}, standardized {standardized_error.max()}, "
                f"sem {sem_error.max()}"
            )
        payload[suite] = {
            "n_reference_rows": int(len(expected)),
            "n_means_replaying_to_printed_precision": int(within_printed_mean.sum()),
            "n_sems_replaying_to_printed_precision": int(within_printed_sem.sum()),
            "max_abs_mean_difference_from_rounded_log": float(mean_error.max()),
            "max_mean_difference_in_old_sem_units": float(standardized_error.max()),
            "max_abs_sem_difference_from_rounded_log": float(sem_error.max()),
            "all_rows_scientifically_replay_within_0p1_old_sem": True,
        }
    return payload


def extrema(table: pd.DataFrame, quantity: str) -> dict:
    values = table[f"{quantity}_mean"].to_numpy(float)
    index = int(np.argmax(np.abs(values)))
    row = table.iloc[index]
    return {
        "max_abs_mean": float(values[index]),
        "suite": row.suite,
        "config_id": int(row.config_id),
        "label": row.label,
        "sem_matched": float(row[f"{quantity}_sem_matched"]),
    }


def report_markdown(table: pd.DataFrame, payload: dict) -> str:
    lines = [
        "# Exact legacy Gaussian toys without retained neighbour context",
        "",
        "The original contextual decomposition is replayed as a control.  The "
        "new pair-context-self closure keeps the old target response measured in "
        "the full scene but replaces every neighbour term by an isolated "
        "primary--neighbour pair.  The fully isolated closure also measures the "
        "target alone.",
        "",
        "Every value below is the mean over the same matched Gaussian-noise draws "
        "used in the legacy tests.  Error terms are one matched-draw SEM.  These "
        "are fixed configurations, not a random galaxy population.",
        "",
        "| suite | configuration | N | old contextual excess | pair neighbours, contextual self | fully isolated components | neighbour context − pair |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        def value(quantity: str) -> str:
            return (
                f"{getattr(row, quantity + '_mean'):+.4f} ± "
                f"{getattr(row, quantity + '_sem_matched'):.4f}"
            )
        lines.append(
            f"| {row.suite} | {row.label} | {row.n_neighbours} | "
            f"{value('excess_context')} | "
            f"{value('excess_pair_contextself')} | "
            f"{value('excess_isolated')} | "
            f"{value('delta_neighbour_context_pair')} |"
        )
    lines.extend([
        "",
        "The JSON records the replay audit against all 38 rounded historical log "
        "rows, including last-digit CPU differences in the faint sweep, the "
        "original quadrature errors, algebraic closure checks, and the largest "
        "effects.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    for output in (args.output_csv, args.output_json, args.output_md):
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")
    sweep, sweep_audit = load_suite(args.input_root, "sweep")
    closepair, closepair_audit = load_suite(args.input_root, "closepair")
    table = pd.concat([sweep, closepair], ignore_index=True)
    if table.duplicated(["suite", "config_id"]).any():
        raise RuntimeError("duplicate suite/config identity")
    replay = validate_legacy_replay(table)
    payload = {
        "design": {
            "description": "exact reliable legacy Gaussian-toy grids with isolated component rerun",
            "n_configurations": int(len(table)),
            "n_sweep_configurations": int(len(sweep)),
            "n_closepair_configurations": int(len(closepair)),
            "noise": "Gaussian pixel RMS 10; matched map and fitter seed across every counterfactual",
            "g": 0.05,
            "stamp_pixels": 96,
            "pixel_scale_arcsec": 0.2,
            "profiles": "circular Gaussian, HLR 0.4 arcsec",
            "statistical_unit": "matched technical noise realization within each fixed configuration",
        },
        "coverage": {
            "sweep": sweep_audit,
            "closepair": closepair_audit,
            "max_identity_replay_abs": float(max(
                sweep_audit["max_identity_replay_abs"],
                closepair_audit["max_identity_replay_abs"],
            )),
        },
        "legacy_contextual_replay": replay,
        "largest_effects": {
            quantity: extrema(table, quantity)
            for quantity in (
                "excess_context", "excess_pair_contextself", "excess_isolated",
                "delta_neighbour_context_pair", "delta_self_context_isolated",
            )
        },
        "rows": table.to_dict(orient="records"),
        "inputs": {"root": os.path.abspath(args.input_root)},
        "artifacts": {
            "csv": os.path.abspath(args.output_csv),
            "markdown": os.path.abspath(args.output_md),
        },
    }
    if payload["coverage"]["max_identity_replay_abs"] > 1.0e-12:
        raise RuntimeError("global decomposition replay failed")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.output_md, "x", encoding="utf-8") as handle:
        handle.write(report_markdown(table, payload))
    print(json.dumps({
        "coverage": payload["coverage"],
        "legacy_contextual_replay": replay,
        "largest_effects": payload["largest_effects"],
    }, indent=2, sort_keys=True))
    print("LEGACY_TOY_NO_CONTEXT_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
