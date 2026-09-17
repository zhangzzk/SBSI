#!/usr/bin/env python3
"""Separate the ConstGold model response into its flow and blending terms.

ConstGold measures only the sum ``R = R_flow + R_blend``, so a shift in ``m``
cannot be assigned to either term from ``m`` alone -- an improvement in ``m``
may be removing an error or cancelling one against the other.

Binning the fixed-g0 cohort by the model's own ``R_blend`` identifies them.
Within bin ``i`` the model residual

    R_measured_i - R_flow_i - R_blend_i

is fitted as ``b * R_flow_i + a * R_blend_i``, so ``b`` is the fractional error
of the flow term and ``a`` the fractional error of the blending term.  The two
separate because ``R_blend`` varies by more than an order of magnitude across
the bins while ``R_flow`` barely moves.  Bin membership is fixed once from a
reference lookup, on truth-side quantities only, so every arm is reported on
the same objects.

A blending bin is not a random slice of the cohort: the quiet bins are
strongly enriched in bright primaries.  ``--magnitude-edge`` therefore crosses
the blending axis with a primary-magnitude axis read from the truth catalogue,
so the flow can be read at fixed primary magnitude and re-weighted to the
cohort rather than quoted on whichever primaries happened to be quiet.

Only the matched-usable branch is reported: the same identities usable in both
ConstGold legs, which is the branch with no selection modelling in it.

For diagnostics at the fixed-cohort boundary, ``--anchor-bin-column`` instead
bins on a measured target stored in the fixed-g0 anchor.  Such a bin is a
conditioning statement about a noisy selection variable, not an independent
calibration slice; the result metadata records that distinction explicitly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.measurement_model import load_measurement_model
from sbsi.shear_map import apply_shear_to_ellipticity

from scripts.evaluate_constgold_fixed_g0_response import (
    file_sha256,
    load_anchor,
    load_measured_leg,
    load_rblend,
    model_means,
    parse_mapping,
    shear_frames,
    sufficient,
    write_json,
)




def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument("--anchor-pattern", required=True)
    parser.add_argument("--flow", type=Path, required=True,
                        help="the one measurement flow, shared by every arm")
    parser.add_argument("--rblend", action="append", required=True,
                        help="label=path; the first is the reference for binning")
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument(
        "--bin-column", default="R_blend",
        help="lookup column the bins are cut on; R_blend is the SIGNED sum and "
             "can be near zero through cancellation, R_blend_abs is the sum of "
             "|per-pair response| and is near zero only when blending really is "
             "negligible",
    )
    parser.add_argument(
        "--anchor-bin-column",
        choices=("g0_flux_radius_arcsec",),
        help="bin on a measured fixed-g0 anchor target instead of a lookup column",
    )
    parser.add_argument(
        "--magnitude-edge", type=float, action="append", default=None,
        help="inner edge of the primary-magnitude axis crossed with the "
             "blending axis; repeat for several. Needs --primary-catalogue.",
    )
    parser.add_argument(
        "--primary-catalogue", default=None,
        help="format string with {case} for the truth catalogue carrying "
             "index_input and r_input; required with --magnitude-edge",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=7301)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def load_column(
    frame: pd.DataFrame, path: Path, case: int, anchor_ids: np.ndarray, column: str
) -> np.ndarray:
    """The same alignment ``load_rblend`` does, for any lookup column."""
    frame = frame.loc[frame["case"] == case]
    if frame["input_index"].duplicated().any():
        raise ValueError(f"duplicate keys for case {case} in {path}")
    aligned = frame.set_index("input_index")[column].reindex(anchor_ids)
    if aligned.isna().any():
        raise RuntimeError(
            f"lookup {path} misses {int(aligned.isna().sum())} anchor keys "
            f"in case {case}"
        )
    values = aligned.to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {column} in {path}")
    return values


def load_anchor_column(
    path: Path,
    column: str,
    expected_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Load a measured fixed-g0 target in sorted anchor-key order."""

    with np.load(path, allow_pickle=False) as stored:
        ids = np.asarray(stored["input_index"], dtype=np.int64)
        target = np.asarray(stored["target"], dtype=np.float64)
    if (
        ids.ndim != 1
        or target.shape != (len(ids), 4)
        or len(np.unique(ids)) != len(ids)
        or not np.isfinite(target).all()
    ):
        raise ValueError(f"invalid fixed-g0 anchor targets in {path}")
    order = np.argsort(ids, kind="stable")
    ids, target = ids[order], target[order]
    if expected_ids is not None and not np.array_equal(ids, expected_ids):
        raise RuntimeError(f"fixed-g0 target identities differ in {path}")
    if column == "g0_flux_radius_arcsec":
        values = 0.2 * target[:, 2]
        if np.any(values <= 0):
            raise ValueError(f"non-positive fixed-g0 radius in {path}")
        return values
    raise ValueError(f"unsupported fixed-g0 anchor column {column!r}")


def sufficient_or_empty(values: np.ndarray, weights: np.ndarray) -> dict:
    """``sufficient`` that reports a zero-weight cell instead of refusing it.

    Crossing blending with primary magnitude leaves cells that are empty in
    some individual cases; they are only ever read summed over cases.
    """
    if weights.size:
        return sufficient(values, weights)
    return {"denominator": 0.0, "numerator": np.zeros(2, dtype=np.float64)}


def load_primary_magnitude(pattern: str, case: int, anchor_ids: np.ndarray) -> np.ndarray:
    """Truth primary magnitude in anchor order; no measurement noise in it."""
    path = Path(pattern.format(case=case))
    frame = pd.read_feather(path, columns=["index_input", "r_input"])
    if frame["index_input"].duplicated().any():
        raise ValueError(f"duplicate index_input in {path}")
    aligned = frame.set_index("index_input")["r_input"].reindex(anchor_ids)
    if aligned.isna().any():
        raise RuntimeError(
            f"truth catalogue {path} misses {int(aligned.isna().sum())} anchor keys"
        )
    values = aligned.to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite r_input in {path}")
    return values


def magnitude_names(inner: tuple[float, ...]) -> list[str]:
    edges = (-np.inf, *inner, np.inf)
    return [
        "r_p" + ("<%g" % edges[1] if index == 0 else
                 ">=%g" % edges[-2] if index == len(edges) - 2 else
                 "[%g,%g)" % (edges[index], edges[index + 1]))
        for index in range(len(edges) - 1)
    ]


def bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Quantile edges, open at both ends, strictly increasing."""
    if bins < 3:
        raise ValueError("at least three bins are needed to identify two terms")
    inner = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    if not np.all(np.diff(inner) > 0):
        raise ValueError("R_blend quantile edges are not strictly increasing")
    return np.concatenate(([-np.inf], inner, [np.inf]))


def stack(per_case: dict, cases, cells, terms) -> tuple[np.ndarray, np.ndarray]:
    """Sufficient statistics as (case, cell, term, direction) arrays."""
    numerator = np.zeros((len(cases), len(cells), len(terms), 2, 2), dtype=np.float64)
    denominator = np.zeros((len(cases), len(cells), len(terms), 2), dtype=np.float64)
    for case_index, case in enumerate(cases):
        for cell_index, cell in enumerate(cells):
            entry = per_case[str(case)][cell]
            for term_index, term in enumerate(terms):
                for side, direction in enumerate(("plus", "minus")):
                    record = entry[term][direction]
                    numerator[case_index, cell_index, term_index, side] = np.asarray(
                        record["numerator"], dtype=np.float64)
                    denominator[case_index, cell_index, term_index, side] = record[
                        "denominator"]
    return numerator, denominator


def group_matrix(groups, cells) -> np.ndarray:
    """0/1 membership of each reported group in the crossed cells."""
    position = {cell: index for index, cell in enumerate(cells)}
    matrix = np.zeros((len(groups), len(cells)), dtype=np.float64)
    for row, members in enumerate(groups.values()):
        for cell in members:
            matrix[row, position[cell]] = 1.0
    return matrix


def group_responses(numerator: np.ndarray, denominator: np.ndarray,
                    matrix: np.ndarray, h: float, *, strict: bool = True):
    """First response component and object count, per (group, term).

    ``strict`` refuses an empty group, which is what a real empty cell means.
    A bootstrap replicate can empty a small cell by chance without the cell
    being empty in the data, so there the group is returned as ``nan`` and the
    nan-aware summary downstream reports how often that happened.
    """
    summed_n = np.tensordot(matrix, numerator.sum(axis=0), axes=([1], [0]))
    summed_d = np.tensordot(matrix, denominator.sum(axis=0), axes=([1], [0]))
    empty = summed_d <= 0
    if strict and empty.any():
        raise RuntimeError("a reported group is empty in some term or direction")
    means = summed_n[..., 0] / np.where(empty, np.nan, summed_d)
    return (means[..., 0] - means[..., 1]) / (2.0 * h), summed_d[:, 0, 0]


def fit_terms(flow: np.ndarray, blend: np.ndarray, residual: np.ndarray,
              weights: np.ndarray) -> dict:
    """Weighted least squares of residual on the two model terms."""
    design = np.column_stack((flow, blend))
    root = np.sqrt(weights)
    solution, *_ = np.linalg.lstsq(design * root[:, None], residual * root, rcond=None)
    blend_only, *_ = np.linalg.lstsq(blend[:, None] * root[:, None], residual * root,
                                     rcond=None)
    flow_only, *_ = np.linalg.lstsq(flow[:, None] * root[:, None], residual * root,
                                    rcond=None)
    return {
        "flow_fractional_error": float(solution[0]),
        "blend_fractional_error": float(solution[1]),
        "blend_only_fractional_error": float(blend_only[0]),
        "flow_only_fractional_error": float(flow_only[0]),
    }


FIT_NAMES = ("flow_fractional_error", "blend_fractional_error",
             "blend_only_fractional_error", "flow_only_fractional_error")


def summarize(per_case: dict, cases, cells, groups, blend_keys, labels, *,
              h, replicates, seed) -> dict:
    """Responses and ``m`` for every reported group, bootstrapped over cases."""
    cases = tuple(cases)
    terms = ["measured", "flow"] + [f"blend::{label}" for label in labels]
    numerator, denominator = stack(per_case, cases, cells, terms)
    matrix = group_matrix(groups, cells)
    names = list(groups)
    blend_rows = [names.index(key) for key in blend_keys]

    def m_percent(values: np.ndarray) -> np.ndarray:
        """(group, arm) percent bias from the (group, term) response table."""
        return np.column_stack([
            100.0 * (values[:, 0] / (values[:, 1] + values[:, 2 + index]) - 1.0)
            for index in range(len(labels))
        ])

    def fits_of(values: np.ndarray, objects: np.ndarray) -> dict:
        flow = values[blend_rows, 1]
        measured = values[blend_rows, 0]
        weights = objects[blend_rows]
        return {label: fit_terms(flow, values[blend_rows, 2 + index],
                                 measured - flow - values[blend_rows, 2 + index],
                                 weights)
                for index, label in enumerate(labels)}

    point, objects = group_responses(numerator, denominator, matrix, h)
    point_m = m_percent(point)
    point_fits = fits_of(point, objects)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cases), size=(replicates, len(cases)))
    m_samples = np.empty((replicates, len(names), len(labels)))
    fit_samples = {label: {name: np.empty(replicates) for name in FIT_NAMES}
                   for label in labels}
    for index, choice in enumerate(draws):
        values, counts = group_responses(numerator[choice], denominator[choice],
                                         matrix, h, strict=False)
        m_samples[index] = m_percent(values)
        if not np.isfinite(values[blend_rows]).all():
            for label in labels:
                for name in FIT_NAMES:
                    fit_samples[label][name][index] = np.nan
            continue
        for label, entry in fits_of(values, counts).items():
            for name, value in entry.items():
                fit_samples[label][name][index] = value

    def described(value, samples):
        usable = samples[np.isfinite(samples)]
        if usable.size < 2:
            return {"value": float(value), "standard_error": None, "ci95": None,
                    "resampled": int(usable.size)}
        entry = {"value": float(value),
                 "standard_error": float(usable.std(ddof=1)),
                 "ci95": [float(edge) for edge in np.quantile(usable, (0.025, 0.975))]}
        if usable.size < samples.size:
            entry["resampled"] = int(usable.size)
        return entry

    reported = {}
    for row, name in enumerate(names):
        entry = {"objects": float(objects[row]),
                 "R_measured": float(point[row, 0]),
                 "R_flow": float(point[row, 1])}
        for index, label in enumerate(labels):
            entry[f"R_blend::{label}"] = float(point[row, 2 + index])
            entry[f"m_percent::{label}"] = described(point_m[row, index],
                                                     m_samples[:, row, index])
        reported[name] = entry

    return {
        "cases": len(cases),
        "groups": reported,
        "bins": {key: reported[key] for key in blend_keys},
        "all": reported["all"],
        "fits": {
            label: {name: described(point_fits[label][name], fit_samples[label][name])
                    for name in FIT_NAMES}
            for label in labels
        },
    }


def main(argv=None):
    args = parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("ConstGold flow evaluation must run under Slurm")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.h <= 0 or args.draws < 2 or args.draws % 2 or args.batch_size <= 0:
        raise ValueError("positive h/batch and even draws >=2 required")
    cases = tuple(dict.fromkeys(int(case) for case in args.case))
    rblend_paths = parse_mapping(args.rblend, "--rblend")
    labels = list(rblend_paths)
    reference = labels[0]
    if not args.flow.is_file():
        raise FileNotFoundError(args.flow)

    tables = {}
    for label, path in rblend_paths.items():
        columns = ["case", "input_index", "R_blend"]
        #  Only the reference arm is binned on, so only it has to carry the bin
        #  column; a comparison arm built before the column existed is still a
        #  valid second blending model on the same cells.
        if (
            label == reference
            and args.anchor_bin_column is None
            and args.bin_column not in columns
        ):
            columns.append(args.bin_column)
        table = pd.read_feather(path, columns=columns)
        if table.duplicated(["case", "input_index"]).any():
            raise ValueError(f"duplicate R_blend keys in {path}")
        if not set(cases) <= set(table["case"].unique()):
            raise ValueError(f"R_blend lookup {path} does not cover all cases")
        tables[label] = table.loc[table["case"].isin(cases)].copy()
    if args.anchor_bin_column is None:
        edge_values = tables[reference][args.bin_column].to_numpy(np.float64)
        quantiles_of = str(rblend_paths[reference])
        binning_column = args.bin_column
    else:
        edge_values = np.concatenate(
            [
                load_anchor_column(
                    Path(args.anchor_pattern.format(case=case)),
                    args.anchor_bin_column,
                )
                for case in cases
            ]
        )
        quantiles_of = args.anchor_pattern
        binning_column = args.anchor_bin_column
    edges = bin_edges(edge_values, args.bins)
    del edge_values
    keys = tuple(f"bin{index:02d}" for index in range(args.bins))
    print("BIN_EDGES " + " ".join(f"{edge:.6g}" for edge in edges), flush=True)

    if args.magnitude_edge and not args.primary_catalogue:
        raise ValueError("--magnitude-edge needs --primary-catalogue")
    magnitude_inner = tuple(sorted(args.magnitude_edge or ()))
    if len(set(magnitude_inner)) != len(magnitude_inner):
        raise ValueError("repeated --magnitude-edge")
    if magnitude_inner:
        magnitudes = tuple(magnitude_names(magnitude_inner))
        cells = tuple(f"{key}|{name}" for key in keys for name in magnitudes)
        print("MAGNITUDE_BANDS " + " ".join(magnitudes), flush=True)
    else:
        magnitudes = ()
        cells = keys
    #  Reported groups, in order: the blending bins (what the two-term fit is
    #  cut on), then -- when the magnitude axis is on -- the magnitude bands
    #  marginalised over blending, then every crossed cell, then the cohort.
    groups = {key: [cell for cell in cells if cell.split("|")[0] == key]
              for key in keys}
    for name in magnitudes:
        groups[name] = [cell for cell in cells if cell.split("|")[1] == name]
    if magnitudes:
        groups.update({cell: [cell] for cell in cells})
    groups["all"] = list(cells)

    device = torch.device(args.device)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    bundle = load_measurement_model(args.flow, device=device)
    if bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES):
        raise ValueError("flow feature contract differs from the fixed-g0 domain")

    per_case = {}
    for case in cases:
        print(f"SPLIT_CASE_START case={case}", flush=True)
        anchor_ids, context = load_anchor(Path(args.anchor_pattern.format(case=case)))
        plus_frame, minus_frame = shear_frames(context, args.h)
        legs = {direction: load_measured_leg(args.constgold_root, case, sign * args.h,
                                             anchor_ids)
                for direction, sign in (("plus", 1.0), ("minus", -1.0))}
        common = legs["plus"]["usable"] & legs["minus"]["usable"]
        if not common.any():
            raise RuntimeError(f"case {case} has no matched usable fixed-g0 objects")

        base_e1, base_e2 = context[:, 0], context[:, 1]
        truth_delta = {}
        for direction, sign in (("plus", 1.0), ("minus", -1.0)):
            e1, e2 = apply_shear_to_ellipticity(base_e1, base_e2, sign * args.h, 0.0)
            truth_delta[direction] = np.column_stack((e1 - base_e1, e2 - base_e2))

        plus_mean, minus_mean = model_means(
            bundle, plus_frame, minus_frame, draws=args.draws,
            batch_size=args.batch_size, seed=args.sampling_seed + 10_000_019 * case)
        flow_values = {"plus": plus_mean, "minus": minus_mean}
        rblend = {label: load_rblend(tables[label], rblend_paths[label], case, anchor_ids)
                  for label in labels}
        if args.anchor_bin_column is None:
            binned_on = load_column(
                tables[reference],
                rblend_paths[reference],
                case,
                anchor_ids,
                args.bin_column,
            )
        else:
            binned_on = load_anchor_column(
                Path(args.anchor_pattern.format(case=case)),
                args.anchor_bin_column,
                anchor_ids,
            )
        assignment = np.digitize(binned_on, edges[1:-1], right=False)

        if magnitudes:
            band = np.digitize(
                load_primary_magnitude(args.primary_catalogue, case, anchor_ids),
                np.asarray(magnitude_inner), right=False)
            cell_of = assignment * len(magnitudes) + band
        else:
            cell_of = assignment

        per_case[str(case)] = {}
        for cell_index, cell in enumerate((*cells, "all")):
            mask = common if cell == "all" else (common & (cell_of == cell_index))
            count = int(mask.sum())
            if cell == "all" and not count:
                raise RuntimeError(f"case {case} has no matched usable objects")
            weights = np.ones(count)
            entry = {"measured": {}, "flow": {}}
            for direction in ("plus", "minus"):
                entry["measured"][direction] = sufficient_or_empty(
                    legs[direction]["values"][mask], weights)
                entry["flow"][direction] = sufficient_or_empty(
                    flow_values[direction][mask], weights)
            for label in labels:
                entry[f"blend::{label}"] = {
                    direction: sufficient_or_empty(
                        rblend[label][mask, None] * truth_delta[direction][mask], weights)
                    for direction in ("plus", "minus")
                }
            per_case[str(case)][cell] = entry
        del plus_mean, minus_mean, flow_values
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"SPLIT_CASE_DONE case={case} matched={int(common.sum())}", flush=True)

    result = {
        "format_version": 1,
        "purpose": (
            "separate the ConstGold model response into its flow and blending "
            f"terms by binning on {binning_column}"
        ),
        "h": float(args.h),
        "draws": int(args.draws),
        "sampling_seed": int(args.sampling_seed),
        "branch": "matched_usable",
        "binning": {
            "column": binning_column,
            "source": (
                "fixed-g0 anchor target"
                if args.anchor_bin_column is not None
                else "reference R_blend lookup"
            ),
            "magnitude_edges": [float(edge) for edge in magnitude_inner],
            "magnitude_bands": list(magnitudes),
            "primary_catalogue": args.primary_catalogue,
            "reference_arm": reference,
            "edges": [float(edge) for edge in edges],
            "quantiles_of": quantiles_of,
        },
        "inputs": {
            "flow": str(args.flow),
            "flow_sha256": file_sha256(args.flow),
            "rblend": {label: {"path": str(path), "sha256": file_sha256(path)}
                       for label, path in rblend_paths.items()},
        },
        "cases": list(cases),
        "summary": summarize(per_case, cases, cells, groups, keys, labels, h=args.h,
                             replicates=args.n_boot, seed=args.bootstrap_seed),
        "per_case_sufficient": per_case,
        "limitations": [
            (
                "Bins are defined on the fixed-g0 measured FLUX_RADIUS that also "
                "enters S0. This deliberately conditions on a noisy selection "
                "variable and is a boundary diagnostic, not calibration on an "
                "independent covariate."
                if args.anchor_bin_column is not None
                else "Bins are defined on the reference arm's chosen column, a "
                "truth-side model quantity with no measurement noise in it, so "
                "binning induces no selection on the measured shapes; the bins are "
                "nonetheless not a random partition of the cohort and R_flow is not "
                "constant across them."
            ),
            "The two-term fit assumes each term is wrong by a single multiplicative "
            "factor over the whole cohort; a term that is wrong in a way that "
            "correlates with R_blend differently from a scaling will be misattributed.",
            "The magnitude axis is truth r_input of the primary only; a flow "
            "error that tracks size or ellipticity at fixed magnitude is not "
            "resolved by it.",
            "One flow-training seed and one antithetic integration seed.",
            "Only the matched-usable branch is reported.",
        ],
    }
    write_json(output, result)
    print(f"CONSTGOLD_SPLIT_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
