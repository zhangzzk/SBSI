"""Held-out full-matrix response audit for a trained measurement flow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .flow_response_target import _load_leg, estimate_response_matrix
from .measurement_model import load_measurement_model
from .training import build_shifted_context


ZERO_COLUMNS = [
    "case", "input_index", "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "e1_input_rot0_p", "e2_input_rot0_p",
    "e1_input_rot0_s", "e2_input_rot0_s", "gamma1_input_p", "gamma2_input_p",
    "gamma1_input_s", "gamma2_input_s", "neighbored", "distance", "polarization_angle",
    "nbr_flux_near", "nbr_flux_far", "nbr_flux_max",
    "measured_ngmix_g1", "measured_ngmix_g2",
]


def _mean_model_matrix(bundle, frame, delta=0.02, batch_size=65536):
    metadata = bundle.metadata
    condition_features = metadata["condition_features"]
    kwargs = dict(
        pixel_rms=0.312,
        pixel_size=0.2,
        zero_mag=30.0,
        psf_fwhm=0.73,
        moffat_beta=2.224,
    )
    contexts = [
        build_shifted_context(
            frame, direction, sign * delta, bundle.condition_preprocessor,
            condition_features, kwargs,
        )
        for direction, sign in [((1.0, 0.0), 1.0), ((1.0, 0.0), -1.0),
                                ((0.0, 1.0), 1.0), ((0.0, 1.0), -1.0)]
    ]
    scales = tuple(float(value) for value in bundle.target_transform.scales[:2])
    total = np.zeros(4, dtype=np.float64)
    count = 0
    bundle.model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            tensors = [
                torch.as_tensor(context[start:stop], dtype=torch.float32, device=bundle.device)
                for context in contexts
            ]
            p1, m1, p2, m2 = [bundle.model._shift(context) for context in tensors]
            inv = 1.0 / (2.0 * delta)
            matrix = torch.stack(
                [
                    (p1[:, 0] - m1[:, 0]) * scales[0] * inv,
                    (p2[:, 0] - m2[:, 0]) * scales[0] * inv,
                    (p1[:, 1] - m1[:, 1]) * scales[1] * inv,
                    (p2[:, 1] - m2[:, 1]) * scales[1] * inv,
                ],
                dim=1,
            )
            total += matrix.double().sum(dim=0).cpu().numpy()
            count += len(matrix)
    return (total / count).reshape(2, 2)


def _shear_state_masks(frame, tolerance=1.0e-12):
    """Classify rows by whether the primary and secondary carry applied shear."""

    primary = frame[["gamma1_input_p", "gamma2_input_p"]].to_numpy(dtype=np.float64)
    secondary = frame[["gamma1_input_s", "gamma2_input_s"]].to_numpy(dtype=np.float64)
    primary_sheared = np.isfinite(primary).all(axis=1) & (
        np.linalg.norm(np.nan_to_num(primary), axis=1) > tolerance
    )
    secondary_sheared = np.isfinite(secondary).all(axis=1) & (
        np.linalg.norm(np.nan_to_num(secondary), axis=1) > tolerance
    )
    return primary_sheared, secondary_sheared


def _forward_model_delta(bundle, frame, applied, batch_size=65536):
    """Return the model-mean shape change for the matching forward ``0 -> g`` map."""

    applied = np.asarray(applied, dtype=np.float64)
    magnitude = np.linalg.norm(applied, axis=1)
    if not np.isfinite(magnitude).all() or (magnitude <= 0).any():
        raise ValueError("applied primary shear must be finite and nonzero")
    delta = float(np.median(magnitude))
    if not np.allclose(magnitude, delta, rtol=0.0, atol=1.0e-12):
        raise ValueError("primary-only rows do not share one applied-shear magnitude")
    direction = applied / magnitude[:, None]
    metadata = bundle.metadata
    kwargs = dict(
        pixel_rms=0.312,
        pixel_size=0.2,
        zero_mag=30.0,
        psf_fwhm=0.73,
        moffat_beta=2.224,
    )
    zero = build_shifted_context(
        frame, (direction[:, 0], direction[:, 1]), 0.0,
        bundle.condition_preprocessor, metadata["condition_features"], kwargs,
    )
    sheared = build_shifted_context(
        frame, (direction[:, 0], direction[:, 1]), delta,
        bundle.condition_preprocessor, metadata["condition_features"], kwargs,
    )
    scales = np.asarray(bundle.target_transform.scales[:2], dtype=np.float64)
    result = np.empty((len(frame), 2), dtype=np.float64)
    bundle.model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            zero_tensor = torch.as_tensor(
                zero[start:stop], dtype=torch.float32, device=bundle.device
            )
            sheared_tensor = torch.as_tensor(
                sheared[start:stop], dtype=torch.float32, device=bundle.device
            )
            zero_mean = bundle.model._shift(zero_tensor)[:, :2].double().cpu().numpy()
            sheared_mean = bundle.model._shift(sheared_tensor)[:, :2].double().cpu().numpy()
            result[start:stop] = (sheared_mean - zero_mean) * scales
    return result


def _response_from_sufficient(normal, rhs):
    if np.linalg.cond(normal) > 1.0e8:
        raise ValueError("applied-shear directions do not span a stable 2D response")
    return rhs @ np.linalg.inv(normal)


def _bootstrap_primary_only_m(case_sufficient, n_boot, seed):
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=np.float64)
    count = len(case_sufficient)
    for draw in range(n_boot):
        selected = rng.integers(0, count, size=count)
        normal = sum((case_sufficient[index][0] for index in selected), np.zeros((2, 2)))
        simulation_rhs = sum(
            (case_sufficient[index][1] for index in selected), np.zeros((2, 2))
        )
        flow_rhs = sum(
            (case_sufficient[index][2] for index in selected), np.zeros((2, 2))
        )
        simulation = 0.5 * np.trace(_response_from_sufficient(normal, simulation_rhs))
        flow = 0.5 * np.trace(_response_from_sufficient(normal, flow_rhs))
        values[draw] = simulation / flow - 1.0
    return float(values.std(ddof=1))


def audit_primary_only_forward_response(
    model_path: Path,
    g0_catalogue: Path,
    sheared_catalogue: Path,
    response_target: Path,
    output: Path,
    *,
    device="cuda",
    batch_size=65536,
    n_boot=10_000,
    bootstrap_seed=0,
):
    """Score a flow on held-out primary-sheared/secondary-unsheared forward rows."""

    target = np.load(response_target)
    validation_cases = [int(value) for value in target["validation_cases"]]
    zero = _load_leg(g0_catalogue, ZERO_COLUMNS, validation_cases)
    sheared_columns = [
        "case", "input_index", "gamma1_input_p", "gamma2_input_p",
        "gamma1_input_s", "gamma2_input_s", "measured_ngmix_g1", "measured_ngmix_g2",
    ]
    sheared = _load_leg(sheared_catalogue, sheared_columns, validation_cases)
    primary_sheared, secondary_sheared = _shear_state_masks(sheared)
    primary_only = primary_sheared & ~secondary_sheared
    shear_state_counts = {
        "loaded_validation_rows": int(len(sheared)),
        "primary_only_rows_before_match_and_cuts": int(primary_only.sum()),
        "both_sheared_rows_before_match_and_cuts": int(
            (primary_sheared & secondary_sheared).sum()
        ),
        "primary_unsheared_rows_before_match_and_cuts": int((~primary_sheared).sum()),
    }
    sheared = sheared.loc[primary_only].reset_index(drop=True)
    zero_shape = zero[
        ["case", "input_index", "measured_ngmix_g1", "measured_ngmix_g2"]
    ].copy()
    measured = zero_shape.merge(
        sheared,
        on=["case", "input_index"],
        how="inner",
        suffixes=("_zero", "_sheared"),
        validate="one_to_one",
    )
    matched_before_cuts = len(measured)
    frame = measured[["case", "input_index"]].merge(
        zero, on=["case", "input_index"], how="left", validate="one_to_one"
    )
    if frame[ZERO_COLUMNS[2:]].isna().all(axis=1).any():
        raise RuntimeError("matched primary-only rows were absent from the zero-shear frame")

    bundle = load_measurement_model(str(model_path), device=device)
    cuts = bundle.metadata.get("selection_cuts")
    if cuts is None:
        raise ValueError("checkpoint lacks recorded selection_cuts")
    keep = (
        (frame["r_input_p"] > cuts[1][0]) & (frame["r_input_p"] < cuts[1][1])
        & (frame["Re_input_p"] > cuts[3][0]) & (frame["Re_input_p"] < cuts[3][1])
        & (((frame["distance"] > cuts[4][0]) & (frame["distance"] < cuts[4][1]))
           | (~frame["neighbored"].astype(bool)))
    )
    frame = frame.loc[keep].reset_index(drop=True)
    measured = frame[["case", "input_index"]].merge(
        measured, on=["case", "input_index"], how="left", validate="one_to_one"
    )
    applied = measured[["gamma1_input_p", "gamma2_input_p"]].to_numpy(float)
    simulation_delta = measured[
        ["measured_ngmix_g1_sheared", "measured_ngmix_g2_sheared"]
    ].to_numpy(float) - measured[
        ["measured_ngmix_g1_zero", "measured_ngmix_g2_zero"]
    ].to_numpy(float)
    flow_delta = _forward_model_delta(bundle, frame, applied, batch_size=batch_size)

    case_sufficient = []
    per_case = {}
    for case in validation_cases:
        selected = measured["case"].astype(int).to_numpy() == case
        if selected.sum() < 100:
            raise RuntimeError(f"validation case {case} has fewer than 100 primary-only rows")
        gamma = applied[selected]
        normal = gamma.T @ gamma
        simulation_rhs = simulation_delta[selected].T @ gamma
        flow_rhs = flow_delta[selected].T @ gamma
        simulation_matrix = _response_from_sufficient(normal, simulation_rhs)
        flow_matrix = _response_from_sufficient(normal, flow_rhs)
        simulation_response = float(0.5 * np.trace(simulation_matrix))
        flow_response = float(0.5 * np.trace(flow_matrix))
        per_case[str(case)] = {
            "n_rows": int(selected.sum()),
            "simulation_matrix": simulation_matrix.tolist(),
            "flow_matrix": flow_matrix.tolist(),
            "simulation_response": simulation_response,
            "flow_response": flow_response,
            "m": simulation_response / flow_response - 1.0,
        }
        case_sufficient.append((normal, simulation_rhs, flow_rhs))
        print(f"case {case}: primary-only rows={int(selected.sum()):,}", flush=True)

    normal = sum((value[0] for value in case_sufficient), np.zeros((2, 2)))
    simulation_rhs = sum((value[1] for value in case_sufficient), np.zeros((2, 2)))
    flow_rhs = sum((value[2] for value in case_sufficient), np.zeros((2, 2)))
    simulation_matrix = _response_from_sufficient(normal, simulation_rhs)
    flow_matrix = _response_from_sufficient(normal, flow_rhs)
    simulation_response = float(0.5 * np.trace(simulation_matrix))
    flow_response = float(0.5 * np.trace(flow_matrix))
    result = {
        "model": str(model_path),
        "g0_catalogue": str(g0_catalogue),
        "sheared_catalogue": str(sheared_catalogue),
        "response_target": str(response_target),
        "population": "seed-501 validation cases; primary sheared; secondary unsheared",
        "difference_convention": "forward 0->g on simulation and model sides",
        "matrix_convention": "rows=measured e1/e2; columns=applied g1/g2",
        "validation_cases": validation_cases,
        "shear_state_counts": shear_state_counts,
        "matched_rows_before_model_cuts": int(matched_before_cuts),
        "n_rows": int(len(frame)),
        "applied_shear_magnitude": float(np.median(np.linalg.norm(applied, axis=1))),
        "simulation_matrix": simulation_matrix.tolist(),
        "flow_matrix": flow_matrix.tolist(),
        "simulation_response": simulation_response,
        "flow_response": flow_response,
        "m": simulation_response / flow_response - 1.0,
        "flow_over_simulation_minus_one": flow_response / simulation_response - 1.0,
        "bootstrap_standard_error_m": _bootstrap_primary_only_m(
            case_sufficient, n_boot=n_boot, seed=bootstrap_seed
        ),
        "bootstrap_replicates": int(n_boot),
        "bootstrap_seed": int(bootstrap_seed),
        "per_case": per_case,
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "n_rows", "simulation_matrix", "flow_matrix", "simulation_response",
        "flow_response", "m", "bootstrap_standard_error_m",
    )}, indent=2))
    return output


def audit_flow_response(
    model_path: Path,
    g0_catalogue: Path,
    sheared_catalogue: Path,
    response_target: Path,
    output: Path,
    *,
    device="cuda",
    delta=0.02,
):
    """Compare model and simulation matrices on target-declared validation cases."""

    target = np.load(response_target)
    validation_cases = [int(value) for value in target["validation_cases"]]
    zero = _load_leg(g0_catalogue, ZERO_COLUMNS, validation_cases)
    sheared = _load_leg(
        sheared_catalogue,
        ["case", "input_index", "gamma1_input_p", "gamma2_input_p",
         "measured_ngmix_g1", "measured_ngmix_g2"],
        validation_cases,
    )
    # Reuse the zero-leg scan: the source is tens of millions of rows, so a
    # second pass merely to recover the two measured columns is material I/O.
    zero_shape = zero[
        ["case", "input_index", "measured_ngmix_g1", "measured_ngmix_g2"]
    ].copy()
    measured = zero_shape.merge(
        sheared,
        on=["case", "input_index"],
        how="inner",
        suffixes=("_zero", "_sheared"),
        validate="one_to_one",
    )
    common = measured[["case", "input_index"]]
    frame = common.merge(zero, on=["case", "input_index"], how="left", validate="one_to_one")
    if frame[ZERO_COLUMNS[2:]].isna().all(axis=1).any():
        raise RuntimeError("matched response rows were absent from the zero-shear flow frame")

    bundle = load_measurement_model(str(model_path), device=device)
    cuts = bundle.metadata.get("selection_cuts")
    if cuts is None:
        raise ValueError("checkpoint lacks recorded selection_cuts")
    keep = (
        (frame["r_input_p"] > cuts[1][0]) & (frame["r_input_p"] < cuts[1][1])
        & (frame["Re_input_p"] > cuts[3][0]) & (frame["Re_input_p"] < cuts[3][1])
        & (((frame["distance"] > cuts[4][0]) & (frame["distance"] < cuts[4][1]))
           | (~frame["neighbored"].astype(bool)))
    )
    frame = frame.loc[keep].reset_index(drop=True)
    measured = frame[["case", "input_index"]].merge(
        measured, on=["case", "input_index"], how="left", validate="one_to_one"
    )

    sim_case = []
    model_case = []
    used_cases = []
    for case in validation_cases:
        select = frame["case"].astype(int).to_numpy() == case
        if select.sum() < 100:
            continue
        sim_select = measured["case"].astype(int).to_numpy() == case
        delta_e = measured.loc[sim_select, [
            "measured_ngmix_g1_sheared", "measured_ngmix_g2_sheared",
        ]].to_numpy(float) - measured.loc[sim_select, [
            "measured_ngmix_g1_zero", "measured_ngmix_g2_zero",
        ]].to_numpy(float)
        applied = measured.loc[sim_select, ["gamma1_input_p", "gamma2_input_p"]].to_numpy(float)
        sim_case.append(estimate_response_matrix(delta_e, applied))
        model_case.append(_mean_model_matrix(bundle, frame.loc[select].reset_index(drop=True), delta))
        used_cases.append(case)
        print(f"case {case}: rows={int(select.sum()):,}", flush=True)

    sim_case = np.asarray(sim_case)
    model_case = np.asarray(model_case)
    if len(used_cases) != len(validation_cases):
        raise RuntimeError(
            f"audit used {len(used_cases)} of {len(validation_cases)} validation cases"
        )
    result = {
        "model": str(model_path),
        "g0_catalogue": str(g0_catalogue),
        "sheared_catalogue": str(sheared_catalogue),
        "response_target": str(response_target),
        "matrix_convention": "rows=measured e1/e2; columns=applied g1/g2",
        "validation_cases": used_cases,
        "n_rows": int(len(frame)),
        "delta": float(delta),
        "simulation_matrix": sim_case.mean(axis=0).tolist(),
        "simulation_case_standard_error": (
            sim_case.std(axis=0, ddof=1) / np.sqrt(len(sim_case))
        ).tolist(),
        "model_matrix": model_case.mean(axis=0).tolist(),
        "model_case_standard_error": (
            model_case.std(axis=0, ddof=1) / np.sqrt(len(model_case))
        ).tolist(),
        "model_minus_simulation": (model_case.mean(axis=0) - sim_case.mean(axis=0)).tolist(),
        "per_case_simulation_matrix": sim_case.tolist(),
        "per_case_model_matrix": model_case.tolist(),
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "n_rows", "simulation_matrix", "model_matrix", "model_minus_simulation"
    )}, indent=2))
    return output


__all__ = ["audit_flow_response", "audit_primary_only_forward_response"]
