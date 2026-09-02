#!/usr/bin/env python
"""Audit shape, magnitude, and size derivative losses on Flow E's held-out cases."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sbsi.flow_coupling_target import _load_leg
from sbsi.flow_training import _fold_catalogue_shear_into_shape
from sbsi.measurement_model import add_measurement_target_features, load_measurement_model
from sbsi.preprocessing import rescale
from sbsi.training import build_shifted_context


KEYS = ["case", "input_index"]
FLOW_COLUMNS = [
    *KEYS,
    "Re_input_p",
    "Re_input_s",
    "r_input_p",
    "r_input_s",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "e1_input_rot0_p",
    "e2_input_rot0_p",
    "e1_input_rot0_s",
    "e2_input_rot0_s",
    "gamma1_input_p",
    "gamma2_input_p",
    "gamma1_input_s",
    "gamma2_input_s",
    "neighbored",
    "distance",
    "polarization_angle",
    "nbr_flux_near",
    "nbr_flux_far",
    "nbr_flux_max",
    "r_blend",
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_flux_radius",
]
SHEARED_COLUMNS = [
    *KEYS,
    "gamma1_input_p",
    "gamma2_input_p",
    "gamma1_input_s",
    "gamma2_input_s",
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_flux_radius",
]
RESCALE_KWARGS = dict(
    pixel_rms=0.312,
    pixel_size=0.2,
    zero_mag=30.0,
    psf_fwhm=0.73,
    moffat_beta=2.224,
)


def _response_matrix(delta, applied):
    normal = applied.T @ applied
    if np.linalg.cond(normal) > 1.0e8:
        raise ValueError("applied shear does not span a stable response matrix")
    return delta.T @ applied @ np.linalg.inv(normal)


def _bin_ids(frame, target):
    shape = tuple(int(value) for value in target["counts"].shape)
    fi = np.clip(np.digitize(frame["r_input_p"], target["edges_flux"]) - 1, 0, shape[0] - 1)
    si = np.clip(np.digitize(frame["Re_input_p"], target["edges_size"]) - 1, 0, shape[1] - 1)
    ci = np.clip(np.digitize(frame["r_blend"], target["edges_crowd"]) - 1, 0, shape[2] - 1)
    return (fi * shape[1] + si) * shape[2] + ci


def _model_derivatives(bundle, frame, delta, batch_size):
    contexts = [
        build_shifted_context(
            frame,
            direction,
            sign * delta,
            bundle.condition_preprocessor,
            bundle.metadata["condition_features"],
            RESCALE_KWARGS,
        )
        for direction, sign in (
            ((1.0, 0.0), 1.0),
            ((1.0, 0.0), -1.0),
            ((0.0, 1.0), 1.0),
            ((0.0, 1.0), -1.0),
        )
    ]
    scales = np.asarray(bundle.target_transform.scales, dtype=np.float64)
    derivatives = np.empty((len(frame), 4, 2), dtype=np.float64)
    bundle.model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            tensors = [
                torch.as_tensor(context[start:stop], dtype=torch.float32, device=bundle.device)
                for context in contexts
            ]
            p1, m1, p2, m2 = [bundle.model._shift(value) for value in tensors]
            derivatives[start:stop, :, 0] = (
                (p1 - m1).double().cpu().numpy() * scales / (2.0 * delta)
            )
            derivatives[start:stop, :, 1] = (
                (p2 - m2).double().cpu().numpy() * scales / (2.0 * delta)
            )
    return derivatives


def _nll(bundle, frame, batch_size):
    prepared = rescale(frame.copy(), **RESCALE_KWARGS)
    prepared = add_measurement_target_features(prepared)
    return float(-bundle.log_prob(prepared, batch_size=batch_size).mean())


def audit(model_path, g0_catalogue, sheared_catalogue, response_target, coupling_target,
          output, *, device="cuda", delta=0.02, batch_size=65536):
    with np.load(response_target, allow_pickle=False) as rt:
        validation_cases = np.asarray(rt["validation_cases"], dtype=np.int64)
        primary_mag_max = float(rt["primary_mag_max"])
        primary_re_min = float(rt["primary_re_min"])
        response_grid = {key: np.asarray(rt[key]) for key in (
            "edges_flux", "edges_size", "edges_crowd", "counts"
        )}
    with np.load(coupling_target, allow_pickle=False) as ct:
        if str(ct["case_role"].item()) != "validation":
            raise ValueError("coupling audit target must use case_role='validation'")
        for key in ("edges_flux", "edges_size", "edges_crowd"):
            np.testing.assert_array_equal(ct[key], response_grid[key])
        coupling_mag = np.asarray(ct["coupling_mag"], dtype=np.float64).reshape(-1)
        coupling_size = np.asarray(ct["coupling_size"], dtype=np.float64).reshape(-1)

    zero = _load_leg(g0_catalogue, FLOW_COLUMNS, validation_cases)
    sheared = _load_leg(sheared_catalogue, SHEARED_COLUMNS, validation_cases)
    before_zero, before_sheared = len(zero), len(sheared)
    renamed = sheared.rename(columns={
        name: f"{name}_sheared" for name in SHEARED_COLUMNS if name not in KEYS
    })
    frame = zero.merge(renamed, on=KEYS, how="inner", validate="one_to_one")
    matched = len(frame)
    keep = (
        (frame["r_input_p"] > 18.0)
        & (frame["r_input_p"] < primary_mag_max)
        & (frame["Re_input_p"] > primary_re_min)
        & (frame["Re_input_p"] < 1.5)
        & (
            ((frame["distance"] > 0.0) & (frame["distance"] < 5.0))
            | (~frame["neighbored"].astype(bool))
        )
    )
    finite_columns = [
        "r_input_p", "Re_input_p", "r_blend", "e1_input_rot0_p", "e2_input_rot0_p",
        "gamma1_input_p_sheared", "gamma2_input_p_sheared",
        "measured_ngmix_g1", "measured_ngmix_g2",
        "measured_ngmix_g1_sheared", "measured_ngmix_g2_sheared",
        "measured_mag_auto", "measured_mag_auto_sheared",
        "measured_flux_radius", "measured_flux_radius_sheared",
    ]
    keep &= np.isfinite(frame[finite_columns]).all(axis=1)
    keep &= (frame["measured_flux_radius"] > 0.0) & (frame["measured_flux_radius_sheared"] > 0.0)
    frame = frame.loc[keep].reset_index(drop=True)

    bundle = load_measurement_model(str(model_path), device=device)
    derivatives = _model_derivatives(bundle, frame, delta, batch_size)
    shape_model = derivatives[:, :2, :]
    mag_model = derivatives[:, 2, :]
    size_model = derivatives[:, 3, :]
    bin_id = _bin_ids(frame, response_grid)
    n_cells = int(np.prod(response_grid["counts"].shape))

    applied = frame[["gamma1_input_p_sheared", "gamma2_input_p_sheared"]].to_numpy(float)
    shape_delta = frame[["measured_ngmix_g1_sheared", "measured_ngmix_g2_sheared"]].to_numpy(float)
    shape_delta -= frame[["measured_ngmix_g1", "measured_ngmix_g2"]].to_numpy(float)
    shape_squared_sum = 0.0
    shape_count = 0
    used_cells = 0
    for index in range(n_cells):
        selected = bin_id == index
        count = int(selected.sum())
        if count < 100:
            continue
        simulation = _response_matrix(shape_delta[selected], applied[selected])
        model = shape_model[selected].mean(axis=0)
        shape_squared_sum += count * float(np.square(model - simulation).sum())
        shape_count += count
        used_cells += 1
    shape_loss = shape_squared_sum / (2.0 * shape_count)

    b_mag = coupling_mag[bin_id]
    b_size = coupling_size[bin_id]
    e1 = frame["e1_input_rot0_p"].to_numpy(float)
    e2 = frame["e2_input_rot0_p"].to_numpy(float)
    size_terms = (
        np.square(size_model[:, 0] - b_size * e1)
        + np.square(size_model[:, 1] - b_size * e2)
    )
    magnitude_terms = (
        np.square(mag_model[:, 0] - b_mag * e1)
        + np.square(mag_model[:, 1] - b_mag * e2)
    )
    size_loss = float(size_terms.mean())
    magnitude_loss = float(magnitude_terms.mean())
    theta_loss = size_loss + magnitude_loss

    zero_nll_frame = frame[FLOW_COLUMNS].copy()
    sheared_nll_frame = frame[FLOW_COLUMNS].copy()
    for name in SHEARED_COLUMNS:
        if name not in KEYS:
            sheared_nll_frame[name] = frame[f"{name}_sheared"].to_numpy()
    sheared_nll_frame = _fold_catalogue_shear_into_shape(sheared_nll_frame)
    nll_zero = _nll(bundle, zero_nll_frame, batch_size)
    nll_sheared = _nll(bundle, sheared_nll_frame, batch_size)

    result = {
        "model": str(model_path),
        "g0_catalogue": str(g0_catalogue),
        "sheared_catalogue": str(sheared_catalogue),
        "response_target": str(response_target),
        "coupling_target": str(coupling_target),
        "population": "Flow-E 40 grouped validation cases; matched selected finite rows",
        "validation_cases": validation_cases.tolist(),
        "rows_g0_before_match": before_zero,
        "rows_sheared_before_match": before_sheared,
        "matched_rows_before_cuts": matched,
        "unmatched_g0_rows": before_zero - matched,
        "unmatched_sheared_rows": before_sheared - matched,
        "n_rows": int(len(frame)),
        "shape_cells_used": used_cells,
        "shape_response_loss": float(shape_loss),
        "magnitude_derivative_loss": magnitude_loss,
        "log_size_derivative_loss": size_loss,
        "photometric_derivative_loss": theta_loss,
        "nll_g0": nll_zero,
        "nll_sheared": nll_sheared,
        "nll_balanced": 0.5 * (nll_zero + nll_sheared),
        "weighted_objective_r450_t500": 0.5 * (nll_zero + nll_sheared)
        + 450.0 * shape_loss + 500.0 * theta_loss,
        "weighted_objective_r500_t500": 0.5 * (nll_zero + nll_sheared)
        + 500.0 * shape_loss + 500.0 * theta_loss,
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--g0-catalogue", type=Path, required=True)
    parser.add_argument("--sheared-catalogue", type=Path, required=True)
    parser.add_argument("--response-target", type=Path, required=True)
    parser.add_argument("--coupling-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args(argv)
    audit(
        args.model,
        args.g0_catalogue,
        args.sheared_catalogue,
        args.response_target,
        args.coupling_target,
        args.output,
        device=args.device,
        delta=args.delta,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
