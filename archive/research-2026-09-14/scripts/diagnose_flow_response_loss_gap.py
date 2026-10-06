"""Decompose the plain-flow training-versus-audit response-loss gap.

The response trainer is supervised by a matrix grid estimated on 160 cases,
whereas the held-out audit reconstructs a new matrix in each cell from 40
cases.  This script puts the target, held-out ruler, and frozen model on the
same rows and reports the exact quadratic decomposition

    L(model, heldout) = L(model, target) + L(target, heldout) + cross.

It also compares the trainer's finite antithetic-normal mean estimator with the
Sobol-QMC audit estimator on the same frozen model and rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.flow_training import complete_flow_context_means
from sbsi.measurement_model import load_measurement_model
from sbsi.training import build_shifted_context
from scripts.audit_flow_derivative_losses import (
    FLOW_COLUMNS,
    RESCALE_KWARGS,
    _full_model_derivatives,
    _load_leg,
    _response_matrix,
)


KEYS = ["case", "input_index"]
BASE_COLUMNS = FLOW_COLUMNS


def _bin_ids(frame: pd.DataFrame, target: dict[str, np.ndarray]) -> np.ndarray:
    shape = target["counts"].shape
    fi = np.clip(
        np.digitize(frame["r_input_p"], target["edges_flux"]) - 1,
        0,
        shape[0] - 1,
    )
    si = np.clip(
        np.digitize(frame["Re_input_p"], target["edges_size"]) - 1,
        0,
        shape[1] - 1,
    )
    ci = np.clip(
        np.digitize(frame["r_blend"], target["edges_crowd"]) - 1,
        0,
        shape[2] - 1,
    )
    return ((fi * shape[1] + si) * shape[2] + ci).astype(np.int64)


def _load_matched(
    zero_path: Path,
    sheared_path: Path,
    cases: np.ndarray,
    primary_mag_max: float,
    primary_re_min: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zero = _load_leg(zero_path, BASE_COLUMNS, cases)
    sheared = _load_leg(sheared_path, BASE_COLUMNS, cases)
    sheared = sheared.rename(
        columns={
            "measured_ngmix_g1": "measured_ngmix_g1_sheared",
            "measured_ngmix_g2": "measured_ngmix_g2_sheared",
            "measured_mag_auto": "measured_mag_auto_sheared",
            "measured_flux_radius": "measured_flux_radius_sheared",
            "gamma1_input_p": "gamma1_input_p_sheared",
            "gamma2_input_p": "gamma2_input_p_sheared",
        }
    )
    keep_sheared = [
        *KEYS,
        "measured_ngmix_g1_sheared",
        "measured_ngmix_g2_sheared",
        "measured_mag_auto_sheared",
        "measured_flux_radius_sheared",
        "gamma1_input_p_sheared",
        "gamma2_input_p_sheared",
    ]
    frame = zero.merge(
        sheared[keep_sheared], on=KEYS, how="inner", validate="one_to_one"
    )
    true_domain = (
        (frame["r_input_p"] > 18.0)
        & (frame["r_input_p"] < primary_mag_max)
        & (frame["Re_input_p"] > primary_re_min)
        & (frame["Re_input_p"] < 1.5)
        & (
            ((frame["distance"] > 0.0) & (frame["distance"] < 5.0))
            | (~frame["neighbored"].astype(bool))
        )
    )
    shape_finite = np.isfinite(
        frame[
            [
                "r_input_p",
                "Re_input_p",
                "r_blend",
                "measured_ngmix_g1",
                "measured_ngmix_g2",
                "measured_ngmix_g1_sheared",
                "measured_ngmix_g2_sheared",
                "gamma1_input_p_sheared",
                "gamma2_input_p_sheared",
            ]
        ]
    ).all(axis=1)
    target_domain = frame.loc[true_domain & shape_finite].reset_index(drop=True)
    joint_finite = np.isfinite(
        target_domain[
            [
                "measured_mag_auto",
                "measured_flux_radius",
                "measured_mag_auto_sheared",
                "measured_flux_radius_sheared",
            ]
        ]
    ).all(axis=1)
    joint_finite &= (target_domain["measured_flux_radius"] > 0.0) & (
        target_domain["measured_flux_radius_sheared"] > 0.0
    )
    audit_domain = target_domain.loc[joint_finite].reset_index(drop=True)
    return target_domain, audit_domain


def _cell_matrices(
    frame: pd.DataFrame,
    target: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    bin_id = _bin_ids(frame, target)
    n_cells = int(np.prod(target["counts"].shape))
    matrices = np.full((n_cells, 2, 2), np.nan, dtype=np.float64)
    counts = np.bincount(bin_id, minlength=n_cells).astype(np.int64)
    delta = frame[
        ["measured_ngmix_g1_sheared", "measured_ngmix_g2_sheared"]
    ].to_numpy(float) - frame[["measured_ngmix_g1", "measured_ngmix_g2"]].to_numpy(float)
    applied = frame[
        ["gamma1_input_p_sheared", "gamma2_input_p_sheared"]
    ].to_numpy(float)
    for cell in np.flatnonzero(counts > 0):
        selected = bin_id == cell
        matrices[cell] = _response_matrix(delta[selected], applied[selected])
    return matrices, counts


def loss_decomposition(
    model: np.ndarray,
    target: np.ndarray,
    simulation: np.ndarray,
    counts: np.ndarray,
    *,
    min_count: int = 20,
) -> dict[str, float | int]:
    """Weighted Frobenius loss and its exact target-mediated decomposition."""

    model = np.asarray(model, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    simulation = np.asarray(simulation, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    valid = (
        (counts >= min_count)
        & np.isfinite(model).all(axis=(1, 2))
        & np.isfinite(target).all(axis=(1, 2))
        & np.isfinite(simulation).all(axis=(1, 2))
    )
    weight = counts[valid]
    mt = model[valid] - target[valid]
    tv = target[valid] - simulation[valid]
    mv = mt + tv
    norm = 2.0 * weight.sum()
    model_target = float((weight * np.square(mt).sum(axis=(1, 2))).sum() / norm)
    target_simulation = float((weight * np.square(tv).sum(axis=(1, 2))).sum() / norm)
    model_simulation = float((weight * np.square(mv).sum(axis=(1, 2))).sum() / norm)
    cross = float((weight * (mt * tv).sum(axis=(1, 2))).sum() / weight.sum())
    return {
        "cells": int(valid.sum()),
        "rows": int(weight.sum()),
        "model_vs_target": model_target,
        "target_vs_simulation": target_simulation,
        "cross_term": cross,
        "model_vs_simulation": model_simulation,
        "closure_error": model_simulation - model_target - target_simulation - cross,
    }


def _target_only_loss(
    target: np.ndarray,
    simulation: np.ndarray,
    counts: np.ndarray,
    min_count: int = 20,
) -> dict[str, float | int]:
    valid = (
        (counts >= min_count)
        & np.isfinite(target).all(axis=(1, 2))
        & np.isfinite(simulation).all(axis=(1, 2))
    )
    weight = counts[valid].astype(np.float64)
    error = target[valid] - simulation[valid]
    return {
        "cells": int(valid.sum()),
        "rows": int(weight.sum()),
        "loss": float(
            (weight * np.square(error).sum(axis=(1, 2))).sum()
            / (2.0 * weight.sum())
        ),
    }


def _model_cell_means(
    derivatives: np.ndarray,
    bin_id: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    shape = np.asarray(derivatives[:, :2, :], dtype=np.float64)
    counts = np.bincount(bin_id, minlength=n_cells).astype(np.int64)
    result = np.full((n_cells, 2, 2), np.nan, dtype=np.float64)
    for cell in np.flatnonzero(counts > 0):
        result[cell] = shape[bin_id == cell].mean(axis=0)
    return result, counts


def _antithetic_derivatives(
    bundle,
    frame: pd.DataFrame,
    delta: float,
    batch_size: int,
    draws: int,
    seed: int,
) -> np.ndarray:
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
    result = np.empty((len(frame), len(scales), 2), dtype=np.float64)
    bundle.model.eval()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(frame), batch_size)):
            stop = min(start + batch_size, len(frame))
            tensors = [
                torch.as_tensor(value[start:stop], dtype=torch.float32, device=bundle.device)
                for value in contexts
            ]
            p1, m1, p2, m2 = complete_flow_context_means(
                bundle.model, tensors, draws, seed=seed + batch_index
            )
            result[start:stop, :, 0] = (
                (p1 - m1).double().cpu().numpy() * scales / (2.0 * delta)
            )
            result[start:stop, :, 1] = (
                (p2 - m2).double().cpu().numpy() * scales / (2.0 * delta)
            )
    return result


def _top_cell_rows(
    model: np.ndarray,
    target_matrix: np.ndarray,
    simulation: np.ndarray,
    counts: np.ndarray,
    target: dict[str, np.ndarray],
    n_top: int = 15,
) -> list[dict[str, object]]:
    total = counts.sum()
    contribution = (
        counts * np.square(model - simulation).sum(axis=(1, 2)) / (2.0 * total)
    )
    shape = target["counts"].shape
    rows = []
    for cell in np.argsort(np.nan_to_num(contribution, nan=-np.inf))[::-1][:n_top]:
        fi = int(cell // (shape[1] * shape[2]))
        rem = int(cell % (shape[1] * shape[2]))
        si, ci = divmod(rem, shape[2])
        rows.append(
            {
                "cell": int(cell),
                "flux_bin": fi,
                "size_bin": int(si),
                "crowd_bin": int(ci),
                "count": int(counts[cell]),
                "audit_loss_contribution": float(contribution[cell]),
                "model_matrix": model[cell].tolist(),
                "training_target_matrix": target_matrix[cell].tolist(),
                "heldout_simulation_matrix": simulation[cell].tolist(),
            }
        )
    return rows


def diagnose(args) -> dict[str, object]:
    with np.load(args.response_target, allow_pickle=False) as payload:
        target = {key: np.asarray(payload[key]) for key in payload.files}
    target_matrix = target["Rsim_matrix"].reshape(-1, 2, 2).astype(np.float64)
    train_cases = target["train_cases"].astype(np.int64)
    validation_cases = target["validation_cases"].astype(np.int64)
    primary_mag_max = float(target["primary_mag_max"])
    primary_re_min = float(target["primary_re_min"])

    train_target_domain, train_audit_domain = _load_matched(
        args.g0_catalogue,
        args.sheared_catalogue,
        train_cases,
        primary_mag_max,
        primary_re_min,
    )
    val_target_domain, val_audit_domain = _load_matched(
        args.g0_catalogue,
        args.sheared_catalogue,
        validation_cases,
        primary_mag_max,
        primary_re_min,
    )
    train_matrix, train_counts = _cell_matrices(train_target_domain, target)
    val_target_matrix, val_target_counts = _cell_matrices(val_target_domain, target)
    val_audit_matrix, val_audit_counts = _cell_matrices(val_audit_domain, target)

    if len(val_audit_domain) > args.sample_rows:
        rng = np.random.default_rng(args.sample_seed)
        index = np.sort(
            rng.choice(len(val_audit_domain), size=args.sample_rows, replace=False)
        )
        sample = val_audit_domain.iloc[index].reset_index(drop=True)
    else:
        sample = val_audit_domain.copy()
    sample_matrix, sample_counts = _cell_matrices(sample, target)
    sample_bin = _bin_ids(sample, target)
    n_cells = len(target_matrix)

    bundle = load_measurement_model(str(args.model), device=args.device)
    estimators: dict[str, object] = {}
    qmc = _full_model_derivatives(
        bundle,
        sample,
        args.delta,
        args.batch_size,
        args.qmc_draws,
        args.sample_seed,
        difference="central",
    )
    qmc_cells, qmc_counts = _model_cell_means(qmc, sample_bin, n_cells)
    estimators[f"sobol_qmc_{args.qmc_draws}"] = {
        **loss_decomposition(
            qmc_cells, target_matrix, sample_matrix, qmc_counts, min_count=20
        ),
        "mean_response_matrix": qmc[:, :2, :].mean(axis=0).tolist(),
    }
    for draws in args.antithetic_draws:
        derivative = _antithetic_derivatives(
            bundle,
            sample,
            args.delta,
            args.batch_size,
            draws,
            args.sample_seed,
        )
        cells, counts = _model_cell_means(derivative, sample_bin, n_cells)
        estimators[f"antithetic_normal_{draws}"] = {
            **loss_decomposition(
                cells, target_matrix, sample_matrix, counts, min_count=20
            ),
            "mean_response_matrix": derivative[:, :2, :].mean(axis=0).tolist(),
        }

    result = {
        "model": str(args.model),
        "response_target": str(args.response_target),
        "definitions": {
            "training_target": "matrix grid estimated on matched finite-shape rows in 160 training cases",
            "target_domain_validation": "same response-target domain on the 40 held-out cases",
            "audit_domain_validation": "target domain plus finite positive photometry in both legs",
            "loss": "count-weighted squared Frobenius error divided by 2",
        },
        "rows": {
            "train_target_domain": int(len(train_target_domain)),
            "train_audit_domain": int(len(train_audit_domain)),
            "validation_target_domain": int(len(val_target_domain)),
            "validation_audit_domain": int(len(val_audit_domain)),
            "validation_sample": int(len(sample)),
        },
        "saved_target_reproduction": _target_only_loss(
            target_matrix, train_matrix, train_counts, min_count=20
        ),
        "target_vs_validation_target_domain": _target_only_loss(
            target_matrix, val_target_matrix, val_target_counts, min_count=20
        ),
        "target_vs_validation_audit_domain": _target_only_loss(
            target_matrix, val_audit_matrix, val_audit_counts, min_count=20
        ),
        "target_vs_validation_sample": _target_only_loss(
            target_matrix, sample_matrix, sample_counts, min_count=20
        ),
        "estimators": estimators,
        "top_qmc_audit_cells": _top_cell_rows(
            qmc_cells,
            target_matrix,
            sample_matrix,
            qmc_counts,
            target,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--g0-catalogue", type=Path, required=True)
    parser.add_argument("--sheared-catalogue", type=Path, required=True)
    parser.add_argument("--response-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sample-rows", type=int, default=200_000)
    parser.add_argument("--sample-seed", type=int, default=8301)
    parser.add_argument("--qmc-draws", type=int, default=64)
    parser.add_argument("--antithetic-draws", type=int, nargs="*", default=[16, 64])
    diagnose(parser.parse_args(argv))


if __name__ == "__main__":
    main()
