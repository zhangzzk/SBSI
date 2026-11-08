"""Full-component response targets for measurement-flow regularization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from .training import split_group_values


KEY_COLUMNS = ["case", "input_index"]


def estimate_response_matrix(delta_e, applied_g, weights=None):
    """Weighted least-squares estimate of ``delta_e = R @ applied_g``."""

    delta_e = np.asarray(delta_e, dtype=np.float64)
    applied_g = np.asarray(applied_g, dtype=np.float64)
    if delta_e.ndim != 2 or delta_e.shape[1] != 2:
        raise ValueError("delta_e must have shape (n, 2)")
    if applied_g.shape != delta_e.shape:
        raise ValueError("applied_g must have the same shape as delta_e")
    if weights is None:
        weights = np.ones(len(delta_e), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(delta_e).all(axis=1) & np.isfinite(applied_g).all(axis=1)
    finite &= np.isfinite(weights) & (weights > 0)
    d = delta_e[finite]
    g = applied_g[finite]
    w = weights[finite]
    normal = (g * w[:, None]).T @ g
    if np.linalg.cond(normal) > 1.0e8:
        raise ValueError("applied-shear directions do not span a stable 2D response")
    return (d * w[:, None]).T @ g @ np.linalg.inv(normal)


def _load_leg(path, columns, cases):
    parts = []
    cases = set(int(case) for case in cases)
    with ipc.open_file(path) as reader:
        available = set(reader.schema.names)
        missing = sorted(set(columns) - available)
        if missing:
            raise KeyError(f"{path} lacks required columns: {missing}")
        for index in range(reader.num_record_batches):
            table = pa.Table.from_batches([reader.get_batch(index)]).select(columns)
            frame = table.to_pandas()
            frame = frame[frame["case"].astype(int).isin(cases)]
            if not frame.empty:
                parts.append(frame)
    if not parts:
        raise RuntimeError(f"no requested cases were loaded from {path}")
    frame = pd.concat(parts, ignore_index=True)
    duplicate = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicate.any():
        raise ValueError(f"{path} has {int(duplicate.sum()):,} duplicate keyed rows")
    return frame


def _quantile_edges(values, bins):
    edges = np.quantile(np.asarray(values, dtype=float), np.linspace(0.0, 1.0, bins + 1))
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    if not np.all(np.diff(edges) > 0):
        raise ValueError("quantile grid has repeated edges")
    return edges


def build_response_target(
    g0_catalogue: Path,
    sheared_catalogue: Path,
    output: Path,
    *,
    all_cases=range(200),
    split_seed=501,
    validation_size=0.2,
    n_flux=6,
    n_size=6,
    n_crowd=5,
    min_count=500,
    primary_mag_max=25.8,
    primary_re_min=0.5,
):
    """Build trace and full-matrix response grids from CRN-matched legs."""

    all_cases = sorted(set(int(case) for case in all_cases))
    train_cases, validation_cases = split_group_values(all_cases, split_seed, validation_size)
    train_cases = sorted(int(case) for case in train_cases)
    validation_cases = sorted(int(case) for case in validation_cases)

    base_columns = [
        "case", "input_index", "r_input_p", "Re_input_p", "r_blend",
        "neighbored", "distance", "measured_ngmix_g1", "measured_ngmix_g2",
    ]
    zero = _load_leg(g0_catalogue, base_columns, train_cases).rename(columns={
        "measured_ngmix_g1": "e1_zero", "measured_ngmix_g2": "e2_zero",
    })
    sheared = _load_leg(
        sheared_catalogue,
        [*base_columns, "gamma1_input_p", "gamma2_input_p"],
        train_cases,
    ).rename(columns={
        "measured_ngmix_g1": "e1_sheared", "measured_ngmix_g2": "e2_sheared",
    })
    before_zero, before_sheared = len(zero), len(sheared)
    frame = zero.merge(
        sheared[
            ["case", "input_index", "e1_sheared", "e2_sheared",
             "gamma1_input_p", "gamma2_input_p"]
        ],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    print(
        f"response matching: g0={before_zero:,}, sheared={before_sheared:,}, "
        f"matched={len(frame):,}, dropped_g0={before_zero-len(frame):,}, "
        f"dropped_sheared={before_sheared-len(frame):,}"
    )

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
    frame = frame.loc[keep].reset_index(drop=True)
    delta_e = frame[["e1_sheared", "e2_sheared"]].to_numpy(float) - frame[
        ["e1_zero", "e2_zero"]
    ].to_numpy(float)
    gamma = frame[["gamma1_input_p", "gamma2_input_p"]].to_numpy(float)
    finite = np.isfinite(delta_e).all(axis=1) & np.isfinite(gamma).all(axis=1)
    finite &= np.isfinite(frame[["r_input_p", "Re_input_p", "r_blend"]]).all(axis=1)
    frame = frame.loc[finite].reset_index(drop=True)
    delta_e = delta_e[finite]
    gamma = gamma[finite]
    if len(frame) < min_count:
        raise ValueError("too few matched finite rows for a response target")

    edges_flux = _quantile_edges(frame["r_input_p"], n_flux)
    edges_size = _quantile_edges(frame["Re_input_p"], n_size)
    edges_crowd = _quantile_edges(frame["r_blend"], n_crowd)
    fi = np.clip(np.digitize(frame["r_input_p"], edges_flux) - 1, 0, n_flux - 1)
    si = np.clip(np.digitize(frame["Re_input_p"], edges_size) - 1, 0, n_size - 1)
    ci = np.clip(np.digitize(frame["r_blend"], edges_crowd) - 1, 0, n_crowd - 1)
    bin_id = (fi * n_size + si) * n_crowd + ci

    global_matrix = estimate_response_matrix(delta_e, gamma)
    matrices = np.empty((n_flux * n_size * n_crowd, 2, 2), dtype=np.float64)
    counts = np.bincount(bin_id, minlength=len(matrices)).astype(np.int64)
    fallback = np.zeros(len(matrices), dtype=bool)
    for index in range(len(matrices)):
        selected = bin_id == index
        if counts[index] < min_count:
            matrices[index] = global_matrix
            fallback[index] = True
        else:
            matrices[index] = estimate_response_matrix(delta_e[selected], gamma[selected])
    matrices = matrices.reshape(n_flux, n_size, n_crowd, 2, 2)
    scalar = 0.5 * (matrices[..., 0, 0] + matrices[..., 1, 1])
    counts = counts.reshape(n_flux, n_size, n_crowd)
    fallback = fallback.reshape(n_flux, n_size, n_crowd)

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        edges_flux=edges_flux,
        edges_size=edges_size,
        edges_crowd=edges_crowd,
        crowd_col=np.asarray("r_blend"),
        Rsim=scalar,
        Rsim_matrix=matrices,
        counts=counts,
        fallback=fallback,
        global_R=0.5 * np.trace(global_matrix),
        global_R_matrix=global_matrix,
        primary_mag_max=float(primary_mag_max),
        primary_re_min=float(primary_re_min),
        split_seed=int(split_seed),
        validation_size=float(validation_size),
        train_cases=np.asarray(train_cases, dtype=np.int64),
        validation_cases=np.asarray(validation_cases, dtype=np.int64),
        matched_rows=int(len(frame)),
        g0_catalogue=np.asarray(str(g0_catalogue)),
        sheared_catalogue=np.asarray(str(sheared_catalogue)),
    )
    print(f"global response matrix:\n{global_matrix}")
    print(
        f"wrote {output}: grid={n_flux}x{n_size}x{n_crowd}, "
        f"fallback_cells={int(fallback.sum())}/{fallback.size}, "
        f"train_cases={len(train_cases)}, validation_cases={len(validation_cases)}"
    )
    return output


__all__ = ["build_response_target", "estimate_response_matrix"]
