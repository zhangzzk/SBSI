"""Magnitude and size derivative targets for response-aware flow training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


KEY_COLUMNS = ["case", "input_index"]
ZERO_COLUMNS = [
    *KEY_COLUMNS,
    "r_input_p",
    "Re_input_p",
    "r_blend",
    "neighbored",
    "distance",
    "e1_input_rot0_p",
    "e2_input_rot0_p",
    "measured_mag_auto",
    "measured_flux_radius",
]
SHEARED_COLUMNS = [
    *KEY_COLUMNS,
    "gamma1_input_p",
    "gamma2_input_p",
    "measured_mag_auto",
    "measured_flux_radius",
]


def _load_leg(path: Path, columns, cases):
    cases = set(int(case) for case in cases)
    parts = []
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


def centered_slope(y, x):
    """Least-squares slope with a fitted intercept, rejecting non-finite rows."""

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(x)
    if finite.sum() < 2:
        return np.nan
    y = y[finite]
    x = x[finite]
    xc = x - x.mean()
    denominator = float(xc @ xc)
    if denominator <= 0.0:
        return np.nan
    return float((y - y.mean()) @ xc / denominator)


def load_coupling_frame(
    g0_catalogue: Path,
    sheared_catalogue: Path,
    cases,
    *,
    primary_mag_max: float,
    primary_re_min: float,
):
    """Load, match, cut, and derive photometric finite differences for one case set."""

    zero = _load_leg(g0_catalogue, ZERO_COLUMNS, cases).rename(
        columns={
            "measured_mag_auto": "mag_zero",
            "measured_flux_radius": "size_zero",
        }
    )
    sheared = _load_leg(sheared_catalogue, SHEARED_COLUMNS, cases).rename(
        columns={
            "measured_mag_auto": "mag_sheared",
            "measured_flux_radius": "size_sheared",
        }
    )
    before_zero, before_sheared = len(zero), len(sheared)
    frame = zero.merge(sheared, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    print(
        f"coupling matching: g0={before_zero:,}, sheared={before_sheared:,}, "
        f"matched={len(frame):,}, dropped_g0={before_zero-len(frame):,}, "
        f"dropped_sheared={before_sheared-len(frame):,}",
        flush=True,
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
    positive_size = (frame["size_zero"] > 0.0) & (frame["size_sheared"] > 0.0)
    frame["delta_mag"] = frame["mag_sheared"] - frame["mag_zero"]
    frame["delta_log_size"] = np.nan
    frame.loc[positive_size, "delta_log_size"] = (
        np.log(frame.loc[positive_size, "size_sheared"])
        - np.log(frame.loc[positive_size, "size_zero"])
    )
    frame["orientation_shear"] = (
        frame["e1_input_rot0_p"] * frame["gamma1_input_p"]
        + frame["e2_input_rot0_p"] * frame["gamma2_input_p"]
    )
    finite_columns = [
        "r_input_p",
        "Re_input_p",
        "r_blend",
        "delta_mag",
        "delta_log_size",
        "orientation_shear",
    ]
    finite = np.isfinite(frame[finite_columns]).all(axis=1)
    frame = frame.loc[finite].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("no matched finite rows remain after the Flow-E cuts")
    return frame, before_zero, before_sheared


def _estimate_grid(frame, edges_flux, edges_size, edges_crowd, min_count):
    shape = (len(edges_flux) - 1, len(edges_size) - 1, len(edges_crowd) - 1)
    fi = np.clip(np.digitize(frame["r_input_p"], edges_flux) - 1, 0, shape[0] - 1)
    si = np.clip(np.digitize(frame["Re_input_p"], edges_size) - 1, 0, shape[1] - 1)
    ci = np.clip(np.digitize(frame["r_blend"], edges_crowd) - 1, 0, shape[2] - 1)
    bin_id = (fi * shape[1] + si) * shape[2] + ci

    x = frame["orientation_shear"].to_numpy(float)
    delta_mag = frame["delta_mag"].to_numpy(float)
    delta_size = frame["delta_log_size"].to_numpy(float)
    global_mag = centered_slope(delta_mag, x)
    global_size = centered_slope(delta_size, x)
    if not np.isfinite(global_mag) or not np.isfinite(global_size):
        raise ValueError("global coupling slopes are not finite")

    count = np.bincount(bin_id, minlength=int(np.prod(shape))).astype(np.int64)
    coupling_mag = np.full(count.shape, global_mag, dtype=np.float64)
    coupling_size = np.full(count.shape, global_size, dtype=np.float64)
    fallback = count < min_count
    for index in np.flatnonzero(~fallback):
        selected = bin_id == index
        coupling_mag[index] = centered_slope(delta_mag[selected], x[selected])
        coupling_size[index] = centered_slope(delta_size[selected], x[selected])
    if not np.isfinite(coupling_mag).all() or not np.isfinite(coupling_size).all():
        raise ValueError("a populated coupling cell produced a non-finite slope")
    return (
        coupling_mag.reshape(shape),
        coupling_size.reshape(shape),
        count.reshape(shape),
        fallback.reshape(shape),
        global_mag,
        global_size,
    )


def build_coupling_target(
    g0_catalogue: Path,
    sheared_catalogue: Path,
    response_target: Path,
    output: Path,
    *,
    case_role="train",
    min_count=300,
):
    """Build cell-resolved ``d(mag, log-size)/dg`` orientation targets.

    The response target supplies Flow E's exact grid, domain, and grouped case split.
    ``case_role='train'`` is suitable for optimization; ``'validation'`` creates an
    independently measured target for the untouched cases.
    """

    response_target = Path(response_target)
    with np.load(response_target, allow_pickle=False) as shape_target:
        required = {
            "edges_flux",
            "edges_size",
            "edges_crowd",
            "crowd_col",
            "train_cases",
            "validation_cases",
            "primary_mag_max",
            "primary_re_min",
        }
        missing = sorted(required - set(shape_target.files))
        if missing:
            raise KeyError(f"{response_target} lacks required fields: {missing}")
        crowd_col = str(shape_target["crowd_col"].item())
        if crowd_col != "r_blend":
            raise ValueError(f"expected Flow-E crowd column 'r_blend', got {crowd_col!r}")
        if case_role not in {"train", "validation"}:
            raise ValueError("case_role must be 'train' or 'validation'")
        cases = np.asarray(shape_target[f"{case_role}_cases"], dtype=np.int64)
        edges_flux = np.asarray(shape_target["edges_flux"], dtype=np.float64)
        edges_size = np.asarray(shape_target["edges_size"], dtype=np.float64)
        edges_crowd = np.asarray(shape_target["edges_crowd"], dtype=np.float64)
        primary_mag_max = float(shape_target["primary_mag_max"])
        primary_re_min = float(shape_target["primary_re_min"])

    frame, before_zero, before_sheared = load_coupling_frame(
        g0_catalogue,
        sheared_catalogue,
        cases,
        primary_mag_max=primary_mag_max,
        primary_re_min=primary_re_min,
    )
    mag, size, counts, fallback, global_mag, global_size = _estimate_grid(
        frame, edges_flux, edges_size, edges_crowd, min_count
    )

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        edges_flux=edges_flux,
        edges_size=edges_size,
        edges_crowd=edges_crowd,
        crowd_col=np.asarray(crowd_col),
        coupling_mag=mag,
        coupling_size=size,
        counts=counts,
        fallback=fallback,
        b_mag_global=float(global_mag),
        b_size_global=float(global_size),
        case_role=np.asarray(case_role),
        cases=cases,
        min_count=int(min_count),
        matched_rows=int(len(frame)),
        unmatched_g0_rows=int(before_zero - len(frame)),
        unmatched_sheared_rows=int(before_sheared - len(frame)),
        primary_mag_max=primary_mag_max,
        primary_re_min=primary_re_min,
        slope_predictor=np.asarray("e1_input_rot0_p*gamma1_input_p + e2_input_rot0_p*gamma2_input_p"),
        response_target=np.asarray(str(response_target)),
        g0_catalogue=np.asarray(str(g0_catalogue)),
        sheared_catalogue=np.asarray(str(sheared_catalogue)),
    )
    print(
        f"global couplings: mag={global_mag:+.6f}, log_size={global_size:+.6f}",
        flush=True,
    )
    print(
        f"wrote {output}: role={case_role}, grid={'x'.join(map(str, mag.shape))}, "
        f"rows={len(frame):,}, fallback_cells={int(fallback.sum())}/{fallback.size}, "
        f"mag_range={mag.min():+.4f}..{mag.max():+.4f}, "
        f"size_range={size.min():+.4f}..{size.max():+.4f}",
        flush=True,
    )
    return output


__all__ = ["build_coupling_target", "centered_slope", "load_coupling_frame"]
