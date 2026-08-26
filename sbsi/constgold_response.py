"""Direct ConstGold response closure for a measurement-flow checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import torch

from .coordinates import ellipticity_from_axis_ratio_angle
from .measurement_model import load_measurement_model
from .preprocessing import source_select_selection
from .training import build_shifted_context


CONSTGOLD_COLUMNS = (
    "case", "input_index", "neighbored", "distance",
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s",
    "axis_ratio_input_p", "axis_ratio_input_s",
    "position_angle_input_p", "position_angle_input_s",
    "polarization_angle", "applied_g1", "applied_g2",
    "measured_e1_plus", "measured_e2_plus",
    "measured_e1_minus", "measured_e2_minus",
)
RESCALE_KWARGS = {
    "pixel_rms": 0.312,
    "pixel_size": 0.2,
    "zero_mag": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
}


def _keys(case, input_index):
    case = np.asarray(case, dtype=np.int64)
    input_index = np.asarray(input_index, dtype=np.int64)
    if (case < 0).any() or (input_index < 0).any() or (input_index >= 2**32).any():
        raise ValueError("case/input_index cannot be encoded as nonnegative 32-bit keyed rows")
    return (case << np.int64(32)) | input_index


@dataclass(frozen=True)
class KeyedLookup:
    """Sorted integer-key lookup with explicit unmatched-row reporting."""

    keys: np.ndarray
    values: dict[str, np.ndarray]

    @classmethod
    def load(cls, paths, value_columns, *, min_case, max_case):
        parts = []
        required = ["case", "input_index", *value_columns]
        for path in paths:
            path = Path(path)
            with ipc.open_file(pa.memory_map(str(path))) as reader:
                missing = sorted(set(required) - set(reader.schema.names))
                if missing:
                    raise KeyError(f"{path} lacks required columns: {missing}")
                for batch_index in range(reader.num_record_batches):
                    table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(required)
                    keep = pc.and_(
                        pc.greater_equal(table["case"], pa.scalar(min_case)),
                        pc.less(table["case"], pa.scalar(max_case)),
                    )
                    table = table.filter(keep)
                    if table.num_rows:
                        parts.append(table.to_pandas())
        if not parts:
            raise RuntimeError("keyed lookup contains no requested cases")
        frame = pd.concat(parts, ignore_index=True)
        key = _keys(frame["case"], frame["input_index"])
        order = np.argsort(key, kind="stable")
        key = key[order]
        if len(key) > 1 and np.any(key[1:] == key[:-1]):
            raise ValueError("keyed lookup has duplicate (case,input_index) rows")
        values = {
            name: frame[name].to_numpy(dtype=np.float64, copy=False)[order]
            for name in value_columns
        }
        return cls(key, values)

    def match(self, case, input_index):
        query = _keys(case, input_index)
        position = np.searchsorted(self.keys, query)
        clipped = np.minimum(position, len(self.keys) - 1)
        matched = (position < len(self.keys)) & (self.keys[clipped] == query)
        values = {
            name: array[clipped]
            for name, array in self.values.items()
        }
        return matched, values


@dataclass
class _CaseAccumulator:
    count: int = 0
    simulation_sum: float = 0.0
    flow_sum: float = 0.0
    blend_sum: float = 0.0
    simulation_cross_sum: float = 0.0
    flow_cross_sum: float = 0.0
    normal: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=np.float64))
    simulation_rhs: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 2), dtype=np.float64)
    )
    flow_rhs: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=np.float64))

    def add(self, simulation_delta, flow_delta, applied, blend):
        magnitude = np.linalg.norm(applied, axis=1)
        direction = applied / magnitude[:, None]
        perpendicular = np.column_stack((-direction[:, 1], direction[:, 0]))
        denominator = 2.0 * magnitude
        self.count += len(applied)
        self.simulation_sum += float(
            np.sum(np.einsum("ij,ij->i", simulation_delta, direction) / denominator)
        )
        self.flow_sum += float(
            np.sum(np.einsum("ij,ij->i", flow_delta, direction) / denominator)
        )
        self.simulation_cross_sum += float(
            np.sum(np.einsum("ij,ij->i", simulation_delta, perpendicular) / denominator)
        )
        self.flow_cross_sum += float(
            np.sum(np.einsum("ij,ij->i", flow_delta, perpendicular) / denominator)
        )
        self.blend_sum += float(np.sum(blend))
        applied_difference = 2.0 * applied
        self.normal += applied_difference.T @ applied_difference
        self.simulation_rhs += simulation_delta.T @ applied_difference
        self.flow_rhs += flow_delta.T @ applied_difference

    def summary(self):
        if self.count <= 0:
            raise RuntimeError("cannot summarize an empty ConstGold case")
        matrix_rank = int(np.linalg.matrix_rank(self.normal))
        simulation_matrix = None
        flow_matrix = None
        model_matrix = None
        if matrix_rank == 2:
            inverse = np.linalg.inv(self.normal)
            simulation_matrix_array = self.simulation_rhs @ inverse
            flow_matrix_array = self.flow_rhs @ inverse
            simulation_matrix = simulation_matrix_array.tolist()
            flow_matrix = flow_matrix_array.tolist()
            model_matrix = (flow_matrix_array + self.blend_sum / self.count * np.eye(2)).tolist()
        simulation = self.simulation_sum / self.count
        flow = self.flow_sum / self.count
        blend = self.blend_sum / self.count
        return {
            "n_rows": self.count,
            "simulation_response": simulation,
            "flow_response": flow,
            "blend_response": blend,
            "model_response": flow + blend,
            "m": simulation / (flow + blend) - 1.0,
            "simulation_cross_response": self.simulation_cross_sum / self.count,
            "flow_cross_response": self.flow_cross_sum / self.count,
            "applied_shear_matrix_rank": matrix_rank,
            "simulation_matrix": simulation_matrix,
            "flow_matrix": flow_matrix,
            "model_matrix": model_matrix,
        }


def _iter_constgold(path, *, min_case, max_case):
    with ipc.open_file(pa.memory_map(str(path))) as reader:
        missing = sorted(set(CONSTGOLD_COLUMNS) - set(reader.schema.names))
        if missing:
            raise KeyError(f"{path} lacks required columns: {missing}")
        for batch_index in range(reader.num_record_batches):
            table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(
                CONSTGOLD_COLUMNS
            )
            keep = pc.and_(
                pc.greater_equal(table["case"], pa.scalar(min_case)),
                pc.less(table["case"], pa.scalar(max_case)),
            )
            table = table.filter(keep)
            if table.num_rows:
                yield table.to_pandas()


def _flow_delta(bundle, frame, applied, *, batch_size):
    magnitude = np.linalg.norm(applied, axis=1)
    if not np.isfinite(magnitude).all() or (magnitude <= 0).any():
        raise ValueError("ConstGold applied shear must be finite and nonzero")
    if not np.allclose(magnitude, np.median(magnitude), rtol=0.0, atol=1.0e-12):
        raise ValueError("ConstGold batch does not have one common shear magnitude")
    direction = applied / magnitude[:, None]
    delta = float(np.median(magnitude))
    plus = build_shifted_context(
        frame, (direction[:, 0], direction[:, 1]), delta,
        bundle.condition_preprocessor, bundle.metadata["condition_features"],
        RESCALE_KWARGS,
    )
    minus = build_shifted_context(
        frame, (direction[:, 0], direction[:, 1]), -delta,
        bundle.condition_preprocessor, bundle.metadata["condition_features"],
        RESCALE_KWARGS,
    )
    scales = np.asarray(bundle.target_transform.scales[:2], dtype=np.float64)
    result = np.empty((len(frame), 2), dtype=np.float64)
    bundle.model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            p = torch.as_tensor(plus[start:stop], dtype=torch.float32, device=bundle.device)
            m = torch.as_tensor(minus[start:stop], dtype=torch.float32, device=bundle.device)
            pmean = bundle.model._shift(p)[:, :2].double().cpu().numpy()
            mmean = bundle.model._shift(m)[:, :2].double().cpu().numpy()
            result[start:stop] = (pmean - mmean) * scales
    return result


def _bootstrap_m(case_summaries, *, n_boot, seed):
    cases = sorted(case_summaries)
    counts = np.asarray([case_summaries[case].count for case in cases], dtype=np.float64)
    simulation = np.asarray(
        [case_summaries[case].simulation_sum for case in cases], dtype=np.float64
    )
    flow = np.asarray([case_summaries[case].flow_sum for case in cases], dtype=np.float64)
    blend = np.asarray([case_summaries[case].blend_sum for case in cases], dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        selected = rng.integers(0, len(cases), size=len(cases))
        n = counts[selected].sum()
        values[index] = simulation[selected].sum() / (
            flow[selected].sum() + blend[selected].sum()
        ) - 1.0
        if n <= 0:
            raise RuntimeError("empty case bootstrap draw")
    return float(values.std(ddof=1))


def evaluate_constgold_response(
    model_path,
    catalogue_path,
    blend_lookup_paths,
    crowd_lookup_paths,
    *,
    min_case=40,
    max_case=140,
    batch_size=65536,
    device="cuda",
    n_boot=10_000,
    bootstrap_seed=0,
):
    """Evaluate antithetic flow+blend closure on the registered ConstGold population."""

    bundle = load_measurement_model(str(model_path), device=device)
    cuts = bundle.metadata.get("selection_cuts")
    if cuts is None:
        raise ValueError("checkpoint lacks selection_cuts")
    blend_lookup = KeyedLookup.load(
        blend_lookup_paths, ["R_blend"], min_case=min_case, max_case=max_case
    )
    crowd_columns = ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"]
    crowd_lookup = KeyedLookup.load(
        crowd_lookup_paths, crowd_columns, min_case=min_case, max_case=max_case
    )

    per_case: dict[int, _CaseAccumulator] = {}
    selected_before_lookup = 0
    dropped_blend = 0
    dropped_crowd = 0
    for raw in _iter_constgold(catalogue_path, min_case=min_case, max_case=max_case):
        frame = source_select_selection(raw, cuts=cuts)
        if frame.empty:
            continue
        selected_before_lookup += len(frame)
        blend_matched, blend_values = blend_lookup.match(frame["case"], frame["input_index"])
        dropped_blend += int((~blend_matched).sum())
        frame = frame.loc[blend_matched].reset_index(drop=True)
        blend = blend_values["R_blend"][blend_matched]
        crowd_matched, crowd_values = crowd_lookup.match(frame["case"], frame["input_index"])
        dropped_crowd += int((~crowd_matched).sum())
        frame = frame.loc[crowd_matched].reset_index(drop=True)
        blend = blend[crowd_matched]
        for name in crowd_columns:
            frame[name] = crowd_values[name][crowd_matched]
        if frame.empty:
            continue
        finite = np.isfinite(blend)
        finite &= np.isfinite(frame[crowd_columns].to_numpy(dtype=np.float64)).all(axis=1)
        if not finite.all():
            raise ValueError("matched ConstGold blend/crowding lookup contains non-finite values")

        e1, e2 = ellipticity_from_axis_ratio_angle(
            frame["axis_ratio_input_p"].to_numpy(float),
            frame["position_angle_input_p"].to_numpy(float),
        )
        frame["e1_input_rot0_p"] = e1
        frame["e2_input_rot0_p"] = e2
        frame["gamma1_input_p"] = 0.0
        frame["gamma2_input_p"] = 0.0
        applied = frame[["applied_g1", "applied_g2"]].to_numpy(dtype=np.float64)
        simulation_delta = frame[
            ["measured_e1_plus", "measured_e2_plus"]
        ].to_numpy(dtype=np.float64) - frame[
            ["measured_e1_minus", "measured_e2_minus"]
        ].to_numpy(dtype=np.float64)
        flow_delta = _flow_delta(bundle, frame, applied, batch_size=batch_size)
        finite = (
            np.isfinite(applied).all(axis=1)
            & np.isfinite(simulation_delta).all(axis=1)
            & np.isfinite(flow_delta).all(axis=1)
        )
        if not finite.all():
            raise ValueError("ConstGold model/simulation response contains non-finite values")
        cases = frame["case"].to_numpy(dtype=np.int64)
        for case in np.unique(cases):
            selected = cases == case
            accumulator = per_case.setdefault(int(case), _CaseAccumulator())
            accumulator.add(
                simulation_delta[selected], flow_delta[selected], applied[selected], blend[selected]
            )

    expected_cases = list(range(min_case, max_case))
    if sorted(per_case) != expected_cases:
        missing = sorted(set(expected_cases) - set(per_case))
        raise RuntimeError(f"ConstGold evaluation is missing cases: {missing}")
    total = _CaseAccumulator()
    for accumulator in per_case.values():
        total.count += accumulator.count
        total.simulation_sum += accumulator.simulation_sum
        total.flow_sum += accumulator.flow_sum
        total.blend_sum += accumulator.blend_sum
        total.simulation_cross_sum += accumulator.simulation_cross_sum
        total.flow_cross_sum += accumulator.flow_cross_sum
        total.normal += accumulator.normal
        total.simulation_rhs += accumulator.simulation_rhs
        total.flow_rhs += accumulator.flow_rhs
    result = total.summary()
    result.update({
        "model": str(model_path),
        "catalogue": str(catalogue_path),
        "blend_lookups": [str(path) for path in blend_lookup_paths],
        "crowd_lookups": [str(path) for path in crowd_lookup_paths],
        "case_range": [int(min_case), int(max_case)],
        "selected_before_lookup": int(selected_before_lookup),
        "dropped_unmatched_blend": int(dropped_blend),
        "dropped_unmatched_crowd": int(dropped_crowd),
        "bootstrap_standard_error_m": _bootstrap_m(
            per_case, n_boot=n_boot, seed=bootstrap_seed
        ),
        "bootstrap_replicates": int(n_boot),
        "bootstrap_seed": int(bootstrap_seed),
        "matrix_convention": "rows=measured e1/e2; columns=applied g1/g2",
        "per_case": {
            str(case): per_case[case].summary() for case in sorted(per_case)
        },
    })
    return result


__all__ = ["KeyedLookup", "evaluate_constgold_response"]
