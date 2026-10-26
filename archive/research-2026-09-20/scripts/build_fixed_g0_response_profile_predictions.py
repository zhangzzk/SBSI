#!/usr/bin/env python3
"""Cache staged-flow and v2-emulator responses for notebook profile plots."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES, matched_key_indices
from sbsi.flow_paired_shape import physical_context_means
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from scripts.build_constgold_fixed_g0_blend_lookup import CONDITIONS


BLEND_RESPONSE_COLUMNS = [
    "case",
    "input_index",
    "shear_angle",
    "r_input_p",
    "r_input_s",
    "Re_input_p",
    "Re_input_s",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance",
    "delta_et1",
    "delta_et2",
    "measured_e1_m",
    "measured_e2_m",
    "measured_e1_p",
    "measured_e2_p",
]
BLEND_MEASUREMENT_COLUMNS = [
    "measured_e1_m",
    "measured_e2_m",
    "measured_e1_p",
    "measured_e2_p",
]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--response-catalogue", type=Path, required=True)
    parser.add_argument("--response-shear", type=float, default=0.2)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--emulator-model", type=Path, required=True)
    parser.add_argument("--emulator-metadata", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=7301)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def load_flow_case(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        result = {name: stored[name] for name in stored.files}
    expected = {"case", "input_index", "context", "target", "gamma"}
    if set(result) != expected:
        raise ValueError(f"unexpected fixed-g0 flow arrays in {path}")
    return result


def projected_response(delta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    if delta.shape != gamma.shape or delta.ndim != 2 or delta.shape[1] != 2:
        raise ValueError("aligned two-component response differences required")
    squared = np.einsum("ij,ij->i", gamma, gamma)
    if not np.isfinite(delta).all() or not np.isfinite(gamma).all() or np.any(squared <= 0):
        raise ValueError("finite response differences and nonzero shears required")
    return np.einsum("ij,ij->i", delta, gamma) / squared


@torch.no_grad()
def predict_self_case(
    bundle,
    zero: dict[str, np.ndarray],
    sheared: dict[str, np.ndarray],
    *,
    draws: int,
    batch_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    left, right, counts = matched_key_indices(
        zero["case"],
        zero["input_index"],
        sheared["case"],
        sheared["input_index"],
    )
    gamma = np.asarray(sheared["gamma"][right], dtype=np.float64)
    eligible = np.isfinite(gamma).all(axis=1)
    eligible &= np.einsum("ij,ij->i", gamma, gamma) > 0.0
    left, right, gamma = left[eligible], right[eligible], gamma[eligible]
    zero_context = bundle.context_tensor(
        pd.DataFrame(zero["context"][left], columns=FLOW_FEATURES)
    )
    shear_context = bundle.context_tensor(
        pd.DataFrame(sheared["context"][right], columns=FLOW_FEATURES)
    )
    predicted = np.empty((len(left), 2), dtype=np.float64)
    for block, start in enumerate(range(0, len(left), batch_size)):
        stop = min(start + batch_size, len(left))
        means = physical_context_means(
            bundle.model,
            [zero_context[start:stop], shear_context[start:stop]],
            draws,
            seed + 1_000_003 * block,
        )
        predicted[start:stop] = (
            means[1][:, :2] - means[0][:, :2]
        ).cpu().numpy()
    frame = pd.DataFrame(
        {
            "case": zero["case"][left].astype(np.int16),
            "input_index": zero["input_index"][left].astype(np.int64),
            "R_self_model": projected_response(predicted, gamma),
        }
    )
    if frame.duplicated(["case", "input_index"]).any():
        raise RuntimeError("model self-response cache contains duplicate keys")
    return frame, {
        **counts,
        "response_rows": int(len(frame)),
        "zero_or_nonfinite_shear_dropped": int((~eligible).sum()),
    }


def response_anchor_for_case(path: Path, case: int) -> np.ndarray:
    anchor = pd.read_feather(path, columns=["case", "input_index"])
    selected = anchor.loc[anchor["case"] == case, "input_index"]
    if selected.empty or selected.duplicated().any():
        raise RuntimeError(f"invalid response anchor for case {case}")
    return np.sort(selected.to_numpy(np.int64))


def valid_blend_response_pairs(frame: pd.DataFrame) -> np.ndarray:
    """Reproduce the finite, physical two-leg response-label validity rule."""
    finite = np.isfinite(
        frame[BLEND_RESPONSE_COLUMNS].to_numpy(np.float64, copy=False)
    ).all(axis=1)
    measured = frame[BLEND_MEASUREMENT_COLUMNS].to_numpy(np.float64, copy=False)
    finite &= np.square(measured[:, :2]).sum(axis=1) < 1.0
    finite &= np.square(measured[:, 2:]).sum(axis=1) < 1.0
    return finite


def predict_blend_cases(
    predictor,
    response_catalogue: Path,
    anchor_by_case: dict[int, np.ndarray],
    cases: tuple[int, ...],
    response_shear: float,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Predict on the exact pair rows that define the measured blend target."""
    allowed = np.asarray(sorted(cases), dtype=np.int64)
    partial = []
    counts = {
        case: {
            "anchor_rows": int(len(anchor_by_case[case])),
            "case_window_pair_rows": 0,
            "anchored_pair_rows": 0,
            "invalid_pair_rows_dropped": 0,
            "valid_pair_rows": 0,
        }
        for case in cases
    }
    previous_max = None
    with ipc.open_file(response_catalogue) as reader:
        missing = sorted(set(BLEND_RESPONSE_COLUMNS) - set(reader.schema.names))
        if missing:
            raise KeyError(f"response catalogue lacks {missing}")
        case_column = reader.schema.get_field_index("case")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            case_values = batch.column(case_column).to_numpy(zero_copy_only=False)
            batch_min, batch_max = int(case_values.min()), int(case_values.max())
            if previous_max is not None and batch_min < previous_max:
                raise RuntimeError("response catalogue is not ordered by case")
            previous_max = batch_max
            if batch_max < allowed.min():
                continue
            if batch_min > allowed.max():
                break
            frame = pa.Table.from_batches([batch]).select(
                BLEND_RESPONSE_COLUMNS
            ).to_pandas()
            frame = frame.loc[
                np.isin(frame["case"].to_numpy(np.int64), allowed)
            ].copy()
            if frame.empty:
                continue
            case_values = frame["case"].to_numpy(np.int64)
            anchored = np.zeros(len(frame), dtype=bool)
            for case in np.unique(case_values):
                local = case_values == case
                counts[int(case)]["case_window_pair_rows"] += int(local.sum())
                anchored[local] = np.isin(
                    frame.loc[local, "input_index"].to_numpy(np.int64),
                    anchor_by_case[int(case)],
                    assume_unique=False,
                )
            frame = frame.loc[anchored].copy()
            if frame.empty:
                continue
            case_values = frame["case"].to_numpy(np.int64)
            valid = valid_blend_response_pairs(frame)
            for case in np.unique(case_values):
                local = case_values == case
                counts[int(case)]["anchored_pair_rows"] += int(local.sum())
                counts[int(case)]["invalid_pair_rows_dropped"] += int(
                    (local & ~valid).sum()
                )
                counts[int(case)]["valid_pair_rows"] += int((local & valid).sum())
            frame = frame.loc[valid].copy()
            if frame.empty:
                continue
            predicted = predictor.predict_on_pairs(
                frame,
                task="response",
                rescaled=False,
                warn_extrapolation=False,
            )
            response = predicted["response"].to_numpy(np.float64)
            if len(response) != len(frame) or not np.isfinite(response).all():
                raise RuntimeError("response emulator returned invalid pair predictions")
            part = frame.loc[:, ["case", "input_index"]].copy()
            part["R_blend_model"] = response
            part["R_blend_measured"] = (
                frame["delta_et1"].to_numpy(np.float64) / response_shear
            )
            partial.append(
                part.groupby(["case", "input_index"], sort=False, as_index=False).agg(
                    R_blend_model=("R_blend_model", "sum"),
                    R_blend_measured=("R_blend_measured", "sum"),
                    n_pairs=("R_blend_model", "size"),
                )
            )
            if (batch_index + 1) % 100 == 0:
                print(
                    f"RESPONSE_PROFILE_BLEND_SCAN "
                    f"batch={batch_index + 1}/{reader.num_record_batches}",
                    flush=True,
                )
    if not partial:
        raise RuntimeError("response catalogue yielded no valid anchored pairs")
    result = pd.concat(partial, ignore_index=True).groupby(
        ["case", "input_index"], sort=False, as_index=False
    ).agg(
        R_blend_model=("R_blend_model", "sum"),
        R_blend_measured=("R_blend_measured", "sum"),
        n_pairs=("n_pairs", "sum"),
    )
    reports = {}
    for case in cases:
        local = result.loc[result["case"] == case]
        unexpected = np.setdiff1d(
            local["input_index"].to_numpy(np.int64),
            anchor_by_case[case],
            assume_unique=False,
        )
        if len(unexpected):
            raise RuntimeError(f"case {case} response pairs include non-anchor primaries")
        reports[str(case)] = {
            **counts[case],
            "response_objects": int(len(local)),
            "anchor_without_finite_pairs": int(
                len(anchor_by_case[case]) - len(local)
            ),
            "mean_measured_R_blend": float(local["R_blend_measured"].mean()),
            "mean_model_R_blend": float(local["R_blend_model"].mean()),
        }
    if result.duplicated(["case", "input_index"]).any():
        raise RuntimeError("model blend-response cache contains duplicate keys")
    return result.drop(columns=["R_blend_measured", "n_pairs"]), reports


def main(argv=None) -> None:
    args = parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("response-profile prediction must run under Slurm")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    cases = tuple(dict.fromkeys(int(case) for case in args.case))
    if len(cases) != len(args.case):
        raise ValueError("cases must be unique")
    if args.draws < 2 or args.draws % 2 or args.batch_size <= 0:
        raise ValueError("even draws >=2 and positive batch size required")
    if not np.isfinite(args.response_shear) or args.response_shear <= 0:
        raise ValueError("response shear must be positive and finite")
    for path in (
        args.flow,
        args.emulator_model,
        args.emulator_metadata,
        args.response_catalogue,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_path = args.domain_root / "manifest.json"
    domain = json.loads(manifest_path.read_text())
    population = domain["population"]
    if (
        population["truth_analysis_cut"] is not None
        or population["sheared_leg_recut"]
        or population["mag_auto_max"] != 25.8
        or population["flux_radius_min_pixels"] != 3.0
    ):
        raise ValueError("prediction cache requires the corrected fixed-g0 v2 domain")

    device = torch.device(args.device)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    bundle = load_measurement_model(args.flow, device=device)
    if bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES):
        raise ValueError("flow feature contract differs from the fixed-g0 domain")
    paths = ModelPaths(
        flow_checkpoints=(args.flow,),
        emulator_model=args.emulator_model,
        emulator_metadata=args.emulator_metadata,
    )
    predictor = load_emulator(paths, conditions=CONDITIONS, device="cpu")

    response_anchor_path = args.domain_root / "response_anchor.feather"
    self_parts = []
    reports = {}
    anchor_by_case = {
        case: response_anchor_for_case(response_anchor_path, case) for case in cases
    }
    for case in cases:
        print(f"RESPONSE_PROFILE_MODEL_START case={case}", flush=True)
        zero = load_flow_case(args.domain_root / "flow" / "g0" / f"case{case:03d}.npz")
        sheared = load_flow_case(
            args.domain_root / "flow" / "g05" / f"case{case:03d}.npz"
        )
        self_frame, self_report = predict_self_case(
            bundle,
            zero,
            sheared,
            draws=args.draws,
            batch_size=args.batch_size,
            seed=args.sampling_seed + 10_000_019 * case,
        )
        self_parts.append(self_frame)
        reports[str(case)] = {"R_self": self_report}
        print(
            f"RESPONSE_PROFILE_MODEL_DONE case={case} self={len(self_frame)}",
            flush=True,
        )

    blend_table, blend_reports = predict_blend_cases(
        predictor,
        args.response_catalogue,
        anchor_by_case,
        cases,
        args.response_shear,
    )
    for case in cases:
        reports[str(case)]["R_blend"] = blend_reports[str(case)]

    output_root.mkdir(parents=True)
    self_table = pd.concat(self_parts, ignore_index=True)
    self_path = output_root / "R_self_model.feather"
    blend_path = output_root / "R_blend_model.feather"
    self_table.to_feather(self_path)
    blend_table.to_feather(blend_path)
    result = {
        "format_version": 1,
        "cases": list(cases),
        "domain": {
            "anchor": (
                "measured g=0 detected, MAG_AUTO<25.8, strict "
                "FLUX_RADIUS>3.0 pixels (0.6 arcsec)"
            ),
            "truth_analysis_cut": None,
            "sheared_leg_recut": False,
        },
        "self_prediction": (
            "dot(E_model[e_g]-E_model[e_0],g)/|g|^2 using common antithetic draws"
        ),
        "blend_prediction": (
            "sum of v2 response-emulator predictions on the exact finite "
            "response-catalogue pair rows used by the measured target"
        ),
        "draws": int(args.draws),
        "sampling_seed": int(args.sampling_seed),
        "inputs": {
            "domain_manifest": str(manifest_path.resolve()),
            "domain_manifest_sha256": file_sha256(manifest_path),
            "flow": str(args.flow.resolve()),
            "flow_sha256": file_sha256(args.flow),
            "emulator_model": str(args.emulator_model.resolve()),
            "emulator_model_sha256": file_sha256(args.emulator_model),
            "emulator_metadata": str(args.emulator_metadata.resolve()),
            "emulator_metadata_sha256": file_sha256(args.emulator_metadata),
            "response_catalogue": str(args.response_catalogue.resolve()),
            "response_catalogue_sha256": file_sha256(args.response_catalogue),
        },
        "outputs": {
            "R_self": str(self_path),
            "R_self_rows": int(len(self_table)),
            "R_self_sha256": file_sha256(self_path),
            "R_blend": str(blend_path),
            "R_blend_rows": int(len(blend_table)),
            "R_blend_sha256": file_sha256(blend_path),
        },
        "case_reports": reports,
    }
    write_json(output_root / "manifest.json", result)
    print(
        f"RESPONSE_PROFILE_MODEL_COMPLETE output={output_root} "
        f"self={len(self_table):,} blend={len(blend_table):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
