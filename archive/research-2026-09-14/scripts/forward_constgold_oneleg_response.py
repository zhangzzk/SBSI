#!/usr/bin/env python
"""Forward-model one-leg shear response on a frozen ConstGold sample.

This is deliberately not an inverse catalogue-likelihood fit.  It holds the
truth conditions of the supplied ConstGold identities fixed, shears their
intrinsic primary shapes by ``+/- h``, samples the complete conditional
measurement model with common random numbers, adds the fixed catalogue blend
shift, reapplies the measured-output selection at each sign, and differences
the selected means.

The source sample is already both-detected and plus-leg selected.  Detection
is therefore conditioned upon rather than applied a second time.  The result
is the forward response conditional on this frozen plus-selected population.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import torch

from sbsi.coordinates import ellipticity_from_axis_ratio_angle
from sbsi.measurement_model import load_measurement_model
from sbsi.shear_map import apply_shear_to_ellipticity
from sbsi.training import build_shifted_context


RESCALE_KWARGS = {
    "pixel_rms": 0.312,
    "pixel_size": 0.2,
    "zero_mag": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
}
SOURCE_COLUMNS = (
    "case",
    "input_index",
    "neighbored",
    "distance",
    "Re_input_p",
    "Re_input_s",
    "r_input_p",
    "r_input_s",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "axis_ratio_input_p",
    "axis_ratio_input_s",
    "position_angle_input_p",
    "position_angle_input_s",
    "polarization_angle",
    "applied_g1",
    "applied_g2",
    "measured_e1_plus",
    "measured_e2_plus",
    "measured_mag_auto_plus",
    "measured_flux_radius_plus",
)
CROWD_COLUMNS = ("nbr_flux_near", "nbr_flux_far", "nbr_flux_max")
OBSERVED_COLUMNS = (
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_log_flux_radius",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _keys(case, input_index) -> np.ndarray:
    case = np.asarray(case, dtype=np.int64)
    index = np.asarray(input_index, dtype=np.int64)
    if (case < 0).any() or (index < 0).any() or (index >= 2**32).any():
        raise ValueError("case/input_index cannot be packed into nonnegative 32-bit keys")
    return (case << np.int64(32)) | index


def _scan_requested_rows(
    path: Path, columns, requested_keys: np.ndarray, *, require_all: bool = True
) -> pd.DataFrame:
    """Scan an Arrow/Feather file and return requested rows in request order."""

    requested = np.asarray(requested_keys, dtype=np.int64)
    if requested.ndim != 1 or not len(requested):
        raise ValueError("requested keys must be a non-empty vector")
    if len(np.unique(requested)) != len(requested):
        raise ValueError("requested keys contain duplicate identities")
    sorted_requested = np.sort(requested)
    required = tuple(dict.fromkeys(("case", "input_index", *columns)))
    parts = []
    with pa.memory_map(str(path), "r") as stream:
        reader = ipc.open_file(stream)
        missing = sorted(set(required) - set(reader.schema.names))
        if missing:
            raise KeyError(f"{path} lacks required columns: {missing}")
        for batch_index in range(reader.num_record_batches):
            table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(required)
            batch_keys = _keys(table["case"].to_numpy(), table["input_index"].to_numpy())
            position = np.searchsorted(sorted_requested, batch_keys)
            clipped = np.minimum(position, len(sorted_requested) - 1)
            keep = (position < len(sorted_requested)) & (
                sorted_requested[clipped] == batch_keys
            )
            if keep.any():
                parts.append(table.filter(pa.array(keep)).to_pandas())
    if not parts:
        raise RuntimeError(f"{path} matched none of the requested identities")
    frame = pd.concat(parts, ignore_index=True)
    found_keys = _keys(frame["case"], frame["input_index"])
    if len(np.unique(found_keys)) != len(found_keys):
        raise ValueError(f"{path} contains duplicate requested identities")
    if not require_all:
        return frame.reset_index(drop=True)
    order = pd.Series(np.arange(len(frame)), index=found_keys).reindex(requested)
    missing_count = int(order.isna().sum())
    if missing_count:
        raise RuntimeError(f"{path} is missing {missing_count:,} requested identities")
    return frame.iloc[order.to_numpy(dtype=np.int64)].reset_index(drop=True)


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list of unique ints")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--constgold", type=Path, required=True)
    parser.add_argument("--crowd-lookup", type=Path, required=True)
    parser.add_argument("--rblend-lookup", type=Path, action="append", required=True)
    parser.add_argument("--measurement-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--draw-ladder", type=_parse_ints, default=(16, 32, 64))
    parser.add_argument(
        "--sampling-seeds", type=_parse_ints, default=(4101, 4102, 4103, 4104)
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _validate_args(args):
    if not np.isfinite(args.h) or args.h <= 0:
        raise ValueError("h must be finite and positive")
    if args.draws <= 0 or args.batch_size <= 0 or args.n_boot <= 0:
        raise ValueError("draws, batch-size, and n-boot must be positive")
    if any(value <= 0 or value > args.draws for value in args.draw_ladder):
        raise ValueError("draw-ladder entries must lie in [1, draws]")
    if tuple(sorted(args.draw_ladder)) != args.draw_ladder:
        raise ValueError("draw-ladder must be strictly increasing")
    if args.draw_ladder[-1] != args.draws:
        raise ValueError("draw-ladder must end at draws")
    if args.max_objects is not None and args.max_objects <= 0:
        raise ValueError("max-objects must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")


def _target_indices(bundle):
    names = tuple(bundle.target_transform.target_names)
    expected = (
        "measured_ngmix_g1",
        "measured_ngmix_g2",
        "measured_mag_auto",
        "measured_log_flux_radius",
    )
    if names != expected:
        raise ValueError(f"unexpected measurement target order: {names}")
    return {name: names.index(name) for name in names}


def _draw_raw(bundle, context: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    tensor = torch.as_tensor(context, dtype=torch.float32, device=bundle.device)
    with torch.no_grad():
        standardized = bundle.model.sample(tensor, n_samples=n_draws, qmc=True)
    shape = standardized.shape
    flat = standardized.detach().cpu().numpy().reshape(-1, shape[-1])
    raw = bundle.target_transform.inverse_transform_array(flat)
    return raw.reshape(shape)


def _empty_sufficient(cases: np.ndarray, ladder) -> dict:
    unique = np.unique(cases)
    return {
        int(rung): {
            int(case): {
                "plus_count": 0.0,
                "plus_e1": 0.0,
                "plus_e2": 0.0,
                "minus_count": 0.0,
                "minus_e1": 0.0,
                "minus_e2": 0.0,
            }
            for case in unique
        }
        for rung in ladder
    }


def _accumulate_sign(target, case, draws, passed, sign):
    for value in np.unique(case):
        selected = case == value
        mask = passed[selected]
        subset = draws[selected]
        target[int(value)][f"{sign}_count"] += float(mask.sum())
        target[int(value)][f"{sign}_e1"] += float(subset[..., 0][mask].sum(dtype=np.float64))
        target[int(value)][f"{sign}_e2"] += float(subset[..., 1][mask].sum(dtype=np.float64))


def _pooled_model(sufficient, h):
    sums = {
        name: sum(row[name] for row in sufficient.values())
        for name in next(iter(sufficient.values()))
    }
    if sums["plus_count"] <= 0 or sums["minus_count"] <= 0:
        raise RuntimeError("one forward sign has no selected flow draws")
    plus = np.asarray(
        [sums["plus_e1"], sums["plus_e2"]], dtype=np.float64
    ) / sums["plus_count"]
    minus = np.asarray(
        [sums["minus_e1"], sums["minus_e2"]], dtype=np.float64
    ) / sums["minus_count"]
    response = (plus - minus) / (2.0 * h)
    intercept = 0.5 * (plus + minus)
    return {
        "plus_mean": plus.tolist(),
        "minus_mean": minus.tolist(),
        "intercept": intercept.tolist(),
        "response": response.tolist(),
        "plus_selected_draws": int(sums["plus_count"]),
        "minus_selected_draws": int(sums["minus_count"]),
    }


def _observed_by_case(sample: pd.DataFrame):
    result = {}
    for case, frame in sample.groupby("case", sort=True):
        result[int(case)] = {
            "count": int(len(frame)),
            "e1": float(frame["measured_ngmix_g1"].sum()),
            "e2": float(frame["measured_ngmix_g2"].sum()),
        }
    return result


def _pooled_observed(observed, h):
    count = sum(row["count"] for row in observed.values())
    mean = np.asarray(
        [sum(row["e1"] for row in observed.values()), sum(row["e2"] for row in observed.values())]
    ) / count
    return {
        "mean": mean.tolist(),
        "one_leg_response": (mean / h).tolist(),
        "n_objects": int(count),
    }


def _bootstrap(observed, model, *, h, n_boot, seed):
    cases = np.asarray(sorted(observed), dtype=np.int64)
    if set(cases) != set(model):
        raise ValueError("observed and model case sets differ")
    obs_count = np.asarray([observed[int(case)]["count"] for case in cases], dtype=float)
    obs_e1 = np.asarray([observed[int(case)]["e1"] for case in cases], dtype=float)
    fields = next(iter(model.values())).keys()
    mod = {
        name: np.asarray([model[int(case)][name] for case in cases], dtype=float)
        for name in fields
    }
    rng = np.random.default_rng(seed)
    values = np.empty((n_boot, 5), dtype=np.float64)
    for draw in range(n_boot):
        selected = rng.integers(0, len(cases), size=len(cases))
        raw = obs_e1[selected].sum() / obs_count[selected].sum() / h
        plus = mod["plus_e1"][selected].sum() / mod["plus_count"][selected].sum()
        minus = mod["minus_e1"][selected].sum() / mod["minus_count"][selected].sum()
        forward = (plus - minus) / (2.0 * h)
        observed_plus = obs_e1[selected].sum() / obs_count[selected].sum()
        values[draw] = (raw, forward, forward - raw, plus, plus - observed_plus)
    names = (
        "raw_one_leg_response_e1",
        "forward_response_e1",
        "forward_minus_raw_response_e1",
        "model_plus_mean_e1",
        "model_minus_observed_plus_mean_e1",
    )
    return {
        name: {
            "standard_error": float(values[:, index].std(ddof=1)),
            "ci95_percentile": np.quantile(values[:, index], [0.025, 0.975]).tolist(),
        }
        for index, name in enumerate(names)
    }


def main(argv=None):
    args = parse_args(argv)
    _validate_args(args)
    sample = pd.read_parquet(args.sample)
    required_sample = {"case", "input_index", *OBSERVED_COLUMNS}
    missing = sorted(required_sample - set(sample))
    if missing:
        raise KeyError(f"frozen sample lacks columns: {missing}")
    if args.max_objects is not None:
        sample = sample.iloc[: args.max_objects].reset_index(drop=True)
    if sample[["case", "input_index"]].duplicated().any():
        raise ValueError("frozen sample identities are not unique")
    requested = _keys(sample["case"], sample["input_index"])

    print(f"loading truth for {len(sample):,} frozen identities", flush=True)
    truth = _scan_requested_rows(args.constgold, SOURCE_COLUMNS, requested)
    print("loading keyed crowding conditions", flush=True)
    crowd = _scan_requested_rows(args.crowd_lookup, CROWD_COLUMNS, requested)
    print("loading keyed catalogue blend response", flush=True)
    rblend_parts = []
    for path in args.rblend_lookup:
        try:
            rblend_parts.append(
                _scan_requested_rows(
                    path, ("R_blend",), requested, require_all=False
                )
            )
        except RuntimeError as error:
            if "matched none" not in str(error):
                raise
    if not rblend_parts:
        raise RuntimeError("no R_blend shard matched the frozen identities")
    rblend = pd.concat(rblend_parts, ignore_index=True)
    rblend_keys = _keys(rblend["case"], rblend["input_index"])
    if len(np.unique(rblend_keys)) != len(rblend_keys):
        raise ValueError("R_blend shards overlap on requested identities")
    rblend = rblend.set_index(rblend_keys).reindex(requested)
    unmatched_rblend = int(rblend["R_blend"].isna().sum())
    if unmatched_rblend:
        raise RuntimeError(f"R_blend shards miss {unmatched_rblend:,} frozen identities")
    rblend = rblend.reset_index(drop=True)

    if not np.array_equal(_keys(truth["case"], truth["input_index"]), requested):
        raise RuntimeError("truth rows are not aligned to the frozen sample")
    if not np.array_equal(_keys(crowd["case"], crowd["input_index"]), requested):
        raise RuntimeError("crowd rows are not aligned to the frozen sample")
    source_measurement = np.column_stack(
        (
            truth["measured_e1_plus"].to_numpy(float),
            truth["measured_e2_plus"].to_numpy(float),
            truth["measured_mag_auto_plus"].to_numpy(float),
            np.log(truth["measured_flux_radius_plus"].to_numpy(float)),
        )
    )
    frozen_measurement = sample[list(OBSERVED_COLUMNS)].to_numpy(float)
    source_max_abs_difference = np.max(
        np.abs(source_measurement - frozen_measurement), axis=0
    )
    if not np.allclose(source_measurement, frozen_measurement, rtol=0.0, atol=2e-12):
        raise RuntimeError(
            "frozen plus measurements do not reproduce the ConstGold source; "
            f"max differences {source_max_abs_difference.tolist()}"
        )

    frame = truth.copy()
    for name in CROWD_COLUMNS:
        frame[name] = crowd[name].to_numpy(float)
    intrinsic_e1, intrinsic_e2 = ellipticity_from_axis_ratio_angle(
        frame["axis_ratio_input_p"].to_numpy(float),
        frame["position_angle_input_p"].to_numpy(float),
    )
    frame["e1_input_rot0_p"] = intrinsic_e1
    frame["e2_input_rot0_p"] = intrinsic_e2
    frame["gamma1_input_p"] = 0.0
    frame["gamma2_input_p"] = 0.0
    finite_context = np.isfinite(
        frame[
            [
                "Re_input_p",
                "r_input_p",
                "sersic_n_input_p",
                "e1_input_rot0_p",
                "e2_input_rot0_p",
                *CROWD_COLUMNS,
            ]
        ].to_numpy(float)
    ).all(axis=1)
    if not finite_context.all():
        raise ValueError(f"{int((~finite_context).sum())} truth contexts are non-finite")
    rblend_value = rblend["R_blend"].to_numpy(float)
    if not np.isfinite(rblend_value).all():
        raise ValueError("R_blend contains non-finite values")

    bundle = load_measurement_model(str(args.measurement_model), device=args.device)
    target_index = _target_indices(bundle)
    plus_context = build_shifted_context(
        frame,
        (1.0, 0.0),
        args.h,
        bundle.condition_preprocessor,
        bundle.metadata["condition_features"],
        RESCALE_KWARGS,
    )
    minus_context = build_shifted_context(
        frame,
        (1.0, 0.0),
        -args.h,
        bundle.condition_preprocessor,
        bundle.metadata["condition_features"],
        RESCALE_KWARGS,
    )
    plus_e1, plus_e2 = apply_shear_to_ellipticity(
        intrinsic_e1, intrinsic_e2, args.h, 0.0
    )
    minus_e1, minus_e2 = apply_shear_to_ellipticity(
        intrinsic_e1, intrinsic_e2, -args.h, 0.0
    )
    plus_shift = rblend_value[:, None] * np.column_stack(
        (plus_e1 - intrinsic_e1, plus_e2 - intrinsic_e2)
    )
    minus_shift = rblend_value[:, None] * np.column_stack(
        (minus_e1 - intrinsic_e1, minus_e2 - intrinsic_e2)
    )

    cases = sample["case"].to_numpy(np.int64)
    all_sufficient = {}
    log_radius_min = float(np.log(0.75 / RESCALE_KWARGS["pixel_size"]))
    for sampling_seed in args.sampling_seeds:
        sufficient = _empty_sufficient(cases, args.draw_ladder)
        print(f"sampling seed {sampling_seed}", flush=True)
        for start in range(0, len(frame), args.batch_size):
            stop = min(start + args.batch_size, len(frame))
            paired_seed = int(sampling_seed) * 1_000_003 + start
            plus = _draw_raw(
                bundle, plus_context[start:stop], args.draws, paired_seed
            )
            minus = _draw_raw(
                bundle, minus_context[start:stop], args.draws, paired_seed
            )
            plus[..., target_index["measured_ngmix_g1"]] += plus_shift[start:stop, None, 0]
            plus[..., target_index["measured_ngmix_g2"]] += plus_shift[start:stop, None, 1]
            minus[..., target_index["measured_ngmix_g1"]] += minus_shift[start:stop, None, 0]
            minus[..., target_index["measured_ngmix_g2"]] += minus_shift[start:stop, None, 1]
            for rung in args.draw_ladder:
                for sign, draws in (("plus", plus[:, :rung]), ("minus", minus[:, :rung])):
                    passed = (
                        np.isfinite(draws).all(axis=2)
                        & (
                            np.hypot(
                                draws[..., target_index["measured_ngmix_g1"]],
                                draws[..., target_index["measured_ngmix_g2"]],
                            )
                            < 0.6
                        )
                        & (draws[..., target_index["measured_mag_auto"]] < 25.8)
                        & (
                            draws[..., target_index["measured_log_flux_radius"]]
                            >= log_radius_min
                        )
                    )
                    _accumulate_sign(
                        sufficient[int(rung)], cases[start:stop], draws, passed, sign
                    )
            if start == 0 or stop == len(frame) or stop % 20_000 < args.batch_size:
                print(f"  rows {stop:,}/{len(frame):,}", flush=True)
        all_sufficient[int(sampling_seed)] = sufficient

    observed_case = _observed_by_case(sample)
    observed = _pooled_observed(observed_case, args.h)
    per_seed = {}
    for sampling_seed, by_rung in all_sufficient.items():
        per_seed[str(sampling_seed)] = {}
        for rung, sufficient in by_rung.items():
            pooled = _pooled_model(sufficient, args.h)
            pooled["plus_selection_fraction"] = pooled["plus_selected_draws"] / (
                len(frame) * rung
            )
            pooled["minus_selection_fraction"] = pooled["minus_selected_draws"] / (
                len(frame) * rung
            )
            pooled["forward_minus_raw_response_e1"] = (
                pooled["response"][0] - observed["one_leg_response"][0]
            )
            pooled["model_minus_observed_plus_mean_e1"] = (
                pooled["plus_mean"][0] - observed["mean"][0]
            )
            pooled["case_bootstrap"] = _bootstrap(
                observed_case,
                sufficient,
                h=args.h,
                n_boot=args.n_boot,
                seed=args.bootstrap_seed + sampling_seed + rung,
            )
            per_seed[str(sampling_seed)][str(rung)] = pooled

    final_rung = args.draw_ladder[-1]
    combined_case = _empty_sufficient(cases, (final_rung,))[final_rung]
    for sufficient in all_sufficient.values():
        for case, row in sufficient[final_rung].items():
            for name, value in row.items():
                combined_case[case][name] += value
    combined = _pooled_model(combined_case, args.h)
    total_draws = len(args.sampling_seeds) * final_rung
    combined["draws_per_truth_context"] = int(total_draws)
    combined["plus_selection_fraction"] = combined["plus_selected_draws"] / (
        len(frame) * total_draws
    )
    combined["minus_selection_fraction"] = combined["minus_selected_draws"] / (
        len(frame) * total_draws
    )
    combined["forward_minus_raw_response_e1"] = (
        combined["response"][0] - observed["one_leg_response"][0]
    )
    combined["model_minus_observed_plus_mean_e1"] = (
        combined["plus_mean"][0] - observed["mean"][0]
    )
    combined["case_bootstrap"] = _bootstrap(
        observed_case,
        combined_case,
        h=args.h,
        n_boot=args.n_boot,
        seed=args.bootstrap_seed,
    )
    individual_final = np.asarray(
        [per_seed[str(seed)][str(final_rung)]["response"][0] for seed in args.sampling_seeds]
    )
    combined["sampling_seed_response_e1"] = individual_final.tolist()
    sampling_seed_sd = (
        float(individual_final.std(ddof=1)) if len(individual_final) > 1 else 0.0
    )
    combined["sampling_seed_standard_deviation_e1"] = sampling_seed_sd
    combined["sampling_seed_sem_e1"] = float(
        sampling_seed_sd / np.sqrt(len(individual_final))
    )

    result = {
        "analysis": "complete-model forward response conditional on a frozen plus-selected ConstGold sample",
        "difference_convention": "central +/-h model response; one-leg +h raw measurement comparator",
        "detection_treatment": (
            "conditioned: ConstGold identities are both-detected and the measurement flow is conditional on detection"
        ),
        "selection_treatment": (
            "parent identities are frozen after plus-leg selection; model cuts are reapplied independently to +/- draws"
        ),
        "h": float(args.h),
        "model": {
            "measurement_flow": str(args.measurement_model),
            "measurement_flow_sha256": _sha256(args.measurement_model),
            "r_blend": "fixed keyed ConstGold V3 response; shift=R_blend*[e(g)-e(0)]",
        },
        "cuts": {
            "abs_shape_max": 0.6,
            "measured_mag_max": 25.8,
            "measured_radius_min_arcsec": 0.75,
            "measured_log_flux_radius_min": log_radius_min,
        },
        "sampling": {
            "method": "randomly shifted Sobol QMC with common random numbers across signs",
            "draws_per_seed": int(args.draws),
            "draw_ladder": list(args.draw_ladder),
            "sampling_seeds": list(args.sampling_seeds),
            "batch_size": int(args.batch_size),
        },
        "population": {
            "n_objects": int(len(sample)),
            "n_cases": int(sample["case"].nunique()),
            "case_range": [int(sample["case"].min()), int(sample["case"].max()) + 1],
            "unmatched_truth": 0,
            "unmatched_crowd": 0,
            "unmatched_rblend": 0,
            "source_measurement_max_abs_difference": source_max_abs_difference.tolist(),
            "conditional_population_warning": (
                "the 100k parent identities were sampled after passing the observed plus-leg cuts"
            ),
        },
        "observed_plus": observed,
        "combined_final": combined,
        "per_sampling_seed_and_rung": per_seed,
        "inputs": {
            "sample": str(args.sample),
            "sample_sha256": _sha256(args.sample),
            "constgold": str(args.constgold),
            "crowd_lookup": str(args.crowd_lookup),
            "rblend_lookups": [str(path) for path in args.rblend_lookup],
        },
        "bootstrap": {
            "unit": "ConstGold case",
            "replicates": int(args.n_boot),
            "seed": int(args.bootstrap_seed),
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    sample[["case", "input_index"]].assign(
        R_blend=rblend_value,
        intrinsic_e1=intrinsic_e1,
        intrinsic_e2=intrinsic_e2,
    ).to_parquet(args.output / "matched_truth_summary.parquet", index=False)
    print(json.dumps({
        "observed_plus": observed,
        "combined_final": combined,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
