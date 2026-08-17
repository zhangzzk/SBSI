"""Draw a reproducible uniform sample from the V2.2 half-shear pair catalogue.

The response catalogue is too large to load as a pandas frame.  This script
streams Arrow record batches, applies the exact V2.2 regression support, and
uses iid random priorities to retain an exactly uniform fixed-size sample of
eligible pair rows.  The frozen V2.2 prediction and the original noisy forward
label are attached for later comparison with a clean two-object toy response.

No constgold catalogue is read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


BLENDEMU_ROOT = "/home/z/Zekang.Zhang/blendemu"
MODEL_DIR = os.path.join(BLENDEMU_ROOT, "models")
COND = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}
PAIR_FEATURES = [
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance",
]
SOURCE_COLUMNS = [
    "RA_input_p", "RA_input_s", "DEC_input_p", "DEC_input_s",
    "Re_input_p", "Re_input_s",
    "axis_ratio_input_p", "axis_ratio_input_s",
    "position_angle_input_p", "position_angle_input_s",
    "sersic_n_input_p", "sersic_n_input_s",
    "r_input_p", "r_input_s",
]
READ_COLUMNS = [
    "case", "input_index", *SOURCE_COLUMNS, "polarization_angle",
    "distance", "shear_angle", "delta_et1", "delta_et2",
]
CUTS = {
    "r_input_s": (13.0, 29.0),
    "r_input_p": (18.0, 25.8),
    "Re_input_s": (0.0, 10.0),
    "Re_input_p": (0.5, 1.5),
    "distance": (0.0, 10.0),
}


def eligible_mask(frame: pd.DataFrame, case_min: int, case_max: int) -> np.ndarray:
    """Exact finite, strict-bound V2.2 regression population mask."""
    mask = frame.case.between(case_min, case_max, inclusive="both").to_numpy()
    for column, (lower, upper) in CUTS.items():
        values = frame[column].to_numpy(float)
        mask &= np.isfinite(values) & (values > lower) & (values < upper)
    finite_columns = [*SOURCE_COLUMNS, "delta_et1", "delta_et2"]
    mask &= np.isfinite(frame[finite_columns].to_numpy(float)).all(axis=1)
    return mask


def keep_smallest_priority(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    """Keep the rows with the smallest iid priorities without a full sort."""
    if len(frame) <= size:
        return frame
    values = frame.sample_priority.to_numpy(float)
    chosen = np.argpartition(values, size - 1)[:size]
    return frame.iloc[chosen].copy()


def uniform_priority_sample(
    catalogue: str,
    sample_size: int,
    seed: int,
    case_min: int,
    case_max: int,
) -> tuple[pd.DataFrame, dict]:
    """Stream ``catalogue`` and return a uniform sample of eligible rows."""
    rng = np.random.default_rng(seed)
    kept: pd.DataFrame | None = None
    n_scanned = 0
    n_eligible = 0
    processed_batches = 0
    with ipc.open_file(catalogue) as reader:
        missing = set(READ_COLUMNS) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            batch = pa.Table.from_batches([reader.get_batch(batch_index)]).select(
                READ_COLUMNS
            ).to_pandas()
            row_number = n_scanned + np.arange(len(batch), dtype=np.int64)
            n_scanned += len(batch)
            mask = eligible_mask(batch, case_min, case_max)
            if not np.any(mask):
                continue
            processed_batches += 1
            candidate = batch.loc[mask].copy()
            candidate.insert(0, "catalogue_row", row_number[mask])
            candidate.insert(1, "sample_priority", rng.random(len(candidate)))
            n_eligible += len(candidate)
            candidate = keep_smallest_priority(candidate, sample_size)
            kept = candidate if kept is None else keep_smallest_priority(
                pd.concat([kept, candidate], ignore_index=True), sample_size
            )
            if processed_batches % 200 == 0:
                print(
                    f"processed={processed_batches} selected batches; "
                    f"scanned={n_scanned:,}; eligible={n_eligible:,}",
                    flush=True,
                )
    if kept is None or len(kept) != sample_size:
        raise RuntimeError(
            f"sample has {0 if kept is None else len(kept):,} rows, "
            f"expected {sample_size:,}"
        )
    kept = kept.sort_values("sample_priority", kind="mergesort").reset_index(drop=True)
    kept.insert(0, "sample_id", np.arange(len(kept), dtype=np.int64))
    if kept.catalogue_row.duplicated().any():
        raise RuntimeError("uniform sample contains duplicate catalogue rows")
    audit = {
        "n_scanned": int(n_scanned),
        "n_eligible": int(n_eligible),
        "n_sample": int(len(kept)),
        "processed_selected_batches": int(processed_batches),
    }
    return kept, audit


def score_v22(frame: pd.DataFrame, tag: str) -> np.ndarray:
    """Score the frozen raw-feature pair sample with BlendEMU."""
    if BLENDEMU_ROOT not in sys.path:
        sys.path.insert(0, BLENDEMU_ROOT)
    from blendemu.inference import BlendingPredictor

    predictor = BlendingPredictor.load(MODEL_DIR, tag=tag, conditions=COND, device="cpu")
    cuts, _, _ = predictor._select("regression")
    expected = [list(CUTS[name]) for name in (
        "r_input_s", "r_input_p", "Re_input_s", "Re_input_p", "distance"
    )]
    if not np.array_equal(np.asarray(cuts, float), np.asarray(expected, float)):
        raise RuntimeError(f"{tag} cuts drifted: {cuts} != {expected}")
    prediction = predictor.predict_on_pairs(
        frame[PAIR_FEATURES], task="response"
    )["response"].to_numpy(float)
    if not np.isfinite(prediction).all():
        raise RuntimeError("V2.2 produced a non-finite sampled-pair prediction")
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--case-min", type=int, default=40)
    parser.add_argument("--case-max", type=int, default=199)
    parser.add_argument("--catalogue-shear", type=float, default=0.2)
    parser.add_argument("--tag", default="lsst_r_extnbr_v22")
    parser.add_argument("--expected-eligible", type=int, default=37_852_393)
    parser.add_argument("--output-feather", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    for path in (args.output_feather, args.output_json):
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")
    if args.sample_size < 1000 or args.case_min > args.case_max:
        raise ValueError("sample must have >=1000 rows and a valid case window")
    if args.catalogue_shear <= 0.0:
        raise ValueError("catalogue shear must be positive")

    sample, audit = uniform_priority_sample(
        args.catalogue, args.sample_size, args.seed, args.case_min, args.case_max
    )
    if args.expected_eligible and audit["n_eligible"] != args.expected_eligible:
        raise RuntimeError(
            f"eligible population {audit['n_eligible']:,} != frozen expectation "
            f"{args.expected_eligible:,}"
        )
    sample["R_label_forward"] = sample.delta_et1.to_numpy(float) / args.catalogue_shear
    sample["R_label_null"] = sample.delta_et2.to_numpy(float) / args.catalogue_shear
    sample["R_emulator_v22"] = score_v22(sample, args.tag)
    if sample.case.nunique() < args.case_max - args.case_min + 1:
        raise RuntimeError("uniform sample does not cover every training case")

    Path(args.output_feather).parent.mkdir(parents=True, exist_ok=True)
    sample.to_feather(args.output_feather)
    payload = {
        "design": (
            "iid-priority uniform sample of pair rows after the exact frozen V2.2 "
            "regression cuts; original forward label and frozen emulator prediction attached"
        ),
        "catalogue": os.path.abspath(args.catalogue),
        "catalogue_shear": float(args.catalogue_shear),
        "case_window": [int(args.case_min), int(args.case_max)],
        "regression_cuts": {key: list(value) for key, value in CUTS.items()},
        "sample_seed": int(args.seed),
        "sampling": "one iid U(0,1) priority per eligible row; globally smallest n retained",
        "tag": args.tag,
        "sample_size": int(args.sample_size),
        "n_cases_sampled": int(sample.case.nunique()),
        **audit,
        "output_feather": os.path.abspath(args.output_feather),
    }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("HALFSHEAR_NOISELESS_PAIR_TOY_SAMPLE_DONE", flush=True)


if __name__ == "__main__":
    main()
