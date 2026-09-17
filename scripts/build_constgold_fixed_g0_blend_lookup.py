#!/usr/bin/env python3
"""Build an anchor-keyed ConstGold R_blend lookup for a fixed-g0 emulator.

This deliberately restores the useful part of the archived ConstGold lookup
builder under the replacement domain contract.  The complete truth scene is
kept as the neighbour catalogue, while only exact keys selected in the g=0
flow anchor are used as response primaries.  No truth-property cut is applied.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from sbsi.coordinates import ellipticity_from_axis_ratio_angle
from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.models import ModelPaths, load_emulator


CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument(
        "--input-pattern", required=True, help="format string containing {case}"
    )
    parser.add_argument(
        "--anchor-pattern", required=True, help="case NPZ format string containing {case}"
    )
    parser.add_argument("--measurement-model", type=Path, required=True)
    parser.add_argument("--emulator-metadata", type=Path, required=True)
    parser.add_argument("--emulator-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def load_anchor(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        ids = np.asarray(data["input_index"], dtype=np.int64)
        context = np.asarray(data["context"], dtype=np.float64)
    if (
        ids.ndim != 1
        or context.shape != (len(ids), len(FLOW_FEATURES))
        or len(np.unique(ids)) != len(ids)
        or not np.isfinite(context).all()
    ):
        raise ValueError(f"invalid fixed-g0 anchor arrays in {path}")
    order = np.argsort(ids, kind="stable")
    return ids[order], context[order]


def prepare_truth(path: Path) -> pd.DataFrame:
    raw = feather.read_table(path).to_pandas()
    rename = {name: name.replace("_input", "") for name in raw.columns}
    truth = raw.rename(columns=rename)
    required = {
        "index",
        "RA",
        "DEC",
        "redshift",
        "Re",
        "axis_ratio",
        "position_angle",
        "sersic_n",
        "r",
    }
    missing = sorted(required - set(truth.columns))
    if missing:
        raise KeyError(f"{path} lacks truth fields: {missing}")
    if truth["index"].duplicated().any():
        raise ValueError(f"{path} has duplicate source identities")
    return truth


def verify_anchor_truth(
    truth: pd.DataFrame, anchor_ids: np.ndarray, anchor_context: np.ndarray
) -> pd.DataFrame:
    aligned = truth.set_index("index", drop=False).reindex(anchor_ids)
    if aligned["index"].isna().any():
        raise RuntimeError("fixed-g0 anchor contains identities absent from ConstGold")
    e1, e2 = ellipticity_from_axis_ratio_angle(
        aligned["axis_ratio"].to_numpy(float),
        aligned["position_angle"].to_numpy(float),
    )
    expected = np.column_stack(
        (
            e1,
            e2,
            aligned["sersic_n"].to_numpy(float),
            aligned["r"].to_numpy(float),
            aligned["Re"].to_numpy(float)
            * np.sqrt(aligned["axis_ratio"].to_numpy(float)),
        )
    )
    if not np.allclose(expected, anchor_context[:, :5], rtol=2e-6, atol=2e-6):
        difference = float(np.max(np.abs(expected - anchor_context[:, :5])))
        raise RuntimeError(
            "ConstGold truth is not identity-compatible with the fixed-g0 anchor; "
            f"maximum context difference={difference}"
        )
    return aligned.reset_index(drop=True)


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("duplicate cases are not allowed")

    metadata = json.loads(args.emulator_metadata.read_text())
    regression = metadata.get("tasks", {}).get("regression", {})
    domain = regression.get("population_domain", {})
    if regression.get("cuts") is not None:
        raise ValueError("fixed-g0 blend lookup requires emulator cuts=null")
    if regression.get("truth_analysis_cuts_applied") is not False:
        raise ValueError("fixed-g0 blend lookup requires no truth analysis cut")
    if domain.get("sheared_leg_recut") is not False:
        raise ValueError("fixed-g0 blend lookup refuses a sheared-leg recut")

    paths = ModelPaths(
        flow_checkpoints=(args.measurement_model,),
        emulator_metadata=args.emulator_metadata,
        emulator_model=args.emulator_model,
    )
    paths.validate(verify_hashes=False)
    predictor = load_emulator(paths, conditions=CONDITIONS, device=args.device)

    parts = []
    reports = []
    for case in args.cases:
        anchor_path = Path(args.anchor_pattern.format(case=case))
        truth_path = Path(args.input_pattern.format(case=case))
        if not anchor_path.is_file() or not truth_path.is_file():
            raise FileNotFoundError(
                anchor_path if not anchor_path.is_file() else truth_path
            )
        anchor_ids, anchor_context = load_anchor(anchor_path)
        truth = prepare_truth(truth_path)
        primaries = verify_anchor_truth(truth, anchor_ids, anchor_context)

        predicted = predictor.predict_response(primaries, truth)
        primary_column = "index_input_p"
        if primary_column not in predicted:
            raise RuntimeError(
                f"response prediction lacks primary key {primary_column!r}"
            )
        grouped = predicted.groupby(primary_column, sort=False)["response"]
        response = grouped.sum()
        #  R_blend is a SIGNED sum over a primary's pairs, so it can sit near
        #  zero through cancellation rather than through an absence of
        #  blending.  The absolute sum bounds how much a per-pair emulator
        #  error can move this primary, and the pair count says over how many
        #  terms.  Both are additive columns; R_blend itself is untouched.
        absolute = predicted.assign(
            magnitude=predicted["response"].abs()
        ).groupby(primary_column, sort=False)["magnitude"].sum()
        pairs = grouped.size()
        unexpected = np.setdiff1d(
            response.index.to_numpy(np.int64), anchor_ids, assume_unique=False
        )
        if len(unexpected):
            raise RuntimeError(
                f"response prediction returned {len(unexpected)} non-anchor primaries"
            )
        pair_primary_count = len(response)
        response = response.reindex(anchor_ids, fill_value=0.0)
        absolute = absolute.reindex(anchor_ids, fill_value=0.0)
        pairs = pairs.reindex(anchor_ids, fill_value=0)
        values = response.to_numpy(np.float64)
        absolute_values = absolute.to_numpy(np.float64)
        pair_counts = pairs.to_numpy(np.int64)
        if not np.isfinite(values).all():
            raise RuntimeError("non-finite fixed-g0 blending response")
        if not np.isfinite(absolute_values).all():
            raise RuntimeError("non-finite fixed-g0 absolute blending response")
        if (absolute_values + 1e-12 < np.abs(values)).any():
            raise RuntimeError("absolute blending sum below the signed sum")
        parts.append(
            pd.DataFrame(
                {
                    "case": np.full(len(anchor_ids), case, dtype=np.int32),
                    "input_index": anchor_ids,
                    "R_blend": values,
                    "R_blend_abs": absolute_values,
                    "blend_pairs": pair_counts,
                }
            )
        )
        report = {
            "case": int(case),
            "anchor_rows": int(len(anchor_ids)),
            "pair_primary_rows": int(pair_primary_count),
            "isolated_zero_rows": int(len(anchor_ids) - pair_primary_count),
            "mean_R_blend": float(values.mean()),
            "mean_R_blend_abs": float(absolute_values.mean()),
            "mean_blend_pairs": float(pair_counts.mean()),
        }
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["case", "input_index"]).any():
        raise RuntimeError("generated lookup contains duplicate keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_feather(output)
    manifest = {
        "format_version": 1,
        "population": "exact fixed-g0 flow-anchor keys; no truth cut; no sheared-leg recut",
        "cases": [int(case) for case in args.cases],
        "rows": int(len(result)),
        "inputs": {
            "anchor_pattern": args.anchor_pattern,
            "input_pattern": args.input_pattern,
            "measurement_model": str(args.measurement_model.resolve()),
            "measurement_model_sha256": file_sha256(args.measurement_model),
            "emulator_model": str(args.emulator_model.resolve()),
            "emulator_model_sha256": file_sha256(args.emulator_model),
            "emulator_metadata": str(args.emulator_metadata.resolve()),
            "emulator_metadata_sha256": file_sha256(args.emulator_metadata),
        },
        "case_reports": reports,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"CONSTGOLD_FIXED_G0_RBLEND_DONE output={output} rows={len(result):,}")


if __name__ == "__main__":
    main()
