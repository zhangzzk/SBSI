#!/usr/bin/env python
"""Recover a unique truth scene catalogue from a paired SBSI catalogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRUTH_COLUMNS = (
    "RA",
    "DEC",
    "redshift",
    "r",
    "Re",
    "sersic_n",
    "axis_ratio",
    "position_angle",
)


def extract_scene_catalogue(
    paired: pd.DataFrame,
    *,
    cases=None,
    group_column: str = "case",
    primary_id_column: str = "input_index",
    secondary_id_column: str = "input_index_sec",
    primary_mag_min: float,
    primary_mag_max: float,
    primary_re_min: float,
    primary_re_max: float,
) -> tuple[pd.DataFrame, dict]:
    """Union primary and secondary truth rows and assign primary prior mass.

    Galaxies outside the flow's primary domain remain in the scene with zero
    prior mass, so they still contribute as neighbours.  Multiple records for
    one ``(group, input_index)`` must carry identical truth values.
    """

    required = {group_column, primary_id_column, secondary_id_column}
    required.update(f"{name}_input_p" for name in TRUTH_COLUMNS)
    required.update(f"{name}_input_s" for name in TRUTH_COLUMNS)
    missing = sorted(required - set(paired))
    if missing:
        raise KeyError(f"paired catalogue lacks scene columns: {missing}")
    if not (
        primary_mag_min < primary_mag_max
        and primary_re_min < primary_re_max
    ):
        raise ValueError("primary-domain bounds must be strictly increasing")

    selected = paired
    if cases is not None:
        cases = tuple(cases)
        selected = selected.loc[selected[group_column].isin(cases)]
    if selected.empty:
        raise ValueError("no paired rows remain after the requested case selection")

    def side(suffix: str, id_column: str) -> pd.DataFrame:
        columns = [group_column, id_column, *(f"{name}_input_{suffix}" for name in TRUTH_COLUMNS)]
        out = selected.loc[:, columns].copy()
        out = out.rename(
            columns={
                id_column: "input_index",
                **{f"{name}_input_{suffix}": name for name in TRUTH_COLUMNS},
            }
        )
        out = out.loc[out["input_index"].notna()]
        return out

    primary = side("p", primary_id_column)
    secondary = side("s", secondary_id_column)
    combined = pd.concat((primary, secondary), ignore_index=True)
    combined["input_index"] = combined["input_index"].astype(np.int64)
    keys = [group_column, "input_index"]

    distinct_records = combined.drop_duplicates()
    multiplicity = distinct_records.groupby(keys, sort=False, dropna=False).size()
    conflicts = multiplicity[multiplicity > 1]
    if len(conflicts):
        examples = [tuple(value) for value in conflicts.index[:5]]
        raise ValueError(
            f"{len(conflicts)} input galaxies have conflicting truth records; "
            f"first keys: {examples}"
        )
    scene = distinct_records.drop_duplicates(keys, keep="first").reset_index(drop=True)
    finite = np.isfinite(scene.loc[:, TRUTH_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    if not finite.all():
        raise ValueError(f"{int((~finite).sum())} recovered galaxies have non-finite truth values")

    eligible = (
        scene["r"].between(primary_mag_min, primary_mag_max, inclusive="neither")
        & scene["Re"].between(primary_re_min, primary_re_max, inclusive="neither")
    )
    scene["prior_weight"] = eligible.astype(np.float64)
    report = {
        "n_paired_rows_selected": int(len(selected)),
        "n_primary_records": int(len(primary)),
        "n_secondary_records": int(len(secondary)),
        "n_unique_scene_galaxies": int(len(scene)),
        "n_positive_prior_atoms": int(eligible.sum()),
        "n_zero_weight_neighbours": int((~eligible).sum()),
        "groups": [value.item() if hasattr(value, "item") else value for value in scene[group_column].unique()],
        "primary_domain": {
            "mag": [float(primary_mag_min), float(primary_mag_max)],
            "Re": [float(primary_re_min), float(primary_re_max)],
        },
    }
    return scene, report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-catalogue", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--cases", required=True, help="comma-separated case values")
    parser.add_argument("--group-column", default="case")
    parser.add_argument("--primary-id-column", default="input_index")
    parser.add_argument("--secondary-id-column", default="input_index_sec")
    parser.add_argument("--primary-mag-min", type=float, required=True)
    parser.add_argument("--primary-mag-max", type=float, required=True)
    parser.add_argument("--primary-re-min", type=float, required=True)
    parser.add_argument("--primary-re-max", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cases = tuple(int(value) for value in args.cases.split(",") if value)
    if not cases:
        raise SystemExit("--cases must contain at least one value")
    columns = [args.group_column, args.primary_id_column, args.secondary_id_column]
    columns.extend(f"{name}_input_p" for name in TRUTH_COLUMNS)
    columns.extend(f"{name}_input_s" for name in TRUTH_COLUMNS)
    paired = pd.read_feather(args.paired_catalogue, columns=columns)
    scene, report = extract_scene_catalogue(
        paired,
        cases=cases,
        group_column=args.group_column,
        primary_id_column=args.primary_id_column,
        secondary_id_column=args.secondary_id_column,
        primary_mag_min=args.primary_mag_min,
        primary_mag_max=args.primary_mag_max,
        primary_re_min=args.primary_re_min,
        primary_re_max=args.primary_re_max,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".feather":
        scene.to_feather(output)
    elif output.suffix == ".parquet":
        scene.to_parquet(output, index=False)
    else:
        raise SystemExit("--output must end in .feather or .parquet")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"scene catalogue -> {output}")


if __name__ == "__main__":
    main()
