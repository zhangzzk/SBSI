"""Empirically test whether stored constgold rows survived the bright-neighbour cut.

For each requested case and shear leg, reproduce blendemu's exact detected-object
cut (FLUX_AUTO ratio > 5 within 3 arcsec), map rejected SExtractor rows back to
stable input IDs, and intersect them with the stored constant response catalogue.
Zero overlap is direct evidence that the cut is baked into that artifact.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

from blendemu.utils import remove_detection_w_bright_neighbour


TILE = "tile180.0_-0.5"


def response_cases(path: str, cases: set[int]) -> pd.DataFrame:
    parts = []
    with ipc.open_file(pa.memory_map(path)) as reader:
        for i in range(reader.num_record_batches):
            table = pa.Table.from_batches([reader.get_batch(i)]).select(["case", "input_index"])
            mask = None
            for case in sorted(cases):
                this = pc.equal(table["case"], pa.scalar(case))
                mask = this if mask is None else pc.or_(mask, this)
            table = table.filter(mask)
            if table.num_rows:
                parts.append(table)
    if not parts:
        raise RuntimeError("none of the requested cases occur in the response catalogue")
    out = pa.concat_tables(parts).to_pandas()
    if out.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate response keys")
    return out


def rejected_input_ids(base: str, case: int, shear: str) -> tuple[np.ndarray, int]:
    cat = os.path.join(base, f"case{case}_{shear}", "real0", "catalogues")
    scat = pd.read_feather(os.path.join(
        cat, "Shapes", f"shape_catalogue_detect_position_all_{TILE}.feather"
    ))
    match = pd.read_feather(os.path.join(
        cat, "CrossMatch", f"{TILE}_rot0_matched.feather"
    ))
    reject_rows = remove_detection_w_bright_neighbour(
        scat["X_WORLD"].array, scat["Y_WORLD"].array, scat["FLUX_AUTO"].array,
        ratio_max=5, r_min=0, r_max=3 / 3600,
    )
    rejected = match.loc[
        np.isin(match["id_detec"].to_numpy(dtype=int) - 1, reject_rows), "id_input"
    ].to_numpy(dtype=int)
    return np.unique(rejected), len(scat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--cases", nargs="+", type=int, required=True)
    ap.add_argument("--shears", nargs=2, default=["0.02", "-0.02"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    stored = response_cases(args.catalogue, set(args.cases))
    rows = []
    rejected_union = set()
    overlap_union = set()
    for case in args.cases:
        stored_ids = set(stored.loc[stored["case"] == case, "input_index"].astype(int))
        case_rejected = set()
        leg_rows = []
        for shear in args.shears:
            rejected, n_detect = rejected_input_ids(args.base, case, shear)
            rejected_set = set(map(int, rejected))
            overlap = rejected_set & stored_ids
            case_rejected |= rejected_set
            rejected_union |= {(case, value) for value in rejected_set}
            overlap_union |= {(case, value) for value in overlap}
            leg_rows.append({
                "shear": shear,
                "n_shape_detections": int(n_detect),
                "n_rejected_input_ids": int(len(rejected_set)),
                "n_rejected_present_in_stored": int(len(overlap)),
            })
        rows.append({
            "case": case,
            "n_stored": int(len(stored_ids)),
            "n_rejected_union": int(len(case_rejected)),
            "n_rejected_union_present_in_stored": int(len(case_rejected & stored_ids)),
            "legs": leg_rows,
        })
        print(
            f"case {case}: stored={len(stored_ids):,}, rejected union={len(case_rejected):,}, "
            f"overlap={len(case_rejected & stored_ids):,}", flush=True,
        )

    result = {
        "status": "artifact audit only",
        "cut": "detected neighbour FLUX_AUTO > 5x primary within 3 arcsec, separately per leg",
        "cases": args.cases,
        "n_stored_in_cases": int(len(stored)),
        "n_unique_rejected_case_ids": int(len(rejected_union)),
        "n_rejected_case_ids_present_in_stored": int(len(overlap_union)),
        "cut_baked_in_if_zero_overlap": bool(len(overlap_union) == 0),
        "per_case": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
