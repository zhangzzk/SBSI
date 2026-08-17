"""Rescore direct neighbour-response anchors with pair-other-flux BlendEMU.

The stored direct truth and V2.2 prediction are immutable.  V2.2 is replayed
from each anchor input field as an exact control.  For each deployed pair, a
precomputed full-scene shell lookup is joined by primary key, the designated
secondary is subtracted, and the frozen ``other3abs`` emulator is scored.  No
truth quantity is used to construct a feature or prediction.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu.inference import BlendingPredictor  # noqa: E402
from blendemu.scene_features import (  # noqa: E402
    PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS,
    pair_other_absolute_flux,
)


COND = dict(
    pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
    moffat_beta=2.224, pixel_rms=0.312,
)
TILE = "tile180.0_-0.5"
SCENE_COLUMNS = ["logflux_near_0_1", "logflux_mid_1_3", "logflux_far_3_10"]
BASE_TAG = "lsst_r_extnbr_v22"
CANDIDATE_TAG = "lsst_r_extnbr_v22_other3abs"


def input_frame(base, case, sign):
    path = os.path.join(
        base, f"case{case}_{str(float(sign))}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )
    frame = pd.read_feather(path)
    return frame.rename(columns={column: column.replace("_input", "") for column in frame})


def read_scene(path, case_min, case_max):
    scene = pd.read_feather(path, columns=["case", "input_index", *SCENE_COLUMNS])
    scene = scene[scene.case.between(case_min, case_max)].copy()
    if scene.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate anchor scene lookup key")
    return scene


def score_case(frame, scene_case, baseline, candidate, anchors):
    pairs = baseline.predict_response(frame, frame)
    primary = next(
        column for column in pairs if column.startswith("index") and column.endswith("_p")
    )
    scene_by_id = scene_case.set_index("input_index", verify_integrity=True)
    primary_id = pairs[primary].to_numpy(np.int64)
    missing = np.setdiff1d(np.unique(primary_id), scene_by_id.index.to_numpy(np.int64))
    if len(missing):
        raise RuntimeError(f"scene lookup misses {len(missing):,} deployed primaries")
    total_scene = scene_by_id.loc[primary_id, SCENE_COLUMNS].to_numpy(float)
    other = pair_other_absolute_flux(
        total_scene,
        pairs["r_input_p"].to_numpy(float), pairs["r_input_s"].to_numpy(float),
        pairs["distance"].to_numpy(float), zero_point=COND["zero_point"],
    )
    for index, name in enumerate(PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS):
        pairs[name] = other[:, index].astype(np.float32)
    pairs[f"R_blend_{BASE_TAG}"] = pairs.pop("response").to_numpy(float)
    candidate_scored = candidate.predict_on_pairs(
        pairs, task="response", rescaled=True, warn_extrapolation=False,
    )
    pairs[f"R_blend_{CANDIDATE_TAG}"] = candidate_scored["response"].to_numpy(float)

    anchor_index = pd.Index(np.asarray(anchors, np.int64), name="input_index")
    group = pairs.groupby(primary, sort=False)
    out = pd.DataFrame(index=anchor_index)
    out["n_pairs"] = group.size().reindex(anchor_index, fill_value=0).to_numpy(np.int16)
    for tag in (BASE_TAG, CANDIDATE_TAG):
        column = f"R_blend_{tag}"
        out[column] = group[column].sum().reindex(
            anchor_index, fill_value=0.0,
        ).to_numpy(float)
    return out.reset_index(), len(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--scene-lookup", required=True)
    ap.add_argument("--case-min", type=int, default=200)
    ap.add_argument("--case-max", type=int, default=299)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    truth = pd.read_feather(args.response)
    truth = truth[truth.case.between(args.case_min, args.case_max)].copy()
    required = {"case", "input_index", "R_blend_truth", f"R_blend_{BASE_TAG}"}
    missing = required - set(truth)
    if missing:
        raise KeyError(f"response table lacks {sorted(missing)}")
    if truth.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate anchor truth key")
    scene = read_scene(args.scene_lookup, args.case_min, args.case_max)
    baseline = BlendingPredictor.load(
        f"{BE}/models", tag=BASE_TAG, conditions=COND, device="cpu",
    )
    candidate = BlendingPredictor.load(
        f"{BE}/models", tag=CANDIDATE_TAG, conditions=COND, device="cpu",
    )

    parts = []
    for case, observed in truth.groupby("case", sort=True):
        frame = input_frame(args.base, int(case), args.g)
        scored, n_pairs = score_case(
            frame, scene[scene.case == case], baseline, candidate,
            observed["input_index"].to_numpy(np.int64),
        )
        joined = observed.merge(scored, on="input_index", validate="one_to_one",
                                suffixes=("_stored", ""))
        stored = joined[f"R_blend_{BASE_TAG}_stored"].to_numpy(float)
        replay = joined[f"R_blend_{BASE_TAG}"].to_numpy(float)
        replay_max = float(np.max(np.abs(stored - replay)))
        if not np.allclose(stored, replay, rtol=1e-7, atol=5e-7):
            raise RuntimeError(f"case {case}: V2.2 replay failed; max abs={replay_max:.3e}")
        joined = joined.drop(columns=f"R_blend_{BASE_TAG}_stored")
        parts.append(joined)
        print(
            f"case {int(case)}: anchors={len(joined):,} pairs={n_pairs:,} "
            f"truth={joined.R_blend_truth.mean():+.6f} "
            f"v22={joined[f'R_blend_{BASE_TAG}'].mean():+.6f} "
            f"other3abs={joined[f'R_blend_{CANDIDATE_TAG}'].mean():+.6f} "
            f"replay={replay_max:.2e}", flush=True,
        )
    output = pd.concat(parts, ignore_index=True)
    output.to_feather(args.output)
    print(
        f"wrote {args.output}: {len(output):,} rows over {output.case.nunique()} cases",
        flush=True,
    )
    print("ANCHORBLEND_OTHER3ABS_RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
