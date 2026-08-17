"""Rescore coherent-neighbour anchors with distance-definition model variants.

The deployed V2.2 response emulator was trained with primary-detected-centroid
separations but is queried with input-to-input separations.  Existing ``indist``
models repair that definition and, for the ``wc*`` variants, up-weight rare
close pairs.  This script compares those frozen models on the independent
anchorblend instrument without reading constgold or fitting a correction.

The stored anchor response table supplies only the direct coherent-neighbour
truth.  Pair predictions are replayed from the corresponding input fields.  A
V2.2 replay must agree bit-for-bit with the stored V2.2 prediction before any
new model result is accepted.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable

import numpy as np
import pandas as pd

BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu.inference import BlendingPredictor  # noqa: E402

COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"
SHELL_EDGES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])


def shell_name(lo: float, hi: float) -> str:
    def fmt(x: float) -> str:
        return str(x).replace(".0", "").replace(".", "p")
    return f"d{fmt(lo)}_{fmt(hi)}"


def aggregate_pairs(pairs: pd.DataFrame, anchor_ids: Iterable[int],
                    response_columns: Iterable[str]) -> pd.DataFrame:
    """Aggregate pair responses and counts by anchor and distance shell."""
    anchors = pd.Index(np.asarray(list(anchor_ids), dtype=np.int64), name="input_index")
    primary = next(c for c in pairs if c.startswith("index") and c.endswith("_p"))
    if not np.isfinite(pairs["distance"].to_numpy(float)).all():
        raise RuntimeError("non-finite pair distance")
    out = pd.DataFrame(index=anchors)
    grouped = pairs.groupby(primary, sort=False)
    out["n_pairs"] = grouped.size().reindex(anchors, fill_value=0).to_numpy(np.int32)
    for column in response_columns:
        out[column] = grouped[column].sum().reindex(anchors, fill_value=0.0).to_numpy(float)
    distance = pairs["distance"].to_numpy(float)
    for lo, hi in zip(SHELL_EDGES[:-1], SHELL_EDGES[1:]):
        name = shell_name(lo, hi)
        mask = (distance >= lo) & (distance < hi)
        shell = pairs.loc[mask]
        shell_grouped = shell.groupby(primary, sort=False)
        out[f"n_{name}"] = shell_grouped.size().reindex(anchors, fill_value=0).to_numpy(np.int32)
        for column in response_columns:
            out[f"{column}_{name}"] = (
                shell_grouped[column].sum().reindex(anchors, fill_value=0.0).to_numpy(float)
            )
    return out.reset_index()


def input_frame(base: str, case: int, sign: float) -> pd.DataFrame:
    path = os.path.join(
        base, f"case{case}_{str(float(sign))}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )
    frame = pd.read_feather(path)
    return frame.rename(columns={c: c.replace("_input", "") for c in frame.columns})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", required=True, help="stored direct-anchor response Feather")
    ap.add_argument("--base", required=True, help="anchor simulation root")
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--tags", nargs="+", default=[
        "lsst_r_extnbr_v22", "lsst_r_extnbr_indist",
        "lsst_r_extnbr_indist_wc5", "lsst_r_extnbr_indist_wc20",
    ])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")
    truth = pd.read_feather(args.response)
    required = {"case", "input_index", "R_blend_truth", "R_blend_lsst_r_extnbr_v22"}
    missing = required - set(truth)
    if missing:
        raise KeyError(f"response table lacks {sorted(missing)}")
    if truth.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate anchor keys in response table")

    predictors = {
        tag: BlendingPredictor.load(
            "/home/z/Zekang.Zhang/blendemu/models", tag=tag,
            conditions=COND, device="cpu",
        )
        for tag in args.tags
    }
    base_tag = "lsst_r_extnbr_v22"
    if base_tag not in predictors:
        raise ValueError(f"tags must include replay control {base_tag}")

    parts = []
    response_columns = [f"R_blend_{tag}" for tag in args.tags]
    for case, observed in truth.groupby("case", sort=True):
        observed = observed.copy()
        anchors = observed["input_index"].to_numpy(np.int64)
        frame = input_frame(args.base, int(case), args.g)
        pairs = predictors[base_tag].predict_response(frame, frame)
        pairs[f"R_blend_{base_tag}"] = pairs.pop("response")
        for tag, predictor in predictors.items():
            if tag == base_tag:
                continue
            scored = predictor.predict_on_pairs(
                pairs, task="response", rescaled=True, warn_extrapolation=False,
            )
            pairs[f"R_blend_{tag}"] = scored["response"].to_numpy(float)

        aggregate = aggregate_pairs(pairs, anchors, response_columns)
        joined = observed.merge(aggregate, on="input_index", how="left",
                                validate="one_to_one", suffixes=("_stored", ""))
        replay = joined[f"R_blend_{base_tag}"].to_numpy(float)
        stored = joined[f"R_blend_{base_tag}_stored"].to_numpy(float)
        if not np.array_equal(replay, stored):
            max_abs = float(np.max(np.abs(replay - stored)))
            raise RuntimeError(f"case {case}: V2.2 replay differs; max abs={max_abs:.3e}")
        joined = joined.drop(columns=f"R_blend_{base_tag}_stored")
        parts.append(joined)
        means = " ".join(
            f"{tag}={joined[f'R_blend_{tag}'].mean():+.5f}" for tag in args.tags
        )
        print(
            f"case {int(case)}: anchors={len(joined):,} pairs={len(pairs):,} "
            f"truth={joined['R_blend_truth'].mean():+.5f} {means}", flush=True,
        )

    out = pd.concat(parts, ignore_index=True)
    out.to_feather(args.output)
    print(f"wrote {args.output}: rows={len(out):,} cases={out['case'].nunique()}")
    print("ANCHORBLEND_DISTANCE_RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
