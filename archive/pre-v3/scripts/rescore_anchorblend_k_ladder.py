"""Test whether the deployed nearest-20 cap omits coherent-neighbour response.

One KD-tree query at the largest requested ``k`` is ranked before regression
cuts.  Smaller arms then keep exactly the raw neighbour-query slots that their
native query would have returned.  The k=20 arm must replay the stored V2.2
anchor prediction bit-for-bit, making larger-k differences an isolated
inference-aperture test.  Nothing is fitted and constgold is never read.
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

from blendemu import data_utils, nz_utils  # noqa: E402
from blendemu.inference import BlendingPredictor  # noqa: E402

COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"
TAG = "lsst_r_extnbr_v22"


def input_frame(base: str, case: int, sign: float) -> pd.DataFrame:
    path = os.path.join(
        base, f"case{case}_{str(float(sign))}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )
    frame = pd.read_feather(path)
    return frame.rename(columns={c: c.replace("_input", "") for c in frame.columns})


def score_ladder(predictor: BlendingPredictor, frame: pd.DataFrame,
                 anchors: np.ndarray, ks: list[int]) -> pd.DataFrame:
    cuts, r_max, native_k = predictor._select("regression")
    if r_max != 10 or native_k != 20:
        raise RuntimeError(f"unexpected deployed selection r_max={r_max}, k={native_k}")
    max_k = max(ks)
    raw = nz_utils.make_reg_features(
        frame, frame, r_max=float(r_max) / 3600.0, k=max_k,
    )
    primary = next(c for c in raw if c.startswith("index") and c.endswith("_p"))
    # The self match occupies query slot zero and is removed by r_min=0.  The
    # remaining rows are in increasing-distance query order, hence cumcount+1
    # restores the original KD-tree slot number used by native k arms.
    raw["query_slot"] = raw.groupby(primary, sort=False).cumcount().to_numpy(np.int16) + 1
    pairs = data_utils.source_select_reg(raw, cuts=cuts)
    pairs["distance"] *= 3600.0
    pairs = data_utils.rescale(
        pairs, zero_mag=COND["zero_point"], pixel_rms=COND["pixel_rms"],
        pixel_size=COND["pixel_size"], psf_fwhm=COND["psf_fwhm"],
        moffat_beta=COND["moffat_beta"],
    )
    scored = predictor.predict_on_pairs(
        pairs, task="response", rescaled=True, warn_extrapolation=False,
    )
    pairs["response"] = scored["response"].to_numpy(float)

    anchor_index = pd.Index(np.asarray(anchors, np.int64), name="input_index")
    out = pd.DataFrame(index=anchor_index)
    for k in ks:
        use = pairs["query_slot"].to_numpy(int) < int(k)
        group = pairs.loc[use].groupby(primary, sort=False)
        out[f"R_blend_k{k}"] = group["response"].sum().reindex(
            anchor_index, fill_value=0.0,
        ).to_numpy(float)
        out[f"n_pairs_k{k}"] = group.size().reindex(
            anchor_index, fill_value=0,
        ).to_numpy(np.int16)
    out["raw_neighbours_within_10"] = raw.groupby(primary, sort=False).size().reindex(
        anchor_index, fill_value=0,
    ).to_numpy(np.int16)
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--ks", type=int, nargs="+", default=[20, 32, 48, 64])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")
    if sorted(set(args.ks)) != args.ks or args.ks[0] != 20:
        raise ValueError("ks must be unique, sorted, and start with replay control k=20")

    truth = pd.read_feather(args.response)
    predictor = BlendingPredictor.load(
        "/home/z/Zekang.Zhang/blendemu/models", tag=TAG,
        conditions=COND, device="cpu",
    )
    parts = []
    for case, observed in truth.groupby("case", sort=True):
        observed = observed.copy()
        frame = input_frame(args.base, int(case), args.g)
        ladder = score_ladder(
            predictor, frame, observed["input_index"].to_numpy(np.int64), args.ks,
        )
        joined = observed.merge(ladder, on="input_index", validate="one_to_one")
        stored = joined[f"R_blend_{TAG}"].to_numpy(float)
        replay = joined["R_blend_k20"].to_numpy(float)
        replay_max_abs = float(np.max(np.abs(stored - replay)))
        if not np.allclose(stored, replay, rtol=1.0e-7, atol=5.0e-7):
            raise RuntimeError(
                f"case {case}: k20 replay failed; max abs={replay_max_abs:.3e}"
            )
        joined["k20_replay_max_abs_case"] = replay_max_abs
        parts.append(joined)
        means = " ".join(f"k{k}={joined[f'R_blend_k{k}'].mean():+.6f}" for k in args.ks)
        capped = np.mean(joined["raw_neighbours_within_10"].to_numpy(int) >= 20)
        print(
            f"case {int(case)}: anchors={len(joined):,} truth={joined.R_blend_truth.mean():+.6f} "
            f"{means} frac_raw_n>=20={capped:.3%} "
            f"k20-replay-max={replay_max_abs:.3e}", flush=True,
        )
    out = pd.concat(parts, ignore_index=True)
    out.to_feather(args.output)
    print(f"wrote {args.output}: {len(out):,} rows over {out.case.nunique()} cases")
    print("ANCHORBLEND_K_LADDER_DONE", flush=True)


if __name__ == "__main__":
    main()
