"""Replay frozen V2.2 constgold pairs and retain per-primary response dominance.

The selection coordinate is identical to the anchored-scene diagnostic:

    max(abs(R_pair)) / max(second_largest(abs(R_pair)), 1e-12)

This is an evaluation feature, not a correction.  The stored V2.2 total response
lookup is replayed and checked key-for-key so pair-selection drift cannot pass
silently.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as pf

from sbs_shear.paths import CONST_SIM_DIR


BLENDEMU_ROOT = "/home/z/Zekang.Zhang/blendemu"
MODEL_DIR = os.path.join(BLENDEMU_ROOT, "models")
TILE = "tile180.0_-0.5"
COND = dict(
    pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
    moffat_beta=2.224, pixel_rms=0.312,
)


def input_feather(case: int, sign: str, base: str) -> str:
    return (
        f"{base}/case{case}_{sign}/real0/catalogues/input/"
        f"gals_info_{TILE}.feather"
    )


def summarize_pair_responses(primary: np.ndarray, response: np.ndarray) -> pd.DataFrame:
    """Return exact sums and top-two absolute responses for every primary."""
    primary = np.asarray(primary, dtype=np.int64)
    response = np.asarray(response, dtype=float)
    if len(primary) != len(response) or not len(primary):
        raise ValueError("primary and response must be non-empty arrays of equal length")
    if not np.isfinite(response).all():
        raise ValueError("non-finite pair response")

    absolute = np.abs(response)
    # Sorting by (primary, absolute response) makes the last two rows in each
    # primary block the exact largest and runner-up, including tied maxima.
    order = np.lexsort((absolute, primary))
    p = primary[order]
    r = response[order]
    a = absolute[order]
    starts = np.r_[0, np.flatnonzero(p[1:] != p[:-1]) + 1]
    ends = np.r_[starts[1:], len(p)]
    counts = ends - starts
    largest = a[ends - 1]
    runner = np.zeros(len(starts), dtype=float)
    multiple = counts >= 2
    runner[multiple] = a[ends[multiple] - 2]
    total = np.add.reduceat(r, starts)
    absolute_total = np.add.reduceat(a, starts)
    return pd.DataFrame({
        "input_index": p[starts],
        "R_blend": total,
        "n_pairs": counts.astype(np.int16),
        "dominant_abs_response": largest,
        "runner_up_abs_response": runner,
        "R_abs_sum": absolute_total,
        "dominant_to_runner_up_abs_response": (
            largest / np.maximum(runner, 1.0e-12)
        ),
        "top_abs_fraction": largest / absolute_total,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--reference-lookup", required=True)
    ap.add_argument("--base", default=CONST_SIM_DIR)
    ap.add_argument("--sign", default="0.02")
    ap.add_argument("--tag", default="lsst_r_extnbr_v22")
    # The original lookup was serialized after a pandas groupby, while this
    # diagnostic sorts the same float32 pair responses before reducing them.
    # Those different summation orders differ at a few 1e-7 per object.
    ap.add_argument("--replay-max-atol", type=float, default=1.0e-6)
    ap.add_argument("--replay-mean-atol", type=float, default=1.0e-8)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing existing output {args.output}")

    if BLENDEMU_ROOT not in sys.path:
        sys.path.insert(0, BLENDEMU_ROOT)
    from blendemu.inference import BlendingPredictor

    predictor = BlendingPredictor.load(
        MODEL_DIR, tag=args.tag, conditions=COND, device="cpu",
    )
    cuts, rmax, k = predictor._select("regression")
    if not np.isclose(float(rmax), 10.0) or int(k) != 20:
        raise RuntimeError(f"expected frozen V2.2 r_max=10/k=20, got {rmax}/{k}")

    reference = pf.read_table(
        args.reference_lookup, columns=["case", "input_index", "R_blend"],
    ).to_pandas()
    reference = reference[reference["case"].isin(args.cases)].copy()
    if reference.duplicated(["case", "input_index"]).any():
        raise RuntimeError("reference lookup has duplicate keys")

    parts: list[pd.DataFrame] = []
    for case in args.cases:
        path = input_feather(case, args.sign, args.base)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        field = pf.read_table(path).to_pandas().rename(
            columns=lambda name: name.replace("_input", "")
        )
        pairs = predictor.predict_response(field, field)
        primary_columns = [name for name in pairs if name.startswith("index")]
        if len(primary_columns) < 2:
            raise RuntimeError(f"case {case}: cannot identify pair IDs {primary_columns}")
        primary_column = primary_columns[0]
        summary = summarize_pair_responses(
            pairs[primary_column].to_numpy(np.int64),
            pairs["response"].to_numpy(float),
        )
        summary.insert(0, "case", case)

        expected = reference.loc[reference["case"] == case].sort_values("input_index")
        actual = summary.sort_values("input_index")
        if not np.array_equal(
            expected["input_index"].to_numpy(np.int64),
            actual["input_index"].to_numpy(np.int64),
        ):
            raise RuntimeError(f"case {case}: replay/reference keys differ")
        difference = (
            actual["R_blend"].to_numpy(float)
            - expected["R_blend"].to_numpy(float)
        )
        maximum = float(np.max(np.abs(difference)))
        mean_absolute = float(np.mean(np.abs(difference)))
        if maximum > args.replay_max_atol or mean_absolute > args.replay_mean_atol:
            raise RuntimeError(
                f"case {case}: lookup replay mismatch max={maximum:.3e}, "
                f"meanabs={mean_absolute:.3e}"
            )
        parts.append(summary)
        cut_fraction = float(np.mean(
            summary["dominant_to_runner_up_abs_response"].to_numpy(float) > 20.0
        ))
        print(
            f"case{case}: {len(summary):,} primaries, {len(pairs):,} pairs, "
            f"ratio>20={cut_fraction:.2%}, replay={maximum:.2e}/{mean_absolute:.2e}",
            flush=True,
        )

    output = pd.concat(parts, ignore_index=True)
    output.to_feather(args.output)
    print(
        f"wrote {args.output}: {len(output):,} rows over "
        f"{output['case'].nunique()} cases; regression cuts={cuts}"
    )
    print("V22_CONSTGOLD_RESPONSE_DOMINANCE_DONE", flush=True)


if __name__ == "__main__":
    main()
