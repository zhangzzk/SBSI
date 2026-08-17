"""Emit the deployed per-pair V2.2 response manifest for a coherent-anchor tree.

``localize_anchorblend_response_dominance.py`` needs, per anchor, the full list of
deployed pairs with their predicted responses so it can form the single-pair
dominance ratio ``max|R_pair| / runner-up|R_pair|``.  For cases 400--499 that list
came for free from the one-active derisk manifest
(``prepare_anchorblend_oneactive_orthogonal.py``), which builds it as a by-product
of a much heavier design: it requires two extra rendered roots and assigns
orthogonal u/v shear directions.

This script extracts only the part the dominance table uses -- the pair list and
its predicted responses -- straight from a single anchor tree, so the g=0.02
extension blocks can be scored without rendering anything extra.  The predictor
call, the deployed-configuration assertion (``r_max = 10``, ``k = 20``) and the
non-overlapping-neighbourhood check are the same as in the one-active script.

Nothing is fitted, no correction is applied and constgold is not opened.  Output
is written as ``pairs_case{case}.feather`` inside ``--manifest-dir``, matching the
layout the dominance builder expects (which also reads ``anchors_case{case}.feather``
from that directory).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
BLENDEMU_ROOT = "/home/z/Zekang.Zhang/blendemu"
if BLENDEMU_ROOT not in sys.path:
    sys.path.insert(0, BLENDEMU_ROOT)

from blendemu.inference import BlendingPredictor  # noqa: E402
from scripts.prepare_v22_matched_decomposition import (  # noqa: E402
    generated_path,
    input_path,
    pair_id_columns,
)

COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
MODEL_DIR = "/home/z/Zekang.Zhang/blendemu/models"
KEEP = ["case", "anchor_index", "secondary_index", "distance", "response", "n_pairs"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True,
                    help="coherent-anchor tree holding gals{case}_{g}.feather "
                         "and anchors_case{case}.feather")
    ap.add_argument("--manifest-dir", required=True,
                    help="output directory for pairs_case{case}.feather")
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.02,
                    help="rendered leg whose latent catalogue is read; the "
                         "latents are shear-independent, so the choice only "
                         "picks a filename")
    ap.add_argument(
        "--catalogue-source", choices=("generated", "rendered"),
        default="generated",
        help="catalogue used for neighbour finding and model inference. "
             "'rendered' reads the exact simulator input catalogue used by "
             "build_anchorblend_response.py; use it when pair sums must replay "
             "the stored scene prediction exactly",
    )
    ap.add_argument(
        "--pair-prefix", default="pairs",
        help="output prefix before _case{case}.feather (default: pairs)",
    )
    ap.add_argument("--tag", default="lsst_r_extnbr_v22")
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args()

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        target = manifest_dir / f"{args.pair_prefix}_case{case}.feather"
        if target.exists():
            raise SystemExit(f"REFUSING to overwrite {target}")

    predictor = BlendingPredictor.load(
        MODEL_DIR, tag=args.tag, conditions=COND, device="cpu",
    )
    cuts, rmax, kmax = predictor._select("regression")
    if float(rmax) != 10.0 or int(kmax) != 20:
        raise RuntimeError(f"expected deployed r_max=10/k=20, got {rmax}/{kmax}")

    per_case = []
    for case in args.cases:
        catalogue_path = (
            generated_path(args.base, case, +args.g)
            if args.catalogue_source == "generated"
            else input_path(args.base, case, +args.g)
        )
        field_raw = pd.read_feather(catalogue_path)
        field = field_raw.rename(
            columns={c: c.replace("_input", "") for c in field_raw}
        )
        anchors = pd.read_feather(
            os.path.join(args.base, f"anchors_case{case}.feather")
        )
        if anchors["index"].duplicated().any():
            raise RuntimeError(f"case {case}: duplicate anchor")
        anchor_ids = anchors["index"].to_numpy(np.int64)
        if not np.isin(anchor_ids, field["index"].to_numpy(np.int64)).all():
            raise RuntimeError(f"case {case}: anchors missing from scene")

        anchor_frame = field[field["index"].isin(anchor_ids)].copy()
        pair_raw = predictor.predict_response(anchor_frame, field)
        pcol, scol = pair_id_columns(pair_raw)
        pairs = pair_raw.rename(
            columns={pcol: "anchor_index", scol: "secondary_index"}).copy()
        pairs["anchor_index"] = pairs["anchor_index"].astype(np.int64)
        pairs["secondary_index"] = pairs["secondary_index"].astype(np.int64)
        if pairs.duplicated(["anchor_index", "secondary_index"]).any():
            raise RuntimeError(f"case {case}: duplicate deployed pair")
        if not pairs["anchor_index"].isin(anchor_ids).all():
            raise RuntimeError(f"case {case}: predictor returned non-anchor primary")
        if not np.isfinite(pairs["response"].to_numpy(float)).all():
            raise RuntimeError(f"case {case}: non-finite deployed response")
        owner_count = pairs.groupby("secondary_index")["anchor_index"].nunique()
        if len(owner_count) and int(owner_count.max()) > 1:
            raise RuntimeError(f"case {case}: overlapping anchor neighbourhoods")

        pairs["case"] = case
        pairs["n_pairs"] = pairs.groupby("anchor_index")[
            "secondary_index"].transform("size")
        pairs = pairs[KEEP].reset_index(drop=True)
        pairs.to_feather(
            manifest_dir / f"{args.pair_prefix}_case{case}.feather"
        )
        per_case.append({
            "case": case,
            "n_anchors": int(pairs.anchor_index.nunique()),
            "n_pairs": int(len(pairs)),
            "mean_pairs_per_anchor": float(pairs.groupby("anchor_index").size().mean()),
        })
        print(f"case {case}: anchors={per_case[-1]['n_anchors']:,} "
              f"pairs={per_case[-1]['n_pairs']:,} "
              f"per_anchor={per_case[-1]['mean_pairs_per_anchor']:.2f}", flush=True)

    if args.summary_json:
        summary = {
            "design": "deployed per-pair V2.2 responses for coherent anchors",
            "base": args.base,
            "manifest_dir": args.manifest_dir,
            "tag": args.tag,
            "g": args.g,
            "catalogue_source": args.catalogue_source,
            "pair_prefix": args.pair_prefix,
            "regression_cuts": cuts,
            "deployed_r_max_arcsec": float(rmax),
            "deployed_k": int(kmax),
            "per_case": per_case,
        }
        with open(args.summary_json, "x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1, sort_keys=True)
            handle.write("\n")
        print("WROTE", args.summary_json)
    print("ANCHORBLEND_PAIR_RESPONSES_DONE", flush=True)


if __name__ == "__main__":
    main()
