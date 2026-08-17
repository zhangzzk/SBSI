"""Prepare matched scenes with exactly one active deployed neighbour per anchor.

For every sparse anchor from an existing coherent-anchor simulation, one of the
deployed V2.2 neighbours is selected with equal probability by a stable hash.
The same neighbour is assigned antithetic shear in two orthogonal spin-2
directions in two otherwise identical output roots.  Primaries and all other
rendered sources remain unsheared.  The immutable pair manifest retains every
deployed pair so the selected response can later be Horvitz--Thompson weighted
back to the full per-anchor neighbour sum.
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
    assert_same_latents,
    generated_path,
    pair_id_columns,
    random_pair_directions,
)


COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
MODEL_DIR = "/home/z/Zekang.Zhang/blendemu/models"


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorised SplitMix64 finaliser for stable, row-order-independent ranks."""
    z = np.asarray(values, dtype=np.uint64).copy()
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return z


def select_one_pair(pairs: pd.DataFrame, case: int, seed: int) -> pd.DataFrame:
    """Mark one uniformly hashed secondary for every anchor with deployed pairs."""
    required = {"anchor_index", "secondary_index"}
    if missing := required - set(pairs.columns):
        raise KeyError(f"pair table lacks {sorted(missing)}")
    if pairs.duplicated(["anchor_index", "secondary_index"]).any():
        raise RuntimeError("duplicate deployed pair")
    out = pairs.copy()
    mask64 = (1 << 64) - 1
    anchor = out["anchor_index"].to_numpy(np.uint64)
    secondary = out["secondary_index"].to_numpy(np.uint64)
    case_term = np.uint64(((case + 1) * 0xD1B54A32D192ED03) & mask64)
    seed_term = np.uint64(seed & mask64)
    key = secondary + anchor * np.uint64(0x9E3779B185EBCA87) + case_term + seed_term
    out["selection_hash"] = splitmix64(key)
    out["n_pairs"] = out.groupby("anchor_index")["secondary_index"].transform("size")
    ordered = out.sort_values(
        ["anchor_index", "selection_hash", "secondary_index"], kind="mergesort",
    )
    selected_index = ordered.groupby("anchor_index", sort=False).head(1).index
    out["selected"] = False
    out.loc[selected_index, "selected"] = True
    if not (out.loc[out.selected].groupby("anchor_index").size() == 1).all():
        raise RuntimeError("failed to select exactly one pair per non-empty anchor")
    out["selection_probability"] = 1.0 / out["n_pairs"].to_numpy(float)
    return out


def assign_selected_shear(frame: pd.DataFrame, selected: pd.DataFrame,
                          direction: str, sign: float, g: float) -> pd.DataFrame:
    """Return a leg in which only selected secondaries receive shear."""
    if direction not in {"u", "v"}:
        raise ValueError(f"unknown direction {direction!r}")
    if selected["secondary_index"].duplicated().any():
        raise RuntimeError("one source is selected by more than one anchor")
    out = frame.copy()
    out[["g1", "g2"]] = 0.0
    columns = [f"{direction}1", f"{direction}2"]
    vectors = selected.set_index("secondary_index", verify_integrity=True)[columns]
    mapped = vectors.reindex(out["index"].to_numpy(np.int64))
    present = mapped[columns[0]].notna().to_numpy()
    if int(present.sum()) != len(selected):
        raise RuntimeError(
            f"selected-direction coverage {present.sum()} != {len(selected)}"
        )
    out.loc[present, "g1"] = sign * g * mapped.loc[present, columns[0]].to_numpy(float)
    out.loc[present, "g2"] = sign * g * mapped.loc[present, columns[1]].to_numpy(float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-base", required=True)
    ap.add_argument("--base-u", required=True)
    ap.add_argument("--base-v", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--tag", default="lsst_r_extnbr_v22")
    ap.add_argument("--selection-seed", type=int, default=2026081201)
    ap.add_argument("--direction-seed", type=int, default=2026081202)
    args = ap.parse_args()
    if len({os.path.abspath(args.reference_base), os.path.abspath(args.base_u),
            os.path.abspath(args.base_v)}) != 3:
        raise ValueError("reference, u and v roots must be distinct")

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=False)
    predictor = BlendingPredictor.load(
        MODEL_DIR, tag=args.tag, conditions=COND, device="cpu",
    )
    cuts, rmax, kmax = predictor._select("regression")
    if float(rmax) != 10.0 or int(kmax) != 20:
        raise RuntimeError(f"expected deployed r_max=10/k=20, got {rmax}/{kmax}")

    summary = {
        "design": "one uniformly selected deployed neighbour per coherent anchor; antithetic u/v orthogonal spin-2 directions",
        "reference_base": args.reference_base,
        "bases": {"u": args.base_u, "v": args.base_v},
        "cases": args.cases,
        "g": args.g,
        "tag": args.tag,
        "selection_seed": args.selection_seed,
        "direction_seed": args.direction_seed,
        "selection_estimator": "Horvitz-Thompson with weight n_pairs",
        "regression_cuts": cuts,
        "deployed_r_max_arcsec": float(rmax),
        "deployed_k": int(kmax),
        "per_case": [],
    }
    for case in args.cases:
        reference = pd.read_feather(generated_path(args.reference_base, case, +args.g))
        source_anchors = pd.read_feather(
            os.path.join(args.reference_base, f"anchors_case{case}.feather")
        )
        if source_anchors["index"].duplicated().any():
            raise RuntimeError(f"case {case}: duplicate reference anchor")
        anchor_ids = source_anchors["index"].to_numpy(np.int64)
        if not np.isin(anchor_ids, reference["index"].to_numpy(np.int64)).all():
            raise RuntimeError(f"case {case}: reference anchors missing from scene")

        for mode, base in (("u", args.base_u), ("v", args.base_v)):
            for sign in (+args.g, -args.g):
                path = generated_path(base, case, sign)
                frame = pd.read_feather(path)
                assert_same_latents(reference, frame, f"case {case} {mode} {sign:+g}")
                render_tree = os.path.join(base, f"case{case}_{str(float(sign))}")
                if os.path.exists(render_tree):
                    raise FileExistsError(f"refusing rendered tree {render_tree}")

        field = reference.rename(columns={c: c.replace("_input", "") for c in reference})
        anchor_frame = field[field["index"].isin(anchor_ids)].copy()
        pair_raw = predictor.predict_response(anchor_frame, field)
        pcol, scol = pair_id_columns(pair_raw)
        pairs = pair_raw.rename(columns={
            pcol: "anchor_index", scol: "secondary_index",
        }).copy()
        pairs["anchor_index"] = pairs["anchor_index"].astype(np.int64)
        pairs["secondary_index"] = pairs["secondary_index"].astype(np.int64)
        if not pairs["anchor_index"].isin(anchor_ids).all():
            raise RuntimeError(f"case {case}: predictor returned non-anchor primary")
        if not np.isfinite(pairs["response"].to_numpy(float)).all():
            raise RuntimeError(f"case {case}: non-finite deployed response")
        owner_count = pairs.groupby("secondary_index")["anchor_index"].nunique()
        if len(owner_count) and int(owner_count.max()) > 1:
            raise RuntimeError(f"case {case}: overlapping anchor neighbourhoods")
        pairs = select_one_pair(pairs, case, args.selection_seed)
        chosen = pairs.loc[pairs.selected].copy()
        u1, u2 = random_pair_directions(
            case, chosen["secondary_index"].to_numpy(np.int64), 0,
            args.direction_seed,
        )
        chosen["u1"], chosen["u2"] = u1, u2
        chosen["v1"], chosen["v2"] = -u2, u1
        for name in ("u", "v"):
            norm = np.hypot(chosen[f"{name}1"], chosen[f"{name}2"])
            if not np.allclose(norm, 1.0, rtol=0.0, atol=2e-15):
                raise RuntimeError(f"case {case}: non-unit {name} direction")
        dot = chosen["u1"] * chosen["v1"] + chosen["u2"] * chosen["v2"]
        if not np.allclose(dot, 0.0, rtol=0.0, atol=2e-15):
            raise RuntimeError(f"case {case}: u/v directions are not orthogonal")
        direction_columns = chosen.set_index(
            ["anchor_index", "secondary_index"], verify_integrity=True,
        )[["u1", "u2", "v1", "v2"]]
        pairs = pairs.join(direction_columns, on=["anchor_index", "secondary_index"])
        keep = [
            "anchor_index", "secondary_index", "distance", "response", "n_pairs",
            "selection_hash", "selection_probability", "selected",
            "u1", "u2", "v1", "v2",
        ]
        pair_out = pairs[keep].copy()
        pair_out.insert(0, "case", int(case))
        pair_out.to_feather(manifest_dir / f"pairs_case{case}.feather")
        source_anchors.to_feather(manifest_dir / f"anchors_case{case}.feather")
        source_anchors.to_feather(os.path.join(args.base_u, f"anchors_case{case}.feather"))
        source_anchors.to_feather(os.path.join(args.base_v, f"anchors_case{case}.feather"))

        for mode, base in (("u", args.base_u), ("v", args.base_v)):
            for sign in (+1.0, -1.0):
                path = generated_path(base, case, sign * args.g)
                frame = pd.read_feather(path)
                assigned = assign_selected_shear(frame, chosen, mode, sign, args.g)
                assigned.to_feather(path)

        counts = pairs.groupby("anchor_index")["secondary_index"].size()
        summary["per_case"].append({
            "case": int(case),
            "n_anchors": int(len(anchor_ids)),
            "n_anchors_with_pairs": int(counts.size),
            "n_pairs": int(len(pairs)),
            "n_selected_pairs": int(len(chosen)),
            "mean_pairs_per_anchor": float(len(pairs) / len(anchor_ids)),
            "max_pairs_per_anchor": int(counts.max()) if len(counts) else 0,
            "exact_model_sum_per_anchor": float(pairs["response"].sum() / len(anchor_ids)),
            "ht_selected_model_sum_per_anchor": float(
                (chosen["response"] * chosen["n_pairs"]).sum() / len(anchor_ids)
            ),
        })
        print(
            f"case {case}: anchors={len(anchor_ids):,} with-pairs={counts.size:,} "
            f"pairs={len(pairs):,} selected={len(chosen):,} "
            f"<k>={len(pairs)/len(anchor_ids):.3f}", flush=True,
        )

    with open(manifest_dir / "design.json", "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"wrote immutable design -> {manifest_dir}")
    print("ANCHORBLEND_ONEACTIVE_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
