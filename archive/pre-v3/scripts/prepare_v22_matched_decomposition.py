"""Prepare matched V2.2 response-decomposition simulation catalogues.

The generated catalogues in every mode start from the same intrinsic FS2 scene.
Sparse V2.2-domain anchors are separated by more than twice the deployed 10 arcsec
BlendEMU aperture.  The exact V2.2 ``predict_response`` pair table defines which
neighbours are in scope.  Four antithetic shear treatments are then assigned:

``total``
    Anchor and all of its deployed neighbours receive coherent +/-g1.
``self``
    Only the anchor receives coherent +/-g1.
``neighbour``
    Only the deployed neighbours receive coherent +/-g1.
``pair0`` (and optional further ``pairN`` modes)
    Deployed neighbours receive seeded random spin-2 directions, antithetic
    between legs; anchors remain unsheared.  These are Hutchinson probes of the
    sum of individual-pair derivatives.

All other rendered sources remain present but unsheared.  Therefore the later
same-anchor contrasts separate flow error, emulator error, self--neighbour
interaction, and multi-neighbour non-additivity without changing the light scene.
No constgold quantity is read.
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
from sbs_shear.anchorblend import assert_anchor_spacing, sparse_anchor_mask  # noqa: E402


COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
MODEL_DIR = "/home/z/Zekang.Zhang/blendemu/models"
TILE = "tile180.0_-0.5"
IDENTITY_EXCLUDE = {"g1", "g2"}


def label(g: float) -> str:
    return str(float(g))


def generated_path(base: str, case: int, sign: float) -> str:
    return os.path.join(base, f"gals{case}_{label(sign)}.feather")


def input_path(base: str, case: int, sign: float) -> str:
    return os.path.join(
        base, f"case{case}_{label(sign)}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )


def v22_primary_mask(frame: pd.DataFrame) -> np.ndarray:
    mag = frame["r"].to_numpy(float)
    size = frame["Re"].to_numpy(float)
    return (np.isfinite(mag) & np.isfinite(size)
            & (mag > 18.0) & (mag < 25.8)
            & (size > 0.5) & (size < 1.5))


def assert_same_latents(reference: pd.DataFrame, other: pd.DataFrame, label_: str) -> None:
    """Require byte-equivalent latent catalogues after stable-ID alignment."""
    if set(reference.columns) != set(other.columns):
        raise RuntimeError(f"{label_}: generated schemas differ")
    if reference["index"].duplicated().any() or other["index"].duplicated().any():
        raise RuntimeError(f"{label_}: duplicate stable input index")
    rhs = other.set_index("index").reindex(reference["index"]).reset_index()
    if rhs["index"].isna().any():
        raise RuntimeError(f"{label_}: stable input IDs differ")
    for col in reference.columns:
        if col in IDENTITY_EXCLUDE:
            continue
        if not reference[col].reset_index(drop=True).equals(rhs[col].reset_index(drop=True)):
            raise RuntimeError(f"{label_}: latent column {col!r} differs")


def pair_id_columns(pairs: pd.DataFrame) -> tuple[str, str]:
    candidates = [c for c in pairs if c.startswith("index") and c.endswith("_p")]
    secondaries = [c for c in pairs if c.startswith("index") and c.endswith("_s")]
    if len(candidates) != 1 or len(secondaries) != 1:
        raise RuntimeError(
            f"cannot identify unique pair IDs in columns {pairs.columns.tolist()}"
        )
    return candidates[0], secondaries[0]


def random_pair_directions(case: int, secondary_ids: np.ndarray, probe: int,
                           seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable spin-2 directions, independent across case/probe/secondary."""
    secondary_ids = np.asarray(secondary_ids, np.uint64)
    # SplitMix-style integer hashing avoids dependence on pair-table row order.
    mask64 = (1 << 64) - 1
    case_term = np.uint64(((case + 1) * 0x9E3779B185EBCA87) & mask64)
    probe_term = np.uint64(((probe + 1) * 0xD1B54A32D192ED03) & mask64)
    z = (secondary_ids
         + np.uint64(seed)
         + case_term + probe_term)
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    unit = ((z >> np.uint64(11)).astype(np.float64)
            * (1.0 / float(1 << 53)))
    angle = 2.0 * np.pi * unit
    return np.cos(angle), np.sin(angle)


def assign_shear(frame: pd.DataFrame, mode: str, sign: float, g: float,
                 anchors: np.ndarray, neighbour_ids: np.ndarray,
                 random_by_id: pd.DataFrame | None) -> pd.DataFrame:
    """Return one antithetic treatment leg from a common latent catalogue."""
    out = frame.copy()
    out[["g1", "g2"]] = 0.0
    ids = out["index"].to_numpy(np.int64)
    anchor = np.isin(ids, anchors)
    neighbour = np.isin(ids, neighbour_ids)
    direction = 1.0 if sign > 0 else -1.0
    if mode == "total":
        selected = anchor | neighbour
        out.loc[selected, "g1"] = direction * g
    elif mode == "self":
        out.loc[anchor, "g1"] = direction * g
    elif mode == "neighbour":
        out.loc[neighbour, "g1"] = direction * g
    elif mode.startswith("pair"):
        if random_by_id is None:
            raise ValueError(f"{mode}: random directions are required")
        mapped = pd.DataFrame({"index": ids}).merge(
            random_by_id, on="index", how="left", validate="one_to_one"
        )
        present = mapped["u1"].notna().to_numpy()
        if int(present.sum()) != int(neighbour.sum()):
            raise RuntimeError(
                f"{mode}: random-direction coverage {present.sum()} != neighbours {neighbour.sum()}"
            )
        out.loc[present, "g1"] = direction * g * mapped.loc[present, "u1"].to_numpy(float)
        out.loc[present, "g2"] = direction * g * mapped.loc[present, "u2"].to_numpy(float)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", action="append", required=True,
                    help="MODE=PATH; required modes are total,self,neighbour and >=1 pairN")
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--min-separation", type=float, default=20.01)
    ap.add_argument("--tag", default="lsst_r_extnbr_v22")
    ap.add_argument("--random-seed", type=int, default=25876)
    ap.add_argument("--manifest-dir", required=True)
    args = ap.parse_args()

    bases: dict[str, str] = {}
    for item in args.base:
        mode, sep, path = item.partition("=")
        if not sep or not mode or not path:
            raise ValueError(f"--base must be MODE=PATH, got {item!r}")
        if mode in bases:
            raise ValueError(f"duplicate mode {mode!r}")
        bases[mode] = path
    required = {"total", "self", "neighbour"}
    if not required.issubset(bases) or not any(m.startswith("pair") for m in bases):
        raise ValueError(f"need {sorted(required)} and at least one pairN mode; got {sorted(bases)}")
    pair_modes = sorted(m for m in bases if m.startswith("pair"))
    for i, mode in enumerate(pair_modes):
        if mode != f"pair{i}":
            raise ValueError(f"pair modes must be contiguous pair0..pairN, got {pair_modes}")

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if any(manifest_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty manifest directory {manifest_dir}")

    predictor = BlendingPredictor.load(
        MODEL_DIR, tag=args.tag, conditions=COND, device="cpu",
    )
    cuts, rmax, k = predictor._select("regression")
    if float(rmax) != 10.0 or int(k) != 20:
        raise RuntimeError(f"expected deployed r_max=10/k=20, got {rmax}/{k}")

    summary = {
        "tag": args.tag, "g": args.g, "cases": args.cases,
        "min_separation_arcsec": args.min_separation,
        "random_seed": args.random_seed, "modes": sorted(bases),
        "deployed_r_max_arcsec": float(rmax), "deployed_k": int(k),
        "regression_cuts": cuts, "per_case": [],
    }
    for case in args.cases:
        # Every generated sign and mode must be the identical latent scene.
        reference = pd.read_feather(generated_path(bases["total"], case, +args.g))
        for mode, base in bases.items():
            for sign in (+args.g, -args.g):
                path = generated_path(base, case, sign)
                frame = pd.read_feather(path)
                assert_same_latents(reference, frame, f"case {case} {mode} {sign:+g}")
                render_tree = os.path.join(base, f"case{case}_{label(sign)}")
                if os.path.exists(render_tree):
                    raise FileExistsError(f"refusing already-rendered tree {render_tree}")

        eligible = v22_primary_mask(reference)
        anchor_mask = sparse_anchor_mask(
            reference["RA"], reference["DEC"], eligible,
            min_separation_arcsec=args.min_separation,
        )
        minimum = assert_anchor_spacing(
            reference["RA"], reference["DEC"], anchor_mask,
            min_separation_arcsec=args.min_separation,
        )
        anchors = reference.loc[anchor_mask, "index"].to_numpy(np.int64)
        if len(anchors) < 100:
            raise RuntimeError(f"case {case}: only {len(anchors)} anchors")

        field = reference.rename(columns={c: c.replace("_input", "") for c in reference.columns})
        anchor_frame = field[field["index"].isin(anchors)].copy()
        pairs = predictor.predict_response(anchor_frame, field)
        pcol, scol = pair_id_columns(pairs)
        pairs = pairs.rename(columns={pcol: "anchor_index", scol: "secondary_index"})
        pairs["anchor_index"] = pairs["anchor_index"].astype(np.int64)
        pairs["secondary_index"] = pairs["secondary_index"].astype(np.int64)
        if not pairs["anchor_index"].isin(anchors).all():
            raise RuntimeError(f"case {case}: predictor returned a non-anchor primary")
        if pairs.duplicated(["anchor_index", "secondary_index"]).any():
            raise RuntimeError(f"case {case}: duplicate deployed pair")
        count = pairs.groupby("anchor_index").size()
        if int(count.max()) > int(k):
            raise RuntimeError(f"case {case}: pair count exceeds deployed k={k}")
        owner_count = pairs.groupby("secondary_index")["anchor_index"].nunique()
        if len(owner_count) and int(owner_count.max()) > 1:
            bad = owner_count[owner_count > 1].head().to_dict()
            raise RuntimeError(
                f"case {case}: anchor neighbourhoods overlap despite spacing: {bad}"
            )
        neighbour_ids = pairs["secondary_index"].drop_duplicates().to_numpy(np.int64)

        pair_out = pairs[["anchor_index", "secondary_index", "distance", "response"]].copy()
        pair_out.insert(0, "case", case)
        for probe, mode in enumerate(pair_modes):
            u1, u2 = random_pair_directions(case, neighbour_ids, probe, args.random_seed)
            direction = pd.DataFrame({"index": neighbour_ids, "u1": u1, "u2": u2})
            probe_pairs = pair_out[["secondary_index"]].merge(
                direction.rename(columns={"index": "secondary_index"}),
                on="secondary_index", how="left", validate="many_to_one",
            )
            pair_out[f"u1_{mode}"] = probe_pairs["u1"].to_numpy(float)
            pair_out[f"u2_{mode}"] = probe_pairs["u2"].to_numpy(float)
        pair_out.to_feather(manifest_dir / f"pairs_case{case}.feather")
        reference.loc[anchor_mask, ["index", "r", "Re"]].assign(case=case).to_feather(
            manifest_dir / f"anchors_case{case}.feather"
        )

        for mode, base in bases.items():
            random_by_id = None
            if mode.startswith("pair"):
                probe = int(mode[4:])
                random_by_id = pd.DataFrame({
                    "index": neighbour_ids,
                    "u1": random_pair_directions(case, neighbour_ids, probe,
                                                  args.random_seed)[0],
                    "u2": random_pair_directions(case, neighbour_ids, probe,
                                                  args.random_seed)[1],
                })
            for sign in (+args.g, -args.g):
                path = generated_path(base, case, sign)
                frame = pd.read_feather(path)
                assigned = assign_shear(
                    frame, mode, sign, args.g, anchors, neighbour_ids, random_by_id,
                )
                assigned.to_feather(path)

        summary["per_case"].append({
            "case": case, "n_sources": len(reference), "n_eligible": int(eligible.sum()),
            "n_anchors": len(anchors), "minimum_anchor_spacing_arcsec": minimum,
            "n_pairs": len(pair_out), "n_anchors_with_pairs": int(count.size),
            "mean_pairs_per_anchor": float(len(pair_out) / len(anchors)),
            "max_pairs_per_anchor": int(count.max()) if len(count) else 0,
        })
        print(
            f"case {case}: sources={len(reference):,} eligible={eligible.sum():,} "
            f"anchors={len(anchors):,} pairs={len(pair_out):,} "
            f"<k>={len(pair_out)/len(anchors):.3f} maxk={int(count.max()) if len(count) else 0} "
            f"minsep={minimum:.3f}\"",
            flush=True,
        )

    with open(manifest_dir / "design.json", "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote matched design -> {manifest_dir}")
    print("V22_MATCHED_DECOMP_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
