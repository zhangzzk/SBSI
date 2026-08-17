#!/usr/bin/env python3
"""Prepare a supplemental coherent-anchor arm for V2 minus the old anchor domain.

The source cases are regenerated with their original deterministic seeds.  This
script verifies every latent column against the existing coherent-anchor cases,
resets the old anchor shear, and then keeps only sparse anchors in the missing
part of the full V2 primary box.  The existing anchor measurements therefore
remain an untouched stratum rather than being replaced.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sbs_shear.anchorblend import assert_anchor_spacing, sparse_anchor_mask
from sbs_shear.population import EXTENDED_PAIR_CUTS, primary_mask


def label(value: float) -> str:
    return str(float(value))


def v2_primary(frame: pd.DataFrame) -> np.ndarray:
    mag = frame["r"].to_numpy(float)
    size = frame["Re"].to_numpy(float)
    return (
        np.isfinite(mag) & np.isfinite(size)
        & (mag > 18.0) & (mag < 26.0)
        & (size > 0.3) & (size < 1.5)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--reference-base", required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--g", type=float, default=0.02)
    parser.add_argument("--min-separation", type=float, default=20.0)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    reference = Path(args.reference_base).resolve()
    audit_rows = []
    for case in args.cases:
        manifest_path = base / f"anchors_case{case}.feather"
        audit_path = base / f"anchor_population_case{case}.json"
        if manifest_path.exists() or audit_path.exists():
            raise FileExistsError(f"case {case}: supplemental outputs already exist")
        original_manifest = pd.read_feather(
            reference / f"anchors_case{case}.feather"
        )
        legs = []
        for sign in (args.g, -args.g):
            new_path = base / f"gals{case}_{label(sign)}.feather"
            old_path = reference / f"gals{case}_{label(sign)}.feather"
            new = pd.read_feather(new_path)
            old = pd.read_feather(old_path)
            if list(new.columns) != list(old.columns):
                raise RuntimeError(f"case {case}: regenerated/source schemas differ")
            for column in new.columns.difference(["g1", "g2"], sort=False):
                if not new[column].equals(old[column]):
                    raise RuntimeError(
                        f"case {case}: regenerated latent column {column} differs"
                    )
            new.loc[:, "g1"] = float(sign)
            new.loc[:, "g2"] = 0.0
            legs.append((new_path, new))

        plus = legs[0][1]
        population_frame = plus.rename(
            columns={"r": "r_input_p", "Re": "Re_input_p"}
        )
        old_domain = primary_mask(population_frame)
        full_v2 = v2_primary(plus)
        if np.any(old_domain & ~full_v2):
            raise RuntimeError(f"case {case}: old anchor domain is not nested in V2")
        complement = full_v2 & ~old_domain
        anchors = sparse_anchor_mask(
            plus["RA"], plus["DEC"], complement,
            min_separation_arcsec=args.min_separation,
        )
        if int(anchors.sum()) < 100:
            raise RuntimeError(f"case {case}: only {int(anchors.sum())} complement anchors")
        minimum = assert_anchor_spacing(
            plus["RA"], plus["DEC"], anchors,
            min_separation_arcsec=args.min_separation,
        )

        original_ids = set(original_manifest["index"].astype(np.int64))
        original_row = plus["index"].isin(original_ids).to_numpy()
        expected_old_ids = set(plus.loc[original_row, "index"].astype(np.int64))
        if expected_old_ids != original_ids:
            raise RuntimeError(f"case {case}: original anchor identities are not retained")
        if not np.all(old_domain[original_row]):
            raise RuntimeError(f"case {case}: an original anchor lies outside its domain")
        if np.any(anchors & original_row):
            raise RuntimeError(f"case {case}: complement overlaps original anchors")

        for path, frame in legs:
            frame.loc[anchors, ["g1", "g2"]] = 0.0
            frame.to_feather(path)
        identity = ["index", "cata_idx", "RA", "DEC", "r", "Re"]
        n_v2 = int(full_v2.sum())
        n_old = int(old_domain.sum())
        n_complement = int(complement.sum())
        n_anchors = int(anchors.sum())
        manifest = plus.loc[anchors, identity].assign(
            case=case,
            stratum="V2_minus_original_anchor_domain",
            n_parent_v2=n_v2,
            n_parent_original_domain=n_old,
            n_parent_complement=n_complement,
            inverse_sampling_weight=float(n_complement / n_anchors),
        )
        manifest.to_feather(manifest_path)
        payload = {
            "case": case,
            "n_parent_v2": n_v2,
            "n_parent_original_domain": n_old,
            "n_parent_complement": n_complement,
            "n_original_anchors_retained_in_existing_arm": len(original_manifest),
            "n_supplemental_anchors": n_anchors,
            "supplemental_inverse_sampling_weight": float(n_complement / n_anchors),
            "minimum_supplemental_anchor_separation_arcsec": minimum,
            "domains_partition_v2": n_old + n_complement == n_v2,
        }
        with audit_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        audit_rows.append(payload)
        print(
            f"case {case}: V2={n_v2:,} old={n_old:,} complement={n_complement:,} "
            f"supplemental_anchors={n_anchors:,} minsep={minimum:.3f}",
            flush=True,
        )
    print(
        "V2_COMPLEMENT_ANCHOR_PREP_DONE "
        f"cases={len(audit_rows)} supplemental={sum(x['n_supplemental_anchors'] for x in audit_rows):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
