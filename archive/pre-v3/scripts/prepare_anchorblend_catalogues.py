"""Turn fresh constant-shear inputs into neighbour-only shear inputs.

The two generated legs have identical galaxies and positions and opposite
coherent shear.  A sparse, deterministic set of V2.1 galaxies is selected as
anchors and assigned zero shear in both legs.  With anchor separation greater
than twice the response aperture, all sources near any anchor remain sheared.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from sbs_shear.anchorblend import assert_anchor_spacing, sparse_anchor_mask
from sbs_shear.population import EXTENDED_PAIR_CUTS, primary_mask


def label(g):
    return str(float(g))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.02)
    ap.add_argument("--min-separation", type=float, default=20.0)
    ap.add_argument("--audit-only", action="store_true",
                    help="verify an already patched set without writing")
    args = ap.parse_args()

    for case in args.cases:
        manifest = os.path.join(args.base, f"anchors_case{case}.feather")
        if not args.audit_only and os.path.exists(manifest):
            raise SystemExit(f"REFUSING to overwrite existing manifest {manifest}")
        if not args.audit_only:
            for sign in (args.g, -args.g):
                rendered = os.path.join(args.base, f"case{case}_{label(sign)}")
                if os.path.exists(rendered):
                    raise SystemExit(f"REFUSING: render tree already exists: {rendered}")

        plus_path = os.path.join(args.base, f"gals{case}_{label(args.g)}.feather")
        minus_path = os.path.join(args.base, f"gals{case}_{label(-args.g)}.feather")
        plus = pd.read_feather(plus_path)
        minus = pd.read_feather(minus_path)
        identity = ["index", "cata_idx", "RA", "DEC", "r", "Re"]
        if list(plus.columns) != list(minus.columns):
            raise RuntimeError(f"case {case}: +/- schemas differ")
        # The only permitted leg difference before anchor patching is applied shear.
        # This includes morphology, redshift, position angle, and catalogue identity,
        # not merely the columns later used for selection and matching.
        for col in plus.columns.difference(["g1", "g2"], sort=False):
            if not plus[col].equals(minus[col]):
                raise RuntimeError(f"case {case}: +/- latent column {col} differs")
        smag = plus["r"].to_numpy(float)
        sre = plus["Re"].to_numpy(float)
        secondary_support = (
            np.isfinite(smag) & np.isfinite(sre)
            & (smag > EXTENDED_PAIR_CUTS.secondary_mag[0])
            & (smag < EXTENDED_PAIR_CUTS.secondary_mag[1])
            & (sre > EXTENDED_PAIR_CUTS.secondary_re[0])
            & (sre < EXTENDED_PAIR_CUTS.secondary_re[1])
        )
        unsupported_count = int((~secondary_support).sum())
        if unsupported_count:
            print(
                f"case {case}: NOTE {unsupported_count} rendered source(s) outside deployed "
                "secondary support; response analysis excludes anchors within 10 arcsec",
                flush=True,
            )

        if args.audit_only:
            saved = pd.read_feather(manifest)
            if saved["index"].duplicated().any():
                raise RuntimeError(f"case {case}: duplicate saved anchor ids")
            anchors = plus["index"].isin(saved["index"]).to_numpy()
            if anchors.sum() != len(saved):
                raise RuntimeError(f"case {case}: saved anchor ids do not match input")
            minimum = assert_anchor_spacing(
                plus["RA"], plus["DEC"], anchors,
                min_separation_arcsec=args.min_separation,
            )
            non = ~anchors
            for frame, sign in ((plus, +1.0), (minus, -1.0)):
                if np.max(np.abs(frame.loc[non, "g1"].to_numpy(float)
                                 - sign * args.g)) > 1e-12:
                    raise RuntimeError(f"case {case}: non-anchor g1 is not {sign*args.g:+g}")
                if np.any(frame.loc[anchors, ["g1", "g2"]].to_numpy(float)):
                    raise RuntimeError(f"case {case}: an anchor retained nonzero shear")
            print(
                f"case {case}: audited anchors={anchors.sum():,}, "
                f"minimum separation={minimum:.3f} arcsec",
                flush=True,
            )
            continue

        for frame, sign in ((plus, +1.0), (minus, -1.0)):
            if np.max(np.abs(frame["g1"].to_numpy(float) - sign * args.g)) > 1e-12:
                raise RuntimeError(f"case {case}: generated g1 is not constant {sign*args.g:+g}")
            if np.max(np.abs(frame["g2"].to_numpy(float))) > 1e-12:
                raise RuntimeError(f"case {case}: generated g2 is not zero")

        # Apply the same intrinsic primary population as the flow, R_blend, and
        # constgold comparison: ordinary 18<r<28, Re<1.5 support intersected
        # with V2.1's Re>0.5 and intrinsic S/N>10 curved domain.
        eligible = primary_mask(plus.rename(columns={"r": "r_input_p", "Re": "Re_input_p"}))
        anchors = sparse_anchor_mask(
            plus["RA"], plus["DEC"], eligible,
            min_separation_arcsec=args.min_separation,
        )
        if anchors.sum() < 10:
            raise RuntimeError(f"case {case}: only {anchors.sum()} anchors selected")
        minimum = assert_anchor_spacing(
            plus["RA"], plus["DEC"], anchors,
            min_separation_arcsec=args.min_separation,
        )
        for frame in (plus, minus):
            frame.loc[anchors, ["g1", "g2"]] = 0.0

        # Verify the exact neighbour-only shear construction before modifying data.
        non = ~anchors
        if np.max(np.abs(plus.loc[non, "g1"].to_numpy(float)
                         + minus.loc[non, "g1"].to_numpy(float))) > 1e-12:
            raise RuntimeError("non-anchor +/- shears are not antithetic")
        if np.any(plus.loc[anchors, ["g1", "g2"]].to_numpy(float)) \
                or np.any(minus.loc[anchors, ["g1", "g2"]].to_numpy(float)):
            raise RuntimeError("an anchor retained nonzero shear")

        plus.to_feather(plus_path)
        minus.to_feather(minus_path)
        plus.loc[anchors, identity].assign(case=case).to_feather(manifest)
        print(
            f"case {case}: eligible={eligible.sum():,}, anchors={anchors.sum():,}, "
            f"minimum separation={minimum:.3f} arcsec",
            flush=True,
        )
    print("ANCHORBLEND_AUDIT_DONE" if args.audit_only else "ANCHORBLEND_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
