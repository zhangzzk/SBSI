#!/usr/bin/env python
"""Plot selected complete-likelihood predictions against measured properties."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_mock_constgold_properties import (
    _comparison_summary,
    _distribution_summary,
    _draw_corner,
    _draw_radius_tail,
)


TARGETS = (
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_log_flux_radius",
)


def _load(
    path: Path, pixel_size: float, *, abs_shape_max: float | None
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(set(TARGETS) - set(frame))
    if missing:
        raise KeyError(f"{path} lacks measurement columns: {missing}")
    result = pd.DataFrame(
        {
            "g1": frame[TARGETS[0]].to_numpy(float),
            "g2": frame[TARGETS[1]].to_numpy(float),
            "mag": frame[TARGETS[2]].to_numpy(float),
            "radius": pixel_size * np.exp(frame[TARGETS[3]].to_numpy(float)),
        }
    )
    values = result.to_numpy(float)
    selected = (
        np.isfinite(values).all(axis=1)
        & (values[:, 2] < 25.8)
        & (values[:, 3] >= 0.75)
    )
    if abs_shape_max is not None:
        selected &= np.hypot(values[:, 0], values[:, 1]) < abs_shape_max
    if not selected.all():
        raise ValueError(f"{path} contains {int((~selected).sum())} rows outside cuts")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--disable-shape-cut", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    abs_shape_max = None if args.disable_shape_cut else 0.6
    predicted = _load(
        args.predicted, args.pixel_size, abs_shape_max=abs_shape_max
    )
    measured = _load(args.measured, args.pixel_size, abs_shape_max=abs_shape_max)
    if len(predicted) != len(measured):
        raise ValueError("predicted and measured contour samples must have equal size")
    manifest = json.loads(args.sample_manifest.read_text())
    if len(predicted) != int(manifest["sample_size_each"]):
        raise ValueError("sample size does not match manifest")
    args.output.mkdir(parents=True)
    cut_label = (
        r"$m<25.8$, $r_\mathrm{flux}\geq0.75''$; no $|e|$ cut"
        if args.disable_shape_cut
        else r"$|e|<0.6$, $m<25.8$, $r_\mathrm{flux}\geq0.75''$"
    )
    title = (
        r"Complete-likelihood prediction versus ConstGold at $g=(0.02,0)$"
        f"\n20 cases; 100,000 selected measurements each; {cut_label}"
    )
    limits = None
    if args.disable_shape_cut:
        pooled = pd.concat([predicted, measured], ignore_index=True)
        lower, upper = np.quantile(
            pooled[["g1", "g2"]].to_numpy(float), [0.001, 0.999], axis=0
        )
        span = float(max(np.max(np.abs(lower)), np.max(np.abs(upper))))
        radius_hi = float(np.quantile(pooled["radius"], 0.997))
        mag_lo = float(np.quantile(pooled["mag"], 0.001))
        limits = [(-span, span), (-span, span), (mag_lo, 25.8), (0.75, radius_hi)]
    _draw_corner(predicted, measured, args.output, title=title, limits=limits)
    _draw_radius_tail(predicted, measured, args.output, title=title)
    summary = {
        "comparison": (
            "complete forward prediction versus ConstGold measured distribution"
        ),
        "cases": [40, 60],
        "sample_size_each": int(len(predicted)),
        "shear": [0.02, 0.0],
        "selection": (
            "MAG_AUTO<25.8; measured radius>=0.75 arcsec; no |e| cut"
            if args.disable_shape_cut
            else "|e|<0.6; MAG_AUTO<25.8; measured radius>=0.75 arcsec"
        ),
        "predicted": _distribution_summary(predicted),
        "measured": _distribution_summary(measured),
        "differences": _comparison_summary(predicted, measured),
        "sample_manifest": manifest,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
