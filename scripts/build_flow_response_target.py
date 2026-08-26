#!/usr/bin/env python
"""Build trace and full-2x2 response targets from mixed-shear flow legs."""

from __future__ import annotations

import argparse
from pathlib import Path

from sbsi.flow_response_target import build_response_target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0-catalogue", type=Path, required=True)
    parser.add_argument("--sheared-catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--split-seed", type=int, default=501)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--n-flux", type=int, default=6)
    parser.add_argument("--n-size", type=int, default=6)
    parser.add_argument("--n-crowd", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=500)
    parser.add_argument("--primary-mag-max", type=float, default=25.8)
    parser.add_argument("--primary-re-min", type=float, default=0.5)
    args = parser.parse_args(argv)
    build_response_target(
        args.g0_catalogue,
        args.sheared_catalogue,
        args.output,
        all_cases=range(args.cases),
        split_seed=args.split_seed,
        validation_size=args.validation_size,
        n_flux=args.n_flux,
        n_size=args.n_size,
        n_crowd=args.n_crowd,
        min_count=args.min_count,
        primary_mag_max=args.primary_mag_max,
        primary_re_min=args.primary_re_min,
    )


if __name__ == "__main__":
    main()
