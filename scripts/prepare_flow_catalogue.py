#!/usr/bin/env python
"""Prepare a compact mixed-shear flow leg from explicit external catalogues."""

from __future__ import annotations

import argparse
from pathlib import Path

from sbsi.flow_catalogue import prepare_flow_catalogue


DEFAULT_COLUMNS = [
    "case", "input_index", "input_index_sec", "shear_case", "detected",
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "redshift_input_p", "redshift_input_s",
    "e1_input_rot0_p", "e2_input_rot0_p", "e1_input_rot0_s", "e2_input_rot0_s",
    "gamma1_input_p", "gamma2_input_p", "gamma1_input_s", "gamma2_input_s",
    "neighbored", "distance", "polarization_angle", "shear_component_convention",
    "measured_ngmix_g1", "measured_ngmix_g2", "measured_mag_auto",
    "measured_flux_radius",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crowd-lookup", type=Path, required=True)
    parser.add_argument("--rblend-lookup", type=Path, required=True)
    parser.add_argument("--max-case", type=int, default=200)
    args = parser.parse_args(argv)
    prepare_flow_catalogue(
        args.input,
        args.output,
        columns=DEFAULT_COLUMNS,
        lookups={
            args.crowd_lookup: ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"],
            args.rblend_lookup: ["r_blend"],
        },
        max_case=args.max_case,
    )


if __name__ == "__main__":
    main()
