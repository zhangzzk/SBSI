#!/usr/bin/env python
"""Build Flow-E magnitude/log-size derivative targets on its frozen case split."""

import argparse
from pathlib import Path

from sbsi.flow_coupling_target import build_coupling_target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0-catalogue", type=Path, required=True)
    parser.add_argument("--sheared-catalogue", type=Path, required=True)
    parser.add_argument("--response-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-role", choices=("train", "validation"), default="train")
    parser.add_argument("--min-count", type=int, default=300)
    args = parser.parse_args(argv)
    build_coupling_target(
        args.g0_catalogue,
        args.sheared_catalogue,
        args.response_target,
        args.output,
        case_role=args.case_role,
        min_count=args.min_count,
    )


if __name__ == "__main__":
    main()
