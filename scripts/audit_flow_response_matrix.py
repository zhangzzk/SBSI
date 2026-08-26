#!/usr/bin/env python
"""Audit a trained flow's full matrix on held-out mixed-shear cases."""

import argparse
from pathlib import Path

from sbsi.flow_response_audit import audit_flow_response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--g0-catalogue", type=Path, required=True)
    parser.add_argument("--sheared-catalogue", type=Path, required=True)
    parser.add_argument("--response-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--delta", type=float, default=0.02)
    args = parser.parse_args(argv)
    audit_flow_response(
        args.model, args.g0_catalogue, args.sheared_catalogue,
        args.response_target, args.output, device=args.device, delta=args.delta,
    )


if __name__ == "__main__":
    main()
