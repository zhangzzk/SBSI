#!/usr/bin/env python
"""Audit one flow on held-out primary-only forward half-shear simulations."""

import argparse
from pathlib import Path

from sbsi.flow_response_audit import audit_primary_only_forward_response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--g0-catalogue", type=Path, required=True)
    parser.add_argument("--sheared-catalogue", type=Path, required=True)
    parser.add_argument("--response-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    audit_primary_only_forward_response(
        args.model,
        args.g0_catalogue,
        args.sheared_catalogue,
        args.response_target,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        n_boot=args.n_boot,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
