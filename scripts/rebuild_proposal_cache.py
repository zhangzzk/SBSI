#!/usr/bin/env python
"""Rebuild a disk proposal coordinate table at a different dispersion floor.

The assembled proposal table is a *summary* of the prepared shards: the flow
draws are already reduced to a per-atom mean and standard deviation, and the
floor is applied to that summary afterwards.  Changing the floor therefore
needs neither the flow nor a GPU.

It cannot be done by re-running ``run_disk_inference.py assemble``, which
refuses every shard receipt whose recorded implementation differs from the
current tree and whose own source is hash-pinned.  This rebuilds *only* the
proposal table, from the same shard arrays in the same order, and writes it to
a new directory; every other assembled artifact is left untouched and keeps
its recorded hash.

Before writing anything the rebuild is checked against the shipped table by
reproducing it at the floor it was built with.  An exact match is what makes
the rebuilt table at any other floor trustworthy, and it is a stronger
statement than re-flooring the shipped table in memory can make: a fractional
floor ranks atoms by ``sigma / |x|``, an order the earlier absolute floor can
permute, so a re-floor reproduces a fresh build only up to the atoms that
floor already bound.  This starts from the unfloored shard arrays instead.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sbsi.catalogue_sampling import (
    ProposalCoordinateTable,
    floored_dispersion,
    fractional_mask,
)
from sbsi.disk_inference_store import coordinate_table, file_hash, write_json

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/likelihood_v3_6_like.json"


def load_shards(prepared, assembled):
    """Concatenate the shard summaries in the order ``assemble`` used."""
    manifest = json.loads((assembled / "manifest.json").read_text())
    paths = sorted(prepared.glob("shard_*/manifest.json"))
    receipts = manifest["receipts"]
    if len(paths) != len(receipts):
        raise ValueError(f"expected {len(receipts)} prepared shards, found {len(paths)}")
    values, dispersion = [], []
    for index, (path, expected) in enumerate(zip(paths, receipts)):
        receipt = json.loads(path.read_text())
        if receipt != expected or receipt["source_shard"] != index:
            raise ValueError(f"prepared shard receipt differs from the assembled record: {path}")
        for name in ("values.npy", "dispersion.npy"):
            if file_hash(path.parent / name) != receipt["output_sha256"][name]:
                raise ValueError(f"prepared artifact hash differs: {path.parent / name}")
        values.append(np.load(path.parent / "values.npy"))
        dispersion.append(np.load(path.parent / "dispersion.npy"))
        print(f"SHARD {index} rows={len(values[-1])}", flush=True)
    values, dispersion = np.concatenate(values), np.concatenate(dispersion)
    if len(values) != manifest["n_rows"]:
        raise ValueError(f"rebuilt {len(values)} atoms, assembled manifest records {manifest['n_rows']}")
    return values, dispersion, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, required=True,
        help="directory holding the per-shard preparation output")
    parser.add_argument("--assembled", type=Path, required=True,
        help="assembled directory whose proposal table is being rebuilt")
    parser.add_argument("--output", type=Path, required=True,
        help="new proposal directory to write; must not exist")
    parser.add_argument("--dispersion-floor-percentile", type=float, default=50.0)
    parser.add_argument("--fractional-floor", nargs="*",
        default=("measured_flux_from_mag_auto",))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing proposal table")

    values, dispersion, manifest = load_shards(args.prepared, args.assembled)
    names = json.loads(CONFIG.read_text())["measurement_model"]["target_names"]

    # `coordinate_table` is the function `assemble` itself called, so this is
    # the shipped table recomputed rather than a second implementation of it.
    reference = coordinate_table(values, dispersion, names)
    shipped = ProposalCoordinateTable.load(args.assembled / "proposal")
    for field in ("values", "center", "scale", "dispersion"):
        rebuilt, original = getattr(reference, field), getattr(shipped, field)
        if rebuilt.shape != original.shape or not np.array_equal(rebuilt, original):
            raise ValueError(f"rebuild does not reproduce the shipped table: {field}")
    if tuple(shipped.target_names) != tuple(names):
        raise ValueError("shipped table target names differ from the likelihood configuration")
    print(f"VERIFIED rebuild reproduces {args.assembled / 'proposal'} exactly", flush=True)

    fractional = fractional_mask(names, args.fractional_floor)
    table = ProposalCoordinateTable(
        values, tuple(names), reference.center, reference.scale,
        dispersion=floored_dispersion(values, dispersion,
            percentile=args.dispersion_floor_percentile, fractional=fractional,
            fallback=np.maximum(1.0e-3 * reference.scale, np.finfo(np.float64).eps)),
        statistic=reference.statistic, dispersion_statistic=reference.dispersion_statistic,
        n_flow_samples=reference.n_flow_samples,
        metadata=dict(
            rebuilt_from="prepared shard summaries",
            source_assembled=str(args.assembled.resolve()),
            source_proposal_sha256=manifest["output_sha256"]["proposal/coordinates.npz"],
            source_identity=manifest["identity"],
            dispersion_floor_percentile=float(args.dispersion_floor_percentile),
            fractional_floor_targets=[str(name) for name in args.fractional_floor],
            reproduces_source_at_build_floor=True,
        ))
    table.save(args.output)
    write_json(args.output.parent / f"{args.output.name}_receipt.json", dict(
        status="complete", n_atoms=int(len(values)),
        dispersion_floor_percentile=float(args.dispersion_floor_percentile),
        fractional_floor_targets=[str(name) for name in args.fractional_floor],
        source_assembled=str(args.assembled.resolve()),
        source_proposal_sha256=manifest["output_sha256"]["proposal/coordinates.npz"],
        output_sha256={name: file_hash(args.output / name)
                       for name in ("coordinates.npz", "manifest.json")}))

    ratio = table.dispersion / shipped.dispersion
    print(f"REBUILT atoms={len(values)} output={args.output}", flush=True)
    for i, name in enumerate(names):
        bound = float(np.mean(ratio[:, i] > 1.0))
        print(f"  {name}: floor binds on {100 * bound:.1f}% of atoms, "
              f"median scatter x{float(np.median(ratio[:, i])):.3f}", flush=True)


if __name__ == "__main__":
    main()
