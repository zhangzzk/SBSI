#!/usr/bin/env python
"""Combine disjoint atom shards into one exact normalization stencil cache."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from sbsi.selection_normalization import ExactPopulationNormalization


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shard_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "selection_normalization_shard.json" if path.is_dir() else path


def combine(shard_paths: list[str | Path]) -> ExactPopulationNormalization:
    if not shard_paths:
        raise ValueError("at least one normalization shard is required")
    records = []
    for name in shard_paths:
        path = _shard_path(name)
        payload = json.loads(path.read_text())
        if payload.get("version") != 1 or payload.get("method") != (
            "exact_detected_selected_mass_atom_shard"
        ):
            raise ValueError(f"not a supported normalization shard: {path}")
        records.append((int(payload["partition"]["index"]), path, payload))
    records.sort()
    reference = records[0][2]
    count = int(reference["partition"]["count"])
    if len(records) != count or [row[0] for row in records] != list(range(count)):
        raise ValueError("normalization shards do not contain every unique shard index")

    expected_start = 0
    point_keys = [
        (row["name"], round(float(row["g1"]), 14), round(float(row["g2"]), 14))
        for row in reference["points"]
    ]
    contributions = {key: [] for key in point_keys}
    sources = []
    for index, path, payload in records:
        partition = payload["partition"]
        if (
            payload["identity"] != reference["identity"]
            or payload["center"] != reference["center"]
            or float(payload["finite_difference_step"])
            != float(reference["finite_difference_step"])
            or int(partition["count"]) != count
            or int(partition["n_active_total"])
            != int(reference["partition"]["n_active_total"])
            or int(partition["n_atoms_total"])
            != int(reference["partition"]["n_atoms_total"])
        ):
            raise ValueError(f"normalization shard identity mismatch: {path}")
        start = int(partition["active_start"])
        stop = int(partition["active_stop"])
        if start != expected_start or stop <= start:
            raise ValueError("normalization atom shards are not contiguous and non-empty")
        expected_start = stop
        actual_keys = [
            (row["name"], round(float(row["g1"]), 14), round(float(row["g2"]), 14))
            for row in payload["points"]
        ]
        if actual_keys != point_keys:
            raise ValueError(f"normalization stencil mismatch: {path}")
        for key, row in zip(point_keys, payload["points"]):
            value = float(row["partial_detected_and_selected_mass"])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid partial normalization mass in {path}")
            contributions[key].append(value)
        sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    if expected_start != int(reference["partition"]["n_active_total"]):
        raise ValueError("normalization atom shards do not cover the active catalogue")

    points = {
        (g1, g2): math.fsum(contributions[(name, g1, g2)])
        for name, g1, g2 in point_keys
    }
    return ExactPopulationNormalization(
        points=points,
        finite_difference_step=float(reference["finite_difference_step"]),
        identity=dict(reference["identity"]),
        source={
            "method": "sum_of_disjoint_atom_shards",
            "n_shards": count,
            "n_active_atoms": int(reference["partition"]["n_active_total"]),
            "n_atoms": int(reference["partition"]["n_atoms_total"]),
            "shards": sources,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    normalization = combine(args.shard)
    normalization.save(args.output)
    print(Path(args.output).read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
