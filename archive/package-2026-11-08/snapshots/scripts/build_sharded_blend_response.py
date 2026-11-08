#!/usr/bin/env python
"""Build the atom-aligned BlendEMU response cache for a *sharded* scene prior.

`build_catalogue_blend_response.py` evaluates one scene store.  It cannot be
used on the global inference scene: `merge_sharded_prior_model_qmc.py` writes
that scene with an empty neighbour graph on purpose (the flow and detection
views were precomputed per shard, where neighbours were still available, so the
graph itself was dropped).  Running the single-scene builder there yields zero
response pairs and an all-zero cache that would still report `r_blend.enabled`.

This builder goes back to the shards, where the graph survives, computes the
response for each shard's active atoms, and concatenates them in the *same
order the merge uses* -- shard order, `np.flatnonzero(weights > 0)` within a
shard -- so the result is aligned to the merged scene row for row.

Partial results are written per shard, so a run can be resumed or split across
several jobs by shard range and merged afterwards.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from sbsi.catalogue_blend import CatalogueBlendResponse, CATALOGUE_BLEND_CACHE_VERSION
from sbsi.forward_catalogue import EmulatorPairingConfig
from sbsi.models import ModelPaths, load_emulator
from sbsi.scene_prior import ScenePrior


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-manifest", required=True)
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument("--emulator-model", required=True)
    parser.add_argument("--emulator-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixel-size", type=float, required=True)
    parser.add_argument("--zero-point", type=float, required=True)
    parser.add_argument("--psf-fwhm", type=float, required=True)
    parser.add_argument("--moffat-beta", type=float, required=True)
    parser.add_argument("--pixel-rms", type=float, required=True)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument(
        "--shard-stop",
        type=int,
        default=None,
        help="exclusive; default is every shard in the manifest",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="assemble the finished per-shard parts into the aligned cache",
    )
    parser.add_argument(
        "--inference-scene-store",
        default=None,
        help=(
            "merged inference scene store this cache is for; required by --merge. "
            "run_inference.py checks a blend cache against the scene it is used "
            "with, so the merge must state which scene that is -- and it proves "
            "the claim before stamping it (same default_prior_manifest_sha256, "
            "same n_positive_prior_atoms) rather than asserting it"
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def _part_path(root: Path, index: int) -> Path:
    return root / "parts" / f"shard_{index:02d}.npz"


def _build_shard(args, shard, index, conditions, root):
    """Evaluate one shard and save its active-atom response slice."""

    scene_root = Path(shard["root"]) / "scene_store"
    prior = ScenePrior.load(scene_root)
    active = np.flatnonzero(prior.weights > 0)
    expected = int(shard["n_positive_prior_atoms"])
    if len(active) != expected:
        raise RuntimeError(
            f"shard {index} active count changed: {len(active)} != {expected}"
        )

    emulator_model = Path(args.emulator_model)
    emulator_metadata = Path(args.emulator_metadata)
    paths = ModelPaths(
        flow_checkpoints=(Path(args.measurement_model),),
        emulator_model=emulator_model,
        emulator_metadata=emulator_metadata,
    )
    emulator = load_emulator(paths, conditions=conditions, device=args.device)
    pairing = EmulatorPairingConfig.from_emulator(emulator, task="regression")
    response = CatalogueBlendResponse.from_emulator(
        prior,
        emulator,
        config=pairing,
        active_indices=active,
        metadata={
            "shard_index": index,
            "shard_cases": shard["cases"],
            "shard_root": str(shard["root"]),
            "scene_sha256": {
                name: _sha256(scene_root / name)
                for name in ("manifest.json", "galaxies.parquet", "neighbours.npz")
            },
            "emulator_sha256": {
                "model": _sha256(emulator_model),
                "metadata": _sha256(emulator_metadata),
            },
            "conditions": conditions,
            "pairing_config": asdict(pairing),
        },
    )
    # Only the active rows survive the merge, so only they are kept.  Storing
    # the full shard vector would invite an accidental misalignment later.
    part = _part_path(root, index)
    part.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        part,
        values=response.values[active],
        active=active,
        n_scene_rows=np.int64(len(response.values)),
    )
    (part.with_suffix(".json")).write_text(
        json.dumps({"metadata": response.metadata, "report": response.report}, indent=2)
        + "\n"
    )
    return response.report


def _bind_to_inference_scene(args, manifest, n_rows):
    """Verify the merged cache really is row-aligned to the inference scene.

    ``run_inference.py`` refuses a blend cache whose recorded ``scene_sha256``
    does not match the scene store it is used with, which is the right check:
    an atom-aligned cache is meaningless against a different atom ordering.
    The cache is built from the per-shard scene stores, because the merged
    scene deliberately carries an empty neighbour graph, so the merge is the
    only place that can record which merged scene the result belongs to.

    It is recorded only after the binding is demonstrated.  Both sides are
    derived from one prior manifest, and ``merge_sharded_prior_model_qmc.py``
    concatenates active rows shard by shard in manifest order -- exactly what
    ``_merge`` does.  So agreement on the manifest hash and on the active-atom
    count is what makes the two orderings the same ordering.
    """

    scene_root = Path(args.inference_scene_store)
    scene_manifest = json.loads((scene_root / "manifest.json").read_text())
    scene_metadata = scene_manifest.get("metadata") or {}
    prior_manifest_sha256 = _sha256(args.prior_manifest)

    recorded = scene_metadata.get("default_prior_manifest_sha256")
    if recorded != prior_manifest_sha256:
        raise RuntimeError(
            "the inference scene store was built from a different prior manifest: "
            f"scene records {recorded}, this build used {prior_manifest_sha256}"
        )
    scene_atoms = scene_metadata.get("n_positive_prior_atoms")
    if scene_atoms != n_rows:
        raise RuntimeError(
            f"the inference scene store holds {scene_atoms} active atoms but the "
            f"merged response has {n_rows} rows"
        )
    if int(manifest["n_positive_prior_atoms"]) != n_rows:
        raise RuntimeError(
            f"prior manifest says {manifest['n_positive_prior_atoms']} atoms, "
            f"merged response has {n_rows} rows"
        )

    scene_hashes = {
        name: _sha256(scene_root / name)
        for name in ("manifest.json", "galaxies.parquet", "neighbours.npz")
    }
    return scene_root, scene_hashes, prior_manifest_sha256


def _merge(args, manifest, root):
    if args.inference_scene_store is None:
        raise SystemExit("--merge requires --inference-scene-store")
    shards = manifest["shards"]
    values, reports, pairings = [], [], []
    for index, shard in enumerate(shards):
        part = _part_path(root, index)
        if not part.exists():
            raise SystemExit(f"missing shard part {part}; build it before merging")
        loaded = np.load(part)
        got = np.asarray(loaded["values"], dtype=np.float64)
        expected = int(shard["n_positive_prior_atoms"])
        if got.shape != (expected,):
            raise RuntimeError(
                f"shard {index} part has {got.shape} rows, manifest says {expected}"
            )
        values.append(got)
        payload = json.loads(part.with_suffix(".json").read_text())
        reports.append(payload["report"])
        # run_inference.py checks the cache's pairing configuration against the
        # emulator it loads, so the merge has to carry one forward -- and it is
        # only meaningful if every shard used the same one.
        pairings.append(payload["metadata"]["pairing_config"])
    merged = np.concatenate(values)
    expected_total = int(manifest["n_positive_prior_atoms"])
    if merged.shape != (expected_total,):
        raise RuntimeError(
            f"merged response has {merged.shape} rows, manifest says {expected_total}"
        )

    scene_root, scene_hashes, prior_manifest_sha256 = _bind_to_inference_scene(
        args, manifest, len(merged)
    )
    conditions = {
        "pixel_size": args.pixel_size,
        "zero_point": args.zero_point,
        "psf_fwhm": args.psf_fwhm,
        "moffat_beta": args.moffat_beta,
        "pixel_rms": args.pixel_rms,
    }

    pairing_config = pairings[0]
    for index, other in enumerate(pairings[1:], start=1):
        if other != pairing_config:
            raise RuntimeError(
                f"shard {index} was built with a different emulator pairing "
                "configuration than shard 0"
            )

    n_pairs = int(sum(r["n_response_pairs"] for r in reports))
    report = {
        "n_scene_rows": expected_total,
        "n_active_atoms": expected_total,
        "n_response_pairs": n_pairs,
        "n_nonzero_atoms": int(np.count_nonzero(merged)),
        "active_mean": float(merged.mean()),
        "active_min": float(merged.min()),
        "active_max": float(merged.max()),
        "shard_reports": reports,
    }
    np.save(root / "r_blend.npy", merged)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": CATALOGUE_BLEND_CACHE_VERSION,
                "values": "r_blend.npy",
                "n_rows": expected_total,
                "metadata": {
                    "purpose": "sharded_blend_response_aligned_to_merged_inference_scene",
                    "prior_manifest": str(args.prior_manifest),
                    "prior_manifest_sha256": prior_manifest_sha256,
                    "alignment": (
                        "concatenated over shards in manifest order, active rows "
                        "selected as flatnonzero(weights > 0) within each shard, "
                        "matching merge_sharded_prior_model_qmc.py"
                    ),
                    "inference_scene_store": str(scene_root),
                    "scene_sha256": scene_hashes,
                    "emulator_sha256": {
                        "model": _sha256(args.emulator_model),
                        "metadata": _sha256(args.emulator_metadata),
                    },
                    "conditions": conditions,
                    "pairing_config": pairing_config,
                },
                "report": report,
            },
            indent=2,
        )
        + "\n"
    )
    return report


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.output)
    manifest = json.loads(Path(args.prior_manifest).read_text())
    conditions = {
        "pixel_size": args.pixel_size,
        "zero_point": args.zero_point,
        "psf_fwhm": args.psf_fwhm,
        "moffat_beta": args.moffat_beta,
        "pixel_rms": args.pixel_rms,
    }

    if args.merge:
        report = _merge(args, manifest, root)
        print(json.dumps({"output": str(root), **{k: v for k, v in report.items() if k != "shard_reports"}}, indent=2))
        return

    shards = manifest["shards"]
    stop = len(shards) if args.shard_stop is None else int(args.shard_stop)
    for index in range(int(args.shard_start), min(stop, len(shards))):
        if _part_path(root, index).exists():
            print(json.dumps({"shard_index": index, "status": "already built"}))
            continue
        report = _build_shard(args, shards[index], index, conditions, root)
        print(json.dumps({"shard_index": index, **report}), flush=True)


if __name__ == "__main__":
    main()
