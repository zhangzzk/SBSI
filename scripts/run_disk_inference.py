#!/usr/bin/env python
"""V3.6 preparation and inference using the shared v1.2 tilted estimator.

The legacy driver intentionally continues to reject the disk likelihood.
This entry point accepts only the pinned V3.6 artifacts and uncut subset store.
"""
import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_disk_likelihood import DiskCatalogueLikelihood
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.catalogue_likelihood import CatalogueSelection, OutputCut
from sbsi.catalogue_null import run_adaptive_section5
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable
from sbsi.crowding import FLOW_FEATURES
from sbsi.disk_inference_store import (FrozenDiskCache, ShardedDiskResponse, coordinate_table,
    file_hash, load_source_shard, load_subset_manifest, stencil, write_json)
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V36_LIKE, load_detection_classifier, load_disk_response
from sbsi.output_conditioned_response import TRUTH_FEATURES
from sbsi.selection_normalization import ExactPopulationNormalization, detected_selected_mass_shard


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/likelihood_v3_6_like.json"
NUMERICS = REPO / "configs/inference_v1_2_16k.json"


def implementation():
    paths = sorted((REPO / "sbsi").glob("*.py")) + [Path(__file__)]
    return {str(p.relative_to(REPO)): file_hash(p) for p in paths}


# Files neither preparation nor assembly executes, so a change confined to
# them cannot alter a cached artifact.  `catalogue_null` holds the estimator,
# which runs long after the cache is written and is not imported by
# `disk_inference_store` at all; `assemble` reaches `catalogue_sampling` only
# for `ProposalCoordinateTable`'s constructor and `save`.  Every cached
# artifact is separately verified by hash in `run`, so this relaxes a proxy
# for that check and not the check itself.  Their content is deliberately not
# pinned: the inference releases change these two files by design, and pinning
# them would tie a cache to the one release that happened to build it.
RUN_STAGE_IMPLEMENTATION = frozenset({"sbsi/catalogue_null.py", "sbsi/catalogue_sampling.py"})


def check_prepared_identity(saved, current):
    """Accept the audited inference-only LRU fix, and run-stage-only changes.

    Preserve the producer's full identity; do not rewrite old cache receipts.
    Every scientific input and every preparation implementation file must
    match.  The four preparation functions must remain byte-identical to the
    original.
    """
    if saved == current:
        return "identical"
    before, after = dict(saved), dict(current)
    old_code, new_code = before.pop("implementation_sha256"), after.pop("implementation_sha256")
    if before != after or set(old_code) != set(new_code):
        raise ValueError("prepared scientific identity differs")
    changed = {name for name in old_code if old_code[name] != new_code[name]}
    driver = "scripts/run_disk_inference.py"
    density = "sbsi/catalogue_disk_likelihood.py"
    run_stage = changed & RUN_STAGE_IMPLEMENTATION
    if (changed - run_stage != {driver, density}
            or old_code[driver] != "9a32b73a9ae05582cc455037657e1b19ffb8edb829da780208b18fa3f8a55cb4"
            or old_code[density] != "bfb4e2ba2c8f4be44f844d1001555265b570183f34974c5294b9abc84087dea2"
            or new_code[density] != "7b6b059d4af4e1a99ca54b60970c55033657290eedf29fd6b193cffddf50784c"):
        raise ValueError("unapproved preparation/runtime implementation difference")
    source = Path(__file__).read_text()
    names = {"identity", "prepare", "assemble", "seed_stream"}
    parts = [ast.get_source_segment(source, node) for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if sha256("\n".join(parts).encode()).hexdigest() != "8a4c398b22e7415d0ed19a4df23b53adc221bb43d7dc3bb0705ec325b8bb8ab1":
        raise ValueError("preparation functions changed; rebuild caches")
    if not run_stage:
        return "audited_two_entry_response_lru_v1"
    return "audited_two_entry_response_lru_v1+run_stage:" + ",".join(sorted(run_stage))


def identity(args, subset):
    config = json.loads(CONFIG.read_text())
    V36_LIKE.validate()
    models = dict(measurement=file_hash(V36_LIKE.flow_checkpoints[0]),
        emulator=file_hash(V36_LIKE.emulator_model), metadata=file_hash(V36_LIKE.emulator_metadata),
        detector=file_hash(V36_LIKE.detection_classifiers[0]))
    if (models["measurement"] != config["measurement_model"]["sha256"]
            or models["emulator"] != config["emulator"]["model"]["sha256"]
            or models["metadata"] != config["emulator"]["metadata"]["sha256"]
            or models["detector"] != config["emulator"]["detection_model"]["sha256"]
            or subset["conditions"] != config["observing_conditions"]):
        raise ValueError("model preset, likelihood config or observing conditions differ")
    geometry = config["geometry"]
    if (subset["flow_features"] != list(FLOW_FEATURES) or subset["pair_features"] != list(TRUTH_FEATURES)
            or subset["pairing"]["k"] != geometry["disk_pair_neighbours"]
            or subset["pairing"]["r_max_arcsec"] != geometry["disk_pair_radius_arcsec"]
            or subset["pairing"]["cuts"] is not None):
        raise ValueError("prepared pair geometry or feature order differs from V3.6")
    measured = pd.read_parquet(args.input / "measurements.parquet")
    names = config["measurement_model"]["target_names"]
    cut = OutputCut.from_specs(names, specs=config["measured_selection"]["bounds"])
    raw = measured[names].to_numpy(float)
    if (not np.isfinite(raw).all() or not cut(raw).all() or np.any(raw[:, 2:] <= 0)
            or np.any(np.square(raw[:, :2]).sum(1) >= 1)):
        raise ValueError("frozen observations violate requested selection/usability")
    return dict(subset_sha256=file_hash(args.subset), models=models,
        input_sha256={name: file_hash(args.input / name) for name in ("measurements.parquet", "truth.parquet", "image_mock_manifest.json")},
        center=raw[:, :2].mean(0).tolist(), h=.001, n_observations=len(raw),
        likelihood_config_sha256=file_hash(CONFIG), inference_config_sha256=file_hash(NUMERICS),
        implementation_sha256=implementation(),
        selection_samples=64, proposal_samples=128,
        seed_convention="SeedSequence([base_seed, source_shard]); reset per shear for CRN",
        selection_seed=8101, coordinate_seed=8201, row_chunk=8192)


def seed_stream(base, shard):
    seed = int(np.random.SeedSequence([base, shard]).generate_state(1)[0])
    torch.manual_seed(seed)
    return seed


def prepare(args):
    started = time.monotonic()
    subset = load_subset_manifest(args.subset)
    frozen = identity(args, subset)
    if args.output.exists():
        raise ValueError("refusing to overwrite preparation")
    flow = load_measurement_model(V36_LIKE.flow_checkpoints[0], device=args.device)
    detector = load_detection_classifier(V36_LIKE, device=args.device)
    response_model = load_disk_response(V36_LIKE, device=args.device, nthread=4)
    print(f"PREP_VERIFY shard={args.shard_index}", flush=True)
    cache, _ = load_source_shard(subset, args.shard_index, detector=detector,
        model=response_model, limit=args.limit_atoms)
    args.output.mkdir(parents=True)
    n = len(cache.zero_flow)
    cache.zero_flow.to_parquet(args.output / "flow_zero.parquet")
    values, dispersion = np.empty((n, 4)), np.empty((n, 4))
    seed_stream(8201, args.shard_index)
    for start in range(0, n, frozen["row_chunk"]):
        stop = min(start+frozen["row_chunk"], n)
        draws = np.asarray(flow.sample(cache.zero_flow.iloc[start:stop], n_samples=128,
            batch_size=frozen["row_chunk"], qmc=True), dtype=float)
        values[start:stop], dispersion[start:stop] = draws.mean(1), draws.std(1)
        if start == 0 or stop == n or stop % (32*frozen["row_chunk"]) == 0:
            print(f"COORDINATES shard={args.shard_index} rows={stop}/{n}", flush=True)
    np.save(args.output / "values.npy", values)
    np.save(args.output / "dispersion.npy", dispersion)
    config = json.loads(CONFIG.read_text())
    cut = OutputCut.from_specs(flow.target_transform.target_names, specs=config["measured_selection"]["bounds"])
    selection = CatalogueSelection(cut, n_samples=64,
        seed=int(np.random.SeedSequence([8101, args.shard_index]).generate_state(1)[0]), row_chunk=frozen["row_chunk"])
    points = stencil(frozen["center"], frozen["h"])
    probabilities, masses = [], []
    for i, point in enumerate(points):
        view = cache.get(*point)
        probabilities.append(view.detection_probability)
        mass = detected_selected_mass_shard(selection, flow, view, cache.prior.weights,
            active_indices=np.arange(n, dtype=np.int64), random_offset_rows=0)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("invalid selected mass")
        masses.append(mass)
        cache.discard_views([point])
        print(f"NORMALIZATION shard={args.shard_index} node={i}/{len(points)} mass={mass:.9g} elapsed={time.monotonic()-started:.1f}s", flush=True)
    np.save(args.output / "probability.npy", np.asarray(probabilities))
    files = ("flow_zero.parquet", "values.npy", "dispersion.npy", "probability.npy")
    write_json(args.output / "manifest.json", dict(status="complete", identity=frozen,
        source_shard=args.shard_index, n_rows=n, points=points, selected_mass=masses,
        output_sha256={name: file_hash(args.output / name) for name in files},
        subset_receipt=subset["shards"][args.shard_index], elapsed_seconds=time.monotonic()-started,
        pilot_only=args.limit_atoms is not None))
    print(f"PREP_COMPLETE {args.output}", flush=True)


def assemble(args):
    subset = load_subset_manifest(args.subset)
    frozen = identity(args, subset)
    expected = 1 if args.pilot else len(subset["shards"])
    paths = sorted(args.prepared.glob("shard_*/manifest.json"))
    if len(paths) != expected or args.output.exists():
        raise ValueError("exact prepared shard coverage and new output required")
    receipts, zeros, values, dispersions, probabilities, masses = [], [], [], [], [], []
    for i, path in enumerate(paths):
        receipt = json.loads(path.read_text())
        if (receipt["status"] != "complete" or receipt["identity"] != frozen
                or receipt["source_shard"] != i or receipt["pilot_only"] != args.pilot
                or receipt["subset_receipt"] != subset["shards"][i]
                or receipt["points"] != [list(p) for p in stencil(frozen["center"], frozen["h"]) ]
                or (not args.pilot and receipt["n_rows"] != subset["shards"][i]["n_rows"])):
            raise ValueError(f"preparation identity/coverage differs: {path}")
        for name, expected_hash in receipt["output_sha256"].items():
            if file_hash(path.parent / name) != expected_hash:
                raise ValueError(f"prepared artifact hash differs: {path.parent/name}")
        receipts.append(receipt)
        zeros.append(pd.read_parquet(path.parent / "flow_zero.parquet"))
        values.append(np.load(path.parent / "values.npy"))
        dispersions.append(np.load(path.parent / "dispersion.npy"))
        probabilities.append(np.load(path.parent / "probability.npy"))
        masses.append(receipt["selected_mass"])
    n = sum(r["n_rows"] for r in receipts)
    names = json.loads(CONFIG.read_text())["measurement_model"]["target_names"]
    coords = coordinate_table(np.concatenate(values), np.concatenate(dispersions), names)
    args.output.mkdir(parents=True)
    coords.save(args.output / "proposal")
    np.save(args.output / "zero.npy", pd.concat(zeros, ignore_index=True).to_numpy())
    np.save(args.output / "probability.npy", np.concatenate(probabilities, axis=1))
    # A deliberately smaller smoke prior has equal 1/n masses, not 1/24m.
    combined_mass = np.sum(masses, axis=0)*subset["n_rows"]/n
    points = stencil(frozen["center"], frozen["h"])
    ExactPopulationNormalization(dict(zip(points, combined_mass)), frozen["h"], frozen,
        dict(shards=[str(p.resolve()) for p in paths], pilot_only=args.pilot)).save(args.output / "normalization.json")
    files = ("proposal/coordinates.npz", "proposal/manifest.json", "zero.npy", "probability.npy", "normalization.json")
    write_json(args.output / "manifest.json", dict(status="complete", identity=frozen, n_rows=n,
        pilot_only=args.pilot, points=points, receipts=receipts,
        output_sha256={name: file_hash(args.output / name) for name in files}))
    print(f"ASSEMBLED atoms={n} output={args.output}", flush=True)


def run(args):
    started = time.monotonic()
    subset = load_subset_manifest(args.subset)
    frozen = identity(args, subset)
    prepared = json.loads((args.prepared / "manifest.json").read_text())
    compatibility = check_prepared_identity(prepared["identity"], frozen)
    if (prepared["status"] != "complete"
            or prepared["pilot_only"] != args.pilot or args.output.exists()):
        raise ValueError("complete matching preparation, explicit pilot mode and new output required")
    for name, expected in prepared["output_sha256"].items():
        if file_hash(args.prepared / name) != expected:
            raise ValueError(f"prepared hash differs: {name}")
    flow = load_measurement_model(V36_LIKE.flow_checkpoints[0], device=args.device)
    model = load_disk_response(V36_LIKE, device=args.device, nthread=4)
    responses = []
    for receipt in prepared["receipts"]:
        source = receipt["subset_receipt"]
        root = Path(source["root"])
        for name in ("pair_indptr.npy", "pair_features.npy"):
            if file_hash(root / name) != source["output_sha256"][name]:
                raise ValueError(f"response source hash differs: {root/name}")
        ptr = np.load(root / "pair_indptr.npy", mmap_mode="r")[:receipt["n_rows"]+1]
        features = np.load(root / "pair_features.npy", mmap_mode="r")[:ptr[-1]]
        responses.append(CatalogueDiskResponse(ptr, features, model))
    response = ShardedDiskResponse(responses)
    probability = np.load(args.prepared / "probability.npy", mmap_mode="r")
    zero = pd.DataFrame(np.load(args.prepared / "zero.npy", mmap_mode="r"), columns=FLOW_FEATURES)
    cache = FrozenDiskCache(zero, dict(zip(map(tuple, prepared["points"]), probability)), subset["conditions"])
    config = json.loads(CONFIG.read_text())
    cut = OutputCut.from_specs(flow.target_transform.target_names, specs=config["measured_selection"]["bounds"])
    likelihood = DiskCatalogueLikelihood(flow, cache, disk_response=response, selection=CatalogueSelection(cut))
    likelihood.population_normalization = ExactPopulationNormalization.from_payload(
        json.loads((args.prepared / "normalization.json").read_text()))
    if likelihood.population_normalization.identity != prepared["identity"]:
        raise ValueError("normalizer identity differs")
    # A proposal table rebuilt at another dispersion floor is still a proposal
    # over the same atoms in the same order: it changes which atoms are
    # scored into reach, never what any of them is worth.  The prepared
    # table's own hash is verified above either way.
    proposal_source = args.proposal if args.proposal is not None else args.prepared / "proposal"
    coords = ProposalCoordinateTable.load(proposal_source)
    if coords.values.shape[0] != response.n_atoms:
        raise ValueError("proposal table does not cover the prepared atoms")
    # Keep historical float64 tilted score arithmetic, not the synthetic timing probe's fp32.
    proposal = DefensiveLocalProposal(coords, cache.prior.weights,
        local_base_weights=cache.get(0., 0.).detection_probability, score_dtype=torch.float64)
    mock = MockCatalogue.load(args.input)
    stop = min(args.start+args.count, len(mock.measurements))
    if args.start < 0 or args.count < 1 or stop <= args.start:
        raise ValueError("nonempty valid observation window required")
    window = MockCatalogue(mock.measurements.iloc[args.start:stop].reset_index(drop=True),
        mock.truth.iloc[args.start:stop].reset_index(drop=True))
    if args.compile_flow:
        flow.compile_log_prob(mode=None, dynamic=True)
    args.output.mkdir(parents=True)
    ladder = (512, 1024, 2048, 4096, 8192, 16384)
    print(f"INFERENCE_START atoms={response.n_atoms} observations={len(window.measurements)} load_seconds={time.monotonic()-started:.1f}", flush=True)
    moments = run_adaptive_section5(likelihood, window, proposal, center=frozen["center"], h=frozen["h"],
        draw_ladder=ladder, n_candidates=1024, proposal_prefilter_candidates=131072,
        epsilon=.1, proposal_seed=8701, min_ess=32., max_weight_fraction=.5,
        candidate_backend="torch", estimator_mode="tilted_stratified", tilt_delta=.1,
        retain_full_ladder=True, full_information=True, object_id_offset=args.start,
        object_chunk=args.object_chunk, atom_chunk=4096,
        progress=lambda done, total, elapsed: print(f"INFERENCE rows={done}/{total} seconds={elapsed:.2f}", flush=True))
    arrays = {name: getattr(moments, name) for name in ("score", "information", "draw_counts", "unique_counts",
        "ladder_score", "ladder_information", "weight_ess", "weight_max_fraction", "weight_relative_error", "weight_pareto_k")}
    arrays["object_rows"] = np.arange(args.start, stop)
    arrays["weight_diagnostic_draws"] = np.asarray(moments.weight_diagnostic_draws)
    np.savez_compressed(args.output / "one_step_moments.npz", **arrays)
    injected = mock.truth[["injected_g1", "injected_g2"]].drop_duplicates().to_numpy()
    if injected.shape != (1, 2):
        raise ValueError("one injected shear required")
    result = dict(status="moments_complete", pilot_only=args.pilot,
        preparation_identity=prepared["identity"], runtime_cache_compatibility=compatibility,
        initial_center=dict(center=frozen["center"], strategy="mean_observed_shape_full_frozen_input"),
        injected_shear=injected[0].tolist(), config=dict(adaptive_draw_ladder=list(ladder),
            likelihood="v3.6-like", estimator="v1.2-infer-16k", tilt_score_precision="float64",
            compile_flow=args.compile_flow, object_chunk=args.object_chunk),
        model_sha256=frozen["models"], model_cache_sha256=file_hash(args.prepared / "manifest.json"),
        scene_sha256=frozen["subset_sha256"],
        prepared_proposal_sha256=prepared["output_sha256"]["proposal/coordinates.npz"],
        proposal_source=str(proposal_source.resolve()),
        proposal_cache_sha256=file_hash(proposal_source / "coordinates.npz"),
        proposal_metadata=dict(coords.metadata or {}),
        mock_input_sha256=frozen["input_sha256"], implementation_sha256=frozen["implementation_sha256"],
        observation_partition=dict(start=args.start, stop=stop, n_partition=stop-args.start, n_total=len(mock.measurements)),
        one_step_moments=dict(path="one_step_moments.npz", sha256=file_hash(args.output / "one_step_moments.npz")),
        result=dict(elapsed_seconds=moments.elapsed_seconds, phase_seconds=dict(moments.phase_seconds)),
        total_seconds=time.monotonic()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        note="Per-object moments only; combine disjoint partitions and require positive-definite total information.")
    resolved = json.loads(NUMERICS.read_text())
    resolved["prior"] = {"manifest": str(args.subset.resolve()), "sha256": frozen["subset_sha256"],
                         "n_atoms": response.n_atoms, "pilot_only": args.pilot}
    resolved["execution"].update(compile_flow=args.compile_flow, object_chunk=args.object_chunk,
                                 tilt_score_precision="float64")
    result.update(pipeline_release="custom", pipeline_base_release="v1.2-infer-16k",
        pipeline_config_sha256=frozen["inference_config_sha256"],
        pipeline_resolved_config_sha256=sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest(),
        likelihood_release="v3.6-like", likelihood_config_sha256=frozen["likelihood_config_sha256"],
        pipeline_config=resolved, likelihood_component_sha256=frozen["models"],
        pipeline_implementation_sha256=frozen["implementation_sha256"],
        likelihood_implementation_sha256=frozen["implementation_sha256"])
    write_json(args.output / "result.json", result)
    print(f"INFERENCE_COMPLETE {args.output} seconds={moments.elapsed_seconds:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "assemble", "run"))
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--proposal", type=Path,
        help="proposal table to draw from instead of the prepared one, as written by "
             "scripts/rebuild_proposal_cache.py; the prepared table is still hash-verified")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-atoms", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--object-chunk", type=int, default=16)
    parser.add_argument("--compile-flow", action="store_true")
    args = parser.parse_args()
    if args.action != "prepare" and args.prepared is None:
        parser.error("--prepared is required for assemble/run")
    torch.set_num_threads(4)
    {"prepare": prepare, "assemble": assemble, "run": run}[args.action](args)


if __name__ == "__main__":
    main()
