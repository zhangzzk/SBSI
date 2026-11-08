#!/usr/bin/env python
"""Build new model/QMC caches using verified neighbour-complete truth features.

Only deterministic zero-shear conditions are reused. Classifier probabilities
and all flow proposal coordinates are evaluated with the requested models.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sbsi.catalogue_blend import CatalogueBlendResponse
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache, CatalogueModelView
from sbsi.catalogue_sampling import ProposalCoordinateTable
from sbsi.flow_size_condition import CIRCULARIZED_FEATURE, circularized_radius
from sbsi.measurement_model import load_measurement_model
from sbsi.scene_prior import ScenePrior
from sbsi.selection_model import load_selection_model_ensemble
from sbsi.coordinates import ellipticity_from_axis_ratio_angle


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ProgressFlow:
    def __init__(self, bundle):
        self.bundle = bundle
        self.rows = 0
        self.started = time.monotonic()

    def __getattr__(self, name):
        return getattr(self.bundle, name)

    def sample(self, frame, **kwargs):
        result = self.bundle.sample(frame, **kwargs)
        previous = self.rows
        self.rows += len(frame)
        if previous == 0 or self.rows // 131072 != previous // 131072:
            print(f"QMC rows={self.rows} elapsed={time.monotonic()-self.started:.1f}s", flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-store", type=Path, required=True)
    parser.add_argument("--source-model-cache", type=Path, required=True)
    parser.add_argument("--source-merge-report", type=Path, required=True)
    parser.add_argument("--blend-response-cache", type=Path, required=True)
    parser.add_argument("--likelihood-config", type=Path, required=True)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--measurement-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    config = json.loads(args.likelihood_config.read_text())
    inference = json.loads(args.inference_config.read_text())
    detection = config["emulator"]["detection_model"]
    if detection.get("backend") != "sbsi_selection_model_ensemble" or detection.get("aggregation") != "arithmetic_mean_probability":
        raise ValueError("requires an explicit arithmetic-mean probability ensemble")
    paths = [(args.measurement_model, config["measurement_model"]["sha256"])]
    paths += [(Path(member["path"]), member["sha256"]) for member in detection["members"]]
    paths += [(Path(config["emulator"][name]["path"]), config["emulator"][name]["sha256"]) for name in ("model", "metadata")]
    for path, expected in paths:
        if file_hash(path) != expected:
            raise ValueError(f"model hash mismatch: {path}")
    scene_hashes = {name: file_hash(args.scene_store / name) for name in ("manifest.json", "galaxies.parquet", "neighbours.npz")}
    source = json.loads((args.source_model_cache / "manifest.json").read_text())
    if source["metadata"].get("scene_sha256") != scene_hashes:
        raise ValueError("source conditions are not bound to this scene")
    if source["metadata"].get("compaction") != "positive_atoms_only_after_neighbour_complete_shard_views":
        raise ValueError("source does not declare neighbour-complete compaction")
    geometry = config["geometry"]
    for key, expected in (("conditions", config["observing_conditions"]),
                          ("flow_neighbour_radius_arcsec", geometry["flow_neighbour_radius_arcsec"]),
                          ("crowding_radii_arcsec", geometry["crowding_radii_arcsec"])):
        if source.get(key) != expected:
            raise ValueError(f"source geometry mismatch: {key}")
    zero = [entry for entry in source["views"] if entry["g1"] == entry["g2"] == 0.0]
    if len(zero) != 1:
        raise ValueError("source must contain exactly one zero view")
    source_flow = args.source_model_cache / zero[0]["flow"]
    merge_report = json.loads(args.source_merge_report.read_text())
    source_key = (
        "zero_conditions" if source.get("kind") == "neighbour_complete_zero_conditions" else "model_cache"
    )
    for name in ("manifest.json", source_flow.name):
        if file_hash(args.source_model_cache / name) != merge_report["output_sha256"][source_key][name]:
            raise ValueError(f"source cache differs from its compaction report: {name}")
    print("Loading aligned prior and neighbour-complete zero conditions", flush=True)
    prior = ScenePrior.load(args.scene_store)
    frame = pd.read_parquet(source_flow).set_index("primary_row")
    expected_index = pd.Index(np.arange(len(prior.galaxies), dtype=np.int64), name="primary_row")
    if not frame.index.equals(expected_index):
        raise ValueError("source zero view is not aligned to prior rows")
    for name, truth_name in (("Re_input_p", "Re"), ("r_input_p", "r"), ("sersic_n_input_p", "sersic_n")):
        np.testing.assert_allclose(frame[name], prior.galaxies[truth_name], rtol=0, atol=1e-12)
    e1, e2 = ellipticity_from_axis_ratio_angle(prior.galaxies.axis_ratio, prior.galaxies.position_angle)
    np.testing.assert_allclose(frame.e1_input_p, e1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(frame.e2_input_p, e2, rtol=0, atol=1e-12)
    frame[CIRCULARIZED_FEATURE] = circularized_radius(prior.galaxies.Re, prior.galaxies.axis_ratio)
    response = CatalogueBlendResponse.load(args.blend_response_cache)
    blend_hashes = {name: file_hash(args.blend_response_cache / name) for name in ("manifest.json", "r_blend.npy")}
    if response.metadata.get("scene_sha256") != scene_hashes:
        raise ValueError("response cache is not bound to this scene")
    expected_emulator = {name: config["emulator"][name]["sha256"] for name in ("model", "metadata")}
    if response.metadata.get("emulator_sha256") != expected_emulator or response.metadata.get("conditions") != config["observing_conditions"]:
        raise ValueError("response cache has different model/conditions")
    bundle = load_measurement_model(args.measurement_model, device=args.device)
    detector = load_selection_model_ensemble([Path(member["path"]) for member in detection["members"]], device=args.device)
    flow_frame = frame.loc[:, bundle.condition_preprocessor.feature_names].copy()
    frame["R_blend"] = response.values
    detector_frame = frame.loc[:, detector.preprocessor.feature_names].copy()
    del frame
    print(f"Evaluating three-member classifier on {len(flow_frame)} atoms", flush=True)
    probability = detector.predict_proba(detector_frame)
    cache = CatalogueModelCache(
        prior, detector=detector, conditions=config["observing_conditions"],
        detection_radius_arcsec=geometry["detection_radius_arcsec"],
        flow_neighbour_radius_arcsec=geometry["flow_neighbour_radius_arcsec"],
        crowding_radii_arcsec=geometry["crowding_radii_arcsec"], blend_response=response,
        flow_features=bundle.condition_preprocessor.feature_names,
    )
    cache._views[(0.0, 0.0)] = CatalogueModelView(0.0, 0.0, flow_frame, detector_frame, probability, response.shape_shift_at(prior, 0.0, 0.0))
    cache.validate_model_features(bundle)
    model_hashes = {"measurement": config["measurement_model"]["sha256"],
                    "emulator": expected_emulator["model"], "emulator_metadata": expected_emulator["metadata"]}
    source_identity = {"root": str(args.source_model_cache), "manifest_sha256": file_hash(args.source_model_cache / "manifest.json"),
                       "merge_report_sha256": file_hash(args.source_merge_report),
                       "flow_sha256": file_hash(source_flow), "reuse": "deterministic neighbour-complete truth conditions only"}
    cache.save(args.output / "model_cache", metadata={
        "scene_sha256": scene_hashes,
        "model_sha256": {**model_hashes, "detection": {"aggregation": detection["aggregation"], "members": [member["sha256"] for member in detection["members"]]}},
        "r_blend": {"enabled": True, "cache_sha256": blend_hashes}, "source_zero_conditions": source_identity,
    })
    proposal = inference["proposal"]
    print("Building new flow QMC proposal coordinates", flush=True)
    coordinates = ProposalCoordinateTable.from_flow(
        CatalogueLikelihood(ProgressFlow(bundle), cache), target_names=bundle.target_transform.target_names,
        n_flow_samples=proposal["flow_samples"], statistic=proposal["location_statistic"],
        dispersion_statistic=proposal["dispersion_statistic"], row_chunk=proposal["row_chunk"], seed=proposal["coordinate_seed"],
        metadata={"model_sha256": model_hashes, "scene_sha256": scene_hashes, "conditions": config["observing_conditions"],
                  "flow_neighbour_radius_arcsec": geometry["flow_neighbour_radius_arcsec"], "crowding_radii_arcsec": geometry["crowding_radii_arcsec"],
                  "coordinate_config": {"n_flow_samples": proposal["flow_samples"], "statistic": proposal["location_statistic"],
                                        "dispersion_statistic": proposal["dispersion_statistic"], "seed": proposal["coordinate_seed"]}},
    )
    coordinates.save(args.output / "proposal_cache")
    (args.output / "completion.json").write_text(json.dumps({"n_atoms": len(prior.galaxies), "source": source_identity,
        "implementation_sha256": file_hash(__file__), "likelihood_config_sha256": file_hash(args.likelihood_config)}, indent=2) + "\n")
    print("MODEL_QMC_CACHE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
