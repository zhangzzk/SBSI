"""Minimal frozen-pipeline loader for recovered Infer V1 diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from plot_infer_v1_sampling import _sha256, _verify_hashes
from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.scene_prior import ScenePrior


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_pipeline(reference_path: Path, device: str):
    """Reproduce the final loader used by the archived ladder artifact."""

    reference = json.loads(reference_path.read_text())
    if reference.get("status") not in (None, "complete") or (
        reference.get("status") is None and "result" not in reference
    ):
        raise RuntimeError("reference inference result is not complete")
    config = reference["config"]
    identity = reference.get("identity", reference)
    unsupported = {
        "blend_response_cache": config.get("blend_response_cache"),
        "selection_cache": config.get("selection_cache"),
        "cut_abs_ehat": config.get("cut_abs_ehat"),
        "cut_bound": config.get("cut_bound"),
    }
    if any(bool(value) for value in unsupported.values()):
        raise RuntimeError("diversification diagnostic requires no measured cut")
    _verify_hashes(config["scene_store"], identity["scene_sha256"], "scene")
    _verify_hashes(config["model_cache"], identity["model_cache_sha256"], "model cache")
    _verify_hashes(
        config["proposal_cache"], identity["proposal_cache_sha256"], "proposal cache"
    )
    _verify_hashes(config["mock_input"], identity["mock_input_sha256"], "likelihood mock")
    model_files = {
        "measurement": config["measurement_model"],
        "emulator": config["emulator_model"],
        "emulator_metadata": config["emulator_metadata"],
    }
    if {name: _sha256(path) for name, path in model_files.items()} != identity[
        "model_sha256"
    ]:
        raise RuntimeError("model hashes do not match the reference result")
    conditions = reference.get("conditions") or {
        name: config[name]
        for name in ("pixel_size", "zero_point", "psf_fwhm", "moffat_beta", "pixel_rms")
    }
    prior = ScenePrior.load(config["scene_store"])
    flow = load_measurement_model(config["measurement_model"], device=device)
    detector = load_emulator(
        ModelPaths(
            flow_checkpoints=(Path(config["measurement_model"]),),
            emulator_model=Path(config["emulator_model"]),
            emulator_metadata=Path(config["emulator_metadata"]),
        ),
        conditions=conditions,
        device=device,
    )
    cache = CatalogueModelCache.load(config["model_cache"], prior=prior)
    cache.attach_detector(detector)
    cache.validate_model_features(flow)
    cache.validate_detection_shear_invariance()
    likelihood = CatalogueLikelihood(flow, cache)
    coordinates = ProposalCoordinateTable.load(config["proposal_cache"])
    proposal = DefensiveLocalProposal(
        coordinates,
        prior.weights,
        local_base_weights=cache.get(0.0, 0.0).detection_probability,
    )
    mock = MockCatalogue.load(config["mock_input"])
    if config.get("compile_flow", False):
        flow.compile_log_prob(mode=None, dynamic=True)
    metadata = {
        "inference_version": config["inference_version"],
        "reference_result": str(reference_path),
        "reference_result_sha256": _sha256(reference_path),
        "diagnostic_sha256": _sha256(Path(__file__)),
        "model_sha256": identity["model_sha256"],
        "model_cache_sha256": identity["model_cache_sha256"],
        "proposal_cache_sha256": identity["proposal_cache_sha256"],
        "mock_input_sha256": identity["mock_input_sha256"],
        "implementation_sha256": identity["implementation_sha256"],
    }
    return config, identity, likelihood, proposal, mock, metadata
