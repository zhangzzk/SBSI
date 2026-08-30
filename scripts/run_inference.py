#!/usr/bin/env python
"""Run the v1.1-infer one-step full-2D catalogue-likelihood estimator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from sbsi.catalogue_blend import CatalogueBlendResponse
from sbsi.catalogue_closure import MockCatalogue, generate_mock_catalogue
from sbsi.catalogue_likelihood import (
    CatalogueLikelihood,
    CatalogueModelCache,
    CatalogueSelection,
    OutputCut,
)
from sbsi.catalogue_null import (
    estimate_one_step_adaptive_section5,
)
from sbsi.catalogue_sampling import (
    TAIL_PROPOSAL_ATOM_CHUNK,
    TAIL_PROPOSAL_OBJECT_CHUNK,
    TAIL_PROPOSAL_RECIPE,
    DefensiveLocalProposal,
    ProposalCoordinateTable,
    SpatialCellPartition,
)
from sbsi.amortized_proposal import ContrastiveRetrievalProposal
from sbsi.complement_diagnostic import (
    COMPLEMENT_DIAGNOSTIC_SEED_OFFSET,
    PILOT_STRATUM_DIAGNOSTIC_SEED_OFFSET,
    diagnose_complement,
)
from sbsi.learned_retrieval_diagnostic import (
    LEARNED_RETRIEVAL_CENTER_CHECKPOINT_SHA256,
    LEARNED_RETRIEVAL_CENTER_PROTOCOL,
    learned_retrieval_center_summary,
)
from sbsi.spatial_neyman_diagnostic import (
    SPATIAL_NEYMAN_PROTOCOL,
    diagnose_spatial_neyman_complement,
    spatial_neyman_summary,
)
from sbsi.spatial_h256_u8_ps2_diagnostic import (
    SPATIAL_H256_U8_PS2_PROTOCOL,
    diagnose_spatial_h256_u8_ps2_complement,
    spatial_h256_u8_ps2_center_summary,
)
from sbsi.forward_catalogue import EmulatorPairingConfig
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.scene_prior import SHEAR_TRANSFORM, ScenePrior


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INFERENCE_CONFIG = REPOSITORY_ROOT / "configs" / "inference.json"
DEFAULT_LIKELIHOOD_CONFIG = REPOSITORY_ROOT / "configs" / "likelihood.json"
SPATIAL_NEYMAN_PARTITION_FILE = "spatial_partition.npz"
SPATIAL_NEYMAN_PARTITION_MANIFEST = "manifest.json"
SPATIAL_NEYMAN_PARTITION_AUDIT = "partition_audit.json"


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair(value: str) -> tuple[float, float]:
    parts = tuple(float(item) for item in value.split(","))
    if len(parts) != 2 or not np.isfinite(parts).all():
        raise argparse.ArgumentTypeError("expected two finite comma-separated values")
    return parts


def _matrix2(value: str) -> tuple[tuple[float, float], tuple[float, float]]:
    parts = tuple(float(item) for item in value.split(","))
    if len(parts) != 4 or not np.isfinite(parts).all():
        raise argparse.ArgumentTypeError(
            "expected four finite comma-separated row-major values"
        )
    matrix = np.asarray(parts, dtype=np.float64).reshape(2, 2)
    if abs(float(np.linalg.det(matrix))) <= 1e-12:
        raise argparse.ArgumentTypeError("mean-shape response matrix must be invertible")
    return tuple(tuple(map(float, row)) for row in matrix)


def _optional_float(text):
    """Argparse type accepting a float or ``none`` for an absent cut."""

    return None if str(text).strip().lower() in {"none", "off", ""} else float(text)


def _mean_observed_shape(measurements, target_names):
    """Return the cheap data-derived centre used before one likelihood step.

    This is deliberately only an optimizer starting point.  It is not reported as a
    calibrated shear: the measurement response and additive terms are left for the
    catalogue likelihood score and full 2x2 curvature to correct.
    """

    names = ("measured_ngmix_g1", "measured_ngmix_g2")
    missing = sorted(set(names) - set(target_names))
    if missing:
        raise ValueError(
            f"mean-observed initial centre requires flow targets {names}; missing {missing}"
        )
    values = measurements.loc[:, names].to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("mean-observed initial centre requires nonempty shape measurements")
    if not np.isfinite(values).all():
        raise ValueError("mean-observed initial centre received non-finite shapes")
    center = values.mean(axis=0, dtype=np.float64)
    if not np.isfinite(center).all():
        raise ValueError("mean-observed initial centre is non-finite")
    return (float(center[0]), float(center[1])), names


def _calibrated_mean_observed_shape(
    measurements,
    target_names,
    *,
    offset,
    response,
):
    """Map the raw mean shape to an optimizer-only shear pilot.

    The affine calibration only improves the expansion centre.  The catalogue
    likelihood still supplies the final score, full information, and estimate.
    """

    raw, names = _mean_observed_shape(measurements, target_names)
    offset_array = np.asarray(offset, dtype=np.float64)
    response_array = np.asarray(response, dtype=np.float64)
    if offset_array.shape != (2,) or response_array.shape != (2, 2):
        raise ValueError("mean-shape calibration has the wrong shape")
    if not np.isfinite(offset_array).all() or not np.isfinite(response_array).all():
        raise ValueError("mean-shape calibration must be finite")
    try:
        center = np.linalg.solve(response_array, np.asarray(raw) - offset_array)
    except np.linalg.LinAlgError as error:
        raise ValueError("mean-shape response matrix must be invertible") from error
    if not np.isfinite(center).all():
        raise ValueError("calibrated mean-shape initial centre is non-finite")
    return tuple(map(float, center)), names, raw


def _full_ladder_one_step_report(moments):
    """Summarize common-draw full-2D Newton estimates at every retained rung."""

    if moments.ladder_score is None:
        return None
    center = np.asarray(moments.center, dtype=np.float64)
    report = {}
    previous_estimate = None
    previous_influence = None
    for rung_index, n_draws in enumerate(moments.draw_ladder):
        score = moments.ladder_score[rung_index]
        information = moments.ladder_information[rung_index]
        information_sum = information.sum(axis=0, dtype=np.float64)
        information_sum = 0.5 * (information_sum + information_sum.T)
        eigenvalues = np.linalg.eigvalsh(information_sum)
        entry = {
            "information_eigenvalues": eigenvalues.tolist(),
            "positive_definite": bool(eigenvalues[0] > 0),
        }
        try:
            step = np.linalg.solve(
                information_sum, score.sum(axis=0, dtype=np.float64)
            )
            estimate = center + step
            residual = score - np.einsum("nij,j->ni", information, step)
            influence = np.linalg.solve(
                information_sum / len(score), residual.T
            ).T
            robust_se = np.std(influence, axis=0, ddof=1) / np.sqrt(len(score))
        except np.linalg.LinAlgError:
            entry.update(
                {
                    "estimate": None,
                    "robust_standard_error": None,
                    "previous_rung_shift": None,
                    "previous_rung_paired_standard_error": None,
                    "previous_rung_pull": None,
                }
            )
            report[str(n_draws)] = entry
            previous_estimate = None
            previous_influence = None
            continue
        entry["estimate"] = estimate.tolist()
        entry["robust_standard_error"] = robust_se.tolist()
        if previous_estimate is not None:
            shift = estimate - previous_estimate
            paired = influence - previous_influence
            paired_se = np.std(paired, axis=0, ddof=1) / np.sqrt(len(paired))
            entry["previous_rung_shift"] = shift.tolist()
            entry["previous_rung_paired_standard_error"] = paired_se.tolist()
            entry["previous_rung_pull"] = np.divide(
                shift,
                paired_se,
                out=np.full(2, np.nan),
                where=paired_se > 0,
            ).tolist()
        report[str(n_draws)] = entry
        previous_estimate = estimate
        previous_influence = influence
    return report


def _file_hashes(root: str | Path, names) -> dict[str, str]:
    root = Path(root)
    return {name: _sha256(root / name) for name in names}


def _load_release_config(path: str | Path, *, kind: str) -> dict:
    """Load one small release file and reject incomplete identities early."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {kind} config {path}: {error}") from error
    if payload.get("schema_version") != 1:
        raise ValueError(f"{kind} config must use schema_version 1")
    if not isinstance(payload.get("release"), str) or not payload["release"].strip():
        raise ValueError(f"{kind} config must declare a nonempty release")
    return payload


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _inference_defaults(config: dict) -> dict:
    """Translate the release document directly to argparse destinations."""

    try:
        proposal = config["proposal"]
        estimator = config["estimator"]
        execution = config["execution"]
        if estimator["step"] != "one_step_full_2d":
            raise ValueError("only the one_step_full_2d estimator is supported")
        return {
            "proposal_flow_samples": int(proposal["flow_samples"]),
            "proposal_statistic": proposal["location_statistic"],
            "proposal_dispersion_statistic": proposal["dispersion_statistic"],
            "proposal_row_chunk": int(proposal["row_chunk"]),
            "proposal_coordinate_seed": int(proposal["coordinate_seed"]),
            "proposal_candidates": int(proposal["candidates"]),
            "proposal_prefilter_candidates": int(proposal["prefilter_candidates"]),
            "proposal_seed": int(proposal["seed"]),
            "proposal_epsilon": float(proposal["epsilon"]),
            "proposal_method": proposal["method"],
            "initial_strategy": estimator["initial_strategy"],
            "h": float(estimator["finite_difference_h"]),
            "adaptive_draw_ladder": tuple(map(int, estimator["draw_ladder"])),
            "adaptive_min_ess": float(estimator["minimum_ess"]),
            "adaptive_max_weight_fraction": float(
                estimator["maximum_weight_fraction"]
            ),
            "adaptive_allocation": estimator["allocation"],
            "adaptive_pilot_draws": int(estimator["pilot_draws"]),
            "adaptive_pilot_seed": int(estimator["pilot_seed"]),
            "adaptive_pilot_safety_factor": float(
                estimator["pilot_safety_factor"]
            ),
            "adaptive_bias_correction": estimator["bias_correction"],
            "retain_full_ladder": bool(estimator["retain_full_ladder"]),
            "compile_flow": bool(execution["compile_flow"]),
            "candidate_backend": execution["candidate_backend"],
            "object_chunk": int(execution["object_chunk"]),
            "atom_chunk": int(execution["atom_chunk"]),
            "precision": execution["precision"],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid inference config: {error}") from error


def _likelihood_defaults(config: dict) -> dict:
    try:
        geometry = config["geometry"]
        conditions = config["observing_conditions"]
        emulator = config["emulator"]
        crowding = geometry["crowding_radii_arcsec"]
        return {
            "emulator_model": str(_repo_path(emulator["model"]["path"])),
            "emulator_metadata": str(_repo_path(emulator["metadata"]["path"])),
            "detection_radius_arcsec": float(
                geometry["detection_radius_arcsec"]
            ),
            "flow_neighbour_radius_arcsec": float(
                geometry["flow_neighbour_radius_arcsec"]
            ),
            "crowding_near_arcsec": float(crowding[0]),
            "crowding_far_arcsec": float(crowding[1]),
            "pixel_size": float(conditions["pixel_size"]),
            "zero_point": float(conditions["zero_point"]),
            "psf_fwhm": float(conditions["psf_fwhm"]),
            "moffat_beta": float(conditions["moffat_beta"]),
            "pixel_rms": float(conditions["pixel_rms"]),
        }
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"invalid likelihood config: {error}") from error


def _validate_artifact(path: str | Path, spec: dict, *, name: str) -> str:
    expected = spec.get("sha256")
    actual = _sha256(path)
    if not isinstance(expected, str) or actual != expected:
        raise RuntimeError(
            f"{name} does not match the likelihood release: expected {expected}, "
            f"found {actual}"
        )
    return actual


def _json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _resolved_pipeline_config(args, source: dict) -> dict:
    resolved = {
        "schema_version": source["schema_version"],
        "release": source["release"],
        "description": source.get("description"),
        "prior": source.get("prior"),
        "proposal": {
            "flow_samples": args.proposal_flow_samples,
            "location_statistic": args.proposal_statistic,
            "dispersion_statistic": args.proposal_dispersion_statistic,
            "row_chunk": args.proposal_row_chunk,
            "coordinate_seed": args.proposal_coordinate_seed,
            "candidates": args.proposal_candidates,
            "prefilter_candidates": args.proposal_prefilter_candidates,
            "seed": args.proposal_seed,
            "epsilon": args.proposal_epsilon,
            "method": args.proposal_method,
        },
        "estimator": {
            "initial_strategy": args.initial_strategy,
            "step": "one_step_full_2d",
            "finite_difference_h": args.h,
            "draw_ladder": list(args.adaptive_draw_ladder),
            "minimum_ess": args.adaptive_min_ess,
            "maximum_weight_fraction": args.adaptive_max_weight_fraction,
            "allocation": args.adaptive_allocation,
            "pilot_draws": args.adaptive_pilot_draws,
            "pilot_seed": args.adaptive_pilot_seed,
            "pilot_safety_factor": args.adaptive_pilot_safety_factor,
            "bias_correction": args.adaptive_bias_correction,
            "retain_full_ladder": args.retain_full_ladder,
        },
        "execution": {
            "precision": args.precision,
            "compile_flow": args.compile_flow,
            "candidate_backend": args.candidate_backend,
            "object_chunk": args.object_chunk,
            "atom_chunk": args.atom_chunk,
        },
    }
    if args.estimator_mode != "mixture":
        # Screens deviate on purpose.  The key is added only when it is not the
        # release default, so an ordinary run still matches the release
        # document exactly and reports v1.1-infer rather than custom.
        resolved["estimator"]["estimator_mode"] = args.estimator_mode
        if args.estimator_mode == "tilted_stratified":
            # The expanded complement proposal is a scientific recipe, not a
            # deployment knob.  Copy through JSON so tuples become stable
            # manifest lists and callers cannot mutate the module constant.
            resolved["estimator"]["tail_proposal"] = json.loads(
                json.dumps(asdict(TAIL_PROPOSAL_RECIPE))
            )
            resolved["execution"]["tail_proposal_atom_chunk"] = (
                TAIL_PROPOSAL_ATOM_CHUNK
            )
            resolved["execution"]["tail_proposal_object_chunk"] = (
                TAIL_PROPOSAL_OBJECT_CHUNK
            )
    return resolved


def _result_arguments(args) -> dict:
    """JSON-safe invocation details, excluding the retired naming switch."""

    return {
        name: value
        for name, value in vars(args).items()
        if name != "legacy_inference_version"
    }


def _selection_cache_hashes(root: str | Path) -> dict[str, str]:
    """Hash a selection manifest and every probability array it names."""

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    names = ["manifest.json"]
    names.extend(entry["probability"] for entry in manifest.get("entries", ()))
    return _file_hashes(root, names)


def _model_cache_hashes(root: str | Path) -> dict[str, str]:
    """Hash a model-cache manifest and every table it references."""

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    names = ["manifest.json"]
    for view in manifest.get("views", ()):
        names.extend((view["flow"], view["detection"]))
    return _file_hashes(root, names)


def _mock_input_hashes(root: str | Path) -> dict[str, str]:
    """Hash a frozen mock and whichever provenance manifest it carries."""

    root = Path(root)
    names = ["measurements.parquet", "truth.parquet"]
    for manifest in ("image_mock_manifest.json", "likelihood_mock_manifest.json"):
        if (root / manifest).is_file():
            names.append(manifest)
    return _file_hashes(root, names)


def _spatial_partition_manifest_payload(
    partition: SpatialCellPartition,
    *,
    partition_cache_sha256: str,
    provenance: dict,
) -> dict:
    """Return the path-independent manifest for an audited spatial partition."""

    return {
        "schema_version": 1,
        "kind": "SBSI/outcome-blind-spatial-cell-partition/v1",
        "protocol_label": SPATIAL_NEYMAN_PROTOCOL["label"],
        "partition_file": SPATIAL_NEYMAN_PARTITION_FILE,
        "partition_cache_sha256": partition_cache_sha256,
        "partition_sha256": partition.partition_sha256,
        "partition_recipe_sha256": partition.recipe_sha256,
        "hash_contract": partition.hash_contract,
        "n_cells": int(len(partition.counts)),
        "n_active": int(len(partition.cell_id)),
        "cell_count_minimum": int(np.min(partition.counts)),
        "cell_count_maximum": int(np.max(partition.counts)),
        "detected_mass_minimum": float(np.min(partition.detected_mass)),
        "detected_mass_maximum": float(np.max(partition.detected_mass)),
        "prior_mass_minimum": float(np.min(partition.prior_mass)),
        "prior_mass_maximum": float(np.max(partition.prior_mass)),
        "input_sha256": {
            "active_indices_sha256": partition.active_indices_sha256,
            "coordinates_sha256": partition.coordinates_sha256,
            "coordinate_names_sha256": partition.coordinate_names_sha256,
            "detected_weights_sha256": partition.detected_weights_sha256,
            "prior_weights_sha256": partition.prior_weights_sha256,
        },
        "array_sha256": {
            "assignment_sha256": partition.assignment_sha256,
            "counts_sha256": partition.counts_sha256,
            "detected_mass_sha256": partition.detected_mass_sha256,
            "prior_mass_sha256": partition.prior_mass_sha256,
        },
        "partition_metadata": partition.metadata,
        "provenance": provenance,
        "science_rows_read": 0,
        "candidate_queries": 0,
        "target_flow_evaluations": 0,
    }


def _write_spatial_partition_audit(
    output: Path,
    proposal: DefensiveLocalProposal,
    *,
    provenance: dict,
) -> dict:
    """Build and freeze the outcome-blind partition, then write its audit."""

    started = time.perf_counter()
    if proposal.tree is not None:
        raise RuntimeError("spatial partition audit unexpectedly built a KD tree")
    partition = SpatialCellPartition.build(proposal, n_cells=int(SPATIAL_NEYMAN_PROTOCOL["n_cells"]))
    partition_path = output / SPATIAL_NEYMAN_PARTITION_FILE
    partition.save(partition_path)
    partition_cache_sha256 = _sha256(partition_path)
    manifest = _spatial_partition_manifest_payload(
        partition,
        partition_cache_sha256=partition_cache_sha256,
        provenance=provenance,
    )
    manifest_path = output / SPATIAL_NEYMAN_PARTITION_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_sha256 = _sha256(manifest_path)
    audit = {
        "status": "complete",
        "outcome_blind": True,
        "partition_sha256": partition.partition_sha256,
        "partition_cache_sha256": partition_cache_sha256,
        "partition_manifest_sha256": manifest_sha256,
        "partition_recipe_sha256": partition.recipe_sha256,
        "n_cells": int(len(partition.counts)),
        "n_active": int(len(partition.cell_id)),
        "cell_count_minimum": int(np.min(partition.counts)),
        "cell_count_maximum": int(np.max(partition.counts)),
        "detected_mass_minimum": float(np.min(partition.detected_mass)),
        "detected_mass_maximum": float(np.max(partition.detected_mass)),
        "prior_mass_minimum": float(np.min(partition.prior_mass)),
        "prior_mass_maximum": float(np.max(partition.prior_mass)),
        "science_rows_read": 0,
        "candidate_queries": 0,
        "target_flow_evaluations": 0,
        "elapsed_seconds": float(time.perf_counter() - started),
        "files": {
            SPATIAL_NEYMAN_PARTITION_FILE: partition_cache_sha256,
            SPATIAL_NEYMAN_PARTITION_MANIFEST: manifest_sha256,
        },
    }
    audit_path = output / SPATIAL_NEYMAN_PARTITION_AUDIT
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def _load_verified_spatial_partition(
    root: str | Path,
    proposal: DefensiveLocalProposal,
    *,
    protocol: dict | None = None,
) -> tuple[SpatialCellPartition, float]:
    """Load a frozen partition and verify bytes, semantics, and live inputs."""

    started = time.perf_counter()
    expected = SPATIAL_NEYMAN_PROTOCOL if protocol is None else protocol
    frozen_hashes = {
        name: expected.get(name)
        for name in (
            "partition_sha256",
            "partition_cache_sha256",
            "partition_manifest_sha256",
        )
    }
    missing = sorted(name for name, value in frozen_hashes.items() if value is None)
    if missing:
        raise SystemExit("the spatial-Neyman protocol lacks frozen partition hashes: " + ", ".join(missing))
    root = Path(root)
    manifest_path = root / SPATIAL_NEYMAN_PARTITION_MANIFEST
    partition_path = root / SPATIAL_NEYMAN_PARTITION_FILE
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load spatial-Neyman partition manifest: {error}") from error
    try:
        actual_manifest_sha256 = _sha256(manifest_path)
        actual_cache_sha256 = _sha256(partition_path)
    except OSError as error:
        raise SystemExit(f"cannot hash spatial-Neyman partition artifacts: {error}") from error
    if actual_manifest_sha256 != frozen_hashes["partition_manifest_sha256"]:
        raise SystemExit(
            "spatial-Neyman partition manifest SHA-256 mismatch: "
            f"{actual_manifest_sha256} != "
            f"{frozen_hashes['partition_manifest_sha256']}"
        )
    if actual_cache_sha256 != frozen_hashes["partition_cache_sha256"]:
        raise SystemExit(
            "spatial-Neyman partition cache SHA-256 mismatch: "
            f"{actual_cache_sha256} != {frozen_hashes['partition_cache_sha256']}"
        )
    manifest_identity = {
        "partition_sha256": manifest.get("partition_sha256"),
        "partition_cache_sha256": manifest.get("partition_cache_sha256"),
        "partition_recipe_sha256": manifest.get("partition_recipe_sha256"),
        "n_cells": manifest.get("n_cells"),
        "n_active": manifest.get("n_active"),
    }
    expected_identity = {
        "partition_sha256": frozen_hashes["partition_sha256"],
        "partition_cache_sha256": frozen_hashes["partition_cache_sha256"],
        "partition_recipe_sha256": expected["partition_recipe_sha256"],
        "n_cells": expected["n_cells"],
        "n_active": expected["active_atoms"],
    }
    if manifest_identity != expected_identity:
        raise SystemExit(
            "spatial-Neyman partition manifest identity mismatch: "
            + json.dumps(
                {"expected": expected_identity, "observed": manifest_identity},
                sort_keys=True,
            )
        )
    try:
        partition = SpatialCellPartition.load(partition_path, proposal=proposal)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"spatial-Neyman partition failed semantic verification: {error}") from error
    manifest_embedded_identity = {
        "hash_contract": manifest.get("hash_contract"),
        "input_sha256": manifest.get("input_sha256"),
        "array_sha256": manifest.get("array_sha256"),
        "cell_count_minimum": manifest.get("cell_count_minimum"),
        "cell_count_maximum": manifest.get("cell_count_maximum"),
        "detected_mass_minimum": manifest.get("detected_mass_minimum"),
        "detected_mass_maximum": manifest.get("detected_mass_maximum"),
        "prior_mass_minimum": manifest.get("prior_mass_minimum"),
        "prior_mass_maximum": manifest.get("prior_mass_maximum"),
    }
    partition_embedded_identity = {
        "hash_contract": partition.hash_contract,
        "input_sha256": {
            "active_indices_sha256": partition.active_indices_sha256,
            "coordinates_sha256": partition.coordinates_sha256,
            "coordinate_names_sha256": partition.coordinate_names_sha256,
            "detected_weights_sha256": partition.detected_weights_sha256,
            "prior_weights_sha256": partition.prior_weights_sha256,
        },
        "array_sha256": {
            "assignment_sha256": partition.assignment_sha256,
            "counts_sha256": partition.counts_sha256,
            "detected_mass_sha256": partition.detected_mass_sha256,
            "prior_mass_sha256": partition.prior_mass_sha256,
        },
        "cell_count_minimum": int(np.min(partition.counts)),
        "cell_count_maximum": int(np.max(partition.counts)),
        "detected_mass_minimum": float(np.min(partition.detected_mass)),
        "detected_mass_maximum": float(np.max(partition.detected_mass)),
        "prior_mass_minimum": float(np.min(partition.prior_mass)),
        "prior_mass_maximum": float(np.max(partition.prior_mass)),
    }
    if manifest_embedded_identity != partition_embedded_identity:
        raise SystemExit(
            "spatial-Neyman partition manifest does not match embedded assignment, input, or mass identities"
        )
    if (
        partition.partition_sha256 != frozen_hashes["partition_sha256"]
        or partition.recipe_sha256 != expected["partition_recipe_sha256"]
        or len(partition.cell_id) != expected["active_atoms"]
        or len(partition.counts) != expected["n_cells"]
    ):
        raise SystemExit("spatial-Neyman partition embedded identity does not match the frozen protocol")
    minimum_count = int(np.min(partition.counts))
    maximum_count = int(np.max(partition.counts))
    if minimum_count <= 16_384 or maximum_count > 262_144:
        raise SystemExit(
            "spatial-Neyman partition cell counts fail the frozen execution "
            f"bounds: min={minimum_count}, max={maximum_count}"
        )
    return partition, float(time.perf_counter() - started)


def _validate_loaded_mock_manifest(
    root: str | Path,
    mock: MockCatalogue,
    *,
    measurement_model_sha256: str,
    target_names,
) -> dict | None:
    """Validate a frozen image mock against its preparation manifest."""

    root = Path(root)
    manifest_path = root / "image_mock_manifest.json"
    if not manifest_path.is_file():
        if mock.kind == "image":
            raise RuntimeError("image mock lacks image_mock_manifest.json provenance")
        return None

    payload = json.loads(manifest_path.read_text())
    if payload.get("mock_kind") != mock.kind:
        raise RuntimeError("image mock kind does not match its source manifest")
    if payload.get("shear_transform") != mock.shear_transform:
        raise RuntimeError("image mock shear transform does not match its source manifest")
    if payload.get("measurement_model_sha256") != measurement_model_sha256:
        raise RuntimeError(
            "image mock was prepared for a different measurement-flow checkpoint"
        )
    if tuple(payload.get("target_names", ())) != tuple(target_names):
        raise RuntimeError("image mock target names do not match the measurement flow")

    expected_output_hashes = payload.get("output_sha256")
    actual_output_hashes = _file_hashes(
        root,
        ("measurements.parquet", "truth.parquet"),
    )
    if expected_output_hashes != actual_output_hashes:
        raise RuntimeError("image mock files do not match their source manifest")

    injected = mock.truth[["injected_g1", "injected_g2"]].drop_duplicates()
    manifest_injection = np.asarray(
        [payload.get("injected_g1"), payload.get("injected_g2")], dtype=float
    )
    if (
        len(injected) != 1
        or not np.isfinite(manifest_injection).all()
        or not np.allclose(
            injected.iloc[0].to_numpy(dtype=float),
            manifest_injection,
            rtol=0,
            atol=1e-14,
        )
    ):
        raise RuntimeError("image mock injected shear does not match its source manifest")
    return payload


def _validate_loaded_likelihood_manifest(
    root: str | Path,
    mock: MockCatalogue,
    *,
    generation_identity: dict,
    implementation_sha256: dict[str, str],
) -> dict:
    """Require full provenance before reusing a likelihood-generated mock."""

    root = Path(root)
    manifest_path = root / "likelihood_mock_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "likelihood mock lacks likelihood_mock_manifest.json provenance; "
            "generate it in-process with the current runner"
        )
    payload = json.loads(manifest_path.read_text())
    if payload.get("mock_kind") != "likelihood" or mock.kind != "likelihood":
        raise RuntimeError("likelihood mock kind does not match its source manifest")
    if payload.get("generation_identity") != generation_identity:
        raise RuntimeError(
            "likelihood mock was generated with different scene/model/selection settings"
        )
    if payload.get("implementation_sha256") != implementation_sha256:
        raise RuntimeError(
            "likelihood mock was generated by a different implementation"
        )
    actual_hashes = _file_hashes(
        root,
        ("measurements.parquet", "truth.parquet"),
    )
    if payload.get("output_sha256") != actual_hashes:
        raise RuntimeError("likelihood mock files do not match their source manifest")
    return payload


def _pipeline_implementation_hashes() -> dict[str, str]:
    """Hash the numerical inference implementation independently of the model."""

    root = REPOSITORY_ROOT
    names = (
        "configs/inference.json",
        "scripts/run_inference.py",
        "sbsi/amortized_proposal.py",
        "sbsi/catalogue_null.py",
        "sbsi/catalogue_likelihood.py",
        "sbsi/catalogue_sampling.py",
        "sbsi/learned_retrieval_diagnostic.py",
        "sbsi/spatial_neyman_diagnostic.py",
        "sbsi/spatial_h256_u8_ps2_diagnostic.py",
    )
    return {name: _sha256(root / name) for name in names}


def _likelihood_implementation_hashes() -> dict[str, str]:
    """Hash only code that changes generation by the named likelihood."""

    root = REPOSITORY_ROOT
    names = (
        "configs/likelihood.json",
        "sbsi/catalogue_likelihood.py",
        "sbsi/catalogue_blend.py",
        "sbsi/catalogue_closure.py",
        "sbsi/scene_prior.py",
        "sbsi/shear_map.py",
        "sbsi/measurement_model.py",
        "sbsi/models.py",
    )
    return {name: _sha256(root / name) for name in names}


def parse_args(argv=None):
    release_parser = argparse.ArgumentParser(add_help=False)
    release_parser.add_argument(
        "--inference-config", default=str(DEFAULT_INFERENCE_CONFIG)
    )
    release_parser.add_argument(
        "--likelihood-config", default=str(DEFAULT_LIKELIHOOD_CONFIG)
    )
    release_args, _ = release_parser.parse_known_args(argv)
    try:
        inference_config = _load_release_config(
            release_args.inference_config, kind="inference"
        )
        likelihood_config = _load_release_config(
            release_args.likelihood_config, kind="likelihood"
        )
        release_defaults = {
            **_inference_defaults(inference_config),
            **_likelihood_defaults(likelihood_config),
        }
    except ValueError as error:
        release_parser.error(str(error))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-config",
        default=release_args.inference_config,
        help="numerical release configuration (default: configs/inference.json)",
    )
    parser.add_argument(
        "--likelihood-config",
        default=release_args.likelihood_config,
        help="likelihood release configuration (default: configs/likelihood.json)",
    )
    parser.add_argument(
        "--inference-version",
        dest="legacy_inference_version",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--scene-store", required=True)
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument("--emulator-metadata")
    parser.add_argument("--emulator-model")
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--proposal-cache", required=True)
    parser.add_argument("--detection-radius-arcsec", type=float, default=3.0)
    parser.add_argument("--flow-neighbour-radius-arcsec", type=float, default=7.0)
    parser.add_argument("--crowding-near-arcsec", type=float, default=3.0)
    parser.add_argument("--crowding-far-arcsec", type=float, default=7.0)
    parser.add_argument("--proposal-flow-samples", type=int, default=16)
    parser.add_argument(
        "--proposal-statistic", choices=("mean", "median"), default="median"
    )
    parser.add_argument(
        "--proposal-dispersion-statistic",
        choices=("robust_iqr", "std"),
        default="robust_iqr",
        help=(
            "per-atom flow-draw width used by Gaussian uncertainty reranking; "
            "std with --proposal-statistic mean gives literal Gaussian moments"
        ),
    )
    parser.add_argument("--proposal-row-chunk", type=int, default=8192)
    parser.add_argument("--proposal-coordinate-seed", type=int, default=8201)
    parser.add_argument(
        "--blend-response-cache",
        default=None,
        help=(
            "optional fixed atom-aligned R_blend cache; image measurements already "
            "contain pixel blending and are never shifted again"
        ),
    )
    parser.add_argument(
        "--selection-cache",
        default=None,
        help="optional cache for per-atom measured-cut pass probabilities",
    )
    parser.add_argument(
        "--cut-abs-ehat",
        type=_optional_float,
        default=None,
        help="keep measurements with |measured shape| below this value; default none",
    )
    parser.add_argument(
        "--cut-bound",
        action="append",
        default=[],
        metavar="NAME:LO:HI",
        help=(
            "keep LO <= flow output NAME < HI; empty LO/HI is unbounded; "
            "repeatable and ANDed with --cut-abs-ehat"
        ),
    )
    parser.add_argument("--selection-samples", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=8101)
    parser.add_argument("--selection-row-chunk", type=int, default=8192)
    parser.add_argument(
        "--mock-input",
        default=None,
        help=(
            "saved mock directory; when omitted, generate a frozen likelihood mock "
            "from the configured full catalogue likelihood"
        ),
    )
    parser.add_argument("--n-detected", type=int, default=10000)
    parser.add_argument("--injected-g1", type=float, default=0.0)
    parser.add_argument("--injected-g2", type=float, default=0.0)
    parser.add_argument("--scene-seed", type=int, default=1001)
    parser.add_argument("--detection-seed", type=int, default=2001)
    parser.add_argument("--flow-seed", type=int, default=3001)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="generate and freeze the likelihood mock, then stop before inference",
    )
    parser.add_argument(
        "--observation-start",
        type=int,
        default=0,
        help="first observation used by this inference partition",
    )
    parser.add_argument(
        "--observation-stop",
        type=int,
        default=None,
        help="exclusive observation bound; default uses the complete mock",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial", type=_pair, default=(0.0, 0.0))
    parser.add_argument(
        "--initial-strategy",
        choices=(
            "fixed",
            "mean_observed_shape",
            "calibrated_mean_observed_shape",
        ),
        default="fixed",
        help=(
            "fixed uses --initial; mean_observed_shape uses the raw mean measured "
            "ngmix shape; calibrated_mean_observed_shape first applies the supplied "
            "affine response and offset"
        ),
    )
    parser.add_argument(
        "--mean-shape-offset",
        type=_pair,
        default=(0.0, 0.0),
        help="population additive offset c1,c2 for the calibrated mean-shape pilot",
    )
    parser.add_argument(
        "--mean-shape-response",
        type=_matrix2,
        default=((1.0, 0.0), (0.0, 1.0)),
        help="row-major 2x2 response mapping shear to mean measured shape",
    )
    parser.add_argument("--h", type=float, default=0.001)
    parser.add_argument("--draws", type=int, default=16384, help=argparse.SUPPRESS)
    parser.add_argument("--proposal-candidates", type=int, default=32768)
    parser.add_argument(
        "--proposal-prefilter-candidates",
        type=int,
        default=None,
        help=(
            "query this many cheap location neighbours, rerank them with cached "
            "flow dispersion, and exactly evaluate only --proposal-candidates"
        ),
    )
    parser.add_argument("--proposal-seed", type=int, default=8701)
    parser.add_argument("--proposal-epsilon", type=float, default=0.1)
    parser.add_argument(
        "--proposal-method",
        choices=("initial_center_posterior_adapted",),
        default="initial_center_posterior_adapted",
    )
    parser.add_argument("--bandwidth", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--max-iterations", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--tolerance", type=float, default=1e-4, help=argparse.SUPPRESS)
    parser.add_argument("--max-step", type=float, default=0.01, help=argparse.SUPPRESS)
    parser.add_argument("--shear-bound", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--max-backtracks", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument(
        "--adaptive-one-step",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--adaptive-draw-ladder", nargs="+", type=int, default=(512, 1024, 2048)
    )
    parser.add_argument(
        "--estimator-mode",
        choices=("mixture", "stratified", "tilted_stratified"),
        default="mixture",
        help=(
            "'mixture' is the v1.1-infer defensive proposal.  'stratified' "
            "sums the candidate support exactly and spends every draw on its "
            "complement; it requires the production-prefix allocation, a "
            "retained full ladder, and no finite-draw bias correction, and it "
            "marks the run as a custom pipeline.  'tilted_stratified' uses "
            "the same split with the fixed component-conditioned "
            "T1/T2/prior deterministic-MIS whole-catalogue complement recipe."
        ),
    )
    parser.add_argument("--adaptive-min-ess", type=float, default=32.0)
    parser.add_argument("--adaptive-max-weight-fraction", type=float, default=0.5)
    parser.add_argument(
        "--adaptive-allocation",
        choices=("production_prefix", "independent_pilot"),
        default="production_prefix",
    )
    parser.add_argument("--adaptive-pilot-draws", type=int, default=512)
    parser.add_argument("--adaptive-pilot-seed", type=int, default=18701)
    parser.add_argument("--adaptive-pilot-safety-factor", type=float, default=1.0)
    parser.add_argument(
        "--adaptive-bias-correction",
        choices=("none", "richardson_1_over_m"),
        default="none",
    )
    parser.add_argument(
        "--candidate-backend", choices=("scipy", "torch"), default="scipy"
    )
    parser.add_argument(
        "--retain-full-ladder",
        action="store_true",
        help=(
            "evaluate the deepest adaptive prefix for every object and retain "
            "common-draw score/information at every nested rung"
        ),
    )
    parser.add_argument(
        "--compile-flow",
        action="store_true",
        help="compile the dynamic-shape fp32 model.log_prob inference path",
    )
    parser.add_argument("--object-chunk", type=int, default=128)
    parser.add_argument(
        "--diagnose-complement",
        metavar="DIRECTORY",
        default=None,
        help=(
            "instead of running the estimator, measure where the target mass "
            "and the heavy weight tail sit outside the exact candidate "
            "stratum, and write the per-object table there"
        ),
    )
    parser.add_argument(
        "--diagnose-complement-global-draws",
        type=int,
        default=131072,
        help="prior draws per object used to reach beyond the prefilter",
    )
    parser.add_argument(
        "--diagnose-complement-tail-proposal",
        action="store_true",
        help=(
            "also diagnose the same fixed conditioned deterministic-MIS tail "
            "proposal used by estimator-mode tilted_stratified"
        ),
    )
    parser.add_argument(
        "--diagnose-complement-pilot-stratum",
        action="store_true",
        help=(
            "diagnose the fixed discarded-independent-pilot enlargement of "
            "the exact stratum before fresh prior and tail-proposal draws; "
            "requires --diagnose-complement-tail-proposal"
        ),
    )
    parser.add_argument(
        "--diagnose-complement-exact-oracle",
        action="store_true",
        help=(
            "on a declared 512-object complement diagnostic, exactly sum all "
            "active atoms for the frozen outcome-independent 64-object subset; "
            "requires --diagnose-complement-tail-proposal"
        ),
    )
    parser.add_argument(
        "--diagnose-complement-learned-center-checkpoint",
        default=None,
        help=(
            "run the frozen development-only exact learned-R/conditioned-DM3 "
            "centre diagnostic using this contrastive checkpoint"
        ),
    )
    parser.add_argument(
        "--audit-spatial-neyman-partition",
        action="store_true",
        help=(
            "build and audit the outcome-blind frozen spatial-cell partition "
            "under --output, then stop before loading a mock or evaluating "
            "candidate/target likelihoods"
        ),
    )
    parser.add_argument(
        "--spatial-neyman-partition-cache",
        default=None,
        help=(
            "frozen outcome-blind SpatialCellPartition cache required by the "
            "spatial-Neyman complement diagnostic"
        ),
    )
    parser.add_argument(
        "--diagnose-complement-spatial-neyman",
        action="store_true",
        help=(
            "run the frozen development-only spatially stratified discarded-"
            "pilot Neyman centre diagnostic; requires its partition cache and "
            "the frozen exact oracle"
        ),
    )
    parser.add_argument(
        "--spatial-neyman-object-chunk",
        type=int,
        default=8,
        help="observations per frozen spatial-proposal batch",
    )
    parser.add_argument(
        "--spatial-neyman-atom-chunk",
        type=int,
        default=262144,
        help="active atoms per frozen spatial-proposal slab",
    )
    parser.add_argument(
        "--spatial-h256-u8-ps2-partition-cache",
        default=None,
        help=(
            "reviewed outcome-blind SpatialCellPartition cache required by "
            "the frozen H256-U8-PS2 complement diagnostic"
        ),
    )
    parser.add_argument(
        "--diagnose-complement-spatial-h256-u8-ps2",
        action="store_true",
        help=(
            "run the frozen development-only no-pilot H256-U8-PS2 integer-"
            "CDF centre diagnostic; requires its reviewed partition and exact oracle"
        ),
    )
    parser.add_argument(
        "--spatial-h256-u8-ps2-object-chunk",
        type=int,
        default=32,
        help="observations per frozen H256-U8-PS2 integer-CDF batch",
    )
    parser.add_argument(
        "--spatial-h256-u8-ps2-atom-chunk",
        type=int,
        default=262144,
        help="active atoms per frozen four-cell H256-U8-PS2 slab",
    )
    parser.add_argument(
        "--learned-retrieval-object-chunk",
        type=int,
        default=8,
        help="observations per global learned-score matrix",
    )
    parser.add_argument(
        "--learned-retrieval-atom-chunk",
        type=int,
        default=262144,
        help="active atoms encoded per learned-retrieval block",
    )
    parser.add_argument("--atom-chunk", type=int, default=4096)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-point", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # Kept as a rejected-value compatibility option for older launchers.  The
    # current likelihood is deliberately fp32-only.
    parser.add_argument("--precision", choices=("fp32",), help=argparse.SUPPRESS)
    parser.set_defaults(**release_defaults)
    return parser.parse_args(argv)


def _validate_exact_oracle_request(args) -> None:
    """Fail closed before exact-mode row identities can be changed by a cut."""

    if not args.diagnose_complement_exact_oracle:
        return
    if args.diagnose_complement is None:
        raise SystemExit(
            "--diagnose-complement-exact-oracle requires --diagnose-complement"
        )
    if not (
        args.diagnose_complement_tail_proposal
        or args.diagnose_complement_spatial_neyman
        or args.diagnose_complement_spatial_h256_u8_ps2
    ):
        raise SystemExit(
            "--diagnose-complement-exact-oracle requires "
            "--diagnose-complement-tail-proposal or "
            "--diagnose-complement-spatial-neyman or "
            "--diagnose-complement-spatial-h256-u8-ps2"
        )
    if (
        args.cut_abs_ehat is not None
        or bool(args.cut_bound)
        or args.selection_cache is not None
    ):
        raise SystemExit(
            "--diagnose-complement-exact-oracle requires measured cuts and "
            "selection caches to be disabled so frozen absolute row ids are preserved"
        )


def _validate_pilot_stratum_request(args) -> None:
    """Keep the development-only pilot arm behind one fixed boolean."""

    if not args.diagnose_complement_pilot_stratum:
        return
    if args.diagnose_complement is None:
        raise SystemExit(
            "--diagnose-complement-pilot-stratum requires --diagnose-complement"
        )
    if not args.diagnose_complement_tail_proposal:
        raise SystemExit(
            "--diagnose-complement-pilot-stratum requires "
            "--diagnose-complement-tail-proposal"
        )
    if args.diagnose_complement_exact_oracle:
        raise SystemExit(
            "--diagnose-complement-pilot-stratum cannot be combined with "
            "--diagnose-complement-exact-oracle"
        )
    if args.diagnose_complement_spatial_neyman:
        raise SystemExit(
            "--diagnose-complement-pilot-stratum cannot be combined with "
            "--diagnose-complement-spatial-neyman"
        )
    if args.diagnose_complement_spatial_h256_u8_ps2:
        raise SystemExit(
            "--diagnose-complement-pilot-stratum cannot be combined with "
            "--diagnose-complement-spatial-h256-u8-ps2"
        )
    if (
        args.cut_abs_ehat is not None
        or bool(args.cut_bound)
        or args.selection_cache is not None
    ):
        raise SystemExit(
            "--diagnose-complement-pilot-stratum requires measured cuts and "
            "selection caches to be disabled so frozen absolute row ids are preserved"
        )


def _validate_learned_retrieval_center_request(args) -> None:
    """Fail closed around the one frozen learned-R development protocol."""

    checkpoint = args.diagnose_complement_learned_center_checkpoint
    if checkpoint is None:
        return
    expected = LEARNED_RETRIEVAL_CENTER_PROTOCOL
    effective_prefilter = (
        args.proposal_candidates
        if args.proposal_prefilter_candidates is None
        else args.proposal_prefilter_candidates
    )
    observed = {
        "diagnostic_enabled": args.diagnose_complement is not None,
        "tail_proposal": bool(args.diagnose_complement_tail_proposal),
        "exact_oracle": bool(args.diagnose_complement_exact_oracle),
        "pilot_stratum": bool(args.diagnose_complement_pilot_stratum),
        "spatial_neyman": bool(args.diagnose_complement_spatial_neyman),
        "spatial_h256_u8_ps2": bool(
            args.diagnose_complement_spatial_h256_u8_ps2
        ),
        "candidate_backend": args.candidate_backend,
        "universe_start": int(args.observation_start),
        "universe_stop": (
            None if args.observation_stop is None else int(args.observation_stop)
        ),
        "n_candidates": int(args.proposal_candidates),
        "prefilter_candidates": int(effective_prefilter),
        "n_global_draws": int(args.diagnose_complement_global_draws),
        "proposal_seed": int(args.proposal_seed),
        "initial_strategy": args.initial_strategy,
        "diagnostic_object_chunk": int(args.object_chunk),
        "flow_atom_chunk": int(args.atom_chunk),
        "retrieval_object_chunk": int(args.learned_retrieval_object_chunk),
        "retrieval_atom_chunk": int(args.learned_retrieval_atom_chunk),
    }
    required = {
        "diagnostic_enabled": True,
        "tail_proposal": True,
        "exact_oracle": True,
        "pilot_stratum": False,
        "spatial_neyman": False,
        "spatial_h256_u8_ps2": False,
        "candidate_backend": expected["candidate_backend"],
        "universe_start": expected["universe_start"],
        "universe_stop": expected["universe_stop"],
        "n_candidates": expected["n_candidates"],
        "prefilter_candidates": expected["prefilter_candidates"],
        "n_global_draws": expected["n_global_draws"],
        "proposal_seed": expected["proposal_seed"],
        "initial_strategy": expected["initial_strategy"],
        "diagnostic_object_chunk": expected["diagnostic_object_chunk"],
        "flow_atom_chunk": expected["flow_atom_chunk"],
        "retrieval_object_chunk": expected["retrieval_object_chunk"],
        "retrieval_atom_chunk": expected["retrieval_atom_chunk"],
    }
    mismatched = {
        name: {"expected": required[name], "observed": observed[name]}
        for name in required
        if observed[name] != required[name]
    }
    if mismatched:
        raise SystemExit(
            "learned-retrieval centre protocol mismatch: "
            + json.dumps(mismatched, sort_keys=True)
        )
    if (
        args.cut_abs_ehat is not None
        or bool(args.cut_bound)
        or args.selection_cache is not None
    ):
        raise SystemExit(
            "the learned-retrieval centre protocol requires measured cuts and "
            "selection caches to be disabled"
        )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(f"missing learned-retrieval checkpoint: {checkpoint_path}")
    actual_sha256 = _sha256(checkpoint_path)
    if actual_sha256 != LEARNED_RETRIEVAL_CENTER_CHECKPOINT_SHA256:
        raise SystemExit(
            "learned-retrieval checkpoint SHA-256 mismatch: "
            f"{actual_sha256} != {LEARNED_RETRIEVAL_CENTER_CHECKPOINT_SHA256}"
        )
    if args.mock_input is None:
        raise SystemExit(
            "the learned-retrieval centre protocol requires its frozen mock input"
        )
    try:
        actual_mock_sha256 = _mock_input_hashes(args.mock_input)
    except OSError as error:
        raise SystemExit(
            "cannot hash the learned-retrieval centre frozen mock input: "
            f"{error}"
        ) from error
    expected_mock_sha256 = expected["mock_input_sha256"]
    if actual_mock_sha256 != expected_mock_sha256:
        raise SystemExit(
            "learned-retrieval mock-input SHA-256 mismatch: "
            f"{json.dumps(actual_mock_sha256, sort_keys=True)} != "
            f"{json.dumps(expected_mock_sha256, sort_keys=True)}"
        )


def _validate_spatial_neyman_partition_audit_request(args) -> None:
    """Keep the outcome-blind partition build separate from all science rows."""

    if not args.audit_spatial_neyman_partition:
        return
    conflicts = {
        "--prepare-only": bool(args.prepare_only),
        "--diagnose-complement": args.diagnose_complement is not None,
        "--diagnose-complement-tail-proposal": bool(args.diagnose_complement_tail_proposal),
        "--diagnose-complement-pilot-stratum": bool(args.diagnose_complement_pilot_stratum),
        "--diagnose-complement-exact-oracle": bool(args.diagnose_complement_exact_oracle),
        "--diagnose-complement-spatial-neyman": bool(args.diagnose_complement_spatial_neyman),
        "--diagnose-complement-spatial-h256-u8-ps2": bool(
            args.diagnose_complement_spatial_h256_u8_ps2
        ),
        "--diagnose-complement-learned-center-checkpoint": (
            args.diagnose_complement_learned_center_checkpoint is not None
        ),
        "--mock-input": args.mock_input is not None,
        "--spatial-neyman-partition-cache": (args.spatial_neyman_partition_cache is not None),
        "--spatial-h256-u8-ps2-partition-cache": (
            args.spatial_h256_u8_ps2_partition_cache is not None
        ),
    }
    active = sorted(name for name, enabled in conflicts.items() if enabled)
    if active:
        raise SystemExit(
            "--audit-spatial-neyman-partition is outcome-blind and cannot be "
            "combined with " + ", ".join(active)
        )
    if args.candidate_backend != "torch":
        raise SystemExit("--audit-spatial-neyman-partition requires --candidate-backend torch")


def _validate_spatial_neyman_request(args) -> None:
    """Fail closed around the frozen spatial-Neyman centre diagnostic."""

    if not args.diagnose_complement_spatial_neyman:
        return
    expected = SPATIAL_NEYMAN_PROTOCOL
    effective_prefilter = (
        args.proposal_candidates
        if args.proposal_prefilter_candidates is None
        else args.proposal_prefilter_candidates
    )
    observed = {
        "diagnostic_enabled": args.diagnose_complement is not None,
        "tail_proposal": bool(args.diagnose_complement_tail_proposal),
        "exact_oracle": bool(args.diagnose_complement_exact_oracle),
        "pilot_stratum": bool(args.diagnose_complement_pilot_stratum),
        "learned_retrieval": (args.diagnose_complement_learned_center_checkpoint is not None),
        "partition_audit": bool(args.audit_spatial_neyman_partition),
        "spatial_h256_u8_ps2": bool(
            args.diagnose_complement_spatial_h256_u8_ps2
        ),
        "candidate_backend": args.candidate_backend,
        "universe_start": int(args.observation_start),
        "universe_stop": (None if args.observation_stop is None else int(args.observation_stop)),
        "n_candidates": int(args.proposal_candidates),
        "prefilter_candidates": int(effective_prefilter),
        "n_global_draws": int(args.diagnose_complement_global_draws),
        "proposal_seed": int(args.proposal_seed),
        "initial_strategy": args.initial_strategy,
        "diagnostic_object_chunk": int(args.object_chunk),
        "flow_atom_chunk": int(args.atom_chunk),
        "spatial_object_chunk": int(args.spatial_neyman_object_chunk),
        "spatial_atom_chunk": int(args.spatial_neyman_atom_chunk),
    }
    required = {
        "diagnostic_enabled": True,
        "tail_proposal": False,
        "exact_oracle": True,
        "pilot_stratum": False,
        "learned_retrieval": False,
        "partition_audit": False,
        "spatial_h256_u8_ps2": False,
        "candidate_backend": expected["candidate_backend"],
        "universe_start": expected["universe_start"],
        "universe_stop": expected["universe_stop"],
        "n_candidates": expected["n_candidates"],
        "prefilter_candidates": expected["prefilter_candidates"],
        "n_global_draws": expected["n_global_draws"],
        "proposal_seed": expected["proposal_seed"],
        "initial_strategy": expected["initial_strategy"],
        "diagnostic_object_chunk": expected["diagnostic_object_chunk"],
        "flow_atom_chunk": expected["flow_atom_chunk"],
        "spatial_object_chunk": expected["spatial_object_chunk"],
        "spatial_atom_chunk": expected["spatial_atom_chunk"],
    }
    mismatched = {
        name: {"expected": required[name], "observed": observed[name]}
        for name in required
        if observed[name] != required[name]
    }
    if mismatched:
        raise SystemExit("spatial-Neyman centre protocol mismatch: " + json.dumps(mismatched, sort_keys=True))
    if args.cut_abs_ehat is not None or bool(args.cut_bound) or args.selection_cache is not None:
        raise SystemExit(
            "the spatial-Neyman centre protocol requires measured cuts and selection caches to be disabled"
        )
    partition_root = (
        None if args.spatial_neyman_partition_cache is None else Path(args.spatial_neyman_partition_cache)
    )
    if partition_root is None:
        raise SystemExit("--diagnose-complement-spatial-neyman requires --spatial-neyman-partition-cache")
    manifest_path = partition_root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing spatial-Neyman partition manifest: {manifest_path}")
    partition_path = partition_root / SPATIAL_NEYMAN_PARTITION_FILE
    if not partition_path.is_file():
        raise SystemExit(f"missing spatial-Neyman partition cache: {partition_path}")
    frozen_partition_hashes = {
        name: expected.get(name)
        for name in (
            "partition_sha256",
            "partition_cache_sha256",
            "partition_manifest_sha256",
        )
    }
    missing_frozen_hashes = sorted(name for name, value in frozen_partition_hashes.items() if value is None)
    if missing_frozen_hashes:
        raise SystemExit(
            "the spatial-Neyman protocol lacks frozen partition hashes "
            "(" + ", ".join(missing_frozen_hashes) + "); run and review the "
            "outcome-blind audit first"
        )
    actual_manifest_sha256 = _sha256(manifest_path)
    if actual_manifest_sha256 != frozen_partition_hashes["partition_manifest_sha256"]:
        raise SystemExit(
            "spatial-Neyman partition manifest SHA-256 mismatch: "
            f"{actual_manifest_sha256} != "
            f"{frozen_partition_hashes['partition_manifest_sha256']}"
        )
    if args.mock_input is None:
        raise SystemExit("the spatial-Neyman centre protocol requires its frozen mock input")
    try:
        actual_mock_sha256 = _mock_input_hashes(args.mock_input)
    except OSError as error:
        raise SystemExit(f"cannot hash the spatial-Neyman frozen mock input: {error}") from error
    expected_mock_sha256 = expected["mock_input_sha256"]
    if actual_mock_sha256 != expected_mock_sha256:
        raise SystemExit(
            "spatial-Neyman mock-input SHA-256 mismatch: "
            f"{json.dumps(actual_mock_sha256, sort_keys=True)} != "
            f"{json.dumps(expected_mock_sha256, sort_keys=True)}"
        )


def _validate_spatial_h256_u8_ps2_request(args) -> None:
    """Fail closed around the separate frozen H256-U8-PS2 diagnostic."""

    if not args.diagnose_complement_spatial_h256_u8_ps2:
        return
    expected = SPATIAL_H256_U8_PS2_PROTOCOL
    effective_prefilter = (
        args.proposal_candidates
        if args.proposal_prefilter_candidates is None
        else args.proposal_prefilter_candidates
    )
    observed = {
        "diagnostic_enabled": args.diagnose_complement is not None,
        "tail_proposal": bool(args.diagnose_complement_tail_proposal),
        "exact_oracle": bool(args.diagnose_complement_exact_oracle),
        "pilot_stratum": bool(args.diagnose_complement_pilot_stratum),
        "learned_retrieval": (
            args.diagnose_complement_learned_center_checkpoint is not None
        ),
        "partition_audit": bool(args.audit_spatial_neyman_partition),
        "spatial_neyman": bool(args.diagnose_complement_spatial_neyman),
        "old_partition_cache": args.spatial_neyman_partition_cache is not None,
        "candidate_backend": args.candidate_backend,
        "universe_start": int(args.observation_start),
        "universe_stop": (
            None
            if args.observation_stop is None
            else int(args.observation_stop)
        ),
        "n_candidates": int(args.proposal_candidates),
        "prefilter_candidates": int(effective_prefilter),
        "n_global_draws": int(args.diagnose_complement_global_draws),
        "proposal_seed": int(args.proposal_seed),
        "initial_strategy": args.initial_strategy,
        "diagnostic_object_chunk": int(args.object_chunk),
        "flow_atom_chunk": int(args.atom_chunk),
        "spatial_object_chunk": int(
            args.spatial_h256_u8_ps2_object_chunk
        ),
        "spatial_atom_chunk": int(args.spatial_h256_u8_ps2_atom_chunk),
    }
    required = {
        "diagnostic_enabled": True,
        "tail_proposal": False,
        "exact_oracle": True,
        "pilot_stratum": False,
        "learned_retrieval": False,
        "partition_audit": False,
        "spatial_neyman": False,
        "old_partition_cache": False,
        "candidate_backend": expected["candidate_backend"],
        "universe_start": expected["universe_start"],
        "universe_stop": expected["universe_stop"],
        "n_candidates": expected["n_candidates"],
        "prefilter_candidates": expected["prefilter_candidates"],
        "n_global_draws": expected["n_global_draws"],
        "proposal_seed": expected["proposal_seed"],
        "initial_strategy": expected["initial_strategy"],
        "diagnostic_object_chunk": expected["diagnostic_object_chunk"],
        "flow_atom_chunk": expected["flow_atom_chunk"],
        "spatial_object_chunk": expected["spatial_object_chunk"],
        "spatial_atom_chunk": expected["spatial_atom_chunk"],
    }
    mismatched = {
        name: {"expected": required[name], "observed": observed[name]}
        for name in required
        if observed[name] != required[name]
    }
    if mismatched:
        raise SystemExit(
            "H256-U8-PS2 centre protocol mismatch: "
            + json.dumps(mismatched, sort_keys=True)
        )
    if (
        args.cut_abs_ehat is not None
        or bool(args.cut_bound)
        or args.selection_cache is not None
    ):
        raise SystemExit(
            "the H256-U8-PS2 centre protocol requires measured cuts and "
            "selection caches to be disabled"
        )
    partition_root = (
        None
        if args.spatial_h256_u8_ps2_partition_cache is None
        else Path(args.spatial_h256_u8_ps2_partition_cache)
    )
    if partition_root is None:
        raise SystemExit(
            "--diagnose-complement-spatial-h256-u8-ps2 requires "
            "--spatial-h256-u8-ps2-partition-cache"
        )
    manifest_path = partition_root / SPATIAL_NEYMAN_PARTITION_MANIFEST
    partition_path = partition_root / SPATIAL_NEYMAN_PARTITION_FILE
    if not manifest_path.is_file():
        raise SystemExit(
            f"missing H256-U8-PS2 partition manifest: {manifest_path}"
        )
    if not partition_path.is_file():
        raise SystemExit(
            f"missing H256-U8-PS2 partition cache: {partition_path}"
        )
    actual_manifest_sha256 = _sha256(manifest_path)
    if actual_manifest_sha256 != expected["partition_manifest_sha256"]:
        raise SystemExit(
            "H256-U8-PS2 partition manifest SHA-256 mismatch: "
            f"{actual_manifest_sha256} != "
            f"{expected['partition_manifest_sha256']}"
        )
    if args.mock_input is None:
        raise SystemExit("H256-U8-PS2 requires its frozen mock input")
    try:
        actual_mock_sha256 = _mock_input_hashes(args.mock_input)
    except OSError as error:
        raise SystemExit(
            f"cannot hash the H256-U8-PS2 frozen mock input: {error}"
        ) from error
    if actual_mock_sha256 != expected["mock_input_sha256"]:
        raise SystemExit(
            "H256-U8-PS2 mock-input SHA-256 mismatch: "
            f"{json.dumps(actual_mock_sha256, sort_keys=True)} != "
            f"{json.dumps(expected['mock_input_sha256'], sort_keys=True)}"
        )


def main(argv=None):
    args = parse_args(argv)
    inference_config = _load_release_config(
        args.inference_config, kind="inference"
    )
    likelihood_config = _load_release_config(
        args.likelihood_config, kind="likelihood"
    )
    resolved_pipeline_config = _resolved_pipeline_config(args, inference_config)
    pipeline_release = (
        inference_config["release"]
        if resolved_pipeline_config == inference_config
        else "custom"
    )
    pipeline_config_sha256 = _sha256(args.inference_config)
    likelihood_config_sha256 = _sha256(args.likelihood_config)
    pipeline_implementation_hashes = _pipeline_implementation_hashes()
    likelihood_implementation_hashes = _likelihood_implementation_hashes()
    if args.observation_start < 0:
        raise SystemExit("--observation-start must be non-negative")
    _validate_spatial_neyman_partition_audit_request(args)
    _validate_exact_oracle_request(args)
    _validate_pilot_stratum_request(args)
    _validate_learned_retrieval_center_request(args)
    _validate_spatial_neyman_request(args)
    _validate_spatial_h256_u8_ps2_request(args)
    if args.prepare_only and args.mock_input is not None:
        raise SystemExit("--prepare-only generates a new mock and rejects --mock-input")
    if args.selection_cache is not None and not (
        args.cut_abs_ehat is not None or args.cut_bound
    ):
        raise SystemExit("--selection-cache requires a measured cut")
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    conditions = {
        "pixel_size": args.pixel_size,
        "zero_point": args.zero_point,
        "psf_fwhm": args.psf_fwhm,
        "moffat_beta": args.moffat_beta,
        "pixel_rms": args.pixel_rms,
    }
    likelihood_defaults = _likelihood_defaults(likelihood_config)
    geometry_names = tuple(
        name
        for name in likelihood_defaults
        if name not in {"emulator_model", "emulator_metadata"}
    )
    resolved_likelihood_defaults = {
        name: getattr(args, name) for name in geometry_names
    }
    expected_likelihood_defaults = {
        name: likelihood_defaults[name] for name in geometry_names
    }
    if resolved_likelihood_defaults != expected_likelihood_defaults:
        raise RuntimeError(
            "likelihood geometry, observing conditions, and emulator paths must "
            "match the named likelihood config; use a different config to change them"
        )
    if likelihood_config.get("shear_transform") != SHEAR_TRANSFORM:
        raise RuntimeError("likelihood config uses a different shear transform")
    emulator_config = likelihood_config["emulator"]
    _validate_artifact(
        args.measurement_model,
        likelihood_config["measurement_model"],
        name="measurement model",
    )
    _validate_artifact(
        args.emulator_model,
        emulator_config["model"],
        name="response emulator",
    )
    _validate_artifact(
        args.emulator_metadata,
        emulator_config["metadata"],
        name="emulator metadata",
    )
    detection_model_path = _repo_path(emulator_config["detection_model"]["path"])
    _validate_artifact(
        detection_model_path,
        emulator_config["detection_model"],
        name="detection model",
    )
    prior = ScenePrior.load(args.scene_store)
    flow = load_measurement_model(args.measurement_model, device=args.device)
    expected_targets = tuple(likelihood_config["measurement_model"]["target_names"])
    if tuple(flow.target_transform.target_names) != expected_targets:
        raise RuntimeError(
            "measurement-flow target order does not match the likelihood release"
        )
    output_cut = None
    if args.cut_abs_ehat is not None or args.cut_bound:
        try:
            output_cut = OutputCut.from_specs(
                flow.target_transform.target_names,
                abs_shape=args.cut_abs_ehat,
                specs=args.cut_bound,
            )
        except (KeyError, ValueError) as error:
            raise SystemExit(f"invalid measured selection: {error}") from error

    model_hashes = {
        "measurement": _sha256(args.measurement_model),
        "emulator": _sha256(args.emulator_model),
        "emulator_metadata": _sha256(args.emulator_metadata),
    }
    likelihood_component_hashes = {
        **model_hashes,
        "detection": _sha256(detection_model_path),
    }
    scene_hashes = _file_hashes(
        args.scene_store,
        ("manifest.json", "galaxies.parquet", "neighbours.npz"),
    )
    blend_response = None
    blend_cache_hashes = None
    if args.blend_response_cache is not None:
        blend_cache_hashes = _file_hashes(
            args.blend_response_cache,
            ("manifest.json", "r_blend.npy"),
        )
        blend_response = CatalogueBlendResponse.load(args.blend_response_cache)
        expected_blend_metadata = {
            "scene_sha256": scene_hashes,
            "emulator_sha256": {
                "model": model_hashes["emulator"],
                "metadata": model_hashes["emulator_metadata"],
            },
            "conditions": conditions,
        }
        actual_blend_metadata = {
            name: blend_response.metadata.get(name)
            for name in expected_blend_metadata
        }
        if actual_blend_metadata != expected_blend_metadata:
            raise RuntimeError(
                "blend-response cache identity does not match the supplied scene, "
                "emulator, or observing conditions"
            )

    detector = load_emulator(
        ModelPaths(
            flow_checkpoints=(Path(args.measurement_model),),
            emulator_model=Path(args.emulator_model),
            emulator_metadata=Path(args.emulator_metadata),
        ),
        conditions=conditions,
        device=args.device,
    )
    if blend_response is not None:
        expected_pairing = json.loads(
            json.dumps(
                asdict(
                    EmulatorPairingConfig.from_emulator(
                        detector, task="regression"
                    )
                )
            )
        )
        if blend_response.metadata.get("pairing_config") != expected_pairing:
            raise RuntimeError(
                "blend-response cache pairing configuration does not match the "
                "supplied emulator"
            )
        expected_counts = {
            "n_scene_rows": int(len(prior.galaxies)),
            "n_active_atoms": int(np.count_nonzero(prior.weights > 0)),
        }
        actual_counts = {
            name: blend_response.report.get(name) for name in expected_counts
        }
        if actual_counts != expected_counts:
            raise RuntimeError(
                "blend-response cache does not cover the supplied scene/prior support"
            )
    cache_path = Path(args.model_cache)
    cache_created = False
    if (cache_path / "manifest.json").is_file():
        cache = CatalogueModelCache.load(
            cache_path,
            prior=prior,
            blend_response=blend_response,
        )
        for name, expected in (
            ("model_sha256", model_hashes),
            ("scene_sha256", scene_hashes),
        ):
            if cache.metadata.get(name) != expected:
                raise RuntimeError(
                    f"model cache {name} does not match the supplied catalogue/models"
                )
        expected_geometry = (
            conditions,
            float(args.detection_radius_arcsec),
            float(args.flow_neighbour_radius_arcsec),
            (float(args.crowding_near_arcsec), float(args.crowding_far_arcsec)),
            tuple(flow.condition_preprocessor.feature_names),
        )
        actual_geometry = (
            cache.conditions,
            cache.detection_radius_arcsec,
            cache.flow_neighbour_radius_arcsec,
            cache.crowding_radii_arcsec,
            tuple(cache.flow_features or ()),
        )
        if actual_geometry != expected_geometry:
            raise RuntimeError(
                "model cache geometry/features do not match the requested inference setup"
            )
        cache.attach_detector(detector)
    else:
        if cache_path.exists() and any(cache_path.iterdir()):
            raise RuntimeError(
                f"model cache path exists without a manifest and is not empty: {cache_path}"
            )
        cache = CatalogueModelCache(
            prior,
            detector=detector,
            conditions=conditions,
            detection_radius_arcsec=args.detection_radius_arcsec,
            flow_neighbour_radius_arcsec=args.flow_neighbour_radius_arcsec,
            crowding_radii_arcsec=(
                args.crowding_near_arcsec,
                args.crowding_far_arcsec,
            ),
            blend_response=blend_response,
            flow_features=flow.condition_preprocessor.feature_names,
        )
        cache.get(0.0, 0.0)
        cache_created = True
    cache.validate_model_features(flow)
    detection_features = cache.validate_detection_shear_invariance()
    if cache_created:
        cache.save(
            cache_path,
            metadata={
                "purpose": "numerical_catalogue_likelihood_recenter_base",
                "model_sha256": model_hashes,
                "scene_sha256": scene_hashes,
                "flow_features": list(flow.condition_preprocessor.feature_names),
                "selection": None,
                "r_blend": {
                    "enabled": blend_response is not None,
                    "cache_sha256": blend_cache_hashes,
                },
            },
        )
    model_cache_hashes = _model_cache_hashes(cache_path)
    selection = None
    selection_path = None if args.selection_cache is None else Path(args.selection_cache)
    selection_initial_shears = ()
    selection_identity = {
        "scene_sha256": scene_hashes,
        "model_sha256": model_hashes,
        "blend_response_sha256": blend_cache_hashes,
        "conditions": conditions,
        "detection_radius_arcsec": float(args.detection_radius_arcsec),
        "flow_neighbour_radius_arcsec": float(args.flow_neighbour_radius_arcsec),
        "crowding_radii_arcsec": [
            float(args.crowding_near_arcsec),
            float(args.crowding_far_arcsec),
        ],
        "selection_samples": int(args.selection_samples),
        "selection_seed": int(args.selection_seed),
        "selection_row_chunk": int(args.selection_row_chunk),
    }
    if output_cut is not None:
        if selection_path is not None and (selection_path / "manifest.json").is_file():
            selection = CatalogueSelection.load(selection_path, output_cut=output_cut)
            requested_sampling = (
                int(args.selection_samples),
                int(args.selection_seed),
                int(args.selection_row_chunk),
            )
            cached_sampling = (
                selection.n_samples,
                selection.seed,
                selection.row_chunk,
            )
            if (
                selection.metadata != selection_identity
                or cached_sampling != requested_sampling
            ):
                raise RuntimeError(
                    "selection cache identity/configuration does not match the supplied "
                    "catalogue, models, R_blend cache, cut, or sampling settings"
                )
        else:
            selection = CatalogueSelection(
                output_cut,
                n_samples=args.selection_samples,
                seed=args.selection_seed,
                row_chunk=args.selection_row_chunk,
            )
            selection.metadata = selection_identity
        selection_initial_shears = selection.available_shears
    likelihood = CatalogueLikelihood(flow, cache, selection=selection)
    likelihood_mock_identity = {
        "likelihood_release": likelihood_config["release"],
        "likelihood_config_sha256": likelihood_config_sha256,
        "shear_transform": SHEAR_TRANSFORM,
        "target_names": list(likelihood.target_names),
        "model_sha256": model_hashes,
        "model_cache_sha256": model_cache_hashes,
        "scene_sha256": scene_hashes,
        "blend_response_sha256": blend_cache_hashes,
        "conditions": conditions,
        "detection_radius_arcsec": float(args.detection_radius_arcsec),
        "flow_neighbour_radius_arcsec": float(args.flow_neighbour_radius_arcsec),
        "crowding_radii_arcsec": [
            float(args.crowding_near_arcsec),
            float(args.crowding_far_arcsec),
        ],
        "selection_cut_key": (
            None
            if output_cut is None
            else json.loads(json.dumps(output_cut.key()))
        ),
    }
    proposal_path = Path(args.proposal_cache)
    proposal_identity = {
        "model_sha256": model_hashes,
        "scene_sha256": scene_hashes,
        "conditions": conditions,
        "flow_neighbour_radius_arcsec": float(args.flow_neighbour_radius_arcsec),
        "crowding_radii_arcsec": [
            float(args.crowding_near_arcsec),
            float(args.crowding_far_arcsec),
        ],
        "coordinate_config": {
            "n_flow_samples": int(args.proposal_flow_samples),
            "statistic": args.proposal_statistic,
            "dispersion_statistic": args.proposal_dispersion_statistic,
            "seed": int(args.proposal_coordinate_seed),
        },
    }
    if (proposal_path / "manifest.json").is_file():
        coordinates = ProposalCoordinateTable.load(proposal_path)
        for name, expected in proposal_identity.items():
            actual = coordinates.metadata.get(name)
            if name == "coordinate_config" and actual is not None:
                actual = dict(actual)
                actual.setdefault(
                    "dispersion_statistic", coordinates.dispersion_statistic
                )
            if actual != expected:
                raise RuntimeError(
                    f"proposal cache {name} does not match the supplied catalogue/models"
                )
    else:
        if proposal_path.exists() and any(proposal_path.iterdir()):
            raise RuntimeError(
                f"proposal cache path exists without a manifest and is not empty: {proposal_path}"
            )
        coordinates = ProposalCoordinateTable.from_flow(
            likelihood,
            target_names=likelihood.target_names,
            n_flow_samples=args.proposal_flow_samples,
            statistic=args.proposal_statistic,
            dispersion_statistic=args.proposal_dispersion_statistic,
            row_chunk=args.proposal_row_chunk,
            seed=args.proposal_coordinate_seed,
            metadata=proposal_identity,
        )
        coordinates.save(proposal_path)
    proposal_cache_hashes = _file_hashes(
        proposal_path,
        ("manifest.json", "coordinates.npz"),
    )
    missing_proposal_targets = sorted(
        set(coordinates.target_names) - set(likelihood.target_names)
    )
    if missing_proposal_targets:
        raise RuntimeError(
            f"proposal cache uses unknown flow targets: {missing_proposal_targets}"
        )

    # The spatial partition is outcome-blind.  Build or verify it before a mock
    # is loaded, before rows are sliced, and before any candidate query or
    # target-flow evaluation can occur.  The spatial Torch path never needs the
    # 12.76-million-row coordinate KD tree.
    proposal = None
    spatial_partition = None
    spatial_partition_load_seconds = None
    proposal_initialization_seconds = None
    if (
        args.audit_spatial_neyman_partition
        or args.diagnose_complement_spatial_neyman
        or args.diagnose_complement_spatial_h256_u8_ps2
    ):
        proposal_started = time.perf_counter()
        proposal = DefensiveLocalProposal(
            coordinates,
            prior.weights,
            local_base_weights=cache.get(0.0, 0.0).detection_probability,
            build_coordinate_tree=False,
        )
        proposal_initialization_seconds = time.perf_counter() - proposal_started
        if args.audit_spatial_neyman_partition:
            audit = _write_spatial_partition_audit(
                output,
                proposal,
                provenance={
                    "pipeline_release": "partition_audit",
                    "pipeline_base_release": inference_config["release"],
                    "pipeline_config_sha256": pipeline_config_sha256,
                    "pipeline_resolved_config_sha256": _json_sha256(
                        resolved_pipeline_config
                    ),
                    "likelihood_release": likelihood_config["release"],
                    "likelihood_config_sha256": likelihood_config_sha256,
                    "conditions": conditions,
                    "model_sha256": model_hashes,
                    "model_cache_sha256": model_cache_hashes,
                    "scene_sha256": scene_hashes,
                    "proposal_cache_sha256": proposal_cache_hashes,
                    "likelihood_component_sha256": likelihood_component_hashes,
                    "pipeline_implementation_sha256": (
                        pipeline_implementation_hashes
                    ),
                    "proposal_initialization_seconds": float(
                        proposal_initialization_seconds
                    ),
                    "coordinate_tree_built": False,
                },
            )
            print(json.dumps(audit, indent=2))
            return 0
        spatial_partition, spatial_partition_load_seconds = (
            _load_verified_spatial_partition(
                (
                    args.spatial_h256_u8_ps2_partition_cache
                    if args.diagnose_complement_spatial_h256_u8_ps2
                    else args.spatial_neyman_partition_cache
                ),
                proposal,
                protocol=(
                    SPATIAL_H256_U8_PS2_PROTOCOL
                    if args.diagnose_complement_spatial_h256_u8_ps2
                    else SPATIAL_NEYMAN_PROTOCOL
                ),
            )
        )

    generated_mock = args.mock_input is None
    if generated_mock:
        mock = generate_mock_catalogue(
            likelihood,
            n_detected=args.n_detected,
            g1=args.injected_g1,
            g2=args.injected_g2,
            scene_seed=args.scene_seed,
            detection_seed=args.detection_seed,
            flow_seed=args.flow_seed,
        )
    else:
        mock = MockCatalogue.load(args.mock_input)
    if mock.shear_transform != SHEAR_TRANSFORM:
        raise RuntimeError("saved mock does not use the current shape-only shear map")
    injected = mock.truth[["injected_g1", "injected_g2"]].drop_duplicates()
    if len(injected) != 1:
        raise RuntimeError("saved mock contains more than one injected shear")
    injected_shear = injected.iloc[0].to_numpy(float)
    if generated_mock and not np.allclose(
        injected_shear,
        [args.injected_g1, args.injected_g2],
        rtol=0,
        atol=1e-14,
    ):
        raise RuntimeError("generated mock does not retain its requested injected shear")

    if not generated_mock:
        if mock.kind == "image":
            _validate_loaded_mock_manifest(
                args.mock_input,
                mock,
                measurement_model_sha256=model_hashes["measurement"],
                target_names=likelihood.target_names,
            )
        elif mock.kind == "likelihood":
            _validate_loaded_likelihood_manifest(
                args.mock_input,
                mock,
                generation_identity=likelihood_mock_identity,
                implementation_sha256=likelihood_implementation_hashes,
            )
        else:
            raise RuntimeError(f"unsupported saved mock kind {mock.kind!r}")

    source_mock_hashes = (
        None if generated_mock else _mock_input_hashes(args.mock_input)
    )

    selection_report = None
    if output_cut is not None:
        values = mock.measurements.loc[:, likelihood.target_names].to_numpy(float)
        selected = np.asarray(output_cut(values), dtype=bool)
        if selected.shape != (len(mock.measurements),):
            raise RuntimeError("measured selection returned the wrong mock shape")
        n_input = len(mock.measurements)
        n_retained = int(selected.sum())
        if n_retained == 0:
            raise RuntimeError("measured selection retains no mock objects")
        if n_retained != n_input:
            mock = MockCatalogue(
                measurements=mock.measurements.loc[selected].reset_index(drop=True),
                truth=mock.truth.loc[selected].reset_index(drop=True),
            )
        selection_report = {
            "cut_key": output_cut.key(),
            "description": output_cut.describe(),
            "n_input": int(n_input),
            "n_retained": n_retained,
            "keep_fraction": float(n_retained / n_input),
            "n_samples": selection.n_samples,
            "seed": selection.seed,
            "row_chunk": selection.row_chunk,
        }

    if mock.kind == "likelihood":
        required = {"scene_row", "r_blend", "blend_shift_g1", "blend_shift_g2"}
        missing = sorted(required - set(mock.truth))
        if blend_response is not None and missing:
            raise RuntimeError(
                f"saved R_blend likelihood mock lacks truth columns: {missing}"
            )
        if not missing:
            rows = mock.truth["scene_row"].to_numpy(dtype=np.int64)
            recorded = mock.truth["r_blend"].to_numpy(float)
            if blend_response is None:
                if np.any(recorded != 0.0):
                    raise RuntimeError(
                        "saved likelihood mock contains R_blend but inference disabled it"
                    )
            else:
                expected_shift = cache.get(
                    float(injected_shear[0]), float(injected_shear[1])
                ).blend_shift[rows]
                if not np.allclose(
                    recorded,
                    blend_response.values[rows],
                    rtol=0,
                    atol=0,
                ) or not np.allclose(
                    mock.truth[["blend_shift_g1", "blend_shift_g2"]],
                    expected_shift,
                    rtol=0,
                    atol=1e-14,
                ):
                    raise RuntimeError(
                        "saved likelihood mock R_blend truth does not match the supplied cache"
                    )

    initial_raw_mean = None
    if args.initial_strategy == "mean_observed_shape":
        initial, initial_targets = _mean_observed_shape(
            mock.measurements, likelihood.target_names
        )
        initial_raw_mean = initial
    elif args.initial_strategy == "calibrated_mean_observed_shape":
        initial, initial_targets, initial_raw_mean = _calibrated_mean_observed_shape(
            mock.measurements,
            likelihood.target_names,
            offset=args.mean_shape_offset,
            response=args.mean_shape_response,
        )
    else:
        initial = tuple(map(float, args.initial))
        initial_targets = None
    initial_report = {
        "strategy": args.initial_strategy,
        "center": list(initial),
        "raw_mean_shape": (
            None if initial_raw_mean is None else list(initial_raw_mean)
        ),
        "target_names": None if initial_targets is None else list(initial_targets),
        "n_objects": int(len(mock.measurements)),
        "affine_calibration": (
            {
                "offset": list(args.mean_shape_offset),
                "response": [list(row) for row in args.mean_shape_response],
                "role": "optimizer_start_only",
            }
            if args.initial_strategy == "calibrated_mean_observed_shape"
            else None
        ),
    }

    n_observations_total = len(mock.measurements)
    observation_stop = (
        n_observations_total
        if args.observation_stop is None
        else int(args.observation_stop)
    )
    if not (0 <= args.observation_start < observation_stop <= n_observations_total):
        raise RuntimeError(
            "observation partition must satisfy 0 <= start < stop <= mock size"
        )
    observation_partition = {
        "start": int(args.observation_start),
        "stop": observation_stop,
        "n_partition": observation_stop - int(args.observation_start),
        "n_total": int(n_observations_total),
        "proposal_object_id_offset": int(args.observation_start),
    }

    if args.prepare_only:
        if args.observation_start != 0 or observation_stop != n_observations_total:
            raise RuntimeError("--prepare-only requires the complete generated mock")
        mock_root = output / "mock"
        mock.save(mock_root)
        mock_hashes = {
            name: _sha256(mock_root / name)
            for name in ("measurements.parquet", "truth.parquet")
        }
        likelihood_mock_manifest = {
            "mock_kind": "likelihood",
            "generation_identity": likelihood_mock_identity,
            "implementation_sha256": likelihood_implementation_hashes,
            "injected_g1": float(injected_shear[0]),
            "injected_g2": float(injected_shear[1]),
            "scene_seed": int(args.scene_seed),
            "detection_seed": int(args.detection_seed),
            "flow_seed": int(args.flow_seed),
            "output_sha256": mock_hashes,
        }
        (mock_root / "likelihood_mock_manifest.json").write_text(
            json.dumps(likelihood_mock_manifest, indent=2) + "\n"
        )
        payload = {
            "status": "prepared",
            "pipeline_release": pipeline_release,
            "pipeline_base_release": inference_config["release"],
            "pipeline_config_sha256": pipeline_config_sha256,
            "pipeline_resolved_config_sha256": _json_sha256(
                resolved_pipeline_config
            ),
            "likelihood_release": likelihood_config["release"],
            "likelihood_config_sha256": likelihood_config_sha256,
            "pipeline_config": resolved_pipeline_config,
            "config": _result_arguments(args),
            "initial_center": initial_report,
            "observation_partition": observation_partition,
            "injected_shear": injected_shear.tolist(),
            "model_sha256": model_hashes,
            "model_cache_sha256": model_cache_hashes,
            "scene_sha256": scene_hashes,
            "proposal_cache_sha256": proposal_cache_hashes,
            "mock_sha256": mock_hashes,
            "likelihood_component_sha256": likelihood_component_hashes,
            "pipeline_implementation_sha256": pipeline_implementation_hashes,
            "likelihood_implementation_sha256": likelihood_implementation_hashes,
            "implementation_sha256": pipeline_implementation_hashes,
        }
        (output / "preparation.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({
            "status": "prepared",
            "n_observations": n_observations_total,
            "initial_center": initial_report["center"],
            "injected_shear": injected_shear.tolist(),
        }, indent=2))
        return

    if args.observation_start != 0 or observation_stop != n_observations_total:
        rows = slice(int(args.observation_start), observation_stop)
        mock = MockCatalogue(
            mock.measurements.iloc[rows].reset_index(drop=True),
            mock.truth.iloc[rows].reset_index(drop=True),
        )

    # Proposal trees are unnecessary when this invocation only freezes a mock.
    # On the 12.76-million-atom default prior they are material enough to defer
    # until an inference partition actually needs candidate queries.
    if proposal is None:
        proposal_started = time.perf_counter()
        if args.diagnose_complement_learned_center_checkpoint is not None:
            proposal = ContrastiveRetrievalProposal(
                coordinates,
                prior.weights,
                local_base_weights=cache.get(0.0, 0.0).detection_probability,
                likelihood=likelihood,
                checkpoint=args.diagnose_complement_learned_center_checkpoint,
                center=initial,
                contrastive_pool=args.proposal_candidates,
                retrieval_object_chunk=args.learned_retrieval_object_chunk,
                rerank_object_chunk=args.object_chunk,
                atom_chunk=args.learned_retrieval_atom_chunk,
                build_coordinate_tree=False,
            )
        else:
            proposal = DefensiveLocalProposal(
                coordinates,
                prior.weights,
                local_base_weights=cache.get(0.0, 0.0).detection_probability,
            )
        proposal_initialization_seconds = time.perf_counter() - proposal_started
    elif not (
        args.diagnose_complement_spatial_neyman
        or args.diagnose_complement_spatial_h256_u8_ps2
    ):
        raise RuntimeError("prebuilt proposal is reserved for frozen spatial modes")
    elif proposal.tree is not None or spatial_partition is None:
        raise RuntimeError("spatial pre-work did not freeze its tree-free partition")

    if args.compile_flow:
        flow.compile_log_prob(mode=None, dynamic=True)

    if args.diagnose_complement is not None:
        # A measurement, not a release: the estimator never runs, so this
        # writes its own directory and no `result.json` identity block.
        if args.diagnose_complement_spatial_h256_u8_ps2:
            diagnostic = diagnose_spatial_h256_u8_ps2_complement(
                likelihood,
                mock,
                proposal,
                partition=spatial_partition,
                partition_cache_sha256=SPATIAL_H256_U8_PS2_PROTOCOL[
                    "partition_cache_sha256"
                ],
                partition_manifest_sha256=SPATIAL_H256_U8_PS2_PROTOCOL[
                    "partition_manifest_sha256"
                ],
                partition_load_seconds=float(spatial_partition_load_seconds),
                proposal_initialization_seconds=float(
                    proposal_initialization_seconds
                ),
                confirmation_mock_sha256=source_mock_hashes,
                center=initial,
                observation_start=0,
                observation_stop=len(mock.measurements),
                object_id_offset=int(args.observation_start),
                object_chunk=args.object_chunk,
                atom_chunk=args.atom_chunk,
                spatial_atom_chunk=args.spatial_h256_u8_ps2_atom_chunk,
                spatial_object_chunk=args.spatial_h256_u8_ps2_object_chunk,
                candidate_backend=args.candidate_backend,
                n_candidates=args.proposal_candidates,
                prefilter_candidates=(
                    args.proposal_candidates
                    if args.proposal_prefilter_candidates is None
                    else args.proposal_prefilter_candidates
                ),
                proposal_seed=args.proposal_seed,
                production_seed=SPATIAL_H256_U8_PS2_PROTOCOL[
                    "production_seed"
                ],
            )
        elif args.diagnose_complement_spatial_neyman:
            diagnostic = diagnose_spatial_neyman_complement(
                likelihood,
                mock,
                proposal,
                partition=spatial_partition,
                partition_cache_sha256=SPATIAL_NEYMAN_PROTOCOL["partition_cache_sha256"],
                partition_manifest_sha256=SPATIAL_NEYMAN_PROTOCOL["partition_manifest_sha256"],
                partition_load_seconds=float(spatial_partition_load_seconds),
                proposal_initialization_seconds=float(proposal_initialization_seconds),
                confirmation_mock_sha256=source_mock_hashes,
                center=initial,
                # The runner has already sliced the frozen absolute window;
                # preserve absolute object ids for every counter-based stream.
                observation_start=0,
                observation_stop=len(mock.measurements),
                object_id_offset=int(args.observation_start),
                object_chunk=args.object_chunk,
                atom_chunk=args.atom_chunk,
                spatial_atom_chunk=args.spatial_neyman_atom_chunk,
                spatial_object_chunk=args.spatial_neyman_object_chunk,
                candidate_backend=args.candidate_backend,
                n_candidates=args.proposal_candidates,
                prefilter_candidates=(
                    args.proposal_candidates
                    if args.proposal_prefilter_candidates is None
                    else args.proposal_prefilter_candidates
                ),
                proposal_seed=args.proposal_seed,
                pilot_seed=SPATIAL_NEYMAN_PROTOCOL["pilot_seed"],
                production_seed=SPATIAL_NEYMAN_PROTOCOL["production_seed"],
            )
        else:
            diagnostic = diagnose_complement(
                likelihood,
                mock,
                proposal,
                center=initial,
                n_candidates=args.proposal_candidates,
                prefilter_candidates=(
                    args.proposal_candidates
                    if args.proposal_prefilter_candidates is None
                    else args.proposal_prefilter_candidates
                ),
                n_global_draws=args.diagnose_complement_global_draws,
                seed=int(args.proposal_seed) + COMPLEMENT_DIAGNOSTIC_SEED_OFFSET,
                pilot_seed=(
                    int(args.proposal_seed) + PILOT_STRATUM_DIAGNOSTIC_SEED_OFFSET
                    if args.diagnose_complement_pilot_stratum
                    else None
                ),
                # `mock` is already the local partition.  The previous code sliced
                # it and then applied the original absolute bounds a second time,
                # producing an empty/wrong diagnostic for nonzero partitions.
                observation_start=0,
                observation_stop=len(mock.measurements),
                object_id_offset=int(args.observation_start),
                object_chunk=args.object_chunk,
                atom_chunk=args.atom_chunk,
                candidate_backend=args.candidate_backend,
                include_tail_proposal=args.diagnose_complement_tail_proposal,
                include_pilot_stratum=(args.diagnose_complement_pilot_stratum),
                include_exact_oracle=args.diagnose_complement_exact_oracle,
                confirmation_mock_sha256=source_mock_hashes,
                learned_retrieval_center=(
                    {
                        "label": LEARNED_RETRIEVAL_CENTER_PROTOCOL["label"],
                        "initial_strategy": args.initial_strategy,
                        "checkpoint": str(Path(args.diagnose_complement_learned_center_checkpoint).resolve()),
                        "checkpoint_sha256": (LEARNED_RETRIEVAL_CENTER_CHECKPOINT_SHA256),
                        "retrieval_object_chunk": int(args.learned_retrieval_object_chunk),
                        "retrieval_atom_chunk": int(args.learned_retrieval_atom_chunk),
                        "retrieval_matmul_precision": (proposal.retrieval_matmul_precision),
                        "mock_input_sha256": dict(source_mock_hashes),
                        "proposal_initialization_seconds": float(proposal_initialization_seconds),
                    }
                    if args.diagnose_complement_learned_center_checkpoint is not None
                    else None
                ),
            )
        destination = Path(args.diagnose_complement)
        destination.mkdir(parents=True, exist_ok=True)
        table_path = destination / "per_object.parquet"
        diagnostic.table.to_parquet(table_path)
        diagnostic_summary = diagnostic.summary()
        if args.diagnose_complement_learned_center_checkpoint is not None:
            learned_summary = learned_retrieval_center_summary(diagnostic)
            diagnostic_summary["learned_retrieval_center"] = learned_summary
            diagnostic_summary["gates"] = learned_summary["gates"]
        if args.diagnose_complement_spatial_neyman:
            spatial_summary = spatial_neyman_summary(diagnostic)
            diagnostic_summary["spatial_neyman"] = spatial_summary
            diagnostic_summary["gates"] = spatial_summary["gates"]
        if args.diagnose_complement_spatial_h256_u8_ps2:
            spatial_ps_summary = spatial_h256_u8_ps2_center_summary(diagnostic)
            diagnostic_summary["spatial_h256_u8_ps2"] = spatial_ps_summary
            diagnostic_summary["gates"] = spatial_ps_summary["gates"]
        summary = {
            "method": (
                "held-out complement spatial H256-U8-PS2 diagnostic"
                if args.diagnose_complement_spatial_h256_u8_ps2
                else (
                    "held-out complement spatial-Neyman diagnostic"
                    if args.diagnose_complement_spatial_neyman
                    else "held-out complement tail-proposal diagnostic"
                )
            ),
            "metadata": diagnostic.metadata,
            "summary": diagnostic_summary,
            "provenance": {
                "pipeline_release": "diagnostic",
                "pipeline_base_release": inference_config["release"],
                "pipeline_config_sha256": pipeline_config_sha256,
                "pipeline_resolved_config_sha256": _json_sha256(
                    resolved_pipeline_config
                ),
                "likelihood_release": likelihood_config["release"],
                "likelihood_config_sha256": likelihood_config_sha256,
                "pipeline_config": resolved_pipeline_config,
                "config": _result_arguments(args),
                "conditions": conditions,
                "selection": selection_report,
                "initial_center": initial_report,
                "observation_partition": observation_partition,
                "mock_kind": mock.kind,
                "mock_generated": generated_mock,
                "injected_shear": injected_shear.tolist(),
                "model_sha256": model_hashes,
                "model_cache_sha256": model_cache_hashes,
                "scene_sha256": scene_hashes,
                "proposal_cache_sha256": proposal_cache_hashes,
                "mock_input_sha256": source_mock_hashes,
                "spatial_partition_sha256": (
                    {
                        "partition_sha256": (
                            SPATIAL_H256_U8_PS2_PROTOCOL["partition_sha256"]
                            if args.diagnose_complement_spatial_h256_u8_ps2
                            else SPATIAL_NEYMAN_PROTOCOL["partition_sha256"]
                        ),
                        "partition_cache_sha256": (
                            SPATIAL_H256_U8_PS2_PROTOCOL[
                                "partition_cache_sha256"
                            ]
                            if args.diagnose_complement_spatial_h256_u8_ps2
                            else SPATIAL_NEYMAN_PROTOCOL[
                                "partition_cache_sha256"
                            ]
                        ),
                        "partition_manifest_sha256": (
                            SPATIAL_H256_U8_PS2_PROTOCOL[
                                "partition_manifest_sha256"
                            ]
                            if args.diagnose_complement_spatial_h256_u8_ps2
                            else SPATIAL_NEYMAN_PROTOCOL[
                                "partition_manifest_sha256"
                            ]
                        ),
                    }
                    if (
                        args.diagnose_complement_spatial_neyman
                        or args.diagnose_complement_spatial_h256_u8_ps2
                    )
                    else None
                ),
                "likelihood_component_sha256": likelihood_component_hashes,
                "pipeline_implementation_sha256": pipeline_implementation_hashes,
                "likelihood_implementation_sha256": (
                    likelihood_implementation_hashes
                ),
                "diagnostic_implementation_sha256": {
                    "sbsi/complement_diagnostic.py": _sha256(
                        REPOSITORY_ROOT / "sbsi/complement_diagnostic.py"
                    ),
                    **(
                        {
                            "jobs/job_complement_exact_oracle.sh": _sha256(
                                REPOSITORY_ROOT
                                / "jobs/job_complement_exact_oracle.sh"
                            )
                        }
                        if args.diagnose_complement_exact_oracle
                        else {}
                    ),
                    **(
                        {
                            "jobs/job_complement_pilot_stratum.sh": _sha256(
                                REPOSITORY_ROOT
                                / "jobs/job_complement_pilot_stratum.sh"
                            )
                        }
                        if args.diagnose_complement_pilot_stratum
                        else {}
                    ),
                    **(
                        {
                            "sbsi/amortized_proposal.py": _sha256(
                                REPOSITORY_ROOT / "sbsi/amortized_proposal.py"
                            ),
                            "sbsi/learned_retrieval_diagnostic.py": _sha256(
                                REPOSITORY_ROOT
                                / "sbsi/learned_retrieval_diagnostic.py"
                            ),
                            "jobs/job_complement_learned_retrieval_center.sh": (
                                _sha256(
                                    REPOSITORY_ROOT
                                    / "jobs/job_complement_learned_retrieval_center.sh"
                                )
                            ),
                        }
                        if args.diagnose_complement_learned_center_checkpoint
                        is not None
                        else {}
                    ),
                    **(
                        {
                            "sbsi/spatial_neyman_diagnostic.py": _sha256(
                                REPOSITORY_ROOT / "sbsi/spatial_neyman_diagnostic.py"
                            ),
                            "jobs/job_complement_spatial_neyman_center.sh": (
                                _sha256(REPOSITORY_ROOT / "jobs/job_complement_spatial_neyman_center.sh")
                            ),
                        }
                        if args.diagnose_complement_spatial_neyman
                        else {}
                    ),
                    **(
                        {
                            "sbsi/spatial_h256_u8_ps2_diagnostic.py": _sha256(
                                REPOSITORY_ROOT
                                / "sbsi/spatial_h256_u8_ps2_diagnostic.py"
                            ),
                            "jobs/job_complement_spatial_h256_u8_ps2_center.sh": (
                                _sha256(
                                    REPOSITORY_ROOT
                                    / "jobs/job_complement_spatial_h256_u8_ps2_center.sh"
                                )
                            ),
                        }
                        if args.diagnose_complement_spatial_h256_u8_ps2
                        else {}
                    ),
                },
                "output_sha256": {table_path.name: _sha256(table_path)},
            },
        }
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.proposal_method != "initial_center_posterior_adapted":
        raise ValueError(
            "v1.1-infer requires the initial-center posterior proposal"
        )
    result = estimate_one_step_adaptive_section5(
        likelihood,
        mock,
        proposal,
        center=initial,
        h=args.h,
        draw_ladder=args.adaptive_draw_ladder,
        n_candidates=args.proposal_candidates,
        proposal_prefilter_candidates=args.proposal_prefilter_candidates,
        epsilon=args.proposal_epsilon,
        proposal_seed=args.proposal_seed,
        min_ess=args.adaptive_min_ess,
        max_weight_fraction=args.adaptive_max_weight_fraction,
        allocation_method=args.adaptive_allocation,
        pilot_draws=args.adaptive_pilot_draws,
        pilot_seed=args.adaptive_pilot_seed,
        pilot_safety_factor=args.adaptive_pilot_safety_factor,
        bias_correction=args.adaptive_bias_correction,
        candidate_backend=args.candidate_backend,
        estimator_mode=args.estimator_mode,
        retain_full_ladder=args.retain_full_ladder,
        object_id_offset=int(args.observation_start),
        object_chunk=args.object_chunk,
        atom_chunk=args.atom_chunk,
        tail_proposal_atom_chunk=TAIL_PROPOSAL_ATOM_CHUNK,
        tail_proposal_object_chunk=TAIL_PROPOSAL_OBJECT_CHUNK,
    )
    moment_path = output / "one_step_moments.npz"
    np.savez_compressed(
        moment_path,
        score=result.moments.score,
        information=result.moments.information,
        draw_counts=result.moments.draw_counts,
        unique_counts=result.moments.unique_counts,
        **(
            {
                "ladder_score": result.moments.ladder_score,
                "ladder_information": result.moments.ladder_information,
            }
            if result.moments.ladder_score is not None
            else {}
        ),
        # Per-object weight diagnostics, so a heavy tail can be traced back to
        # the objects that carry it rather than only read as a percentile.
        **(
            {
                "weight_diagnostic_draws": np.asarray(
                    result.moments.weight_diagnostic_draws, dtype=np.int64
                ),
                "weight_ess": result.moments.weight_ess,
                "weight_max_fraction": result.moments.weight_max_fraction,
                "weight_relative_error": result.moments.weight_relative_error,
                "weight_pareto_k": result.moments.weight_pareto_k,
            }
            if result.moments.weight_diagnostic_draws is not None
            else {}
        ),
        # Preserve the full per-object/component audit, not only the
        # percentile reduction embedded in result.json.
        **(
            {
                "proposal_component_counts": (
                    result.moments.proposal_component_counts
                ),
                "proposal_component_complement_mass": (
                    result.moments.proposal_component_complement_mass
                ),
            }
            if result.moments.proposal_component_counts is not None
            else {}
        ),
    )
    moment_report = {
        "path": moment_path.name,
        "sha256": _sha256(moment_path),
    }
    result_elapsed = result.moments.elapsed_seconds
    result_reason = "one_full_2d_newton_step"
    result_iterations = 1
    result_evaluations = 9
    if (
        selection is not None
        and selection_path is not None
        and selection.available_shears != selection_initial_shears
    ):
        selection.save(selection_path)
    selection_cache_hashes = (
        None
        if selection_path is None
        else _selection_cache_hashes(selection_path)
    )
    if selection_report is not None:
        probabilities = []
        for g1, g2 in selection.available_shears:
            probability = selection.cached_probability(g1, g2)
            detected_mass = (
                prior.weights
                * cache.get(g1, g2).detection_probability
                * probability
            )
            probabilities.append(
                {
                    "g1": g1,
                    "g2": g2,
                    "prior_mean_p_pass": float(np.sum(prior.weights * probability)),
                    "detected_and_selected_mass": float(detected_mass.sum()),
                }
            )
        selection_report["probabilities"] = probabilities
        selection_report["cache_sha256"] = selection_cache_hashes

    mock_root = output / "mock"
    mock.save(mock_root)
    mock_hashes = {
        name: _sha256(mock_root / name)
        for name in ("measurements.parquet", "truth.parquet")
    }
    if generated_mock:
        likelihood_mock_manifest = {
            "mock_kind": "likelihood",
            "generation_identity": likelihood_mock_identity,
            "implementation_sha256": likelihood_implementation_hashes,
            "injected_g1": float(injected_shear[0]),
            "injected_g2": float(injected_shear[1]),
            "scene_seed": int(args.scene_seed),
            "detection_seed": int(args.detection_seed),
            "flow_seed": int(args.flow_seed),
            "output_sha256": mock_hashes,
        }
        (mock_root / "likelihood_mock_manifest.json").write_text(
            json.dumps(likelihood_mock_manifest, indent=2) + "\n"
        )

    payload = {
        "pipeline_release": pipeline_release,
        "pipeline_base_release": inference_config["release"],
        "pipeline_config_sha256": pipeline_config_sha256,
        "pipeline_resolved_config_sha256": _json_sha256(resolved_pipeline_config),
        "likelihood_release": likelihood_config["release"],
        "likelihood_config_sha256": likelihood_config_sha256,
        "pipeline_config": resolved_pipeline_config,
        "method": "adaptive one-step full-2d catalogue likelihood",
        "config": _result_arguments(args),
        "conditions": conditions,
        "selection": selection_report,
        "initial_center": initial_report,
        "observation_partition": observation_partition,
        "one_step_moments": moment_report,
        "full_ladder": _full_ladder_one_step_report(result.moments),
        "execution": {
            "precision": "fp32",
            "compiled_log_prob": bool(args.compile_flow),
            "compile_dynamic": True if args.compile_flow else None,
        },
        "r_blend": {
            "enabled": blend_response is not None,
            "cache": args.blend_response_cache,
            "cache_sha256": blend_cache_hashes,
            "metadata": None if blend_response is None else blend_response.metadata,
            "report": None if blend_response is None else blend_response.report,
        },
        "detection_features": list(detection_features),
        "mock_kind": mock.kind,
        "mock_generated": generated_mock,
        "injected_shear": injected_shear.tolist(),
        "difference": (np.asarray(result.estimate) - injected_shear).tolist(),
        "result": result.to_dict(),
        "model_sha256": model_hashes,
        "model_cache_sha256": model_cache_hashes,
        "scene_sha256": scene_hashes,
        "proposal_cache_sha256": proposal_cache_hashes,
        "mock_sha256": mock_hashes,
        "mock_input_sha256": source_mock_hashes,
        "likelihood_component_sha256": likelihood_component_hashes,
        "pipeline_implementation_sha256": pipeline_implementation_hashes,
        "likelihood_implementation_sha256": likelihood_implementation_hashes,
        "implementation_sha256": pipeline_implementation_hashes,
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "injected_shear": payload["injected_shear"],
                "estimate": list(result.estimate),
                "difference": payload["difference"],
                "converged": None,
                "reason": result_reason,
                "iterations": result_iterations,
                "likelihood_evaluations": result_evaluations,
                "elapsed_seconds": result_elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
