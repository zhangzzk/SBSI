#!/usr/bin/env python
"""Run the v1.1-infer one-step full-2D catalogue-likelihood estimator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

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
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable
from sbsi.complement_diagnostic import diagnose_complement
from sbsi.forward_catalogue import EmulatorPairingConfig
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.selection_model import load_selection_model, load_selection_model_ensemble
from sbsi.likelihood_landscape import likelihood_landscape
from sbsi.scene_prior import SHEAR_TRANSFORM, ScenePrior
from sbsi.selection_normalization import (
    detected_selected_mass_shard,
    load_population_normalization,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INFERENCE_CONFIG = REPOSITORY_ROOT / "configs" / "inference.json"
DEFAULT_LIKELIHOOD_CONFIG = REPOSITORY_ROOT / "configs" / "likelihood.json"


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
        raise argparse.ArgumentTypeError("expected four finite comma-separated row-major values")
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
        raise ValueError(f"mean-observed initial centre requires flow targets {names}; missing {missing}")
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
            step = np.linalg.solve(information_sum, score.sum(axis=0, dtype=np.float64))
            estimate = center + step
            residual = score - np.einsum("nij,j->ni", information, step)
            influence = np.linalg.solve(information_sum / len(score), residual.T).T
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
        defaults = {
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
            # The production budget is the deepest rung.  `--draws` is a
            # hidden screen knob; leaving it at a fixed 16,384 happened to
            # match v1.1-infer's ladder top and would silently exceed the
            # ladder of any release that stops shallower.
            "draws": int(estimator["draw_ladder"][-1]),
            "adaptive_min_ess": float(estimator["minimum_ess"]),
            "adaptive_max_weight_fraction": float(estimator["maximum_weight_fraction"]),
            "adaptive_allocation": estimator["allocation"],
            "adaptive_pilot_draws": int(estimator["pilot_draws"]),
            "adaptive_pilot_seed": int(estimator["pilot_seed"]),
            "adaptive_pilot_safety_factor": float(estimator["pilot_safety_factor"]),
            "adaptive_bias_correction": estimator["bias_correction"],
            "retain_full_ladder": bool(estimator["retain_full_ladder"]),
            "compile_flow": bool(execution["compile_flow"]),
            "candidate_backend": execution["candidate_backend"],
            "object_chunk": int(execution["object_chunk"]),
            "atom_chunk": int(execution["atom_chunk"]),
            "precision": execution["precision"],
        }
        # v1.1-infer omits these three and inherits the argparse defaults, so
        # its resolved document stays byte-identical to the file on disk.  A
        # release that names a stratified complement declares them instead,
        # and `_resolved_pipeline_config` echoes them back, so an ordinary run
        # of such a release reports its own name rather than "custom".
        mode = estimator.get("estimator_mode")
        if mode is not None:
            defaults["estimator_mode"] = str(mode)
            if mode in ("tilted_stratified", "priority_stratified"):
                defaults["estimator_tilt_delta"] = float(estimator["tilt_delta"])
                defaults["estimator_tilt_temperature"] = float(estimator["tilt_temperature"])
            elif "tilt_delta" in estimator or "tilt_temperature" in estimator:
                raise ValueError(f"estimator_mode {mode} does not take a complement tilt")
        return defaults
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid inference config: {error}") from error


def _likelihood_defaults(config: dict) -> dict:
    try:
        geometry = config["geometry"]
        conditions = config["observing_conditions"]
        emulator = config["emulator"]
        crowding = geometry["crowding_radii_arcsec"]
        defaults = {
            "emulator_model": str(_repo_path(emulator["model"]["path"])),
            "emulator_metadata": str(_repo_path(emulator["metadata"]["path"])),
            "detection_radius_arcsec": float(geometry["detection_radius_arcsec"]),
            "detection_neighbour_selection": str(geometry.get("detection_neighbour_selection", "nearest")),
            "detection_impact_exponent": float(geometry.get("detection_impact_exponent", 2.0)),
            "flow_neighbour_radius_arcsec": float(geometry["flow_neighbour_radius_arcsec"]),
            "crowding_near_arcsec": float(crowding[0]),
            "crowding_far_arcsec": float(crowding[1]),
            "pixel_size": float(conditions["pixel_size"]),
            "zero_point": float(conditions["zero_point"]),
            "psf_fwhm": float(conditions["psf_fwhm"]),
            "moffat_beta": float(conditions["moffat_beta"]),
            "pixel_rms": float(conditions["pixel_rms"]),
        }
        selection = config.get("measured_selection")
        if isinstance(selection, dict):
            if selection.get("mode") != "output_cut_with_population_normalization":
                raise ValueError("unsupported measured-selection mode")
            bounds = selection.get("bounds")
            if not isinstance(bounds, list) or not all(isinstance(bound, str) and bound for bound in bounds):
                raise ValueError("measured-selection bounds must be nonempty strings")
            defaults["cut_bound"] = list(bounds)
        return defaults
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"invalid likelihood config: {error}") from error


def _validate_artifact(path: str | Path, spec: dict, *, name: str) -> str:
    expected = spec.get("sha256")
    actual = _sha256(path)
    if not isinstance(expected, str) or actual != expected:
        raise RuntimeError(
            f"{name} does not match the likelihood release: expected {expected}, found {actual}"
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
    if args.proposal_candidate_source != "location_prefilter":
        # The released proposal uses a location prefilter followed by the
        # Gaussian reranker.  A direct whole-catalogue proxy shortlist has no
        # prefilter; recording that explicitly makes the run custom rather
        # than silently inheriting the release's 131,072 value.
        resolved["proposal"]["candidate_source"] = args.proposal_candidate_source
        resolved["proposal"]["prefilter_candidates"] = None
    if args.estimator_mode != "mixture":
        # Screens deviate on purpose.  The key is added only when it is not the
        # release default, so an ordinary run still matches the release
        # document exactly and reports v1.1-infer rather than custom.
        resolved["estimator"]["estimator_mode"] = args.estimator_mode
        if args.estimator_mode in ("tilted_stratified", "priority_stratified"):
            resolved["estimator"]["tilt_delta"] = float(args.estimator_tilt_delta)
            resolved["estimator"]["tilt_temperature"] = float(args.estimator_tilt_temperature)
    return resolved


def _result_arguments(args) -> dict:
    """JSON-safe invocation details, excluding the retired naming switch."""

    return {name: value for name, value in vars(args).items() if name != "legacy_inference_version"}


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
        raise RuntimeError("image mock was prepared for a different measurement-flow checkpoint")
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
    manifest_injection = np.asarray([payload.get("injected_g1"), payload.get("injected_g2")], dtype=float)
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
        raise RuntimeError("likelihood mock was generated with different scene/model/selection settings")
    if payload.get("implementation_sha256") != implementation_sha256:
        raise RuntimeError("likelihood mock was generated by a different implementation")
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
        "sbsi/catalogue_null.py",
        "sbsi/catalogue_likelihood.py",
        "sbsi/catalogue_sampling.py",
        "sbsi/selection_normalization.py",
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
    release_parser.add_argument("--inference-config", default=str(DEFAULT_INFERENCE_CONFIG))
    release_parser.add_argument("--likelihood-config", default=str(DEFAULT_LIKELIHOOD_CONFIG))
    release_args, _ = release_parser.parse_known_args(argv)
    try:
        inference_config = _load_release_config(release_args.inference_config, kind="inference")
        likelihood_config = _load_release_config(release_args.likelihood_config, kind="likelihood")
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
    parser.add_argument(
        "--detection-neighbour-selection",
        choices=("nearest", "impact"),
        default="nearest",
    )
    parser.add_argument("--detection-impact-exponent", type=float, default=2.0)
    parser.add_argument("--flow-neighbour-radius-arcsec", type=float, default=7.0)
    parser.add_argument("--crowding-near-arcsec", type=float, default=3.0)
    parser.add_argument("--crowding-far-arcsec", type=float, default=7.0)
    parser.add_argument("--proposal-flow-samples", type=int, default=16)
    parser.add_argument("--proposal-statistic", choices=("mean", "median"), default="median")
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
        "--selection-normalization-cache",
        default=None,
        help=(
            "validated exact-stencil or local-quadratic cache for log B_W(g); "
            "avoids rebuilding the full-catalogue measured-selection integral "
            "inside every observation process"
        ),
    )
    parser.add_argument(
        "--selection-normalization-shard-index",
        type=int,
        default=None,
        help=(
            "prepare only this zero-based atom shard of the exact nine-view "
            "selection normalization; requires --selection-normalization-shards"
        ),
    )
    parser.add_argument(
        "--selection-normalization-shards",
        type=int,
        default=None,
        help="number of GPU atom shards in distributed normalization preparation",
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
        "--object-subset",
        default=None,
        help=(
            "path to a .npy of absolute mock row indices to evaluate.  The "
            "hybrid recompute pass evaluates only the bright objects, which "
            "are not a contiguous range; the ids are carried into the draw "
            "seeds so each object samples exactly the atoms it would sample "
            "inside a full pass.  Mutually exclusive with the observation "
            "range flags."
        ),
    )
    parser.add_argument(
        "--allow-indefinite-partition-summary",
        action="store_true",
        help=(
            "persist additive moments for a deliberately partial object subset "
            "even when that subset is not a standalone positive-definite Newton "
            "problem; the downstream combined solve must still be positive definite"
        ),
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
    parser.add_argument(
        "--proposal-candidate-source",
        choices=("location_prefilter", "whole_catalogue_gaussian_proxy"),
        default="location_prefilter",
        help=(
            "choose the exact shortlist from the released location-prefilter "
            "reranker or rank every active atom directly with the same "
            "diagonal-Gaussian proxy"
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
    parser.add_argument("--adaptive-draw-ladder", nargs="+", type=int, default=(512, 1024, 2048))
    parser.add_argument(
        "--estimator-mode",
        choices=("mixture", "stratified", "tilted_stratified", "priority_stratified"),
        default="mixture",
        help=(
            "'mixture' is the v1.1-infer defensive proposal.  'stratified' "
            "sums the candidate support exactly and spends every draw on its "
            "complement; it requires the production-prefix allocation, a "
            "retained full ladder, and no finite-draw bias correction, and it "
            "marks the run as a custom pipeline.  'tilted_stratified' is "
            "the same split with the complement drawn from the tilted "
            "whole-catalogue proposal instead of the flat prior.  "
            "'priority_stratified' draws that complement without replacement "
            "by priority sampling, which caps the importance ratio by "
            "construction; it needs a single-rung draw ladder because its "
            "threshold depends on the rung."
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
    parser.add_argument("--candidate-backend", choices=("scipy", "torch"), default="scipy")
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
        "--estimator-tilt-delta",
        type=float,
        default=0.1,
        help="defensive prior fraction of the tilted complement proposal",
    )
    parser.add_argument(
        "--estimator-tilt-temperature",
        type=float,
        default=1.0,
        help="flatten the tilted complement proposal; above one broadens it",
    )
    parser.add_argument(
        "--diagnose-complement-tilt-temperature",
        type=float,
        default=1.0,
        help="flatten the tilted component; above one broadens its coverage",
    )
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
        "--diagnose-complement-tilt",
        type=float,
        default=None,
        metavar="DELTA",
        help=(
            "also draw the complement from a whole-catalogue proposal mixing "
            "the diagonal Gaussian proxy with DELTA of the detected prior, and "
            "report its tail index beside the prior-only one"
        ),
    )
    parser.add_argument(
        "--diagnose-complement-exact-top",
        type=int,
        default=0,
        metavar="N",
        help=(
            "also report the tilted complement after the N atoms the "
            "whole-catalogue score ranks highest are moved into the exact "
            "stratum, which measures how much of the remaining tail is carried "
            "by a handful of dominant atoms the candidate shortlist missed"
        ),
    )
    parser.add_argument("--atom-chunk", type=int, default=4096)
    # cont.334: the exact landscape showed a bright galaxy's likelihood rests on
    # 10-40 catalogue atoms out of 12,760,990, against 82,707-338,272 for a faint
    # one.  That is a property of the prior, not of the sampler, and no reported
    # error bar currently carries it.  Splitting the active support into two
    # complementary halves and running the same scene against each measures the
    # catalogue-induced scatter directly: halving the support inflates it by
    # sqrt(2), so the paired difference is a controlled stress test rather than
    # an estimate.  Half 0 and half 1 partition the same permutation, so the two
    # arms are exact complements and share no atom.
    parser.add_argument(
        "--catalogue-half",
        type=int,
        choices=(0, 1),
        default=None,
        help="restrict the prior to one half of its active support (0 or 1)",
    )
    parser.add_argument("--catalogue-half-seed", type=int, default=0)
    # cont.337: precision of the two whole-catalogue inner products that build
    # the tilted proposal.  `q` enters the weight as a ratio rather than a
    # finite difference, so a slightly different `q` used consistently to draw
    # and to weight is a different valid proposal, not a less accurate answer --
    # unlike the h=0.001 stencil, where float32 has no headroom.  A float32 arm
    # is not bit-paired with a float64 one.
    parser.add_argument(
        "--tilt-score-precision",
        choices=("float64", "float32"),
        default="float64",
    )
    # cont.331: the exact whole-catalogue landscape.  `--likelihood-landscape`
    # names an output directory and turns the run into a measurement; the
    # estimator never executes.  Rows are positions in the observation window,
    # given explicitly so the selection policy stays with whoever chose them.
    # The atom chunk is much wider than the estimator's because there is one
    # observation in flight rather than a hundred, so a 12.76-million-atom row
    # would otherwise be walked in three thousand launches.
    parser.add_argument("--likelihood-landscape", default=None)
    parser.add_argument("--likelihood-landscape-rows", nargs="+", type=int, default=())
    parser.add_argument("--likelihood-landscape-head", type=int, default=4096)
    parser.add_argument("--likelihood-landscape-atom-chunk", type=int, default=1 << 20)
    # Shortlist sizes to score against the exact profile: how much of the true
    # mass the estimator's exactly-summed stratum would have captured.
    parser.add_argument("--likelihood-landscape-shortlists", nargs="+", type=int, default=())
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-point", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Kept as a rejected-value compatibility option for older launchers.  The
    # current likelihood is deliberately fp32-only.
    parser.add_argument("--precision", choices=("fp32",), help=argparse.SUPPRESS)
    parser.set_defaults(**release_defaults)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    inference_config = _load_release_config(args.inference_config, kind="inference")
    likelihood_config = _load_release_config(args.likelihood_config, kind="likelihood")
    resolved_pipeline_config = _resolved_pipeline_config(args, inference_config)
    pipeline_release = (
        inference_config["release"] if resolved_pipeline_config == inference_config else "custom"
    )
    pipeline_config_sha256 = _sha256(args.inference_config)
    likelihood_config_sha256 = _sha256(args.likelihood_config)
    pipeline_implementation_hashes = _pipeline_implementation_hashes()
    likelihood_implementation_hashes = _likelihood_implementation_hashes()
    if args.observation_start < 0:
        raise SystemExit("--observation-start must be non-negative")
    if args.prepare_only and args.mock_input is not None:
        raise SystemExit("--prepare-only generates a new mock and rejects --mock-input")
    if args.selection_cache is not None and not (args.cut_abs_ehat is not None or args.cut_bound):
        raise SystemExit("--selection-cache requires a measured cut")
    if args.selection_normalization_cache is not None and not (
        args.cut_abs_ehat is not None or args.cut_bound
    ):
        raise SystemExit("--selection-normalization-cache requires a measured cut")
    normalization_shard_mode = (
        args.selection_normalization_shard_index is not None
        or args.selection_normalization_shards is not None
    )
    if normalization_shard_mode:
        if args.selection_normalization_shard_index is None or args.selection_normalization_shards is None:
            raise SystemExit("distributed normalization requires both shard index and shard count")
        if args.selection_normalization_shards <= 0 or not (
            0 <= args.selection_normalization_shard_index < args.selection_normalization_shards
        ):
            raise SystemExit("distributed normalization shard index/count are invalid")
        if args.selection_normalization_cache is not None:
            raise SystemExit("normalization shard preparation cannot also consume a normalization cache")
        if args.mock_input is None:
            raise SystemExit("normalization shard preparation requires --mock-input")
        if not (args.cut_abs_ehat is not None or args.cut_bound):
            raise SystemExit("normalization shard preparation requires a measured cut")
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
        name for name in likelihood_defaults if name not in {"emulator_model", "emulator_metadata"}
    )
    resolved_likelihood_defaults = {name: getattr(args, name) for name in geometry_names}
    expected_likelihood_defaults = {name: likelihood_defaults[name] for name in geometry_names}
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
    detection_spec = emulator_config["detection_model"]
    detection_backend = detection_spec.get("backend", "blendemu")
    detection_model_paths = []
    if detection_backend == "sbsi_selection_model_ensemble":
        if detection_spec.get("aggregation") != "arithmetic_mean_probability":
            raise RuntimeError("selection-model ensemble aggregation must be arithmetic_mean_probability")
        members = detection_spec.get("members")
        if not isinstance(members, list) or not members:
            raise RuntimeError("selection-model ensemble requires a nonempty members list")
        for index, member in enumerate(members):
            if not isinstance(member, dict) or "path" not in member:
                raise RuntimeError(f"selection-model ensemble member {index} lacks a path")
            path = _repo_path(member["path"])
            _validate_artifact(path, member, name=f"detection model member {index}")
            detection_model_paths.append(path)
        detection_identity = {
            "aggregation": "arithmetic_mean_probability",
            "members": [_sha256(path) for path in detection_model_paths],
        }
    else:
        detection_model_path = _repo_path(detection_spec["path"])
        _validate_artifact(
            detection_model_path,
            detection_spec,
            name="detection model",
        )
        detection_model_paths.append(detection_model_path)
        detection_identity = _sha256(detection_model_path)
    prior = ScenePrior.load(args.scene_store)
    # The blend-response cache is built over the full active support, so a half
    # still lies inside it.  Keep the full count for the cache identity check
    # below -- shrinking the prior must not read as a cache that fails to cover
    # the scene.
    full_active_atoms = int(np.count_nonzero(prior.weights > 0))
    if args.catalogue_half is not None:
        active = np.flatnonzero(prior.weights > 0).astype(np.int64)
        order = np.random.default_rng(args.catalogue_half_seed).permutation(active.size)
        cut = active.size // 2
        keep = active[np.sort(order[:cut] if args.catalogue_half == 0 else order[cut:])]
        halved = np.zeros_like(prior.weights)
        halved[keep] = prior.weights[keep]
        prior = prior.reweighted(
            halved,
            metadata={
                "catalogue_half": int(args.catalogue_half),
                "catalogue_half_seed": int(args.catalogue_half_seed),
                "catalogue_half_atoms": int(keep.size),
                "catalogue_full_atoms": full_active_atoms,
            },
        )
    flow = load_measurement_model(args.measurement_model, device=args.device)
    expected_targets = tuple(likelihood_config["measurement_model"]["target_names"])
    if tuple(flow.target_transform.target_names) != expected_targets:
        raise RuntimeError("measurement-flow target order does not match the likelihood release")
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
        "detection": detection_identity,
    }
    scene_hashes = _file_hashes(
        args.scene_store,
        ("manifest.json", "galaxies.parquet", "neighbours.npz"),
    )
    blend_response = None
    blend_cache_hashes = None
    if (
        likelihood_config.get("blend_response") == "required_fixed_atom_cache"
        and args.blend_response_cache is None
    ):
        raise RuntimeError("this likelihood candidate requires --blend-response-cache")
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
        actual_blend_metadata = {name: blend_response.metadata.get(name) for name in expected_blend_metadata}
        if actual_blend_metadata != expected_blend_metadata:
            raise RuntimeError(
                "blend-response cache identity does not match the supplied scene, "
                "emulator, or observing conditions"
            )

    emulator = load_emulator(
        ModelPaths(
            flow_checkpoints=(Path(args.measurement_model),),
            emulator_model=Path(args.emulator_model),
            emulator_metadata=Path(args.emulator_metadata),
        ),
        conditions=conditions,
        device=args.device,
    )
    cache_model_hashes = model_hashes if detection_backend == "blendemu" else likelihood_component_hashes
    if detection_backend == "blendemu":
        detector = emulator
    elif detection_backend == "sbsi_selection_model":
        detector = load_selection_model(detection_model_paths[0], device=args.device)
    elif detection_backend == "sbsi_selection_model_ensemble":
        detector = load_selection_model_ensemble(detection_model_paths, device=args.device)
    else:
        raise RuntimeError(f"unsupported detection-model backend {detection_backend!r}")
    if blend_response is not None:
        expected_pairing = json.loads(
            json.dumps(asdict(EmulatorPairingConfig.from_emulator(emulator, task="regression")))
        )
        if blend_response.metadata.get("pairing_config") != expected_pairing:
            raise RuntimeError(
                "blend-response cache pairing configuration does not match the supplied emulator"
            )
        expected_counts = {
            "n_scene_rows": int(len(prior.galaxies)),
            "n_active_atoms": full_active_atoms,
        }
        actual_counts = {name: blend_response.report.get(name) for name in expected_counts}
        if actual_counts != expected_counts:
            raise RuntimeError("blend-response cache does not cover the supplied scene/prior support")
    cache_path = Path(args.model_cache)
    cache_created = False
    if (cache_path / "manifest.json").is_file():
        cache = CatalogueModelCache.load(
            cache_path,
            prior=prior,
            blend_response=blend_response,
        )
        for name, expected in (
            ("model_sha256", cache_model_hashes),
            ("scene_sha256", scene_hashes),
        ):
            if cache.metadata.get(name) != expected:
                raise RuntimeError(f"model cache {name} does not match the supplied catalogue/models")
        expected_geometry = (
            conditions,
            float(args.detection_radius_arcsec),
            args.detection_neighbour_selection,
            float(args.detection_impact_exponent),
            float(args.flow_neighbour_radius_arcsec),
            (float(args.crowding_near_arcsec), float(args.crowding_far_arcsec)),
            tuple(flow.condition_preprocessor.feature_names),
        )
        actual_geometry = (
            cache.conditions,
            cache.detection_radius_arcsec,
            cache.detection_neighbour_selection,
            cache.detection_impact_exponent,
            cache.flow_neighbour_radius_arcsec,
            cache.crowding_radii_arcsec,
            tuple(cache.flow_features or ()),
        )
        if actual_geometry != expected_geometry:
            raise RuntimeError("model cache geometry/features do not match the requested inference setup")
        cache.attach_detector(detector)
    else:
        if cache_path.exists() and any(cache_path.iterdir()):
            raise RuntimeError(f"model cache path exists without a manifest and is not empty: {cache_path}")
        cache = CatalogueModelCache(
            prior,
            detector=detector,
            conditions=conditions,
            detection_radius_arcsec=args.detection_radius_arcsec,
            detection_neighbour_selection=args.detection_neighbour_selection,
            detection_impact_exponent=args.detection_impact_exponent,
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
    detection_features = tuple(cache.detection_features)
    if detection_backend == "blendemu":
        cache.validate_detection_shear_invariance()
    if cache_created:
        cache.save(
            cache_path,
            metadata={
                "purpose": "numerical_catalogue_likelihood_recenter_base",
                "model_sha256": cache_model_hashes,
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
        "detection_neighbour_selection": args.detection_neighbour_selection,
        "detection_impact_exponent": float(args.detection_impact_exponent),
        "flow_neighbour_radius_arcsec": float(args.flow_neighbour_radius_arcsec),
        "crowding_radii_arcsec": [
            float(args.crowding_near_arcsec),
            float(args.crowding_far_arcsec),
        ],
        "selection_samples": int(args.selection_samples),
        "selection_seed": int(args.selection_seed),
        "selection_row_chunk": int(args.selection_row_chunk),
    }
    if detection_backend != "blendemu":
        selection_identity["detection_model"] = {
            "backend": detection_backend,
            "sha256": likelihood_component_hashes["detection"],
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
            if selection.metadata != selection_identity or cached_sampling != requested_sampling:
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
    population_normalization = None
    population_normalization_hash = None
    if args.selection_normalization_cache is not None:
        normalization_path = Path(args.selection_normalization_cache)
        population_normalization = load_population_normalization(normalization_path)
        normalization_identity = {
            **selection_identity,
            "cut_key": output_cut.key(),
        }
        if dict(population_normalization.identity) != normalization_identity:
            raise RuntimeError(
                "selection normalization cache identity does not match the supplied "
                "catalogue, models, R_blend cache, cut, or sampling settings"
            )
        if not np.isclose(
            population_normalization.finite_difference_step,
            float(args.h),
            rtol=0,
            atol=1e-15,
        ):
            raise RuntimeError("selection normalization cache finite-difference step does not match --h")
        population_normalization_hash = _sha256(normalization_path)
    likelihood = CatalogueLikelihood(flow, cache, selection=selection)
    likelihood.population_normalization = population_normalization
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
        "detection_neighbour_selection": args.detection_neighbour_selection,
        "detection_impact_exponent": float(args.detection_impact_exponent),
        "flow_neighbour_radius_arcsec": float(args.flow_neighbour_radius_arcsec),
        "crowding_radii_arcsec": [
            float(args.crowding_near_arcsec),
            float(args.crowding_far_arcsec),
        ],
        "selection_cut_key": (None if output_cut is None else json.loads(json.dumps(output_cut.key()))),
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
                actual.setdefault("dispersion_statistic", coordinates.dispersion_statistic)
            if actual != expected:
                raise RuntimeError(f"proposal cache {name} does not match the supplied catalogue/models")
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
    missing_proposal_targets = sorted(set(coordinates.target_names) - set(likelihood.target_names))
    if missing_proposal_targets:
        raise RuntimeError(f"proposal cache uses unknown flow targets: {missing_proposal_targets}")
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
            raise RuntimeError(f"saved R_blend likelihood mock lacks truth columns: {missing}")
        if not missing:
            rows = mock.truth["scene_row"].to_numpy(dtype=np.int64)
            recorded = mock.truth["r_blend"].to_numpy(float)
            if blend_response is None:
                if np.any(recorded != 0.0):
                    raise RuntimeError("saved likelihood mock contains R_blend but inference disabled it")
            else:
                expected_shift = cache.get(float(injected_shear[0]), float(injected_shear[1])).blend_shift[
                    rows
                ]
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
        initial, initial_targets = _mean_observed_shape(mock.measurements, likelihood.target_names)
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
        "raw_mean_shape": (None if initial_raw_mean is None else list(initial_raw_mean)),
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
    observation_stop = n_observations_total if args.observation_stop is None else int(args.observation_stop)
    if not (0 <= args.observation_start < observation_stop <= n_observations_total):
        raise RuntimeError("observation partition must satisfy 0 <= start < stop <= mock size")
    subset_rows = None
    if args.object_subset is not None:
        if args.observation_start != 0 or args.observation_stop is not None:
            raise RuntimeError(
                "--object-subset selects the rows itself; do not also pass "
                "--observation-start/--observation-stop"
            )
        subset_rows = np.asarray(np.load(args.object_subset), dtype=np.int64)
        if subset_rows.ndim != 1 or subset_rows.size == 0:
            raise RuntimeError("--object-subset must be a non-empty 1-D index array")
        if np.unique(subset_rows).size != subset_rows.size:
            raise RuntimeError("--object-subset must not repeat a row")
        if subset_rows.min() < 0 or subset_rows.max() >= n_observations_total:
            raise RuntimeError("--object-subset row out of range for this mock")
    elif args.allow_indefinite_partition_summary:
        raise RuntimeError("--allow-indefinite-partition-summary is restricted to --object-subset workers")
    observation_partition = {
        "start": int(args.observation_start),
        "stop": observation_stop,
        "n_partition": (
            int(subset_rows.size)
            if subset_rows is not None
            else observation_stop - int(args.observation_start)
        ),
        "n_total": int(n_observations_total),
        "proposal_object_id_offset": int(args.observation_start),
        "object_subset": (None if subset_rows is None else str(Path(args.object_subset).resolve())),
        "object_subset_sha256": (None if subset_rows is None else _sha256(args.object_subset)),
    }

    if normalization_shard_mode:
        if selection is None:
            raise RuntimeError("normalization shard preparation requires measured selection")
        active = np.flatnonzero(prior.weights > 0).astype(np.int64)
        n_active = int(active.size)
        n_chunks = (n_active + selection.row_chunk - 1) // selection.row_chunk
        shard_index = int(args.selection_normalization_shard_index)
        n_shards = int(args.selection_normalization_shards)
        first_chunk = shard_index * n_chunks // n_shards
        last_chunk = (shard_index + 1) * n_chunks // n_shards
        active_start = min(first_chunk * selection.row_chunk, n_active)
        active_stop = min(last_chunk * selection.row_chunk, n_active)
        shard_active = active[active_start:active_stop]
        if shard_active.size == 0:
            raise RuntimeError(
                f"normalization shard {shard_index} has no active atoms; "
                "request no more shards than row chunks"
            )
        views = {
            "zero": (float(initial[0]), float(initial[1])),
            "g1_plus": (float(initial[0] + args.h), float(initial[1])),
            "g1_minus": (float(initial[0] - args.h), float(initial[1])),
            "g2_plus": (float(initial[0]), float(initial[1] + args.h)),
            "g2_minus": (float(initial[0]), float(initial[1] - args.h)),
            "pp": (float(initial[0] + args.h), float(initial[1] + args.h)),
            "pm": (float(initial[0] + args.h), float(initial[1] - args.h)),
            "mp": (float(initial[0] - args.h), float(initial[1] + args.h)),
            "mm": (float(initial[0] - args.h), float(initial[1] - args.h)),
        }
        protected = set(cache.available_shears)
        masses = []
        for name, (g1, g2) in views.items():
            view = cache.get(g1, g2)
            mass = detected_selected_mass_shard(
                selection,
                flow,
                view,
                prior.weights,
                active_indices=shard_active,
                random_offset_rows=active_start,
            )
            masses.append(
                {
                    "name": name,
                    "g1": g1,
                    "g2": g2,
                    "partial_detected_and_selected_mass": mass,
                }
            )
            if cache._key(g1, g2) not in protected:
                cache.discard_views(((g1, g2),))
        payload = {
            "version": 1,
            "method": "exact_detected_selected_mass_atom_shard",
            "center": list(initial),
            "finite_difference_step": float(args.h),
            "identity": {**selection_identity, "cut_key": output_cut.key()},
            "partition": {
                "index": shard_index,
                "count": n_shards,
                "active_start": active_start,
                "active_stop": active_stop,
                "n_active_total": n_active,
                "n_atoms_total": int(len(prior.weights)),
                "row_chunk": int(selection.row_chunk),
            },
            "points": masses,
        }
        destination = output / "selection_normalization_shard.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return

    if args.prepare_only:
        if args.observation_start != 0 or observation_stop != n_observations_total:
            raise RuntimeError("--prepare-only requires the complete generated mock")
        mock_root = output / "mock"
        mock.save(mock_root)
        mock_hashes = {name: _sha256(mock_root / name) for name in ("measurements.parquet", "truth.parquet")}
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
            "pipeline_resolved_config_sha256": _json_sha256(resolved_pipeline_config),
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
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "n_observations": n_observations_total,
                    "initial_center": initial_report["center"],
                    "injected_shear": injected_shear.tolist(),
                },
                indent=2,
            )
        )
        return

    if subset_rows is not None:
        mock = MockCatalogue(
            mock.measurements.iloc[subset_rows].reset_index(drop=True),
            mock.truth.iloc[subset_rows].reset_index(drop=True),
        )
    elif args.observation_start != 0 or observation_stop != n_observations_total:
        rows = slice(int(args.observation_start), observation_stop)
        mock = MockCatalogue(
            mock.measurements.iloc[rows].reset_index(drop=True),
            mock.truth.iloc[rows].reset_index(drop=True),
        )

    # Proposal trees are unnecessary when this invocation only freezes a mock.
    # On the 12.76-million-atom default prior they are material enough to defer
    # until an inference partition actually needs candidate queries.
    proposal = DefensiveLocalProposal(
        coordinates,
        prior.weights,
        local_base_weights=cache.get(0.0, 0.0).detection_probability,
        score_dtype=(torch.float32 if args.tilt_score_precision == "float32" else torch.float64),
    )

    if args.compile_flow:
        flow.compile_log_prob(mode=None, dynamic=True)

    if args.likelihood_landscape is not None:
        if not len(args.likelihood_landscape_rows):
            raise SystemExit(
                "--likelihood-landscape needs --likelihood-landscape-rows; "
                "scoring every atom for every observation is not affordable"
            )
        # A measurement, not a release.  cont.331: every convergence number so
        # far is a property of `c/q`, which cannot say whether the bright
        # objects are hard to sample or simply have a target that no sampler
        # integrates well.  Scoring every active atom exactly removes the
        # proposal from the question.
        landscape = likelihood_landscape(
            likelihood,
            mock,
            proposal,
            center=initial,
            rows=args.likelihood_landscape_rows,
            observation_start=args.observation_start,
            head=args.likelihood_landscape_head,
            atom_chunk=args.likelihood_landscape_atom_chunk,
            coordinate_columns=proposal.coordinates.target_names,
            shortlist_sizes=args.likelihood_landscape_shortlists,
            prefilter_candidates=(
                args.proposal_candidates
                if args.proposal_prefilter_candidates is None
                else args.proposal_prefilter_candidates
            ),
            candidate_backend=args.candidate_backend,
        )
        destination = Path(args.likelihood_landscape)
        destination.mkdir(parents=True, exist_ok=True)
        landscape.table.to_parquet(destination / "per_object.parquet")
        landscape.observed.to_parquet(destination / "observed.parquet")
        np.savez_compressed(
            destination / "mass_profile.npz",
            head_rank=landscape.head_rank,
            head_share=landscape.head_share,
            head_atom=landscape.head_atom,
            head_coordinates=landscape.head_coordinates,
        )
        summary = {"metadata": landscape.metadata, "summary": landscape.summary()}
        (destination / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    if args.diagnose_complement is not None:
        # A measurement, not a release: the estimator never runs, so this
        # writes its own directory and no `result.json` identity block.
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
            seed=int(args.proposal_seed) + 7_000_003,
            observation_start=args.observation_start,
            observation_stop=args.observation_stop,
            object_chunk=args.object_chunk,
            atom_chunk=args.atom_chunk,
            candidate_backend=args.candidate_backend,
            tilt_delta=args.diagnose_complement_tilt,
            tilt_temperature=args.diagnose_complement_tilt_temperature,
            exact_top=args.diagnose_complement_exact_top,
        )
        destination = Path(args.diagnose_complement)
        destination.mkdir(parents=True, exist_ok=True)
        diagnostic.table.to_parquet(destination / "per_object.parquet")
        summary = {"metadata": diagnostic.metadata, "summary": diagnostic.summary()}
        (destination / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    if args.proposal_method != "initial_center_posterior_adapted":
        raise ValueError("v1.1-infer requires the initial-center posterior proposal")
    result = estimate_one_step_adaptive_section5(
        likelihood,
        mock,
        proposal,
        center=initial,
        require_positive_definite=not args.allow_indefinite_partition_summary,
        h=args.h,
        draw_ladder=args.adaptive_draw_ladder,
        n_candidates=args.proposal_candidates,
        proposal_prefilter_candidates=(
            None
            if args.proposal_candidate_source == "whole_catalogue_gaussian_proxy"
            else args.proposal_prefilter_candidates
        ),
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
        candidate_source=args.proposal_candidate_source,
        estimator_mode=args.estimator_mode,
        tilt_delta=args.estimator_tilt_delta,
        tilt_temperature=args.estimator_tilt_temperature,
        retain_full_ladder=args.retain_full_ladder,
        object_id_offset=0 if subset_rows is not None else int(args.observation_start),
        object_ids=subset_rows,
        object_chunk=args.object_chunk,
        atom_chunk=args.atom_chunk,
        progress=lambda completed, total, elapsed: print(
            json.dumps({"event": "inference_progress", "completed": completed,
                        "total": total, "elapsed_seconds": elapsed}), flush=True
        ),
    )
    moment_path = output / "one_step_moments.npz"
    np.savez_compressed(
        moment_path,
        score=result.moments.score,
        information=result.moments.information,
        draw_counts=result.moments.draw_counts,
        unique_counts=result.moments.unique_counts,
        # Absolute mock rows, so a partition or a bright-only subset can be
        # aligned against another pass without reconstructing its selection.
        object_rows=(
            subset_rows
            if subset_rows is not None
            else np.arange(int(args.observation_start), observation_stop, dtype=np.int64)
        ),
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
                "weight_diagnostic_draws": np.asarray(result.moments.weight_diagnostic_draws, dtype=np.int64),
                "weight_ess": result.moments.weight_ess,
                "weight_max_fraction": result.moments.weight_max_fraction,
                "weight_relative_error": result.moments.weight_relative_error,
                "weight_pareto_k": result.moments.weight_pareto_k,
            }
            if result.moments.weight_diagnostic_draws is not None
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
    selection_cache_hashes = None if selection_path is None else _selection_cache_hashes(selection_path)
    if selection_report is not None:
        probabilities = []
        for g1, g2 in selection.available_shears:
            probability = selection.cached_probability(g1, g2)
            detected_mass = prior.weights * cache.get(g1, g2).detection_probability * probability
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
        normalization_report = None
        if population_normalization is not None:
            normalization_report = {
                "path": str(Path(args.selection_normalization_cache).resolve()),
                "sha256": population_normalization_hash,
                "source": dict(population_normalization.source),
            }
            if hasattr(population_normalization, "available_shears"):
                normalization_report.update(
                    {
                        "method": "exact_distributed_detected_selected_mass",
                        "available_shears": [
                            list(point) for point in population_normalization.available_shears
                        ],
                    }
                )
            else:
                normalization_report.update(
                    {
                        "method": "local_quadratic_log_detected_selected_mass",
                        "center": population_normalization.center.tolist(),
                        "trust_box": {
                            "minimum": population_normalization.trust_min.tolist(),
                            "maximum": population_normalization.trust_max.tolist(),
                        },
                        "validation": dict(population_normalization.validation),
                    }
                )
        selection_report["normalization_cache"] = normalization_report
        selection_report["normalization_surrogate"] = (
            normalization_report
            if normalization_report is not None
            and normalization_report["method"] == "local_quadratic_log_detected_selected_mass"
            else None
        )

    source_mock_hashes = None
    if not generated_mock:
        source_root = Path(args.mock_input)
        source_mock_hashes = {
            name: _sha256(source_root / name) for name in ("measurements.parquet", "truth.parquet")
        }
        source_image_manifest = source_root / "image_mock_manifest.json"
        if source_image_manifest.is_file():
            source_mock_hashes[source_image_manifest.name] = _sha256(source_image_manifest)
        source_likelihood_manifest = source_root / "likelihood_mock_manifest.json"
        if source_likelihood_manifest.is_file():
            source_mock_hashes[source_likelihood_manifest.name] = _sha256(source_likelihood_manifest)
    mock_root = output / "mock"
    mock.save(mock_root)
    mock_hashes = {name: _sha256(mock_root / name) for name in ("measurements.parquet", "truth.parquet")}
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
