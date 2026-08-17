#!/usr/bin/env python3
"""Shared helpers for the frozen-V2.2 physical-scene-proxy correction.

The scene proxy is

    Q_d2 = sum_j (F_s,j / F_p) / d_j**2,

where the sum uses every deployed V2.2 neighbour around a labelled primary,
including the unsheared half of the half-shear renderer catalogue.  Only the
labelled/sheared pairs enter the supervised loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.v22_grouped_rscene_common import sha256


PROXY_DEFINITION = "sum_j (F_s,j/F_p) / distance_arcsec_j**2"
PROXY_FEATURE = "log10_sum_flux_s_over_flux_p_over_distance2"
CONDITIONAL_FEATURES = [
    "Re_input_p_scaled",
    "Re_input_s_scaled",
    "r_input_p_scaled",
    "r_input_s_scaled",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance_scaled",
    "v22_pair_prediction",
    "log10_flux_s_over_flux_p",
    "log10_Re_s_over_Re_p",
    PROXY_FEATURE,
    "log1p_full_n_pairs",
]

PSF_FWHM_ARCSEC = 0.73
MOFFAT_BETA = 2.224
PSF_HALF_LIGHT_ARCSEC = (
    np.sqrt(
        (2.0 ** (1.0 / (MOFFAT_BETA - 1.0)) - 1.0)
        / (2.0 ** (1.0 / MOFFAT_BETA) - 1.0)
    )
    / 2.0
    * PSF_FWHM_ARCSEC
)


def load_full_metadata(full_cache: Path, source_cache: Path) -> dict[str, Any]:
    path = full_cache / "metadata.json"
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["source_tag"] != "lsst_r_extnbr_v22":
        raise RuntimeError("full-neighbour cache is not frozen V2.2")
    if metadata["case_window"] != [0, 199]:
        raise RuntimeError("full-neighbour cache does not cover cases 0--199")
    if metadata["source_metadata_sha256"] != sha256(source_cache / "metadata.json"):
        raise RuntimeError("full-neighbour/source-cache provenance mismatch")
    neighbour = metadata["neighbour_definition"]
    if (
        neighbour["population"]
        != "all rendered objects (both half-shear populations)"
        or float(neighbour["r_max_arcsec"]) != 10.0
        or int(neighbour["k"]) != 20
    ):
        raise RuntimeError("full-neighbour population or deployed support drifted")
    return metadata


def full_case_arrays(
    full_cache: Path,
    metadata: dict[str, Any],
    case: int,
) -> dict[str, np.ndarray]:
    arrays = {}
    for name, pattern in metadata["arrays"].items():
        arrays[name] = np.load(
            full_cache / pattern.format(case=case), mmap_mode="r"
        )
    return arrays


def scene_proxy_from_full_arrays(
    x_scaled: np.ndarray,
    x_aux: np.ndarray,
    pair_scene: np.ndarray,
    scene_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Q_d2 and log10(Q_d2) once per full-neighbour scene."""
    scaled = np.asarray(x_scaled, dtype=np.float64)
    aux = np.asarray(x_aux, dtype=np.float64)
    pair_scene = np.asarray(pair_scene, dtype=np.int64)
    count = np.asarray(scene_count, dtype=np.int64)
    if scaled.ndim != 2 or scaled.shape[1] != 7:
        raise ValueError("unexpected full-neighbour x_scaled shape")
    if aux.shape != (len(scaled), 2) or pair_scene.shape != (len(scaled),):
        raise ValueError("full-neighbour auxiliary arrays do not align")
    if int(count.sum()) != len(scaled) or np.any(count <= 0):
        raise RuntimeError("full-neighbour scene counts do not close")
    if len(scaled) and (
        pair_scene.min() < 0 or pair_scene.max() >= len(count)
    ):
        raise RuntimeError("full-neighbour pair-to-scene index is invalid")

    # BlendEMU stores distance_scaled = d / sqrt(Re_p**2 + Re_psf**2)
    # and Re_input_p_scaled = Re_p / sqrt(Re_p**2 + Re_psf**2).
    # Invert those two stored coordinates to recover d in arcsec without
    # rerunning the 90-million-pair neighbour query.
    re_fraction = scaled[:, 0]
    if np.any((re_fraction <= 0.0) | (re_fraction >= 1.0)):
        raise RuntimeError("scaled primary radius cannot be inverted")
    post_psf_radius = PSF_HALF_LIGHT_ARCSEC / np.sqrt(1.0 - re_fraction**2)
    distance_arcsec = scaled[:, 6] * post_psf_radius
    if np.any(~np.isfinite(distance_arcsec)) or np.any(distance_arcsec <= 0.0):
        raise RuntimeError("recovered angular separation is invalid")
    pair_proxy = np.power(10.0, aux[:, 0]) / np.square(distance_arcsec)
    scene_proxy = np.bincount(
        pair_scene, weights=pair_proxy, minlength=len(count)
    ).astype(np.float64)
    if np.any(~np.isfinite(scene_proxy)) or np.any(scene_proxy <= 0.0):
        raise RuntimeError("physical scene proxy is non-finite or non-positive")
    return scene_proxy, np.log10(scene_proxy)


def aligned_full_scene_context(
    source_primary: np.ndarray,
    full_cache: Path,
    full_metadata: dict[str, Any],
    case: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align full-scene log10(Q), Q, and multiplicity to source pair rows."""
    arrays = full_case_arrays(full_cache, full_metadata, case)
    scene_primary = np.asarray(arrays["scene_primary"], dtype=np.int64)
    if np.any(scene_primary[1:] <= scene_primary[:-1]):
        raise RuntimeError(f"case {case}: full-neighbour primaries are not ordered")
    proxy, log_proxy = scene_proxy_from_full_arrays(
        arrays["x_scaled"], arrays["x_aux"], arrays["pair_scene"],
        arrays["scene_count"],
    )
    primary = np.asarray(source_primary, dtype=np.int64)
    mapping = np.searchsorted(scene_primary, primary)
    if np.any(mapping >= len(scene_primary)) or not np.array_equal(
        scene_primary[mapping], primary
    ):
        raise RuntimeError(f"case {case}: source primary lacks full-scene context")
    count = np.asarray(arrays["scene_count"], dtype=np.int16)
    return log_proxy[mapping], proxy[mapping], count[mapping]


def proxy_conditional_matrix(
    x_scaled: np.ndarray,
    x_raw: np.ndarray,
    pair_prediction: np.ndarray,
    scene_log_proxy: np.ndarray,
    full_n_pairs: np.ndarray,
) -> np.ndarray:
    """Build the old 12-feature correction coordinate with P_s replaced by Q."""
    scaled = np.asarray(x_scaled, dtype=np.float32)
    raw = np.asarray(x_raw, dtype=np.float32)
    pair = np.asarray(pair_prediction, dtype=np.float32)
    log_proxy = np.asarray(scene_log_proxy, dtype=np.float32)
    count = np.asarray(full_n_pairs, dtype=np.float32)
    n_rows = len(scaled)
    if scaled.shape != (n_rows, 7) or raw.shape != scaled.shape:
        raise ValueError("unexpected labelled-pair feature shape")
    for value in (pair, log_proxy, count):
        if value.shape != (n_rows,):
            raise ValueError("pair/context feature shape mismatch")
    if np.any(raw[:, :2] <= 0.0) or np.any(count <= 0.0):
        raise RuntimeError("supported pair has non-positive size or multiplicity")
    output = np.empty((n_rows, len(CONDITIONAL_FEATURES)), dtype=np.float32)
    output[:, :7] = scaled
    output[:, 7] = pair
    output[:, 8] = -0.4 * (raw[:, 3] - raw[:, 2])
    output[:, 9] = np.log10(raw[:, 1] / raw[:, 0])
    output[:, 10] = log_proxy
    output[:, 11] = np.log1p(count)
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite proxy-correction feature")
    return output
