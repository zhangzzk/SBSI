"""Shape-response evaluation for the SBSI measurement flow.

This is the load-bearing response library (it used to live at
``scripts/response_ratio_diagnostic.py``, a "diagnostic" name that undersold it).
It provides the induced first-moment projection used to form the flow self-response
``R_flow`` that enters the certified, parameter-free decomposition

    m = R_sim / (R_flow + R_blend) - 1.

Public API
----------
load_sheared_sample(...)   Stream + select + reservoir-sample a sheared catalogue.
model_mean_proj(...)       ``< E[e_hat | S_{s*ghat}(intrinsic)] . ghat >`` -- the induced
                           first moment projected on the per-object applied-shear direction.
flow_response(...)         The antithetic +/-g secant ``R_flow = (m_+g - m_-g)/(2g)``,
                           optionally per-object, with optional Common-Random-Numbers reseeding.

See ``SBI_shear_response.md`` for the response-aware (Sobolev) derivation this supervises.
``scripts/response_ratio_diagnostic.py`` remains as a thin back-compat shim + CLI that
re-exports these names, so ``from scripts.response_ratio_diagnostic import ...`` still works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from .measurement_model import raw_columns_for_measurement_targets
from .preprocessing import (
    DEFAULT_SELECTION_CUTS,
    raw_columns_for_selection_features,
    rescale,
    source_select_selection,
)
from .shear_map import apply_shear_to_ellipticity


def load_sheared_sample(catalogue, bundle, max_rows, shear_threshold, seed, max_read_batches=None,
                        snr_min=None):
    """Stream a sheared catalogue, apply the standard cuts + detected, and reservoir-
    sample.  Keep intrinsic shape, applied-shear truth, and the measured targets.
    snr_min applies a measured-quality cut (measured_flux_auto/fluxerr_auto > snr_min)
    on the SHEARED measured quantity -- a shear-dependent selection."""
    rng = np.random.default_rng(seed)
    condition_features = bundle.condition_preprocessor.feature_names
    target_features = bundle.target_transform.target_names

    with ipc.open_file(catalogue) as reader:
        available = set(reader.schema.names)
        needed = set()
        needed |= raw_columns_for_selection_features(condition_features, available_columns=available)
        needed |= raw_columns_for_measurement_targets(target_features)
        needed |= {"detected", "gamma1_input_p", "gamma2_input_p"}
        needed |= {"e1_input_rot0_p", "e2_input_rot0_p"}
        needed |= {"r_input_p", "Re_input_p", "distance", "neighbored"}
        needed |= {"measured_flux_auto", "measured_fluxerr_auto", "measured_mag_auto"}
        read_columns = sorted(c for c in needed if c in available)

        reservoir = None
        raw_rows = 0
        for bi in range(reader.num_record_batches):
            if max_read_batches is not None and bi >= max_read_batches:
                break
            batch = pa.Table.from_batches([reader.get_batch(bi)]).select(read_columns).to_pandas()
            raw_rows += len(batch)
            batch = source_select_selection(batch, cuts=DEFAULT_SELECTION_CUTS)
            if len(batch) == 0:
                continue
            batch = batch[batch["detected"].astype(bool)].reset_index(drop=True)
            if len(batch) == 0:
                continue
            if snr_min is not None and "measured_flux_auto" in batch.columns:
                snr = batch["measured_flux_auto"].to_numpy(float) / batch["measured_fluxerr_auto"].to_numpy(float)
                batch = batch[np.isfinite(snr) & (snr > snr_min)].reset_index(drop=True)
                if len(batch) == 0:
                    continue
            gmag = np.hypot(batch["gamma1_input_p"].to_numpy(float), batch["gamma2_input_p"].to_numpy(float))
            batch = batch[gmag > shear_threshold].reset_index(drop=True)
            if len(batch) == 0:
                continue
            batch = batch.copy()
            batch["__key"] = rng.random(len(batch))
            reservoir = batch if reservoir is None else pd.concat([reservoir, batch], ignore_index=True)
            if len(reservoir) > 2 * max_rows:
                reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)

    if reservoir is None:
        raise SystemExit(f"No sheared rows selected from {catalogue}")
    if len(reservoir) > max_rows:
        reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)
    reservoir = reservoir.drop(columns="__key").reset_index(drop=True)
    print(f"  raw scanned={raw_rows:,}  kept (sheared)={len(reservoir):,}")
    return reservoir


def _shape_target_indices(names):
    """Locate the (e1,e2)-like shape target pair among the flow's target names,
    supporting both SExtractor (measured_e1_image/e2_image) and ngmix
    (measured_ngmix_g1/g2) conventions."""
    for c1, c2 in (("measured_e1_image", "measured_e2_image"),
                   ("measured_ngmix_g1", "measured_ngmix_g2"),
                   ("measured_galsim_g1", "measured_galsim_g2")):
        if c1 in names and c2 in names:
            return names.index(c1), names.index(c2)
    raise KeyError(f"No known shape target pair in {names}")


def model_mean_proj(bundle, base, s, ghat1, ghat2, intrinsic, rescale_kwargs,
                    n_samples, batch_size, return_proj=False):
    """< E[e_hat | S_{s*ghat}(intrinsic)] . ghat >  -- induced flow first moment
    projected onto the per-object applied-shear direction.

    Returns (global_mean, sem).  With return_proj=True also returns the per-object
    projection array `proj` (shape N,), letting callers form a per-object response
    (proj_{+g} - proj_{-g})/(2g).  Because the global mean is exactly np.mean(proj),
    the scalar response is identical whether taken from the two means or from the
    per-object array -- so exposing proj never changes the certified global R_flow."""
    frame = base.copy()
    e1p, e2p = apply_shear_to_ellipticity(intrinsic[0], intrinsic[1], s * ghat1, s * ghat2)
    frame["e1_input_rot0_p"] = e1p
    frame["e2_input_rot0_p"] = e2p
    frame = rescale(frame, **rescale_kwargs)
    draws = bundle.sample(frame, n_samples=n_samples, batch_size=batch_size)  # (N, n_samples, dim)
    mean = draws.mean(axis=1)  # (N, dim) in engineered target units
    i1, i2 = _shape_target_indices(bundle.target_transform.target_names)
    proj = mean[:, i1] * ghat1 + mean[:, i2] * ghat2
    gmean, sem = float(np.mean(proj)), float(np.std(proj) / np.sqrt(len(proj)))
    if return_proj:
        return gmean, sem, proj
    return gmean, sem


def flow_response(bundle, base, g, ghat1, ghat2, intrinsic, rescale_kwargs,
                  n_samples, batch_size, reseed=None, return_perobj=False):
    """Antithetic +/-g secant of the induced first moment -- the flow self-response

        R_flow = ( <proj(+g)> - <proj(-g)> ) / (2 g).

    This is the canonical way to read the trained flow's shape-response; it folds the
    two ``model_mean_proj`` legs the certified harvest evaluates inline.  ``reseed``, if
    given, is called with no arguments immediately before each leg so both legs share
    Common Random Numbers (flow-sampling noise cancels in the difference) -- variance
    reduction only, it does not bias R_flow.  With return_perobj=True also returns the
    per-object secant ``(proj_{+g} - proj_{-g})/(2 g)``; its mean is identically the
    scalar R_flow, so exposing it never changes the certified number.
    """
    if reseed is not None:
        reseed()
    mp, _, projp = model_mean_proj(bundle, base, +g, ghat1, ghat2, intrinsic,
                                   rescale_kwargs, n_samples, batch_size, return_proj=True)
    if reseed is not None:
        reseed()
    mm, _, projm = model_mean_proj(bundle, base, -g, ghat1, ghat2, intrinsic,
                                   rescale_kwargs, n_samples, batch_size, return_proj=True)
    R = (mp - mm) / (2 * g)
    if return_perobj:
        return R, (projp - projm) / (2 * g)
    return R
