#!/usr/bin/env python
"""Joint measurement+detection forward model (SBSI reframe, GOALS.md, locked 2026-07-22).

Trains ``sbs_shear.forward_model.SetConditionedForwardModel`` -- one DeepSets scene trunk feeding
(a) a ConditionalMeanFlow over the measured observables whose explicit mean head carries the
response, and (b) a Bernoulli detection head -- with a MULTI-TERM loss that supervises the
SEPARABLE response pieces against their OWN firewall-clean targets:

  L = L_NLL + lam_r * L_shape_resp + lam_d * L_BCE + lam_s * L_sel_resp

  L_NLL        -mean log p(meas e1, e2, meas mag, meas size | scene)     [FLOW stream, detected]
  L_shape_resp per-(mag x size)-bin || R_model - R_self ||^2  (dims 0,1) [FLOW stream]
  L_BCE        BCE(detection_logit, detected)                            [DET stream, det+undet]
  L_sel_resp   per-bin || dP/dgamma_model - b_true ||^2                  [DET stream]

Vs the diagnostic prototype (scripts/train_forward_prototype.py) this reframe changes:
  1. 4D measurement targets: measured shape + measured mag + measured size. Mag/size move
     INPUT->OUTPUT, killing the errors-in-variables floor a measured-conditioned model has on
     true-property cuts (diagnosed 2026-07-22). Their shear response R_theta (mean-head Jacobian
     cols 2,3) is READ OUT for validation, not pinned in this first build.
  2. PRIMARY-ONLY shear: the analytic S_{+/-delta} shift shears ONLY the primary intrinsic shape
     (neighbour fixed) so the flow learns R_self alone -- no R_blend double-count. R_blend stays
     the separate per-pair BlendEMU emulator.
  3. True-property primary cut (Re_input_p > 0.3, true mag r_input_p < 26); neighbours full-pop.

FIREWALL: train on HALF-SHEAR only (det_meas g=0 train leg + half-shear R_sim lookup).
constgold (constant +/-0.02) is the held-out acceptance set and is NEVER read here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa: E402
from sbs_shear.forward_model import SetConditionedForwardModel  # noqa: E402
from sbs_shear.measurement_model import TargetStandardizer  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    rescale,
    source_select_selection,
)
from sbs_shear.scene_model import SetFeatureStandardizer  # noqa: E402
from sbs_shear.selection_model import TabularPreprocessor  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

# --- ORIENTED scene features (shared by both streams -> shared trunk dims). ------------------
# The measured shape target is orientation-full, so the primary shape must be the ORIENTED
# sky-basis e1/e2 (NOT the rotation-invariant e_abs_p) for the mean head to carry a shear
# response -- see the ORIENTED-variant note in measurement_model.py.  Gamma_* features are
# omitted so the same list works for the FLOW catalogue (no applied-shear columns) and DET.
PRIMARY_FEATURES = [
    "Re_input_p_scaled",
    "r_input_p_scaled",
    "sersic_n_input_p",
    "e1_input_p",
    "e2_input_p",
    "redshift_input_p",
    "neighbor_count",             # 0/1 for the nearest-pair prototype
    "nearest_distance_scaled",    # distance_scaled if neighbored else NaN->fill
]
NEIGHBOR_FEATURES = [
    "distance_scaled",
    "e1_input_s",
    "e2_input_s",
    "Re_input_s_scaled",
    "r_input_s_scaled",
    "flux_ratio",
    "sersic_n_input_s",
    "redshift_input_s",
    "relative_position_angle_cos2",
    "relative_position_angle_sin2",
]
# 4D measurement targets. dims 0,1 = measured shape (response PINNED to R_sim in the loss);
# dims 2,3 = measured mag + measured log-size, moved INPUT->OUTPUT to kill the errors-in-variables
# floor. Their shear response R_theta (mean-head Jacobian cols 2,3) is READ OUT for validation
# (eval_joint_triad), not pinned here. Order matters: shape first so target scales[0,1] = the
# shape components the response loss uses.
SHAPE_TARGETS = ["measured_e1_m", "measured_e2_m"]
MEAS_EXTRA_TARGETS = ["measured_mag_auto", "measured_log_flux_radius"]
MEASUREMENT_TARGETS = SHAPE_TARGETS + MEAS_EXTRA_TARGETS
GAMMA_SELF = 0.05                                     # delta_et = measured_e(0.05) - measured_e(0)

# Shifts needed per stream: baseline (delta=0) for NLL/BCE + 4 for the central-secant response.
SHIFTS = {
    "0":   ((1.0, 0.0), 0.0),
    "e1p": ((1.0, 0.0), +1.0),
    "e1m": ((1.0, 0.0), -1.0),
    "e2p": ((0.0, 1.0), +1.0),
    "e2m": ((0.0, 1.0), -1.0),
}

RESCALE_KW = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)


# ----------------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------------
class Reservoir:
    def __init__(self, max_rows, rng):
        self.max = int(max_rows) if max_rows else 0
        self.rng = rng
        self.buf = None

    def add(self, df):
        if len(df) == 0:
            return
        df = df.copy()
        df["__k"] = self.rng.random(len(df))
        self.buf = df if self.buf is None else pd.concat([self.buf, df], ignore_index=True)
        if self.max and len(self.buf) > 2 * self.max:
            self.buf = self.buf.nlargest(self.max, "__k").reset_index(drop=True)

    def finalize(self):
        if self.buf is None:
            raise RuntimeError("empty reservoir -- no rows passed the cuts")
        b = self.buf
        if self.max and len(b) > self.max:
            b = b.nlargest(self.max, "__k")
        return b.drop(columns="__k").reset_index(drop=True)


def _candidate_columns(kind):
    common = [
        "case", "neighbored", "distance", "polarization_angle", "shear_component_convention",
        "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
        "sersic_n_input_p", "sersic_n_input_s", "redshift_input_p", "redshift_input_s",
        # shape: rot0 (DET) OR axis_ratio/position_angle (FLOW)
        "e1_input_rot0_p", "e2_input_rot0_p", "e1_input_rot0_s", "e2_input_rot0_s",
        "axis_ratio_input_p", "position_angle_input_p",
        "axis_ratio_input_s", "position_angle_input_s",
        "r_blend", "nbr_flux_near", "nbr_flux_far",   # crowd/blend axis + shell fluxes (augmented cats)
    ]
    if kind == "flow":
        # measured_ngmix_g{1,2} = the g=0 measured shape in a full-population det_meas catalogue
        # (renamed to measured_e{1,2}_m in load_stream); delta_et* only exist in the pairs product.
        return common + ["measured_e1_m", "measured_e2_m", "delta_et1", "delta_et2",
                         "measured_ngmix_g1", "measured_ngmix_g2",
                         "measured_mag_auto", "measured_flux_radius"]
    return common + ["detected", "shear_case"]


def load_stream(path, kind, max_case, train_case_max, max_rows_tr, max_rows_va, seed, true_cut=None):
    rng = np.random.default_rng(seed)
    res_tr = Reservoir(max_rows_tr, rng)
    res_va = Reservoir(max_rows_va, rng)
    want = _candidate_columns(kind)
    raw_scanned = 0
    t0 = time.time()
    with ipc.open_file(path) as reader:
        avail = set(reader.schema.names)
        cols = [c for c in want if c in avail]
        for bi in range(reader.num_record_batches):
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            raw_scanned += len(b)
            minc = int(b["case"].min())
            b = b[b["case"].astype(int) < max_case]
            if len(b) == 0:
                if minc >= max_case:      # catalogues are written case-ascending -> stop early
                    break
                continue
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            if true_cut is not None:
                # acceptance TRAINING cut on TRUE primary properties (shear-invariant). r_input_p IS
                # the true magnitude (preprocessing.mag2flux takes it as mag), so mag<max == r_input_p<max.
                re_min, mag_max = true_cut
                b = b[(b["Re_input_p"].to_numpy(float) > re_min)
                      & (b["r_input_p"].to_numpy(float) < mag_max)]
                if len(b) == 0:
                    continue
            if kind == "flow":
                # full-population det_meas carries the g=0 shape as measured_ngmix_g{1,2}
                if "measured_e1_m" not in b.columns and "measured_ngmix_g1" in b.columns:
                    b = b.rename(columns={"measured_ngmix_g1": "measured_e1_m",
                                          "measured_ngmix_g2": "measured_e2_m"})
                # measurement density is DETECTED-only: p(obs | scene, detected=1)
                if "detected" in b.columns:
                    b = b[b["detected"].astype(bool)]
                # measured size -> log (positive, heavy-tailed); mag_auto is already a log-magnitude
                if "measured_flux_radius" in b.columns:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        b["measured_log_flux_radius"] = np.log(b["measured_flux_radius"].to_numpy(float))
                # require finite on the 4D targets present + delta_et* if this is a pairs product
                fcols = [c for c in ["measured_e1_m", "measured_e2_m",
                                     "measured_mag_auto", "measured_log_flux_radius",
                                     "delta_et1", "delta_et2"] if c in b.columns]
                fin = np.isfinite(b[fcols].to_numpy(float)).all(axis=1)
                b = b.loc[fin]
            if len(b) == 0:
                continue
            case = b["case"].astype(int)
            res_tr.add(b.loc[case < train_case_max])
            res_va.add(b.loc[case >= train_case_max])
    tr, va = res_tr.finalize(), res_va.finalize()
    print(f"  [{kind}] scanned={raw_scanned:,}  train={len(tr):,} (cases<{train_case_max})  "
          f"val={len(va):,} (cases {train_case_max}-{max_case - 1})  {time.time() - t0:.1f}s",
          flush=True)
    return tr, va


def intrinsic_shape(frame, suffix):
    """Intrinsic (unsheared, rot0) sky-basis ellipticity for primary ('p') / neighbour ('s')."""
    r0e1, r0e2 = f"e1_input_rot0_{suffix}", f"e2_input_rot0_{suffix}"
    if r0e1 in frame.columns and r0e2 in frame.columns:
        return frame[r0e1].to_numpy(float), frame[r0e2].to_numpy(float)
    ar, pa_ = f"axis_ratio_input_{suffix}", f"position_angle_input_{suffix}"
    e1, e2 = ellipticity_from_axis_ratio_angle(frame[ar].to_numpy(float), frame[pa_].to_numpy(float))
    return np.asarray(e1, float), np.asarray(e2, float)


def shifted_feature_frame(base, intr, gdir, delta, primary_only=False):
    """Copy base, apply S_{delta*gdir} to the primary intrinsic shape (and, unless primary_only,
    the neighbour intrinsic shape too), rebuild scene features, add the scene-summary columns.
    delta=0 -> unshifted.  primary_only=True leaves the neighbour fixed so the induced response
    is R_self ALONE (no R_blend double-count) -- the reframe firewall."""
    f = base.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        # isolated rows carry NaN neighbour shapes -> NaN under the Mobius map; they are masked
        # out downstream, so silence the benign divide/invalid warnings here.
        e1p, e2p = apply_shear_to_ellipticity(intr["e1p"], intr["e2p"], delta * gdir[0], delta * gdir[1])
        if primary_only:
            e1s, e2s = intr["e1s"], intr["e2s"]           # neighbour NOT sheared -> isolate R_self
        else:
            e1s, e2s = apply_shear_to_ellipticity(intr["e1s"], intr["e2s"], delta * gdir[0], delta * gdir[1])
    f["e1_input_rot0_p"], f["e2_input_rot0_p"] = e1p, e2p
    f["e1_input_rot0_s"], f["e2_input_rot0_s"] = e1s, e2s
    f = rescale(f, **RESCALE_KW)
    nbg = f["neighbored"].astype(bool).to_numpy()
    f["neighbor_count"] = nbg.astype(np.float32)
    f["nearest_distance_scaled"] = np.where(nbg, f["distance_scaled"].to_numpy(float), np.nan)
    return f, nbg


def neighbor_padded(frame, nbg, nbr_std):
    raw = frame[NEIGHBOR_FEATURES].to_numpy(np.float32)
    std = nbr_std.transform_array(raw)          # fills NaN with fit means, then standardizes
    std[~nbg] = 0.0                             # empty set -> zeros (masked out anyway)
    return std[:, None, :], nbg.astype(np.float32)[:, None]


def precompute_stream(base, kind, primary_pre, nbr_std, delta, device,
                      target_std=None, flux_edges=None, size_edges=None, mag_edges=None,
                      dist_edges=None, decorrelate=False, primary_only=False,
                      det_flux_edges=None, det_size_edges=None, det_dist_edges=None,
                      theta_size_grid=None, theta_mag_grid=None, crowd_edges=None):
    """Return a dict of GPU tensors: 5 shifted (primary, neighbour) pairs + mask + stream labels.
    primary_only=True shears only the primary in the +/-delta response contexts (isolate R_self)."""
    intr = {}
    intr["e1p"], intr["e2p"] = intrinsic_shape(base, "p")
    intr["e1s"], intr["e2s"] = intrinsic_shape(base, "s")
    out = {}
    mask = None
    for name, (gdir, sgn) in SHIFTS.items():
        f, nbg = shifted_feature_frame(base, intr, gdir, sgn * delta, primary_only=primary_only)
        out[f"p_{name}"] = torch.as_tensor(primary_pre.transform_frame(f), dtype=torch.float32, device=device)
        npad, m = neighbor_padded(f, nbg, nbr_std)
        out[f"n_{name}"] = torch.as_tensor(npad, dtype=torch.float32, device=device)
        if mask is None:
            mask = torch.as_tensor(m, dtype=torch.float32, device=device)
    out["mask"] = mask
    if kind == "flow":
        out["tgt"] = torch.as_tensor(target_std.transform_frame(base), dtype=torch.float32, device=device)
        if "delta_et1" in base.columns:                                # empirical two-leg (pairs) path
            rself = base["delta_et1"].to_numpy(float) / GAMMA_SELF
            out["rself"] = torch.as_tensor(rself, dtype=torch.float32, device=device)
        flux = base["r_input_p"].to_numpy(float)
        size = base["Re_input_p"].to_numpy(float)
        nf, ns = len(flux_edges) - 1, len(size_edges) - 1
        fi = np.clip(np.digitize(flux, flux_edges) - 1, 0, nf - 1)
        si = np.clip(np.digitize(size, size_edges) - 1, 0, ns - 1)
        if crowd_edges is not None:
            # 3D grid (flux, size, blend); blend axis = QUANTILE bins of r_blend (neighbour-blend
            # severity, V1-style). Isolated / NaN r_blend -> bin 0 (weakest blend, highest response).
            rb = base["r_blend"].to_numpy(float)
            nblend = len(crowd_edges) - 1
            ci = np.clip(np.digitize(rb, crowd_edges) - 1, 0, nblend - 1)
            ci = np.where(np.isfinite(rb), ci, 0)
            binid = (fi * ns + si) * nblend + ci
        elif dist_edges is not None:
            # 3D grid (flux, size, blend); blend bin 0 = isolated, 1..ndist = distance quantiles.
            # C-order flatten of Rsim shape (nf, ns, nblend): (fi*ns + si)*nblend + bi.
            nbg = base["neighbored"].astype(bool).to_numpy()
            dist = base["distance"].to_numpy(float)
            ndist = len(dist_edges) - 1
            bi = np.where(nbg, 1 + np.clip(np.digitize(dist, dist_edges) - 1, 0, ndist - 1), 0)
            nblend = ndist + 1
            binid = (fi * ns + si) * nblend + bi
        else:
            binid = fi * ns + si
        out["binid"] = torch.as_tensor(binid.astype(np.int64), device=device)
        if decorrelate:
            e1i, e2i = intrinsic_shape(base, "p")                      # g=0 intrinsic (rot0) shape
            w = decorrelation_weights(np.hypot(e1i, e2i), size)
            out["w"] = torch.as_tensor(w, dtype=torch.float32, device=device)
        if theta_size_grid is not None:                                # R_theta coupling supervision
            e1i, e2i = intrinsic_shape(base, "p")                      # g=0 intrinsic (rot0) shape, per obj
            out["e1_int"] = torch.as_tensor(np.asarray(e1i, np.float32), device=device)
            out["e2_int"] = torch.as_tensor(np.asarray(e2i, np.float32), device=device)
            out["theta_bsize"] = torch.as_tensor(theta_size_grid.reshape(-1)[binid].astype(np.float32), device=device)
            out["theta_bmag"] = torch.as_tensor(theta_mag_grid.reshape(-1)[binid].astype(np.float32), device=device)
    else:
        out["detected"] = torch.as_tensor(base["detected"].astype(np.float32).to_numpy(), dtype=torch.float32, device=device)
        mag = base["r_input_p"].to_numpy(float)
        nmb = len(mag_edges) - 1
        mi = np.clip(np.digitize(mag, mag_edges) - 1, 0, nmb - 1)
        out["magbin"] = torch.as_tensor(mi.astype(np.int64), device=device)
        if det_flux_edges is not None:
            # BLEND-RESOLVED detection target: per-object bin into grid_b[flux, size, blend].
            # blend 0 = isolated, 1..ndist = distance quantiles; C-order (fi*ns+si)*nblend+bi ==
            # grid_b.reshape(-1) (identical to the flow branch's binid and harvest_det_response.py).
            size = base["Re_input_p"].to_numpy(float)
            nfg, nsg = len(det_flux_edges) - 1, len(det_size_edges) - 1
            ndist = len(det_dist_edges) - 1
            fi = np.clip(np.digitize(mag, det_flux_edges) - 1, 0, nfg - 1)
            si = np.clip(np.digitize(size, det_size_edges) - 1, 0, nsg - 1)
            nbg = base["neighbored"].astype(bool).to_numpy()
            dist = base["distance"].to_numpy(float)
            bi = np.where(nbg, 1 + np.clip(np.digitize(dist, det_dist_edges) - 1, 0, ndist - 1), 0)
            gb = (fi * nsg + si) * (ndist + 1) + bi
            out["detbin"] = torch.as_tensor(gb.astype(np.int64), device=device)
        else:
            out["detbin"] = out["magbin"]          # default: mag-only supervision (unchanged behaviour)
    out["_n"] = int(mask.shape[0])
    return out


# ----------------------------------------------------------------------------------------------
# Losses / responses
# ----------------------------------------------------------------------------------------------
def decorrelation_weights(e_abs, size, nbins=40, clip=10.0):
    """Per-row weights that make g=0 scene-shape magnitude |e| and size independent, so the mean
    head learns the shape response across the full (shape x size) support instead of extrapolating
    from the correlated manifold.  Port of train_measurement_model.compute_decorrelation_weights
    (the documented isolated-bias fix); uses g=0 truth only, no sheared-shear information."""
    good = np.isfinite(e_abs) & np.isfinite(size)
    ea = np.quantile(e_abs[good], np.linspace(0, 1, nbins + 1)); ea[0] -= 1e-9; ea[-1] += 1e-9
    eb = np.quantile(size[good], np.linspace(0, 1, nbins + 1)); eb[0] -= 1e-9; eb[-1] += 1e-9
    ia = np.clip(np.digitize(e_abs, ea) - 1, 0, nbins - 1)
    ib = np.clip(np.digitize(size, eb) - 1, 0, nbins - 1)
    joint = np.zeros((nbins, nbins)); np.add.at(joint, (ia, ib), good.astype(float))
    pa_ = joint.sum(1); pb = joint.sum(0); N = max(joint.sum(), 1.0)
    wij = (pa_[ia] * pb[ib]) / (N * np.maximum(joint[ia, ib], 1.0))
    w = np.where(good, wij, 1.0)
    w = np.clip(w, 1.0 / clip, clip)
    return (w / w.mean()).astype(np.float32)


def binned_mse(values, binid, targets, nbins, weights=None, equal_weight=False):
    w = torch.ones_like(values) if weights is None else weights
    sum_b = torch.zeros(nbins, device=values.device).index_add_(0, binid, w * values)
    cnt_b = torch.zeros(nbins, device=values.device).index_add_(0, binid, w)
    mean_b = torch.where(cnt_b > 0, sum_b / cnt_b.clamp_min(1e-8), targets)
    if equal_weight:
        # weight each OCCUPIED bin equally (not by population count): stops high-response, low-count
        # cells (faint/isolated) from being drowned out by the many bright near-zero-response cells.
        occ = (cnt_b > 0).to(values.dtype)
        return ((mean_b - targets) ** 2 * occ).sum() / occ.sum().clamp_min(1.0)
    return ((mean_b - targets) ** 2 * cnt_b).sum() / cnt_b.sum().clamp_min(1.0)


def _flow_mu_shifts(model, D, idx):
    """The 4 mean-head evaluations at the +/-delta shear contexts (dir-1: e1p/e1m, dir-2: e2p/e2m).
    Shared by the shape response (dims 0,1) and the R_theta coupling (dims 2,3) -> ONE set of passes."""
    m = D["mask"][idx]
    # pass the shifted primary tensor as the shape-skip source (no-op unless model has the skip head)
    return (model.mu(model.context(D["p_e1p"][idx], D["n_e1p"][idx], m), D["p_e1p"][idx]),
            model.mu(model.context(D["p_e1m"][idx], D["n_e1m"][idx], m), D["p_e1m"][idx]),
            model.mu(model.context(D["p_e2p"][idx], D["n_e2p"][idx], m), D["p_e2p"][idx]),
            model.mu(model.context(D["p_e2m"][idx], D["n_e2m"][idx], m), D["p_e2m"][idx]))


def flow_response_perobj(model, D, idx, sc, delta):
    mu1p, mu1m, mu2p, mu2m = _flow_mu_shifts(model, D, idx)
    return 0.5 * ((mu1p[:, 0] - mu1m[:, 0]) * sc[0] + (mu2p[:, 1] - mu2m[:, 1]) * sc[1]) / (2.0 * delta)


def theta_coupling_residual(mu1p, mu1m, mu2p, mu2m, D, idx, sc23, delta):
    """R_theta ORIENTATION-COUPLING loss (owner: pin the coupling directly, 2026-07-23). The measured
    mag/size (dims 2,3) shear response is a spin-2 coupling to the intrinsic shape:
    d<prop>/d_gamma1 = b*e1_int, d<prop>/d_gamma2 = b*e2_int, with b = the per-cell half-shear slope
    (theta_bsize/theta_bmag). Central diff already cancels the coherent even part; this pins the odd
    (size-cut selection-driving) part.  sc23 = target scales for dims 2,3 (un-standardize)."""
    sc2, sc3 = sc23
    inv = 1.0 / (2.0 * delta)
    dM1 = (mu1p[:, 2] - mu1m[:, 2]) * sc2 * inv
    dM2 = (mu2p[:, 2] - mu2m[:, 2]) * sc2 * inv
    dS1 = (mu1p[:, 3] - mu1m[:, 3]) * sc3 * inv
    dS2 = (mu2p[:, 3] - mu2m[:, 3]) * sc3 * inv
    e1, e2 = D["e1_int"][idx], D["e2_int"][idx]
    bS, bM = D["theta_bsize"][idx], D["theta_bmag"][idx]
    return ((dS1 - bS * e1) ** 2 + (dS2 - bS * e2) ** 2
            + (dM1 - bM * e1) ** 2 + (dM2 - bM * e2) ** 2).mean()


def flow_loss(model, D, idx, sc, delta, bin_tgt, nbins, equal_weight=False, sc23=None, lam_theta=0.0):
    """Return (L_NLL, L_shape_resp, L_theta) as differentiable tensors.  When D carries decorrelation
    weights ('w'), the NLL and per-bin shape response are weighted by them.  equal_weight weights each
    occupied (flux,size,blend) response cell EQUALLY so rare high-response cells are not under-fit.
    L_theta (0 unless lam_theta>0) pins the measured mag/size orientation coupling (dims 2,3)."""
    ctx0 = model.context(D["p_0"][idx], D["n_0"][idx], D["mask"][idx])
    lp = model.log_prob_obs(D["tgt"][idx], ctx0, D["p_0"][idx])
    wi = D.get("w")
    if wi is not None:
        wi = wi[idx]
        nll = -(wi * lp).sum() / wi.sum().clamp_min(1e-8)
    else:
        nll = -lp.mean()
    mu1p, mu1m, mu2p, mu2m = _flow_mu_shifts(model, D, idx)
    r_i = 0.5 * ((mu1p[:, 0] - mu1m[:, 0]) * sc[0] + (mu2p[:, 1] - mu2m[:, 1]) * sc[1]) / (2.0 * delta)
    resp = binned_mse(r_i, D["binid"][idx], bin_tgt, nbins, weights=wi, equal_weight=equal_weight)
    theta = torch.zeros((), device=r_i.device)
    if lam_theta > 0.0 and sc23 is not None:
        theta = theta_coupling_residual(mu1p, mu1m, mu2p, mu2m, D, idx, sc23, delta)
    return nll, resp, theta


def det_response_perobj(model, D, idx, delta):
    m = D["mask"][idx]
    l1p = model.detection_logit(model.context(D["p_e1p"][idx], D["n_e1p"][idx], m))
    l1m = model.detection_logit(model.context(D["p_e1m"][idx], D["n_e1m"][idx], m))
    l2p = model.detection_logit(model.context(D["p_e2p"][idx], D["n_e2p"][idx], m))
    l2m = model.detection_logit(model.context(D["p_e2m"][idx], D["n_e2m"][idx], m))
    return 0.5 * ((torch.sigmoid(l1p) - torch.sigmoid(l1m))
                  + (torch.sigmoid(l2p) - torch.sigmoid(l2m))) / (2.0 * delta)


def det_loss(model, D, idx, delta, b_true, nmb, equal_weight=False):
    """Return (L_BCE, L_sel_resp) as differentiable tensors."""
    logit0 = model.detection_logit(model.context(D["p_0"][idx], D["n_0"][idx], D["mask"][idx]))
    bce = F.binary_cross_entropy_with_logits(logit0, D["detected"][idx])
    dp = det_response_perobj(model, D, idx, delta)
    sel = binned_mse(dp, D["detbin"][idx], b_true, nmb, equal_weight=equal_weight)
    return bce, sel


def batches(n, bs, device, shuffle):
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    return [idx[i:i + bs] for i in range(0, n, bs)]


# ----------------------------------------------------------------------------------------------
# Evaluation (V1-V4)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def eval_flow(model, D, sc, delta, bin_tgt, nbins, bs, empty_set=False, sc23=None):
    model.eval()
    r_all, bin_all, nll_sum, n, theta_sum = [], [], 0.0, 0, 0.0
    for idx in batches(D["_n"], bs, D["mask"].device, shuffle=False):
        m = torch.zeros_like(D["mask"][idx]) if empty_set else D["mask"][idx]
        ctx0 = model.context(D["p_0"][idx], D["n_0"][idx], m)
        nll_sum += float((-model.log_prob_obs(D["tgt"][idx], ctx0, D["p_0"][idx])).sum())
        n += len(idx)
        mu1p = model.mu(model.context(D["p_e1p"][idx], D["n_e1p"][idx], m), D["p_e1p"][idx])
        mu1m = model.mu(model.context(D["p_e1m"][idx], D["n_e1m"][idx], m), D["p_e1m"][idx])
        mu2p = model.mu(model.context(D["p_e2p"][idx], D["n_e2p"][idx], m), D["p_e2p"][idx])
        mu2m = model.mu(model.context(D["p_e2m"][idx], D["n_e2m"][idx], m), D["p_e2m"][idx])
        r_i = 0.5 * ((mu1p[:, 0] - mu1m[:, 0]) * sc[0] + (mu2p[:, 1] - mu2m[:, 1]) * sc[1]) / (2.0 * delta)
        r_all.append(r_i)
        bin_all.append(D["binid"][idx])
        if sc23 is not None and not empty_set and "e1_int" in D:
            theta_sum += float(theta_coupling_residual(mu1p, mu1m, mu2p, mu2m, D, idx, sc23, delta)) * len(idx)
    r = torch.cat(r_all)
    binid = torch.cat(bin_all)
    per_bin = np.full(nbins, np.nan)
    cnt = np.zeros(nbins, np.int64)
    b_np = binid.cpu().numpy()
    r_np = r.cpu().numpy()
    for k in range(nbins):
        sel = b_np == k
        cnt[k] = int(sel.sum())
        if cnt[k]:
            per_bin[k] = float(r_np[sel].mean())
    return {"R_global": float(r.mean()), "R_bin": per_bin, "bin_count": cnt,
            "nll": nll_sum / max(n, 1), "theta": theta_sum / max(n, 1)}


@torch.no_grad()
def eval_det(model, D, delta, b_true, nmb, bs):
    model.eval()
    dp_all, bin_all, bce_sum, n = [], [], 0.0, 0
    for idx in batches(D["_n"], bs, D["mask"].device, shuffle=False):
        logit0 = model.detection_logit(model.context(D["p_0"][idx], D["n_0"][idx], D["mask"][idx]))
        bce_sum += float(F.binary_cross_entropy_with_logits(logit0, D["detected"][idx], reduction="sum"))
        n += len(idx)
        dp_all.append(det_response_perobj(model, D, idx, delta))
        bin_all.append(D["detbin"][idx])
    dp = torch.cat(dp_all)
    binid = torch.cat(bin_all)
    per_bin = np.full(nmb, np.nan)
    cnt = np.zeros(nmb, np.int64)
    b_np = binid.cpu().numpy()
    d_np = dp.cpu().numpy()
    for k in range(nmb):
        sel = b_np == k
        cnt[k] = int(sel.sum())
        if cnt[k]:
            per_bin[k] = float(d_np[sel].mean())
    return {"dP_global": float(dp.mean()), "dP_bin": per_bin, "bin_count": cnt,
            "bce": bce_sum / max(n, 1)}


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
def _shape_skip_config(args):
    """(dim, idx, hidden) for the direct intrinsic-shape skip head, or (0, None, 0) if off.
    idx = columns of e1_input_p/e2_input_p in the standardized PRIMARY feature vector."""
    if not getattr(args, "shape_skip", False):
        return 0, None, 0
    idx = [PRIMARY_FEATURES.index("e1_input_p"), PRIMARY_FEATURES.index("e2_input_p")]
    return len(idx), idx, int(getattr(args, "shape_skip_hidden", 0))


def build_model(primary_dim, neighbor_dim, args, device):
    skip_dim, skip_idx, skip_hidden = _shape_skip_config(args)
    return SetConditionedForwardModel(
        shape_skip_dim=skip_dim, shape_skip_idx=skip_idx, shape_skip_hidden=skip_hidden,
        target_dim=len(MEASUREMENT_TARGETS),
        primary_dim=primary_dim,
        neighbor_dim=neighbor_dim,
        context_dim=args.context_dim,
        set_hidden_dim=args.set_hidden_dim,
        set_neighbor_layers=2,
        set_context_layers=2,
        flow_hidden_dim=args.flow_hidden_dim,
        flow_layers=args.flow_layers,
        n_flows=args.n_flows,
        mean_hidden=args.mean_hidden,
        det_hidden=args.det_hidden,
        det_layers=2,
        activation="silu",
        pooling="sum",
        base_flow="affine",
    ).to(device)


def train_one(mode, Ftr, Fva, Dtr, Dva, sc, args, primary_dim, neighbor_dim,
              bin_tgt, nbins, b_true, nmb, lam_r, lam_s, lam_d, epochs, device, tag="",
              sc23=None, lam_theta=0.0):
    """mode in {joint, flow_only, det_only}.  Returns (model, history, best_val)."""
    torch.manual_seed(args.seed)
    model = build_model(primary_dim, neighbor_dim, args, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    hist = {"train_total": [], "val_nll": [], "val_shape_resp": [], "val_bce": [], "val_sel_resp": [],
            "val_R": [], "val_dP": []}
    best_val, best_state, wait = np.inf, None, 0
    swa_k = max(int(getattr(args, "swa_last_k", 0)), 0)   # SWA: last-K epoch weight average (0=off, V1-style)
    swa_snaps = []
    do_flow = mode in ("joint", "flow_only")
    do_det = mode in ("joint", "det_only")
    d = args.delta
    print(f"\n--- train [{tag or mode}] lam_r={lam_r} lam_s={lam_s} lam_d={lam_d} epochs={epochs} ---", flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        fb = batches(Ftr["_n"], args.batch_size, device, shuffle=True) if do_flow else []
        db = batches(Dtr["_n"], args.batch_size, device, shuffle=True) if do_det else []
        nsteps = max(len(fb), len(db))
        tot = 0.0
        for i in range(nsteps):
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            if do_flow:
                nll, resp, theta = flow_loss(model, Ftr, fb[i % len(fb)], sc, d, bin_tgt, nbins,
                                             equal_weight=args.flow_equal_weight, sc23=sc23, lam_theta=lam_theta)
                loss = loss + nll + lam_r * resp + lam_theta * theta
            if do_det:
                bce, sel = det_loss(model, Dtr, db[i % len(db)], d, b_true, nmb,
                                    equal_weight=args.det_equal_weight)
                loss = loss + lam_d * bce + lam_s * sel
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            tot += float(loss.detach())
        # validation
        vnll = vshape = vtheta = vbce = vsel = vR = vdP = np.nan
        sel_obj = 0.0
        if do_flow:
            ef = eval_flow(model, Fva, sc, d, bin_tgt, nbins, args.batch_size, sc23=sc23)
            vnll, vR = ef["nll"], ef["R_global"]
            vshape = float(np.nanmean((ef["R_bin"] - bin_tgt.cpu().numpy()) ** 2))
            vtheta = ef.get("theta", np.nan)
            sel_obj += vnll + lam_r * vshape + (lam_theta * vtheta if np.isfinite(vtheta) else 0.0)
        if do_det:
            ed = eval_det(model, Dva, d, b_true, nmb, args.batch_size)
            vbce, vdP = ed["bce"], ed["dP_global"]
            vsel = float(np.nanmean((ed["dP_bin"] - b_true.cpu().numpy()) ** 2))
            sel_obj += vbce + lam_s * vsel
        hist["train_total"].append(tot / max(nsteps, 1))
        hist["val_nll"].append(vnll); hist["val_shape_resp"].append(vshape)
        hist["val_bce"].append(vbce); hist["val_sel_resp"].append(vsel)
        hist["val_R"].append(vR); hist["val_dP"].append(vdP)
        if ep % args.log_every == 0 or ep == 1 or ep == epochs:
            print(f"  ep{ep:03d} tot={tot / max(nsteps,1):.4f} | "
                  f"nll={vnll:.4f} shp_resp={vshape:.2e} theta={vtheta:.2e} <R>={vR:+.4f} | "
                  f"bce={vbce:.4f} sel_resp={vsel:.2e} <dP>={vdP:+.4f}", flush=True)
        if swa_k > 0:                                     # SWA: keep last-K end-of-epoch snapshots (CPU)
            swa_snaps.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            swa_snaps = swa_snaps[-swa_k:]
        if sel_obj < best_val:
            best_val, wait = sel_obj, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  early stop @ ep{ep}", flush=True)
                break
    if swa_k > 0 and swa_snaps:                           # SWA average (no BatchNorm -> plain weight mean)
        newest = swa_snaps[-1]
        swa_state = {}
        for key, ref in newest.items():
            if torch.is_floating_point(ref):
                swa_state[key] = torch.stack([sn[key].to(torch.float64) for sn in swa_snaps], 0).mean(0).to(ref.dtype)
            else:
                swa_state[key] = ref.clone()
        model.load_state_dict(swa_state)
        print(f"  SWA: averaged last {len(swa_snaps)} epoch snapshots", flush=True)
    elif best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow-catalogue",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
                            "det_meas_ngmix_g0.0_train.feather",
                    help="HALF-SHEAR g=0 leg with the 4D measured observables (shape+mag+size). "
                         "Its ngmix estimator must MATCH the --target-npz R_sim (non-ap7 here). "
                         "The shape response comes from the firewall-clean --target-npz lookup.")
    ap.add_argument("--det-catalogue",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather")
    ap.add_argument("--btrue-npz",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/btrue_detection.npz")
    ap.add_argument("--btrue-grid", action="store_true",
                    help="supervise the detection-response head on the BLEND-RESOLVED b_true grid "
                         "grid_b[flux,size,blend] (blend 0=isolated + distance bins) instead of mag_b "
                         "(mag only). Fixes the blend-resolved detection non-closure (Goal 2). "
                         "DIAGNOSTIC: dP is additive, NEVER wired into certified m=R_sim/(R_flow+R_blend)-1.")
    ap.add_argument("--det-equal-weight", action="store_true",
                    help="weight each OCCUPIED detection-target cell EQUALLY in the selection-response "
                         "loss (not by population count). With the fine blend grid the high-response "
                         "cells (faint/isolated) are low-count and get drowned out by count-weighting; "
                         "equal weight aligns the training loss with the (already equal-weighted) "
                         "early-stop metric. Recommended with --btrue-grid.")
    ap.add_argument("--flow-equal-weight", action="store_true",
                    help="weight each OCCUPIED (flux,size,blend) response cell EQUALLY in the flow "
                         "shape-response loss (not by count). Fixes R_flow UNDER-fitting the rare "
                         "high-response LARGE-size cells (size cuts: additive residual +4..+6%). "
                         "CERTIFIED-path change: aligns with the equal-weighted early-stop metric; "
                         "a-priori (not tuned on |m|); needs re-certification of GLOBAL m before adoption.")
    ap.add_argument("--outdir", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto")
    ap.add_argument("--max-case", type=int, default=20)
    ap.add_argument("--train-case-max", type=int, default=16)   # train cases <16, val 16-19
    ap.add_argument("--det-max-case", type=int, default=None,
                    help="det-stream case cap (default: --max-case). Decouple when the det catalogue "
                         "has fewer cases than the flow catalogue (else the det val split is empty).")
    ap.add_argument("--det-train-case-max", type=int, default=None,
                    help="det-stream train/val split (default: --train-case-max).")
    ap.add_argument("--max-rows-flow", type=int, default=1_000_000)
    ap.add_argument("--max-rows-det", type=int, default=1_000_000)
    ap.add_argument("--max-rows-val", type=int, default=400_000)
    ap.add_argument("--n-flux", type=int, default=4)
    ap.add_argument("--n-size", type=int, default=3)
    # --- production-guided isolated recipe (cont.60), gated: default None/off = pairs-only behaviour ---
    ap.add_argument("--target-npz", default=None,
                    help="isolated-separated snc self-response grid (compute_response_target_blend.py: "
                         "Rsim[flux,size,blend], blend bin 0=isolated + distance bins). When set, the "
                         "per-bin shape-response target is this LOOKUP (both-legs-free) instead of the "
                         "empirical delta_et1 -- so isolated objects need only the g=0 leg.")
    ap.add_argument("--decorrelate", action="store_true",
                    help="weight the flow loss by the g=0 shape<->size decorrelation weights "
                         "(train_measurement_model.compute_decorrelation_weights) -- the documented "
                         "isolated-bias fix. Uses g=0 truth only.")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--lam-r", type=float, default=50.0)
    ap.add_argument("--lam-s", type=float, default=50.0)
    ap.add_argument("--lam-d", type=float, default=1.0)
    ap.add_argument("--flow-only", action="store_true",
                    help="train the MAIN model as flow-only (no detection head/loss): skips det batches, "
                         "isolates the flow's shape/mag-size response. lam_d/lam_s forced to 0.")
    ap.add_argument("--lam-theta", type=float, default=0.0,
                    help="weight of the R_theta orientation-coupling loss (measured mag/size shear "
                         "response, dims 2,3). 0 = off (unchanged behaviour).")
    ap.add_argument("--theta-target-npz", default=None,
                    help="coupling target from build_theta_coupling_target.py (per-cell b_size/b_mag "
                         "on the SAME grid as --target-npz). Required when --lam-theta>0.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--sens-epochs", type=int, default=20)
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=7.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-5)
    ap.add_argument("--max-grad-norm", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--swa-last-k", type=int, default=0,
                    help="SWA: average the last K end-of-epoch weight snapshots as the final model (0=off, V1-style).")
    ap.add_argument("--context-dim", type=int, default=128)
    ap.add_argument("--set-hidden-dim", type=int, default=128)
    ap.add_argument("--flow-hidden-dim", type=int, default=128)
    ap.add_argument("--flow-layers", type=int, default=3)
    ap.add_argument("--n-flows", type=int, default=6)
    ap.add_argument("--mean-hidden", type=int, default=64)
    ap.add_argument("--det-hidden", type=int, default=128)
    ap.add_argument("--shape-skip", action="store_true",
                    help="ABLATION FIX (worktree): add a DIRECT intrinsic-shape (e1/e2_input_p) skip "
                         "into the mean head, bypassing the DeepSets trunk so the shape->response "
                         "mapping is not smeared. Off = baseline V2.")
    ap.add_argument("--shape-skip-hidden", type=int, default=0,
                    help="hidden width of the shape-skip head (0 = direct linear map).")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=421)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="joint")
    ap.add_argument("--true-re-min", type=float, default=0.3,
                    help="acceptance TRAIN cut: keep primary TRUE size Re_input_p > this.")
    ap.add_argument("--true-mag-max", type=float, default=26.0,
                    help="acceptance TRAIN cut: keep primary TRUE mag r_input_p < this (r_input_p IS the mag).")
    ap.add_argument("--no-true-cut", action="store_true",
                    help="disable the true-property primary cut (train on the full source-selected population).")
    ap.add_argument("--shear-both", action="store_true",
                    help="shear BOTH primary and neighbour in the response contexts (prototype behaviour). "
                         "Default shears the PRIMARY ONLY so the flow learns R_self alone (reframe firewall).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.outdir, exist_ok=True)
    print(f"device={device}  outdir={args.outdir}", flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    true_cut = None if args.no_true_cut else (args.true_re_min, args.true_mag_max)
    primary_only = not args.shear_both
    print(f"true_cut={true_cut}  primary_only_shear={primary_only}", flush=True)

    # ---- load ----
    print("Loading streams (case-disjoint train/val)...", flush=True)
    Ftr_df, Fva_df = load_stream(args.flow_catalogue, "flow", args.max_case, args.train_case_max,
                                 args.max_rows_flow, args.max_rows_val, args.seed, true_cut=true_cut)
    Dtr_df, Dva_df = load_stream(args.det_catalogue, "det",
                                 args.det_max_case if args.det_max_case is not None else args.max_case,
                                 args.det_train_case_max if args.det_train_case_max is not None else args.train_case_max,
                                 args.max_rows_det, args.max_rows_val, args.seed + 1, true_cut=true_cut)
    if args.target_npz is None and "delta_et1" not in Ftr_df.columns:
        raise SystemExit("--target-npz is REQUIRED with a g=0 det_meas flow catalogue (no per-object "
                         "delta_et): pass the firewall-clean half-shear self-response lookup from "
                         "compute_deltaet_target.py.")

    # ---- sanity numbers ----
    r_self_global = (float(Ftr_df["delta_et1"].to_numpy(float).mean() / GAMMA_SELF)
                     if "delta_et1" in Ftr_df.columns else np.nan)   # set from --target-npz below
    det_frac = float(Dtr_df["detected"].astype(float).mean())
    print(f"SANITY  <R_self>(flow train)={r_self_global:.4f}  det_frac(det train)={det_frac:.4f}", flush=True)
    assert 0.05 < det_frac < 0.95, f"det_frac out of range: {det_frac}"

    # ---- fit preprocessors on the UNION of the two train streams (delta=0 features) ----
    def base_feat_frame(df):
        intr = {}
        intr["e1p"], intr["e2p"] = intrinsic_shape(df, "p")
        intr["e1s"], intr["e2s"] = intrinsic_shape(df, "s")
        f, nbg = shifted_feature_frame(df, intr, (1.0, 0.0), 0.0)
        return f, nbg

    Ff0, Fnbg = base_feat_frame(Ftr_df)
    Df0, Dnbg = base_feat_frame(Dtr_df)
    union_primary = pd.concat([Ff0[PRIMARY_FEATURES], Df0[PRIMARY_FEATURES]], ignore_index=True)
    primary_pre = TabularPreprocessor.fit(union_primary, PRIMARY_FEATURES, add_missing_indicators=True)
    nbr_raw = [Ff0.loc[Fnbg, NEIGHBOR_FEATURES].to_numpy(np.float32),
               Df0.loc[Dnbg, NEIGHBOR_FEATURES].to_numpy(np.float32)]
    nbr_std = SetFeatureStandardizer.fit(nbr_raw, NEIGHBOR_FEATURES)
    target_std = TargetStandardizer.fit(Ff0, MEASUREMENT_TARGETS)
    sc = (float(target_std.scales[0]), float(target_std.scales[1]))   # shape-component scales (dims 0,1)
    sc23 = (float(target_std.scales[2]), float(target_std.scales[3]))  # mag, log-size scales (dims 2,3) for R_theta
    primary_dim = primary_pre.output_dim
    neighbor_dim = nbr_std.dim
    print(f"primary_dim={primary_dim} neighbor_dim={neighbor_dim} target_scales={sc}", flush=True)

    # ---- response-target bins ----
    dist_edges = None
    crowd_edges = None
    if args.target_npz:
        # PRODUCTION-GUIDED path: per-bin self-response is an isolated-separated snc LOOKUP grid
        # Rsim[flux, size, blend] (blend 0=isolated + distance bins), both-legs-free.
        tz = np.load(args.target_npz)
        Rsim = tz["Rsim"].astype(float)
        flux_edges = tz["edges_flux"].astype(float)
        size_edges = tz["edges_size"].astype(float)
        _cc = tz["crowd_col"] if "crowd_col" in tz.files else ""
        crowd_col = _cc.item() if hasattr(_cc, "item") and getattr(_cc, "shape", None) == () else str(_cc)
        if crowd_col:                                    # r_blend / nbr_flux crowd-axis target (V1-style)
            crowd_edges = tz["edges_crowd"].astype(float)
            print(f"  CROWD-TYPE target: 3rd axis = {crowd_col}, edges={np.round(crowd_edges,4).tolist()}", flush=True)
        else:
            dist_edges = tz["edges_dist"].astype(float)  # distance quantile edges for BLENDED gals
        nf, ns, nblend = Rsim.shape
        nbins = Rsim.size
        r_self_global = float(tz["global_R"]) if "global_R" in tz.files else float(np.nanmean(Rsim))
        shape_target = np.where(np.isfinite(Rsim), Rsim, r_self_global).reshape(-1)   # C-order
        bin_tgt = torch.as_tensor(shape_target, dtype=torch.float32, device=device)
        print(f"TARGET-NPZ {args.target_npz}: grid ({nf}x{ns}x{nblend}) nbins={nbins} "
              f"<R>={r_self_global:.4f}  decorrelate={args.decorrelate}", flush=True)
        print(f"  Rsim (flux x size x [iso|dist]) = {np.round(shape_target, 4).tolist()}", flush=True)
    else:
        flux = Ftr_df["r_input_p"].to_numpy(float)
        size = Ftr_df["Re_input_p"].to_numpy(float)
        flux_edges = np.quantile(flux, np.linspace(0, 1, args.n_flux + 1)); flux_edges[0] -= 1e-6; flux_edges[-1] += 1e-6
        size_edges = np.quantile(size, np.linspace(0, 1, args.n_size + 1)); size_edges[0] -= 1e-6; size_edges[-1] += 1e-6
        nf, ns = args.n_flux, args.n_size
        nbins = nf * ns
        fi = np.clip(np.digitize(flux, flux_edges) - 1, 0, nf - 1)
        si = np.clip(np.digitize(size, size_edges) - 1, 0, ns - 1)
        bidx = fi * ns + si
        rself = Ftr_df["delta_et1"].to_numpy(float) / GAMMA_SELF
        shape_target = np.full(nbins, r_self_global)
        for k in range(nbins):
            m = bidx == k
            if m.any():
                shape_target[k] = float(rself[m].mean())
        bin_tgt = torch.as_tensor(shape_target, dtype=torch.float32, device=device)
        print(f"shape R_self target per (flux x size) bin ({nf}x{ns}):", np.round(shape_target, 4).tolist(), flush=True)

    # ---- R_theta orientation-coupling target (dims 2,3), on the SAME grid as the shape target ----
    theta_size_grid = theta_mag_grid = None
    if args.lam_theta > 0.0:
        if not args.theta_target_npz:
            raise SystemExit("--lam-theta>0 requires --theta-target-npz (build_theta_coupling_target.py)")
        if dist_edges is None and crowd_edges is None:
            raise SystemExit("--lam-theta needs the 3D (flux x size x blend) --target-npz grid")
        tt = np.load(args.theta_target_npz)
        theta_size_grid = tt["coupling_size"].astype(np.float32)
        theta_mag_grid = tt["coupling_mag"].astype(np.float32)
        if theta_size_grid.shape != (nf, ns, nblend):
            raise SystemExit(f"theta grid {theta_size_grid.shape} != shape grid {(nf, ns, nblend)}; "
                             "rebuild the coupling target from the SAME --target-npz")
        print(f"THETA-TARGET {args.theta_target_npz}: b_size {float(np.nanmin(theta_size_grid)):+.3f}.."
              f"{float(np.nanmax(theta_size_grid)):+.3f} (global {float(tt['b_size_global']):+.3f}), "
              f"b_mag~{float(tt['b_mag_global']):+.4f}, lam_theta={args.lam_theta}", flush=True)

    z = np.load(args.btrue_npz)
    mag_edges = z["mag_edges"]
    det_flux_edges = det_size_edges = det_dist_edges = None
    if args.btrue_grid:
        # BLEND-RESOLVED detection supervision: grid_b[flux, size, blend], blend 0=isolated + dist bins.
        # Per-object bin (det branch) = (fi*ns+si)*nblend+bi, C-order matching grid_b.reshape(-1).
        grid_b = z["grid_b"].astype(float)                       # (nf, ns, nblend)
        det_flux_edges = z["grid_edges_flux"].astype(float)
        det_size_edges = z["grid_edges_size"].astype(float)
        det_dist_edges = z["grid_edges_dist"].astype(float)
        global_b = float(z["global_b"])
        nmb = grid_b.size
        b_true_np = np.where(np.isfinite(grid_b), grid_b, global_b).reshape(-1)   # C-order; NaN->global
        b_true = torch.as_tensor(b_true_np, dtype=torch.float32, device=device)
        print(f"selection b_true BLEND-RESOLVED grid {grid_b.shape} nbins={nmb} global_b={global_b:.4f}\n"
              f"  edges flux/size/dist={np.round(det_flux_edges,3).tolist()}/"
              f"{np.round(det_size_edges,3).tolist()}/{np.round(det_dist_edges,3).tolist()}\n"
              f"  grid_b%={np.round(b_true_np*100,2).tolist()}", flush=True)
    else:
        b_true_np = z["mag_b"]
        nmb = len(b_true_np)
        b_true = torch.as_tensor(b_true_np, dtype=torch.float32, device=device)
        print(f"selection b_true per mag bin ({nmb}): {np.round(b_true_np, 4).tolist()}  global_b={float(z['global_b']):.4f}",
              flush=True)

    # ---- precompute shifted-context feature tensors on device (reused for all trainings) ----
    print("Precomputing shifted-context feature tensors...", flush=True)
    t0 = time.time()
    Ftr = precompute_stream(Ftr_df, "flow", primary_pre, nbr_std, args.delta, device,
                            target_std=target_std, flux_edges=flux_edges, size_edges=size_edges,
                            dist_edges=dist_edges, decorrelate=args.decorrelate, primary_only=primary_only,
                            theta_size_grid=theta_size_grid, theta_mag_grid=theta_mag_grid, crowd_edges=crowd_edges)
    Fva = precompute_stream(Fva_df, "flow", primary_pre, nbr_std, args.delta, device,
                            target_std=target_std, flux_edges=flux_edges, size_edges=size_edges,
                            dist_edges=dist_edges, decorrelate=args.decorrelate, primary_only=primary_only,
                            theta_size_grid=theta_size_grid, theta_mag_grid=theta_mag_grid, crowd_edges=crowd_edges)
    Dtr = precompute_stream(Dtr_df, "det", primary_pre, nbr_std, args.delta, device, mag_edges=mag_edges,
                            det_flux_edges=det_flux_edges, det_size_edges=det_size_edges, det_dist_edges=det_dist_edges,
                            primary_only=primary_only)
    Dva = precompute_stream(Dva_df, "det", primary_pre, nbr_std, args.delta, device, mag_edges=mag_edges,
                            det_flux_edges=det_flux_edges, det_size_edges=det_size_edges, det_dist_edges=det_dist_edges,
                            primary_only=primary_only)
    print(f"  precompute done in {time.time() - t0:.1f}s", flush=True)

    common = dict(Ftr=Ftr, Fva=Fva, Dtr=Dtr, Dva=Dva, sc=sc, sc23=sc23, args=args,
                  primary_dim=primary_dim, neighbor_dim=neighbor_dim,
                  bin_tgt=bin_tgt, nbins=nbins, b_true=b_true, nmb=nmb, device=device)

    # ---- main model (joint, or flow-only when --flow-only) ----
    _mode = "flow_only" if args.flow_only else "joint"
    _lam_s = 0.0 if args.flow_only else args.lam_s
    _lam_d = 0.0 if args.flow_only else args.lam_d
    print(f"MAIN MODEL mode={_mode}  lam_r={args.lam_r} lam_s={_lam_s} lam_d={_lam_d} lam_theta={args.lam_theta}", flush=True)
    joint, joint_hist, _ = train_one(_mode, lam_r=args.lam_r, lam_s=_lam_s, lam_d=_lam_d,
                                     epochs=args.epochs, tag="joint", lam_theta=args.lam_theta, **common)

    # ---- single-head baselines (V3) ----
    baselines = {}
    if not args.no_baselines:
        fo, fo_hist, _ = train_one("flow_only", lam_r=args.lam_r, lam_s=0.0, lam_d=0.0,
                                   epochs=args.epochs, tag="flow_only", **common)
        do_, do_hist, _ = train_one("det_only", lam_r=0.0, lam_s=args.lam_s, lam_d=args.lam_d,
                                    epochs=args.epochs, tag="det_only", **common)
        baselines = {"flow_only": (fo, fo_hist), "det_only": (do_, do_hist)}

    # ---- 2x2 lambda sensitivity (globals only; NOT tuned on any m) ----
    sens = {}
    if not args.no_sensitivity:
        for lr_ in (args.lam_r, 2 * args.lam_r):
            for ls_ in (args.lam_s, 2 * args.lam_s):
                key = f"lr{int(lr_)}_ls{int(ls_)}"
                if abs(lr_ - args.lam_r) < 1e-9 and abs(ls_ - args.lam_s) < 1e-9:
                    m = joint  # reuse main run for the (50,50) corner
                else:
                    m, _, _ = train_one("joint", lam_r=lr_, lam_s=ls_, lam_d=args.lam_d,
                                        epochs=args.sens_epochs, tag=f"sens_{key}", **common)
                ef = eval_flow(m, Fva, sc, args.delta, bin_tgt, nbins, args.batch_size)
                ed = eval_det(m, Dva, args.delta, b_true, nmb, args.batch_size)
                sens[key] = {"R_global": ef["R_global"], "dP_global": ed["dP_global"]}
                print(f"SENS {key}: <R>={ef['R_global']:+.4f} <dP>={ed['dP_global']:+.4f}", flush=True)

    # ---- final validation V1-V4 on the JOINT model ----
    print("\n===== VALIDATION (held-out cases) =====", flush=True)
    ef = eval_flow(joint, Fva, sc, args.delta, bin_tgt, nbins, args.batch_size)
    ed = eval_det(joint, Dva, args.delta, b_true, nmb, args.batch_size)
    ef_empty = eval_flow(joint, Fva, sc, args.delta, bin_tgt, nbins, args.batch_size, empty_set=True)

    # val-data R_self per shape bin (reference truth on the val split)
    if args.target_npz or "delta_et1" not in Fva_df.columns:
        # no per-object empirical response label; the bin target IS the snc-lookup truth
        rself_bin_val = shape_target.copy()
        rself_val_global = r_self_global
    else:
        vflux = Fva_df["r_input_p"].to_numpy(float); vsize = Fva_df["Re_input_p"].to_numpy(float)
        vfi = np.clip(np.digitize(vflux, flux_edges) - 1, 0, nf - 1)
        vsi = np.clip(np.digitize(vsize, size_edges) - 1, 0, ns - 1)
        vbin = vfi * ns + vsi
        vrself = Fva_df["delta_et1"].to_numpy(float) / GAMMA_SELF
        rself_bin_val = np.full(nbins, np.nan)
        for k in range(nbins):
            m = vbin == k
            if m.any():
                rself_bin_val[k] = float(vrself[m].mean())
        rself_val_global = float(vrself.mean())

    # V3 baseline comparison
    v3 = {}
    if baselines:
        fo, _ = baselines["flow_only"]; do_, _ = baselines["det_only"]
        ef_fo = eval_flow(fo, Fva, sc, args.delta, bin_tgt, nbins, args.batch_size)
        ed_do = eval_det(do_, Dva, args.delta, b_true, nmb, args.batch_size)
        v3 = {"joint_nll": ef["nll"], "flowbase_nll": ef_fo["nll"],
              "joint_R": ef["R_global"], "flowbase_R": ef_fo["R_global"],
              "joint_bce": ed["bce"], "detbase_bce": ed_do["bce"],
              "joint_dP": ed["dP_global"], "detbase_dP": ed_do["dP_global"]}

    # ---- reporting ----
    def _fmt(a):
        return np.round(np.asarray(a, float), 4).tolist()

    lines = []
    lines.append("SBSI joint measurement+detection forward model (reframe, GOALS.md; experimental -- not yet certified)")
    lines.append(f"seed={args.seed} delta={args.delta} lam_r={args.lam_r} lam_s={args.lam_s} lam_d={args.lam_d}")
    lines.append(f"flow train/val rows={Ftr['_n']}/{Fva['_n']}  det train/val rows={Dtr['_n']}/{Dva['_n']}")
    lines.append(f"SANITY <R_self>(train)={r_self_global:.4f}  det_frac(train)={det_frac:.4f}  <R_self>(val)={rself_val_global:.4f}")
    lines.append("")
    lines.append("V1 SHAPE RESPONSE (isolated+blend self-response), joint model, val:")
    lines.append(f"   <R_model> global = {ef['R_global']:+.4f}   (full-pop <R_self> = {rself_val_global:.4f} "
                 f"is NOT the true-cut mean; ignore its rel.err)")
    # HONEST target: R_sim count-weighted by the ACTUAL (true-cut) val population, not the full-pop
    # global. Under the true cut (bright+large) the population mean response is much higher than the
    # full-pop 0.281, so comparing to 0.281 is meaningless. This is the shape-only global closure.
    _cf = ef["bin_count"].astype(float)
    _tw = float(np.nansum(_cf * shape_target) / max(_cf.sum(), 1.0))
    lines.append(f"   <R_sim> count-weighted by the TRUE-CUT val population = {_tw:+.4f}"
                 f"   shape-only global m = {(ef['R_global'] / _tw - 1) * 100:+.2f}%"
                 f"   (no R_blend/R_detect yet; this is HALF-SHEAR closure, not the constgold deliverable)")
    lines.append(f"   R_model per (flux x size) bin: {_fmt(ef['R_bin'])}")
    lines.append(f"   R_self  per bin (val truth):   {_fmt(rself_bin_val)}")
    lines.append(f"   R_self  per bin (train target): {_fmt(shape_target)}")
    lines.append("")
    # dP_global == dp.mean() over EVERY val object == COUNT-weighted (population) global.
    # Compare it apples-to-apples with b_true weighted by the SAME val mag histogram, and
    # with the npz global_b (which is a differently-weighted detected-population number).
    cnt = ed["bin_count"].astype(float)
    b_true_popw = float(np.nansum(cnt * b_true_np) / max(cnt.sum(), 1.0))
    b_true_unw = float(np.mean(b_true_np))
    lines.append("V2 SELECTION RESPONSE dP(s=1)/dgamma, joint model, val:")
    lines.append(f"   <dP/dg> count-weighted (val population) = {ed['dP_global']:+.4f}")
    lines.append(f"   b_true  count-weighted by val mag-hist  = {b_true_popw:+.4f}   "
                 f"(model/target = {ed['dP_global'] / b_true_popw if b_true_popw else float('nan'):.2f}x)")
    lines.append(f"   b_true  global_b (npz, detected-pop)    = {float(z['global_b']):+.4f}")
    lines.append(f"   b_true  unweighted per-bin mean         = {b_true_unw:+.4f}")
    lines.append(f"   dP/dg per mag bin:   {_fmt(ed['dP_bin'])}")
    lines.append(f"   b_true per mag bin:  {_fmt(b_true_np)}")
    lines.append("")
    lines.append("V3 NO-TRADE-OFF (joint vs single-head baselines, val):")
    if v3:
        lines.append(f"   NLL  joint={v3['joint_nll']:.4f}  flow_only_baseline={v3['flowbase_nll']:.4f}"
                     f"   (delta={v3['joint_nll'] - v3['flowbase_nll']:+.4f})")
        lines.append(f"   <R>  joint={v3['joint_R']:+.4f}  flow_only_baseline={v3['flowbase_R']:+.4f}")
        lines.append(f"   BCE  joint={v3['joint_bce']:.4f}  det_only_baseline={v3['detbase_bce']:.4f}"
                     f"   (delta={v3['joint_bce'] - v3['detbase_bce']:+.4f})")
        lines.append(f"   <dP> joint={v3['joint_dP']:+.4f}  det_only_baseline={v3['detbase_dP']:+.4f}")
    else:
        lines.append("   (baselines skipped)")
    lines.append("")
    lines.append("V4 ISOLATED vs BLENDED (neighbour present vs empty set), joint model, val:")
    lines.append(f"   <R_model> present(mask=1) = {ef['R_global']:+.4f}   empty(mask=0) = {ef_empty['R_global']:+.4f}"
                 f"   diff = {ef['R_global'] - ef_empty['R_global']:+.4f}")
    lines.append("")
    if sens:
        lines.append("2x2 lambda sensitivity (global <R>, <dP>):")
        for k, v in sens.items():
            lines.append(f"   {k}: <R>={v['R_global']:+.4f}  <dP>={v['dP_global']:+.4f}")
    report = "\n".join(lines)
    print("\n" + report, flush=True)

    # ---- persist ----
    ckpt_path = os.path.join(args.outdir, f"forward_{args.tag}_joint.pt")
    torch.save({
        "state_dict": joint.state_dict(),
        "model_config": dict(target_dim=len(MEASUREMENT_TARGETS), primary_dim=primary_dim, neighbor_dim=neighbor_dim,
                             context_dim=args.context_dim, set_hidden_dim=args.set_hidden_dim,
                             flow_hidden_dim=args.flow_hidden_dim, flow_layers=args.flow_layers,
                             n_flows=args.n_flows, mean_hidden=args.mean_hidden, det_hidden=args.det_hidden,
                             shape_skip_dim=_shape_skip_config(args)[0],
                             shape_skip_idx=_shape_skip_config(args)[1],
                             shape_skip_hidden=_shape_skip_config(args)[2]),
        "primary_preprocessor": primary_pre.to_state(),
        "neighbor_preprocessor": nbr_std.to_state(),
        "target_transform": target_std.to_state(),
        "primary_features": PRIMARY_FEATURES, "neighbor_features": NEIGHBOR_FEATURES,
        "shape_targets": SHAPE_TARGETS, "measurement_targets": MEASUREMENT_TARGETS, "delta": args.delta,
        "flux_edges": flux_edges, "size_edges": size_edges, "mag_edges": mag_edges,
        "metadata": {"reframe": "joint measurement+detection forward model (GOALS.md, 2026-07-22)",
                     "primary_only_shear": bool(primary_only), "true_cut": true_cut,
                     "firewall": "trained on half-shear only; constgold never read"},
    }, ckpt_path)

    val_path = os.path.join(args.outdir, f"validation_{args.tag}.txt")
    with open(val_path, "w") as fh:
        fh.write(report + "\n")
    npz_path = os.path.join(args.outdir, f"validation_{args.tag}.npz")
    np.savez(
        npz_path,
        R_global=ef["R_global"], R_bin=ef["R_bin"], R_bin_count=ef["bin_count"],
        rself_val_global=rself_val_global, rself_bin_val=rself_bin_val, shape_target_train=shape_target,
        dP_global=ed["dP_global"], dP_bin=ed["dP_bin"], dP_bin_count=ed["bin_count"],
        b_true=b_true_np, b_true_popw=b_true_popw, b_true_unw=b_true_unw,
        mag_edges=mag_edges, global_b=float(z["global_b"]),
        flux_edges=flux_edges, size_edges=size_edges,
        R_empty_global=ef_empty["R_global"], R_empty_bin=ef_empty["R_bin"],
        det_frac=det_frac, r_self_global_train=r_self_global,
        joint_val_nll=ef["nll"], joint_val_bce=ed["bce"],
        v3=json.dumps(v3), sensitivity=json.dumps(sens),
        joint_history=json.dumps(joint_hist),
    )
    print(f"\nSaved checkpoint: {ckpt_path}", flush=True)
    print(f"Saved validation: {val_path}", flush=True)
    print(f"Saved metrics:    {npz_path}", flush=True)
    print("JOINT_FORWARD_DONE", flush=True)


if __name__ == "__main__":
    main()
