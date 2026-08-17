"""FLOW #2 -- the blending-response flow (`Gold-V3.md`): a firewalled drop-in for BlendEMU.

WHAT IT MODELS
--------------
    p(primary's measured shape | primary true props, ONE neighbour's true props, separation)

trained on the g=0 leg, where the rendered shape IS the intrinsic shape. The deliverable is not the
density but its NEIGHBOUR-shear derivative: shear the neighbour's intrinsic shape by delta, hold the
primary fixed, and read the mean head's shift. That derivative is `R_blend` for the pair, and it is
summed over a primary's neighbours exactly as `build_blend_lookup.py` sums the emulator today.

WHY THIS CAN BEAT THE EMULATOR WHERE IT FAILS
---------------------------------------------
BlendEMU under-predicts the per-pair response by ~-37% below 1" (WORKLOG 2026-07-28i..l; reconfirmed
2026-08-02c at the fiducial tag: -36.0% below 0.5", -38.6% at 0.5-1"). Neither in-domain retraining
nor close-pair loss weighting moves it, so it is a representational limit. Its features are seven
scalars -- `Re_input_p/s`, `r_input_p/s`, `sersic_n_input_p/s`, `distance` -- with **no shape
information at all**. Flow #2 conditions on both galaxies' ORIENTED intrinsic shapes and trains on
our own detected population with our own estimator, which is the standing explanation for the
deficit: our labels are conditioned on both objects being detected, and at sub-arcsecond separation
detection depends on the relative orientation the emulator cannot see, so the population it averages
over is not ours (`Gold-V3.md` "The live mechanism").

Not a guarantee. The emulator's per-pair accuracy is good from 1.5" outward, and the honest test is
whether flow #2 fixes the close pairs WITHOUT degrading those -- exactly the trade the emulator's
close-pair weighting failed at (-37.2% at 4x weight while 1-2" degraded from -5.4% to -10.1%).

ONE DELIBERATE DEPARTURE FROM THE `Gold-V3.md` SPEC: PER-PAIR, NOT BINNED, SUPERVISION
--------------------------------------------------------------------------------------
That document specifies a binned response target, mirroring flow #1, and estimates "~100 bins over
4.8M rows ⇒ S/N ≈ 12 per bin". **That arithmetic is wrong by more than an order of magnitude.** It
compares a signal of `R_blend*g ≈ 6e-4` against a per-bin error of `5e-5`, but the measured per-pair
label scatter is `std = 3.93` in response units, i.e. `0.197` in `de` units, so 48k rows per bin give
`8.9e-4`, not `5e-5`. The true per-bin S/N at that design is ~0.7.

The binning is also unnecessary here. Flow #1 needs it because its target is a POPULATION MEAN that
only exists per cell. Flow #2's label exists PER PAIR and is unbiased, so a plain squared-error
regression converges to `E[truth | features]`, which is the conditional mean we want -- with the
network's smoothness pooling information across the feature space instead of a hand-drawn grid. That
removes the bin-design problem rather than solving it.

What does NOT go away is the statistical floor ON THE LABELS. On the full ap7 sample the label mean
is determined to ~3% globally and ~6% below 1"; no *measurement* against these labels can resolve
better than that, so a reported agreement tighter than the sem is not a demonstrated agreement.

A CHAIN-RULE ARGUMENT THAT LOOKED LIKE AN ESCAPE FROM THAT FLOOR -- AND DOES NOT SURVIVE MEASUREMENT
---------------------------------------------------------------------------------------------------
The response factorises:

    R_blend = d(mu_p)/d(g_s) = [ d(mu_p)/d(e_s) ] * [ d(e_s)/d(g_s) ]

The second factor is the analytic Mobius Jacobian -- exact, no data needed. The first is a property
of the g=0 density alone: how the primary's measured shape depends on the neighbour's intrinsic
shape, i.e. blending contamination, measurable on tens of millions of g=0 rows at ordinary
shape-noise SNR rather than through a `de/|g|` difference inflated by 1/0.05. On that basis I claimed
the NLL supplies the response's dominant channel at high precision and the noisy labels merely
calibrate it.

**MEASURED, AND IT IS FALSE** (job 15478188, `--response-weight 0`, i.e. NLL only). The
density-derived response is **70-85% too low in every separation bin and almost completely flat**:
0.0132, 0.0087, 0.0069, 0.0062, 0.0056, 0.0052, 0.0050, 0.0049 from <0.5" out to 7", against a truth
that runs 0.0721 down to 0.0042 with a bump at 2-3". The mathematics is right; the inference about
training was wrong.

The reason is a signal-to-signal problem, not a signal-to-noise one. `d(mu_p)/d(e_s) ~ 0.04` implies a
correlation of only ~0.035 between the neighbour's shape and the primary's measured shape, against
~0.8 for the primary's OWN shape. Maximum likelihood spends its capacity on the strong direction and
under-fits the weak one -- the same conditional-mean under-fit `ConditionalMeanFlow` exists to
counter, and the explicit mean head is not enough on its own here.

**So the response supervision carries the load, and `--response-weight` is the load-bearing knob, not
a regulariser.** The label floor above therefore does bind. Stated plainly because the earlier version
of this docstring claimed the opposite.

The causal reading of `d(mu_p)/d(e_s)` is still legitimate in this suite -- intrinsic shapes are drawn
independently per object, so `e_s` has no confounder among the conditioning variables -- and the
structural advantage over BlendEMU stands: the emulator has NO shape information at all, neither
galaxy's, so that channel is closed to it entirely. What does not stand is the claim that the channel
comes for free from the density.

FIREWALL: trained on half-shear legs only. constgold is never read here, so this model can be scored
on constgold `m` without circularity -- the same standing that BlendEMU has.
"""
from __future__ import annotations

import numpy as np
import torch

from .measurement_model import ConditionalMeanFlow

# Raw catalogue columns the pair set carries. Order is FIXED -- the checkpoint stores
# standardization keyed to this order, and `build_blend_pairset.py` writes the same names.
PRIMARY = ["r_input_p", "Re_input_p", "sersic_n_input_p", "e1_input_rot0_p", "e2_input_rot0_p"]
NEIGHBOUR = ["r_input_s", "Re_input_s", "sersic_n_input_s", "e1_input_rot0_s", "e2_input_rot0_s"]
GEOMETRY = ["distance"]
RAW = PRIMARY + NEIGHBOUR + GEOMETRY

# The columns the shear map acts on. Everything else is shear-invariant, so the response is carried
# entirely by these; if they were absent the shifted and unshifted contexts would be IDENTICAL and
# the response would be exactly zero for any parameters (`Gold-V3.md`, "the blocker").
#
# TWO channels, not one (added 2026-08-02i for the MERGED flow). Shifting the NEIGHBOUR's shape gives
# the blending response; shifting the PRIMARY's own shape gives the self response -- the quantity
# flow #1 exists to produce. Both channels have always been present in this model's conditioning;
# only the neighbour one was ever supervised or read.
SHAPE_S = ("e1_input_rot0_s", "e2_input_rot0_s")
SHAPE_P = ("e1_input_rot0_p", "e2_input_rot0_p")
SHAPE_COLUMNS = {"s": SHAPE_S, "p": SHAPE_P}

TARGETS = ["measured_ngmix_g1", "measured_ngmix_g2"]

DIST_FLOOR = 0.01      # arcsec; the catalogue reaches 0.0045" and log needs a floor

# CROWDING BLOCK (added 2026-08-02j). Constants copied from `scripts/build_crowding_lookup.py` so
# flow #2's crowding scalars are the SAME quantity flow #1 conditions on -- same shells, same zero
# point, same noise normalisation -- rather than a lookalike that happens to correlate.
#
# WHY THESE EXIST HERE AT ALL. Step 2 (WORKLOG 2026-08-02i) measured a pair model getting the
# crowding trend BACKWARDS on the self response: the truth falls 15% from k=1 to k=6-9 while the
# model rises 1%. That is not an optimisation shortfall -- a pair model reads the response from a
# context naming ONE neighbour and has no input that counts them. Flow #1 hit the identical wall
# (+13% to -66% across crowding) and solved it with exactly these scalars.
#
# All four are functions of MAGNITUDES, DISTANCES and COUNTS only -- no shape enters -- so they are
# invariant under either shear shift and cannot open a third response channel. That invariance is
# what `test_crowding_features_are_shear_invariant` checks, because a leak here would be silent.
PIXEL_RMS, PIXEL_SIZE, PSF_FWHM, MOFFAT_BETA, ZERO_MAG = 0.312, 0.2, 0.73, 2.224, 30.0
NEAR_ARCSEC = 3.0                 # inner shell 0-3"; the pair set's own aperture (7") is the outer
CROWDING = ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max", "log_k"]


def aperture_rms():
    """Noise scale the shell fluxes are expressed in. Same expression as build_crowding_lookup."""
    factor = np.sqrt((2 ** (1 / (MOFFAT_BETA - 1)) - 1) / (2 ** (1 / MOFFAT_BETA) - 1)) / 2
    psf_size = PSF_FWHM * factor
    return PIXEL_RMS * (psf_size / PIXEL_SIZE) ** 2 * np.pi


def crowding_columns(frame):
    """Per-PRIMARY crowding scalars, derived from the pair set's own neighbour rows.

    The pair set already holds every annotated neighbour of a primary inside 7" as a separate row
    keyed by `pid`, so the shell sums can be formed by grouping those rows -- no second pass over the
    130 GB legs, and the features describe exactly the neighbours this model is shown.

    Returns a dict of four arrays, one value per ROW (constant across a primary's rows).
    """
    pid = np.asarray(frame["pid"], dtype=np.int64)
    mag_s = np.asarray(frame["r_input_s"], dtype=np.float64)
    dist = np.asarray(frame["distance"], dtype=np.float64)
    n = int(pid.max()) + 1 if len(pid) else 0

    flux = 10.0 ** (-0.4 * (mag_s - ZERO_MAG))
    near = dist < NEAR_ARCSEC
    f_near = np.bincount(pid[near], weights=flux[near], minlength=n)
    f_far = np.bincount(pid[~near], weights=flux[~near], minlength=n)
    f_max = np.zeros(n, dtype=np.float64)
    np.maximum.at(f_max, pid, flux)          # brightest single neighbour anywhere in the aperture
    # Prefer the STORED neighbour count. Recomputing it from the rows present would silently disagree
    # with the stored value on any subsampled frame, and `log_k` would then describe the sample rather
    # than the galaxy.
    if "k" in frame:
        kk = np.asarray(frame["k"], dtype=np.float64)
    else:
        kk = np.bincount(pid, minlength=n).astype(np.float64)[pid]

    ar = aperture_rms()
    return {
        "nbr_flux_near": np.log10(1.0 + f_near[pid] / ar),
        "nbr_flux_far": np.log10(1.0 + f_far[pid] / ar),
        "nbr_flux_max": np.log10(1.0 + f_max[pid] / ar),
        "log_k": np.log(np.maximum(kk, 1.0)),
    }


def feature_names(derived=True, crowding=False):
    """Column names of the standardized context, in order."""
    names = []
    for c in RAW:
        names.append("log_" + c if c in ("Re_input_p", "Re_input_s", "sersic_n_input_p",
                                         "sersic_n_input_s", "distance") else c)
    if derived:
        # Deterministic functions of the columns above -- no new information, but they are the
        # physically natural variables (flux ratio; separation in units of the primary's size) and a
        # small MLP finds the close-pair structure far faster with them present. Neither depends on
        # either galaxy's SHAPE, so both are invariant under the shear shift and cannot leak a
        # second, unintended response channel.
        names += ["dmag_s_minus_p", "log_dist_over_Re_p"]
    if crowding:
        # Appended LAST so every existing checkpoint's column indices are unchanged; the shape
        # channels are found by name anyway, but keeping the prefix stable makes old and new
        # standardizers directly comparable.
        names += list(CROWDING)
    return names


def build_features(frame, derived=True, crowding=False):
    """Raw pair columns -> the (N, D) design matrix, BEFORE standardization."""
    cols = []
    for c in RAW:
        v = np.asarray(frame[c], dtype=np.float64)
        if c in ("Re_input_p", "Re_input_s", "sersic_n_input_p", "sersic_n_input_s"):
            v = np.log(np.clip(v, 1e-3, None))
        elif c == "distance":
            v = np.log(np.clip(v, DIST_FLOOR, None))
        cols.append(v)
    if derived:
        rp = np.asarray(frame["r_input_p"], dtype=np.float64)
        rs = np.asarray(frame["r_input_s"], dtype=np.float64)
        dd = np.clip(np.asarray(frame["distance"], dtype=np.float64), DIST_FLOOR, None)
        rep = np.clip(np.asarray(frame["Re_input_p"], dtype=np.float64), 1e-3, None)
        cols.append(rs - rp)
        cols.append(np.log(dd / rep))
    if crowding:
        cr = (frame if all(c in frame for c in CROWDING) else crowding_columns(frame))
        for c in CROWDING:
            cols.append(np.asarray(cr[c], dtype=np.float64))
    return np.column_stack(cols).astype(np.float32)


def shape_column_indices(derived=True, which="s", crowding=False):
    """Positions of one galaxy's two shape columns inside the design matrix.

    `which` is "s" for the neighbour (the blend-response channel) or "p" for the primary (the
    self-response channel). Defaults to the neighbour so every pre-2026-08-02i caller is unchanged.
    """
    names = feature_names(derived, crowding)
    c1, c2 = SHAPE_COLUMNS[which]
    return [names.index(c1), names.index(c2)]


def apply_shear(e1, e2, g1, g2):
    """Mobius reduced-shear map eps' = (eps + g)/(1 + conj(g) eps), on numpy arrays.

    Same map as `sbs_shear.shear_map.apply_shear_to_ellipticity`, written out in real components so
    it stays cheap on the two columns this module shifts.
    """
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    dr = 1.0 + g1 * e1 + g2 * e2
    di = g1 * e2 - g2 * e1
    nr, ni = e1 + g1, e2 + g2
    den = dr * dr + di * di
    return (nr * dr + ni * di) / den, (ni * dr - nr * di) / den


def shifted_shape_columns(frame, delta, which="s"):
    """Standardization-free shifted shapes for the four finite-difference legs of one channel.

    Returns a dict of (e1, e2) pairs keyed 'e1+', 'e1-', 'e2+', 'e2-', where 'e1+' shears the chosen
    galaxy's intrinsic shape by (+delta, 0) and so on. A CENTRAL difference is used because the
    Mobius map is nonlinear in the shape and a forward difference leaves an O(delta) term that does
    not cancel. `which` selects the neighbour ("s", the blend channel) or the primary ("p", the self
    channel); it defaults to the neighbour so pre-2026-08-02i callers are unchanged.
    """
    c1, c2 = SHAPE_COLUMNS[which]
    e1 = np.asarray(frame[c1], dtype=np.float64)
    e2 = np.asarray(frame[c2], dtype=np.float64)
    out = {}
    for key, (g1, g2) in (("e1+", (delta, 0.0)), ("e1-", (-delta, 0.0)),
                          ("e2+", (0.0, delta)), ("e2-", (0.0, -delta))):
        out[key] = apply_shear(e1, e2, g1, g2)
    return out


class Standardizer:
    """Column-wise (x - mean)/scale, fitted on the training split only."""

    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)

    @classmethod
    def fit(cls, x):
        m = np.nanmean(x, axis=0)
        s = np.nanstd(x, axis=0)
        s = np.where(np.isfinite(s) & (s > 1e-8), s, 1.0)
        return cls(m, s)

    def transform(self, x):
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)

    def to_state(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state(cls, st):
        return cls(st["mean"], st["scale"])


def build_model(context_dim, mean_hidden=256, hidden_dim=128, n_layers=3, n_flows=6,
                derived=True, blind_flow_to_neighbour_shape=True,
                blind_flow_to_primary_shape=False, crowding=False):
    """The flow. Config is returned alongside so the checkpoint can rebuild it exactly.

    The residual flow is kept BLIND to the neighbour's two shape columns so the neighbour-shape
    response lives entirely in the explicit mean head and cannot be re-absorbed or shrunk by the
    density term -- the same device flow #1 uses for the primary's shape
    (`--flow-blind-features e1_input_p e2_input_p`).

    THE PRIMARY'S SHAPE IS BLINDED ONLY IN DUAL MODE (2026-08-02i). In blend-only training it is not
    a response channel, so leaving it visible costs nothing and it genuinely predicts the residual
    scatter -- that is the pre-2026-08-02i default and it is preserved. The moment the SELF response
    is supervised, the same argument that blinds the neighbour's shape applies to the primary's with
    more force, because the self channel is the STRONG one (~0.86 against ~0.14) and is exactly what
    the density term would otherwise absorb. Getting this wrong would not raise; it would quietly
    shrink the larger of the two responses.
    """
    # NOTE `n_layers`, not `condition_layers`: ConditionalAffineFlow takes **kwargs and would
    # SILENTLY IGNORE a misspelled depth argument, leaving the default depth in place with nothing
    # raised. Keep these names matched to that signature.
    cfg = dict(flow_type="mean_affine", target_dim=len(TARGETS), context_dim=int(context_dim),
               mean_hidden=int(mean_hidden), hidden_dim=int(hidden_dim),
               n_layers=int(n_layers), n_flows=int(n_flows))
    drop = []
    if blind_flow_to_neighbour_shape:
        drop += shape_column_indices(derived, "s", crowding)
    if blind_flow_to_primary_shape:
        drop += shape_column_indices(derived, "p", crowding)
    if drop:
        cfg["flow_drop_indices"] = sorted(set(drop))
    kw = dict(cfg)
    kw.pop("flow_type")
    kw["base_flow"] = "affine"
    return ConditionalMeanFlow(**kw), cfg


def response_from_contexts(model, ctx, shifted, scales, delta):
    """Per-pair response in physical (measured-shape) units -- `R_blend` or `R_self`.

    The function is CHANNEL-AGNOSTIC: it reads only the shifted contexts it is handed, so which
    response comes out is decided entirely by which two columns were shifted to build them.

    `ctx` is the unshifted standardized context; `shifted` maps 'e1+','e1-','e2+','e2-' to contexts
    differing from `ctx` ONLY in that channel's two shape columns. The estimator is the trace/2
    responsivity -- the response of measured e1 to an e1-shift averaged with that of measured e2 to
    an e2-shift -- which is exactly the quantity the half-shear label measures, because projecting
    on a shear direction that is isotropically distributed relative to the pair averages the
    response tensor down to its isotropic part.

    Mirrors `r_i` in `train_measurement_model_swa_s1_truecond.epoch_response`, including the factor
    convention, so the two flows' responses are the same kind of number and can be added.
    """
    mu = {k: model._mu(v) for k, v in shifted.items()}
    sc0, sc1 = float(scales[0]), float(scales[1])
    return 0.25 * ((mu["e1+"][:, 0] - mu["e1-"][:, 0]) * sc0
                   + (mu["e2+"][:, 1] - mu["e2-"][:, 1]) * sc1) / delta


def save_blend_flow(path, model, model_config, feat_std, target_mean, target_scale, metadata):
    torch.save({"state_dict": model.state_dict(),
                "model_config": dict(model_config),
                "feature_standardizer": feat_std.to_state(),
                "feature_names": metadata.get("feature_names", []),
                "target_mean": np.asarray(target_mean, dtype=np.float64).tolist(),
                "target_scale": np.asarray(target_scale, dtype=np.float64).tolist(),
                "metadata": dict(metadata)}, path)


def load_blend_flow(path, device="cpu"):
    """Returns (model, feature_standardizer, target_scale, metadata)."""
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=device)
    cfg = dict(ck["model_config"])
    kw = dict(cfg)
    kw.pop("flow_type", None)
    kw["base_flow"] = "affine"
    model = ConditionalMeanFlow(**kw)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return (model, Standardizer.from_state(ck["feature_standardizer"]),
            np.asarray(ck["target_scale"], dtype=np.float64), ck.get("metadata", {}))
