"""The V2.1 deliverable domain: one definition, imported by everything that uses it.

WHY THIS FILE EXISTS. The V2 domain (`primary true mag < 26`, `Re > 0.3`) is a BOX, so every
consumer could spell it as two numbers and they happened to agree. V2.1 replaces the magnitude
limit with a signal-to-noise limit, and S/N is a function of BOTH true magnitude and true size --
a curve in the (mag, Re) plane, not a box. A curve cannot be re-typed in four places without
drifting, and AGENTS.md documents what drift costs: a model scored on a population it was not
trained for reads as a model error (the 7"-capped emulator, +3.84 pt of a 4.79-pt move).

So the flow trainer, the response-target builder, the emulator retrain, and the constgold
evaluators all call `in_domain()` here. If the domain changes, it changes once.

THE V2.1 DOMAIN (owner, 2026-08-04), on the PRIMARY only -- neighbours stay full-population,
which is the deliverable definition in GOALS.md:

  * true size    Re_input_p > 0.5"   (2.5 pixels at 0.2"/pixel; resolution R = 0.474)
  * true S/N     sn_true(mag, Re) > 10

"true size > 2.5" is 2.5 PIXELS. The owner quoted it alongside "resolution factor roughly 0.5",
and that is what fixes the unit: `Re_input_p` is a half-light radius in ARCSEC, the Moffat PSF
half-light radius is 0.527" (below), and

    R = Re^2 / (Re^2 + Re_psf^2)  =  0.474 at Re = 0.50" = 2.5 px
                                 =  0.245 at Re = 0.30" (the V2 cut)

so only the pixel reading reproduces "roughly 0.5". 2.5 arcsec is impossible -- the catalogue's
size cut tops out at 1.5".

WHAT `sn_true` IS. An aperture S/N built from TRUE flux and TRUE size via the standard CCD noise
equation -- sky noise over the aperture PLUS the source's own Poisson noise:

    flux   = 10^(-0.4 (mag - ZERO_MAG))                       [ADU]
    Re_c   = sqrt(Re^2 + PSF_RE^2)                            [PSF-convolved half-light radius]
    N_pix  = pi Re_c^2 / PIXEL_SIZE^2                         [pixels in the half-light ellipse]
    S/N    = flux / sqrt(SN_SKY_VAR * N_pix  +  flux / SN_GAIN)

THE SOURCE TERM IS NOT OPTIONAL, and leaving it out is not a small error. The first version of this
module used the sky-limited form `flux / (PIXEL_RMS sqrt(N_pix))`. Regressed against the
catalogue's own measured S/N it gave a slope of 0.826, not 1 -- i.e. the wrong SHAPE, which no
multiplicative constant can repair -- with the residual sliding monotonically from +0.12 dex at the
faint end to -0.08 at the bright end. That is the signature of bright objects carrying their own
shot noise. Adding `flux / SN_GAIN` moved the slope to 1.004 and the offset to -0.005.

Cutting on TRUE properties is the point: a cut on a measured quantity moves its own boundary with
shear and so manufactures selection bias, which is the effect being measured. A true-property cut
has pass_plus == pass_minus by construction (the NULL in `eval_selection_constgold.py`).

CIRCULARIZED ON PURPOSE. True ellipticity would enter through the aperture axes, but it moves the
S/N by ~1.6% (0.017 mag) at a typical axis ratio -- an order below the 0.097 dex scatter below --
while forcing a fifth column through four consumers, two of which (the emulator's cut list, the
response target's bin axes) are written in (mag, Re) only. Not worth the coupling.

THE TWO CONSTANTS ARE FITTED, NOT GUESSED, and both are physically interpretable:

  * SN_SKY_VAR is the effective sky variance per pixel. It comes out at 5.37 x PIXEL_RMS^2, and
    that factor is the APERTURE CONVENTION: SExtractor's Kron AUTO aperture has ~5.4x the area of
    the half-light ellipse `N_pix` is written in (~2.3x in linear size), which is what a Kron
    aperture should be. It is not a fudge absorbing an unexplained discrepancy.
  * SN_GAIN is the effective gain converting ADU to photo-electrons for the source shot-noise term.

Both are fitted ONCE against `measured_flux_auto / measured_fluxerr_auto` by
`scripts/eval_domain_v21.py`, which reports the slope, offset and residual scatter alongside them,
and are recorded below with the run that produced them. Per AGENTS.md "Numerical Integrity", a
constant derived and reported alongside its error is fine; a constant pasted in unlabelled is not
-- so do not edit these without rerunning that script and updating the provenance block.

WHAT THE 0.097 dex (25%) SCATTER MEANS. It is the spread of REAL measured S/N about the proxy at
fixed true properties -- Sersic profile shape, ellipticity, neighbours, and the noise realisation,
none of which the proxy sees. It does NOT make the cut fuzzy: the cut is on the proxy, which is
exact and shear-independent by construction. It means "S/N > 10" is "EXPECTED S/N > 10", which is
the right object for defining a sample, and is why the cut must never be re-expressed as a cut on
measured S/N.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PIXEL_SIZE", "PIXEL_RMS", "ZERO_MAG", "PSF_FWHM", "MOFFAT_BETA", "PSF_RE",
    "V21_RE_MIN", "V21_SN_MIN", "SN_SKY_VAR", "SN_GAIN",
    "MAG_COLUMN", "RE_COLUMN",
    "moffat_fwhm_to_re", "resolution", "aperture_pixels", "sn_true", "sn_limiting_mag",
    "in_domain", "select_frame", "metadata", "describe",
]

# The PRIMARY's true properties. Named here rather than defaulted per-caller so that "which column
# is the primary" is answered once -- passing the `_s` columns would cut the NEIGHBOUR population,
# which the deliverable explicitly does not do, and would fail silently.
MAG_COLUMN = "r_input_p"
RE_COLUMN = "Re_input_p"

# --- Observing conditions of the FS2/LSST-r simulation -------------------------------------
# These are the sim's own parameters, not choices: they appear identically in the blendemu
# config (`rescale:` block of configs/fs2_lsst_r_extnbr_indom.yaml) and in the RK/COND dicts of
# ~10 scripts. Changing one changes which sim the numbers describe.
PIXEL_SIZE = 0.2      # arcsec / pixel
PIXEL_RMS = 0.312     # sky RMS per pixel, ADU
ZERO_MAG = 30.0       # magnitude zero point
PSF_FWHM = 0.73       # arcsec
MOFFAT_BETA = 2.224


def moffat_fwhm_to_re(fwhm=PSF_FWHM, beta=MOFFAT_BETA):
    """Moffat FWHM -> half-light radius. Mirrors `blendemu.data_utils.moffat_fwhm2Re`."""
    factor = np.sqrt((2 ** (1 / (beta - 1)) - 1) / (2 ** (1 / beta) - 1)) / 2
    return fwhm * factor


PSF_RE = float(moffat_fwhm_to_re())     # 0.5268 arcsec = 2.634 pixels

# --- The domain ----------------------------------------------------------------------------
V21_RE_MIN = 0.5      # arcsec == 2.5 pixels; resolution R = 0.474
V21_SN_MIN = 10.0     # on sn_true()

# --- Fitted S/N constants ------------------------------------------------------------------
# PROVENANCE: fitted by `scripts/eval_domain_v21.py`, Slurm job 15519546, 2026-08-04, on the whole
# g=0 training catalogue `det_meas_crowd_conc_g0.0_train_full.feather` (31,411,766 rows; 4,771,152
# in the calibration sample -- detected, nearest neighbour > 3", 8 < measured S/N < 200).
#
# Fit quality, all printed by that job:
#     free slope   1.0035   (gate: the form is only right if this is 1)
#     offset      -0.0045 dex
#     residual sd  0.0967 dex = 24.9%     <- irreducible; see the docstring
# Stable: an 8-batch smoke fit on 79,742 rows gave 0.52303 / 10.5848, i.e. both constants moved
# under 1% when the sample grew 60x.
#
# DO NOT edit these by hand. Rerun that script, quote its job id, and update this block --
# `sn_true` raises while they are None precisely so an unfitted pair cannot be papered over.
SN_SKY_VAR = 0.52699   # effective sky variance per pixel, ADU^2 (= 5.414 x PIXEL_RMS^2)
SN_GAIN = 10.7036      # effective gain, electrons per ADU, for the source shot-noise term


def resolution(re_arcsec):
    """PSF resolution factor R = Re^2 / (Re^2 + Re_psf^2), in [0, 1).

    0 is unresolved, 1 is PSF-free. R = 0.474 at the V2.1 size cut.
    """
    re2 = np.asarray(re_arcsec, dtype=float) ** 2
    return re2 / (re2 + PSF_RE ** 2)


def aperture_pixels(re_arcsec):
    """Pixels in the PSF-convolved half-light ellipse: N_pix = pi Re_c^2 / PIXEL_SIZE^2.

    The absolute normalisation does not need to match SExtractor's Kron aperture -- the ratio is
    absorbed by the fitted `SN_SKY_VAR` (it lands at 5.37 x PIXEL_RMS^2, i.e. a 5.4x larger area).
    What matters is that the SHAPE, Re_c^2, is right, and that is what the slope-1 check tests.
    """
    re_c2 = np.asarray(re_arcsec, dtype=float) ** 2 + PSF_RE ** 2
    return np.pi * re_c2 / PIXEL_SIZE ** 2


def _constants(sky_var, gain):
    sky_var = SN_SKY_VAR if sky_var is None else sky_var
    gain = SN_GAIN if gain is None else gain
    if sky_var is None or gain is None:
        raise RuntimeError(
            "sbs_shear.domain.SN_SKY_VAR / SN_GAIN are unset -- run scripts/eval_domain_v21.py and "
            "record the fitted pair before using sn_true(). Refusing to substitute a default, "
            "which would move the S/N = 10 boundary without saying so.")
    return float(sky_var), float(gain)


def sn_true(mag, re_arcsec, sky_var=None, gain=None):
    """Aperture S/N from TRUE magnitude and TRUE half-light radius (arcsec), CCD noise equation.

    `sky_var` / `gain` override the module constants -- used by the fitting script itself, which
    must evaluate the form before it has constants to report. Everything else should leave them
    None so there is exactly one pair of numbers in play.
    """
    sky_var, gain = _constants(sky_var, gain)
    flux = 10.0 ** (-0.4 * (np.asarray(mag, dtype=float) - ZERO_MAG))
    var = sky_var * aperture_pixels(re_arcsec) + flux / gain
    return flux / np.sqrt(var)


def sn_limiting_mag(re_arcsec, sn_min=V21_SN_MIN, sky_var=None, gain=None):
    """The magnitude at which `sn_true` equals `sn_min`, at a given true size.

    The domain cut is `mag < sn_limiting_mag(Re)`. Useful for reporting the curve, and for
    sanity-checking a consumer that can only express box cuts.

    Closed form: S/N = f / sqrt(s N + f/g) at S/N = q is the positive root of
    f^2 - (q^2/g) f - q^2 s N = 0.
    """
    sky_var, gain = _constants(sky_var, gain)
    n_pix = aperture_pixels(re_arcsec)
    q2 = float(sn_min) ** 2
    b = q2 / gain
    flux_min = 0.5 * (b + np.sqrt(b ** 2 + 4.0 * q2 * sky_var * n_pix))
    return -2.5 * np.log10(flux_min) + ZERO_MAG


def in_domain(mag, re_arcsec, re_min=V21_RE_MIN, sn_min=V21_SN_MIN, sky_var=None, gain=None):
    """Boolean mask of the V2.1 primary domain. NaNs are OUT (comparisons on NaN are False).

    `mag` / `re_arcsec` are the PRIMARY true properties -- `r_input_p` and `Re_input_p`. Passing a
    secondary's columns here would cut the neighbour population, which the deliverable does not do.
    """
    re = np.asarray(re_arcsec, dtype=float)
    return (re > re_min) & (sn_true(mag, re, sky_var=sky_var, gain=gain) > sn_min)


def select_frame(frame, re_min=V21_RE_MIN, sn_min=V21_SN_MIN,
                 mag_col=MAG_COLUMN, re_col=RE_COLUMN, sky_var=None, gain=None):
    """Drop the rows of a catalogue batch that fall outside the V2.1 primary domain.

    THE single entry point for applying the domain. The flow trainer, the response-target builder,
    the emulator retrain and the constgold evaluators all call this, so none of them carries its
    own copy of "0.5" and "10" -- which is the whole point, since the S/N half of the domain is a
    curve and would not survive being retyped four times.

    Applied AFTER the usual box selection (`source_select_selection`), which keeps the upper limits
    (mag < 28, Re < 1.5") and the pair-distance cut. This only ever narrows further.

    Raises if a required column is absent rather than passing the frame through unfiltered: a
    silently un-narrowed population is exactly the failure AGENTS.md documents, where a model gets
    scored on rows it was never trained for and the discrepancy reads as model error.
    """
    for col in (mag_col, re_col):
        if col not in frame.columns:
            raise KeyError(
                f"select_frame: column {col!r} is missing, so the V2.1 domain cannot be applied. "
                f"Refusing to return an unfiltered frame. Available: {sorted(frame.columns)[:12]}...")
    keep = in_domain(frame[mag_col].to_numpy(dtype=float),
                     frame[re_col].to_numpy(dtype=float),
                     re_min=re_min, sn_min=sn_min, sky_var=sky_var, gain=gain)
    return frame[keep].reset_index(drop=True)


def metadata(re_min=V21_RE_MIN, sn_min=V21_SN_MIN, sky_var=None, gain=None):
    """The domain as a JSON-able dict, to be stamped into every checkpoint and product.

    A number is only reproducible if you can tell which population produced it. Anything written
    under V2.1 carries this so a later reader does not have to infer the domain from the filename.
    """
    s, g = _constants(sky_var, gain)
    return {
        "domain": "v2.1",
        "primary_re_min": float(re_min),
        "primary_sn_min": float(sn_min),
        "sn_sky_var": s,
        "sn_gain": g,
        "psf_re": PSF_RE,
        "resolution_at_re_min": float(resolution(re_min)),
        "description": describe(re_min=re_min, sn_min=sn_min, sky_var=s, gain=g),
    }


def describe(re_min=V21_RE_MIN, sn_min=V21_SN_MIN, sky_var=None, gain=None):
    """One-line, printable statement of the active domain -- for job logs and metadata.

    Every consumer prints this, so a mismatched domain shows up in the log rather than in `m`.
    """
    txt = (f"V2.1 domain: primary true Re > {re_min:g}\" "
           f"(= {re_min / PIXEL_SIZE:g} px, resolution R = {float(resolution(re_min)):.3f}) "
           f"AND sn_true > {sn_min:g}")
    try:
        s, g = _constants(sky_var, gain)
    except RuntimeError:
        return txt + "  [SN_SKY_VAR/SN_GAIN UNSET]"
    lim = sn_limiting_mag(np.array([re_min, 1.0, 1.5]), sn_min=sn_min, sky_var=s, gain=g)
    return (txt + f"  [sky_var={s:.4f}, gain={g:.3f}; the S/N cut is mag < "
            f"{lim[0]:.2f} at Re={re_min:g}\", {lim[1]:.2f} at 1.0\", {lim[2]:.2f} at 1.5\"]")
