"""The V2.1 domain: unit contract, the S/N inverse, and the guard on the unfitted constants.

The domain is imported by the flow trainer, the response-target builder, the emulator retrain and
the constgold evaluators. A silent change here re-populates all four at once and shows up as a
model error, not as a domain error -- which is the failure mode AGENTS.md spends most of its
"Fiducial Model" section on. Hence tests on the parts that could drift silently.
"""

import numpy as np
import pytest

from sbs_shear import domain as D

# Smoke-fit values. Used ONLY to exercise the maths -- they are passed explicitly so these tests
# neither depend on nor validate whatever is currently adopted in the module.
SKY, GAIN = 0.52303, 10.5848


def test_psf_size_matches_the_moffat_conversion():
    # Re_psf is what turns "2.5 pixels" into "resolution 0.5"; if it drifts, the domain moves.
    assert D.PSF_RE == pytest.approx(0.5268, abs=1e-4)
    assert D.PSF_RE / D.PIXEL_SIZE == pytest.approx(2.634, abs=1e-3)


def test_the_size_cut_is_2p5_pixels_and_resolution_about_half():
    # This is the whole reason V21_RE_MIN is 0.5 and not 2.5: the owner's "2.5" was in PIXELS,
    # and only that reading reproduces "resolution factor roughly 0.5".
    assert D.V21_RE_MIN / D.PIXEL_SIZE == 2.5
    assert float(D.resolution(D.V21_RE_MIN)) == pytest.approx(0.474, abs=0.005)
    # 2.5 ARCSEC would be off the end of the catalogue's size range and resolution ~0.96.
    assert float(D.resolution(2.5)) > 0.9


def test_resolution_is_half_at_the_psf_size_and_monotone():
    assert float(D.resolution(D.PSF_RE)) == pytest.approx(0.5, abs=1e-12)
    r = D.resolution(np.array([0.1, 0.3, 0.5, 1.0, 1.5]))
    assert np.all(np.diff(r) > 0)
    assert 0.0 < r[0] and r[-1] < 1.0


def test_sn_true_refuses_to_run_on_unfitted_constants(monkeypatch):
    # The guard that stops a plausible-looking default from silently moving the S/N=10 boundary.
    monkeypatch.setattr(D, "SN_SKY_VAR", None)
    monkeypatch.setattr(D, "SN_GAIN", None)
    with pytest.raises(RuntimeError, match="unset"):
        D.sn_true(24.0, 0.6)
    with pytest.raises(RuntimeError, match="unset"):
        D.in_domain(24.0, 0.6)
    # ... but an explicit override still works, which is how the fitting script bootstraps.
    assert np.isfinite(D.sn_true(24.0, 0.6, sky_var=SKY, gain=GAIN))


def test_limiting_mag_inverts_sn_true_exactly():
    # sn_limiting_mag solves a quadratic in flux by hand; a sign slip there would shift the whole
    # magnitude limit while still looking perfectly reasonable. Round-trip it.
    re = np.array([0.5, 0.7, 1.0, 1.5])
    for sn_min in (5.0, 10.0, 20.0):
        mag = D.sn_limiting_mag(re, sn_min=sn_min, sky_var=SKY, gain=GAIN)
        back = D.sn_true(mag, re, sky_var=SKY, gain=GAIN)
        assert back == pytest.approx(np.full_like(re, sn_min), rel=1e-9)


def test_sn_falls_with_magnitude_and_with_size():
    # Fainter is lower S/N; bigger is lower S/N at fixed flux (light spread over more sky pixels).
    mags = np.array([22.0, 23.0, 24.0, 25.0, 26.0])
    sn = D.sn_true(mags, 0.6, sky_var=SKY, gain=GAIN)
    assert np.all(np.diff(sn) < 0)
    sizes = np.array([0.5, 0.8, 1.2, 1.5])
    sn = D.sn_true(24.0, sizes, sky_var=SKY, gain=GAIN)
    assert np.all(np.diff(sn) < 0)


def test_source_noise_term_matters_and_grows_towards_the_bright_end():
    # The sky-limited form (gain -> infinity) was WRONG: slope 0.826 against measured S/N. Guard
    # the fix, because dropping the term again would still produce entirely plausible numbers.
    # How much it costs, at Re = 0.6": 6.5% of the S/N at mag 26, 28% at mag 24, 84% at mag 20.
    mags = np.array([20.0, 22.0, 24.0, 26.0])
    sky_only = D.sn_true(mags, 0.6, sky_var=SKY, gain=1e12)
    full = D.sn_true(mags, 0.6, sky_var=SKY, gain=GAIN)
    source_frac = 1.0 - full / sky_only
    assert np.all(full < sky_only)
    assert np.all(np.diff(source_frac) < 0)     # brighter -> more source-dominated
    assert source_frac[-1] < 0.10               # mag 26: sky-dominated, but not negligible
    assert source_frac[0] > 0.50                # mag 20: source noise dominates outright


def test_in_domain_is_the_two_cuts_and_nan_is_out():
    mag = np.array([24.0, 24.0, 27.0, np.nan, 24.0])
    re = np.array([0.6, 0.4, 0.6, 0.6, np.nan])
    got = D.in_domain(mag, re, sky_var=SKY, gain=GAIN)
    expected = (re > D.V21_RE_MIN) & (D.sn_true(mag, re, sky_var=SKY, gain=GAIN) > D.V21_SN_MIN)
    assert np.array_equal(got, np.nan_to_num(expected, nan=False).astype(bool))
    assert got[0] and not got[1]        # size cut bites
    assert not got[2]                   # S/N cut bites
    assert not got[3] and not got[4]    # NaN in either property is OUT


def test_in_domain_is_a_subset_of_the_v2_box():
    # Measured on the catalogue: 100.00% of V2.1 lies inside V2 (mag<26, Re>0.3). That is what
    # makes the two directly comparable row-for-row, so assert it holds on a grid too.
    mag, re = np.meshgrid(np.linspace(18, 28, 60), np.linspace(0.05, 1.5, 40))
    v21 = D.in_domain(mag.ravel(), re.ravel(), sky_var=SKY, gain=GAIN)
    v2 = (mag.ravel() < 26.0) & (re.ravel() > 0.3)
    assert np.all(v2[v21]), "V2.1 must be strictly inside the V2 box"


def test_describe_names_the_state_it_is_in(monkeypatch):
    # Every consumer prints this, so a mismatched or unfitted domain lands in the log rather than
    # in `m`. It must never look calibrated when it is not.
    monkeypatch.setattr(D, "SN_SKY_VAR", None)
    monkeypatch.setattr(D, "SN_GAIN", None)
    assert "UNSET" in D.describe()
    txt = D.describe(sky_var=SKY, gain=GAIN)
    assert "UNSET" not in txt and "2.5 px" in txt and "mag <" in txt
