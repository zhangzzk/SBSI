"""GalSBI image reference builder on synthetic constant-shear catalogues."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbsi.disk.precision import validate_shape_encoding

PATH = Path(__file__).resolve().parents[1]/'scripts/prepare_galsbi_image_reference.py'
SPEC = importlib.util.spec_from_file_location('prepare_galsbi_image_reference_under_test', PATH)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)
CUTS = dict(magnitude_limit=25., min_radius_pixels=3., max_radius_pixels=100., max_measured_ellipticity=.99)


def write_leg(root, case, shear, n=400, *, seed=0, tweak=None):
    rng = np.random.default_rng(seed)
    paths = prep.leg_paths(root, case, shear)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    truth = pd.DataFrame(dict(index_input=np.arange(n)[::-1].copy(), RA_input=np.linspace(0, 1, n),
        DEC_input=np.linspace(1, 2, n), redshift_input=np.full(n, .5), Re_input=np.full(n, .4),
        axis_ratio_input=np.full(n, .7), position_angle_input=np.zeros(n), sersic_n_input=np.ones(n),
        r_input=np.linspace(20, 26, n), gamma1_input=np.full(n, shear), gamma2_input=np.zeros(n)))
    if tweak:
        tweak(truth)
    truth.to_feather(paths['truth'])
    # detections: every third input undetected; one crossmatch row without shape; one shape without crossmatch
    detected = np.arange(0, n, 3)
    number = np.arange(1, len(detected)+1)
    pd.DataFrame(dict(id_detec=np.r_[number, 10**6], id_input=np.r_[detected, 1])).to_feather(paths['crossmatch'])
    rng2 = np.random.default_rng(seed+1)
    g = rng2.uniform(-.75, .75, (len(number)+1, 2))
    g[0] = [.995, 0.]          # |e| >= 0.99: usable, not selected
    g[1] = [1.2, 0.]           # outside the unit disk: unusable
    mag = rng2.uniform(19, 26, len(number)+1)
    radius = rng2.uniform(1, 8, len(number)+1)
    radius[2] = 3.             # strict radius bound
    mag[3] = 25.               # strict magnitude bound
    pd.DataFrame(dict(NUMBER=np.r_[number, 10**7], NGMIX_G1=g[:, 0], NGMIX_G2=g[:, 1], MAG_AUTO=mag,
                      FLUX_RADIUS=radius)).to_feather(paths['shapes'])


def make(root, cases=(5, 6), **kw):
    for case in cases:
        for _, shear in prep.LEGS:
            write_leg(root, case, shear, seed=case, **kw)


def test_reference_is_inference_ready_and_applies_normalization_cuts(tmp_path):
    make(tmp_path/'sims')
    summary = prep.build(tmp_path/'sims', [5, 6], tmp_path/'ref', fraction=.6, seed=3,
                         model_name='like-1.0-galsbi', cuts=CUTS)
    report = json.loads((tmp_path/'ref/selection_report.json').read_text())
    for name, shear in prep.LEGS:
        leg = tmp_path/'ref'/name
        protocol = json.loads((leg/'protocol.json').read_text())
        result = json.loads((leg/'result.json').read_text())
        obs, truth = pd.read_parquet(leg/'observations.parquet'), pd.read_parquet(leg/'truth.parquet')
        assert list(obs.columns) == [*prep.MEASURED, prep.LOG_GAP_COLUMN]
        assert result['observation_count'] == len(obs) == len(truth) == summary[name] > 0
        assert result['observations_sha256'] == prep.digest(leg/'observations.parquet')
        assert protocol['reference_kind'] == 'galsbi_population_v1' and protocol['chains'] == 4
        assert protocol['conditions']['psf_fwhm'] == .71 and protocol['injected_shear'] == [shear, 0.]
        assert protocol['model_identity']['name'] == 'like-1.0-galsbi'
        e = np.hypot(obs.measured_ngmix_g1, obs.measured_ngmix_g2)
        assert (e < .99).all() and (obs.measured_flux_radius > 3).all() and (obs.measured_flux_radius < 100).all()
        assert (obs.measured_flux_from_mag_auto > 100).all() and (truth.measured_mag_auto < 25).all()
        validate_shape_encoding(obs[prep.MEASURED[:2]].to_numpy(), obs[prep.LOG_GAP_COLUMN].to_numpy())
        assert (truth.injected_g1 == shear).all()
        for case in report['cases']:
            r = case[name]
            assert r['crossmatch_without_shape'] == 1 and r['shape_without_crossmatch'] == 1
            assert r['unusable'] >= 1
    # The same identity draw on both legs (shared noise seed => same rows here).
    plus, minus = (pd.read_parquet(tmp_path/'ref'/n/'truth.parquet') for n in ('plus', 'minus'))
    pd.testing.assert_frame_equal(plus[['source_case', 'source_input_index']],
                                  minus[['source_case', 'source_input_index']])
    # The draw does not depend on measurements: a smaller fraction gives a nested subset.
    prep.build(tmp_path/'sims', [5, 6], tmp_path/'small', fraction=.3, seed=3, model_name='like-1.0-galsbi', cuts=CUTS)
    small = pd.read_parquet(tmp_path/'small/plus/truth.parquet')
    key = lambda f: set(zip(f.source_case, f.source_input_index))
    assert key(small) < key(plus)


def test_legs_with_different_unsheared_inputs_are_rejected(tmp_path):
    make(tmp_path/'sims', cases=(5,))
    write_leg(tmp_path/'sims', 5, -.02, seed=5, tweak=lambda t: t.__setitem__('Re_input', t.Re_input*1.01))
    with pytest.raises(ValueError, match='differ between legs'):
        prep.build(tmp_path/'sims', [5], tmp_path/'ref', fraction=1., seed=3, model_name='like-1.0-galsbi', cuts=CUTS)


def test_declared_shear_is_checked(tmp_path):
    make(tmp_path/'sims', cases=(5,))
    write_leg(tmp_path/'sims', 5, .02, seed=5, tweak=lambda t: t.__setitem__('gamma2_input', 1e-3))
    with pytest.raises(ValueError, match='declared'):
        prep.build(tmp_path/'sims', [5], tmp_path/'ref', fraction=1., seed=3, model_name='like-1.0-galsbi', cuts=CUTS)
