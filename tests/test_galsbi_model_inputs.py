"""Model scene-input contract: GalSBI-trained models see rendered magnitude and Sersic n."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sbsi.galsbi.likelihood import scene_model_inputs
from sbsi.galsbi.model_inputs import SceneMarks, render_sersic_n
from sbsi.galsbi.scenes import GalSBISceneBatch, GalSBIScenePrior
from sbsi.models import get_model

CONDITIONS = dict(pixel_size=.2, zero_point=30., psf_fwhm=.71, moffat_beta=2.458, pixel_rms=.6159)


def writer_render(n):
    """Verbatim arithmetic of SBSI 1c16564 scripts/write_galsbi_scenes.py."""
    render = (0.3, 6.2)
    grid = np.geomspace(*render, 122)
    k = np.rint(np.log(np.clip(n, *render)/render[0])/np.log(grid[1]/grid[0])).astype(int)
    return grid[k]


def test_rendered_sersic_grid_matches_scene_writer_and_clips():
    n = np.r_[0.2, 0.3, np.geomspace(.31, 6.1, 997), 6.2, 9.]
    out = render_sersic_n(n)
    np.testing.assert_array_equal(out, writer_render(n))
    assert out[0] == pytest.approx(.3) and out[-1] == pytest.approx(6.2)
    assert len(np.unique(out)) <= 122


def marks(n, m_r, n_sersic):
    return dict(a=np.full(n, .2), psi=np.linspace(.3, 5., n), z=np.ones(n), R_kpc=np.ones(n),
                n=np.asarray(n_sersic, float), m_r=np.asarray(m_r, float))


class Shifted:
    """Stand-in contract: band offset per galaxy and rendered n."""

    def magnitude(self, data):
        return np.asarray(data['m_r'])+.3+.1*np.asarray(data['n'])

    def sersic_n(self, data):
        return render_sersic_n(np.asarray(data['n']))


def test_marks_replace_magnitude_sersic_crowding_and_emulator_pairs_everywhere():
    prior = GalSBIScenePrior(SimpleNamespace(), aperture_arcsec=10.)
    targets = marks(2, [22., 23.5], [1.03, 4.4])
    neighbours = marks(25, np.linspace(20, 27, 25), np.linspace(.25, 7., 25))
    offsets = np.column_stack((np.linspace(.1, 9.9, 25), np.zeros(25)))
    scenes = GalSBISceneBatch(targets, neighbours, [0, 13, 25], offsets)
    rule = Shifted()
    substituted = GalSBISceneBatch(
        {**targets, 'm_r': rule.magnitude(targets), 'n': rule.sersic_n(targets)},
        {**neighbours, 'm_r': rule.magnitude(neighbours), 'n': rule.sersic_n(neighbours)}, [0, 13, 25], offsets)
    distance = lambda z: np.full(len(z), 1000.)
    mapped = scene_model_inputs(scenes, prior, distance, CONDITIONS, response_radius_arcsec=10.,
                                response_neighbours=20, marks=rule)
    expected = scene_model_inputs(substituted, prior, distance, CONDITIONS, response_radius_arcsec=10.,
                                  response_neighbours=20)
    pd.testing.assert_frame_equal(mapped[0], expected[0], check_exact=True)
    np.testing.assert_array_equal(mapped[1], expected[1])
    np.testing.assert_array_equal(mapped[2], expected[2])
    native = scene_model_inputs(scenes, prior, distance, CONDITIONS, response_radius_arcsec=10.,
                                response_neighbours=20)
    assert not np.array_equal(native[0].to_numpy(), mapped[0].to_numpy())
    # Default (None) and the native contract give the historical inputs bit for bit.
    pd.testing.assert_frame_equal(native[0], scene_model_inputs(scenes, prior, distance, CONDITIONS,
        response_radius_arcsec=10., response_neighbours=20, marks=SceneMarks(None))[0], check_exact=True)


def test_galsbi_presets_declare_contract_and_fs2_presets_do_not():
    like = get_model('like-1.0-galsbi')
    assert like.scene_inputs['magnitude'] == 'LSST_i'
    assert like.observing_conditions['psf_fwhm'] == .71
    assert get_model('galsbi-multistage').scene_inputs == like.scene_inputs
    assert get_model('V3.9-like').scene_inputs is None and get_model('V3.9-like').observing_conditions is None
    with pytest.raises(ValueError, match='model_band'):
        SceneMarks(like.scene_inputs, SimpleNamespace(), verify_table=False)
    with pytest.raises(ValueError, match='already'):
        SceneMarks(like.scene_inputs, SimpleNamespace(model_band=print), verify_table=False)
    # Drawn scenes must carry the hooked magnitude; native m_r is never silently used.
    marks = SceneMarks.__new__(SceneMarks)
    marks.kind = 'galsbi_resim_lsst_i'
    with pytest.raises(KeyError, match='m_model'):
        marks.magnitude(dict(m_r=np.ones(2)))


def test_inference_rejects_reference_conditions_other_than_the_model_training_ones():
    path = Path(__file__).resolve().parents[1]/'scripts/run_galsbi_inference.py'
    spec = importlib.util.spec_from_file_location('run_galsbi_inference_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    like = get_model('like-1.0-galsbi')
    module.validate_conditions(dict(CONDITIONS), like)
    fs2 = dict(CONDITIONS, psf_fwhm=.73, moffat_beta=2.224, pixel_rms=.312)
    with pytest.raises(ValueError, match='training conditions'):
        module.validate_conditions(fs2, like)
    module.validate_conditions(fs2, get_model('V3.9-like'))

