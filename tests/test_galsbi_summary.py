"""Compression sufficiency, geometry and unchanged likelihood replay."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.catalogue_disk_likelihood import DiskCatalogueLikelihood
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.flow_size_condition import circularized_radius
from sbsi.galsbi_scene_prior import GalSBISceneBatch, GalSBIScenePrior
from sbsi.galsbi_summary import (SummaryLikelihood, compress_scenes, native_shapes,
                                scene_model_inputs)
from sbsi.output_conditioned_response import output_features
from sbsi.v39_inference import FEATURES, V39CatalogueCache

TARGETS = ['measured_ngmix_g1', 'measured_ngmix_g2', 'measured_flux_radius', 'measured_flux_from_mag_auto']


class Flow:
    device = torch.device('cpu')
    target_transform = SimpleNamespace(target_names=TARGETS, means=np.zeros(4), scales=np.ones(4))
    condition_preprocessor = SimpleNamespace(feature_names=FEATURES, means=np.zeros(8), scales=np.ones(8),
        log_features=(), raw_tensor_from_frame=lambda frame, device: torch.as_tensor(frame.to_numpy(), device=device))

    def context_tensor(self, frame):
        return torch.as_tensor(frame[list(FEATURES)].to_numpy(), dtype=torch.float64)

    def target_tensor(self, frame):
        return torch.as_tensor(frame[TARGETS].to_numpy(), dtype=torch.float64)

    def log_prob_tensor(self, target, context, **kwargs):
        center = torch.cat((context[:, :2], torch.ones((len(context), 2))), dim=1)
        return -.5*((target-center)**2).sum(1)-2*np.log(2*np.pi)


class Detector:
    preprocessor = SimpleNamespace(feature_names=FEATURES)

    def predict_proba(self, frame):
        return .5+.1*frame.e1_input_p.to_numpy()+.01*frame.nbr_close_050.to_numpy()


class Trees:
    response_coordinate = 'disk_velocity'
    output_thresholds = (np.array([1.], np.float32), np.array([24.], np.float32))
    transformed_outputs = staticmethod(output_features)

    def predict_pairs(self, pairs, radius, flux):
        return .2*pairs.iloc[:, 0].to_numpy()+(radius > np.e)*.1+(flux < 100)*.2


def test_nine_summaries_replay_full_pair_likelihood_and_reject_stale_auxiliaries():
    zero = pd.DataFrame([[.2, -.1, 1., 22., .4, .1, .2, .3],
                         [-.1, .3, 2., 23., .5, .2, .4, .8]], columns=FEATURES)
    response = CatalogueDiskResponse([0, 0, 3], np.arange(21).reshape(3, 7), Trees())
    obs = pd.DataFrame([[.1, .2, 4., 100.], [-.1, .3, 5., 80.]], columns=TARGETS)
    summary = compress_scenes(zero, response, obs[TARGETS[2:]].to_numpy())
    assert summary.values.shape == (2, 9) and summary.values[0, -1] == 0
    cache = V39CatalogueCache(zero, np.ones(2), {}, detector=Detector())
    reference = DiskCatalogueLikelihood(Flow(), cache, disk_response=response)
    compressed = SummaryLikelihood(Flow(), Detector(), density_batch_size=4)
    for g1, g2 in ((0., 0.), (.02, -.01), (-.015, .02)):
        expected = reference.conditional_log_likelihood(obs, g1, g2, atom_indices=np.arange(2))
        expected += np.log(cache.get(g1, g2).detection_probability)
        np.testing.assert_allclose(compressed.log_kernel(summary, obs, g1, g2), expected, atol=1e-12)
    changed = obs.copy()
    changed.iloc[0, 2] += .1
    with pytest.raises(ValueError, match='different observed'):
        compressed.log_kernel(summary, changed, .02, 0.)


def marks(a):
    a = np.asarray(a)
    return dict(a=a, psi=np.linspace(.3, 5., len(a)), z=np.ones(len(a)),
                R_kpc=np.ones(len(a)), n=np.ones(len(a)), m_r=np.full(len(a), 22.))


def test_native_mapping_matches_ellipse_axes_and_sbsi_circularization():
    data = marks([0., .2, .8])
    shape = native_shapes(data, lambda z: np.full(len(z), 1000.))
    eps = np.hypot(shape['e1'], shape['e2'])
    q = np.sqrt((1-data['a'])/(1+data['a']))
    np.testing.assert_allclose(eps, (1-q)/(1+q), atol=1e-15)
    np.testing.assert_allclose(shape['circularized'], circularized_radius(shape['semimajor'], q))
    assert shape['circularized'][2] < shape['circularized'][0]


def test_only_response_uses_nearest_twenty_and_empty_neighbours_work():
    prior = GalSBIScenePrior(SimpleNamespace(), aperture_arcsec=10.)
    scenes = GalSBISceneBatch(marks([.2, .3]), marks(np.full(25, .1)),
                             [0, 25, 25], np.column_stack((np.linspace(.1, 10, 25), np.zeros(25))))
    conditions = dict(pixel_size=.2, zero_point=30., psf_fwhm=.73, moffat_beta=2.224, pixel_rms=.312)
    zero, ptr, pairs = scene_model_inputs(scenes, prior, lambda z: np.full(len(z), 1000.),
        conditions, response_radius_arcsec=10., response_neighbours=20)
    np.testing.assert_array_equal(ptr, [0, 20, 20])
    np.testing.assert_allclose(zero.iloc[:, 5:], prior.v39_crowding(scenes))
    assert pairs.shape == (20, 7)
    empty = GalSBISceneBatch(marks([.2]), marks([]), [0, 0], np.empty((0, 2)))
    _, ptr, pairs = scene_model_inputs(empty, prior, lambda z: np.full(len(z), 1000.),
        conditions, response_radius_arcsec=10., response_neighbours=20)
    np.testing.assert_array_equal(ptr, [0, 0])
    assert pairs.shape == (0, 7)


def test_redshift_sed_moves_preserve_full_likelihood_summaries_with_neighbours():
    from test_galsbi_prior import prior_fixture
    from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates

    galaxy = prior_fixture()
    # Nontrivial colour and redshift dependence: fixing r magnitude requires
    # changing M_B, and fixing angular radius requires changing native R_kpc.
    galaxy.photometry = lambda z, M, c: dict(m_r=M+40+z+3*c[:, 0],
        m_i=M+40+2*c[:, 1], M_i=M+2*c[:, 1])
    prior = GalSBIScenePrior(galaxy)
    def angular(z):
        return 1000*z/(1+z)
    coords = GalSBITargetCoordinates(galaxy, angular)
    sampled, _ = galaxy.sample(512, 290)
    target = {k: v[:8] for k, v in sampled.items() if k != 'proposal_index'}
    neighbours = {k: v[8:40] for k, v in sampled.items() if k != 'proposal_index'}
    scene = GalSBISceneBatch(target, neighbours, np.arange(9)*4,
        np.column_stack((np.tile([.2, .8, 2., 9.], 8), np.zeros(32))))
    x = coords.encode(target)
    moved, _ = coords.propose_sed(x, np.random.default_rng(291))
    moved[:, 0] += .15
    _, data, _ = coords.log_mixture_prior(moved)
    proposal = GalSBISceneBatch(data, neighbours, scene.indptr, scene.offset_arcsec)
    conditions = dict(pixel_size=.2, zero_point=30., psf_fwhm=.73, moffat_beta=2.224, pixel_rms=.312)
    summaries = []
    for candidate in (scene, proposal):
        zero, ptr, pairs = scene_model_inputs(candidate, prior, angular, conditions,
            response_radius_arcsec=10., response_neighbours=20)
        summaries.append(compress_scenes(zero, CatalogueDiskResponse(ptr, pairs, Trees()), [4., 100.]))
    np.testing.assert_allclose(summaries[0].values, summaries[1].values, rtol=0, atol=1e-12)
    assert not np.allclose(target['M_B'], data['M_B'])
    assert not np.allclose(target['R_kpc'], data['R_kpc'])
    observed = pd.DataFrame([[.1, .2, 4., 100.]], columns=TARGETS)
    likelihood = SummaryLikelihood(Flow(), Detector(), density_batch_size=16)
    for shear in ((0., 0.), (.02, -.01)):
        np.testing.assert_allclose(likelihood.log_kernel(summaries[0], observed, *shear),
                                   likelihood.log_kernel(summaries[1], observed, *shear), atol=1e-12)
