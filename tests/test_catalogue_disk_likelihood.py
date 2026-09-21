from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import logsumexp

from sbsi.catalogue_disk_likelihood import DiskCatalogueLikelihood
from sbsi.catalogue_likelihood import CatalogueModelView, CatalogueSelection, OutputCut
from sbsi.disk_response_transport import disk_response_transport
from sbsi.shear_map import apply_shear_to_ellipticity


TARGETS = ["measured_ngmix_g1", "measured_ngmix_g2", "measured_flux_radius", "measured_flux_from_mag_auto"]
FEATURES = ["e1_input_p", "e2_input_p"]


class Flow:
    device = torch.device("cpu")
    target_transform = SimpleNamespace(target_names=TARGETS, means=np.array([.1, -.2, 4., 300.]),
                                      scales=np.array([.3, .4, 2., 100.]))
    condition_preprocessor = SimpleNamespace(feature_names=FEATURES, means=np.zeros(2), scales=np.ones(2),
        log_features=(), raw_tensor_from_frame=lambda frame, device: torch.as_tensor(frame.to_numpy(), device=device))

    def context_tensor(self, frame):
        return torch.as_tensor(frame[FEATURES].to_numpy(), dtype=torch.float64)

    def target_tensor(self, frame):
        return torch.as_tensor((frame[TARGETS].to_numpy()-self.target_transform.means)/self.target_transform.scales)

    def log_prob_tensor(self, target, context, **kwargs):
        center = torch.cat((context, torch.zeros_like(context)), dim=1)
        return -.5*((target-center)**2).sum(1) - 2*np.log(2*np.pi)


class Response:
    n_atoms = 2
    calls = 0

    def predict(self, atoms, radius, flux):
        self.calls += 1
        return .2 + .4*atoms + .001*radius + .000001*flux, {}


def setup():
    raw = pd.DataFrame([[.1, .2], [-.2, .1]], columns=FEATURES)
    raw.index.name = "primary_row"
    def get(g1, g2):
        flow = raw.copy()
        flow[FEATURES] = np.column_stack(apply_shear_to_ellipticity(raw.iloc[:, 0], raw.iloc[:, 1], g1, g2))
        return CatalogueModelView(g1, g2, flow, raw, np.array([.3+g1, .8-g2]))
    cache = SimpleNamespace(blend_response=None, prior=SimpleNamespace(galaxies=raw, weights=np.array([.3, .7])),
        get=get, _key=lambda g1, g2: (float(g1), float(g2)), validate_model_features=lambda flow: None)
    observed = pd.DataFrame([[.25, -.1, 3.2, 100.], [-.05, .3, 6., 250.]], columns=TARGETS)
    return DiskCatalogueLikelihood(Flow(), cache, disk_response=Response()), observed


def test_every_density_path_matches_direct_inverse_jacobian_and_state_mass():
    likelihood, observed = setup()
    g1, g2 = .02, -.01
    indices = np.array([[0, 1], [1, 0]])
    q = np.array([[.4, .6], [.6, .4]])
    manual = np.empty((2, 2))
    for i in range(2):
        for k in range(2):
            atom = indices[i, k]
            physical = torch.as_tensor(observed.iloc[[i]].to_numpy())
            b = .2+.4*atom+.001*physical[0, 2]+.000001*physical[0, 3]
            e, ld = disk_response_transport(physical[:, :2], b*torch.tensor([[g1, g2]], dtype=torch.float64), inverse=True)
            base = torch.cat((e, physical[:, 2:]), dim=1)
            flow = likelihood.flow_model
            context = flow.context_tensor(likelihood.cache.get(g1, g2).flow.iloc[[atom]])
            means = torch.as_tensor(flow.target_transform.means)
            scales = torch.as_tensor(flow.target_transform.scales)
            manual[i, k] = float(flow.log_prob_tensor((base-means)/scales, context)+ld)
    conditional = likelihood.conditional_log_likelihood(observed, g1, g2, atom_indices=indices[:, 0])
    np.testing.assert_allclose(conditional, manual[:, 0], atol=1e-12)
    mass = likelihood.cache.prior.weights * likelihood.cache.get(g1, g2).detection_probability
    expected = manual + np.log(mass[indices]/q)
    actual = likelihood.log_importance_weights(observed, g1, g2, atom_indices=indices, proposal_probability=q)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    sampled = likelihood.log_likelihood(observed, g1, g2, atom_indices=indices, proposal_probability=q,
                                        object_chunk=1, atom_chunk=1)
    np.testing.assert_allclose(sampled, logsumexp(expected, axis=1)-np.log(2)-np.log(mass.sum()), atol=1e-12)
    exact = likelihood.log_likelihood(observed, g1, g2, object_chunk=1, atom_chunk=1)
    np.testing.assert_allclose(exact, logsumexp(manual+np.log(mass[indices]), axis=1)-np.log(mass.sum()), atol=1e-12)


def test_stencil_reuses_output_response_but_not_detection_probability():
    likelihood, observed = setup()
    indices = np.array([[0, 1], [1, 0]])
    for g in [0., .001, -.001, .002]:
        likelihood.log_importance_weights_tensor(observed, g, 0., atom_indices=indices, proposal_probability=np.ones((2, 2)))
    assert likelihood.disk_response.calls == 1
    assert likelihood._tensor_view(.001, 0.).context.data_ptr() == likelihood._tensor_view(0., 0.).context.data_ptr()
    assert not torch.equal(likelihood._tensor_view(.001, 0.).log_detected_mass,
                           likelihood._tensor_view(0., 0.).log_detected_mass)


def test_ragged_padding_and_invalid_physical_outputs():
    likelihood, observed = setup()
    observed.iloc[0, 0] = 1.1
    valid = np.array([[True, False], [True, False]])
    result = likelihood.log_importance_weights_tensor(observed, .02, 0., atom_indices=np.array([[0, 1], [1, 0]]),
        proposal_probability=np.ones((2, 2)), atom_valid=valid)
    assert torch.isneginf(result[0, 0]) and torch.isfinite(result[1, 0])
    assert torch.isneginf(result[:, 1]).all()
    conditional = likelihood.conditional_log_likelihood(observed, .02, 0., atom_indices=np.array([0, 1]))
    assert np.isneginf(conditional[0]) and np.isfinite(conditional[1])


def test_alternating_exact_and_complement_response_grids_are_cached_and_bounded():
    likelihood, observed = setup()
    grids = [np.array([[0], [1]]), np.array([[1, 0], [0, 1]])]
    for g in [0., .001, -.001, .002]:
        for indices in grids:
            actual = likelihood.log_importance_weights(observed, g, 0., atom_indices=indices,
                proposal_probability=np.ones(indices.shape))
            fresh, _ = setup()
            expected = fresh.log_importance_weights(observed, g, 0., atom_indices=indices,
                proposal_probability=np.ones(indices.shape))
            np.testing.assert_array_equal(actual, expected)
    assert likelihood.disk_response.calls == 2
    assert len(likelihood._response_cache) == 2
    changed = observed.copy()
    changed.iloc[:, 2] += .1
    likelihood.log_importance_weights(changed, 0., 0., atom_indices=grids[0], proposal_probability=np.ones((2, 1)))
    assert likelihood.disk_response.calls == 3
    assert len(likelihood._response_cache) == 2


def test_rejects_shape_selection_until_transport_normalizer_is_implemented():
    likelihood, _ = setup()
    cut = OutputCut(TARGETS, abs_shape=.8)
    with pytest.raises(ValueError, match="radius/flux"):
        DiskCatalogueLikelihood(likelihood.flow_model, likelihood.cache,
            disk_response=Response(), selection=CatalogueSelection(cut))
