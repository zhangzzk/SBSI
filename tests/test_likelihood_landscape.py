"""The landscape must describe the target, not the machinery that reads it.

Its entire purpose is to answer a question the sampler's own diagnostics
cannot: whether a hard object is hard because the proposal misses it or
because its likelihood concentrates on a handful of atoms.  That answer is
only worth having if the concentration statistics are the ones they claim to
be, and if the atoms reported as carrying the mass really are the largest.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.likelihood_landscape import _mass_statistics, likelihood_landscape


class _Coordinates:
    def __init__(self, values):
        self.values = values
        self.target_names = tuple(f"t{i}" for i in range(values.shape[1]))
        self.center = np.zeros(values.shape[1])
        self.scale = np.ones(values.shape[1])


class _Proposal:
    def __init__(self, coordinates, n_atoms):
        self.coordinates = coordinates
        self.active_indices = np.arange(n_atoms)


class _Likelihood:
    """Returns a fixed log-weight per atom, ignoring the shear view."""

    def __init__(self, log_c):
        self._log_c = np.asarray(log_c, dtype=np.float64)

    def observed_target_tensor(self, observed):
        return torch.zeros((len(observed), 1), dtype=torch.float64)

    def log_importance_weights_tensor(
        self, observed, g1, g2, *, atom_indices, proposal_probability, **kwargs
    ):
        # The caller passes a proposal of one, so the returned value is the
        # exact log target and nothing has to be undone downstream.
        assert np.allclose(proposal_probability, 1.0)
        return torch.as_tensor(self._log_c[np.asarray(atom_indices)])


class _Mock:
    def __init__(self, measurements):
        self.measurements = measurements


@pytest.fixture
def landscape_inputs():
    n_atoms = 512
    rng = np.random.default_rng(3)
    coordinates = _Coordinates(rng.normal(size=(n_atoms, 2)))
    proposal = _Proposal(coordinates, n_atoms)
    # One atom carries essentially everything; the rest are far below it.
    log_c = np.full(n_atoms, -60.0)
    log_c[7] = 0.0
    log_c[11] = -1.0
    mock = _Mock(pd.DataFrame({"measured_mag_auto": [20.0, 25.0]}))
    return _Likelihood(log_c), mock, proposal, log_c


def test_mass_statistics_count_the_atoms_that_carry_the_mass():
    share = np.array([0.5, 0.4, 0.05, 0.05])
    stats = _mass_statistics(share)

    assert stats["atoms_for_50_percent"] == 1
    assert stats["atoms_for_90_percent"] == 2
    assert stats["atoms_for_99_percent"] == 4
    assert stats["top1_share"] == pytest.approx(0.5)
    assert stats["n_eff"] == pytest.approx(1.0 / (0.25 + 0.16 + 0.0025 + 0.0025))


def test_a_point_mass_target_reports_one_effective_atom(landscape_inputs):
    likelihood, mock, proposal, log_c = landscape_inputs

    result = likelihood_landscape(
        likelihood, mock, proposal, center=(0.0, 0.0), rows=[0], head=8
    )

    row = result.table.iloc[0]
    # exp(0) against exp(-1) and 510 atoms of exp(-60): the second atom holds
    # 1/(1+e) of the mass, so a single atom is short of half and two clear it.
    assert row["n_eff"] == pytest.approx(1.0 / ((1 / (1 + np.e)) ** 2 + (np.e / (1 + np.e)) ** 2), rel=1e-6)
    assert row["atoms_for_50_percent"] == 1
    assert row["atoms_for_99_percent"] == 2
    assert result.head_atom[0, 0] == 7
    assert result.head_atom[0, 1] == 11


def test_the_reported_head_is_sorted_and_matches_the_true_ordering(landscape_inputs):
    likelihood, mock, proposal, log_c = landscape_inputs

    result = likelihood_landscape(
        likelihood, mock, proposal, center=(0.0, 0.0), rows=[0, 1], head=16
    )

    for position in range(2):
        share = result.head_share[position]
        assert np.all(np.diff(share) <= 0)
        expected = np.argsort(log_c)[::-1][:16]
        assert set(result.head_atom[position, :2]) == set(expected[:2])
    assert result.metadata["flow_evaluations"] == 2 * len(proposal.active_indices)
    assert list(result.table["observation"]) == [0, 1]


def test_absolute_identity_survives_a_windowed_run(landscape_inputs):
    likelihood, mock, proposal, _ = landscape_inputs

    result = likelihood_landscape(
        likelihood, mock, proposal, center=(0.0, 0.0), rows=[1],
        observation_start=25000, head=4,
    )

    assert list(result.table["observation"]) == [25001]
    assert list(result.table["window_row"]) == [1]


def test_rows_outside_the_window_are_refused(landscape_inputs):
    likelihood, mock, proposal, _ = landscape_inputs

    with pytest.raises(ValueError, match="index the observation window"):
        likelihood_landscape(
            likelihood, mock, proposal, center=(0.0, 0.0), rows=[2]
        )
    with pytest.raises(ValueError, match="non-empty"):
        likelihood_landscape(
            likelihood, mock, proposal, center=(0.0, 0.0), rows=[]
        )


class _Candidates:
    def __init__(self, indices):
        self.indices = indices


class _ShortlistProposal(_Proposal):
    """Ranks atoms by a fixed order, standing in for the proxy reranker."""

    def __init__(self, coordinates, n_atoms, order):
        super().__init__(coordinates, n_atoms)
        self._order = np.asarray(order, dtype=np.int64)
        self.flow_model = None

    def candidates(self, observed, *, n_candidates, prefilter_candidates, torch_device):
        return _Candidates(self._order[None, :n_candidates])


def test_shortlist_coverage_reports_the_mass_the_exact_stratum_would_capture(
    landscape_inputs,
):
    likelihood, mock, proposal, log_c = landscape_inputs
    n_atoms = len(proposal.active_indices)
    # A ranking that finds atom 11 immediately but buries the dominant atom 7,
    # which is exactly the failure the landscape exists to detect.
    order = np.concatenate([[11], np.setdiff1d(np.arange(n_atoms), [7, 11]), [7]])
    ranked = _ShortlistProposal(proposal.coordinates, n_atoms, order)

    result = likelihood_landscape(
        likelihood, mock, ranked, center=(0.0, 0.0), rows=[0], head=4,
        shortlist_sizes=(1, n_atoms), prefilter_candidates=n_atoms,
        candidate_backend="numpy",
    )

    row = result.table.iloc[0]
    # Atom 11 alone carries exp(-1)/(exp(0)+exp(-1)) of the mass, up to the
    # negligible floor of the 510 atoms at exp(-60).
    assert row["shortlist_1_mass"] == pytest.approx(1 / (1 + np.e), rel=1e-6)
    assert not row["shortlist_1_holds_top1"]
    assert row[f"shortlist_{n_atoms}_mass"] == pytest.approx(1.0, rel=1e-9)
    assert row[f"shortlist_{n_atoms}_holds_top1"]


def test_a_shortlist_larger_than_the_prefilter_is_refused(landscape_inputs):
    likelihood, mock, proposal, _ = landscape_inputs

    with pytest.raises(ValueError, match="prefilter cannot be smaller"):
        likelihood_landscape(
            likelihood, mock, proposal, center=(0.0, 0.0), rows=[0],
            shortlist_sizes=(64,), prefilter_candidates=32,
        )
