from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from sbsi.amortized_proposal import (
    AmortizedProposal,
    ContrastiveRetrievalProposal,
    topk_unique_candidates,
)
from sbsi.catalogue_sampling import ProposalCandidates, ProposalCoordinateTable


TARGET_NAMES = ("measured_a", "measured_b")
ATOM_FEATURE_NAMES = ("atom_a", "atom_b")


@dataclass(frozen=True)
class _TensorView:
    context: torch.Tensor
    log_detected_mass: torch.Tensor


class _ConditionPreprocessor:
    feature_names = ATOM_FEATURE_NAMES


class _FlowModel:
    device = "cpu"
    condition_preprocessor = _ConditionPreprocessor()


class _Likelihood:
    tensor_native_available = True
    flow_model = _FlowModel()

    def __init__(self, context: np.ndarray):
        context_tensor = torch.as_tensor(context, dtype=torch.float32)
        self._view = _TensorView(
            context=context_tensor,
            log_detected_mass=torch.zeros(len(context_tensor)),
        )
        self.exact_score_calls = 0

    def _tensor_view(self, g1: float, g2: float) -> _TensorView:
        assert (g1, g2) == (0.0, 0.0)
        return self._view

    @staticmethod
    def observed_target_tensor(observed) -> torch.Tensor:
        return torch.as_tensor(observed, dtype=torch.float32)

    def log_importance_weights_tensor(self, *args, **kwargs):
        self.exact_score_calls += 1
        raise AssertionError("direct learned retrieval must not evaluate the flow")


def _coordinates(n_atoms: int) -> ProposalCoordinateTable:
    values = np.column_stack(
        (
            np.arange(n_atoms, dtype=np.float64),
            -np.arange(n_atoms, dtype=np.float64),
        )
    )
    return ProposalCoordinateTable(
        values=values,
        target_names=TARGET_NAMES,
        center=np.zeros(2),
        scale=np.ones(2),
    )


def _ranked_model() -> AmortizedProposal:
    """Return a model whose learned score increases with atom feature zero."""

    model = AmortizedProposal(
        query_dim=len(TARGET_NAMES),
        atom_dim=len(ATOM_FEATURE_NAMES),
        embedding_dim=3,
        hidden_dim=4,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # SiLU is strictly increasing on the non-negative atom values used by
        # these tests. The embedding dot product remains exactly zero, so the
        # atom-bias tower alone defines a known strict ranking.
        model.atom_bias[0].weight[0, 0] = 1.0
        model.atom_bias[2].weight[0, 0] = 1.0
    return model


def _write_checkpoint(
    path: Path,
    *,
    target_names=TARGET_NAMES,
    atom_feature_names=ATOM_FEATURE_NAMES,
    g=(0.0, 0.0),
) -> Path:
    model = _ranked_model()
    torch.save(
        {
            "query_dim": len(TARGET_NAMES),
            "atom_dim": len(ATOM_FEATURE_NAMES),
            "embedding_dim": model.embedding_dim,
            "hidden_dim": 4,
            "target_names": list(target_names),
            "atom_feature_names": list(atom_feature_names),
            "g": list(g),
            "state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _proposal(tmp_path: Path, *, retrieval_object_chunk: int = 2):
    n_atoms = 6
    context = np.column_stack(
        (
            np.arange(n_atoms, dtype=np.float32),
            np.zeros(n_atoms, dtype=np.float32),
        )
    )
    likelihood = _Likelihood(context)
    # Atom one is deliberately inactive. It has a high-enough learned score
    # to expose an accidental full-row rather than active-support retrieval.
    prior = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    proposal = ContrastiveRetrievalProposal(
        _coordinates(n_atoms),
        prior,
        local_base_weights=np.ones(n_atoms),
        likelihood=likelihood,
        checkpoint=_write_checkpoint(tmp_path / "model.pt"),
        center=(0.0, 0.0),
        contrastive_pool=4,
        retrieval_object_chunk=retrieval_object_chunk,
        rerank_object_chunk=1,
        atom_chunk=2,
    )
    return proposal, likelihood


def test_synthetic_checkpoint_loads_recovered_silu_architecture(tmp_path):
    proposal, _ = _proposal(tmp_path)

    assert isinstance(proposal.model.query_encoder[1], nn.SiLU)
    assert isinstance(proposal.model.atom_encoder[1], nn.SiLU)
    assert isinstance(proposal.model.atom_bias[1], nn.SiLU)
    assert proposal.model.embedding_dim == 3
    assert proposal.checkpoint == (tmp_path / "model.pt").resolve()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("target_names", ("wrong_a", "wrong_b"), "target names do not match"),
        (
            "atom_feature_names",
            ("wrong_a", "wrong_b"),
            "atom features do not match",
        ),
        ("g", (0.01, 0.0), "must declare shear-free g=0 training"),
    ),
)
def test_checkpoint_schema_mismatches_fail_closed(
    tmp_path, field, value, message
):
    n_atoms = 3
    likelihood = _Likelihood(np.zeros((n_atoms, 2), dtype=np.float32))
    checkpoint_arguments = {field: value}
    checkpoint = _write_checkpoint(tmp_path / "bad.pt", **checkpoint_arguments)

    with pytest.raises(RuntimeError, match=message):
        ContrastiveRetrievalProposal(
            _coordinates(n_atoms),
            np.ones(n_atoms),
            local_base_weights=np.ones(n_atoms),
            likelihood=likelihood,
            checkpoint=checkpoint,
            center=(0.0, 0.0),
            contrastive_pool=2,
            retrieval_object_chunk=1,
            rerank_object_chunk=1,
            atom_chunk=2,
        )


def test_direct_candidates_are_unique_deterministic_learned_prefix(tmp_path):
    proposal, likelihood = _proposal(tmp_path, retrieval_object_chunk=1)
    observed = np.array([[0.0, 0.0], [3.0, -2.0]], dtype=np.float32)

    first = proposal.candidates(
        observed,
        n_candidates=3,
        prefilter_candidates=999,
        torch_device="cpu",
    )
    second = proposal.candidates(
        observed,
        n_candidates=3,
        prefilter_candidates=1,
        torch_device=torch.device("cpu"),
    )

    expected = np.array([[5, 4, 3], [5, 4, 3]])
    assert isinstance(first, ProposalCandidates)
    assert not hasattr(first, "log_target")
    np.testing.assert_array_equal(first.indices, expected)
    np.testing.assert_array_equal(second.indices, expected)
    np.testing.assert_allclose(first.distances, second.distances)
    assert all(len(np.unique(row)) == len(row) for row in first.indices)
    assert 1 not in first.indices
    assert likelihood.exact_score_calls == 0
    assert proposal.candidate_flow_evaluations == 0


def test_direct_candidates_validate_device_and_k_bounds(tmp_path):
    proposal, _ = _proposal(tmp_path)
    observed = np.zeros((1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="requires its likelihood Torch device"):
        proposal.candidates(observed, n_candidates=1, torch_device=None)
    with pytest.raises(ValueError, match="requires its likelihood Torch device"):
        proposal.candidates(observed, n_candidates=1, torch_device="cuda")
    with pytest.raises(ValueError, match="requested K exceeds"):
        proposal.candidates(observed, n_candidates=0, torch_device="cpu")
    with pytest.raises(ValueError, match="requested K exceeds"):
        proposal.candidates(observed, n_candidates=5, torch_device="cpu")


def test_retrieval_precision_is_pinned_without_leaking_global_state(tmp_path):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        proposal, _ = _proposal(tmp_path)
        assert proposal.retrieval_matmul_precision == "high"
        assert torch.get_float32_matmul_precision() == "highest"

        proposal.candidates(
            np.zeros((1, 2), dtype=np.float32),
            n_candidates=2,
            torch_device="cpu",
        )
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(previous)


def test_learned_score_gaps_never_enter_adapted_probabilities(tmp_path):
    proposal, _ = _proposal(tmp_path)
    observed = np.zeros((1, 2), dtype=np.float32)
    indices = np.array([[5, 4, 3, 2]], dtype=np.int64)

    proposal._contrastive_candidates = lambda targets: (
        indices,
        np.array([[1000.0, 20.0, 2.0, -500.0]]),
    )
    wide_gap = proposal.candidates(
        observed, n_candidates=3, torch_device="cpu"
    )
    proposal._contrastive_candidates = lambda targets: (
        indices,
        np.array([[3.0, 2.999, 2.998, 2.997]]),
    )
    narrow_gap = proposal.candidates(
        observed, n_candidates=3, torch_device="cpu"
    )
    assert not np.allclose(wide_gap.distances, narrow_gap.distances)

    exact_log_target = np.log(np.array([[0.2, 0.3, 0.5]]))
    wide_draw = proposal.draw_adapted(
        wide_gap,
        exact_log_target,
        n_draws=256,
        epsilon=0.25,
        seed=7821,
    )
    narrow_draw = proposal.draw_adapted(
        narrow_gap,
        exact_log_target,
        n_draws=256,
        epsilon=0.25,
        seed=7821,
    )

    np.testing.assert_array_equal(wide_draw.indices, narrow_draw.indices)
    np.testing.assert_array_equal(
        wide_draw.global_component, narrow_draw.global_component
    )
    np.testing.assert_array_equal(wide_draw.local_member, narrow_draw.local_member)
    np.testing.assert_allclose(wide_draw.probability, narrow_draw.probability)


def test_topk_unique_candidates_is_repeatable_and_removes_duplicates():
    indices = np.array([[9, 2, 9, 7, 3], [4, 4, 8, 1, 6]])
    score = np.array([[1.0, 5.0, 1.0, 4.0, 3.0], [6.0, 6.0, 2.0, 5.0, 4.0]])

    first_indices, first_score = topk_unique_candidates(indices, score, 3)
    second_indices, second_score = topk_unique_candidates(indices, score, 3)

    np.testing.assert_array_equal(first_indices, [[2, 7, 3], [4, 1, 6]])
    np.testing.assert_allclose(first_score, [[5.0, 4.0, 3.0], [6.0, 5.0, 4.0]])
    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_array_equal(first_score, second_score)
    assert all(len(np.unique(row)) == len(row) for row in first_indices)
