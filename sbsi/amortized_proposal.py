"""Compact learned retrieval with exact local likelihood reranking."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .catalogue_likelihood import CatalogueLikelihood
from .catalogue_sampling import (
    DefensiveLocalProposal,
    ProposalCandidates,
    ProposalCoordinateTable,
)


class AmortizedProposal(nn.Module):
    """Small two-tower retrieval score with precomputable atom embeddings."""

    def __init__(self, query_dim, atom_dim, embedding_dim=16, hidden_dim=64):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.atom_bias = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_scale = nn.Parameter(torch.zeros(()))

    def encode_query(self, query):
        return self.query_encoder(query)

    def encode_atom(self, atom):
        return self.atom_encoder(atom), self.atom_bias(atom).squeeze(-1)

    def forward(self, query, atom, log_detected_mass):
        query_embedding = self.encode_query(query)
        atom_embedding, atom_bias = self.encode_atom(atom)
        similarity = torch.einsum("bd,bsd->bs", query_embedding, atom_embedding)
        scale = self.log_scale.exp() / np.sqrt(self.embedding_dim)
        return log_detected_mass + atom_bias + scale * similarity


def encode_atoms(model, atom_context, active, chunk, device):
    """Encode the active catalogue once for global maximum-inner-product search."""

    embedding = torch.empty(
        (len(active), model.embedding_dim), dtype=torch.float32, device=device
    )
    bias = torch.empty(len(active), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(active), chunk):
            stop = min(start + chunk, len(active))
            rows = torch.as_tensor(
                active[start:stop], dtype=torch.long, device=device
            )
            embedding[start:stop], bias[start:stop] = model.encode_atom(
                atom_context[rows]
            )
    return embedding, bias


def topk_unique_candidates(indices, score, k):
    """Select exact-score top-K atoms per row after masking duplicate pool entries."""

    indices = np.asarray(indices, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if indices.ndim != 2 or score.shape != indices.shape:
        raise ValueError("candidate indices and scores must be aligned matrices")
    if k <= 0 or k > indices.shape[1]:
        raise ValueError("K must lie inside the candidate-pool width")
    selected_indices = np.empty((len(indices), int(k)), dtype=np.int64)
    selected_score = np.empty((len(indices), int(k)), dtype=np.float64)
    for row in range(len(indices)):
        _, first = np.unique(indices[row], return_index=True)
        if len(first) < k:
            raise RuntimeError("candidate union contains fewer than K unique atoms")
        unique_score = np.full(indices.shape[1], -np.inf, dtype=np.float64)
        unique_score[first] = score[row, first]
        selected = np.argpartition(unique_score, -int(k))[-int(k) :]
        order = np.argsort(-unique_score[selected], kind="stable")
        selected = selected[order]
        selected_indices[row] = indices[row, selected]
        selected_score[row] = unique_score[selected]
    return selected_indices, selected_score


class HybridExactRerankProposal(DefensiveLocalProposal):
    """Infer V1 safety retrieval plus global learned retrieval and exact reranking.

    The learned score is used only to retrieve a broad global prefix.  The union
    with the established diagonal-Gaussian support is evaluated by the actual
    catalogue likelihood at the declared inference centre, and only its exact
    top-K atoms are returned.  Downstream ``draw_adapted`` therefore continues
    to assign exact local probabilities and exact importance corrections.
    """

    def __init__(
        self,
        coordinates: ProposalCoordinateTable,
        prior_weights: np.ndarray,
        *,
        local_base_weights: np.ndarray,
        likelihood: CatalogueLikelihood,
        checkpoint: str | Path,
        center: Sequence[float],
        contrastive_pool: int = 65536,
        retrieval_object_chunk: int = 16,
        rerank_object_chunk: int = 16,
        atom_chunk: int = 65536,
        build_coordinate_tree: bool = True,
        retrieval_matmul_precision: str = "high",
    ):
        super().__init__(
            coordinates,
            prior_weights,
            local_base_weights=local_base_weights,
            build_coordinate_tree=build_coordinate_tree,
        )
        if not likelihood.tensor_native_available:
            raise TypeError("hybrid exact reranking requires a tensor-native flow")
        center = np.asarray(center, dtype=np.float64)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("hybrid proposal centre must be a finite two-vector")
        if (
            contrastive_pool <= 0
            or retrieval_object_chunk <= 0
            or rerank_object_chunk <= 0
            or atom_chunk <= 0
        ):
            raise ValueError("hybrid proposal sizes must be positive")
        self.likelihood = likelihood
        self.center = (float(center[0]), float(center[1]))
        self.contrastive_pool = min(int(contrastive_pool), len(self.active_indices))
        self.retrieval_object_chunk = int(retrieval_object_chunk)
        self.rerank_object_chunk = int(rerank_object_chunk)
        self.atom_chunk = int(atom_chunk)
        if retrieval_matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("invalid learned-retrieval matmul precision")
        self.retrieval_matmul_precision = retrieval_matmul_precision
        self.checkpoint = Path(checkpoint).resolve()
        device = torch.device(likelihood.flow_model.device)
        payload = torch.load(self.checkpoint, map_location=device, weights_only=False)
        expected_targets = list(coordinates.target_names)
        expected_features = list(
            likelihood.flow_model.condition_preprocessor.feature_names
        )
        if payload.get("target_names") != expected_targets:
            raise RuntimeError("amortized checkpoint target names do not match")
        if payload.get("atom_feature_names") != expected_features:
            raise RuntimeError("amortized checkpoint atom features do not match")
        if payload.get("g") != [0.0, 0.0]:
            raise RuntimeError("amortized retriever must declare shear-free g=0 training")
        tensor_view = likelihood._tensor_view(0.0, 0.0)
        self.model = AmortizedProposal(
            query_dim=int(payload["query_dim"]),
            atom_dim=int(payload["atom_dim"]),
            embedding_dim=int(payload["embedding_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
        ).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self._active_tensor = torch.as_tensor(
            self.active_indices, dtype=torch.long, device=device
        )
        previous_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision(self.retrieval_matmul_precision)
            self._atom_embedding, atom_bias = encode_atoms(
                self.model,
                tensor_view.context,
                self.active_indices,
                self.atom_chunk,
                device,
            )
        finally:
            torch.set_float32_matmul_precision(previous_precision)
        self._contrastive_base = (
            tensor_view.log_detected_mass.index_select(0, self._active_tensor)
            + atom_bias
        )
        self.candidate_flow_evaluations = 0
        self.candidate_flow_seconds = 0.0
        self.candidate_calls = 0
        self.retrieval_batches = 0
        self.retrieval_objects = 0

    def _contrastive_candidates(self, observed_targets):
        rows = []
        score_rows = []
        scale = self.model.log_scale.exp() / np.sqrt(self.model.embedding_dim)
        previous_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision(self.retrieval_matmul_precision)
            with torch.no_grad():
                for start in range(
                    0, len(observed_targets), self.retrieval_object_chunk
                ):
                    stop = min(
                        start + self.retrieval_object_chunk,
                        len(observed_targets),
                    )
                    self.retrieval_batches += 1
                    self.retrieval_objects += stop - start
                    query = self.model.encode_query(observed_targets[start:stop])
                    score = self._contrastive_base[None, :] + scale * torch.matmul(
                        query, self._atom_embedding.T
                    )
                    selected = torch.topk(
                        score,
                        k=self.contrastive_pool,
                        dim=1,
                        largest=True,
                        sorted=True,
                    ).indices
                    rows.append(
                        self._active_tensor[selected]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64)
                    )
                    score_rows.append(
                        torch.gather(score, 1, selected)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    del query, score, selected
        finally:
            torch.set_float32_matmul_precision(previous_precision)
        return np.concatenate(rows, axis=0), np.concatenate(score_rows, axis=0)

    def candidates(
        self,
        observed,
        *,
        n_candidates: int,
        prefilter_candidates: int | None = None,
        torch_device: str | torch.device | None = None,
    ) -> ProposalCandidates:
        device = torch.device(self.likelihood.flow_model.device)
        if torch_device is None or torch.device(torch_device) != device:
            raise ValueError("hybrid exact reranking requires its likelihood Torch device")
        self.candidate_calls += 1
        diagonal = super().candidates(
            observed,
            n_candidates=int(n_candidates),
            prefilter_candidates=prefilter_candidates,
            torch_device=device,
        )
        observed_targets = self.likelihood.observed_target_tensor(observed)
        contrastive, _ = self._contrastive_candidates(observed_targets)
        pool = np.concatenate((diagonal.indices, contrastive), axis=1)
        flow_started = time.perf_counter()
        score = self.likelihood.log_importance_weights_tensor(
            observed,
            *self.center,
            atom_indices=pool,
            proposal_probability=np.ones_like(pool, dtype=np.float64),
            observed_targets=observed_targets,
            object_chunk=min(len(pool), self.rerank_object_chunk),
            atom_chunk=self.atom_chunk,
        ).detach().cpu().numpy().astype(np.float64)
        self.candidate_flow_seconds += time.perf_counter() - flow_started
        self.candidate_flow_evaluations += int(pool.size)
        selected, selected_score = topk_unique_candidates(
            pool, score, int(n_candidates)
        )
        distances = selected_score[:, :1] - selected_score
        finite = np.isfinite(distances)
        row_max = np.max(np.where(finite, distances, 0.0), axis=1, keepdims=True)
        distances = np.where(finite, distances, row_max + 1.0)
        return ProposalCandidates(
            selected,
            distances,
            distances[:, -1].copy(),
        )


class ContrastiveRetrievalProposal(HybridExactRerankProposal):
    """Use the learned model only to retrieve support; weight it exactly later.

    This deliberately omits the diagonal safety arm and the broad exact-rerank
    pool.  The learned score chooses K atoms, while ``draw_adapted`` continues
    to compute the actual Like V3.2 target and exact local probabilities on
    those atoms.  Learned scores are therefore never importance weights.
    """

    def candidates(
        self,
        observed,
        *,
        n_candidates: int,
        prefilter_candidates: int | None = None,
        torch_device: str | torch.device | None = None,
    ) -> ProposalCandidates:
        del prefilter_candidates
        device = torch.device(self.likelihood.flow_model.device)
        if torch_device is None or torch.device(torch_device) != device:
            raise ValueError("contrastive retrieval requires its likelihood Torch device")
        if n_candidates <= 0 or n_candidates > self.contrastive_pool:
            raise ValueError("requested K exceeds the encoded contrastive prefix")
        self.candidate_calls += 1
        observed_targets = self.likelihood.observed_target_tensor(observed)
        indices, learned_score = self._contrastive_candidates(observed_targets)
        indices = indices[:, : int(n_candidates)]
        learned_score = learned_score[:, : int(n_candidates)]
        distances = learned_score[:, :1] - learned_score
        return ProposalCandidates(
            indices,
            distances,
            distances[:, -1].copy(),
        )


__all__ = [
    "AmortizedProposal",
    "ContrastiveRetrievalProposal",
    "HybridExactRerankProposal",
    "encode_atoms",
    "topk_unique_candidates",
]
