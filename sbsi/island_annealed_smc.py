"""Outcome-independent finite-state island annealed-SMC prototype.

The implementation deliberately has no catalogue, likelihood, CLI, or release
integration.  It estimates the normalizing constant of a finite non-negative
target from a full-support base law with a fixed temperature schedule.  Each
island is an independent bootstrap particle filter, so the mean of the island
normalizer estimates is unbiased in exact arithmetic and the island sample
variance divided by the island count estimates the variance of that mean.  The
log of the mean normalizer is not itself an unbiased log-normalizer estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


_RNG_VERSION = 1
_DOMAIN_INITIAL = 1
_DOMAIN_RESAMPLE = 2
_DOMAIN_MOVE_KIND = 3
_DOMAIN_INDEPENDENCE_PROPOSAL = 4
_DOMAIN_GRAPH_PROPOSAL = 5
_DOMAIN_MOVE_ACCEPT = 6


def _exact_nonnegative_integer(value, name: str, maximum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        or int(value) > maximum
    ):
        raise ValueError(f"{name} must be a non-negative integer <= {maximum}")
    return int(value)


def _positive_integer(value, name: str) -> int:
    parsed = _exact_nonnegative_integer(value, name, np.iinfo(np.int64).max)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _u64_words(value: int) -> tuple[int, int]:
    return value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF


def _keyed_rng(
    *,
    seed: int,
    object_id: int,
    island_id: int,
    stage: int,
    domain: int,
    move: int = 0,
) -> np.random.Generator:
    seed_low, seed_high = _u64_words(seed)
    object_low, object_high = _u64_words(object_id)
    island_low, island_high = _u64_words(island_id)
    words = [
        seed_low,
        seed_high,
        object_low,
        object_high,
        island_low,
        island_high,
        int(stage),
        int(domain),
        int(move),
        _RNG_VERSION,
    ]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if np.isneginf(maximum):
        return -np.inf
    return maximum + float(
        np.log(np.exp(values - maximum, dtype=np.float64).sum(dtype=np.float64))
    )


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if np.isneginf(numerator) and np.isneginf(denominator):
        return 0.0
    return numerator - denominator


def _log_bridge_density(
    log_q0: np.ndarray, log_target: np.ndarray, beta: float
) -> np.ndarray:
    if beta == 0.0:
        return log_q0.copy()
    if beta == 1.0:
        return log_target.copy()
    return (1.0 - beta) * log_q0 + beta * log_target


def _normalize_problem(
    log_target, q0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.ascontiguousarray(log_target, dtype=np.float64)
    base = np.ascontiguousarray(q0, dtype=np.float64)
    if target.ndim != 1 or not len(target):
        raise ValueError("log_target must be a non-empty one-dimensional array")
    if base.shape != target.shape:
        raise ValueError("q0 must have the same one-dimensional shape as log_target")
    if np.isnan(target).any() or np.isposinf(target).any():
        raise ValueError("log_target must contain only finite values or -inf")
    if not np.isfinite(target).any():
        raise ValueError("the finite-state target must have positive total mass")
    if not np.isfinite(base).all() or (base <= 0.0).any():
        raise ValueError("q0 must be finite and strictly positive on every state")
    total = float(base.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("q0 must have a finite positive total")
    base = np.ascontiguousarray(base / total, dtype=np.float64)
    if not np.isfinite(base).all() or (base <= 0.0).any():
        raise ValueError("normalizing q0 lost full support")
    return target, base, np.log(base)


def _validate_betas(betas) -> np.ndarray:
    result = np.ascontiguousarray(betas, dtype=np.float64)
    if (
        result.ndim != 1
        or len(result) < 2
        or not np.isfinite(result).all()
        or result[0] != 0.0
        or result[-1] != 1.0
        or not np.all(np.diff(result) > 0.0)
    ):
        raise ValueError(
            "betas must be a finite strictly increasing vector from 0 to 1"
        )
    return result


@dataclass(frozen=True)
class UndirectedGraph:
    """Canonical finite undirected graph used by the optional local MH move."""

    n_states: int
    edges: np.ndarray
    _indptr: np.ndarray = field(init=False, repr=False)
    _indices: np.ndarray = field(init=False, repr=False)
    _degree: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        n_states = _positive_integer(self.n_states, "n_states")
        raw = np.asarray(self.edges)
        if raw.ndim != 2 or raw.shape[1:] != (2,) or raw.dtype.kind not in "iu":
            raise ValueError("edges must be an integer array with shape (n_edges, 2)")
        edges = np.ascontiguousarray(raw, dtype=np.int64).copy()
        if len(edges):
            if (edges < 0).any() or (edges >= n_states).any():
                raise ValueError("graph edge endpoint is outside the state space")
            edges.sort(axis=1)
            if np.any(edges[:, 0] == edges[:, 1]):
                raise ValueError("graph self-edges are not allowed")
            order = np.lexsort((edges[:, 1], edges[:, 0]))
            edges = np.ascontiguousarray(edges[order])
            if np.any(np.all(edges[1:] == edges[:-1], axis=1)):
                raise ValueError("graph edges must be unique as undirected pairs")

        degree = np.zeros(n_states, dtype=np.int64)
        if len(edges):
            np.add.at(degree, edges[:, 0], 1)
            np.add.at(degree, edges[:, 1], 1)
        indptr = np.empty(n_states + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(degree, out=indptr[1:])
        indices = np.empty(int(indptr[-1]), dtype=np.int64)
        cursor = indptr[:-1].copy()
        for left, right in edges:
            indices[cursor[left]] = right
            cursor[left] += 1
            indices[cursor[right]] = left
            cursor[right] += 1
        for state in range(n_states):
            indices[indptr[state] : indptr[state + 1]].sort()
        for array in (edges, degree, indptr, indices):
            array.setflags(write=False)
        object.__setattr__(self, "n_states", n_states)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "_degree", degree)
        object.__setattr__(self, "_indptr", indptr)
        object.__setattr__(self, "_indices", indices)

    @property
    def degree(self) -> np.ndarray:
        return self._degree

    def neighbors(self, state: int) -> np.ndarray:
        parsed = _exact_nonnegative_integer(
            state, "state", np.iinfo(np.int64).max
        )
        if parsed >= self.n_states:
            raise ValueError("state is outside the graph")
        return self._indices[self._indptr[parsed] : self._indptr[parsed + 1]]


@dataclass(frozen=True)
class FiniteStateIslandSMCResult:
    """Immutable result of independent fixed-schedule SMC islands."""

    normalizer: float
    log_normalizer: float
    variance: float
    log_variance: float
    standard_error: float
    island_ids: np.ndarray
    island_normalizer: np.ndarray
    island_log_normalizer: np.ndarray
    stage_log_increment: np.ndarray
    stage_effective_sample_size: np.ndarray
    stage_move_acceptance_rate: np.ndarray
    stage_unique_ancestor_count: np.ndarray
    collapsed_stage: np.ndarray
    terminal_particles: np.ndarray
    betas: np.ndarray

    def __post_init__(self):
        arrays = {
            "island_ids": np.ascontiguousarray(self.island_ids, dtype=np.int64),
            "island_normalizer": np.ascontiguousarray(
                self.island_normalizer, dtype=np.float64
            ),
            "island_log_normalizer": np.ascontiguousarray(
                self.island_log_normalizer, dtype=np.float64
            ),
            "stage_log_increment": np.ascontiguousarray(
                self.stage_log_increment, dtype=np.float64
            ),
            "stage_effective_sample_size": np.ascontiguousarray(
                self.stage_effective_sample_size, dtype=np.float64
            ),
            "stage_move_acceptance_rate": np.ascontiguousarray(
                self.stage_move_acceptance_rate, dtype=np.float64
            ),
            "stage_unique_ancestor_count": np.ascontiguousarray(
                self.stage_unique_ancestor_count, dtype=np.int64
            ),
            "collapsed_stage": np.ascontiguousarray(
                self.collapsed_stage, dtype=np.int64
            ),
            "terminal_particles": np.ascontiguousarray(
                self.terminal_particles, dtype=np.int64
            ),
            "betas": np.ascontiguousarray(self.betas, dtype=np.float64),
        }
        islands = len(arrays["island_ids"])
        stages = len(arrays["betas"]) - 1
        try:
            _validate_betas(arrays["betas"])
        except ValueError as error:
            raise ValueError("finite-state island SMC result betas are invalid") from error
        if (
            islands < 2
            or (arrays["island_ids"] < 0).any()
            or len(np.unique(arrays["island_ids"])) != islands
            or arrays["island_normalizer"].shape != (islands,)
            or arrays["island_log_normalizer"].shape != (islands,)
            or arrays["stage_log_increment"].shape != (islands, stages)
            or arrays["stage_effective_sample_size"].shape != (islands, stages)
            or arrays["stage_move_acceptance_rate"].shape != (islands, stages)
            or arrays["stage_unique_ancestor_count"].shape != (islands, stages)
            or arrays["collapsed_stage"].shape != (islands,)
            or arrays["terminal_particles"].ndim != 2
            or arrays["terminal_particles"].shape[0] != islands
            or np.isnan(arrays["island_normalizer"]).any()
            or (arrays["island_normalizer"] < 0.0).any()
            or np.isnan(arrays["island_log_normalizer"]).any()
            or np.isposinf(arrays["island_log_normalizer"]).any()
            or np.isnan(arrays["stage_log_increment"]).any()
            or np.isposinf(arrays["stage_log_increment"]).any()
            or not np.isfinite(arrays["stage_effective_sample_size"]).all()
            or (arrays["stage_effective_sample_size"] < 0.0).any()
            or not np.isfinite(arrays["stage_move_acceptance_rate"]).all()
            or (arrays["stage_move_acceptance_rate"] < 0.0).any()
            or (arrays["stage_move_acceptance_rate"] > 1.0).any()
            or (arrays["stage_unique_ancestor_count"] < 0).any()
            or (arrays["collapsed_stage"] < 0).any()
            or (arrays["collapsed_stage"] > stages).any()
            or (arrays["terminal_particles"] < 0).any()
        ):
            raise ValueError("finite-state island SMC result arrays are invalid")
        if (
            np.isnan(self.normalizer)
            or self.normalizer < 0.0
            or np.isnan(self.log_normalizer)
            or np.isposinf(self.log_normalizer)
            or np.isnan(self.variance)
            or self.variance < 0.0
            or np.isnan(self.log_variance)
            or np.isnan(self.standard_error)
            or self.standard_error < 0.0
        ):
            raise ValueError("finite-state island SMC result scalars are invalid")
        with np.errstate(over="ignore"):
            expected_island_normalizer = np.exp(
                arrays["island_log_normalizer"], dtype=np.float64
            )
        expected_log_from_stages = arrays["stage_log_increment"].sum(
            axis=1, dtype=np.float64
        )
        (
            expected_normalizer,
            expected_log_normalizer,
            expected_variance,
            expected_log_variance,
            expected_standard_error,
            _,
        ) = _summarize_islands(
            arrays["island_ids"], arrays["island_log_normalizer"]
        )
        scalar_observed = np.asarray(
            (
                self.normalizer,
                self.log_normalizer,
                self.variance,
                self.log_variance,
                self.standard_error,
            ),
            dtype=np.float64,
        )
        scalar_expected = np.asarray(
            (
                expected_normalizer,
                expected_log_normalizer,
                expected_variance,
                expected_log_variance,
                expected_standard_error,
            ),
            dtype=np.float64,
        )
        if (
            not np.array_equal(
                arrays["island_normalizer"], expected_island_normalizer
            )
            or not np.allclose(
                arrays["island_log_normalizer"],
                expected_log_from_stages,
                rtol=64.0 * np.finfo(np.float64).eps,
                atol=64.0 * np.finfo(np.float64).eps,
            )
            or not np.allclose(
                scalar_observed,
                scalar_expected,
                rtol=64.0 * np.finfo(np.float64).eps,
                atol=0.0,
                equal_nan=False,
            )
        ):
            raise ValueError("finite-state island SMC result identities are invalid")
        for name, array in arrays.items():
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def _summarize_islands(
    island_ids: np.ndarray, island_log_normalizer: np.ndarray
) -> tuple[float, float, float, float, float, np.ndarray]:
    order = np.argsort(island_ids, kind="stable")
    logs = island_log_normalizer[order]
    islands = len(logs)
    finite = np.isfinite(logs)
    if not finite.any():
        return 0.0, -np.inf, 0.0, -np.inf, 0.0, np.zeros(islands)
    shift = float(logs[finite].max())
    scaled = np.zeros(islands, dtype=np.float64)
    scaled[finite] = np.exp(logs[finite] - shift, dtype=np.float64)
    mean_scaled = float(scaled.mean(dtype=np.float64))
    log_normalizer = shift + float(np.log(mean_scaled))
    centered = scaled - mean_scaled
    squared_sum = float(np.square(centered).sum(dtype=np.float64))
    if squared_sum == 0.0:
        log_variance = -np.inf
        variance = 0.0
        standard_error = 0.0
    else:
        with np.errstate(over="ignore"):
            log_variance = (
                2.0 * shift
                + float(np.log(squared_sum))
                - float(np.log(islands * (islands - 1)))
            )
            variance = float(np.exp(log_variance))
            standard_error = float(np.exp(0.5 * log_variance))
    with np.errstate(over="ignore"):
        normalizer = float(np.exp(log_normalizer))
        island_normalizer = np.exp(island_log_normalizer, dtype=np.float64)
    return (
        normalizer,
        log_normalizer,
        variance,
        log_variance,
        standard_error,
        island_normalizer,
    )


def _validate_graph_probability(
    graph: Optional[UndirectedGraph], graph_move_probability: float, n_states: int
) -> float:
    probability = float(graph_move_probability)
    if not np.isfinite(probability) or probability < 0.0 or probability >= 1.0:
        raise ValueError("graph_move_probability must lie in [0, 1)")
    if graph is not None and graph.n_states != n_states:
        raise ValueError("graph state count does not match the finite target")
    if probability > 0.0 and (graph is None or not len(graph.edges)):
        raise ValueError("a nonempty graph is required for graph moves")
    return probability


def annealed_mh_transition_matrix(
    log_target,
    q0,
    *,
    beta: float,
    graph: Optional[UndirectedGraph] = None,
    graph_move_probability: float = 0.0,
) -> np.ndarray:
    """Return the exact one-move finite-state MH transition matrix.

    This dense helper is intended for tiny-state algebra and detailed-balance
    audits.  Production-sized code should use :func:`run_island_annealed_smc`.
    """

    target, base, log_q0 = _normalize_problem(log_target, q0)
    beta = float(beta)
    if not np.isfinite(beta) or beta < 0.0 or beta > 1.0:
        raise ValueError("beta must lie in [0, 1]")
    graph_probability = _validate_graph_probability(
        graph, graph_move_probability, len(target)
    )
    log_weight = target - log_q0
    log_gamma = _log_bridge_density(log_q0, target, beta)
    states = len(target)

    independence = np.zeros((states, states), dtype=np.float64)
    for current in range(states):
        off_diagonal = 0.0
        current_log_weight = 0.0 if beta == 0.0 else beta * log_weight[current]
        for proposal in range(states):
            if proposal == current:
                continue
            proposal_log_weight = (
                0.0 if beta == 0.0 else beta * log_weight[proposal]
            )
            log_ratio = _safe_log_ratio(
                float(proposal_log_weight), float(current_log_weight)
            )
            probability = base[proposal] * min(1.0, float(np.exp(min(0.0, log_ratio))))
            independence[current, proposal] = probability
            off_diagonal += probability
        independence[current, current] = 1.0 - off_diagonal

    if graph_probability == 0.0:
        return independence

    graph_matrix = np.zeros((states, states), dtype=np.float64)
    for current in range(states):
        neighbors = graph.neighbors(current)
        if not len(neighbors):
            graph_matrix[current, current] = 1.0
            continue
        off_diagonal = 0.0
        for proposal in neighbors:
            log_ratio = _safe_log_ratio(
                float(log_gamma[proposal]), float(log_gamma[current])
            )
            log_ratio += float(
                np.log(graph.degree[current]) - np.log(graph.degree[proposal])
            )
            probability = (1.0 / len(neighbors)) * min(
                1.0, float(np.exp(min(0.0, log_ratio)))
            )
            graph_matrix[current, proposal] = probability
            off_diagonal += probability
        graph_matrix[current, current] = 1.0 - off_diagonal
    return (
        (1.0 - graph_probability) * independence
        + graph_probability * graph_matrix
    )


def _move_particles(
    particles: np.ndarray,
    *,
    log_target: np.ndarray,
    q0: np.ndarray,
    log_q0: np.ndarray,
    beta: float,
    graph: Optional[UndirectedGraph],
    graph_probability: float,
    seed: int,
    object_id: int,
    island_id: int,
    stage: int,
    moves_per_stage: int,
) -> tuple[np.ndarray, float]:
    particles = particles.copy()
    accepted = 0
    attempted = len(particles) * moves_per_stage
    log_weight = log_target - log_q0
    log_gamma = _log_bridge_density(log_q0, log_target, beta)
    for move in range(moves_per_stage):
        kind_rng = _keyed_rng(
            seed=seed,
            object_id=object_id,
            island_id=island_id,
            stage=stage,
            domain=_DOMAIN_MOVE_KIND,
            move=move,
        )
        graph_move = kind_rng.random(len(particles)) < graph_probability
        proposal = particles.copy()

        independence_positions = np.flatnonzero(~graph_move)
        if len(independence_positions):
            proposal_rng = _keyed_rng(
                seed=seed,
                object_id=object_id,
                island_id=island_id,
                stage=stage,
                domain=_DOMAIN_INDEPENDENCE_PROPOSAL,
                move=move,
            )
            proposal[independence_positions] = proposal_rng.choice(
                len(q0), size=len(independence_positions), p=q0
            )

        graph_positions = np.flatnonzero(graph_move)
        if len(graph_positions):
            graph_rng = _keyed_rng(
                seed=seed,
                object_id=object_id,
                island_id=island_id,
                stage=stage,
                domain=_DOMAIN_GRAPH_PROPOSAL,
                move=move,
            )
            for position in graph_positions:
                neighbors = graph.neighbors(int(particles[position]))
                if len(neighbors):
                    proposal[position] = neighbors[
                        graph_rng.integers(0, len(neighbors))
                    ]

        log_acceptance = np.empty(len(particles), dtype=np.float64)
        for position, (current, candidate) in enumerate(
            zip(particles, proposal)
        ):
            if graph_move[position]:
                ratio = _safe_log_ratio(
                    float(log_gamma[candidate]), float(log_gamma[current])
                )
                ratio += float(
                    np.log(graph.degree[current])
                    - np.log(graph.degree[candidate])
                ) if graph.degree[current] and graph.degree[candidate] else 0.0
            else:
                ratio = _safe_log_ratio(
                    float(beta * log_weight[candidate]),
                    float(beta * log_weight[current]),
                )
            log_acceptance[position] = min(0.0, ratio)
        accept_rng = _keyed_rng(
            seed=seed,
            object_id=object_id,
            island_id=island_id,
            stage=stage,
            domain=_DOMAIN_MOVE_ACCEPT,
            move=move,
        )
        log_uniform = np.log(accept_rng.random(len(particles)))
        accept = log_uniform < log_acceptance
        particles[accept] = proposal[accept]
        accepted += int(np.count_nonzero(accept))
    return particles, accepted / attempted


def run_island_annealed_smc(
    log_target,
    q0,
    *,
    betas: Sequence[float],
    n_islands: int,
    particles_per_island: int,
    seed: int,
    object_id: int = 0,
    island_ids=None,
    graph: Optional[UndirectedGraph] = None,
    graph_move_probability: float = 0.0,
    moves_per_stage: int = 1,
    island_chunk: Optional[int] = None,
) -> FiniteStateIslandSMCResult:
    """Estimate a finite target normalizer with independent SMC islands.

    Resampling is multinomial at every fixed temperature.  At every
    *nonterminal* temperature it is followed by a fixed mixture of independence
    MH from ``q0`` and, optionally, a reversible degree-corrected move on
    ``graph``.  Mutating after the terminal weight cannot change a normalizer,
    so that pure-cost transition is deliberately omitted.  Random streams are
    keyed by the explicit object, island, stage, move, and domain, so outer
    island order and chunking do not alter any island.
    """

    target, base, log_q0 = _normalize_problem(log_target, q0)
    beta_values = _validate_betas(betas)
    islands = _positive_integer(n_islands, "n_islands")
    if islands < 2:
        raise ValueError("n_islands must be at least two to estimate variance")
    particles_count = _positive_integer(
        particles_per_island, "particles_per_island"
    )
    moves = _positive_integer(moves_per_stage, "moves_per_stage")
    seed = _exact_nonnegative_integer(seed, "seed", np.iinfo(np.uint64).max)
    object_id = _exact_nonnegative_integer(
        object_id, "object_id", np.iinfo(np.uint64).max
    )
    graph_probability = _validate_graph_probability(
        graph, graph_move_probability, len(target)
    )

    if island_ids is None:
        ids = np.arange(islands, dtype=np.int64)
    else:
        raw_ids = np.asarray(island_ids)
        if raw_ids.ndim != 1 or len(raw_ids) != islands or raw_ids.dtype.kind not in "iu":
            raise ValueError("island_ids must be a one-dimensional integer vector")
        if raw_ids.dtype.kind == "u" and np.any(
            raw_ids > np.iinfo(np.int64).max
        ):
            raise ValueError("island_ids must fit non-negative int64")
        ids = np.ascontiguousarray(raw_ids, dtype=np.int64)
        if (ids < 0).any() or len(np.unique(ids)) != islands:
            raise ValueError("island_ids must be unique non-negative integers")
    chunk = islands if island_chunk is None else _positive_integer(
        island_chunk, "island_chunk"
    )
    stages = len(beta_values) - 1
    if stages > np.iinfo(np.uint32).max or moves > np.iinfo(np.uint32).max:
        raise ValueError("stage and move counts must fit uint32 RNG keys")

    island_log_normalizer = np.zeros(islands, dtype=np.float64)
    stage_log_increment = np.zeros((islands, stages), dtype=np.float64)
    stage_ess = np.zeros((islands, stages), dtype=np.float64)
    stage_acceptance = np.zeros((islands, stages), dtype=np.float64)
    stage_unique_ancestors = np.zeros((islands, stages), dtype=np.int64)
    collapsed_stage = np.zeros(islands, dtype=np.int64)
    terminal_particles = np.empty(
        (islands, particles_count), dtype=np.int64
    )
    log_weight = target - log_q0

    for chunk_start in range(0, islands, chunk):
        chunk_stop = min(chunk_start + chunk, islands)
        for position in range(chunk_start, chunk_stop):
            island_id = int(ids[position])
            initial_rng = _keyed_rng(
                seed=seed,
                object_id=object_id,
                island_id=island_id,
                stage=0,
                domain=_DOMAIN_INITIAL,
            )
            particles = initial_rng.choice(
                len(base), size=particles_count, p=base
            ).astype(np.int64, copy=False)
            ancestors = np.arange(particles_count, dtype=np.int64)
            log_estimate = 0.0

            for stage in range(1, stages + 1):
                delta = beta_values[stage] - beta_values[stage - 1]
                log_incremental_weight = delta * log_weight[particles]
                log_total_weight = _logsumexp(log_incremental_weight)
                if np.isneginf(log_total_weight):
                    stage_log_increment[position, stage - 1] = -np.inf
                    collapsed_stage[position] = stage
                    log_estimate = -np.inf
                    break
                log_mean_weight = log_total_weight - np.log(particles_count)
                stage_log_increment[position, stage - 1] = log_mean_weight
                log_estimate += log_mean_weight
                normalized_weight = np.exp(
                    log_incremental_weight - log_total_weight,
                    dtype=np.float64,
                )
                stage_ess[position, stage - 1] = 1.0 / float(
                    np.square(normalized_weight).sum(dtype=np.float64)
                )
                resample_rng = _keyed_rng(
                    seed=seed,
                    object_id=object_id,
                    island_id=island_id,
                    stage=stage,
                    domain=_DOMAIN_RESAMPLE,
                )
                parent = resample_rng.choice(
                    particles_count,
                    size=particles_count,
                    replace=True,
                    p=normalized_weight,
                )
                particles = particles[parent]
                ancestors = ancestors[parent]
                stage_unique_ancestors[position, stage - 1] = len(
                    np.unique(ancestors)
                )
                if stage < stages:
                    particles, stage_acceptance[position, stage - 1] = (
                        _move_particles(
                            particles,
                            log_target=target,
                            q0=base,
                            log_q0=log_q0,
                            beta=float(beta_values[stage]),
                            graph=graph,
                            graph_probability=graph_probability,
                            seed=seed,
                            object_id=object_id,
                            island_id=island_id,
                            stage=stage,
                            moves_per_stage=moves,
                        )
                    )
            island_log_normalizer[position] = log_estimate
            terminal_particles[position] = particles

    (
        normalizer,
        log_normalizer,
        variance,
        log_variance,
        standard_error,
        island_normalizer,
    ) = _summarize_islands(ids, island_log_normalizer)
    return FiniteStateIslandSMCResult(
        normalizer=normalizer,
        log_normalizer=log_normalizer,
        variance=variance,
        log_variance=log_variance,
        standard_error=standard_error,
        island_ids=ids,
        island_normalizer=island_normalizer,
        island_log_normalizer=island_log_normalizer,
        stage_log_increment=stage_log_increment,
        stage_effective_sample_size=stage_ess,
        stage_move_acceptance_rate=stage_acceptance,
        stage_unique_ancestor_count=stage_unique_ancestors,
        collapsed_stage=collapsed_stage,
        terminal_particles=terminal_particles,
        betas=beta_values,
    )


__all__ = [
    "FiniteStateIslandSMCResult",
    "UndirectedGraph",
    "annealed_mh_transition_matrix",
    "run_island_annealed_smc",
]
