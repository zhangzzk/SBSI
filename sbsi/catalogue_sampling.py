"""Target-specific defensive importance sampling for catalogue-prior inference.

The local proposal is built in a low-dimensional measured-property space.  A
global catalogue-prior component guarantees support:

``q_i(j) = epsilon pi_j + (1-epsilon) q_local,i(j)``.

Every likelihood contribution carries the exact ``pi_j / q_i(j)`` correction.
The same per-object draws are reused at every finite-difference shear point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.special import logsumexp
import torch

from .catalogue_closure import MockCatalogue
from .catalogue_likelihood import CatalogueLikelihood, CatalogueScore


@dataclass(frozen=True)
class ProposalCoordinateTable:
    """Flow-predicted location of every prior atom in measured space."""

    values: np.ndarray
    target_names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    dispersion: Optional[np.ndarray] = None
    statistic: str = "median"
    dispersion_statistic: str = "robust_iqr"
    n_flow_samples: int = 0
    metadata: Optional[Mapping] = None

    def __post_init__(self):
        values = np.asarray(self.values, dtype=np.float64)
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        dispersion = (
            None
            if self.dispersion is None
            else np.asarray(self.dispersion, dtype=np.float64)
        )
        names = tuple(self.target_names)
        if values.ndim != 2 or values.shape[1] != len(names) or not len(values):
            raise ValueError("proposal coordinates must have shape (n_atoms, n_targets)")
        if center.shape != (len(names),) or scale.shape != (len(names),):
            raise ValueError("proposal center and scale do not match target dimension")
        if not np.isfinite(values).all() or not np.isfinite(center).all():
            raise ValueError("proposal coordinates and centers must be finite")
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("proposal scales must be finite and positive")
        if dispersion is not None:
            if dispersion.shape != values.shape:
                raise ValueError("proposal dispersion must align with coordinates")
            if not np.isfinite(dispersion).all() or (dispersion <= 0).any():
                raise ValueError("proposal dispersion must be finite and positive")
        if len(set(names)) != len(names):
            raise ValueError("proposal target names must be unique")
        if self.statistic not in {"mean", "median"}:
            raise ValueError("statistic must be 'mean' or 'median'")
        if self.dispersion_statistic not in {"robust_iqr", "std"}:
            raise ValueError(
                "dispersion_statistic must be 'robust_iqr' or 'std'"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "target_names", names)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "dispersion", dispersion)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def standardized(self) -> np.ndarray:
        return (self.values - self.center) / self.scale

    @classmethod
    def from_flow(
        cls,
        likelihood: CatalogueLikelihood,
        *,
        target_names: Sequence[str],
        n_flow_samples: int = 32,
        statistic: str = "median",
        dispersion_statistic: str = "robust_iqr",
        row_chunk: int = 8192,
        seed: int = 7101,
        dispersion_floor_percentile: float = 1.0,
        metadata: Optional[Mapping] = None,
    ) -> "ProposalCoordinateTable":
        """Estimate per-atom measured summaries using full flow draws.

        This intentionally does not read a flow's explicit mean head: that head
        need not equal the mean of the complete conditional density.  QMC flow
        draws are summarized instead and can be cached once.
        """

        names = tuple(target_names)
        if not names:
            raise ValueError("at least one proposal target is required")
        missing = sorted(set(names) - set(likelihood.target_names))
        if missing:
            raise KeyError(f"flow does not predict proposal targets: {missing}")
        if n_flow_samples <= 0 or row_chunk <= 0:
            raise ValueError("n_flow_samples and row_chunk must be positive")
        if not 0 <= dispersion_floor_percentile <= 100:
            raise ValueError("dispersion_floor_percentile must lie in [0, 100]")
        if statistic not in {"mean", "median"}:
            raise ValueError("statistic must be 'mean' or 'median'")
        if dispersion_statistic not in {"robust_iqr", "std"}:
            raise ValueError(
                "dispersion_statistic must be 'robust_iqr' or 'std'"
            )

        target_index = [likelihood.target_names.index(name) for name in names]
        view = likelihood.cache.get(0.0, 0.0)
        values = np.empty((len(view.flow), len(names)), dtype=np.float64)
        dispersion = np.empty_like(values)
        active_indices = np.flatnonzero(likelihood.cache.prior.weights > 0).astype(
            np.int64
        )
        if not len(active_indices):
            raise ValueError("catalogue prior has no positive-mass atoms")
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        reducer = np.median if statistic == "median" else np.mean
        for start in range(0, len(active_indices), row_chunk):
            stop = min(start + row_chunk, len(active_indices))
            rows = active_indices[start:stop]
            frame = view.flow.iloc[rows]
            try:
                draws = likelihood.flow_model.sample(
                    frame, n_samples=n_flow_samples, batch_size=row_chunk, qmc=True
                )
            except TypeError:
                draws = likelihood.flow_model.sample(frame, n_samples=n_flow_samples)
            selected = np.asarray(draws, dtype=float)[..., target_index]
            values[rows] = reducer(selected, axis=1)
            standard_deviation = np.std(selected, axis=1)
            if dispersion_statistic == "std":
                dispersion[rows] = standard_deviation
            else:
                q25, q75 = np.percentile(selected, [25, 75], axis=1)
                block_dispersion = (q75 - q25) / 1.3489795003921634
                dispersion[rows] = np.where(
                    np.isfinite(block_dispersion) & (block_dispersion > 0),
                    block_dispersion,
                    standard_deviation,
                )

        support_values = values[active_indices]
        center = np.median(support_values, axis=0)
        q25, q75 = np.percentile(support_values, [25, 75], axis=0)
        scale = (q75 - q25) / 1.3489795003921634
        fallback = np.std(support_values, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, fallback)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        support_dispersion = dispersion[active_indices]
        positive_dispersion = np.where(
            np.isfinite(support_dispersion) & (support_dispersion > 0),
            support_dispersion,
            np.nan,
        )
        dispersion_floor = np.nanpercentile(
            positive_dispersion, dispersion_floor_percentile, axis=0
        )
        dispersion_floor = np.where(
            np.isfinite(dispersion_floor) & (dispersion_floor > 0),
            dispersion_floor,
            np.maximum(1.0e-3 * scale, np.finfo(np.float64).eps),
        )
        dispersion[active_indices] = np.maximum(
            np.where(
                np.isfinite(support_dispersion) & (support_dispersion > 0),
                support_dispersion,
                dispersion_floor,
            ),
            dispersion_floor,
        )
        # Inactive rows are retained only for alignment with the full scene.
        # They never enter the active-only proposal tree.
        values[likelihood.cache.prior.weights <= 0] = center
        dispersion[likelihood.cache.prior.weights <= 0] = scale
        return cls(
            values,
            names,
            center,
            scale,
            dispersion=dispersion,
            statistic=statistic,
            dispersion_statistic=dispersion_statistic,
            n_flow_samples=int(n_flow_samples),
            metadata=metadata,
        )

    def save(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        arrays = {
            "values": self.values,
            "center": self.center,
            "scale": self.scale,
        }
        if self.dispersion is not None:
            arrays["dispersion"] = self.dispersion
        np.savez_compressed(root / "coordinates.npz", **arrays)
        manifest = {
            "version": 4 if self.dispersion is not None else 2,
            "target_names": list(self.target_names),
            "statistic": self.statistic,
            "dispersion_statistic": self.dispersion_statistic,
            "n_flow_samples": self.n_flow_samples,
            "metadata": dict(self.metadata or {}),
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ProposalCoordinateTable":
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("version") not in (2, 3, 4):
            raise ValueError(
                f"unsupported proposal-coordinate version {manifest.get('version')!r}"
            )
        arrays = np.load(root / "coordinates.npz")
        return cls(
            arrays["values"],
            tuple(manifest["target_names"]),
            arrays["center"],
            arrays["scale"],
            dispersion=(
                arrays["dispersion"]
                if manifest.get("version") in (3, 4)
                else None
            ),
            statistic=manifest["statistic"],
            dispersion_statistic=manifest.get(
                "dispersion_statistic", "robust_iqr"
            ),
            n_flow_samples=int(manifest["n_flow_samples"]),
            metadata=manifest.get("metadata"),
        )


@dataclass(frozen=True)
class ProposalDraw:
    """Per-object atoms and their full mixture probabilities."""

    indices: np.ndarray
    probability: np.ndarray
    local_member: np.ndarray
    global_component: np.ndarray
    candidate_radius: np.ndarray
    seed: int
    local_position: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.local_position is None:
            position = np.full(self.indices.shape, -1, dtype=np.int32)
        else:
            position = np.asarray(self.local_position, dtype=np.int32)
            if position.shape != self.indices.shape:
                raise ValueError("local positions must align with proposal draws")
            if np.any((position >= 0) != self.local_member):
                raise ValueError("local positions and membership flags disagree")
        object.__setattr__(self, "local_position", position)

    def prefix(self, n_draws: int) -> "ProposalDraw":
        if n_draws <= 0 or n_draws > self.indices.shape[1]:
            raise ValueError("prefix draw count is outside the stored ladder")
        return ProposalDraw(
            self.indices[:, :n_draws],
            self.probability[:, :n_draws],
            self.local_member[:, :n_draws],
            self.global_component[:, :n_draws],
            self.candidate_radius,
            self.seed,
            self.local_position[:, :n_draws],
        )

    def take(self, rows: np.ndarray) -> "ProposalDraw":
        """Select observed-object rows without changing their draw streams."""

        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or ((rows < 0) | (rows >= len(self.indices))).any():
            raise ValueError("proposal row selection is invalid")
        return ProposalDraw(
            self.indices[rows],
            self.probability[rows],
            self.local_member[rows],
            self.global_component[rows],
            self.candidate_radius[rows],
            self.seed,
            self.local_position[rows],
        )

    def coalesce(self) -> "CoalescedProposalDraw":
        """Collapse repeated atom ids while retaining prefix multiplicities."""

        unique_rows = []
        probability_rows = []
        inverse_rows = []
        local_position_rows = []
        first_rows = []
        for row in range(len(self.indices)):
            unique, first, inverse = np.unique(
                self.indices[row], return_index=True, return_inverse=True
            )
            unique_rows.append(unique)
            probability_rows.append(self.probability[row, first])
            inverse_rows.append(inverse.astype(np.int32, copy=False))
            local_position_rows.append(self.local_position[row, first])
            first_rows.append(first.astype(np.int32, copy=False))
        width = max(map(len, unique_rows))
        indices = np.empty((len(unique_rows), width), dtype=np.int64)
        probability = np.ones((len(unique_rows), width), dtype=np.float64)
        valid = np.zeros((len(unique_rows), width), dtype=bool)
        local_position = np.full((len(unique_rows), width), -1, dtype=np.int32)
        first_position = np.zeros((len(unique_rows), width), dtype=np.int32)
        for row, unique in enumerate(unique_rows):
            count = len(unique)
            indices[row, :count] = unique
            probability[row, :count] = probability_rows[row]
            valid[row, :count] = True
            local_position[row, :count] = local_position_rows[row]
            first_position[row, :count] = first_rows[row]
            if count < width:
                indices[row, count:] = unique[0]
                probability[row, count:] = probability_rows[row][0]
                first_position[row, count:] = first_rows[row][0]
        return CoalescedProposalDraw(
            indices=indices,
            probability=probability,
            valid=valid,
            inverse=np.stack(inverse_rows),
            local_position=local_position,
            first_position=first_position,
            source=self,
        )


@dataclass(frozen=True)
class CoalescedProposalDraw:
    """Unique sampled atoms plus exact nested-prefix accounting."""

    indices: np.ndarray
    probability: np.ndarray
    valid: np.ndarray
    inverse: np.ndarray
    local_position: np.ndarray
    first_position: np.ndarray
    source: ProposalDraw
    _count_cache: dict[int, np.ndarray] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _category_cache: dict[int, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def counts(self, n_draws: int) -> np.ndarray:
        if n_draws <= 0 or n_draws > self.inverse.shape[1]:
            raise ValueError("prefix draw count is outside the coalesced draw")
        if n_draws in self._count_cache:
            return self._count_cache[n_draws]
        counts = np.zeros(self.indices.shape, dtype=np.int32)
        for row in range(len(counts)):
            np.add.at(counts[row], self.inverse[row, :n_draws], 1)
        self._count_cache[n_draws] = counts
        return counts

    def category_counts(
        self, n_draws: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Counts of outside-local and global-component draws per unique atom."""

        if n_draws <= 0 or n_draws > self.inverse.shape[1]:
            raise ValueError("prefix draw count is outside the coalesced draw")
        if n_draws in self._category_cache:
            return self._category_cache[n_draws]
        outside = np.zeros(self.indices.shape, dtype=np.int32)
        global_component = np.zeros(self.indices.shape, dtype=np.int32)
        for row in range(len(outside)):
            inv = self.inverse[row, :n_draws]
            np.add.at(outside[row], inv, (~self.source.local_member[row, :n_draws]).astype(np.int32))
            np.add.at(
                global_component[row],
                inv,
                self.source.global_component[row, :n_draws].astype(np.int32),
            )
        result = (outside, global_component)
        self._category_cache[n_draws] = result
        return result


@dataclass(frozen=True)
class ProposalCandidates:
    """Deterministic nearest-neighbour support for an adapted proposal."""

    indices: np.ndarray
    distances: np.ndarray
    radius: np.ndarray

    def __post_init__(self):
        indices = np.asarray(self.indices, dtype=np.int64)
        distances = np.asarray(self.distances, dtype=np.float64)
        radius = np.asarray(self.radius, dtype=np.float64)
        if indices.ndim != 2 or distances.shape != indices.shape:
            raise ValueError("candidate indices and distances must be aligned matrices")
        if radius.shape != (len(indices),):
            raise ValueError("candidate radius must have one value per object")
        if not np.isfinite(distances).all() or not np.isfinite(radius).all():
            raise ValueError("candidate distances must be finite")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "radius", radius)

    def take(self, rows: np.ndarray) -> "ProposalCandidates":
        """Select observed-object rows while preserving candidate order."""

        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or ((rows < 0) | (rows >= len(self.indices))).any():
            raise ValueError("candidate row selection is invalid")
        return ProposalCandidates(
            self.indices[rows],
            self.distances[rows],
            self.radius[rows],
        )


@dataclass(frozen=True)
class CandidateSupportDiagnostics:
    """Posterior mass retained by nested and ideally reranked candidates.

    Every mass is normalized to the largest evaluated candidate set, not to
    the full catalogue.  The ``optimal`` columns answer whether a small
    support could work if it were ranked perfectly, separately from whether
    the current distance ranking finds that support.
    """

    prefix_sizes: tuple[int, ...]
    distance_prefix_mass: np.ndarray
    optimal_prefix_mass: np.ndarray
    reference_effective_atoms: np.ndarray
    reference_max_mass_fraction: np.ndarray

    def __post_init__(self):
        sizes = tuple(int(value) for value in self.prefix_sizes)
        distance = np.asarray(self.distance_prefix_mass, dtype=np.float64)
        optimal = np.asarray(self.optimal_prefix_mass, dtype=np.float64)
        effective = np.asarray(self.reference_effective_atoms, dtype=np.float64)
        maximum = np.asarray(self.reference_max_mass_fraction, dtype=np.float64)
        if not sizes or any(value <= 0 for value in sizes):
            raise ValueError("candidate prefix sizes must be positive")
        if distance.ndim != 2 or distance.shape != optimal.shape:
            raise ValueError("candidate mass arrays must be aligned matrices")
        if distance.shape[1] != len(sizes):
            raise ValueError("candidate mass columns must match prefix sizes")
        if effective.shape != (len(distance),) or maximum.shape != effective.shape:
            raise ValueError("reference diagnostics must contain one value per object")
        arrays = (distance, optimal, effective, maximum)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("candidate-support diagnostics must be finite")
        if (distance < 0).any() or (distance > 1 + 1e-12).any():
            raise ValueError("distance-prefix masses must lie in [0, 1]")
        if (optimal < 0).any() or (optimal > 1 + 1e-12).any():
            raise ValueError("optimal-prefix masses must lie in [0, 1]")
        object.__setattr__(self, "prefix_sizes", sizes)
        object.__setattr__(self, "distance_prefix_mass", distance)
        object.__setattr__(self, "optimal_prefix_mass", optimal)
        object.__setattr__(self, "reference_effective_atoms", effective)
        object.__setattr__(self, "reference_max_mass_fraction", maximum)


def candidate_support_diagnostics(
    log_local_target: np.ndarray,
    prefix_sizes: Sequence[int],
) -> CandidateSupportDiagnostics:
    """Compare current nested support with the best subset of the same size.

    ``log_local_target`` must follow the current candidate order and contain
    ``log(pi * Pdet * L)``.  The final column is the reference support.  The
    optimal mass uses the largest target values within that support and does
    not prescribe an implementable ranking; it is an upper bound for proposal
    design.
    """

    target = np.asarray(log_local_target, dtype=np.float64)
    sizes = tuple(sorted({int(value) for value in prefix_sizes}))
    if target.ndim != 2 or not len(target) or not target.shape[1]:
        raise ValueError("candidate target must be a non-empty matrix")
    if np.isnan(target).any() or np.isposinf(target).any():
        raise ValueError("candidate target may be finite or -inf only")
    if not sizes or sizes[0] <= 0 or sizes[-1] > target.shape[1]:
        raise ValueError("candidate prefix sizes fall outside the target width")
    normalizer = logsumexp(target, axis=1)
    if not np.isfinite(normalizer).all():
        raise ValueError("every object must have finite mass on the reference support")
    posterior = np.exp(target - normalizer[:, None])
    distance_mass = np.column_stack(
        [posterior[:, :size].sum(axis=1) for size in sizes]
    )
    # Sorting once is inexpensive for this bounded diagnostic and gives every
    # optimal prefix from the same ordering.
    ranked = np.sort(posterior, axis=1)[:, ::-1]
    cumulative = np.cumsum(ranked, axis=1)
    optimal_mass = np.column_stack([cumulative[:, size - 1] for size in sizes])
    positive = posterior > 0
    entropy = -np.sum(
        np.where(
            positive,
            posterior * np.log(np.where(positive, posterior, 1.0)),
            0.0,
        ),
        axis=1,
    )
    return CandidateSupportDiagnostics(
        prefix_sizes=sizes,
        distance_prefix_mass=distance_mass,
        optimal_prefix_mass=optimal_mass,
        reference_effective_atoms=np.exp(entropy),
        reference_max_mass_fraction=ranked[:, 0],
    )


class DefensiveLocalProposal:
    """Nearest-neighbour kernel proposal mixed with the full catalogue prior."""

    def __init__(
        self,
        coordinates: ProposalCoordinateTable,
        prior_weights: np.ndarray,
        *,
        local_base_weights: Optional[np.ndarray] = None,
    ):
        prior = np.asarray(prior_weights, dtype=np.float64)
        if prior.shape != (len(coordinates.values),) or (prior < 0).any() or prior.sum() <= 0:
            raise ValueError("prior weights must align with proposal coordinates")
        prior = prior / prior.sum()
        if local_base_weights is None:
            local = prior.copy()
        else:
            local = np.asarray(local_base_weights, dtype=np.float64)
            if local.shape != prior.shape or (local < 0).any() or not np.isfinite(local).all():
                raise ValueError("local base weights must be finite, non-negative, and aligned")
            local = prior * local
            if local.sum() <= 0:
                raise ValueError("local base weights vanish over the catalogue")
        self.coordinates = coordinates
        self.prior_weights = prior
        self.local_base_weights = local
        # Zero-mass rows may still be present because they supply the clustered
        # neighbour environment of positive prior atoms. They are not prior
        # support and must not consume candidate slots or be sampled.
        self.active_indices = np.flatnonzero(prior > 0).astype(np.int64)
        self.tree = KDTree(coordinates.standardized[self.active_indices])
        self._uncertainty_mips_tree = None
        self._uncertainty_mips_indices = None
        self._torch_standardized = {}
        self._torch_reranking_tables = {}
        self.global_cdf = np.cumsum(self.prior_weights)
        self.global_cdf[-1] = 1.0
        # A reusable atom-id -> local-candidate-position table avoids sorting
        # every 32k--131k candidate row for every observed object.
        self._candidate_lookup = np.full(len(prior), -1, dtype=np.int32)

    def _build_uncertainty_mips_tree(self) -> None:
        """Build an exact lifted-space index for the Gaussian proxy score."""

        if self._uncertainty_mips_tree is not None:
            return
        if self.coordinates.dispersion is None:
            raise ValueError("direct uncertainty ranking requires a version-3 proposal cache")
        support = self.active_indices[self.local_base_weights[self.active_indices] > 0]
        means = self.coordinates.values[support]
        dispersion = self.coordinates.dispersion[support]
        inverse_variance = np.reciprocal(np.square(dispersion))
        constant = (
            np.log(self.local_base_weights[support])
            - np.log(dispersion).sum(axis=1)
            - 0.5 * np.square(means / dispersion).sum(axis=1)
        )
        atoms = np.concatenate(
            (
                means * inverse_variance,
                -0.5 * inverse_variance,
                constant[:, None],
            ),
            axis=1,
        )
        norm_square = np.square(atoms).sum(axis=1)
        radius_square = float(np.max(norm_square)) * (1.0 + 8.0 * np.finfo(float).eps)
        lifted = np.column_stack(
            (atoms, np.sqrt(np.maximum(0.0, radius_square - norm_square)))
        )
        self._uncertainty_mips_tree = KDTree(lifted)
        self._uncertainty_mips_indices = support

    def uncertainty_candidates(
        self,
        observed,
        *,
        n_candidates: int,
    ) -> ProposalCandidates:
        """Query the heteroscedastic Gaussian proxy directly over all atoms.

        Expanding the proxy log density makes it an inner product between
        ``[x, x^2, 1]`` and atom-specific coefficients.  One extra coordinate
        converts maximum-inner-product search to exact Euclidean nearest
        neighbours, avoiding the broad location-only prefilter.
        """

        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        self._build_uncertainty_mips_tree()
        values = self._observed_values(observed)
        query = np.concatenate(
            (values, np.square(values), np.ones((len(values), 1))), axis=1
        )
        query = np.column_stack((query, np.zeros(len(query))))
        k = min(int(n_candidates), len(self._uncertainty_mips_indices))
        distances, positions = self._uncertainty_mips_tree.query(
            query, k=k, workers=-1
        )
        distances = np.asarray(distances).reshape(len(values), k)
        positions = np.asarray(positions, dtype=np.int64).reshape(len(values), k)
        indices = self._uncertainty_mips_indices[positions]
        return ProposalCandidates(indices, distances, distances[:, -1].copy())

    def _local_positions(
        self, candidates: np.ndarray, drawn: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return direct candidate positions and membership for sampled ids."""

        self._candidate_lookup[candidates] = np.arange(
            len(candidates), dtype=np.int32
        )
        positions = self._candidate_lookup[drawn].copy()
        self._candidate_lookup[candidates] = -1
        return positions, positions >= 0

    def _observed_values(self, observed) -> np.ndarray:
        if isinstance(observed, pd.DataFrame):
            missing = sorted(set(self.coordinates.target_names) - set(observed))
            if missing:
                raise KeyError(f"observations lack proposal targets: {missing}")
            values = observed.loc[:, self.coordinates.target_names].to_numpy(dtype=float)
        else:
            values = np.asarray(observed, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.coordinates.target_names):
            raise ValueError("observed proposal coordinates have the wrong shape")
        if not np.isfinite(values).all():
            raise ValueError("observed proposal coordinates must be finite")
        return values

    def _torch_uncertainty_rerank(
        self,
        values: np.ndarray,
        positions: torch.Tensor,
        *,
        n_candidates: int,
        device: torch.device,
    ) -> ProposalCandidates:
        """Rerank a location prefilter without transferring it to the host."""

        if self.coordinates.dispersion is None:
            raise ValueError(
                "uncertainty reranking requires a version-3 proposal cache"
            )
        key = str(device)
        tables = self._torch_reranking_tables.get(key)
        if tables is None:
            # Keep the established float64 Gaussian score.  For the 12.76M-
            # atom Infer V1 prior these reusable tables cost about 1 GiB,
            # materially less than each temporary location-distance slab.
            tables = (
                torch.as_tensor(
                    np.ascontiguousarray(
                        self.coordinates.values[self.active_indices]
                    ),
                    dtype=torch.float64,
                    device=device,
                ),
                torch.as_tensor(
                    np.ascontiguousarray(
                        self.coordinates.dispersion[self.active_indices]
                    ),
                    dtype=torch.float64,
                    device=device,
                ),
                torch.as_tensor(
                    np.ascontiguousarray(
                        self.local_base_weights[self.active_indices]
                    ),
                    dtype=torch.float64,
                    device=device,
                ).log(),
                torch.as_tensor(
                    np.ascontiguousarray(self.active_indices),
                    dtype=torch.long,
                    device=device,
                ),
            )
            self._torch_reranking_tables[key] = tables
        atom_values, atom_dispersion, log_base, active_indices = tables
        observed = torch.as_tensor(
            np.ascontiguousarray(values),
            dtype=torch.float64,
            device=device,
        )
        with torch.no_grad():
            selected_dispersion = atom_dispersion[positions]
            residual = (
                observed[:, None, :] - atom_values[positions]
            ) / selected_dispersion
            approximate_log_target = (
                log_base[positions]
                - torch.log(selected_dispersion).sum(dim=2)
                - 0.5 * torch.square(residual).sum(dim=2)
            )
            selected_score, selected = torch.topk(
                approximate_log_target,
                k=n_candidates,
                dim=1,
                largest=True,
                sorted=True,
            )
            selected_position = torch.gather(positions, 1, selected)
            indices = active_indices[selected_position]
            distances = selected_score[:, :1] - selected_score
            finite = torch.isfinite(distances)
            row_max = torch.max(
                torch.where(finite, distances, 0.0), dim=1, keepdim=True
            ).values
            distances = torch.where(finite, distances, row_max + 1.0)
        indices_np = indices.cpu().numpy().astype(np.int64)
        distances_np = distances.cpu().numpy().astype(np.float64)
        return ProposalCandidates(
            indices_np,
            distances_np,
            distances_np[:, -1].copy(),
        )

    def candidates(
        self,
        observed,
        *,
        n_candidates: int,
        prefilter_candidates: Optional[int] = None,
        torch_device: Optional[str | torch.device] = None,
    ) -> ProposalCandidates:
        """Return deterministic local support for posterior adaptation.

        By default this is the original nearest-location support.  When a
        larger ``prefilter_candidates`` is supplied, the cheap location query
        is reranked by a diagonal heteroscedastic approximation using cached
        per-atom flow dispersion and detected-prior mass.  Only the final
        ``n_candidates`` require exact flow likelihood evaluations.
        """

        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        values = self._observed_values(observed)
        standardized = (values - self.coordinates.center) / self.coordinates.scale
        k = min(int(n_candidates), len(self.active_indices))
        prefilter = (
            k
            if prefilter_candidates is None
            else min(int(prefilter_candidates), len(self.active_indices))
        )
        if prefilter < k:
            raise ValueError("prefilter_candidates must be at least n_candidates")
        if torch_device is None:
            distances, positions = self.tree.query(
                standardized, k=prefilter, workers=-1
            )
            distances = np.asarray(distances).reshape(len(values), prefilter)
            positions = np.asarray(positions, dtype=np.int64).reshape(
                len(values), prefilter
            )
        else:
            device = torch.device(torch_device)
            key = str(device)
            active = self._torch_standardized.get(key)
            if active is None:
                active = torch.as_tensor(
                    np.ascontiguousarray(
                        self.coordinates.standardized[self.active_indices]
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                self._torch_standardized[key] = active
            query = torch.as_tensor(
                np.ascontiguousarray(standardized),
                dtype=active.dtype,
                device=device,
            )
            with torch.no_grad():
                matrix = torch.cdist(query, active)
                distance_tensor, position_tensor = torch.topk(
                    matrix,
                    k=prefilter,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
            del matrix, query
            if prefilter > k:
                del distance_tensor
                return self._torch_uncertainty_rerank(
                    values,
                    position_tensor,
                    n_candidates=k,
                    device=device,
                )
            distances = distance_tensor.cpu().numpy().astype(np.float64)
            positions = position_tensor.cpu().numpy().astype(np.int64)
            del distance_tensor, position_tensor
        indices = self.active_indices[positions]
        if prefilter > k:
            if self.coordinates.dispersion is None:
                raise ValueError(
                    "uncertainty reranking requires a version-3 proposal cache"
                )
            atom_dispersion = self.coordinates.dispersion[indices]
            residual = (values[:, None, :] - self.coordinates.values[indices]) / atom_dispersion
            base = self.local_base_weights[indices]
            approximate_log_target = np.full(base.shape, -np.inf, dtype=np.float64)
            positive = base > 0
            gaussian = (
                -np.log(atom_dispersion).sum(axis=2)
                - 0.5 * np.square(residual).sum(axis=2)
            )
            approximate_log_target[positive] = (
                np.log(base[positive]) + gaussian[positive]
            )
            selected = np.argpartition(
                approximate_log_target, kth=prefilter - k, axis=1
            )[:, -k:]
            selected_score = np.take_along_axis(
                approximate_log_target, selected, axis=1
            )
            order = np.argsort(-selected_score, axis=1)
            selected = np.take_along_axis(selected, order, axis=1)
            selected_score = np.take_along_axis(selected_score, order, axis=1)
            indices = np.take_along_axis(indices, selected, axis=1)
            # These values are only diagnostics for the posterior-adapted path.
            # Shift the approximate negative log target so the best is zero.
            distances = selected_score[:, :1] - selected_score
            finite = np.isfinite(distances)
            row_max = np.max(np.where(finite, distances, 0.0), axis=1, keepdims=True)
            distances = np.where(finite, distances, row_max + 1.0)
        else:
            indices = indices[:, :k]
            distances = distances[:, :k]
        return ProposalCandidates(
            indices,
            distances,
            distances[:, -1].copy(),
        )

    def draw_adapted(
        self,
        candidates: ProposalCandidates,
        log_local_target: np.ndarray,
        *,
        n_draws: int,
        epsilon: float,
        seed: int,
        object_offset: int = 0,
        object_ids: Optional[np.ndarray] = None,
    ) -> ProposalDraw:
        """Draw from a zero-shear posterior-adapted defensive proposal.

        ``log_local_target`` is the exact unnormalised detected likelihood
        ``log(pi * Pdet * L0)`` on the deterministic candidate support.  The
        resulting proposal is

        ``q_i(j) = epsilon*pi_j + (1-epsilon)*q_local,i(j)``.

        The full mixture probability is returned for every draw, so this
        adaptation changes variance but not the marginalized likelihood.
        """

        target = np.asarray(log_local_target, dtype=np.float64)
        if target.shape != candidates.indices.shape:
            raise ValueError("local target must align with proposal candidates")
        if np.isnan(target).any() or np.isposinf(target).any():
            raise ValueError("local target may be finite or -inf, but not NaN/+inf")
        if n_draws <= 0:
            raise ValueError("n_draws must be positive")
        if not (0 < epsilon <= 1):
            raise ValueError("epsilon must satisfy 0 < epsilon <= 1")
        if object_offset < 0:
            raise ValueError("object_offset must be non-negative")

        n_objects, k = candidates.indices.shape
        if object_ids is None:
            absolute_ids = object_offset + np.arange(n_objects, dtype=np.int64)
        else:
            absolute_ids = np.asarray(object_ids, dtype=np.int64)
            if object_offset != 0:
                raise ValueError("object_offset and explicit object_ids cannot be combined")
            if absolute_ids.shape != (n_objects,) or (absolute_ids < 0).any():
                raise ValueError("object_ids must contain one non-negative id per row")
        indices = np.empty((n_objects, n_draws), dtype=np.int64)
        probability = np.empty((n_objects, n_draws), dtype=np.float64)
        local_member = np.zeros((n_objects, n_draws), dtype=bool)
        local_position = np.full((n_objects, n_draws), -1, dtype=np.int32)
        global_component = np.empty((n_objects, n_draws), dtype=bool)

        for row in range(n_objects):
            local_log = target[row]
            normalizer = logsumexp(local_log)
            if np.isfinite(normalizer):
                local_probability = np.exp(local_log - normalizer)
            else:
                # This can only occur for an observation outside numerical
                # flow support. Retain a valid defensive proposal so the
                # subsequent likelihood reports the problem without bias.
                local_probability = self.local_base_weights[candidates.indices[row]]
                total = float(local_probability.sum())
                if total <= 0:
                    local_probability = np.ones(k, dtype=np.float64)
                    total = float(k)
                local_probability = local_probability / total

            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(absolute_ids[row])])
            )
            uniforms = rng.random((n_draws, 3))
            is_global = uniforms[:, 0] < epsilon
            global_component[row] = is_global
            if is_global.any():
                indices[row, is_global] = np.searchsorted(
                    self.global_cdf, uniforms[is_global, 1], side="right"
                )
            if (~is_global).any():
                local_cdf = np.cumsum(local_probability)
                local_cdf[-1] = 1.0
                positions = np.searchsorted(
                    local_cdf, uniforms[~is_global, 2], side="right"
                )
                indices[row, ~is_global] = candidates.indices[row, positions]

            position, member = self._local_positions(
                candidates.indices[row], indices[row]
            )
            local_position[row] = position
            local_member[row] = member
            local_at_draw = np.zeros(n_draws, dtype=np.float64)
            local_at_draw[member] = local_probability[position[member]]
            probability[row] = (
                epsilon * self.prior_weights[indices[row]]
                + (1.0 - epsilon) * local_at_draw
            )

        return ProposalDraw(
            indices,
            probability,
            local_member,
            global_component,
            candidates.radius,
            int(seed),
            local_position,
        )

    def draw_global(
        self,
        candidates: ProposalCandidates,
        *,
        n_draws: int,
        seed: int,
        object_offset: int = 0,
    ) -> ProposalDraw:
        """Draw catalogue-prior atoms and mark candidate-set membership.

        This supports stratified evidence estimation: candidate atoms are
        summed exactly, while prior draws contribute only when they fall in
        the complement.  Per-object streams retain exact nested prefixes.
        """

        if n_draws <= 0:
            raise ValueError("n_draws must be positive")
        if object_offset < 0:
            raise ValueError("object_offset must be non-negative")
        n_objects = len(candidates.indices)
        indices = np.empty((n_objects, n_draws), dtype=np.int64)
        probability = np.empty((n_objects, n_draws), dtype=np.float64)
        local_member = np.empty((n_objects, n_draws), dtype=bool)
        local_position = np.empty((n_objects, n_draws), dtype=np.int32)
        for row in range(n_objects):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(object_offset) + row])
            )
            indices[row] = np.searchsorted(
                self.global_cdf, rng.random(n_draws), side="right"
            )
            probability[row] = self.prior_weights[indices[row]]
            position, member = self._local_positions(
                candidates.indices[row], indices[row]
            )
            local_position[row] = position
            local_member[row] = member
        return ProposalDraw(
            indices=indices,
            probability=probability,
            local_member=local_member,
            global_component=np.ones_like(local_member),
            candidate_radius=candidates.radius,
            seed=int(seed),
            local_position=local_position,
        )

    def draw(
        self,
        observed,
        *,
        n_draws: int,
        n_candidates: int,
        epsilon: float,
        bandwidth: float,
        seed: int,
        object_offset: int = 0,
    ) -> ProposalDraw:
        if n_draws <= 0 or n_candidates <= 0:
            raise ValueError("n_draws and n_candidates must be positive")
        if not (0 < epsilon <= 1):
            raise ValueError("epsilon must satisfy 0 < epsilon <= 1")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        candidate_set = self.candidates(observed, n_candidates=n_candidates)
        candidates = candidate_set.indices
        distances = candidate_set.distances
        values = self._observed_values(observed)
        k = candidates.shape[1]
        indices = np.empty((len(values), n_draws), dtype=np.int64)
        probability = np.empty((len(values), n_draws), dtype=np.float64)
        local_member = np.zeros((len(values), n_draws), dtype=bool)
        local_position = np.full((len(values), n_draws), -1, dtype=np.int32)
        global_component = np.empty((len(values), n_draws), dtype=bool)
        radius = candidate_set.radius
        if object_offset < 0:
            raise ValueError("object_offset must be non-negative")
        for row in range(len(values)):
            candidate = candidates[row]
            base = self.local_base_weights[candidate]
            log_local = np.full(k, -np.inf, dtype=np.float64)
            positive = base > 0
            log_local[positive] = (
                np.log(base[positive])
                - 0.5 * (distances[row, positive] / bandwidth) ** 2
            )
            peak = np.max(log_local)
            if not np.isfinite(peak):
                local_probability = self.prior_weights[candidate]
            else:
                local_probability = np.exp(log_local - peak)
            if local_probability.sum() <= 0:
                local_probability = np.ones(k, dtype=np.float64)
            local_probability = local_probability / local_probability.sum()

            # Each object/draw consumes one fixed-width random record.  Separate
            # object streams prevent the requested maximum draw count from
            # shifting later objects' streams; fixed-width records make every
            # field of draw(n) an exact prefix of draw(m) for n < m.
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(object_offset) + row])
            )
            uniforms = rng.random((n_draws, 3))
            is_global = uniforms[:, 0] < epsilon
            global_component[row] = is_global
            if is_global.any():
                indices[row, is_global] = np.searchsorted(
                    self.global_cdf, uniforms[is_global, 1], side="right"
                )
            if (~is_global).any():
                local_cdf = np.cumsum(local_probability)
                local_cdf[-1] = 1.0
                sampled_local_position = np.searchsorted(
                    local_cdf, uniforms[~is_global, 2], side="right"
                )
                indices[row, ~is_global] = candidate[sampled_local_position]

            position, member = self._local_positions(candidate, indices[row])
            local_position[row] = position
            local_member[row] = member
            local_at_draw = np.zeros(n_draws, dtype=np.float64)
            local_at_draw[member] = local_probability[position[member]]
            probability[row] = (
                epsilon * self.prior_weights[indices[row]]
                + (1.0 - epsilon) * local_at_draw
            )

        return ProposalDraw(
            indices,
            probability,
            local_member,
            global_component,
            radius,
            int(seed),
            local_position,
        )


@dataclass(frozen=True)
class ImportanceDiagnostics:
    mean_ess: float
    median_ess: float
    p10_ess: float
    mean_ess_fraction: float
    p90_max_weight_fraction: float
    mean_outside_local_contribution: float
    mean_global_draw_contribution: float


def importance_diagnostics(log_weights: np.ndarray, draw: ProposalDraw) -> ImportanceDiagnostics:
    log_weights = np.asarray(log_weights, dtype=np.float64)
    if log_weights.shape != draw.indices.shape:
        raise ValueError("diagnostic weights do not align with proposal draws")
    peak = np.max(log_weights, axis=1, keepdims=True)
    finite = np.isfinite(peak[:, 0])
    normalized = np.zeros_like(log_weights)
    normalized[finite] = np.exp(log_weights[finite] - peak[finite])
    total = normalized.sum(axis=1, keepdims=True)
    good = finite & (total[:, 0] > 0)
    normalized[good] /= total[good]
    ess = np.zeros(len(log_weights), dtype=np.float64)
    ess[good] = 1.0 / np.sum(normalized[good] ** 2, axis=1)
    max_fraction = normalized.max(axis=1)
    outside = np.sum(normalized * (~draw.local_member), axis=1)
    global_draw = np.sum(normalized * draw.global_component, axis=1)
    return ImportanceDiagnostics(
        mean_ess=float(np.mean(ess)),
        median_ess=float(np.median(ess)),
        p10_ess=float(np.percentile(ess, 10)),
        mean_ess_fraction=float(np.mean(ess) / log_weights.shape[1]),
        p90_max_weight_fraction=float(np.percentile(max_fraction, 90)),
        mean_outside_local_contribution=float(np.mean(outside)),
        mean_global_draw_contribution=float(np.mean(global_draw)),
    )


def select_adaptive_draw_counts(
    log_weights: np.ndarray,
    ladder: Sequence[int],
    *,
    min_ess: float,
    max_weight_fraction: float,
) -> np.ndarray:
    """Choose the first acceptable nested prefix for every object.

    This is the production draw-doubling policy.  It is evaluated from the
    zero-shear weights, then the chosen prefix is held fixed across all five
    Section 5 stencil views.  Validation continues to report every fixed rung.
    Rows that never meet both criteria receive the largest available prefix.
    """

    log_weights = np.asarray(log_weights, dtype=np.float64)
    ladder = tuple(sorted({int(value) for value in ladder}))
    if log_weights.ndim != 2 or not ladder or ladder[0] <= 0:
        raise ValueError("log_weights must be a matrix and ladder must be positive")
    if ladder[-1] > log_weights.shape[1]:
        raise ValueError("draw ladder exceeds the available weight prefixes")
    if min_ess <= 0 or not (0 < max_weight_fraction <= 1):
        raise ValueError("adaptive ESS and maximum-weight thresholds are invalid")
    selected = np.full(len(log_weights), ladder[-1], dtype=np.int64)
    unresolved = np.ones(len(log_weights), dtype=bool)
    for n_draws in ladder:
        prefix = log_weights[:, :n_draws]
        peak = np.max(prefix, axis=1, keepdims=True)
        finite = np.isfinite(peak[:, 0])
        normalized = np.zeros_like(prefix)
        normalized[finite] = np.exp(prefix[finite] - peak[finite])
        total = normalized.sum(axis=1, keepdims=True)
        good = finite & (total[:, 0] > 0)
        normalized[good] /= total[good]
        ess = np.zeros(len(prefix), dtype=np.float64)
        ess[good] = 1.0 / np.square(normalized[good]).sum(axis=1)
        maximum = normalized.max(axis=1)
        passed = unresolved & (ess >= min_ess) & (maximum <= max_weight_fraction)
        selected[passed] = n_draws
        unresolved[passed] = False
    return selected


def select_independent_pilot_draw_counts(
    pilot_log_weights: np.ndarray,
    ladder: Sequence[int],
    *,
    min_ess: float,
    max_weight_fraction: float,
    safety_factor: float = 1.0,
) -> np.ndarray:
    """Allocate a fixed production draw count from an independent pilot.

    The pilot must use a random stream independent of the later production
    importance draw.  Its effective-sample-size fraction is projected to each
    production rung.  The observed maximum normalized pilot weight is diluted
    in proportion to the larger production sample, with ``safety_factor``
    providing a conservative margin for both diagnostics.  Because the chosen
    rung is fixed before any production weight is inspected, this avoids the
    optional-stopping coupling of :func:`select_adaptive_draw_counts`.
    """

    weights = np.asarray(pilot_log_weights, dtype=np.float64)
    ladder = tuple(sorted({int(value) for value in ladder}))
    if weights.ndim != 2 or weights.shape[1] <= 0:
        raise ValueError("pilot_log_weights must be a non-empty matrix")
    if not ladder or ladder[0] <= 0:
        raise ValueError("draw ladder must be positive")
    if min_ess <= 0 or not (0 < max_weight_fraction <= 1):
        raise ValueError("adaptive ESS and maximum-weight thresholds are invalid")
    if not np.isfinite(safety_factor) or safety_factor < 1.0:
        raise ValueError("pilot safety factor must be finite and at least one")

    peak = np.max(weights, axis=1, keepdims=True)
    finite = np.isfinite(peak[:, 0])
    normalized = np.zeros_like(weights)
    normalized[finite] = np.exp(weights[finite] - peak[finite])
    total = normalized.sum(axis=1, keepdims=True)
    good = finite & (total[:, 0] > 0)
    normalized[good] /= total[good]
    ess = np.zeros(len(weights), dtype=np.float64)
    ess[good] = 1.0 / np.square(normalized[good]).sum(axis=1)
    maximum = normalized.max(axis=1)
    pilot_draws = weights.shape[1]
    ess_fraction = ess / pilot_draws

    selected = np.full(len(weights), ladder[-1], dtype=np.int64)
    unresolved = np.ones(len(weights), dtype=bool)
    for n_draws in ladder:
        projected_ess = n_draws * ess_fraction / safety_factor
        projected_maximum = np.minimum(
            1.0,
            maximum * pilot_draws * safety_factor / n_draws,
        )
        passed = (
            unresolved
            & good
            & (projected_ess >= min_ess)
            & (projected_maximum <= max_weight_fraction)
        )
        selected[passed] = n_draws
        unresolved[passed] = False
    return selected


@dataclass(frozen=True)
class ImportanceRung:
    n_draws: int
    evaluation_shear: float
    estimated_shear: float
    fisher_estimated_shear: float
    score_sum: float
    information_sum: float
    score_square_sum: float
    score_z: float
    robust_standard_error: float
    model_standard_error: float
    closure_pull: float
    score_sum_error_exact: Optional[float]
    information_sum_error_exact: Optional[float]
    shear_error_exact: Optional[float]
    diagnostics: ImportanceDiagnostics


@dataclass(frozen=True)
class ImportanceLadderResult:
    injected_shear: float
    exact_estimated_shear: Optional[float]
    proposal_seed: int
    epsilon: float
    n_candidates: int
    bandwidth: float
    target_names: tuple[str, ...]
    rungs: tuple[ImportanceRung, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["target_names"] = list(self.target_names)
        return payload


@dataclass(frozen=True)
class ImportanceProfilePoint:
    shear: float
    log_likelihood_sum: float


@dataclass(frozen=True)
class ImportanceProfileRung:
    n_draws: int
    points: tuple[ImportanceProfilePoint, ...]
    grid_estimated_shear: float
    estimated_shear: float
    quadratic_information: Optional[float]


@dataclass(frozen=True)
class ImportanceProfileResult:
    """One fixed-draw likelihood profile over trial directional shears."""

    injected_shear: float
    proposal_seed: int
    epsilon: float
    n_candidates: int
    bandwidth: float
    target_names: tuple[str, ...]
    rungs: tuple[ImportanceProfileRung, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["target_names"] = list(self.target_names)
        return payload


@dataclass(frozen=True)
class ImportanceCurvatureRung:
    """Per-object curvature decomposition at several expansion centres."""

    n_draws: int
    centers: tuple[float, ...]
    marginalized_score: np.ndarray
    marginalized_information: np.ndarray
    posterior_conditional_information: np.ndarray
    posterior_score_variance: np.ndarray
    decomposition_residual: np.ndarray
    true_atom_score: Optional[np.ndarray]
    true_atom_information: Optional[np.ndarray]


@dataclass(frozen=True)
class ImportanceCurvatureScanResult:
    """Small retained profile plus streamed posterior curvature terms."""

    injected_shear: float
    proposal_seed: int
    epsilon: float
    n_candidates: int
    bandwidth: float
    target_names: tuple[str, ...]
    h: float
    shears: tuple[float, ...]
    marginalized_log_likelihood: Mapping[int, np.ndarray]
    true_atom_log_likelihood: Optional[np.ndarray]
    rungs: tuple[ImportanceCurvatureRung, ...]


@dataclass(frozen=True)
class ImportanceConvergenceAssessment:
    """Explicit multi-axis stopping decision for an importance study."""

    passed: bool
    checks: Mapping[str, bool]
    metrics: Mapping[str, float]
    failures: tuple[str, ...]


def assess_importance_convergence(
    results: Sequence[ImportanceLadderResult],
    *,
    max_exact_shear_error: float,
    max_rung_change: float,
    max_seed_spread: float,
    max_candidate_change: float,
    min_ess_fraction: float,
    max_p90_weight_fraction: float,
) -> ImportanceConvergenceAssessment:
    """Apply predeclared stopping gates across M, K, seed, and the oracle.

    At least two draw rungs, two independent seeds per candidate count, and two
    candidate counts are required.  ESS is only one gate; it cannot override a
    failure of score stability or exact-oracle agreement.
    """

    results = tuple(results)
    if not results:
        raise ValueError("at least one importance result is required")
    thresholds = (
        max_exact_shear_error,
        max_rung_change,
        max_seed_spread,
        max_candidate_change,
        min_ess_fraction,
        max_p90_weight_fraction,
    )
    if not np.isfinite(thresholds).all() or any(value < 0 for value in thresholds):
        raise ValueError("convergence thresholds must be finite and non-negative")

    top = [result.rungs[-1] for result in results]
    exact_available = all(rung.shear_error_exact is not None for rung in top)
    exact_error = (
        max(abs(rung.shear_error_exact) for rung in top)
        if exact_available
        else float("inf")
    )
    enough_rungs = all(len(result.rungs) >= 2 for result in results)
    rung_change = (
        max(
            abs(result.rungs[-1].estimated_shear - result.rungs[-2].estimated_shear)
            for result in results
        )
        if enough_rungs
        else float("inf")
    )

    by_candidates: dict[int, list[ImportanceLadderResult]] = {}
    for result in results:
        by_candidates.setdefault(result.n_candidates, []).append(result)
    enough_seeds = all(len({arm.proposal_seed for arm in arms}) >= 2 for arms in by_candidates.values())
    seed_spread = (
        max(
            np.ptp([arm.rungs[-1].estimated_shear for arm in arms])
            for arms in by_candidates.values()
        )
        if enough_seeds
        else float("inf")
    )
    candidate_counts = sorted(by_candidates)
    enough_candidates = len(candidate_counts) >= 2
    candidate_means = {
        count: float(np.mean([arm.rungs[-1].estimated_shear for arm in by_candidates[count]]))
        for count in candidate_counts
    }
    candidate_change = (
        max(
            abs(candidate_means[right] - candidate_means[left])
            for left, right in zip(candidate_counts[:-1], candidate_counts[1:])
        )
        if enough_candidates
        else float("inf")
    )
    ess_fraction = min(rung.diagnostics.mean_ess_fraction for rung in top)
    max_weight = max(rung.diagnostics.p90_max_weight_fraction for rung in top)

    checks = {
        "exact_oracle": bool(
            exact_available and exact_error <= max_exact_shear_error
        ),
        "draw_ladder": bool(enough_rungs and rung_change <= max_rung_change),
        "independent_seeds": bool(enough_seeds and seed_spread <= max_seed_spread),
        "candidate_expansion": bool(
            enough_candidates and candidate_change <= max_candidate_change
        ),
        "ess_fraction": bool(ess_fraction >= min_ess_fraction),
        "weight_concentration": bool(max_weight <= max_p90_weight_fraction),
    }
    metrics = {
        "max_exact_shear_error": float(exact_error),
        "max_last_rung_change": float(rung_change),
        "max_seed_spread": float(seed_spread),
        "max_candidate_change": float(candidate_change),
        "min_mean_ess_fraction": float(ess_fraction),
        "max_p90_weight_fraction": float(max_weight),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return ImportanceConvergenceAssessment(not failures, checks, metrics, failures)


def run_importance_ladder(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    ladder: Sequence[int],
    n_candidates: int,
    epsilon: float,
    bandwidth: float,
    proposal_seed: int,
    center: Sequence[float] = (0.0, 0.0),
    direction: Sequence[float] = (1.0, 0.0),
    delta: float = 0.01,
    richardson: bool = True,
    compare_exact: bool = True,
    reference_score: Optional[CatalogueScore] = None,
    object_chunk: int = 64,
    atom_chunk: int = 4096,
) -> ImportanceLadderResult:
    """Run a nested draw ladder, holding every prefix fixed across shear."""

    ladder = tuple(sorted({int(value) for value in ladder}))
    if not ladder or ladder[0] <= 0:
        raise ValueError("ladder must contain positive draw counts")
    center = np.asarray(center, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center must be a finite two-vector")
    norm = float(np.linalg.norm(direction))
    if direction.shape != (2,) or norm == 0:
        raise ValueError("direction must be a nonzero two-vector")
    direction = direction / norm
    evaluation_shear = float(center @ direction)
    injected_vector = mock.truth[["injected_g1", "injected_g2"]].iloc[0].to_numpy(float)
    injected = float(injected_vector @ direction)

    draw = proposal.draw(
        mock.measurements,
        n_draws=ladder[-1],
        n_candidates=n_candidates,
        epsilon=epsilon,
        bandwidth=bandwidth,
        seed=proposal_seed,
    )
    log_weights = likelihood.log_importance_weights(
        mock.measurements,
        0.0,
        0.0,
        atom_indices=draw.indices,
        proposal_probability=draw.probability,
        object_chunk=object_chunk,
    )

    exact_score = None
    exact_estimate = None
    if reference_score is not None and not compare_exact:
        raise ValueError("reference_score requires compare_exact=True")
    if compare_exact:
        exact_score = reference_score
        if exact_score is None:
            exact_score = likelihood.score_and_information(
                mock.measurements,
                center=center,
                direction=direction,
                delta=delta,
                richardson=richardson,
                object_chunk=object_chunk,
                atom_chunk=atom_chunk,
            )
        exact_estimate = evaluation_shear + float(
            exact_score.score.sum() / exact_score.information.sum()
        )

    rungs = []
    for n_draws in ladder:
        prefix = draw.prefix(n_draws)
        score = likelihood.score_and_information(
            mock.measurements,
            center=center,
            direction=direction,
            delta=delta,
            richardson=richardson,
            atom_indices=prefix.indices,
            proposal_probability=prefix.probability,
            object_chunk=object_chunk,
            atom_chunk=atom_chunk,
        )
        score_sum = float(score.score.sum())
        information_sum = float(score.information.sum())
        score_square_sum = float(np.square(score.score).sum())
        estimate = evaluation_shear + score_sum / information_sum
        fisher_estimate = evaluation_shear + score_sum / score_square_sum
        score_z = score_sum / np.sqrt(score_square_sum)
        n = len(score.score)
        if n < 2:
            robust_standard_error = float("nan")
        else:
            offset = estimate - evaluation_shear
            residual = np.asarray(score.score, dtype=float) - offset * np.asarray(
                score.information, dtype=float
            )
            robust_standard_error = float(
                np.sqrt(
                    n
                    / (n - 1)
                    * float(np.square(residual).sum())
                    / information_sum**2
                )
            )
        model_standard_error = (
            float(1.0 / np.sqrt(information_sum))
            if information_sum > 0
            else float("nan")
        )
        closure_pull = (
            float((estimate - injected) / robust_standard_error)
            if np.isfinite(robust_standard_error) and robust_standard_error > 0
            else float("nan")
        )
        diagnostics = importance_diagnostics(log_weights[:, :n_draws], prefix)
        rungs.append(
            ImportanceRung(
                n_draws=n_draws,
                evaluation_shear=evaluation_shear,
                estimated_shear=float(estimate),
                fisher_estimated_shear=float(fisher_estimate),
                score_sum=score_sum,
                information_sum=information_sum,
                score_square_sum=score_square_sum,
                score_z=float(score_z),
                robust_standard_error=robust_standard_error,
                model_standard_error=model_standard_error,
                closure_pull=closure_pull,
                score_sum_error_exact=None if exact_score is None else score_sum - float(exact_score.score.sum()),
                information_sum_error_exact=None if exact_score is None else information_sum - float(exact_score.information.sum()),
                shear_error_exact=None if exact_estimate is None else float(estimate - exact_estimate),
                diagnostics=diagnostics,
            )
        )
    return ImportanceLadderResult(
        injected_shear=injected,
        exact_estimated_shear=exact_estimate,
        proposal_seed=int(proposal_seed),
        epsilon=float(epsilon),
        n_candidates=int(n_candidates),
        bandwidth=float(bandwidth),
        target_names=proposal.coordinates.target_names,
        rungs=tuple(rungs),
    )


def _summarize_profile_rung(
    n_draws: int, points: Sequence[ImportanceProfilePoint]
) -> ImportanceProfileRung:
    points = tuple(sorted(points, key=lambda point: point.shear))
    shear = np.asarray([point.shear for point in points], dtype=float)
    log_likelihood = np.asarray(
        [point.log_likelihood_sum for point in points], dtype=float
    )
    best = int(np.argmax(log_likelihood))
    grid_estimate = float(shear[best])
    estimate = grid_estimate
    information = None
    if 0 < best < len(points) - 1:
        local_shear = shear[best - 1 : best + 2]
        local_log_likelihood = log_likelihood[best - 1 : best + 2]
        coefficient = np.polyfit(
            local_shear,
            local_log_likelihood - local_log_likelihood[1],
            deg=2,
        )
        if coefficient[0] < 0:
            candidate = float(-coefficient[1] / (2.0 * coefficient[0]))
            if local_shear[0] <= candidate <= local_shear[-1]:
                estimate = candidate
                information = float(-2.0 * coefficient[0])
    return ImportanceProfileRung(
        n_draws=int(n_draws),
        points=points,
        grid_estimated_shear=grid_estimate,
        estimated_shear=estimate,
        quadratic_information=information,
    )


def run_importance_profiles(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    shears: Sequence[float],
    ladder: Sequence[int],
    n_candidates: int,
    epsilon: float,
    bandwidth: float,
    proposal_seeds: Sequence[int],
    direction: Sequence[float] = (1.0, 0.0),
    object_chunk: int = 64,
) -> tuple[ImportanceProfileResult, ...]:
    """Evaluate a bounded-memory directional likelihood profile.

    Proposal draws are held fixed across every shear point and rung.  All
    proposal seeds are evaluated while a model view is resident, after which
    dynamically created views are released.  This avoids retaining a large
    catalogue table for every point in a shear scan.
    """

    shears = tuple(sorted({float(value) for value in shears}))
    ladder = tuple(sorted({int(value) for value in ladder}))
    seeds = tuple(dict.fromkeys(int(value) for value in proposal_seeds))
    if not shears or not np.isfinite(shears).all():
        raise ValueError("profile shears must be non-empty and finite")
    if not ladder or ladder[0] <= 0:
        raise ValueError("ladder must contain positive draw counts")
    if not seeds:
        raise ValueError("at least one proposal seed is required")
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (2,) or not np.isfinite(direction).all():
        raise ValueError("direction must be a finite two-vector")
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        raise ValueError("direction must be nonzero")
    direction = direction / norm
    injected_vector = mock.truth[["injected_g1", "injected_g2"]].iloc[0].to_numpy(float)
    injected = float(injected_vector @ direction)

    draws = {
        seed: proposal.draw(
            mock.measurements,
            n_draws=ladder[-1],
            n_candidates=n_candidates,
            epsilon=epsilon,
            bandwidth=bandwidth,
            seed=seed,
        )
        for seed in seeds
    }
    points = {
        seed: {n_draws: [] for n_draws in ladder}
        for seed in seeds
    }
    protected = set(likelihood.cache.available_shears)
    for shear in shears:
        g1 = float(shear * direction[0])
        g2 = float(shear * direction[1])
        key = likelihood.cache._key(g1, g2)
        view = likelihood.cache.get(g1, g2)
        detected_mass = (
            likelihood.cache.prior.weights * view.detection_probability
        )
        selected_mass = detected_mass * likelihood.selection_probability(g1, g2)
        total_selected_mass = float(selected_mass.sum())
        if total_selected_mass <= 0:
            raise RuntimeError(
                "the catalogue prior has zero detected-and-selected probability"
            )
        log_b = float(np.log(total_selected_mass))
        for seed, draw in draws.items():
            log_weights = likelihood.log_importance_weights(
                mock.measurements,
                g1,
                g2,
                atom_indices=draw.indices,
                proposal_probability=draw.probability,
                object_chunk=object_chunk,
            )
            for n_draws in ladder:
                log_numerator = (
                    logsumexp(log_weights[:, :n_draws], axis=1)
                    - np.log(n_draws)
                )
                points[seed][n_draws].append(
                    ImportanceProfilePoint(
                        shear=float(shear),
                        log_likelihood_sum=float(
                            np.sum(log_numerator - log_b)
                        ),
                    )
                )
            del log_weights
        if key not in protected:
            likelihood.cache.discard_views((key,))
            del view
            gc.collect()

    return tuple(
        ImportanceProfileResult(
            injected_shear=injected,
            proposal_seed=seed,
            epsilon=float(epsilon),
            n_candidates=int(n_candidates),
            bandwidth=float(bandwidth),
            target_names=proposal.coordinates.target_names,
            rungs=tuple(
                _summarize_profile_rung(n_draws, points[seed][n_draws])
                for n_draws in ladder
            ),
        )
        for seed in seeds
    )


def run_importance_curvature_scan(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    centers: Sequence[float],
    h: float,
    profile_shears: Sequence[float],
    ladder: Sequence[int],
    n_candidates: int,
    epsilon: float,
    bandwidth: float,
    proposal_seed: int,
    direction: Sequence[float] = (1.0, 0.0),
    truth_atom_indices: Optional[np.ndarray] = None,
    object_chunk: int = 64,
    decomposition_chunk: int = 64,
) -> ImportanceCurvatureScanResult:
    """Measure where finite-shear curvature comes from without atom tensors.

    At every requested centre this evaluates the same importance atoms at
    ``centre-h``, ``centre``, and ``centre+h``.  The marginalized observed
    information is decomposed as

    ``I = E_q[I_flow] - Var_q(s_flow)``

    using the posterior atom weights ``q`` at that centre.  Only the three
    temporary object-by-atom weight arrays are resident at once.  The returned
    arrays are object-by-centre or object-by-profile-point, which are small.

    The decomposition assumes the prior, detector, and population
    normalization are shear-invariant.  This is exactly the current
    shape-only, no-measured-selection closure and is checked explicitly.
    """

    centers = tuple(sorted({float(value) for value in centers}))
    ladder = tuple(sorted({int(value) for value in ladder}))
    requested_shears = {float(value) for value in profile_shears}
    if not centers or not np.isfinite(centers).all():
        raise ValueError("centers must be non-empty and finite")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if not ladder or ladder[0] <= 0:
        raise ValueError("ladder must contain positive draw counts")
    if not np.isfinite(tuple(requested_shears)).all():
        raise ValueError("profile shears must be finite")
    if object_chunk <= 0 or decomposition_chunk <= 0:
        raise ValueError("chunk sizes must be positive")
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (2,) or not np.isfinite(direction).all():
        raise ValueError("direction must be a finite two-vector")
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        raise ValueError("direction must be nonzero")
    direction = direction / norm
    injected_vector = mock.truth[["injected_g1", "injected_g2"]].iloc[0].to_numpy(float)
    injected = float(injected_vector @ direction)

    for center in centers:
        requested_shears.update((center - h, center, center + h))
    shears = tuple(sorted(requested_shears))
    draw = proposal.draw(
        mock.measurements,
        n_draws=ladder[-1],
        n_candidates=n_candidates,
        epsilon=epsilon,
        bandwidth=bandwidth,
        seed=proposal_seed,
    )
    truth_indices = None
    if truth_atom_indices is not None:
        truth_indices = np.asarray(truth_atom_indices, dtype=np.int64)
        if truth_indices.shape != (len(mock.measurements),):
            raise ValueError("truth_atom_indices must contain one row per observation")

    protected = set(likelihood.cache.available_shears)
    profile = {
        n_draws: np.full((len(shears), len(mock.measurements)), np.nan, dtype=np.float64)
        for n_draws in ladder
    }
    true_profile = (
        None
        if truth_indices is None
        else np.full((len(shears), len(mock.measurements)), np.nan, dtype=np.float64)
    )
    shear_index = {shear: index for index, shear in enumerate(shears)}

    def evaluate(shear: float):
        g1 = float(shear * direction[0])
        g2 = float(shear * direction[1])
        view = likelihood.cache.get(g1, g2)
        selected_mass = (
            likelihood.cache.prior.weights
            * view.detection_probability
            * likelihood.selection_probability(g1, g2)
        )
        total = float(selected_mass.sum())
        if total <= 0:
            raise RuntimeError("the catalogue prior has zero selected probability")
        log_b = float(np.log(total))
        log_weights = likelihood.log_importance_weights(
            mock.measurements,
            g1,
            g2,
            atom_indices=draw.indices,
            proposal_probability=draw.probability,
            object_chunk=object_chunk,
        )
        row = shear_index[shear]
        for n_draws in ladder:
            profile[n_draws][row] = (
                logsumexp(log_weights[:, :n_draws], axis=1)
                - np.log(n_draws)
                - log_b
            )
        if true_profile is not None:
            true_profile[row] = likelihood.conditional_log_likelihood(
                mock.measurements,
                g1,
                g2,
                atom_indices=truth_indices,
                object_chunk=max(object_chunk, 256),
            )
        return log_weights, log_b, np.asarray(view.detection_probability)

    rung_buffers = {
        n_draws: {
            name: np.empty((len(centers), len(mock.measurements)), dtype=np.float64)
            for name in (
                "score",
                "information",
                "conditional_information",
                "score_variance",
                "residual",
            )
        }
        for n_draws in ladder
    }
    true_scores = (
        None
        if truth_indices is None
        else np.empty((len(centers), len(mock.measurements)), dtype=np.float64)
    )
    true_information = (
        None
        if truth_indices is None
        else np.empty((len(centers), len(mock.measurements)), dtype=np.float64)
    )
    evaluated = set()
    for center_index, center in enumerate(centers):
        triple = (center - h, center, center + h)
        evaluated_triple = [evaluate(value) for value in triple]
        evaluated.update(triple)
        (minus_weights, minus_log_b, minus_detection), (
            zero_weights,
            zero_log_b,
            zero_detection,
        ), (plus_weights, plus_log_b, plus_detection) = evaluated_triple
        if not (
            np.array_equal(minus_detection, zero_detection)
            and np.array_equal(zero_detection, plus_detection)
            and abs(minus_log_b - zero_log_b) <= 1.0e-12
            and abs(zero_log_b - plus_log_b) <= 1.0e-12
        ):
            raise RuntimeError(
                "curvature decomposition requires shear-invariant detection and normalization"
            )

        if true_profile is not None:
            minus_true, zero_true, plus_true = (
                true_profile[shear_index[value]] for value in triple
            )
            true_scores[center_index] = (plus_true - minus_true) / (2.0 * h)
            true_information[center_index] = -(
                plus_true - 2.0 * zero_true + minus_true
            ) / h**2

        for n_draws in ladder:
            minus_ll, zero_ll, plus_ll = (
                profile[n_draws][shear_index[value]] for value in triple
            )
            marginalized_score = (plus_ll - minus_ll) / (2.0 * h)
            marginalized_information = -(
                plus_ll - 2.0 * zero_ll + minus_ll
            ) / h**2
            conditional_information = np.empty(len(mock.measurements), dtype=np.float64)
            score_variance = np.empty(len(mock.measurements), dtype=np.float64)
            for start in range(0, len(mock.measurements), decomposition_chunk):
                stop = min(start + decomposition_chunk, len(mock.measurements))
                minus = minus_weights[start:stop, :n_draws]
                zero = zero_weights[start:stop, :n_draws]
                plus = plus_weights[start:stop, :n_draws]
                log_norm = logsumexp(zero, axis=1, keepdims=True)
                posterior = np.exp(zero - log_norm)
                finite = np.isfinite(minus) & np.isfinite(zero) & np.isfinite(plus)
                atom_score = np.zeros_like(zero)
                atom_information = np.zeros_like(zero)
                atom_score[finite] = (plus[finite] - minus[finite]) / (2.0 * h)
                atom_information[finite] = -(
                    plus[finite] - 2.0 * zero[finite] + minus[finite]
                ) / h**2
                mean_score = np.sum(posterior * atom_score, axis=1)
                conditional_information[start:stop] = np.sum(
                    posterior * atom_information, axis=1
                )
                score_variance[start:stop] = np.sum(
                    posterior * np.square(atom_score - mean_score[:, None]), axis=1
                )
            reconstructed = conditional_information - score_variance
            buffers = rung_buffers[n_draws]
            buffers["score"][center_index] = marginalized_score
            buffers["information"][center_index] = marginalized_information
            buffers["conditional_information"][center_index] = conditional_information
            buffers["score_variance"][center_index] = score_variance
            buffers["residual"][center_index] = marginalized_information - reconstructed
        del minus_weights, zero_weights, plus_weights, evaluated_triple
        for value in triple:
            key = likelihood.cache._key(
                value * direction[0], value * direction[1]
            )
            if key not in protected:
                likelihood.cache.discard_views((key,))
        gc.collect()

    for shear in shears:
        if shear in evaluated:
            continue
        log_weights, _, _ = evaluate(shear)
        del log_weights
        key = likelihood.cache._key(shear * direction[0], shear * direction[1])
        if key not in protected:
            likelihood.cache.discard_views((key,))
        gc.collect()

    rungs = []
    for n_draws in ladder:
        buffers = rung_buffers[n_draws]
        rungs.append(
            ImportanceCurvatureRung(
                n_draws=n_draws,
                centers=centers,
                marginalized_score=buffers["score"],
                marginalized_information=buffers["information"],
                posterior_conditional_information=buffers["conditional_information"],
                posterior_score_variance=buffers["score_variance"],
                decomposition_residual=buffers["residual"],
                true_atom_score=true_scores,
                true_atom_information=true_information,
            )
        )
    return ImportanceCurvatureScanResult(
        injected_shear=injected,
        proposal_seed=int(proposal_seed),
        epsilon=float(epsilon),
        n_candidates=int(n_candidates),
        bandwidth=float(bandwidth),
        target_names=proposal.coordinates.target_names,
        h=float(h),
        shears=shears,
        marginalized_log_likelihood=profile,
        true_atom_log_likelihood=true_profile,
        rungs=tuple(rungs),
    )


__all__ = [
    "CandidateSupportDiagnostics",
    "CoalescedProposalDraw",
    "DefensiveLocalProposal",
    "ImportanceDiagnostics",
    "ImportanceConvergenceAssessment",
    "ImportanceCurvatureRung",
    "ImportanceCurvatureScanResult",
    "ImportanceLadderResult",
    "ImportanceProfilePoint",
    "ImportanceProfileResult",
    "ImportanceProfileRung",
    "ImportanceRung",
    "ProposalCoordinateTable",
    "ProposalCandidates",
    "ProposalDraw",
    "assess_importance_convergence",
    "candidate_support_diagnostics",
    "importance_diagnostics",
    "run_importance_ladder",
    "run_importance_curvature_scan",
    "run_importance_profiles",
    "select_adaptive_draw_counts",
    "select_independent_pilot_draw_counts",
]
