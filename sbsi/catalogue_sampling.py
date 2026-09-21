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


# How many observations the tilted proposal builds its whole-catalogue mixture
# for at once.  The mixture is one float64 row per observation over every active
# atom, so the cost in memory is `chunk * n_atoms * 8` bytes twice over (the
# mixture and its cumulative sum); 32 rows over 12.76M atoms is about 6.5 GB,
# which fits the A40s this runs on.  Driving it one observation at a time cost
# 78% of the estimator's wall clock (cont.328).
TILTED_OBJECT_CHUNK = 32


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

    def _row_major_bins(self, n_draws: int) -> tuple[np.ndarray, int]:
        """Flatten (row, unique-atom) into one bincount axis.

        `np.add.at` is the unbuffered scatter-add, and it is roughly an order of
        magnitude slower than `np.bincount`; the loops these replaced ran it once
        per object per ladder rung, on the reduction's critical path.  Offsetting
        each row by `row * width` turns the whole chunk into a single flat
        histogram, which is the same arithmetic done in one pass.
        """
        n_rows, width = self.indices.shape
        offsets = np.arange(n_rows, dtype=np.int64)[:, None] * width
        return (offsets + self.inverse[:, :n_draws]).ravel(), n_rows * width

    def counts(self, n_draws: int) -> np.ndarray:
        if n_draws <= 0 or n_draws > self.inverse.shape[1]:
            raise ValueError("prefix draw count is outside the coalesced draw")
        if n_draws in self._count_cache:
            return self._count_cache[n_draws]
        bins, size = self._row_major_bins(n_draws)
        counts = (
            np.bincount(bins, minlength=size)
            .astype(np.int32)
            .reshape(self.indices.shape)
        )
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
        bins, size = self._row_major_bins(n_draws)
        # Weighted bincount sums in float64; every weight is 0 or 1 and every
        # total is bounded by n_draws, so the cast back to int32 is exact.
        outside = (
            np.bincount(
                bins,
                weights=(~self.source.local_member[:, :n_draws]).ravel(),
                minlength=size,
            )
            .astype(np.int32)
            .reshape(self.indices.shape)
        )
        global_component = (
            np.bincount(
                bins,
                weights=self.source.global_component[:, :n_draws].ravel(),
                minlength=size,
            )
            .astype(np.int32)
            .reshape(self.indices.shape)
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


class DefensiveLocalProposal:
    """Nearest-neighbour kernel proposal mixed with the full catalogue prior."""

    def __init__(
        self,
        coordinates: ProposalCoordinateTable,
        prior_weights: np.ndarray,
        *,
        local_base_weights: Optional[np.ndarray] = None,
        score_dtype=torch.float64,
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
        if score_dtype not in (torch.float64, torch.float32):
            raise ValueError("score_dtype must be torch.float64 or torch.float32")
        self.score_dtype = score_dtype
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
        self._tilted_proxies = {}
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
            # atom v1.1-infer prior these reusable tables cost about 1 GiB,
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

    def whole_catalogue_proxy_candidates(
        self,
        observed,
        *,
        n_candidates: int,
        device: Optional[str | torch.device] = None,
        object_chunk: int = TILTED_OBJECT_CHUNK,
    ) -> ProposalCandidates:
        """Return the Gaussian-proxy top-K over every active atom.

        This is the direct whole-catalogue analogue of the uncertainty
        reranker in :meth:`candidates`: it uses the same detected-prior mass,
        cached flow mean and diagonal dispersion, but does not first restrict
        the ranking to a nearest-location prefilter.  The query is chunked over
        observations because one dense score row spans the complete active
        catalogue.
        """

        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        chunk = int(object_chunk)
        if chunk <= 0:
            raise ValueError("object_chunk must be positive")
        values = self._observed_values(observed)
        proxy = self._tilted_proxy(device)
        count = min(int(n_candidates), len(self.active_indices))
        indices = np.empty((len(values), count), dtype=np.int64)
        distances = np.empty((len(values), count), dtype=np.float64)
        for start in range(0, len(values), chunk):
            stop = min(start + chunk, len(values))
            selected = proxy.top_score_candidates(values[start:stop], count)
            indices[start:stop] = selected.indices
            distances[start:stop] = selected.distances
        return ProposalCandidates(
            indices,
            distances,
            distances[:, -1].copy(),
        )

    def whole_catalogue_proxy_candidates_and_tilted_draw(
        self,
        observed,
        *,
        n_candidates: int,
        n_draws: int,
        delta: float,
        seed: int,
        temperature: float = 1.0,
        object_offset: int = 0,
        object_ids: Optional[np.ndarray] = None,
        device: Optional[str | torch.device] = None,
        object_chunk: int = TILTED_OBJECT_CHUNK,
    ) -> tuple[ProposalCandidates, ProposalDraw]:
        """Obtain direct top-K and tilted complement draws from one proxy pass."""

        if n_candidates <= 0 or n_draws <= 0:
            raise ValueError("n_candidates and n_draws must be positive")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must lie strictly between zero and one")
        if object_offset < 0:
            raise ValueError("object_offset must be non-negative")
        chunk = int(object_chunk)
        if chunk <= 0:
            raise ValueError("object_chunk must be positive")
        values = self._observed_values(observed)
        n_objects = len(values)
        if object_ids is None:
            absolute_ids = object_offset + np.arange(n_objects, dtype=np.int64)
        else:
            absolute_ids = np.asarray(object_ids, dtype=np.int64)
            if object_offset != 0:
                raise ValueError("object_offset and explicit object_ids cannot be combined")
            if absolute_ids.shape != (n_objects,) or (absolute_ids < 0).any():
                raise ValueError("object_ids must contain one non-negative id per row")

        count = min(int(n_candidates), len(self.active_indices))
        candidate_indices = np.empty((n_objects, count), dtype=np.int64)
        candidate_distances = np.empty((n_objects, count), dtype=np.float64)
        drawn_indices = np.empty((n_objects, n_draws), dtype=np.int64)
        drawn_probability = np.empty((n_objects, n_draws), dtype=np.float64)
        proxy = self._tilted_proxy(device)
        for start in range(0, n_objects, chunk):
            stop = min(start + chunk, n_objects)
            uniforms = np.empty((stop - start, n_draws), dtype=np.float64)
            for offset, row in enumerate(range(start, stop)):
                rng = np.random.default_rng(
                    np.random.SeedSequence([int(seed), int(absolute_ids[row])])
                )
                uniforms[offset] = rng.random((n_draws, 3))[:, 1]
            candidates, drawn, probability = proxy.top_and_draw_uniforms_batch(
                values[start:stop],
                uniforms,
                n_candidates=count,
                delta=float(delta),
                temperature=float(temperature),
            )
            candidate_indices[start:stop] = candidates.indices
            candidate_distances[start:stop] = candidates.distances
            drawn_indices[start:stop] = drawn
            drawn_probability[start:stop] = probability

        candidates = ProposalCandidates(
            candidate_indices,
            candidate_distances,
            candidate_distances[:, -1].copy(),
        )
        local_member = np.empty((n_objects, n_draws), dtype=bool)
        local_position = np.empty((n_objects, n_draws), dtype=np.int32)
        for row in range(n_objects):
            position, member = self._local_positions(
                candidates.indices[row], drawn_indices[row]
            )
            local_position[row] = position
            local_member[row] = member
        draw = ProposalDraw(
            indices=drawn_indices,
            probability=drawn_probability,
            local_member=local_member,
            global_component=np.ones_like(local_member),
            candidate_radius=candidates.radius,
            seed=int(seed),
            local_position=local_position,
        )
        return candidates, draw

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

    def draw_stratified(
        self,
        candidates: ProposalCandidates,
        *,
        n_draws: int,
        seed: int,
        object_offset: int = 0,
        object_ids: Optional[np.ndarray] = None,
    ) -> ProposalDraw:
        """Draw the sampled complement of an exact candidate stratum.

        The defensive mixture spends ``1-epsilon`` of its draws inside the
        candidate support, where the exact sum is already affordable, and only
        ``epsilon`` of them on the complement that carries the rest of the
        target mass.  This estimator splits the sum instead of blending the
        proposal:

        ``A_i = sum_{j in S_i} c_j + (1/M) sum_t 1[j_t not in S_i] c_jt/pi_jt``

        Every draw comes from the detected prior ``pi`` and carries ``pi`` as
        its proposal probability.  Draws landing inside ``S_i`` are flagged
        ``local_member``; the consumer discards them and sums that stratum
        exactly, so the estimator stays unbiased while the dominant stratum
        contributes no variance at all.

        The record layout matches :meth:`draw_adapted` and the atom is read
        from the same uniform column, so at one seed the retained complement
        draws are exactly the mixture's own global-component draws.  The two
        estimators are therefore paired by common random numbers, and every
        field of ``draw(n)`` remains an exact prefix of ``draw(m)`` for n < m.
        """

        if n_draws <= 0:
            raise ValueError("n_draws must be positive")
        if object_offset < 0:
            raise ValueError("object_offset must be non-negative")
        n_objects = len(candidates.indices)
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
        local_member = np.empty((n_objects, n_draws), dtype=bool)
        local_position = np.empty((n_objects, n_draws), dtype=np.int32)
        for row in range(n_objects):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(absolute_ids[row])])
            )
            # Same fixed-width three-column record as draw_adapted, and the
            # atom is taken from that method's global-component column.
            uniforms = rng.random((n_draws, 3))
            indices[row] = np.searchsorted(
                self.global_cdf, uniforms[:, 1], side="right"
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

    def draw_tilted(
        self,
        candidates: ProposalCandidates,
        observed,
        *,
        n_draws: int,
        delta: float,
        seed: int,
        temperature: float = 1.0,
        object_offset: int = 0,
        object_ids: Optional[np.ndarray] = None,
        device: Optional[str | torch.device] = None,
        object_chunk: int = TILTED_OBJECT_CHUNK,
    ) -> ProposalDraw:
        """Draw the complement from a tilted whole-catalogue proposal.

        :meth:`draw_stratified` samples the complement from the detected prior
        ``pi``, whose ratio ``c_j / pi_j`` is unbounded because ``pi`` knows
        nothing about the observation.  cont.317 measured that this is what
        keeps the tail index above the 0.7 reliability threshold, and that
        about 80% of the mass the exact stratum misses lies outside the
        location prefilter, so no prefilter-restricted proposal can reach it.

        This draw replaces ``pi`` with

        ``q_i(j) = (1 - delta) softmax(s_i)_j + delta pi_j``

        where ``s_i`` is the diagonal Gaussian proxy the reranker already
        computes -- here evaluated over every active atom rather than only the
        prefilter, which costs two matmuls and no flow evaluations.  The
        defensive ``delta pi`` component keeps full support, and ``q`` is
        reported exactly, so the complement estimator stays unbiased and only
        its variance changes.

        The record layout, the per-object seeding, and the nested-prefix
        property match :meth:`draw_stratified`, and the atom is drawn from the
        same uniform column, so the two arms remain paired by common random
        numbers at one seed (``doc/CONVENTIONS.md`` section 6d) even though
        they map those uniforms to different atoms.
        """

        if n_draws <= 0:
            raise ValueError("n_draws must be positive")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must lie strictly between zero and one")
        if object_offset < 0:
            raise ValueError("object_offset must be non-negative")
        n_objects = len(candidates.indices)
        if object_ids is None:
            absolute_ids = object_offset + np.arange(n_objects, dtype=np.int64)
        else:
            absolute_ids = np.asarray(object_ids, dtype=np.int64)
            if object_offset != 0:
                raise ValueError("object_offset and explicit object_ids cannot be combined")
            if absolute_ids.shape != (n_objects,) or (absolute_ids < 0).any():
                raise ValueError("object_ids must contain one non-negative id per row")
        values = self._observed_values(observed)
        if len(values) != n_objects:
            raise ValueError("observations and candidate rows must align")
        proxy = self._tilted_proxy(device)
        indices = np.empty((n_objects, n_draws), dtype=np.int64)
        probability = np.empty((n_objects, n_draws), dtype=np.float64)
        local_member = np.empty((n_objects, n_draws), dtype=bool)
        local_position = np.empty((n_objects, n_draws), dtype=np.int32)
        # The uniform stream is still drawn per object from its own absolute
        # id, so the nested prefix and the pairing against `draw_stratified`
        # are unchanged; only the mixture build and the lookup are batched.
        chunk = int(object_chunk)
        if chunk <= 0:
            raise ValueError("object_chunk must be positive")
        for start in range(0, n_objects, chunk):
            stop = min(start + chunk, n_objects)
            block = np.empty((stop - start, n_draws), dtype=np.float64)
            for offset, row in enumerate(range(start, stop)):
                rng = np.random.default_rng(
                    np.random.SeedSequence([int(seed), int(absolute_ids[row])])
                )
                block[offset] = rng.random((n_draws, 3))[:, 1]
            drawn, drawn_probability = proxy.draw_uniforms_batch(
                values[start:stop],
                block,
                delta=float(delta),
                temperature=float(temperature),
            )
            indices[start:stop] = drawn
            probability[start:stop] = drawn_probability
            for row in range(start, stop):
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

    def draw_priority(
        self,
        candidates: ProposalCandidates,
        observed,
        *,
        n_draws: int,
        delta: float,
        seed: int,
        temperature: float = 1.0,
        object_offset: int = 0,
        object_ids: Optional[np.ndarray] = None,
        device: Optional[str | torch.device] = None,
        object_chunk: int = TILTED_OBJECT_CHUNK,
    ) -> ProposalDraw:
        """Priority-sample the complement without replacement, bounding the ratio.

        :meth:`draw_tilted` still samples with replacement from ``q``, so a
        single draw contributes ``c_j / (M q_j)`` and nothing caps that ratio:
        an atom the proxy underrates by a large factor produces an arbitrarily
        large weight whenever it happens to be drawn.  cont.313 measured the
        consequence -- a Pareto tail index above 0.7 on a large minority of
        objects, i.e. infinite weight variance.

        Priority sampling (Duffield, Lund and Thorup) removes the ``1 / M``
        and the repetition together.  Each atom draws its own uniform
        ``u_j`` and receives the priority key ``q_j / u_j``; the ``M`` atoms
        with the largest keys are retained and ``tau`` is the ``(M+1)``-th
        largest key.  Conditional on ``tau`` an atom is retained exactly when
        ``u_j < q_j / tau``, so

        ``sum_{j retained} c_j / min(1, q_j / tau)``

        is unbiased for the complement.  Two properties follow that the
        with-replacement draw does not have.  Every atom with ``q_j >= tau`` is
        retained with certainty and enters at weight exactly ``c_j``, carrying
        no variance at all -- the scheme discovers its own exact stratum rather
        than being handed one.  And no retained atom is inflated by more than
        ``tau / q_j``, which is bounded by the proxy's own ordering rather than
        by luck.

        The atoms of ``candidates`` are removed from the race, because the
        estimator sums that stratum exactly; leaving them in would spend about
        ``K / M`` of the budget on draws that are then discarded.  Removing
        them is a deterministic function of the observation and the catalogue,
        never of the uniforms, so the retained set's inclusion probabilities
        are unchanged in form and the estimator stays unbiased.

        The returned ``probability`` is ``min(1, q_j / tau) / n_draws``.  The
        division cancels the ``- log(n_draws)`` that
        :func:`~sbsi.catalogue_null._stratified_logsumexp` applies to every
        complement term, so that reduction produces ``log c_j - log p_j`` with
        no change to the estimator code path.  Retained atoms are distinct by
        construction, so their coalesced counts are one.

        Two limitations, both recorded in the work log.  The scheme consumes
        one uniform per active atom rather than one per draw, so it cannot
        share a uniform stream with :meth:`draw_tilted` and the two arms are
        paired only by seed and object id, not by common random numbers
        (``doc/CONVENTIONS.md`` section 6d).  And the retained set is nested in
        ``M`` while the weights are not, because ``tau`` depends on ``M``; a
        nested ladder would need a per-rung threshold, so rungs must be run as
        separate single-rung evaluations until that is implemented.
        """

        if n_draws <= 0:
            raise ValueError("n_draws must be positive")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must lie strictly between zero and one")
        if object_offset < 0:
            raise ValueError("object_offset must be non-negative")
        n_objects = len(candidates.indices)
        if object_ids is None:
            absolute_ids = object_offset + np.arange(n_objects, dtype=np.int64)
        else:
            absolute_ids = np.asarray(object_ids, dtype=np.int64)
            if object_offset != 0:
                raise ValueError("object_offset and explicit object_ids cannot be combined")
            if absolute_ids.shape != (n_objects,) or (absolute_ids < 0).any():
                raise ValueError("object_ids must contain one non-negative id per row")
        values = self._observed_values(observed)
        if len(values) != n_objects:
            raise ValueError("observations and candidate rows must align")
        proxy = self._tilted_proxy(device)
        indices = np.empty((n_objects, n_draws), dtype=np.int64)
        probability = np.empty((n_objects, n_draws), dtype=np.float64)
        local_member = np.empty((n_objects, n_draws), dtype=bool)
        local_position = np.empty((n_objects, n_draws), dtype=np.int32)
        chunk = int(object_chunk)
        if chunk <= 0:
            raise ValueError("object_chunk must be positive")
        for start in range(0, n_objects, chunk):
            stop = min(start + chunk, n_objects)
            drawn, inclusion = proxy.select_priority_batch(
                values[start:stop],
                n_select=n_draws,
                delta=float(delta),
                temperature=float(temperature),
                seed=int(seed),
                object_ids=absolute_ids[start:stop],
                excluded=[candidates.indices[row] for row in range(start, stop)],
            )
            indices[start:stop] = drawn
            probability[start:stop] = inclusion / float(n_draws)
            for row in range(start, stop):
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

    def _tilted_proxy(self, device: Optional[str | torch.device]) -> "WholeCatalogueProxy":
        """Cache the whole-catalogue proxy tables per device."""

        resolved = torch.device("cpu" if device is None else device)
        key = f"{resolved}:{self.score_dtype}"
        proxy = self._tilted_proxies.get(key)
        if proxy is None:
            proxy = WholeCatalogueProxy(self, resolved, score_dtype=self.score_dtype)
            self._tilted_proxies[key] = proxy
        return proxy

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



def _priority_row_seed(seed: int, object_id: int) -> int:
    """A stable 63-bit generator seed for one object's priority race.

    Derived through ``SeedSequence`` from the same ``(seed, object_id)`` pair
    the other draws use, so the race is reproducible on a fixed device and
    independent of how observations are chunked.  It is not reproducible
    across devices, because the uniforms come from a torch generator rather
    than from numpy: one uniform per active atom per object is far too many to
    generate on the host.
    """

    state = np.random.SeedSequence([int(seed), int(object_id)]).generate_state(
        2, dtype=np.uint32
    )
    return int((int(state[0]) << 31) | (int(state[1]) >> 1))


class WholeCatalogueProxy:
    """Whole-catalogue diagonal Gaussian proxy score, as a device matvec.

    ``_torch_uncertainty_rerank`` scores a 131,072-atom prefilter with

    ``s_j = log(pi_j Pdet_j) - sum_d log sigma_jd
            - 0.5 sum_d ((x_d - mu_jd) / sigma_jd)^2``

    and then discards everything it does not keep.  Expanding the square makes
    the observation-dependent part two inner products, so the same score is
    affordable over every active atom instead of only the prefilter:

    ``s_j = c_j - 0.5 (x^2 . a1_j - 2 x . a2_j)`` with
    ``a1 = 1/sigma^2``, ``a2 = mu/sigma^2`` and ``c`` collecting the per-atom
    constants.  This is the only whole-catalogue score available without new
    flow evaluations, which is what makes a tail proposal over the full
    catalogue affordable at all.
    """

    def __init__(self, proposal, device, score_dtype=torch.float64):
        if proposal.coordinates.dispersion is None:
            raise ValueError("a tilted tail proposal needs a version-3 proposal cache")
        if score_dtype not in (torch.float64, torch.float32):
            raise ValueError("score_dtype must be torch.float64 or torch.float32")
        # cont.337: the two inner products below are the whole-catalogue term
        # and, per this class's docstring, 78% of the estimator's wall clock.
        # The A40 runs float64 at 1/64 of its float32 rate, so their precision
        # is the dominant cost knob.  Carrying `a1`/`a2` in float32 is safe in a
        # way the h=0.001 stencil is not: `q` enters the weight as a ratio, not
        # as a finite difference, and any strictly positive `q` used
        # consistently to draw and to weight leaves the estimator unbiased --
        # float32 selects a slightly different but equally valid proposal
        # rather than a less accurate answer.  The per-atom constants, the
        # prior floor, the shift and the softmax normalisation stay float64,
        # because those accumulate over 12,760,990 terms where float32 would
        # lose ~sqrt(N)*eps.
        self.score_dtype = score_dtype
        values = proposal.coordinates.values[proposal.active_indices]
        dispersion = proposal.coordinates.dispersion[proposal.active_indices]
        inverse_variance = np.reciprocal(np.square(dispersion))
        constant = (
            np.log(
                np.where(
                    proposal.local_base_weights[proposal.active_indices] > 0,
                    proposal.local_base_weights[proposal.active_indices],
                    np.nan,
                )
            )
            - np.log(dispersion).sum(axis=1)
            - 0.5 * (np.square(values) * inverse_variance).sum(axis=1)
        )
        # Atoms with no detection probability carry no target mass either, so
        # dropping them from the tilted component loses nothing; the defensive
        # prior component still covers them.
        self.constant = torch.as_tensor(
            np.nan_to_num(constant, nan=-np.inf), dtype=torch.float64, device=device
        )
        self.a1 = torch.as_tensor(
            np.ascontiguousarray(inverse_variance), dtype=score_dtype, device=device
        )
        self.a2 = torch.as_tensor(
            np.ascontiguousarray(values * inverse_variance),
            dtype=score_dtype,
            device=device,
        )
        self.active = torch.as_tensor(
            np.ascontiguousarray(proposal.active_indices),
            dtype=torch.long,
            device=device,
        )
        self.prior = torch.as_tensor(
            np.ascontiguousarray(proposal.prior_weights[proposal.active_indices]),
            dtype=torch.float64,
            device=device,
        )

    def score(self, observation: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(observation, dtype=self.score_dtype, device=self.a1.device)
        quadratic = (
            torch.square(x) @ self.a1.T - 2.0 * (x @ self.a2.T)
        ).to(torch.float64)
        return self.constant - 0.5 * quadratic

    def top_score_candidates(
        self, observations: np.ndarray, n_candidates: int
    ) -> ProposalCandidates:
        """Rank all active atoms by the pure diagonal-Gaussian proxy score."""

        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        values = np.atleast_2d(np.ascontiguousarray(observations, dtype=np.float64))
        score = self.score(values)
        count = min(int(n_candidates), int(score.shape[1]))
        selected_score, positions = torch.topk(
            score, count, dim=1, largest=True, sorted=True
        )
        distances = selected_score[:, :1] - selected_score
        finite = torch.isfinite(distances)
        row_max = torch.max(
            torch.where(finite, distances, 0.0), dim=1, keepdim=True
        ).values
        distances = torch.where(finite, distances, row_max + 1.0)
        indices = self.active[positions]
        indices_np = indices.cpu().numpy().astype(np.int64)
        distances_np = distances.cpu().numpy().astype(np.float64)
        return ProposalCandidates(
            indices_np,
            distances_np,
            distances_np[:, -1].copy(),
        )

    def top_and_draw_uniforms_batch(
        self,
        observations: np.ndarray,
        uniforms: np.ndarray,
        *,
        n_candidates: int,
        delta: float,
        temperature: float = 1.0,
    ):
        """Rank and draw from one materialization of the proxy score matrix.

        The score is ranked first, then transformed in place into the tilted
        mixture.  This avoids the second pair of whole-catalogue matrix
        products previously paid by the direct-top-K production arm.
        """

        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must lie strictly between zero and one")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive and finite")
        values = np.atleast_2d(np.ascontiguousarray(observations, dtype=np.float64))
        score = self.score(values)
        count = min(int(n_candidates), int(score.shape[1]))
        selected_score, positions = torch.topk(
            score, count, dim=1, largest=True, sorted=True
        )
        distances = selected_score[:, :1] - selected_score
        finite_distance = torch.isfinite(distances)
        row_max = torch.max(
            torch.where(finite_distance, distances, 0.0), dim=1, keepdim=True
        ).values
        distances = torch.where(finite_distance, distances, row_max + 1.0)
        candidates = ProposalCandidates(
            self.active[positions].cpu().numpy().astype(np.int64),
            distances.cpu().numpy().astype(np.float64),
            distances[:, -1].cpu().numpy().astype(np.float64),
        )
        del selected_score, positions, distances

        # Reuse the dense score allocation as q.  The stable in-place softmax
        # keeps peak memory at the existing score+cumulative footprint.
        if temperature != 1.0:
            score.div_(float(temperature))
        score.nan_to_num_(nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        maximum = torch.max(score, dim=1, keepdim=True).values
        if not torch.isfinite(maximum).all():
            raise RuntimeError("whole-catalogue proxy has no finite atom in a row")
        score.sub_(maximum).exp_()
        score.div_(score.sum(dim=1, keepdim=True))
        score.mul_(1.0 - delta).add_(self.prior.unsqueeze(0), alpha=delta)
        cumulative = torch.cumsum(score, dim=1)
        cumulative[:, -1] = 1.0
        uniform_tensor = torch.as_tensor(
            np.atleast_2d(np.ascontiguousarray(uniforms, dtype=np.float64)),
            dtype=torch.float64,
            device=score.device,
        )
        if uniform_tensor.shape[0] != score.shape[0]:
            raise ValueError("one uniform row is needed per observation")
        drawn_positions = torch.searchsorted(cumulative, uniform_tensor, right=True)
        drawn_positions.clamp_(max=score.shape[1] - 1)
        drawn = self.active[drawn_positions].cpu().numpy()
        probability = torch.gather(score, 1, drawn_positions).cpu().numpy()
        return candidates, drawn, probability

    def mixture(
        self,
        observations: np.ndarray,
        *,
        delta: float,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Build ``q = (1-delta) softmax(s/T) + delta pi`` for a batch of rows.

        One row of the returned ``(B, n_atoms)`` tensor per observation.  This
        is the single expensive operation in the tilted proposal -- cont.328
        measured it at 78% of the estimator's wall clock when driven one
        observation at a time -- and it is the only place the mixture is
        formed, so the batched and single-row paths cannot drift apart.
        """

        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive and finite")
        x = torch.as_tensor(
            np.atleast_2d(np.ascontiguousarray(observations, dtype=np.float64)),
            dtype=self.score_dtype,
            device=self.a1.device,
        )
        quadratic = (
            torch.square(x) @ self.a1.T - 2.0 * (x @ self.a2.T)
        ).to(torch.float64)
        score = self.constant.unsqueeze(0) - 0.5 * quadratic
        if temperature != 1.0:
            score /= float(temperature)
        # Atoms the proxy drops carry -inf; softmax sends them to zero, but the
        # shift has to come from a finite maximum or the whole row becomes NaN.
        score = torch.where(
            torch.isfinite(score),
            score,
            torch.tensor(-np.inf, dtype=score.dtype, device=score.device),
        )
        mixture = torch.softmax(score, dim=1)
        mixture *= 1.0 - delta
        mixture += delta * self.prior.unsqueeze(0)
        return mixture

    def select_priority_batch(
        self,
        observations: np.ndarray,
        *,
        n_select: int,
        delta: float,
        temperature: float = 1.0,
        seed: int,
        object_ids: np.ndarray,
        excluded=None,
    ):
        """Retain the ``n_select`` atoms with the largest priority keys.

        Returns absolute atom ids and their inclusion probabilities
        ``min(1, q_j / tau)``, one row per observation.  See
        :meth:`DefensiveLocalProposal.draw_priority` for the estimator this
        feeds and why the ratio it produces is bounded.

        The mixture is built for the whole chunk in one pair of matmuls, as in
        :meth:`draw_uniforms_batch`, but the race itself runs one row at a time
        so the per-atom uniform buffer stays ``O(n_atoms)`` rather than
        ``O(B n_atoms)``.  Each row seeds its own generator from its absolute
        object id, so the retained set does not depend on how observations are
        chunked.
        """

        if n_select <= 0:
            raise ValueError("n_select must be positive")
        mixture = self.mixture(
            observations, delta=delta, temperature=temperature
        )
        n_rows, n_atoms = mixture.shape
        if n_select + 1 > n_atoms:
            raise ValueError(
                "priority sampling needs more active atoms than draws: "
                f"n_select={n_select}, active={n_atoms}"
            )
        object_ids = np.asarray(object_ids, dtype=np.int64)
        if object_ids.shape != (n_rows,):
            raise ValueError("object_ids must carry one id per observation row")
        device = mixture.device
        indices = np.empty((n_rows, n_select), dtype=np.int64)
        inclusion = np.empty((n_rows, n_select), dtype=np.float64)
        generator = torch.Generator(device=device)
        tiny = float(np.finfo(np.float64).tiny)
        for row in range(n_rows):
            weight = mixture[row]
            if excluded is not None:
                # A deterministic removal: the exact stratum is summed with no
                # proposal correction, so racing it here would only produce
                # draws the estimator discards.
                dropped = np.asarray(excluded[row], dtype=np.int64)
                if dropped.size:
                    position = torch.searchsorted(
                        self.active,
                        torch.as_tensor(dropped, dtype=torch.long, device=device),
                    ).clamp_max(n_atoms - 1)
                    present = self.active[position] == torch.as_tensor(
                        dropped, dtype=torch.long, device=device
                    )
                    weight = weight.clone()
                    weight[position[present]] = 0.0
            generator.manual_seed(_priority_row_seed(seed, int(object_ids[row])))
            uniform = torch.rand(
                n_atoms, generator=generator, device=device, dtype=torch.float64
            ).clamp_min(tiny)
            key = weight / uniform
            top = torch.topk(key, n_select + 1, sorted=True)
            threshold = top.values[n_select]
            chosen = top.indices[:n_select]
            if not torch.isfinite(threshold) or threshold <= 0.0:
                raise RuntimeError(
                    "priority threshold is not a positive finite key: "
                    f"object_id={int(object_ids[row])}, tau={float(threshold)}"
                )
            indices[row] = self.active[chosen].cpu().numpy()
            inclusion[row] = (
                (weight[chosen] / threshold).clamp_max(1.0).cpu().numpy()
            )
        return indices, inclusion

    def top_atoms(
        self,
        observation: np.ndarray,
        n_top: int,
        *,
        delta: float,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Return the ``n_top`` atom ids carrying the largest proposal mass.

        The exact stratum exists because a term summed exactly contributes no
        variance.  The candidate shortlist chooses that stratum with the
        reranker, which only ever sees the prefilter; this picks it with the
        whole-catalogue score instead, so it can reach the atoms the prefilter
        never offered.  Membership depends only on the observation and the
        catalogue, never on the draws, so moving these atoms into the exact sum
        leaves the estimator unbiased.
        """

        if n_top <= 0:
            return np.empty(0, dtype=np.int64)
        mixture = self.mixture(
            observation, delta=delta, temperature=temperature
        )[0]
        count = min(int(n_top), int(mixture.numel()))
        positions = torch.topk(mixture, count, sorted=False).indices
        return self.active[positions].cpu().numpy()

    def draw(
        self,
        observation: np.ndarray,
        *,
        n_draws: int,
        delta: float,
        rng,
        temperature: float = 1.0,
    ):
        """Sample ``q = (1-delta) softmax(s/T) + delta pi`` over every atom.

        Returns absolute atom ids and their exact mixture probability, so the
        importance ratio ``c_j / q_j`` stays exact and the estimator unbiased.
        """

        return self.draw_uniforms(
            observation, rng.random(n_draws), delta=delta, temperature=temperature
        )

    def draw_uniforms(
        self,
        observation: np.ndarray,
        uniforms,
        *,
        delta: float,
        temperature: float = 1.0,
    ):
        """Map an explicit uniform stream to atoms, for common random numbers.

        Taking the uniforms rather than an rng lets a caller reuse the stream
        another proposal consumed, so two estimator arms stay paired.

        ``temperature`` above one flattens the tilted component.  The proxy is
        a Gaussian approximation to the target, so where that approximation is
        poor the tilted component is over-confident and the ratio ``c_j / q_j``
        spikes; flattening trades some efficiency on the easy objects for
        coverage on the hard ones.  It changes only the proposal, never the
        estimator, so unbiasedness is untouched at any value.
        """

        indices, probability = self.draw_uniforms_batch(
            np.atleast_2d(observation),
            np.atleast_2d(uniforms),
            delta=delta,
            temperature=temperature,
        )
        return indices[0], probability[0]

    def draw_uniforms_batch(
        self,
        observations: np.ndarray,
        uniforms: np.ndarray,
        *,
        delta: float,
        temperature: float = 1.0,
    ):
        """Map one uniform stream per observation to atoms, all rows at once.

        Identical mathematics to the single-row path -- which now calls this
        with one row -- but the mixture, its cumulative sum and the inverse-CDF
        lookup are done for a whole chunk of observations in one set of kernel
        launches.  cont.328 measured ~10x on that step, which was 78% of the
        estimator's wall clock.  The arithmetic is reassociated across rows, so
        the reported probabilities move by of order 1e-15 relative; the drawn
        atoms are unchanged.
        """

        mixture = self.mixture(
            observations, delta=delta, temperature=temperature
        )
        cumulative = torch.cumsum(mixture, dim=1)
        # Guard the inverse-CDF lookup against the cumulative sum finishing a
        # rounding step below one, which would send a uniform near one past the
        # last atom.
        cumulative[:, -1] = 1.0
        drawn = torch.as_tensor(
            np.ascontiguousarray(
                np.atleast_2d(np.asarray(uniforms, dtype=np.float64))
            ),
            dtype=torch.float64,
            device=cumulative.device,
        )
        if drawn.shape[0] != cumulative.shape[0]:
            raise ValueError("one uniform row is needed per observation")
        positions = torch.searchsorted(cumulative, drawn, right=True)
        positions = torch.clamp(positions, max=mixture.shape[1] - 1)
        return (
            self.active[positions].cpu().numpy(),
            torch.gather(mixture, 1, positions).cpu().numpy(),
        )


def pareto_tail_index(weights, *, minimum_tail: int = 5):
    """Generalized-Pareto shape ``k`` of each row's upper weight tail.

    This is the Zhang and Stephens (2009) empirical-Bayes estimator used as the
    PSIS ``k-hat`` diagnostic.  It measures how heavy the importance-weight tail
    is, which is what decides whether a draw ladder can converge at all:
    ``k >= 0.5`` means the weight variance is infinite, so the estimator has no
    root-M rate and a 1/M finite-draw bias correction is not applicable;
    ``k >= 1`` means the mean is infinite as well.  The practical reliability
    threshold is ``min(1 - 1/log10(S), 0.7)`` for a sample of size ``S``.

    ``weights`` is a nonnegative ``(rows, S)`` array or tensor of raw importance
    weights, not log weights and not normalized; the shape is invariant to
    rescaling a row.  The fit uses the largest ``min(0.2 S, 3 sqrt(S))`` of
    them.  Rows whose tail is degenerate -- fewer than ``minimum_tail``
    exceedances, or no spread above the threshold -- return NaN rather than a
    fitted value, because for those rows the diagnostic is undefined rather
    than good.
    """

    tensor = torch.as_tensor(weights)
    if tensor.ndim != 2:
        raise ValueError("weights must be a (rows, draws) matrix")
    tensor = tensor.to(torch.float64)
    if torch.isnan(tensor).any() or (tensor < 0).any():
        raise ValueError("weights must be nonnegative and free of NaN")
    n_rows, n_draws = tensor.shape
    n_tail = int(min(0.2 * n_draws, 3.0 * np.sqrt(n_draws)))
    if n_tail < int(minimum_tail):
        return np.full(n_rows, np.nan, dtype=np.float64)

    # Sort where the caller's data already is -- this is the expensive step for
    # a wide draw matrix -- then fit on the CPU, where the tail is small.
    ordered, _ = torch.sort(tensor, dim=1)
    threshold = ordered[:, n_draws - n_tail - 1 : n_draws - n_tail]
    exceedance = (ordered[:, n_draws - n_tail :] - threshold).detach().cpu()
    # The fit assumes a continuous tail.  It places its parameter grid on the
    # largest exceedance and on the tail's own first quartile, so a row whose
    # lower tail is tied at the threshold -- a weight vector taking only a few
    # distinct values -- divides by zero there and returns a number with no
    # meaning.  Such rows are undefined, not light-tailed.
    quartile_index = int(np.floor(n_tail / 4.0 + 0.5)) - 1
    usable = (exceedance[:, -1] > 0) & (exceedance[:, quartile_index] > 0)
    khat = torch.full((n_rows,), float("nan"), dtype=torch.float64)
    if not bool(usable.any()):
        return khat.numpy()
    x = exceedance[usable].clamp_min(torch.finfo(torch.float64).tiny)

    # Zhang and Stephens place a grid of candidate theta values using the
    # largest observation and the tail's own first quartile, then average them
    # by profile likelihood instead of maximizing.
    prior = 3.0
    n_grid = 30 + int(np.sqrt(n_tail))
    grid = torch.arange(1, n_grid + 1, dtype=torch.float64)
    quartile = x[:, quartile_index].unsqueeze(1)
    theta = 1.0 / x[:, -1].unsqueeze(1) + (
        1.0 - torch.sqrt(n_grid / (grid - 0.5))
    ) / (prior * quartile)
    shape = torch.log1p(-theta.unsqueeze(2) * x.unsqueeze(1)).mean(dim=2)
    profile = n_tail * (torch.log(-theta / shape) - shape - 1.0)
    weight = torch.softmax(profile, dim=1)
    theta_hat = (weight * theta).sum(dim=1)
    fitted = torch.log1p(-theta_hat.unsqueeze(1) * x).mean(dim=1)
    # Weakly informative prior on k, shrinking a short tail toward 1/2.
    adjust = 10.0
    khat[usable] = (fitted * n_tail + 0.5 * adjust) / (n_tail + adjust)
    return khat.numpy()


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

__all__ = [
    "CoalescedProposalDraw",
    "DefensiveLocalProposal",
    "ImportanceDiagnostics",
    "ImportanceConvergenceAssessment",
    "ImportanceLadderResult",
    "ImportanceProfilePoint",
    "ImportanceProfileResult",
    "ImportanceProfileRung",
    "ImportanceRung",
    "ProposalCoordinateTable",
    "ProposalCandidates",
    "ProposalDraw",
    "assess_importance_convergence",
    "importance_diagnostics",
    "run_importance_ladder",
    "run_importance_profiles",
    "select_adaptive_draw_counts",
    "select_independent_pilot_draw_counts",
]
