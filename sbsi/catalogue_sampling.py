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
from hashlib import sha256
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
class TailProposalRecipe:
    """Immutable provenance for complement-conditioned deterministic MIS.

    Every component except the last retains the detected-prior base measure
    and tempers only the observation-dependent Gaussian proxy likelihood.  The
    final component is the catalogue prior.  ``component_cycle`` fixes source
    counts in every complete cycle; all samples are nevertheless weighted by
    the balance-mixture probability, not by their source probability.
    """

    name: str
    component_names: tuple[str, ...]
    component_weights: tuple[float, ...]
    component_betas: tuple[Optional[float], ...]
    component_cycle: tuple[int, ...]
    conditioning: str = "component"
    base_measure: str = "prior_times_detection"
    temperature_scope: str = "proxy_likelihood_only"
    denominator: str = "balance_mixture"

    def __post_init__(self):
        names = tuple(self.component_names)
        weights = tuple(float(value) for value in self.component_weights)
        betas = tuple(
            None if value is None else float(value)
            for value in self.component_betas
        )
        cycle = tuple(int(value) for value in self.component_cycle)
        if (
            len(names) < 2
            or len(weights) != len(names)
            or len(betas) != len(names)
        ):
            raise ValueError(
                "the tail recipe needs aligned proxy and final-prior components"
            )
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("tail-recipe component names must be unique and non-empty")
        if names[-1] != "prior":
            raise ValueError("the final tail-recipe component must be named prior")
        if not all(np.isfinite(weights)) or any(value <= 0.0 for value in weights):
            raise ValueError("tail-recipe component weights must be finite and positive")
        if not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1.0e-15):
            raise ValueError("tail-recipe component weights must sum to one")
        if (
            betas[-1] is not None
            or any(
                value is None or not np.isfinite(value) or value < 0.0
                for value in betas[:-1]
            )
        ):
            raise ValueError(
                "proxy components need non-negative betas and the prior beta must be None"
            )
        if not cycle or any(value < 0 or value >= len(names) for value in cycle):
            raise ValueError("tail-recipe cycle contains an invalid component id")
        frequencies = np.bincount(cycle, minlength=len(names)) / len(cycle)
        if not np.allclose(frequencies, weights, rtol=0.0, atol=1.0e-15):
            raise ValueError("tail-recipe cycle frequencies must equal component weights")
        if self.conditioning != "component":
            raise ValueError("tail components must be conditioned separately")
        if self.base_measure != "prior_times_detection":
            raise ValueError("proxy components must use the prior-times-detection base")
        if self.temperature_scope != "proxy_likelihood_only":
            raise ValueError("temperature must apply only to the proxy likelihood")
        if self.denominator != "balance_mixture":
            raise ValueError("deterministic MIS requires the balance-mixture denominator")
        object.__setattr__(self, "component_names", names)
        object.__setattr__(self, "component_weights", weights)
        object.__setattr__(self, "component_betas", betas)
        object.__setattr__(self, "component_cycle", cycle)

    @property
    def cycle_length(self) -> int:
        return len(self.component_cycle)

    @property
    def component_counts(self) -> tuple[int, ...]:
        counts = np.bincount(
            self.component_cycle, minlength=len(self.component_names)
        )
        return tuple(int(value) for value in counts)

    @property
    def proxy_temperatures(self) -> tuple[float, ...]:
        return tuple(
            np.inf if value == 0.0 else 1.0 / value
            for value in self.component_betas
            if value is not None
        )


TAIL_PROPOSAL_RECIPE = TailProposalRecipe(
    name="conditioned_dm_mis_t1_t2_prior_4_3_1_v2",
    component_names=("proxy_t1", "proxy_t2", "prior"),
    component_weights=(0.5, 0.375, 0.125),
    component_betas=(1.0, 0.5, None),
    component_cycle=(0, 1, 0, 2, 1, 0, 1, 0),
)


TAIL_PROPOSAL_ATOM_CHUNK = 262144
# A proposal batch owns, at its peak, one float64 probability slab per
# component plus likelihood/cumulative workspaces.  At the fixed 262,144-atom
# slab, the validated four-component stress recipe used 2.44 GB total CUDA
# allocation including the static whole-catalogue tables, while turning the two
# scans into matrix--matrix rather than hundreds of matrix--vector operations.
TAIL_PROPOSAL_OBJECT_CHUNK = 32


# The pilot is discarded, but using the same fixed DM3 law makes its primary
# second-moment score directly relevant to the production proposal.  Keep an
# explicit alias so resolved provenance can name both roles without copying a
# second recipe that might silently drift.
PILOT_STRATUM_PROPOSAL_RECIPE = TAIL_PROPOSAL_RECIPE


PILOT_STRATUM_HASH_CONTRACT = "sha256_domain_shape_big_endian_v1"
PILOT_DUPLICATE_LOG_PROBABILITY_ATOL = 1.0e-12


@dataclass(frozen=True)
class PilotStratumRecipe:
    """Frozen independent-pilot rule for enlarging an exact stratum.

    The pilot proposal is used only to discover atoms.  A consumer must sum the
    returned stratum exactly and generate its production complement from an
    independent random stream; this class deliberately contains no production
    seed or estimator state.
    """

    name: str
    pilot_draws: int
    max_anchors: int
    n_promotions: int
    neighbors_per_anchor: int
    pilot_proposal: TailProposalRecipe
    atom_chunk: int = TAIL_PROPOSAL_ATOM_CHUNK
    object_chunk: int = TAIL_PROPOSAL_OBJECT_CHUNK
    score_name: str = "log_target_squared_over_proposal"
    ranking: str = "unique_max_explicit_score_desc_atom_id"
    promotion_strategy: str = "ranked_anchors_with_coordinate_neighbors"
    coordinate_metric: str = "standardized_proposal_euclidean"
    neighbor_order: str = "ranked_anchor_then_distance_atom_id"
    hash_contract: str = PILOT_STRATUM_HASH_CONTRACT

    def __post_init__(self):
        name = str(self.name)
        pilot_draws = int(self.pilot_draws)
        max_anchors = int(self.max_anchors)
        n_promotions = int(self.n_promotions)
        neighbors_per_anchor = int(self.neighbors_per_anchor)
        atom_chunk = int(self.atom_chunk)
        object_chunk = int(self.object_chunk)
        if not name:
            raise ValueError("pilot-stratum recipe name must be non-empty")
        if not isinstance(self.pilot_proposal, TailProposalRecipe):
            raise TypeError("pilot proposal must be a TailProposalRecipe")
        if pilot_draws <= 0 or pilot_draws % self.pilot_proposal.cycle_length:
            raise ValueError(
                "pilot draws must be a positive multiple of the pilot-proposal cycle"
            )
        if max_anchors <= 0 or max_anchors > pilot_draws:
            raise ValueError("maximum pilot anchors must lie in [1, pilot_draws]")
        if n_promotions <= 0:
            raise ValueError("pilot promotions must be positive")
        if neighbors_per_anchor <= 0:
            raise ValueError("neighbors per anchor must be positive")
        if atom_chunk <= 0 or object_chunk <= 0:
            raise ValueError("pilot proposal chunk sizes must be positive")
        if not str(self.score_name):
            raise ValueError("pilot score name must be non-empty")
        if self.ranking != "unique_max_explicit_score_desc_atom_id":
            raise ValueError("unsupported pilot-anchor ranking")
        promotion_strategy = str(self.promotion_strategy)
        if promotion_strategy == "ranked_anchors_with_coordinate_neighbors":
            if self.coordinate_metric != "standardized_proposal_euclidean":
                raise ValueError("unsupported pilot-neighbor coordinate metric")
            if self.neighbor_order != "ranked_anchor_then_distance_atom_id":
                raise ValueError("unsupported pilot-neighbor ordering")
        elif promotion_strategy == "ranked_atoms_then_active_id_fill":
            if neighbors_per_anchor != 1:
                raise ValueError(
                    "direct pilot promotion requires one atom per anchor"
                )
            if self.coordinate_metric != "none" or self.neighbor_order != "none":
                raise ValueError(
                    "direct pilot promotion must disable coordinate neighbours"
                )
        else:
            raise ValueError("unsupported pilot promotion strategy")
        if self.hash_contract != PILOT_STRATUM_HASH_CONTRACT:
            raise ValueError("unsupported pilot-stratum hash contract")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "pilot_draws", pilot_draws)
        object.__setattr__(self, "max_anchors", max_anchors)
        object.__setattr__(self, "n_promotions", n_promotions)
        object.__setattr__(self, "neighbors_per_anchor", neighbors_per_anchor)
        object.__setattr__(self, "atom_chunk", atom_chunk)
        object.__setattr__(self, "object_chunk", object_chunk)
        object.__setattr__(self, "score_name", str(self.score_name))
        object.__setattr__(self, "promotion_strategy", promotion_strategy)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        digest = sha256()
        digest.update(b"SBSI/pilot-stratum-recipe/v1\0")
        digest.update(payload)
        return digest.hexdigest()


PILOT_STRATUM_RECIPE = PilotStratumRecipe(
    name="independent_pilot_exact_stratum_direct_second_moment_v2",
    pilot_draws=8192,
    max_anchors=4096,
    n_promotions=4096,
    neighbors_per_anchor=1,
    pilot_proposal=PILOT_STRATUM_PROPOSAL_RECIPE,
    promotion_strategy="ranked_atoms_then_active_id_fill",
    coordinate_metric="none",
    neighbor_order="none",
)


def _array_manifest_sha256(
    label: str,
    values: np.ndarray,
    *,
    dtype: str,
    sort_rows: bool = False,
) -> str:
    """Hash an array with explicit domain, shape, and endian conventions."""

    array = np.asarray(values)
    if sort_rows:
        if array.ndim != 2:
            raise ValueError("only matrices can be canonically sorted by row")
        array = np.sort(array, axis=1)
    canonical = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    shape = np.ascontiguousarray(canonical.shape, dtype=np.dtype(">u8"))
    digest = sha256()
    digest.update(b"SBSI/pilot-stratum-array/v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(shape.tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _object_array_manifest_sha256(
    label: str,
    values: np.ndarray,
    *,
    dtype: str,
    sort_rows: bool = False,
) -> tuple[str, ...]:
    """Hash rows independently so provenance survives object slicing."""

    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("object manifests require a matrix")
    return tuple(
        _array_manifest_sha256(
            label,
            array[row : row + 1],
            dtype=dtype,
            sort_rows=sort_rows,
        )
        for row in range(len(array))
    )


def _candidate_exclusion_manifest_sha256(
    indices: np.ndarray,
) -> tuple[str, ...]:
    """Canonical per-object identity of a conditioned proposal exclusion."""

    return _object_array_manifest_sha256(
        "object_conditioned_exclusion_indices",
        indices,
        dtype=">i8",
        sort_rows=True,
    )


@dataclass(frozen=True)
class PilotStratumProvenance:
    """Canonical hashes for one pilot-promotion result."""

    hash_contract: str
    recipe_sha256: str
    pilot_seed: int
    pilot_atom_chunk: int
    pilot_object_chunk: int
    call_pilot_draw_indices_sha256: str
    call_pilot_probability_sha256: str
    call_pilot_scores_sha256: str
    call_base_stratum_indices_sha256: str
    call_anchor_indices_sha256: str
    call_promoted_indices_sha256: str
    call_final_stratum_indices_sha256: str
    object_pilot_draw_indices_sha256: tuple[str, ...]
    object_pilot_probability_sha256: tuple[str, ...]
    object_pilot_scores_sha256: tuple[str, ...]
    object_base_stratum_indices_sha256: tuple[str, ...]
    object_anchor_indices_sha256: tuple[str, ...]
    object_promoted_indices_sha256: tuple[str, ...]
    object_final_stratum_indices_sha256: tuple[str, ...]

    def __post_init__(self):
        if self.hash_contract != PILOT_STRATUM_HASH_CONTRACT:
            raise ValueError("pilot provenance uses an unsupported hash contract")
        pilot_atom_chunk = int(self.pilot_atom_chunk)
        pilot_object_chunk = int(self.pilot_object_chunk)
        if pilot_atom_chunk <= 0 or pilot_object_chunk <= 0:
            raise ValueError("pilot provenance chunk sizes must be positive")
        object_hash_fields = (
            "object_pilot_draw_indices_sha256",
            "object_pilot_probability_sha256",
            "object_pilot_scores_sha256",
            "object_base_stratum_indices_sha256",
            "object_anchor_indices_sha256",
            "object_promoted_indices_sha256",
            "object_final_stratum_indices_sha256",
        )
        object_hashes = tuple(
            tuple(getattr(self, name)) for name in object_hash_fields
        )
        if len({len(values) for values in object_hashes}) != 1:
            raise ValueError("pilot provenance object hashes must have equal lengths")
        hashes = (
            self.recipe_sha256,
            self.call_pilot_draw_indices_sha256,
            self.call_pilot_probability_sha256,
            self.call_pilot_scores_sha256,
            self.call_base_stratum_indices_sha256,
            self.call_anchor_indices_sha256,
            self.call_promoted_indices_sha256,
            self.call_final_stratum_indices_sha256,
            *(value for values in object_hashes for value in values),
        )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("pilot provenance hashes must be lowercase SHA-256")
        object.__setattr__(self, "pilot_seed", int(self.pilot_seed))
        object.__setattr__(self, "pilot_atom_chunk", pilot_atom_chunk)
        object.__setattr__(self, "pilot_object_chunk", pilot_object_chunk)
        for name, values in zip(object_hash_fields, object_hashes):
            object.__setattr__(self, name, values)


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
    component_id: Optional[np.ndarray] = None
    component_complement_mass: Optional[np.ndarray] = None
    component_recipe: Optional[TailProposalRecipe] = None
    atom_chunk: Optional[int] = None
    object_chunk: Optional[int] = None
    object_excluded_candidate_indices_sha256: Optional[tuple[str, ...]] = None

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
        if (self.component_id is None) != (self.component_complement_mass is None):
            raise ValueError(
                "component ids and complement-mass diagnostics must be stored together"
            )
        if (self.component_id is None) != (self.component_recipe is None):
            raise ValueError(
                "component diagnostics and recipe provenance must be stored together"
            )
        if self.component_recipe is None:
            if (
                self.atom_chunk is not None
                or self.object_chunk is not None
                or self.object_excluded_candidate_indices_sha256 is not None
            ):
                raise ValueError(
                    "component chunk provenance requires a component recipe"
                )
        else:
            if not isinstance(self.component_recipe, TailProposalRecipe):
                raise TypeError("component recipe must be a TailProposalRecipe")
            if self.atom_chunk is None or self.object_chunk is None:
                raise ValueError(
                    "component recipe provenance requires atom and object chunks"
                )
            if self.object_excluded_candidate_indices_sha256 is None:
                raise ValueError(
                    "component recipe provenance requires conditioned-exclusion hashes"
                )
            atom_chunk = int(self.atom_chunk)
            object_chunk = int(self.object_chunk)
            if atom_chunk <= 0 or object_chunk <= 0:
                raise ValueError("component provenance chunk sizes must be positive")
            object.__setattr__(self, "atom_chunk", atom_chunk)
            object.__setattr__(self, "object_chunk", object_chunk)
            exclusion_hashes = tuple(
                self.object_excluded_candidate_indices_sha256
            )
            if len(exclusion_hashes) != len(self.indices) or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in exclusion_hashes
            ):
                raise ValueError(
                    "conditioned-exclusion hashes must contain one lowercase SHA-256 per row"
                )
            object.__setattr__(
                self,
                "object_excluded_candidate_indices_sha256",
                exclusion_hashes,
            )
        if self.component_id is not None:
            component_id = np.asarray(self.component_id, dtype=np.int16)
            complement_mass = np.asarray(
                self.component_complement_mass, dtype=np.float64
            )
            if component_id.shape != self.indices.shape:
                raise ValueError("component ids must align with proposal draws")
            if complement_mass.ndim != 2 or complement_mass.shape[0] != len(self.indices):
                raise ValueError(
                    "component complement masses need one row per observed object"
                )
            n_components = len(self.component_recipe.component_names)
            if complement_mass.shape[1] != n_components:
                raise ValueError(
                    "component complement masses do not match the component recipe"
                )
            if (
                (component_id < 0).any()
                or (component_id >= n_components).any()
            ):
                raise ValueError("component ids are outside the diagnostic component axis")
            schedule = np.resize(
                np.asarray(
                    self.component_recipe.component_cycle, dtype=np.int16
                ),
                self.indices.shape[1],
            )
            if not np.array_equal(
                component_id,
                np.broadcast_to(schedule, component_id.shape),
            ):
                raise ValueError(
                    "component schedule does not match the recorded component recipe"
                )
            if (
                not np.isfinite(complement_mass).all()
                or (complement_mass < 0.0).any()
                or (complement_mass > 1.0 + 1.0e-12).any()
            ):
                raise ValueError("component complement masses must lie in [0, 1]")
            object.__setattr__(self, "component_id", component_id)
            object.__setattr__(self, "component_complement_mass", complement_mass)

    def prefix(self, n_draws: int) -> "ProposalDraw":
        if n_draws <= 0 or n_draws > self.indices.shape[1]:
            raise ValueError("prefix draw count is outside the stored ladder")
        return ProposalDraw(
            indices=self.indices[:, :n_draws],
            probability=self.probability[:, :n_draws],
            local_member=self.local_member[:, :n_draws],
            global_component=self.global_component[:, :n_draws],
            candidate_radius=self.candidate_radius,
            seed=self.seed,
            local_position=self.local_position[:, :n_draws],
            component_id=(
                None
                if self.component_id is None
                else self.component_id[:, :n_draws]
            ),
            component_complement_mass=self.component_complement_mass,
            component_recipe=self.component_recipe,
            atom_chunk=self.atom_chunk,
            object_chunk=self.object_chunk,
            object_excluded_candidate_indices_sha256=(
                self.object_excluded_candidate_indices_sha256
            ),
        )

    def take(self, rows: np.ndarray) -> "ProposalDraw":
        """Select observed-object rows without changing their draw streams."""

        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or ((rows < 0) | (rows >= len(self.indices))).any():
            raise ValueError("proposal row selection is invalid")
        return ProposalDraw(
            indices=self.indices[rows],
            probability=self.probability[rows],
            local_member=self.local_member[rows],
            global_component=self.global_component[rows],
            candidate_radius=self.candidate_radius[rows],
            seed=self.seed,
            local_position=self.local_position[rows],
            component_id=(
                None if self.component_id is None else self.component_id[rows]
            ),
            component_complement_mass=(
                None
                if self.component_complement_mass is None
                else self.component_complement_mass[rows]
            ),
            component_recipe=self.component_recipe,
            atom_chunk=self.atom_chunk,
            object_chunk=self.object_chunk,
            object_excluded_candidate_indices_sha256=(
                None
                if self.object_excluded_candidate_indices_sha256 is None
                else tuple(
                    self.object_excluded_candidate_indices_sha256[int(row)]
                    for row in rows
                )
            ),
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


@dataclass(frozen=True)
class PilotStratumPromotion:
    """Fixed-width exact stratum selected by a discarded pilot.

    Invalid anchor slots are padded with ``-1`` and identified by
    ``anchor_valid``.  Promoted and final-stratum matrices never use padding:
    each row has exactly the same width, is unique, and contains active atoms.
    """

    candidates: ProposalCandidates
    recipe: PilotStratumRecipe
    base_size: int
    anchor_indices: np.ndarray
    anchor_valid: np.ndarray
    promoted_indices: np.ndarray
    pilot_unique_counts: np.ndarray
    pilot_finite_score_counts: np.ndarray
    fallback_fill_counts: np.ndarray
    max_neighbor_query_width: np.ndarray
    provenance: PilotStratumProvenance

    def __post_init__(self):
        if not isinstance(self.candidates, ProposalCandidates):
            raise TypeError("promoted candidates must be ProposalCandidates")
        if not isinstance(self.recipe, PilotStratumRecipe):
            raise TypeError("promotion recipe must be a PilotStratumRecipe")
        if not isinstance(self.provenance, PilotStratumProvenance):
            raise TypeError("promotion provenance must be PilotStratumProvenance")
        candidate_indices = np.array(
            self.candidates.indices, dtype=np.int64, copy=True
        )
        candidate_distances = np.array(
            self.candidates.distances, dtype=np.float64, copy=True
        )
        candidate_radius = np.array(
            self.candidates.radius, dtype=np.float64, copy=True
        )
        for value in (candidate_indices, candidate_distances, candidate_radius):
            value.setflags(write=False)
        candidates = ProposalCandidates(
            candidate_indices, candidate_distances, candidate_radius
        )
        base_size = int(self.base_size)
        anchors = np.array(self.anchor_indices, dtype=np.int64, copy=True)
        anchor_valid = np.array(self.anchor_valid, dtype=bool, copy=True)
        promoted = np.array(self.promoted_indices, dtype=np.int64, copy=True)
        unique_counts = np.array(self.pilot_unique_counts, dtype=np.int64, copy=True)
        finite_counts = np.array(
            self.pilot_finite_score_counts, dtype=np.int64, copy=True
        )
        fallback_counts = np.array(
            self.fallback_fill_counts, dtype=np.int64, copy=True
        )
        max_query_width = np.array(
            self.max_neighbor_query_width, dtype=np.int64, copy=True
        )
        n_objects, final_size = candidates.indices.shape
        if base_size <= 0 or base_size > final_size:
            raise ValueError("pilot promotion base size is invalid")
        if anchors.ndim != 2 or anchor_valid.shape != anchors.shape:
            raise ValueError("pilot anchors and validity must be aligned matrices")
        if len(anchors) != n_objects:
            raise ValueError("pilot anchor rows must align with candidates")
        if promoted.shape != (n_objects, final_size - base_size):
            raise ValueError("promoted atom matrix has the wrong fixed width")
        for values, name in (
            (unique_counts, "unique pilot counts"),
            (finite_counts, "finite pilot-score counts"),
            (fallback_counts, "deterministic fallback-fill counts"),
            (max_query_width, "maximum neighbor-query widths"),
        ):
            if values.shape != (n_objects,) or (values < 0).any():
                raise ValueError(f"{name} must contain one non-negative value per row")
        if (finite_counts > unique_counts).any():
            raise ValueError("finite pilot-score counts cannot exceed unique counts")
        if np.any(anchors[~anchor_valid] != -1) or np.any(
            anchors[anchor_valid] < 0
        ):
            raise ValueError("invalid anchor padding or atom ids")
        for row in range(n_objects):
            final = candidates.indices[row]
            if len(np.unique(final)) != len(final):
                raise ValueError("final exact-stratum rows must be unique")
            if len(np.unique(promoted[row])) != len(promoted[row]):
                raise ValueError("promoted atom rows must be unique")
            if not np.array_equal(final[base_size:], promoted[row]):
                raise ValueError("promoted atoms must be the final stratum suffix")
            valid_anchors = anchors[row, anchor_valid[row]]
            if not np.isin(valid_anchors, promoted[row]).all():
                raise ValueError("every consumed pilot anchor must be promoted")
        if self.recipe.promotion_strategy == "ranked_atoms_then_active_id_fill":
            anchor_counts = anchor_valid.sum(axis=1)
            expected_anchor_counts = np.minimum(
                finite_counts,
                min(self.recipe.max_anchors, promoted.shape[1]),
            )
            if not np.array_equal(anchor_counts, expected_anchor_counts):
                raise ValueError(
                    "direct promotion anchor counts do not match finite pilot scores"
                )
            if not np.array_equal(
                fallback_counts, promoted.shape[1] - anchor_counts
            ):
                raise ValueError(
                    "direct promotion fallback counts do not fill the stratum"
                )
            if np.any(max_query_width != 0):
                raise ValueError("direct promotion cannot record neighbor queries")
        provenance = self.provenance
        if provenance.recipe_sha256 != self.recipe.sha256:
            raise ValueError("pilot provenance recipe hash does not match the recipe")
        if (
            provenance.pilot_atom_chunk != self.recipe.atom_chunk
            or provenance.pilot_object_chunk != self.recipe.object_chunk
        ):
            raise ValueError("pilot provenance chunks do not match the recipe")
        base = candidates.indices[:, :base_size]
        call_manifests = {
            "call_base_stratum_indices_sha256": _array_manifest_sha256(
                "call_base_stratum_indices", base, dtype=">i8", sort_rows=True
            ),
            "call_anchor_indices_sha256": _array_manifest_sha256(
                "call_anchor_indices", anchors, dtype=">i8"
            ),
            "call_promoted_indices_sha256": _array_manifest_sha256(
                "call_promoted_indices", promoted, dtype=">i8", sort_rows=True
            ),
            "call_final_stratum_indices_sha256": _array_manifest_sha256(
                "call_final_stratum_indices",
                candidates.indices,
                dtype=">i8",
                sort_rows=True,
            ),
        }
        object_manifests = {
            "object_base_stratum_indices_sha256": _object_array_manifest_sha256(
                "object_base_stratum_indices", base, dtype=">i8", sort_rows=True
            ),
            "object_anchor_indices_sha256": _object_array_manifest_sha256(
                "object_anchor_indices", anchors, dtype=">i8"
            ),
            "object_promoted_indices_sha256": _object_array_manifest_sha256(
                "object_promoted_indices", promoted, dtype=">i8", sort_rows=True
            ),
            "object_final_stratum_indices_sha256": _object_array_manifest_sha256(
                "object_final_stratum_indices",
                candidates.indices,
                dtype=">i8",
                sort_rows=True,
            ),
        }
        for name, expected in {**call_manifests, **object_manifests}.items():
            if getattr(provenance, name) != expected:
                raise ValueError(
                    f"pilot provenance {name} does not match the promotion"
                )
        for value in (
            anchors,
            anchor_valid,
            promoted,
            unique_counts,
            finite_counts,
            fallback_counts,
            max_query_width,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "base_size", base_size)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "anchor_indices", anchors)
        object.__setattr__(self, "anchor_valid", anchor_valid)
        object.__setattr__(self, "promoted_indices", promoted)
        object.__setattr__(self, "pilot_unique_counts", unique_counts)
        object.__setattr__(self, "pilot_finite_score_counts", finite_counts)
        object.__setattr__(self, "fallback_fill_counts", fallback_counts)
        object.__setattr__(self, "max_neighbor_query_width", max_query_width)

    @property
    def promotion_size(self) -> int:
        return self.promoted_indices.shape[1]

    @property
    def anchor_counts(self) -> np.ndarray:
        counts = self.anchor_valid.sum(axis=1)
        counts.setflags(write=False)
        return counts


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

    def _validate_active_atom_rows(self, indices: np.ndarray, *, label: str) -> None:
        """Require every atom id in a matrix to belong to active support."""

        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 2:
            raise ValueError(f"{label} must be a matrix")
        positions = np.searchsorted(self.active_indices, values)
        valid = positions < len(self.active_indices)
        if valid.any():
            valid[valid] &= self.active_indices[positions[valid]] == values[valid]
        if not valid.all():
            raise ValueError(f"{label} must contain only active atom ids")

    def _canonical_coordinate_neighbors(
        self, anchor_indices: np.ndarray, *, n_neighbors: int
    ) -> np.ndarray:
        """Return exact-distance neighbors with atom-id tie breaking.

        Each anchor uses a bounded ``KDTree.query`` probe.  If the requested
        kth distance ties the probe boundary, the probe doubles until it finds
        a strictly larger distance or reaches the full active support.  Exact
        float64 squared distances and absolute atom ids then define the final
        order.  Processing one anchor at a time bounds temporary workspace by
        ``O(n_active * n_coordinates)`` even when every catalogue coordinate is
        tied; it never materializes one such workspace per anchor.
        """

        anchors = np.asarray(anchor_indices, dtype=np.int64)
        if anchors.ndim != 1 or not len(anchors):
            raise ValueError("coordinate-neighbor anchors must be a nonempty vector")
        if n_neighbors <= 0:
            raise ValueError("coordinate-neighbor count must be positive")
        self._validate_active_atom_rows(anchors[None, :], label="neighbor anchors")
        n_active = len(self.active_indices)
        k = min(int(n_neighbors), n_active)
        queries = (
            self.coordinates.values[anchors] - self.coordinates.center
        ) / self.coordinates.scale
        result = np.empty((len(anchors), k), dtype=np.int64)
        initial_probe = min(n_active, max(k + 1, 2 * k))
        for row, query in enumerate(queries):
            probe = initial_probe
            while True:
                distances, tree_positions = self.tree.query(
                    query, k=probe, workers=-1
                )
                distances = np.asarray(distances, dtype=np.float64).reshape(probe)
                tree_positions = np.asarray(
                    tree_positions, dtype=np.int64
                ).reshape(probe)
                if probe == n_active:
                    break
                kth_distance = distances[k - 1]
                probe_boundary = distances[-1]
                margin = (
                    32.0
                    * np.finfo(np.float64).eps
                    * max(1.0, abs(kth_distance), abs(probe_boundary))
                )
                if probe_boundary > kth_distance + margin:
                    break
                probe = min(n_active, max(probe + 1, 2 * probe))
            atom_ids = self.active_indices[tree_positions]
            atom_standardized = (
                self.coordinates.values[atom_ids] - self.coordinates.center
            ) / self.coordinates.scale
            difference = atom_standardized - query
            distance_squared = np.einsum(
                "ij,ij->i", difference, difference, optimize=True
            )
            order = np.lexsort((atom_ids, distance_squared))
            result[row] = atom_ids[order[:k]]
        return result

    def promote_exact_stratum(
        self,
        candidates: ProposalCandidates,
        observed,
        pilot_draw: ProposalDraw,
        pilot_scores: np.ndarray,
        *,
        recipe: PilotStratumRecipe = PILOT_STRATUM_RECIPE,
    ) -> PilotStratumPromotion:
        """Promote pilot-discovered atoms into an exact stratum.

        ``pilot_scores`` is intentionally explicit.  The selected development
        recipe expects ``log(c^2/q)`` (equivalently ``2*log(c/q)+log(q)``), but
        callers can instantiate another immutable recipe for a predeclared
        ranking or quota.  Duplicate pilot atoms use their maximum score, then
        atom id breaks ties. The recipe either consumes ranked atoms directly
        with an ascending-support uniqueness fill, or uses the retained
        development-only coordinate-cluster strategy.

        The pilot is selection data only.  This method returns no estimator
        weights and performs no production draw; consumers must exact-sum the
        final stratum and independently sample a newly conditioned complement.
        """

        if not isinstance(candidates, ProposalCandidates):
            raise TypeError("base exact stratum must be ProposalCandidates")
        if not isinstance(pilot_draw, ProposalDraw):
            raise TypeError("pilot draw must be a ProposalDraw")
        if not isinstance(recipe, PilotStratumRecipe):
            raise TypeError("pilot recipe must be a PilotStratumRecipe")
        base = np.asarray(candidates.indices, dtype=np.int64)
        pilot_indices = np.asarray(pilot_draw.indices, dtype=np.int64)
        pilot_probability = np.asarray(pilot_draw.probability, dtype=np.float64)
        scores = np.asarray(pilot_scores, dtype=np.float64)
        n_objects, base_size = base.shape
        expected_draw_shape = (n_objects, recipe.pilot_draws)
        if pilot_indices.shape != expected_draw_shape or scores.shape != expected_draw_shape:
            raise ValueError(
                "pilot atoms and explicit scores must match the recipe draw matrix"
            )
        if np.isnan(scores).any() or np.isposinf(scores).any():
            raise ValueError("pilot scores may be finite or -inf, but not NaN/+inf")
        if (
            pilot_probability.shape != expected_draw_shape
            or pilot_draw.local_member.shape != expected_draw_shape
            or not np.isfinite(pilot_probability).all()
            or (pilot_probability <= 0.0).any()
        ):
            raise ValueError("pilot proposal probabilities must be positive and finite")
        if base_size <= 0:
            raise ValueError("base exact stratum must be nonempty")
        if any(len(np.unique(row)) != base_size for row in base):
            raise ValueError("base exact-stratum rows must be unique")
        self._validate_active_atom_rows(base, label="base exact stratum")
        if base_size > len(self.active_indices):
            raise ValueError("base exact stratum is wider than active support")
        if pilot_draw.component_recipe != recipe.pilot_proposal:
            raise ValueError(
                "pilot draw component recipe does not match the frozen pilot recipe"
            )
        if (
            pilot_draw.atom_chunk != recipe.atom_chunk
            or pilot_draw.object_chunk != recipe.object_chunk
        ):
            raise ValueError(
                "pilot draw execution chunks do not match the frozen pilot recipe"
            )
        if (
            pilot_draw.object_excluded_candidate_indices_sha256
            != _candidate_exclusion_manifest_sha256(base)
        ):
            raise ValueError(
                "pilot draw conditioned-exclusion support does not match the base stratum"
            )
        if pilot_draw.component_id is None:
            raise ValueError("pilot draw must record its deterministic component schedule")
        schedule = np.tile(
            np.asarray(recipe.pilot_proposal.component_cycle, dtype=np.int16),
            recipe.pilot_draws // recipe.pilot_proposal.cycle_length,
        )
        if not np.array_equal(
            pilot_draw.component_id,
            np.broadcast_to(schedule, expected_draw_shape),
        ):
            raise ValueError("pilot component schedule does not match its frozen recipe")
        if (
            pilot_draw.component_complement_mass is None
            or pilot_draw.component_complement_mass.shape
            != (n_objects, len(recipe.pilot_proposal.component_names))
        ):
            raise ValueError("pilot complement-mass diagnostics do not match its recipe")
        self._validate_active_atom_rows(pilot_indices, label="pilot draws")
        values = self._observed_values(observed)
        if len(values) != n_objects:
            raise ValueError("observations and exact-stratum rows must align")
        promotion_size = min(
            recipe.n_promotions, len(self.active_indices) - base_size
        )
        anchor_width = min(recipe.max_anchors, promotion_size)
        promoted = np.empty((n_objects, promotion_size), dtype=np.int64)
        anchors = np.full((n_objects, anchor_width), -1, dtype=np.int64)
        anchor_valid = np.zeros_like(anchors, dtype=bool)
        unique_counts = np.empty(n_objects, dtype=np.int64)
        finite_counts = np.empty(n_objects, dtype=np.int64)
        fallback_fill_counts = np.zeros(n_objects, dtype=np.int64)
        max_query_width = np.zeros(n_objects, dtype=np.int64)
        seen = np.zeros(len(self.prior_weights), dtype=bool)

        for row in range(n_objects):
            base_row = base[row]
            seen[base_row] = True
            selected_count = 0
            try:
                if promotion_size == 0:
                    unique_counts[row] = 0
                    finite_counts[row] = 0
                    continue
                if seen[pilot_indices[row]].any() or pilot_draw.local_member[row].any():
                    raise ValueError(
                        "pilot draw must be conditioned on the base-stratum complement"
                    )
                unique, inverse = np.unique(
                    pilot_indices[row], return_inverse=True
                )
                log_probability = np.log(pilot_probability[row])
                minimum_log_probability = np.full(
                    len(unique), np.inf, dtype=np.float64
                )
                maximum_log_probability = np.full(
                    len(unique), -np.inf, dtype=np.float64
                )
                np.minimum.at(minimum_log_probability, inverse, log_probability)
                np.maximum.at(maximum_log_probability, inverse, log_probability)
                if np.any(
                    maximum_log_probability - minimum_log_probability
                    > PILOT_DUPLICATE_LOG_PROBABILITY_ATOL
                ):
                    raise ValueError(
                        "duplicate pilot atom has inconsistent balance probability"
                    )
                unique_score = np.full(len(unique), -np.inf, dtype=np.float64)
                np.maximum.at(unique_score, inverse, scores[row])
                unique_counts[row] = len(unique)
                finite_counts[row] = int(np.isfinite(unique_score).sum())
                order = np.lexsort((unique, -unique_score))
                ranked = unique[order[np.isfinite(unique_score[order])]][
                    : recipe.max_anchors
                ]

                consumed = []
                if recipe.promotion_strategy == "ranked_atoms_then_active_id_fill":
                    # The clustered v1 result showed that score-ranked pilot
                    # atoms carried much more target mass per slot than their
                    # coordinate neighbours. Consume ranked unique atoms
                    # directly, then fill a uniqueness deficit deterministically.
                    for anchor in ranked[:anchor_width]:
                        atom = int(anchor)
                        consumed.append(atom)
                        anchors[row, len(consumed) - 1] = atom
                        anchor_valid[row, len(consumed) - 1] = True
                        if seen[atom]:
                            raise RuntimeError(
                                "direct pilot anchor already belongs to the stratum"
                            )
                        promoted[row, selected_count] = atom
                        selected_count += 1
                        seen[atom] = True
                        if selected_count == promotion_size:
                            break
                    if selected_count < promotion_size:
                        for atom in self.active_indices:
                            atom = int(atom)
                            if not seen[atom]:
                                promoted[row, selected_count] = atom
                                selected_count += 1
                                fallback_fill_counts[row] += 1
                                seen[atom] = True
                                if selected_count == promotion_size:
                                    break
                else:
                    # Batch tree queries without making their return order part
                    # of the algorithm. Each cluster is anchor + nearest atoms.
                    cluster_query = min(
                        len(self.active_indices), recipe.neighbors_per_anchor
                    )
                    if len(ranked):
                        max_query_width[row] = cluster_query
                    anchor_batch = 64

                    def consume_anchor_range(anchor_values):
                        nonlocal selected_count
                        for start in range(0, len(anchor_values), anchor_batch):
                            stop = min(start + anchor_batch, len(anchor_values))
                            batch = anchor_values[start:stop]
                            neighbor_rows = self._canonical_coordinate_neighbors(
                                batch, n_neighbors=cluster_query
                            )
                            for anchor, neighbors in zip(batch, neighbor_rows):
                                if len(consumed) >= anchor_width:
                                    return
                                consumed.append(int(anchor))
                                anchors[row, len(consumed) - 1] = int(anchor)
                                anchor_valid[row, len(consumed) - 1] = True
                                cluster = np.concatenate(
                                    (
                                        np.asarray([anchor], dtype=np.int64),
                                        neighbors[neighbors != anchor][
                                            : max(
                                                recipe.neighbors_per_anchor - 1,
                                                0,
                                            )
                                        ],
                                    )
                                )
                                for atom in cluster:
                                    atom = int(atom)
                                    if not seen[atom]:
                                        promoted[row, selected_count] = atom
                                        selected_count += 1
                                        seen[atom] = True
                                        if selected_count == promotion_size:
                                            return

                    consume_anchor_range(ranked[:anchor_width])

                    # A degenerate pilot with fewer finite anchors receives an
                    # explicit ascending-support fill.
                    if (
                        selected_count < promotion_size
                        and len(consumed)
                        < min(recipe.max_anchors, promotion_size)
                    ):
                        for atom in self.active_indices:
                            atom = int(atom)
                            if not seen[atom]:
                                promoted[row, selected_count] = atom
                                selected_count += 1
                                fallback_fill_counts[row] += 1
                                seen[atom] = True
                                if selected_count == promotion_size:
                                    break

                    # Heavy overlap deepens only the consumed anchor
                    # neighborhoods; it never consumes another ranked atom.
                    query_k = cluster_query
                    while selected_count < promotion_size:
                        if not consumed:
                            raise RuntimeError(
                                "pilot promotion has no usable finite anchor"
                            )
                        if query_k == len(self.active_indices):
                            raise RuntimeError(
                                "pilot promotion could not fill exact stratum"
                            )
                        query_k = min(
                            len(self.active_indices), max(query_k + 1, 2 * query_k)
                        )
                        max_query_width[row] = query_k
                        neighbor_rows = self._canonical_coordinate_neighbors(
                            np.asarray(consumed, dtype=np.int64),
                            n_neighbors=query_k,
                        )
                        before = selected_count
                        for neighbors in neighbor_rows:
                            for atom in neighbors:
                                atom = int(atom)
                                if not seen[atom]:
                                    promoted[row, selected_count] = atom
                                    selected_count += 1
                                    seen[atom] = True
                                    if selected_count == promotion_size:
                                        break
                            if selected_count == promotion_size:
                                break
                        if (
                            selected_count == before
                            and query_k == len(self.active_indices)
                        ):
                            raise RuntimeError(
                                "pilot promotion exhausted active support"
                            )
            finally:
                seen[base_row] = False
                # Only the initialized prefix is marked if validation raises.
                if selected_count:
                    seen[promoted[row, :selected_count]] = False

        final_indices = (
            np.concatenate((base, promoted), axis=1)
            if promotion_size
            else base.copy()
        )
        # Candidate constructors may use uncertainty scores rather than metric
        # distances.  Once support is promoted, every distance and the radius
        # must have one meaning: Euclidean separation in the standardized
        # proposal coordinates.  Index raw coordinates so this never
        # materializes the full catalogue-sized standardized table.
        observed_standardized = (
            values - self.coordinates.center
        ) / self.coordinates.scale
        final_standardized = (
            self.coordinates.values[final_indices] - self.coordinates.center
        ) / self.coordinates.scale
        final_distances = np.sqrt(
            np.square(
                final_standardized - observed_standardized[:, None, :]
            ).sum(axis=2)
        )
        final_radius = np.max(final_distances, axis=1)
        final_candidates = ProposalCandidates(
            final_indices, final_distances, final_radius
        )

        provenance = PilotStratumProvenance(
            hash_contract=PILOT_STRATUM_HASH_CONTRACT,
            recipe_sha256=recipe.sha256,
            pilot_seed=int(pilot_draw.seed),
            pilot_atom_chunk=pilot_draw.atom_chunk,
            pilot_object_chunk=pilot_draw.object_chunk,
            call_pilot_draw_indices_sha256=_array_manifest_sha256(
                "call_pilot_draw_indices", pilot_indices, dtype=">i8"
            ),
            call_pilot_probability_sha256=_array_manifest_sha256(
                "call_pilot_probability", pilot_probability, dtype=">f8"
            ),
            call_pilot_scores_sha256=_array_manifest_sha256(
                "call_pilot_scores", scores, dtype=">f8"
            ),
            call_base_stratum_indices_sha256=_array_manifest_sha256(
                "call_base_stratum_indices", base, dtype=">i8", sort_rows=True
            ),
            call_anchor_indices_sha256=_array_manifest_sha256(
                "call_anchor_indices", anchors, dtype=">i8"
            ),
            call_promoted_indices_sha256=_array_manifest_sha256(
                "call_promoted_indices", promoted, dtype=">i8", sort_rows=True
            ),
            call_final_stratum_indices_sha256=_array_manifest_sha256(
                "call_final_stratum_indices",
                final_indices,
                dtype=">i8",
                sort_rows=True,
            ),
            object_pilot_draw_indices_sha256=_object_array_manifest_sha256(
                "object_pilot_draw_indices", pilot_indices, dtype=">i8"
            ),
            object_pilot_probability_sha256=_object_array_manifest_sha256(
                "object_pilot_probability", pilot_probability, dtype=">f8"
            ),
            object_pilot_scores_sha256=_object_array_manifest_sha256(
                "object_pilot_scores", scores, dtype=">f8"
            ),
            object_base_stratum_indices_sha256=_object_array_manifest_sha256(
                "object_base_stratum_indices",
                base,
                dtype=">i8",
                sort_rows=True,
            ),
            object_anchor_indices_sha256=_object_array_manifest_sha256(
                "object_anchor_indices", anchors, dtype=">i8"
            ),
            object_promoted_indices_sha256=_object_array_manifest_sha256(
                "object_promoted_indices",
                promoted,
                dtype=">i8",
                sort_rows=True,
            ),
            object_final_stratum_indices_sha256=_object_array_manifest_sha256(
                "object_final_stratum_indices",
                final_indices,
                dtype=">i8",
                sort_rows=True,
            ),
        )
        return PilotStratumPromotion(
            candidates=final_candidates,
            recipe=recipe,
            base_size=base_size,
            anchor_indices=anchors,
            anchor_valid=anchor_valid,
            promoted_indices=promoted,
            pilot_unique_counts=unique_counts,
            pilot_finite_score_counts=finite_counts,
            fallback_fill_counts=fallback_fill_counts,
            max_neighbor_query_width=max_query_width,
            provenance=provenance,
        )

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
        # The whole-catalogue sampler and the reranker use the same expanded
        # Gaussian coefficients.  Sharing them avoids a second roughly-GiB
        # device cache for the production catalogue.
        proxy = self._tilted_proxy(device)
        with torch.no_grad():
            approximate_log_target = proxy.score_positions(values, positions)
            selected_score, selected = torch.topk(
                approximate_log_target,
                k=n_candidates,
                dim=1,
                largest=True,
                sorted=True,
            )
            selected_position = torch.gather(positions, 1, selected)
            indices = proxy.active[selected_position]
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
    ) -> ProposalDraw:
        """Draw the complement from a tilted whole-catalogue proposal.

        :meth:`draw_stratified` samples the complement from the detected prior
        ``pi``, whose ratio ``c_j / pi_j`` has no useful observation-informed
        bound.  The finite catalogue makes every realized ratio finite, but
        cont.317 measured that their extreme concentration is what
        keeps the tail index above the 0.7 reliability threshold, and that
        about 80% of the mass the exact stratum misses lies outside the
        location prefilter, so no prefilter-restricted proposal can reach it.

        This draw replaces ``pi`` with a whole-catalogue defensive proposal

        ``q_i(j) = (1 - delta) softmax(s_i)_j + delta pi_j``

        where ``s_i`` is the diagonal Gaussian proxy the reranker already
        computes -- here evaluated over every active atom rather than only the
        prefilter, which costs two matmuls and no flow evaluations.  The
        defensive ``delta pi`` component keeps full support.  Because the
        candidate stratum ``S_i`` is already summed exactly, sampling ``q``
        there and then assigning the draw zero weight is dominated by drawing
        from its conditional distribution on the complement.  This method
        therefore samples

        ``q_i(j | j not in S_i) = q_i(j) / (1 - q_i(S_i))``

        and reports that conditional probability exactly.  Every nondegenerate
        draw reaches the sampled complement, while the estimator stays
        unbiased because its importance denominator is the distribution that
        actually generated the atom.

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
        for row in range(n_objects):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(absolute_ids[row])])
            )
            uniforms = rng.random((n_draws, 3))
            if candidates.indices.shape[1] == len(self.active_indices):
                # The exact stratum covers the full prior.  There is no
                # conditional complement to sample and the consumer will drop
                # these sentinel draws as local members before any flow call.
                drawn = np.full(
                    n_draws, candidates.indices[row, 0], dtype=np.int64
                )
                drawn_probability = np.ones(n_draws, dtype=np.float64)
            else:
                drawn, drawn_probability = proxy.draw_uniforms(
                    values[row],
                    uniforms[:, 1],
                    delta=float(delta),
                    temperature=float(temperature),
                    exclude_indices=candidates.indices[row],
                )
            indices[row] = drawn
            probability[row] = drawn_probability
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

    def draw_tilted_mis(
        self,
        candidates: ProposalCandidates,
        observed,
        *,
        n_draws: int,
        seed: int,
        object_offset: int = 0,
        object_ids: Optional[np.ndarray] = None,
        device: Optional[str | torch.device] = None,
        recipe: TailProposalRecipe = TAIL_PROPOSAL_RECIPE,
        atom_chunk: int = TAIL_PROPOSAL_ATOM_CHUNK,
        object_chunk: int = TAIL_PROPOSAL_OBJECT_CHUNK,
    ) -> ProposalDraw:
        """Draw a complement-conditioned deterministic-mixture MIS stream.

        The selected fixed recipe draws four samples from the T=1 proxy, three
        from the T=2 proxy, and one from ``pi`` in every eight-sample cycle.
        Each component is normalized separately on ``S^c`` before the mixture
        is formed. Every returned probability is the balance mixture

        ``qbar_C = 1/2 r1_C + 3/8 r2_C + 1/8 pi_C``,

        irrespective of which component generated that sample.  Fixed source
        counts make this deterministic-mixture estimator exactly unbiased and
        no more variable than iid sampling from ``qbar_C``.  Object-specific
        fixed-width random records preserve nested prefixes, object-batch
        invariance at one fixed atom-reduction chunk, and common random numbers
        with the other complement samplers.  The returned draw records both
        execution chunks because changing the atom chunk can change last-bit
        balance probabilities through floating-point reduction order.

        ``component_complement_mass`` records each *unconditioned* component's
        mass on ``S^c``.  When ``S`` is the full active support, a local-member
        sentinel with unit probability is returned for every scheduled source;
        consumers already discard such rows before evaluating the flow.
        """

        if not isinstance(recipe, TailProposalRecipe):
            raise TypeError("recipe must be a TailProposalRecipe")
        if n_draws <= 0 or n_draws % recipe.cycle_length:
            raise ValueError(
                f"n_draws must be a positive multiple of {recipe.cycle_length}"
            )
        if atom_chunk <= 0:
            raise ValueError("atom_chunk must be positive")
        if object_chunk <= 0:
            raise ValueError("object_chunk must be positive")
        if object_offset < 0:
            raise ValueError("object_offset must be non-negative")
        n_objects = len(candidates.indices)
        if object_ids is None:
            absolute_ids = object_offset + np.arange(n_objects, dtype=np.int64)
        else:
            absolute_ids = np.asarray(object_ids, dtype=np.int64)
            if object_offset != 0:
                raise ValueError(
                    "object_offset and explicit object_ids cannot be combined"
                )
            if absolute_ids.shape != (n_objects,) or (absolute_ids < 0).any():
                raise ValueError(
                    "object_ids must contain one non-negative id per row"
                )
        values = self._observed_values(observed)
        if len(values) != n_objects:
            raise ValueError("observations and candidate rows must align")

        proxy = self._tilted_proxy(device)
        schedule = np.tile(
            np.asarray(recipe.component_cycle, dtype=np.int16),
            n_draws // recipe.cycle_length,
        )
        indices = np.empty((n_objects, n_draws), dtype=np.int64)
        probability = np.empty((n_objects, n_draws), dtype=np.float64)
        component_id = np.broadcast_to(
            schedule, (n_objects, n_draws)
        ).copy()
        component_complement_mass = np.empty(
            (n_objects, len(recipe.component_names)), dtype=np.float64
        )
        local_member = np.zeros((n_objects, n_draws), dtype=bool)
        local_position = np.full((n_objects, n_draws), -1, dtype=np.int32)

        if candidates.indices.shape[1] == len(self.active_indices):
            # Candidate matrices have a common width.  If that width is the
            # active-support size, every valid row is a full-support stratum.
            # Still validate every row so duplicates or inactive ids cannot be
            # hidden behind the sentinel shortcut.
            for row in range(n_objects):
                excluded_position = proxy._excluded_positions(
                    candidates.indices[row]
                )
                if len(excluded_position) != len(self.active_indices):
                    raise RuntimeError("full-width candidate support is incomplete")
                if not len(candidates.indices[row]):
                    raise RuntimeError("full support has no sentinel candidate")
                indices[row].fill(int(candidates.indices[row, 0]))
                probability[row].fill(1.0)
                component_complement_mass[row].fill(0.0)
                local_member[row].fill(True)
                local_position[row].fill(0)
        else:
            # Random records stay object-local and fixed-width.  Batching only
            # changes how their atom uniforms are mapped through the proposal;
            # it cannot shift another object's stream or a nested prefix.
            for object_start in range(0, n_objects, int(object_chunk)):
                object_stop = min(object_start + int(object_chunk), n_objects)
                atom_uniforms = np.empty(
                    (object_stop - object_start, n_draws), dtype=np.float64
                )
                for local_row, absolute_id in enumerate(
                    absolute_ids[object_start:object_stop]
                ):
                    rng = np.random.default_rng(
                        np.random.SeedSequence([int(seed), int(absolute_id)])
                    )
                    atom_uniforms[local_row] = rng.random((n_draws, 3))[:, 1]
                (
                    indices[object_start:object_stop],
                    probability[object_start:object_stop],
                    component_complement_mass[object_start:object_stop],
                ) = proxy.draw_conditioned_mis_uniforms_batched(
                    values[object_start:object_stop],
                    atom_uniforms,
                    component_id[object_start:object_stop],
                    candidates.indices[object_start:object_stop],
                    recipe=recipe,
                    atom_chunk=int(atom_chunk),
                )

        return ProposalDraw(
            indices=indices,
            probability=probability,
            local_member=local_member,
            global_component=component_id == len(recipe.component_names) - 1,
            candidate_radius=candidates.radius,
            seed=int(seed),
            local_position=local_position,
            component_id=component_id,
            component_complement_mass=component_complement_mass,
            component_recipe=recipe,
            atom_chunk=int(atom_chunk),
            object_chunk=int(object_chunk),
            object_excluded_candidate_indices_sha256=(
                _candidate_exclusion_manifest_sha256(candidates.indices)
            ),
        )

    def _tilted_proxy(self, device: Optional[str | torch.device]) -> "WholeCatalogueProxy":
        """Cache the whole-catalogue proxy tables per device."""

        resolved = torch.device("cpu" if device is None else device)
        key = str(resolved)
        proxy = self._tilted_proxies.get(key)
        if proxy is None:
            proxy = WholeCatalogueProxy(self, resolved)
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

    def __init__(self, proposal, device):
        if proposal.coordinates.dispersion is None:
            raise ValueError("a tilted tail proposal needs a version-3 proposal cache")
        values = proposal.coordinates.values[proposal.active_indices]
        dispersion = proposal.coordinates.dispersion[proposal.active_indices]
        inverse_variance = np.reciprocal(np.square(dispersion))
        log_base = np.log(
            np.where(
                proposal.local_base_weights[proposal.active_indices] > 0,
                proposal.local_base_weights[proposal.active_indices],
                np.nan,
            )
        )
        gaussian_constant = (
            -np.log(dispersion).sum(axis=1)
            - 0.5 * (np.square(values) * inverse_variance).sum(axis=1)
        )
        # Atoms with no detection probability carry no target mass either, so
        # dropping them from the tilted component loses nothing; the defensive
        # prior component still covers them.
        self.log_base = torch.as_tensor(
            np.nan_to_num(log_base, nan=-np.inf),
            dtype=torch.float64,
            device=device,
        )
        self.gaussian_constant = torch.as_tensor(
            np.ascontiguousarray(gaussian_constant),
            dtype=torch.float64,
            device=device,
        )
        # ``constant`` is retained for callers that inspect the expanded T=1
        # score, while tempering uses the two terms above separately.
        self.constant = self.log_base + self.gaussian_constant
        self.a1 = torch.as_tensor(
            np.ascontiguousarray(inverse_variance), dtype=torch.float64, device=device
        )
        self.a2 = torch.as_tensor(
            np.ascontiguousarray(values * inverse_variance),
            dtype=torch.float64,
            device=device,
        )
        self.active = torch.as_tensor(
            np.ascontiguousarray(proposal.active_indices),
            dtype=torch.long,
            device=device,
        )
        self.active_indices = np.asarray(proposal.active_indices, dtype=np.int64)
        self.prior = torch.as_tensor(
            np.ascontiguousarray(proposal.prior_weights[proposal.active_indices]),
            dtype=torch.float64,
            device=device,
        )

    def score(self, observation: np.ndarray) -> torch.Tensor:
        """Return the untempered ``log(pi Pdet) + proxy log likelihood``."""

        return self.log_base + self.proxy_likelihood(observation)

    def proxy_likelihood(self, observation: np.ndarray) -> torch.Tensor:
        """Observation-dependent diagonal-Gaussian log likelihood."""

        x = torch.as_tensor(observation, dtype=torch.float64, device=self.a1.device)
        return self.gaussian_constant - 0.5 * (
            torch.square(x) @ self.a1.T - 2.0 * (x @ self.a2.T)
        )

    def score_positions(
        self, observations: np.ndarray, positions: torch.Tensor
    ) -> torch.Tensor:
        """Score an active-position slab using the shared proxy coefficients."""

        x = torch.as_tensor(
            np.ascontiguousarray(observations),
            dtype=torch.float64,
            device=self.a1.device,
        )
        if x.ndim != 2 or positions.ndim != 2 or len(x) != len(positions):
            raise ValueError("observations and proxy positions must be aligned matrices")
        selected_a1 = self.a1[positions]
        selected_a2 = self.a2[positions]
        likelihood = self.gaussian_constant[positions] - 0.5 * (
            (torch.square(x)[:, None, :] * selected_a1).sum(dim=2)
            - 2.0 * (x[:, None, :] * selected_a2).sum(dim=2)
        )
        return self.log_base[positions] + likelihood

    def component_log_scores(
        self, observation: np.ndarray, recipe: TailProposalRecipe
    ) -> tuple[torch.Tensor, ...]:
        """Dense component scores, primarily for exact small-catalogue checks."""

        likelihood = self.proxy_likelihood(observation)
        return tuple(
            self.log_base + float(beta) * likelihood
            for beta in recipe.component_betas[:-1]
        )

    def _excluded_positions(self, exclude_indices: np.ndarray) -> np.ndarray:
        excluded = np.asarray(exclude_indices, dtype=np.int64)
        if excluded.ndim != 1:
            raise ValueError("excluded atom ids must be one-dimensional")
        if len(np.unique(excluded)) != len(excluded):
            raise ValueError("excluded atom ids must be unique")
        positions = np.searchsorted(self.active_indices, excluded)
        valid = positions < len(self.active_indices)
        if valid.any():
            valid[valid] &= self.active_indices[positions[valid]] == excluded[valid]
        if not valid.all():
            raise ValueError("excluded atom ids must belong to active support")
        return np.sort(positions.astype(np.int64, copy=False))

    def _proxy_likelihood_block(
        self, observation: torch.Tensor, start: int, stop: int
    ) -> torch.Tensor:
        return self.gaussian_constant[start:stop] - 0.5 * (
            torch.square(observation) @ self.a1[start:stop].T
            - 2.0 * (observation @ self.a2[start:stop].T)
        )

    def _proxy_likelihood_blocks(
        self, observations: torch.Tensor, start: int, stop: int
    ) -> torch.Tensor:
        """Score one atom slab for a fixed batch of observations.

        The catalogue coefficients are read once for the whole observation
        batch.  This is mathematically the row-wise form of
        :meth:`_proxy_likelihood_block`; GEMM may change the last few floating
        point bits relative to independent GEMV calls, but neither the random
        records nor the probability being evaluated changes.
        """

        return self.gaussian_constant[start:stop][None, :] - 0.5 * (
            torch.square(observations) @ self.a1[start:stop].T
            - 2.0 * (observations @ self.a2[start:stop].T)
        )

    def _exclusion_mask(
        self,
        excluded_positions: Sequence[np.ndarray],
        start: int,
        stop: int,
    ) -> torch.Tensor:
        """Return a row-specific mask for one streamed atom slab."""

        n_rows = len(excluded_positions)
        mask = torch.zeros(
            (n_rows, stop - start),
            dtype=torch.bool,
            device=self.a1.device,
        )
        flat_rows = []
        flat_positions = []
        for row, excluded in enumerate(excluded_positions):
            left = np.searchsorted(excluded, start, side="left")
            right = np.searchsorted(excluded, stop, side="left")
            if right > left:
                flat_rows.append(np.full(right - left, row, dtype=np.int64))
                flat_positions.append(excluded[left:right] - start)
        if flat_rows:
            rows = torch.as_tensor(
                np.concatenate(flat_rows), dtype=torch.long, device=self.a1.device
            )
            positions = torch.as_tensor(
                np.concatenate(flat_positions),
                dtype=torch.long,
                device=self.a1.device,
            )
            mask[rows, positions] = True
        return mask

    @staticmethod
    def _padded_draw_matrix(
        draw_lists: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad ragged per-row draw ids and mark the entries that are real."""

        width = max((len(draws) for draws in draw_lists), default=0)
        indices = np.zeros((len(draw_lists), width), dtype=np.int64)
        valid = np.zeros((len(draw_lists), width), dtype=bool)
        for row, draws in enumerate(draw_lists):
            count = len(draws)
            if count:
                indices[row, :count] = draws
                valid[row, :count] = True
        return indices, valid

    def draw_conditioned_mis_uniforms_batched(
        self,
        observations: np.ndarray,
        uniforms: np.ndarray,
        component_id: np.ndarray,
        exclude_indices: np.ndarray,
        *,
        recipe: TailProposalRecipe,
        atom_chunk: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map several object streams through exact conditioned DM-MIS.

        Atom slabs are streamed twice, exactly as in the scalar constructor.
        The first pass obtains each object's component normalizers and block
        probabilities.  The second pass samples within selected blocks and
        evaluates the full balance density.  Only the observation axis is
        batched: exclusions, component conditioning, fixed source counts and
        object-local uniform records remain separate.

        At peak, the method owns ``O(B * atom_chunk * n_components)`` device
        memory for a proposal object batch of size ``B``.  It never constructs
        an ``n_objects * n_atoms`` table.
        """

        if atom_chunk <= 0:
            raise ValueError("atom_chunk must be positive")
        observations = np.asarray(observations, dtype=np.float64)
        uniforms = np.asarray(uniforms, dtype=np.float64)
        component_id = np.asarray(component_id, dtype=np.int16)
        exclude_indices = np.asarray(exclude_indices, dtype=np.int64)
        if observations.ndim != 2 or observations.shape[1] != self.a1.shape[1]:
            raise ValueError("observations have the wrong proxy shape")
        n_objects = len(observations)
        if not n_objects:
            raise ValueError("a conditioned MIS batch cannot be empty")
        if (
            uniforms.ndim != 2
            or uniforms.shape[0] != n_objects
            or component_id.shape != uniforms.shape
            or exclude_indices.ndim != 2
            or exclude_indices.shape[0] != n_objects
            or not np.isfinite(observations).all()
            or not np.isfinite(uniforms).all()
        ):
            raise ValueError(
                "observations, uniforms, component ids, and exclusions must align"
            )
        if ((uniforms < 0.0) | (uniforms >= 1.0)).any():
            raise ValueError("uniforms must lie in [0, 1)")
        if (
            (component_id < 0).any()
            or (component_id >= len(recipe.component_names)).any()
        ):
            raise ValueError("component ids are invalid for the tail recipe")

        excluded_positions = [
            self._excluded_positions(exclude_indices[row])
            for row in range(n_objects)
        ]
        n_atoms = len(self.active_indices)
        if any(len(excluded) == n_atoms for excluded in excluded_positions):
            raise ValueError("cannot sample an empty catalogue complement")
        bounds = [
            (start, min(start + int(atom_chunk), n_atoms))
            for start in range(0, n_atoms, int(atom_chunk))
        ]
        n_blocks = len(bounds)
        n_proxy = len(recipe.component_betas) - 1
        n_components = len(recipe.component_names)
        observation_tensor = torch.as_tensor(
            np.ascontiguousarray(observations),
            dtype=torch.float64,
            device=self.a1.device,
        )

        complement_log_blocks = torch.empty(
            (n_objects, n_proxy, n_blocks),
            dtype=torch.float64,
            device=self.a1.device,
        )
        full_log_z = torch.full(
            (n_objects, n_proxy),
            -torch.inf,
            dtype=torch.float64,
            device=self.a1.device,
        )
        prior_block_mass = torch.empty(
            (n_objects, n_blocks),
            dtype=torch.float64,
            device=self.a1.device,
        )

        with torch.no_grad():
            for block, (start, stop) in enumerate(bounds):
                likelihood = self._proxy_likelihood_blocks(
                    observation_tensor, start, stop
                )
                mask = self._exclusion_mask(excluded_positions, start, stop)
                prior = self.prior[start:stop][None, :].expand(n_objects, -1)
                prior_block_mass[:, block] = prior.masked_fill(mask, 0.0).sum(
                    dim=1
                )
                for component, beta in enumerate(recipe.component_betas[:-1]):
                    score = (
                        self.log_base[start:stop][None, :]
                        + float(beta) * likelihood
                    )
                    block_full = torch.logsumexp(score, dim=1)
                    block_complement = torch.logsumexp(
                        score.masked_fill(mask, -torch.inf), dim=1
                    )
                    full_log_z[:, component] = torch.logaddexp(
                        full_log_z[:, component], block_full
                    )
                    complement_log_blocks[:, component, block] = block_complement

            complement_log_z = torch.logsumexp(
                complement_log_blocks, dim=2
            )
            invalid_full = torch.isnan(full_log_z) | torch.isposinf(full_log_z)
            invalid_complement = torch.isnan(complement_log_z) | torch.isposinf(
                complement_log_z
            )
            if bool((invalid_full | invalid_complement).any()):
                bad_components = torch.nonzero(
                    (invalid_full | invalid_complement).any(dim=0)
                ).flatten().cpu().tolist()
                names = ", ".join(
                    recipe.component_names[int(component)]
                    for component in bad_components
                )
                raise RuntimeError(
                    "proxy component produced a NaN or positive-infinite "
                    f"normalizer: {names}"
                )
            if bool(
                (torch.isneginf(full_log_z) & ~torch.isneginf(complement_log_z)).any()
            ):
                raise RuntimeError(
                    "proxy complement has mass while its full support does not"
                )

            prior_complement_mass = prior_block_mass.sum(dim=1)
            if not bool(
                (
                    torch.isfinite(prior_complement_mass)
                    & (prior_complement_mass > 0.0)
                ).all()
            ):
                raise RuntimeError("catalogue prior has zero complement mass")

            proxy_fallback_tensor = torch.isneginf(complement_log_z)
            prior_block_probability = (
                prior_block_mass / prior_complement_mass[:, None]
            )
            proxy_block_probability = torch.exp(
                complement_log_blocks - complement_log_z[:, :, None]
            )
            proxy_block_probability = torch.where(
                proxy_fallback_tensor[:, :, None],
                prior_block_probability[:, None, :],
                proxy_block_probability,
            )
            block_probability_tensor = torch.cat(
                (proxy_block_probability, prior_block_probability[:, None, :]),
                dim=1,
            )

            proxy_diagnostics = torch.where(
                proxy_fallback_tensor,
                torch.zeros_like(complement_log_z),
                torch.exp(complement_log_z - full_log_z),
            )
            diagnostics_tensor = torch.cat(
                (proxy_diagnostics, prior_complement_mass[:, None]), dim=1
            )

        # One synchronization per proposal object batch, rather than once per
        # atom block and object.  Normalizing again in float64 ensures every
        # row's CDF terminates at exactly one after roundoff.
        block_probability = block_probability_tensor.cpu().numpy()
        block_probability /= block_probability.sum(axis=2, keepdims=True)
        diagnostics = np.clip(
            diagnostics_tensor.cpu().numpy(), 0.0, 1.0
        )
        proxy_fallback = proxy_fallback_tensor.cpu().numpy()

        selected_block = np.empty(uniforms.shape, dtype=np.int64)
        residual_uniform = np.empty(uniforms.shape, dtype=np.float64)
        block_cdf = np.cumsum(block_probability, axis=2)
        block_cdf[:, :, -1] = 1.0
        for row in range(n_objects):
            for component in range(n_components):
                draws = np.flatnonzero(component_id[row] == component)
                if not len(draws):
                    continue
                blocks = np.searchsorted(
                    block_cdf[row, component],
                    uniforms[row, draws],
                    side="right",
                )
                blocks = np.minimum(blocks, n_blocks - 1)
                selected_block[row, draws] = blocks
                lower = np.where(
                    blocks > 0,
                    block_cdf[
                        row, component, np.maximum(blocks - 1, 0)
                    ],
                    0.0,
                )
                mass = block_probability[row, component, blocks]
                if (mass <= 0.0).any():
                    raise RuntimeError("conditioned MIS selected a zero-mass block")
                residual_uniform[row, draws] = np.minimum(
                    (uniforms[row, draws] - lower) / mass,
                    np.nextafter(1.0, 0.0),
                )

        drawn_indices_device = torch.empty(
            uniforms.shape, dtype=torch.long, device=self.a1.device
        )
        drawn_probability_device = torch.empty(
            uniforms.shape, dtype=torch.float64, device=self.a1.device
        )
        drawn_local_position = torch.empty(
            uniforms.shape, dtype=torch.long, device=self.a1.device
        )
        mixture_weight = torch.as_tensor(
            recipe.component_weights,
            dtype=torch.float64,
            device=self.a1.device,
        )
        normalized_block_probability = torch.as_tensor(
            np.ascontiguousarray(block_probability),
            dtype=torch.float64,
            device=self.a1.device,
        )

        with torch.no_grad():
            for block in np.unique(selected_block):
                start, stop = bounds[int(block)]
                active_rows = np.flatnonzero(
                    (selected_block == int(block)).any(axis=1)
                )
                active_rows_device = torch.as_tensor(
                    active_rows, dtype=torch.long, device=self.a1.device
                )
                likelihood = self._proxy_likelihood_blocks(
                    observation_tensor[active_rows_device], start, stop
                )
                mask = self._exclusion_mask(
                    [excluded_positions[row] for row in active_rows], start, stop
                )
                prior = self.prior[start:stop][None, :].expand(
                    len(active_rows), -1
                ).masked_fill(mask, 0.0)
                prior_total = prior.sum(dim=1, keepdim=True)
                prior_within = torch.where(
                    prior_total > 0.0,
                    prior / prior_total,
                    torch.zeros_like(prior),
                )

                within = []
                for component, beta in enumerate(recipe.component_betas[:-1]):
                    score = (
                        self.log_base[start:stop][None, :]
                        + float(beta) * likelihood
                    ).masked_fill(mask, -torch.inf)
                    component_within = torch.zeros_like(score)
                    regular = np.flatnonzero(
                        ~proxy_fallback[active_rows, component]
                        & (
                            block_probability[
                                active_rows, component, int(block)
                            ]
                            > 0.0
                        )
                    )
                    if len(regular):
                        regular_device = torch.as_tensor(
                            regular, dtype=torch.long, device=self.a1.device
                        )
                        component_within[regular_device] = torch.softmax(
                            score[regular_device], dim=1
                        )
                    fallback = np.flatnonzero(
                        proxy_fallback[active_rows, component]
                    )
                    if len(fallback):
                        fallback_device = torch.as_tensor(
                            fallback, dtype=torch.long, device=self.a1.device
                        )
                        component_within[fallback_device] = prior_within[
                            fallback_device
                        ]
                    within.append(component_within)
                within.append(prior_within)
                within_probability = torch.stack(within, dim=0)

                # Search every object's within-block CDF in one kernel per
                # source component.  Ragged draw lists are padded only across
                # the small proposal object batch and discarded immediately.
                for component in range(n_components):
                    draw_lists = [
                        np.flatnonzero(
                            (component_id[row] == component)
                            & (selected_block[row] == int(block))
                        )
                        for row in active_rows
                    ]
                    draw_matrix, valid = self._padded_draw_matrix(draw_lists)
                    if not valid.any():
                        continue
                    residual = np.zeros(draw_matrix.shape, dtype=np.float64)
                    for row, draws in enumerate(draw_lists):
                        if len(draws):
                            residual[row, : len(draws)] = residual_uniform[
                                active_rows[row], draws
                            ]
                    cumulative = torch.cumsum(
                        within_probability[component], dim=1
                    )
                    cumulative[:, -1] = 1.0
                    local = torch.searchsorted(
                        cumulative,
                        torch.as_tensor(
                            np.ascontiguousarray(residual),
                            dtype=torch.float64,
                            device=self.a1.device,
                        ),
                        right=True,
                    ).clamp(max=stop - start - 1)
                    valid_device = torch.as_tensor(
                        valid, dtype=torch.bool, device=self.a1.device
                    )
                    draw_matrix_device = torch.as_tensor(
                        draw_matrix, dtype=torch.long, device=self.a1.device
                    )
                    row_matrix_device = active_rows_device[:, None].expand_as(
                        draw_matrix_device
                    )
                    target_rows = row_matrix_device[valid_device]
                    target_draws = draw_matrix_device[valid_device]
                    target_local = local[valid_device]
                    drawn_local_position[target_rows, target_draws] = target_local
                    drawn_indices_device[target_rows, target_draws] = self.active[
                        start + target_local
                    ]

                all_draw_lists = [
                    np.flatnonzero(selected_block[row] == int(block))
                    for row in active_rows
                ]
                draw_matrix, valid = self._padded_draw_matrix(all_draw_lists)
                valid_device = torch.as_tensor(
                    valid, dtype=torch.bool, device=self.a1.device
                )
                draw_matrix_device = torch.as_tensor(
                    draw_matrix, dtype=torch.long, device=self.a1.device
                )
                row_matrix_device = active_rows_device[:, None].expand_as(
                    draw_matrix_device
                )
                local = drawn_local_position[
                    row_matrix_device, draw_matrix_device
                ]
                # Padded draw ids are arbitrary placeholders.  Keep their
                # gather index valid even though their qbar entries are never
                # copied to the returned stream.
                local = torch.where(
                    valid_device, local, torch.zeros_like(local)
                )
                qbar = torch.zeros(
                    draw_matrix.shape,
                    dtype=torch.float64,
                    device=self.a1.device,
                )
                for component in range(n_components):
                    within_at_draw = torch.gather(
                        within_probability[component], 1, local
                    )
                    qbar += (
                        mixture_weight[component]
                        * normalized_block_probability[
                            active_rows_device, component, int(block)
                        ][:, None]
                        * within_at_draw
                    )
                drawn_probability_device[
                    row_matrix_device[valid_device],
                    draw_matrix_device[valid_device],
                ] = qbar[valid_device]

        drawn_indices = drawn_indices_device.cpu().numpy()
        drawn_probability = drawn_probability_device.cpu().numpy()
        if (
            not np.isfinite(drawn_probability).all()
            or (drawn_probability <= 0.0).any()
        ):
            raise RuntimeError(
                "conditioned MIS produced an invalid balance probability"
            )
        for row in range(n_objects):
            if np.isin(drawn_indices[row], exclude_indices[row]).any():
                raise RuntimeError(
                    "conditioned MIS sampled its excluded exact stratum"
                )
        return drawn_indices, drawn_probability, diagnostics

    def draw_conditioned_mis_uniforms(
        self,
        observation: np.ndarray,
        uniforms: np.ndarray,
        component_id: np.ndarray,
        exclude_indices: np.ndarray,
        *,
        recipe: TailProposalRecipe,
        atom_chunk: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map uniforms through separately conditioned DM-MIS components.

        This is a blocked two-pass sampler.  The first pass stores only one
        log mass per component and atom block.  The second revisits blocks hit
        by the draw stream, samples within each block, and evaluates the exact
        hierarchical balance-mixture probability.  Its working memory is
        ``O(atom_chunk)`` rather than ``O(n_objects * n_atoms)``.
        """

        if atom_chunk <= 0:
            raise ValueError("atom_chunk must be positive")
        uniforms = np.asarray(uniforms, dtype=np.float64)
        component_id = np.asarray(component_id, dtype=np.int16)
        if (
            uniforms.ndim != 1
            or component_id.shape != uniforms.shape
            or not np.isfinite(uniforms).all()
        ):
            raise ValueError("uniforms and component ids must be aligned vectors")
        if ((uniforms < 0.0) | (uniforms >= 1.0)).any():
            raise ValueError("uniforms must lie in [0, 1)")
        if (
            (component_id < 0).any()
            or (component_id >= len(recipe.component_names)).any()
        ):
            raise ValueError("component ids are invalid for the tail recipe")
        excluded = self._excluded_positions(exclude_indices)
        n_atoms = len(self.active_indices)
        if len(excluded) == n_atoms:
            raise ValueError("cannot sample an empty catalogue complement")

        observation_tensor = torch.as_tensor(
            np.ascontiguousarray(observation),
            dtype=torch.float64,
            device=self.a1.device,
        )
        if observation_tensor.shape != (self.a1.shape[1],):
            raise ValueError("observation has the wrong proxy dimension")
        bounds = [
            (start, min(start + int(atom_chunk), n_atoms))
            for start in range(0, n_atoms, int(atom_chunk))
        ]
        n_proxy = len(recipe.component_betas) - 1
        complement_log_blocks: list[list[torch.Tensor]] = [
            [] for _ in range(n_proxy)
        ]
        full_log_blocks: list[list[torch.Tensor]] = [
            [] for _ in range(n_proxy)
        ]
        prior_blocks: list[torch.Tensor] = []

        with torch.no_grad():
            for start, stop in bounds:
                likelihood = self._proxy_likelihood_block(
                    observation_tensor, start, stop
                )
                left = np.searchsorted(excluded, start, side="left")
                right = np.searchsorted(excluded, stop, side="left")
                block_excluded = excluded[left:right] - start
                mask = None
                if len(block_excluded):
                    mask = torch.zeros(
                        stop - start, dtype=torch.bool, device=self.a1.device
                    )
                    mask[
                        torch.as_tensor(
                            block_excluded,
                            dtype=torch.long,
                            device=self.a1.device,
                        )
                    ] = True
                for component, beta in enumerate(recipe.component_betas[:-1]):
                    score = self.log_base[start:stop] + float(beta) * likelihood
                    full_log_blocks[component].append(torch.logsumexp(score, dim=0))
                    if mask is not None:
                        score = score.masked_fill(mask, -torch.inf)
                    complement_log_blocks[component].append(
                        torch.logsumexp(score, dim=0)
                    )
                prior = self.prior[start:stop]
                if mask is not None:
                    prior = prior.masked_fill(mask, 0.0)
                prior_blocks.append(prior.sum())

            full_log_z = [
                torch.logsumexp(torch.stack(blocks), dim=0)
                for blocks in full_log_blocks
            ]
            complement_log_z = [
                torch.logsumexp(torch.stack(blocks), dim=0)
                for blocks in complement_log_blocks
            ]
            for component, (full_mass, complement_mass) in enumerate(
                zip(full_log_z, complement_log_z)
            ):
                invalid_full = torch.isnan(full_mass) | torch.isposinf(full_mass)
                invalid_complement = torch.isnan(complement_mass) | torch.isposinf(
                    complement_mass
                )
                if bool(invalid_full | invalid_complement):
                    raise RuntimeError(
                        "proxy component produced a NaN or positive-infinite "
                        f"normalizer: {recipe.component_names[component]}"
                    )
                if bool(
                    torch.isneginf(full_mass)
                    & ~torch.isneginf(complement_mass)
                ):
                    raise RuntimeError(
                        "proxy complement has mass while its full support does not"
                    )
            prior_block_mass = torch.stack(prior_blocks)
            prior_complement_mass = prior_block_mass.sum()
            if not bool(
                torch.isfinite(prior_complement_mass)
                & (prior_complement_mass > 0.0)
            ):
                raise RuntimeError("catalogue prior has zero complement mass")

            diagnostics = np.empty(len(recipe.component_names), dtype=np.float64)
            for component in range(n_proxy):
                if not bool(torch.isneginf(complement_log_z[component])):
                    diagnostics[component] = float(
                        torch.exp(
                            complement_log_z[component] - full_log_z[component]
                        ).item()
                    )
                else:
                    diagnostics[component] = 0.0
            diagnostics[-1] = float(prior_complement_mass.item())
            diagnostics[:] = np.clip(diagnostics, 0.0, 1.0)

            prior_block_probability = (
                prior_block_mass / prior_complement_mass
            ).cpu().numpy()
            block_probability = np.empty(
                (len(recipe.component_names), len(bounds)), dtype=np.float64
            )
            proxy_fallback = np.zeros(n_proxy, dtype=bool)
            for component in range(n_proxy):
                if not bool(torch.isneginf(complement_log_z[component])):
                    block_probability[component] = torch.exp(
                        torch.stack(complement_log_blocks[component])
                        - complement_log_z[component]
                    ).cpu().numpy()
                else:
                    # Only an exact log(0) support mass reaches this branch.
                    # NaN and +inf are cache/numerical defects rejected above.
                    # Falling back to pi_C preserves support and the fixed
                    # source count without inventing a likelihood score; the
                    # complement-mass diagnostic deliberately remains 0.
                    proxy_fallback[component] = True
                    block_probability[component] = prior_block_probability
            block_probability[-1] = prior_block_probability
        block_probability /= block_probability.sum(axis=1, keepdims=True)

        block_cdf = np.cumsum(block_probability, axis=1)
        block_cdf[:, -1] = 1.0
        selected_block = np.empty(len(uniforms), dtype=np.int64)
        residual_uniform = np.empty(len(uniforms), dtype=np.float64)
        for component in range(len(recipe.component_names)):
            draws = np.flatnonzero(component_id == component)
            if not len(draws):
                continue
            blocks = np.searchsorted(
                block_cdf[component], uniforms[draws], side="right"
            )
            blocks = np.minimum(blocks, len(bounds) - 1)
            selected_block[draws] = blocks
            lower = np.where(
                blocks > 0, block_cdf[component, np.maximum(blocks - 1, 0)], 0.0
            )
            mass = block_probability[component, blocks]
            residual_uniform[draws] = np.minimum(
                (uniforms[draws] - lower) / mass,
                np.nextafter(1.0, 0.0),
            )

        drawn_indices_device = torch.empty(
            len(uniforms), dtype=torch.long, device=self.a1.device
        )
        drawn_probability_device = torch.empty(
            len(uniforms), dtype=torch.float64, device=self.a1.device
        )
        mixture_weight = torch.as_tensor(
            recipe.component_weights,
            dtype=torch.float64,
            device=self.a1.device,
        )
        with torch.no_grad():
            for block in np.unique(selected_block):
                start, stop = bounds[int(block)]
                likelihood = self._proxy_likelihood_block(
                    observation_tensor, start, stop
                )
                left = np.searchsorted(excluded, start, side="left")
                right = np.searchsorted(excluded, stop, side="left")
                block_excluded = excluded[left:right] - start
                mask = torch.zeros(
                    stop - start, dtype=torch.bool, device=self.a1.device
                )
                if len(block_excluded):
                    mask[
                        torch.as_tensor(
                            block_excluded,
                            dtype=torch.long,
                            device=self.a1.device,
                        )
                    ] = True
                prior = self.prior[start:stop].masked_fill(mask, 0.0)
                prior_total = prior.sum()
                prior_within = torch.where(
                    prior_total > 0.0, prior / prior_total, torch.zeros_like(prior)
                )
                within = []
                for component, beta in enumerate(recipe.component_betas[:-1]):
                    if proxy_fallback[component]:
                        within.append(prior_within)
                        continue
                    score = (
                        self.log_base[start:stop] + float(beta) * likelihood
                    ).masked_fill(mask, -torch.inf)
                    if block_probability[component, int(block)] > 0.0:
                        within.append(torch.softmax(score, dim=0))
                    else:
                        within.append(torch.zeros_like(score))
                within.append(prior_within)
                within_probability = torch.stack(within)
                block_component_probability = torch.as_tensor(
                    block_probability[:, int(block)],
                    dtype=torch.float64,
                    device=self.a1.device,
                )[:, None]

                for component in range(len(recipe.component_names)):
                    draws = np.flatnonzero(
                        (component_id == component) & (selected_block == block)
                    )
                    if not len(draws):
                        continue
                    draw_positions = torch.as_tensor(
                        draws, dtype=torch.long, device=self.a1.device
                    )
                    cumulative = torch.cumsum(
                        within_probability[component], dim=0
                    )
                    cumulative[-1] = 1.0
                    local = torch.searchsorted(
                        cumulative,
                        torch.as_tensor(
                            np.ascontiguousarray(residual_uniform[draws]),
                            dtype=torch.float64,
                            device=self.a1.device,
                        ),
                        right=True,
                    ).clamp(max=stop - start - 1)
                    drawn_indices_device[draw_positions] = self.active[start + local]
                    actual_component_probability = (
                        block_component_probability * within_probability[:, local]
                    )
                    drawn_probability_device[draw_positions] = (
                        mixture_weight[:, None] * actual_component_probability
                    ).sum(dim=0)

        # Keep block/component work on one device stream; synchronizing here
        # once per object avoids hundreds of tiny host transfers per catalogue
        # pass at production draw depth.
        drawn_indices = drawn_indices_device.cpu().numpy()
        drawn_probability = drawn_probability_device.cpu().numpy()

        if not np.isfinite(drawn_probability).all() or (drawn_probability <= 0.0).any():
            raise RuntimeError("conditioned MIS produced an invalid balance probability")
        if np.isin(drawn_indices, np.asarray(exclude_indices, dtype=np.int64)).any():
            raise RuntimeError("conditioned MIS sampled its excluded exact stratum")
        return drawn_indices, drawn_probability, diagnostics

    def draw(
        self,
        observation: np.ndarray,
        *,
        n_draws: int,
        delta: float,
        rng,
        temperature: float = 1.0,
        exclude_indices: Optional[np.ndarray] = None,
    ):
        """Sample ``q = (1-delta) softmax(s/T) + delta pi`` over every atom.

        Returns absolute atom ids and their exact mixture probability, so the
        importance ratio ``c_j / q_j`` stays exact and the estimator unbiased.
        """

        return self.draw_uniforms(
            observation,
            rng.random(n_draws),
            delta=delta,
            temperature=temperature,
            exclude_indices=exclude_indices,
        )

    def draw_uniforms(
        self,
        observation: np.ndarray,
        uniforms,
        *,
        delta: float,
        temperature: float = 1.0,
        exclude_indices: Optional[np.ndarray] = None,
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

        ``exclude_indices`` conditions the mixture on the complement of a
        deterministic exact stratum.  The returned probability is normalized
        on that complement, not the unconditional mixture probability.
        """

        if not 0.0 < delta < 1.0:
            raise ValueError("delta must lie strictly between zero and one")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive and finite")
        uniforms = np.asarray(uniforms, dtype=np.float64)
        if uniforms.ndim != 1 or not np.isfinite(uniforms).all():
            raise ValueError("uniforms must be a finite one-dimensional array")
        if ((uniforms < 0.0) | (uniforms >= 1.0)).any():
            raise ValueError("uniforms must lie in [0, 1)")
        # Temperature deliberately leaves the detected-prior base measure
        # untouched: only the approximate likelihood is flattened.
        score = self.log_base + self.proxy_likelihood(observation) / float(temperature)
        finite = torch.isfinite(score)
        shifted = torch.where(finite, score, torch.tensor(-np.inf, device=score.device))
        tilted = torch.softmax(shifted, dim=0)
        mixture = (1.0 - delta) * tilted + delta * self.prior
        if exclude_indices is not None:
            positions = self._excluded_positions(exclude_indices)
            if len(positions) == len(self.active_indices):
                raise ValueError("cannot sample an empty catalogue complement")
            mixture[
                torch.as_tensor(positions, dtype=torch.long, device=score.device)
            ] = 0.0
            normalizer = mixture.sum()
            normalizer_value = float(normalizer)
            if not np.isfinite(normalizer_value) or normalizer_value <= 0.0:
                raise RuntimeError("tilted proposal has zero complement mass")
            mixture = mixture / normalizer
        cumulative = torch.cumsum(mixture, dim=0)
        cumulative[-1] = 1.0
        drawn = torch.as_tensor(
            np.ascontiguousarray(uniforms),
            dtype=torch.float64,
            device=score.device,
        )
        positions = torch.searchsorted(cumulative, drawn, right=True)
        positions = torch.clamp(positions, max=len(mixture) - 1)
        return (
            self.active[positions].cpu().numpy(),
            mixture[positions].cpu().numpy(),
        )


def pareto_tail_index(weights, *, minimum_tail: int = 5):
    """Generalized-Pareto shape ``k`` of each row's upper weight tail.

    This is the Zhang and Stephens (2009) empirical-Bayes estimator used as the
    PSIS ``k-hat`` diagnostic.  It measures how heavy the importance-weight tail
    is.  In a continuous heavy-tail model, ``k >= 0.5`` corresponds to
    infinite variance and ``k >= 1`` to an infinite mean.  This estimator has
    finite support, so its moments are mathematically finite; here k-hat is an
    operational warning that the sampled regime behaves like a heavy tail and
    that root-M or 1/M extrapolation is not yet trustworthy.  The practical
    reliability threshold is ``min(1 - 1/log10(S), 0.7)`` for a sample of size
    ``S``.

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
    "PILOT_STRATUM_HASH_CONTRACT",
    "PILOT_DUPLICATE_LOG_PROBABILITY_ATOL",
    "PILOT_STRATUM_PROPOSAL_RECIPE",
    "PILOT_STRATUM_RECIPE",
    "PilotStratumPromotion",
    "PilotStratumProvenance",
    "PilotStratumRecipe",
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
